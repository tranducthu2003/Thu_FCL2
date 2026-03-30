import os
import itertools
import logging
import argparse
import pickle

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from sklearn.linear_model import LinearRegression

from utils.data_utils import *

# ─────────────────────────────────────────────────────────────────────────────
# Logger — vừa print terminal, vừa ghi file
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('./outputs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./outputs/drift_run.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def get_model_path(saving_dir: str, client_id: int, task: int) -> str:
    """
    D:/FCL/checkpoints_client_task/client_0_task_4.pt
    """
    return os.path.join(saving_dir, f'client_{client_id}_task_{task}.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Các hàm đo — KHÔNG thay đổi
# ─────────────────────────────────────────────────────────────────────────────
def get_resnet18_blocks(_model):
    return {
        'block0': torch.nn.Sequential(_model.conv1, _model.bn1, _model.relu, _model.maxpool),
        'block1': _model.layer1,
        'block2': _model.layer2,
        'block3': _model.layer3,
        'block4': _model.layer4,
    }

def compute_width(_model, _target_layer: int):
    layers = [_model.layer1, _model.layer2, _model.layer3, _model.layer4]
    if not (0 <= _target_layer < len(layers)):
        raise IndexError(f"_target_layer phải trong [0, 3], nhận được: {_target_layer}")
    layer = layers[_target_layer]
    if isinstance(layer, torch.nn.Sequential):
        block = layer[-1]
        if isinstance(block, BasicBlock):
            return torch.norm(block.conv2.weight, p='fro').item()
    raise TypeError(f"Unexpected type: {type(layer)}")

def load_resnet18_from_checkpoint(ckpt_path: str) -> torch.nn.Module:
    """
    Chỉ load phần backbone (base.*), bỏ qua classifier (head/fc).
    """
    model  = resnet18(weights=None)  # fc mặc định 1000, không quan trọng vì không dùng
    raw_sd = torch.load(ckpt_path, map_location='cpu')

    # Strip prefix 'base.' → key chuẩn torchvision, bỏ qua 'head.*'
    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith('base.'):
            new_sd[k[len('base.'):]] = v   # base.conv1.weight → conv1.weight
        # bỏ qua head.* hoàn toàn

    # strict=False: bỏ qua fc (không load), bỏ qua num_batches_tracked
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    # Chỉ warn key thực sự quan trọng (bỏ qua fc và num_batches_tracked)
    real_missing = [
        k for k in missing
        if 'num_batches_tracked' not in k and not k.startswith('fc.')
    ]
    if real_missing:
        logger.warning(f'  [WARN] Missing backbone keys: {real_missing}')

    model.eval()
    return model

def compute_feature_resnet18(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    _model             : torchvision ResNet18 instance
    _model_task_index  : index task (dùng để đặt tên cache file)
    _dataset           : DataLoader hoặc list of (features, targets)
    _target_layer_index: 'block0'..'block4'
    seed               : seed value
    args               : argparse namespace (cần args.partition_options, args.backbone)
    """
    cache_dir = os.getenv("CACHE_DIR", "./cache")
    fname = (
        f'{cache_dir}/{args.partition_options}-{seed}-{args.backbone}'
        f'-task{_model_task_index}-layer{_target_layer_index}.pkl'
    )

    # ── Load cache nếu đã có ──────────────────────────────────────────────────
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    # ── Wrap ResNet18 layers thành blocks dict (giống code gốc) ──────────────
    blocks = get_resnet18_blocks(_model)

    _model.eval()
    outputs = []

    with torch.no_grad():
        for data in tqdm(_dataset, desc=f'Computing feature of M_{_model_task_index}^{_target_layer_index}...'):
            features, targets = data
            features = features.unsqueeze(0)  # (1, C, H, W)

            # ── Forward từng block, dừng lại ở target layer ──────────────────
            for block_name, operations in blocks.items():
                features = operations(features)
                if block_name == _target_layer_index:
                    break

            # Flatten & lưu
            outputs.append(
                torch.flatten(features, 1).squeeze().detach().cpu().numpy()
            )

    outputs = np.array(outputs)

    # ── Lưu cache ─────────────────────────────────────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(outputs, f)

    return outputs

def compute_eta(_feature_t):
    feature_dim = _feature_t.shape[-1]
    norms = np.linalg.norm(_feature_t, ord=2, axis=-1)
    mn, mx = norms.min(), norms.max()
    return mn, mx, mn / np.sqrt(feature_dim), mx / np.sqrt(feature_dim)

def compute_sigma(_feature_t, _feature_tprime):
    return np.linalg.norm(_feature_t - _feature_tprime, ord=2, axis=-1).max()

def compute_eps(_feature_t, _feature_tprime):
    reg = LinearRegression(fit_intercept=False).fit(_feature_t, _feature_tprime)
    transformed = reg.predict(_feature_tprime)
    return np.linalg.norm(_feature_t - transformed, ord=2, axis=-1).min()


# ─────────────────────────────────────────────────────────────────────────────
# Hàm đo chính
# ─────────────────────────────────────────────────────────────────────────────
def measure_all_representation_drift(args):
    output_file = f'./outputs/representation_drift-{args.partition_options}-{args.backbone}.csv'

    # Header CSV (chỉ ghi lần đầu)
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('client,block,t,tprime,'
                    'eta_min,eta_max,eta_min_norm,eta_max_norm,'
                    'sigma,eps,width_t,width_tprime\n')

    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    # [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    num_blocks = 5  # block0=stem, block1..4=layer1..4

    total = args.num_clients * len(task_pairs) * num_blocks
    done  = 0

    for client_id in range(args.num_clients):
        logger.info('=' * 60)
        logger.info(f'  CLIENT {client_id:>2} / {args.num_clients - 1}'
                    f'   ({len(task_pairs)} task-pairs × {num_blocks} blocks)')
        logger.info('=' * 60)
        args.client = client_id

        for (t, tprime) in task_pairs:
            logger.info(f'  ┌── Task pair ({t}, {tprime})')

            # Kiểm tra checkpoint tồn tại trước khi load
            ckpt_t  = get_model_path(args.saving_dir, client_id, t)
            ckpt_tp = get_model_path(args.saving_dir, client_id, tprime)
            for ckpt in [ckpt_t, ckpt_tp]:
                if not os.path.isfile(ckpt):
                    logger.error(f'  [MISSING] {ckpt}')
                    continue

            # Load model t
            model_t      = load_resnet18_from_checkpoint(ckpt_t)
            model_t.eval()
            logger.info(f'  │  model_t  ← {ckpt_t}')

            # Load model t'
            model_tprime = load_resnet18_from_checkpoint(ckpt_tp)
            model_tprime.eval()
            logger.info(f'  │  model_t\' ← {ckpt_tp}')

            # Load probe data (data của task t)
            test_data_t = read_client_data_FCL_cifar10(
                client_id, task=t,
                classes_per_task=args.cpt,
                count_labels=False, train=False
            )
            test_data_tprime = read_client_data_FCL_cifar10(
                client_id, task=tprime,
                classes_per_task=args.cpt,
                count_labels=False, train=False 
            )
            for k in range(num_blocks):
                target_layer = f'block{k}'
                try:
                    # Extract features
                    feat_t = compute_feature_resnet18(
                        model_t,      t,      test_data_t, target_layer, args.seed, args)
                    feat_tp = compute_feature_resnet18(
                        model_tprime, tprime, test_data_t, target_layer, args.seed, args)

                    # Width: block0 (stem) không có BasicBlock → nan
                    if k == 0:
                        width_t = width_tp = float('nan')
                    else:
                        width_t  = compute_width(model_t,      k - 1)
                        width_tp = compute_width(model_tprime, k - 1)

                    eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_t)
                    sigma = compute_sigma(feat_t, feat_tp)
                    eps   = compute_eps(feat_t, feat_tp)

                    done += 1
                    progress = f'[{done}/{total}]'

                    # ── Terminal + log file ───────────────────────────────────
                    logger.info(
                        f'  │  {progress} {target_layer} | '
                        f'σ={sigma:.4f}  ε={eps:.4f}  '
                        f'η_min={eta_min_n:.4f}  η_max={eta_max_n:.4f}  '
                        f'W_t={width_t:.4f}  W_t\'={width_tp:.4f}'
                    )

                    # ── WandB: log từng metric theo (client, task_pair, block) ─
                    if args.use_wandb:
                        wandb.log({
                            'client':        client_id,
                            'block':         k,
                            't':             t,
                            'tprime':        tprime,
                            'sigma':         sigma,
                            'eps':           eps,
                            'eta_min_norm':  eta_min_n,
                            'eta_max_norm':  eta_max_n,
                            'width_t':       width_t,
                            'width_tprime':  width_tp,
                            # key tổng hợp để filter trên wandb dashboard
                            'pair':          f'({t},{tprime})',
                            'client_block':  f'c{client_id}_b{k}',
                        })

                    # ── CSV ───────────────────────────────────────────────────
                    with open(output_file, 'a') as f:
                        row = [client_id, k, t, tprime,
                               eta_min, eta_max, eta_min_n, eta_max_n,
                               sigma, eps, width_t, width_tp]
                        f.write(','.join(map(str, row)) + '\n')

                except Exception as e:
                    logger.error(
                        f'  │  [SKIP] client={client_id} {target_layer} '
                        f't={t} t\'={tprime} | {e}'
                    )
                    continue

            logger.info(f'  └── Task pair ({t}, {tprime}) done')

    logger.info(f'\n✅  Hoàn thành! CSV → {output_file}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(args.seed)

    logger.info('-' * 60)
    logger.info(f'  partition_options  : {args.partition_options}')
    logger.info(f'  Backbone  : {args.backbone}')
    logger.info(f'  Clients   : {args.num_clients}')
    logger.info(f'  Tasks     : {args.num_tasks}')
    logger.info(f'  CPT       : {args.cpt}')
    logger.info(f'  Saving dir: {args.saving_dir}')
    logger.info('-' * 60)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f'{args.partition_options}-{args.backbone}-drift',
            config=vars(args),
        )

    if args.backbone == 'ResNet18':
        measure_all_representation_drift(args)
    else:
        raise ValueError(f'Backbone chưa hỗ trợ: {args.backbone}')

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Representation Drift Measurement')

    # Paths
    parser.add_argument('--saving_dir', type=str,
                        default=r'D:\FCL\checkpoints_client_task',
                        help='Thư mục chứa checkpoint (client_{c}_task_{t}.pt)')

    # Partition options / model
    parser.add_argument('--partition_options',    type=str, default='hetero')
    parser.add_argument('--backbone',    type=str, default='ResNet18')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_tasks',   type=int, default=5)
    parser.add_argument('--cpt',         type=int, default=2,
                        help='Classes per task')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--classes',     type=int, default=10,
                        help='Tổng số classes (dùng để khởi tạo ResNet18 đúng num_classes)')
    # WandB
    parser.add_argument('--use_wandb',type=bool, default=False, help='Có log lên WandB không')

    args = parser.parse_args()
    main(args)


# def compute_eta(_feature_t):
#     feature_dim = _feature_t.shape[-1]
#     outputs = np.linalg.norm(_feature_t, ord=2, axis=-1)
#     min_val = min(outputs)
#     max_val = max(outputs)
#     return min_val, max_val,min_val /np.sqrt(feature_dim), max_val / np.sqrt(feature_dim)

# def compute_sigma(_feature_t,_feature_tprime):
#     outputs = np.linalg.norm(_feature_t-_feature_tprime,ord=2, axis=-1)
#     return max(outputs)

# def compute_eps(_feature_t,_feature_tprime):
#     reg = LinearRegression(fit_intercept=False).fit(_feature_t, _feature_tprime)
    
#     _feature_tprime_transformed = reg.predict(_feature_tprime)

#     eps = np.linalg.norm(_feature_t- _feature_tprime_transformed, ord=2, axis=-1)
#     eps = np.max(eps)
#     return eps