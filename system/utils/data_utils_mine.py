# system/utils/data_utils_mine.py
# -*- coding: utf-8 -*-
"""
HeteroScope-style partitioning (Section 3), minimal and self-contained.
- No parser or env dependency. All knobs are defined below.
- Same public API as the repo's data_utils: read_client_data_FCL_cifar10/100/imagenet1k.
- Defaults mirror the codebase partition (use all images of selected classes).
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

# ====== HeteroScope knobs (define them here) ======
HETERO_SEED: int   = 42          # global seed
HETERO_ALPHA: float = 10       # label-skew Dirichlet alpha (used only if PER_TASK_SAMPLES > 0)
HETERO_PSI: float   = 1       # order disorder ψ ∈ [0,1] (adjacent-swap probability)
HETERO_OMEGA: float = 0.5       # task overlap ω ∈ [0,1] (adjacent Jaccard approx)
HETERO_RHO: float   = 0.5       # recurrence ρ ∈ [0,1] (draw from any earlier tasks)
HETERO_USE_ALL_PER_CLASS: bool = True  # if True, use all images per selected class (codebase-like)
HETERO_PER_TASK_SAMPLES: int   = 0     # if >0 and USE_ALL=False, sample this many per (client,task)
HETERO_NUM_TASKS: int | None   = None  # if None, T := ceil(K / cpt). Set an int to override.

# ====== helpers ======
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def _dataset_root_and_K(dataset: str) -> Tuple[Path, int]:
    ds = dataset.lower()
    if ds == "cifar10":  return Path("dataset/cifar10-classes"), 10
    if ds == "cifar100": return Path("dataset/cifar100-classes"), 100
    # assume ImageNet-1K folders are prebuilt as .npy per class (0..999)
    return Path("dataset/imagenet1k-classes"), 1000

def _glob_npy_for_class(root: Path, cls: int) -> Path:
    p = root / f"{cls}.npy"
    if p.exists(): return p
    cand = sorted(root.glob(f"{cls}_*.npy"))
    if not cand: raise FileNotFoundError(f"Missing .npy for class {cls} under {root}")
    return cand[0]

def _load_class_np(root: Path, cls: int) -> np.ndarray:
    arr = np.load(_glob_npy_for_class(root, cls), allow_pickle=True)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    elif arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0,2,3,1))
    elif arr.ndim == 3:
        if   arr.shape[-1] == 3: arr = arr[None, ...]
        elif arr.shape[0]  == 3: arr = np.transpose(arr, (1,2,0))[None, ...]
        else: raise ValueError(f"Unexpected shape for class {cls}: {arr.shape}")
    else:
        raise ValueError(f"Unexpected shape for class {cls}: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr  # (N, H, W, 3) uint8

def _to_tensordataset(images_hwc_uint8: np.ndarray, labels_int: np.ndarray) -> TensorDataset:
    x = torch.tensor(np.transpose(images_hwc_uint8, (0,3,1,2)), dtype=torch.float32) / 255.0
    y = torch.tensor(labels_int.astype(np.int64), dtype=torch.long)
    return TensorDataset(x, y)

# ====== core generator: Y_t and π_c ======
_Y_CACHE: Dict[Tuple[int,int,int,float,float,int], List[List[int]]] = {}  # key: (K,cpt,T,omega,rho,seed)
def _build_Y_sets(K: int, cpt: int, T: int, omega: float, rho: float, seed: int) -> List[List[int]]:
    key = (K, cpt, T, omega, rho, seed)
    if key in _Y_CACHE: return _Y_CACHE[key]
    r = _rng(seed + 1)
    pool = list(range(K))
    r.shuffle(pool)
    Y: List[List[int]] = []
    for t in range(T):
        chosen: List[int] = []
        if t > 0 and omega > 0.0:
            k_overlap = min(int(round(omega * cpt)), len(Y[-1]))
            if k_overlap > 0:
                chosen.extend(r.choice(Y[-1], size=k_overlap, replace=False).tolist())
                chosen = list(dict.fromkeys(chosen))
        if t > 0 and rho > 0.0:
            prev = sorted({k for yy in Y for k in yy})
            k_rec = int(round(rho * cpt))
            prev_cands = [k for k in prev if k not in chosen]
            if prev_cands and k_rec > 0:
                chosen.extend(r.choice(prev_cands, size=min(k_rec, len(prev_cands)), replace=False).tolist())
        need = cpt - len(chosen)
        take = min(need, len(pool))
        if take > 0:
            picks = pool[:take]
            chosen.extend(picks)
            del pool[:take]
        while len(chosen) < cpt:  # if pool exhausted, fill from any not yet in 'chosen'
            rest = [k for k in range(K) if k not in chosen]
            chosen.append(r.choice(rest))
        Y.append(list(dict.fromkeys(chosen))[:cpt])
    _Y_CACHE[key] = Y
    return Y

_PI_CACHE: Dict[Tuple[int,int,float,int], List[int]] = {}  # key: (client_id, T, psi, seed)
def _pi_for_client(client_id: int, T: int, psi: float, seed: int) -> List[int]:
    key = (client_id, T, float(psi), seed)
    if key in _PI_CACHE:
        return _PI_CACHE[key]

    rng = _rng(seed + 123456 + client_id)
    order = list(range(T))

    if psi <= 0.0 or T <= 1:
        _PI_CACHE[key] = order
        return order

    if psi >= 1.0:
        # maximal disorder: independent random permutation per client
        order = rng.permutation(T).tolist()
        _PI_CACHE[key] = order
        return order

    # partial disorder: choose a random number of disjoint adjacent swaps
    # m ~ Binomial(T-1, psi), at least 1 when psi>0
    m = max(1, rng.binomial(T - 1, psi))

    # pick m distinct positions to swap (0..T-2), then apply in random order
    positions = rng.choice(T - 1, size=m, replace=False)
    for idx in rng.permutation(positions):
        order[idx], order[idx + 1] = order[idx + 1], order[idx]

    _PI_CACHE[key] = order
    return order

# ====== sampling within a (client, task) ======
def _alloc_counts_for_labels(labels: List[int], per_task_samples: int, alpha: float, seed: int) -> Dict[int,int]:
    if per_task_samples <= 0:  # unused when USE_ALL=True
        return {k: 0 for k in labels}
    r = _rng(seed)
    a = np.ones(len(labels)) * max(1e-6, float(alpha))
    w = r.dirichlet(a)
    c = np.floor(w * per_task_samples).astype(int)
    rem = per_task_samples - int(c.sum())
    for i in range(rem):
        c[i % len(c)] += 1
    return {labels[i]: int(c[i]) for i in range(len(labels))}

# ====== build dataset for (client, task) ======
def _assemble(dataset: str, root: Path, K: int,
              client_id: int, task: int, cpt: int, count_labels: bool) -> Tuple[TensorDataset, Dict[str,Any]]:
    T = HETERO_NUM_TASKS if (HETERO_NUM_TASKS is not None) else math.ceil(K / max(1, cpt))
    Y_sets = _build_Y_sets(K, cpt, T, HETERO_OMEGA, HETERO_RHO, HETERO_SEED)
    pi_c   = _pi_for_client(client_id, T, HETERO_PSI, HETERO_SEED)
    if task < 0 or task >= T: raise ValueError(f"task {task} out of range [0,{T-1}]")
    t_true = pi_c[task]
    labels = list(Y_sets[t_true])

    xs, ys = [], []
    counts: Dict[int,int] = {}
    # decide per-class counts if sampling mode is enabled
    target = None
    if not HETERO_USE_ALL_PER_CLASS and HETERO_PER_TASK_SAMPLES > 0:
        target = _alloc_counts_for_labels(labels, HETERO_PER_TASK_SAMPLES, HETERO_ALPHA,
                                          seed=HETERO_SEED + 97 * (client_id + 31 * (task + 1)))

    for k in labels:
        arr = _load_class_np(root, k)   # (N, H, W, 3)
        n = int(arr.shape[0])
        if target is None:
            take_idx = np.arange(n)
        else:
            want = min(int(target.get(k, 0)), n)
            if want <= 0:
                counts[k] = 0
                continue
            r = _rng(HETERO_SEED + 313 * (client_id + 17 * (task + 1) + 7 * k))
            take_idx = r.choice(n, size=want, replace=False)
        xs.append(arr[take_idx])
        ys.append(np.full((len(take_idx),), fill_value=int(k), dtype=np.int64))
        counts[k] = int(len(take_idx))

    if xs:
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
    else:
        # empty edge case
        X = np.zeros((0, 3, 32, 32), dtype=np.float32)
        Y = np.zeros((0,), dtype=np.int64)
        info = {
            "labels": list(labels),            # <-- add 'labels' (required by server)
            "classes": list(labels),           # <-- keep 'classes' for compatibility
            "counts": {},                      # optional
            "label_count": {},                 # optional alias
            "global_task_index": int(t_true),
            "task_order_pi": list(pi_c),
        }
        return TensorDataset(torch.tensor(X), torch.tensor(Y)), info

    ds = _to_tensordataset(X, Y)

    # Always provide 'labels' (server expects it) and 'classes' (back-compat)
    info: Dict[str, Any] = {
        "labels": list(labels),                  # <-- required by server
        "classes": list(labels),                 # <-- compatibility
        "global_task_index": int(t_true),
        "task_order_pi": list(pi_c),
    }

    # Counts: return both 'counts' and 'label_count'
    if count_labels:
        if target is None:
            uniq, cnt = np.unique(Y, return_counts=True)
            cnt_map = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
        else:
            cnt_map = counts
        info["counts"] = cnt_map
        info["label_count"] = dict(cnt_map)      # alias

    return ds, info

# ====== public API (match codebase names) ======
def read_client_data_FCL_cifar10(client_id: int, task: int, classes_per_task: int, count_labels: bool=False):
    root, K = _dataset_root_and_K("CIFAR10")
    return _assemble("CIFAR10", root, K, client_id, task, classes_per_task, count_labels)

def read_client_data_FCL_cifar100(client_id: int, task: int, classes_per_task: int, count_labels: bool=False):
    root, K = _dataset_root_and_K("CIFAR100")
    return _assemble("CIFAR100", root, K, client_id, task, classes_per_task, count_labels)

def read_client_data_FCL_imagenet1k(client_id: int, task: int, classes_per_task: int, count_labels: bool=False):
    root, K = _dataset_root_and_K("IMAGENET1K")
    return _assemble("IMAGENET1K", root, K, client_id, task, classes_per_task, count_labels)
