# Federated Continual Learning Benchmark

This branch provides a clean Federated Continual Learning (FCL) benchmark with ready-to-run baselines, reproducible data preparation, colored training progress, and two data-partition modes, Tuan and Lucaz.

## 1, Installation

### 1.1, Environment

```bash
git clone https://github.com/chickbong221/FCL.git
cd FCL
git checkout Lucaz-FCL-PFLlib

# Python 3.9+ recommended
python -m venv .env
source .env/bin/activate

pip install -r requirements.txt
# optional, for pretty console tables and progress bars
pip install rich
# optional, for Weights & Biases logging
pip install wandb
```

### 1.2, Repo layout, important paths

```
FCL/
  system/
    main.py                      , entry point
    flcore/
      servers/                   , FedAvg, etc.
      clients/                   , client logic
      trainmodel/                , models
    utils/
      data_utils.py              , CURRENT partition (codebase split)
      data_utils_mine.py         , MINE partition (HeteroScope, all knobs in this file)
      partition_viz.py           , optional, partition visualization (if you added it)
  dataset/
    cifar10_npy.py               , builds per-class .npy
    cifar100_npy.py              , builds per-class .npy
    cifar10-classes/             , 0.npy … 9.npy
    cifar100-classes/            , 0.npy … 99.npy
    imagenet1k-classes/          , 0.npy … 999.npy (if you prepare it)
  hparams/
    FedAvg.json                  , example config, see below
```

## 2, Data preparation

### 2.1, CIFAR-10 and CIFAR-100

```bash
# CIFAR-10
cd dataset
python cifar10_npy.py        # creates dataset/cifar10-classes/{0..9}.npy

# CIFAR-100
python cifar100_npy.py       # creates dataset/cifar100-classes/{0..99}.npy
cd ..
```

### 2.2, ImageNet-1K (optional)

Place per-class `.npy` files under `dataset/imagenet1k-classes/0.npy ... 999.npy` following the same format as CIFAR. The loader expects HWC uint8 or CHW float that can be standardized.

## 3, Configuration

A minimal `hparams/FedAvg.json` example you can copy and edit:

```json
{
  "optimizer": "sgd",
  "device": "cuda",
  "device_id": "0",
  "dataset": "CIFAR100",
  "num_classes": 100,
  "model": "CNN",
  "model_str": "CNN",
  "batch_size": 128,
  "local_learning_rate": 0.01,
  "global_rounds": 100,
  "local_epochs": 5,
  "algorithm": "FedAvg",
  "join_ratio": 1.0,
  "random_join_ratio": false,
  "num_clients": 10,
  "eval_gap": 1,
  "out_folder": "out",
  "client_drop_rate": 0.0,
  "time_threthold": 10000
}
```

Notes, set `device="cuda"` only if you have a GPU, otherwise use `"cpu"`. The server keeps the global model on CPU by default, and each client moves to GPU only during its local `train()`.

## 4, Running one experiment

### 4.1, The quick run

```bash
# codebase/current partition, CIFAR-100, FedAvg
python system/main.py --cfp ./hparams/FedAvg.json --partition_options current
```

Useful optional flags:

* `--wandb True`, log to Weights & Biases
* `--log True`, print key metrics
* `--offlog True`, save CSVs under `out/<exp>/...`
* `--cpt <int>`, classes per task, for example `--cpt 10`
* `--nt <int>`, number of tasks (if not set, defaults by dataset or computed)

Colored progress and per-round tables appear automatically when `rich` is installed.

### 4.2, Expected outputs

* Console shows round headers, selected clients, per-client loss/acc/time/samples, and round summaries.
* Files saved under `out/<DATASET>_<ALGO>_<MODEL>_...`
* If you turned on offline logging, per-client and global CSVs are created in the run folder.

## 5, Data partition modes

You can pick between the **current** (codebase) split and **mine** (HeteroScope) split with one flag:

```bash
--partition_options current    , use utils/data_utils.py
--partition_options mine       , use utils/data_utils_mine.py
```

### 5.1, Current partition, how it works

* There is a precomputed **class order** per client, e.g., arrays under `dataset/class_order/*.npy` (depending on dataset).
* For a given `client` and `task`, the loader slices `cpt` consecutive class IDs from that client’s class order and builds the task dataset from per-class `.npy`.
* All images of the selected classes are used by default.

Example, CIFAR-10 with `cpt=2`, five clients, ten classes. Suppose a client’s class order is `[6, 5, 3, 2, 9, 7, 8, 1, 4, 0]`. Then:

* Task 0 → classes `[6, 5]`
* Task 1 → `[3, 2]`
* Task 2 → `[9, 7]`
* Task 3 → `[8, 1]`
* Task 4 → `[4, 0]`

Clients usually have different class orders, so Task 0 class pairs differ across clients.

### 5.2, Mine partition (HeteroScope), how it works

All knobs are defined at the top of `utils/data_utils_mine.py` and take effect without touching the parser:

```python
# utils/data_utils_mine.py
HETERO_SEED  = 42
HETERO_ALPHA = 10.0       # only used if sub-sampling is enabled
HETERO_PSI   = 1.0        # order disorder in [0,1]
HETERO_OMEGA = 0.0        # task overlap in [0,1]
HETERO_RHO   = 0.0        # recurrence in [0,1]
HETERO_USE_ALL_PER_CLASS = True
HETERO_PER_TASK_SAMPLES  = 0
HETERO_NUM_TASKS = None   # if None, T = ceil(K/cpt)
```

What each does:

* `HETERO_PSI`, order disorder, controls how each client **permutes** the global task order
  0 keeps all clients aligned, 1 produces a **random permutation per client**
* `HETERO_OMEGA`, task overlap, makes adjacent global tasks share about `⌊ω·cpt⌋` labels
* `HETERO_RHO`, recurrence, reuses about `⌊ρ·cpt⌋` labels from earlier tasks (not just the previous one)
* `HETERO_USE_ALL_PER_CLASS=True`, the task uses **all samples** of its labels
  If you want budgeted sampling instead, set `HETERO_USE_ALL_PER_CLASS=False` and `HETERO_PER_TASK_SAMPLES > 0`
* `HETERO_ALPHA`, Dirichlet α, only relevant when sub-sampling is enabled, smaller α gives heavier skew
* `HETERO_NUM_TASKS`, override total tasks if you want, else it’s computed as `ceil(K/cpt)`

Example, CIFAR-10 with `cpt=2`, `HETERO_PSI=1.0`, `HETERO_OMEGA=0`, `HETERO_RHO=0`:

* The generator builds global task label sets `Y₀, Y₁, …, Y₄`, each with two labels
* Each client draws a **random permutation** πᶜ of `[0,1,2,3,4]`
* Client c’s Task 0 uses the labels from `Y_{πᶜ(0)}`, so Task 0 labels differ across clients
* With `HETERO_USE_ALL_PER_CLASS=True`, each chosen label contributes all of its samples

If you want visible overlap between adjacent tasks, set `HETERO_OMEGA` to something like `0.3`, then consecutive global tasks share one label when `cpt=2`.

### 5.3, Switching between modes at runtime

No code changes are needed once your server’s `set_clients(...)` imports by `partition_options`. Just run:

```bash
# current
python system/main.py --cfp ./hparams/FedAvg.json --partition_options current

# mine
python system/main.py --cfp ./hparams/FedAvg.json --partition_options mine
```

### 5.4, Quick sanity print (optional)

To verify the split before training, you can print Task 0 labels for the first few clients. Drop this right after `server.set_clients(...)` in your server code, or temporarily in `main.py` after server creation:

```python
print("\n[Partition sanity] Task 0 labels per client:")
for cid, u in enumerate(server.clients[:10]):
    print(f"  client {cid}: {u.task_dict.get(0)}")
```

## 6, Training progress UI

The server uses a colored progress bar and tables when `rich` is installed. You’ll see, per round, a header with the current task and selected clients, a per-client table with loss, accuracy, time and samples, and a summary panel. If `rich` is not installed, it falls back to simple prints.

Install once:

```bash
pip install rich
```

## 7, Tips and troubleshooting

* If **mine** looks identical to **current**, check these quickly
  1, ensure you actually run with `--partition_options mine`,
  2, confirm your `utils/data_utils_mine.py` is being imported by the server during `set_clients`,
  3, set `HETERO_PSI > 0` to misalign task order across clients,
  4, set `HETERO_OMEGA > 0` or `HETERO_RHO > 0` to make global tasks overlap or recur.
* CUDA out-of-memory during startup usually means the global model was moved to GPU at construction. Keep the global model on CPU, and move only the selected client to GPU during `train()`.
* If evaluation raises a device mismatch, move the batch to the **model’s device** inside `test_metrics` and `train_metrics`:

  ```python
  dev = next(self.model.parameters()).device
  x = x.to(dev)
  y = y.to(dev)
  ```
* Check dataset paths
  `dataset/cifar10-classes/0.npy` … `dataset/cifar100-classes/0.npy` … exist and are readable.

## 8, Reproducibility

* The codebase split is driven by fixed class-order arrays and will be deterministic given the same random seeds inside the training loop.
* The HeteroScope split uses `HETERO_SEED` at the top of `utils/data_utils_mine.py`. Change this once per experiment to re-draw the global task label sets and client permutations.

## 9, Minimal end-to-end recipe

```bash
# 1, environment
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
pip install rich

# 2, data
cd dataset
python cifar10_npy.py
python cifar100_npy.py
cd ..

# 3, config
# edit hparams/FedAvg.json with your dataset, num_clients, cpt, etc.

# 4a, run with codebase partition
python system/main.py --cfp ./hparams/FedAvg.json --partition_options current --log True --offlog True

# 4b, run with mine partition, knobs set in utils/data_utils_mine.py
python system/main.py --cfp ./hparams/FedAvg.json --partition_options mine --log True --offlog True
```