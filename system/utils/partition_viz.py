# system/utils/partition_viz.py
# -*- coding: utf-8 -*-
"""
Partition visualization & reporting for FCL runs.
- Introspects the server's clients to recover the Task -> Class mapping they've been assigned.
- Prints a readable table before training starts.
- Saves heatmaps (presence & counts), CSV, TXT and JSON into ./figures (or custom fig_dir).

Usage in system/main.py (before server.train()):
    from system.utils.partition_viz import visualize_and_print_partition
    visualize_and_print_partition(server, args, fig_dir="figures")

This file is read-only with respect to training; it won’t modify datasets or clients.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

# Headless-safe matplotlib import
import matplotlib
matplotlib.use("Agg")  # no GUI required
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch  # only used for label extraction if tensors are returned
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ---------- Utilities ----------

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_int(x: Any) -> Optional[int]:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor) and x.numel() == 1:
        return int(x.item())
    return None

def _dataset_label_array(ds: Any) -> Optional[np.ndarray]:
    """
    Try common ways to get labels from a dataset object without iterating:
    - ds.targets / ds.labels / ds.y / ds.ys
    - ds.tensors (TensorDataset) -> second tensor
    Returns np.ndarray of ints if found, else None.
    """
    for attr in ("targets", "labels", "y", "ys"):
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            try:
                arr = np.array(arr)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                return arr.astype(int)
            except Exception:
                pass
    if hasattr(ds, "tensors"):
        try:
            tens = ds.tensors
            if isinstance(tens, (tuple, list)) and len(tens) >= 2:
                y = tens[1]
                if TORCH_AVAILABLE and isinstance(y, torch.Tensor):
                    return y.detach().cpu().numpy().astype(int)
        except Exception:
            pass
    return None

def _dataset_unique_labels(ds: Any, scan_cap: int = 50000) -> Tuple[List[int], Optional[Dict[int, int]]]:
    """
    Return (sorted_unique_labels, optional_counts) for a dataset.
    Tries zero-copy attributes first; otherwise iterates items (capped).
    Counts dict may be None if we can't cheaply compute counts.
    """
    arr = _dataset_label_array(ds)
    if arr is not None:
        # Fast path: we have all labels
        uniq, counts = np.unique(arr.astype(int), return_counts=True)
        return uniq.tolist(), {int(k): int(v) for k, v in zip(uniq, counts)}

    # Slow path: iterate items
    label_list: List[int] = []
    n = len(ds) if hasattr(ds, "__len__") else scan_cap
    n = min(n, scan_cap)
    for i in range(n):
        try:
            item = ds[i]
        except Exception:
            break
        lbl = None
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            lbl = _to_int(item[1])
        else:
            lbl = _to_int(item)
        if lbl is not None:
            label_list.append(lbl)
    if not label_list:
        return [], None
    uniq, counts = np.unique(np.array(label_list, dtype=int), return_counts=True)
    return uniq.tolist(), {int(k): int(v) for k, v in zip(uniq, counts)}

def _looks_like_dataloader(x: Any) -> bool:
    return hasattr(x, "dataset") and hasattr(x, "__iter__")

def _looks_like_dataset(x: Any) -> bool:
    return hasattr(x, "__getitem__") and hasattr(x, "__len__")

def _extract_task_datasets_from_client(client: Any) -> List[Any]:
    """
    Heuristics to collect per-task datasets (train side) from a client object.
    Checks common attribute names first; otherwise scans attributes for lists/dicts of datasets or dataloaders.
    Returns a list ordered by task index (if dicts, sorted by key).
    """
    # 1) Known attribute names (ordered)
    candidate_attr_names = [
        "task_train_loaders", "train_loaders", "trainloaders",
        "task_train_datasets", "train_datasets", "train_sets",
        "train_data", "train_data_local_dict"
    ]
    for name in candidate_attr_names:
        if hasattr(client, name):
            obj = getattr(client, name)
            # dict -> sort by key
            if isinstance(obj, dict):
                items = [obj[k] for k in sorted(obj)]
            elif isinstance(obj, (list, tuple)):
                items = list(obj)
            else:
                items = [obj]
            # map dataloaders -> datasets
            ds_list = []
            for it in items:
                if _looks_like_dataloader(it):
                    ds_list.append(getattr(it, "dataset"))
                elif _looks_like_dataset(it):
                    ds_list.append(it)
            if ds_list:
                return ds_list

    # 2) Fallback: scan all attributes for plausible lists/dicts
    ds_list = []
    for k, v in vars(client).items():
        if isinstance(v, dict):
            vals = list(v.values())
        elif isinstance(v, (list, tuple)):
            vals = list(v)
        else:
            vals = [v]
        tmp = []
        for it in vals:
            if _looks_like_dataloader(it):
                tmp.append(getattr(it, "dataset"))
            elif _looks_like_dataset(it):
                tmp.append(it)
        # Heuristic: if it looks like multiple datasets and len > 1, consider it tasks
        if len(tmp) > 1:
            ds_list = tmp
            break
    return ds_list

def _dataset_name_map(dataset_name: str) -> Dict[int, str]:
    name = (dataset_name or "").lower()
    if "cifar10" in name:
        return {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
    # CIFAR-100 / ImageNet: names are long; default to "class_{id}"
    return {}

def _fmt_label_list(ids: List[int], name_map: Dict[int, str]) -> str:
    if not ids:
        return "[]"
    pieces = []
    for k in ids:
        nm = name_map.get(int(k))
        pieces.append(f"{k}" if nm is None else f"{k}:{nm}")
    return "[" + ", ".join(pieces) + "]"


# ---------- Main API ----------

def visualize_and_print_partition(server: Any, args: Any, fig_dir: str = "figures") -> Dict[str, Any]:
    """
    Build a full partition report from the actual datasets held by each client.
    Prints a table and saves figures & files to fig_dir.
    Returns the report dict for further use.
    """
    out_dir = _ensure_dir(fig_dir)
    num_clients = len(getattr(server, "clients", []))
    if num_clients == 0:
        print("[PartitionViz] No clients found on server; skipping.")
        return {}

    dataset_name = getattr(args, "dataset", "")
    name_map = _dataset_name_map(dataset_name)
    num_classes = getattr(args, "num_classes", None)

    # 1) Collect per-(client, task) label info
    # report["clients"][cid]["tasks"][tid] = {"labels": [...], "counts": {...}}
    report: Dict[str, Any] = {"dataset": dataset_name, "num_clients": num_clients, "clients": {}}
    max_tasks = 0
    global_class_ids: set = set()

    for cid, client in enumerate(server.clients):
        ds_list = _extract_task_datasets_from_client(client)
        if not ds_list:
            print(f"[PartitionViz] WARNING: Could not locate task datasets for client {cid}.")
        client_entry = {"tasks": {}}
        for tid, ds in enumerate(ds_list):
            labels, counts = _dataset_unique_labels(ds)
            labels = sorted(set(int(k) for k in labels))
            for k in labels:
                global_class_ids.add(int(k))
            client_entry["tasks"][tid] = {"labels": labels, "counts": counts}
            max_tasks = max(max_tasks, tid + 1)
        report["clients"][str(cid)] = client_entry

    if num_classes is None:
        # infer from union of labels we saw
        num_classes = max(global_class_ids) + 1 if global_class_ids else 0
    report["num_classes_inferred"] = int(num_classes)

    # 2) Print a readable table (and write to TXT)
    lines = []
    header = f"=== Data Partition Plan (dataset={dataset_name}, clients={num_clients}, classes={num_classes}) ==="
    print(header)
    lines.append(header)
    for cid in range(num_clients):
        cdict = report["clients"].get(str(cid), {})
        tasks = cdict.get("tasks", {})
        print(f"Client {cid}:")
        lines.append(f"Client {cid}:")
        if not tasks:
            print("  (no tasks discovered)")
            lines.append("  (no tasks discovered)")
            continue
        for tid in sorted(tasks):
            labels = tasks[tid]["labels"]
            pretty = _fmt_label_list(labels, name_map)
            print(f"  Task {tid}: |classes|={len(labels)}  {pretty}")
            lines.append(f"  Task {tid}: |classes|={len(labels)}  {pretty}")
    (out_dir / "partition_table.txt").write_text("\n".join(lines), encoding="utf-8")

    # 3) Write CSV (long format)
    # columns: client, task, class_id, count(optional)
    csv_lines = ["client,task,class_id,count"]
    for cid in range(num_clients):
        tasks = report["clients"].get(str(cid), {}).get("tasks", {})
        for tid, info in tasks.items():
            labels = info["labels"]
            counts = info.get("counts") or {}
            for k in labels:
                cnt = counts.get(int(k), "")
                csv_lines.append(f"{cid},{tid},{int(k)},{cnt}")
    (out_dir / "partition_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # 4) Save JSON
    with open(out_dir / "partition_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 5) Heatmaps
    # Build matrices with shape [num_clients * max_tasks, num_classes]
    if num_classes and max_tasks:
        H = num_clients * max_tasks
        presence = np.zeros((H, num_classes), dtype=np.float32)
        counts   = np.zeros((H, num_classes), dtype=np.float32)
        ytick_labels = []
        row = 0
        for cid in range(num_clients):
            tasks = report["clients"].get(str(cid), {}).get("tasks", {})
            for tid in range(max_tasks):
                info = tasks.get(tid)
                ytick_labels.append(f"C{cid}-T{tid}")
                if info:
                    for k in info["labels"]:
                        presence[row, int(k)] = 1.0
                    if info.get("counts"):
                        for k, v in info["counts"].items():
                            if 0 <= int(k) < num_classes:
                                counts[row, int(k)] = float(v)
                row += 1

        def _save_heatmap(M: np.ndarray, title: str, path: Path, vmin: float, vmax: float):
            plt.figure(figsize=(min(18, 1 + num_classes * 0.25), min(2 + H * 0.25, 16)))
            plt.imshow(M, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="Greys")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.yticks(np.arange(H), ytick_labels, fontsize=6)
            # Only show some x-ticks if many classes
            if num_classes <= 50:
                plt.xticks(np.arange(num_classes), [str(i) for i in range(num_classes)], rotation=90, fontsize=6)
            else:
                step = max(1, num_classes // 50)
                ticks = np.arange(0, num_classes, step)
                plt.xticks(ticks, [str(i) for i in ticks], rotation=90, fontsize=6)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path, dpi=250)
            plt.close()

        _save_heatmap(presence, "Class presence by (client, task)", out_dir / "partition_presence_heatmap.png", 0.0, 1.0)

        if counts.max() > 0:
            # Normalize per-row to highlight relative class density
            norm_counts = counts / (counts.max(axis=1, keepdims=True) + 1e-8)
            _save_heatmap(norm_counts, "Per-(client,task) class counts (row-normalized)", out_dir / "partition_counts_heatmap.png", 0.0, 1.0)
        else:
            # If counts unavailable, still save a placeholder copy of presence as counts map
            _save_heatmap(presence, "Per-(client,task) class counts (binary only)", out_dir / "partition_counts_heatmap.png", 0.0, 1.0)

    print(f"[PartitionViz] Wrote partition artifacts to: {out_dir.resolve()}")
    return report
