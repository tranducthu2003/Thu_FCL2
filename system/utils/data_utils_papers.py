"""
Placeholder for 'most_papers' partition scheme.

Typical choices in many papers:
  - Dirichlet label skew: draw class proportions per client from Dir(α).
  - Shard-based/pathological: each client gets K class shards only.
  - IID: uniform random split.

IMPORTANT: Keep function signatures identical to data_utils.read_client_data_FCL_*.
Return formats must match your current code (train_data, label_info).
"""

from typing import Any, Dict, Tuple

# If you need shared state across calls (e.g., a fixed split plan), define a module-level cache:
_SPLIT_PLAN_CACHE = {}   # e.g., keyed by (dataset_name, num_clients, seed, alpha, ...)

def read_client_data_FCL_cifar10(
    client_id: int,
    task: int,
    classes_per_task: int,
    count_labels: bool = False,
) -> Tuple[Any, Dict]:
    """
    TODO: Implement your 'most_papers' partition for CIFAR-10.

    Suggested approach (Dirichlet α example):
      1) On first call, build a plan that assigns, for every client, a per-task class set or data indices
         by sampling class proportions from a Dirichlet(α) and then mapping into tasks of size `classes_per_task`.
      2) Put that plan into _SPLIT_PLAN_CACHE.
      3) Here, return the slice for (client_id, task).

    Must return: (train_data, label_info_dict)
    """
    raise NotImplementedError("Implement CIFAR-10 'most_papers' partitioning here.")


def read_client_data_FCL_cifar100(
    client_id: int,
    task: int,
    classes_per_task: int,
    count_labels: bool = False,
) -> Tuple[Any, Dict]:
    """TODO: Implement 'most_papers' partition for CIFAR-100."""
    raise NotImplementedError("Implement CIFAR-100 'most_papers' partitioning here.")


def read_client_data_FCL_imagenet1k(
    client_id: int,
    task: int,
    classes_per_task: int,
    count_labels: bool = False,
) -> Tuple[Any, Dict]:
    """TODO: Implement 'most_papers' partition for ImageNet-1K."""
    raise NotImplementedError("Implement ImageNet-1K 'most_papers' partitioning here.")
