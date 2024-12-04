import logging
from dataclasses import dataclass
import numpy as np
import typing as t
from torch.utils.data import Dataset, Subset

from config import DatasetConfig, DatasetModelSpec, DATA_PATH
from fldatasets.torchvision import get_cifar10, get_cifar100, get_fast_cifar10, get_emnist
from fldatasets.medmnist import fetch_medmnist_dataset

# from fldatasets.flamby import (
#     fetch_flamby_pooled,
#     fetch_flamby_federated,
#     get_flamby_model_spec,
# )

@dataclass
class DatasetPair:
    train: Subset
    test: Subset


def load_raw_dataset(cfg: DatasetConfig) -> tuple[DatasetPair, DatasetModelSpec]:
    """Fetch and split requested datasets.

    Args:
        cfg: DatasetConfig"

    Returns: raw_train, raw_test, model_spec
    """

    match cfg.name:
        case "cifar10":
            train, test = get_cifar10(cfg)
            model_spec = DatasetModelSpec(num_classes=10, in_channels=3)
        case "fast_cifar10":
            train, test = get_fast_cifar10(cfg)
            model_spec = DatasetModelSpec(num_classes=10, in_channels=3)
        case "cifar100":
            train, test = get_cifar100(cfg)
            model_spec = DatasetModelSpec(num_classes=100, in_channels=3)
        case "medmnist":
            train, test, model_spec = fetch_medmnist_dataset(cfg.name, DATA_PATH)
        case "emnist":
            train, test = get_emnist(cfg)
            model_spec = DatasetModelSpec(num_classes=62, in_channels=1)
        case "fedisic":
            from fldatasets.flamby import fetch_flamby_federated, fetch_flamby_pooled, get_flamby_model_spec

            train, test = fetch_flamby_pooled(cfg.name, DATA_PATH)
            model_spec = get_flamby_model_spec(cfg.name, DATA_PATH)
        case _:
            raise NotImplementedError()

    if cfg.subsample_fraction < 1.0:
        get_subset = lambda set, fraction: Subset(set, np.random.randint(0, len(set) - 1, int(fraction * len(set))))  # type: ignore

        train = get_subset(train, cfg.subsample_fraction)
        test = get_subset(test, cfg.subsample_fraction)
    else:
        train = Subset(train, np.arange(len(train))) # type: ignore
        test = Subset(test, np.arange(len(test))) # type: ignore

    return DatasetPair(train=train, test=test), model_spec


def subsample_dataset(dataset: Dataset, fraction: float):
    return Subset(dataset, np.random.randint(0, len(dataset) - 1, int(fraction * len(dataset))))  # type: ignore
