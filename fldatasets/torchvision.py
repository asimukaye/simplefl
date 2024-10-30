import torch
import logging
import torchvision
from torch.utils.data import Dataset, Subset
import numpy as np

from torchvision import transforms
from torchvision.datasets import CIFAR10, VisionDataset, CIFAR100

# dataset wrapper module
from config import DatasetConfig, DATA_PATH


class VisionClfDataset(Subset):
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.targets = self.dataset.targets  # type: ignore
        self.indices = np.arange(len(self.dataset))  # type: ignore
        self.class_to_idx = dataset.class_to_idx

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, targets

    def __len__(self):
        return len(self.dataset)  # type: ignore


def get_cifar10(cfg: DatasetConfig):
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train = CIFAR10(
        root=DATA_PATH, train=True, download=True, transform=train_transforms
    )
    test = CIFAR10(
        root=DATA_PATH, train=False, download=True, transform=test_transforms
    )
    return VisionClfDataset(train, "CIFAR10"), VisionClfDataset(test, "CIFAR10")


# RFFL version
class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        from torch import from_numpy

        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # https://github.com/kuangliu/pytorch-cifar/issues/16
        # https://github.com/kuangliu/pytorch-cifar/issues/8
        for i, (mean, std) in enumerate(
            zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ):
            self.data[:, i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print(
            "CIFAR10 data shape {}, targets shape {}".format(
                self.data.shape, self.targets.shape
            )
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class FastCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        from torch import from_numpy

        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # https://github.com/kuangliu/pytorch-cifar/issues/16
        # https://github.com/kuangliu/pytorch-cifar/issues/8
        for i, (mean, std) in enumerate(
            zip((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ):
            self.data[:, i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print(
            "CIFAR10 data shape {}, targets shape {}".format(
                self.data.shape, self.targets.shape
            )
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def get_fast_cifar10(cfg: DatasetConfig):
    train = FastCIFAR10(root=DATA_PATH, train=True, download=True)
    test = FastCIFAR10(root=DATA_PATH, train=False, download=True)
    return VisionClfDataset(train, "CIFAR10"), VisionClfDataset(test, "CIFAR10")


def get_cifar100(cfg: DatasetConfig):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train = CIFAR100(
        root=DATA_PATH, train=True, download=True, transform=train_transform
    )
    test = CIFAR100(
        root=DATA_PATH, train=False, download=True, transform=test_transform
    )
    return VisionClfDataset(train, "CIFAR100"), VisionClfDataset(test, "CIFAR100")
