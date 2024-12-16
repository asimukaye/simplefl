from dataclasses import dataclass, field, asdict, MISSING
import typing as t
from functools import partial
import os
import torch
from torch.nn import (
    Module,
    CrossEntropyLoss,
    NLLLoss,
    MSELoss,
    L1Loss,
    BCELoss,
    BCEWithLogitsLoss,
    CTCLoss,
    KLDivLoss,
    MultiMarginLoss,
    SmoothL1Loss,
    TripletMarginLoss,
    CosineEmbeddingLoss,
    PoissonNLLLoss,
)
from torch.optim import Optimizer, SGD, Adam, AdamW, Adadelta, Adagrad, RMSprop
from torch.optim.lr_scheduler import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
)

## Root level module. Should not have any dependencies on other modules except utils

from utils import auto_configure_device

OPTIMIZER_MAP = {
    "adam": Adam,
    "sgd": SGD,
    "adamw": AdamW,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "rmsprop": RMSprop,
}

LRSCHEDULER_MAP = {
    "step": StepLR,
    "multistep": MultiStepLR,
    "exponential": ExponentialLR,
    "cosine": CosineAnnealingLR,
    "plateau": ReduceLROnPlateau,
    "cyclic": CyclicLR,
    "onecycle": OneCycleLR,
    "cosine_warmup": CosineAnnealingWarmRestarts,
}

LOSS_MAP = {
    "crossentropy": CrossEntropyLoss,
    "nll": NLLLoss,
    "mse": MSELoss,
    "l1": L1Loss,
    "bce": BCELoss,
    "bcelogits": BCEWithLogitsLoss,
    "ctc": CTCLoss,
    "kl": KLDivLoss,
    "margin": MultiMarginLoss,
    "smoothl1": SmoothL1Loss,
    "huber": SmoothL1Loss,
    "triplet": TripletMarginLoss,
    "hinge": MultiMarginLoss,
    "cosine": CosineEmbeddingLoss,
    "poisson": PoissonNLLLoss,
}

SEED = 42

DATA_PATH = os.getcwd()+"/data"


@dataclass
class DatasetModelSpec:
    num_classes: int
    in_channels: int


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 0.01
    batch_size: int = 128
    eval_batch_size: int = 128
    optimizer: str = "sgd"
    loss_name: str = "crossentropy"
    scheduler: str = "exponential"
    lr_decay: float = 0.977
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = auto_configure_device()
        self.optim_partial: partial[Optimizer] = partial(
            OPTIMIZER_MAP[self.optimizer], lr=self.lr
        )
        self.loss_fn: Module = LOSS_MAP[self.loss_name]()
        if self.scheduler == "exponential":
            self.scheduler_partial: partial[LRScheduler] = partial(
                LRSCHEDULER_MAP[self.scheduler], gamma=self.lr_decay
            )


@dataclass
class DatasetConfig:
    name: str
    seed: int = SEED
    subsample_fraction: float = 1.0


@dataclass
class SplitConfig:
    name: str
    num_splits: int = 0  # should be equal to num_clients
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    # test_fractions: list[float] = field(init=False, default_factory=list)
    def __post_init__(self):
        # self.test_fractions = [1.0 / self.num_splits] * self.num_splits
        self.dataset_name = '---'

@dataclass
class NoisyImageSplitConfig(SplitConfig):
    name: str = "noisy_image"
    num_noisy_clients: int = 1
    noise_mu: float = 0.0
    noise_sigma: float | list = 0.1


@dataclass
class NoisyLabelSplitConfig(SplitConfig):
    name: str = "noisy_label"
    num_noisy_clients: int = 1
    noise_flip_percent: float | list = 0.1


@dataclass
class PathoSplitConfig(SplitConfig):
    name: str = "patho"
    num_class_per_client: int = 2


@dataclass
class DirichletSplitConfig(SplitConfig):
    name: str = "dirichlet"
    alpha: float = 1.0  # concentration parameter


@dataclass
class DataImbalanceSplitConfig(SplitConfig):
    name: str = "data_imbalance"
    num_imbalanced_clients: int = 1


######### STRATEGY CONFIGS #########


@dataclass
class Config:
    train: TrainConfig
    dataset: DatasetConfig
    split: SplitConfig
    model: str
    num_clients: int
    num_rounds: int
    name: str = "---"
    desc: str = ""
    train_fraction: float = 1.0
    eval_fraction: float = 1.0
    seed: int = SEED
    checkpoint_every: int = 20

    def __post_init__(self):
        ## Define internal config variables here
        self.use_wandb = True
        self.resumed = False
        self.split.num_splits = self.num_clients
        if self.split.name == "natural":
            self.split.dataset_name = self.dataset.name
