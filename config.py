from dataclasses import dataclass, field, asdict, MISSING
import typing as t
from functools import partial
import yaml
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
DATA_PATH = "data"


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
class SplitConfig:
    name: str = "none"
    num_splits: int = 0  # should be equal to num_clients
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    # test_fractions: list[float] = field(init=False, default_factory=list)


IIDSplitConfig = SplitConfig
IIDSplitConfig.name = "iid"


@dataclass
class NoisyImageSplitConfig(SplitConfig):
    name = "noisy_image"
    num_noisy_clients: int = 1
    noise_mu: float = 0.0
    noise_sigma: float = 0.1


@dataclass
class NoisyLabelSplitConfig(SplitConfig):
    name = "noisy_label"
    num_noisy_clients: int = 1
    noise_flip_percent: float = 0.1


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


@dataclass
class DatasetConfig:
    name: str = "fast_cifar10"
    seed: int = SEED
    subsample_fraction: float = 1.0


######### STRATEGY CONFIGS #########


@dataclass
class Config:
    train: TrainConfig
    dataset: DatasetConfig
    split: SplitConfig
    model: str = "rffl_cnn"
    desc: str = ""
    name: str = "fedavg"
    num_clients: int = 6
    num_rounds: int = 200
    train_fraction: float = 1.0
    eval_fraction: float = 1.0
    seed: int = SEED
    checkpoint_every: int = 20

    def __post_init__(self):
        ## Define internal config variables here
        self.use_wandb = True
        self.split.num_splits = self.num_clients


@dataclass
class CGSVConfig(Config):
    beta: float = 1.0
    alpha: float = 0.95
    gamma: float = 0.25


@dataclass
class FHGConfig(Config):
    name: str = "fedhigrad"
    branches: list[int] = field(default_factory=lambda: [1, 2, 3])


CONFIG_MAP = {
    "fedavg": Config,
    "cgsv": CGSVConfig,
    "rffl": Config,
    "fedhigrad": FHGConfig,
}


def set_debug_config(cfg: Config) -> Config:
    cfg.use_wandb = False

    cfg.dataset.subsample_fraction = 0.05
    cfg.train.epochs = 1
    cfg.num_rounds = 3
    return cfg


def get_default_config(strategy: str) -> Config:
    return Config(
        TrainConfig(), DatasetConfig(), SplitConfig(), model="rffl_cnn", name=strategy
    )


def get_fedavg_config() -> Config:
    cfg = Config(
        desc="FedAvg on CIFAR10, Dirichle split 0.01",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        seed=SEED,
        train=TrainConfig(
            epochs=1,
            lr=0.01,
            batch_size=128,
            eval_batch_size=128,
            device="auto",
            optimizer="sgd",
            loss_name="crossentropy",
            scheduler="exponential",
            lr_decay=0.977,
        ),
        split=DirichletSplitConfig(alpha=0.01),
        # split=IIDSplitConfig(),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    return cfg


def get_fedhigrad_config() -> FHGConfig:
    cfg = FHGConfig(
        desc="FedAvg on CIFAR10, Dirichlet split, alpha=0.1",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        seed=SEED,
        train=TrainConfig(
            epochs=1,
            lr=0.01,
            batch_size=128,
            eval_batch_size=128,
            device="auto",
            optimizer="sgd",
            loss_name="crossentropy",
            scheduler="exponential",
            lr_decay=0.977,
        ),
        split=DirichletSplitConfig(name="dirichlet", alpha=1.0),
        dataset=DatasetConfig(
            name="fast_cifar10",
            seed=SEED,
            subsample_fraction=1.0,
        ),
    )
    return cfg


def load_config(cfg_dir: str) -> Config:
    with open(cfg_dir + "/config.yaml", "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    strategy = cfg_dict["name"]
    print("Strategy: ", strategy)
    print("Resuming from: ", cfg_dir)
    cfg = CONFIG_MAP[strategy](**cfg_dict)
    return cfg


def compile_config(strategy: str) -> Config:
    """Compile the configuration dictionary"""
    print("Strategy: ", strategy)
    print("\n")

    match strategy:
        case "fedavg":
            cfg = get_fedavg_config()
            # cfg.split.num_splits = cfg.num_clients
            # return cfg
        case "fedhigrad":
            cfg = get_fedhigrad_config()
        case _:
            raise NotImplementedError
    return cfg
