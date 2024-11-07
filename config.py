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
    name: str
    num_splits: int = 0  # should be equal to num_clients
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    # test_fractions: list[float] = field(init=False, default_factory=list)


# IIDSplitConfig = SplitConfig
# IIDSplitConfig.name = "iid"


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
    name: str = ""
    num_clients: int = -1
    num_rounds: int = -1
    train_fraction: float = 1.0
    eval_fraction: float = 1.0
    seed: int = SEED
    checkpoint_every: int = 20

    def __post_init__(self):
        ## Define internal config variables here
        self.use_wandb = True
        self.resumed = False
        self.split.num_splits = self.num_clients


@dataclass
class CGSVConfig(Config):
    name: str = "cgsv"
    beta: float = 1.0
    alpha: float = 0.95
    gamma: float = 0.15
    use_reputation: bool = True
    use_sparsify: bool = True
    fedopt_debug: bool = False
    normalize_delta: bool = True


@dataclass
class ShapfedConfig(Config):
    name: str = "shapfed"
    alpha: float = 0.95
    gamma: float = 0.15
    compute_every: int = 1


@dataclass
class FHGConfig(Config):
    name: str = "fedhigrad"
    branches: list[int] = field(default_factory=lambda: [2, 2])
    enable_weights: bool = True
    alpha: float = 0.95
    phi_method: str = "mean"


SPLIT_MAP = {
    "iid": SplitConfig,
    "noisy_image": NoisyImageSplitConfig,
    "noisy_label": NoisyLabelSplitConfig,
    "dirichlet": DirichletSplitConfig,
    "data_imbalance": DataImbalanceSplitConfig,
    "patho": PathoSplitConfig,
}
CONFIG_MAP = {
    "fedavg": Config,
    "fedopt": Config,
    "cgsv": CGSVConfig,
    "rffl": CGSVConfig,
    "fedhigrad": FHGConfig,
    "centralized": Config,
}


def set_debug_config(cfg: Config) -> Config:
    cfg.use_wandb = False

    cfg.dataset.subsample_fraction = 0.05
    # cfg.dataset.subsample_fraction = 1.0
    cfg.train.epochs = 1
    cfg.num_rounds = 10
    return cfg


def get_default_config(strategy: str) -> Config:
    return Config(
        TrainConfig(),
        DatasetConfig(),
        SplitConfig(name="iid"),
        model="rffl_cnn",
        name=strategy,
    )


def get_centralized_config() -> Config:
    cfg = Config(
        name="centralized",
        # model="rffl_cnn",
        # model="twocnn",
        # model="twocnnv2",
        model="fednet",
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
        # split=DirichletSplitConfig(alpha=0.01),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            # name="cifar10",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = f"Centralized on CIFAR10, Noisy Image split {cfg.split.num_noisy_clients} Noise, mu=0.0, sigma={cfg.split.noise_sigma}"
    # cfg.desc = f"Centralized on CIFAR10, not fast cifar, checking peak accuracy, pooling enabled"
    # cfg.desc = f"Centralized on fast cifar, two cnn v2 model"
    cfg.desc = f"Centralized on fast cifar, fednet model"
    return cfg


def get_standalone_config() -> Config:
    cfg = Config(
        name="standalone",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=300,
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
        split=DirichletSplitConfig(alpha=0.05),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=NoisyImageSplitConfig(num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            # name="cifar10",
            subsample_fraction=1.0,
        ),
    )

    cfg.desc = f"Standalone run on fast cifar10, dirichlet"
    return cfg


def get_fedavg_config() -> Config:
    cfg = Config(
        name="fedavg",
        model="rffl_cnn",
        # model="twocnn",
        # model="twocnnv2",
        # model="fednet",
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
        split=DirichletSplitConfig(alpha=0.05),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = "FedAvg on fast cifar10 iid, twocnn"
    # cfg.desc = "FedAvg on fast cifar10 iid, fednet"
    # cfg.desc = "FedAvg on fast cifar10 iid, twocnnv2"
    cfg.desc = "FedAvg on fast cifar10 iid, rffl_cnn, dirichlet 0.05"
    # cfg.desc = (
    #     "FedAvg on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    # )

    return cfg


def get_fedopt_config() -> Config:
    cfg = Config(
        name="fedopt",
        # desc="FedOpt on fast CIFAR10, Dirichlet split, alpha=0.01",
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
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    cfg.desc="FedOpt on fast CIFAR10, Dirichlet split"
    # cfg.desc = "FedOpt on fast CIFAR10, IID, modified to add deltas to clients"

    return cfg


def get_cgsv_config() -> CGSVConfig:
    cfg = CGSVConfig(
        name="cgsv",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        alpha=0.95,
        beta=1.0,
        gamma=0.15,
        use_sparsify=True,
        use_reputation=True,
        fedopt_debug=False,
        normalize_delta=True,
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
        # split=DirichletSplitConfig(alpha=0.1),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        split=NoisyImageSplitConfig(
            num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        ),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = (
    # f"CGSV on CIFAR10, Noisy Image split 3 Noise, mu=0.0, sigma=3.0, as is in paper"
    # )
    
    cfg.desc = (
        "CGSV on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    )

    # cfg.desc = f"CGSV on CIFAR10, Noisy Image split 3 Noise, mu=0.0, sigma=3.0, normalizing phis, no delta normalize"
    # cfg.desc = f"CGSV on CIFAR10, Dirichlet split"
    # cfg.desc = f"CGSV on CIFAR10 IID, sweep gama={cfg.gamma}"
    # cfg.desc = f"CGSV on fast CIFAR10 IID, use reputation but no sparsify"
    return cfg


def get_shapfed_config() -> ShapfedConfig:
    cfg = ShapfedConfig(
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        alpha=0.95,
        gamma=0.15,
        compute_every=1,
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
        # split=DirichletSplitConfig(alpha=0.01),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    cfg.desc = f"shapfed on fast cifar10 with iid split"
    cfg.desc = (
        "Shapfed on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    )

    return cfg


def get_fedhigrad_config() -> FHGConfig:
    cfg = FHGConfig(
        name="fedhigrad",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        enable_weights=True,
        branches=[4],
        alpha=0.95,
        # phi_method="norm",
        phi_method="delta/sigma",
        seed=SEED,
        train=TrainConfig(
            epochs=1,
            lr=0.05,
            batch_size=128,
            eval_batch_size=128,
            device="auto",
            optimizer="sgd",
            loss_name="crossentropy",
            scheduler="exponential",
            lr_decay=0.977,
        ),
        split=DirichletSplitConfig(name="dirichlet", alpha=0.01),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=1.0),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        # split=NoisyLabelSplitConfig(num_noisy_clients=2, noise_flip_percent=0.1),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            seed=SEED,
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = f"Fedhigrad on fast CIFAR10, [2, 2], Noisy Image split"
    # cfg.desc = f"Fedhigrad on fast CIFAR10, Noisy Image split sweep
    # cfg.desc = "Fedhigrad on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    cfg.desc = "Fedhigrad on fast cifar10, [4], dirichlet split 0.01, delta/sigma"
    return cfg


def load_config(cfg_dir: str) -> Config:
    with open(cfg_dir + "/config.yaml", "r") as f:
        cfg_dict: dict = yaml.load(f, Loader=yaml.FullLoader)
    strategy = cfg_dict["name"]
    print("Strategy: ", strategy)
    print("Resuming from: ", cfg_dir)
    tr = TrainConfig(**cfg_dict.pop("train"))
    ds = DatasetConfig(**cfg_dict.pop("dataset"))
    sp_name = cfg_dict["split"]["name"]
    if sp_name not in SPLIT_MAP:
        raise NotImplementedError
    sp = SPLIT_MAP[sp_name](**cfg_dict.pop("split"))

    cfg = CONFIG_MAP[strategy](train=tr, dataset=ds, split=sp, **cfg_dict)
    print("Config loaded: ", cfg)
    return cfg


def compile_config(strategy: str) -> Config:
    """Compile the configuration dictionary"""
    print("Strategy: ", strategy)
    print("\n")

    match strategy:
        case "fedavg":
            cfg = get_fedavg_config()
        case "fedopt":
            cfg = get_fedopt_config()
        case "cgsv":
            cfg = get_cgsv_config()
        case "fedhigrad":
            cfg = get_fedhigrad_config()
        case "centralized":
            cfg = get_centralized_config()
        case "standalone":
            cfg = get_standalone_config()
        case "shapfed":
            cfg = get_shapfed_config()
        case _:
            raise NotImplementedError
    return cfg
