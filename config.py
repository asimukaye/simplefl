from dataclasses import dataclass, field, asdict, MISSING
import typing as t
from functools import partial
import yaml

from rootconfig import *


def set_debug_config(cfg: Config) -> Config:
    cfg.use_wandb = False

    cfg.dataset.subsample_fraction = 0.05
    # cfg.dataset.subsample_fraction = 1.0
    cfg.train.epochs = 1
    cfg.num_rounds = 3
    return cfg


def get_default_config(strategy: str) -> Config:
    return Config(
        TrainConfig(),
        DatasetConfig(name="fast_cifar10"),
        SplitConfig(name="iid"),
        model="rffl_cnn",
        name=strategy,
        num_clients=6,
        num_rounds=500,
        desc="Default config RFFL CNN, IID, fast CIFAR10",
    )


def get_centralized_config() -> Config:
    cfg = Config(
        name="centralized",
        # model="rffl_cnn",
        # model="twocnn",
        # model="twocnnv2",
        # model="fednet",
        model="resnet34",
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
            # name="fast_cifar10",
            name="fedisic",
            # name="cifar10",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = f"Centralized on CIFAR10, Noisy Image split {cfg.split.num_noisy_clients} Noise, mu=0.0, sigma={cfg.split.noise_sigma}"
    # cfg.desc = f"Centralized on CIFAR10, not fast cifar, checking peak accuracy, pooling enabled"
    # cfg.desc = f"Centralized on fast cifar, two cnn v2 model"
    cfg.desc = f"Centralized on fedisic, resnet34 model"
    return cfg


def get_standalone_config() -> Config:
    cfg = Config(
        name="standalone",
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
        # split=DirichletSplitConfig(alpha=0.05),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=NoisyLabelSplitConfig(num_noisy_clients=3, noise_flip_percent=0.2),
        split=NoisyLabelSplitConfig(
            num_noisy_clients=5, noise_flip_percent=[0.25, 0.2, 0.15, 0.1, 0.05]
        ),
        # split=NoisyImageSplitConfig(num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            # name="cifar10",
            subsample_fraction=1.0,
        ),
    )

    cfg.desc = f"Standalone run on fast cifar10, varying noisy label"
    return cfg


def get_fedavg_config() -> Config:
    cfg = Config(
        name="fedavg",
        model="rffl_cnn",
        # model="twocnn",
        # model="twocnnv2",
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
        # split=DirichletSplitConfig(alpha=0.05),
        # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=3.0),
        # split=NoisyLabelSplitConfig(num_noisy_clients=2, noise_flip_percent=0.1),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        split=NoisyLabelSplitConfig(
            num_noisy_clients=5, noise_flip_percent=[0.25, 0.2, 0.15, 0.1, 0.05]
        ),
        # split=SplitConfig(name="iid"),
        # split=SplitConfig(name="natural"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            # name="fedisic",
            # name="emnist",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = "FedAvg on fast cifar10 iid, twocnn"
    # cfg.desc = "FedAvg on fast cifar10 iid, fednet"
    # cfg.desc = "FedAvg on fast cifar10 iid, twocnnv2"
    # cfg.desc = "FedAvg on fast cifar10 iid, rffl_cnn, dirichlet 0.05"
    cfg.desc = "FedAvg on varying label flip noise"
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
    cfg.desc = "FedOpt on fast CIFAR10, Dirichlet split"
    # cfg.desc = "FedOpt on fast CIFAR10, IID, modified to add deltas to clients"

    return cfg


from cgsv import CGSVConfig


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
        # split=NoisyLabelSplitConfig(num_noisy_clients=2, noise_flip_percent=0.1),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        split=NoisyLabelSplitConfig(
            num_noisy_clients=5, noise_flip_percent=[0.25, 0.2, 0.15, 0.1, 0.05]
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

    # cfg.desc = (
    #     "CGSV on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    # )
    cfg.desc = "CGSV on fast cifar10 varying label noise"

    # cfg.desc = f"CGSV on CIFAR10, Noisy Image split 3 Noise, mu=0.0, sigma=3.0, normalizing phis, no delta normalize"
    # cfg.desc = f"CGSV on CIFAR10, Dirichlet split"
    # cfg.desc = f"CGSV on CIFAR10 IID, sweep gama={cfg.gamma}"
    # cfg.desc = f"CGSV on fast CIFAR10 IID, use reputation but no sparsify"
    return cfg


from shapfed import ShapfedConfig


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
        split=NoisyLabelSplitConfig(
            num_noisy_clients=5, noise_flip_percent=[0.25, 0.2, 0.15, 0.1, 0.05]
        ),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = f"shapfed on fast cifar10 with iid split"
    # cfg.desc = f"shapfed on fast cifar10 with dirichlet split"
    # cfg.desc = (
    #     "Shapfed on fast cifar10 varying noise, mu=0.0, sigma=[2.5, 2.0, 1.5, 1.0, 0.5]"
    # )
    cfg.desc = "Shapfed on fast cifar10 varying label noise"
    return cfg


from fedhigrad import FHGConfig


def get_fedhigrad_config() -> FHGConfig:
    cfg = FHGConfig(
        name="fedhigrad",
        model="rffl_cnn",
        num_clients=6,
        num_rounds=500,
        enable_weights=True,
        branches=[4],
        alpha=0.95,
        aggregate=True,
        sigma_floor=0.01,
        beta=0.4,
        phi_method="norm",
        # phi_method="delta/sigma",
        # phi_method="delta/sigma_delta",
        # phi_method="delta+sigma",
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
        # split=DirichletSplitConfig(name="dirichlet", alpha=1.0),
        # # split=NoisyImageSplitConfig(num_noisy_clients=3, noise_mu=0.0, noise_sigma=1.0),
        # split=NoisyImageSplitConfig(
        #     num_noisy_clients=5, noise_mu=0.0, noise_sigma=[2.5, 2.0, 1.5, 1.0, 0.5]
        # ),
        # split=NoisyLabelSplitConfig(num_noisy_clients=3, noise_flip_percent=0.2),
        split=NoisyLabelSplitConfig(
            num_noisy_clients=5, noise_flip_percent=[0.25, 0.2, 0.15, 0.1, 0.05]
        ),
        # split=SplitConfig(name="iid"),
        dataset=DatasetConfig(
            name="fast_cifar10",
            seed=SEED,
            subsample_fraction=1.0,
        ),
    )
    # cfg.desc = f"Fedhigrad on fast CIFAR10, [2, 2], Noisy Image split"
    # cfg.desc = f"Fedhigrad on fast CIFAR10, noisy label, phi delta + inv sigma"
    # cfg.desc = f"Fedhigrad on fast CIFAR10, iid, delta/sigma"
    # cfg.desc = f"Fedhigrad on fast CIFAR10, dirichlet 1 , delta/sigma"
    cfg.desc = f"Fedhigrad on fast CIFAR10, varying label noise , 1/sigma"
    # cfg.desc = "Fedhigrad on fast cifar10 varying noise sigma=[2.5, 2.0, 1.5, 1.0, 0.5], delta+invsigma"
    # cfg.desc = "Fedhigrad on fast cifar10, [4], dirichlet split 0.01, delta/sigma"
    # cfg.desc = "Fedhigrad on fast cifar10, [4], dirichle, delta/ sigma"
    # cfg.desc = "Fedhigrad on fast cifar10, [4], dirichlet, delta/del_sigma"
    # cfg.desc = "Fedhigrad on fast cifar10, [4], dirichlet, delta+sigma"
    return cfg


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

CONFIG_FN_MAP = {
    "fedavg": get_fedavg_config,
    "fedopt": get_fedopt_config,
    "cgsv": get_cgsv_config,
    "fedhigrad": get_fedhigrad_config,
    "centralized": get_centralized_config,
    "standalone": get_standalone_config,
    "shapfed": get_shapfed_config,
}


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
