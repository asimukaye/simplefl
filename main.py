import argparse
import time
import os
import typing as t
from dataclasses import asdict, dataclass, field
from operator import attrgetter
import logging
import random
import json
import sys
import numpy as np
import torch
import wandb
import yaml

from torch.nn import Module
from torch.backends import cudnn, mps
from torch.utils.data import Dataset

from config import compile_config, load_config, set_debug_config, CGSVConfig, Config
from data import load_raw_dataset
from model import init_model
from icecream import install, ic

install()
ic.configureOutput(includeContext=True)

HOME_DIR = os.getcwd()
# logger = logging.getLogger(__name__)
def setattr_nested(base, path, value):
    """Accept a dotted path to a nested attribute to set."""
    path, _, target = path.rpartition('.')
    for attrname in path.split('.'):
        base = getattr(base, attrname)
    setattr(base, target, value)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"[SEED] Simulator global seed is set to: {seed}!")


def get_wandb_run_id(root_dir=".") -> str:
    with open(root_dir + "/wandb/wandb-resume.json", "r") as f:
        wandb_json = json.load(f)
    return wandb_json["run_id"]


def setup_output_dirs(suffix="debug"):
    # os.makedirs("output", exist_ok=True)
    experiment_date = time.strftime("%y-%m-%d")
    experiment_time = time.strftime("%H-%M-%S")
    output_dir = f"output/{experiment_date}/{suffix}/{experiment_time}"
    os.makedirs(output_dir, exist_ok=False)
    print(f"Logging to {output_dir}")
    os.chdir(output_dir)


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    verbose_formatter = logging.Formatter(
        "%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("log.txt")
    file_handler.mode = "a"
    file_handler.setLevel(level)
    file_handler.setFormatter(verbose_formatter)
    root_logger.addHandler(file_handler)

    # exit(0)


def launch_experiment(strategy: str, cfg: Config, resume_from: str, debug: bool):

    start_time = time.time()
    set_seed(cfg.seed)

    dataset, model_spec = load_raw_dataset(cfg.dataset)
    model_instance = init_model(
        cfg.model, model_spec.in_channels, model_spec.num_classes, "xavier", 1.0
    )

    if resume_from:
        resumed = True
        os.chdir(args.resume_from)
        run_id = get_wandb_run_id(args.resume_from)
        wandb_resume_mode = "must"
    else:
        resumed = False
        if debug:
            setup_output_dirs("debug")
        else:
            setup_output_dirs(args.strategy)
        run_id = None
        wandb_resume_mode = True
        with open("config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f)

    if debug:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)

    if cfg.use_wandb:
        run = wandb.init(
            project="simplefl",
            job_type=args.strategy,
            # tags=tags,
            config=asdict(cfg),
            resume=wandb_resume_mode,
            notes=cfg.desc,
            id=run_id,
        )

    match strategy:
        case "fedavg":
            from fedavg import run_fedavg

            out = run_fedavg(dataset, model_instance, cfg, resumed)
        case "cgsv":
            from cgsv import run_cgsv

            assert isinstance(cfg, CGSVConfig)
            out = run_cgsv(dataset, model_instance, cfg, resumed)
        case _:
            raise NotImplementedError

    os.chdir(HOME_DIR)
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--strategy", type=str, default="fedavg")
    parser.add_argument("-r", "--resume_from", type=str, default="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--sweep_param", type=str, default="")
    parser.add_argument("-v", "--sweep_values", nargs="+")
    parser.add_argument("--rounds", default=0)

    args = parser.parse_args()

    if args.resume_from:
        cfg = load_config(args.resume_from)
        if args.rounds > cfg.num_rounds:
            print("Resuming from rounds")
            cfg.num_rounds = args.rounds
    else:
        cfg = compile_config(args.strategy)


    if args.debug:
        print("_____DEBUG MODE______\n")
        cfg = set_debug_config(cfg)

 
    if args.sweep_param:
        # assert hasattr(cfg, args.sweep_param), f"{args.sweep_param} not in config"
        val = attrgetter(args.sweep_param)(cfg)
        # astype = type(getattr(cfg, args.sweep_param))
        astype = type(val)
        sweep_values = [astype(x) for x in args.sweep_values]
        print("_____SWEEP MODE______\n")

        print(yaml.dump(asdict(cfg)))

        print(f"Sweeping over: {args.sweep_param} with values:{sweep_values}")

        input("Press Enter to continue...")

        for sweep_value in sweep_values:
            setattr_nested(cfg, args.sweep_param, sweep_value)
            print(f"Running with {args.sweep_param}={sweep_value}")
            launch_experiment(args.strategy, cfg, args.resume_from, args.debug)
    else:
        print(yaml.dump(asdict(cfg)))
        input("Press Enter to continue...")
        launch_experiment(args.strategy, cfg, args.resume_from, args.debug)

    # for sweep_value in sweep_values:
    #     setattr(cfg, args.sweep_param, sweep_value)
    #     print(f"Running with {args.sweep_param}={sweep_value}")
