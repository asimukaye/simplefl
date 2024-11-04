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
from wandb.sdk.wandb_run import Run
import yaml

from torch.nn import Module
from torch.backends import cudnn, mps
from torch.utils.data import Dataset

from config import (
    compile_config,
    load_config,
    set_debug_config,
    CGSVConfig,
    Config,
    FHGConfig,
)
from data import load_raw_dataset
from model import init_model
from icecream import install, ic

install()
ic.configureOutput(includeContext=True)

HOME_DIR = os.getcwd()


# logger = logging.getLogger(__name__)
def setattr_nested(base, path: str, value):
    """Accept a dotted path to a nested attribute to set."""
    splits = path.split(".")
    intermediates = splits[:-1]
    target = splits[-1]
    for attrname in intermediates:
        base = getattr(base, attrname)
    setattr(base, target, value)


def getattr_nested(base: t.Any, path: str) -> t.Any:
    splits = path.split(".")
    for attrname in splits:
        base = getattr(base, attrname)
    return base


def hasattr_nested(base, path: str):
    splits = path.split(".")
    for attrname in splits:
        if not hasattr(base, attrname):
            return False
        base = getattr(base, attrname)
    return True


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
        os.chdir(resume_from)
        run_id = get_wandb_run_id(resume_from)
        wandb_resume_mode = "must"
    else:
        if debug:
            setup_output_dirs("debug")
        else:
            setup_output_dirs(strategy)
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
            job_type=strategy,
            # tags=tags,
            config=asdict(cfg),
            resume=wandb_resume_mode,
            notes=cfg.desc,
            id=run_id,
        )
        assert isinstance(run, Run)

    match strategy:
        case "fedavg":
            from fedavg import run_fedavg

            out = run_fedavg(dataset, model_instance, cfg)
        case "fedopt":
            from fedopt import run_fedopt

            out = run_fedopt(dataset, model_instance, cfg)
        case "fedhigrad":
            from fedhigrad import run_fedhigrad

            assert isinstance(cfg, FHGConfig)
            out = run_fedhigrad(dataset, model_instance, cfg)
        case "cgsv":
            from cgsv import run_cgsv

            assert isinstance(cfg, CGSVConfig)
            out = run_cgsv(dataset, model_instance, cfg)
        case "centralized":
            from centralized import run_centralized

            assert isinstance(cfg, Config)
            out = run_centralized(dataset, model_instance, cfg)
        case _:
            raise NotImplementedError

    if cfg.use_wandb:
        wandb.finish()
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
    parser.add_argument("-p2", "--sweep_param2", type=str, default="")
    parser.add_argument("-v2", "--sweep_values2", nargs="+")
    parser.add_argument("--rounds", default=0)

    args = parser.parse_args()

    if args.resume_from:
        cfg = load_config(args.resume_from)
        cfg.resumed = True
        if args.rounds > cfg.num_rounds:
            print("Resuming from rounds")
            cfg.num_rounds = args.rounds
    else:
        cfg = compile_config(args.strategy)

    if args.debug:
        print("_____DEBUG MODE______\n")
        cfg = set_debug_config(cfg)

    if args.sweep_param:
        assert len(args.sweep_values) > 0
        assert hasattr_nested(
            cfg, args.sweep_param
        ), f"{args.sweep_param} not in config"
        # val = attrgetter(args.sweep_param)(cfg)
        val = getattr_nested(cfg, args.sweep_param)
        astype = type(val)
        sweep_values = [astype(x) for x in args.sweep_values]
        print("_____SWEEP MODE______\n")

        print(f"Sweeping over param: {args.sweep_param} with values:{sweep_values}")

        if args.sweep_param2:
            assert len(args.sweep_values2) > 0
            assert hasattr_nested(
                cfg, args.sweep_param2
            ), f"{args.sweep_param2} not in config"
            val2 = getattr_nested(cfg, args.sweep_param2)
            astype2 = type(val2)
            sweep_values2 = [astype2(x) for x in args.sweep_values2]

            print(
                f"Sweeping over param2: {args.sweep_param2} with values:{sweep_values2}"
            )

            print(yaml.dump(asdict(cfg)))

            input("Press Enter to continue...")

            for sweep_value in sweep_values:
                setattr_nested(cfg, args.sweep_param, sweep_value)
                for sweep_value2 in sweep_values2:
                    setattr_nested(cfg, args.sweep_param2, sweep_value2)
                    print(
                        f"Running with {args.sweep_param}={sweep_value} and {args.sweep_param2}={sweep_value2}"
                    )
                    print(yaml.dump(asdict(cfg)))

                    launch_experiment(args.strategy, cfg, args.resume_from, args.debug)
        else:
            print(yaml.dump(asdict(cfg)))

            input("Press Enter to continue...")

            for sweep_value in sweep_values:
                setattr_nested(cfg, args.sweep_param, sweep_value)
                print(f"Running with {args.sweep_param}={sweep_value}")
                print(yaml.dump(asdict(cfg)))

                launch_experiment(args.strategy, cfg, args.resume_from, args.debug)
    else:
        print(yaml.dump(asdict(cfg)))
        input("Press Enter to continue...")
        launch_experiment(args.strategy, cfg, args.resume_from, args.debug)

    logging.info("Done!")
