import argparse
import time
import os
import typing as t
from dataclasses import asdict, dataclass, field
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

from config import compile_config, load_config, set_debug_config
from data import load_raw_dataset
from model import init_model
from icecream import install, ic

install()
ic.configureOutput(includeContext=True)

# logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--strategy", type=str, default="fedavg")
    parser.add_argument("-r", "--resume_from", type=str, default="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--rounds", default=0)

    args = parser.parse_args()

    start_time = time.time()

    if args.resume_from:
        cfg = load_config(args.resume_from)
    else:
        cfg = compile_config(args.strategy)

    if args.debug:
        print("_____DEBUG MODE______\n")
        cfg = set_debug_config(cfg)

    if args.rounds > cfg.num_rounds:
        print("Overriding rounds")
        cfg.num_rounds = args.rounds

    print(yaml.dump(asdict(cfg)))
    input("Press Enter to continue...")

    set_seed(cfg.seed)

    dataset, model_spec = load_raw_dataset(cfg.dataset)
    model_instance = init_model(
        cfg.model, model_spec.in_channels, model_spec.num_classes, "xavier", 1.0
    )

    if args.resume_from:
        resumed = True
        os.chdir(args.resume_from)
        run_id = get_wandb_run_id(args.resume_from)
        wandb_resume_mode = "must"
    else:
        resumed = False
        if args.debug:
            setup_output_dirs("debug")
        else:
            setup_output_dirs(args.strategy)
        run_id = None
        wandb_resume_mode = True
        with open("config.yaml", "w") as f:
            yaml.dump(asdict(cfg), f)

    if args.debug:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)
    print(cfg.use_wandb)

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
    # Run.log

    match args.strategy:
        case "fedavg":
            from fedavg import run_fedavg

            out = run_fedavg(dataset, model_instance, cfg, resumed)
        case _:
            raise NotImplementedError

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time}")
