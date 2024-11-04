import logging
from copy import deepcopy
import time
import torch
from torch.nn import Module
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import wandb
from pandas import json_normalize
from config import Config
from data import DatasetPair
from split import get_client_datasets, pool_datasets
from utils import find_server_checkpoint, make_server_checkpoint_dirs
from fedavg import simple_evaluator, simple_trainer

logger = logging.getLogger(__name__)


def run_centralized(
    dataset: DatasetPair,
    model: Module,
    cfg: Config,
):

    if cfg.resumed:
        server_ckpt_pth = find_server_checkpoint()
        checkpoint = torch.load(server_ckpt_pth)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = cfg.train.optim_partial(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # epoch = checkpoint['epoch']
        start_epoch = checkpoint["round"]
        # Find a way to avoid this result manager round bug repeatedly
    else:
        make_server_checkpoint_dirs()
        start_epoch = 0
        optimizer = cfg.train.optim_partial(model.parameters())

    # split and repool to match any data modifications
    client_datasets = get_client_datasets(cfg.split, dataset)
    pooled_set = pool_datasets(client_datasets)

    # pooled_set = dataset

    logger.info(f"Server pooled dataset train size: {len(pooled_set.train)}")

    train_loader = DataLoader(
        dataset=pooled_set.train, batch_size=cfg.train.batch_size, shuffle=True
    )

    test_loader = DataLoader(
        dataset=pooled_set.test, batch_size=cfg.train.eval_batch_size, shuffle=False
    )
    # keeping the same amount of training duration as federated:
    total_epochs = cfg.train.epochs * cfg.num_rounds
    step_count = 0

    metrics = {
        "loss": {"train": {}, "eval": {}},
        "accuracy": {"train": {}, "eval": {}},
        "round": 0,
    }

    for curr_round in range(start_epoch, total_epochs):
        logger.info(f"-------- Round: {curr_round} --------\n")

        loop_start = time.time()

        metrics["round"] = curr_round

        train_result = simple_trainer(model, train_loader, cfg.train, optimizer)

        metrics["loss"]["train"]["server"] = train_result["loss"]
        metrics["accuracy"]["train"]["server"] = train_result["accuracy"]
        logger.info(f"SERVER TRAIN: {train_result}")

        # result_manager.log_general_result(
        #     train_result, "post_train", "sim", "central_train"
        # )

        # with get_time():
        # params = dict(model.named_parameters())

        eval_result = simple_evaluator(model, test_loader, cfg.train)

        metrics["loss"]["eval"]["server"] = eval_result["loss"]
        metrics["accuracy"]["eval"]["server"] = eval_result["accuracy"]
        logger.info(f"SERVER EVAL: {eval_result}")
        if cfg.use_wandb:
            flat = json_normalize(metrics, sep="/").to_dict(orient="records")[0]
            wandb.log(flat, step=step_count, commit=True)
            step_count += 1

        if curr_round % cfg.checkpoint_every == 0:
            torch.save(
                {
                    "round": curr_round,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"ckpts/server/ckpt_r{curr_round:003}.pt",
            )

        loop_end = time.time() - loop_start
        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------"
        )

    return eval_result
