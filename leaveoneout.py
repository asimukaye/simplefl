from copy import deepcopy
import time
import logging
import random
from pandas import json_normalize
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run
from utils import (
    generate_client_ids,
    make_client_checkpoint_dirs,
    make_server_checkpoint_dirs,
    find_client_checkpoint,
    find_server_checkpoint,
    load_server_checkpoint,
    save_checkpoint,
    get_accuracy,
)
from data import DatasetPair
from split import get_client_datasets
from config import TrainConfig, Config
from fedavg import simple_trainer, simple_evaluator

logger = logging.getLogger(__name__)

class Client:
    def __init__(
        self,
        cid: str,
        dataset: DatasetPair,
        model: Module,
        train_cfg: TrainConfig,
    ):
        self.dataset = dataset

        self.model = model
        self.cid = cid
        self.tr_cfg = train_cfg
        self.optimizer = self.tr_cfg.optim_partial(
            self.model.parameters(), lr=self.tr_cfg.lr
        )
        self.data_size = len(self.dataset.train)
        self.test_size = len(self.dataset.test)
        self.train_loader = DataLoader(
            self.dataset.train, batch_size=self.tr_cfg.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.dataset.test, batch_size=self.tr_cfg.eval_batch_size, shuffle=False
        )
        self.start_epoch = 0

    def train(self, _round: int) -> dict:

        results = {}
        for epoch in range(self.tr_cfg.epochs):
            result = simple_trainer(
                self.model, self.train_loader, self.tr_cfg, self.optimizer
            )

            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                save_checkpoint(
                    _round, self.model, self.optimizer, f"c_{self.cid}", epoch
                )
            # logger.info(f"TRAIN {epoch}: {result}")
            results[epoch] = result

        return results

    def evaluate(self):
        result = simple_evaluator(self.model, self.test_loader, self.tr_cfg)
        # logger.info(f"EVAL: {result}")

        return result

    def load_checkpoint(self, ckpt: str, root_dir="."):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # _round = checkpoint["round"]
        self.start_epoch = checkpoint["epoch"]


def run_standolone(dataset: DatasetPair, model: Module, cfg: Config):

    global_model = model
    global_model.to(cfg.train.device)
    global_model.eval()
    global_model.zero_grad()

    client_ids = generate_client_ids(cfg.num_clients)

    client_datasets = get_client_datasets(cfg.split, dataset)

    ## Create Clients
    clients: dict[str, Client] = {}
    # NOTE: IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    for cid, dataset in zip(client_ids, client_datasets):
        clients[cid] = Client(
            train_cfg=cfg.train,
            cid=cid,
            dataset=dataset,
            model=deepcopy(model),
        )

    if cfg.resumed:
        server_ckpt = find_server_checkpoint()
        start_round = load_server_checkpoint(server_ckpt, global_model)
        for cid, client in clients.items():
            ckpt = find_client_checkpoint(cid)
            if ckpt:
                client.load_checkpoint(ckpt)
            else:
                client.load_checkpoint(server_ckpt)
    else:
        make_server_checkpoint_dirs()
        make_client_checkpoint_dirs(client_ids=client_ids)
        start_round = 0

    ## Create Server Loader
    server_loader = DataLoader(
        dataset=dataset.test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )
    server_optimizer = cfg.train.optim_partial(
        global_model.parameters(), lr=cfg.train.lr
    )
    server_scheduler = cfg.train.scheduler_partial(server_optimizer)

    # Fedavg weights
    data_sizes = [clients[cid].data_size for cid in client_ids]
    total_size = sum(data_sizes)
    weights = {cid: size / total_size for cid, size in zip(client_ids, data_sizes)}

    logger.info(f"Client data sizes: {data_sizes}")

    # Define relevant x axes for logging

    step_count = 0
    total_epochs = 0
    phase_count = 0
    # Define metrics to log

    metrics = {
        "loss": {"train": {}, "eval": {}, "eval_global": {}},
        "accuracy": {"train": {}, "eval": {}, "eval_global": {}},
        "round": 0,
        "normalized_accuracy": {},
        "normalized_inv_loss": {},
        "total_epochs": total_epochs,
        "phase": phase_count,
    }

    ## Start Training
    for curr_round in range(start_round, cfg.num_rounds):
        logger.info(f"-------- Round: {curr_round} --------\n")

        loop_start = time.time()

        metrics["round"] = curr_round
        #### CLIENTS TRAIN ####
        # select all clients
        train_ids = client_ids

        train_results = {}
        for cid in train_ids:
            train_results[cid] = clients[cid].train(curr_round)

        for epoch in range(cfg.train.epochs):
            for cid in train_ids:
                result = train_results[cid][epoch]

                metrics["loss"]["train"][cid] = result["loss"]
                metrics["accuracy"]["train"][cid] = result["accuracy"]

            for metric in ["loss", "accuracy"]:
                m_list = [metrics[metric]["train"][cid] for cid in client_ids]
                metrics[metric]["train"]["mean"] = sum(m_list) / len(m_list)
                logger.info(
                    f"CLIENT TRAIN mean {metric}: {metrics[metric]['train']['mean']}"
                )

            metrics["total_epochs"] = total_epochs
            total_epochs += 1
            if cfg.use_wandb:
                flat = json_normalize(metrics, sep="/").to_dict(orient="records")[0]
                wandb.log(flat, step=step_count, commit=False)
                step_count += 1

        ### CLIENTS EVALUATE###
        # eval_ids = client_ids
        # for cid in eval_ids:
        #     eval_result_post = clients[cid].evaluate()
        #     metrics["loss"]["eval"][cid] = eval_result_post["loss"]
        #     metrics["accuracy"]["eval"][cid] = eval_result_post["accuracy"]

        # for metric in ["loss", "accuracy"]:
        #     m_list = list(metrics[metric]["eval"].values())
        #     mean = np.mean(m_list)
        #     metrics[metric]["eval"]["mean"] = mean
        #     logger.info(f"CLIENT EVAL post mean {metric}: {mean}")

        # Global data EVALUATE
        for cid, client in clients.items():
            global_result = simple_evaluator(client.model, server_loader, cfg.train)
            metrics["loss"]["eval_global"][cid] = global_result["loss"]
            metrics["accuracy"]["eval_global"][cid] = global_result["accuracy"]

        for metric in ["loss", "accuracy"]:
            m_array = np.array([metrics[metric]["eval_global"][cid] for cid in client_ids])
            if metric == "loss":
                inv_m_array = 1/(m_array+1e-6)
                norm_inv_arr = inv_m_array/inv_m_array.sum()
                metrics["normalized_inv_loss"] = {cid: norm_inv_arr[i] for i, cid in enumerate(client_ids)}
            else:
                norm_arr = m_array/m_array.sum()
                metrics["normalized_accuracy"] = {cid: norm_arr[i] for i, cid in enumerate(client_ids)}

            mean = m_array.mean()
            metrics[metric]["eval_global"]["mean"] = mean
            logger.info(f"CLIENT Global EVAL post {metric}: {mean}")

        # ic(metrics)

        if cfg.use_wandb:
            flat = json_normalize(metrics, sep="/").to_dict(orient="records")[0]
            wandb.log(flat, step=step_count, commit=True)
            step_count += 1

        if curr_round % cfg.checkpoint_every == 0:
            # save_checkpoint(curr_round, global_model, server_optimizer, "server")
            for cid, client in clients.items():
                save_checkpoint(curr_round, client.model, client.optimizer, f"c_{cid}")

            # save_checkpoint(curr_round, global_model, server_optimizer, "server", True)

        loop_end = time.time() - loop_start

        # ic("Post", metrics["accuracy"]["eval"]["mean"], metrics["phase"])

        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------\n"
        )
    torch.save(global_model.state_dict(), f"final_model.pt")
