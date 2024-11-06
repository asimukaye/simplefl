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
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
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
from config import TrainConfig, FHGConfig
from fedavg import simple_evaluator, simple_trainer
from cgsv import flatten, unflatten

logger = logging.getLogger(__name__)


def model_mean_std(
    target_model: Module, input_models: list[Module]
) -> tuple[Module, Tensor]:
    flat_list = []
    for inmodel in input_models:
        flat_model = flatten(inmodel.parameters())
        flat_list.append(flat_model)
    flat_std, flat_mean = torch.std_mean(torch.stack(flat_list), dim=0)
    # target_model = unflatten(flat_mean, target_model.parameters())
    vector_to_parameters(flat_mean, target_model.parameters())
    return target_model, flat_std


class FHGClient:
    def __init__(
        self,
        cid: str,
        dataset: DatasetPair,
        model: Module,
        train_cfg: TrainConfig,
        branches: list[int],
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
        self.flat_model_std = torch.zeros_like(
            parameters_to_vector(self.model.parameters())
        )
        self.branches = branches
        self.start_epoch = 0

    def train(self, _round: int) -> tuple[dict, Tensor]:

        results = {}

        prev_models = [deepcopy(self.model)]
        # for epoch in range(self.tr_cfg.epochs):
        for branch_level, num_branches in enumerate(self.branches):

            # r_list = []
            for prev_m in prev_models:

                r_list_loss = []
                r_list_acc = []
                new_models = []
                models = [deepcopy(prev_m) for _ in range(num_branches)]
                optimizers = [self.tr_cfg.optim_partial(m.parameters()) for m in models]
                for model, optim in zip(models, optimizers):
                    result = simple_trainer(
                        model, self.train_loader, self.tr_cfg, optim
                    )
                    r_list_loss.append(result["loss"])
                    r_list_acc.append(result["accuracy"])
                    new_models.append(model)

            # ic("Branch", branch_level)
            # ic(len(new_models))
            prev_models = new_models
            results[branch_level] = {
                "loss": np.mean(r_list_loss),
                "accuracy": np.mean(r_list_acc),
            }

            logger.info(f"TRAIN level {branch_level}: {results[branch_level]}")
            # logger.info(f"TRAIN {epoch}: {result}")
            results[branch_level] = result

        self.model, self.flat_model_std = model_mean_std(self.model, new_models)

        return results, self.flat_model_std

    def evaluate(self):
        result = simple_evaluator(self.model, self.test_loader, self.tr_cfg)
        return result

    def load_checkpoint(self, ckpt: str, root_dir="."):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # _round = checkpoint["round"]
        self.start_epoch = checkpoint["epoch"]


def run_fedhigrad(dataset: DatasetPair, model: Module, cfg: FHGConfig):

    global_model = model
    global_model.to(cfg.train.device)
    global_model.eval()
    global_model.zero_grad()

    client_ids = generate_client_ids(cfg.num_clients)

    client_datasets = get_client_datasets(cfg.split, dataset)

    ## Create Clients
    clients: dict[str, FHGClient] = {}
    # NOTE: IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    for cid, dataset in zip(client_ids, client_datasets):
        clients[cid] = FHGClient(
            train_cfg=cfg.train,
            cid=cid,
            dataset=dataset,
            model=deepcopy(model),
            branches=cfg.branches,
        )

    # Checkpointing
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
    shard_sizes = [clients[cid].data_size for cid in client_ids]
    total_size = sum(shard_sizes)
    logger.info(f"Client data sizes: {shard_sizes}")

    shard_sizes = torch.tensor(shard_sizes).float()
    relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))



    # Define relevant x axes for logging

    step_count = 0
    total_epochs = 0
    phase_count = 0

    weights = torch.div(shard_sizes, torch.sum(shard_sizes)).to(cfg.train.device)

    # rs = torch.zeros(cfg.num_clients, device=cfg.train.device)
    rs = torch.clone(weights)
    phis = torch.zeros_like(rs)

    
    # Define metrics to log

    metrics = {
        "loss": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "accuracy": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "round": 0,
        "weights": {},
        "phis": {},
        "rs": {},
        "total_epochs": total_epochs,
        "phase": phase_count,
    }
    for i, cid in enumerate(client_ids):
        metrics["weights"][cid] = weights[i].item()
        metrics["phis"][cid] = phis[i].item()
        metrics["rs"][cid] = rs[i].item()

    ## Start Training
    for curr_round in range(start_round, cfg.num_rounds):
        logger.info(f"-------- Round: {curr_round} --------\n")

        loop_start = time.time()

        metrics["round"] = curr_round
        #### CLIENTS TRAIN ####
        # select all clients
        train_ids = client_ids

        train_results = {}
        model_std = {}
        for cid in train_ids:
            train_results[cid], model_std[cid] = clients[cid].train(curr_round)

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

        ### CLIENTS EVALUATE local performance before aggregation ###
        for cid in train_ids:
            eval_result_pre = clients[cid].evaluate()
            metrics["loss"]["eval"][cid] = eval_result_pre["loss"]
            metrics["loss"]["eval_pre"][cid] = eval_result_pre["loss"]
            metrics["accuracy"]["eval"][cid] = eval_result_pre["accuracy"]
            metrics["accuracy"]["eval_pre"][cid] = eval_result_pre["accuracy"]

        for metric in ["loss", "accuracy"]:
            m_list = [metrics[metric]["eval"][cid] for cid in client_ids]
            # mean = sum(m_list) / len(m_list)
            mean = np.mean(m_list)
            metrics[metric]["eval"]["mean"] = mean
            metrics[metric]["eval_pre"]["mean"] = mean
            logger.info(f"CLIENT EVAL mean {metric}: {mean}")

        metrics["phase"] = phase_count
        phase_count += 1

        if cfg.use_wandb:
            flat = json_normalize(metrics, sep="/").to_dict(orient="records")[0]
            wandb.log(flat, step=step_count, commit=False)
            step_count += 1

        #### AGGREGATE ####

        clients_deltas: dict[str, list[Tensor]] = {}

        for cid, client in clients.items():
            cdelta = []
            for cparam, gparam in zip(
                client.model.parameters(), global_model.parameters()
            ):
                delta = cparam.data - gparam.data
                cdelta.append(delta)

            clients_deltas[cid] = cdelta

        if cfg.enable_weights:
            if curr_round == 0:
                weights = torch.div(shard_sizes, torch.sum(shard_sizes))
            else:
                weights = rs

            if cfg.phi_method == "mean":
                stacked = torch.stack(list(model_std.values()))
                # ic(stacked.shape)
                # start = time.time()
                sigma_rms = torch.pow(torch.mean(stacked.square(), dim=1), 0.5)
                inv_sigma = torch.div(1, sigma_rms+1e-8)
                phis = torch.div(inv_sigma, torch.sum(inv_sigma))
                
                # ic("time 1", time.time() - start)
                # ic("Mean inv sigma", phis)

                # start2 = time.time()

                # norm = stacked.norm(dim=1)
                # inv_norm = torch.div(1, norm+1e-8)
                # norm_phis = torch.div(inv_norm, torch.sum(inv_norm))
                # ic("time 2", time.time() - start2)

            elif cfg.phi_method == "norm":
                stacked = torch.stack(list(model_std.values()))
                norm = stacked.norm(dim=1)
                inv_norm = torch.div(1, norm+1e-8)
                phis = torch.div(inv_norm, torch.sum(inv_norm))
                # ic("Norm inv sigmas", phis)

            elif cfg.phi_method == "per_parameter":
                pass
            elif cfg.phi_method == "per_layer":
                pass
            elif cfg.phi_method == "":
                pass
            else:
                raise ValueError("Invalid phi method")

            rs = cfg.alpha * rs + (1 - cfg.alpha) * phis
            rs = torch.div(rs, rs.sum())

            # ic("RS", rs)

        for k, gparam in enumerate(global_model.parameters()):
            temp_delta = torch.zeros_like(gparam.data)
            # for deltas in clients_deltas.values():
            for c, deltas in enumerate(clients_deltas.values()):
                temp_delta.add_(weights[c] * deltas[k].data)
            gparam.data.add_(temp_delta)

        # server_optimizer.step()
        # server_scheduler.step()

        ### send the global model to all clients
        for cid, client in clients.items():
            for cparam, gparam in zip(
                client.model.parameters(), global_model.parameters()
            ):
                cparam.data.copy_(gparam.data)
            
        for i, cid in enumerate(client_ids):
            metrics["weights"][cid] = weights[i].item()
            metrics["phis"][cid] = phis[i].item()
            metrics["rs"][cid] = rs[i].item()


        ### CLIENTS EVALUATE post aggregation###
        eval_ids = client_ids
        for cid in eval_ids:
            eval_result_post = clients[cid].evaluate()

            metrics["loss"]["eval"][cid] = eval_result_post["loss"]
            metrics["loss"]["eval_post"][cid] = eval_result_post["loss"]
            metrics["accuracy"]["eval"][cid] = eval_result_post["accuracy"]
            metrics["accuracy"]["eval_post"][cid] = eval_result_post["accuracy"]

        for metric in ["loss", "accuracy"]:
            m_list = [metrics[metric]["eval"][cid] for cid in client_ids]
            mean = np.mean(m_list)
            metrics[metric]["eval"]["mean"] = mean
            metrics[metric]["eval_post"]["mean"] = mean
            logger.info(f"CLIENT EVAL post mean {metric}: {mean}")

        # SERVER EVALUATE
        server_result = simple_evaluator(global_model, server_loader, cfg.train)

        metrics["loss"]["eval"]["server"] = server_result["loss"]
        metrics["accuracy"]["eval"]["server"] = server_result["accuracy"]
        metrics["phase"] = phase_count
        phase_count += 1

        logger.info(f"SERVER EVAL: {server_result}")
        if cfg.use_wandb:
            flat = json_normalize(metrics, sep="/").to_dict(orient="records")[0]
            wandb.log(flat, step=step_count, commit=True)
            step_count += 1

        if curr_round % cfg.checkpoint_every == 0:
            save_checkpoint(curr_round, global_model, server_optimizer, "server")

        loop_end = time.time() - loop_start


        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------\n"
        )
    torch.save(global_model.state_dict(), f"final_model.pt")