from copy import deepcopy
import time
import logging
import math
import random
from pandas import json_normalize
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm
from torch.linalg import norm
import torch.nn.functional as F

import wandb
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
from config import TrainConfig, CGSVConfig
from fedavg import simple_trainer, simple_evaluator

logger = logging.getLogger(__name__)


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
    assert len(grad_update_1) == len(
        grad_update_2
    ), "Lengths of the two grad_updates not equal"

    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight


def add_update_to_model(model, update, weight=1.0, device=None):
    if not update:
        return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False  # two models have different weights
    return True


def sign(grad):
    return [torch.sign(update) for update in grad]


def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]

    return grad_update


def l2norm(grad):
    return torch.sqrt(torch.sum(torch.pow(flatten(grad), 2)))


def cosine_similarity(grad1, grad2, normalized=False):
    """
    Input: two sets of gradients of the same shape
    Output range: [-1, 1]
    """

    cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10)
    if normalized:
        return (cos_sim + 1) / 2.0
    else:
        return

def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

    if mode == 'all':
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat([update.data.view(-1).abs()
                                    for update in grad_update])
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)
        
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order) # type: ignore
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == 'layer': # layer wise largest-values criterion
        grad_update = deepcopy(grad_update)

        mask_percentile = max(0, mask_percentile) # type: ignore
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, 
                               min(mask_order, len(layer_mod)-1)) # type: ignore
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update
    

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
            self.dataset.test, batch_size=self.tr_cfg.eval_batch_size, shuffle=True
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


def run_cgsv(dataset: DatasetPair, in_model: Module, cfg: CGSVConfig, resumed=False):

    global_model = in_model
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
            model=deepcopy(in_model),
        )

    if resumed:
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
    shard_sizes = torch.tensor(shard_sizes).float()
    relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))
    weights_log = {cid: rel_size for cid, rel_size in zip(client_ids, relative_shard_sizes)}
    D = sum([p.numel() for p in global_model.parameters()])
    rs_list = []
    rs = torch.zeros(cfg.num_clients, device=cfg.train.device)
    past_phis = []
    qs_list = []

    logger.info(f"Client data sizes: {shard_sizes}")

    # Define relevant x axes for logging

    step_count = 0
    total_epochs = 0
    phase_count = 0
    # Define metrics to log

    metrics = {
        "loss": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "accuracy": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "round": 0,
        "weights": weights_log,
        "q_ratios": weights_log,
        "rs": weights_log,
        "phis": weights_log,
        "total_epochs": total_epochs,
        "phase": phase_count,
    }

    ## Start Training
    for curr_round in range(start_round, cfg.num_rounds):
        logger.info(f"-------- Round: {curr_round} --------\n")

        loop_start = time.time()

        metrics["round"] = curr_round

        gradients = []

        #### CLIENTS TRAIN ####
        # select all clients
        train_ids = client_ids

        train_results = {}
        for cid in train_ids:
            client = clients[cid]
            backup = deepcopy(client.model)

            train_results[cid] = clients[cid].train(curr_round)

            gradient = compute_grad_update(
                old_model=backup, new_model=client.model, device=cfg.train.device
            )

            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7  # to prevent division by zero
            if norm_value > cfg.gamma:
                gradient = unflatten(
                    torch.multiply(
                        torch.tensor(cfg.gamma), torch.div(flattened, norm_value)
                    ),
                    gradient,
                )

                client.model.load_state_dict(backup.state_dict())
                add_update_to_model(client.model, gradient, device=cfg.train.device)

            gradients.append(gradient)

        for epoch in range(cfg.train.epochs):
            for cid in train_ids:
                result = train_results[cid][epoch]

                metrics["loss"]["train"][cid] = result["loss"]
                metrics["accuracy"]["train"][cid] = result["accuracy"]

            for metric in ["loss", "accuracy"]:
                m_list = list(metrics[metric]["train"].values())
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
            m_list = list(metrics[metric]["eval"].values())
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

        aggregated_gradient = [
            torch.zeros(param.shape).to(cfg.train.device)
            for param in global_model.parameters()
        ]

        if not cfg.use_reputation:
            # fedavg
            for gradient, weight in zip(gradients, relative_shard_sizes):
                add_gradient_updates(aggregated_gradient, gradient, weight=weight.item())

        else:
            if curr_round == 0:
                weights = torch.div(shard_sizes, torch.sum(shard_sizes))
            else:
                weights = rs

            for gradient, weight in zip(gradients, weights):
                add_gradient_updates(aggregated_gradient, gradient, weight=weight.item())


            flat_aggre_grad = flatten(aggregated_gradient)

            phis = torch.zeros(cfg.num_clients, device=cfg.train.device)
            for i, gradient in enumerate(gradients):
                phis[i] = F.cosine_similarity(
                    flatten(gradient), flat_aggre_grad, 0, 1e-10
                )

            past_phis.append(phis)

            rs = cfg.alpha * rs + (1 - cfg.alpha) * phis
            rs = torch.div(rs, rs.sum())

            # r_threshold.append(threshold * (1.0 / len(R_set)))
            q_ratios = torch.div(rs, torch.max(rs))

            rs_list.append(rs)
            qs_list.append(q_ratios)

        #THIS LINE WAS MISSING IN RFFL
        # update the global model
        add_update_to_model(global_model, aggregated_gradient, device=cfg.train.device)

        for i, cid in enumerate(client_ids):

            if cfg.use_sparsify and cfg.use_reputation:

                q_ratio = q_ratios[i]
                reward_gradient = mask_grad_update_by_order(
                    aggregated_gradient, mask_percentile=q_ratio, mode="layer"
                )

            elif cfg.use_sparsify and not cfg.use_reputation:

                # relative_shard_sizes[i] the relative dataset weight of the local dataset
                reward_gradient = mask_grad_update_by_order(
                    aggregated_gradient,
                    mask_percentile=relative_shard_sizes[i],
                    mode="layer",
                )

            else:  # not use_sparsify
                # the reward_gradient is the whole gradient
                reward_gradient = aggregated_gradient

            add_update_to_model(clients[cid].model, reward_gradient)

            metrics["q_ratios"][cid] = q_ratios[i].item()
            metrics["weights"][cid] = weights[i].item()
            metrics["phis"][cid] = phis[i].item()
            metrics["rs"][cid] = rs[i].item()


        # server_optimizer.step()
        # server_scheduler.step()

        ### CLIENTS EVALUATE post aggregation###
        eval_ids = client_ids
        for cid in eval_ids:
            eval_result_post = clients[cid].evaluate()

            metrics["loss"]["eval"][cid] = eval_result_post["loss"]
            metrics["loss"]["eval_post"][cid] = eval_result_post["loss"]
            metrics["accuracy"]["eval"][cid] = eval_result_post["accuracy"]
            metrics["accuracy"]["eval_post"][cid] = eval_result_post["accuracy"]

        for metric in ["loss", "accuracy"]:
            m_list = list(metrics[metric]["eval"].values())
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

        # ic("Post", metrics["accuracy"]["eval"]["mean"], metrics["phase"])

        logger.info(
            f"------------ Round {curr_round} completed in time: {loop_end} ------------\n"
        )
    torch.save(global_model.state_dict(), f"final_model.pt")
