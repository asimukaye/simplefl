from copy import deepcopy
from dataclasses import dataclass
import time
import logging
import random
from itertools import combinations

from pandas import json_normalize
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.nn import Module, CrossEntropyLoss, Parameter
from torch.optim import Optimizer
from torch.nn import functional as F
from tqdm import tqdm
import wandb

from utils import (
    generate_client_ids,
    make_client_checkpoint_dirs,
    make_server_checkpoint_dirs,
    find_client_checkpoint,
    find_server_checkpoint,
    load_server_checkpoint,
    save_checkpoint,
)
from data import DatasetPair
from split import get_client_datasets
from rootconfig import TrainConfig, Config
from fedavg import simple_evaluator, simple_trainer

# from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class ShapfedConfig(Config):
    name: str = "shapfed"
    alpha: float = 0.95
    gamma: float = 0.15
    compute_every: int = 1


class ShapfedClient:
    def __init__(
        self,
        cid: str,
        dataset: DatasetPair,
        model: Module,
        train_cfg: TrainConfig,
    ):
        self.dataset = dataset

        self.model = deepcopy(model)
        self.model.to(train_cfg.device)
        self.cid = cid
        self.client_id = int(cid)
        self.tr_cfg = deepcopy(train_cfg)
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
        initial_weights = deepcopy(self.get_weights())

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

        final_weights = self.get_weights()
        self.compute_full_grad(initial_weights, final_weights)

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

    def get_weights(self) -> dict[str, Parameter]:
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def compute_full_grad(self, initial_weights, final_weights):
        self.full_grads = {
            name: (1 / self.optimizer.param_groups[0]["lr"])
            * (initial_weights[name] - final_weights[name])
            for name in initial_weights
        }

    def get_gradients(self):
        return self.full_grads


# Server Class
class Server:
    def __init__(self, clients: list[ShapfedClient], model: Module, tcfg: TrainConfig):
        self.model = model
        self.model.to(tcfg.device)
        self.clients = clients
        self.tcfg = tcfg

        self.stored_grads = None

    def aggregate_gradients(self, coefficients):
        total_grads = None

        for client_id, client in enumerate(self.clients):
            client_grads = client.get_gradients()

            if total_grads is None:
                total_grads = {
                    name: torch.zeros_like(grad) for name, grad in client_grads.items()
                }

            for name, grad in client_grads.items():
                total_grads[name] += coefficients[client_id] * grad

        # Store the aggregated gradients
        self.stored_grads = total_grads

    def aggregate(self, coefficients):
        total_weights = {}

        for client_id, client in enumerate(self.clients):
            client_weights = client.get_weights()

            if not total_weights:
                total_weights = {
                    name: torch.zeros_like(param)
                    for name, param in client_weights.items()
                }

            for name, param in client_weights.items():
                if total_weights[name].dtype == torch.float32:
                    total_weights[name] += coefficients[client_id] * param.to(
                        torch.float32
                    )
                elif total_weights[name].dtype == torch.int64:
                    total_weights[name] += int(coefficients[client_id]) * param.to(
                        torch.int64
                    )

        prev_weights = self.model.state_dict()
        eta = 1.0
        for name, param in total_weights.items():
            prev_weights[name] = (1 - eta) * prev_weights[name] + eta * param

        self.model.load_state_dict(prev_weights)

    def broadcast(self, coefficients):
        for client in self.clients:
            for global_param, client_param in zip(
                self.model.parameters(), client.model.parameters()
            ):
                # personalization
                client_param.data = (
                    1 - coefficients[client.client_id]
                ) * client_param.data + coefficients[
                    client.client_id
                ] * global_param.data

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        criterion = CrossEntropyLoss()

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.tcfg.device), target.to(self.tcfg.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss += criterion(output, target).item()

        accuracy = 100 * correct / total
        avg_loss = loss / len(dataloader)
        return accuracy, avg_loss


def compute_cssv_cifar(clients: list[ShapfedClient], weights, original_model: Module):
    n = len(clients)
    tcfg = deepcopy(clients[0].tr_cfg)
    num_classes = 10  # clients[0].model.state_dict()['linear.weight'].shape[0]
    similarity_matrix = torch.zeros((n, num_classes))  # One similarity value per class

    # weight_layer_name = "linear.weight"
    # bias_layer_name = "linear.bias"

    weight_layer_name = "fc3.weight"
    bias_layer_name = "fc3.bias"

    subsets = [subset for subset in combinations(range(n), n)]
    for subset in subsets:
        # Create a temporary server for this subset
        subset_clients = [clients[i] for i in subset]
        curr_weights = [weights[j] for j in subset]
        # normalized_curr_weights = softmax(curr_weights)  # curr_weights / np.sum(curr_weights)
        normalized_curr_weights = curr_weights / np.sum(curr_weights)

        temp_server = Server(subset_clients, original_model, tcfg=tcfg)
        temp_server.aggregate(coefficients=normalized_curr_weights)
        temp_server.aggregate_gradients(coefficients=normalized_curr_weights)

        for cls_id in range(num_classes):
            # Use gradients instead of weights and biases
            w1_grad = torch.cat(
                [
                    temp_server.stored_grads[weight_layer_name][cls_id].view(-1),
                    temp_server.stored_grads[bias_layer_name][cls_id].view(-1),
                ]
            ).view(1, -1)

            w1_grad = F.normalize(w1_grad, p=2)

            for client_id in range(len(subset)):
                w2_grad = torch.cat(
                    [
                        subset_clients[client_id]
                        .get_gradients()[weight_layer_name][cls_id]
                        .view(-1),
                        subset_clients[client_id]
                        .get_gradients()[bias_layer_name][cls_id]
                        .view(-1),
                    ]
                ).view(1, -1)
                w2_grad = F.normalize(w2_grad, p=2)

                # Compute cosine similarity with gradients
                sim = F.cosine_similarity(w1_grad, w2_grad).item()
                similarity_matrix[client_id][cls_id] = sim

    shapley_values = torch.mean(similarity_matrix, dim=1).numpy()
    return shapley_values, similarity_matrix


def run_shapfed(dataset: DatasetPair, model: Module, cfg: ShapfedConfig):

    global_model = deepcopy(model)
    global_model.to(cfg.train.device)
    global_model.eval()
    global_model.zero_grad()

    client_ids = generate_client_ids(cfg.num_clients)

    client_datasets = get_client_datasets(cfg.split, dataset)
    shapley_values = None
    mu = 0.5
    ## Create Clients
    # clients: dict[str, ShapfedClient] = {}
    clients: list[ShapfedClient] = []
    # NOTE: IMPORTANT Sharing models without deepcopy could potentially have same references to parameters
    # for cid, dataset in zip(client_ids, client_datasets):
    #     clients[cid] = ShapfedClient(
    #         train_cfg=cfg.train,
    #         cid=cid,
    #         dataset=dataset,
    #         model=deepcopy(model),
    #     )

    # backup_epoch
    for cid, dataset in zip(client_ids, client_datasets):
        clients.append(
            ShapfedClient(
                train_cfg=cfg.train,
                cid=cid,
                dataset=dataset,
                model=deepcopy(model),
            )
        )
    if cfg.resumed:
        server_ckpt = find_server_checkpoint()
        start_round = load_server_checkpoint(server_ckpt, global_model)
        # for cid, client in clients.items():
        for cid, client in zip(client_ids, clients):
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

    server = Server(clients, global_model, tcfg=cfg.train)
    # Fedavg weights
    # data_sizes = [clients[cid].data_size for cid in client_ids]
    data_sizes = [client.data_size for client in clients]
    total_size = sum(data_sizes)
    weights = [1 / cfg.num_clients for _ in range(cfg.num_clients)]

    logger.info(f"Client data sizes: {data_sizes}")

    # Define relevant x axes for logging

    step_count = 0
    total_epochs = 0
    phase_count = 0

    # Define metrics to log

    metrics = {
        "loss": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "accuracy": {"train": {}, "eval": {}, "eval_pre": {}, "eval_post": {}},
        "round": 0,
        "weights": {},
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
        for cid, client in zip(client_ids, clients):
            if curr_round == 0:
                client.tr_cfg.epochs = 5
            else:
                client.tr_cfg.epochs = cfg.train.epochs
            train_results[cid] = client.train(curr_round)

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
        for cid, client in zip(client_ids, clients):
            eval_result_pre = client.evaluate()
            metrics["loss"]["eval"][cid] = eval_result_pre["loss"]
            metrics["loss"]["eval_pre"][cid] = eval_result_pre["loss"]
            metrics["accuracy"]["eval"][cid] = eval_result_pre["accuracy"]
            metrics["accuracy"]["eval_pre"][cid] = eval_result_pre["accuracy"]

        for metric in ["loss", "accuracy"]:
            m_list = [metrics[metric]["eval"][cid] for cid in client_ids]
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

        # clients_deltas: dict[str, list[Tensor]] = {}

        # for cid, client in zip(client_ids, clients):
        #     cdelta = []
        #     for cparam, gparam in zip(
        #         client.model.parameters(), global_model.parameters()
        #     ):
        #         delta = cparam.data - gparam.data
        #         cdelta.append(delta)

        #     clients_deltas[cid] = cdelta

        ##  Testing

        # aggregated_delta = [
        #         torch.zeros(param.shape).to(cfg.train.device)
        #         for param in global_model.parameters()
        #     ]
        # Aggregate
        if curr_round % cfg.compute_every == 0:  # every round
            # Compute Shapley values for each client every 5 rounds [to be efficient]
            temp_shapley_values, temp_class_shapley_values = compute_cssv_cifar(
                clients, weights, global_model
            )
            if shapley_values is None:
                shapley_values = np.array(temp_shapley_values)
                class_shapley_values = np.array(temp_class_shapley_values)
            else:
                shapley_values = mu * shapley_values + (1 - mu) * temp_shapley_values
                class_shapley_values = mu * class_shapley_values + (1 - mu) * np.array(
                    temp_class_shapley_values
                )

            # normalized_shapley_values = softmax(shapley_values)  # shapley_values / np.sum(shapley_values)
            normalized_shapley_values = shapley_values / np.sum(shapley_values)
            broadcast_normalized_shapley_values = shapley_values / np.max(
                shapley_values
            )

        print(shapley_values, normalized_shapley_values)
        print(class_shapley_values)

        weights = normalized_shapley_values
        server.aggregate(coefficients=weights)

        # if one wants to allow the highest-contribution participant to receive full update
        # server.broadcast(coefficients = broadcast_normalized_shapley_values)

        server.broadcast(coefficients=weights)

        ### CLIENTS EVALUATE post aggregation###
        eval_ids = client_ids
        for cid, client in zip(client_ids, clients):
            eval_result_post = client.evaluate()

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
