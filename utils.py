import logging
import os
import glob
import torch.backends
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
import torch
import subprocess
import pandas as pd
from io import StringIO
from sklearn.metrics import accuracy_score
## Root level module. Should not have any dependencies on other modules


def get_free_gpus(min_memory_reqd=4096):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    # min_memory_reqd = 10000
    ids = gpu_df.index[gpu_df["memory.free"] > min_memory_reqd]
    for id in ids:
        logging.debug(
            "Returning GPU:{} with {} free MiB".format(
                id, gpu_df.iloc[id]["memory.free"]
            )
        )
    return ids.to_list()


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode()), names=["memory.used", "memory.free"], skiprows=1
    )
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    logging.debug("Returning GPU:{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"]))  # type: ignore
    return idx


def auto_configure_device():

    if torch.cuda.is_available():
        # Set visible GPUs
        # TODO: MAke the gpu configurable
        gpu_ids = get_free_gpus()
        # logging.info('Selected GPUs:')
        logging.info("Selected GPUs:" + ",".join(map(str, gpu_ids)))

        # Disabling below line due to cluster policies
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        if torch.cuda.device_count() > 1:
            device = f"cuda:{get_free_gpu()}"
        else:
            device = "cuda"

    else:
        device = "cpu"
    logging.info(f"Auto Configured device to: {device}")
    return device


def generate_client_ids(num_clients):
    return [f"{idx:04}" for idx in range(num_clients)]


#### Checkpointing functions


def make_client_checkpoint_dirs(root_dir=".", client_ids=[]):
    # os.makedirs(f"{}/client_ckpts", exist_ok=True)
    for cid in client_ids:
        os.makedirs(f"ckpts/c_{cid}")


def make_server_checkpoint_dirs(root_dir="."):
    os.makedirs("ckpts/server")


def checkpoint_dirs_exist(root_dir=".") -> bool:
    return os.path.exists(root_dir + "/server_ckpts") or os.path.exists(
        root_dir + "/client_ckpts"
    )


def find_server_checkpoint(root_dir=".") -> str:
    server_ckpts = sorted(glob.glob(root_dir + "/ckpts/server/ckpt_*"))
    if server_ckpts:
        logging.info(f"------ Found server checkpoint: {server_ckpts[-1]} ------")
        return server_ckpts[-1]
    else:
        # logging.debug("------------ No server checkpoint found. ------------")
        raise FileNotFoundError("No server checkpoint found")


def find_client_checkpoint(client_id: str, root_dir=".") -> str:
    client_ckpts = sorted(glob.glob(f"{root_dir}/ckpts/c_{client_id}/ckpt_*"))
    if client_ckpts:
        logging.info(
            f"------ Found client {client_id} checkpoint: {client_ckpts[-1]} ------"
        )
        return client_ckpts[-1]
    else:
        logging.warning(f"------------ No client {client_id} checkpoint found. Use global checkpoints ------------")

        return ""


def save_checkpoint(
    _round: int,
    model: Module,
    optimizer: Optimizer,
    suffix="server",
    epoch=0,
    root_dir=".",
):

    torch.save(
        {
            "epoch": epoch,
            "round": _round,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{root_dir}/ckpts/{suffix}/ckpt_r{_round:003}_e{epoch:003}.pt",
    )


def load_server_checkpoint(ckpt, global_model: Module) -> int:

    global_ckpt = torch.load(ckpt)
    global_model.load_state_dict(global_ckpt["model_state_dict"])
    _round = global_ckpt["round"]

    return _round


def get_accuracy(outputs: Tensor, targets: Tensor):
    """Calculate accuracy from outputs and targets.
    Args:
        outputs (Tensor): model outputs
        targets (Tensor): target labels
    Returns:
        accuracy (float): accuracy value"""
    preds = outputs.argmax(dim=1)
    acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
    return acc
