from dataclasses import dataclass
from flamby.datasets.fed_isic2019 import FedIsic2019, Isic2019Raw
import albumentations
import os, random
import logging
from torch.utils.data import Subset, ConcatDataset, Dataset
import torch
from PIL import Image
from rootconfig import DatasetModelSpec

logger = logging.getLogger(__name__)


@dataclass
class DatasetPair:
    train: Subset
    test: Subset



class FastFedIsic2019(Isic2019Raw):
    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        debug: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        data_path: str = None,
    ):
        """Cf class docstring"""
        sz = 200
        if train:
            augmentations = albumentations.Compose(
                [
                    albumentations.RandomScale(0.07),
                    albumentations.Rotate(50),
                    albumentations.RandomBrightnessContrast(0.15, 0.1),
                    albumentations.Flip(p=0.5),
                    albumentations.Affine(shear=0.1),
                    albumentations.RandomCrop(sz, sz),
                    albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        else:
            augmentations = albumentations.Compose(
                [
                    albumentations.CenterCrop(sz, sz),
                    albumentations.Normalize(always_apply=True),
                ]
            )

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=augmentations,
            data_path=data_path,
        )

        self.center = center
        self.train_test = "train" if train else "test"
        self.pooled = pooled
        self.key = self.train_test + "_" + str(self.center)
        df = pd.read_csv(self.dic["train_test_split"])

        if self.pooled:
            df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)

        if not self.pooled:
            assert center in range(6)
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        images = df2.image.tolist()
        self.image_paths = [
            os.path.join(self.dic["input_preprocessed"], image_name + ".jpg")
            for image_name in images
        ]
        self.targets = df2.target
        self.centers = df2.center
        self.targets = torch.Tensor(self.targets).long()


        self.data = torch.stack([self._load_image(path) for path in self.image_paths])


    def _load_image(self, path):
        return torch.tensor(Image.open(path)[:, :, ::-1].copy(), dtype=torch.float32).permute(2, 0, 1)



def fetch_flamby_pooled(dataset_name: str, root: str) -> tuple[Subset, Subset]:
    logger.debug(f"[DATA LOAD] Fetching dataset: {dataset_name.upper()}")

    match dataset_name:
        case "fedisic":
            train_dataset = FedIsic2019(data_path=root, pooled=True, train=True)
            test_dataset = FedIsic2019(data_path=root, pooled=True, train=True)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return train_dataset, test_dataset


def custom_pooled(sharded_sets) -> tuple[Dataset, Dataset]:
    train_sets = []
    test_sets = []
    for train, test in sharded_sets:
        train_sets.append(train)
        test_sets.append(test)
    train_dataset = ConcatDataset(train_sets)
    test_dataset = ConcatDataset(test_sets)

    return train_dataset, test_dataset


def fetch_flamby_federated(dataset_name: str, root: str, num_splits: int):
    logger.debug(f"[DATA LOAD] Fetching dataset: {dataset_name.upper()}")

    client_datasets = []
    match dataset_name:
        case "fedisic":
            assert num_splits < 7, "FedIsic2019 only supports upto 6 centres"
            for i in range(num_splits):
                train_dataset = FedIsic2019(
                    center=i, data_path=root, pooled=False, train=True
                )
                test_dataset = FedIsic2019(
                    center=i, data_path=root, pooled=False, train=False
                )
                client_datasets.append(DatasetPair(train_dataset, test_dataset))
            pooled_test = FedIsic2019(data_path=root, pooled=True, train=False)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return client_datasets, pooled_test


def get_flamby_model_spec(dataset_name: str, root: str) -> DatasetModelSpec:
    match dataset_name:
        case "fedisic":
            train_dataset = FedIsic2019(data_path=root, pooled=True)
            num_classes = len(torch.unique(torch.as_tensor(train_dataset.targets)))
            assert num_classes == 8
            model_spec = DatasetModelSpec(num_classes=num_classes, in_channels=3)
        case _:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    return model_spec
