from typing import List
from dataset import ChessRecognitionDataset
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from torchvision import transforms

pl.seed_everything(42, workers=True)


class ChessDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for ChessReD.

    Args:
        dataroot (str): Path to ChessReD directory.
        batch_size (int): Number of samples per batch.
    """

    def __init__(self, dataroot: str, batch_size: int, workers: int, px_resize: int) -> None:
        """
        Args:
            dataroot (str): Path to ChessReD directory.
            batch_size (int): Number of samples per batch.
            workers (int): Number of workers for dataloading.
        """
        super().__init__()
        self.dataroot = dataroot
        self.transform = transforms.Compose([
            transforms.Resize(px_resize, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])

        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str) -> None:
        """PyTorch Lightning required method to setup the dataset at `stage`.

        Args:
            stage (str): Stage at which the data module is loaded.
        """
        if stage == "fit":
            self.chess_train = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="train", transform=self.transform)

            self.chess_val = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="val", transform=self.transform)

        if stage == "test":
            self.chess_test = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="test", transform=self.transform)

        if stage == "predict":
            self.chess_predict = ChessRecognitionDataset(
                dataroot=self.dataroot,
                split="test", transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_train, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_val, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_test, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_predict, batch_size=self.batch_size,
            num_workers=self.workers)



class ChessCornerDataModule(pl.LightningDataModule):
    def __init__(self, dataroot: str, batch_size: int, workers: int) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers

        self.transform = transforms.Compose([
            transforms.Resize(1024, antialias=None),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.47225544, 0.51124555, 0.55296206],
                std=[0.27787283, 0.27054584, 0.27802786]),
        ])

    def _get_subset_with_corners(self, dataset: ChessRecognitionDataset) -> Subset:
        image_ids_with_corners = set(dataset.corners["image_id"])
        valid_indices: List[int] = [
            i for i, img_id in enumerate(dataset.split_img_ids)
            if img_id in image_ids_with_corners
        ]
        return Subset(dataset, valid_indices)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            full_train = ChessRecognitionDataset(
                dataroot=self.dataroot, split="train", transform=self.transform)
            self.chess_train = self._get_subset_with_corners(full_train)

            full_val = ChessRecognitionDataset(
                dataroot=self.dataroot, split="val", transform=self.transform)
            self.chess_val = self._get_subset_with_corners(full_val)

        if stage == "test":
            full_test = ChessRecognitionDataset(
                dataroot=self.dataroot, split="test", transform=self.transform)
            self.chess_test = self._get_subset_with_corners(full_test)

        if stage == "predict":
            full_predict = ChessRecognitionDataset(
                dataroot=self.dataroot, split="test", transform=self.transform)
            self.chess_predict = self._get_subset_with_corners(full_predict)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_train, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_val, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_test, batch_size=self.batch_size,
            num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.chess_predict, batch_size=self.batch_size,
            num_workers=self.workers)
