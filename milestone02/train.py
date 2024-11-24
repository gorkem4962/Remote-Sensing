import logging

import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.optim import Adam

from data_BEN import BENDataModule
from data_EuroSAT import EuroSATDataModule


class APP4RSTask2SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple, output_dim: int):
        super().__init__()
        c, h, w = input_shape
        assert h == w, "Only square images are supported"
        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.pool(x)
        x = self.linear(x)
        return x


class APP4RSTask2LightningModule(LightningModule):
    """
    Simple CNN model for the APP4RS task 2
    Note: This model is not optimized for performance, it is just a simple example
    You can replace it with your own model or use as is, no changes are required
    """

    def __init__(self, datamodule: pl.LightningDataModule, network: nn.Module, task: str):
        super().__init__()
        self.datamodule = datamodule
        self.network = network
        assert task in ["slc", "mlc"], ("Task must be either 'slc' (single-label classification) or 'mlc' "
                                        "(multi-label classification)")
        self.loss = nn.CrossEntropyLoss() if task == "slc" else nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-3)
        return optim

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return self.datamodule.test_dataloader()


def main(
        dataset: str,
        lmdb_path: str,
        metadata_parquet_path: str,
):
    """
    Train a simple CNN model on the given dataset.

    :param dataset: The dataset to train on, either "BEN" or "EUROSAT"
    :param lmdb_path: the path to the lmdb file
    :param metadata_parquet_path: the path to the metadata parquet file
    :return: the trained model
    """
    # TODO: This code is not deterministic. There are multiple sources of randomness that need to be fixed.
    #       Fix the sources of randomness to make the training deterministic.
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)  # suppress some prints
    dm_kwargs = {
        "lmdb_path": lmdb_path,
        "metadata_parquet_path": metadata_parquet_path,
        "ds_type": "indexable_lmdb",
        "batch_size": 32,
        "num_workers": 4,
        "bandorder": ["B02", "B03", "B04", "B08"]
    }

    if dataset == "BEN":
        dm = BENDataModule(**dm_kwargs)
    elif dataset == "EUROSAT":
        dm = EuroSATDataModule(**dm_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    network = APP4RSTask2SimpleCNN(input_shape=(len(dm_kwargs["bandorder"]), 120, 120),
                                   output_dim=19 if dataset == "BEN" else 10)
    model = APP4RSTask2LightningModule(datamodule=dm, network=network, task="mlc" if dataset == "BEN" else "slc")

    logger = CSVLogger(
        save_dir="logs/",
        name="lightning_logs",
    )

    trainer = Trainer(max_epochs=2, logger=logger, enable_progress_bar=False, enable_model_summary=False,
                      limit_train_batches=100, limit_val_batches=32)
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    return model
