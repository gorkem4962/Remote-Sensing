# partial functions
from hashlib import md5
from typing import List, Literal, Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


def _hash(data):
    return md5(str(data).encode()).hexdigest()


EUROSAT_CLASSES = [
    "Forest",
    "AnnualCrop",
    "Highway",
    "HerbaceousVegetation",
    "Pasture",
    "Residential",
    "River",
    "Industrial",
    "PermanentCrop",
    "SeaLake"
]
EUROSAT_CLASSES.sort()

EUROSAT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


class EuroSATIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the EuroSAT dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        pass

    def __len__(self):
        # TODO: Implement the length of the dataset.
        return ...

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # TODO: Implement the __getitem__ method for the dataset.
        return ...


class EuroSATIndexableTifDataset(Dataset):
    def __init__(self, base_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the EuroSAT dataset using tif files.

        :param base_path: path to the source EuroSAT dataset (root of the zip file)
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        # Hint: You don't have metadata. Where do you get the labels from? How do you split the dataset?
        pass

    def __len__(self):
        # TODO: Implement the length of the dataset.
        return ...

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # TODO: Implement the __getitem__ method for the dataset.
        return ...


class EuroSATIterableLMDBDataset(IterableDataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None,
                 with_keys=False):
        """
        IterableDataset for the EuroSAT dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: order of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        pass

    def __len__(self):
        # TODO: Implement the length of the dataset.
        return ...

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: an iterator over the dataset, e.g. via `yield` where each item is a (patch, label) tuple where patch is
            a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # TODO: Implement the iterator for the dataset.
        return ...


class EuroSATDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            bandorder: List,
            ds_type: Literal['iterable_lmdb', 'indexable_tif', 'indexable_lmdb'],
            base_path: Optional[str] = None,
            lmdb_path: Optional[str] = None,
            metadata_parquet_path: Optional[str] = None,
    ):
        """
        DataModule for the EuroSAT dataset.

        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param ds_type: type of dataset to use, one of 'iterable_lmdb', 'indexable_tif', 'indexable_lmdb'
        :param base_path: path to the source BigEarthNet dataset (root of the tar file), for tif dataset
        :param lmdb_path: path to the converted lmdb file, for lmdb dataset
        :param metadata_parquet_path: path to the metadata parquet file, for lmdb dataset
        """
        super().__init__()
        # TODO: Store the parameters as attributes as needed.
        pass

    def setup(self, stage=None):
        # TODO: Create dataset objects for the train, validation and test splits.
        pass

    def train_dataloader(self):
        # TODO: Return a DataLoader for the training dataset with the correct parameters for training neural networks.
        return ...

    def val_dataloader(self):
        # TODO: Return a DataLoader for the validation dataset with the correct parameters for training neural networks.
        return ...

    def test_dataloader(self):
        # TODO: Return a DataLoader for the test dataset with the correct parameters for training neural networks.
        return ...


############################################ DON'T CHANGE CODE BELOW HERE ############################################

def main(
        lmdb_path: str,
        metadata_parquet_path: str,
        tif_base_path: str,
        bandorder: List,
        sample_indices: List[int],
        num_batches: int,
        seed: int,
        timing_samples: int,
):
    """
    Test the EuroSAT dataset classes

    :param lmdb_path: path to the converted lmdb file
    :param metadata_parquet_path: path to the metadata parquet file
    :param tif_base_path: path to the source BigEarthNet dataset (root of the tar file)
    :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
    :param sample_indices: indices of samples to check for correctness
    :param num_batches: number of batches to check in the dataloaders for correctness
    :param seed: seed for the dataloaders for reproducibility
    :param timing_samples: number of samples to check during timing
    :return: None
    """
    import time

    # check values of sample_indices
    for split in ['train', 'validation', 'test', None]:
        print(f"\nSplit: {split}")
        for DS in [EuroSATIndexableLMDBDataset, EuroSATIndexableTifDataset, EuroSATIterableLMDBDataset]:
            paths = {
                "base_path": tif_base_path,
            } if DS == EuroSATIndexableTifDataset else {
                "lmdb_path": lmdb_path,
                "metadata_parquet_path": metadata_parquet_path,
            }
            ds = DS(
                bandorder=bandorder,
                split=split,
                **paths
            )
            total_str = ""
            if DS == EuroSATIterableLMDBDataset:
                for i, (x, y) in enumerate(ds):
                    total_str += _hash(x) + _hash(y)
                    if i >= len(sample_indices):
                        break
            else:
                for i in sample_indices:
                    x, y = ds[i]
                    total_str += _hash(x) + _hash(y)

            # check timing
            t0 = time.time()
            for i, _ in enumerate(iter(ds)):
                if i >= timing_samples:
                    break
            ds_type = "IterableLMDB " if DS == EuroSATIterableLMDBDataset \
                else "IndexableTif " if DS == EuroSATIndexableTifDataset \
                else "IndexableLMDB"
            print(f"{split}-{ds_type}: {_hash(total_str)} @ {time.time() - t0:.2f}s")

    print()
    for ds_type in ['indexable_lmdb', 'indexable_tif', 'iterable_lmdb']:
        # seed the dataloaders for reproducibility
        torch.manual_seed(seed)
        dm = EuroSATDataModule(
            batch_size=1,
            num_workers=0,
            bandorder=bandorder,
            ds_type=ds_type,
            lmdb_path=lmdb_path,
            metadata_parquet_path=metadata_parquet_path,
            base_path=tif_base_path
        )
        dm.setup()
        total_str = ""
        for i in range(num_batches):
            for x, y in dm.train_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.val_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.test_dataloader():
                total_str += _hash(x) + _hash(y)
                break
        print(f"datamodule-{ds_type:<14}: {_hash(total_str)}")
