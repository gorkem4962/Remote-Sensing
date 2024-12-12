# partial functions

from hashlib import md5
from typing import List, Literal, Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from pathlib import Path
import os
import lmdb
import pandas as pd
import numpy as np
import pickle
import rasterio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
from torch.utils.data import Sampler


# additional imports


def _hash(data):
    return md5(str(data).encode()).hexdigest()


BEN_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]
BEN_CLASSES.sort()
assert len(BEN_CLASSES) == 19, f"Expected 19 classes, got {len(BEN_CLASSES)}"

BEN_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]


class BENIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.bandorder = bandorder
        self.split = split
        self.transform = transform
        # Load metadata
        metadata_temp =  pd.read_parquet(self.metadata_parquet_path)
        if self.split is None: 
            self.metadata = metadata_temp
        else: 
            self.metadata = metadata_temp.loc[metadata_temp['split'].isin([self.split])]


        # Open LMDB environment
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
       
        # Map band names to indices
       
        
        

    def __len__(self):
        # TODO: Implement the length of the dataset.

        return len(self.metadata)



    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: Index of the item in the dataset.
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,).
        """
        
       
        if idx >= len(self.metadata) or idx < 0:
         raise IndexError(f"Index {idx} is out of bounds for metadata of size {len(metadata)}")

        sample = self.metadata.iloc[idx]
        band_tensors = []  # To hold transformed tensors of each band
        label_list = []
        # Target shape for resizing (e.g., 120x120 for consistency)
        target_height, target_width = 120, 120

        patch_id = sample['patch_id']

        # Retrieve the band data from the LMDB database
        with self.env.begin() as txn:
            data = txn.get(patch_id.encode('utf-8'))
            if data is None:
                raise KeyError(f"Sample ID: {patch_id} not found in LMDB.")
            
            dictonary = pickle.loads(data)
            
            for band in self.bandorder:
                band_image = dictonary[band]  # Assuming `band` retrieves the correct image data
                label = sample['labels']
                
                # Convert band image to tensor
                band_image_tensor = torch.tensor(band_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                
                # Resize using torch.nn.functional.interpolate
                band_image_resized = F.interpolate(
                    band_image_tensor, 
                    size=(target_height, target_width), 
                    mode='nearest'
                ).squeeze(0)  # Remove added channel dimension
                
                # Append resized band tensor and label
                band_tensors.append(band_image_resized)
                label_list.append(label)

        # Concatenate all bands along the channel dimension (C, H, W)
        image = torch.cat(band_tensors, dim=0)
        
        # Convert labels to indices

        labels_indices = [BEN_CLASSES.index(item) for arr in label_list for item in arr]
        
        labels = torch.tensor(labels_indices)

        return image, labels

        


class BENIndexableTifDataset(Dataset):
     
    


    def __init__(self, base_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using tif files.

        :param base_path: path to the source BigEarthNet dataset (root of the tar file)
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        self.base_path = base_path
        self.bandorder = bandorder
        self.split = split 
        self.transform = transform 
        
        dataframe = pd.read_parquet("untracked-files/BigEarthNet.parquet")
        self.metadata = dataframe 
        if self.split is None:
           self.metadata_for_split = dataframe
        else: 
           self.metadata_for_split = dataframe.loc[dataframe['split'].isin([self.split])]
        # Reset the indices
        self.metadata_for_split.reset_index(drop=True, inplace=True)

# Verify the new indices


    


    def __len__(self):
        # TODO: Implement the length of the dataset.
        counter = len(self.metadata_for_split)
       
        return counter

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        if idx < 0 or idx >= len(self.metadata_for_split):
         raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.metadata_for_split)}")
        
        # TODO: Implement the __getitem__ method for the dataset.
        
        band_tensors = []  # To hold transformed tensors of each band
        label_list = []
         # Target shape for resizing (e.g., 120x120 for consistency)
        target_height, target_width = 120, 120
        input_data_path = self.base_path
        path_to_DataSet = os.path.join(input_data_path, "BigEarthNet-Lithuania-Summer-S2")
        
        patch_id = self.metadata_for_split.iloc[idx]['patch_id']
        sample = self.metadata_for_split.iloc[idx]['labels']
       
       
        tile = patch_id.rsplit('_', 2)[0]
        path_to_tile = os.path.join(path_to_DataSet,tile)
        path_to_patch_2 = os.path.join(path_to_tile,patch_id)
        path_to_patch = os.path.join(path_to_patch_2,patch_id)
        for band in self.bandorder:
            
            path_to_band = path_to_patch+ f"_{band}.tif"

         # Retrieve the band data from the LMDB database
            with rasterio.open(path_to_band) as txn:
             data = txn.read()
             if data is None:
                raise KeyError("Sample ID: not found in tif files.")
             
            # Unpack the stored record
          
            band_image = data  # NumPy array
            label = self.metadata_for_split.iloc[idx]['labels']
            

         # Resize the band image using NumPy or manual interpolation
            if band_image.shape[1:] != (target_height, target_width):
            # Convert NumPy array to PyTorch tensor and add a batch dimension
             band_image_tensor = torch.tensor(band_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Resize using interpolate
             band_image_resized = F.interpolate(
                band_image_tensor, 
                size=(target_height, target_width), 
                mode='nearest'
             ).squeeze(0).numpy()  # Remove batch dimension and convert back to NumPy
            else:
             band_image_resized = band_image

        # Convert to tensor
            band_tensor = torch.tensor(band_image_resized, dtype=torch.float32)
            label_list.append(label)
        

        # Append the tensor to the list
            band_tensors.append(band_tensor)

    # Concatenate all bands along the channel dimension (C, H, W)
        image = torch.cat(band_tensors, dim=0)
        
        labels_indices =  [BEN_CLASSES.index(item) for arr in label_list for item in arr]
        labels = torch.tensor(labels_indices)
    # Retrieve label from metadata
     

        return image, labels
      

              
        


class BENIterableLMDBDataset(IterableDataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None,
                 with_keys=False):
        """
        IterableDataset for the BigEarthNet dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: order of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        # TODO: Implement the constructor for the dataset.
        # Hint: Be aware when to initialize what.
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.with_keys = with_keys
        self.bandorder = bandorder
        self.split = split 
        # Load and filter metadata
        metadata_2 = pd.read_parquet(metadata_parquet_path)
        if self.split is None:
            self.metadata = metadata_2
        else:
            self.metadata = metadata_2.loc[metadata_2['split'].isin([self.split])]
       
        
        # Open the LMDB environment
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.current_index = 0

    def __len__(self):
        # TODO: Implement the length of the dataset.
        
        return len(self.metadata)



    def __iter__(self):
        """
        Iterate over the dataset.

        :return: an iterator over the dataset, yielding (patch, label) tuples where patch is a tensor of shape (C, H, W)
                and label is a tensor of shape (N,). If `self.with_keys` is True, yields (sample_id, patch, label).
        """
        
        
        while self.current_index < len(self.metadata):
            idx = self.current_index
            self.current_index += 1
            sample = self.metadata.iloc[idx]
            #print(f"Sample {idx}, Label: {sample['labels']}")
            band_tensors = []  # To hold transformed tensors of each band
            label_list = []
            target_height, target_width = 120, 120  # Target shape for resizing

            patch_id = sample['patch_id']

            # Retrieve the band data from the LMDB database
            with self.env.begin() as txn:
                data = txn.get(patch_id.encode('utf-8'))
                if data is None:
                    raise KeyError(f"Sample ID: {patch_id} not found in LMDB.")
                
                dictonary = pickle.loads(data)
                
                for band in self.bandorder:
                    band_image = dictonary[band]  # Assuming `band` retrieves the correct image data
                    label = sample['labels']
                    
                    # Convert band image to tensor
                    band_image_tensor = torch.tensor(band_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                    
                    # Resize using torch.nn.functional.interpolate
                    band_image_resized = F.interpolate(
                        band_image_tensor, 
                        size=(target_height, target_width), 
                        mode='nearest'
                    ).squeeze(0)  # Remove added channel dimension
                    
                    # Append resized band tensor and label
                    band_tensors.append(band_image_resized)
                    label_list.append(label)

            # Concatenate all bands along the channel dimension (C, H, W)
            image = torch.cat(band_tensors, dim=0)
            
            # Convert labels to indices
            labels_indices = [BEN_CLASSES.index(item) for arr in label_list for item in arr]
            labels = torch.tensor(labels_indices)

            if self.with_keys:
                yield patch_id, image, labels
            else:
                yield image, labels
    


class BENDataModule(LightningDataModule):

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
        DataModule for the BigEarthNet dataset.

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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bandorder = bandorder
        self.ds_type = ds_type
        self.base_path = base_path
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path

        # Initialize the dataset objects for train, validation, and test (None to be initialized later)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    
    def setup(self, stage=None):
        # TODO: Create dataset objects for the train, validation and test splits.
        if self.ds_type == 'indexable_lmdb':
            # Use BENIndexableLMDBDataset
            self.train_dataset = BENIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='train'
            )
            self.val_dataset = BENIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset = BENIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='test'
            )

        elif self.ds_type == 'indexable_tif':
            # Use BENIndexableTifDataset
            self.train_dataset = BENIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split = 'train'
            )
            self.val_dataset = BENIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset = BENIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split='test'
            )

        elif self.ds_type == 'iterable_lmdb':
            # Use BENIterableLMDBDataset
            self.train_dataset = BENIterableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='train'
            )
            self.val_dataset = BENIterableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset = BENIterableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='test'
            )
    
   
    def train_dataloader(self):
                
    # Create a sampler that respects the dataset length
        
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Use the custom sampler
        )
    def val_dataloader(self):
        # TODO: Return a DataLoader for the validation dataset with the correct parameters for training neural networks.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False  # Validation data should not be shuffled
        )

    def test_dataloader(self):
        # TODO: Return a DataLoader for the test dataset with the correct parameters for training neural networks.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False  # Validation data should not be shuffled
        )


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
    Test the BigEarthNet dataset classes.

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
        for DS in [BENIndexableLMDBDataset, BENIndexableTifDataset, BENIterableLMDBDataset]:
            paths = {
                "base_path": tif_base_path,
            } if DS == BENIndexableTifDataset else {
                "lmdb_path": lmdb_path,
                "metadata_parquet_path": metadata_parquet_path,
            }
            # create dataset
            ds = DS(
                bandorder=bandorder,
                split=split,
                **paths
            )
            # check values of sample_indices, collect hashes
            total_str = ""
            if DS == BENIterableLMDBDataset:
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
            ds_type = "IterableLMDB " if DS == BENIterableLMDBDataset \
                else "IndexableTif " if DS == BENIndexableTifDataset \
                else "IndexableLMDB"
            print(f"{split}-{ds_type}: {_hash(total_str)} @ {time.time() - t0:.2f}s")
    
    print()
   
    for ds_type in ['indexable_lmdb', 'indexable_tif', 'iterable_lmdb']:
        # seed the dataloaders for reproducibility
        
        torch.manual_seed(seed)
        dm = BENDataModule(
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
