# partial functions
from hashlib import md5
from typing import List, Literal, Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

import lmdb
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms
from typing import List

import os 
import rasterio
from pathlib import Path
from torch.utils.data import DataLoader


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

EUROSAT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08","B8A", "B09", "B10", "B11", "B12"]
# just modified eursat_bands, such that it fits with the argument banddorder of main function 

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
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.bandorder = bandorder
        self.split = split
        self.transform = transform

        # Load metadata
        self.metadata = pd.read_parquet(metadata_parquet_path)

        if self.split:
            self.metadata = self.metadata[self.metadata['split'] == self.split]
        
        # Open the LMDB environment
        self.env = None
      

    def __len__(self):
        # TODO: Implement the length of the dataset.
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
       
        
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        
          # Assuming 'label' contains the class index (adjust accordingly)

        # Retrieve the image data from LMDB
        

        sample = self.metadata.iloc[idx]
        band_tensors = []  # To hold transformed tensors of each band
        label_list = []
        # Target shape for resizing (e.g., 120x120 for consistency)
        
        patch_id = sample['file_name']
        
        

        # Retrieve the band data from the LMDB database
        with self.env.begin() as txn:
            data = txn.get(patch_id.encode('utf-8'))
            
            if data is None:
                raise KeyError(f"Sample ID: {patch_id} not found in LMDB.")
            
            dictonary = pickle.loads(data)
            
            
            for band in self.bandorder:
                
                band_image = dictonary[band]
                  # Assuming `band` retrieves the correct image data
                label = sample['class']
                assert isinstance(label, str), f"Unexpected label format: {label}"

                # Convert band image to tensor
                band_image_tensor = torch.tensor(band_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                
                # Resize using torch.nn.functional.interpolate
                  # Remove added channel dimension
                
                # Append resized band tensor and label
                band_tensors.append(band_image_tensor)
                label_list.append(label)

        # Concatenate all bands along the channel dimension (C, H, W)
        image = torch.cat(band_tensors, dim=0)
        if self.transform:
         image = self.transform(image)
        
        flattened_label_list = [sublist for sublist in label_list ]
        label_indices = [EUROSAT_CLASSES.index(label) for label in flattened_label_list]
        label_indices_tensor = torch.tensor(label_indices)

        # Step 4: Create a tensor for all possible indices (for membership checking)
        all_indices = torch.tensor(range(len(EUROSAT_CLASSES)))

        # Step 5: Use torch.isin for membership checking (return 1 for present, 0 for not present)
        labels_indices = torch.isin(all_indices, label_indices_tensor).int()
        labels_indices = labels_indices.float()

        return image, labels_indices



class EuroSATIndexableTifDataset(Dataset):


    def split_dataset(self):
        dataset_with_path = []
        for class_folder in os.listdir(self.base_path):
            class_path = Path(self.base_path) / class_folder
            
            # Ensure it's a directory
            if not class_path.is_dir():
                continue

            tif_files = sorted(class_path.glob("*.tif"), key=lambda f: int(f.stem.split('_')[-1]))
            
            num_files = len(tif_files)
            if self.split is None:
                dataset_with_path += tif_files
            elif self.split == 'train':
                dataset_with_path += tif_files[:int(0.7 * num_files)]
            elif self.split == 'validation':
                dataset_with_path += tif_files[int(0.7 * num_files):int(0.85 * num_files)]
            elif self.split == 'test':
                dataset_with_path += tif_files[int(0.85 * num_files):]
        return dataset_with_path

       
       
        
        # List all .tif files in this class directory
        



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
        self.base_path = base_path
        self.bandorder = bandorder
        self.split = split 
        self.transform = transform 
        self.dataset_with_path = self.split_dataset()
        
        
    def __len__(self):
        # TODO: Implement the length of the dataset.
        return len(self.dataset_with_path)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        # TODO: Implement the __getitem__ method for the dataset.
        
        
        
        band_tensors = []  # To hold transformed tensors of each band
        label_list = []
         # Target shape for resizing (e.g., 120x120 for consistency)
        
        input_data_path = self.dataset_with_path[idx]
        patch_id = os.path.basename(input_data_path)
        patch_id_without_tif = os.path.splitext(patch_id)[0]
       
        label  = patch_id_without_tif.split("_")[0]
        
        
        with rasterio.open(input_data_path) as txn:
             data = txn.read()
             
             
            # Unpack the stored record
          
         # NumPy array
        
        band_order_indices = [EUROSAT_BANDS.index(band) for band in self.bandorder]
        for band_index in band_order_indices:
            
            
        # Convert to tensor
            
            band_tensor = torch.tensor(data[band_index], dtype=torch.float32)
            label_list.append(label)
        

        # Append the tensor to the list
            band_tensors.append(band_tensor)

    # Concatenate all bands along the channel dimension (C, H, W)
        image = torch.stack(band_tensors, dim=0)
        if self.transform:
         image = self.transform(image)
        flattened_label_list = [sublist for sublist in label_list]
        label_indices = [EUROSAT_CLASSES.index(label) for label in flattened_label_list]
        label_indices_tensor = torch.tensor(label_indices)

        # Step 4: Create a tensor for all possible indices (for membership checking)
        all_indices = torch.tensor(range(len(EUROSAT_CLASSES)))

        # Step 5: Use torch.isin for membership checking (return 1 for present, 0 for not present)
        labels_indices = torch.isin(all_indices, label_indices_tensor).int()
        labels_indices = labels_indices.float()
        
        return image,labels_indices


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
        self.env = None
        self.current_index = 0

    def __len__(self):
        # TODO: Implement the length of the dataset.
         return len(self.metadata)

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: an iterator over the dataset, e.g. via `yield` where each item is a (patch, label) tuple where patch is
            a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        while self.current_index < len(self.metadata):
            idx = self.current_index
            self.current_index += 1
            sample = self.metadata.iloc[idx]
            #print(f"Sample {idx}, Label: {sample['labels']}")
            band_tensors = []  # To hold transformed tensors of each band
            label_list = []
            

            patch_id = sample['file_name']

            # Retrieve the band data from the LMDB database
            with self.env.begin() as txn:
                data = txn.get(patch_id.encode('utf-8'))
                if data is None:
                    raise KeyError(f"Sample ID: {patch_id} not found in LMDB.")
                
                dictonary = pickle.loads(data)
                
                for band in self.bandorder:
                    band_image = dictonary[band]  # Assuming `band` retrieves the correct image data
                    label = sample['class']
                    
                    # Convert band image to tensor
                    band_image_tensor = torch.tensor(band_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                    
                    # Append resized band tensor and label
                    band_tensors.append(band_image_tensor)
                    label_list.append(label)

            # Concatenate all bands along the channel dimension (C, H, W)
            image = torch.cat(band_tensors, dim=0)
            if self.transform:
              image = self.transform(image)
            # Convert labels to indices
            flattened_label_list = [sublist for sublist in label_list]
            label_indices = [EUROSAT_CLASSES.index(label) for label in flattened_label_list]
            label_indices_tensor = torch.tensor(label_indices)

            # Step 4: Create a tensor for all possible indices (for membership checking)
            all_indices = torch.tensor(range(len(EUROSAT_CLASSES)))

            # Step 5: Use torch.isin for membership checking (return 1 for present, 0 for not present)
            labels_indices = torch.isin(all_indices, label_indices_tensor).int()
            labels_indices = labels_indices.float()

            if self.with_keys:
                yield patch_id, image, labels_indices
            else:
                yield image, labels_indices
    
        


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
            self.train_dataset = EuroSATIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='train'
            )
            self.val_dataset = EuroSATIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset = EuroSATIndexableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='test'
            )

        elif self.ds_type == 'indexable_tif':
            # Use BENIndexableTifDataset
            self.train_dataset = EuroSATIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split = 'train'
            )
            self.val_dataset =EuroSATIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset = EuroSATIndexableTifDataset(
                base_path=self.base_path,
                bandorder=self.bandorder,
                split='test'
            )

        elif self.ds_type == 'iterable_lmdb':
            # Use BENIterableLMDBDataset
            self.train_dataset = EuroSATIterableLMDBDataset (
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='train'
            )
            self.val_dataset = EuroSATIterableLMDBDataset(
                lmdb_path=self.lmdb_path,
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder=self.bandorder,
                split='validation'
            )
            self.test_dataset =EuroSATIterableLMDBDataset(
                lmdb_path=self.lmdb_path, 
                metadata_parquet_path=self.metadata_parquet_path,
                bandorder= self.bandorder,
                split='test'
            )

    def train_dataloader(self):
        # TODO: Return a D 
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
            # Use the custom sampler
        ) 

    def val_dataloader(self):
        # TODO: Return a DataLoader for the validation dataset with the correct parameters for training neural networks.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0 # Validation data should not be shuffled
        )

    def test_dataloader(self):
        # TODO: Return a DataLoader for the test dataset with the correct parameters for training neural networks.
         return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False ,
            persistent_workers=self.num_workers > 0 # Validation data should not be shuffled
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
