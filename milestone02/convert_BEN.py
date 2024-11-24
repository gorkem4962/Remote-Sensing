import os
import lmdb
import pandas as pd
import rasterio
from tqdm import tqdm
import re
import pickle

# expected resolutions for the BigEarthNet dataset
expected_resolutions = {
    'B01': 20,
    'B02': 120,
    'B03': 120,
    'B04': 120,
    'B05': 60,
    'B06': 60,
    'B07': 60,
    'B08': 120,
    'B8A': 60,
    'B09': 20,
    'B11': 60,
    'B12': 60
}

def validate_tif(file_path):
    """
    Validate a .tif file's resolution matches the expected resolution.
    
    :param file_path: Path to the .tif file
    :return: Boolean indicating validity
    """
    try:
        with rasterio.open(file_path) as src:
            band = os.path.basename(file_path).split('_')[-1].split('.')[0]
            if band in expected_resolutions:
                return src.res[0] == expected_resolutions[band]
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
    return False

def create_lmdb(input_data_path, output_lmdb_path):
    """
    Create an LMDB database from .tif files in the dataset.

    :param input_data_path: Path to the source dataset
    :param output_lmdb_path: Path to the LMDB output
    :return: Dictionary with dataset split counts
    """
    # Ensure the directory for LMDB exists
    if not os.path.exists(os.path.dirname(output_lmdb_path)):
        os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)

    # Open LMDB environment with a reasonable map_size
    env = lmdb.open(output_lmdb_path, map_size=int(1e9))  # Adjust map_size if needed

    stats = {"train": 0, "validation": 0, "test": 0}
    with env.begin(write=True) as txn:
        idx = 0
        for root, _, files in tqdm(os.walk(input_data_path)):
            for file in files:
                if file.endswith('.tif') and validate_tif(os.path.join(root, file)):
                    with rasterio.open(os.path.join(root, file)) as src:
                        data = src.read()
                        metadata = src.meta
                    
                    key = f"sample_{idx}".encode('utf-8')
                    value = pickle.dumps({"data": data, "metadata": metadata})
                    txn.put(key, value)
                    
                    # Randomly assign to train, validation, or test
                    if idx % 10 == 0:
                        stats["test"] += 1
                    elif idx % 10 in [1, 2]:
                        stats["validation"] += 1
                    else:
                        stats["train"] += 1
                    
                    idx += 1

    env.close()
    return stats

def create_parquet(input_data_path: str,output_parquet_path: str):

    df_parquet = pd.read_parquet(os.path.join(input_data_path, "lithuania_summer.parquet"))

    if not os.path.exists(os.path.dirname(output_parquet_path)):
     os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    
    path_to_S2 = os.path.join( input_data_path,"BigEarthNet-Lithuania-Summer-S2")
    path_to_tile_1 = os.path.join(path_to_S2,"S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG")
    path_to_tile_2 = os.path.join(path_to_S2,"S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA")
    
    len_tile_1 = len(os.listdir(path_to_tile_1))
    len_tile_2 = len(os.listdir(path_to_tile_2))

    len_total = len_tile_1 + len_tile_2
    new_parquet = df_parquet.head(len_total)
    new_parquet.to_parquet(output_parquet_path, index=False)



def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the BigEarthNet dataset to LMDB and Parquet format with proper metadata columns.
    
    :param input_data_path: Path to the source BigEarthNet dataset (root directory of .tif files)
    :param output_lmdb_path: Path to the destination LMDB file
    :param output_parquet_path: Path to the destination Parquet file
    :return: None
    """
    
    stats = {"train": 0, "validation": 0, "test": 0}
   
    create_parquet(input_data_path,output_parquet_path)
    # Load the metadata from the Parquet file
    
    # Make sure the output paths exist
    stats = create_lmdb(input_data_path, output_lmdb_path)
    
    
    print("# Dataset Statistics:")
    print(f"# Total samples: {sum(stats.values())}")
    print(f"# Train samples: {stats['train']}")
    print(f"# Validation samples: {stats['validation']}")
    print(f"# Test samples: {stats['test']}") 
