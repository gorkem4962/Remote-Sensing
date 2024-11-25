import os
import lmdb
import pandas as pd
import rasterio
from tqdm import tqdm
import re
import pickle
import shutil
from pathlib import Path

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

def delete_directory(output_lmdb_path):
  if os.path.exists(output_lmdb_path):
     if os.path.isdir(output_lmdb_path):
        shutil.rmtree(output_lmdb_path)  # Remove directory and all its contents
        print(f"Deleted directory: {output_lmdb_path}")



def create_lmdb(input_data_path, output_lmdb_path, output_parquet_path):
    """
    Create an LMDB database from .tif files in the dataset.

    :param input_data_path: Path to the source dataset
    :param output_lmdb_path: Path to the LMDB output
    :param output_parquet_path: Path to the parquet file containing split info
    :return: Dictionary with dataset split counts
    """

    # Ensure the directory for LMDB exists
    if not os.path.exists(os.path.dirname(output_lmdb_path)):
        os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)

    df_parquet = pd.read_parquet(output_parquet_path)

    # Open LMDB environment with a reasonable map_size
    env = lmdb.open(output_lmdb_path, map_size=int(20**10), lock=True)  # Adjust map_size if needed
    path_to_DataSet = Path(os.path.join(input_data_path, "BigEarthNet-Lithuania-Summer-S2"))
    stats = {"train": 0, "validation": 0, "test": 0}

    try:
        # Begin the first transaction
        txn = env.begin(write=True)

        for sub_path in path_to_DataSet.iterdir():
            for sub_path_order in sub_path.iterdir():
                for sub_sub_path in sub_path_order.iterdir():
                    file = os.path.basename(sub_sub_path)
                    file_without_tif = os.path.splitext(file)[0]
                    patch_id = file_without_tif.rsplit('_', 1)[0]

                    # Get the row corresponding to patch_id 
                    
                    split_values = df_parquet.loc[df_parquet['patch_id'].str.startswith(patch_id), 'split'].values
                    if len(split_values) == 0:
                        print(f"Warning: No split info for {patch_id}")
                        continue  # Skip if no split value is found

                    split_value = split_values[0]

                    # Open the .tif file and extract data and metadata
                    with rasterio.open(sub_sub_path) as src:
                        data = src.read()
                        metadata_row = df_parquet.loc[df_parquet['patch_id'].str.startswith(patch_id)]
                        metadata = metadata_row.iloc[0].to_dict()
                        

                    if split_value in stats:
                        stats[split_value] += 1

                    # Store the data in LMDB
                    key = f"{file}".encode('utf-8')
                    value = pickle.dumps({"data": data, "metadata": metadata})
                    txn.put(key, value)

                

                    # Commit every 1000 files
                
        # Commit remaining data after the loop
        

    except Exception as e:
        print(f"An error occurred: {e}")
        txn.abort()  # Abort the transaction in case of an error

    finally:
        env.close()  # Ensure the environment is closed properly

    # Normalize stats (divide by 12 as per your logic)
    stats = {key: value / 12 for key, value in stats.items()}
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
    delete_directory(output_lmdb_path)
    
    
    stats = {"train": 0, "validation": 0, "test": 0}
   
    create_parquet(input_data_path,output_parquet_path)
    # Load the metadata from the Parquet file
    
    # Make sure the output paths exist
    stats = create_lmdb(input_data_path, output_lmdb_path,output_parquet_path)
    
   

    
    print("# Dataset Statistics:")
    print(f"# Total samples: {sum(stats.values())}")
    print(f"# Train samples: {stats['train']}")
    print(f"# Validation samples: {stats['validation']}")
    print(f"# Test samples: {stats['test']}") 
