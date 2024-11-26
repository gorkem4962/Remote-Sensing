import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import lmdb
import pickle
from tqdm import tqdm
import shutil
def read_lmdb(lmdb_path):
    """
    Open an LMDB file and print some of its contents.

    :param lmdb_path: Path to the LMDB file
    """
    # Open the LMDB environment in read-only mode
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin(write=False) as txn:
        # Get the first key
        cursor = txn.cursor()
        for key, value in cursor:  # Iterate through the key-value pairs
            print(f"Key: {key.decode('utf-8')}")  # Decode key to string
            data = pickle.loads(value)  # Deserialize value using pickle
            
            # Extract metadata and data for demonstration
            metadata = data.get("metadata", {})
            image_data = data.get("data", None)  # This is the raster data
            
            print("Metadata:", metadata)
            print("Image data shape:", image_data.shape if image_data is not None else "No data")
            
            # Stop after printing the first entry
            
    
    # Close the environment
    env.close()


def delete_directory(output_lmdb_path):
  if os.path.exists(output_lmdb_path):
     if os.path.isdir(output_lmdb_path):
        shutil.rmtree(output_lmdb_path)  # Remove directory and all its contents
        print(f"Deleted directory: {output_lmdb_path}")

def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the EuroSAT dataset to lmdb and parquet format.

    :param input_data_path: path to the source EuroSAT dataset (root of the extracted zip file)
    :param output_lmdb_path: path to the destination EuroSAT lmdb file
    :param output_parquet_path: path to the destination EuroSAT parquet file
    :return: None
    """
    
    delete_directory(output_lmdb_path)
    delete_directory(output_parquet_path)

    # Ensure the directory for LMDB exists
    if not os.path.exists(os.path.dirname(output_lmdb_path)):
        os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)

    # Initialize the LMDB environment
    env = lmdb.open(output_lmdb_path, map_size=int(20**10), lock=True)
    
    # Initialize the DataFrame to hold metadata
    metadata = []

    # Prepare stats to count splits
    stats = {"train": 0, "validation": 0, "test": 0}
    
    # Traverse through each class folder (e.g., AnnualCrop, Forest, etc.)
    for class_folder in os.listdir(input_data_path):
        class_path = Path(input_data_path) / class_folder
        
        # Ensure it's a directory
        if not class_path.is_dir():
            continue

       
       

        # List all .tif files in this class directory
        tif_files = sorted(class_path.glob("*.tif"), key=lambda f: int(f.stem.split('_')[-1]))
        
        # Split the files into train, validation, and test sets
        num_files = len(tif_files)
        train_files = tif_files[:int(0.7 * num_files)]
        validation_files = tif_files[int(0.7 * num_files):int(0.85 * num_files)]
        test_files = tif_files[int(0.85 * num_files):]

        # Update stats
        stats["train"] += len(train_files)
        stats["validation"] += len(validation_files)
        stats["test"] += len(test_files)

        # Combine all files in this class
        all_files = train_files + validation_files + test_files
        
        with env.begin(write=True) as txn:
            # Process each file and store it in LMDB
            for file in tqdm(all_files, desc=f"Processing {class_folder}", unit="file"):
                with rasterio.open(file) as src:
                    data = src.read()
                    metadata_info = {
                        "file_name": file.name,
                        "class": class_folder,
                        "split": "train" if file in train_files else ("validation" if file in validation_files else "test"),
                    }

                # Serialize and store in LMDB
                key = file.name.encode('utf-8')
                value = pickle.dumps({"data": data, "metadata": src.meta})
                txn.put(key, value)
                
                # Append metadata for parquet
                metadata.append(metadata_info)

    # Close the LMDB environment
    env.close()

    # Create a DataFrame from the metadata
    df = pd.DataFrame(metadata)

    # Write the metadata to a Parquet file
    df.to_parquet(output_parquet_path)

    # Print dataset statistics
    num_keys = len(metadata)
    num_train_samples = stats["train"]
    num_validation_samples = stats["validation"]
    num_test_samples = stats["test"]

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}") 

    
