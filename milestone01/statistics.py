import rasterio
import re
import numpy as np
import pandas as pd
import os
from rasterio.windows import Window
from rasterio.transform import Affine



'''
def statistics_calculate(path_to_statistics, path_to_BigEarthNet):
    """
    Calculates statistics (mean and standard deviation) for each band across all patches in a dataset.

    Parameters:
        path_to_statistics (str): Path to the CSV file containing metadata (e.g., tile, patch IDs).
        path_to_BigEarthNet (str): Root path to the BigEarthNet dataset.

    Returns:
        None; prints band means and standard deviations rounded to the nearest integer.
    """
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                  'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    data_frame = pd.read_parquet(path_to_statistics)
    mean_of_bands = np.zeros((12, len(data_frame)))
    list_of_all_bands = []

    for j in range(len(data_frame)):
        patch_id = data_frame.iloc[j]['patch_id']
        parts = patch_id.split('_')

        tile = '_'.join(parts[:-2])

        path_to_patch = os.path.join(path_to_BigEarthNet, tile, str(patch_id))
        
        if not os.path.exists(path_to_patch):
            print(f"Path not found: {path_to_patch}")
            continue

        for i, band_name in enumerate(band_names):
            band_filename = f"{patch_id}_{band_name}.tif"
            band_path = os.path.join(path_to_patch, band_filename)
            try:
                with rasterio.open(band_path) as band_rasterio:
                    nodata_value = band_rasterio.nodata
                    band_numpy = band_rasterio.read(1)
                    
                    if nodata_value is not None:
                        band_numpy_without_nodata = np.ma.masked_equal(band_numpy, nodata_value)
                    else:
                        band_numpy_without_nodata = band_numpy

                    mean_of_bands[i][j] = np.mean(band_numpy_without_nodata)
            except FileNotFoundError:
                print(f"Band file not found: {band_path}")
                mean_of_bands[i][j] = np.nan
            except Exception as e:
                print(f"Error processing {band_path}: {e}")
                mean_of_bands[i][j] = np.nan

    band_mean = np.nanmean(mean_of_bands, axis=1)
    band_std_dev = np.nanstd(mean_of_bands, axis=1)

    for i, band_name in enumerate(band_names):
        print(f'{band_name} mean: {int(round(band_mean[i]))}')
        print(f'{band_name} std-dev: {int(round(band_std_dev[i]))}')
'''
import os
import numpy as np
import pandas as pd
import rasterio

def statistics_calculate(path_to_statistics, path_to_BigEarthNet):
    """
    Calculates statistics (mean and standard deviation) for each band across all patches in a dataset.

    Parameters:
        path_to_statistics (str): Path to the Parquet file containing metadata (e.g., tile, patch IDs).
        path_to_BigEarthNet (str): Root path to the BigEarthNet dataset.

    Returns:
        None; prints band means and standard deviations rounded to the nearest integer.
    """
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                  'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    # Load the statistics data
    data_frame = pd.read_parquet(path_to_statistics)

    # Create a dictionary to hold all values for each band
    all_band_values = {band: [] for band in band_names}

    for j in range(len(data_frame)):
        patch_id = data_frame.iloc[j]['patch_id']
        parts = patch_id.split('_')

        tile = '_'.join(parts[:-2])

        # Construct the path to the patch
        path_to_patch = os.path.join(path_to_BigEarthNet, tile, str(patch_id))
        
        if not os.path.exists(path_to_patch):
            print(f"Path not found: {path_to_patch}")
            continue

        for i, band_name in enumerate(band_names):
            band_filename = f"{patch_id}_{band_name}.tif"
            band_path = os.path.join(path_to_patch, band_filename)
            try:
                with rasterio.open(band_path) as band_rasterio:
                    nodata_value = band_rasterio.nodata
                    band_numpy = band_rasterio.read(1)
                    
                    # Mask nodata values
                    if nodata_value is not None:
                        band_numpy = band_numpy[band_numpy != nodata_value]
                    
                    # Append valid values to the band list
                    all_band_values[band_name].extend(band_numpy.flatten().tolist())
            except FileNotFoundError:
                print(f"Band file not found: {band_path}")
            except Exception as e:
                print(f"Error processing {band_path}: {e}")

    # Calculate and print statistics for each band
    for band_name in band_names:
        if all_band_values[band_name]:  # Ensure there are valid values
            band_data = np.array(all_band_values[band_name])
            band_mean = np.mean(band_data)
            band_std_dev = np.std(band_data)

            print(f'{band_name} mean: {int(round(band_mean))}')
            print(f'{band_name} std-dev: {int(round(band_std_dev))}')
        else:
            print(f"No valid data for {band_name}.")

statistics_calculate("untracked-files/milestone01/metadata.parquet","untracked-files/milestone01/BigEarthNet-v2.0-S2-with-errors")