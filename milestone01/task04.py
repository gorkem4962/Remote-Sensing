import rasterio
import re
import numpy as np
import pandas as pd
import os
from rasterio.windows import Window
from rasterio.transform import Affine


def checkup_metadata(path_patch_id, meta_data, counter):
    """
    Checks if the given patch ID exists in the metadata DataFrame.

    Parameters:
        path_patch_id (str): Path containing the patch ID.
        meta_data (DataFrame): DataFrame containing metadata with a 'patch_id' column.
        counter (int): Current count of mismatches.

    Returns:
        int: Updated mismatch counter.
    """
    path_patch_id = path_patch_id.split('/')[-1]

    # Assert that meta_data contains 'patch_id' column
    assert 'patch_id' in meta_data.columns, "'patch_id' column not found in metadata."
    
    for patch_id_parquet in meta_data['patch_id']:
        if patch_id_parquet == path_patch_id:
            return counter  # Return if a match is found

    return counter + 1  # Increment counter if no match is found


def check_no_Data(dataset):
    """
    Checks if the nodata value exists in the raster dataset's first band.

    Parameters:
        dataset (rasterio.io.DatasetReader): Opened rasterio dataset.

    Returns:
        bool: True if the nodata value is found in the dataset's first band, False otherwise.
    """
    nodata_value = dataset.nodata
    band_numpy = dataset.read(1)

    # Assert the dataset and nodata value are valid
    assert nodata_value is not None, "Nodata value is None."
    assert band_numpy.size > 0, "Band data is empty."
    
    return round(nodata_value) in band_numpy


def count_bands_in_band_data(file_path):
    """
    Opens a raster file and returns the number of bands.

    Parameters:
        file_path (str): Path to the raster file.

    Returns:
        int: Number of bands in the raster file.
    """
    # Assert that the file exists
    assert os.path.exists(file_path), f"File not found: {file_path}"

    with rasterio.open(file_path) as dataset:
        band_count = dataset.count

        # Assert that the band count is greater than zero
        assert band_count > 0, "Band count should be greater than zero."

        return band_count


def calculate_num_of_errors(path, meta_parquet):
    """
    Calculates the number of errors in a dataset including:
    - Incorrect resolutions (wrong size).
    - Bands with nodata values.
    - Missing metadata.

    Parameters:
        path (Path): Path to the dataset directory.
        meta_parquet (DataFrame): Metadata containing 'patch_id'.

    Returns:
        tuple: (num_wrongsize, num_nodata, num_npdataset)
    """
    num_wrongsize = 0
    num_nodata = 0
    num_npdataset = 0
    
    band_resolution = {
        "B02": "10m/px", "B03": "10m/px", "B04": "10m/px", "B08": "10m/px",
        "B05": "20m/px", "B06": "20m/px", "B07": "20m/px", "B8A": "20m/px",
        "B11": "20m/px", "B12": "20m/px", "B01": "60m/px", "B09": "60m/px"
    }
    
    pattern = r"B(?:\d{2}|8A)"
    
    for sub_path in path.iterdir():
        if not sub_path.is_dir():
            continue

        for sub_path_order in sub_path.iterdir():
            num_npdataset = checkup_metadata(str(sub_path_order), meta_parquet, num_npdataset)

            if not sub_path_order.is_dir():
                continue

            for sub_sub_path in sub_path_order.iterdir():
                match = re.findall(pattern, str(sub_sub_path))

                

                if match:
                    right_value = band_resolution.get(match[0])
                    
                    with rasterio.open(str(sub_sub_path)) as dataset:
                        transform = dataset.transform
                        meters_per_pixel_x = round(transform.a)
                        meters_per_pixel_x_str = f"{meters_per_pixel_x}m/px"
                        
                        if meters_per_pixel_x_str != right_value:
                            num_wrongsize += 1

                            if check_no_Data(dataset):
                                num_nodata += 1
                                break
                else:
                    print("No matches found.")
    
    # Assert final counts are non-negative
    assert num_wrongsize >= 0, "num_wrongsize cannot be negative."
    assert num_nodata >= 0, "num_nodata cannot be negative."
    assert num_npdataset >= 0, "num_npdataset cannot be negative."
    
    return num_wrongsize, num_nodata, num_npdataset


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

    data_frame = pd.read_csv(path_to_statistics)
    mean_of_bands = np.zeros((12, len(data_frame)))

    for j in range(len(data_frame)):
        tile = data_frame.iloc[j]['tile']
        patch_id = data_frame.iloc[j]['patch_id']
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




def save_tile(tile_data, transform, out_file, meta):
    """
    Saves a subset of raster data (tile) to a new GeoTIFF file.

    Parameters:
        tile_data (numpy.ndarray): The array containing the tile data.
        transform (Affine): The affine transformation matrix for the tile.
        out_file (str): Path where the tile should be saved.
        meta (dict): Metadata from the original raster file.

    Returns:
        None
    """
    # Update the metadata for the new tile
    tile_meta = meta.copy()
    tile_meta.update({
        'height': tile_data.shape[0],
        'width': tile_data.shape[1],
        'transform': transform
    })

    # Ensure the metadata has required fields and valid values
    if 'driver' not in tile_meta:
        tile_meta['driver'] = 'GTiff'
    if 'dtype' not in tile_meta:
        tile_meta['dtype'] = tile_data.dtype
    if 'count' not in tile_meta:
        tile_meta['count'] = 1

    # Assertions to validate metadata and input data
    assert tile_data.size > 0, "Tile data is empty."
    assert isinstance(transform, Affine), "Transform must be an Affine object."
    assert 'height' in tile_meta and tile_meta['height'] > 0, "Invalid height in tile metadata."
    assert 'width' in tile_meta and tile_meta['width'] > 0, "Invalid width in tile metadata."
    assert 'driver' in tile_meta and tile_meta['driver'] == 'GTiff', "Driver should be 'GTiff'."

    # Write the tile data to a new file
    with rasterio.open(out_file, 'w', **tile_meta) as dest:
        dest.write(tile_data, 1)  # Write the first (and only) band


def retiling(path_to_file):
    """
    Splits a GeoTIFF file into four smaller tiles and saves them as separate files.

    Parameters:
        path_to_file (str): Path to the GeoTIFF file to be re-tiled.

    
    """
    re_tiled_directory = "untracked-files/re-tiled"
    if not os.path.exists(re_tiled_directory):
        os.makedirs(re_tiled_directory)

    # Extract file prefix for naming the tiles
    Geotiff_file_prefix = os.path.basename(path_to_file).replace('.tif', '')

    # Assertions to validate the input file and output directory
    assert os.path.exists(path_to_file), f"Input file does not exist: {path_to_file}"
    assert os.path.isdir(re_tiled_directory), f"Failed to create output directory: {re_tiled_directory}"

    with rasterio.open(path_to_file) as src:
        # Read the original data and metadata
        data = src.read(1)  # Read the first (and only) band
        original_transform = src.transform
        meta = src.meta.copy()  # Copy the metadata for later modification

        # Assertions to ensure the input file is valid
        assert data.size > 0, "Raster data is empty."
        assert isinstance(original_transform, Affine), "Original transform must be an Affine object."
        assert 'count' in meta and meta['count'] > 0, "Invalid metadata: count should be greater than zero."

    # Define new transforms and slices for each tile
    tiles = {
        "_A": (data[:60, :60], Affine(10.0, 0.0, original_transform.c, 0.0, -10.0, original_transform.f)),
        "_B": (data[:60, 60:], Affine(10.0, 0.0, original_transform.c + 600, 0.0, -10.0, original_transform.f)),
        "_C": (data[60:, :60], Affine(10.0, 0.0, original_transform.c, 0.0, -10.0, original_transform.f - 600)),
        "_D": (data[60:, 60:], Affine(10.0, 0.0, original_transform.c + 600, 0.0, -10.0, original_transform.f - 600))
    }

    # Save each tile as a new GeoTIFF file
    for name, (tile_data, transform) in tiles.items():
        out_file = os.path.join(re_tiled_directory, f"{Geotiff_file_prefix}{name}.tif")

        # Assertions to validate tile data and output file path
        assert tile_data.size > 0, f"Tile {name} is empty."
        assert isinstance(transform, Affine), f"Invalid transform for tile {name}."

        save_tile(tile_data, transform, out_file, meta)

    re_tiled_directory = "untracked-files/re-tiled"
    if not os.path.exists(re_tiled_directory):
      os.makedirs(re_tiled_directory)
    
    Geotiff_file_prefix = os.path.basename(path_to_file)
    Geotiff_file_prefix = Geotiff_file_prefix.replace('.tif', '')

    with rasterio.open(path_to_file) as src:
        # Read the original data and metadata
        data = src.read(1)  # Read the first (and only) band
        original_transform = src.transform
        meta = src.meta.copy()  # Copy the metadata for later modification
        
    # Define new transforms for each tile
    tiles = {
        "_A": (data[:60, :60], Affine(10.0, 0.0, original_transform.c, 0.0, -10.0, original_transform.f)),
        "_B": (data[:60, 60:], Affine(10.0, 0.0, original_transform.c + 600, 0.0, -10.0, original_transform.f)),
        "_C": (data[60:, :60], Affine(10.0, 0.0, original_transform.c, 0.0, -10.0, original_transform.f - 600)),
        "_D": (data[60:, 60:], Affine(10.0, 0.0, original_transform.c + 600, 0.0, -10.0, original_transform.f - 600))
    }
    
    # Save each tile as a new TIFF file
    for name, (tile_data, transform) in tiles.items():
        out_file = f'untracked-files/re-tiled/{Geotiff_file_prefix}{name}.tif'
        save_tile(tile_data, transform, out_file, meta)