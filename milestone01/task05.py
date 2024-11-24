import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os.path




def overlap_per_rectangle(rectangle1,rectangle2):
    """
    Determines if two rectangles overlap.

    Parameters:
        rectangle1 (dict): A dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax' defining the first rectangle.
        rectangle2 (dict): A dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax' defining the second rectangle.

    Returns:
        bool: True if the rectangles overlap, False otherwise.
    """
    xmin1 = rectangle1['xmin']
    ymin1 = rectangle1['ymin']
    xmax1 = rectangle1['xmax']
    ymax1 = rectangle1['ymax']

    xmin2 = rectangle2['xmin']
    ymin2 = rectangle2['ymin']
    xmax2 = rectangle2['xmax']
    ymax2 = rectangle2['ymax']

    horizontal_overlap = (xmax1 > xmin2 and xmax2 > xmin1) or (xmax2 > xmin1 and xmax1 > xmin2)
    vertical_overlap = (ymax1 > ymin2 and ymax2 > ymin1) or (ymax2 > ymin1 and ymax1 > ymin2)
    
    
    return horizontal_overlap or vertical_overlap

def is_overlapping(bbox1, bbox2):
  """
    Checks if any bounding box in bbox1 overlaps with any bounding box in bbox2.

    Parameters:
        bbox1 (list[dict]): A list of bounding boxes for the first set.
        bbox2 (list[dict]): A list of bounding boxes for the second set.

    Returns:
        bool: True if there is at least one overlap, False otherwise.
  """
  geometry_bbox1 = pd.Series(bbox1)
  geometry_bbox2 = pd.Series(bbox2)
  

# Iterate through the Series and access values in each dictionary
  for bbox1 in geometry_bbox1:
      for bbox2 in geometry_bbox2:
        if overlap_per_rectangle(bbox1,bbox2):
            return True
    
  return False
    
  
 
def count_unique_overlapping_patches(df):
    """
    Counts the number of unique overlapping pairs of bounding boxes in the dataset.

    Parameters:
        df (list[dict]): A list of dictionaries, each containing a 'geometry_bbox' key with bounding box data.

    Returns:
        int: The number of unique overlapping bounding box pairs.
    """
    overlap_count = 0
    n = len(df)
    overlapping_pairs = set()  # To track unique overlapping pairs

    for i in range(n):
        
        for j in range(i + 1, n):  # Only check each pair once
            bbox1 = df[i]['geometry_bbox']
            bbox2 = df[j]['geometry_bbox']
           
           
            
            if is_overlapping(bbox1, bbox2):
                
                # Sort the indices to avoid (i, j) and (j, i) duplicates
                overlapping_pairs.add(tuple(sorted((i, j))))

    # The number of unique overlaps
    overlap_count = len(overlapping_pairs)
    return overlap_count


def print_reference_map(path):
  """
    Computes statistics about the dataset, including the number of overlaps and average number of labels.

    Parameters:
        path (str): Directory path containing parquet files.

    Returns:
        None
  """

  unlabeled_num = [122,123,124,131,132,133,141,142,332,334,335,423,999]
  counter = 0
  overlap_count = 0
  list_dataframes = []
  assert os.path.isdir(path), f"Provided path is not a directory: {path}"

  for i,parquet_file in enumerate(os.listdir(path)):
    
        parquet_file_path = os.path.join(path, parquet_file)
        assert parquet_file.endswith('.parquet'), f"Invalid file format: {parquet_file}"
        parquet_file = pq.ParquetFile(parquet_file_path)
        df = parquet_file.read().to_pandas()
        # Print before filtering
         # Filter out rows with DN values in unlabeled_num
        df_filtered = df[~df['DN'].isin(unlabeled_num)]
        counter += df_filtered.size
        list_dataframes.append(df)
  print("geom-num-overlaps: ", count_unique_overlapping_patches(list_dataframes))
  print("geom-average-num-labels: ", round(counter/len(os.listdir(path)),2))

def dict_numpatches_of_labels(df):
    """
    Creates a dictionary showing the number of patches with a given number of labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'labels' column with lists of labels.

    Returns:
        dict: Dictionary mapping the count of labels (0-9) to their occurrences.
    """
    assert 'labels' in df.columns, "The DataFrame must contain a 'labels' column."
    df['label_count'] = df['labels'].apply(len)
    

# Count the occurrences of each label count (from 0 to 9)
    label_count_dict = df['label_count'].value_counts().reindex(range(10), fill_value=0).to_dict()

    return label_count_dict