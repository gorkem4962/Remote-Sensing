import pandas as pd
import numpy as np
from pathlib import Path
from task03 import *
from task04 import *
from task05 import *
from task06 import *





path = Path("untracked-files/milestone01/BigEarthNet-v2.0-S2-with-errors")
df = pd.read_parquet("untracked-files/milestone01/metadata.parquet")

df['season'] = df['patch_id'].apply(get_season_from_patch_id)

season_counts = df['season'].value_counts()

max_labels, average_labels = calculate_labels(df['labels'].tolist())
print("average-num-labels: ", average_labels) 
print("maximum-num-labels: ", max_labels) 


print(f"spring: {season_counts.get('spring', 0)} samples")
print(f"summer: {season_counts.get('summer', 0)} samples")
print(f"fall: {season_counts.get('fall', 0)} samples")
print(f"winter: {season_counts.get('winter', 0)} samples")




num_wrong_size,num_no_data,num_np_dataset = calculate_num_of_errors(path,df)
print("wrong-size: " + str(num_wrong_size))

print("with-no-data: " + str(num_no_data)) 
print("not-part-of-dataset: " + str(num_np_dataset))


statistics_calculate("untracked-files/milestone01/patches_for_stats.csv.gz","untracked-files/milestone01/BigEarthNet-v2.0-S2-with-errors")


retiling("untracked-files/milestone01/BigEarthNet-v2.0-S2-with-errors/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_33_29/S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_33_29_B02.tif")

print_reference_map("untracked-files/milestone01/geoparquets")

split(df)

