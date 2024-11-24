import pandas as pd
import numpy as np


def dict_numpatches_of_labels(df):
    """
    Counts the occurrences of patches with a given number of labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'labels' column where each entry is a list of labels.

    Returns:
        dict: A dictionary where the keys are the number of labels (0-9) and the values are the counts of patches with that many labels.
    """
    # Ensure 'labels' column exists
    assert 'labels' in df.columns, "The DataFrame must contain a 'labels' column."

    # Create a new column to count the number of labels for each patch
    df['label_count'] = df['labels'].apply(len)

    # Count the occurrences of each label count (0-9)
    label_count_dict = df['label_count'].value_counts().reindex(range(10), fill_value=0).to_dict()

    return label_count_dict


def split(df):
  """
    Splits a DataFrame into training and testing subsets based on the number of labels per patch.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'labels' (list of labels) and 'patch_id' columns.

    Returns:
        tuple: Two DataFrames, `train_df` and `test_df`, for training and testing respectively.
   """
    # Ensure required columns exist
  assert 'labels' in df.columns, "The DataFrame must contain a 'labels' column."
  assert 'patch_id' in df.columns, "The DataFrame must contain a 'patch_id' column."
  train_df = pd.DataFrame()
  test_df = pd.DataFrame()
  label_count_dict = dict_numpatches_of_labels(df)
# Loop through each label count in the dictionary
  
  

  for label_count, num_patches in label_count_dict.items():
    # Filter the patches with the current label count
      patches_with_label_count = df[df['label_count'] == label_count]
      
      
    # Determine split sizes
      test_size = num_patches // 5 # 20% testsize means dividing with 5
      train_size = num_patches - test_size  # Remainder goes to test
     
    # Split patches for the current label count
      train_subset = patches_with_label_count.iloc[:train_size]
      test_subset = patches_with_label_count.iloc[train_size:train_size + test_size]
      
    # Append to the respective train and test DataFrames
      train_df = pd.concat([train_df, train_subset])
      test_df = pd.concat([test_df, test_subset])
     
      split_df = pd.DataFrame({'train': train_df['patch_id'].reset_index(drop=True), 
                         'test': test_df['patch_id'].reset_index(drop=True)})

# Save to the specified location
      split_df.to_csv('untracked-files/split.csv', index=False)
      
  return train_df,test_df