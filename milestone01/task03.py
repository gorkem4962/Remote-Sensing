import re
from datetime import datetime

def get_season_from_patch_id(patch_id):
    """
    Extracts the season based on the date encoded in a patch ID.
    The patch ID should contain an 8-digit date in the format YYYYMMDD.
    The function determines the season ('spring', 'summer', 'fall', 'winter')
    based on the month of the extracted date.

    Parameters:
        patch_id (str): A string containing a patch ID with an embedded 8-digit date.

    Returns:
        str: The season corresponding to the date in the patch ID.
        None: If no valid date is found in the patch ID.
    """
    # Regular expression to search for an 8-digit value
    match = re.search(r'\d{8}', patch_id)
    
    if match:
        date_str = match.group(0)
        
        # Assert that the date string is valid
        assert len(date_str) == 8, "Date string extracted is not 8 digits long."
        
        date = datetime.strptime(date_str, "%Y%m%d")
        
        # Assert that the date is correctly parsed
        assert isinstance(date, datetime), "Date parsing failed."
        
        # Determine the season based on the month
        if date.month in [3, 4, 5]:
            return 'spring'
        elif date.month in [6, 7, 8]:
            return 'summer'
        elif date.month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
    
    # Assert that no match was found if returning None
    assert match is None, "Unexpectedly returning None when match is not None."
    return None

def calculate_labels(labels):
    """
    Calculates the maximum number of labels associated with any patch
    and the average number of labels across all patches.

    Parameters:
        labels (list of lists): A list where each element is a list of labels for a patch.

    Returns:
        tuple: A tuple containing:
            - max_labels (int): Maximum number of labels for any single patch.
            - average_labels (float): Average number of labels per patch, rounded to 2 decimal places.
    """
    # Assert input is a list of lists
    assert isinstance(labels, list), "Input labels must be a list."
   

    max_labels = max(len(patch) for patch in labels)
    total_labels = sum(len(patch) for patch in labels)
    average_labels = total_labels / len(labels)
    
    # Assert that the calculations are valid
    assert max_labels >= 0, "Maximum labels calculated is negative."
    assert total_labels >= 0, "Total labels calculated is negative."
    assert len(labels) > 0, "Labels list is empty, division by zero avoided."

    return max_labels, round(average_labels, 2)
