# imports


def main(input_data_path: str, output_lmdb_path: str, output_parquet_path: str):
    """
    Convert the EuroSAT dataset to lmdb and parquet format.

    :param input_data_path: path to the source EuroSAT dataset (root of the extracted zip file)
    :param output_lmdb_path: path to the destination EuroSAT lmdb file
    :param output_parquet_path: path to the destination EuroSAT parquet file
    :return: None
    """

    # TODO: Read the tif files and write them to the lmdb file.
    # TODO: Create a split for the dataset.
    #       For every class the first 70% of the samples are used for training, the next 15% for validation and the last
    #       15% for testing. The samples should be ordered by the number in their filename number, 1,2,3,...,n - be
    #       aware, that the order of 1,2,3,...,10,...,n is not the same as when you sort them as strings.
    # TODO: Write the metadata to a parquet file.
    # TODO: Print the number of samples in the dataset and the number of samples in each split.

    num_keys = ...
    num_train_samples = ...
    num_validation_samples = ...
    num_test_samples = ...

    print(f"#samples: {num_keys}")
    print(f"#samples_train: {num_train_samples}")
    print(f"#samples_validation: {num_validation_samples}")
    print(f"#samples_test: {num_test_samples}")
