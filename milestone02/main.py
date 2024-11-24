import os

import convert_BEN
import convert_EuroSAT
import data_BEN
import data_EuroSAT
import train

# ============================================= Environment Parameters ============================================= #
# Information about usage:
# Instead of using command line arguments, we will use environment variables to pass the parameters to the functions.
# This way, you can run the functions directly from this file and it is easier to test them for you and for us.
# For auto-tests, please comment out the following lines or remove them, as they otherwise overwrite our settings.
# Some of the variables you will set to the same values as you don't have access to the real reference that we have
# on the server. For auto-tests, the _REF keys will be used to compare the results. For local tests, just set the
# non-_REF keys and _REF keys to the same values.

os.environ['EUROSAT_SRC'] = "/home/name/Downloads/EuroSAT_MS/"
os.environ['EUROSAT_LMDB'] = "untracked-files/EuroSAT.lmdb"
os.environ['EUROSAT_LMDB_REF'] = "untracked-files/EuroSAT.lmdb"  # GT, same as EUROSAT_LMDB
os.environ['EUROSAT_PARQUET'] = "untracked-files/EuroSAT.parquet"
os.environ['EUROSAT_PARQUET_REF'] = "untracked-files/EuroSAT.parquet"  # GT, same as EUROSAT_PARQUET

os.environ['BEN_SRC'] = "untracked-files/BigEarthNet-Lithuania-Summer"
os.environ['BEN_LMDB'] = "untracked-files/BigEarthNet.lmdb"
os.environ['BEN_LMDB_REF'] = "untracked-files/BigEarthNet.lmdb"  # GT, same as BEN_LMDB
os.environ['BEN_PARQUET'] = "untracked-files/BigEarthNet.parquet"
os.environ['BEN_PARQUET_REF'] = "untracked-files/BigEarthNet.parquet"  # GT, same as BEN_PARQUET

os.environ['BEN_BANDS'] = "['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']"
os.environ['EUROSAT_BANDS'] = "['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']"
os.environ['BEN_SAMPLES'] = "[0, 15, 42, 150, 1231]"
os.environ['EUROSAT_SAMPLES'] = "[0, 31, 84, 329, 1935]"
os.environ['RANDOM_SEED'] = "42"
# ================================================================================================================== #

if __name__ == '__main__':
    test_conversion_ben = True
    test_conversion_eurosat = True
    test_data_ben = True
    test_data_eurosat = True
    test_training_ben = True
    test_training_eurosat = True
    # You may run your code here

    # ============================================== Don't change this ============================================== #
    if test_conversion_ben:
        print("\nConverting BEN")
        convert_BEN.main(
            input_data_path=os.environ['BEN_SRC'],
            output_lmdb_path=os.environ['BEN_LMDB'],
            output_parquet_path=os.environ['BEN_PARQUET']
        )

    if test_conversion_eurosat:
        print("\nConverting EuroSAT")
        convert_EuroSAT.main(
            input_data_path=os.environ['EUROSAT_SRC'],
            output_lmdb_path=os.environ['EUROSAT_LMDB'],
            output_parquet_path=os.environ['EUROSAT_PARQUET']
        )

    if test_data_ben:
        print("\nTesting BEN data")
        data_BEN.main(
            lmdb_path=os.environ['BEN_LMDB_REF'],
            metadata_parquet_path=os.environ['BEN_PARQUET_REF'],
            tif_base_path=os.environ['BEN_SRC'],
            bandorder=eval(os.environ['BEN_BANDS']),
            sample_indices=eval(os.environ['BEN_SAMPLES']),
            num_batches=5,
            seed=int(os.environ['RANDOM_SEED']),
            timing_samples=100
        )

    if test_data_eurosat:
        print("\nTesting EuroSAT data")
        data_EuroSAT.main(
            lmdb_path=os.environ['EUROSAT_LMDB_REF'],
            metadata_parquet_path=os.environ['EUROSAT_PARQUET_REF'],
            tif_base_path=os.environ['EUROSAT_SRC'],
            bandorder=eval(os.environ['EUROSAT_BANDS']),
            sample_indices=eval(os.environ['EUROSAT_SAMPLES']),
            num_batches=5,
            seed=int(os.environ['RANDOM_SEED']),
            timing_samples=100
        )

    if test_training_ben:
        import torch
        print("\nTraining BEN")
        model_1 = train.main(
            dataset='BEN',
            lmdb_path=os.environ['BEN_LMDB_REF'],
            metadata_parquet_path=os.environ['BEN_PARQUET_REF'],
        )
        model_2 = train.main(
            dataset='BEN',
            lmdb_path=os.environ['BEN_LMDB_REF'],
            metadata_parquet_path=os.environ['BEN_PARQUET_REF'],
        )
        # check that the output on both models is very similar
        inp = torch.randn((1, 4, 120, 120))
        out_1 = model_1(inp)
        out_2 = model_2(inp)
        print(f"Model output difference: {torch.abs(out_1 - out_2).mean()}")
        print(f"Models are {'not ' if model_1.state_dict().__str__() != model_2.state_dict().__str__() else ''}"
              f"completely equal")

    if test_training_eurosat:
        import torch
        print("\nTraining EuroSAT")
        model_1 = train.main(
            dataset='EUROSAT',
            lmdb_path=os.environ['EUROSAT_LMDB_REF'],
            metadata_parquet_path=os.environ['EUROSAT_PARQUET_REF'],
        )
        model_2 = train.main(
            dataset='EUROSAT',
            lmdb_path=os.environ['EUROSAT_LMDB_REF'],
            metadata_parquet_path=os.environ['EUROSAT_PARQUET_REF'],
        )
        # check that the output on both models is very similar
        inp = torch.randn((1, 4, 120, 120))
        out_1 = model_1(inp)
        out_2 = model_2(inp)
        print(f"Model output difference: {torch.abs(out_1 - out_2).mean()}")
        print(f"Models are {'not ' if model_1.state_dict().__str__() != model_2.state_dict().__str__() else ''}"
              f"completely equal")

    # =============================================================================================================== #
