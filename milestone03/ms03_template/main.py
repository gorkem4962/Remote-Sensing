import subprocess


# executing  python experiments.py
def run_training():
    # Define the command to run your training script with arguments
    command = [
        "python", "experiments.py",  # replace with your script's filename
        "--task", "?",
        "--logging_dir", "/path/to/logging_dir",
        "--dataset", "dataset",
        "--lmdb_path", "/path/to/lmdb",
        "--metadata_parquet_path", "/path/to/metadata",
        "--num_channels", "33",
        "--num_classes", "1000",
        "--num_workers", "0",
        "--batch_size", "1",
        "--arch_name", "abcdefg",
        "--epochs", "1",
        "--learning_rate", "0.0",
        "--weight_decay", "0.0"
    ]

    # Run the command synchronously and capture output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the process completed successfully
    if result.returncode == 0:
        print("Training completed successfully.")
        print(result.stdout)
        print("Error:")
        print(result.stdout)
    else:
        print("Training output:")
        print(result.stdout)
        print("Error occurred during training:")
        print(result.stderr)


if __name__ == "__main__":
    run_training()

# COMMAND 
# python experiments.py --task mlc --num_classes 3 --learning_rate 2.0 --weight_decay 2.0
