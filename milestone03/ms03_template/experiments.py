import argparse

from lightning.pytorch import Trainer

from base import BaseModel
import torch
from network import CustomCNN
import sys

parser = argparse.ArgumentParser(prog='APP4RS', description='Run Experiments.')

parser.add_argument('--task', type=str, default='slc')
parser.add_argument('--logging_dir', type=str)

parser.add_argument('--dataset', type=str)
parser.add_argument('--lmdb_path', type=str)
parser.add_argument('--metadata_parquet_path', type=str)

parser.add_argument('--num_channels', type=int)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--batch_size', type=int)

parser.add_argument('--arch_name', type=str)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--epochs', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--weight_decay', type=float)
args = parser.parse_args()

def experiments():
    args = parser.parse_args()
    
    #network = None
    print("NUM CHANNELS: ", args.num_channels)
    sys.exit(-1)
    network = CustomCNN(args.num_channels,args.num_classes)
    datamodule = None
    model = BaseModel(args, datamodule, network)

    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        enable_checkpointing=True,
        max_epochs=args.epochs,
    )

    # training
    trainer.fit(model)


if __name__ == "__main__":
    experiments()
