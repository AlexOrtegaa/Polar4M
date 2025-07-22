from src.utils.helpers import create_identifiers
from src.models.quantizer_model.vq_vae import VQVAE
from src.models.training import train
from src.data.data_loader import MultimodalDataset
from torch.utils.data import DataLoader
from settings import (CONFIGS_DIR, METRICS_DIR, CHECKPOINTS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)

import numpy as np

import argparse
import torch
import os
import yaml


def _load_data(datafile_path, batch_size):
    data = np.load(datafile_path)

    train_ids = np.load('./src/data/identifiers/example_ids.npy')
    val_ids = np.load(VAL_IDS_PATH)
    test_ids = np.load(TEST_IDS_PATH)


    train_dataset = MultimodalDataset(data, train_ids)
    val_dataset = MultimodalDataset(data, val_ids)
    test_dataset = MultimodalDataset(data, test_ids)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    config_parser = argparse.ArgumentParser(
        description="Parser to read configuration 'yaml' files"
    )
    config_parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='',
    )

    train_parser = argparse.ArgumentParser(
        description='Parser to run VQ-VAE on POSSUM data'
    )

    train_parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of training epochs',
    )
    train_parser.add_argument(
        '--name_model',
        type=str,
        help='Folder name used to save the model checkpoints and metrics',
    )
    train_parser.add_argument(
        '--datafile_path',
        type=str,
        help='Path to the data file containing the modality to be trained.',
    )

    args_config, remaining_args = config_parser.parse_known_args()

    if args_config.config:
        with open(f'{CONFIGS_DIR}/{args_config.config}.yaml', 'r') as f:
            config_yaml = yaml.safe_load(f)
            train_parser.set_defaults(**config_yaml)

    args = train_parser.parse_args(remaining_args)
    num_epochs = args.num_epochs
    name_model = args.name_model
    datafile_path = args.datafile_path

    if  (not os.path.exists(IDENTIFIERS_DIR)) \
            or (not os.path.exists(TRAIN_IDS_PATH)) \
                or  (not os.path.exists(VAL_IDS_PATH)) \
                    or  (not os.path.exists(TEST_IDS_PATH)):
        create_identifiers()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    os.makedirs(f'{CHECKPOINTS_DIR}/{name_model}', exist_ok=True)
    os.makedirs(f'{METRICS_DIR}/{name_model}', exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = _load_data(datafile_path, batch_size = 4096)

    model = VQVAE(
        in_channels=1,
        hidden_channels=64,
        residual_channels=32,
        num_residual_layers=5,
        num_embeddings=512,
        dim_embeddings=128,
        beta=0.25
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        num_epochs=num_epochs,
        name_folder=name_model,
        metrics_dir=METRICS_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        device=device,
    )

if __name__ == "__main__":
    main()