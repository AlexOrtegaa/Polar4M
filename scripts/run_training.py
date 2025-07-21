from src.utils.helpers import create_identifiers
from src.models.quantizer_model.vq_vae import VQVAE
from src.models.training import train
from src.data.data_loader import MultimodalDataset
from torch.utils.data import DataLoader

import numpy as np

import argparse
import torch
import os



def _load_data(datafile_path, batch_size):
    data = np.load(datafile_path)

    train_ids = np.load('./src/data/identifiers/train_ids.npy')
    val_ids = np.load('./src/data/identifiers/val_ids.npy')
    test_ids = np.load('./src/data/identifiers/test_ids.npy')


    train_dataset = MultimodalDataset(data, train_ids)
    val_dataset = MultimodalDataset(data, val_ids)
    test_dataset = MultimodalDataset(data, test_ids)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description='Running VQ-VAE for POSSUM data')

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Number of training epochs',
    )

    parser.add_argument(
        '--name_folder',
        type=str,
        default='/vq_vae',
        help='Folder name used to save the model checkpoints and metrics',
    )

    parser.add_argument(
        '--checkpoints_path',
        type=str,
        default='./checkpoints',
        help='Path to the folder where checkpoints will be saved.',
    )
    parser.add_argument(
        '--metrics_path',
        type=str,
        default='./metrics',
        help='Path to the folder where metrics will be saved.',
    )
    parser.add_argument(
        '--datafile_path',
        type=str,
        default='../data/modalities/stokes_i.npy',
        help='Path to the data file containing the modality to be trained.',
    )

    if  (not os.path.exists('./src/data/identifiers/'))\
            or (not os.path.exists('./src/data/identifiers/train_ids.npy')) \
                or  (not os.path.exists('./src/data/identifiers/val_ids.npy')) \
                    or  (not os.path.exists('./src/data/identifiers/test_ids.npy')):
        create_identifiers()

    args = parser.parse_args()
    num_epochs = args.num_epochs
    name_folder = args.name_folder
    checkpoints_path = args.checkpoints_path
    metrics_path = args.metrics_path
    datafile_path = args.datafile_path

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    os.makedirs(f'{checkpoints_path}{name_folder}', exist_ok=True)
    os.makedirs(f'{metrics_path}{name_folder}', exist_ok=True)

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
        name_folder=name_folder,
        metrics_path=metrics_path,
        checkpoints_path=checkpoints_path,
        device=device,
    )

if __name__ == "__main__":
    main()