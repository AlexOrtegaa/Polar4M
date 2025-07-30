from src.constants import LEN_DATASET, LEN_TRAIN_DATASET, LEN_VAL_DATASET
from sklearn.model_selection import train_test_split
from src.data.data_loader import MultimodalDataset
from torch.utils.data import DataLoader
from settings import (CONFIGS_DIR, METRICS_DIR, CHECKPOINTS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)
from src.data.scaler import MapScaler

import numpy as np

import argparse
import torch
import os
import yaml

import torch
import os



def create_identifiers():
    ids = np.arange(LEN_DATASET)
    train_and_val_ids, test_ids = train_test_split(
        ids,
        train_size=LEN_TRAIN_DATASET+LEN_VAL_DATASET,
    )
    train_ids, val_ids = train_test_split(
        train_and_val_ids,
        train_size=LEN_TRAIN_DATASET,
    )


    os.makedirs(IDENTIFIERS_DIR, exist_ok=True)

    np.save(TRAIN_IDS_PATH, train_ids)
    np.save(VAL_IDS_PATH, val_ids)
    np.save(TEST_IDS_PATH, test_ids)

    return


def _load_data(
        datafile_path,
        batch_size,
        num_workers,
        prefetch_factor,
        pin_memory,
):
    data = np.load(datafile_path)
    
    scaler = MapScaler()
    scaler.max_min_fit(data)
    data =  scaler.max_min_transform(data)

    train_ids = np.load(TRAIN_IDS_PATH)
    val_ids = np.load(VAL_IDS_PATH)
    test_ids = np.load(TEST_IDS_PATH)


    train_dataset = MultimodalDataset(data, train_ids)
    val_dataset = MultimodalDataset(data, val_ids)
    test_dataset = MultimodalDataset(data, test_ids)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader

def _get_args():
    config_parser = argparse.ArgumentParser(
        description="Parser to read configuration 'yaml' files"
    )
    config_parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='',
    )

    parser = argparse.ArgumentParser(
        description='Parser to run VQ-VAE on POSSUM data'
    )
    parser.add_argument(
        '--datafile_path',
        type=str,
        help='Path to the data file containing the modality to be trained.',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--name_model',
        type=str,
        help='Folder name used to save the model checkpoints and metrics',
    )
    parser.add_argument(
        '--load_pretrained_model',
        type=lambda x: str(x) if x is not None else None,
        default=None,
        help='Name of the pretrained model to load. If provided, the model will be started from this'
    )
    parser.add_argument(
        '--load_epoch',
        type=lambda x: int(x) if x is not None else None,
        default=None,
        help='Epoch number of the pretrained model to load. If provided, the model will be loaded from this epoch.'
    )
    parser.add_argument(
        '--wandb_id',
        type=lambda x: str(x) if x is not None else None,
        default=None,
        help='wandb id to match run'
    )
    parser.add_argument(
        '--load_shift',
        type=int,
        default=0,
        help='If a training has been tarted from a pretrained model, then you might want '
             'to count the epochs starting at load_shift when saving metrics and checkpoints. If you are starting a new run set it to 0, '
             'but if you are extending a past run, set it properly with wandb. This parameter must match the wandb run with the '
             'id given by the "id" parameter for continual training. The units for this value are num epochs. Often, it should match "load__epoch" if given.'
             'Finally, note it does not affect the training number of epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4096,
        help='The batch size for the model to train.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for the dataloader to parallelize.',
    )
    parser.add_argument(
        '--prefetch_factor',
        type=lambda x: int(x) if x is not None else None,
        default=2,
        help='Number of batches each worker loads in advance while the model is training on the current data.'
    )
    parser.add_argument(
        '--pin_memory',
        type=bool,
        default=False,
        help='Set to true when the model is run on a GPU. '
             'It allows for faster data transfer between CPU and GPU. '
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate initial value'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay value for AdamW'
    )
    parser.add_argument(
        '--T_0',
        type=int,
        default=100,
        help='Iterations until first restart'
    )
    parser.add_argument(
        '--T_mult',
        type=int,
        default=1,
        help='Factor by which T_i increases after a restart'
    )
    parser.add_argument(
        '--eta_min',
        type=float,
        default=1e-8,
        help='Minimum learning rate'
    )
    parser.add_argument(
        '--last_epoch',
        type=int,
        default=-1,
        help='Index of last epoch'
    )


    args_config, remaining_args = config_parser.parse_known_args()

    if args_config.config:
        with open(f'{CONFIGS_DIR}/{args_config.config}.yaml', 'r') as f:
            config_yaml = yaml.safe_load(f)
            parser.set_defaults(**config_yaml)

    args = parser.parse_args(remaining_args)
    args.config_path = args_config.config

    return args


def load_pretrained(
        model,
        optimizer,
        scheduler,
        args,
        checkpoints_dir,
        device,
):
    if args.load_pretrained_model is not None:
        if args.load_epoch is None:
            raise ValueError("If a pretrained model is specified, load_epoch must be specified.")

        load_model(
            model,
            args.load_pretrained_model,
            args.load_epoch,
            checkpoints_dir,
            device,
        )

        load_optimizer(
            optimizer,
            args.load_pretrained_model,
            args.load_epoch,
            checkpoints_dir,
            device,
        )

        load_scheduler(
            scheduler,
            args.load_pretrained_model,
            args.load_epoch,
            checkpoints_dir,
            device,
        )


def load_model(
        model,
        load_pretrained_model,
        load_epoch,
        checkpoints_dir,
        device,
):
    path = f'{checkpoints_dir}/{load_pretrained_model}/epoch_{load_epoch}.pth.tar'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return

def load_optimizer(
        optimizer,
        load_pretrained_model,
        load_epoch,
        checkpoints_dir,
        device,
):
    path = f'{checkpoints_dir}/{load_pretrained_model}/epoch_{load_epoch}.pth.tar'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return

def load_scheduler(
        scheduler,
        load_pretrained_model,
        load_epoch,
        checkpoints_dir,
        device,
):
    path = f'{checkpoints_dir}/{load_pretrained_model}/epoch_{load_epoch}.pth.tar'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return



