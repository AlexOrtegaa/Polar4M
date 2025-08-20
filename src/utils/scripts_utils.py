from src.constants import LEN_DATASET, LEN_TRAIN_DATASET, LEN_VAL_DATASET
from sklearn.model_selection import train_test_split
from src.data.data_loader import ModalityDataset
from torch.utils.data import DataLoader
from settings import (CONFIGS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

import numpy as np

import argparse
import yaml

import torch
import os
import joblib



arguments_groups_dict = {
    'training': [
        'num_epochs',
        'name_model',
        'accelerator',
        'devices',
        'precision',
        'deterministic',
    ],
    'optimizer': [
        'optimizer_name',
        'lr',
        'weight_decay',
        'betas',
        'eps',
        'momentum',
    ],
    'scheduler': [
        'scheduler_name',
        'T_0',
        'T_mult',
        'eta_min',
        'last_epoch',
    ],
    'data': [
        'datafile_path',
        'batch_size',
        'num_workers',
        'prefetch_factor',
        'pin_memory',
        'qt_path',
    ],
    'architecture': [
        'in_channels',
        'hidden_channels',
        'residual_channels',
        'num_residual_layers',
        'codebook_size',
        'codebook_dim',
        'commitment_weight',
        'orthogonal_reg_weight',
        'sample_codebook_temp',
    ],
    'checkpoints': [
        'every_n_epochs',
        'save_top_k',
    ],
}


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

def apply_standard_scaler(
        data
):
    shape = data.shape

    scaler = StandardScaler()
    data = data.reshape(-1, 1)
    scaler.fit(data)

    return scaler.transform(data).reshape(shape)

def load_or_fit_quantile_transformer(
        data,
        qt_path,
        n_quantiles=int(1e4),
        random_state=0,
):

    if os.path.exists(qt_path):
        qt = joblib.load(qt_path)
    else:
        qt = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=n_quantiles,
            random_state=random_state
        )
        qt.fit(data.reshape(-1, 1))
        joblib.dump(qt, qt_path)

    return qt

def apply_quantile_transformation(
    data,
    qt,
):
    shape = data.shape

    data = data.reshape(-1, 1)
    data = qt.transform(data)

    return data.reshape(shape)

def quantile_transform_splits(
        train,
        val,
        test,
        qt_path,
):
    qt = load_or_fit_quantile_transformer(
        train,
        qt_path,
    )

    train = apply_quantile_transformation(train, qt)
    val = apply_quantile_transformation(val, qt)
    test = apply_quantile_transformation(test, qt)

    return train, val, test


def _load_data(
        datafile_path,
        batch_size,
        num_workers,
        prefetch_factor,
        pin_memory,
        qt_path,
):
    data = np.load(datafile_path)

    data = apply_standard_scaler(data)

    train_ids = np.load(TRAIN_IDS_PATH)
    val_ids = np.load(VAL_IDS_PATH)
    test_ids = np.load(TEST_IDS_PATH)

    data[train_ids], data[val_ids], data[test_ids] = quantile_transform_splits(
        data[train_ids],
        data[val_ids],
        data[test_ids],
        qt_path,
    )

    train_dataset = ModalityDataset(data, train_ids)
    val_dataset = ModalityDataset(data, val_ids)
    test_dataset = ModalityDataset(data, test_ids)

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

def betas_type(s):
    parts = s.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Betas must be two comma-separated floats, e.g., '0.9,0.999'.")
    return tuple(float(part) for part in parts)

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
    # -- Training arguments --
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--name_model',
        type=str,
        default='vq_vae',
        help='Folder name used to save the model checkpoints.',
    )
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        help='Folder name used to save the model checkpoints.',
    )
    parser.add_argument(
        '--devices',
        type=int,
        default=1,
        help='Number of devices to use for training. '
             'If set to -1, it will use all available devices.',
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=32,
        help='Precision to use for training. '
             'If set to 16, it will use mixed precision training.',
    )
    parser.add_argument(
        '--deterministic',
        type=bool,
        default=False,
        help='If set to True, it will use deterministic training. '
             'It will ensure that the training is reproducible, but it might be slower.',
    )

    # -- Hyperparameters arguments--
    # Optimizer parameters

    parser.add_argument(
        '--optimizer_name',
        type=str,
        default='adamw',
        help='Name of the optimizer to use.'
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
        help='Weight decay value'
    )
    parser.add_argument(
        '--betas',
        type=betas_type,
        default=(0.9, 0.999),
        help='Betas parameters for Adam optimizer, provided as comma-separated values, e.g., "0.9,0.999".'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-8,
        help='Epsilon parameter for Adam and AdamW optimizers.'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum parameter for SGD optimizer.'
    )

    # Scheduler parameters
    parser.add_argument(
        '--scheduler_name',
        type=str,
        default='CosineAnnealingWarmRestarts',
        help='Name of the learning rate scheduler to use.'
    )
    parser.add_argument(
        '--T_0',
        type=int,
        default=100,
        help='Number of iterations until the first restart for CosineAnnealingWarmRestarts scheduler.'
    )
    parser.add_argument(
        '--T_mult',
        type=int,
        default=1,
        help='Factor by which T_i increases after a restart for CosineAnnealingWarmRestarts scheduler.'
    )
    parser.add_argument(
        '--eta_min',
        type=float,
        default=1e-6,
        help='Minimum learning rate for scheduler.'
    )
    parser.add_argument(
        '--last_epoch',
        type=int,
        default=-1,
        help='Index of the last epoch for the scheduler.'
    )

    # -- Data arguments --
    parser.add_argument(
        '--datafile_path',
        type=str,
        default="./src/data/datasets/clean_stokes_i.npy",
        help='Path to the data file containing the modality to be trained.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='The batch size for the model to train.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=3,
        help='Number of workers for the dataloader to parallelize.',
    )
    parser.add_argument(
        '--prefetch_factor',
        type=lambda x: int(x) if x is not None else None,
        default=16,
        help='Number of batches each worker loads in advance while the model is training on the current data.'
    )
    parser.add_argument(
        '--pin_memory',
        type=bool,
        default=True,
        help='Set to true when the model is run on a GPU. '
             'It allows for faster data transfer between CPU and GPU. '
    )
    parser.add_argument(
        '--qt_path',
        type=str,
        default='./transforms/qt_stokes_i.pkl',
        help='Path to the Normal Quantile Transform model. If the path is not existent, the Normal Quantile Transform model will be computed '
             'based on the training data and saved under such a path. The Normal Quantile Transform model is used during the pre-processing of the data to ensure '
             'the data exists within the same range.'
    )

    # -- Architecture arguments --
    parser.add_argument(
        '--in_channels',
        type=int,
        default=1,
        help='Number of input channels for the model. For example, 1 for Stokes I, 2 for Stokes I and Q, etc.'
    )
    parser.add_argument(
        '--hidden_channels',
        type=int,
        default=16,
        help='Number of hidden channels for the encoder side of the model.'
    )
    parser.add_argument(
        '--residual_channels',
        type=int,
        default=8,
        help="Number of channels for the encoder's and decoder's residual section."
    )
    parser.add_argument(
        '--num_residual_layers',
        type=int,
        default=60,
        help='Number of residual layers in the encoder and decoder.'
    )
    parser.add_argument(
        '--codebook_size',
        type=int,
        default=30,
        help='Size of the codebook used for vector quantization.'
    )
    parser.add_argument(
        '--codebook_dim',
        type=int,
        default=3,
        help='Dimension of each code in the codebook.'
    )
    parser.add_argument(
        '--commitment_weight',
        type=float,
        default=1.0,
        help='Weight for the commitment loss in vector quantization.'
    )
    parser.add_argument(
        '--orthogonal_reg_weight',
        type=float,
        default=0.2,
        help='Weight for the orthogonal regularization loss.'
    )
    parser.add_argument(
        '--sample_codebook_temp',
        type=float,
        default=1.0,
        help='Temperature parameter for sampling from the codebook. Note the Gumbel trick is used to sample.'
    )

    # -- Checkpoint arguments --
    every_n_epochs: 50
    save_top_k: -1
    parser.add_argument(
        '--every_n_epochs',
        type=int,
        default=50,
        help='Save a checkpoint every n epochs. If set to -1, it will save a checkpoint every epoch.'
    )
    parser.add_argument(
        '--save_top_k',
        type=int,
        default=-1,
        help='Save the last k checkpoints. If set to -1, it will save all checkpoints.'
    )

    """
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
        '--load_shift',
        type=int,
        default=0,
        help='If a training has been tarted from a pretrained model, then you might want '
             'to count the epochs starting at "load_shift" when saving metrics and checkpoints.'
             'Finally, note it does not affect the training number of epochs.'
    )
    """

    args_config, remaining_args = config_parser.parse_known_args()

    if args_config.config:
        with open(f'{CONFIGS_DIR}/{args_config.config}.yaml', 'r') as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)

    args = parser.parse_args(remaining_args)

    config['training']['config_path'] = args_config.config

    return config


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



