from src.constants import LEN_DATASET, LEN_TRAIN_DATASET, LEN_VAL_DATASET
from sklearn.model_selection import train_test_split

import numpy as np

import torch
import os



def checkpoint_save(epoch, model, optimizer, name_folder, checkpoint_path):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{checkpoint_path}{name_folder}/epoch_{epoch + 1}.pth.tar')


def loss_save(epoch, loss, name_folder, metrics_path):
    np.save(f'{metrics_path}{name_folder}/epoch_{epoch+1}.npy', loss)


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


    os.makedirs('./src/data/identifiers/', exist_ok=True)

    np.save(f'./src/data/identifiers/train_ids.npy', train_ids)
    np.save(f'./src/data/identifiers/val_ids.npy', val_ids)
    np.save(f'./src/data/identifiers/test_ids.npy', test_ids)

    return