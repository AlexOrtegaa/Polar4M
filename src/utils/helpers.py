from src.constants import LEN_DATASET, LEN_TRAIN_DATASET, LEN_VAL_DATASET
from sklearn.model_selection import train_test_split
from settings import IDENTIFIERS_DIR, TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH
import numpy as np

import torch
import os



def checkpoint_save(
        epoch,
        model,
        optimizer,
        name_model,
        checkpoints_dir
):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{checkpoints_dir}/{name_model}/epoch_{epoch + 1}.pth.tar')


def loss_save(
        epoch,
        loss,
        name_model,
        metrics_dir
):
    np.save(f'{metrics_dir}/{name_model}/epoch_{epoch + 1}.npy', loss)


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