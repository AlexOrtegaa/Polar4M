from src.lightning_modules.vq_vae_lightning import LightningVQVAE
from src.utils.scripts_utils import create_identifiers
from src.utils.scripts_utils import _load_data, _get_args
from src.models.quantizer_model.vq_vae import VQVAE
from src.models.training import train
from settings import (METRICS_DIR, CHECKPOINTS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import os



def main():
    args, config = _get_args()

    if  (not os.path.exists(IDENTIFIERS_DIR)) \
            or (not os.path.exists(TRAIN_IDS_PATH)) \
                or  (not os.path.exists(VAL_IDS_PATH)) \
                    or  (not os.path.exists(TEST_IDS_PATH)):
        create_identifiers()

    os.makedirs(f'{CHECKPOINTS_DIR}/{config['training']['name_model']}', exist_ok=True)
    os.makedirs(f'{METRICS_DIR}/{config['training']['name_model']}', exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['qt_path']), exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = _load_data(
        datafile_path = config['data']['datafile_path'],
        batch_size = config['data']['batch_size'],
        num_workers = config['data']['num_workers'],
        prefetch_factor = config['data']['prefetch_factor'],
        pin_memory = config['data']['pin_memory'],
        qt_path = config['data']['qt_path'],
    )

    model = LightningVQVAE(
        hparams = config['hparams'],
        architecture_config = config['architecture_config'],
    )

    wandb_logger = WandbLogger(project='my_vqvae_project')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{CHECKPOINTS_DIR}/{config['training']['name_model']}',  # metric to monitor
        filename="epoch_{epoch:04d}",
        every_n_epochs=config['checkpoints']['every_n_epochs'],
        save_top_k=config['checkpoints']['save_top_k'],
    )

    trainer = Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=len(train_dataloader),
        deterministic=config['training']['deterministic'],
    )

    trainer.fit(model, train_dataloader)

    print("\a🤖 Training finished!")
    return

if __name__ == "__main__":
    main()
