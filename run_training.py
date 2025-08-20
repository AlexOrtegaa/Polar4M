from src.lightning_modules.vq_vae_lightning import LightningVQVAE
from src.models.quantizer_model.vq_vae import VQVAE
from src.utils.scripts_utils import create_identifiers
from src.utils.scripts_utils import _load_data, _get_args
from settings import (CHECKPOINTS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.scripts_utils import build_config

import os
import torch



def main():
    args = _get_args()
    config = build_config(args)
    torch.autograd.set_detect_anomaly(True)

    if  (not os.path.exists(IDENTIFIERS_DIR)) \
            or (not os.path.exists(TRAIN_IDS_PATH)) \
                or  (not os.path.exists(VAL_IDS_PATH)) \
                    or  (not os.path.exists(TEST_IDS_PATH)):
        create_identifiers()

    os.makedirs(f"{CHECKPOINTS_DIR}/{config['training']['name_model']}", exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['qt_path']), exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = _load_data(
        datafile_path = config['data']['datafile_path'],
        batch_size = config['data']['batch_size'],
        num_workers = config['data']['num_workers'],
        prefetch_factor = config['data']['prefetch_factor'],
        pin_memory = config['data']['pin_memory'],
        qt_path = config['data']['qt_path'],
        seed = config['training']['seed'],
    )

    model = LightningVQVAE(
        hparams = config['hparams'],
        architecture_config = config['architecture_config'],
    )
    opt_and_sched = model.configure_optimizers()
    print(opt_and_sched)

    #device = torch.device("cpu")
    """x = torch.randn(1, 1, 64, 64, device=device, requires_grad=True)
    vq_module = VQVAE(
        in_channels= 1,
        hidden_channels= 16,
        residual_channels= 8,
        num_residual_layers= 60,
        codebook_size= 30,
        codebook_dim= 3,
        commitment_weight= 1,
        orthogonal_reg_weight= 0.2,
        sample_codebook_temp= 1,
    ).to(device)

    outputs = vq_module(x)

    # Pick the loss
    loss = outputs['loss']

    print("Forward pass done. Loss:", loss)

    # Backward pass
    loss.backward()
    print("Backward pass succeeded!")"""

    wandb_logger = WandbLogger(project=config['training']['name_model'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{CHECKPOINTS_DIR}/{config['training']['name_model']}",  # metric to monitor
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
