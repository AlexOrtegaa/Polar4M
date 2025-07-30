from src.utils.scripts_utils import create_identifiers
from src.utils.scripts_utils import _load_data, _get_args, load_pretrained
from src.models.quantizer_model.vq_vae import VQVAE
from src.models.training import train
from settings import (METRICS_DIR, CHECKPOINTS_DIR, IDENTIFIERS_DIR,
                      TRAIN_IDS_PATH, VAL_IDS_PATH, TEST_IDS_PATH)


import torch
import os



def main():
    args = _get_args()

    if  (not os.path.exists(IDENTIFIERS_DIR)) \
            or (not os.path.exists(TRAIN_IDS_PATH)) \
                or  (not os.path.exists(VAL_IDS_PATH)) \
                    or  (not os.path.exists(TEST_IDS_PATH)):
        create_identifiers()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    os.makedirs(f'{CHECKPOINTS_DIR}/{args.name_model}', exist_ok=True)
    os.makedirs(f'{METRICS_DIR}/{args.name_model}', exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = _load_data(
        datafile_path = args.datafile_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        prefetch_factor = args.prefetch_factor,
        pin_memory = args.pin_memory,

    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        last_epoch=args.last_epoch,
    )

    load_pretrained(
        model,
        optimizer,
        scheduler,
        args,
        CHECKPOINTS_DIR,
        device,
    )

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        name_model=args.name_model,
        metrics_dir=METRICS_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        device=device,
        args=args,
    )

if __name__ == "__main__":
    main()