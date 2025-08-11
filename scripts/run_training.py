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
    os.makedirs(os.path.dirname(args.qt_path), exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = _load_data(
        datafile_path = args.datafile_path,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        prefetch_factor = args.prefetch_factor,
        pin_memory = args.pin_memory,
        qt_path=args.qt_path,
    )

    model = VQVAE(
        in_channels=1,
        hidden_channels=16,
        residual_channels=8,
        num_residual_layers=60,
        codebook_size=30,
        codebook_dim=3,
        commitment_weight=1,
        orthogonal_reg_weight=0.2,
        sample_codebook_temp=1
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of params: {total_params}")

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-5,
        total_steps=42000,
        div_factor=1e1,
        final_div_factor=1e4,
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