from tqdm import trange
from src.utils.train_val_utils import loss_save, checkpoint_save
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np

import wandb


def train(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        num_epochs,
        name_model,
        metrics_dir,
        checkpoints_dir,
        device,
        args,
):
    if args.load_pretrained_model:
        if (args.load_epoch is None) or (args.wandb_id is None) or (args.load_shift == 0):
            raise ValueError("If a pretrained model is specified, load_epoch, id, and load_shift must be specified properly."
                             "The purpose of this is to easily extend a past run.")
        wandb.init(
            id=args.wandb_id,
            project=name_model,
            resume='must'
        )
        wandb.config.update(
            {"epochs" :wandb.config.epochs + num_epochs},
            allow_val_change = True
        )

    else:
        wandb.init(project=name_model)
        wandb.config.epochs = num_epochs

    wandb.watch(model, log="all")

    pbar = trange(num_epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        total_per_batch_loss = 0
        recon_per_batch_loss = 0
        vq_per_batch_loss = 0
        perplexity_per_batch_loss = 0

        for batch_idx, batch_images in enumerate(train_dataloader):

            batch_images = batch_images.unsqueeze(1).to(device)

            vq_loss, output, perplexity = model(batch_images)
            recon_loss = F.mse_loss(output, batch_images)
            loss = vq_loss + recon_loss

            total_per_batch_loss += loss.item()
            recon_per_batch_loss += recon_loss.item()
            vq_per_batch_loss += vq_loss.item()
            perplexity_per_batch_loss += perplexity.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        total_per_batch_loss /= len(train_dataloader)
        recon_per_batch_loss /= len(train_dataloader)
        vq_per_batch_loss /= len(train_dataloader)
        perplexity_per_batch_loss /= len(train_dataloader)

        wandb.log({
            "train_vq_loss": vq_per_batch_loss,
            "train_perplexity": perplexity_per_batch_loss,
            "train_recon_loss": recon_per_batch_loss,
            "train_recon_loss": total_per_batch_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=num_epochs + args.load_shift)

        loss_save(
            epoch + args.load_shift,
            np.array([total_per_batch_loss, recon_per_batch_loss, vq_per_batch_loss]),
            name_model,
            metrics_dir
        )

        pbar.set_postfix({
            "Training loss (per batch)": f"{total_per_batch_loss:.2f}",
        })

        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            checkpoint_save(
                epoch + args.load_shift,
                model,
                optimizer,
                scheduler,
                name_model,
                checkpoints_dir
            )

    tqdm.write("\a🤖 Training finished!")
    return