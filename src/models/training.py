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
):


    wandb.init(project=name_model)
    wandb.watch(model, log="all")

    wandb.config.epochs = num_epochs

    pbar = trange(num_epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        total_per_batch_loss = 0
        recon_per_batch_loss = 0
        vq_per_batch_loss = 0

        for batch_idx, batch_images in enumerate(train_dataloader):

            batch_images = batch_images.unsqueeze(1).to(device)

            vq_loss, output, perplexity = model(batch_images)
            recon_loss = F.mse_loss(output, batch_images)
            loss = vq_loss + recon_loss
            total_per_batch_loss += loss.item()
            recon_per_batch_loss += recon_loss.item()
            vq_per_batch_loss += vq_loss.item()


            wandb.log({
                "train_vq_loss": vq_loss.item(),
                "train_perplexity": perplexity,
                "train_recon_loss": recon_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        total_per_batch_loss /= len(train_dataloader)
        recon_per_batch_loss /= len(train_dataloader)
        vq_per_batch_loss /= len(train_dataloader)

        loss_save(
            epoch,
            np.array([total_per_batch_loss, recon_per_batch_loss, vq_per_batch_loss]),
            name_model,
            metrics_dir
        )

        pbar.set_postfix({
            "Training loss (per batch)": f"{total_per_batch_loss:.2f}",
        })

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            checkpoint_save(
                epoch,
                model,
                optimizer,
                scheduler,
                name_model,
                checkpoints_dir
            )

    tqdm.write("\a🤖 Training finished!")
    return