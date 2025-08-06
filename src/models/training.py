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

    wandb.init(project=name_model)
    wandb.config.epochs = num_epochs

    wandb.watch(model, log="all")

    pbar = trange(num_epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        loss_per_epoch = 0
        recon_loss_per_epoch = 0
        vq_loss_per_epoch = 0
        avg_perplexity = 0
        commit_loss_per_epoch = 0
        orthogonal_reg_loss_per_epoch = 0

        for batch_idx, batch_images in enumerate(train_dataloader):

            batch_images = batch_images.unsqueeze(1).to(device)

            (quantize, output,
             loss,
             recon_loss,
             vq_loss, commit_loss, orthogonal_reg_loss,
             embed_ind, perplexity) = model(batch_images)

            loss_per_epoch += loss.item()
            recon_loss_per_epoch += recon_loss.item()
            vq_loss_per_epoch += vq_loss.item()
            commit_loss_per_epoch += commit_loss.item()
            orthogonal_reg_loss_per_epoch += orthogonal_reg_loss.item()
            avg_perplexity += perplexity.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        print(avg_perplexity)
        avg_perplexity /= len(train_dataloader)
        print(len(train_dataloader))
        print(avg_perplexity)

        wandb.log({
            "loss_per_epoch": loss_per_epoch,
            "recon_loss_per_epoch": recon_loss_per_epoch,
            "vq_loss_per_epoch": vq_loss_per_epoch,
            "commit_loss_per_epoch": commit_loss_per_epoch,
            "orthogonal_reg_loss_per_epoch": orthogonal_reg_loss_per_epoch,
            "avg_perplexity": avg_perplexity,
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        loss_save(
            epoch + args.load_shift,
            np.array(
                [
                 loss_per_epoch,
                 recon_loss_per_epoch,
                 vq_loss_per_epoch,
                 commit_loss_per_epoch,
                 orthogonal_reg_loss_per_epoch,
                 avg_perplexity,
                 ]),
            name_model,
            metrics_dir
        )

        pbar.set_postfix({
            "Training loss": f"{loss_per_epoch:.2f}",
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