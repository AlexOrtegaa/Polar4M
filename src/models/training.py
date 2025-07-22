from tqdm import trange
from src.utils.helpers import loss_save, checkpoint_save
from tqdm import tqdm

import torch.nn.functional as F



def train(
        model,
        optimizer,
        train_dataloader,
        num_epochs,
        name_folder,
        metrics_dir,
        checkpoints_dir,
        device
):
    model.train()

    pbar = trange(num_epochs, desc="Training", unit="epoch")
    for epoch in pbar:
        per_batch_loss = 0

        for batch_idx, batch_images in enumerate(train_dataloader):

            batch_images = batch_images.unsqueeze(1).to(device)

            vq_loss, output, perplexity = model(batch_images)
            loss = vq_loss + F.mse_loss(output, batch_images)
            per_batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        per_batch_loss /= len(train_dataloader)
        loss_save(epoch, per_batch_loss, name_folder, metrics_dir)

        pbar.set_postfix({
            "Training loss (per batch)": f"{per_batch_loss:.2f}",
        })

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            checkpoint_save(epoch, model, optimizer, name_folder, checkpoints_dir)

    tqdm.write("\a🤖 Training finished!")
    return