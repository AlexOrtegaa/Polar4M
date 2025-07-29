import numpy as np

import torch



def checkpoint_save(
        epoch,
        model,
        optimizer,
        scheduler,
        name_model,
        checkpoints_dir
):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f'{checkpoints_dir}/{name_model}/epoch_{epoch + 1}.pth.tar')


def loss_save(
        epoch,
        loss,
        name_model,
        metrics_dir
):
    np.save(f'{metrics_dir}/{name_model}/epoch_{epoch + 1}.npy', loss)

