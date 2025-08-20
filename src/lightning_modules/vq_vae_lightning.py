from src.models.quantizer_model.vq_vae import VQVAE

import pytorch_lightning as L

import torch

optimizer_dict = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}

optimizer_params_dict = {
    'adam': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 0,
    },
    'adamw': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-08,
        'weight_decay': 1e-2,
    },
    'sgd': {
        'lr': 1e-2,
        'momentum': 0.9,
        'weight_decay': 0,
    },
}

lr_scheduler_dict = {
    'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
}

lr_scheduler_params_dict = {
    'CosineAnnealingWarmRestarts': {
        'T_0': 10,
        'T_mult': 1,
        'eta_min': 0.0,
        'last_epoch': -1,
    },
    'OneCycleLR': {
        'pct_start': 0.3,
        'anneal_strategy': 'cos',
        'div_factor': 1e1,
        'final_div_factor': 1e4,
        'total_steps': 1e3,
    },
}


class LightningVQVAE(L.LightningModule):
    def __init__(
            self,
            hparams,
            architecture_config,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = VQVAE(**architecture_config)
        return

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):


        x = batch.unsqueeze(1)

        outputs = self(x)

        optimizer = self.optimizers()

        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", outputs['recon_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_vq_loss", outputs['vq_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_commit_loss", outputs['commit_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_orthogonal_reg_loss", outputs['orthogonal_reg_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", outputs['perplexity'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=True)

        return outputs['loss']

    def predict_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)
        outputs = self(x)
        return outputs

    def configure_optimizers(self):

        optimizer_cls = optimizer_dict.get(self.hparams.optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer {self.hparams.optimizer_name} is not supported. Choose from {list(optimizer_dict.keys())}.")

        optimizer_kwargs = {}
        optimizer_params = optimizer_params_dict.get(self.hparams.optimizer_name, {})
        optimizer_kwargs = merge_hparams_with_defaults(self.hparams, optimizer_kwargs, optimizer_params)

        optimizer = optimizer_cls(self.parameters(), **optimizer_kwargs)



        if hasattr(self.hparams, 'lr_scheduler_name') and self.hparams.lr_scheduler_name is not None:
            lr_scheduler_cls = lr_scheduler_dict.get(self.lr_scheduler_name)
            if lr_scheduler_cls is None:
                raise ValueError(f"LR scheduler {self.scheduler_name} is not supported. Choose from {list(lr_scheduler_dict.keys())}.")

            lr_scheduler_kwargs = {}
            lr_scheduler_params = lr_scheduler_params_dict.get(self.lr_scheduler_name, {})
            lr_scheduler_kwargs = merge_hparams_with_defaults(self.hparams, lr_scheduler_kwargs, lr_scheduler_params)
            lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_kwargs)

            return [optimizer], [lr_scheduler]

        return [optimizer]


def merge_hparams_with_defaults(hparams, target_kwargs, default_params):
    for param, value in default_params.items():
        if hasattr(hparams, param):
            target_kwargs[param] = getattr(hparams, param)
        else:
            target_kwargs[param] = value
    return target_kwargs
