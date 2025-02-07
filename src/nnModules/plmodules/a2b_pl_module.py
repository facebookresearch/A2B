# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import torch as th
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import grad_norm, rank_zero_info


class A2B_LightningModule(pl.LightningModule):
    def __init__(
        self,
        a2b_model: DictConfig,
        metrics: DictConfig,
        optimization: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        cfg = OmegaConf.merge(a2b_model, metrics, optimization)
        # init model and loss metrics
        self.model = hydra.utils.instantiate(a2b_model)
        self.loss_metrics = hydra.utils.instantiate(metrics)
        assert (
            self.loss_metrics.loss_funcs is not None
        ), f"Loss module does not have loss functions"

        self.save_hyperparameters(cfg, ignore="loss_metrics", logger=False)

        # save
        self.validation_step_outputs = []
        self.validation_step_ground_truths = []

    def forward(self, ref_amb: th.Tensor) -> th.Tensor:
        return self.model(ref_amb=ref_amb)

    def unpack_data(self, data: dict):
        """
        Unpack data given by the DataLoader
        """
        return (data["amb_audio"], data["bin_audio"])

    def common_eval_step(
        self,
        modeloutputs: Dict,
        targets: th.Tensor,
        intermediate_targets: th.Tensor = None,
        eval_type: str = "train",
    ):
        loss, loss_dict = self.loss_metrics(modeloutputs, targets, intermediate_targets)
        loss_dict[f"{eval_type}_accumulated_loss"] = loss
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int):
        """Defines logic to execute on each training batch."""
        amb_in, bin_target = self.unpack_data(batch)
        modeloutputs = self.model(amb_in)
        loss, loss_dict = self.common_eval_step(
            modeloutputs,
            bin_target,
            intermediate_targets=None,
            eval_type=f"train",
        )
        # logging
        for k, v in loss_dict.items():
            loss_dict[
                k
            ] = v.detach()  # TODO: remove when PL version is updated to > v1.2
        self.log(
            "train_loss",
            loss.detach(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": loss, **loss_dict}

    def validation_step(self, batch: Dict, batch_idx: int):
        """Defines logic to execute on each validation batch."""
        (amb_in, bin_target)= self.unpack_data(batch)
        modeloutputs = self.model(amb_in)
        loss, loss_dict = self.common_eval_step(
            modeloutputs,
            bin_target,
            intermediate_targets=None,
            eval_type=f"validation",
        )
        # logging
        for k, v in loss_dict.items():
            loss_dict[
                k
            ] = v.detach()  # TODO: remove when PL version is updated to > v1.2
        self.log(
            "val_loss",
            loss.detach(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"loss": loss, **loss_dict}

    def configure_optimizers(self):
        """Required by pytorch-lightning."""
        logger.info("Configuring optimizers...")
        logger.info(f"optimizer: {self.hparams}")
        optimizer_type = self.hparams.optimizer
        parameters = list(self.model.parameters()) + list(
            self.loss_metrics.parameters()
        )
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        if optimizer_type == "adam":
            adam_opt = th.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.hparams.weight_decay,
                amsgrad=False,
            )
            return {
                "optimizer": adam_opt,
                "monitor": "val_loss",
            }
        else:
            logger.error(f"Invalid optimizer")
            sys.exit()
