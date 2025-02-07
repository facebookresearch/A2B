# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import logging
import os
import warnings
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer, seed_everything)
from pytorch_lightning.callbacks import (ModelCheckpoint, ModelSummary,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only


def get_rank():
    # called if it is slurm job
    local_rank = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ["SLURM_PROCID"])
    node_id = int(os.environ["SLURM_NODEID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    return node_id, local_rank, global_rank, world_size


def set_exp_dir(root_out_dir):
    if "SLURM_JOB_ID" in os.environ:
        node_id, local_rank, global_rank, world_size = get_rank()
        is_master = global_rank == 0
    else:
        from pytorch_lightning.utilities import rank_zero_only

        is_master = rank_zero_only.rank == 0
    logger.info(f"is_master: {is_master}")
    out_dir = Path(root_out_dir)
    experiment_dir = out_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    existing_ver = list()
    for d in experiment_dir.iterdir():
        if d.name.startswith("v") and d.name[1:].isdecimal() and d.is_dir():
            existing_ver.append(int(d.name[1:]))
    if is_master:
        current_ver = max(existing_ver) + 1 if existing_ver else 0
        output_dir = experiment_dir / f"v{current_ver}"
        output_dir.mkdir()
    else:
        # Use the same directory for output with the main process.
        current_ver = max(existing_ver, default=0)
        output_dir = experiment_dir / f"v{current_ver}"
    return output_dir, is_master


def main(cfg: DictConfig):
    # Seed
    seed_everything(cfg.seed, workers=True)
    # Only the process with LOCAL_RANK = 0 will print logs
    root_out_dir = cfg["training"]["artifacts_dir"]
    output_dir, is_master = set_exp_dir(root_out_dir)
    if is_master:
        logger.info(f"Output logs in: {output_dir}")
        logger.info(f"Checkpoint will be saved in: {output_dir}/checkpoints/")
        logger.info(f"Tensorboard logger initialized in: {output_dir}/tb_logs")
        logger.info(
            f"The model is distributed to {cfg.training.devices} GPUs with {cfg.training.accelerator} backend."
        )
        logger.info(f"pytorch_lightning version={pl.__version__}")
        logger.info(f"torch version={torch.__version__}")
        # Dump experiment configurations for reproducibility
        with open(output_dir / "cfg.yaml", "w") as yaml_file:
            yaml_file.write(OmegaConf.to_yaml(cfg))
    # Tensorbaord
    tb_logger = TensorBoardLogger(
        save_dir=output_dir / "tb_logs", name="lightning_logs", log_graph=False
    )

    # Checkpoint
    callbacks: List[Callback] = []
    if cfg.model_checkpoint is None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints/"),
            filename="A2B-{epoch:02d}",
            verbose=True,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=cfg.get("ckpt_every_n_train_steps", 20000),
            save_top_k=-1,
            save_last=True,
        )  # save all models
    else:
        checkpoint_callback = hydra.utils.instantiate(cfg.model_checkpoint,
            dirpath=os.path.join(output_dir, "checkpoints/"),
            filename="A2B-{epoch:02d}",
            verbose=True,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=cfg.get("ckpt_every_n_train_steps", None),
            save_top_k=-1,
            save_last=True,
        )
    callbacks += [
        checkpoint_callback, 
        TQDMProgressBar(refresh_rate=cfg.progress_bar_refresh_rate),
        ModelSummary(max_depth=2),
    ]

    # Other callback
    if cfg.callbacks is not None:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Intialize PL Model
    logger.info (f" Intialize model <{cfg.model._target_}>.")
    network = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Intialize Data
    logger.info (f" Intialize data <{cfg.data._target_}>.")
    data: LightningDataModule = hydra.utils.instantiate(cfg.data, exp_dir=f"{output_dir}", _recursive_=False)

    # Intialize PL Trainer]
    plugins = []
    trainer = Trainer(
        devices=cfg["training"]["devices"],
        strategy=cfg["training"]["strategy"]
        if cfg["training"]["strategy"] != "ddp"
        else DDPStrategy(find_unused_parameters=cfg["find_unused_parameters"]),
        accelerator="gpu",
        max_epochs=cfg["training"]["epochs"],
        callbacks=callbacks,
        plugins=plugins,
        logger=[tb_logger],
        **cfg.trainer,
    )

    if cfg["training"]["strategy"] == "ddp":
        logger.info(f"{trainer.global_rank},{trainer.world_size},{os.environ.get('SLURM_NTASKS', 'NOT SLURM')},{trainer.strategy.num_processes},{trainer.strategy.num_nodes}")
    # Fit
    if cfg.ckpt_path is not None:
        logger.info(f"Resuming training fro checkpoint < {cfg.ckpt_path} >.")
        trainer.fit(network, data, ckpt_path=cfg.ckpt_path)
    else:
        logger.info("Start new training.")
        trainer.fit(network, data)
    # Make sure everything closed properly
    logger.info("Finished training.")
    logger.info(f"Best model ckpt at <{trainer.checkpoint_callback.best_model_path}>")


base_config = OmegaConf.create(
    {
        "config_name": "default",
        "config_dir": "configs",
        "ckpt_path": None,
        "model_checkpoint": None,
        "finetuning": False,
        "progress_bar_refresh_rate": 100,
        "training": {"accelerator": "gpu", "verbose": True, "lr_scheduler": None},
        "seed": 1024,
        "find_unused_parameters": False,
        "callbacks": None,
        "trainer": {
            "num_sanity_val_steps": 0,
            "deterministic": False,
            "benchmark": True,
            "enable_model_summary": True,
        },  # PL.Trainer specific parameters
    }
)

# python ./tools/train.py config_name="models/debug_model" training.devices=1,3
if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_config, conf_cli)
    conf_yaml = OmegaConf.load(
        os.path.join(conf.config_dir, "%s.yaml" % conf.config_name)
    )
    conf = OmegaConf.merge(conf, conf_yaml, conf_cli)
    main(conf)
