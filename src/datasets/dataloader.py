# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets_conf: DictConfig,
        train_dataloader_conf: Optional[DictConfig] = None,
        val_dataloader_conf: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        other_args = datasets_conf.copy()
        del other_args["_target_"]
        self.val_j = OmegaConf.create(
            {
                "_target_": datasets_conf._target_,
                "ds_metadata_jsonfile": f"{datasets_conf.root_dir_validation}/metadata.json",
                **other_args,
                **kwargs,
            }
        )
        self.train_j = OmegaConf.create(
            {
                "_target_": datasets_conf._target_,
                "ds_metadata_jsonfile": f"{datasets_conf.root_dir_training}/metadata.json",
                **other_args,
                **kwargs,
            }
        )

        self.train_dataloader_conf = train_dataloader_conf or OmegaConf.create(
            {
                "batch_size": 1,
                "num_workers": 1,
                "shuffle": True,
                "pin_memory": False,
                "persistent_workers": False,
            }
        )
        self.val_dataloader_conf = val_dataloader_conf or OmegaConf.create(
            {"batch_size": 1, "num_workers": 1, "shuffle": False, "pin_memory": False}
        )

    def prepare_data(self):
        """called only once and on 1 GPU in distributed"""
        pass

    def setup(self, stage: Optional[str] = None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        if stage == "fit" or stage is None:
            self.val = hydra.utils.instantiate(self.val_j)
            self.train = hydra.utils.instantiate(self.train_j)

    def train_dataloader(self):
        """returns training dataloader"""
        return DataLoader(self.train, **self.train_dataloader_conf)

    def val_dataloader(self):
        """returns validation dataloader"""
        return DataLoader(self.val, **self.val_dataloader_conf)


if __name__ == "__main__":
    from pytorch_lightning import LightningDataModule

    ds_config = OmegaConf.load(
        "/mnt/home/idgebru/git/Earful/src/configs/models/ambient_noncausal_sym_v0.yaml"
    )
    dm: LightningDataModule = DataModule(**ds_config.data)
    dm.prepare_data()
    dm.setup(stage="fit")

    for batch in dm.train_dataloader():
        print(batch.keys())
        break
