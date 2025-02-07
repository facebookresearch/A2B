# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import json
import os
from bisect import bisect_left, bisect_right
from itertools import product
from pathlib import Path
from loguru import logger

import numpy as np
import torch as th
import torchaudio as ta
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

# must be one of ["ffmpeg", "sox", "soundfile"].
# soundfile is recommended for best performace
# ffmpeg doesn't work for multichannel audio
# sox is slow for training
TA_BACKEND = "soundfile"

# Bytedance Raw is downloaded from: https://zenodo.org/records/10106181


class AmbixDataset(Dataset):
    def __init__(self, ds_metadata_jsonfile: str, **kwargs):
        self.ds_metadata_jsonfile = ds_metadata_jsonfile
        self.kwargs = kwargs
        self.exp_dir = self.kwargs.get("exp_dir", None)
        self.in_out_scale = self.kwargs.get("in_out_scale", [1.0, 1.0])
        self.mic_offset = self.kwargs.get("mic_offset", [0, 0])
        logger.info(f"IN:OUT Scale: {self.in_out_scale}, Offset: {self.mic_offset}")
        self.ds_metadata = dict()
        self.ds_items = dict()
        self.total_items = 0
        self.dataset_segments_idxs = list()
        self.dataset_segments_keys = list()
        self.__load()
        self.__do_data_indexing()

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        (amb_audio, bin_audio) = self._load_data(idx)
        return {
            "amb_audio": th.clamp(amb_audio*self.in_out_scale[0], min=-1.0, max=1.0),
            "bin_audio": th.clamp(bin_audio*self.in_out_scale[1], min=-1.0, max=1.0),
        }

    def __load(self):
        self.ds_metadata = json.load(open(self.ds_metadata_jsonfile))
        self.dataset_segments_idxs = list()
        self.dataset_segments_keys = list()
        self.ds_items = dict()
        self.total_items = self.ds_metadata["total_items"]
        for i, (dkey, ditem) in enumerate(self.ds_metadata["datasets"].items()):
            # construct data items
            this_items = list()
            dataset_recordings = ditem["recording_ids"]
            chunk_size = ditem["audio_chunk_size"]
            hop = ditem["audio_hop_size"]
            for idx, rec_id in enumerate(dataset_recordings):
                st_idx, end_idx = ditem["audio_chunk_start_end_idx"][idx]
                chunk_idx = range(st_idx, (end_idx - chunk_size - 1), hop)
                this_items.extend(
                    list(
                        product(
                            [idx],
                            chunk_idx,
                        )
                    )
                )
            self.ds_items[dkey] = this_items
            assert len(this_items) == ditem["size"]
        if self.exp_dir is not None:
            conf = OmegaConf.create(
                {
                    "exp_dir": str(Path(self.exp_dir)),
                    **self.kwargs,
                }
            )
            # Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
            with open(str(Path(self.exp_dir) / f"dataset.yaml"), "w") as yaml_file:
                yaml_file.write(OmegaConf.to_yaml(conf))

    def __do_data_indexing(self):
        self.dataset_segments_idxs = list()
        self.dataset_segments_keys = list()
        for i, (dkey, ditem) in enumerate(self.ds_metadata["datasets"].items()):
            self.dataset_segments_idxs.append(ditem["size"])
            self.dataset_segments_keys.append(dkey)
        for i in range(1, len(self.dataset_segments_idxs)):
            self.dataset_segments_idxs[i] += self.dataset_segments_idxs[i - 1]
        logger.info(f"Total: {self.dataset_segments_idxs}")
        assert self.dataset_segments_idxs[-1] == self.total_items

    def _get_idx(self, idx):
        idx = idx + 1
        if idx <= self.dataset_segments_idxs[0]:
            return 0, idx - 1
        else:
            item_idx, ds_idx = None, None
            ds_idx = bisect_left(self.dataset_segments_idxs, idx)
            item_idx = idx - self.dataset_segments_idxs[ds_idx - 1] - 1
            return ds_idx, item_idx

    def _load_data(self, idx):
        ds_idx, item_idx = self._get_idx(idx)
        dkey = self.dataset_segments_keys[ds_idx]
        rootpath = Path(self.ds_metadata["datasets"][dkey]["rootpath"])
        # order is: rec_id, frame_st
        rec_idx, frame_st = self.ds_items[dkey][item_idx]
        rec_id = self.ds_metadata["datasets"][dkey]["recording_ids"][rec_idx]
        # get amb recording
        wav_fn = Path(rootpath, f"{rec_id:04d}", "ambisonics.wav")
        #logger.info(f"Reading {wav_fn}")
        amb_in, _ = ta.load(
            wav_fn,
            frame_offset=frame_st + self.mic_offset[0],
            num_frames=self.ds_metadata["chunk_size"],
            channels_first=True,
            backend=f"{TA_BACKEND}",
        )  # [ch, tsamples]
        # get target binaural recording
        wav_fn = Path(rootpath, f"{rec_id:04d}", "binaural.wav")
        #logger.info(f"Reading {wav_fn}")
        bin_target, _ = ta.load(
            wav_fn,
            frame_offset=frame_st + self.mic_offset[1],
            num_frames=self.ds_metadata["chunk_size"],
            channels_first=True,
            backend=f"{TA_BACKEND}",
        )  # [2, tsamples]
        assert amb_in.shape[-1] == bin_target.shape[-1], f"Raw: {amb_in.shape}, Target: {bin_target.shape}"
        return (amb_in, bin_target)


if __name__ == "__main__":
    fn = "/mnt/home/idgebru/exported_a2b_datasets_icassp/urbansounds/validation/metadata.json"
    ds = AmbixDataset(
        ds_metadata_jsonfile=fn,
        exp_dir="/mnt/home/idgebru/",
    )
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(ds, batch_size=40, shuffle=True)
    print(len(ds))
    for data in train_dataloader:
        ambient_ref = data["ambient_ref"]
        ambient_target = data["ambient_target"]
        print(
            ambient_ref.shape,
            ambient_target.shape,
        )
        # pprint(data["target_head_rot"].numpy())
        break
