# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import json
import os
from itertools import product
from json import JSONEncoder
from pathlib import Path
from typing import Dict, List, Set, Tuple

import hydra
import numpy as np
import soundfile as sf
import torchaudio as ta
from omegaconf import DictConfig, OmegaConf

from loguru import logger

# must be one of ["ffmpeg", "sox", "soundfile"].
# soundfile is recommended for best performace
# ffmpeg doesn't work for multichannel audio
# sox is slow for training
TA_BACKEND = "soundfile"

def merge_lists(*args):
    ret = []
    for arg in args:
        ret.extend(arg)
    return ret

OmegaConf.register_new_resolver("merge", merge_lists, replace=True)
OmegaConf.register_new_resolver(
    "range", lambda start, end, step: [*range(start, end + 1, step)], replace=True
)


audio_rates = [
    48000,
]
audio_rates_to_names = {
    audio_rate: f"_{audio_rate//1000}k" for audio_rate in audio_rates
}
# audio files with sampling rate of 96khz do not have any filename tags.
audio_rates_to_names[48000] = ""


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def _load_audio_info(path: str) -> dict:
    # get length of file in samples
    info = {}
    si = sf.info(path)
    info["samplerate"] = si.samplerate
    info["duration"] = si.duration
    info["samples"] = int(info["duration"] * info["samplerate"])
    return info


class ParseAmbixBinauralRecordings:
    def __init__(
        self,
        datasets_config: DictConfig,
        sampling_rate=48000,
        chunk_size=9600,
        overlap=0.5,
        **kwargs,
    ):
        self.ds_config = datasets_config
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap)
        self.hop = self.chunk_size - self.overlap
        self.kwargs = kwargs
        self.total_items = 0
        self.sampling_rate = int(self.kwargs.get("sampling_rate", 48000))
        self.metadata = {"datasets": {}, "total_items": 0, "chunk_size": chunk_size}
        self.__parse()

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        ds_idx = bisect_right(self.dataset_segments_idxs, idx)
        if ds_idx == 0:
            item_idx = idx
        else:
            item_idx = idx - self.dataset_segments_idxs[ds_idx - 1]
        return ds_idx, item_idx

    def __parse(self):
        for i, dataset in enumerate(self.ds_config):
            ds_metadata = process_one(
                dataset, self.chunk_size, self.hop
            )
            self.metadata["datasets"][f"ds_{i}_{dataset.name}"] = ds_metadata
            self.total_items += ds_metadata["size"]
        self.metadata["total_items"] = self.total_items

    def dump_to_file(self, outdirfullpath):
        os.makedirs(outdirfullpath, exist_ok=True)
        # Writing manifest to json file
        metadata_file = os.path.join(outdirfullpath, "metadata.json")
        with open(metadata_file, "w") as outfile:
            json.dump(self.metadata, outfile, indent=4)


def process_one(dataset: DictConfig, chunk_size: int, hop: int):
    ds_audio_chunks = []
    total_items = 0
    logger.info(f"Processing Positions: {dataset.recording_ids}")
    # print(f"Processing Positions: {type(dataset.positions)}")
    for idx, rec_id in enumerate(dataset.recording_ids):
        wavfile = Path(dataset.root_path, f"{rec_id:04d}", f"binaural.wav")
        audio_info = _load_audio_info(wavfile)
        if dataset.start is None:
            st_idx = 0
        elif dataset.start < 0:
            st_idx = int(audio_info["samples"] + int(audio_info["samplerate"] * dataset.start))
        else:
            st_idx = max(0, int(audio_info["samplerate"] * dataset.start))
        if dataset.length is None:
            end_idx = int(audio_info["samples"])
        elif dataset.length < 0:
            # leave out the last lenght frames
            end_idx = int(audio_info["samples"] + int(audio_info["samplerate"] * dataset.length))
        else:
            end_idx = min(
                st_idx + int(audio_info["samplerate"] * dataset.length),
                audio_info["samples"],
            )
        chunk_idx = range(st_idx, (end_idx - chunk_size - 1), hop)
        this_items = list(product([idx], chunk_idx))
        logger.info(f"this dataset.chunks: {len(chunk_idx)}, st_idx: {st_idx}, end_idx: {end_idx}")
        assert len(this_items) > 0, f" No items found for this dataset {wavfile} : {len(chunk_idx)}, st_idx: {st_idx}, end_idx: {end_idx}"
        # print(f"Length this: {len(this_items)}")
        ds_audio_chunks.append((st_idx, end_idx))
        total_items += len(this_items)
    # print(f"Length ds: {total_items}")

    recording_ids = (
        dataset.recording_ids
        if type(dataset.recording_ids) == list
        else OmegaConf.to_object(dataset.recording_ids)
    )
    ds_metadata = {
        "name": dataset.name,
        "rootpath": dataset.root_path,
        "samplerate": audio_info["samplerate"],
        "recording_ids": recording_ids,
        "audio_chunk_start_end_idx": ds_audio_chunks,
        "audio_chunk_size": chunk_size,
        "audio_hop_size": hop,
        "size": total_items,
    }
    return ds_metadata

