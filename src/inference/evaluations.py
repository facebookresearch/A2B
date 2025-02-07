# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import json
import os
from json import JSONEncoder
from pathlib import Path
import shutil
import hydra
import numpy as np
import torch as th
import soundfile as sf
import torchaudio as ta
import tqdm
from loguru import logger
from omegaconf import DictConfig, OmegaConf
TA_BACKEND = "soundfile"

from inference.report_metrics import report



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def merge_lists(*args):
    ret = []
    for arg in args:
        ret.extend(arg)
    return ret


OmegaConf.register_new_resolver("merge", merge_lists, replace=True)
OmegaConf.register_new_resolver(
    "range", lambda start, end, step: [*range(start, end + 1, step)], replace=True
)


def load_checkpoint(ckpt_path, model, default_device="cpu"):
    # https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html#extract-nn-module-from-lightning-checkpoints
    checkpoint = th.load(ckpt_path, map_location=th.device(default_device))
    model_weights = checkpoint["state_dict"]
    # remove "model." prefix
    for key in list(model_weights):
        model_weights[key.replace("model.", "", 1)] = model_weights.pop(key)
    # 1. filter out unnecessary keys saved with PL checkpoints
    model_dict = model.state_dict()
    checkpoint_dict = {k: v for k, v in model_weights.items() if k in model_dict}
    # 2. load state_dict
    model.load_state_dict(checkpoint_dict, strict=True)
    model.eval()
    model.to(default_device)
    logger.info(f"Loaded model from {ckpt_path}")
    logger.info(f"Model loaded to {default_device}")
    logger.info(f"Model: {model}")
    return model


# def _load_audio_info(path: str) -> dict:
#     # get length of file in samples
#     info = {}
#     si = ta.info(str(path))
#     info["samplerate"] = si.sample_rate
#     info["samples"] = si.num_frames
#     info["duration"] = info["samples"] / info["samplerate"]
#     return info

def _load_audio_info(path: str) -> dict:
    # get length of file in samples
    info = {}
    si = sf.info(path)
    info["samplerate"] = si.samplerate
    info["duration"] = si.duration
    info["samples"] = int(info["duration"] * info["samplerate"])
    return info

@logger.catch
def offline_rendering(
    network, ref_amb, device: str = "cuda:0",
):
    outputs = []
    ref_amb = ref_amb.unsqueeze(0).to(device)  # [1, AmbCh, K]
    with th.no_grad():
        modeloutputs = network(ref_amb)
    if isinstance(modeloutputs, dict):
        this_out = modeloutputs["output"]
    else:
        this_out = modeloutputs
    this_out = th.clamp(this_out, min=-1, max=1).cpu()
    return this_out


@logger.catch
def load_audio_data(testset_cfg: DictConfig, rec_id, ref_input_channels, in_out_scale=[1.0, 1.0]):
    if rec_id is None:
        wav_fn_ref = Path(testset_cfg.wav_path)
        wav_fn_gnd = Path(testset_cfg.wav_path_gnd)
    else:
        wav_fn_ref = Path(testset_cfg.wav_path, f"{rec_id:04d}", "ambisonics.wav")
        if testset_cfg.bin_from_video:
            wav_fn_gnd = Path(
                testset_cfg.wav_path, f"{rec_id:04d}", "binaural_from_video.wav"
            )
        else:
            wav_fn_gnd = Path(testset_cfg.wav_path, f"{rec_id:04d}", "binaural.wav")
    logger.info(f"Reading {wav_fn_ref}")
    audio_info = _load_audio_info(wav_fn_ref)
    if testset_cfg.start is None:
        frame_st = 0
    elif testset_cfg.start < 0:
        frame_st = int(
            audio_info["samples"] + int(audio_info["samplerate"] * testset_cfg.start)
        )
    else:
        frame_st = int(testset_cfg.start * audio_info["samplerate"])
    #
    if testset_cfg.length is None:
        num_frames = audio_info["samples"] - frame_st
    elif testset_cfg.length < 0:
        num_frames = (
            audio_info["samples"]
            - frame_st
            + int(audio_info["samplerate"] * testset_cfg.length)
        )
    else:
        num_frames = min(
            int(testset_cfg.length * audio_info["samplerate"]), audio_info["samples"],
        )
    ambient, rfs = ta.load(
        wav_fn_ref, frame_offset=frame_st, num_frames=num_frames, channels_first=True, backend=f"{TA_BACKEND}",
    )  # [32, tsamples]
    ambient_ref = ambient * testset_cfg.scale * in_out_scale[0]
    assert (
        ambient_ref.shape[0] == ref_input_channels
    ), f"Expects {ref_input_channels} channels audio but got {ambient_ref.shape[0]}!"
    # get groundturth
    gnd, rfs_gnd = ta.load(
        wav_fn_gnd, frame_offset=frame_st, num_frames=num_frames, channels_first=True, backend=f"{TA_BACKEND}",
    )  # [2, tsamples]
    gnd = gnd * testset_cfg.scale * in_out_scale[1]
    return ambient_ref, rfs, gnd, wav_fn_ref


def main(cfg: DictConfig):
    exp_dir = Path(cfg.ckpt_path)
    cfg_file = (exp_dir.parent.parent).joinpath("cfg.yaml")
    dataset_file = (exp_dir.parent.parent).joinpath("dataset.yaml")
    model_cfg = OmegaConf.load(str(cfg_file))
    logger.info(f"Loading the following config:\n{OmegaConf.to_yaml(model_cfg)}")
    conf = OmegaConf.merge(cfg, model_cfg)
    # GPU/CPU
    if not th.cuda.is_available():
        cfg["device"] = None

    # load model
    logger.info(f"Loading checkpoint: {cfg.ckpt_path} to {cfg['device']}")
    network = hydra.utils.instantiate(model_cfg.model.a2b_model, _recursive_=False)
    network = load_checkpoint(cfg.ckpt_path, network, cfg["device"])
    logger.info(f"Model loaded to {cfg['device']}")

    outcnt = 0
    logger.info(conf.ref_audio)
    all_error_metrics = dict()
    sdr, snr, dild, ditd, lre = [], [], [], [], []
    ref_input_channels = model_cfg.model.a2b_model.ref_input_channels
    # clean old results
    if cfg.output.fld_path is not None:
        path = Path(cfg.output.fld_path)
        if path.exists():
            shutil.rmtree(cfg.output.fld_path, ignore_errors=True)

    in_out_scale = model_cfg.data.datasets_conf.in_out_scale

    for test_set in conf.ref_audio:
        for rec in test_set["recording_ids"]:
            # load ref audio
            test_set.bin_from_video = model_cfg.data.datasets_conf.get(
                "bin_from_video", False
            )
            ref_audio, rfs, gnd, ref_fn_name = load_audio_data(test_set, rec_id=rec, ref_input_channels=ref_input_channels, in_out_scale=in_out_scale)
            logger.info(
                f"ref_audio: {ref_audio.shape[1]/rfs}, audio shape={ref_audio.shape},"
            )
            assert rfs == 48000, f"Model accepts only 48000Hz audio signals!"
            audio_len = ref_audio.shape[-1] / rfs
            # Render
            out = offline_rendering(network, ref_audio, device=cfg.device,)
            out = out.squeeze(0)
            if cfg.output.fld_path is None:
                fld_path = f"{model_cfg['training']['artifacts_dir']}/rendering/"
                cfg.output.fld_path = fld_path
            else:
                fld_path = cfg.output.fld_path
            fld_path = f"{fld_path}/Result{outcnt:04d}"
            Path(fld_path).mkdir(parents=True, exist_ok=True)
            fn = f"{fld_path}/render.wav"

            logger.info(f"==== Result{outcnt:04d} ===")
            logger.info(f"Input: {ref_fn_name}")
            logger.info(f"Output: {fn}")
            ta.save(fn, out, rfs, channels_first=True, encoding="PCM_F")

            fn = f"{fld_path}/gnd.wav"
            ta.save(fn, gnd, rfs, channels_first=True, encoding="PCM_F")

            fn = f"{fld_path}/input_2ch.wav"
            ta.save(fn, ref_audio[0:2,:], rfs, channels_first=True, encoding="PCM_F")

            if cfg.save_input:
                fn = f"{fld_path}/input.wav"
                ta.save(fn, ref_audio, rfs, channels_first=True, encoding="PCM_F", backend=f"{TA_BACKEND}")

            # Report metrics
            logger.info(f"Out: {out.shape}, Gnd: {gnd.shape}")
            error_metrics = report(out.unsqueeze(0), gnd.unsqueeze(0), verbose=False)
            logger.info(f"Error metrics: {error_metrics}")
            sdr.append(error_metrics["sdr"])
            dild.append(error_metrics["dild"])
            ditd.append(error_metrics["ditd"])
            snr.append(error_metrics["snr"])
            lre.append(error_metrics["lre"])

            all_error_metrics[str(outcnt)] = {"error": error_metrics, "name": rec}
            ##
            del out
            del gnd
            del ref_audio
            outcnt = outcnt + 1
    # Save error metrics    
    mean_sdr = np.mean(np.array(sdr))
    mean_dild = np.mean(np.array(dild))
    mean_ditd = np.mean(np.array(ditd))
    mean_snr = np.mean(np.array(snr))
    mean_lre = np.mean(np.array(lre))
    logger.info(f"Mean SDR: {mean_sdr}")
    logger.info(f"Mean SNR: {mean_snr}")
    logger.info(f"Mean DILD: {mean_dild}")
    logger.info(f"Mean DITD: {mean_ditd}")
    logger.info(f"Mean LRE: {mean_lre}")


    fld_path = cfg.output.fld_path
    all_error_metrics["mean_snr"] = mean_snr.astype(float)
    all_error_metrics["mean_sdr"] = mean_sdr.astype(float)
    all_error_metrics["mean_ditd"] = mean_ditd.astype(float)
    all_error_metrics["mean_dild"] = mean_dild.astype(float)
    all_error_metrics["mean_lre"] = mean_lre.astype(float)
    all_error_metrics["model_ckpt"] = cfg.ckpt_path
    with open(f"{fld_path}/error_metrics.json", "w") as f:
        json.dump(all_error_metrics, f, cls=NumpyArrayEncoder, indent=4)


base_config = OmegaConf.create(
    {
        "ckpt_path": None,
        "device": "cpu",
        "config_name": None,
        "config_dir": "configs/evaluations/",
        "save_input": False,
    }
)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_config, conf_cli)
    assert conf.ckpt_path is not None, "Please provide a checkpoint path!"
    assert conf.config_name is not None, "Please provide a config name!"

    conf_yaml = OmegaConf.load(Path(conf.config_dir) / f"{conf.config_name}.yaml")

    conf = OmegaConf.merge(conf, conf_yaml, conf_cli)
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(conf)}")
    main(conf)

    logger.info(f"========== Summary =========")
    logger.info(f"Output written to: {conf.output.fld_path}")
    logger.info(f"https://hewen.sb.facebook.net:8082/visualize/{conf.output.fld_path}")
    logger.info(f"========== End =========")

# python inference/evaluations.py config_name="bytedance_paper" ckpt_path="/mnt/home/idgebru/a2b_exp_dumps/model_training/bytedance/v3/checkpoints/last.ckpt" device="cuda:2"
# python inference/evaluations.py config_name="urbansounds" ckpt_path="/mnt/home/idgebru/a2b_exp_dumps/model_training/urbansounds/v3/checkpoints/last.ckpt" device="cuda:2"
# python inference/evaluations.py config_name="audessy_mk32" ckpt_path="/mnt/home/idgebru/a2b_exp_dumps/model_training/audessy_mk32_kemar/v2/checkpoints/last.ckpt" device="cuda:2"
# python inference/evaluations.py config_name="hearsay_mk128" ckpt_path="/mnt/home/idgebru/a2b_exp_dumps/model_training/hearsay_mk128_kemar/v2/checkpoints/last.ckpt" device="cuda:7"
# python inference/evaluations.py config_name="n2s_mk128" ckpt_path="/mnt/home/idgebru/a2b_exp_dumps/model_training/n2ss_mk128_kemar/v2/checkpoints/last.ckpt" device="cuda:7"
