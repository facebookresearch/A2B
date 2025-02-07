# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import numpy as np
import torch as th
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy import signal

import nnModules.layers.losses as audioLosses
import utils.audio_transforms

# Signal-Distortion Ratio (SDR)
# Log Spectral Distance  (LSD)
# DITD (ms)
# DILD
eps = 1e-8


def measure_sdr(predicted_binaural, gt_binaural, zero_mean=True):
    assert predicted_binaural.shape == gt_binaural.shape
    assert predicted_binaural.shape[0] == 2
    predicted_binaural = predicted_binaural.ravel()
    gt_binaural = gt_binaural.ravel()
    sdr = gt_binaural**2 / ((gt_binaural - predicted_binaural)**2 + eps)
    sdr = 10.0 * np.log10(sdr + eps).astype(float)
    return  sdr.mean()


def rssq(x):
    #root-sum-of-squares (RSS) 
    return np.sqrt(np.sum(x**2))

def measure_snr_matlab(predicted_binaural, gt_binaural):
    assert predicted_binaural.shape == gt_binaural.shape
    assert predicted_binaural.shape[0] == 2
    predicted_binaural = predicted_binaural.ravel()
    gt_binaural = gt_binaural.ravel()
    noise = gt_binaural - predicted_binaural
    snr = 20.0 * np.log10(rssq(gt_binaural) / (rssq(noise) + eps)).astype(float)
    return snr


def measure_snr(predicted_binaural, gt_binaural):
    assert predicted_binaural.shape == gt_binaural.shape
    assert predicted_binaural.shape[0] == 2
    mse_distance = np.mean((gt_binaural - predicted_binaural)**2, axis=1)
    bin_snr = 10.0 * np.log10((np.mean(gt_binaural ** 2, axis=1) + eps) / (mse_distance + eps))
    return np.mean(bin_snr).astype(float)

def lag_finder(y1, y2, sr, fn):
    n = len(y1)
    corr = signal.correlate(y1, y2, mode="full", method="fft")
    lags = signal.correlation_lags(y1.size, y2.size, mode="full")
    delay = lags[np.argmax(corr)]
    return delay

def itd(binaural_signal, rfs: int = 48000):
    assert binaural_signal.shape[0] == 2
    n = binaural_signal.shape[1]
    xcorr = signal.correlate(binaural_signal[0, :], binaural_signal[1, :], mode="same")
    acorr_l = signal.correlate(
        binaural_signal[0, :], binaural_signal[0, :], mode="same"
    )
    acorr_r = signal.correlate(
        binaural_signal[1, :], binaural_signal[1, :], mode="same"
    )
    mid_idx = int(n / 2)
    corr = xcorr / np.sqrt(acorr_l[mid_idx] * acorr_r[mid_idx])
    delay_arr = np.linspace(-0.5 * n / rfs, 0.5 * n / rfs, n)
    delay = delay_arr[np.argmax(corr)]
    return delay


def diff_itd(predicted_binaural, gt_binaural, rfs: int = 48000):
    assert predicted_binaural.shape == gt_binaural.shape
    assert predicted_binaural.shape[0] == 2

    pred_delay = itd(predicted_binaural, rfs)
    gt_delay = itd(gt_binaural, rfs)
    return np.abs(pred_delay - gt_delay)


def lsd(predicted_binaural: th.Tensor, gt_binaural: th.Tensor, stft_params: DictConfig):
    assert predicted_binaural.shape[-1] == gt_binaural.shape[-1]
    assert gt_binaural.shape[0] == 2
    assert predicted_binaural.shape[0] == 2

    loss = audioLosses.LogSpectralDistance(**stft_params)
    diff_ild_loss = loss(predicted_binaural, gt_binaural)
    return diff_ild_loss


def dild(
    data: th.Tensor, gt: th.Tensor, stft_params: DictConfig = None, eps: float = 1e-8
):
    sfft_calc = utils.audio_transforms.FourierTransform(**stft_params)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    data_stft = sfft_calc.stft(
        data, onesided=True
    )  # shape of (batch, nch=2, freq, time, 2)
    data_mag = th.sqrt(
        th.clamp(data_stft[..., 0].pow(2) + data_stft[..., 1].pow(2), min=eps)
    )  # shape of (batch, nch=2, freq, time)
    data_mag = data_mag.permute(0, 2, 3, 1)

    data_ild_lr = data_mag[..., 0] - data_mag[..., 1]
    data_ild_lr = data_ild_lr.view(-1)
    data_ild_rl = data_mag[..., 1] - data_mag[..., 0]
    data_ild_rl = data_ild_rl.view(-1)

    # gt
    gt_stft = sfft_calc.stft(
        gt, onesided=True
    )  # shape of (batch, nch=2, freq, time, 2)
    gt_mag = th.sqrt(
        th.clamp(gt_stft[..., 0].pow(2) + gt_stft[..., 1].pow(2), min=eps)
    )  # shape of (batch, nch=2, freq, time)
    gt_mag = gt_mag.permute(0, 2, 3, 1)

    gt_ild_lr = gt_mag[..., 0] - data_mag[..., 1]
    gt_ild_lr = gt_ild_lr.view(-1)
    gt_ild_rl = gt_mag[..., 1] - data_mag[..., 0]
    gt_ild_rl = gt_ild_rl.view(-1)

    # mask
    mask_lr = (data_ild_lr > eps) & (gt_ild_lr > eps)
    indices = th.nonzero(mask_lr).view(-1)
    data_ild_lr = data_ild_lr[indices]
    data_ild_lr = data_ild_lr.view(-1)
    gt_ild_lr = gt_ild_lr[indices]
    gt_ild_lr = gt_ild_lr.view(-1)

    mask_rl = (data_ild_rl > eps) & (gt_ild_rl > eps)
    indices = th.nonzero(mask_rl).view(-1)
    data_ild_rl = data_ild_rl[indices]
    data_ild_rl = data_ild_rl.view(-1)
    gt_ild_rl = gt_ild_rl[indices]
    gt_ild_rl = gt_ild_rl.view(-1)

    diff_ild_loss = th.abs(th.log(th.mean(data_ild_lr) / th.mean(gt_ild_lr))) + th.abs(th.log(th.mean(data_ild_rl) / th.mean(gt_ild_rl)))
    diff_ild_loss = diff_ild_loss/2.0 * 10.0
    return diff_ild_loss

def lre(predicted_binaural: th.Tensor, gt_binaural: th.Tensor):
    assert predicted_binaural.shape[-1] == gt_binaural.shape[-1]
    assert gt_binaural.shape[0] == 2
    pred_lr_ratio = 10 * th.log10((predicted_binaural[0,:].pow(2).sum(-1)+ eps) / (predicted_binaural[1,:].pow(2).sum(-1) + eps))
    tgt_lr_ratio = 10 * th.log10((gt_binaural[0,:].pow(2).sum(-1)+ 1e-5) / (gt_binaural[1,:].pow(2).sum(-1) + 1e-5))
    lr_ratio = (pred_lr_ratio - tgt_lr_ratio).abs().mean()
    return lr_ratio

def report(
    predicted_binaural: th.Tensor,
    gt_binaural: th.Tensor,
    stft_params: DictConfig = OmegaConf.create(
        {"fft_bins": 2048, "sample_rate": 48000,}
    ),
    verbose: bool = False,
):
    assert predicted_binaural.shape[0] == 2
    assert predicted_binaural.shape == gt_binaural.shape

    results = {}
    rfs = stft_params.get("sample_rate", 48000)
    #  LSD
    lsd_loss = lsd(predicted_binaural, gt_binaural, stft_params)
    results["lsd"] = lsd_loss.cpu().numpy()
    if verbose:
        logger.info(f"LSD: {lsd_loss}")

    #  DILD
    dild_loss = dild(predicted_binaural, gt_binaural, stft_params)
    if verbose:
        logger.info(f"DILD: {dild_loss}")
    results["dild"] = dild_loss.cpu().numpy()

    #  LRE
    lre_loss = lre(predicted_binaural, gt_binaural)
    if verbose:
        logger.info(f"LRE: {lre_loss}")
    results["lre"] = lre_loss.cpu().numpy()

    #### Numpy
    predicted_binaural = predicted_binaural.squeeze(0).detach().cpu().numpy()
    gt_binaural = gt_binaural.squeeze(0).detach().cpu().numpy()

    #  SNR
    snr = measure_snr(predicted_binaural, gt_binaural)
    if verbose: 
        logger.info(f"SNR: {snr}")
    results["snr"] = snr

    # snr matlab
    snr_matlab = measure_snr_matlab(predicted_binaural, gt_binaural)
    if verbose:
        logger.info(f"SNR Matlab: {snr_matlab}")
    results["snr_matlab"] = snr_matlab
    # SDR
    sdr = measure_sdr(gt_binaural, predicted_binaural)
    if verbose:
        logger.info(f"SDR: {snr}")
    results["sdr"] = sdr

    # DITD
    diff_itd_loss = diff_itd(predicted_binaural, gt_binaural, rfs=rfs)
    if verbose:
        logger.info(f"DITD(ms): {diff_itd_loss * 1000}")
    results["ditd"] = diff_itd_loss * 1000.0

    return results


if __name__ == "__main__":

    b = 1
    nch = 2
    rfs = 48000
    predicted_binaural = th.randn(b, nch, rfs)
    gt_binaural = th.randn(b, nch, 48000)
    stft_params = OmegaConf.create({"fft_bins": 2048,})
    #  test LSD
    lsd_loss = lsd(predicted_binaural, gt_binaural, stft_params)
    logger.info(f"GND: {gt_binaural.shape}, Predicted: {predicted_binaural.shape}")
    logger.info(f"LogSpectralDistance: {lsd_loss}")

    # test DILD
    dild_loss = dild(predicted_binaural, gt_binaural, stft_params)
    logger.info(f"DILD: {dild_loss}")

    # test LRE
    lre_loss = lre(predicted_binaural, gt_binaural)
    logger.info(f"LRE: {lre_loss}")

    #### Numpy
    predicted_binaural = np.random.randn(nch, rfs)
    gt_binaural = np.random.randn(nch, 48000)

    # test SDR
    sdr = measure_sdr(predicted_binaural, gt_binaural)
    logger.info(f"sdr: {sdr}")

    # test SNR
    snr = measure_snr(predicted_binaural, gt_binaural)
    logger.info(f"snr: {snr}")

    # test SNR matlab
    snr_matlab = measure_snr_matlab(predicted_binaural, gt_binaural)
    logger.info(f"snr_matlab: {snr_matlab}")

    # test ITD
    itd_gt = itd(gt_binaural, rfs=rfs)
    logger.info(f"itd: {itd_gt}, \\$mp$ ={1.0/rfs}")

    # test diff_ITD
    diff_itd_loss = diff_itd(predicted_binaural, gt_binaural, rfs=rfs)
    logger.info(f"diff_itd: {diff_itd_loss}")

    # test DITD
    # predicted_binaural = th.randn(10, nch, 48000)
    # gt_binaural = th.randn(10, nch, 48000)
    # ditd = DITD(predicted_binaural, gt_binaural, rfs=48000)
    # logger.info(f"ditd: {ditd}")
