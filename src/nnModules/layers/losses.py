# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


from typing import Iterable, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
import torchaudio as ta

import utils.audio_transforms
from loguru import logger


def _unpad(data, mask_left=0, mask_right=0):
    data = data[..., mask_left:]
    if mask_right > 0:
        data = data[..., :-mask_right]
    return data


def tupleset(t: Iterable, i: int, value) -> Tuple:
    lst = list(t)
    lst[i] = value
    return tuple(lst)

class L1Loss(th.nn.Module):
    """
    l1 loss
    """

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        return th.mean(th.abs(data.view(B, -1) - target.view(B, -1)), dim=-1)


class L2Loss(th.nn.Module):
    """
    l2 loss
    """

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        # logger.info(f"data:{data.shape}, target:{target.shape}")
        return th.mean((data.view(B, -1) - target.view(B, -1)).pow(2), dim=-1)


class LogSTFTMagnitudeDiffLoss(th.nn.Module):
    """Log STFT magnitude difference loss module."""

    def __init__(self, mask_left=0, mask_right=0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.eps = 1e-8
        self.axis = kwargs.get("axis", "time")

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        data_stft_real, data_stft_imag = data_stft[..., 0], data_stft[..., 1]
        target_stft = self.fft.stft(target, onesided=True)
        target_stft_real, target_stft_imag = target_stft[..., 0], target_stft[..., 1]
        # print(f"data:{data.shape},data_sftf:{data_stft.shape}, target:{target.shape}, target_sftf:{target_stft.shape}")
        data_mag = th.sqrt(
            th.clamp(data_stft_real**2 + data_stft_imag**2, min=self.eps)
        ).transpose(2, 1)
        target_mag = th.sqrt(
            th.clamp(target_stft_real**2 + target_stft_imag**2, min=self.eps)
        ).transpose(2, 1)

        log_stft_mag = F.l1_loss(th.log(data_mag), th.log(target_mag))
        return log_stft_mag

class SpectralConvergengeLoss(th.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self, mask_left=0, mask_right=0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.eps = 1e-6
        self.axis = kwargs.get("axis", "time")

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        data_stft_real, data_stft_imag = data_stft[..., 0], data_stft[..., 1]
        target_stft = self.fft.stft(target, onesided=True)
        target_stft_real, target_stft_imag = target_stft[..., 0], target_stft[..., 1]
        # print(f"data:{data.shape},data_sftf:{data_stft.shape}, target:{target.shape}, target_sftf:{target_stft.shape}")
        data_mag = th.sqrt(
            th.clamp(data_stft_real**2 + data_stft_imag**2, min=self.eps)
        ).transpose(2, 1)
        target_mag = th.sqrt(
            th.clamp(target_stft_real**2 + target_stft_imag**2, min=self.eps)
        ).transpose(2, 1)

        spc_loss = th.norm(data_mag - target_mag, p="fro") / th.norm(
            target_mag, p="fro"
        )
        return spc_loss


class LogSpectralDistance(th.nn.Module):
    """log-spectral distance between two waveforms."""

    def __init__(self, mask_left=0, mask_right=0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.eps = 1e-6
        self.axis = kwargs.get("axis", "time")

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        # normalize
        data = data / th.sqrt(th.sum(data**2.0, -1, keepdim=True))
        target = target / th.sqrt(th.sum(target**2.0, -1, keepdim=True))
        # stft
        data_stft = self.fft.stft(data, onesided=True)
        data_stft_real, data_stft_imag = data_stft[..., 0], data_stft[..., 1]
        target_stft = self.fft.stft(target, onesided=True)
        target_stft_real, target_stft_imag = target_stft[..., 0], target_stft[..., 1]
        # print(f"data:{data.shape},data_sftf:{data_stft.shape}, target:{target.shape}, target_sftf:{target_stft.shape}")
        data_mag = th.sqrt(
            th.clamp(data_stft_real**2 + data_stft_imag**2, min=self.eps)
        ).transpose(2, 1)
        target_mag = th.sqrt(
            th.clamp(target_stft_real**2 + target_stft_imag**2, min=self.eps)
        ).transpose(2, 1)

        spc_loss = th.sqrt(th.mean((th.log(data_mag) - th.log(target_mag))**2.0))
        return spc_loss


class CumIntegralLoss(th.nn.Module):
    """Cumulative integral loss module."""

    def __init__(self, mask_left=0, mask_right=0, **kwargs):
        super().__init__()

    def cumtrapz(self, y: th.Tensor, axis: int = -1) -> th.Tensor:
        """
        y: Values to integrate.
        """
        d = th.arange(0, y.shape[axis]).to(y.device)
        d = th.diff(d, n=1, dim=axis)
        nd = len(y.shape)
        slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
        slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
        res = th.cumsum(d * (y[slice1] + y[slice2]) / 2.0, dim=axis)
        shape = list(res.shape)
        shape[axis] = 1
        res = th.cat([th.zeros(shape, dtype=res.dtype).to(y.device), res], dim=axis)
        return res

    def forward(self, data, target, **kwargs):
        data_int = self.cumtrapz(data**2.0, axis=-1)
        target_int = self.cumtrapz(target**2.0, axis=-1)
        loss = F.l1_loss(data_int, target_int)
        return loss


class ComplexSTFTL1Loss(th.nn.Module):
    def __init__(self, mask_left=0, mask_right=0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.eps = 1e-6

    def forward(self, data, target, **kwargs):
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        target_stft = self.fft.stft(target, onesided=True)
        return F.l1_loss(target_stft, data_stft)


class STFTMagnitudeLoss(th.nn.Module):
    """STFT magnitude loss module.
    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L2"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, mask_left=0, mask_right=0, log=True, **kwargs):
        super().__init__()
        self.log = log
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.eps = 1e-6

    def forward(self, data, target, **kwargs):
        # eng_data = th.sum(data**2, dim=-1).mean(-1)
        # eng_target = th.sum(target**2, dim=-1).mean(-1)
        # print(f"Data: {data.shape}, {eng_data}")
        # print(f"Traget: {target.shape}, {eng_target}")
        B = data.shape[0]
        data_stft = th.view_as_complex(self.fft.stft(data, onesided=True))
        target_stft = th.view_as_complex(self.fft.stft(target, onesided=True))
        data_mag = th.sqrt(
            th.clamp((data_stft.real**2) + (data_stft.imag**2), min=self.eps)
        )
        target_mag = th.sqrt(
            th.clamp((target_stft.real**2) + (target_stft.imag**2), min=self.eps)
        )
        if self.log:
            data_mag = th.log(data_mag + self.eps)
            target_mag = th.log(target_mag + self.eps)
        return th.mean(th.abs(data_mag.view(B, -1) - target_mag.view(B, -1)), dim=-1)


class Binaural_Difference_ILDLoss(th.nn.Module):
    def __init__(self, nch: int = 2, mask_left: int = 0, mask_right: int = 0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.nch = nch
        self.eps = 1e-6
        self.w_log_mag = kwargs.get('w_log_mag',1.0)
        self.w_lin_mag = kwargs.get('w_lin_mag',0.0)
        print(f"w_log_mag: {self.w_log_mag}, w_lin_mag: {self.w_lin_mag}")

    def forward(self, data, target, **kwargs):
        assert data.shape[1] == self.nch
        assert target.shape[1] == self.nch
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        target_stft = self.fft.stft(
            target, onesided=True
        )  # shape of (batch, nch=2, freq, time, 2)
        data_mag = th.sqrt(
            th.clamp(
                data_stft[..., 0].pow(2) + data_stft[..., 1].pow(2), min=self.eps
            )
        )
        target_mag = th.sqrt(
            th.clamp(
                target_stft[..., 0].pow(2) + target_stft[..., 1].pow(2), min=self.eps
            )
        )  # shape of (batch, nch=2, freq, time)
        target_mag = target_mag.permute(0,2,3,1)
        data_mag = data_mag.permute(0,2,3,1)

        target_ild_lr = target_mag[...,0] - target_mag[...,1]
        target_ild_lr = target_ild_lr.view(-1)
        data_ild_lr = data_mag[...,0] - data_mag[...,1]
        data_ild_lr = data_ild_lr.view(-1)
        target_ild_rl = target_mag[...,1] - target_mag[...,0]
        target_ild_rl = target_ild_rl.view(-1)
        data_ild_rl = data_mag[...,1] - data_mag[...,0]
        data_ild_rl = data_ild_rl.view(-1)

        log_loss_lr, log_loss_rl = 0.0, 0.0
        lin_loss_lr, lin_loss_rl = 0.0, 0.0
        # case left channel is louder: use *lr
        mask_lr = (target_ild_lr > self.eps) & (data_ild_lr > self.eps)
        indices = th.nonzero(mask_lr).view(-1)
        if indices.shape[0] > 0:
            target, data = th.index_select(target_ild_lr.view(-1), 0, indices), th.index_select(data_ild_lr.view(-1), 0, indices)
            log_loss_lr = F.l1_loss(th.log(data), th.log(target)) if self.w_log_mag else 0.0
            lin_loss_lr = F.l1_loss(data, target) if self.w_lin_mag else 0.0
        # case right channel is louder: use *rl
        mask_rl = (target_ild_rl > self.eps) & (data_ild_rl > self.eps)
        indices = th.nonzero(mask_rl).view(-1)
        if indices.shape[0] > 0:
            target, data = th.index_select(target_ild_rl.view(-1), 0, indices), th.index_select(data_ild_rl.view(-1), 0, indices)
            log_loss_rl = F.l1_loss(th.log(data), th.log(target)) if self.w_log_mag else 0.0
            lin_loss_rl = F.l1_loss(data, target) if self.w_lin_mag else 0.0
        diff_ild_loss = self.w_lin_mag * (lin_loss_lr + lin_loss_rl) + self.w_log_mag * (log_loss_lr + log_loss_rl)
        return diff_ild_loss


class Binaural_Difference_ILDLoss_Log(th.nn.Module):
    def __init__(self, nch: int = 2, mask_left: int = 0, mask_right: int = 0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.nch = nch
        self.eps = 1e-8
        self.ild_eps = kwargs.get("eps", 1e-6)
        self.w_lin_mag = kwargs.get("w_lin_mag", 1.0)
        self.mask_left = mask_left
        self.mask_right = mask_right

    def forward(self, data, target, **kwargs):
        data = _unpad(data, mask_left=self.mask_left, mask_right=self.mask_right)
        target = _unpad(target, mask_left=self.mask_left, mask_right=self.mask_right)
        assert data.shape[1] == self.nch
        assert target.shape[1] == self.nch
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        target_stft = self.fft.stft(
            target, onesided=True
        )  # shape of (batch, nch=2, freq, time, 2)
        data_mag = th.sqrt(
            th.clamp(data_stft[..., 0].pow(2) + data_stft[..., 1].pow(2), min=self.eps)
        )
        target_mag = th.sqrt(
            th.clamp(
                target_stft[..., 0].pow(2) + target_stft[..., 1].pow(2), min=self.eps
            )
        )  # shape of (batch, nch=2, freq, time)
        target_mag = target_mag.permute(0, 2, 3, 1)
        data_mag = data_mag.permute(0, 2, 3, 1)

        target_ild_lr = th.log(target_mag[..., 0] / target_mag[..., 1])
        target_ild_lr = target_ild_lr.view(-1)
        data_ild_lr = th.log(data_mag[..., 0] / data_mag[..., 1])
        data_ild_lr = data_ild_lr.view(-1)
        target_ild_rl = th.log(target_mag[..., 1] / target_mag[..., 0])
        target_ild_rl = target_ild_rl.view(-1)
        data_ild_rl = th.log(data_mag[..., 1] / data_mag[..., 0])
        data_ild_rl = data_ild_rl.view(-1)

        lin_loss_lr, lin_loss_rl = 0.0, 0.0
        # prob = th.rand(1)
        # if prob <= 0.5:
        # case left channel is louder: use *lr
        mask_lr = target_ild_lr > (1.0 + self.eps)
        indices = th.nonzero(mask_lr).view(-1)
        if indices.shape[0] > 0:
            target, data = (
                th.index_select(target_ild_lr.view(-1), 0, indices),
                th.index_select(data_ild_lr.view(-1), 0, indices),
            )
            lin_loss_lr = F.mse_loss(data, target)
        else:
            print(
                f"LR this shouldn't be the case. check ild_eps: {self.ild_eps}. {th.max(target_ild_lr)}, {th.max(data_ild_lr)}"
            )
        # case right channel is louder: use *rl
        mask_rl = target_ild_rl > (1.0 + self.eps)
        indices = th.nonzero(mask_rl).view(-1)
        if indices.shape[0] > 0:
            target, data = (
                th.index_select(target_ild_rl.view(-1), 0, indices),
                th.index_select(data_ild_rl.view(-1), 0, indices),
            )

            lin_loss_rl = F.mse_loss(data, target)
        else:
            print(
                f"RL this shouldn't be the case. check ild_eps: {self.ild_eps}. {th.max(target_ild_rl)}, {th.max(data_ild_rl)}"
            )
        return self.w_lin_mag * (lin_loss_lr +  lin_loss_rl)


class Binaural_CoherenceLoss(th.nn.Module):
    def __init__(self, nch: int = 2, mask_left: int = 0, mask_right: int = 0, **kwargs):
        super().__init__()
        self.fft = utils.audio_transforms.FourierTransform(**kwargs)
        self.nch = nch
        self.eps = 1e-6
        self.w_log_mag = kwargs.get('w_log_mag',1.0)
        self.w_lin_mag = kwargs.get('w_lin_mag',0.0)
        print(f"w_log_mag: {self.w_log_mag}, w_lin_mag: {self.w_lin_mag}")

    def forward(self, data, target, **kwargs):
        assert data.shape[1] == self.nch
        assert target.shape[1] == self.nch
        B = data.shape[0]
        data_stft = self.fft.stft(data, onesided=True)
        target_stft = self.fft.stft(
            target, onesided=True
        )  # shape of (batch, nch=2, freq, time, 2)
        # conj
        data_conj = th.view_as_real(th.conj(th.view_as_complex(data_stft)))
        target_conj = th.view_as_real(th.conj(th.view_as_complex(target_stft)))
        # mag
        data_mag = th.sqrt(th.clamp(data_stft[..., 0].pow(2) + data_stft[..., 1].pow(2), min=self.eps))
        target_mag =  th.sqrt(th.clamp(target_stft[..., 0].pow(2) + target_stft[..., 1].pow(2), min=self.eps))  # shape of (batch, nch=2, freq, time)
        target_mag = target_mag.permute(0,2,3,1)
        data_mag = data_mag.permute(0,2,3,1) # (batch, time, freq, 2)
        print(f"target_conj: {target_conj.shape}, data_stft: {data_stft.shape}")
        coherence = target_conj * data_stft
        coherence_mag =  th.sqrt(th.clamp(coherence[..., 0].pow(2) + coherence[..., 1].pow(2), min=self.eps))
        coherence_mag = coherence_mag.permute(0,2,3,1) # (batch, time, freq, 2)
        coherence_loss = 1.0 - coherence_mag / (target_mag * data_mag)
        return th.mean(coherence_loss)




"""
itentifier: one of the valid loss identifiers in losses.keys()
params: a dict containing kwargs for the specific loss
returns: an instance of the loss specified by identifier
"""


def create(identifier, params={}):
    losses = {
        "l1": L1Loss,
        "l2": L2Loss,
        "LogSTFTMagnitudeDiffLoss": LogSTFTMagnitudeDiffLoss,
        "SpectralConvergengeLoss": SpectralConvergengeLoss,
        "CumIntegralLoss": CumIntegralLoss,
        "ComplexSTFTL1Loss": ComplexSTFTL1Loss,
        "STFTMagnitudeLoss": STFTMagnitudeLoss,
        "Binaural_Difference_ILDLoss": Binaural_Difference_ILDLoss,
        "Binaural_Difference_ILDLoss_Log": Binaural_Difference_ILDLoss_Log,
        "LogSpectralDistance": LogSpectralDistance,
    }
    if not identifier in losses.keys():
        raise Exception(
            f"{identifier} not found. Available loss identifiers are {list(losses.keys())}."
        )
    return losses[identifier](**params)


if __name__ == "__main__":
    # test LogSpectralDistance
    nch = 2
    data = th.randn(10, nch, 48000)
    target = th.randn(10, nch, 48000)
    stft_params = {
        "fft_bins": 2048,
    }
    loss = LogSpectralDistance(**stft_params)
    diff_ild_loss = loss(data, target)
    print(f"in: {data.shape}, {target.shape}")
    print(f"loss: {diff_ild_loss}")
    print(f"out: LogSpectralDistance:{diff_ild_loss.shape}")
