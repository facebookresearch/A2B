# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def div_byn_counter(x: int, n: int):
    assert n != 1, f"n must be greater than 1, n:{n}"
    assert x != 1, f"x must be greater than 1, x:{x}"
    ret = []
    while x > n:
        x = x / n
        ret.append(x)
    return ret


class ClampedSineAct(nn.Module):
    def __init__(self, w0=1.0, thr=None):
        super().__init__()
        self.w0 = w0
        self.thr = thr

    def forward(self, x):
        if self.thr is not None:
            return th.clamp(th.sin(self.w0 * x), min=-1.0 * self.thr, max=self.thr)
        else:
            return th.sin(self.w0 * x)


class QuadChannelwiseInputDecompNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        ch_divider: int = 1,
        n_groups: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 32,
        drop_p: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # santiy check for progressive channel grouping
        self.n_groups = n_groups
        levels = div_byn_counter(in_channels // ch_divider, self.n_groups)
        logger.info(f"levels: {levels}")
        assert levels[-1] == self.n_groups, f"Last level must be {self.n_groups}"
        self.n_levels = len(levels) + 1

        self.ch_divider = ch_divider
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.bias = kwargs.get("bias", False)
        self.kernel_size = kwargs.get("conv_len", 1)

        self.first = nn.Conv1d(
            self.ch_divider,
            hidden_channels,
            kernel_size=self.kernel_size,
            bias=self.bias,
        )
        self.act = ClampedSineAct()
        self.second = nn.Conv1d(
            hidden_channels, out_channels, kernel_size=1, bias=self.bias
        )
        self.progressive_ch = nn.ModuleList()
        this_ch = out_channels * self.n_groups
        for i in range(0, self.n_levels):
            self.progressive_ch.append(
                nn.Conv1d(this_ch, self.out_channels, kernel_size=1, bias=self.bias)
            )

        self.drop_layer = nn.Dropout1d(p=drop_p)

        self.padlen = (self.kernel_size - 1) // 2
        self.pad = nn.ConstantPad1d([self.padlen, self.padlen], 0.0)

        self.init_weights()

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.in_channels
        shape = x.size()
        # print(f"main waveform in: {x.shape}")
        # pack to batch dim
        x = x.view(
            shape[0] * shape[1] // self.ch_divider, self.ch_divider, shape[-1]
        )  # x: B x C x T -> BC//D x D x T
        out = self.act(
            self.second(self.act(self.first(self.pad(x))))
        )  # BC//D x D x T => BC//D x H x T => BC//D x C_OUT x T
        # unpack progressive
        this_shape = out.shape
        # print(f"main progressive in: {out.shape}")
        for i in range(0, self.n_levels):
            if i == 0:
                out = out.view(
                    shape[0] * shape[1] // self.ch_divider // self.n_groups,
                    self.out_channels * self.n_groups,
                    shape[-1],
                )
            else:
                out = out.view(
                    this_shape[0] // self.n_groups,
                    self.out_channels * self.n_groups,
                    shape[-1],
                )
            # print(f"progressive in: {out.shape}")
            out = self.drop_layer(
                self.act(self.progressive_ch[i](out))
            )  # BC//D//ng x C_OUT*ng x T => BC//D//ng x C_OUT x T
            this_shape = out.shape
            # print(f"progressive out: {out.shape}")
        return out

    def init_weights(self):
        # mean of 0 and a standard deviation of 1/sqrt(number of inputs to NN)
        self.first.weight.data.normal_(0.0, 1.0 / math.sqrt(self.ch_divider))
        self.second.weight.data.normal_(
            0.0, 1.0 / math.sqrt(self.ch_divider * self.hidden_channels)
        )
        for i in range(0, self.n_levels):
            self.progressive_ch[i].weight.data.normal_(
                0.0, 1.0 / math.sqrt(self.out_channels)
            )


class OutputCompNet(nn.Module):
    def __init__(self, skip_channels, out_channels):
        super().__init__()
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.first = nn.Conv1d(
            skip_channels, skip_channels, kernel_size=1, groups=1, bias=False
        )
        self.act_first = ClampedSineAct()
        self.last = nn.Conv1d(
            skip_channels, out_channels, kernel_size=1, groups=2, bias=False
        )
        self.init_weights()

    def init_weights(self):
        self.first.weight.data.normal_(0.0, 1.0 / math.sqrt(self.skip_channels))
        self.last.weight.data.normal_(0.0, 1.0 / math.sqrt(self.out_channels))

    def forward(self, x):
        out = self.last(self.act_first(self.first(x)))
        return out


class CasualFiLMedWaveNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        conv_len: int = 7,
        dilation: int = 1,
        drop_p: float = 0.0,
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.conv_len = conv_len
        self.dilation = dilation
        self.scale_parameter = scale_parameter
        self.padlen = [self.dilation * (self.conv_len - 1), 0]  # Left padding

        self.dilated_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.conv_len,
            dilation=self.dilation,
        )
        self.skip_conv = nn.Conv1d(self.in_channels, self.skip_channels, kernel_size=1)
        self.equalize_channels = (
            nn.Conv1d(self.out_channels, self.in_channels, kernel_size=1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.act = ClampedSineAct()
        # self.drop_layer = nn.Dropout1d(p=drop_p)
        self.scale = th.nn.Parameter(th.ones(self.in_channels) * 0.5)

    def forward(self, x: th.Tensor, scale: th.Tensor, inference: bool = False):
        out = self.act(self.dilated_conv(F.pad(x, pad=self.padlen)))
        out = self.drop_layer(self.equalize_channels(out))
        res = (out + x) * self.scale[None, :, None]
        return res, self.skip_conv(out)

    def _receptive_field(self):
        return (self.conv_len - 1) * self.dilation + 1


class CasualWaveNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        conv_len: int = 7,
        dilation: int = 1,
        drop_p: float = 0.0,
        groups: int = 1,
        scale_parameter: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.conv_len = conv_len
        self.dilation = dilation
        self.scale_parameter = scale_parameter
        self.bias = bias
        self.padlen = [self.dilation * (self.conv_len - 1), 0]  # Left padding

        self.dilated_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.conv_len,
            dilation=self.dilation,
            bias=self.bias,
        )
        self.skip_conv = nn.Conv1d(
            self.in_channels, self.skip_channels, kernel_size=1, bias=self.bias
        )
        self.equalize_channels = (
            nn.Conv1d(self.out_channels, self.in_channels, kernel_size=1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.act = ClampedSineAct()
        self.drop_layer = nn.Dropout1d(p=drop_p)
        if self.scale_parameter:
            self.scale = th.nn.Parameter(th.ones(self.in_channels) * 0.5)
        else:
            self.register_buffer("scale", th.ones(self.in_channels) * 0.5)

    def forward(self, x: th.Tensor):
        out = self.act(self.dilated_conv(F.pad(x, pad=self.padlen)))
        out = self.drop_layer(self.equalize_channels(out))
        res = (out + x) * self.scale[None, :, None]
        return res, self.skip_conv(out)

    def _receptive_field(self):
        return (self.conv_len - 1) * self.dilation + 1


class NonCasualWaveNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        conv_len: int = 7,
        dilation: int = 1,
        drop_p: float = 0.0,
        groups: int = 1,
        scale_parameter: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.conv_len = conv_len
        self.dilation = dilation
        self.scale_parameter = scale_parameter
        self.bias = bias

        self.padlen = (self.conv_len - 1) // 2 * dilation
        self.pad = nn.ConstantPad1d([self.padlen, self.padlen], 0.0)

        self.dilated_conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.conv_len,
            dilation=self.dilation,
            bias=self.bias,
        )
        self.skip_conv = nn.Conv1d(
            self.in_channels, self.skip_channels, kernel_size=1, bias=self.bias
        )
        self.equalize_channels = (
            nn.Conv1d(self.out_channels, self.in_channels, kernel_size=1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.act = ClampedSineAct()
        self.drop_layer = nn.Dropout1d(p=drop_p)
        if self.scale_parameter:
            self.scale = th.nn.Parameter(th.ones(self.in_channels) * 0.5)
        else:
            self.register_buffer("scale", th.ones(self.in_channels) * 0.5)

    def forward(self, x: th.Tensor):
        out = self.act(self.dilated_conv(self.pad(x)))
        out = self.drop_layer(self.equalize_channels(out))
        res = (out + x) * self.scale[None, :, None]
        return res, self.skip_conv(out)

    def _receptive_field(self):
        return (self.conv_len - 1) * self.dilation + 1


class Vanilla_WaveNets(nn.Module):
    def __init__(
        self,
        n_layers: int = 6,
        dilation_cycle_length: int = 3,
        in_channels: int = 64,
        skip_channels: int = 64,
        out_channels: int = 64,
        conv_len: int = 7,
        is_casual: bool = True,
        drop_p: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dilation_cycle_length = dilation_cycle_length
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.conv_len = conv_len
        self.convs = nn.ModuleList()
        self.rectv_field = 0
        self.is_casual = is_casual
        self.bias = bias

        layer_cnt = 0
        for i in range(self.n_layers):
            dilation = 2 ** (i % self.dilation_cycle_length)
            layer_cnt += 1
            if self.is_casual:
                self.convs.append(
                    CasualWaveNetBlock(
                        in_channels,
                        out_channels,
                        skip_channels,
                        conv_len=conv_len,
                        dilation=dilation,
                        drop_p=drop_p,
                        groups=1,
                        scale_parameter=True if layer_cnt < self.n_layers else False,
                        bias=self.bias,
                    )
                )
            else:
                self.convs.append(
                    NonCasualWaveNetBlock(
                        in_channels,
                        out_channels,
                        skip_channels,
                        conv_len=conv_len,
                        dilation=dilation,
                        drop_p=drop_p,
                        groups=1,
                        scale_parameter=True if layer_cnt < self.n_layers else False,
                        bias=self.bias,
                    )
                )
            self.rectv_field += self.convs[-1]._receptive_field() - 1

    def forward(self, x: th.Tensor):
        skips = []
        for i, layer in enumerate(self.convs):
            x, skip = layer(x)
            skips += [skip]
        return x, skips

    def receptive_field(self):
        return self.rectv_field


class A2B_Render(nn.Module):
    def __init__(
        self,
        ref_input_channels: int = 1,
        target_out_channels: int = 1,
        intermediate: bool = False,
        encoder_params: DictConfig = OmegaConf.create(
            {
                "ch_divider": 2,
                "n_groups": 2,
                "hidden_channels": 48,
                "out_channels": 64,
                "drop_p": 0.2,
                "bias": False,
            }
        ),
        wavenet_params: DictConfig = OmegaConf.create(
            {
                "n_layers": 6,
                "dilation_cycle_length": 3,
                "in_channels": 64,
                "out_channels": 65,
                "skip_channels": 64,
                "conv_len": 3,
                "is_casual": False,
                "drop_p": 0.0,
                "bias": True,
            }
        ),
    ):
        super().__init__()
        self.intermediate = intermediate
        self.input_net = QuadChannelwiseInputDecompNet(
            in_channels=ref_input_channels,
            ch_divider=encoder_params.ch_divider,
            n_groups=encoder_params.n_groups,
            hidden_channels=encoder_params.hidden_channels,
            out_channels=encoder_params.out_channels,
            drop_p=encoder_params.drop_p,
            bias=encoder_params.bias,
        )
        assert (
            encoder_params.out_channels == wavenet_params.in_channels
        ), f"encoder out_channels {encoder_params.out_channels} and wavenet in_channels {wavenet_params.in_channels} must to be equal to each other"
        assert (
            ref_input_channels % encoder_params.ch_divider
        ) == 0, f"ref_input_channels {ref_input_channels} must be divisible by ch_divider {encoder_params.ch_divider}"
        # waveform to intermediate features
        self.wavenet = Vanilla_WaveNets(
            n_layers=wavenet_params.n_layers,
            dilation_cycle_length=wavenet_params.dilation_cycle_length,
            in_channels=wavenet_params.in_channels,
            out_channels=wavenet_params.out_channels,
            skip_channels=wavenet_params.skip_channels,
            conv_len=wavenet_params.conv_len,
            is_casual=wavenet_params.is_casual,
            drop_p=wavenet_params.drop_p,
            bias=wavenet_params.bias,
        )
        # binaural
        if self.intermediate:
            self.output_stages = nn.ModuleList()
            for i in range(wavenet_params.n_layers):
                self.output_stages.append(
                    OutputCompNet(
                        skip_channels=wavenet_params.skip_channels,
                        out_channels=target_out_channels,
                    )
                )
        else:
            self.output_stages = OutputCompNet(
                skip_channels=wavenet_params.skip_channels,
                out_channels=target_out_channels,
            )

    def forward(self, waveform: th.Tensor) -> th.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B, 1, T]
        wave_feat = self.input_net(waveform)
        _, skips = self.wavenet(wave_feat)
        if self.intermediate:
            out_final = []
            for i in range(1, len(self.output_stages) + 1):
                this_sum = th.sum(th.stack(skips[:i], dim=0), dim=0) / math.sqrt(i)
                out_final.append(self.output_stages[i - 1](this_sum))
            return {"output": out_final[-1], "intermediate": out_final[:-1]}
        else:
            this_sum = th.sum(th.stack(skips, dim=0), dim=0) / math.sqrt(len(skips))
            out_final = self.output_stages(this_sum)
            return {"output": out_final}


if __name__ == "__main__":
    ref_input_channels = 128
    target_out_channels = 2
    waveform = th.randn(32, ref_input_channels, 12)

    encoder_params: DictConfig = OmegaConf.create(
        {
            "ch_divider": 4,
            "n_groups": 2,
            "hidden_channels": 32,
            "out_channels": 32,
            "drop_p": 0.25,
            "bias": False,
        }
    )

    wavenet_params: DictConfig = OmegaConf.create(
        {
            "n_layers": 24,
            "dilation_cycle_length": 12,
            "in_channels": encoder_params.out_channels,
            "out_channels": 35,
            "skip_channels": 32,
            "conv_len": 3,
            "n_blocks": 1,
            "is_casual": True,
            "drop_p": 0.0,
            "scale_parameter": False,
            "bias": True,
        }
    )

    model = A2B_Render(
        ref_input_channels=ref_input_channels,
        target_out_channels=target_out_channels,
        encoder_params=encoder_params,
        wavenet_params=wavenet_params,
    )
    logger.info(model)
    #
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Number of parameters: {params}")
    #
    out_final = model(waveform)
    if model.intermediate:
        logger.info(
            f"input: {waveform.shape},out_final: {out_final['output'].shape}, intermediate: {len(out_final['intermediate'])}"
        )
    else:
        logger.info(f"input: {waveform.shape},out_final: {out_final['output'].shape}")
    logger.info(f"receptive_field: {model.wavenet.receptive_field()}")
