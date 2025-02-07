# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


from typing import Dict, List, Tuple, TypeVar, Union

import torch as th
from pytorch_lightning.utilities import rank_zero_info

import nnModules.layers.losses as audioLosses
from nnModules.metrics.base_metrics import BaseMetrics


class AmbientSoundsMetrics(BaseMetrics):
    def __init__(self, diff: bool = False, intermediate: bool = False, **kwargs):
        """custom loss metrics"""
        super().__init__()
        self.hparams = {"diff": diff, "intermediate": intermediate}
        self.loss_funcs = None
        self.inter_loss_funcs = None
        self.diff_loss_funcs = None
        self.all_weights = None
        self.create_metrics()

    def create_metrics(self):
        sample_rate = 48000
        fft_params_2048 = {
            "fft_bins": 2048,
            "hop_length": 32,
            "sample_rate": sample_rate,
        }
        fft_params_1024 = {
            "fft_bins": 1024,
            "hop_length": 32,
            "sample_rate": sample_rate,
        }
        fft_params_512 = {"fft_bins": 512, "hop_length": 64, "sample_rate": sample_rate}
        fft_params_256 = {"fft_bins": 256, "hop_length": 32, "sample_rate": sample_rate}
        fft_params_128 = {"fft_bins": 128, "hop_length": 64, "sample_rate": sample_rate}
        fft_params_64 = {"fft_bins": 64, "hop_length": 32, "sample_rate": sample_rate}
        fft_params_32 = {"fft_bins": 32, "hop_length": 16, "sample_rate": sample_rate}
        params_ml = {"mask_left": 1024, "mask_right": 0}
        params_mr = {"mask_left": 0, "mask_right": 1024}
        params_nomask = {"mask_left": 0, "mask_right": 0}

        self.loss_funcs = th.nn.ModuleDict(
            {
                "l2": audioLosses.create("l2"),
                "LogSpectralDistance_1024": audioLosses.create("LogSpectralDistance", fft_params_1024),
                "LogSTFTMagnitudeDiffLoss_64": audioLosses.create(
                    "LogSTFTMagnitudeDiffLoss", fft_params_64
                ),
                "LogSTFTMagnitudeDiffLoss_512": audioLosses.create(
                    "LogSTFTMagnitudeDiffLoss", fft_params_512
                ),
                "LogSTFTMagnitudeDiffLoss_1024": audioLosses.create(
                    "LogSTFTMagnitudeDiffLoss", fft_params_1024
                ),
                "SpectralConvergengeLoss2048": audioLosses.create(
                    "SpectralConvergengeLoss", fft_params_2048
                ),
                "SpectralConvergengeLoss512": audioLosses.create(
                    "SpectralConvergengeLoss", fft_params_512
                ),
                "SpectralConvergengeLoss64": audioLosses.create(
                    "SpectralConvergengeLoss", fft_params_64
                ),
                "SpectralConvergengeLoss1024": audioLosses.create(
                    "SpectralConvergengeLoss", fft_params_1024
                ),
                "SpectralConvergengeLoss2048": audioLosses.create(
                    "SpectralConvergengeLoss", fft_params_2048
                ),
                # "ild_2048_l": audioLosses.create(
                #     "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_2048, "w_log_mag":0.1, "w_lin_mag":10.0}
                # ),
                # "ild_512_l": audioLosses.create(
                #     "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_512, "w_log_mag":0.1, "w_lin_mag":10.0}
                # ),
                # "ild_128_l": audioLosses.create(
                #     "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_128 , "w_log_mag":0.1, "w_lin_mag":10.0}
                # ),
                # "ild_32_l": audioLosses.create(
                #     "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_32 , "w_log_mag":0.1, "w_lin_mag":10.0}
                # ),
            }
        )
        self.loss_weights = {
            "l2": 20.0,
            "LogSpectralDistance_1024": 1.0,
            "SpectralConvergengeLoss64": 10.0,
            "SpectralConvergengeLoss512": 10.0,
            "SpectralConvergengeLoss1024": 10.0,
            "SpectralConvergengeLoss2048": 10.0,
            "LogSTFTMagnitudeDiffLoss_64": 1.0,
            "LogSTFTMagnitudeDiffLoss_512": 1.0,
            "LogSTFTMagnitudeDiffLoss_1024": 1.0,
            "LogSTFTMagnitudeDiffLoss_2048": 1.0,
            # "ild_2048_l": 1.0,
            # "ild_512_l": 1.0,
            # "ild_128_l": 1.0,
            # "ild_32_l": 1.0,
        }

        if self.hparams["intermediate"]:
            rank_zero_info(f"Intermediate loss is created")
            self.inter_loss_funcs = th.nn.ModuleDict(
                {
                    "l2": audioLosses.create("l2"),
                    "LogSpectralDistance_1024": audioLosses.create("LogSpectralDistance", fft_params_1024),
                    "LogSTFTMagnitudeDiffLoss_64": audioLosses.create(
                        "LogSTFTMagnitudeDiffLoss", fft_params_64
                    ),
                    "LogSTFTMagnitudeDiffLoss_512": audioLosses.create(
                        "LogSTFTMagnitudeDiffLoss", fft_params_512
                    ),
                    "LogSTFTMagnitudeDiffLoss_1024": audioLosses.create(
                        "LogSTFTMagnitudeDiffLoss", fft_params_1024
                    ),
                    "SpectralConvergengeLoss512": audioLosses.create(
                        "SpectralConvergengeLoss", fft_params_512
                    ),
                    "SpectralConvergengeLoss64": audioLosses.create(
                        "SpectralConvergengeLoss", fft_params_64
                    ),
                    "SpectralConvergengeLoss1024": audioLosses.create(
                        "SpectralConvergengeLoss", fft_params_1024
                    ),
                    "ild_2048_l": audioLosses.create(
                        "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_2048, "w_log_mag":0.1, "w_lin_mag":10.0}
                    ),
                    "ild_512_l": audioLosses.create(
                        "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_512, "w_log_mag":0.1, "w_lin_mag":10.0}
                    ),
                    "ild_128_l": audioLosses.create(
                        "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_128 , "w_log_mag":0.1, "w_lin_mag":10.0}
                    ),
                    "ild_32_l": audioLosses.create(
                        "Binaural_Difference_ILDLoss", {**params_nomask, **fft_params_32 , "w_log_mag":0.1, "w_lin_mag":10.0}
                    ),
                }
            )

            self.inter_loss_weights = {
                "l2": 20.0,
                "LogSpectralDistance_1024": 1.0,
                "SpectralConvergengeLoss64": 10.0,
                "SpectralConvergengeLoss512": 10.0,
                "SpectralConvergengeLoss1024": 20.0,
                "LogSTFTMagnitudeDiffLoss_64": 1.0,
                "LogSTFTMagnitudeDiffLoss_512": 1.0,
                "LogSTFTMagnitudeDiffLoss_1024": 1.0,
                "ild_2048_l": 1.0,
                "ild_512_l": 1.0,
                "ild_128_l": 1.0,
                "ild_32_l": 1.0,
            }

        # for later use
        self.all_weights = {k: v for k, v in self.loss_weights.items()}

        if self.hparams["intermediate"]:
            self.all_weights.update(
                {f"inter_loss_{k}": v for k, v in self.inter_loss_weights.items()}
            )

    def forward(
        self,
        modeloutputs: Union[Dict, th.Tensor],
        targets: Union[Dict, th.Tensor],
        intermediate_targets: Union[Dict, th.Tensor] = None,
    ) -> Tuple[th.Tensor, Dict]:
        metrics = {}
        if isinstance(modeloutputs, dict):
            for lname, lfun in self.loss_funcs.items():
                metrics[lname] = lfun(modeloutputs["output"], targets)
        elif isinstance(modeloutputs, th.Tensor):
            for lname, lfun in self.loss_funcs.items():
                metrics[lname] = lfun(modeloutputs, targets)
        else:
            raise Exception("modeloutput should be a dict or Tensor!")

        if (self.hparams["intermediate"] and "intermediate" in modeloutputs.keys()):
            for lname, lfun in self.inter_loss_funcs.items():
                this_loss = 0.0
                for this_inter_out in modeloutputs["intermediate"]:
                    this_loss += lfun(this_inter_out, targets)
                metrics[f"inter_loss_{lname}"] = this_loss/len(modeloutputs["intermediate"])
        # mean aggregate of per-batch loss then apply weight
        loss_dict = {
            k: th.mean(v) for k, v in metrics.items()
        }  # for tensorboard logging
        loss = [v * self.all_weights[k] for k, v in loss_dict.items()]
        # final loss is the sum over all sub-losses (e.g., l2 + phase + ...)
        loss = th.sum(th.stack(loss))
        return loss, loss_dict


if __name__ == "__main__":
    print("test")
    cr = AmbientSoundsMetrics()
    x = th.randn(10, 1, 48000)
    y = th.randn(10, 1, 48000)
    out = cr(x, x)
    print(f"out: {out[1]}")
