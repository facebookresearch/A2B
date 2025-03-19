# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


import numpy as np
import torch as th
import torchaudio as ta


class FourierTransform(th.nn.Module):
    """
    wrapper around torchaudio fft interface
    fft_bins: number of bins for fft
    win_length_ms: length of the Hanning window in ms
    frame_rate_hz: sampling frequency for fft
    causal: center FFT (i.e. padding on both sides) if False
    preemphasis: emphasize high frequencies if > 0 (0.97 frequently used in speech recognition/speech synthesis)
    sample_rate: assume that this is the sample rate for all audio input
    """

    def __init__(
        self,
        fft_bins=2048,
        win_length=None,
        hop_length=None,
        causal=False,
        preemphasis=0.0,
        sample_rate=48000,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.fft_bins = fft_bins
        self.win_length = fft_bins if win_length is None else win_length
        self.hop_length = fft_bins // 4 if hop_length is None else hop_length
        self.causal = causal
        if self.win_length > self.fft_bins:
            print("FourierTransform Warning: fft_bins should be larger than win_length")
        self.register_buffer("hann", th.hann_window(self.win_length))

    def _convert_format(self, data, expected_dims):
        if not type(data) == th.Tensor:
            data = th.Tensor(data)
        if len(data.shape) < expected_dims:
            data = data.unsqueeze(0)
        if not len(data.shape) == expected_dims:
            raise Exception(
                f"FourierTransform: data needs to be a Tensor with {expected_dims} dimensions but got shape {data.shape}"
            )
        return data

    def _preemphasis(self, audio):
        if self.preemphasis > 0:
            return th.cat(
                (audio[:, 0:1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1
            )
        return audio

    def _revert_preemphasis(self, audio):
        if self.preemphasis > 0:
            for i in range(1, audio.shape[1]):
                audio[:, i] = audio[:, i] + self.preemphasis * audio[:, i - 1]
        return audio

    def _magphase(self, complex_stft):
        mag, phase = ta.functional.magphase(complex_stft, 1.0)
        return mag, phase

    """
    wrapper around th.stft
    audio: wave signal as th.Tensor
    """

    def stft(self, audio, onesided=True):
        # pack batch since input to th.sftf must be either a 1-D time sequence or a 2-D batch of time sequences.
        shape = audio.size()
        audio = audio.reshape(-1, shape[-1])
        spec = th.stft(
            audio,
            n_fft=self.fft_bins,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann,
            center=False,
            return_complex=True,
            onesided=onesided,
        )
        spec = th.view_as_real(spec)
        # unpack batch
        spec = spec.reshape(
            shape[:-1] + spec.shape[-3:]
        )  # [batch, channel, nfreq, ntime, 2]
        return spec.contiguous()

    def forward(self, x):
        return self.stft(x, onesided=True)


if __name__ == "__main__":
    # stft check
    rfs = 48000
    fftnet = FourierTransform(
        fft_bins=2048, causal=False, preemphasis=0.0, sample_rate=rfs,
    )

    x = th.randn(10, 1, int(1.0 * rfs))
    out = fftnet(x)
    print(f"in: {x.shape}, out: {out.shape}")
