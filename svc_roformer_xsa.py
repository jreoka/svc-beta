#!/usr/bin/env python3
"""
Standalone Native Stereo Polyphonic Singing Voice Conversion (SVC) in Pure PyTorch (44.1 kHz)
=============================================================================================
Zero pretrained weights or external voice encoders (no HuBERT, ContentVec, RMVPE, or HiFi-GAN).
Everything is trained from scratch using only standard Python + PyTorch audio/math primitives,
adapted to train effectively on as little as a single WAV/FLAC file or a folder of files.

Key Architectural Features:
  1. Native 44.1 kHz Stereo (2-Channel Left/Right + Mid/Side) Processing:
     - Loads and saves native 2-channel stereo WAV/FLAC files (`[2, L]`), automatically
       upmixing mono files if encountered.
     - Extracts stereo Log-Mel features, 2-channel Wiener-Khinchin autocorrelation + 6-harmonic
       spectral comb salience, interaural stereo pan salience, and stereo RMS energy.
  2. Vectorized Learned Sparse Polyphonic RoFormer Pitch Tracker (`StereoRoFormerPitchTracker`):
     - Predicts a sharp polyphonic pitch probability distribution (`f0_logits`), learned
       voice-presence gate (`gate_logits`), sub-bin frequency offset, and stereo pan field
       across `n_f0_bins=240` log-spaced pitch bins (65-1100 Hz).
     - Uses 100% vectorized harmonic-sieve peak extraction, canonical pitch-offset ordering, and
       single-kernel Nadaraya-Watson normalized convolution sample-and-hold (zero Python frame
       loops and zero blocking `.item()` GPU syncs) so voice tracks never swap identities or
       chirp toward random bins during fade-in/out.
  3. Low-VRAM Checkpointed Sample-Rate (44.1 kHz) Time-Domain Harmonic + Formant Synthesizer:
     - Integrates instantaneous phase (`cumsum`) strictly on the fundamental voice trajectories
       `[B, V, L]` (64x smaller than per-harmonic cumsum) and uses gradient checkpointing on the
       sample-rate sinusoidal oscillator bank so `[B, 2, K, L]` buffers are never retained in
       VRAM during backpropagation.
     - Shapes vocal timbre with a smooth 48-band vocal-tract formant filter and bounded
       small-angle polar phase dispersion (`|theta| <= pi/8`), preventing narrowband comb-filter
       vocoder artifacts and 180-degree phase flips.
  4. Prodigy Optimizer (Mishchenko & Defazio, 2023 - https://arxiv.org/abs/2306.06101):
     Built-in parameter-free D-Adaptation AdamW optimizer (uses `prodigyopt` if installed,
     otherwise falls back to the built-in self-contained implementation).
  5. QKVO Rotary Position Embedding (RoPE on Q, K, and V) & Exclusive Self Attention (XSA):
     Applies rotary positional embeddings to Query, Key, and Value projections with inverse
     output rotation and orthogonal self-value projection (arXiv:2603.09078) in fp32.

Usage:
  Train on a folder of WAV/FLAC files (or a single file):
    python svc_roformer_xsa.py --train --data_dir ./singer_data --ckpt_dir ./checkpoints

  Convert a source WAV/FLAC (solo, double-tracked, or multi-voice harmony) using latest checkpoint:
    python svc_roformer_xsa.py --infer --input ./source_vocal.wav --output ./converted.wav --ckpt_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class SVCConfig:
    sample_rate: int = 44100
    audio_channels: int = 2
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mels: int = 128
    n_formant_bands: int = 96
    fmin: float = 40.0
    fmax: float = 20000.0
    f0_min: float = 65.0
    f0_max: float = 1100.0
    n_f0_bins: int = 240
    n_harmonics: int = 128

    dim: int = 384
    content_dim: int = 32
    fsq_levels: int = 15
    pitch_encoder_depth: int = 4
    encoder_depth: int = 4
    decoder_depth: int = 8
    num_heads: int = 8
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.05
    rope_theta: float = 10000.0

    segment_frames: int = 256
    virtual_epoch_size: int = 256


def _ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Ensures a waveform tensor has shape `[2, L]` (native stereo)."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
    return waveform.contiguous()


def _read_wav_stdlib_stereo(path: str) -> Tuple[torch.Tensor, int]:
    """Fallback pure-Python WAV reader supporting 8/16/24/32-bit PCM, returning `[2, L]`."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        mask = signed & 0x800000
        signed = signed - (mask << 1)
        data = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    data = data.reshape(-1, channels).T
    return _ensure_stereo(torch.from_numpy(data)), sr


def _write_wav_stdlib_stereo(path: str, audio: torch.Tensor, sr: int) -> None:
    """Fallback pure-Python 16-bit PCM stereo WAV writer expecting `[2, L]`."""
    wav_np = _ensure_stereo(audio.detach().cpu()).clamp(-1.0, 1.0).numpy()
    interleaved = wav_np.T.reshape(-1)
    pcm16 = (interleaved * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def resample_audio_stereo(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resamples a stereo waveform `[2, L]` from `orig_sr` to `target_sr`."""
    waveform = _ensure_stereo(waveform)
    if orig_sr == target_sr:
        return waveform
    if torchaudio is not None and hasattr(torchaudio.transforms, "Resample"):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, target_freq=target_sr)
        return resampler(waveform)
    orig_len = waveform.shape[-1]
    target_len = int(round(orig_len * float(target_sr) / float(orig_sr)))
    out = F.interpolate(waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False)
    return out.squeeze(0)


def load_audio_stereo(path: str, target_sr: int = 44100) -> torch.Tensor:
    """Loads a WAV or FLAC file as a native stereo float32 tensor `[2, L]` at `target_sr` (44.1 kHz)."""
    ext = Path(path).suffix.lower()
    waveform: Optional[torch.Tensor] = None
    sr: int = target_sr

    if sf is not None:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            waveform = _ensure_stereo(torch.from_numpy(data.T))
        except Exception:
            waveform = None

    if waveform is None and torchaudio is not None:
        try:
            data, sr = torchaudio.load(path)
            waveform = _ensure_stereo(data.float())
        except Exception:
            waveform = None

    if waveform is None:
        if ext == ".wav":
            waveform, sr = _read_wav_stdlib_stereo(path)
        else:
            raise RuntimeError(
                f"Could not load '{path}'. For FLAC files without torchaudio/soundfile, "
                "please ensure either `soundfile` or `torchaudio` is available."
            )

    if sr != target_sr:
        waveform = resample_audio_stereo(waveform, sr, target_sr)

    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / max(1.0, float(peak))
    return waveform.float()


def save_audio_stereo(path: str, waveform: torch.Tensor, sr: int = 44100) -> None:
    """Saves a stereo float32 waveform tensor `[2, L]` to WAV or FLAC at `sr` (44.1 kHz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wav = _ensure_stereo(waveform.detach().cpu().float())
    peak = wav.abs().max()
    if peak > 0.99:
        wav = wav * (0.98 / float(peak))

    if sf is not None:
        try:
            sf.write(path, wav.T.numpy(), sr)
            return
        except Exception:
            pass

    if torchaudio is not None:
        try:
            torchaudio.save(path, wav, sr)
            return
        except Exception:
            pass

    _write_wav_stdlib_stereo(path, wav, sr)


class Prodigy(torch.optim.Optimizer):
    """
    Self-contained PyTorch implementation of the Prodigy optimizer:
    "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (arXiv:2306.06101).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta3: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        slice_p: int = 1,
    ):
        if not 0.0 < d0:
            raise ValueError(f"Invalid d0 value: {d0}")
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            d0=d0,
            d_max=d0,
            d_numerator=0.0,
            d_denom=0.0,
            d_coef=d_coef,
            k=0,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            safeguard_warmup=safeguard_warmup,
            slice_p=slice_p,
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        d_denom = 0.0
        group0 = self.param_groups[0]
        use_bias_correction = group0["use_bias_correction"]
        beta1, beta2 = group0["betas"]
        beta3 = group0["beta3"]
        if beta3 is None:
            beta3 = math.sqrt(beta2)

        k = group0["k"]
        d = group0["d"]
        d_max = group0["d_max"]
        d_coef = group0["d_coef"]
        lr = max(g["lr"] for g in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1.0 - beta2 ** (k + 1)) ** 0.5) / (1.0 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0

        dlr = d * lr * bias_correction
        growth_rate = group0["growth_rate"]
        decouple = group0["decouple"]

        d_numerator = group0["d_numerator"] * beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group["weight_decay"]
            group_lr = group["lr"]
            d0 = group["d0"]
            safeguard_warmup = group["safeguard_warmup"]
            slice_p = group["slice_p"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if decay != 0.0 and not decouple:
                    grad = grad.add(p.data, alpha=decay)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    sliced_init = p.data.flatten()[::slice_p]
                    state["s"] = torch.zeros_like(sliced_init, dtype=torch.float32).detach()
                    state["p0"] = sliced_init.detach().clone()
                    if beta1 > 0:
                        state["exp_avg"] = torch.zeros_like(p.data).detach()
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]
                p0 = state["p0"]

                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p].float()
                    sliced_p = p.data.flatten()[::slice_p].float()
                    delta_numerator += (
                        (d / d0)
                        * dlr
                        * torch.dot(sliced_grad, p0.float() - sliced_p).item()
                    )

                    if beta1 > 0:
                        exp_avg = state["exp_avg"]
                        exp_avg.mul_(beta1).add_(grad, alpha=d * (1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=(d / d0) * d)
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=(d / d0) * dlr)
                    d_denom += s.abs().sum().item()

        if d_denom == 0.0:
            return loss

        global_d_numerator = d_numerator + delta_numerator
        global_d_denom = d_denom

        d_hat = d_coef * global_d_numerator / global_d_denom
        if d == group0["d0"]:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = global_d_denom
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat

            decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)
                if decay != 0.0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(p.grad.data, denom, value=-dlr * d)

            group["k"] = group["k"] + 1

        return loss


def split_decay_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Separates 2D+ weights (decayed) from 1D norms, biases, and phase/gain scalars (no decay)."""
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name or "emb" in name or "alpha" in name or "phase" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def build_prodigy_optimizer(params, lr: float = 1.0, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Uses external `prodigyopt.Prodigy` if installed, else built-in `Prodigy`."""
    if isinstance(params, nn.Module):
        params = split_decay_param_groups(params, weight_decay)
    try:
        from prodigyopt import Prodigy as ExternalProdigy
        return ExternalProdigy(
            params,
            lr=lr,
            weight_decay=weight_decay,
            decouple=True,
            safeguard_warmup=True,
            use_bias_correction=True,
        )
    except ImportError:
        return Prodigy(
            params,
            lr=lr,
            weight_decay=weight_decay,
            decouple=True,
            safeguard_warmup=True,
            use_bias_correction=True,
        )


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float, device: torch.device
) -> torch.Tensor:
    """Constructs a triangular Mel filterbank matrix `[n_mels, n_fft // 2 + 1]`."""
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0.0, sr / 2.0, n_freqs, device=device)
    m_min = hz_to_mel(torch.tensor(fmin, device=device))
    m_max = hz_to_mel(torch.tensor(fmax, device=device))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    f_pts = mel_to_hz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1].clamp(min=1e-6)
    up_slopes = slopes[:, 2:] / f_diff[1:].clamp(min=1e-6)
    fb = torch.maximum(torch.zeros_like(down_slopes), torch.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels]).clamp(min=1e-6)
    fb = fb * enorm.unsqueeze(0)
    return fb.transpose(0, 1).contiguous()


def create_mel_to_fft_basis(
    sr: int, n_fft: int, n_bands: int, device: torch.device
) -> torch.Tensor:
    """
    Constructs a smooth Mel-spaced piecewise-linear interpolation basis `[n_bands, n_fft // 2 + 1]`
    spanning `[0, sr / 2]` so vocal formants (250-3500 Hz) receive high frequency resolution.
    Every FFT bin sums to 1.0 across bands.
    """
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, sr / 2.0, n_freqs, device=device)
    fft_mels = hz_to_mel(fft_freqs)
    m_max = hz_to_mel(torch.tensor(sr / 2.0, device=device))
    band_mels = torch.linspace(0.0, float(m_max.item()), n_bands, device=device)

    basis = torch.zeros(n_bands, n_freqs, device=device, dtype=torch.float32)
    step = band_mels[1] - band_mels[0]
    for b in range(n_bands):
        dist = (fft_mels - band_mels[b]).abs() / step.clamp(min=1e-6)
        basis[b] = (1.0 - dist).clamp(min=0.0)
    basis = basis / basis.sum(dim=0, keepdim=True).clamp(min=1e-6)
    return basis.contiguous()


def create_harmonic_sieve_matrix(n_f0_bins: int, bin_step_log2: float) -> torch.Tensor:
    """
    Constructs a `[n_f0_bins, n_f0_bins]` non-maximum suppression matrix where row `peak_bin`
    suppresses local neighbors, upper harmonics (`2x, 3x, 4x`), and ghost subharmonics (`1/2, 1/3, 1/4`).
    """
    bin_diff = torch.arange(n_f0_bins).unsqueeze(0) - torch.arange(n_f0_bins).unsqueeze(1)
    local_supp = torch.exp(-0.5 * (bin_diff.float() / 5.0) ** 2)
    octave_bins = int(round(1.0 / bin_step_log2))
    fifth_bins = int(round(math.log2(3.0) / bin_step_log2))
    two_oct_bins = int(round(2.0 / bin_step_log2))
    harm2_supp = 0.78 * torch.exp(-0.5 * ((bin_diff.float() - octave_bins) / 4.0) ** 2)
    harm3_supp = 0.65 * torch.exp(-0.5 * ((bin_diff.float() - fifth_bins) / 4.0) ** 2)
    harm4_supp = 0.55 * torch.exp(-0.5 * ((bin_diff.float() - two_oct_bins) / 4.0) ** 2)
    sub2_supp = 0.85 * torch.exp(-0.5 * ((bin_diff.float() + octave_bins) / 4.0) ** 2)
    sub3_supp = 0.75 * torch.exp(-0.5 * ((bin_diff.float() + fifth_bins) / 4.0) ** 2)
    sub4_supp = 0.65 * torch.exp(-0.5 * ((bin_diff.float() + two_oct_bins) / 4.0) ** 2)
    return (
        local_supp + harm2_supp + harm3_supp + harm4_supp + sub2_supp + sub3_supp + sub4_supp
    ).clamp(0.0, 1.0)


class StereoAcousticFeatureExtractor(nn.Module):
    """
    Fast, low-VRAM stereo polyphonic acoustic front-end in pure PyTorch at 44.1 kHz:
      - Stereo Log-Mel Spectrogram `[B, 2, T, n_mels]` (Left & Right)
      - Mid/Side Log-Mel Spectrogram `[B, 2, T, n_mels]` (Mid & Side)
      - 2-Channel (L/R) Harmonic-Comb, Window-Normalized Autocorrelation, and Stereo Pan Salience Maps
      - Sparse 3-Pass Harmonic-Sieve Reference Pitch Distribution `ref_pitch_dist [B, T, n_f0_bins]`
      - Stereo Log Frame RMS Energy `[B, T, 3]` (Left, Right, and Side)
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        window = torch.hann_window(cfg.win_length)
        self.register_buffer("window", window, persistent=False)

        fft_len = 2 * cfg.win_length
        win_spec = torch.fft.rfft(window, n=fft_len)
        win_acf = torch.fft.irfft(win_spec.real ** 2 + win_spec.imag ** 2, n=fft_len)[: cfg.win_length]
        win_acf = (win_acf / win_acf[0].clamp(min=1e-8)).clamp(min=0.12)
        self.register_buffer("win_acf", win_acf, persistent=False)

        fb = create_mel_filterbank(
            cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax, device=torch.device("cpu")
        )
        self.register_buffer("mel_fb", fb, persistent=False)

        m_min = float(hz_to_mel(torch.tensor(cfg.fmin)).item())
        m_max = float(hz_to_mel(torch.tensor(cfg.fmax)).item())
        mel_pts = torch.linspace(m_min, m_max, cfg.n_mels + 2)[1:-1]
        self.register_buffer("mel_center_mels", mel_pts, persistent=False)
        self.register_buffer("mel_center_hz", mel_to_hz(mel_pts), persistent=False)
        self.mel_min = m_min
        self.mel_max = m_max

        log2_min = math.log2(cfg.f0_min)
        log2_max = math.log2(cfg.f0_max)
        log2_grid = torch.linspace(log2_min, log2_max, cfg.n_f0_bins)
        f0_grid = 2.0 ** log2_grid
        self.register_buffer("log2_f0_grid", log2_grid, persistent=False)
        self.register_buffer("f0_grid_hz", f0_grid, persistent=False)
        self.bin_step_log2 = (log2_max - log2_min) / max(1, cfg.n_f0_bins - 1)

        sieve_kernel = create_harmonic_sieve_matrix(cfg.n_f0_bins, self.bin_step_log2)
        self.register_buffer("sieve_kernel", sieve_kernel, persistent=False)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            if wav_f32.ndim == 3:
                B, C, L = wav_f32.shape
                flat = wav_f32.reshape(B * C, L)
                spec = self.stft(flat)
                return spec.view(B, C, spec.shape[-2], spec.shape[-1])

            pad_mode = "reflect" if wav_f32.shape[-1] > (self.cfg.n_fft // 2) else "constant"
            return torch.stft(
                wav_f32,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=self.window.to(wav.device, dtype=torch.float32),
                center=True,
                pad_mode=pad_mode,
                return_complex=True,
            )

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        with torch.amp.autocast(device_type=spec.device.type, enabled=False):
            spec_c64 = spec.to(torch.complex64)
            if spec_c64.ndim == 4:
                B, C, F_bins, T = spec_c64.shape
                flat = spec_c64.reshape(B * C, F_bins, T)
                wav = self.istft(flat, length=length)
                return wav.view(B, C, length)

            return torch.istft(
                spec_c64,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=self.window.to(spec.device, dtype=torch.float32),
                center=True,
                length=length,
            )

    @staticmethod
    def _interp_1d_bins(profile: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        max_idx = profile.shape[-1] - 1
        pos_clamped = positions.clamp(0.0, float(max_idx))
        idx0 = pos_clamped.long()
        idx1 = (idx0 + 1).clamp(max=max_idx)
        frac = (pos_clamped - idx0.float()).view(*([1] * (profile.ndim - 1)), -1)
        v0 = profile[..., idx0]
        v1 = profile[..., idx1]
        return v0 * (1.0 - frac) + v1 * frac

    @torch.no_grad()
    def shift_mel_by_f0_bins(self, log_mel: torch.Tensor, shift_f0_bins: int) -> torch.Tensor:
        """Shifts a log-Mel tensor `[B, C, T, M]` physically along frequency by `shift_f0_bins` log2-F0 bins."""
        if shift_f0_bins == 0:
            return log_mel
        with torch.amp.autocast(device_type=log_mel.device.type, enabled=False):
            mel_f32 = log_mel.float()
            B, C, T, M = mel_f32.shape
            device = mel_f32.device
            factor = 2.0 ** (float(shift_f0_bins) * self.bin_step_log2)
            src_hz = self.mel_center_hz.to(device) / factor
            src_mel = hz_to_mel(src_hz)
            grid_x = 2.0 * (src_mel - self.mel_min) / max(1e-6, self.mel_max - self.mel_min) - 1.0
            grid_x = grid_x.view(1, 1, M).expand(B, T, M)
            grid_y = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1).expand(B, T, M)
            sample_grid = torch.stack([grid_x, grid_y], dim=-1)
            flat_x = mel_f32.reshape(B * C, 1, T, M)
            flat_grid = sample_grid.unsqueeze(1).expand(B, C, T, M, 2).reshape(B * C, T, M, 2)
            return F.grid_sample(
                flat_x, flat_grid, mode="bilinear", padding_mode="border", align_corners=True
            ).view(B, C, T, M).to(log_mel.dtype)

    @torch.no_grad()
    def extract_pitch_salience_and_stereo_energy(
        self, wav: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            B, C, L = wav_f32.shape
            N_bins = self.cfg.n_f0_bins
            device = wav.device
            pad = self.cfg.n_fft // 2
            pad_mode = "reflect" if L > pad else "constant"
            wav_pad = F.pad(wav_f32, (pad, pad), mode=pad_mode)
            frames = wav_pad.unfold(dimension=-1, size=self.cfg.win_length, step=self.cfg.hop_length)
            W = self.cfg.win_length
            win = self.window.to(device, dtype=torch.float32).view(1, 1, 1, W)
            win_frames = (frames - frames.mean(dim=-1, keepdim=True)) * win

            rms_lr = torch.sqrt((frames ** 2).mean(dim=-1) + 1e-8)
            side_frames = 0.5 * (frames[:, 0] - frames[:, 1])
            rms_side = torch.sqrt((side_frames ** 2).mean(dim=-1, keepdim=True) + 1e-8).transpose(1, 2)
            stereo_energy = torch.log(torch.cat([rms_lr, rms_side], dim=1) + 1e-6).transpose(1, 2).contiguous()
            log_energy_mid = 0.5 * (stereo_energy[..., 0] + stereo_energy[..., 1])

            fft_len = 2 * W
            spec_2ch = torch.fft.rfft(win_frames, n=fft_len, dim=-1)
            power_2ch = spec_2ch.real ** 2 + spec_2ch.imag ** 2
            power_sum = power_2ch.sum(dim=1)
            power_max = power_2ch.max(dim=1).values
            power_joint = 0.60 * power_sum + 0.40 * power_max

            acf = torch.fft.irfft(power_joint, n=fft_len, dim=-1)[..., :W]
            r0 = acf[..., 0:1].clamp(min=1e-7)
            win_acf = self.win_acf.to(device, dtype=torch.float32).view(1, 1, W)
            norm_acf = (acf / (r0 * win_acf)).clamp(-1.0, 1.0)

            f0_grid = self.f0_grid_hz.to(device, dtype=torch.float32)
            log2_grid = self.log2_f0_grid.to(device, dtype=torch.float32)

            lags = float(self.cfg.sample_rate) / f0_grid
            acf_at_lag = self._interp_1d_bins(norm_acf, lags)
            acf_half_lag = self._interp_1d_bins(norm_acf, lags * 0.5)
            acf_salience = (acf_at_lag - 0.35 * F.relu(acf_half_lag)).clamp(-1.0, 1.0)

            log_pow = torch.log(power_joint + 1e-7)
            T = log_pow.shape[1]
            smooth_pow = F.avg_pool1d(
                log_pow.reshape(B * T, 1, -1),
                kernel_size=17,
                stride=1,
                padding=8,
            ).view_as(log_pow)
            whitened_pow = (log_pow - smooth_pow).clamp(-4.0, 6.0)

            # Standard stereo convention: -1.0 = Left, +1.0 = Right
            pan_spec = (power_2ch[:, 1] - power_2ch[:, 0]) / (power_sum + 1e-7)

            bin_hz = float(self.cfg.sample_rate) / float(fft_len)
            comb_salience = torch.zeros_like(acf_salience)
            pan_salience = torch.zeros_like(acf_salience)
            pan_weight_sum = torch.full_like(acf_salience, 1e-4)
            h1_salience = torch.zeros_like(acf_salience)
            n_comb_harmonics = 6
            for h_idx in range(1, n_comb_harmonics + 1):
                w_h = 1.0 / (float(h_idx) ** 0.75)
                peak_bins = (float(h_idx) * f0_grid) / bin_hz
                trough_bins = ((float(h_idx) - 0.5) * f0_grid) / bin_hz
                peak_val = self._interp_1d_bins(whitened_pow, peak_bins)
                if h_idx == 1:
                    h1_salience = peak_val
                trough_val = self._interp_1d_bins(whitened_pow, trough_bins)
                comb_salience = comb_salience + w_h * (peak_val - 0.5 * trough_val)

                p_w = (1.0 / float(h_idx)) * F.relu(peak_val).clamp(min=0.05)
                pan_salience = pan_salience + p_w * self._interp_1d_bins(pan_spec, peak_bins)
                pan_weight_sum = pan_weight_sum + p_w

            comb_salience = torch.tanh(comb_salience / 4.0)
            pan_salience = (pan_salience / pan_weight_sum).clamp(-1.0, 1.0)

            joint_salience = 0.60 * acf_salience + 0.40 * comb_salience

            if T > 1:
                sal_t = joint_salience.permute(0, 2, 1).reshape(B * N_bins, 1, T)
                k_time = torch.tensor([0.12, 0.23, 0.30, 0.23, 0.12], device=device, dtype=torch.float32).view(1, 1, 5)
                sal_t = F.pad(sal_t, (2, 2), mode="replicate")
                smoothed_sal = F.conv1d(sal_t, k_time).view(B, N_bins, T).permute(0, 2, 1).contiguous()
            else:
                smoothed_sal = joint_salience.clone()

            raw_acf_peak = acf_at_lag.max(dim=-1).values.clamp(0.0, 1.0)
            energy_floor = log_energy_mid.amax(dim=-1, keepdim=True) - 4.4
            energy_gate = torch.sigmoid((log_energy_mid - energy_floor) * 2.5)
            ref_voicing = ((raw_acf_peak > 0.30) & (log_energy_mid > energy_floor)).float()
            ref_conf = (raw_acf_peak * energy_gate).clamp(0.0, 1.0)

            sieve_mat = self.sieve_kernel.to(device, dtype=torch.float32)
            residual_sal = F.relu(smoothed_sal + 0.20)
            primary_max = residual_sal.amax(dim=-1, keepdim=True).clamp(min=1e-5)
            bin_indices = torch.arange(N_bins, device=device).view(1, 1, N_bins)
            sigma_log2 = 1.25 * self.bin_step_log2

            ref_multi_activity = torch.zeros(B, T, N_bins, device=device, dtype=torch.float32)
            ref_pitch_unnorm = torch.zeros(B, T, N_bins, device=device, dtype=torch.float32)

            for p_iter in range(3):
                peak_val, peak_bin = residual_sal.max(dim=-1)
                local_mask = (bin_indices - peak_bin.unsqueeze(-1)).abs() <= 5
                masked_logits = torch.where(
                    local_mask,
                    smoothed_sal * 12.0,
                    torch.full_like(smoothed_sal, -1e9),
                )
                local_probs = F.softmax(masked_logits, dim=-1)
                peak_log2 = (local_probs * log2_grid.view(1, 1, N_bins)).sum(dim=-1, keepdim=True)

                if p_iter == 0:
                    active_gate = ref_voicing.unsqueeze(-1)
                else:
                    rel_val = (peak_val.unsqueeze(-1) / primary_max).clamp(0.0, 1.0)
                    acf_at_peak = torch.gather(acf_salience, dim=-1, index=peak_bin.unsqueeze(-1))
                    comb_at_peak = torch.gather(comb_salience, dim=-1, index=peak_bin.unsqueeze(-1))
                    h1_at_peak = torch.gather(h1_salience, dim=-1, index=peak_bin.unsqueeze(-1))
                    is_true_secondary = (
                        (rel_val > 0.56)
                        & (peak_val.unsqueeze(-1) > 0.42)
                        & (acf_at_peak > 0.30)
                        & (comb_at_peak > 0.22)
                        & (h1_at_peak > 0.25)
                        & (ref_voicing.unsqueeze(-1) > 0.5)
                    ).float()
                    if T >= 3:
                        sec_1d = is_true_secondary.squeeze(-1).unsqueeze(1)
                        sec_min = -F.max_pool1d(-F.pad(sec_1d, (1, 1), mode="replicate"), kernel_size=3, stride=1)
                        sec_clean = F.max_pool1d(F.pad(sec_min, (1, 1), mode="replicate"), kernel_size=3, stride=1)
                        is_true_secondary = sec_clean.squeeze(1).unsqueeze(-1)
                    active_gate = is_true_secondary * rel_val

                gauss_peak = torch.exp(-0.5 * ((log2_grid.view(1, 1, N_bins) - peak_log2) / sigma_log2) ** 2)
                ref_multi_activity = torch.maximum(ref_multi_activity, gauss_peak * active_gate)
                ref_pitch_unnorm = ref_pitch_unnorm + gauss_peak * (active_gate + 1e-4 * (p_iter == 0))

                supp_profile = sieve_mat[peak_bin]
                residual_sal = residual_sal * (1.0 - 0.92 * supp_profile)

            ref_pitch_dist = ref_pitch_unnorm / ref_pitch_unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            return {
                "acf_salience": acf_salience,
                "comb_salience": comb_salience,
                "pan_salience": pan_salience,
                "stereo_energy": stereo_energy,
                "ref_pitch_dist": ref_pitch_dist,
                "ref_multi_activity": ref_multi_activity,
                "ref_voicing": ref_voicing,
                "ref_conf": ref_conf,
            }

    def compute_stereo_log_mel(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            spec = self.stft(wav.float())
            fb = self.mel_fb.to(wav.device, dtype=torch.float32)

            mag_lr = spec.abs()
            mel_lr = torch.matmul(fb, mag_lr)
            lr_log_mel = torch.log(mel_lr.clamp(min=1e-5)).transpose(-1, -2).contiguous()

            spec_mid = 0.5 * (spec[:, 0] + spec[:, 1])
            spec_side = 0.5 * (spec[:, 0] - spec[:, 1])
            mag_ms = torch.stack([spec_mid.abs(), spec_side.abs()], dim=1)
            mel_ms = torch.matmul(fb, mag_ms)
            ms_log_mel = torch.log(mel_ms.clamp(min=1e-5)).transpose(-1, -2).contiguous()

            return lr_log_mel, ms_log_mel, spec

    @torch.no_grad()
    def augment_mel_for_content(
        self,
        lr_log_mel: torch.Tensor,
        strength_range: Tuple[float, float] = (0.15, 1.0),
        clean_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Speaker- and pitch-destroying acoustic perturbation for disentangled phonetic content:
          - Random stereo L/R swap & Mid/Side width mixing
          - Piecewise-linear VTLN formant scaling + global Mel shift (disrupts F0 & vocal tract length)
          - Random frequency-axis Gaussian blur (smoothes away discrete F0 harmonic peaks)
          - Random spectral tilt + smooth 8-knot parametric EQ + light noise
        """
        with torch.amp.autocast(device_type=lr_log_mel.device.type, enabled=False):
            mel_f32 = lr_log_mel.float()
            B, C, T, M = mel_f32.shape
            device = mel_f32.device

            strength = torch.empty(B, 1, 1, device=device).uniform_(strength_range[0], strength_range[1])
            if clean_prob > 0.0:
                keep_clean = torch.rand(B, 1, 1, device=device) < clean_prob
                strength = torch.where(keep_clean, torch.zeros_like(strength), strength)

            # 1. Stereo pan / width randomization
            mid = 0.5 * (mel_f32[:, 0:1] + mel_f32[:, 1:2])
            side = 0.5 * (mel_f32[:, 0:1] - mel_f32[:, 1:2])
            flip = torch.where(
                torch.rand(B, 1, 1, 1, device=device) < (0.5 * (strength.view(B, 1, 1, 1) > 0.05).float()),
                -1.0,
                1.0,
            )
            side_scale = 1.0 + strength.view(B, 1, 1, 1) * torch.empty(B, 1, 1, 1, device=device).uniform_(-0.90, 0.35)
            mel_mix = torch.cat([mid + flip * side_scale * side, mid - flip * side_scale * side], dim=1)

            # 2. Piecewise VTLN formant warp + global Mel pitch shift
            vtln_alpha = 1.0 + strength * torch.empty(B, 1, 1, device=device).uniform_(-0.22, 0.25)
            pivot = torch.empty(B, 1, 1, device=device).uniform_(0.25, 0.65)
            base_01 = torch.linspace(0.0, 1.0, M, device=device).view(1, 1, M)
            high_slope = (1.0 - pivot * vtln_alpha) / (1.0 - pivot).clamp(min=1e-4)
            warped_01 = torch.where(
                base_01 <= pivot,
                base_01 * vtln_alpha,
                pivot * vtln_alpha + (base_01 - pivot) * high_slope,
            )
            mel_shift = strength * torch.empty(B, 1, 1, device=device).uniform_(-0.14, 0.14)
            warped_grid = (warped_01 * 2.0 - 1.0 + mel_shift).clamp(-1.0, 1.0)
            time_grid = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1).expand(B, T, M)
            sample_grid = torch.stack([warped_grid.expand(B, T, M), time_grid], dim=-1)

            flat_x = mel_mix.reshape(B * C, 1, T, M)
            flat_grid = sample_grid.unsqueeze(1).expand(B, C, T, M, 2).reshape(B * C, T, M, 2)
            warped = F.grid_sample(
                flat_x, flat_grid, mode="bilinear", padding_mode="border", align_corners=True
            ).view(B, C, T, M)

            # 3. Frequency-axis Gaussian blur (destroys F0 harmonic ripples while preserving formants)
            sigma = strength.view(B, 1) * torch.empty(B, 1, device=device).uniform_(0.0, 1.6)
            offsets = torch.arange(-3, 4, device=device, dtype=torch.float32).view(1, 7)
            k_gauss = torch.where(
                sigma > 1e-3,
                torch.exp(-0.5 * (offsets / sigma.clamp(min=1e-3)) ** 2),
                (offsets == 0).float(),
            )
            k_gauss = (k_gauss / k_gauss.sum(dim=-1, keepdim=True).clamp(min=1e-6)).view(B, 1, 1, 1, 7)
            w_pad = F.pad(warped.reshape(B * C, T, M), (3, 3), mode="replicate").view(B, C, T, M + 6)
            blurred = (w_pad.unfold(-1, 7, 1) * k_gauss).sum(dim=-1)

            # 4. Random spectral tilt + smooth 8-knot EQ + light noise on non-silent frames
            non_silent = (mel_f32.amax(dim=-1, keepdim=True) > -10.5).float()
            tilt_axis = torch.linspace(-1.0, 1.0, M, device=device).view(1, 1, 1, M)
            tilt = strength.view(B, 1, 1, 1) * torch.empty(B, 1, 1, 1, device=device).uniform_(-0.75, 0.75) * tilt_axis
            knots = torch.randn(B, 1, 8, device=device) * (0.45 * strength.view(B, 1, 1))
            eq_curve = F.interpolate(knots, size=M, mode="linear", align_corners=True).view(B, 1, 1, M)
            pan_jitter = torch.randn(B, C, 1, 1, device=device) * (0.15 * strength.view(B, 1, 1, 1))
            noise = torch.randn_like(blurred) * (0.06 * strength.view(B, 1, 1, 1))
            return blurred + (tilt + eq_curve + pan_jitter + noise) * non_silent


class QKVRopeEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) applied to Query, Key, AND Value tensors,
    paired with the inverse rotation on the aggregated attention output.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_len: int = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cache is None
            or self._sin_cache is None
            or self._cache_len < seq_len
            or self._cos_cache.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cache = torch.cos(freqs)
            self._sin_cache = torch.sin(freqs)
            self._cache_len = seq_len
        return (
            self._cos_cache[:seq_len].to(dtype=dtype).view(1, 1, seq_len, -1),
            self._sin_cache[:seq_len].to(dtype=dtype).view(1, 1, seq_len, -1),
        )

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        if inverse:
            rx1 = x1 * cos + x2 * sin
            rx2 = -x1 * sin + x2 * cos
        else:
            rx1 = x1 * cos - x2 * sin
            rx2 = x1 * sin + x2 * cos
        return torch.stack((rx1, rx2), dim=-1).flatten(-2)

    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos, sin = self._update_cache(seq_len, q.device, q.dtype)
        q_rot = self.rotate(q, cos, sin, inverse=False)
        k_rot = self.rotate(k, cos, sin, inverse=False)
        v_rot = self.rotate(v, cos, sin, inverse=False)
        return q_rot, k_rot, v_rot, cos, sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm_x * self.weight.float()).to(x.dtype)


class GEGLUFFN(nn.Module):
    def __init__(self, dim: int, mult: float = 8.0 / 3.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(round(dim * mult / 64.0)) * 64
        hidden_dim = max(hidden_dim, 128)
        self.w_gate_val = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w_gate_val(x).chunk(2, dim=-1)
        hidden = F.gelu(gate, approximate="tanh") * val
        return self.w_out(self.dropout(hidden))


class ExclusiveQKVRoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, rope_theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = QKVRopeEmbedding(self.head_dim, theta=rope_theta)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q_rot, k_rot, v_rot, cos, sin = self.rope.forward_qkv(q, k, v)

        y_rot = F.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v_rot,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        y = self.rope.rotate(y_rot, cos, sin, inverse=True)

        y_f32 = y.float()
        v_unit = F.normalize(v.float(), p=2.0, dim=-1, eps=1e-6)
        proj_self = (y_f32 * v_unit).sum(dim=-1, keepdim=True) * v_unit
        z = (y_f32 - proj_self).to(x.dtype)

        out = z.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class XSARoPETransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_mult: float = 8.0 / 3.0, rope_theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = ExclusiveQKVRoPEAttention(dim, num_heads=num_heads, rope_theta=rope_theta, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = GEGLUFFN(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StereoRoFormerPitchTracker(nn.Module):
    """
    100% Vectorized Learned Sparse Polyphonic RoFormer Pitch Tracker.
    Zero Python frame loops (`for t in range(T)`) and zero blocking `.item()` GPU syncs:
      - Predicts `f0_logits [B, T, 240]`, `gate_logits [B, T, 240]`, sub-bin `fine_offset`,
        and `bin_pan` using a lightweight depthwise-separable lateral inhibition head.
      - Extracts the primary vocal track plus up to 2 simultaneous secondary harmony/double
        tracks via 3 vectorized harmonic-sieve passes, orders secondary tracks canonically by
        signed pitch offset from the lead vocal so tracks never swap identities, and holds
        inactive frame pitches smoothly via a single Nadaraya-Watson normalized 1D convolution.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 4 * cfg.n_mels + 3 * cfg.n_f0_bins + 3
        self.in_norm = RMSNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, cfg.dim)

        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=5, padding=2, groups=cfg.dim),
            nn.GELU(),
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=1),
        )

        self.blocks = nn.ModuleList([
            XSARoPETransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                ffn_mult=cfg.ffn_mult,
                rope_theta=cfg.rope_theta,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.pitch_encoder_depth)
        ])
        self.out_norm = RMSNorm(cfg.dim)

        self.bin_feature_proj = nn.Linear(cfg.dim, cfg.n_f0_bins * 2)

        self.lateral_inhibition_net = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=17, padding=8, groups=16),
            nn.GELU(),
            nn.Conv1d(16, 4, kernel_size=1),
        )

        self.salience_skip_gain = nn.Parameter(torch.tensor(2.5))
        self.pan_skip_gain = nn.Parameter(torch.tensor(0.75))
        self.voicing_head = nn.Linear(cfg.dim, 1)

        log2_min = math.log2(cfg.f0_min)
        log2_max = math.log2(cfg.f0_max)
        log2_grid = torch.linspace(log2_min, log2_max, cfg.n_f0_bins)
        self.register_buffer("log2_f0_grid", log2_grid, persistent=False)
        self.bin_step_log2 = (log2_max - log2_min) / max(1, cfg.n_f0_bins - 1)

        nms_sieve = create_harmonic_sieve_matrix(cfg.n_f0_bins, self.bin_step_log2)
        self.register_buffer("nms_sieve", nms_sieve, persistent=False)

        hold_radius = 12
        hold_t = torch.arange(-hold_radius, hold_radius + 1, dtype=torch.float32)
        hold_kernel = torch.exp(-0.5 * (hold_t / 4.5) ** 2)
        hold_kernel = hold_kernel / hold_kernel.sum()
        self.register_buffer("hold_kernel", hold_kernel.view(1, 1, -1), persistent=False)

    def extract_dynamic_voice_tracks(
        self,
        f0_logits: torch.Tensor,
        multi_voice_gate: torch.Tensor,
        fine_offset: torch.Tensor,
        bin_pan: torch.Tensor,
        periodicity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N_bins = f0_logits.shape
        device = f0_logits.device
        log2_grid = self.log2_f0_grid.to(device, dtype=torch.float32).view(1, 1, N_bins)
        bin_idx = torch.arange(N_bins, device=device).view(1, 1, N_bins)
        full_probs = F.softmax(f0_logits, dim=-1)
        sieve_mat = self.nms_sieve.to(device, dtype=torch.float32)

        with torch.no_grad():
            score_field = full_probs * (0.4 + 0.6 * multi_voice_gate)
            if T >= 3:
                sf_t = score_field.permute(0, 2, 1).reshape(B * N_bins, 1, T)
                k3 = torch.tensor([0.25, 0.50, 0.25], device=device, dtype=torch.float32).view(1, 1, 3)
                score_field = F.conv1d(F.pad(sf_t, (1, 1), mode="replicate"), k3).view(B, N_bins, T).permute(0, 2, 1)

            res_score = score_field.clone()
            max_val_0, peak_bin_0 = res_score.max(dim=-1)
            res_score = res_score * (1.0 - 0.92 * sieve_mat[peak_bin_0])

            val_1, bin_1 = res_score.max(dim=-1)
            gate_1 = torch.gather(multi_voice_gate, dim=-1, index=bin_1.unsqueeze(-1)).squeeze(-1)
            rel_1 = val_1 / max_val_0.clamp(min=1e-6)
            act_1 = ((gate_1 > 0.55) & (rel_1 > 0.36) & (periodicity > 0.40)).float()
            res_score = res_score * (1.0 - 0.92 * sieve_mat[bin_1])

            val_2, bin_2 = res_score.max(dim=-1)
            gate_2 = torch.gather(multi_voice_gate, dim=-1, index=bin_2.unsqueeze(-1)).squeeze(-1)
            rel_2 = val_2 / max_val_0.clamp(min=1e-6)
            act_2 = ((gate_2 > 0.55) & (rel_2 > 0.36) & (periodicity > 0.40)).float()

            if T >= 3:
                sec_act = torch.stack([act_1, act_2], dim=1)
                sec_min = -F.max_pool1d(-F.pad(sec_act, (1, 1), mode="replicate"), kernel_size=3, stride=1)
                sec_clean = F.max_pool1d(F.pad(sec_min, (1, 1), mode="replicate"), kernel_size=3, stride=1)
                act_1, act_2 = sec_clean[:, 0], sec_clean[:, 1]

            eff_bin_1 = torch.where(act_1 > 0.5, bin_1, peak_bin_0)
            eff_bin_2 = torch.where(act_2 > 0.5, bin_2, peak_bin_0)
            swap_mask = eff_bin_1 < eff_bin_2
            hi_bin = torch.where(swap_mask, bin_2, bin_1)
            lo_bin = torch.where(swap_mask, bin_1, bin_2)
            hi_act = torch.where(swap_mask, act_2, act_1)
            lo_act = torch.where(swap_mask, act_1, act_2)

            lead_act = (periodicity > 0.35).float()
            cand_bins = torch.stack([peak_bin_0, hi_bin, lo_bin], dim=1)
            cand_active = torch.stack([lead_act, hi_act, lo_act], dim=1)

        V = 3
        p_bins = cand_bins.unsqueeze(-1)
        window_mask = ((bin_idx.unsqueeze(1) - p_bins).abs() <= 6).float()
        masked_probs = full_probs.unsqueeze(1) * window_mask
        sum_window = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        local_probs = masked_probs / sum_window

        fine_v = (local_probs * fine_offset.unsqueeze(1)).sum(dim=-1)
        raw_log2 = (local_probs * log2_grid.unsqueeze(1)).sum(dim=-1) + fine_v

        if T >= 3:
            hk = self.hold_kernel.to(device, dtype=torch.float32)
            pad_r = hk.shape[-1] // 2
            w_hold = cand_active.float()
            lead_ref = raw_log2[:, 0:1, :].detach().expand(B, V, T)
            num = w_hold * raw_log2 + 1e-3 * lead_ref
            den = w_hold + 1e-3
            num_smooth = F.conv1d(F.pad(num.reshape(B * V, 1, T), (pad_r, pad_r), mode="replicate"), hk).view(B, V, T)
            den_smooth = F.conv1d(F.pad(den.reshape(B * V, 1, T), (pad_r, pad_r), mode="replicate"), hk).view(B, V, T)
            held_log2 = num_smooth / den_smooth.clamp(min=1e-6)
            raw_log2 = torch.where(cand_active > 0.5, raw_log2, held_log2)

            pad_l2 = F.pad(raw_log2.reshape(B * V, 1, T), (1, 1), mode="replicate")
            smooth_k = torch.tensor([0.20, 0.60, 0.20], device=device, dtype=torch.float32).view(1, 1, 3)
            smooth_log2 = F.conv1d(pad_l2, smooth_k).view(B, V, T)
        else:
            smooth_log2 = raw_log2

        smooth_log2 = smooth_log2.clamp(math.log2(self.cfg.f0_min), math.log2(self.cfg.f0_max))
        gate_v = (local_probs.detach() * multi_voice_gate.unsqueeze(1)).sum(dim=-1)

        sec_amp = periodicity.unsqueeze(1) * gate_v[:, 1:] * (
            cand_active[:, 1:] + (gate_v[:, 1:] - gate_v[:, 1:].detach())
        )
        if T >= 3:
            pad_a = F.pad(sec_amp.reshape(B * (V - 1), 1, T), (1, 1), mode="replicate")
            smooth_k = torch.tensor([0.25, 0.50, 0.25], device=device, dtype=torch.float32).view(1, 1, 3)
            sec_amp = F.conv1d(pad_a, smooth_k).view(B, V - 1, T)

        track_amp = torch.cat([periodicity.unsqueeze(1), sec_amp], dim=1).clamp(0.0, 1.0)
        track_pan = (local_probs.detach() * bin_pan.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
        track_f0_hz = 2.0 ** smooth_log2
        return track_f0_hz, track_amp, track_pan

    def forward(
        self,
        lr_log_mel: torch.Tensor,
        ms_log_mel: torch.Tensor,
        acf_salience: torch.Tensor,
        comb_salience: torch.Tensor,
        pan_salience: torch.Tensor,
        stereo_energy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _, T, _ = lr_log_mel.shape
        N_bins = self.cfg.n_f0_bins
        mel_l = lr_log_mel[:, 0]
        mel_r = lr_log_mel[:, 1]
        mel_mid = ms_log_mel[:, 0]
        mel_side = ms_log_mel[:, 1]

        feat = torch.cat(
            [mel_l, mel_r, mel_mid, mel_side, acf_salience, comb_salience, pan_salience, stereo_energy],
            dim=-1,
        )

        x = self.in_proj(self.in_norm(feat))
        x = x + self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        h = self.out_norm(x)

        bin_ctx = self.bin_feature_proj(h).view(B, T, 2, N_bins)
        cues_stack = torch.stack(
            [acf_salience.float(), comb_salience.float(), pan_salience.float()],
            dim=2,
        )
        bin_in = torch.cat([bin_ctx.float(), cues_stack], dim=2).reshape(B * T, 5, N_bins)
        bin_out = self.lateral_inhibition_net(bin_in).view(B, T, 4, N_bins)

        raw_logits = bin_out[:, :, 0, :]
        raw_gate = bin_out[:, :, 1, :]
        raw_fine = bin_out[:, :, 2, :]
        raw_pan = bin_out[:, :, 3, :]

        acoustic_prior = 0.60 * acf_salience.float() + 0.40 * comb_salience.float()
        f0_logits = raw_logits + self.salience_skip_gain.float() * acoustic_prior
        gate_logits = raw_gate + 2.0 * acoustic_prior - 1.2

        fine_offset = 0.5 * self.bin_step_log2 * torch.tanh(raw_fine)
        log2_grid = self.log2_f0_grid.to(f0_logits.device, dtype=torch.float32).view(1, 1, N_bins)
        bin_log2_f0 = (log2_grid + fine_offset).clamp(math.log2(self.cfg.f0_min), math.log2(self.cfg.f0_max))

        bin_pan = torch.tanh(raw_pan + self.pan_skip_gain.float() * pan_salience.float())

        voicing_logit = self.voicing_head(h).squeeze(-1).float()
        periodicity = torch.sigmoid(voicing_logit)

        soft_dist = F.softmax(f0_logits, dim=-1)
        multi_voice_gate = torch.sigmoid(gate_logits)
        local_max = F.max_pool1d(
            soft_dist.reshape(B * T, 1, N_bins), kernel_size=5, stride=1, padding=2
        ).view(B, T, N_bins)
        peak_mask = (soft_dist >= (local_max - 1e-6)).float()
        sparse_probs = soft_dist * peak_mask * multi_voice_gate
        sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pitch_probs = soft_dist + (sparse_probs - soft_dist).detach()

        track_f0_hz, track_amp, track_pan = self.extract_dynamic_voice_tracks(
            f0_logits, multi_voice_gate, fine_offset, bin_pan, periodicity
        )

        lead_f0_hz = torch.where(periodicity > 0.35, track_f0_hz[:, 0], torch.zeros_like(periodicity))

        return {
            "f0_logits": f0_logits,
            "gate_logits": gate_logits,
            "pitch_probs": pitch_probs,
            "multi_voice_gate": multi_voice_gate,
            "bin_log2_f0": bin_log2_f0,
            "bin_pan": bin_pan,
            "periodicity": periodicity,
            "voicing_logit": voicing_logit,
            "track_f0_hz": track_f0_hz,
            "track_amp": track_amp,
            "track_pan": track_pan,
            "f0_hz": lead_f0_hz,
        }


class StereoContentEncoder(nn.Module):
    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.fsq_levels = int(getattr(cfg, "fsq_levels", 15))
        self.in_proj = nn.Linear(cfg.n_mels * 3, cfg.dim)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=5, padding=2, groups=cfg.dim),
            nn.GELU(),
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            XSARoPETransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                ffn_mult=cfg.ffn_mult,
                rope_theta=cfg.rope_theta,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.encoder_depth)
        ])
        self.out_norm = RMSNorm(cfg.dim)
        self.bottleneck = nn.Linear(cfg.dim, cfg.content_dim, bias=False)

    @staticmethod
    def _frame_norm(mel: torch.Tensor) -> torch.Tensor:
        """Gain- and chunk-length-invariant per-frame spectral normalization."""
        m = mel.float()
        mu = m.mean(dim=-1, keepdim=True)
        sd = m.std(dim=-1, unbiased=False, keepdim=True) + 0.5
        return (m - mu) / sd

    def forward(
        self, lr_log_mel: torch.Tensor, return_continuous: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        mel_l = self._frame_norm(lr_log_mel[:, 0])
        mel_r = self._frame_norm(lr_log_mel[:, 1])
        mel_mid = self._frame_norm(0.5 * (lr_log_mel[:, 0] + lr_log_mel[:, 1]))
        x_cat = torch.cat([mel_l, mel_r, mel_mid], dim=-1).to(self.in_proj.weight.dtype)

        x = self.in_proj(x_cat)
        x = x + self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        z_cont = torch.tanh(self.bottleneck(self.out_norm(x)).float())
        if self.fsq_levels > 1:
            half = float(self.fsq_levels - 1) * 0.5
            z_q = torch.round(z_cont * half) / half
            z_out = (z_cont + (z_q - z_cont).detach()).to(x.dtype)
        else:
            z_out = z_cont.to(x.dtype)

        if return_continuous:
            return z_out, z_cont
        return z_out


class StereoPitchEnergyConditioner(nn.Module):
    """
    Continuous Polyphonic Pitch & Stereo Energy Conditioner.
    Encodes the extracted voice tracks (`track_f0_hz [B, V, T]`, `track_amp [B, V, T]`,
    `track_pan [B, V, T]`), frame `periodicity [B, T]`, and `stereo_energy [B, T, 3]` into `cfg.dim`.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        freqs = torch.exp(torch.linspace(0.0, math.log(64.0), 32))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(139, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
            RMSNorm(cfg.dim),
        )

    def forward(
        self,
        track_f0_hz: torch.Tensor,
        track_amp: torch.Tensor,
        track_pan: torch.Tensor,
        periodicity: torch.Tensor,
        stereo_energy: torch.Tensor,
    ) -> torch.Tensor:
        log_f0 = torch.log2(track_f0_hz.float().clamp(min=1.0) / 220.0)
        w = track_amp.float().unsqueeze(-1)
        angles = log_f0.unsqueeze(-1) * self.freqs.to(track_f0_hz.device).view(1, 1, 1, -1) * math.pi
        fourier_per_voice = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) * w

        lead_fourier = fourier_per_voice[:, 0]
        lead_log_f0 = log_f0[:, 0].unsqueeze(-1) * w[:, 0]
        lead_pan = track_pan[:, 0].float().unsqueeze(-1) * w[:, 0]

        sec_fourier = fourier_per_voice[:, 1:].sum(dim=1)
        sec_log_f0 = (log_f0[:, 1:].unsqueeze(-1) * w[:, 1:]).sum(dim=1)
        sec_pan = (track_pan[:, 1:].float().unsqueeze(-1) * w[:, 1:]).sum(dim=1)

        feat = torch.cat(
            [
                lead_fourier,
                lead_log_f0,
                lead_pan,
                sec_fourier,
                sec_log_f0,
                sec_pan,
                periodicity.float().unsqueeze(-1),
                track_amp.float().transpose(1, 2),
                (stereo_energy.float() + 4.0) / 4.0,
            ],
            dim=-1,
        )
        return self.mlp(feat.to(self.mlp[0].weight.dtype))


class Snake1d(nn.Module):
    """Numerically safe periodic Snake activation: x + sin^2(alpha * x) / alpha."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha.float()).clamp(1e-3, 100.0)
        xf = x.float()
        return (xf + (torch.sin(alpha * xf) ** 2) / alpha).to(x.dtype)


class _ResidualAcousticHead(nn.Module):
    """Two-layer nonlinear projection head decoupling acoustic spectral targets from transformer hidden dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StereoWaveformRefiner(nn.Module):
    """
    Sample-rate (44.1 kHz) short-receptive-field (~3 ms) Neural Source-Filter Refiner:
      - Receives synthesized `y_dsp`, pure harmonic carrier `harmonic_sig`, and gated white noise,
        plus 2.9 ms-resolved FiLM conditioning from decoder hidden states `h`.
      - Because its receptive field (~3 ms) is shorter than a vocal pitch period, it cannot invent
        or override F0, while eliminating STFT ringing and sharpening plosives & glottal waveforms.
      - Zero-initialized output projection starts identically at `pred_wav = y_dsp`.
    """

    def __init__(self, cfg: SVCConfig, channels: int = 32):
        super().__init__()
        self.channels = channels
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.dim, 64),
            nn.SiLU(),
        )
        self.cond_upsample = nn.Sequential(
            nn.ConvTranspose1d(64, channels, kernel_size=7, stride=4, padding=3),
            nn.SiLU(),
        )
        self.in_conv = nn.Conv1d(6, channels, kernel_size=7, padding=3)
        dilations = (1, 2, 4, 8, 16)
        self.snakes1 = nn.ModuleList([Snake1d(channels) for _ in dilations])
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * d, dilation=d)
            for d in dilations
        ])
        self.films = nn.ModuleList([nn.Conv1d(channels, channels * 2, kernel_size=1) for _ in dilations])
        self.snakes2 = nn.ModuleList([Snake1d(channels) for _ in dilations])
        self.res_projs = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=1) for _ in dilations])
        self.out_snake = Snake1d(channels)
        self.out_conv = nn.Conv1d(channels, 2, kernel_size=7, padding=3)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _refine_core(
        self,
        y_dsp: torch.Tensor,
        harmonic_sig: torch.Tensor,
        stereo_white_noise: torch.Tensor,
        cond_4x: torch.Tensor,
    ) -> torch.Tensor:
        L = y_dsp.shape[-1]
        align_flag = cond_4x.shape[-1] > 1
        cond_up = F.interpolate(cond_4x, size=L, mode="linear", align_corners=align_flag)
        sig_in = torch.cat([y_dsp, harmonic_sig, stereo_white_noise], dim=1)
        x = self.in_conv(sig_in)
        for s1, conv, film, s2, proj in zip(
            self.snakes1, self.convs, self.films, self.snakes2, self.res_projs
        ):
            h_res = conv(s1(x))
            scale, shift = film(cond_up).chunk(2, dim=1)
            h_res = proj(s2(h_res * (1.0 + torch.tanh(scale) * 0.5) + shift))
            x = x + h_res
        return y_dsp + self.out_conv(self.out_snake(x))

    def forward(
        self,
        h_f32: torch.Tensor,
        y_dsp: torch.Tensor,
        harmonic_sig: torch.Tensor,
        stereo_white_noise: torch.Tensor,
    ) -> torch.Tensor:
        cond_4x = self.cond_upsample(self.cond_proj(h_f32).transpose(1, 2))
        if self.training and (h_f32.requires_grad or y_dsp.requires_grad):
            return grad_checkpoint(
                self._refine_core,
                y_dsp,
                harmonic_sig,
                stereo_white_noise,
                cond_4x,
                use_reentrant=False,
            )
        return self._refine_core(y_dsp, harmonic_sig, stereo_white_noise, cond_4x)


class StereoHarmonicPlusNoiseSynthesizer(nn.Module):
    """
    Low-VRAM, High-Speed Sample-Rate (44.1 kHz) Time-Domain Polyphonic Harmonic + Formant Synthesizer:
      - Integrates instantaneous phase (`cumsum`) strictly on the 3 fundamental voice trajectories
        `[B, V, L]`, synthesizing 128 harmonics in 32-harmonic checkpointed slices.
      - Uses L2 Parseval normalization for RMS-exact harmonic amplitudes and learnable glottal phase.
      - Shapes vocal timbre with a 96-band Mel-spaced vocal-tract formant filter (43 dB range) and
        a sample-rate short-receptive-field Snake Neural Source-Filter refiner.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        n_freqs = cfg.n_fft // 2 + 1
        self.n_freqs = n_freqs
        K = cfg.n_harmonics
        n_bands = cfg.n_formant_bands

        basis = create_mel_to_fft_basis(cfg.sample_rate, cfg.n_fft, n_bands, device=torch.device("cpu"))
        self.register_buffer("mel_to_fft_basis", basis, persistent=False)

        self.harm_common_phase = nn.Parameter(torch.zeros(K))
        self.harm_head = _ResidualAcousticHead(cfg.dim, 2 * (K + 1) + K)
        self.noise_filter_head = _ResidualAcousticHead(cfg.dim, 2 * n_bands)
        self.formant_filter_head = _ResidualAcousticHead(cfg.dim, 2 * 2 * n_bands)
        self.mel_pred_head = _ResidualAcousticHead(cfg.dim, 2 * cfg.n_mels)
        self.refiner = StereoWaveformRefiner(cfg, channels=32)

    def _synthesize_harmonic_waveform(
        self,
        harm_amps: torch.Tensor,
        ild_phase_part: torch.Tensor,
        harm_common_phase: torch.Tensor,
        track_f0_hz: torch.Tensor,
        track_amp: torch.Tensor,
        track_pan: torch.Tensor,
        base_phase: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        B, _, T, K = harm_amps.shape
        V = track_f0_hz.shape[1]
        device = harm_amps.device
        align_flag = T > 1
        nyquist_hz = self.cfg.sample_rate * 0.48

        amps_up = F.interpolate(
            harm_amps.reshape(B * 2, T, K).transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=align_flag,
        ).view(B, 2, K, target_length)
        phase_diff_up = F.interpolate(
            ild_phase_part.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=align_flag,
        )
        f0_up = F.interpolate(
            track_f0_hz.float(), size=target_length, mode="linear", align_corners=align_flag
        )
        amp_v_up = F.interpolate(
            track_amp.float(), size=target_length, mode="linear", align_corners=align_flag
        )
        pan_v_up = F.interpolate(
            track_pan.float(), size=target_length, mode="linear", align_corners=align_flag
        )

        k_idx = torch.arange(1, K + 1, device=device, dtype=torch.float32).view(1, K, 1)
        harmonic_sig = torch.zeros(B, 2, target_length, device=device, dtype=torch.float32)

        chunk_k = 32
        for v_idx in range(V):
            if v_idx == 0:
                pan_l = amp_v_up[:, 0:1, :]
                pan_r = amp_v_up[:, 0:1, :]
            else:
                pan_angle = (pan_v_up[:, v_idx : v_idx + 1, :].clamp(-1.0, 1.0) + 1.0) * (math.pi / 4.0)
                pan_l = amp_v_up[:, v_idx : v_idx + 1, :] * torch.cos(pan_angle) * 0.85
                pan_r = amp_v_up[:, v_idx : v_idx + 1, :] * torch.sin(pan_angle) * 0.85

            sig_l = torch.zeros(B, 1, target_length, device=device, dtype=torch.float32)
            sig_r = torch.zeros(B, 1, target_length, device=device, dtype=torch.float32)
            bp_v = base_phase[:, v_idx : v_idx + 1, :]
            f0_v = f0_up[:, v_idx : v_idx + 1, :]

            for k_start in range(0, K, chunk_k):
                k_end = min(K, k_start + chunk_k)
                k_sl = k_idx[:, k_start:k_end, :]
                harm_freqs_v = f0_v * k_sl
                nyq_mask_v = torch.sigmoid((nyquist_hz - harm_freqs_v) * 0.01)

                common_ph = harm_common_phase[k_start:k_end].float().view(1, -1, 1)
                init_offset = float(v_idx) * 2.3999632 * k_sl + common_ph
                phi_center = bp_v * k_sl + init_offset
                pd_sl = phase_diff_up[:, k_start:k_end, :]

                sig_l = sig_l + torch.sum(
                    amps_up[:, 0, k_start:k_end] * nyq_mask_v * torch.sin(phi_center + 0.5 * pd_sl),
                    dim=1,
                    keepdim=True,
                )
                sig_r = sig_r + torch.sum(
                    amps_up[:, 1, k_start:k_end] * nyq_mask_v * torch.sin(phi_center - 0.5 * pd_sl),
                    dim=1,
                    keepdim=True,
                )

            harmonic_sig = harmonic_sig + torch.cat([sig_l * pan_l, sig_r * pan_r], dim=1)

        return harmonic_sig

    def forward(
        self,
        h: torch.Tensor,
        track_f0_hz: torch.Tensor,
        track_amp: torch.Tensor,
        track_pan: torch.Tensor,
        extractor: StereoAcousticFeatureExtractor,
        target_length: int,
        init_phase: Optional[torch.Tensor] = None,
        phase_anchor_idx: int = 0,
        return_phase: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        B, T, _ = h.shape
        K = self.cfg.n_harmonics
        n_bands = self.cfg.n_formant_bands
        device = h.device
        align_flag = T > 1

        with torch.amp.autocast(device_type=device.type, enabled=False):
            h_f32 = h.float()
            pred_mel = (
                self.mel_pred_head(h_f32)
                .view(B, T, 2, self.cfg.n_mels)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            harm_raw = self.harm_head(h_f32)
            amp_part = harm_raw[..., : 2 * (K + 1)].view(B, T, 2, K + 1).permute(0, 2, 1, 3)
            ild_phase_part = torch.tanh(harm_raw[..., 2 * (K + 1) :]) * (math.pi / 4.0)

            overall_gain = F.softplus(amp_part[..., :1]) * 0.35
            harm_amps = F.softplus(amp_part[..., 1:])
            harm_amps = harm_amps / torch.sqrt((harm_amps ** 2).sum(dim=-1, keepdim=True) + 1e-8) * overall_gain

            f0_up_phase = F.interpolate(
                track_f0_hz.float(), size=target_length, mode="linear", align_corners=align_flag
            )
            base_phase = torch.cumsum(f0_up_phase / float(self.cfg.sample_rate), dim=-1) * (2.0 * math.pi)
            if init_phase is not None:
                anchor = base_phase[:, :, phase_anchor_idx : phase_anchor_idx + 1]
                base_phase = base_phase + (init_phase.to(device=device, dtype=torch.float32) - anchor)

            if self.training and h_f32.requires_grad:
                harmonic_sig = grad_checkpoint(
                    self._synthesize_harmonic_waveform,
                    harm_amps,
                    ild_phase_part,
                    self.harm_common_phase,
                    track_f0_hz.float(),
                    track_amp.float(),
                    track_pan.float(),
                    base_phase,
                    target_length,
                    use_reentrant=False,
                )
            else:
                harmonic_sig = self._synthesize_harmonic_waveform(
                    harm_amps,
                    ild_phase_part,
                    self.harm_common_phase,
                    track_f0_hz.float(),
                    track_amp.float(),
                    track_pan.float(),
                    base_phase,
                    target_length,
                )

            lead_voiced_up = F.interpolate(
                track_amp[:, 0:1].float(), size=target_length, mode="linear", align_corners=align_flag
            )

            basis = self.mel_to_fft_basis.to(device=device, dtype=torch.float32)

            noise_bands = (
                F.softplus(self.noise_filter_head(h_f32))
                .view(B, T, 2, n_bands)
                .permute(0, 2, 1, 3)
                .contiguous()
                * 0.20
            )
            noise_env = torch.matmul(noise_bands, basis).permute(0, 1, 3, 2).contiguous()

            mid_noise = torch.randn(B, 1, target_length, device=device, dtype=torch.float32) * 0.25
            side_noise = torch.randn(B, 2, target_length, device=device, dtype=torch.float32) * 0.15
            unvoiced_boost = 1.0 - 0.35 * lead_voiced_up
            stereo_white_noise = (mid_noise + side_noise) * unvoiced_boost

            noise_spec = extractor.stft(stereo_white_noise)
            T_stft = noise_spec.shape[-1]
            if noise_env.shape[-1] != T_stft:
                noise_env = F.interpolate(
                    noise_env.reshape(B * 2, self.n_freqs, -1),
                    size=T_stft,
                    mode="linear",
                    align_corners=align_flag,
                ).view(B, 2, self.n_freqs, T_stft)
            filtered_noise_spec = noise_spec * noise_env

            harm_spec = extractor.stft(harmonic_sig)
            if harm_spec.shape[-1] != T_stft:
                T_stft = min(harm_spec.shape[-1], filtered_noise_spec.shape[-1])
                harm_spec = harm_spec[..., :T_stft]
                filtered_noise_spec = filtered_noise_spec[..., :T_stft]

            base_spec = harm_spec + filtered_noise_spec

            formant_raw = (
                self.formant_filter_head(h_f32)
                .view(B, T, 2, 2, n_bands)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )
            log_gain_band = torch.tanh(formant_raw[:, :, 0]) * 2.5 - 0.3
            phase_disp_band = torch.tanh(formant_raw[:, :, 1]) * (math.pi / 6.0)

            log_gain_fft = torch.matmul(log_gain_band, basis).permute(0, 1, 3, 2).contiguous()
            phase_disp_fft = torch.matmul(phase_disp_band, basis).permute(0, 1, 3, 2).contiguous()

            if log_gain_fft.shape[-1] != T_stft:
                log_gain_fft = F.interpolate(
                    log_gain_fft.reshape(B * 2, self.n_freqs, -1),
                    size=T_stft,
                    mode="linear",
                    align_corners=align_flag,
                ).view(B, 2, self.n_freqs, T_stft)
                phase_disp_fft = F.interpolate(
                    phase_disp_fft.reshape(B * 2, self.n_freqs, -1),
                    size=T_stft,
                    mode="linear",
                    align_corners=align_flag,
                ).view(B, 2, self.n_freqs, T_stft)

            formant_mag = torch.exp(log_gain_fft)
            complex_filt = torch.polar(formant_mag.float(), phase_disp_fft.float())

            out_spec = base_spec.to(torch.complex64) * complex_filt
            y_dsp = extractor.istft(out_spec, length=target_length)

            pred_wav = self.refiner(h_f32, y_dsp, harmonic_sig, stereo_white_noise)
            if return_phase:
                return pred_wav, pred_mel, base_phase
            return pred_wav, pred_mel


class SingingVoiceConverter(nn.Module):
    """
    Full Native Stereo Polyphonic Singing Voice Conversion model (44.1 kHz) with
    learned sparse polyphonic RoFormer pitch tracking, speaker-invariant FSQ content
    encoding, and low-VRAM sample-rate synthesis + Snake waveform refinement.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        self.extractor = StereoAcousticFeatureExtractor(cfg)
        self.pitch_tracker = StereoRoFormerPitchTracker(cfg)
        self.content_encoder = StereoContentEncoder(cfg)
        self.content_proj = nn.Linear(cfg.content_dim, cfg.dim)
        self.pitch_cond = StereoPitchEnergyConditioner(cfg)

        self.target_singer_emb = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

        self.decoder_conv_stem = nn.Sequential(
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=5, padding=2, groups=cfg.dim),
            nn.GELU(),
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=1),
        )
        self.decoder_blocks = nn.ModuleList([
            XSARoPETransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                ffn_mult=cfg.ffn_mult,
                rope_theta=cfg.rope_theta,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.decoder_depth)
        ])
        self.decoder_norm = RMSNorm(cfg.dim)
        self.synthesizer = StereoHarmonicPlusNoiseSynthesizer(cfg)

        self.register_buffer("target_f0_median", torch.tensor(220.0))
        self.register_buffer("target_f0_std", torch.tensor(50.0))
        self.register_buffer("has_f0_stats", torch.tensor(False))
        self.register_buffer("target_rms", torch.tensor(0.08))
        self.register_buffer("has_rms_stats", torch.tensor(False))

    @torch.no_grad()
    def update_f0_stats(self, f0_hz: torch.Tensor, wav: Optional[torch.Tensor] = None) -> None:
        voiced = f0_hz.detach()[f0_hz.detach() > 1.0]
        if voiced.numel() > 16:
            med = voiced.median()
            std = voiced.std().clamp(min=5.0)
            if not bool(self.has_f0_stats.item()):
                self.target_f0_median.copy_(med)
                self.target_f0_std.copy_(std)
                self.has_f0_stats.fill_(True)
            else:
                self.target_f0_median.mul_(0.98).add_(med * 0.02)
                self.target_f0_std.mul_(0.98).add_(std * 0.02)

        if wav is not None:
            rms = torch.sqrt((wav.detach().float() ** 2).mean() + 1e-8).clamp(1e-3, 0.5)
            if not bool(self.has_rms_stats.item()):
                self.target_rms.copy_(rms)
                self.has_rms_stats.fill_(True)
            else:
                self.target_rms.mul_(0.98).add_(rms * 0.02)

    @staticmethod
    def _roll_pad_last(x: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return x
        if shift > 0:
            pad_edge = x[..., :1].expand(*x.shape[:-1], shift)
            return torch.cat([pad_edge, x[..., :-shift]], dim=-1)
        else:
            s = -shift
            pad_edge = x[..., -1:].expand(*x.shape[:-1], s)
            return torch.cat([x[..., s:], pad_edge], dim=-1)

    def _shift_pitch_features(
        self,
        lr_log_mel: torch.Tensor,
        ms_log_mel: torch.Tensor,
        acf_salience: torch.Tensor,
        comb_salience: torch.Tensor,
        pan_salience: torch.Tensor,
        shift_f0_bins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if shift_f0_bins == 0:
            return lr_log_mel, ms_log_mel, acf_salience, comb_salience, pan_salience

        shifted_acf = self._roll_pad_last(acf_salience, shift_f0_bins)
        shifted_comb = self._roll_pad_last(comb_salience, shift_f0_bins)
        shifted_pan = self._roll_pad_last(pan_salience, shift_f0_bins)
        shifted_lr_mel = self.extractor.shift_mel_by_f0_bins(lr_log_mel, shift_f0_bins)
        shifted_ms_mel = self.extractor.shift_mel_by_f0_bins(ms_log_mel, shift_f0_bins)
        return shifted_lr_mel, shifted_ms_mel, shifted_acf, shifted_comb, shifted_pan

    def forward_train(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, L = wav.shape
        target_lr_mel, target_ms_mel, _ = self.extractor.compute_stereo_log_mel(wav)
        pitch_cues = self.extractor.extract_pitch_salience_and_stereo_energy(wav)

        acf_salience = pitch_cues["acf_salience"]
        comb_salience = pitch_cues["comb_salience"]
        pan_salience = pitch_cues["pan_salience"]
        stereo_energy = pitch_cues["stereo_energy"]
        ref_pitch_dist = pitch_cues["ref_pitch_dist"]
        ref_multi_activity = pitch_cues["ref_multi_activity"]
        ref_voicing = pitch_cues["ref_voicing"]
        ref_conf = pitch_cues["ref_conf"]

        T = min(target_lr_mel.shape[2], acf_salience.shape[1])
        target_lr_mel = target_lr_mel[:, :, :T, :]
        target_ms_mel = target_ms_mel[:, :, :T, :]
        acf_salience = acf_salience[:, :T, :]
        comb_salience = comb_salience[:, :T, :]
        pan_salience = pan_salience[:, :T, :]
        stereo_energy = stereo_energy[:, :T, :]
        ref_pitch_dist = ref_pitch_dist[:, :T, :]
        ref_multi_activity = ref_multi_activity[:, :T, :]
        ref_voicing = ref_voicing[:, :T]
        ref_conf = ref_conf[:, :T]

        pitch_out = self.pitch_tracker(
            target_lr_mel, target_ms_mel, acf_salience, comb_salience, pan_salience, stereo_energy
        )
        f0_logits = pitch_out["f0_logits"]
        gate_logits = pitch_out["gate_logits"]
        pitch_probs = pitch_out["pitch_probs"]
        track_f0_hz = pitch_out["track_f0_hz"]
        track_amp = pitch_out["track_amp"]
        track_pan = pitch_out["track_pan"]
        periodicity = pitch_out["periodicity"]
        voicing_logit = pitch_out["voicing_logit"]
        f0_hz = pitch_out["f0_hz"]

        self.update_f0_stats(f0_hz, wav=wav)

        shift_bins = random.choice([-18, -12, -8, -5, 5, 8, 12, 18])
        sh_lr, sh_ms, sh_acf, sh_comb, sh_pan = self._shift_pitch_features(
            target_lr_mel[:1], target_ms_mel[:1], acf_salience[:1], comb_salience[:1], pan_salience[:1], shift_bins
        )
        shifted_pitch_out = self.pitch_tracker(sh_lr, sh_ms, sh_acf, sh_comb, sh_pan, stereo_energy[:1])

        aug_mel_1 = self.extractor.augment_mel_for_content(
            target_lr_mel, strength_range=(0.0, 0.85), clean_prob=0.15
        )
        aug_mel_2 = self.extractor.augment_mel_for_content(
            target_lr_mel, strength_range=(0.45, 1.0), clean_prob=0.0
        )
        z_aug1, z_cont1 = self.content_encoder(aug_mel_1, return_continuous=True)
        _, z_cont2 = self.content_encoder(aug_mel_2, return_continuous=True)
        content_mask = (target_lr_mel.amax(dim=(1, 3)) > -10.5).float().detach()

        soft_gate = torch.sigmoid((periodicity - 0.30) * 12.0).unsqueeze(1)
        gated_track_amp = track_amp * soft_gate

        cond = (
            self.content_proj(z_aug1)
            + self.pitch_cond(track_f0_hz, gated_track_amp, track_pan, periodicity, stereo_energy)
            + self.target_singer_emb
        )

        h = cond + self.decoder_conv_stem(cond.transpose(1, 2)).transpose(1, 2)
        for blk in self.decoder_blocks:
            h = blk(h)
        h = self.decoder_norm(h)

        pred_wav, pred_lr_mel = self.synthesizer(
            h,
            track_f0_hz,
            gated_track_amp,
            track_pan,
            self.extractor,
            target_length=L,
        )
        return {
            "pred_wav": pred_wav,
            "pred_lr_mel": pred_lr_mel,
            "target_lr_mel": target_lr_mel,
            "target_ms_mel": target_ms_mel,
            "z_cont1": z_cont1,
            "z_cont2": z_cont2,
            "content_mask": content_mask,
            "f0_logits": f0_logits,
            "gate_logits": gate_logits,
            "pitch_probs": pitch_probs,
            "shifted_f0_logits": shifted_pitch_out["f0_logits"],
            "shift_bins": shift_bins,
            "track_f0_hz": track_f0_hz,
            "track_amp": track_amp,
            "voicing_logit": voicing_logit,
            "periodicity": periodicity,
            "ref_pitch_dist": ref_pitch_dist,
            "ref_multi_activity": ref_multi_activity,
            "ref_voicing": ref_voicing,
            "ref_conf": ref_conf,
        }

    @torch.no_grad()
    def estimate_global_f0_scale(
        self,
        full_wav: torch.Tensor,
        pitch_shift_semitones: float = 0.0,
        auto_f0_adjust: bool = True,
    ) -> float:
        scale = 2.0 ** (float(pitch_shift_semitones) / 12.0)
        if not auto_f0_adjust or not bool(self.has_f0_stats.item()):
            return scale

        max_probe_len = min(full_wav.shape[-1], self.cfg.sample_rate * 24)
        probe_wav = full_wav[:, :max_probe_len]
        chunk_len = self.cfg.sample_rate * 6
        voiced_vals = []
        for s in range(0, probe_wav.shape[-1], chunk_len):
            seg = probe_wav[:, s : s + chunk_len]
            if seg.shape[-1] < self.cfg.n_fft:
                continue
            lr_mel, ms_mel, _ = self.extractor.compute_stereo_log_mel(seg.unsqueeze(0))
            cues = self.extractor.extract_pitch_salience_and_stereo_energy(seg.unsqueeze(0))
            T = min(lr_mel.shape[2], cues["acf_salience"].shape[1])
            p_out = self.pitch_tracker(
                lr_mel[:, :, :T],
                ms_mel[:, :, :T],
                cues["acf_salience"][:, :T],
                cues["comb_salience"][:, :T],
                cues["pan_salience"][:, :T],
                cues["stereo_energy"][:, :T],
            )
            f0_seg = p_out["f0_hz"][0]
            v = f0_seg[f0_seg > 1.0]
            if v.numel() > 0:
                voiced_vals.append(v)

        if voiced_vals:
            all_voiced = torch.cat(voiced_vals, dim=0)
            if all_voiced.numel() > 8:
                src_med = all_voiced.median().clamp(min=1.0)
                tgt_med = self.target_f0_median.clamp(min=1.0)
                semi_diff = max(-18, min(18, round(12.0 * math.log2(float((tgt_med / src_med).item())))))
                scale = scale * (2.0 ** (semi_diff / 12.0))
        return scale

    @torch.no_grad()
    def convert_chunk(
        self,
        source_wav: torch.Tensor,
        f0_scale: float = 1.0,
        init_phase: Optional[torch.Tensor] = None,
        phase_anchor_idx: int = 0,
        return_phase: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        B, C, L_orig = source_wav.shape
        hop = self.cfg.hop_length
        min_len = max(self.cfg.n_fft + hop, hop * 6)
        target_len = ((max(L_orig, min_len) + hop - 1) // hop) * hop
        pad_rem = target_len - L_orig
        if pad_rem > 0:
            pad_mode = "reflect" if L_orig > pad_rem else "constant"
            source_in = F.pad(source_wav, (0, pad_rem), mode=pad_mode)
        else:
            source_in = source_wav
        L_pad = source_in.shape[-1]

        src_lr_mel, src_ms_mel, _ = self.extractor.compute_stereo_log_mel(source_in)
        pitch_cues = self.extractor.extract_pitch_salience_and_stereo_energy(source_in)

        acf_salience = pitch_cues["acf_salience"]
        comb_salience = pitch_cues["comb_salience"]
        pan_salience = pitch_cues["pan_salience"]
        stereo_energy = pitch_cues["stereo_energy"]

        T = min(src_lr_mel.shape[2], acf_salience.shape[1])
        src_lr_mel = src_lr_mel[:, :, :T, :]
        src_ms_mel = src_ms_mel[:, :, :T, :]
        acf_salience = acf_salience[:, :T, :]
        comb_salience = comb_salience[:, :T, :]
        pan_salience = pan_salience[:, :T, :]
        stereo_energy = stereo_energy[:, :T, :]

        pitch_out = self.pitch_tracker(
            src_lr_mel, src_ms_mel, acf_salience, comb_salience, pan_salience, stereo_energy
        )
        track_f0_hz = (pitch_out["track_f0_hz"] * float(f0_scale)).clamp(
            self.cfg.f0_min * 0.5, self.cfg.f0_max * 1.5
        )
        track_amp = pitch_out["track_amp"]
        track_pan = pitch_out["track_pan"]
        periodicity = pitch_out["periodicity"]

        soft_gate = torch.sigmoid((periodicity - 0.30) * 12.0).unsqueeze(1)
        gated_track_amp = track_amp * soft_gate

        z_content = self.content_encoder(src_lr_mel)
        cond = (
            self.content_proj(z_content)
            + self.pitch_cond(track_f0_hz, gated_track_amp, track_pan, periodicity, stereo_energy)
            + self.target_singer_emb
        )
        h = cond + self.decoder_conv_stem(cond.transpose(1, 2)).transpose(1, 2)
        for blk in self.decoder_blocks:
            h = blk(h)
        h = self.decoder_norm(h)

        synth_out = self.synthesizer(
            h,
            track_f0_hz,
            gated_track_amp,
            track_pan,
            self.extractor,
            target_length=L_pad,
            init_phase=init_phase,
            phase_anchor_idx=phase_anchor_idx,
            return_phase=return_phase,
        )
        if return_phase:
            pred_wav, _, base_phase = synth_out
            return pred_wav[:, :, :L_orig], base_phase[:, :, :L_orig]
        pred_wav, _ = synth_out
        return pred_wav[:, :, :L_orig]


class StereoMultiResolutionSVCLoss(nn.Module):
    """
    Polyphonic Stereo Multi-Resolution STFT (Magnitude + Spectral Convergence + Temporal/Spectral Flux)
    + Phase-Invariant Waveform RMS Envelope + Mel + Two-View VICReg Content + Sparse RoFormer Pitch Field Loss.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        self.fft_sizes = (256, 512, 1024, 2048, 4096)
        self.hops = (64, 128, 256, 512, 1024)
        self.wins = (256, 512, 1024, 2048, 4096)
        for w in self.wins:
            self.register_buffer(f"win_{w}", torch.hann_window(w), persistent=False)

    @staticmethod
    def _spec_metrics(p_spec: torch.Tensor, t_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_mag = p_spec.abs().clamp(min=1e-6)
        t_mag = t_spec.abs().clamp(min=1e-6)

        num = (t_mag - p_mag).pow(2).sum(dim=(-2, -1)).sqrt()
        den = t_mag.pow(2).sum(dim=(-2, -1)).sqrt().clamp(min=1e-6)
        sc = (num / den).mean()

        p_log = torch.log(p_mag)
        t_log = torch.log(t_mag)
        mag = F.l1_loss(p_log, t_log)

        t_flux = F.l1_loss(p_log[..., 1:] - p_log[..., :-1], t_log[..., 1:] - t_log[..., :-1])
        f_flux = F.l1_loss(p_log[..., 1:, :] - p_log[..., :-1, :], t_log[..., 1:, :] - t_log[..., :-1, :])
        flux = 0.5 * (t_flux + f_flux)
        return sc, mag, flux

    @staticmethod
    def _rms_envelope_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        p_ms = torch.stack([pred[:, 0], pred[:, 1], 0.5 * (pred[:, 0] + pred[:, 1]), 0.5 * (pred[:, 0] - pred[:, 1])], dim=1)
        t_ms = torch.stack([tgt[:, 0], tgt[:, 1], 0.5 * (tgt[:, 0] + tgt[:, 1]), 0.5 * (tgt[:, 0] - tgt[:, 1])], dim=1)
        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        for k_size, stride in ((88, 22), (352, 88)):
            if pred.shape[-1] < k_size:
                continue
            p_env = torch.sqrt(F.avg_pool1d(p_ms.pow(2), kernel_size=k_size, stride=stride) + 1e-7)
            t_env = torch.sqrt(F.avg_pool1d(t_ms.pow(2), kernel_size=k_size, stride=stride) + 1e-7)
            loss = loss + F.l1_loss(p_env, t_env) + 0.5 * F.l1_loss(torch.log(p_env), torch.log(t_env))
        return 0.5 * loss

    def _compute_roformer_pitch_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        f0_logits = out["f0_logits"].float()
        gate_logits = out["gate_logits"].float()
        shifted_f0_logits = out["shifted_f0_logits"].float()
        shift_bins = int(out["shift_bins"])
        track_f0_hz = out["track_f0_hz"].float()
        track_amp = out["track_amp"].float()
        voicing_logit = out["voicing_logit"].float()
        ref_pitch_dist = out["ref_pitch_dist"].float()
        ref_multi_activity = out["ref_multi_activity"].float()
        ref_voicing = out["ref_voicing"].float()
        ref_conf = out["ref_conf"].float()

        voiced_weights = (ref_voicing * ref_conf).detach()
        weight_denom = voiced_weights.sum().clamp(min=1.0)

        log_pred_probs = F.log_softmax(f0_logits, dim=-1)
        kl_per_frame = -(ref_pitch_dist.detach() * log_pred_probs).sum(dim=-1)
        dist_loss = (kl_per_frame * voiced_weights).sum() / weight_denom

        pos_weight = torch.tensor(10.0, device=gate_logits.device, dtype=torch.float32)
        gate_bce = F.binary_cross_entropy_with_logits(
            gate_logits, ref_multi_activity.detach(), pos_weight=pos_weight, reduction="none"
        )
        bg_mask = (ref_multi_activity.detach() < 0.05).float()
        sparsity_l1 = (torch.sigmoid(gate_logits) * bg_mask).mean(dim=-1)
        gate_loss = ((gate_bce.mean(dim=-1) + 0.5 * sparsity_l1) * voiced_weights).sum() / weight_denom

        soft_voicing_target = (0.75 * ref_voicing + 0.25 * ref_conf).clamp(0.0, 1.0).detach()
        voicing_loss = F.binary_cross_entropy_with_logits(voicing_logit, soft_voicing_target)

        pred_probs_det = F.softmax(f0_logits[:1].detach(), dim=-1)
        expected_shifted_probs = SingingVoiceConverter._roll_pad_last(pred_probs_det, shift_bins)
        expected_shifted_probs = expected_shifted_probs / expected_shifted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        shifted_log_probs = F.log_softmax(shifted_f0_logits, dim=-1)
        equiv_kl = -(expected_shifted_probs * shifted_log_probs).sum(dim=-1)
        equiv_weights = voiced_weights[:1]
        equiv_loss = (equiv_kl * equiv_weights).sum() / equiv_weights.sum().clamp(min=1.0)

        smooth_loss = torch.tensor(0.0, device=f0_logits.device, dtype=torch.float32)
        if track_f0_hz.shape[2] >= 3:
            track_log2 = torch.log2(track_f0_hz.clamp(min=1.0))
            v_pair = (track_amp[:, :, 1:] * track_amp[:, :, :-1]).detach()
            df1 = track_log2[:, :, 1:] - track_log2[:, :, :-1]
            vel_penalty = F.relu(df1.abs() - 0.10) ** 2
            v_triplet = (track_amp[:, :, 2:] * track_amp[:, :, 1:-1] * track_amp[:, :, :-2]).detach()
            df2 = track_log2[:, :, 2:] - 2.0 * track_log2[:, :, 1:-1] + track_log2[:, :, :-2]
            accel_penalty = df2 ** 2
            smooth_loss = (
                2.0 * (vel_penalty * v_pair).mean()
                + 1.5 * (accel_penalty * v_triplet).mean()
            )

        return 0.50 * dist_loss + 0.80 * gate_loss + 0.40 * voicing_loss + 0.60 * equiv_loss + smooth_loss

    @staticmethod
    def _compute_content_loss(
        z_cont1: torch.Tensor, z_cont2: torch.Tensor, content_mask: torch.Tensor
    ) -> torch.Tensor:
        z1 = z_cont1.float()
        z2 = z_cont2.float()
        z1_u = F.normalize(z1, p=2.0, dim=-1, eps=1e-6)
        z2_u = F.normalize(z2, p=2.0, dim=-1, eps=1e-6)

        inv12 = 1.0 - (z1_u * z2_u.detach()).sum(dim=-1)
        inv21 = 1.0 - (z2_u * z1_u.detach()).sum(dim=-1)
        inv_frame = 0.5 * (inv12 + inv21)
        mask_sum = content_mask.sum().clamp(min=1.0)
        inv_loss = (inv_frame * content_mask).sum() / mask_sum

        z_flat = z1.reshape(-1, z1.shape[-1])
        N, D = z_flat.shape
        if N > 4:
            std_c = torch.sqrt(z_flat.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(0.35 - std_c).mean()
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            cov = (z_centered.t() @ z_centered) / float(N - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = (off_diag ** 2).mean()
        else:
            var_loss = torch.tensor(0.0, device=z1.device)
            cov_loss = torch.tensor(0.0, device=z1.device)

        return inv_loss + 0.5 * var_loss + 0.05 * cov_loss

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        target_wav: torch.Tensor,
        extractor: StereoAcousticFeatureExtractor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_wav = out["pred_wav"]
        pred_lr_mel = out["pred_lr_mel"]
        target_lr_mel = out["target_lr_mel"]
        target_ms_mel = out["target_ms_mel"]

        with torch.amp.autocast(device_type=pred_wav.device.type, enabled=False):
            B, C, L = pred_wav.shape
            pred_f32 = pred_wav.float()
            tgt_f32 = target_wav.float()

            sc_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)
            mag_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)
            flux_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)
            used_res = 0

            for n_fft, hop, win_len in zip(self.fft_sizes, self.hops, self.wins):
                if L <= n_fft // 2:
                    continue
                win = getattr(self, f"win_{win_len}").to(pred_wav.device, dtype=torch.float32)
                p_spec = torch.stft(
                    pred_f32.reshape(B * C, L),
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win_len,
                    window=win,
                    return_complex=True,
                ).view(B, C, n_fft // 2 + 1, -1)
                t_spec = torch.stft(
                    tgt_f32.reshape(B * C, L),
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win_len,
                    window=win,
                    return_complex=True,
                ).view(B, C, n_fft // 2 + 1, -1)

                sc_lr, mag_lr, flux_lr = self._spec_metrics(p_spec, t_spec)

                p_ms = torch.stack([0.5 * (p_spec[:, 0] + p_spec[:, 1]), 0.5 * (p_spec[:, 0] - p_spec[:, 1])], dim=1)
                t_ms = torch.stack([0.5 * (t_spec[:, 0] + t_spec[:, 1]), 0.5 * (t_spec[:, 0] - t_spec[:, 1])], dim=1)
                sc_ms, mag_ms, flux_ms = self._spec_metrics(p_ms, t_ms)

                sc_loss = sc_loss + 0.5 * (sc_lr + sc_ms)
                mag_loss = mag_loss + 0.5 * (mag_lr + mag_ms)
                flux_loss = flux_loss + 0.5 * (flux_lr + flux_ms)
                used_res += 1

            n_res = float(max(1, used_res))
            sc_loss = sc_loss / n_res
            mag_loss = mag_loss / n_res
            flux_loss = flux_loss / n_res

            env_loss = self._rms_envelope_loss(pred_f32, tgt_f32)

            wav_lr_mel, wav_ms_mel, _ = extractor.compute_stereo_log_mel(pred_f32)
            T = min(pred_lr_mel.shape[2], target_lr_mel.shape[2], wav_lr_mel.shape[2])
            mel_loss = (
                F.l1_loss(pred_lr_mel[:, :, :T].float(), target_lr_mel[:, :, :T].float())
                + F.l1_loss(wav_lr_mel[:, :, :T].float(), target_lr_mel[:, :, :T].float())
                + 0.5 * F.l1_loss(wav_ms_mel[:, :, :T].float(), target_ms_mel[:, :, :T].float())
            )

            content_loss = self._compute_content_loss(
                out["z_cont1"], out["z_cont2"], out["content_mask"]
            )
            pitch_loss = self._compute_roformer_pitch_loss(out)

            stft_total = 1.0 * mag_loss + 0.6 * sc_loss + 0.8 * flux_loss + 1.0 * env_loss
            total = (
                stft_total
                + 1.5 * mel_loss
                + 0.6 * content_loss
                + 1.0 * pitch_loss
            )
            stats = {
                "total": float(total.item()),
                "stft": float(stft_total.item()),
                "mel": float(mel_loss.item()),
                "f0": float(pitch_loss.item()),
                "content": float(content_loss.item()),
            }
            return total, stats


class SingerAudioDataset(Dataset):
    """
    Loads a single WAV/FLAC file or a folder of WAV/FLAC files into memory as native
    44.1 kHz stereo (`[2, L]`) and draws un-aliased native-sample-rate crops so target
    vocal formants and high frequencies remain pristine.
    """

    def __init__(self, data_path: str, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        self.segment_samples = cfg.segment_frames * cfg.hop_length

        p = Path(data_path)
        files: List[Path] = []
        if p.is_file():
            files = [p]
        elif p.is_dir():
            valid_exts = {".wav", ".flac"}
            files = sorted({f.resolve() for f in p.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts})
        else:
            raise FileNotFoundError(f"Training path not found: {data_path}")

        if not files:
            raise RuntimeError(f"No .wav or .flac files found in '{data_path}'")

        self.waveforms: List[torch.Tensor] = []
        total_seconds = 0.0
        for f in files:
            wav = load_audio_stereo(str(f), target_sr=cfg.sample_rate)
            if wav.shape[-1] < cfg.n_fft * 2:
                continue
            self.waveforms.append(wav)
            total_seconds += wav.shape[-1] / float(cfg.sample_rate)

        if not self.waveforms:
            raise RuntimeError(f"No valid audio segments loaded from '{data_path}'")

        self.num_files = len(self.waveforms)
        self.total_seconds = total_seconds

    def __len__(self) -> int:
        return max(self.cfg.virtual_epoch_size, len(self.waveforms) * 16)

    def _sample_single_crop(self, wav: torch.Tensor) -> torch.Tensor:
        seg_len = self.segment_samples
        full_len = wav.shape[-1]

        if full_len <= seg_len:
            repeats = (seg_len // full_len) + 2
            crop = wav.repeat(1, repeats)[:, :seg_len]
        elif random.random() < 0.10:
            start = random.randint(0, full_len - seg_len)
            crop = wav[:, start : start + seg_len]
        else:
            best_crop = None
            best_rms = -1.0
            for _ in range(4):
                start = random.randint(0, full_len - seg_len)
                cand = wav[:, start : start + seg_len]
                rms = float(torch.sqrt((cand ** 2).mean() + 1e-12).item())
                if rms > 0.006:
                    best_crop = cand
                    break
                if rms > best_rms:
                    best_rms = rms
                    best_crop = cand
            crop = best_crop

        mid = 0.5 * (crop[0] + crop[1])
        side = 0.5 * (crop[0] - crop[1]) * random.uniform(0.90, 1.10)
        return torch.stack([mid + side, mid - side], dim=0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = self.waveforms[idx % len(self.waveforms)]
        crop = self._sample_single_crop(wav)

        gain = 10.0 ** (random.uniform(-6.0, 2.0) / 20.0)
        crop = (crop * gain).clamp(-0.99, 0.99)
        return crop


class ModelEMA:
    """Exponential Moving Average of model weights for clean, artifact-free inference."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            self.shadow[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, float(1 + step) / float(10 + step))
        one_minus_d = 1.0 - d
        for name, tensor in model.state_dict().items():
            src = tensor.detach()
            if name not in self.shadow or self.shadow[name].shape != src.shape:
                self.shadow[name] = src.clone()
                continue
            if src.is_floating_point():
                self.shadow[name].mul_(d).add_(src, alpha=one_minus_d)
            else:
                self.shadow[name].copy_(src)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow and self.shadow[name].shape == tensor.shape:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    latest_link = d / "latest.pt"
    if latest_link.is_file():
        return str(latest_link)

    def _step_key(p: Path) -> int:
        s = p.stem.split("_")[-1]
        return int(s) if s.isdigit() else int(p.stat().st_mtime)

    ckpts = sorted(d.glob("ckpt_step_*.pt"), key=_step_key, reverse=True)
    return str(ckpts[0]) if ckpts else None


def save_checkpoint_atomic(
    ckpt_dir: str,
    model: SingingVoiceConverter,
    optimizer: torch.optim.Optimizer,
    cfg: SVCConfig,
    epoch: int,
    step: int,
    best_loss: float,
    keep_last: int = 3,
    ema: Optional[ModelEMA] = None,
    ema_loss: Optional[float] = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "ema_loss": ema_loss,
        "timestamp": time.time(),
    }
    if ema is not None:
        payload["ema_state"] = ema.state_dict()

    step_path = os.path.join(ckpt_dir, f"ckpt_step_{step:07d}.pt")
    latest_path = os.path.join(ckpt_dir, "latest.pt")

    fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        shutil.copy2(tmp_path, step_path)
        os.replace(tmp_path, latest_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def _step_key(p: Path) -> int:
        s = p.stem.split("_")[-1]
        return int(s) if s.isdigit() else int(p.stat().st_mtime)

    all_step_ckpts = sorted(Path(ckpt_dir).glob("ckpt_step_*.pt"), key=_step_key)
    if keep_last > 0 and len(all_step_ckpts) > keep_last:
        for old_p in all_step_ckpts[:-keep_last]:
            try:
                old_p.unlink()
            except OSError:
                pass

    return step_path


def load_checkpoint_auto(
    ckpt_dir: str,
    device: torch.device,
    model: Optional[SingingVoiceConverter] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ModelEMA] = None,
    prefer_ema_weights: bool = False,
) -> Tuple[Optional[SingingVoiceConverter], SVCConfig, int, int, float, Optional[str], Optional[float]]:
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        cfg = model.cfg if model is not None else SVCConfig()
        return model, cfg, 0, 0, float("inf"), None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    if cfg_dict:
        valid_keys = set(SVCConfig.__dataclass_fields__.keys())
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = SVCConfig(**filtered_cfg)
    else:
        cfg = model.cfg if model is not None else SVCConfig()

    if model is None:
        model = SingingVoiceConverter(cfg).to(device)

    current_state = model.state_dict()
    saved_state = ckpt.get("ema_state") if (prefer_ema_weights and "ema_state" in ckpt) else ckpt.get("model_state", {})
    compatible_state = {}
    reshaped_count = 0
    for k, v in saved_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            compatible_state[k] = v
        else:
            reshaped_count += 1

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if missing or reshaped_count > 0:
        print(
            f"[Auto-Resume] Loaded {len(compatible_state)} matching tensors; "
            f"initialized {len(missing)} new parameter tensor(s)."
        )

    if ema is not None and "ema_state" in ckpt:
        try:
            ema.load_state_dict(ckpt["ema_state"])
        except Exception:
            pass

    if optimizer is not None and "optimizer_state" in ckpt and not missing and not unexpected and reshaped_count == 0:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[Warn] Could not restore optimizer state ({e}); continuing with fresh optimizer state.")

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))
    ema_loss = ckpt.get("ema_loss", None)
    ema_loss = float(ema_loss) if ema_loss is not None else None
    return model, cfg, epoch, step, best_loss, ckpt_path, ema_loss


def format_terminal_stats(
    epoch: int,
    total_epochs: int,
    step: int,
    loss_stats: Dict[str, float],
    ema_loss: float,
    prodigy_d: float,
    grad_norm: float,
    steps_per_sec: float,
    vram_gb: float,
) -> str:
    return (
        f"[Ep {epoch:03d}/{total_epochs:03d} | Step {step:06d}] "
        f"Loss: {loss_stats['total']:.4f} (EMA {ema_loss:.4f}) | "
        f"STFT: {loss_stats['stft']:.4f} | Mel: {loss_stats['mel']:.4f} | "
        f"F0: {loss_stats['f0']:.4f} | Cnt: {loss_stats['content']:.4f} | "
        f"Prodigy d: {prodigy_d:.2e} | Grad: {grad_norm:.2f} | "
        f"{steps_per_sec:.2f} it/s | VRAM: {vram_gb:.2f} GB"
    )


def train_svc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = SVCConfig(
        sample_rate=args.sample_rate,
        dim=args.dim,
        pitch_encoder_depth=args.pitch_depth,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        segment_frames=args.segment_frames,
    )

    dataset = SingerAudioDataset(args.data_dir, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = SingingVoiceConverter(cfg).to(device)
    ema = ModelEMA(model, decay=0.999)
    optimizer = build_prodigy_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    criterion = StereoMultiResolutionSVCLoss(cfg).to(device)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    ema_loss: Optional[float] = None

    if not args.fresh:
        _, _, start_epoch, global_step, best_loss, loaded_path, ema_loss = load_checkpoint_auto(
            args.ckpt_dir, device, model=model, optimizer=optimizer, ema=ema
        )
        if loaded_path is not None:
            print(f"[Auto-Resume] Resumed from '{loaded_path}' at Epoch {start_epoch}, Step {global_step}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pitch_params = sum(p.numel() for p in model.pitch_tracker.parameters() if p.requires_grad)
    print("=" * 100)
    print(f" Native Stereo Polyphonic SVC (44.1 kHz) | RoPE-QKV + XSA + GEGLU + FSQ + Snake-NSF | Device: {device}")
    print(f" Parameters: {n_params / 1e6:.2f}M (RoFormer Pitch Tracker: {n_pitch_params / 1e6:.2f}M) | Stereo Files: {dataset.num_files} ({dataset.total_seconds:.1f}s)")
    print(f" Optimizer: Prodigy (lr={args.lr}, d0=1e-6) | Checkpoints: {args.ckpt_dir} (every {args.save_every} steps, keep latest {args.keep_ckpts})")
    print("=" * 100)

    current_epoch = start_epoch

    def do_save_checkpoint(reason: str = "latest") -> str:
        nonlocal best_loss
        prev_sig = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            final_loss = ema_loss if ema_loss is not None else best_loss
            is_best = (ema_loss is not None) and (ema_loss < best_loss)
            if is_best:
                best_loss = ema_loss
            saved = save_checkpoint_atomic(
                args.ckpt_dir,
                model,
                optimizer,
                cfg,
                current_epoch,
                global_step,
                min(best_loss, final_loss),
                keep_last=args.keep_ckpts,
                ema=ema,
                ema_loss=ema_loss,
            )
            if is_best:
                shutil.copy2(saved, os.path.join(args.ckpt_dir, "best.pt"))
            return saved
        finally:
            signal.signal(signal.SIGINT, prev_sig)

    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    orig_int = signal.signal(signal.SIGINT, _sigint_handler)
    orig_break = signal.signal(signal.SIGBREAK, _sigint_handler) if hasattr(signal, "SIGBREAK") else None

    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    total_steps = args.max_steps if args.max_steps > 0 else (args.epochs * len(loader))
    total_steps = max(total_steps, global_step)

    pbar = tqdm(
        total=total_steps,
        initial=global_step,
        desc=f"Ep {start_epoch + 1}/{args.epochs}",
        dynamic_ncols=True,
        unit="step",
    )

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            current_epoch = epoch
            pbar.set_description(f"Ep {epoch}/{args.epochs}")
            model.train()
            for wav_batch in loader:
                if interrupted:
                    break

                wav_batch = wav_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model.forward_train(wav_batch)
                    loss, stats = criterion(
                        out,
                        wav_batch,
                        model.extractor,
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
                    optimizer.step()

                global_step += 1
                ema.update(model, global_step)
                pbar.update(1)
                ema_loss = stats["total"] if ema_loss is None else (0.95 * ema_loss + 0.05 * stats["total"])

                if global_step % args.log_every == 0 or global_step == 1:
                    prodigy_d = float(optimizer.param_groups[0].get("d", 1e-6))
                    postfix = {
                        "loss": f"{stats['total']:.4f}",
                        "ema": f"{ema_loss:.4f}",
                        "stft": f"{stats['stft']:.4f}",
                        "mel": f"{stats['mel']:.4f}",
                        "f0": f"{stats['f0']:.4f}",
                        "cnt": f"{stats['content']:.4f}",
                        "d": f"{prodigy_d:.1e}",
                        "grad": f"{grad_norm:.2f}",
                    }
                    if device.type == "cuda":
                        vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        postfix["vram"] = f"{vram_gb:.1f}G"
                    pbar.set_postfix(postfix)

                if global_step % args.save_every == 0:
                    saved_path = do_save_checkpoint("periodic")
                    tqdm.write(f"  --> [Auto-Save] Saved checkpoint '{saved_path}' (step {global_step})")

                if args.max_steps and global_step >= args.max_steps:
                    interrupted = True
                    break

            if interrupted:
                break
    except (KeyboardInterrupt, SystemExit):
        pbar.close()
        print(f"\n[Ctrl+C] Training interrupted by user at step {global_step}. Saving emergency checkpoint...", flush=True)
        try:
            saved_path = do_save_checkpoint("emergency")
            print(f"  --> [Emergency Save] Successfully saved checkpoint '{saved_path}' (step {global_step}, Epoch {current_epoch})", flush=True)
        except Exception as e:
            print(f"  --> [Emergency Save] Failed to save emergency checkpoint: {e}", flush=True)
    else:
        pbar.close()
        saved_path = do_save_checkpoint("done")
        print(f"[Done] Saved latest checkpoint to '{saved_path}' at step {global_step}.", flush=True)
    finally:
        pbar.close()
        signal.signal(signal.SIGINT, orig_int)
        if hasattr(signal, "SIGBREAK") and orig_break is not None:
            signal.signal(signal.SIGBREAK, orig_break)


@torch.no_grad()
def infer_svc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, epoch, step, best_loss, ckpt_path, _ = load_checkpoint_auto(
        args.ckpt_dir, device, prefer_ema_weights=(not args.no_ema)
    )
    if ckpt_path is None or model is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{args.ckpt_dir}'. Train a model first with `--train --data_dir <folder_or_file>`."
        )

    model.eval()
    print(f"[Auto-Infer] Loaded checkpoint '{ckpt_path}' (step {step}, epoch {epoch})")

    src_wav = load_audio_stereo(args.input, target_sr=cfg.sample_rate).to(device)
    total_len = src_wav.shape[-1]

    if bool(model.has_rms_stats.item()):
        src_rms = float(torch.sqrt((src_wav ** 2).mean() + 1e-8).item())
        if src_rms > 1e-5:
            scale_rms = max(0.25, min(4.0, float(model.target_rms.item()) / src_rms))
            peak = float(src_wav.abs().max().item())
            if peak * scale_rms > 0.95:
                scale_rms = 0.95 / max(1e-6, peak)
            src_wav = src_wav * scale_rms

    global_f0_scale = model.estimate_global_f0_scale(
        src_wav,
        pitch_shift_semitones=args.pitch_shift,
        auto_f0_adjust=not args.no_auto_f0,
    )

    chunk_samples = int(math.ceil((args.chunk_seconds * cfg.sample_rate) / cfg.hop_length) * cfg.hop_length)
    context_pad = 4 * cfg.hop_length
    xfade_samples = 2 * cfg.hop_length

    if total_len <= chunk_samples:
        pad_rem = (cfg.hop_length - (total_len % cfg.hop_length)) % cfg.hop_length
        pad_mode = "reflect" if total_len > pad_rem else "constant"
        wav_in = F.pad(src_wav, (0, pad_rem), mode=pad_mode).unsqueeze(0)
        out_wav = model.convert_chunk(wav_in, f0_scale=global_f0_scale).squeeze(0)[:, :total_len]
    else:
        hop_chunk = chunk_samples - xfade_samples
        out_buf = torch.zeros(2, total_len + chunk_samples, device=device)
        weight_buf = torch.zeros(1, total_len + chunk_samples, device=device)

        fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, xfade_samples, device=device)))
        fade_out = 0.5 * (1.0 + torch.cos(torch.linspace(0.0, math.pi, xfade_samples, device=device)))

        starts = list(range(0, total_len, hop_chunk))
        prev_init_phase: Optional[torch.Tensor] = None
        for idx_c, start in enumerate(tqdm(starts, desc="Converting", unit="chunk", dynamic_ncols=True)):
            end = min(total_len, start + chunk_samples)
            ctx_left = min(start, context_pad)
            ctx_right = min(total_len - end, context_pad)

            seg_with_ctx = src_wav[:, start - ctx_left : end + ctx_right]
            raw_len = seg_with_ctx.shape[-1]
            pad_rem = (cfg.hop_length - (raw_len % cfg.hop_length)) % cfg.hop_length
            if pad_rem > 0:
                pad_mode = "reflect" if raw_len > pad_rem else "constant"
                seg_with_ctx = F.pad(seg_with_ctx, (0, pad_rem), mode=pad_mode)

            conv_full, phase_full = model.convert_chunk(
                seg_with_ctx.unsqueeze(0),
                f0_scale=global_f0_scale,
                init_phase=prev_init_phase,
                phase_anchor_idx=ctx_left,
                return_phase=True,
            )
            conv_full = conv_full.squeeze(0)

            valid_len = end - start
            conv = conv_full[:, ctx_left : ctx_left + valid_len]
            if end < total_len and (ctx_left + hop_chunk) < phase_full.shape[-1]:
                prev_init_phase = phase_full[:, :, ctx_left + hop_chunk : ctx_left + hop_chunk + 1].clone()

            w = torch.ones(1, valid_len, device=device)
            if idx_c > 0 and valid_len >= xfade_samples:
                w[:, :xfade_samples] = fade_in
            if end < total_len and valid_len >= xfade_samples:
                w[:, -xfade_samples:] = fade_out

            out_buf[:, start:end] += conv * w
            weight_buf[:, start:end] += w

        out_wav = out_buf[:, :total_len] / weight_buf[:, :total_len].clamp(min=1e-6)

    save_audio_stereo(args.output, out_wav, sr=cfg.sample_rate)
    print(f"[Saved] Converted 44.1 kHz stereo singing voice written to '{args.output}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native Stereo Singing Voice Conversion (44.1 kHz, RoPE on Q/K/V + Exclusive Self Attention + GEGLU + Learned Sparse Polyphonic RoFormer Pitch Tracker + Prodigy)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train on a folder of WAV/FLAC files or a single file")
    mode.add_argument("--infer", action="store_true", help="Convert a WAV/FLAC file using the latest checkpoint")

    parser.add_argument("--data_dir", type=str, default="./singer_data", help="Folder of WAV/FLAC files or path to a single WAV/FLAC file")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Checkpoint directory for auto-save and auto-resume")
    parser.add_argument("--input", type=str, default=None, help="Input source singer WAV/FLAC file for --infer")
    parser.add_argument("--output", type=str, default="./converted_output.wav", help="Output converted stereo WAV/FLAC file for --infer")

    parser.add_argument("--sample_rate", type=int, default=44100, help="Audio sample rate (default: 44100)")
    parser.add_argument("--dim", type=int, default=384, help="Transformer hidden dimension (default: 384)")
    parser.add_argument("--pitch_depth", type=int, default=4, help="RoFormer pitch tracker XSA-RoPE depth (default: 4)")
    parser.add_argument("--encoder_depth", type=int, default=4, help="Content encoder XSA-RoPE depth (default: 4)")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Decoder XSA-RoPE depth (default: 8)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--segment_frames", type=int, default=256, help="Training crop length in STFT frames (default: 256)")

    parser.add_argument("--epochs", type=int, default=10000, help="Total training epochs (default: 10000)")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional max training steps override (0 = disabled)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1.0, help="Prodigy learning rate multiplier (default: 1.0)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Prodigy decoupled weight decay (default: 0.01)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--log_every", type=int, default=5, help="Print terminal training stats every N steps")
    parser.add_argument("--save_every", type=int, default=2000, help="Auto-save checkpoint every N steps (default: 2000)")
    parser.add_argument("--keep_ckpts", "--keep_last", dest="keep_ckpts", type=int, default=3, help="Number of latest checkpoints to keep (default: 3)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and start fresh")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Additional pitch shift in semitones during inference")
    parser.add_argument("--no_auto_f0", action="store_true", help="Disable automatic median F0 register matching")
    parser.add_argument("--no_ema", action="store_true", help="Use raw weights instead of EMA weights during inference")
    parser.add_argument("--chunk_seconds", type=float, default=6.0, help="Inference chunk duration in seconds")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        train_svc(args)
    elif args.infer:
        if not args.input:
            raise ValueError("Please provide `--input <file.wav|file.flac>` when running `--infer`.")
        infer_svc(args)


if __name__ == "__main__":
    main()