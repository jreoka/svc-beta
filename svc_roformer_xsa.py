#!/usr/bin/env python3
"""
Standalone Native Stereo Singing Voice Conversion (SVC) in Pure PyTorch (44.1 kHz)
==================================================================================
Zero pretrained weights or external voice encoders (no HuBERT, ContentVec, RMVPE, or HiFi-GAN).
Everything is trained from scratch using only standard Python + PyTorch audio/math primitives,
adapted to train effectively on as little as a single WAV/FLAC file or a folder of files.

Key Architectural Features:
  1. Native 44.1 kHz Stereo (2-Channel Left/Right + Mid/Side) Processing:
     - Loads and saves native 2-channel stereo WAV/FLAC files (`[2, L]`), automatically
       upmixing mono files if encountered.
     - Extracts stereo Log-Mel features (`L`, `R`, `Mid`, `Side`), joint stereo harmonic-comb +
       autocorrelation pitch salience maps, and per-channel + Mid/Side energy envelopes.
     - Synthesizes phase-locked stereo harmonics (`[B, 2, K, L]`), independent stereo filtered
       noise, and per-channel complex STFT residual masks (`[B, 2, F, T]`), supervised with
       L/R + zero-extra-STFT Mid/Side multi-resolution loss.
  2. Learned RoFormer Pitch & Voicing Tracker (`StereoRoFormerPitchTracker`):
     - Replaces glitchy frame-by-frame argmax pitch tracking with a dedicated neural RoFormer
       pitch and voicing network built on QKV-RoPE + Exclusive Self Attention (XSA) + GEGLU.
     - Ingests stereo Log-Mel (`L`, `R`, `Mid`, `Side`) + joint L+R autocorrelation lag salience +
       6-harmonic spectral comb salience across `n_f0_bins=240` log-spaced pitch bins (65-1100 Hz).
     - Predicts sub-cent continuous F0 via localized soft-argmax expectation + fine residual
       and continuous voicing probability (`periodicity`), trained end-to-end through the
       differentiable harmonic synthesizer, confidence-weighted harmonic salience anchoring,
       self-supervised pitch-shift equivariance, and 2nd-order temporal anti-glitch regularization.
  3. Prodigy Optimizer (Mishchenko & Defazio, 2023 - https://arxiv.org/abs/2306.06101):
     Built-in parameter-free D-Adaptation AdamW optimizer (uses `prodigyopt` if installed,
     otherwise falls back to the built-in self-contained implementation).
  4. QKVO Rotary Position Embedding (RoPE on Q, K, and V):
     Applies complex-plane rotary positional embeddings to Query, Key, AND Value projections
     before scaled dot-product attention, followed by the inverse rotation on the aggregated
     attention output so relative phase offsets remain translation-invariant.
  5. Exclusive Self Attention (XSA, Zhai 2026 - https://arxiv.org/abs/2603.09078):
     Projects each head's attention output orthogonal to the token's own value vector in fp32:
       Vn = F.normalize(V, dim=-1)
       Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
     Eliminates attention similarity bias and forces self-attention to capture exclusively
     contextual information while leaving pointwise features to the residual/GEGLU path.
  6. GEGLU Feed-Forward Network (Shazeer, 2020):
     Gated Linear Unit with exact GELU activation in every Transformer block.

Usage:
  Train on a folder of WAV/FLAC files (or a single file):
    python svc_roformer_xsa.py --train --data_dir ./singer_data --ckpt_dir ./checkpoints

  Convert a source singer WAV/FLAC using the latest checkpoint automatically:
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Optional audio I/O libraries (gracefully falls back to stdlib wave / soundfile / torchaudio)
try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# ==============================================================================
# 1. Configuration
# ==============================================================================


@dataclass
class SVCConfig:
    sample_rate: int = 44100
    audio_channels: int = 2  # Native stereo (Left / Right)
    n_fft: int = 2048
    hop_length: int = 512    # ~11.6 ms frame resolution at 44.1 kHz
    win_length: int = 2048
    n_mels: int = 128
    fmin: float = 40.0
    fmax: float = 16000.0
    f0_min: float = 65.0     # C2
    f0_max: float = 1100.0   # C6
    n_f0_bins: int = 240     # ~20.3 cents per bin across [65 Hz, 1100 Hz]
    n_harmonics: int = 64

    # Transformer Backbone (RoPE on Q, K, V + Exclusive Self Attention + GEGLU)
    dim: int = 384
    content_dim: int = 64
    pitch_encoder_depth: int = 4
    encoder_depth: int = 4
    decoder_depth: int = 8
    num_heads: int = 8
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.05
    rope_theta: float = 10000.0

    # Training Parameters
    segment_frames: int = 256  # ~2.97 seconds per training crop
    virtual_epoch_size: int = 256


# ==============================================================================
# 2. Pure PyTorch Native Stereo Audio I/O & Resampling (44.1 kHz)
# ==============================================================================


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

    data = data.reshape(-1, channels).T  # [C, L]
    return _ensure_stereo(torch.from_numpy(data)), sr


def _write_wav_stdlib_stereo(path: str, audio: torch.Tensor, sr: int) -> None:
    """Fallback pure-Python 16-bit PCM stereo WAV writer expecting `[2, L]`."""
    wav_np = _ensure_stereo(audio.detach().cpu()).clamp(-1.0, 1.0).numpy()  # [2, L]
    interleaved = wav_np.T.reshape(-1)  # [L, 2] -> [2*L]
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
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # [L, C]
            waveform = _ensure_stereo(torch.from_numpy(data.T))
        except Exception:
            waveform = None

    if waveform is None and torchaudio is not None:
        try:
            data, sr = torchaudio.load(path)  # [C, L]
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
    wav = _ensure_stereo(waveform.detach().cpu().float())  # [2, L]
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


# ==============================================================================
# 3. Prodigy Optimizer (Mishchenko & Defazio, 2023 - arXiv:2306.06101)
# ==============================================================================


class Prodigy(torch.optim.Optimizer):
    """
    Self-contained PyTorch implementation of the Prodigy optimizer:
    "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" (arXiv:2306.06101).
    Automatically adapts the global step scale `d` using D-Adaptation with EMA
    weighting (`beta3 = sqrt(beta2)`). Leave `lr=1.0`.
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


def build_prodigy_optimizer(params, lr: float = 1.0, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Uses external `prodigyopt.Prodigy` if installed, else built-in `Prodigy`."""
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


# ==============================================================================
# 4. Pure-PyTorch Stereo Acoustic Front-End (Mel Filterbank, Pitch Salience, STFT)
# ==============================================================================


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
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # [n_freqs, n_mels + 2]

    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1].clamp(min=1e-6)
    up_slopes = slopes[:, 2:] / f_diff[1:].clamp(min=1e-6)
    fb = torch.maximum(torch.zeros_like(down_slopes), torch.minimum(down_slopes, up_slopes))

    # Slaney-style area normalization
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels]).clamp(min=1e-6)
    fb = fb * enorm.unsqueeze(0)
    return fb.transpose(0, 1).contiguous()  # [n_mels, n_freqs]


class StereoAcousticFeatureExtractor(nn.Module):
    """
    Extracts aligned frame-rate stereo acoustic features in pure PyTorch at 44.1 kHz:
      - Stereo Log-Mel Spectrogram `[B, 2, T, n_mels]` (Left & Right)
      - Mid/Side Log-Mel Spectrogram `[B, 2, T, n_mels]` (Mid & Side)
      - Log-F0 Harmonic-Comb & Autocorrelation Pitch Salience Maps `[B, T, 2, n_f0_bins]`
        for the neural RoFormer Pitch Tracker, plus DP-smoothed acoustic reference F0
      - Stereo Log Frame RMS Energy `[B, T, 3]` (Left, Right, and Side/Mid balance)
    All STFT/iSTFT and spectral operations explicitly run with `autocast(enabled=False)` in `fp32`
    so PyTorch complex tensors (`torch.complex64`) never receive `bfloat16` inputs under AMP.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        window = torch.hann_window(cfg.win_length)
        self.register_buffer("window", window, persistent=False)
        fb = create_mel_filterbank(
            cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax, device=torch.device("cpu")
        )
        self.register_buffer("mel_fb", fb, persistent=False)

        # Log2-spaced F0 grid [n_f0_bins] spanning [f0_min, f0_max]
        log2_min = math.log2(cfg.f0_min)
        log2_max = math.log2(cfg.f0_max)
        log2_grid = torch.linspace(log2_min, log2_max, cfg.n_f0_bins)
        f0_grid = 2.0 ** log2_grid
        self.register_buffer("log2_f0_grid", log2_grid, persistent=False)
        self.register_buffer("f0_grid_hz", f0_grid, persistent=False)

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes complex64 STFT in fp32 with centered reflection padding.
        Accepts `[B, 2, L]` (returns `[B, 2, F, T]`) or `[N, L]` (returns `[N, F, T]`).
        """
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            if wav_f32.ndim == 3:
                B, C, L = wav_f32.shape
                flat = wav_f32.reshape(B * C, L)
                spec = self.stft(flat)
                return spec.view(B, C, spec.shape[-2], spec.shape[-1])

            return torch.stft(
                wav_f32,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=self.window.to(wav.device, dtype=torch.float32),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """
        Computes inverse STFT from complex64 spectrum in fp32.
        Accepts `[B, 2, F, T]` (returns `[B, 2, L]`) or `[N, F, T]` (returns `[N, L]`).
        """
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
        """
        Linearly interpolates `profile [B, T, N_src]` at continuous indices `positions [N_dst]`.
        Returns `[B, T, N_dst]`.
        """
        max_idx = profile.shape[-1] - 1
        pos_clamped = positions.clamp(0.0, float(max_idx))
        idx0 = pos_clamped.long()
        idx1 = (idx0 + 1).clamp(max=max_idx)
        frac = (pos_clamped - idx0.float()).view(1, 1, -1)
        v0 = profile[..., idx0]
        v1 = profile[..., idx1]
        return v0 * (1.0 - frac) + v1 * frac

    @torch.no_grad()
    def extract_pitch_salience_and_stereo_energy(
        self, wav: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes rich multi-cue pitch salience representations for the RoFormer Pitch Tracker
        alongside an octave-stabilized DP-smoothed acoustic reference on `wav [B, 2, L]`:
          1. `acf_salience [B, T, n_f0_bins]`: Joint L+R Wiener-Khinchin autocorrelation
             sampled across the log-F0 lag grid with half-lag subharmonic suppression.
          2. `comb_salience [B, T, n_f0_bins]`: 6-harmonic log-spectral comb summation minus
             inter-harmonic trough energy across the log-F0 grid.
          3. `ref_log2_f0 [B, T]`, `ref_voicing [B, T]`, `ref_conf [B, T]`: Forward-backward
             transition-smoothed acoustic pitch anchor for confidence-weighted supervision.
          4. `stereo_energy [B, T, 3]`: `(log_rms_L, log_rms_R, log_rms_Side)`.
        """
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            B, C, L = wav_f32.shape
            device = wav.device
            pad = self.cfg.n_fft // 2
            wav_pad = F.pad(wav_f32, (pad, pad), mode="reflect")  # [B, 2, L + 2*pad]
            frames = wav_pad.unfold(dimension=-1, size=self.cfg.win_length, step=self.cfg.hop_length)
            # frames: [B, 2, T, W]
            W = self.cfg.win_length
            win = self.window.to(device, dtype=torch.float32).view(1, 1, 1, W)
            win_frames = (frames - frames.mean(dim=-1, keepdim=True)) * win

            # Per-channel and Side RMS energy
            rms_lr = torch.sqrt((frames ** 2).mean(dim=-1) + 1e-8)  # [B, 2, T]
            side_frames = 0.5 * (frames[:, 0] - frames[:, 1])       # [B, T, W]
            rms_side = torch.sqrt((side_frames ** 2).mean(dim=-1, keepdim=True) + 1e-8).transpose(1, 2)  # [B, 1, T]
            stereo_energy = torch.log(torch.cat([rms_lr, rms_side], dim=1) + 1e-6).transpose(1, 2).contiguous()  # [B, T, 3]
            log_energy_mid = 0.5 * (stereo_energy[..., 0] + stereo_energy[..., 1])  # [B, T]

            # Joint L+R Wiener-Khinchin autocorrelation via zero-padded rfft
            fft_len = 2 * W
            spec = torch.fft.rfft(win_frames, n=fft_len, dim=-1)    # [B, 2, T, F_acf]
            power = (spec.real ** 2 + spec.imag ** 2).sum(dim=1)    # [B, T, F_acf]
            acf = torch.fft.irfft(power, n=fft_len, dim=-1)[..., :W]
            r0 = acf[..., 0:1].clamp(min=1e-7)
            norm_acf = (acf / r0).clamp(-1.0, 1.0)  # [B, T, W]

            f0_grid = self.f0_grid_hz.to(device, dtype=torch.float32)    # [N_bins]
            log2_grid = self.log2_f0_grid.to(device, dtype=torch.float32)  # [N_bins]

            # 1. Log-F0 Autocorrelation Salience with subharmonic (half-lag) alias suppression
            lags = float(self.cfg.sample_rate) / f0_grid  # [N_bins]
            acf_at_lag = self._interp_1d_bins(norm_acf, lags)
            acf_half_lag = self._interp_1d_bins(norm_acf, lags * 0.5)
            acf_salience = (acf_at_lag - 0.35 * F.relu(acf_half_lag)).clamp(-1.0, 1.0)  # [B, T, N_bins]

            # 2. Multi-Harmonic Spectral Comb Salience on whitened log-power spectrum
            log_pow = torch.log(power + 1e-7)  # [B, T, F_acf]
            # Smooth baseline removal (spectral whitening) so formants don't masquerade as F0
            smooth_pow = F.avg_pool1d(
                log_pow.reshape(B * log_pow.shape[1], 1, -1),
                kernel_size=17,
                stride=1,
                padding=8,
            ).view_as(log_pow)
            whitened_pow = (log_pow - smooth_pow).clamp(-4.0, 6.0)

            bin_hz = float(self.cfg.sample_rate) / float(fft_len)
            comb_salience = torch.zeros_like(acf_salience)
            n_comb_harmonics = 6
            for h_idx in range(1, n_comb_harmonics + 1):
                w_h = 1.0 / (float(h_idx) ** 0.75)
                peak_bins = (float(h_idx) * f0_grid) / bin_hz
                trough_bins = ((float(h_idx) - 0.5) * f0_grid) / bin_hz
                peak_val = self._interp_1d_bins(whitened_pow, peak_bins)
                trough_val = self._interp_1d_bins(whitened_pow, trough_bins)
                comb_salience = comb_salience + w_h * (peak_val - 0.5 * trough_val)

            comb_salience = torch.tanh(comb_salience / 4.0)  # Normalized to [-1, 1], [B, T, N_bins]

            # Combined salience map for octave-stable acoustic reference anchor
            joint_salience = 0.60 * acf_salience + 0.40 * comb_salience  # [B, T, N_bins]

            # Bidirectional exponential smoothing along time to suppress single-frame octave glitches
            T = joint_salience.shape[1]
            smoothed_sal = joint_salience.clone()
            if T > 1:
                # 1D temporal Gaussian-like conv over salience map [B*N_bins, 1, T]
                sal_t = joint_salience.permute(0, 2, 1).reshape(B * self.cfg.n_f0_bins, 1, T)
                k_time = torch.tensor([0.12, 0.23, 0.30, 0.23, 0.12], device=device, dtype=torch.float32).view(1, 1, 5)
                sal_t = F.pad(sal_t, (2, 2), mode="replicate")
                smoothed_sal = F.conv1d(sal_t, k_time).view(B, self.cfg.n_f0_bins, T).permute(0, 2, 1).contiguous()

            # Soft-argmax around peak of smoothed joint salience for sub-bin reference log2_f0
            peak_val, peak_bin = smoothed_sal.max(dim=-1)  # [B, T]
            bin_indices = torch.arange(self.cfg.n_f0_bins, device=device).view(1, 1, -1)
            local_mask = (bin_indices - peak_bin.unsqueeze(-1)).abs() <= 6
            masked_logits = torch.where(
                local_mask,
                smoothed_sal * 12.0,
                torch.full_like(smoothed_sal, -1e9),
            )
            local_probs = F.softmax(masked_logits, dim=-1)
            ref_log2_f0 = (local_probs * log2_grid.view(1, 1, -1)).sum(dim=-1)  # [B, T]

            # Voicing & confidence estimation from raw autocorrelation peak + energy envelope
            raw_acf_peak = acf_at_lag.max(dim=-1).values.clamp(0.0, 1.0)
            energy_floor = log_energy_mid.amax(dim=-1, keepdim=True) - 4.2
            energy_gate = torch.sigmoid((log_energy_mid - energy_floor) * 2.5)
            ref_voicing = ((raw_acf_peak > 0.26) & (log_energy_mid > energy_floor)).float()
            ref_conf = (raw_acf_peak * energy_gate).clamp(0.0, 1.0)
            ref_f0_hz = torch.where(ref_voicing > 0.5, 2.0 ** ref_log2_f0, torch.zeros_like(ref_log2_f0))

            return {
                "acf_salience": acf_salience,
                "comb_salience": comb_salience,
                "stereo_energy": stereo_energy,
                "ref_f0_hz": ref_f0_hz,
                "ref_log2_f0": ref_log2_f0,
                "ref_voicing": ref_voicing,
                "ref_conf": ref_conf,
            }

    def compute_stereo_log_mel(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes stereo Log-Mel and Mid/Side Log-Mel from `wav [B, 2, L]` in fp32.
        Returns:
          lr_log_mel: `[B, 2, T, n_mels]`
          ms_log_mel: `[B, 2, T, n_mels]` (Mid, Side)
          spec:       `[B, 2, F, T]` complex64 STFT
        """
        with torch.amp.autocast(device_type=wav.device.type, enabled=False):
            spec = self.stft(wav.float())  # [B, 2, F, T]
            fb = self.mel_fb.to(wav.device, dtype=torch.float32)

            mag_lr = spec.abs()  # [B, 2, F, T]
            mel_lr = torch.matmul(fb, mag_lr)  # [B, 2, n_mels, T]
            lr_log_mel = torch.log(mel_lr.clamp(min=1e-5)).transpose(-1, -2).contiguous()  # [B, 2, T, n_mels]

            # Zero-extra-STFT Mid/Side complex spectra derived algebraically
            spec_mid = 0.5 * (spec[:, 0] + spec[:, 1])
            spec_side = 0.5 * (spec[:, 0] - spec[:, 1])
            mag_ms = torch.stack([spec_mid.abs(), spec_side.abs()], dim=1)  # [B, 2, F, T]
            mel_ms = torch.matmul(fb, mag_ms)
            ms_log_mel = torch.log(mel_ms.clamp(min=1e-5)).transpose(-1, -2).contiguous()  # [B, 2, T, n_mels]

            return lr_log_mel, ms_log_mel, spec

    @torch.no_grad()
    def augment_mel_for_content(self, lr_log_mel: torch.Tensor) -> torch.Tensor:
        """
        Applies coherent random piecewise frequency warping (formant perturbation),
        spectral tilt perturbation, and random stereo pan perturbation across `[B, 2, T, M]`
        so the Content Encoder learns speaker-invariant, formant-independent linguistic codes.
        """
        with torch.amp.autocast(device_type=lr_log_mel.device.type, enabled=False):
            mel_f32 = lr_log_mel.float()
            B, C, T, M = mel_f32.shape
            device = mel_f32.device

            warp_alpha = torch.empty(B, 1, 1, device=device).uniform_(0.84, 1.16)
            base_grid = torch.linspace(0.0, 1.0, M, device=device).view(1, 1, M)
            warped_grid = torch.pow(base_grid.clamp(min=1e-6), warp_alpha) * 2.0 - 1.0
            time_grid = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1).expand(B, T, M)
            sample_grid = torch.stack([warped_grid.expand(B, T, M), time_grid], dim=-1)

            # Apply same frequency warp to both L and R channels
            flat_x = mel_f32.reshape(B * C, 1, T, M)
            flat_grid = sample_grid.unsqueeze(1).expand(B, C, T, M, 2).reshape(B * C, T, M, 2)
            warped = F.grid_sample(
                flat_x, flat_grid, mode="bilinear", padding_mode="border", align_corners=True
            ).view(B, C, T, M)

            # Random smooth spectral envelope / EQ perturbation + subtle L/R balance offset
            knots = torch.randn(B, 1, 1, 6, device=device) * 0.35
            eq_curve = F.interpolate(knots.view(B, 1, 6), size=M, mode="linear", align_corners=False).view(B, 1, 1, M)
            pan_jitter = torch.randn(B, C, 1, 1, device=device) * 0.15
            return warped + eq_curve + pan_jitter


# ==============================================================================
# 5. RoPE on Q, K, and V + Exclusive Self Attention (XSA) + GEGLU Transformer
# ==============================================================================


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
            freqs = torch.outer(t, self.inv_freq.to(device))  # [T, D/2]
            self._cos_cache = torch.cos(freqs)
            self._sin_cache = torch.sin(freqs)
            self._cache_len = seq_len
        return (
            self._cos_cache[:seq_len].to(dtype=dtype).view(1, 1, seq_len, -1),
            self._sin_cache[:seq_len].to(dtype=dtype).view(1, 1, seq_len, -1),
        )

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Rotates even/odd pairs `[..., D]` by `+theta` (or `-theta` when `inverse=True`)."""
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
        """Applies forward RoPE rotation to all three of Q, K, and V `[B, H, T, D_h]`."""
        seq_len = q.shape[2]
        cos, sin = self._update_cache(seq_len, q.device, q.dtype)
        q_rot = self.rotate(q, cos, sin, inverse=False)
        k_rot = self.rotate(k, cos, sin, inverse=False)
        v_rot = self.rotate(v, cos, sin, inverse=False)
        return q_rot, k_rot, v_rot, cos, sin


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization in fp32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm_x * self.weight.float()).to(x.dtype)


class GEGLUFFN(nn.Module):
    """
    GEGLU Feed-Forward Network (Shazeer 2020):
      GEGLU(x) = (GELU(x W_gate) * (x W_val)) W_out
    """

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
    """
    Multi-Head Self-Attention combining:
      1. RoPE rotational information on Q, K, and V (with inverse rotation after aggregation).
      2. Exclusive Self Attention (XSA, Zhai 2026 - https://arxiv.org/abs/2603.09078):
         Projects attention output `Y` onto the subspace orthogonal to the token's own
         value vector `V`:
             Vn = F.normalize(V, dim=-1)
             Z  = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
    """

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
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D_h]

        q = self.q_norm(q).transpose(1, 2)  # [B, H, T, D_h]
        k = self.k_norm(k).transpose(1, 2)  # [B, H, T, D_h]
        v = v.transpose(1, 2)               # [B, H, T, D_h]

        # Apply RoPE rotational embedding to Q, K, AND V
        q_rot, k_rot, v_rot, cos, sin = self.rope.forward_qkv(q, k, v)

        # Scaled dot-product attention on rotated Q, K, V
        y_rot = F.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v_rot,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        # Inverse RoPE rotation on attention output back to token-local frame
        y = self.rope.rotate(y_rot, cos, sin, inverse=True)

        # Exclusive Self Attention (XSA, arXiv:2603.09078 Algorithm 1):
        # Remove component of attention output `y` parallel to self value vector `v` in fp32
        y_f32 = y.float()
        v_unit = F.normalize(v.float(), p=2.0, dim=-1, eps=1e-6)
        proj_self = (y_f32 * v_unit).sum(dim=-1, keepdim=True) * v_unit
        z = (y_f32 - proj_self).to(x.dtype)

        out = z.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class XSARoPETransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with QKV-RoPE Exclusive Self Attention + GEGLU FFN."""

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


# ==============================================================================
# 6. Neural RoFormer Pitch & Voicing Tracker + Native Stereo SVC Architecture
# ==============================================================================


class StereoRoFormerPitchTracker(nn.Module):
    """
    Learned Neural RoFormer Pitch & Voicing Tracker (`XSARoPETransformerBlock` backbone).
    Eliminates the octave jumps, voicing dropouts, and jitter of raw autocorrelation by learning
    continuous F0 trajectories and voicing confidence directly with the RoFormer architecture:
      - Ingests stereo Log-Mel (`L`, `R`, `Mid`, `Side`), joint autocorrelation pitch-lag salience,
        6-harmonic spectral comb salience across `n_f0_bins=240` log-F0 bins, and stereo RMS energy.
      - Uses a residual depthwise temporal conv stem + stack of QKV-RoPE Exclusive Self-Attention
        (XSA) + GEGLU Transformer blocks so every frame's pitch decision attends to surrounding
        phrase context (vibrato, glides, consonant bridges) while XSA prevents frame-local noise
        from dominating the attention output.
      - Decodes sub-cent continuous `log2_f0` via localized soft-argmax expectation over the
        `n_f0_bins` distribution plus a fine sub-bin residual head and a learned temporal
        smoothing kernel, keeping `f0_hz_cont` and `periodicity` 100% differentiable for end-to-end
        backpropagation from the harmonic synthesizer and multi-resolution STFT loss.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 4 * cfg.n_mels + 2 * cfg.n_f0_bins + 3
        self.in_norm = RMSNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, cfg.dim)

        # Local depthwise conv stem to capture short-term pitch velocity & vibrato phase
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

        # Pitch bin logits head (initialized with a direct residual skip from acoustic salience)
        self.f0_bins_head = nn.Linear(cfg.dim, cfg.n_f0_bins)
        self.salience_skip_gain = nn.Parameter(torch.tensor(2.5))

        # Sub-bin fine pitch offset (in fractions of a bin width) + continuous voicing logit
        self.fine_and_voicing_head = nn.Linear(cfg.dim, 2)

        # Log2-spaced F0 grid buffer [n_f0_bins]
        log2_min = math.log2(cfg.f0_min)
        log2_max = math.log2(cfg.f0_max)
        log2_grid = torch.linspace(log2_min, log2_max, cfg.n_f0_bins)
        self.register_buffer("log2_f0_grid", log2_grid, persistent=False)
        self.bin_step_log2 = (log2_max - log2_min) / max(1, cfg.n_f0_bins - 1)

    def forward(
        self,
        lr_log_mel: torch.Tensor,
        ms_log_mel: torch.Tensor,
        acf_salience: torch.Tensor,
        comb_salience: torch.Tensor,
        stereo_energy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          lr_log_mel:    `[B, 2, T, n_mels]`
          ms_log_mel:    `[B, 2, T, n_mels]`
          acf_salience:  `[B, T, n_f0_bins]`
          comb_salience: `[B, T, n_f0_bins]`
          stereo_energy: `[B, T, 3]`
        Returns dict with:
          f0_hz_cont:    `[B, T]` continuous differentiable F0 in Hz (never zeroed, for synth & conditioner)
          f0_hz:         `[B, T]` voiced-gated F0 in Hz (0.0 on unvoiced frames)
          periodicity:   `[B, T]` continuous voicing probability in [0, 1]
          voicing_logit: `[B, T]` raw voicing logit
          pred_log2_f0:  `[B, T]` continuous log2(F0)
          f0_logits:     `[B, T, n_f0_bins]` pitch distribution logits
        """
        mel_l = lr_log_mel[:, 0]
        mel_r = lr_log_mel[:, 1]
        mel_mid = ms_log_mel[:, 0]
        mel_side = ms_log_mel[:, 1]

        feat = torch.cat(
            [mel_l, mel_r, mel_mid, mel_side, acf_salience, comb_salience, stereo_energy],
            dim=-1,
        )  # [B, T, 4*M + 2*N_bins + 3]

        x = self.in_proj(self.in_norm(feat))
        x = x + self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        h = self.out_norm(x)

        # Combine RoFormer contextual pitch logits with acoustic harmonic-comb + ACF prior
        acoustic_prior = 0.60 * acf_salience.float() + 0.40 * comb_salience.float()
        f0_logits = self.f0_bins_head(h).float() + self.salience_skip_gain.float() * acoustic_prior

        fine_voicing = self.fine_and_voicing_head(h).float()
        fine_offset = 0.5 * self.bin_step_log2 * torch.tanh(fine_voicing[..., 0])  # [B, T]
        voicing_logit = fine_voicing[..., 1]                                       # [B, T]
        periodicity = torch.sigmoid(voicing_logit)                                 # [B, T]

        # Differentiable localized soft-argmax expectation over log2(F0) bins:
        # Restricts expectation to a +/- 9 bin (~180 cents) neighborhood around the modal bin
        # so distant subharmonic/octave modes never pull the expected pitch into an in-between value,
        # while maintaining full softmax gradient flow!
        log2_grid = self.log2_f0_grid.to(f0_logits.device, dtype=torch.float32).view(1, 1, -1)
        full_probs = F.softmax(f0_logits, dim=-1)

        with torch.no_grad():
            peak_bin = f0_logits.argmax(dim=-1, keepdim=True)  # [B, T, 1]
            bin_idx = torch.arange(self.cfg.n_f0_bins, device=f0_logits.device).view(1, 1, -1)
            window_mask = ((bin_idx - peak_bin).abs() <= 9).float()

        # Straight-through localized probability normalization
        masked_probs = full_probs * window_mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-7)
        local_probs = full_probs + (masked_probs - full_probs).detach()

        raw_log2_f0 = (local_probs * log2_grid).sum(dim=-1) + fine_offset  # [B, T]

        # Differentiable 3-tap temporal smoothing on log2_f0 to eliminate frame-level micro-glitches
        if raw_log2_f0.shape[1] >= 3:
            pad_log2 = F.pad(raw_log2_f0.unsqueeze(1), (1, 1), mode="replicate")
            smooth_kernel = torch.tensor([0.20, 0.60, 0.20], device=raw_log2_f0.device, dtype=torch.float32).view(1, 1, 3)
            pred_log2_f0 = F.conv1d(pad_log2, smooth_kernel).squeeze(1)
        else:
            pred_log2_f0 = raw_log2_f0

        pred_log2_f0 = pred_log2_f0.clamp(math.log2(self.cfg.f0_min), math.log2(self.cfg.f0_max))
        f0_hz_cont = 2.0 ** pred_log2_f0
        f0_hz = torch.where(periodicity > 0.35, f0_hz_cont, torch.zeros_like(f0_hz_cont))

        return {
            "f0_hz_cont": f0_hz_cont,
            "f0_hz": f0_hz,
            "periodicity": periodicity,
            "voicing_logit": voicing_logit,
            "pred_log2_f0": pred_log2_f0,
            "f0_logits": f0_logits,
        }


class StereoContentEncoder(nn.Module):
    """
    Extracts speaker-independent phonetic/articulation content codes from stereo Log-Mel frames.
    Applies per-channel InstanceNorm1d across time to strip global channel/formant bias,
    projects concatenated Left + Right + Mid Mel features into `cfg.dim`, and processes
    through XSA-RoPE Transformer blocks into an L2-normalized bottleneck.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.in_norm = nn.InstanceNorm1d(cfg.n_mels * 3, affine=False)
        self.in_proj = nn.Linear(cfg.n_mels * 3, cfg.dim)
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

    def forward(self, lr_log_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
          lr_log_mel: `[B, 2, T, n_mels]`
        Returns:
          z: `[B, T, content_dim]`
        """
        mel_l = lr_log_mel[:, 0]
        mel_r = lr_log_mel[:, 1]
        mel_mid = 0.5 * (mel_l + mel_r)
        x_cat = torch.cat([mel_l, mel_r, mel_mid], dim=-1)  # [B, T, 3 * n_mels]

        x = self.in_norm(x_cat.float().transpose(1, 2)).transpose(1, 2)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        z = self.bottleneck(self.out_norm(x))
        return F.normalize(z.float(), p=2.0, dim=-1).to(x.dtype)


class StereoPitchEnergyConditioner(nn.Module):
    """
    Continuous Fourier-feature + MLP embedding for differentiable `f0_hz_cont`,
    learned `periodicity` (voicing probability), and stereo L/R/Side energy.
    """

    def __init__(self, dim: int):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, math.log(64.0), 32))
        self.register_buffer("freqs", freqs, persistent=False)
        # 32 sin + 32 cos for log_f0 + log_f0 (1) + periodicity (1) + stereo_energy (3) = 69
        self.mlp = nn.Sequential(
            nn.Linear(69, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        f0_hz_cont: torch.Tensor,
        periodicity: torch.Tensor,
        stereo_energy: torch.Tensor,
    ) -> torch.Tensor:
        log_f0 = torch.log2(f0_hz_cont.float().clamp(min=1.0) / 220.0)
        p_weight = periodicity.float().unsqueeze(-1)
        angles = log_f0.unsqueeze(-1) * self.freqs.to(f0_hz_cont.device).view(1, 1, -1) * math.pi
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) * p_weight
        feat = torch.cat(
            [
                fourier,
                log_f0.unsqueeze(-1) * p_weight,
                periodicity.float().unsqueeze(-1),
                stereo_energy.float() / 5.0,
            ],
            dim=-1,
        )
        return self.mlp(feat)


class StereoHarmonicPlusNoiseSynthesizer(nn.Module):
    """
    Native Stereo Differentiable Harmonic-plus-Noise + Per-Channel Complex STFT Residual Synthesizer.
    Generates:
      - Phase-locked stereo harmonic excitation at differentiable `f0_hz_cont` modulated smoothly
        by learned `periodicity`, with per-channel (L/R) harmonic amplitudes and subtle interaural
        harmonic phase offsets (`[-pi/4, pi/4]`) for natural stereo vocal width.
      - Independent stereo filtered aperiodic noise (`[B, 2, F, T]`).
      - Per-channel complex STFT residual masks (`[B, 2, F, T]`) for crisp stereo consonants,
        formants, and spatial imaging.
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        n_freqs = cfg.n_fft // 2 + 1
        self.n_freqs = n_freqs
        K = cfg.n_harmonics

        # Stereo harmonic head: 2 channels * (1 overall gain + K harmonic amplitudes) + K interaural phase offsets
        self.harm_head = nn.Linear(cfg.dim, 2 * (K + 1) + K)
        # Stereo noise filter: 2 channels * n_freqs
        self.noise_filter_head = nn.Linear(cfg.dim, 2 * n_freqs)
        # Per-channel complex STFT residual mask: 2 channels * 2 (real/imag) * n_freqs
        self.complex_mask_head = nn.Linear(cfg.dim, 2 * 2 * n_freqs)
        # Auxiliary stereo Log-Mel predictor: 2 channels * n_mels
        self.mel_pred_head = nn.Linear(cfg.dim, 2 * cfg.n_mels)

    def forward(
        self,
        h: torch.Tensor,
        f0_hz_cont: torch.Tensor,
        periodicity: torch.Tensor,
        extractor: StereoAcousticFeatureExtractor,
        target_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          h:             Decoded hidden states `[B, T, dim]`
          f0_hz_cont:    Continuous differentiable fundamental frequency `[B, T]` in Hz
          periodicity:   Continuous differentiable voicing probability `[B, T]` in [0, 1]
          extractor:     StereoAcousticFeatureExtractor for STFT/iSTFT
          target_length: Output waveform sample count `L`
        Returns:
          pred_wav: `[B, 2, L]` (float32)
          pred_mel: `[B, 2, T, n_mels]` (float32)
        """
        B, T, _ = h.shape
        K = self.cfg.n_harmonics
        device = h.device

        # Disable AMP autocast across the DSP/STFT synthesizer so all linear projections,
        # phase accumulations, and `torch.complex(filt_real, filt_imag)` run strictly in fp32
        with torch.amp.autocast(device_type=device.type, enabled=False):
            h_f32 = h.float()
            pred_mel = (
                self.mel_pred_head(h_f32)
                .view(B, T, 2, self.cfg.n_mels)
                .permute(0, 2, 1, 3)
                .contiguous()
            )

            harm_raw = self.harm_head(h_f32)  # [B, T, 2*(K+1) + K]
            amp_part = harm_raw[..., : 2 * (K + 1)].view(B, T, 2, K + 1).permute(0, 2, 1, 3)  # [B, 2, T, K+1]
            ild_phase_part = torch.tanh(harm_raw[..., 2 * (K + 1) :]) * (math.pi / 4.0)       # [B, T, K]

            overall_gain = F.softplus(amp_part[..., :1]) * 0.15  # [B, 2, T, 1]
            harm_amps = F.softplus(amp_part[..., 1:])            # [B, 2, T, K]
            harm_amps = harm_amps / (harm_amps.sum(dim=-1, keepdim=True) + 1e-6) * overall_gain

            # Upsample continuous F0, voicing periodicity, stereo harmonic amplitudes, and interaural phase to L
            f0_up = F.interpolate(
                f0_hz_cont.float().unsqueeze(1), size=target_length, mode="linear", align_corners=False
            )  # [B, 1, L]
            voiced_sample_mask = F.interpolate(
                periodicity.float().unsqueeze(1), size=target_length, mode="linear", align_corners=False
            ).unsqueeze(1)  # [B, 1, 1, L]
            amps_up = F.interpolate(
                harm_amps.reshape(B * 2, T, K).transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).view(B, 2, K, target_length)
            phase_diff_up = F.interpolate(
                ild_phase_part.transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            )  # [B, K, L]

            # Harmonic frequencies [B, K, L]
            k_idx = torch.arange(1, K + 1, device=device, dtype=torch.float32).view(1, K, 1)
            harm_freqs = f0_up * k_idx  # [B, K, L]

            # Smooth anti-aliasing transition near Nyquist so F0 gradients remain continuous
            nyquist_hz = self.cfg.sample_rate * 0.48
            nyquist_mask = torch.sigmoid((nyquist_hz - harm_freqs) * 0.01).unsqueeze(1)  # [B, 1, K, L]
            amps_up = amps_up * nyquist_mask * voiced_sample_mask

            # Instantaneous phase integration: shared center phase +/- 0.5 * interaural phase offset
            center_phase = torch.cumsum(harm_freqs / float(self.cfg.sample_rate), dim=-1) * (2.0 * math.pi)  # [B, K, L]
            phase_l = center_phase + 0.5 * phase_diff_up
            phase_r = center_phase - 0.5 * phase_diff_up
            stereo_phase = torch.stack([phase_l, phase_r], dim=1)  # [B, 2, K, L]

            harmonic_sig = torch.sum(amps_up * torch.sin(stereo_phase), dim=2)  # [B, 2, L]

            # Stereo filtered aperiodic noise via STFT domain
            noise_env = (
                F.softplus(self.noise_filter_head(h_f32))
                .view(B, T, 2, self.n_freqs)
                .permute(0, 2, 3, 1)
                .contiguous()
                * 0.05
            )  # [B, 2, F, T]

            # Correlated Mid + independent Side white noise for realistic stereo breath/sibilance
            mid_noise = torch.randn(B, 1, target_length, device=device, dtype=torch.float32) * 0.08
            side_noise = torch.randn(B, 2, target_length, device=device, dtype=torch.float32) * 0.06
            stereo_white_noise = mid_noise + side_noise  # [B, 2, L]

            noise_spec = extractor.stft(stereo_white_noise)  # [B, 2, F, T_stft]
            T_stft = noise_spec.shape[-1]
            if noise_env.shape[-1] != T_stft:
                noise_env = F.interpolate(
                    noise_env.reshape(B * 2, self.n_freqs, -1),
                    size=T_stft,
                    mode="linear",
                    align_corners=False,
                ).view(B, 2, self.n_freqs, T_stft)
            filtered_noise_spec = noise_spec * noise_env

            # Per-channel complex STFT modulation of stereo harmonic excitation + filtered noise
            harm_spec = extractor.stft(harmonic_sig)  # [B, 2, F, T_stft]
            if harm_spec.shape[-1] != T_stft:
                T_stft = min(harm_spec.shape[-1], filtered_noise_spec.shape[-1])
                harm_spec = harm_spec[..., :T_stft]
                filtered_noise_spec = filtered_noise_spec[..., :T_stft]

            base_spec = harm_spec + filtered_noise_spec  # [B, 2, F, T_stft]

            mask_raw = (
                self.complex_mask_head(h_f32)
                .view(B, T, 2, 2 * self.n_freqs)
                .permute(0, 2, 3, 1)
                .contiguous()
            )  # [B, 2, 2*F, T]
            if mask_raw.shape[-1] != T_stft:
                mask_raw = F.interpolate(
                    mask_raw.reshape(B * 2, 2 * self.n_freqs, -1),
                    size=T_stft,
                    mode="linear",
                    align_corners=False,
                ).view(B, 2, 2 * self.n_freqs, T_stft)

            mask_real, mask_imag = mask_raw.chunk(2, dim=2)  # each [B, 2, F, T_stft]
            filt_real = (1.0 + 1.5 * torch.tanh(mask_real)).float()
            filt_imag = (1.5 * torch.tanh(mask_imag)).float()
            complex_filt = torch.complex(filt_real, filt_imag)

            out_spec = base_spec.to(torch.complex64) * complex_filt
            pred_wav = extractor.istft(out_spec, length=target_length)  # [B, 2, L]
            return pred_wav, pred_mel


class SingingVoiceConverter(nn.Module):
    """
    Full Native Stereo Singing Voice Conversion model (44.1 kHz) with:
      - StereoRoFormerPitchTracker (Learned XSA + QKV-RoPE + GEGLU Pitch & Voicing Tracker)
      - StereoContentEncoder (XSA + QKV-RoPE + GEGLU)
      - StereoPitchEnergyConditioner
      - Target Singer Decoder (XSA + QKV-RoPE + GEGLU + Learned Target Timbre Embedding)
      - StereoHarmonicPlusNoiseSynthesizer
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        self.extractor = StereoAcousticFeatureExtractor(cfg)
        self.pitch_tracker = StereoRoFormerPitchTracker(cfg)
        self.content_encoder = StereoContentEncoder(cfg)
        self.content_proj = nn.Linear(cfg.content_dim, cfg.dim)
        self.pitch_cond = StereoPitchEnergyConditioner(cfg.dim)

        # Learned target singer global timbre & stereo spatial signature embedding
        self.target_singer_emb = nn.Parameter(torch.randn(1, 1, cfg.dim) * 0.02)

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

        # Running target singer vocal range statistics (for automatic pitch adaptation in inference)
        self.register_buffer("target_f0_median", torch.tensor(220.0))
        self.register_buffer("target_f0_std", torch.tensor(50.0))
        self.register_buffer("has_f0_stats", torch.tensor(False))

    @torch.no_grad()
    def update_f0_stats(self, f0_hz: torch.Tensor) -> None:
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

    @staticmethod
    def _shift_pitch_features(
        lr_log_mel: torch.Tensor,
        ms_log_mel: torch.Tensor,
        acf_salience: torch.Tensor,
        comb_salience: torch.Tensor,
        shift_f0_bins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shifts log-F0 salience maps by `shift_f0_bins` along the pitch-bin axis and shifts
        Log-Mel spectrograms proportionally along the Mel axis for self-supervised pitch
        equivariance training of the RoFormer Pitch Tracker.
        """
        if shift_f0_bins == 0:
            return lr_log_mel, ms_log_mel, acf_salience, comb_salience

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

        shifted_acf = _roll_pad_last(acf_salience, shift_f0_bins)
        shifted_comb = _roll_pad_last(comb_salience, shift_f0_bins)
        # Approximate proportional Mel-bin shift (~0.35 Mel bins per F0 bin in vocal range)
        mel_shift = int(round(shift_f0_bins * 0.35))
        if mel_shift != 0:
            shifted_lr_mel = _roll_pad_last(lr_log_mel, mel_shift)
            shifted_ms_mel = _roll_pad_last(ms_log_mel, mel_shift)
        else:
            shifted_lr_mel = lr_log_mel
            shifted_ms_mel = ms_log_mel
        return shifted_lr_mel, shifted_ms_mel, shifted_acf, shifted_comb

    def forward_train(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single-stage end-to-end stereo training forward pass on `wav [B, 2, L]`:
          1. Extracts target stereo Log-Mel, harmonic-comb + autocorrelation pitch salience,
             and L/R/Side Energy.
          2. Runs `StereoRoFormerPitchTracker` to predict continuous differentiable `f0_hz_cont`,
             `periodicity`, `pred_log2_f0`, and `f0_logits` (plus a pitch-shifted view for
             self-supervised pitch equivariance).
          3. Creates a formant-warped & pan-jittered stereo Mel view for the Content Encoder,
             plus a clean view for content consistency loss.
          4. Decodes stereo target audio conditioned on warped content + RoFormer-predicted
             differentiable F0/Periodicity/Energy + Target Timbre.
        """
        B, C, L = wav.shape
        target_lr_mel, target_ms_mel, _ = self.extractor.compute_stereo_log_mel(wav)
        pitch_cues = self.extractor.extract_pitch_salience_and_stereo_energy(wav)

        acf_salience = pitch_cues["acf_salience"]
        comb_salience = pitch_cues["comb_salience"]
        stereo_energy = pitch_cues["stereo_energy"]
        ref_log2_f0 = pitch_cues["ref_log2_f0"]
        ref_voicing = pitch_cues["ref_voicing"]
        ref_conf = pitch_cues["ref_conf"]

        T = min(target_lr_mel.shape[2], acf_salience.shape[1])
        target_lr_mel = target_lr_mel[:, :, :T, :]
        target_ms_mel = target_ms_mel[:, :, :T, :]
        acf_salience = acf_salience[:, :T, :]
        comb_salience = comb_salience[:, :T, :]
        stereo_energy = stereo_energy[:, :T, :]
        ref_log2_f0 = ref_log2_f0[:, :T]
        ref_voicing = ref_voicing[:, :T]
        ref_conf = ref_conf[:, :T]

        # 1. Neural RoFormer Pitch & Voicing Tracking (end-to-end differentiable)
        pitch_out = self.pitch_tracker(
            target_lr_mel, target_ms_mel, acf_salience, comb_salience, stereo_energy
        )
        f0_hz_cont = pitch_out["f0_hz_cont"]
        f0_hz = pitch_out["f0_hz"]
        periodicity = pitch_out["periodicity"]
        pred_log2_f0 = pitch_out["pred_log2_f0"]
        f0_logits = pitch_out["f0_logits"]
        voicing_logit = pitch_out["voicing_logit"]

        self.update_f0_stats(f0_hz)

        # 2. Self-Supervised Pitch-Shift Equivariance branch for the RoFormer Pitch Tracker
        shift_bins = random.choice([-18, -12, -8, -5, 5, 8, 12, 18])
        sh_lr, sh_ms, sh_acf, sh_comb = self._shift_pitch_features(
            target_lr_mel, target_ms_mel, acf_salience, comb_salience, shift_bins
        )
        shifted_pitch_out = self.pitch_tracker(sh_lr, sh_ms, sh_acf, sh_comb, stereo_energy)
        shift_delta_log2 = float(shift_bins) * self.pitch_tracker.bin_step_log2

        # 3. Speaker-invariant Content Encoding
        aug_lr_mel = self.extractor.augment_mel_for_content(target_lr_mel)
        z_aug = self.content_encoder(aug_lr_mel)
        z_clean = self.content_encoder(target_lr_mel)

        # 4. Target Singer Decoding conditioned on RoFormer-learned F0 & Periodicity
        cond = (
            self.content_proj(z_aug)
            + self.pitch_cond(f0_hz_cont, periodicity, stereo_energy)
            + self.target_singer_emb
        )

        h = cond
        for blk in self.decoder_blocks:
            h = blk(h)
        h = self.decoder_norm(h)

        pred_wav, pred_lr_mel = self.synthesizer(
            h, f0_hz_cont, periodicity, self.extractor, target_length=L
        )
        return {
            "pred_wav": pred_wav,
            "pred_lr_mel": pred_lr_mel,
            "target_lr_mel": target_lr_mel,
            "target_ms_mel": target_ms_mel,
            "z_aug": z_aug,
            "z_clean": z_clean,
            "pred_log2_f0": pred_log2_f0,
            "shifted_pred_log2_f0": shifted_pitch_out["pred_log2_f0"],
            "shift_delta_log2": shift_delta_log2,
            "f0_logits": f0_logits,
            "voicing_logit": voicing_logit,
            "periodicity": periodicity,
            "ref_log2_f0": ref_log2_f0,
            "ref_voicing": ref_voicing,
            "ref_conf": ref_conf,
        }

    @torch.no_grad()
    def convert_chunk(
        self,
        source_wav: torch.Tensor,
        pitch_shift_semitones: float = 0.0,
        auto_f0_adjust: bool = True,
    ) -> torch.Tensor:
        """
        Converts a chunk of stereo source singer waveform `[1, 2, L]` into the trained target
        singer using the learned `StereoRoFormerPitchTracker` for smooth, glitch-free F0 & voicing.
        """
        B, C, L = source_wav.shape
        src_lr_mel, src_ms_mel, _ = self.extractor.compute_stereo_log_mel(source_wav)
        pitch_cues = self.extractor.extract_pitch_salience_and_stereo_energy(source_wav)

        acf_salience = pitch_cues["acf_salience"]
        comb_salience = pitch_cues["comb_salience"]
        stereo_energy = pitch_cues["stereo_energy"]

        T = min(src_lr_mel.shape[2], acf_salience.shape[1])
        src_lr_mel = src_lr_mel[:, :, :T, :]
        src_ms_mel = src_ms_mel[:, :, :T, :]
        acf_salience = acf_salience[:, :T, :]
        comb_salience = comb_salience[:, :T, :]
        stereo_energy = stereo_energy[:, :T, :]

        # Predict smooth, contextual F0 and voicing using the learned RoFormer Pitch Tracker
        pitch_out = self.pitch_tracker(
            src_lr_mel, src_ms_mel, acf_salience, comb_salience, stereo_energy
        )
        f0_hz_cont = pitch_out["f0_hz_cont"]
        periodicity = pitch_out["periodicity"]

        voiced = periodicity > 0.35
        if auto_f0_adjust and bool(self.has_f0_stats.item()) and voiced.any():
            src_med = f0_hz_cont[voiced].median().clamp(min=1.0)
            tgt_med = self.target_f0_median.clamp(min=1.0)
            f0_hz_cont = f0_hz_cont * (tgt_med / src_med)

        if abs(pitch_shift_semitones) > 1e-4:
            scale = 2.0 ** (pitch_shift_semitones / 12.0)
            f0_hz_cont = f0_hz_cont * scale

        f0_hz_cont = f0_hz_cont.clamp(self.cfg.f0_min * 0.5, self.cfg.f0_max * 1.5)

        z_content = self.content_encoder(src_lr_mel)
        h = (
            self.content_proj(z_content)
            + self.pitch_cond(f0_hz_cont, periodicity, stereo_energy)
            + self.target_singer_emb
        )
        for blk in self.decoder_blocks:
            h = blk(h)
        h = self.decoder_norm(h)

        # Smooth voicing gate at inference to suppress unvoiced harmonic bleed while keeping legato transitions
        soft_periodicity = torch.sigmoid((periodicity - 0.30) * 12.0) * periodicity
        pred_wav, _ = self.synthesizer(
            h, f0_hz_cont, soft_periodicity, self.extractor, target_length=L
        )
        return pred_wav


# ==============================================================================
# 7. Stereo Multi-Resolution STFT + Mel + RoFormer Pitch Learning Loss
# ==============================================================================


class StereoMultiResolutionSVCLoss(nn.Module):
    """
    Combines:
      1. Stereo L/R + Zero-Extra-STFT Mid/Side Multi-Resolution STFT Loss
         (Spectral Convergence + Log-Magnitude + Power-Compressed Complex)
      2. Stereo L/R + Mid/Side Log-Mel Reconstruction Loss
      3. Content Invariance Cosine Loss (enforces identical content codes under formant warping)
      4. RoFormer Pitch & Voicing Loss (`pitch_loss`):
         - Gaussian-smoothed categorical KL distribution loss + Huber log2-F0 loss weighted by
           acoustic harmonic confidence (`ref_conf`)
         - Voicing BCE loss against harmonic voicing confidence
         - Self-supervised pitch-shift equivariance loss (`shifted_pred_log2_f0` vs `pred_log2_f0 + delta`)
         - 1st and 2nd-order temporal pitch continuity / anti-glitch regularization on voiced frames
    """

    def __init__(self, cfg: SVCConfig):
        super().__init__()
        self.cfg = cfg
        self.fft_sizes = (512, 1024, 2048)
        self.hops = (128, 256, 512)
        self.wins = (512, 1024, 2048)
        for w in self.wins:
            self.register_buffer(f"win_{w}", torch.hann_window(w), persistent=False)

        log2_grid = torch.linspace(math.log2(cfg.f0_min), math.log2(cfg.f0_max), cfg.n_f0_bins)
        self.register_buffer("log2_f0_grid", log2_grid, persistent=False)
        self.bin_step_log2 = (math.log2(cfg.f0_max) - math.log2(cfg.f0_min)) / max(1, cfg.n_f0_bins - 1)

    @staticmethod
    def _spec_metrics(p_spec: torch.Tensor, t_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_mag = p_spec.abs().clamp(min=1e-6)
        t_mag = t_spec.abs().clamp(min=1e-6)

        sc = torch.norm(t_mag - p_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-6)
        mag = F.l1_loss(torch.log(p_mag), torch.log(t_mag))

        # Power-compressed complex spectrogram loss (c=0.3) using unit phasor to avoid angle() NaNs
        p_unit = p_spec / p_mag
        t_unit = t_spec / t_mag
        p_comp = (p_mag ** 0.3) * p_unit
        t_comp = (t_mag ** 0.3) * t_unit
        comp = F.l1_loss(torch.view_as_real(p_comp), torch.view_as_real(t_comp))
        return sc, mag, comp

    def _compute_roformer_pitch_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_log2_f0 = out["pred_log2_f0"].float()              # [B, T]
        shifted_pred_log2_f0 = out["shifted_pred_log2_f0"].float()
        shift_delta_log2 = float(out["shift_delta_log2"])
        f0_logits = out["f0_logits"].float()                    # [B, T, N_bins]
        voicing_logit = out["voicing_logit"].float()            # [B, T]
        periodicity = out["periodicity"].float()                # [B, T]
        ref_log2_f0 = out["ref_log2_f0"].float()                # [B, T]
        ref_voicing = out["ref_voicing"].float()                # [B, T]
        ref_conf = out["ref_conf"].float()                      # [B, T]

        # 1. Confidence-weighted Gaussian categorical distribution loss over log2_f0 bins
        log2_grid = self.log2_f0_grid.to(pred_log2_f0.device, dtype=torch.float32).view(1, 1, -1)
        sigma_log2 = 1.5 * self.bin_step_log2  # ~30 cents Gaussian blur on target bins
        target_logits = -0.5 * ((log2_grid - ref_log2_f0.unsqueeze(-1)) / sigma_log2) ** 2
        target_probs = F.softmax(target_logits, dim=-1)
        log_pred_probs = F.log_softmax(f0_logits, dim=-1)
        ce_per_frame = -(target_probs * log_pred_probs).sum(dim=-1)  # [B, T]

        voiced_weights = (ref_voicing * ref_conf).detach()
        weight_denom = voiced_weights.sum().clamp(min=1.0)
        bin_dist_loss = (ce_per_frame * voiced_weights).sum() / weight_denom

        # 2. Confidence-weighted continuous log2-F0 Huber loss
        f0_reg_per_frame = F.smooth_l1_loss(pred_log2_f0, ref_log2_f0, reduction="none", beta=0.04)
        f0_reg_loss = (f0_reg_per_frame * voiced_weights).sum() / weight_denom

        # 3. Voicing probability BCE loss against acoustic harmonic voicing target
        soft_voicing_target = (0.75 * ref_voicing + 0.25 * ref_conf).clamp(0.0, 1.0).detach()
        voicing_loss = F.binary_cross_entropy_with_logits(voicing_logit, soft_voicing_target)

        # 4. Self-Supervised Pitch-Shift Equivariance Loss (enforces exact relative pitch tracking)
        expected_shifted = (pred_log2_f0.detach() + shift_delta_log2).clamp(
            math.log2(self.cfg.f0_min), math.log2(self.cfg.f0_max)
        )
        equiv_per_frame = F.smooth_l1_loss(shifted_pred_log2_f0, expected_shifted, reduction="none", beta=0.03)
        equiv_loss = (equiv_per_frame * voiced_weights).sum() / weight_denom

        # 5. Temporal Anti-Glitch Smoothness Loss (penalizes 1st & 2nd-order pitch spikes on voiced frames)
        smooth_loss = torch.tensor(0.0, device=pred_log2_f0.device, dtype=torch.float32)
        if pred_log2_f0.shape[1] >= 3:
            v_pair = (periodicity[:, 1:] * periodicity[:, :-1]).detach()
            df1 = pred_log2_f0[:, 1:] - pred_log2_f0[:, :-1]
            # Penalize frame-to-frame jumps exceeding ~1.2 semitones (0.1 octaves per 11.6ms frame)
            vel_penalty = F.relu(df1.abs() - 0.10) ** 2
            v_triplet = (periodicity[:, 2:] * periodicity[:, 1:-1] * periodicity[:, :-2]).detach()
            df2 = pred_log2_f0[:, 2:] - 2.0 * pred_log2_f0[:, 1:-1] + pred_log2_f0[:, :-2]
            accel_penalty = df2 ** 2
            smooth_loss = (
                2.0 * (vel_penalty * v_pair).mean()
                + 1.5 * (accel_penalty * v_triplet).mean()
            )

        return 0.25 * bin_dist_loss + 2.0 * f0_reg_loss + 0.4 * voicing_loss + 1.0 * equiv_loss + smooth_loss

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
        z_aug = out["z_aug"]
        z_clean = out["z_clean"]

        with torch.amp.autocast(device_type=pred_wav.device.type, enabled=False):
            B, C, L = pred_wav.shape
            pred_f32 = pred_wav.float()
            tgt_f32 = target_wav.float()

            sc_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)
            mag_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)
            comp_loss = torch.tensor(0.0, device=pred_wav.device, dtype=torch.float32)

            for n_fft, hop, win_len in zip(self.fft_sizes, self.hops, self.wins):
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

                # L/R spectral loss
                sc_lr, mag_lr, comp_lr = self._spec_metrics(p_spec, t_spec)

                # Zero-extra-STFT Mid/Side spectral loss derived algebraically from L/R STFTs
                p_ms = torch.stack([0.5 * (p_spec[:, 0] + p_spec[:, 1]), 0.5 * (p_spec[:, 0] - p_spec[:, 1])], dim=1)
                t_ms = torch.stack([0.5 * (t_spec[:, 0] + t_spec[:, 1]), 0.5 * (t_spec[:, 0] - t_spec[:, 1])], dim=1)
                sc_ms, mag_ms, comp_ms = self._spec_metrics(p_ms, t_ms)

                sc_loss = sc_loss + 0.5 * (sc_lr + sc_ms)
                mag_loss = mag_loss + 0.5 * (mag_lr + mag_ms)
                comp_loss = comp_loss + 0.5 * (comp_lr + comp_ms)

            n_res = float(len(self.fft_sizes))
            sc_loss = sc_loss / n_res
            mag_loss = mag_loss / n_res
            comp_loss = comp_loss / n_res

            wav_lr_mel, wav_ms_mel, _ = extractor.compute_stereo_log_mel(pred_f32)
            T = min(pred_lr_mel.shape[2], target_lr_mel.shape[2], wav_lr_mel.shape[2])
            mel_loss = (
                F.l1_loss(pred_lr_mel[:, :, :T].float(), target_lr_mel[:, :, :T].float())
                + F.l1_loss(wav_lr_mel[:, :, :T].float(), target_lr_mel[:, :, :T].float())
                + 0.5 * F.l1_loss(wav_ms_mel[:, :, :T].float(), target_ms_mel[:, :, :T].float())
            )

            # Content invariance loss (1 - cosine similarity)
            content_loss = (1.0 - (z_aug.float() * z_clean.float().detach()).sum(dim=-1)).mean()

            # RoFormer Pitch Tracker loss (in addition to end-to-end STFT/Mel synth gradients)
            pitch_loss = self._compute_roformer_pitch_loss(out)

            total = (
                1.0 * mag_loss
                + 0.5 * sc_loss
                + 1.0 * comp_loss
                + 1.5 * mel_loss
                + 0.5 * content_loss
                + 1.0 * pitch_loss
            )
            stats = {
                "total": float(total.item()),
                "stft": float((mag_loss + 0.5 * sc_loss + comp_loss).item()),
                "mel": float(mel_loss.item()),
                "f0": float(pitch_loss.item()),
                "content": float(content_loss.item()),
            }
            return total, stats


# ==============================================================================
# 8. Single-File / Multi-File Stereo Singer Dataset with Dynamic Augmentation
# ==============================================================================


class SingerAudioDataset(Dataset):
    """
    Loads a single WAV/FLAC file or a folder of WAV/FLAC files into memory as native
    44.1 kHz stereo (`[2, L]`). Works seamlessly even when given just 1 audio file by
    drawing random stereo crops with gentle gain, stereo image width, and micro-pitch/speed
    perturbations across virtual epoch steps.
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
            wav = load_audio_stereo(str(f), target_sr=cfg.sample_rate)  # [2, L]
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = self.waveforms[idx % len(self.waveforms)]  # [2, L_full]
        seg_len = self.segment_samples
        full_len = wav.shape[-1]

        # Random micro speed/pitch resample (+/- 6%) 50% of the time for single-file diversity
        if random.random() < 0.5:
            rate = random.uniform(0.94, 1.06)
            req_len = int(round(seg_len * rate))
        else:
            req_len = seg_len

        if full_len <= req_len:
            repeats = (req_len // full_len) + 2
            crop = wav.repeat(1, repeats)[:, :req_len]
        else:
            start = random.randint(0, full_len - req_len)
            crop = wav[:, start : start + req_len]

        if crop.shape[-1] != seg_len:
            crop = F.interpolate(
                crop.unsqueeze(0), size=seg_len, mode="linear", align_corners=False
            ).squeeze(0)

        # Gentle stereo Mid/Side width augmentation (preserves target stereo character)
        mid = 0.5 * (crop[0] + crop[1])
        side = 0.5 * (crop[0] - crop[1]) * random.uniform(0.85, 1.15)
        crop = torch.stack([mid + side, mid - side], dim=0)

        # Random overall gain scaling
        gain = 10.0 ** (random.uniform(-6.0, 2.0) / 20.0)
        crop = (crop * gain).clamp(-0.99, 0.99)
        return crop


# ==============================================================================
# 9. Checkpoint Management (Atomic Auto-Save & Auto-Resume)
# ==============================================================================


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Finds the most recent checkpoint in `ckpt_dir` (`latest.pt` or highest step `ckpt_step_*.pt`)."""
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
) -> str:
    """Atomically writes `latest.pt`, `ckpt_step_{step:07d}.pt`, and `best.pt` if improved."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "timestamp": time.time(),
    }

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

    # Prune old numbered checkpoints beyond `keep_last`
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
) -> Tuple[Optional[SingingVoiceConverter], SVCConfig, int, int, float, Optional[str]]:
    """Loads the latest checkpoint from `ckpt_dir` if present (gracefully upgrading legacy checkpoints)."""
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        cfg = model.cfg if model is not None else SVCConfig()
        return model, cfg, 0, 0, float("inf"), None

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

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[Auto-Resume] Initialized {len(missing)} new parameter tensor(s) (e.g. RoFormer Pitch Tracker).")

    if optimizer is not None and "optimizer_state" in ckpt and not missing and not unexpected:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[Warn] Could not restore optimizer state ({e}); continuing with fresh optimizer state.")

    epoch = int(ckpt.get("epoch", 0))
    step = int(ckpt.get("step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))
    return model, cfg, epoch, step, best_loss, ckpt_path


# ==============================================================================
# 10. Live Terminal Dashboard & Training / Inference Pipelines
# ==============================================================================


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
    optimizer = build_prodigy_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = StereoMultiResolutionSVCLoss(cfg).to(device)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if not args.fresh:
        _, _, start_epoch, global_step, best_loss, loaded_path = load_checkpoint_auto(
            args.ckpt_dir, device, model=model, optimizer=optimizer
        )
        if loaded_path is not None:
            print(f"[Auto-Resume] Resumed from '{loaded_path}' at Epoch {start_epoch}, Step {global_step}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pitch_params = sum(p.numel() for p in model.pitch_tracker.parameters() if p.requires_grad)
    print("=" * 100)
    print(f" Native Stereo SVC (44.1 kHz) | RoPE-QKV + Exclusive Self-Attention (XSA) + GEGLU | Device: {device}")
    print(f" Parameters: {n_params / 1e6:.2f}M (RoFormer Pitch Tracker: {n_pitch_params / 1e6:.2f}M) | Stereo Files: {dataset.num_files} ({dataset.total_seconds:.1f}s)")
    print(f" Optimizer: Prodigy (lr={args.lr}, d0=1e-6) | Checkpoints: {args.ckpt_dir} (every {args.save_every} steps, keep latest {args.keep_ckpts})")
    print("=" * 100)

    current_epoch = start_epoch
    ema_loss: Optional[float] = None

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
            )
            if is_best:
                shutil.copy2(saved, os.path.join(args.ckpt_dir, "best.pt"))
            return saved
        finally:
            signal.signal(signal.SIGINT, prev_sig)

    # Graceful Ctrl+C emergency checkpoint saving
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

                wav_batch = wav_batch.to(device, non_blocking=True)  # [B, 2, L]
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
        # Final save on normal loop exit
        saved_path = do_save_checkpoint("done")
        print(f"[Done] Saved latest checkpoint to '{saved_path}' at step {global_step}.", flush=True)
    finally:
        pbar.close()
        signal.signal(signal.SIGINT, orig_int)
        if hasattr(signal, "SIGBREAK") and orig_break is not None:
            signal.signal(signal.SIGBREAK, orig_break)


@torch.no_grad()
def infer_svc(args: argparse.Namespace) -> None:
    """
    Loads the latest checkpoint automatically from `args.ckpt_dir` and converts
    a single input WAV/FLAC file into 44.1 kHz native stereo (`[2, L]`) with raised-cosine crossfades.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, cfg, epoch, step, best_loss, ckpt_path = load_checkpoint_auto(args.ckpt_dir, device)
    if ckpt_path is None or model is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{args.ckpt_dir}'. Train a model first with `--train --data_dir <folder_or_file>`."
        )

    model.eval()
    print(f"[Auto-Infer] Loaded checkpoint '{ckpt_path}' (step {step}, epoch {epoch})")

    src_wav = load_audio_stereo(args.input, target_sr=cfg.sample_rate).to(device)  # [2, L]
    total_len = src_wav.shape[-1]

    chunk_samples = int(math.ceil((args.chunk_seconds * cfg.sample_rate) / cfg.hop_length) * cfg.hop_length)
    overlap_samples = chunk_samples // 4

    if total_len <= chunk_samples:
        pad_rem = (cfg.hop_length - (total_len % cfg.hop_length)) % cfg.hop_length
        wav_in = F.pad(src_wav, (0, pad_rem)).unsqueeze(0)  # [1, 2, L_pad]
        out_wav = model.convert_chunk(
            wav_in,
            pitch_shift_semitones=args.pitch_shift,
            auto_f0_adjust=not args.no_auto_f0,
        ).squeeze(0)[:, :total_len]
    else:
        hop_chunk = chunk_samples - overlap_samples
        out_buf = torch.zeros(2, total_len + chunk_samples, device=device)
        weight_buf = torch.zeros(1, total_len + chunk_samples, device=device)

        # Raised-cosine crossfade window [1, chunk_samples]
        fade = torch.ones(1, chunk_samples, device=device)
        if overlap_samples > 0:
            t = torch.linspace(0.0, math.pi, overlap_samples, device=device)
            fade[:, :overlap_samples] = 0.5 * (1.0 - torch.cos(t))
            fade[:, -overlap_samples:] = 0.5 * (1.0 + torch.cos(t))

        starts = list(range(0, max(1, total_len - overlap_samples), hop_chunk))
        for start in tqdm(starts, desc="Converting", unit="chunk", dynamic_ncols=True):
            end = min(total_len, start + chunk_samples)
            seg = src_wav[:, start:end]
            cur_len = seg.shape[-1]
            if cur_len < chunk_samples:
                seg = F.pad(seg, (0, chunk_samples - cur_len))

            conv = model.convert_chunk(
                seg.unsqueeze(0),
                pitch_shift_semitones=args.pitch_shift,
                auto_f0_adjust=not args.no_auto_f0,
            ).squeeze(0)  # [2, chunk_samples]

            out_buf[:, start : start + chunk_samples] += conv * fade
            weight_buf[:, start : start + chunk_samples] += fade

        out_wav = out_buf[:, :total_len] / weight_buf[:, :total_len].clamp(min=1e-6)

    save_audio_stereo(args.output, out_wav, sr=cfg.sample_rate)
    print(f"[Saved] Converted 44.1 kHz stereo singing voice written to '{args.output}'")


# ==============================================================================
# 11. CLI Entrypoint
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native Stereo Singing Voice Conversion (44.1 kHz, RoPE on Q/K/V + Exclusive Self Attention + GEGLU + Learned RoFormer Pitch Tracker + Prodigy)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train on a folder of WAV/FLAC files or a single file")
    mode.add_argument("--infer", action="store_true", help="Convert a WAV/FLAC file using the latest checkpoint")

    # Paths
    parser.add_argument("--data_dir", type=str, default="./singer_data", help="Folder of WAV/FLAC files or path to a single WAV/FLAC file")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Checkpoint directory for auto-save and auto-resume")
    parser.add_argument("--input", type=str, default=None, help="Input source singer WAV/FLAC file for --infer")
    parser.add_argument("--output", type=str, default="./converted_output.wav", help="Output converted stereo WAV/FLAC file for --infer")

    # Model Architecture
    parser.add_argument("--sample_rate", type=int, default=44100, help="Audio sample rate (default: 44100)")
    parser.add_argument("--dim", type=int, default=384, help="Transformer hidden dimension (default: 384)")
    parser.add_argument("--pitch_depth", type=int, default=4, help="RoFormer pitch tracker XSA-RoPE depth (default: 4)")
    parser.add_argument("--encoder_depth", type=int, default=4, help="Content encoder XSA-RoPE depth (default: 4)")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Decoder XSA-RoPE depth (default: 8)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--segment_frames", type=int, default=256, help="Training crop length in STFT frames (default: 256)")

    # Training & Prodigy Hyperparameters
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

    # Inference Options
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Additional pitch shift in semitones during inference")
    parser.add_argument("--no_auto_f0", action="store_true", help="Disable automatic median F0 register matching")
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