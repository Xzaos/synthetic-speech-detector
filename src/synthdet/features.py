"""Audio preprocessing and features. Every clip, real or synthetic, goes
through the same path: 16 kHz mono, silence trimmed, RMS normalised. The
point is that duration, level and leading silence cannot tell the classes
apart, so whatever the detector learns has to come from the signal itself.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

SR = 16_000
CROP = 3 * SR
TARGET_RMS = 0.05
N_MELS = 64


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def trim_silence(x: np.ndarray, top_db: float = 35.0, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Cut leading and trailing frames more than top_db below the loudest frame."""
    if len(x) < frame:
        raise ValueError("clip shorter than one analysis frame")
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    keep = np.flatnonzero(db > db.max() - top_db)
    return x[keep[0] * hop : keep[-1] * hop + frame]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def rms_normalize(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    r = rms(x)
    if r < 1e-6:
        raise ValueError("clip is silent")
    return np.clip(x * (target / r), -1.0, 1.0).astype(np.float32)


def preprocess(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Returns the processed clip and its RMS after trimming but before
    normalisation. The raw level is only used by the cheap-stats baseline, to
    show what the normalisation step removes."""
    x = trim_silence(resample(to_mono(x), sr))
    return rms_normalize(x), rms(x)


def pad_wrap(x: np.ndarray, n: int = CROP) -> np.ndarray:
    """Short clips are repeated to length n rather than zero padded, so the
    amount of padding does not encode the clip's duration."""
    return x if len(x) >= n else np.pad(x, (0, n - len(x)), mode="wrap")


def center_crop(x: np.ndarray, n: int = CROP) -> np.ndarray:
    x = pad_wrap(x, n)
    s = (len(x) - n) // 2
    return x[s : s + n]


def random_crop(x: np.ndarray, rng: np.random.Generator, n: int = CROP) -> np.ndarray:
    x = pad_wrap(x, n)
    s = int(rng.integers(0, len(x) - n + 1))
    return x[s : s + n]


def mu_law(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """8-bit mu-law compand, quantise to 256 levels and expand again."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu)
    y = q / mu * 2 - 1
    return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)


def phone_channel(x: np.ndarray) -> np.ndarray:
    """Bandpass 300 to 3400 Hz, 8 kHz round trip, mu-law, then back to the
    original level so the channel does not reintroduce a loudness cue."""
    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    y = sosfilt(sos, x)
    y = resample_poly(y, 1, 2)
    y = mu_law(np.clip(y, -1, 1))
    y = resample_poly(y, 2, 1)[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return rms_normalize(y.astype(np.float32))


_MEL = {}


def log_mel(wave):
    """wave: float tensor (B, T). Returns (B, 1, N_MELS, frames), normalised
    per clip by a single mean and std so overall level is not a feature."""
    import torch
    import torchaudio

    key = str(wave.device)
    if key not in _MEL:
        _MEL[key] = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=512, win_length=400, hop_length=160, n_mels=N_MELS
        ).to(wave.device)
    m = torch.log(_MEL[key](wave) + 1e-6)
    mean = m.mean(dim=(1, 2), keepdim=True)
    std = m.std(dim=(1, 2), keepdim=True) + 1e-5
    return ((m - mean) / std).unsqueeze(1)


def spec_augment(mel, n_freq: int = 2, max_freq: int = 8, n_time: int = 2, max_time: int = 30, generator=None):
    """Zero (the per-clip mean after normalisation) random frequency bands and
    time spans, independently for each clip in the batch."""
    import torch

    mel = mel.clone()
    b, _, f, t = mel.shape
    for i in range(b):
        for _ in range(n_freq):
            w = int(torch.randint(0, max_freq + 1, (1,), generator=generator))
            s = int(torch.randint(0, max(1, f - w + 1), (1,), generator=generator))
            mel[i, :, s : s + w, :] = 0
        for _ in range(n_time):
            w = int(torch.randint(0, max_time + 1, (1,), generator=generator))
            s = int(torch.randint(0, max(1, t - w + 1), (1,), generator=generator))
            mel[i, :, :, s : s + w] = 0
    return mel


STAT_NAMES = [
    "duration", "rms", "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
    "bandwidth_mean", "bandwidth_std", "zcr",
]


def utterance_stats(x: np.ndarray, raw_rms: float, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Cheap per-clip statistics for the baseline. x is the processed clip
    before cropping, so duration is the trimmed length."""
    pad = np.pad(x, (0, max(0, n_fft - len(x))))
    n = 1 + (len(pad) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(pad[idx] * np.hanning(n_fft), axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    p = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)
    centroid = (p * freqs).sum(axis=1)
    bandwidth = np.sqrt((p * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1))
    cum = np.cumsum(mag, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))
    return np.array([
        len(x) / SR, raw_rms, centroid.mean(), centroid.std(), rolloff.mean(), rolloff.std(),
        bandwidth.mean(), bandwidth.std(), zcr,
    ], dtype=np.float64)
