import numpy as np
import pytest
import torch

from synthdet import features as F

SR = F.SR


def tone(sec=1.0, f=440.0, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def band_energy(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return spec[(freqs >= lo) & (freqs < hi)].sum()


def test_trim_removes_silent_edges():
    x = np.concatenate([np.zeros(SR), tone(0.5), np.zeros(SR)])
    y = F.trim_silence(x)
    assert 0.5 * SR <= len(y) <= 0.5 * SR + 800


def test_trim_rejects_tiny_clip():
    with pytest.raises(ValueError):
        F.trim_silence(np.zeros(100, np.float32))


def test_rms_normalize_hits_target_and_rejects_silence():
    y = F.rms_normalize(tone(amp=0.01))
    assert F.rms(y) == pytest.approx(F.TARGET_RMS, rel=1e-3)
    with pytest.raises(ValueError):
        F.rms_normalize(np.zeros(SR, np.float32))


def test_preprocess_resamples_and_reports_raw_level():
    x = tone(1.0, amp=0.2)
    x8 = F.resample(x, SR, 8000)
    y, raw = F.preprocess(x8, 8000)
    assert abs(len(y) - SR) < 400
    assert raw == pytest.approx(0.2 / np.sqrt(2), rel=0.1)
    assert F.rms(y) == pytest.approx(F.TARGET_RMS, rel=1e-2)


def test_crops_have_fixed_length_and_pad_short_clips():
    short = tone(1.0)
    assert len(F.center_crop(short)) == F.CROP
    assert len(F.random_crop(short, np.random.default_rng(0))) == F.CROP
    long = np.arange(10 * SR, dtype=np.float32)
    c = F.center_crop(long)
    assert c[0] == (len(long) - F.CROP) // 2


def test_random_crop_is_a_window_of_the_clip_and_varies():
    long = np.arange(10 * SR, dtype=np.float32)
    rng = np.random.default_rng(0)
    starts = {int(F.random_crop(long, rng)[0]) for _ in range(20)}
    assert len(starts) > 5
    assert max(starts) <= len(long) - F.CROP


def test_pad_wrap_repeats_signal_instead_of_zeros():
    y = F.pad_wrap(tone(1.0))
    assert np.abs(y[-SR:]).max() > 0.1


def test_phone_channel_removes_out_of_band_energy_and_keeps_level():
    x = np.random.default_rng(1).normal(0, 0.2, SR).astype(np.float32)
    y = F.phone_channel(x)
    assert y.shape == x.shape
    assert band_energy(y, 4200, 8000) < 0.01 * band_energy(y, 300, 3400)
    assert band_energy(y, 0, 200) < 0.05 * band_energy(y, 300, 3400)
    assert F.rms(y) == pytest.approx(F.TARGET_RMS, rel=1e-2)


def test_mu_law_is_quantised_and_close():
    x = np.linspace(-0.9, 0.9, 5000).astype(np.float32)
    y = F.mu_law(x)
    assert len(np.unique(y)) <= 256
    assert np.abs(x - y).max() < 0.05


def test_log_mel_shape_and_per_clip_normalisation():
    noise = np.random.default_rng(0).normal(0, 0.1, F.CROP).astype(np.float32)
    w = torch.from_numpy(np.stack([noise, 5 * noise]))
    m = F.log_mel(w)
    assert m.shape == (2, 1, F.N_MELS, 1 + F.CROP // 160)
    assert torch.allclose(m.mean(dim=(1, 2, 3)), torch.zeros(2), atol=1e-4)
    assert torch.allclose(m[0], m[1], atol=1e-2)


def test_spec_augment_masks_and_keeps_shape():
    m = torch.ones(4, 1, 64, 301)
    out = F.spec_augment(m, generator=torch.Generator().manual_seed(0))
    assert out.shape == m.shape
    assert (out == 0).any() and (out == 1).any()
    assert (m == 1).all()


def test_utterance_stats_separate_low_and_high_tones():
    lo = F.utterance_stats(tone(2.0, 300), 0.1)
    hi = F.utterance_stats(tone(2.0, 3000), 0.1)
    names = F.STAT_NAMES
    assert len(lo) == len(names)
    assert lo[names.index("duration")] == pytest.approx(2.0)
    assert hi[names.index("centroid_mean")] > 5 * lo[names.index("centroid_mean")]
    assert hi[names.index("zcr")] > lo[names.index("zcr")]
