"""Fits the two models on the MMS-vs-real training pairs: a logistic
regression on utterance statistics, and the CNN with random 3 s crops and
SpecAugment-style masking. The CNN checkpoint with the lowest validation EER
is kept, and the score threshold at that EER is saved for the evaluation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import labels, stats_matrix
from .eval import eer_and_threshold
from .features import center_crop, log_mel, random_crop, spec_augment
from .model import Detector, n_params


def fit_baseline(train_clips, val_clips) -> dict:
    from sklearn.linear_model import LogisticRegression

    x, y = stats_matrix(train_clips), labels(train_clips)
    mean, scale = x.mean(0), x.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((x - mean) / scale, y)
    b = {"mean": mean.tolist(), "scale": scale.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    e, thr = eer_and_threshold(baseline_scores(b, val_clips), labels(val_clips))
    b.update(val_eer=e, threshold=thr)
    return b


def baseline_scores(b: dict, clips) -> np.ndarray:
    z = (stats_matrix(clips) - np.array(b["mean"])) / np.array(b["scale"])
    return z @ np.array(b["coef"]) + b["intercept"]


def detector_scores(model, clips, device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(clips), batch):
            w = torch.from_numpy(np.stack([center_crop(c["wave"]) for c in clips[i : i + batch]])).to(device)
            out.append(model(log_mel(w)).float().cpu().numpy())
    return np.concatenate(out)


def train_detector(train_clips, val_clips, out_dir: str, minutes: float = 10.0, epochs: int = 120, batch: int = 32, lr: float = 2e-3, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detector().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(train_clips) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss()
    yv = labels(val_clips)
    y_all = torch.tensor(labels(train_clips), dtype=torch.float32)
    best, history, t0 = None, [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train_clips))
        loss_sum = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            w = torch.from_numpy(np.stack([random_crop(train_clips[j]["wave"], rng) for j in idx])).to(dev)
            mel = spec_augment(log_mel(w), generator=gen)
            loss = bce(model(mel), y_all[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
        s = detector_scores(model, val_clips, dev)
        e, thr = eer_and_threshold(s, yv)
        vloss = bce(torch.from_numpy(s), torch.from_numpy(yv).float()).item()
        history.append({"epoch": ep, "train_loss": loss_sum / len(perm), "val_eer": e, "val_loss": vloss})
        if best is None or (e, vloss) < (best["val_eer"], best["val_loss"]):
            best = {"epoch": ep, "val_eer": e, "val_loss": vloss, "threshold": thr}
            torch.save(model.state_dict(), Path(out_dir) / "detector.pt")
        if time.time() - t0 > minutes * 60:
            break
    meta = {**best, "params": n_params(model), "epochs_run": len(history), "best_epoch": best["epoch"], "train_seconds": time.time() - t0, "history": history}
    return meta


def train_all(train_clips, val_clips, out_dir: str, **kw) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = fit_baseline(train_clips, val_clips)
    (out / "baseline.json").write_text(json.dumps(base, indent=1), encoding="utf-8")
    print(f"baseline val EER {base['val_eer']:.3f}")
    meta = train_detector(train_clips, val_clips, out_dir, **kw)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"detector best val EER {meta['val_eer']:.3f} at epoch {meta['best_epoch']} of {meta['epochs_run']}, {meta['train_seconds']:.0f} s")
