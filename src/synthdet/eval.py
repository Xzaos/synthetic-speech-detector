"""Metrics and the evaluation run. Labels are 1 for fake and 0 for real, and a
higher score means more likely fake. Bootstrap intervals resample clips within
each class, so both classes are always present in a resample.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def roc_points(scores, labels):
    """False-positive and false-negative rates at each distinct threshold
    (predict fake when score >= threshold), plus the thresholds."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n_fake, n_real = y.sum(), (1 - y).sum()
    if n_fake == 0 or n_real == 0:
        raise ValueError("need both classes")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    thr = s[last]
    fpr = np.r_[0.0, fp / n_real]
    fnr = np.r_[1.0, 1 - tp / n_fake]
    return fpr, fnr, np.r_[np.inf, thr]


def eer_and_threshold(scores, labels):
    fpr, fnr, thr = roc_points(scores, labels)
    d = fpr - fnr
    i = int(np.argmax(d >= 0))
    if d[i] == 0 or i == 0:
        return float(fpr[i]), float(thr[i])
    t = -d[i - 1] / (d[i] - d[i - 1])
    eer = fpr[i - 1] + t * (fpr[i] - fpr[i - 1])
    j = i if abs(d[i]) <= abs(d[i - 1]) else i - 1
    return float(eer), float(thr[j])


def eer(scores, labels) -> float:
    return eer_and_threshold(scores, labels)[0]


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("need both classes")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(metric, scores, labels, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        vals[b] = metric(s[idx], y[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"value": float(metric(s, y)), "lo": float(lo), "hi": float(hi)}


def false_accept_rate(fake_scores, threshold: float) -> float:
    """Share of fake clips scored below the threshold, i.e. passed as real."""
    return float(np.mean(np.asarray(fake_scores) < threshold))


def false_reject_rate(real_scores, threshold: float) -> float:
    return float(np.mean(np.asarray(real_scores) >= threshold))


CONDITIONS = [
    ("seen_clean", "mms", False),
    ("unseen_clean", "bulbul", False),
    ("seen_channel", "mms", True),
    ("unseen_channel", "bulbul", True),
]


def score_condition(model, real, fake, channel: bool, n_boot: int):
    """model(items, channel) -> scores. real and fake are lists of items."""
    s_real, s_fake = model(real, channel), model(fake, channel)
    scores = np.r_[s_real, s_fake]
    labels = np.r_[np.zeros(len(s_real), int), np.ones(len(s_fake), int)]
    return {
        "n_real": len(s_real),
        "n_fake": len(s_fake),
        "eer": bootstrap_ci(eer, scores, labels, n_boot),
        "auc": bootstrap_ci(auc, scores, labels, n_boot),
        "real_scores": s_real,
        "fake_scores": s_fake,
    }


def evaluate_model(name, model, val_threshold, real, fakes, n_boot: int):
    out = {"threshold_at_val_eer": val_threshold, "conditions": {}}
    for cond, gen, channel in CONDITIONS:
        r = score_condition(model, real, fakes[gen], channel, n_boot)
        r["frr_real"] = false_reject_rate(r["real_scores"], val_threshold)
        r["far_fake"] = false_accept_rate(r["fake_scores"], val_threshold)
        r["real_scores"], r["fake_scores"] = r["real_scores"].tolist(), r["fake_scores"].tolist()
        out["conditions"][cond] = r
    return out


def fmt_ci(d, pct=False) -> str:
    k = 100 if pct else 1
    return f"{d['value'] * k:.3f} [{d['lo'] * k:.3f}, {d['hi'] * k:.3f}]"


def leak_flag(baseline: dict, auc_cut: float = 0.8) -> bool:
    """The baseline uses no spectro-temporal detail. If it already separates
    MMS from real, the task leaks shortcuts."""
    return baseline["conditions"]["seen_clean"]["auc"]["value"] >= auc_cut


def markdown_report(rep: dict) -> str:
    L = ["# Results", ""]
    meta = rep["meta"]
    L.append(f"Clips: {meta['n_train_pairs']} train sentences, {meta['n_val_pairs']} validation sentences, {meta['n_test']} test sentences.")
    L.append(f"Detector parameters: {meta['params']}. Epochs run: {meta['epochs_run']}, best epoch {meta['best_epoch']}, train time {meta['train_seconds']:.0f} s.")
    L += ["", "EER and ROC AUC (real vs fake) with 95% bootstrap intervals over clips.", ""]
    L += ["| model | condition | EER | AUC |", "|---|---|---|---|"]
    for m in ("baseline", "detector"):
        for cond, _, _ in CONDITIONS:
            c = rep[m]["conditions"][cond]
            L.append(f"| {m} | {cond} | {fmt_ci(c['eer'])} | {fmt_ci(c['auc'])} |")
    L += ["", "Threshold set at the validation EER point (clean validation clips, MMS fakes). Error rates on test clips at that threshold:", ""]
    L += ["| model | threshold | condition | false-accept rate (fake passed as real) | false-reject rate (real flagged) |", "|---|---|---|---|---|"]
    for m in ("baseline", "detector"):
        for cond, _, _ in CONDITIONS:
            c = rep[m]["conditions"][cond]
            L.append(f"| {m} | {rep[m]['threshold_at_val_eer']:.3f} | {cond} | {c['far_fake']:.3f} | {c['frr_real']:.3f} |")
    L += ["", f"Baseline leak check (seen_clean AUC >= 0.8): {'LEAKS' if rep['baseline_leaks'] else 'does not leak'}."]
    return "\n".join(L) + "\n"


def write_report(rep: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(rep))
    (out / "report.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    (out / "report.md").write_text(markdown_report(rep), encoding="utf-8")


def clip_summary(clips) -> dict:
    d = np.array([len(c["wave"]) / 16000 for c in clips])
    r = np.array([c["raw_rms"] for c in clips])
    return {"n": len(clips), "mean_duration_s": float(d.mean()), "mean_raw_rms": float(r.mean())}


def run_eval(data_dir: str, results_dir: str, n_boot: int = 1000) -> dict:
    import torch

    from .data import fake_items, load_clips, real_items, split_train_val, text_overlap, with_channel
    from .model import Detector
    from .train import baseline_scores, detector_scores

    res = Path(results_dir)
    base = json.loads((res / "baseline.json").read_text(encoding="utf-8"))
    tmeta = json.loads((res / "train_meta.json").read_text(encoding="utf-8"))
    train_real = real_items(data_dir, "train")
    train_rows, val_rows = split_train_val(train_real)
    real_rows = real_items(data_dir, "test")
    sets = {
        "real": load_clips(real_rows),
        "mms": load_clips(fake_items(data_dir, "mms", "test", real_rows)),
        "bulbul": load_clips(fake_items(data_dir, "bulbul", "test", real_rows)),
    }
    chan = {}

    def get(clips, channel):
        if not channel:
            return clips
        if id(clips) not in chan:
            chan[id(clips)] = with_channel(clips)
        return chan[id(clips)]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Detector().to(dev)
    net.load_state_dict(torch.load(res / "detector.pt", map_location=dev))
    models = {
        "baseline": (lambda clips, ch: baseline_scores(base, get(clips, ch)), base["threshold"]),
        "detector": (lambda clips, ch: detector_scores(net, get(clips, ch), dev), tmeta["threshold"]),
    }
    rep = {"meta": {
        "n_train_pairs": len(train_rows), "n_val_pairs": len(val_rows), "n_test": len(real_rows),
        "params": tmeta["params"], "epochs_run": tmeta["epochs_run"], "best_epoch": tmeta["best_epoch"],
        "train_seconds": tmeta["train_seconds"], "val_eer_detector": tmeta["val_eer"], "val_eer_baseline": base["val_eer"],
        "test_sentences_also_in_train": text_overlap(real_rows, train_real),
        "clip_summary": {k: clip_summary(v) for k, v in sets.items()},
        "bootstrap_resamples": n_boot,
    }}
    for name, (fn, thr) in models.items():
        rep[name] = evaluate_model(name, fn, thr, sets["real"], {"mms": sets["mms"], "bulbul": sets["bulbul"]}, n_boot)
    rep["baseline_leaks"] = leak_flag(rep["baseline"])
    write_report(rep, results_dir)
    print(markdown_report(rep))
    return rep
