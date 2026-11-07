"""Manifests, sentence pairing and the train/validation split.

Every fake clip is named after the FLEURS clip whose sentence it speaks, so a
pair is (real id, fake id) with the same text. Items are plain dicts: id,
audio, text, label (1 fake, 0 real), system, sent (the real clip id).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from asrbench.data import prepare_hf, read_manifest, write_jsonl

from .features import phone_channel, preprocess, utterance_stats

FLEURS_TAG = {"train": "train_hi", "test": "test_hi"}


def prepare_real(data_dir: str, n_train: int, n_test: int, seed: int = 0) -> None:
    for split, n in (("train", n_train), ("test", n_test)):
        prepare_hf("google/fleurs", "hi_in", split, "hi", "raw_transcription", f"{data_dir}/fleurs", n, seed, FLEURS_TAG[split])


def real_items(data_dir: str, split: str) -> list[dict]:
    path = Path(data_dir) / "fleurs" / FLEURS_TAG[split] / "manifest.jsonl"
    return [
        {"id": r["id"], "audio": r["audio"], "text": r["text"], "label": 0, "system": "fleurs", "sent": r["id"]}
        for r in read_manifest(path)
    ]


def fake_path(data_dir: str, system: str, split: str, sent: str) -> Path:
    return Path(data_dir) / system / split / f"{sent}.wav"


def fake_items(data_dir: str, system: str, split: str, reals: list[dict]) -> list[dict]:
    """Fake clips for the given real items. Missing clips are an error, not
    something to skip, since a dropped pair would bias the comparison."""
    out, missing = [], []
    for r in reals:
        p = fake_path(data_dir, system, split, r["sent"])
        if not p.exists():
            missing.append(r["sent"])
            continue
        out.append({"id": f"{system}_{r['sent']}", "audio": str(p), "text": r["text"], "label": 1, "system": system, "sent": r["sent"]})
    if missing:
        raise FileNotFoundError(f"{len(missing)} {system} {split} clips missing, first: {missing[:3]}. Rerun synth to resume.")
    return out


def split_train_val(reals: list[dict], frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out whole sentences: FLEURS repeats some sentences across
    speakers, and a sentence must not sit on both sides of the split."""
    texts = sorted({r["text"] for r in reals})
    random.Random(seed).shuffle(texts)
    held = set(texts[: max(1, math.ceil(frac * len(texts)))])
    train = [r for r in reals if r["text"] not in held]
    val = [r for r in reals if r["text"] in held]
    return train, val


def text_overlap(a: list[dict], b: list[dict]) -> int:
    return len({r["text"] for r in a} & {r["text"] for r in b})


def load_clips(items: list[dict]) -> list[dict]:
    """Reads and preprocesses every item. Adds wave (processed, uncropped) and
    raw_rms. A clip that fails preprocessing stops the run."""
    out = []
    for it in items:
        x, sr = sf.read(it["audio"], dtype="float32")
        try:
            wave, raw = preprocess(x, sr)
        except ValueError as e:
            raise ValueError(f"{it['id']} ({it['audio']}): {e}") from e
        out.append({**it, "wave": wave, "raw_rms": raw})
    return out


def with_channel(clips: list[dict]) -> list[dict]:
    return [{**c, "wave": phone_channel(c["wave"])} for c in clips]


def stats_matrix(clips: list[dict]) -> np.ndarray:
    return np.stack([utterance_stats(c["wave"], c["raw_rms"]) for c in clips])


def labels(clips: list[dict]) -> np.ndarray:
    return np.array([c["label"] for c in clips], dtype=int)


def write_system_manifest(data_dir: str, system: str, split: str, items: list[dict]) -> None:
    write_jsonl(Path(data_dir) / system / f"{split}_manifest.jsonl", items)
