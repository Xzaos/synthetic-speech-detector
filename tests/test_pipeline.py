import json

import numpy as np
import soundfile as sf

from synthdet.cli import main
from synthdet.data import fake_path, write_jsonl
from synthdet.features import SR


def clip(rng, kind):
    n = int(rng.uniform(3.5, 6) * SR)
    x = rng.normal(0, 1, n)
    if kind == "fake":
        x = np.convolve(x, np.ones(6) / 6, mode="same")
    env = np.abs(np.sin(np.linspace(0, rng.uniform(8, 20), n))) + 0.05
    return (0.1 * x * env).astype(np.float32)


def build(root, n_train=30, n_test=12):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        tag = {"train": "train_hi", "test": "test_hi"}[split]
        rows = []
        for i in range(n):
            uid = f"{tag}_{i:04d}"
            p = root / "fleurs" / tag / "wav" / f"{uid}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, clip(rng, "real"), SR)
            rows.append({"id": uid, "audio": str(p), "text": f"sentence {i}", "lang": "hi"})
            systems = ["mms", "bulbul"] if split == "test" else ["mms"]
            for s in systems:
                fp = fake_path(str(root), s, split, uid)
                fp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(fp, clip(rng, "fake"), SR)
        write_jsonl(root / "fleurs" / tag / "manifest.jsonl", rows)


def test_train_and_eval_end_to_end(tmp_path):
    data, res = tmp_path / "data", tmp_path / "results"
    build(data)
    base = ["--data", str(data), "--results", str(res)]
    main(base + ["train", "--epochs", "4", "--minutes", "5"])
    main(base + ["eval", "--n-boot", "20"])
    rep = json.loads((res / "report.json").read_text(encoding="utf-8"))
    assert rep["meta"]["n_val_pairs"] == 3 and rep["meta"]["n_train_pairs"] == 27
    assert rep["meta"]["params"] < 200_000
    for m in ("baseline", "detector"):
        assert set(rep[m]["conditions"]) == {"seen_clean", "unseen_clean", "seen_channel", "unseen_channel"}
        c = rep[m]["conditions"]["seen_clean"]
        assert c["n_real"] == 12 and c["n_fake"] == 12
        assert c["eer"]["lo"] <= c["eer"]["value"] <= c["eer"]["hi"]
    # smoothed noise vs white noise: the cheap stats alone should separate them
    assert rep["baseline"]["conditions"]["seen_clean"]["auc"]["value"] > 0.9
    assert (res / "report.md").exists()
