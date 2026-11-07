from __future__ import annotations

import argparse

DATA, RESULTS = "data", "results"


def cmd_prepare(a):
    from .data import prepare_real

    prepare_real(a.data, a.n_train, a.n_test, a.seed)


def cmd_synth(a):
    from .data import real_items

    split_reals = {s: real_items(a.data, s) for s in a.splits}
    for split, reals in split_reals.items():
        if a.system == "mms":
            from .synth import synth_mms

            n = synth_mms(reals, a.data, split, a.seed)
        else:
            from .synth import MODEL, synth_bulbul

            speakers = a.speakers.split(",") if a.speakers else None
            n = synth_bulbul(reals, a.data, split, speakers, a.model or MODEL, a.concurrency)
        print(f"{a.system} {split}: wrote {n} new clips of {len(reals)}")


def cmd_train(a):
    from .data import fake_items, load_clips, real_items, split_train_val
    from .train import train_all

    reals = real_items(a.data, "train")
    train_r, val_r = split_train_val(reals, 0.1, a.seed)
    fakes = fake_items(a.data, "mms", "train", reals)
    by_sent = {f["sent"]: f for f in fakes}
    train = load_clips(train_r + [by_sent[r["sent"]] for r in train_r])
    val = load_clips(val_r + [by_sent[r["sent"]] for r in val_r])
    print(f"train clips {len(train)}, val clips {len(val)}")
    train_all(train, val, a.results, minutes=a.minutes, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed)


def cmd_eval(a):
    from .eval import run_eval

    run_eval(a.data, a.results, a.n_boot)


def main(argv=None):
    p = argparse.ArgumentParser(prog="synthdet")
    p.add_argument("--data", default=DATA)
    p.add_argument("--results", default=RESULTS)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="sample FLEURS hi_in train and test clips")
    s.add_argument("--n-train", type=int, default=800)
    s.add_argument("--n-test", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("synth", help="speak the FLEURS sentences with a TTS system")
    s.add_argument("system", choices=["mms", "bulbul"])
    s.add_argument("--splits", nargs="+", default=["train", "test"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", default=None, help="bulbul model id, default bulbul:v3")
    s.add_argument("--speakers", default=None, help="comma separated bulbul speakers")
    s.add_argument("--concurrency", type=int, default=2)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("train", help="fit the baseline and the CNN on MMS vs real")
    s.add_argument("--minutes", type=float, default=10.0)
    s.add_argument("--epochs", type=int, default=120)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="score all conditions and write results/report.*")
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
