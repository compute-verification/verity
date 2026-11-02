"""CPU smoke test of the speedrun harness on the tiny model and a synthetic corpus (seconds, no GPU, no network).

    python smoke.py [--keep DIR]

Builds a corpus of Zipf-distributed tokens with a manifest like prep.py's, trains a tiny dense start with `train.py --save-theta`,
and checks: the kept start's digest round-trips; speedrun.py's evaluation at the trained context equals train.py's; every
sliding-window setting scores each target once; per-window low rank's FLOP count is lora_flops_per_token's case of every
matrix; every class runs to a witness record; a LoRA run's merged weights evaluate to its last checkpoint's loss; a run cut
into chunks resumes to the same result as one run straight through; and kappa.py reads the records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import kappa
import speedrun
import train
from model import CTX, SIZES, LM, counts, load_theta

VAL_WINDOWS = 2
TRAIN_SEQ = 400


def corpus(root: Path) -> None:
    rng = np.random.default_rng(0)
    ranks = np.arange(1, 2001)
    p = 1.0 / ranks / (1.0 / ranks).sum()
    root.mkdir(parents=True, exist_ok=True)
    rng.choice(2000, size=TRAIN_SEQ * CTX + 1, p=p).astype(np.uint16).tofile(root / "train.bin")
    rng.choice(2000, size=VAL_WINDOWS * CTX + 1, p=p).astype(np.uint16).tofile(root / "val.bin")
    sha = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in ("train.bin", "val.bin")}
    (root / "manifest.json").write_text(json.dumps({
        "schema": "learning-frontier/corpus/v1", "synthetic": "Zipf over 2000 token ids (smoke.py)",
        "train": {"tokens": TRAIN_SEQ * CTX + 1, "sha256": sha["train.bin"]},
        "val": {"targets": VAL_WINDOWS * CTX, "val_bytes": 4 * VAL_WINDOWS * CTX, "sha256": sha["val.bin"]}}))


def run(fn, argv: list[str]) -> int:
    print("+", " ".join(argv[-12:]), flush=True)
    return fn(argv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--keep", type=Path, default=None, help="work here and keep it (default: a temporary directory)")
    a = ap.parse_args()
    torch.set_num_threads(1)  # bf16 autocast on several CPU threads varies by up to ~5e-3 from call to call
    work = a.keep or Path(tempfile.mkdtemp(prefix="speedrun-smoke-"))
    if a.keep and work.exists():
        shutil.rmtree(work)
    data, ckpt = work / "data", work / "ckpt"
    corpus(data)
    common = ["--device", "cpu", "--no-compile", "--batch", "1", "--evals", "4"]

    # the start: tiny dense, 0.1 tokens per parameter (10 steps of one sequence), kept at 50% and 100%
    assert run(train.main, ["--data", str(data), "--out", str(work / "base"), "--ckpt-root", str(ckpt), "--size", "tiny",
                            "--family", "dense", "--lr", "3e-3", "--tokens-per-param", "0.1", "--save-theta", "0.5,1",
                            "--commit", "smoke", *common]) == 0
    thetas = json.loads((work / "base/theta.json").read_text())
    curve = {p["step"]: p["val_nats_per_byte"] for p in json.loads((work / "base/curve.json").read_text())}
    assert [t["fraction"] for t in thetas] == [0.5, 1.0], thetas
    for t in thetas:
        state, meta = load_theta(work / "base" / t["file"], t["theta_sha256"])
        assert meta["val_nats_per_byte"] == curve[t["step"]]
    start = work / "base" / thetas[0]["file"]

    # evaluation parity, window coverage, FLOP count
    state, meta = load_theta(start)
    model = LM(SIZES["tiny"])
    model.load_state_dict(state)
    c = train.Corpus(data, verify_train=True)
    ours = speedrun.evaluate(model, c.val, VAL_WINDOWS * CTX, c.manifest["val"]["val_bytes"], "cpu", 1)
    theirs = train.evaluate(model, c, VAL_WINDOWS, 1, "cpu")
    assert abs(ours["val_nats_per_byte"] - theirs[0]) < 1e-5, (ours, theirs)
    assert abs(ours["val_nats_per_byte"] - meta["val_nats_per_byte"]) < 1e-5
    for ctx, stride, n in ((1024, 1024, 4096), (1024, 256, 4096), (2048, 1024, 4096), (1024, 384, 3000)):
        hits = np.zeros(n, int)
        for b, f in speedrun.windows(n, ctx, stride):
            hits[b + f: b + ctx] += 1
        assert (hits == 1).all(), (ctx, stride)
    for size in ("s", "m", "l"):
        sz = SIZES[size]
        plan = {"rank": 32, "blocks": list(range(sz.n_layer)), "embed": True}
        assert speedrun.lora_flops_per_token(sz, plan) == counts(sz, 32)["lowrank_flops_per_token"]
    plans = {size: {b: speedrun.lora_plan(SIZES[size], b) for b in (1e5, 1e6, 1e7, 1e8)} for size in ("s", "m", "l")}
    print(json.dumps({s: {f"{b:g}": (p["rank"], len(p["blocks"]), p["embed"],
                                     speedrun.lora_update_values(SIZES[s], p) * 4) for b, p in ps.items()}
                      for s, ps in plans.items()}))

    # every class
    sr = ["--data", str(data), "--start", str(start), "--start-sha256", meta["theta_sha256"], "--ckpt-root", str(ckpt),
          "--budget-tpp", "0.06", "--data-offset", "20", "--any-corpus", "--commit", "smoke", *common]
    out = work / "runs"
    cases = {"adamw-3e-3": ["--method", "adamw", "--lr", "3e-3"], "adamw-1e-3": ["--method", "adamw", "--lr", "1e-3"],
             "muon-3e-3": ["--method", "muon", "--lr", "3e-3"],
             "lora-1e4-3e-3": ["--method", "lora", "--update-bytes", "1e4", "--lr", "3e-3"],
             "lora-1e6-3e-3": ["--method", "lora", "--update-bytes", "1e6", "--lr", "3e-3"],
             "mezo-1e-4": ["--method", "mezo", "--lr", "1e-4"],
             "icl": ["--method", "icl", "--icl", "1024:1024,1024:512,1024:128,2048:1024:ntk,2048:1024:pi"]}
    for name, extra in cases.items():
        assert run(speedrun.main, [*sr, "--out", str(out / name), *extra]) == 0, name
    w = {name: json.loads((out / name / "witness.json").read_text()) for name in cases}
    for name, x in w.items():
        if x["method"] != "icl":
            assert x["complete"] and abs(x["l0_measured"] - meta["val_nats_per_byte"]) < 1e-5, name
            assert len(x["trajectory"]) == 5 and x["flops"] <= x["budget_flops"], name
    assert w["lora-1e4-3e-3"]["update"]["plan"]["embed"] is False
    assert w["lora-1e6-3e-3"]["update"]["plan"]["embed"] is True
    for name in ("lora-1e4-3e-3", "lora-1e6-3e-3"):
        assert abs(w[name]["merged_val_nats_per_byte"] - w[name]["final_val_nats_per_byte"]) < 2e-3, w[name]
        assert (out / name / "update.pt").exists()
    assert abs(w["icl"]["trajectory"][0]["val_nats_per_byte"] - ours["val_nats_per_byte"]) < 1e-5
    assert w["mezo-1e-4"]["update_bits"] == 32 * w["mezo-1e-4"]["steps"] + 64

    # resume: the same run in one-step chunks (--chunk-minutes 0 saves and exits 3 after every step)
    chunks = 0
    while (rc := run(speedrun.main, [*sr, "--out", str(out / "adamw-3e-3-chunked"), "--method", "adamw", "--lr", "3e-3",
                                     "--chunk-minutes", "0"])) == train.EXIT_CHUNK:
        chunks += 1
    assert rc == 0 and chunks > 1
    chunked = json.loads((out / "adamw-3e-3-chunked/witness.json").read_text())
    # bf16 on CPU isn't bitwise reproducible from one run to the next even on one thread (two straight runs end at different
    # digests), so the resumed run is held to the straight one's losses: a lost optimizer state or data position moves them
    # by far more
    gap = max(abs(p["val_nats_per_byte"] - q["val_nats_per_byte"])
              for p, q in zip(chunked["trajectory"], w["adamw-3e-3"]["trajectory"], strict=True))
    assert gap < 1e-5, f"resumed run differs from the straight one by {gap}"

    shutil.rmtree(out / "adamw-3e-3-chunked")
    assert kappa.main(["--out", str(work / "analysis"), "--include-smoke", str(out)]) == 0
    report = json.loads((work / "analysis/analysis.json").read_text())
    assert len(report["starts"]) == 1 and report["starts"][0]["kappa"], report["starts"]
    print(f"smoke: ok ({work})")
    if not a.keep:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
