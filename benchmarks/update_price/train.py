"""The fine-tune whose updated slots `price.py` describes: SmolLM2-135M, fp32 AdamW on CPU, packed wikitext-103 text.

Everything that decides the slots is fixed here or in the registration: θ (the model at its pinned revision), the data
(the dataset's pinned revision, packed into `--seq`-token chunks in file order, drawn in a seeded permutation), the seed,
the hyperparameters and the thread count. Training runs in float32 throughout (weights, activations, gradients, Adam
moments), with `torch.use_deterministic_algorithms(True)` and no dropout, so the run is a deterministic function of
them on one machine.

Writes, under `--ckpt` (scratch): θ as step 0 and the parameters after each `--save` step, one `.npy` per tensor, and the
Adam moments after each `--state` step; under `--out`: `train.json` (the record a replay needs, per-step loss and
timings, each checkpoint's SHA-256, the token stream's code lengths), and with `--replay N` a second run of the first N
steps from θ in the same process, whose checkpoints must have the same digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

MODEL = ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
DATA = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wikitext-103-raw-v1/train-00000-of-00002.parquet")


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True, help="model and dataset downloads")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--save", default="1,10,100,300,1000")
    p.add_argument("--state", default="1,10,100")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--betas", default="0.9,0.95")
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--replay", type=int, default=100, help="rerun the first N steps from θ and compare digests")
    return p


def lr_at(step: int, a) -> float:
    if step <= a.warmup:
        return a.lr * step / a.warmup
    frac = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def fetch(cache: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    model = Path(snapshot_download(MODEL[0], revision=MODEL[1], cache_dir=cache / "hf",
                                   allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    data = Path(hf_hub_download(DATA[0], DATA[2], revision=DATA[1], repo_type="dataset", cache_dir=cache / "hf"))
    return model, data


def tokens(model_dir: Path, data: Path, need: int) -> np.ndarray:
    """The first `need` tokens of the parquet file's text, in row order."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    rows = pq.read_table(data, columns=["text"]).column("text").to_pylist()
    out, have, i = [], 0, 0
    while have < need:
        block = "".join(rows[i:i + 2000])
        i += 2000
        if not block and i >= len(rows):
            raise SystemExit(f"the data file holds {have} tokens, the run needs {need}")
        ids = tok.encode(block, add_special_tokens=False).ids
        out.append(np.asarray(ids, dtype=np.int64))
        have += len(ids)
    return np.concatenate(out)[:need]


def stored_dtypes(model_dir: Path) -> list[str]:
    """The dtypes θ is published in (the safetensors header's)."""
    with open(model_dir / "model.safetensors", "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    return sorted({v["dtype"] for k, v in head.items() if k != "__metadata__"})


def digest(model) -> str:
    h = hashlib.sha256()
    for _, p in model.named_parameters():
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def save(model, where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, p in model.named_parameters():
        np.save(where / f"{name}.npy", p.detach().numpy())


def save_state(model, opt, where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, p in model.named_parameters():
        st = opt.state[p]
        np.save(where / f"{name}.exp_avg.npy", st["exp_avg"].numpy())
        np.save(where / f"{name}.exp_avg_sq.npy", st["exp_avg_sq"].numpy())


def run(a, model_dir: Path, chunks: np.ndarray, order: np.ndarray, *, write: bool, upto: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.train()
    named = list(model.named_parameters())
    decay = [p for _, p in named if p.ndim >= 2]
    keep = [p for _, p in named if p.ndim < 2]
    b1, b2 = (float(x) for x in a.betas.split(","))
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": keep, "weight_decay": 0.0}],
                            lr=0.0, betas=(b1, b2), eps=a.eps, foreach=False, fused=False)
    saves = {int(s) for s in a.save.split(",") if s}
    states = {int(s) for s in a.state.split(",") if s}
    rec = {"digests": {"0": digest(model)}, "steps": []}
    if write:
        save(model, a.ckpt / "0")
        rec["params"] = [[n, list(p.shape)] for n, p in named]
        rec["theta_dtypes"] = stored_dtypes(model_dir)
    t_all = time.perf_counter()
    for step in range(1, upto + 1):
        idx = order[(step - 1) * a.batch: step * a.batch]
        x = torch.from_numpy(chunks[idx])
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.perf_counter()
        loss = model(input_ids=x, labels=x).loss
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, foreach=False)
        opt.step()
        opt.zero_grad(set_to_none=True)
        t3 = time.perf_counter()
        rec["steps"].append({"step": step, "loss": float(loss.item()), "grad_norm": float(gn), "lr": lr,
                             "fwd_s": t1 - t0, "bwd_s": t2 - t1, "opt_s": t3 - t2})
        if step in saves:
            rec["digests"][str(step)] = digest(model)
            if write:
                save(model, a.ckpt / str(step))
        if write and step in states:
            save_state(model, opt, a.ckpt / f"state{step}")
        if step % 10 == 0 or step == upto:
            print(f"step {step} loss {loss.item():.4f} {(t3 - t0):.2f}s/step", flush=True)
    rec["wall_s"] = time.perf_counter() - t_all
    return rec


def main(argv=None) -> None:
    a = args_parser().parse_args(argv)
    import torch

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a.ckpt.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model_dir, data = fetch(a.cache)
    need = a.steps * a.batch * a.seq
    ids = tokens(model_dir, data, need)
    n_chunks = need // a.seq
    chunks = ids.reshape(n_chunks, a.seq)
    order = np.random.default_rng(a.seed).permutation(n_chunks)
    np.save(a.ckpt / "order.npy", order)
    np.save(a.ckpt / "chunks.npy", chunks.astype(np.uint16))
    vocab = json.loads((model_dir / "config.json").read_text())["vocab_size"]

    rec = run(a, model_dir, chunks, order, write=True, upto=a.steps)
    saves = sorted(int(s) for s in a.save.split(",") if s)
    stream = {}
    for n in saves:
        part = chunks[order[: n * a.batch]]
        seen_n = np.zeros(vocab, dtype=bool)
        seen_n[np.unique(part)] = True
        np.save(a.ckpt / f"seen{n}.npy", seen_n)
        losses = [s["loss"] for s in rec["steps"][:n]]
        stream[str(n)] = {
            "tokens": int(part.size),
            "prequential_bits": float(sum(losses) * a.batch * (a.seq - 1) / math.log(2) + n * a.batch * math.log2(vocab)),
            "raw_bits": float(part.size * math.log2(vocab)),
            "xz_bits": 8 * len(lzma.compress(part.astype("<u2").tobytes(), preset=9 | lzma.PRESET_EXTREME)),
            "unseen_tokens": int(vocab - seen_n.sum()),
        }
    out = {
        "model": {"repo": MODEL[0], "revision": MODEL[1]},
        "data": {"repo": DATA[0], "revision": DATA[1], "file": DATA[2],
                 "tokens_sha256": hashlib.sha256(chunks.astype("<u2").tobytes()).hexdigest()},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
                   if k not in ("ckpt", "out", "cache")},
        "vocab": vocab,
        "machine": {"cpu": _cpu(), "threads": a.threads, "platform": platform.platform(),
                    "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "train": rec,
        "token_stream": stream,
    }
    if a.replay:
        rep = run(a, model_dir, chunks, order, write=False, upto=a.replay)
        out["replay"] = {"steps": a.replay, "wall_s": rep["wall_s"],
                         "digests": rep["digests"],
                         "match": {k: v == rec["digests"].get(k) for k, v in rep["digests"].items()},
                         "step_s": [s["fwd_s"] + s["bwd_s"] + s["opt_s"] for s in rep["steps"]]}
        print("replay match:", out["replay"]["match"], flush=True)
    (a.out / "train.json").write_text(json.dumps(out, indent=1))


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    main()
