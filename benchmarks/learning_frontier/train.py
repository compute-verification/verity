"""Train one learning-frontier configuration and record its loss curve against compute, time, novel state and communication.

Families (`--family`):

- `dense`: AdamW on every parameter, the reference.
- `lowrank`: per-window low rank. Each window of `--window` optimizer steps trains only a rank-`--rank` factor pair per matrix
  (every attention and MLP matrix, and the tied embedding) and the norms; at the window's end the pairs merge into the frozen
  matrices and fresh pairs are drawn (`model.py`). The factors' AdamW state is reset at each merge (the norms' persists), and the
  learning rate re-warms linearly over `--rewarm` steps after each merge. No full-rank warmup. Novel bits per window: every value
  trained in it (factors and norms) at its stored precision, fp32.
- `localsgd`: `--replicas` replicas with their own AdamW, each taking a quarter of every global batch (the same sequences, in the
  same order, as dense); every `--sync` steps the outer step: the replicas' mean delta from the global model is an outer gradient
  for SGD with Nesterov momentum (`--outer-lr`, `--outer-momentum`), and every replica restarts from the new global model. Bits
  communicated per sync: each replica's delta, every parameter at fp32. The global model is evaluated at syncs.

Every family at a size starts from the same initialization (`--seed`) and reads the same sequences in the same order
(`--data-seed`): the training stream cut into CTX + 1 token windows at stride CTX, permuted once. The learning rate warms up
linearly over `--warmup` of the steps and decays by cosine to `--min-lr` of its peak. Held-out loss is in nats per byte of the
held-out text (`prep.py`'s `val_bytes`), evaluated `--evals` times at equal step intervals.

A run trains for at most `--chunk-minutes` of wall-clock, then saves its state under `--ckpt-root` (keyed by its configuration)
and exits 3; started again with the same arguments it resumes there, so a long run is a sequence of short GPU leases. It also
saves at every evaluation, and on SIGTERM (a preempted lease) it saves and exits 143. Exit 0: the run is complete and
`--out` holds `config.json`, `curve.json` and `result.json`.

    python train.py --data DIR --out DIR --ckpt-root DIR --size s --family lowrank --rank 32 --lr 3e-3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CTX, SIZES, VOCAB, LM, counts

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "benchmarks/census/hardware.json"
PEAK_ID = "rtx-pro-6000-server/bf16"
BITS = 32
EXIT_CHUNK, EXIT_TERM = 3, 143
TIMED_AFTER = 10  # steps after each start before the throughput clock runs (compilation, warm caches)


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--family", choices=("dense", "lowrank", "localsgd"), required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--rewarm", type=int, default=20)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--sync", type=int, default=0)
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--tokens-per-param", type=float, default=8.0, help="training tokens per non-embedding parameter")
    ap.add_argument("--batch", type=int, default=64, help="sequences per optimizer step")
    ap.add_argument("--micro", type=int, default=0, help="sequences per forward pass (default: the whole batch)")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--min-lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--chunk-minutes", type=float, default=26.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (a smoke test); 0: the full run")
    ap.add_argument("--val-windows", type=int, default=0, help="held-out windows to evaluate (a smoke test); 0: all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args(argv)
    if (a.family == "lowrank") != (a.rank > 0):
        ap.error("--rank is for --family lowrank, which needs it")
    if (a.family == "localsgd") != (a.sync > 0):
        ap.error("--sync is for --family localsgd, which needs it")
    if a.family == "localsgd" and a.batch % a.replicas:
        ap.error("--batch must split evenly over --replicas")
    return a


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def peak_flops() -> float:
    for row in json.loads(CENSUS.read_text()):
        if row.get("id") == PEAK_ID:
            return float(row["flop_per_second"])
    raise SystemExit(f"train: {CENSUS} has no {PEAK_ID}")


def git_commit() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or f"unknown ({r.stderr.strip()})"


class Corpus:
    def __init__(self, root: Path, verify_train: bool):
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.train = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.fromfile(root / "val.bin", dtype=np.uint16)
        if sha256(root / "val.bin") != self.manifest["val"]["sha256"]:
            raise SystemExit("train: val.bin does not match its manifest")
        if verify_train and sha256(root / "train.bin") != self.manifest["train"]["sha256"]:
            raise SystemExit("train: train.bin does not match its manifest")
        self.n_seq = (len(self.train) - 1) // CTX

    def order(self, seed: int, n: int) -> np.ndarray:
        if n > self.n_seq:
            raise SystemExit(f"train: the run needs {n} sequences and the corpus has {self.n_seq}")
        return np.random.default_rng(seed).permutation(self.n_seq)[:n]

    def batch(self, idx: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = np.stack([self.train[i * CTX: i * CTX + CTX + 1] for i in idx]).astype(np.int64)
        t = torch.from_numpy(rows)
        if device.startswith("cuda"):
            t = t.pin_memory().to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def schedule(a: argparse.Namespace, step: int, n_steps: int) -> float:
    warm = max(1, round(a.warmup * n_steps))
    lr = a.lr * min(1.0, (step + 1) / warm)
    if step >= warm:
        frac = (step - warm) / max(1, n_steps - warm)
        lr = a.lr * (a.min_lr + (1 - a.min_lr) * 0.5 * (1 + math.cos(math.pi * frac)))
    if a.family == "lowrank" and step >= a.window:
        lr *= min(1.0, (step % a.window + 1) / a.rewarm)
    return lr


def adamw(params: list[torch.nn.Parameter], a: argparse.Namespace, fused: bool) -> torch.optim.AdamW:
    decay = [p for p in params if p.ndim >= 2]
    keep = [p for p in params if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": keep, "weight_decay": 0.0}],
                             lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, fused=fused)


def loss_of(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1), reduction=reduction)


@torch.no_grad()
def evaluate(model: torch.nn.Module, corpus: Corpus, windows: int, micro: int, device: str) -> tuple[float, float, int]:
    total = torch.zeros((), dtype=torch.float64, device=device)
    val = torch.from_numpy(corpus.val.astype(np.int64))
    for i in range(0, windows, micro):
        j = min(windows, i + micro)
        rows = torch.stack([val[w * CTX: w * CTX + CTX + 1] for w in range(i, j)]).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            total += loss_of(model, rows[:, :-1], rows[:, 1:], "sum").double()
    nll = float(total)
    targets = windows * CTX
    val_bytes = corpus.manifest["val"]["val_bytes"] if windows * CTX == corpus.manifest["val"]["targets"] else None
    return (nll / val_bytes if val_bytes else float("nan")), nll / targets, targets


class Run:
    """One configuration's model(s), optimizer(s) and bookkeeping, with save and resume."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.size = SIZES[a.size]
        self.counts = counts(self.size, a.rank)
        self.n_steps = max(1, round(a.tokens_per_param * self.counts["non_embedding"] / (a.batch * CTX)))
        self.micro = a.micro or (a.batch // a.replicas if a.family == "localsgd" else a.batch)
        fused = a.device.startswith("cuda")
        torch.manual_seed(a.seed)
        self.model = LM(self.size).to(a.device)
        if a.family == "lowrank":
            self.model.attach(a.rank, a.seed)
        self.replicas, self.opts, self.outer_buf = [], [], []
        if a.family == "localsgd":
            for _ in range(a.replicas):
                r = LM(self.size).to(a.device)
                r.load_state_dict(self.model.state_dict())
                self.replicas.append(r)
                self.opts.append(adamw(list(r.parameters()), a, fused))
            self.outer_buf = [torch.zeros_like(p) for p in self.model.parameters()]
        else:
            self.opts.append(adamw([p for p in self.model.parameters() if p.requires_grad], a, fused))
        self.compiled = [m if a.no_compile else torch.compile(m) for m in (self.replicas or [self.model])]
        self.eval_model = self.compiled[0] if not self.replicas else self.model if a.no_compile else torch.compile(self.model)
        self.step, self.syncs, self.curve, self.ema = 0, 0, [], None
        self.timed_steps, self.timed_s, self.wall_s, self.eval_s = 0, 0.0, 0.0, 0.0
        self.starts, self.train_verified = 0, False

    def key(self, corpus_sha: str) -> str:
        skip = {"out", "ckpt_root", "chunk_minutes", "device", "no_compile", "data"}
        spec = {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(self.a).items()) if k not in skip}
        return hashlib.sha256(json.dumps([spec, corpus_sha], sort_keys=True).encode()).hexdigest()[:20]

    def state(self) -> dict:
        return {"step": self.step, "syncs": self.syncs, "curve": self.curve, "model": self.model.state_dict(),
                "replicas": [r.state_dict() for r in self.replicas], "opts": [o.state_dict() for o in self.opts],
                "outer_buf": self.outer_buf, "timed_steps": self.timed_steps, "timed_s": self.timed_s,
                "wall_s": self.wall_s, "eval_s": self.eval_s, "starts": self.starts, "train_verified": self.train_verified,
                "ema": None if self.ema is None else float(self.ema)}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / "state.pt.tmp"
        torch.save(self.state(), tmp)
        tmp.rename(path / "state.pt")

    def load(self, path: Path) -> bool:
        f = path / "state.pt"
        if not f.exists():
            return False
        s = torch.load(f, map_location=self.a.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        for r, rs in zip(self.replicas, s["replicas"]):
            r.load_state_dict(rs)
        for o, os_ in zip(self.opts, s["opts"]):
            o.load_state_dict(os_)
        for b, bs in zip(self.outer_buf, s["outer_buf"]):
            b.copy_(bs)
        for k in ("step", "syncs", "curve", "timed_steps", "timed_s", "wall_s", "eval_s", "starts", "train_verified"):
            setattr(self, k, s[k])
        self.ema = None if s["ema"] is None else torch.tensor(s["ema"], device=self.a.device)
        return True

    def eval_steps(self) -> list[int]:
        n, e = self.n_steps, self.a.evals
        marks = sorted({max(1, round(n * k / e)) for k in range(1, e + 1)})
        if self.a.family == "localsgd":
            h = self.a.sync
            marks = sorted({min(n, max(h, h * round(m / h))) for m in marks} | {n})
        return marks

    def train_step(self, corpus: Corpus, order: np.ndarray) -> torch.Tensor:
        a, step, b = self.a, self.step, self.a.batch
        lr = schedule(a, step, self.n_steps)
        dev = a.device.split(":")[0]
        losses = []
        groups = [(self.compiled[k], self.opts[k], order[step * b + k * b // a.replicas: step * b + (k + 1) * b // a.replicas])
                  for k in range(a.replicas)] if a.family == "localsgd" else [(self.compiled[0], self.opts[0],
                                                                                  order[step * b: (step + 1) * b])]
        for model, opt, idx in groups:
            for g in opt.param_groups:
                g["lr"] = lr
            params = [p for g in opt.param_groups for p in g["params"]]
            for i in range(0, len(idx), self.micro):
                x, y = corpus.batch(idx[i: i + self.micro], a.device)
                with torch.autocast(dev, dtype=torch.bfloat16):
                    loss = loss_of(model, x, y) * (len(x) / len(idx))
                loss.backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.step += 1
        if a.family == "lowrank" and self.step % a.window == 0 and self.step < self.n_steps:
            self.model.merge(a.seed, self.step // a.window)
            for p in [p for _, pa, pb, _ in self.model.factor_pairs() for p in (pa, pb)]:
                self.opts[0].state.pop(p, None)
        if a.family == "localsgd" and (self.step % a.sync == 0 or self.step == self.n_steps):
            self.outer_step()
        return torch.stack(losses).sum() / len(groups)

    @torch.no_grad()
    def outer_step(self) -> None:
        a = self.a
        glob = list(self.model.parameters())
        reps = [list(r.parameters()) for r in self.replicas]
        delta = [g - sum(r[i] for r in reps) / len(reps) for i, g in enumerate(glob)]
        torch._foreach_mul_(self.outer_buf, a.outer_momentum)
        torch._foreach_add_(self.outer_buf, delta)
        update = torch._foreach_add(delta, self.outer_buf, alpha=a.outer_momentum)
        torch._foreach_add_(glob, update, alpha=-a.outer_lr)
        for r in reps:
            torch._foreach_copy_(r, glob)
        self.syncs += 1

    def tallies(self, step: int, syncs: int) -> dict:
        a, c = self.a, self.counts
        tokens = step * a.batch * CTX
        per_token = c["lowrank_flops_per_token"] if a.family == "lowrank" else c["dense_flops_per_token"]
        trained = c["trained_per_window"] if a.family == "lowrank" else c["total"]
        out = {"step": step, "tokens": tokens, "flops": per_token * tokens, "flops_6n": c["flops_per_token_6n"] * tokens,
               "windows": step / a.window, "novel_bits": BITS * trained * step / a.window,
               "novel_bits_started_windows": BITS * trained * math.ceil(step / a.window)}
        if a.family == "localsgd":
            out |= {"syncs": syncs, "bits_communicated": BITS * a.replicas * c["total"] * syncs}
        else:
            out |= {"bits_communicated_as_data_parallel": BITS * a.replicas * c["total"] * step}
        return out


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    start = time.perf_counter()
    term = {"set": False}
    signal.signal(signal.SIGTERM, lambda *_: term.update(set=True))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run = Run(a)
    manifest = json.loads((a.data / "manifest.json").read_text())
    key = run.key(manifest["train"]["sha256"])
    ckpt = a.ckpt_root / key
    resumed = run.load(ckpt)
    corpus = Corpus(a.data, verify_train=not run.train_verified)
    run.train_verified = True
    run.starts += 1
    n_steps = min(run.n_steps, a.max_steps) if a.max_steps else run.n_steps
    order = corpus.order(a.data_seed, run.n_steps * a.batch)
    windows = a.val_windows or corpus.manifest["val"]["targets"] // CTX
    marks = [m for m in run.eval_steps() if m <= n_steps] or [n_steps]
    peak = peak_flops()
    c = run.counts
    per_token = c["lowrank_flops_per_token"] if a.family == "lowrank" else c["dense_flops_per_token"]
    gpu = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"
    config = {"schema": "learning-frontier/run/v1", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "key": key, "commit": git_commit(), "n_steps": run.n_steps, "eval_steps": run.eval_steps(),
              "micro_batch": run.micro, "counts": c, "flops_per_token": per_token, "bits_per_value": BITS,
              "corpus": manifest, "data_order": {"rule": "the training stream cut into ctx + 1 token windows at stride ctx, "
                                                          "permuted by numpy default_rng(data_seed).permutation; step t reads "
                                                          "permuted windows [t * batch, (t + 1) * batch), a localsgd replica k "
                                                          "the k-th quarter of them", "data_seed": a.data_seed,
                                                 "sequences": int(corpus.n_seq),
                                                 "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()},
              "ctx": CTX, "vocab": VOCAB, "seed": a.seed, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu,
              "peak_flop_per_second": peak, "peak_source": f"benchmarks/census/hardware.json {PEAK_ID}",
              "starts": run.starts, "resumed_at_step": run.step if resumed else None}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    print(f"train: {a.family} size={a.size} lr={a.lr} rank={a.rank} sync={a.sync} steps={run.n_steps} key={key} "
          f"resume={run.step if resumed else 0}", flush=True)

    def sync() -> None:
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    seg_t0, seg_steps, first = None, 0, run.step
    while run.step < n_steps:
        loss = run.train_step(corpus, order)
        run.ema = loss if run.ema is None else 0.98 * run.ema + 0.02 * loss
        if (run.step % 25 == 0 or run.step in marks) and not math.isfinite(float(loss)):
            run.curve.append({"step": run.step, "diverged": True, "train_loss": float(loss)})
            (a.out / "curve.json").write_text(json.dumps(run.curve, indent=1) + "\n")
            print(f"train: diverged at step {run.step}", flush=True)
            break
        if run.step - first == TIMED_AFTER:
            sync()
            seg_t0, seg_steps = time.perf_counter(), 0
        elif seg_t0 is not None:
            seg_steps += 1
        if run.step in marks:
            sync()
            if seg_t0 is not None:
                run.timed_s += time.perf_counter() - seg_t0
                run.timed_steps += seg_steps
            t0 = time.perf_counter()
            nats_byte, nats_token, targets = evaluate(run.eval_model, corpus, windows, run.micro, a.device)
            run.eval_s += time.perf_counter() - t0
            spt = run.timed_s / run.timed_steps if run.timed_steps else float("nan")
            point = run.tallies(run.step, run.syncs) | {
                "val_nats_per_byte": nats_byte, "val_nats_per_token": nats_token, "val_targets": targets,
                "train_loss_ema": float(run.ema), "lr": schedule(a, run.step - 1, run.n_steps),
                "seconds_per_step": spt, "tokens_per_second": a.batch * CTX / spt if run.timed_steps else None,
                "achieved_flop_per_second": per_token * a.batch * CTX / spt if run.timed_steps else None}
            point["mfu"] = point["achieved_flop_per_second"] / peak if point["achieved_flop_per_second"] else None
            run.curve.append(point)
            (a.out / "curve.json").write_text(json.dumps(run.curve, indent=1) + "\n")
            print(json.dumps({k: point[k] for k in ("step", "val_nats_per_byte", "train_loss_ema", "tokens_per_second")}),
                  flush=True)
            if run.step < run.n_steps:
                run.save(ckpt)
            if seg_t0 is not None:
                sync()
                seg_t0, seg_steps = time.perf_counter(), 0
        out_of_time = (time.perf_counter() - start) / 60 > a.chunk_minutes
        if (term["set"] or out_of_time) and run.step < n_steps:
            if seg_t0 is not None:
                sync()
                run.timed_s += time.perf_counter() - seg_t0
                run.timed_steps += seg_steps
            run.wall_s += time.perf_counter() - start
            run.save(ckpt)
            print(f"train: {'SIGTERM' if term['set'] else 'chunk ended'} at step {run.step}; state saved", flush=True)
            return EXIT_TERM if term["set"] else EXIT_CHUNK
    run.wall_s += time.perf_counter() - start
    spt = run.timed_s / run.timed_steps if run.timed_steps else None
    result = {"schema": "learning-frontier/result/v1", "key": key, "family": a.family, "size": a.size, "lr": a.lr,
              "rank": a.rank, "sync": a.sync, "steps": run.step, "complete": run.step >= run.n_steps,
              "diverged": bool(run.curve and run.curve[-1].get("diverged")),
              "final": run.curve[-1] if run.curve else None, "seconds_per_step": spt,
              "tokens_per_second": a.batch * CTX / spt if spt else None,
              "achieved_flop_per_second": per_token * a.batch * CTX / spt if spt else None,
              "mfu": per_token * a.batch * CTX / spt / peak if spt else None,
              "wall_seconds": run.wall_s, "eval_seconds": run.eval_s, "starts": run.starts, "gpu": gpu}
    (a.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    if result["complete"] or result["diverged"]:
        shutil.rmtree(ckpt, ignore_errors=True)
    print(json.dumps({k: result[k] for k in ("family", "size", "lr", "steps", "tokens_per_second", "mfu")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
