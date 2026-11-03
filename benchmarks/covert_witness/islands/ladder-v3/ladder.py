"""The starts ladder (E1 of note:training-20261102T1715Z-draft-training-efficiency-experiments): one warmup-stable-decay
run per size, with cooldown branches that each decay the rate linearly to zero at a saturation (tokens per non-embedding
parameter), so each branch's end is a fully annealed start, registered in covert_witness's start format.

The model is covert_witness's `lf` family (model.py, what `prepare.Net` rebuilds), sized `s`, `m` or `l16`. Block matrices
train with Muon (train.py's), the tied embedding and the norm gains with AdamW. A branch at saturation t has
N = t * non-embedding parameters / tokens per step steps, the last `--cool` of them its cooldown: it forks from the main
run at N - cooldown and, when done, the main run resumes from the weights and optimizer state it forked from, so every
branch is exactly one standalone warmup-stable-decay run of N steps on the same stream prefix. The top branch is the main
run's own end. `--extra` saturations are branches that only add points to the from-scratch curve F (`--extra-cool`).

The stream is ladder_data.py's: blocks of `--group` parts, each block's windows (ctx + 1 tokens at stride ctx, inside a part)
permuted by (`--seed`, block); step k reads the stream's windows [k S, (k + 1) S), S sequences a step, so every size and
branch reads a prefix of one order. Evaluation (`--eval search`) is covert_witness's: bits per byte over the whole search
pool (`prepare.window_nats`, the windows split over the ranks; a registered start's own score is `prepare.evaluate_bpb` on
its registered θ). `--eval pilot` scores the stream's val.bin instead and registers nothing, for choosing hyperparameters.

    bash ladder.sh SIZE BATCH WORK [FLAGS...]   # under gpu-lease: count, then train on every leased GPU

    python ladder.py count --size l16 --batch 512 --micro 32 --out OUT      # one step's MACs under te_core's Counter
    torchrun --nproc_per_node G ladder.py train --size l16 --batch 512 --data DIR --work DIR --out OUT --count OUT/count.json

C0 of a branch is its steps times one step's MACs as `count` measures them: the single-process step (the global batch's
micro-batches, accumulated, then the optimizers' step), eager, under te_core's Counter. A rank's redundant Muon work and the
evaluation passes are not training and are not in it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import prepare  # noqa: E402
from model import CTX, LM, Size  # noqa: E402
from train import Muon  # noqa: E402

SIZES = {s.name: s for s in (Size("s", 512, 6, 8, 1408), Size("m", 768, 8, 12, 2048), Size("l16", 1024, 16, 16, 2816))}
RECIPE = "te-ladder/v1"


def log(**kw) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(kw), flush=True)


# ---- the model and its optimizers ----------------------------------------------------------------------------------------

def build(size: Size, device) -> LM:
    torch.manual_seed(0)
    with torch.device(device):
        return LM(size)


def non_embedding(model: LM) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if n != "embed")


def optimizers(model: LM, a) -> list[torch.optim.Optimizer]:
    ps = dict(model.named_parameters())
    mats = [p for n, p in ps.items() if p.ndim == 2 and n != "embed"]
    norms = [p for p in ps.values() if p.ndim == 1]
    muon = Muon(mats, lr=a.mat_lr, momentum=a.momentum, nesterov=True, wd=a.wd, scale="shape")
    muon.param_groups[0]["base_lr"] = a.mat_lr
    adam = torch.optim.AdamW([{"params": [ps["embed"]], "lr": a.embed_lr, "base_lr": a.embed_lr, "weight_decay": a.embed_wd},
                              {"params": norms, "lr": a.norm_lr, "base_lr": a.norm_lr, "weight_decay": 0.0}],
                             betas=(a.beta1, a.beta2), eps=1e-10, fused=True)
    return [muon, adam]


def set_rates(opts, mult: float, step: int, a) -> None:
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["base_lr"] * mult
    if a.mom_warmup:
        f = min(1.0, step / a.mom_warmup)
        opts[0].param_groups[0]["momentum"] = (1 - f) * a.mom_start + f * a.momentum


def loss_fn(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


def opt_card(a) -> dict:
    return {"matrices": {"optimizer": "muon", "lr": a.mat_lr, "momentum": a.momentum, "nesterov": True, "ns_steps": 5,
                         "scale": "sqrt(max(1, rows/cols))", "weight_decay": a.wd,
                         "momentum_warmup": {"from": a.mom_start, "steps": a.mom_warmup}},
            "embedding": {"optimizer": "adamw", "lr": a.embed_lr, "weight_decay": a.embed_wd},
            "norms": {"optimizer": "adamw", "lr": a.norm_lr, "weight_decay": 0.0},
            "adamw": {"betas": [a.beta1, a.beta2], "eps": 1e-10}, "clip": a.clip,
            "precision": "bf16 autocast, fp32 weights"}


# ---- the stream ------------------------------------------------------------------------------------------------------------

class Stream:
    """ladder_data.py's parts as blocks of `group`, each block's windows permuted by (seed, block)."""

    def __init__(self, root: Path, group: int, seed: int, ctx: int = CTX):
        self.root, self.group, self.seed, self.ctx = root, group, seed, ctx
        self.blocks: list[tuple[list[np.memmap], np.ndarray, np.ndarray]] = []
        self.bounds = [0]
        self.recs: dict[int, dict] = {}

    def part(self, i: int) -> np.memmap:
        rec_path = self.root / f"parts/{i:03d}.json"
        waited = False
        while not rec_path.is_file():
            if not waited:
                log(waiting_for=str(rec_path))
                waited = True
            time.sleep(30)
        rec = json.loads(rec_path.read_text())
        mm = np.memmap(self.root / f"parts/{i:03d}.bin", dtype=np.uint16, mode="r")
        if len(mm) != rec["tokens"]:
            raise SystemExit(f"ladder: part {i} has {len(mm)} tokens, its record {rec['tokens']}")
        self.recs[i] = {k: rec[k] for k in ("index", "file", "file_sha256", "sha256", "tokens", "documents")} | {
            "dropped": len(rec["dropped"])}
        return mm

    def parts_read(self, n: int) -> list[dict]:
        """The records of the parts in the blocks that windows [0, n) draw from."""
        while self.bounds[-1] < n:
            self.extend()
        k = int(np.searchsorted(self.bounds, n - 1, side="right"))
        return [self.recs[i] for i in range(k * self.group)]

    def extend(self) -> None:
        k = len(self.blocks)
        mms = [self.part(i) for i in range(k * self.group, (k + 1) * self.group)]
        counts = np.array([(len(m) - 1) // self.ctx for m in mms], dtype=np.int64)
        perm = np.random.default_rng([self.seed, k]).permutation(int(counts.sum()))
        self.blocks.append((mms, np.concatenate([[0], np.cumsum(counts)]), perm))
        self.bounds.append(self.bounds[-1] + int(counts.sum()))

    def rows(self, start: int, n: int) -> np.ndarray:
        while self.bounds[-1] < start + n:
            self.extend()
        out = np.empty((n, self.ctx + 1), dtype=np.int64)
        picks = []
        for j, w in enumerate(range(start, start + n)):
            k = int(np.searchsorted(self.bounds, w, side="right") - 1)
            _, cum, perm = self.blocks[k]
            idx = int(perm[w - self.bounds[k]])
            p = int(np.searchsorted(cum, idx, side="right") - 1)
            picks.append((k, p, idx - int(cum[p]), j))
        for k, p, off, j in sorted(picks):
            out[j] = self.blocks[k][0][p][off * self.ctx: off * self.ctx + self.ctx + 1]
        return out


def pilot_windows(root: Path, ctx: int = CTX) -> np.ndarray:
    toks = np.fromfile(root / "val.bin", dtype=np.uint16).astype(np.int64)
    n = (len(toks) - 1) // ctx
    return np.stack([toks[i * ctx: i * ctx + ctx + 1] for i in range(n)])


# ---- evaluation ------------------------------------------------------------------------------------------------------------

class Logits:
    """What prepare.window_nats calls: `eval()` and `logits(idx)`."""

    def __init__(self, model: LM):
        self.model = model

    def eval(self):
        return self

    def logits(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)


@torch.no_grad()
def eval_search(model: LM, pool: tuple[np.ndarray, dict], rank: int, world: int, device) -> dict:
    toks, man = pool
    full, tail = prepare.windows(len(toks))
    starts = (full + tail)[rank::world]
    nats = torch.tensor(float(prepare.window_nats(Logits(model), toks, starts, micro=32, device=str(device)).sum()),
                        dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(nats)
    n = float(nats)
    return {"bpb": n / man["bytes"] / math.log(2), "nats_per_token": n / (len(toks) - 1)}


@torch.no_grad()
def eval_pilot(model: LM, rows: np.ndarray, rank: int, world: int, device) -> dict:
    mine = rows[rank::world]
    total = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(mine), 32):
        x = torch.from_numpy(mine[i: i + 32]).to(device)
        total += float(loss_fn(model, x)) * len(x)
    if world > 1:
        dist.all_reduce(total)
    return {"nats_per_token": float(total) / len(rows)}


# ---- one step's MACs -------------------------------------------------------------------------------------------------------

def count(a) -> int:
    from te_core.flops import Counter

    size = SIZES[a.size]
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build(size, dev)
    opts = optimizers(model, a)
    params = list(model.parameters())
    gen = torch.Generator(device="cpu").manual_seed(0)
    batch = torch.randint(0, 50257, (a.batch, CTX + 1), generator=gen).to(dev)
    c = Counter(digests=False)
    with c:
        for step in range(2):
            c.begin_step()
            set_rates(opts, 1.0, step, a)
            for i in range(0, a.batch, a.micro):
                x = c.batch(batch[i: i + a.micro])
                (loss_fn(model, x) * (len(x) / a.batch)).backward()
            if a.clip:
                torch.nn.utils.clip_grad_norm_(params, a.clip)
            for o in opts:
                o.step()
            for o in opts:
                o.zero_grad(set_to_none=True)
            c.end_step()
    rec = c.record()
    steps = rec["steps"]
    tensors = {n: [str(p.dtype), list(p.shape)] for n, p in model.named_parameters()}
    analytic = prepare.train_flops({"config": size.config(), "tensors": tensors}, {"full": sorted(tensors)},
                                   [{"op": "train", "tokens": a.batch * CTX}])["flops"]
    out = {"schema": "te-ladder/count/v1", "size": size.config(), "batch": a.batch, "micro": a.micro, "ctx": CTX,
           "tokens_per_step": a.batch * CTX, "macs_per_step": steps[1]["macs"],
           "steps": [{k: s[k] for k in ("macs", "matmul_macs", "elementwise_macs", "approx_macs", "tokens", "gpu_s")}
                     for s in steps],
           "unknown_ops": rec["unknown_ops"], "foreign_ops": rec["foreign_ops"],
           "analytic_train_flops_per_step": analytic, "two_macs_over_analytic": 2 * steps[1]["macs"] / analytic,
           "non_embedding_params": non_embedding(model), "parameters": sum(p.numel() for p in params),
           "optimizer": opt_card(a),
           "rule": "one training step of the single-process equivalent (the global batch's micro-batches accumulated, "
                   "clip, Muon and AdamW), eager, under te_core.flops.Counter; the second step, once the optimizers' "
                   "state exists"}
    if out["unknown_ops"] or out["foreign_ops"]:
        raise SystemExit(f"ladder count: ops the counter can't price: {out['unknown_ops']} {out['foreign_ops']}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "count.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    return 0


# ---- training ------------------------------------------------------------------------------------------------------------

def plan(a, n_ne: int, seqs: int) -> dict:
    tok = seqs * CTX

    def branch(t: float, f: float, registered: bool) -> dict:
        n = max(1, round(t * n_ne / tok))
        cool = max(1, round(f * n))
        return {"tpp": t, "steps": n, "cool": cool, "fork": n - cool, "registered": registered}

    reg = [float(x) for x in a.branches.split(",") if x.strip()]
    extra = [float(x) for x in a.extra.split(",") if x.strip()]
    bs = [branch(t, a.cool, True) for t in reg] + [branch(t, a.extra_cool, False) for t in extra if t not in reg]
    bs.sort(key=lambda b: (b["fork"], b["steps"]))
    top = max(bs, key=lambda b: b["steps"])
    if not top["registered"]:
        raise SystemExit("ladder: the longest branch must be a registered one")
    if a.warmup >= min(b["fork"] for b in bs):
        raise SystemExit(f"ladder: warmup {a.warmup} reaches a fork ({[b['fork'] for b in bs]})")
    return {"branches": bs, "top": top, "seqs": seqs, "tokens_per_step": tok}


def name_of(size: str, tpp: float) -> str:
    return f"{size}-t{tpp:g}"


def lr_mult(step: int, warm: int, fork: int, end: int) -> float:
    w = min(1.0, (step + 1) / warm) if warm else 1.0
    return w if step < fork else w * (end - step) / (end - fork)


def eval_grid(top_steps: int, n_ne: int, tok: int) -> set[int]:
    """Main-run evaluation steps: 6 a decade of tokens per parameter from 0.5, and every 5% of the run."""
    out = {round(top_steps * i / 20) for i in range(1, 20)}
    t = 0.5
    while (s := round(t * n_ne / tok)) < top_steps:
        if s > 0:
            out.add(s)
        t *= 10 ** (1 / 6)
    return out


def _clone_state(sd: dict) -> dict:
    return {"state": {k: {n: (v.clone() if torch.is_tensor(v) else v) for n, v in s.items()} for k, s in sd["state"].items()},
            "param_groups": [dict(g) for g in sd["param_groups"]]}


class Loss(torch.nn.Module):
    def __init__(self, m: LM):
        super().__init__()
        self.m = m

    def forward(self, x):
        return loss_fn(self.m, x)


def train(a) -> int:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    size = SIZES[a.size]
    if a.batch % (world * a.micro):
        raise SystemExit(f"ladder: --batch {a.batch} is not a multiple of {world} ranks x --micro {a.micro}")
    model = build(size, dev)
    n_ne = non_embedding(model)
    opts = optimizers(model, a)
    params = list(model.parameters())
    pl = plan(a, n_ne, a.batch)
    top = pl["top"]
    stream = Stream(a.data, a.group, a.seed)
    man = json.loads((a.data / "manifest.json").read_text()) if (a.data / "manifest.json").is_file() else None
    count_rec = json.loads(a.count.read_text()) if a.count else None
    if count_rec and (count_rec["batch"] != a.batch or count_rec["size"]["name"] != a.size):
        raise SystemExit("ladder: --count measured another size or batch")
    macs_step = count_rec["macs_per_step"] if count_rec else None
    pool = prepare.held_out(prepare.SEARCH_POOL, "gpt2") if a.eval == "search" else None
    pilot = pilot_windows(a.data) if a.eval == "pilot" else None
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapped = Loss(model)
    ddp = DDP(wrapped, device_ids=[local], gradient_as_bucket_view=True) if world > 1 else wrapped
    fwd = ddp if a.no_compile else torch.compile(ddp)
    per_rank = a.batch // world
    n_micro = per_rank // a.micro

    step, done, evals = 0, set(), []
    ck = a.work / "ckpt.pt"
    if ck.is_file():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if st["plan"] != pl or st["size"] != a.size or st.get("optimizer") != opt_card(a):
            raise SystemExit(f"ladder: {ck} is another run's (plan, size or optimizer differ)")
        model.load_state_dict(st["model"])
        for o, s in zip(opts, st["opts"]):
            o.load_state_dict(s)
        step, done, evals = st["step"], set(st["done"]), st["evals"]
        log(resumed=str(ck), step=step, done=sorted(done))

    def save(at: int) -> None:
        if rank == 0:
            tmp = ck.with_suffix(".tmp")
            torch.save({"plan": pl, "size": a.size, "optimizer": opt_card(a), "step": at, "done": sorted(done), "evals": evals,
                        "model": model.state_dict(), "opts": [o.state_dict() for o in opts]}, tmp)
            os.replace(tmp, ck)
        if world > 1:
            dist.barrier()

    def one_step(k: int, mult: float) -> torch.Tensor:
        set_rates(opts, mult, k, a)
        rows = stream.rows(k * a.batch + rank * per_rank, per_rank)
        x_all = torch.from_numpy(rows).pin_memory().to(dev, non_blocking=True)
        tot = torch.zeros((), device=dev)
        for i in range(n_micro):
            x = x_all[i * a.micro: (i + 1) * a.micro]
            last = i == n_micro - 1
            with (ddp.no_sync() if world > 1 and not last else nullcontext()):
                loss = fwd(x) / n_micro
                loss.backward()
            tot += loss.detach()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
        return tot

    def resync() -> None:
        if world > 1:
            for p in params:
                dist.broadcast(p.data, 0)

    def evaluate(tag: dict) -> dict:
        r = eval_search(model, pool, rank, world, dev) if pool is not None else (
            eval_pilot(model, pilot, rank, world, dev) if pilot is not None else {})
        n = tag["lineage_steps"]
        rec = {**tag, **r, "tokens": n * pl["tokens_per_step"], "tpp": n * pl["tokens_per_step"] / n_ne,
               "macs": macs_step * n if macs_step else None}
        evals.append(rec)
        log(eval=rec)
        if rank == 0:
            (a.out / "curve.json").write_text(json.dumps(evals, indent=1) + "\n")
        return rec

    def register(b: dict) -> None:
        if not b["registered"] or a.eval != "search":
            return
        name = name_of(a.size, b["tpp"])
        if rank == 0:
            theta = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
            with torch.device(dev):
                net = prepare.Net(size.config())
            net.load(theta)
            official = prepare.evaluate_bpb(net, prepare.SEARCH_POOL, "gpt2", device=str(dev))
            del net
            prov = {"recipe": RECIPE, "size": size.config(), "non_embedding_params": n_ne,
                    "tokens_per_param": b["tpp"], "steps": b["steps"], "fork_step": b["fork"],
                    "cooldown_steps": b["cool"], "warmup_steps": a.warmup, "batch_seqs": a.batch, "ctx": CTX,
                    "tokens": b["steps"] * pl["tokens_per_step"],
                    "schedule": f"linear warmup {a.warmup} steps, constant, linear to zero over the last {b['cool']} steps",
                    "optimizer": opt_card(a),
                    "data": {"stream": "ladder_data.py: FineWeb-Edu sample-100BT, GPT-2 tokens, sample-10BT 010/011/013's "
                                       "documents left out", "group": a.group, "seed": a.seed,
                             "windows": [0, b["steps"] * a.batch], "rule": "step k reads the stream's windows [k S, (k+1) S)",
                             "held_out": json.loads((a.data / "held-out.json").read_text()),
                             "parts": stream.parts_read(b["steps"] * a.batch)},
                    "c0": {"macs": macs_step * b["steps"] if macs_step else None, "macs_per_step": macs_step,
                           "counter": "te_core.flops.Counter (ladder.py count)"},
                    "search_pool": official, "run": os.environ.get("RESEARCH_RUN_ID", ""),
                    "commit": os.environ.get("LADDER_COMMIT") or os.environ.get("RESEARCH_SOURCE_SHA", "")}
            reg = prepare.register_start(name, theta, size.config(), "gpt2", prov, root=a.work / "starts")
            log(registered=name, digest=reg["digest"], bpb=official["bpb"])
        if world > 1:
            dist.barrier()

    def branch_run(b: dict) -> None:
        """The cooldown branch `b` from the current state (its fork), then back to that state."""
        saved_m = {k: v.detach().clone() for k, v in model.state_dict().items()}
        saved_o = [_clone_state(o.state_dict()) for o in opts]
        t0 = time.perf_counter()
        for k in range(b["fork"], b["steps"]):
            one_step(k, lr_mult(k, a.warmup, b["fork"], b["steps"]))
        resync()
        if a.eval != "none":
            evaluate({"phase": "annealed", "branch": name_of(a.size, b["tpp"]), "registered": b["registered"],
                      "step": b["steps"], "lineage_steps": b["steps"]})
        register(b)
        log(branch_done=name_of(a.size, b["tpp"]), s=round(time.perf_counter() - t0, 1))
        model.load_state_dict(saved_m)
        for o, s in zip(opts, saved_o):
            o.load_state_dict(s)

    grid = eval_grid(top["steps"], n_ne, pl["tokens_per_step"])
    forks: dict[int, list[dict]] = {}
    for b in pl["branches"]:
        if b is not top:
            forks.setdefault(b["fork"], []).append(b)
    log(plan=pl, non_embedding=n_ne, world=world, per_rank=per_rank, micro=a.micro, macs_per_step=macs_step,
        stream_tokens=man["tokens"] if man else None)
    if man and man["tokens"] < top["steps"] * pl["tokens_per_step"] * (1 + 1 / CTX):
        raise SystemExit(f"ladder: the stream has {man['tokens']} tokens, the top branch reads "
                         f"{top['steps'] * pl['tokens_per_step']}")
    if step == 0 and a.eval != "none":
        evaluate({"phase": "init", "step": 0, "lineage_steps": 0})
    t0 = last = ck_at = time.perf_counter()
    stop = min(a.stop_at or top["steps"], top["steps"])
    while step < stop:
        for b in forks.get(step, []):
            if name_of(a.size, b["tpp"]) not in done:
                save(step)
                branch_run(b)
                done.add(name_of(a.size, b["tpp"]))
                save(step)
        loss = one_step(step, lr_mult(step, a.warmup, top["fork"], top["steps"]))
        step += 1
        if step % 200 == 0:
            resync()
        if step % 50 == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            log(step=step, loss=round(float(loss), 4), lr=round(lr_mult(step - 1, a.warmup, top["fork"], top["steps"]), 4),
                tok_s=round(50 * pl["tokens_per_step"] / (now - last)), s=round(now - t0, 1))
            last = now
            flag = torch.tensor(float(now - ck_at > a.ckpt_min * 60), device=dev)
            if world > 1:
                dist.broadcast(flag, 0)
            if flag.item() and step < top["steps"]:
                save(step)
                ck_at = last = time.perf_counter()
        if step in grid and a.eval != "none" and step < top["steps"]:
            evaluate({"phase": "stable" if step <= top["fork"] else "main-decay", "step": step, "lineage_steps": step})
    if step < top["steps"]:
        save(step)
    if step >= top["steps"] and name_of(a.size, top["tpp"]) not in done:
        resync()
        if a.eval != "none":
            evaluate({"phase": "annealed", "branch": name_of(a.size, top["tpp"]), "registered": True,
                      "step": step, "lineage_steps": step})
        register(top)
        done.add(name_of(a.size, top["tpp"]))
        save(step)
    if rank == 0:
        summary = {"schema": "te-ladder/run/v1", "size": size.config(), "plan": pl, "non_embedding_params": n_ne,
                   "macs_per_step": macs_step, "optimizer": opt_card(a), "args": {k: str(v) for k, v in vars(a).items()},
                   "evals": evals, "done": sorted(done), "world": world, "wall_s": time.perf_counter() - t0}
        (a.out / "ladder.json").write_text(json.dumps(summary, indent=1) + "\n")
        if (a.work / "starts").is_dir():
            for d in sorted((a.work / "starts").iterdir()):
                shutil.copytree(d, a.out / "starts" / d.name, dirs_exist_ok=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("count", "train"), help="count reads only the size, batch and optimizer flags")
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--batch", type=int, required=True, help="sequences a step (global)")
    ap.add_argument("--micro", type=int, default=32, help="sequences a micro-step on one GPU")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mat-lr", type=float, default=0.02)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--mom-start", type=float, default=0.85)
    ap.add_argument("--mom-warmup", type=int, default=300, help="steps over which Muon's momentum rises (0: none)")
    ap.add_argument("--wd", type=float, default=0.0, help="Muon's decoupled weight decay (times its rate)")
    ap.add_argument("--embed-lr", type=float, default=0.006)
    ap.add_argument("--embed-wd", type=float, default=0.0)
    ap.add_argument("--norm-lr", type=float, default=0.006)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--data", type=Path, default=None, help="ladder_data.py's directory (train)")
    ap.add_argument("--work", type=Path, default=None, help="checkpoint and registered starts, persistent (train)")
    ap.add_argument("--count", type=Path, default=None, help="count's count.json (C0)")
    ap.add_argument("--branches", default="20,200", help="registered saturations (tokens per non-embedding parameter)")
    ap.add_argument("--extra", default="", help="saturations that only add points to F")
    ap.add_argument("--cool", type=float, default=0.4, help="a registered branch's cooldown share of its steps")
    ap.add_argument("--extra-cool", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--group", type=int, default=4, help="parts a block (the shuffle's span)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", choices=("search", "pilot", "none"), default="search")
    ap.add_argument("--ckpt-min", type=float, default=30.0)
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this many main-run steps, checkpointed (a throughput check, or a phase on fewer GPUs: the checkpoint resumes on any number of ranks)")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "count":
        return count(a)
    if a.data is None or a.work is None:
        ap.error("train needs --data and --work")
    return train(a)


if __name__ == "__main__":
    sys.exit(main())
