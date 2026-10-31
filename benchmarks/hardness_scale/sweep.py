"""Run presets of train.py configs on this process's GPU, claiming each config from a queue several workers share.

A worker is one `research run --gpus N` through the GPU slots, and claims only configs of N GPUs (`gpus`, torchrun past
one). It walks its presets' configs in order and runs every config
whose claim it wins: the claim is `mkdir QUEUE/<name>` (atomic), so N workers on one node split a preset between them in its
order. A preset is a function of the results already under --shared, so a config whose lr is picked from earlier runs appears
once they finish; it re-resolves before every claim. With nothing left to claim, a worker waits while its presets say more
may come, and exits otherwise.
Each config runs as a child process. Results: $OUT/runs/<name>.json and $OUT/logs/<name>.log, a copy of both under
--shared (outside the run dir, for node-side reads), and a failure record $OUT/failed/<name>.json (and under --shared, where
a preset reads it as an infinite loss). The worker's stdout carries one line per eval and per finished config. A claim stays
when its config fails: rerun it under a new name.

A worker hands its GPU over: after each config, and while it waits, it exits when anyone waits for a GPU (another lane's
job, or newer workers of this sweep, which take over the unclaimed configs).
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools" / "verity" / "check"))
import slot  # noqa: E402
from train import nonemb_params  # noqa: E402

POLL_S = 60

SIZES = {  # d, sequences a step, sequences a forward
    "9m": {"d": 256, "batch": 128, "micro": 64},
    "20m": {"d": 384, "batch": 128, "micro": 64},
    "60m": {"d": 640, "batch": 256, "micro": 64},
    "125m": {"d": 896, "batch": 256, "micro": 32},
    "350m": {"d": 1536, "batch": 256, "micro": 32},
}
TEACHER = "HuggingFaceTB/SmolLM2-1.7B"

ARMS = {
    "base": {},
    "bf16": {"qat": False},
    "noise": {"arr": "noise"},
    "m-mlp-r8": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 8},
    "m-mlp-r4": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 4},
    "m-all-r8": {"arr": "matrix", "where": "all", "rank_frac": 1 / 8},
    "m-all-r4": {"arr": "matrix", "where": "all", "rank_frac": 1 / 4},
    "m-mlph-r8": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 8, "block_frac": 0.5},
    "m-mlpq-r8": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 8, "block_frac": 0.25},
    "res-f17": {"arr": "reservoir", "frozen_frac": 1 / 6},
    "res-f25": {"arr": "reservoir", "frozen_frac": 1 / 4},
    "res-f33": {"arr": "reservoir", "frozen_frac": 1 / 3},
    "res-f50": {"arr": "reservoir", "frozen_frac": 1 / 2},
    "res-f67": {"arr": "reservoir", "frozen_frac": 2 / 3},
    "feat-s25": {"arr": "feature", "feature_share": 0.25},
    "feat-s50": {"arr": "feature", "feature_share": 0.5},
    "feat-s75": {"arr": "feature", "feature_share": 0.75},
    "feat-s100": {"arr": "feature", "feature_share": 1.0},
}


def tokens_d(size):
    return 20 * nonemb_params(SIZES[size]["d"])


def cfg(size, arm, lr, dx=1.0, seed=0, **extra):
    s = SIZES[size]
    lr = float(f"{lr:.3g}")
    name = f"{size}-{arm}-lr{lr:g}" + (f"-Dx{dx:g}" if dx != 1 else "") + (f"-s{seed}" if seed else "")
    for k, v in extra.items():
        name += f"-{k}{v:g}" if isinstance(v, (int, float)) else f"-{k}{v}"
    tokens = int(tokens_d(size) * dx) // (s["batch"] * 1024) * (s["batch"] * 1024)
    return {"name": name, **s, "tokens": tokens, "lr": lr, "seed": seed, **ARMS[arm], **extra}


def smoke():
    """Synthetic tokens, 40 steps: compile, memory and throughput per size and arrangement."""
    out = []
    for size, micro in (("20m", 64), ("60m", 64), ("125m", 32), ("125m", 64)):
        for arm in ("base", "m-all-r4", "res-f50", "feat-s50"):
            c = cfg(size, arm, 3e-3, synthetic=True)
            c.update({"name": f"smoke-{size}-{arm}-mb{micro}", "micro": micro, "batch": micro * 2, "max_steps": 40,
                      "evals": 2, "eval_seqs": micro, "final_eval_seqs": micro * 2})
            out.append(c)
    return out


def ddp_check():
    """Two GPUs: 100 real steps of the 20M baseline (no noise), whose logged train losses (steps 32, 64, 96) should match
    the one-GPU 20m-base-lr0.003's (same schedule and data order), and 350M's throughput and memory for the baseline, a
    matrix arm and distillation."""
    short = {"evals": 2, "eval_seqs": 64, "final_eval_seqs": 128, "ablate": False}
    out = [{**cfg("20m", "base", 3e-3, gpus=2), **short, "max_steps": 100}]
    for arm, extra in (("base", {}), ("m-mlp-r8", {}), ("base", {"teacher": TEACHER, "micro": 16})):
        c = cfg("350m", arm, 2e-3, synthetic=True, gpus=2)
        c.update({**short, **extra, "max_steps": 30, "eval_seqs": c["micro"] if not extra else 16,
                  "final_eval_seqs": 128, "name": c["name"] + ("-kd" if extra else "")})
        out.append(c)
    return out


LR20 = (7.5e-4, 1.5e-3, 3e-3, 6e-3, 1.2e-2, 2.4e-2)
#: two κ per arrangement, matched across them: κ_lin ≈ 1/6 (m-mlpq-r8, res-f17, feat-s50), ≈ 1/3 (m-mlph-r8, res-f33,
#: feat-s100), and the higher m-mlp-r8 (≈ 0.6) and res-f50
MATCHED = ("m-mlp-r8", "m-mlph-r8", "m-mlpq-r8", "res-f17", "res-f33", "res-f50", "feat-s50", "feat-s100")
#: MATCHED in 125M's claim order: the best arrangements at κ_lin ≈ 1/3 and 1/6 first, feature (the worst at 20M) last
T1_125 = ("res-f33", "m-mlph-r8", "res-f17", "m-mlpq-r8", "m-mlp-r8", "res-f50", "feat-s50", "feat-s100")


def lr_sweep(size, lrs, res):
    """A size's Dx1 seed-0 baselines over the geometric grid lrs, extended a step past whichever end wins, and the best lr
    (None until each has finished)."""
    lrs = sorted(lrs)
    q = lrs[1] / lrs[0]
    while True:
        cs = [cfg(size, "base", x) for x in lrs]
        names = [c["name"] for c in cs]
        if any(n not in res for n in names):
            return cs, None
        i = min(range(len(cs)), key=lambda k: res[names[k]])
        if len(lrs) >= 9 or 0 < i < len(lrs) - 1:
            return cs, cs[i]["lr"]
        lrs = [lrs[0] / q] + lrs if i == 0 else lrs + [lrs[-1] * q]


def t1(res):
    """Tier 1 in claim order, and whether configs may still appear. 20M sweeps the baseline's lr (LR20) and picks lr20; 60M
    runs lr20·(384/640)^p for p ∈ {1, ½, 0} and picks lr60 among them, whose exponent p60 sets 125M's lr20·(384/896)^p60,
    checked a half-step of p either side. p = −½ and −1 at 60M are baseline checks only: 60M's optimum is flat from p = ½ to
    −½ (3.2402, 3.2403, 3.2358), and an lr rising with width is not followed on a 0.004 edge. Every other config of a size
    runs at its picked lr. After 60M's arms, the order puts the 125M point of the trend first: 125M's baseline at D with
    60M's two best arms at the higher lr, 125M's arms (T1_125), 125M's D/2, then 20M's lr and seed checks, 60M's, and
    125M's lr check last (only Tier 2 reads its pick)."""
    out, pending = [], False
    s20, lr20 = lr_sweep("20m", LR20, res)
    out += s20
    if lr20 is None:
        return out, True
    s60 = [cfg("60m", "base", lr20 * (384 / 640) ** p) for p in (1, 0.5, 0)]
    out += s60
    lr60 = None if any(c["name"] not in res for c in s60) else min(s60, key=lambda c: res[c["name"]])["lr"]
    c125, checks60, s125 = [], [], []
    if lr60 is None:
        pending = True
    else:
        lr125 = lr20 * (384 / 896) ** (math.log(lr60 / lr20) / math.log(384 / 640))
        out.append(cfg("125m", "base", lr125, 2))
        out += [cfg("60m", "base", lr60, 2)] + [cfg("60m", a, lr60) for a in ("m-all-r8",) + MATCHED]
        out += [cfg("60m", "base", lr60, 0.5), cfg("60m", "noise", lr60)]
        hi = lr20 * (384 / 640) ** -0.5
        top = cfg("60m", "base", lr20 * 640 / 384)
        c125 = [cfg("125m", "base", lr125)] + [cfg("60m", a, hi) for a in ("res-f33", "m-mlph-r8")]
        c125 += [cfg("125m", a, lr125) for a in T1_125] + [cfg("125m", "base", lr125, 0.5)]
        checks60 = [cfg("60m", "base", lr60, seed=1), cfg("60m", "base", hi)]
        checks60.append({**top, "name": top["name"] + "-r2"})  # the first 5e-3 run was cancelled
        s125, best125 = lr_sweep("125m", [lr125 * (384 / 896) ** p for p in (-0.5, 0, 0.5)], res)
        pending |= best125 is None
    rest = [cfg("20m", "base", lr20, dx) for dx in (2, 0.5, 0.25, 0.125)]
    rest += [cfg("9m", "base", lr20 * 1.5, dx) for dx in (1, 2)] + [cfg("9m", "base", lr20)]
    rest += [cfg("20m", a, lr20) for a in MATCHED]
    rest += [cfg("20m", a, lr20) for a in ("m-mlp-r4", "m-all-r8", "m-all-r4", "res-f25", "res-f67", "feat-s25",
                                           "feat-s75", "noise", "bf16")]
    rest += [cfg("20m", a, lr20 * k) for a, k in (("m-mlp-r8", 2), ("m-mlph-r8", 2), ("res-f33", 2), ("res-f33", 0.5),
                                                  ("feat-s50", 2))]
    rest += [cfg("20m", a, lr20, seed=1) for a in ("base", "m-mlp-r8", "res-f33")]
    return out + c125 + rest + checks60 + s125, pending


T2_ARMS = ("res-f33", "m-mlph-r8")  # the two best arrangements at matched κ_lin = 1/3 in Tier 1


def kd(c):
    return {**c, "teacher": TEACHER, "micro": 16, "name": c["name"] + "-kd"}


def t2(res):
    """Tier 2 at 350M, one GPU a run, longest first: the baseline at D, D/2, D/4, the two best arrangements at D (and the
    first at D/4), and distillation from TEACHER of the baseline and the first arrangement at D/4. Its lr is lr20·(384/1536)^p
    with p the exponent of 125M's best checked lr; nothing appears until that check is in."""
    s20, lr20 = lr_sweep("20m", LR20, res)
    if lr20 is None:
        return [], True
    s60 = [cfg("60m", "base", lr20 * (384 / 640) ** p) for p in (1, 0.5, 0)]
    if any(c["name"] not in res for c in s60):
        return [], True
    lr60 = min(s60, key=lambda c: res[c["name"]])["lr"]
    lr125 = lr20 * (384 / 896) ** (math.log(lr60 / lr20) / math.log(384 / 640))
    _, best125 = lr_sweep("125m", [lr125 * (384 / 896) ** p for p in (-0.5, 0, 0.5)], res)
    if best125 is None:
        return [], True
    lr = lr20 * (384 / 1536) ** (math.log(best125 / lr20) / math.log(384 / 896))
    a, b = T2_ARMS
    out = [cfg("350m", "base", lr), cfg("350m", a, lr), cfg("350m", b, lr)]
    out += [kd(cfg("350m", "base", lr, 0.25)), kd(cfg("350m", a, lr, 0.25))]
    out += [cfg("350m", "base", lr, 0.5), cfg("350m", "base", lr, 0.25), cfg("350m", a, lr, 0.25)]
    return out, False


PRESETS = {"smoke": lambda res: (smoke(), False), "ddp-check": lambda res: (ddp_check(), False), "t1": t1, "t2": t2}


def finished(shared: Path) -> dict:
    """Each finished config's final loss under shared; a failed or diverged one's is infinite."""
    res = {p.stem: math.inf for p in (shared / "failed").glob("*.json")}
    for p in (shared / "runs").glob("*.json"):
        try:
            v = json.loads(p.read_text())["final_val"]
        except (OSError, ValueError, KeyError):
            continue
        res[p.stem] = v if math.isfinite(v) else math.inf
    return res


def resolve(presets, shared: Path):
    """The presets' configs in order (first of each name), and whether more may appear."""
    res = finished(shared)
    out, seen, pending = [], set(), False
    for p in presets:
        cs, more = PRESETS[p](res)
        pending |= more
        for c in cs:
            if c["name"] not in seen:
                seen.add(c["name"])
                out.append(c)
    return out, pending


def someone_waits() -> str | None:
    """Who waits for a GPU, if anyone: a queued gpu-lease request (a wait file its waiter holds locked, naming it) or a
    waiter in the GPU slots' line."""
    for f in Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease")).glob("wait.*"):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                return f"gpu-lease {f.name}: {f.read_text(errors='replace').strip()[:200]}"
            except OSError:
                return f"gpu-lease {f.name}"
        finally:
            os.close(fd)
    slots = Path(os.environ.get("CHECK_SLOTS", "/workspace/research/locks/gpu/slots"))
    line = slot.waiting(slots.parent / slot.LINE, datetime.now(timezone.utc))
    return f"slot line: {slot.waiter_text(line[0])}" if line else None


def wait_for_data(data):
    meta = Path(data) / "meta.json"
    told = False
    while True:
        try:
            if json.loads(meta.read_text()).get("complete"):
                return
        except (OSError, ValueError):
            pass
        if not told:
            print(f"waiting for {meta} to be complete", flush=True)
            told = True
        time.sleep(POLL_S)


def run_one(c, out: Path, shared: Path, data: str | None):
    name = c["name"]
    for d in ("runs", "logs", "failed"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (shared / "logs").mkdir(parents=True, exist_ok=True)
    (shared / "runs").mkdir(parents=True, exist_ok=True)
    live = shared / "logs" / f"{name}.log"
    res = out / "runs" / f"{name}.json"
    n = c.get("gpus", 1)
    launch = [sys.executable] + (["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}"] if n > 1 else [])
    cmd = launch + [str(HERE / "train.py"), "--config", json.dumps(c), "--out", str(res)]
    if data:
        cmd += ["--data", data]
    print(f"start {name}", flush=True)
    with open(live, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            log.flush()
            if '"val"' in line or line.startswith(("final", "Traceback")) or "Error" in line:
                print(f"{name} {line.rstrip()}", flush=True)
        rc = p.wait()
    shutil.copy(live, out / "logs" / f"{name}.log")
    if rc == 0 and res.exists():
        shutil.copy(res, shared / "runs" / f".{name}.tmp")
        (shared / "runs" / f".{name}.tmp").rename(shared / "runs" / f"{name}.json")
        print(f"done {name}", flush=True)
        return True
    tail = live.read_text().splitlines()[-40:]
    rec = json.dumps({"name": name, "rc": rc, "cfg": c, "tail": tail}, indent=1)
    (out / "failed" / f"{name}.json").write_text(rec)
    (shared / "failed").mkdir(parents=True, exist_ok=True)
    tmp = shared / "failed" / f".{name}.tmp"
    tmp.write_text(rec)
    tmp.rename(shared / "failed" / f"{name}.json")
    print(f"FAILED {name} rc={rc}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help="comma-separated, run in order")
    ap.add_argument("--queue", required=True, help="the claim directory the node's workers share")
    ap.add_argument("--shared", required=True, help="results and live logs outside the run dir")
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--data", default=os.environ.get("HSC_DATA"))
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    presets = a.preset.split(",")
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown presets {unknown}; known: {sorted(PRESETS)}")
    keep = set(a.only.split(",")) if a.only else None
    out, shared, queue = Path(a.out or "."), Path(a.shared), Path(a.queue)
    if a.list:
        cfgs, pending = resolve(presets, shared)
        res = finished(shared)
        for c in cfgs:
            if keep is None or c["name"] in keep:
                print(c["name"], c["tokens"], "done" if c["name"] in res else "claimed" if (queue / c["name"]).exists() else "")
        print("more may come" if pending else "complete")
        return
    queue.mkdir(parents=True, exist_ok=True)
    gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    ran = failed = 0
    waited = False
    while True:
        cfgs, pending = resolve(presets, shared)
        c = None
        for x in cfgs:
            if (keep is not None and x["name"] not in keep) or x.get("gpus", 1) != gpus:
                continue
            if not x.get("synthetic") and not waited:
                wait_for_data(a.data)
                waited = True
            try:
                (queue / x["name"]).mkdir()
            except FileExistsError:
                continue
            c = x
            break
        if c is not None:
            ok = run_one(c, out, shared, None if c.get("synthetic") else a.data)
            ran += 1
            failed += not ok
        elif not pending:
            break
        if who := someone_waits():
            print(f"yielding to {who}", flush=True)
            break
        if c is None:
            time.sleep(POLL_S)
    print(f"worker done: {ran} run, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
