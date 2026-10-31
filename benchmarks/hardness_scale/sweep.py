"""Run presets of train.py configs on this process's GPU, claiming each config from a queue several workers share.

A worker is one `research run --gpus 1`. It walks its presets' configs in order and runs every config whose claim it wins:
the claim is `mkdir QUEUE/<name>` (atomic), so N workers on one node split a preset between them in its order (a preset
lists its longest configs first).
Each config runs as a child process. Results: $OUT/runs/<name>.json and $OUT/logs/<name>.log, a copy of both under
--shared (outside the run dir, for node-side reads), and a failure record $OUT/failed/<name>.json. The worker's stdout
carries one line per eval and per finished config. A claim stays when its config fails: rerun it under a new name.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import nonemb_params  # noqa: E402

SIZES = {  # d, sequences a step, sequences a forward
    "9m": {"d": 256, "batch": 128, "micro": 64},
    "20m": {"d": 384, "batch": 128, "micro": 64},
    "60m": {"d": 640, "batch": 256, "micro": 64},
    "125m": {"d": 896, "batch": 256, "micro": 32},
}

ARMS = {
    "base": {},
    "bf16": {"qat": False},
    "noise": {"arr": "noise"},
    "m-mlp-r8": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 8},
    "m-mlp-r4": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 4},
    "m-all-r8": {"arr": "matrix", "where": "all", "rank_frac": 1 / 8},
    "m-all-r4": {"arr": "matrix", "where": "all", "rank_frac": 1 / 4},
    "m-mlph-r8": {"arr": "matrix", "where": "mlp", "rank_frac": 1 / 8, "block_frac": 0.5},
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


def t1_20m():
    """Tier 1 at 20M: the baseline's lr and token ladder, every arrangement at several κ, lr and seed checks."""
    lr = 3e-3
    out = [cfg("20m", "base", x) for x in (1.5e-3, 3e-3, 6e-3, 1.2e-2)]
    out += [cfg("20m", "base", lr, dx) for dx in (2, 0.5, 0.25, 0.125)]
    out += [cfg("9m", "base", lr * 1.5, dx) for dx in (1, 2)]
    out += [cfg("20m", a, lr) for a in ("m-mlp-r8", "m-all-r8", "res-f33", "res-f50", "feat-s50", "feat-s100")]
    out += [cfg("20m", a, lr) for a in ("m-mlp-r4", "m-all-r4", "m-mlph-r8", "res-f17", "res-f25", "res-f67",
                                        "feat-s25", "feat-s75", "noise", "bf16")]
    out += [cfg("20m", "m-mlp-r8", lr * 2), cfg("20m", "res-f33", lr * 2), cfg("20m", "res-f33", lr / 2),
            cfg("20m", "feat-s50", lr * 2)]
    out += [cfg("20m", a, lr, seed=1) for a in ("base", "m-mlp-r8", "res-f33")]
    return out


PRESETS = {"smoke": smoke, "t1-20m": t1_20m}


def run_one(c, out: Path, shared: Path, data: str | None):
    name = c["name"]
    for d in ("runs", "logs", "failed"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (shared / "logs").mkdir(parents=True, exist_ok=True)
    (shared / "runs").mkdir(parents=True, exist_ok=True)
    live = shared / "logs" / f"{name}.log"
    res = out / "runs" / f"{name}.json"
    cmd = [sys.executable, str(HERE / "train.py"), "--config", json.dumps(c), "--out", str(res)]
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
        shutil.copy(res, shared / "runs" / f"{name}.json")
        print(f"done {name}", flush=True)
        return True
    tail = live.read_text().splitlines()[-40:]
    (out / "failed" / f"{name}.json").write_text(json.dumps({"name": name, "rc": rc, "cfg": c, "tail": tail}, indent=1))
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
    cfgs = [c for p in a.preset.split(",") for c in PRESETS[p]()]
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["name"] in keep]
    names = [c["name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate config names")
    if a.list:
        for c in cfgs:
            print(c["name"], c["tokens"])
        return
    out, shared, queue = Path(a.out), Path(a.shared), Path(a.queue)
    queue.mkdir(parents=True, exist_ok=True)
    ran = failed = 0
    for c in cfgs:
        try:
            (queue / c["name"]).mkdir()
        except FileExistsError:
            continue
        ok = run_one(c, out, shared, None if c.get("synthetic") else a.data)
        ran += 1
        failed += not ok
    print(f"worker done: {ran} run, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
