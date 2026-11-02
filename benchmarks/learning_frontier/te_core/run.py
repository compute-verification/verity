"""Run one submission inside the harness: the program trains from θ through a `Session`, and everything it computes
through torch is counted (flops.py).

    python -m te_core.run --program train.py --starts starts.json --start small --tier 0 --seed 0 --out OUT \\
        [--evaluator MODULE:FUNCTION] [--device cuda] [-- PROGRAM ARGS...]

The program is a Python file defining `train(te)`; `te` is the `Session`. It builds its model, loads θ and registers it
(`te.load(model)`), declares its step count (`te.plan(n)`), wraps each training step in `with te.step():` and may
name its output model (`te.output(fn)`, default: the registered model's weights) and write counterexample families
(`te.family(name, plan)`). A checkpoint is taken after every 5% of the planned steps: the output model's held-out
bits per byte from the fixed evaluator, uncounted. Tokens computed outside a step, or a run with no step, make the run
invalid.

`--starts` is the integrator's registry of starts, `{name: {path, sha256, val_bpb, ctx, vocab, forward_macs_per_token}}`;
c is `score.C_PER_FORWARD_TOKEN` times the start's `forward_macs_per_token` (measured on the registered model when the
registry lacks it). The evaluator is a function `(state_dict, split, device) -> bits per byte`.

Writes `OUT/run.json` (the count, the checkpoints, the flags, the program's and θ's digests), `OUT/families/*/` with each
family's `verified.json` (checked here, after training, on the run's device, from a live coin) and, with
`--keep-checkpoints` (tier 2), each checkpoint's weights. Torch compilation is disabled and CUDA graph replay refused:
both run work the counter can't see.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from . import VERSION
from .codecs import Plan
from .counterexample import SAMPLE_FINAL, SAMPLE_SEARCH, Family, check
from .flops import Counter, digest, flags, forward_macs_per_token
from .score import block_c

CHECKPOINTS = 20
#: a run past its tier's GPU budget by this factor is stopped at its next step
BUDGET_SLACK = 1.2


class BudgetExceeded(RuntimeError):
    pass


def theta_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode() + b"\0" + digest(state[k]).encode())
    return h.hexdigest()


def load_theta(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state, meta = (obj["state"], obj.get("meta", {})) if "state" in obj else (obj, {})
    return state, meta


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("RESEARCH_COMMIT", "")


class Session:
    """What a program sees. Everything but `step`'s body and the program's own code runs uncounted."""

    def __init__(self, theta: dict[str, torch.Tensor], out: Path, *, seed: int = 0, tier: int = 0, device: str = "cpu",
                 start: dict | None = None, evaluator=None, budget_s: float = 0.0, keep_checkpoints: bool = False,
                 digests: bool = True, args: list[str] | None = None, clock=time.perf_counter):
        self.theta, self.out = theta, Path(out)
        self.seed, self.tier, self.device = seed, tier, device
        self.start = start or {}
        self.evaluator, self.budget_s, self.keep = evaluator, budget_s, keep_checkpoints
        self.args = list(args or [])
        self.clock = clock
        self.counter = Counter(theta, digests=digests)
        self.model = None
        self.output_fn = None
        self.n_steps = 0
        self.cp_steps: set[int] = set()
        self.checkpoints: list[dict] = []
        self.families: list[Family] = []
        self.forward_macs_per_token = float(self.start.get("forward_macs_per_token") or 0)
        self._t0 = clock()
        self.out.mkdir(parents=True, exist_ok=True)

    # -- the program's API

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load θ into `model`, move it to the run's device and register it."""
        with self.counter.pause():
            model.load_state_dict(self.theta)
            model.to(self.device)
        return self.register(model)

    def register(self, model: torch.nn.Module) -> torch.nn.Module:
        self.counter.register(model.state_dict(keep_vars=True))
        self.model = model
        if not self.forward_macs_per_token:
            with self.counter.pause():
                ctx, vocab = int(self.start.get("ctx", 1024)), int(self.start.get("vocab", 50304))
                self.forward_macs_per_token = forward_macs_per_token(model, ctx, vocab, self.device)
            self.start["forward_macs_per_token_source"] = "measured on the registered model"
        return model

    def plan(self, n_steps: int) -> None:
        self.n_steps = int(n_steps)
        self.cp_steps = {max(0, math.ceil(self.n_steps * i / CHECKPOINTS) - 1) for i in range(1, CHECKPOINTS + 1)}

    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        return self.counter.tokens(batch)

    def output(self, fn) -> None:
        """`fn()` returns the output model's state dict (default: the registered model's)."""
        self.output_fn = fn

    def family(self, name: str, plan: Plan, uniform: bool = True) -> Family:
        fam = Family(self.out / "families", name, plan, uniform)
        self.families.append(fam)
        return fam

    def elapsed(self) -> float:
        return self.clock() - self._t0

    def time_left(self) -> float:
        return self.budget_s - self.elapsed() if self.budget_s else math.inf

    @contextmanager
    def step(self):
        if self.model is None:
            raise RuntimeError("register the model (te.load) before the first step")
        self.counter.begin_step()
        yield len(self.counter.steps)
        self.counter.end_step()
        index = len(self.counter.steps) - 1
        if index in self.cp_steps:
            self._checkpoint(index)
        if self.budget_s and self.elapsed() > BUDGET_SLACK * self.budget_s:
            raise BudgetExceeded(f"{self.elapsed():.0f} s past the tier's {self.budget_s:.0f} s budget")

    # -- the harness's side

    def _state(self) -> dict[str, torch.Tensor]:
        state = self.output_fn() if self.output_fn else self.model.state_dict()
        return {k: v.detach() for k, v in state.items()}

    def _checkpoint(self, step: int) -> None:
        with self.counter.pause(), torch.no_grad():
            state = self._state()
            bpb = float(self.evaluator(state, "val", self.device)) if self.evaluator else None
            cp = {"step": step, "val_bpb": bpb, "elapsed_s": self.elapsed(), "output_sha256": theta_sha256(state)}
            if self.keep or step == max(self.cp_steps, default=-1):
                path = self.out / "checkpoints" / f"step{step:07d}.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save({k: v.cpu() for k, v in state.items()}, path)
                cp["path"] = str(path.relative_to(self.out))
            self.checkpoints.append(cp)

    def finish(self, meta: dict) -> dict:
        self.counter.finish()
        rec = self.counter.record()
        problems = []
        if not rec["steps"]:
            problems.append("no step")
        for where in ("setup", "tail"):
            if rec[where]["tokens"] or rec[where]["by"].get("fwd/updated"):
                problems.append(f"tokens or updated MACs outside a step ({where})")
        if rec["unseen_writes"]:
            problems.append(f"weights written outside the dispatcher's view: {rec['unseen_writes'][:3]}")
        c = block_c(self.forward_macs_per_token)
        rec |= meta | {
            "schema": "te-core/run/v1", "te_core": VERSION, "seed": self.seed, "tier": self.tier,
            "start": self.start, "theta_sha256": theta_sha256(self.theta), "device": self.device,
            "forward_macs_per_token": self.forward_macs_per_token, "c": c, "planned_steps": self.n_steps,
            "checkpoints": self.checkpoints, "flags": flags(rec["totals"], float(meta.get("peak_macs_per_s", 0))),
            "invalid": problems,
        }
        n = SAMPLE_FINAL if self.tier >= 2 else SAMPLE_SEARCH
        verified = []
        with self.counter.pause():
            for fam in self.families:
                v = check(fam.dir, rec, self.theta, c, n_sample=n, device=self.device)
                (fam.dir / "verified.json").write_text(json.dumps(v.to_json(), indent=1) + "\n")
                verified.append(v.to_json())
        rec["families"] = verified
        (self.out / "run.json").write_text(json.dumps(rec) + "\n")
        return rec


#: names a program may not contain: the harness's internals, dispatch-mode control, and routes to GPU or host kernels
#: the counter doesn't see
FORBIDDEN = ("counter", "pause", "_python_dispatch", "_pop_mode", "_disable_current_modes", "TorchDispatchMode",
             "_set_dispatch_mode", "CUDAGraph", "cuda.graph", "torch.compile", "_dynamo", "triton", "ctypes", "cupy",
             "numba", "jax", "torch.utils.cpp_extension", "load_inline", "subprocess", "multiprocessing", "os.system")


def screen(source: str) -> list[str]:
    return [name for name in FORBIDDEN if name in source]


def load_program(path: Path):
    path = Path(path).resolve()
    bad = screen(path.read_text())
    if bad:
        raise SystemExit(f"{path}: refused, it names {bad} (te_core/README.md, the program contract)")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("te_program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "train"):
        raise SystemExit(f"{path}: defines no train(te)")
    return mod


def load_evaluator(name: str):
    if not name or name == "none":
        return None
    mod, _, fn = name.partition(":")
    return getattr(importlib.import_module(mod), fn)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--starts", type=Path, help="the starts registry (JSON)")
    ap.add_argument("--start", default="", help="a start's name in --starts")
    ap.add_argument("--theta", type=Path, help="θ's file, instead of --starts/--start")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--evaluator", default="none", help="MODULE:FUNCTION, (state, split, device) -> bits per byte")
    ap.add_argument("--budget-minutes", type=float, default=0.0)
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--peak-macs-per-s", type=float, default=0.0, help="the device's peak, for the unexplained-GPU flag")
    ap.add_argument("--meta", default="{}", help="JSON merged into the record (agent, tags, parent: launch.py's)")
    ap.add_argument("program_args", nargs=argparse.REMAINDER)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    torch.cuda.graphs.CUDAGraph.replay = lambda self: (_ for _ in ()).throw(
        RuntimeError("te_core: CUDA graph replay runs work the counter can't see"))
    if a.theta:
        start = {"name": a.start or a.theta.name, "path": str(a.theta)}
        path = a.theta
    else:
        starts = json.loads(a.starts.read_text())
        start = dict(starts[a.start], name=a.start)
        path = Path(start["path"])
        if not path.is_absolute():
            path = a.starts.parent / path
    theta, _ = load_theta(path)
    if start.get("sha256") and theta_sha256(theta) != start["sha256"]:
        raise SystemExit(f"{path}: θ has sha256 {theta_sha256(theta)}, the registry says {start['sha256']}")
    torch.manual_seed(a.seed)
    args = a.program_args[1:] if a.program_args[:1] == ["--"] else a.program_args
    te = Session(theta, a.out, seed=a.seed, tier=a.tier, device=a.device, start=start,
                 evaluator=load_evaluator(a.evaluator), budget_s=60 * a.budget_minutes,
                 keep_checkpoints=a.keep_checkpoints, args=args)
    program = load_program(a.program)
    meta = json.loads(a.meta) | {"program": a.program.name, "program_sha256": file_sha256(a.program),
                                 "commit": git_commit(), "hardware": torch.cuda.get_device_name() if
                                 torch.cuda.is_available() else "cpu", "peak_macs_per_s": a.peak_macs_per_s}
    stopped = ""
    with te.counter:
        try:
            program.train(te)
        except BudgetExceeded as e:
            stopped = str(e)
    if stopped:
        meta["stopped"] = stopped
    rec = te.finish(meta)
    print(json.dumps({"macs": rec["totals"]["macs"], "steps": len(rec["steps"]), "checkpoints": len(rec["checkpoints"]),
                      "flags": rec["flags"], "invalid": rec["invalid"], "families": [
                          {k: f[k] for k in ("family", "passed", "failed", "covers")} for f in rec["families"]]}))
    return 1 if rec["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
