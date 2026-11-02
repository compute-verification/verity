"""Run one submission inside the harness: the program trains from θ through a `Session`, and everything it computes
through torch is counted (flops.py).

    python -m te_core.run --program train.py --starts starts.json --start small --tier 0 --seed 0 --out OUT \\
        [--evaluator MODULE:FUNCTION] [--device cuda] [-- PROGRAM ARGS...]

The program is a Python file defining `train(te)`; `te` is the `Session`. It builds its model, loads θ and registers it
(`te.load(model)`), declares its step count (`te.plan(n)`), wraps each training step in `with te.step():` and may
name its output model (`te.output(fn)`, default: the registered model's weights) and write counterexample families
(`te.family(name, plan)`). A checkpoint is taken after every 5% of the planned steps: the output model's held-out
bits per byte from the fixed evaluator, uncounted (`--script`: the program's E, written for a separate evaluation job,
cw.py). Tokens computed outside a step, or a run with no step, make the run invalid.

`--starts` is the integrator's registry of starts, `{name: {path, sha256, val_bpb, ctx, vocab, forward_macs_per_token}}`;
c is `score.C_PER_FORWARD_TOKEN` times the start's `forward_macs_per_token` (measured on the registered model when the
registry lacks it). The evaluator is a function `(state_dict, split, device) -> bits per byte`, or None for a split it
keeps sealed (covert_witness's `test`).

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
import re
import resource
import runpy
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
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(str(path)), {}
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
                 digests: bool = True, args: list[str] | None = None, clock=time.perf_counter,
                 families: Path | None = None):
        self.theta, self.out = theta, Path(out)
        self.families_root = Path(families) if families else self.out / "families"
        self.seed, self.tier, self.device = seed, tier, device
        self.start = start or {}
        self.evaluator, self.budget_s, self.keep = evaluator, budget_s, keep_checkpoints
        self.args = list(args or [])
        self.clock = clock
        self.counter = Counter(theta, digests=digests)
        self.bound = bool(theta)
        self.model = None
        self.output_fn = None
        self.n_steps = 0
        self.budget_macs = 0.0
        self.explicit = False
        self.cp_steps: set[int] = set()
        self.marks_done = 0
        self.checkpoints: list[dict] = []
        self.families: list[Family] = []
        self.forward_macs_per_token = float(self.start.get("forward_macs_per_token") or 0)
        self._t0 = clock()
        self.out.mkdir(parents=True, exist_ok=True)

    def bind(self, theta: dict[str, torch.Tensor], start: dict, model: torch.nn.Module,
             named: dict[str, torch.Tensor], extra: dict[str, torch.Tensor] | None = None) -> None:
        """θ known only once a standalone program loads its start: `extra` (the model's buffers) is checked like θ but
        isn't part of the output model."""
        if self.bound:
            raise RuntimeError("te_core: one start per run")
        self.theta = theta
        self.start |= start
        self.counter.theta_cpu = dict(theta) | dict(extra or {})
        with self.uncounted():
            self.counter.register(named)
        self.bound = True
        self._measure(model)

    def _measure(self, model: torch.nn.Module) -> None:
        if not self.forward_macs_per_token:
            with self.counter.pause():
                ctx, vocab = int(self.start.get("ctx", 1024)), int(self.start.get("vocab", 50304))
                self.forward_macs_per_token = forward_macs_per_token(model, ctx, vocab, self.device)
            self.start["forward_macs_per_token_source"] = "measured on the registered model"

    @contextmanager
    def uncounted(self):
        """The harness's own work (the fixed loaders, encoders, decoder and evaluator): neither counted nor timed, and
        outside the dispatch mode. It must not write the program's tensors."""
        with self.counter.pause(), torch.utils._python_dispatch._disable_current_modes():
            yield

    def in_steps(self) -> bool:
        return self.counter.cur is not self.counter.setup

    def current_step(self) -> int:
        return self.counter.cur.index

    def macs_so_far(self) -> int:
        c = self.counter
        return c.setup.macs + sum(s.macs for s in c.steps) + (c.cur.macs if c.cur is not c.setup else 0)

    def progress_bucket(self) -> int:
        """Which 5% of the run the step in progress is in: of the planned steps, else of the MAC budget, else its index."""
        if self.n_steps:
            return math.ceil((self.current_step() + 1) * CHECKPOINTS / self.n_steps)
        if self.budget_macs:
            return math.ceil(self.macs_so_far() * CHECKPOINTS / self.budget_macs)
        return self.current_step()

    def boundary(self) -> None:
        """A standalone program's step boundary (its next batch): end the step in progress if it computed on tokens."""
        if self.explicit or not self.bound:
            return
        c = self.counter
        if c.cur is c.setup:
            c.begin_step()
            return
        if not c.cur.tokens:
            return
        c.end_step()
        self._after_step(len(c.steps) - 1)

    def close(self) -> None:
        """A standalone program returned: its last step ends."""
        c = self.counter
        if not self.explicit and c.cur is not c.setup and c.cur.tokens:
            c.end_step()
            self._after_step(len(c.steps) - 1, budget=False)

    def _after_step(self, index: int, budget: bool = True) -> None:
        due = False
        if self.output_fn is not None or self.model is not None:
            if self.n_steps:
                due = index in self.cp_steps
            elif self.budget_macs:
                reached = min(CHECKPOINTS, int(self.macs_so_far() * CHECKPOINTS // self.budget_macs))
                due, self.marks_done = reached > self.marks_done, max(self.marks_done, reached)
        if due:
            self._checkpoint(index)
        if budget and self.budget_s and self.elapsed() > BUDGET_SLACK * self.budget_s:
            raise BudgetExceeded(f"{self.elapsed():.0f} s past the tier's {self.budget_s:.0f} s budget")

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
        self._measure(model)
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
        fam = Family(self.families_root, name, plan, uniform)
        self.families.append(fam)
        return fam

    def elapsed(self) -> float:
        return self.clock() - self._t0

    def time_left(self) -> float:
        return self.budget_s - self.elapsed() if self.budget_s else math.inf

    @contextmanager
    def step(self):
        if not self.bound:
            raise RuntimeError("register the model (te.load) before the first step")
        self.counter.begin_step()
        yield len(self.counter.steps)
        self.counter.end_step()
        self._after_step(len(self.counter.steps) - 1)

    def explicit_step(self):
        """`hooks.step()`: a standalone program's own boundaries, which replace the data stream's."""
        if self.counter.cur is not self.counter.setup and self.counter.cur.tokens and not self.explicit:
            self.counter.end_step()
            self._after_step(len(self.counter.steps) - 1)
        self.explicit = True
        return self.step()

    # -- the harness's side

    def _state(self) -> dict[str, torch.Tensor]:
        state = self.output_fn() if self.output_fn else self.model.state_dict()
        return {k: v.detach() for k, v in state.items()}

    def _checkpoint(self, step: int) -> None:
        with self.counter.pause(), torch.no_grad():
            state = self._state()
        self.checkpoint_state(step, state, {"source": "output"})

    def checkpoint_state(self, step: int, state: dict[str, torch.Tensor], extra: dict | None = None) -> dict:
        """Score `state` as the output model after step `step` (inclusive), uncounted."""
        with self.uncounted(), torch.no_grad():
            bpb = float(self.evaluator(state, "val", self.device)) if self.evaluator else None
            cp = {"step": step, "macs": self.macs_so_far(), "val_bpb": bpb, "elapsed_s": self.elapsed(),
                  "output_sha256": theta_sha256(state)} | (extra or {})
            test = self.evaluator(state, "test", self.device) if self.tier >= 2 and self.evaluator else None
            if test is not None:
                cp["test_bpb"] = float(test)
            if self.keep or (self.n_steps and step == max(self.cp_steps, default=-1)):
                path = self.out / "checkpoints" / f"step{step:07d}-{len(self.checkpoints):02d}.pt"
                path.parent.mkdir(exist_ok=True)
                torch.save({k: v.cpu() for k, v in state.items()}, path)
                cp["path"] = str(path.relative_to(self.out))
            self.checkpoints.append(cp)
        return cp

    def finish(self, meta: dict, problems: list[str] | None = None) -> dict:
        self.counter.finish()
        rec = self.counter.record()
        problems = list(problems or [])
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
                v = check(fam.dir, rec, self.theta, c, n_sample=n, device=self.device, plan=fam.plan)
                (fam.dir / "verified.json").write_text(json.dumps(v.to_json(), indent=1) + "\n")
                verified.append(v.to_json())
        rec["families"] = verified
        me = resource.getrusage(resource.RUSAGE_SELF)
        # taken last: the evaluation job counts every CPU second of the training job's tree beyond this one as unseen
        rec["totals"]["self_cpu_s"] = me.ru_utime + me.ru_stime
        (self.out / "run.json").write_text(json.dumps(rec) + "\n")
        return rec


#: names a program may not contain: the harness's internals, dispatch-mode control, and routes to GPU or host kernels
#: the counter doesn't see. `torch.compile` is allowed: TORCHDYNAMO_DISABLE, set before torch is imported, makes it
#: return the eager module, and `_dynamo` (which could turn it back on) is refused.
FORBIDDEN = (r"\.counter\b", r"\.pause\b", r"\buncounted\b", r"\bte_core\.(?!(hooks|codecs)\b)",
             r"(?m)from\s+te_core\s+import\s+(?!(hooks|codecs)(\s*,\s*(hooks|codecs))*\s*$)", r"\bhooks\.ACTIVE\b",
             r"\bAdapter\b", "_python_dispatch", "_pop_mode", "_disable_current_modes", "TorchDispatchMode",
             "_set_dispatch_mode", "CUDAGraph", r"cuda\.graph", "_dynamo", "TORCHDYNAMO", "triton", "ctypes", "cupy",
             "numba", "jax", r"torch\.utils\.cpp_extension", "load_inline", "subprocess", "multiprocessing",
             r"os\.system", r"os\.exec", r"os\.spawn", r"os\.posix_spawn", "addaudithook", r"sys\.modules",
             r"\bgc\b\.", "__import__", r"importlib")


def screen(source: str) -> list[str]:
    return [pat for pat in FORBIDDEN if re.search(pat, source)]


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
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--program", type=Path, help="a file defining train(te)")
    which.add_argument("--script", type=Path, help="a standalone covert_witness train.py, run as __main__ (cw.py)")
    ap.add_argument("--budget-macs", type=float, default=0.0,
                    help="--script: the MACs the 5%% checkpoints divide (default --budget-fraction of "
                         "prepare.TRAIN_FLOP_BUDGET / 2)")
    ap.add_argument("--budget-fraction", type=float, default=1.0, help="--script: see --budget-macs")
    ap.add_argument("--start-bpb", type=float, default=None, help="--script: the start's held-out bpb, else measured")
    ap.add_argument("--digests", choices=("on", "off"), default="on",
                    help="digest every updated weight value (flops.KeyedDigest, on the run's device; counterexample "
                         "checks need them)")
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
    ap.add_argument("--families", type=Path, default=None,
                    help="where counterexample families are written (default OUT/families; their verdicts are in run.json)")
    ap.add_argument("--peak-macs-per-s", type=float, default=0.0, help="the device's peak, for the unexplained-GPU flag")
    ap.add_argument("--meta", default="{}", help="JSON merged into the record (agent, tags, parent: launch.py's)")
    ap.add_argument("program_args", nargs=argparse.REMAINDER)
    return ap.parse_args(argv)


def run_script(a, args: list[str]) -> dict:
    """A standalone covert_witness train.py, unmodified, as `__main__` inside the counter (cw.py). Its checkpoints are
    E files for the evaluation job (`cw.score_run`); nothing is scored here."""
    from . import cw, hooks

    script = a.script.resolve()
    bad = screen(script.read_text())
    if bad:
        raise SystemExit(f"{script}: refused, it names {bad} (te_core/README.md, the program contract)")
    prepare = cw.import_prepare(script)
    start = {} if a.start_bpb is None else {"val_bpb": a.start_bpb}
    te = Session({}, a.out, seed=a.seed, tier=a.tier, device=getattr(prepare, "DEVICE", a.device), start=start,
                 budget_s=60 * a.budget_minutes, keep_checkpoints=a.keep_checkpoints, digests=a.digests != "off",
                 args=args, families=a.families)
    te.budget_macs = a.budget_macs or a.budget_fraction * float(getattr(prepare, "TRAIN_FLOP_BUDGET", 0)) / 2
    adapter = cw.Adapter(te, prepare, guard_log=a.out / "guard.log")
    adapter.install()
    hooks.ACTIVE = te
    meta = json.loads(a.meta) | {"program": script.name, "program_sha256": file_sha256(script), "commit": git_commit(),
                                 "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                                 "peak_macs_per_s": a.peak_macs_per_s, "mode": "script", "script_args": args}
    argv0, sys.argv = sys.argv, [str(script), *args]
    sys.path.insert(0, str(script.parent))
    rc = 0
    with te.counter:
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BudgetExceeded as e:
            meta["stopped"] = str(e)
        except Exception as e:  # noqa: BLE001 - the run's record says how the program failed
            rc, meta["error"] = 1, f"{type(e).__name__}: {e}"
        te.close()
    sys.argv = argv0
    hooks.ACTIVE = None
    meta |= adapter.meta() | {"script_rc": rc}
    problems = [f"the script exited {rc}" + (f" ({meta['error']})" if "error" in meta else "")] if rc else []
    if not adapter.submitted and "stopped" not in meta:
        problems.append("the script submitted nothing")
    return te.finish(meta, problems)


def main(argv=None) -> int:
    a = parse(argv)
    torch.cuda.graphs.CUDAGraph.replay = lambda self: (_ for _ in ()).throw(
        RuntimeError("te_core: CUDA graph replay runs work the counter can't see"))
    args = a.program_args[1:] if a.program_args[:1] == ["--"] else a.program_args
    if a.script:
        return report(run_script(a, args))
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
    te = Session(theta, a.out, seed=a.seed, tier=a.tier, device=a.device, start=start,
                 evaluator=load_evaluator(a.evaluator), budget_s=60 * a.budget_minutes,
                 keep_checkpoints=a.keep_checkpoints, digests=a.digests != "off", args=args,
                 families=a.families)
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
    return report(te.finish(meta))


def report(rec: dict) -> int:
    print(json.dumps({"macs": rec["totals"]["macs"], "steps": len(rec["steps"]), "checkpoints": [
        {k: cp.get(k) for k in ("step", "macs", "val_bpb", "source")} for cp in rec["checkpoints"]],
        "flags": rec["flags"], "invalid": rec["invalid"], "families": [
            {k: f[k] for k in ("family", "passed", "failed", "covers")} for f in rec["families"]]}))
    return 1 if rec["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
