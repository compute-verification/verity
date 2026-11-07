"""Run one submission inside the harness: the program trains from θ through a `Harness`, and everything it computes
through torch is counted (flops.py).

    python -m training_slowdown.run --program train.py --starts starts.json --start small --tier 0 --seed 0 --out OUT \\
        [--evaluator MODULE:FUNCTION] [--device cuda] [-- PROGRAM ARGS...]

The program is a Python file defining `train(te)`; `te` is the `Harness`. It builds its model, loads θ and registers it
(`te.load(model)`), declares its step count (`te.plan(n)`), wraps each training step in `with te.step():` and may
name its output model (`te.output(fn)`, default: the registered model's weights) and write counterexample families
(`te.family(name, plan)`). A checkpoint is taken after every 5% of the planned steps: the output model's held-out
bits per byte from the fixed evaluator, uncounted. Tokens computed outside a step, or a run with no step, make the run
invalid.

`--starts` is a registry of starts, `{name: {path, sha256, val_bpb, ctx, vocab, forward_macs_per_token}}`;
c is `C_PER_FORWARD_TOKEN` times the start's `forward_macs_per_token` (`block_c`; measured on the registered model when
the registry lacks it). The evaluator is a function `(state_dict, split, device) -> bits per byte`, or None for a split
it keeps sealed (a held-out `test`).

Writes `OUT/run.json` (the count, the checkpoints, the flags, the program's and θ's digests), `OUT/families/*/` with each
family's `verified.json` (checked here, after training, on the run's device, from a live coin) and, with
`--keep-checkpoints` (tier 2), each checkpoint's weights. Torch compilation is disabled and CUDA graph replay refused:
both run work the counter can't see. For the same reason a run is invalid when it uses an op the counter can't price
(flops.py), reads token data out of torch, starts a thread or a process, or imports a module that names a route
the screen refuses or that it wrote during the run (`Imports`). The screen (`screen`) refuses, in the program and in every
module it imports outside the fixed side, the names that run code it never reads: `exec`, `eval` and `compile` (called
or not), `runpy`, `importlib`, `__import__`, `marshal`, pickle, `FunctionType`, `CodeType`, `__code__` and
`sys.meta_path`; while the program runs no bytecode cache is read, so what runs is the source that was screened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import resource
import subprocess
import sys
import sysconfig
import time
import tokenize
from contextlib import contextmanager
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn.modules.module as nn_module

from . import VERSION
from .codecs import Plan
from .counterexample import SAMPLE_FINAL, SAMPLE_SEARCH, Family, check
from .flops import Counter, _tensors, digest, flags, forward_macs_per_token, guard_exits, leaked, plain

CHECKPOINTS = 20
#: a run's block size c in MACs is this many times its start's forward MACs per token (`block_c`): the bound a
#: counterexample family's rebuild must meet (training_slowdown.counterexample) and the record's `c`
C_PER_FORWARD_TOKEN = 4096
#: a run past its tier's GPU budget by this factor is stopped at its next step
BUDGET_SLACK = 1.2


class BudgetExceeded(RuntimeError):
    pass


def block_c(forward_macs_per_token: float) -> float:
    return C_PER_FORWARD_TOKEN * forward_macs_per_token


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


@contextmanager
def no_function_modes():
    """The block runs with no torch function mode and no tensor subclass's `__torch_function__`, but a mode it enters
    itself (a `torch.device(...)` context) applies: `torch._C.DisableTorchFunction` would switch that off too."""
    modes = [torch._C._pop_torch_function_stack() for _ in range(torch._C._len_torch_function_stack())]
    try:
        with torch._C.DisableTorchFunctionSubclass():
            yield
    finally:
        for mode in reversed(modes):
            torch._C._push_on_torch_function_stack(mode)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unknown ({type(e).__name__})"


class Harness:
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
        self.cp_steps: set[int] = set()
        self.checkpoints: list[dict] = []
        self.families: list[Family] = []
        self.forward_macs_per_token = float(self.start.get("forward_macs_per_token") or 0)
        self.sys_path = list(sys.path)
        self._t0 = clock()
        self.out.mkdir(parents=True, exist_ok=True)

    def _measure(self, model: torch.nn.Module) -> None:
        if not self.forward_macs_per_token:
            with self.counter.pause():
                ctx, vocab = int(self.start.get("ctx", 1024)), int(self.start.get("vocab", 50304))
                self.forward_macs_per_token = forward_macs_per_token(model, ctx, vocab, self.device)
            self.start["forward_macs_per_token_source"] = "measured on the registered model"

    @contextmanager
    def uncounted(self):
        """The harness's own work (the fixed loader, decoder and evaluator): neither counted nor timed, and outside the
        dispatch mode. None of the program's code runs in it: its torch function modes, global module hooks, trace and
        profile functions are off, so is the cyclic collector (its finalizers), and modules import from the sys.path the
        harness began with. It must not write the program's tensors, and it reads only plain values (`plain`)."""
        hooks = {k: v for k, v in vars(nn_module).items() if k.startswith("_global_") and isinstance(v, dict)}
        trace, profile, collecting = sys.gettrace(), sys.getprofile(), gc.isenabled()
        path = list(sys.path)
        for k, v in hooks.items():
            setattr(nn_module, k, type(v)())
        sys.settrace(None)
        sys.setprofile(None)
        gc.disable()
        sys.path[:] = self.sys_path
        try:
            with self.counter.pause(), torch.utils._python_dispatch._disable_current_modes(), no_function_modes():
                yield
        finally:
            sys.path[:] = path
            for k, v in hooks.items():
                setattr(nn_module, k, v)
            sys.settrace(trace)
            sys.setprofile(profile)
            if collecting:
                gc.enable()

    def in_steps(self) -> bool:
        return self.counter.cur is not self.counter.setup

    def current_step(self) -> int:
        return self.counter.cur.index

    def macs_so_far(self) -> int:
        c = self.counter
        return c.setup.macs + sum(s.macs for s in c.steps) + (c.cur.macs if c.cur is not c.setup else 0)

    def progress_bucket(self) -> int:
        """Which 5% of the run the step in progress is in: of the planned steps, else its index."""
        if self.n_steps:
            return math.ceil((self.current_step() + 1) * CHECKPOINTS / self.n_steps)
        return self.current_step()

    def _after_step(self, index: int) -> None:
        due = False
        if self.output_fn is not None or self.model is not None:
            due = bool(self.n_steps) and index in self.cp_steps
        if due:
            self._checkpoint(index)
        if self.budget_s and self.elapsed() > BUDGET_SLACK * self.budget_s:
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

    def batch(self, out):
        """What the start's data stream returned: each integer tensor in it is token data (`Counter.batch`)."""
        for t in _tensors(out):
            if not (t.is_floating_point() or t.is_complex()):
                self.counter.batch(t)
        return out

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

    # -- the harness's side

    def _state(self) -> dict[str, torch.Tensor]:
        state = self.output_fn() if self.output_fn else self.model.state_dict()
        return {k: v.detach() for k, v in state.items()}

    def _checkpoint(self, step: int) -> None:
        with torch.no_grad():
            state = plain(self._state())
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

    def finish(self, meta: dict, problems: list[str] | None = None, notes: list[str] | None = None) -> dict:
        """The run's record, with `problems` (what makes it invalid) and `notes` (flags) besides the counter's own."""
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
        if rec["foreign_ops"]:
            problems.append(f"ops outside torch's own, whose work the counter can't see: {list(rec['foreign_ops'])[:4]}")
        if rec["unknown_ops"]:
            problems.append(f"ops that compute products training_slowdown has no formula for: {list(rec['unknown_ops'])[:4]}")
        leaks = [s["step"] for s in [rec["setup"], *rec["steps"], rec["tail"]] if leaked(s)]
        if leaks:
            problems.append(f"token data read out of torch (numpy, tolist, DLPack) at a value per token or more, in steps "
                            f"{leaks[:5]}: what is computed on it there isn't counted")
        c = block_c(self.forward_macs_per_token)
        rec |= meta | {
            "schema": "training-slowdown/run/v1", "training_slowdown": VERSION, "seed": self.seed, "tier": self.tier,
            "start": self.start, "theta_sha256": theta_sha256(self.theta), "device": self.device,
            "forward_macs_per_token": self.forward_macs_per_token, "c": c, "planned_steps": self.n_steps,
            "checkpoints": self.checkpoints,
            "flags": flags(rec["totals"], float(meta.get("peak_macs_per_s", 0))) + list(notes or []),
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


#: names a program may not contain: the harness's internals, dispatch-mode control, routes to GPU or host kernels the
#: counter doesn't see, threads and processes, and code built from strings. `torch.compile` is allowed:
#: TORCHDYNAMO_DISABLE, set before torch is imported, makes it return the eager module, and `_dynamo` (which could turn
#: it back on) is refused. A screen of source text is a first filter only: the routes it names are caught as they run
#: where they can be (flops.py).
FORBIDDEN = (r"\.counter\b", r"\.pause\b", r"\buncounted\b", r"\btraining_slowdown\.(?!codecs\b)",
             r"(?m)from\s+training_slowdown\s+import\s+(?!codecs\s*$)", r"\bAdapter\b", "_python_dispatch", "_pop_mode", "_disable_current_modes", "TorchDispatchMode",
             "_set_dispatch_mode", "CUDAGraph", r"cuda\.graph", "_dynamo", "TORCHDYNAMO", "triton", "ctypes", "cupy",
             "numba", "jax", r"torch\.utils\.cpp_extension", "load_inline", "subprocess", "multiprocessing",
             r"os\.system", r"os\.exec", r"os\.spawn", r"os\.posix_spawn", r"\bos\.fork", r"\bos\.popen\b", r"\bpty\b",
             "_posixsubprocess", r"\bthreading\b", r"\b_thread\b", r"\bconcurrent\b", r"\btorch\.library\b",
             "TensorBase", "addaudithook", r"sys\.modules", r"\bgc\b\.", "__import__", r"importlib", r"\bbuiltins\b",
             "__builtins__", "__subclasses__", r"(?<![.\w])(exec|eval|compile)\s*\(", r"\bsys\.monitoring\b",
             r"(?m)^\s*from\s+sys\s+import\b.*\bmonitoring\b", r"\btorch\._C\b", r"(?m)^\s*from\s+torch\s+import\b.*\b_C\b",
             r"__bases?__", "__mro__", "__closure__", "cell_contents", r"\bpyarrow\b", r"\bpolars\b", r"\bduckdb\b",
             r"(?m)^\s*(import|from)\s+datasets\b", r"\brunpy\b", r"\bmarshal\b", r"\b_?c?pickle\b", r"\bdill\b",
             r"\bshelve\b", r"\bzipimport\b", "FunctionType", "CodeType", "__code__", r"\bmeta_path\b",
             r"\bpath_hooks\b", r"weights_only\s*=\s*False")
#: names that run source or code built at run time: refused wherever the program names them, called or not (`f = exec`),
#: except as another object's attribute (`re.compile`, `torch.compile`, `model.eval`)
FORBIDDEN_NAMES = frozenset({"exec", "eval", "compile", "execfile", "breakpoint"})


def _code(source: str) -> str:
    """`source` without its comments and docstrings (statements that are only a string), which never run."""
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    starts = (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)
    kept = [t for i, t in enumerate(toks) if t.type != tokenize.COMMENT and not (
        t.type == tokenize.STRING and (i == 0 or toks[i - 1].type in starts)
        and toks[i + 1].type in (tokenize.NEWLINE, tokenize.ENDMARKER))]
    return tokenize.untokenize(kept)


def _names(code: str) -> set[str]:
    """The bare names `code` uses: NAME tokens not after a `.`."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return set(re.findall(r"(?<![.\w])\w+", code))
    return {t.string for i, t in enumerate(toks) if t.type == tokenize.NAME and not (i and toks[i - 1].string == ".")}


def screen(source: str) -> list[str]:
    try:
        code = _code(source)
    except (tokenize.TokenError, SyntaxError, ValueError) as e:
        return [f"source that doesn't tokenize ({type(e).__name__})"]
    return [pat for pat in FORBIDDEN if re.search(pat, code)] + sorted(FORBIDDEN_NAMES & _names(code))


HERE = Path(__file__).resolve().parent


class Imports:
    """The modules first imported while a program runs, each screened like the program (a module beside it that names
    a refused route), unless the fixed side holds it: Python's own paths or training_slowdown (not its tests), unchanged
    since the run began. A module from a file written since the run began is refused:
    its source when it ran may not be what is screened at exit. While the program runs, bytecode caches are neither read
    nor written (`no_bytecode`), so what runs is the source that is screened."""

    def __init__(self):
        self.before = set(sys.modules)
        self.since = time.time()
        paths = {sysconfig.get_path(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
        self.roots = sorted(str(Path(p).resolve()) for p in paths if p)

    def _fixed(self, p: Path) -> bool:
        try:
            if p.stat().st_mtime >= self.since:
                return False
        except OSError:
            return False
        if p.is_relative_to(HERE):
            return "tests" not in p.relative_to(HERE).parts
        return any(str(p).startswith(r + os.sep) for r in self.roots)

    def problems(self) -> list[str]:
        bad = []
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", None)
            if name in self.before or not isinstance(f, str):
                continue
            p = Path(f).resolve()
            if self._fixed(p):
                continue
            if p.is_file() and p.stat().st_mtime >= self.since:
                bad.append(f"{name} ({p}: written during the run)")
            elif p.suffix != ".py":
                bad.append(f"{name} ({p.name}: native code)")
            elif not p.is_file():
                bad.append(f"{name} ({p}: its source is gone)")
            elif hits := screen(p.read_text(errors="replace")):
                bad.append(f"{name} names {hits}")
        return [f"modules the program imported break the program contract: {bad[:4]}"] if bad else []


@contextmanager
def no_bytecode():
    """Bytecode caches unread and unwritten (an empty cache prefix), so a module imported now runs from its source."""
    import tempfile

    prefix, write = sys.pycache_prefix, sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="te-pycache-") as d:
        sys.pycache_prefix, sys.dont_write_bytecode = d, True
        try:
            yield
        finally:
            sys.pycache_prefix, sys.dont_write_bytecode = prefix, write


def load_program(path: Path):
    path = Path(path).resolve()
    bad = screen(path.read_text())
    if bad:
        raise SystemExit(f"{path}: refused, it names {bad} (benchmarks/training_slowdown/README.md, the program contract)")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("training_program", path)
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
    ap.add_argument("--program", type=Path, required=True, help="a file defining train(te)")
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
    ap.add_argument("--meta", default="{}", help="JSON merged into the record")
    ap.add_argument("program_args", nargs=argparse.REMAINDER)
    return ap.parse_args(argv)



def main(argv=None) -> int:
    a = parse(argv)
    torch.cuda.graphs.CUDAGraph.replay = lambda self: (_ for _ in ()).throw(
        RuntimeError("training_slowdown: CUDA graph replay runs work the counter can't see"))
    args = a.program_args[1:] if a.program_args[:1] == ["--"] else a.program_args
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
    te = Harness(theta, a.out, seed=a.seed, tier=a.tier, device=a.device, start=start,
                 evaluator=load_evaluator(a.evaluator), budget_s=60 * a.budget_minutes,
                 keep_checkpoints=a.keep_checkpoints, digests=a.digests != "off", args=args,
                 families=a.families)
    guard_exits()
    imports = Imports()
    with no_bytecode():
        program = load_program(a.program)
    meta = json.loads(a.meta) | {"program": a.program.name, "program_sha256": file_sha256(a.program),
                                 "commit": git_commit(), "hardware": torch.cuda.get_device_name() if
                                 torch.cuda.is_available() else "cpu", "peak_macs_per_s": a.peak_macs_per_s}
    stopped = ""
    with no_bytecode(), te.counter:
        try:
            program.train(te)
        except BudgetExceeded as e:
            stopped = str(e)
    if stopped:
        meta["stopped"] = stopped
    return report(te.finish(meta, imports.problems()))


def report(rec: dict) -> int:
    print(json.dumps({"macs": rec["totals"]["macs"], "steps": len(rec["steps"]), "checkpoints": [
        {k: cp.get(k) for k in ("step", "macs", "val_bpb", "source")} for cp in rec["checkpoints"]],
        "flags": rec["flags"], "invalid": rec["invalid"], "families": [
            {k: f[k] for k in ("family", "passed", "failed", "covers")} for f in rec["families"]]}))
    return 1 if rec["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
