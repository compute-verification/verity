"""Run one job of an attack config inside the harness: the config's route trains its start from θ, and everything it
computes through torch is counted (flops.py).

    python -m training_slowdown.run --config attacks/NAME.toml --seed 0 --split search|final --out OUT \\
        [--starts starts.toml] [--starts-root DIR] [--device cuda] [--tier 0] [--keep-weights]
    python -m training_slowdown.run --retrace RUN_JSON --out OUT [--starts starts.toml] [--device cpu]

A job is two processes. Its own (`outside`) runs no route's code: it reads the job's held-out split, starts the
route's process in the sandbox (`training_slowdown.sandbox`: no network, an empty directory where each registered
final set is) and scores each state that process hands out. The route's process (`inside`) stops before it imports
the route when it finds an interface up, a final set readable or a capability in its own hands, so a job never
trains outside the sandbox; it gets no score back, and its record says what it found (`sandbox`).

The route loop (`train`): the route builds its model and optimizers (`training_slowdown.routes`), the harness loads θ
into the model and registers it, and runs the budget's steps, each the route's `step` on the next batch of the start's
data stream in the job's seed (`training_slowdown.data.Stream`), inside `with te.step():`. A checkpoint is taken after
every 5% of the steps, uncounted: the output model's bits per byte on the job's split (`training_slowdown.evaluate`)
and its E, the weights that differ from θ encoded by the route's plan (rebuilt from its spec, so only the codec
library's code runs). The output model a checkpoint scores is the one E rebuilds from θ: the trained weights under a
plan that keeps them whole, the model the kept bits give under one that keeps less. Tokens computed outside a step, or a run with no step,
make the run invalid. c is `C_PER_FORWARD_TOKEN` times the start's forward MACs per token (`block_c`), measured on
the registered model.

After the last step, outside the count, the job traces its route's step once for the grade (`trace_step`), among
the run's last steps: the record's `trace` is what the capture measures of it, the product rows unpriced and the
step's circuit, and a step the capture refuses is recorded as refused and leaves the job valid. `--retrace` takes a
record's trace again with this tree's capture, from the record's own config text, for a record whose trace a later
grade doesn't read.

Writes `OUT/run.json` (the count, the checkpoints, the flags, the config with its digest and text, the route, θ's
digest, the trace and the job's label). No weights are kept by default: of the last checkpoint the job keeps E, the
bytes that rebuild its output from θ, when E is under 50 MiB (`E_KEEP_BYTES`), and with `--keep-weights` the whole
state. Torch
compilation is disabled and CUDA graph replay refused: both run work the counter can't see. For the same reason a run is
invalid when it uses an op the counter can't price (flops.py), reads token data out of torch, starts a thread or a
process, or imports a module that names a way around the dispatcher or that was written during the run (`Imports`).
The screen (`screen`) refuses, in the route and in every module it imports outside the fixed side, the names that run
code it never reads: `exec`, `eval` and `compile` (called or not), `runpy`, `importlib`, `__import__`, `marshal`,
pickle, `FunctionType`, `CodeType`, `__code__` and `sys.meta_path`; while the job runs no bytecode cache is read, so
what runs is the source that was screened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import tokenize
from contextlib import contextmanager
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn.modules.module as nn_module

from . import VERSION, config, sandbox, starts
from .codecs import Identity, Plan, Unencodable
from .counterexample import SAMPLE_FINAL, SAMPLE_SEARCH, Family, check
from .data import SPLITS, Data, Stream
from .evaluate import Evaluator
from .flops import Counter, _tensors, digest, flags, forward_macs_per_token, guard_exits, leaked, plain

CHECKPOINTS = 20
#: a run's block size c in MACs is this many times its start's forward MACs per token (`block_c`): the bound a
#: counterexample family's rebuild must meet (training_slowdown.counterexample) and the record's `c`
C_PER_FORWARD_TOKEN = 4096
#: a run past its tier's GPU budget by this factor is stopped at its next step
BUDGET_SLACK = 1.2
#: a job keeps its last checkpoint's E under OUT when E is smaller than this
E_KEEP_BYTES = 50 * 2**20
#: the steps the trace's capture takes, the run's last (`trace_step`): the measured one, the one before it, which
#: wrote the values it reads, and the one after it, which reads the values it makes
TRACED_STEPS = 3


class BudgetExceeded(RuntimeError):
    pass


def block_c(forward_macs_per_token: float) -> float:
    return C_PER_FORWARD_TOKEN * forward_macs_per_token


def state_sha256(digests: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(digests):
        h.update(k.encode() + b"\0" + digests[k].encode())
    return h.hexdigest()


def theta_sha256(state: dict[str, torch.Tensor]) -> str:
    return state_sha256({k: digest(v) for k, v in state.items()})


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
    """What the code it counts runs inside. Everything but `step`'s body and the trained code itself runs uncounted."""

    def __init__(self, theta: dict[str, torch.Tensor], out: Path, *, seed: int = 0, tier: int = 0, device: str = "cpu",
                 start: dict | None = None, evaluator=None, split: str = "", budget_s: float = 0.0,
                 keep_weights: bool = False, digests: bool = True, clock=time.perf_counter,
                 families: Path | None = None, record: Path | None = None):
        self.theta, self.out = theta, Path(out)
        self.record = Path(record) if record else self.out / "run.json"
        self.families_root = Path(families) if families else self.out / "families"
        self.seed, self.tier, self.device = seed, tier, device
        self.start = start or {}
        self.evaluator, self.split, self.budget_s, self.keep_weights = evaluator, split, budget_s, keep_weights
        self.encoding: Plan | None = None
        self.theta_digests: dict[str, str] | None = None
        self.problems: list[str] = []
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
                shape = self.start.get("shape", {})
                ctx, vocab = int(shape.get("ctx", 1024)), int(shape.get("vocab", 50304))
                self.forward_macs_per_token = forward_macs_per_token(model, ctx, vocab, self.device)
            self.start["forward_macs_per_token_source"] = "measured on the registered model"

    @contextmanager
    def uncounted(self):
        """The harness's own work (the fixed loader, decoder and evaluator): neither counted nor timed, and outside the
        dispatch mode. None of the route's code runs in it: its torch function modes, global module hooks, trace and
        profile functions are off, so is the cyclic collector (its finalizers), and modules import from the sys.path the
        harness began with. It must not write the route's tensors, and it reads only plain values (`plain`)."""
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

    # -- what the counted code runs through

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

    def encode_with(self, plan: Plan) -> None:
        """Each checkpoint's E: the weights that differ from θ, encoded by `plan` as the codec library rebuilds it from
        its spec."""
        self.encoding = Plan.from_spec(plain(plan.spec()))

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

    def _encode(self, step: int, state: dict[str, torch.Tensor], digests: dict[str, str]) -> tuple[dict, bytes | None, dict]:
        """E of `state`: its weights that differ from θ (a weight θ lacks against zeros), by `encoding`, with its size,
        digest and rebuild MACs; E's bytes; and the weights E rebuilds to other bits than `state`'s, as it rebuilds
        them (none under a plan that keeps the weights whole)."""
        if self.theta_digests is None:
            self.theta_digests = {k: digest(v) for k, v in self.theta.items()}
        changed = sorted(k for k in state if digests[k] != self.theta_digests.get(k))
        bases = {k: self.theta[k] if k in self.theta else torch.zeros(state[k].shape, dtype=state[k].dtype)
                 for k in changed}
        try:
            enc = self.encoding.encode({k: state[k] for k in changed}, bases)
        except Unencodable as e:
            self.problems.append(f"checkpoint {step}: the route's plan has no E of its output ({e})")
            return {"e_bytes": None}, None, {}
        rebuilt = self.encoding.decode(enc, bases)
        stored = all(isinstance(c, Identity) for _, c in self.encoding.rules)
        return ({"e_bytes": len(enc), "e_sha256": hashlib.sha256(enc).hexdigest(), "format": "full" if stored else "codecs",
                 "rebuild_macs": self.encoding.cost_macs(enc, bases), "changed": len(changed)}, enc,
                {k: rebuilt[k] for k in changed if digest(rebuilt[k]) != digests[k]})

    def checkpoint_state(self, step: int, state: dict[str, torch.Tensor], extra: dict | None = None) -> dict:
        """Encode `state`, the model after step `step` (inclusive), and score the output model, uncounted. The
        output model is what E rebuilds from θ: `state` itself under a plan that keeps the weights whole, and under
        one that keeps less (a quantized delta) the model its bits give, which is then what is hashed, scored and
        kept, with `trained_sha256` the digest of `state`. Of the planned steps' last checkpoint the job keeps E
        under `e/` when it is smaller than `E_KEEP_BYTES` (`e_path`), and with `keep_weights` the whole output under
        `checkpoints/` (`path`); of any other, nothing."""
        with self.uncounted(), torch.no_grad():
            digests = {k: digest(v) for k, v in state.items()}
            fields, enc = {}, None
            if self.encoding is not None:
                fields, enc, rebuilt = self._encode(step, state, digests)
                if rebuilt:
                    fields["trained_sha256"] = state_sha256(digests)
                    state = state | rebuilt
                    digests = digests | {k: digest(v) for k, v in rebuilt.items()}
            out_sha256 = state_sha256(digests)
            score = self.evaluator(state, out_sha256) if self.evaluator else None
            cp = {"step": step, "macs": self.macs_so_far(), "bpb": score["bpb"] if score else None, "score": score,
                  "elapsed_s": self.elapsed(), "output_sha256": out_sha256} | (extra or {}) | fields
            if self.n_steps and step == max(self.cp_steps, default=-1):
                if enc is not None and len(enc) < E_KEEP_BYTES:
                    cp["e_path"] = self._keep(f"e/step{step:07d}.bin", lambda path: path.write_bytes(enc))
                if self.keep_weights:
                    cp["path"] = self._keep(f"checkpoints/step{step:07d}.pt",
                                            lambda path: torch.save({k: v.cpu() for k, v in state.items()}, path))
            self.checkpoints.append(cp)
        return cp

    def _keep(self, name: str, write) -> str:
        path = self.out / name
        path.parent.mkdir(exist_ok=True)
        write(path)
        return name

    def finish(self, meta: dict, problems: list[str] | None = None, notes: list[str] | None = None) -> dict:
        """The run's record, with `problems` (what makes it invalid) and `notes` (flags) besides the counter's own."""
        self.counter.finish()
        rec = self.counter.record()
        problems = list(problems or []) + self.problems
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
            "split": self.split, "start": self.start, "theta_sha256": theta_sha256(self.theta), "device": self.device,
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
        self.record.write_text(json.dumps(rec) + "\n")
        return rec


#: names a route may not contain: the harness's internals, dispatch-mode control, ways to GPU or host kernels the
#: counter doesn't see, threads and processes, and code built from strings. `torch.compile` is allowed:
#: TORCHDYNAMO_DISABLE, set before torch is imported, makes it return the eager module, and `_dynamo` (which could turn
#: it back on) is refused. A screen of source text is a first filter only: the ways it names are caught as they run
#: where they can be (flops.py).
FORBIDDEN = (r"\.counter\b", r"\.pause\b", r"\buncounted\b", r"\btraining_slowdown\.(?!(codecs|model)\b)",
             r"(?m)from\s+training_slowdown\s+import\s+(?!(codecs|model)\s*$)", r"\bAdapter\b", "_python_dispatch", "_pop_mode", "_disable_current_modes", "TorchDispatchMode",
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
#: names that run source or code built at run time: refused wherever a route names them, called or not (`f = exec`),
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
    """The modules first imported while a job runs, each screened like its route (a module that names a refused way
    around the dispatcher), unless the fixed side holds it: Python's own paths or training_slowdown (not its tests),
    unchanged since the run began. A module from a file written since the run began is refused: its source when it ran
    may not be what is screened at exit. While the job runs, bytecode caches are neither read nor written
    (`no_bytecode`), so what runs is the source that is screened."""

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
        return [f"modules the job imported break the route contract: {bad[:4]}"] if bad else []


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


def train(te: Harness, cfg: config.Config, route, stream: Stream, model, optimizers, steps: int | None = None) -> None:
    """The route loop (module docstring) on the route's `build`. A route with an `output` gives each checkpoint its
    output model; what that computes is counted, in the step after the checkpoint or in the tail. `steps` stops the
    loop after the budget's first `steps` steps, each run as the whole budget's, with a checkpoint after the last."""
    te.load(model)
    te.encode_with(route.plan(cfg.dials))
    if hasattr(route, "output"):
        te.output(lambda: route.output(model, cfg.dials))
    n, batch = cfg.budget["steps"], cfg.budget["batch"]
    te.plan(n)
    last = n if steps is None else min(steps, n)
    te.cp_steps.add(last - 1)
    for t in range(last):
        with te.step():
            route.step(model, te.batch(stream.next(batch).to(te.device)), optimizers, cfg.dials, t, n)


def trace_step(cfg: config.Config, route, model, optimizers, device: str) -> dict:
    """The record's `trace`: the route's `step` on its model and fresh copies of its optimizers, at the job's batch
    and the start's context, as the capture measures it (`capture_steps`, `capture.products.raw_on_dial`). The
    capture takes the run's last three steps in one graph and measures the middle one, which reads what the step
    before it wrote and makes what the step after it reads (`TRACED_STEPS`; a run of two steps is measured at its
    second, a run of one at its only step, which reads no update). Where the three aren't the run's first steps the
    capture is told so (`mid_run`): the first of them reads the run's own weights, not θ's. A route's `output` is
    traced as the end of every step, so the products it computes are the measured step's, charged like any other,
    though a run calls it at its checkpoints only.

    It keeps the capture's and torch's versions, the version of the code that took the trace (`capture_code`, this
    tree's: a record whose trace is taken again keeps its job's `training_slowdown`), the step and the token axes it
    was traced at, the forward's updated MAC nodes (`StepGraph.structure`'s `updated`), the product rows with their
    operands' dtypes and no price at each k of the console's dial, the ops that break P-match, the products written
    with elementwise ops, the read-backs the capture lists (`data_dependent`, `carried_scalars`; a step the capture
    takes as a function has none, since it refuses one), and the measured step's circuit as the capture measures it
    (`circuit`, `capture.circuit.measure`): the gradient's cut and one token group's partial, the carried state no
    token op reads, the update's profile and its cut, τ, a, and the shares of the step's gates in token ops (σ) and
    in ops an update reaches, at the gate prices of the code that took the trace. Nothing the formalism derives is in
    it. A step the capture refuses gives {"refused": {"reason", "op", "message"}}.

    The batch is traced as its inputs and each window's last target apart, so the token axes are the batch and the
    context, the tokens the counter counts a step."""
    return traced_step(cfg, route, model, optimizers, device)[0]


def traced_step(cfg: config.Config, route, model, optimizers, device: str) -> tuple[dict, object]:
    """`trace_step`'s trace beside the capture's graph it was read from (a `StepGraph`), which holds what the record
    doesn't keep: the parameters the step writes and the ops the capture prices by a rule of thumb. No graph, None,
    for a step the capture refuses."""
    from .capture import CaptureRefused, capture_steps, products

    n, batch, ctx = cfg.budget["steps"], cfg.budget["batch"], int(cfg.start.shape["ctx"])
    ts = range(max(n - TRACED_STEPS, 0), n)
    at = min(1, len(ts) - 1)

    def step(t: int):
        def traced(m, rows, opts):
            route.step(m, torch.cat(rows, dim=1), opts, cfg.dials, t, n)
            if hasattr(route, "output"):
                with torch.no_grad():
                    route.output(m, cfg.dials)
        return traced

    example = (torch.zeros(batch, ctx, dtype=torch.long), torch.zeros(batch, 1, dtype=torch.long))
    try:
        g = capture_steps([step(t) for t in ts], model, example, measured=at, optimizers=optimizers,
                          token_dims=((0, 1), (0,)), device=device, mid_run=ts[0] > 0)
    except CaptureRefused as e:
        return {"refused": {"reason": e.reason, "op": e.op, "message": str(e)}}, None
    raw = products.raw_on_dial(g)
    read_backs = g.circuit.read_backs or {}
    return {"capture": raw["capture"], "capture_code": VERSION, "torch": torch.__version__, "step": ts[at],
            "axes": g.axes, "tokens": g.tokens, "order": raw["order"], "updated": g.structure()["updated"],
            "rows": raw["rows"], "k": raw["k"], "p_match": raw["p_match"], "at": raw["at"],
            "elementwise_products": raw["elementwise_products"],
            "data_dependent": list(read_backs.get("data_dependent", [])),
            "carried_scalars": list(read_backs.get("carried_scalars", [])),
            "circuit": {k: v for k, v in g.formalism.items() if k not in read_backs}}, g


def retrace(a: argparse.Namespace) -> int:
    """`--retrace RUN_JSON --out DIR`: the record with its `trace` taken again by this tree's capture and nothing else
    changed, in `DIR/run.json`. The config is the record's own text; a tree whose route, start or data isn't the
    record's (another digest) is refused, since the step it would trace isn't the one that ran."""
    rec = json.loads(a.retrace.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.inside / f"{rec['config']['name']}.toml"
    path.write_text(rec["config"]["text"])
    with no_bytecode():
        cfg = config.load(path, starts.registry(a.starts))
        if cfg.digest != rec["config"]["digest"]:
            raise SystemExit(f"retrace: this tree reads {a.retrace}'s config as digest {cfg.digest[:16]}, and the "
                             f"record's is {rec['config']['digest'][:16]} (its route's VERSION, its start or its "
                             f"data moved): retrace in a clone at the record's commit, {rec.get('commit')}")
        route = config.route(cfg.route)
        model, optimizers = route.build(cfg.start.shape, cfg.dials)
        rec["trace"] = trace_step(cfg, route, model.to(a.device), optimizers, a.device)
    (a.out / "run.json").write_text(json.dumps(rec) + "\n")
    print(json.dumps({"retraced": str(a.retrace), "capture": rec["trace"].get("capture"),
                      "refused": rec["trace"].get("refused")}))
    return 0


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, help="an attack config (training_slowdown.config)")
    ap.add_argument("--seed", type=int, help="the job's seed: torch's and the data stream's")
    ap.add_argument("--split", choices=SPLITS, help="the held-out split each checkpoint is scored on")
    ap.add_argument("--retrace", type=Path, metavar="RUN_JSON",
                    help="instead of a job: write RUN_JSON to OUT/run.json with its trace taken again (`retrace`)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--starts", type=Path, default=starts.REGISTRY, help="the starts registry (default starts.toml)")
    ap.add_argument("--starts-root", type=Path, default=None,
                    help="where the starts are installed, NAME/{theta.safetensors,start.json} (default the registry's)")
    ap.add_argument("--digests", choices=("on", "off"), default="on",
                    help="digest every updated weight value (flops.KeyedDigest, on the run's device; counterexample "
                         "checks need them)")
    ap.add_argument("--tier", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-weights", action="store_true",
                    help="keep the last checkpoint's whole state under OUT/checkpoints (default: its E alone, when "
                         "under 50 MiB)")
    ap.add_argument("--families", type=Path, default=None,
                    help="where counterexample families are written (default OUT/families; their verdicts are in run.json)")
    ap.add_argument("--peak-macs-per-s", type=float, default=0.0, help="the device's peak, for the unexplained-GPU flag")
    ap.add_argument("--meta", default="{}", help="JSON merged into the record")
    ap.add_argument("--inside", type=Path, metavar="SCRATCH",
                    help="the job gives this itself: be the route's process (`inside`), with the job's scratch directory")
    ap.add_argument("--channel", metavar="FD,FD", help="with --inside: where a state is announced, and where its "
                                                        "scoring is")
    a = ap.parse_args(argv)
    if a.retrace is None and (missing := [f"--{k}" for k in ("config", "seed", "split") if getattr(a, k) is None]):
        ap.error(f"a job needs {', '.join(missing)}")
    return a


class Outside:
    """The evaluator's end inside the sandbox: each checkpoint's state goes out, through a file of the job's scratch
    directory, to the job's own process, which scores it (`outside`). One byte comes back when it has, and no score:
    nothing of a held-out set enters the process a route runs in."""

    def __init__(self, scratch: Path, channel: str):
        states, scored = (int(fd) for fd in channel.split(","))
        self.scratch, self.states, self.scored, self.sent = scratch, os.fdopen(states, "w"), os.fdopen(scored, "rb"), 0

    def __call__(self, state: dict[str, torch.Tensor], weights_digest: str) -> None:
        name = f"state-{self.sent:04d}.pt"
        self.sent += 1
        torch.save({k: v.cpu() for k, v in state.items()}, self.scratch / name)
        self.states.write(json.dumps({"state": name}) + "\n")
        self.states.flush()
        if not self.scored.read(1):
            raise SystemExit("training_slowdown: the job's evaluator, outside the sandbox, stopped")


def outside(a: argparse.Namespace, argv: list[str]) -> int:
    """The job's own process, which runs no route's code. It reads the job's held-out split, starts the route's
    process (`inside`) in the sandbox with every registered final set hidden, scores each state that process hands
    out, under the digest it hashes the state to itself, and writes the record with the scores in (`with_scores`)."""
    reg = starts.registry(a.starts)
    scratch = Path(tempfile.mkdtemp(prefix="training-slowdown-"))
    try:
        hide = [x for d in sandbox.final_sets(reg) for x in ("--hide", str(d))]
        route = [sys.executable, "-m", "training_slowdown.sandbox", *hide, "--",
                 sys.executable, "-m", "training_slowdown.run", *argv, "--inside", str(scratch)]
        env = os.environ | {"PYTHONPATH": os.pathsep.join([str(HERE.parent), *filter(None, sys.path)])}
        if a.retrace is not None:
            return subprocess.run(route, env=env).returncode
        s = config.start_of(a.config, reg)
        evaluator = Evaluator(Data.of(s.data, reg), a.split, starts.load(s, a.starts_root or Path(reg["installed"])),
                              s.shape, a.device)
        states, scored = os.pipe(), os.pipe()
        scores = []
        with subprocess.Popen([*route, "--channel", f"{states[1]},{scored[0]}"], env=env,
                              pass_fds=(states[1], scored[0])) as child:
            os.close(states[1])
            os.close(scored[0])
            with open(states[0]) as lines, open(scored[1], "wb", buffering=0) as done:
                for line in lines:
                    path = scratch / Path(json.loads(line)["state"]).name
                    state = torch.load(path, map_location=a.device, weights_only=True)
                    path.unlink()
                    sha = state_sha256({k: digest(v) for k, v in state.items()})
                    scores.append((sha, evaluator(state, sha)))
                    try:
                        done.write(b"\n")
                    except BrokenPipeError:
                        print(f"training_slowdown: the route's process closed its end before it was told its state "
                              f"{len(scores) - 1} was scored; no further state is read", file=sys.stderr)
                        break
        record = scratch / "run.json"
        if child.returncode or not record.is_file():
            return child.returncode or 1
        rec = with_scores(json.loads(record.read_text()), scores)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "run.json").write_text(json.dumps(rec) + "\n")
        return report(rec)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def with_scores(rec: dict, scores: list[tuple[str, dict]]) -> dict:
    """The route's process's record with each checkpoint's score in, where the states scored, each by the digest the
    job's own process hashed it to, are its checkpoints' outputs one for one; else the record, invalid."""
    if [sha for sha, _ in scores] == [cp["output_sha256"] for cp in rec["checkpoints"]]:
        for cp, (_, score) in zip(rec["checkpoints"], scores):
            cp["bpb"], cp["score"] = score["bpb"], score
    else:
        rec["invalid"].append(f"the {len(scores)} states the evaluator scored aren't the record's "
                              f"{len(rec['checkpoints'])} checkpoints' outputs, one for one")
    return rec


def inside(a: argparse.Namespace) -> int:
    """The route's process. It stops before it imports the route unless it is in the sandbox (`sandbox.refusals`):
    an interface that is up or a registered final set it can read an entry of is a failed job. Then it trains as the
    module's docstring says and leaves its record, the checkpoints unscored, in the job's scratch directory."""
    reg = starts.registry(a.starts)
    hidden = sandbox.final_sets(reg)
    if reached := sandbox.refusals(hidden):
        raise SystemExit(f"training_slowdown: a route's code runs only inside the job's sandbox, and this process is "
                         f"outside it: {'; '.join(reached)}. Start the job without --inside: it makes the sandbox.")
    if a.retrace is not None:
        return retrace(a)
    torch.cuda.graphs.CUDAGraph.replay = lambda self: (_ for _ in ()).throw(
        RuntimeError("training_slowdown: CUDA graph replay runs work the counter can't see"))
    imports = Imports()
    with no_bytecode():
        cfg = config.load(a.config, reg)
        route = config.route(cfg.route)
    s = cfg.start
    theta = starts.load(s, a.starts_root or Path(reg["installed"]))
    data = Data.of(s.data, reg)
    torch.manual_seed(a.seed)
    te = Harness(theta, a.out, seed=a.seed, tier=a.tier, device=a.device, start=s.entry(),
                 evaluator=Outside(a.inside, a.channel) if a.channel else None,
                 split=a.split, budget_s=60 * cfg.budget.get("minutes", 0), keep_weights=a.keep_weights,
                 digests=a.digests != "off", families=a.families, record=a.inside / "run.json")
    stream = Stream(data, a.seed, ctx=int(s.shape["ctx"]))
    guard_exits()
    meta = json.loads(a.meta) | {
        "config": cfg.record(), "route": {"name": cfg.route, "version": cfg.route_version,
                                          "sha256": file_sha256(route.__file__)},
        "label": f"{config.LABEL_KEY}={config.Job(cfg, a.seed, a.split).label}", "commit": git_commit(),
        "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "peak_macs_per_s": a.peak_macs_per_s,
        "sandbox": {"interfaces": sandbox.interfaces(),
                    "final_sets": {str(d): "empty" if d.exists() else "absent" for d in hidden}}}
    stopped = ""
    with no_bytecode():
        with te.counter:
            model, optimizers = route.build(cfg.start.shape, cfg.dials)
            try:
                train(te, cfg, route, stream, model, optimizers)
            except BudgetExceeded as e:
                stopped = str(e)
        # after the last checkpoint and outside the count: nothing the trace runs reaches the record's count or scores
        meta["trace"] = trace_step(cfg, route, model, optimizers, a.device)
    if stopped:
        meta["stopped"] = stopped
    te.finish(meta, imports.problems())
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else [str(x) for x in argv]
    a = parse(argv)
    return inside(a) if a.inside is not None else outside(a, argv)


def report(rec: dict) -> int:
    tr = rec.get("trace") or {}
    print(json.dumps({"macs": rec["totals"]["macs"], "steps": len(rec["steps"]), "checkpoints": [
        {k: cp.get(k) for k in ("step", "macs", "bpb", "e_bytes", "source")} for cp in rec["checkpoints"]],
        "flags": rec["flags"], "invalid": rec["invalid"], "families": [
            {k: f[k] for k in ("family", "passed", "failed", "covers")} for f in rec["families"]],
        "trace": {"refused": tr["refused"]} if "refused" in tr else {k: tr.get(k) for k in ("capture", "tokens")}}))
    return 1 if rec["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
