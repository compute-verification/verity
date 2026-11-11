"""Run one job of an attack config inside the harness: the config's route trains its start from θ, and everything it
computes through torch is counted (flops.py).

    python -m training_slowdown.run --config attacks/NAME.toml --seed 0 --split search|final --out OUT \\
        [--state DIR] [--starts starts.toml] [--starts-root DIR] [--device cuda] [--tier 0] [--keep-weights]
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

With `--state DIR` a stopped job continues (`Kept`). After each checkpoint but the last, which is the job's end, the
job's own process keeps in DIR what continuing needs: the tensors the route's process hands it (the model, the
optimizers' state, the stream's position, torch's, numpy's and Python's generators) and, written by itself, the
record so far with its scores, where the clock stands and, in the record's last segment, the SHA-512 of each part
of the tensors as DIR holds them (`kept_sha512`). DIR is hidden from the route's process as a final set is. A job
that finds a state in DIR continues from it when the state is its own: the same `training_slowdown` VERSION, seed,
split and config digest, tensors whose every part hashes to what the record states, and a model that hashes to its
record's last checkpoint. Any other it refuses, naming what differs, and trains nothing. The route's process then
takes the kept tensors and, before it imports the route, says what each part hashes to as it holds it: the job's
own process ends the job there when that isn't what the record states. It loads θ, puts the tensors back and runs
the steps that remain; it is told the next step, the clock and the digest key, and never the record. Each stretch a
process ran is a segment: the record lists them (`segments`, a stopped one with the digests of the state the next
one continued from), its steps and checkpoints are every segment's in order, and its count is their sum (`joined`),
which a stopped segment's work after its last kept checkpoint is not in. `budget.minutes` runs over all of them.
Told to stop (SIGTERM), the job ends with `STOPPED` and leaves the state; a job that ends itself, with its record or
with a failure, removes it.

After the last step, outside the count, the job traces its route's step once for the grade (`trace_step`): the
record's `trace` is what the capture measures of it, the product rows unpriced, and a step the capture refuses is
recorded as refused and leaves the job valid. `--retrace` takes a record's trace again with this tree's capture, from
the record's own config text, for a record whose trace a later grade doesn't read.

Writes `OUT/run.json` (the count, the checkpoints, the flags, the config with its digest and text, the route, θ's
digest, the trace and the job's label). No weights are kept by default: of the last checkpoint the job keeps E, the
bytes that rebuild its output from θ, when E is under 50 MiB (`E_KEEP_BYTES`), and with `--keep-weights` the whole
state. Torch
compilation is disabled and CUDA graph replay refused: both run work the counter can't see. For the same reason a run is
invalid when it uses an op the counter can't price (flops.py), reads token data out of torch, starts a thread or a
process, or imports a module that names a way around the dispatcher or that was written during the run (`Imports`).
The screen (`training_slowdown.screen`) refuses, in the route and in every module it imports outside the fixed side,
the names that run code it never reads: `exec`, `eval` and `compile` (called or not), `runpy`, `importlib`,
`__import__`, `marshal`, pickle, `FunctionType`, `CodeType`, `__code__` and `sys.meta_path`; while the job runs no
bytecode cache is read, so what runs is the source that was screened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
import torch.nn.modules.module as nn_module

from . import BUDGET_SLACK, STOPPED, VERSION, config, sandbox, starts
from .codecs import Identity, Plan, Unencodable
from .counterexample import SAMPLE_FINAL, SAMPLE_SEARCH, Family, check
from .data import SPLITS, Data, Stream
from .evaluate import Evaluator
from .flops import Counter, _tensors, digest, flags, forward_macs_per_token, guard_exits, leaked, plain, structure_id
from .screen import screen

CHECKPOINTS = 20
#: a run's block size c in MACs is this many times its start's forward MACs per token (`block_c`): the bound a
#: counterexample family's rebuild must meet (training_slowdown.counterexample) and the record's `c`
C_PER_FORWARD_TOKEN = 4096
#: a job keeps its last checkpoint's E under OUT when E is smaller than this
E_KEEP_BYTES = 50 * 2**20


class BudgetExceeded(RuntimeError):
    pass


class Stopped(Exception):
    """The job was told to stop."""


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
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _feed(h, x) -> None:
    """Feed `h` a kept state's value under its kind: a tensor as its dtype, shape and bytes, a mapping as its items
    in their keys' order, a list or a tuple as its items, a number, a string or None as its type and text."""
    if isinstance(x, torch.Tensor):
        t = x.detach().contiguous()
        h.update(f"tensor\0{t.dtype}\0{tuple(t.shape)}\0".encode())
        h.update(t.reshape(-1).view(torch.uint8).cpu().numpy())
    elif isinstance(x, dict):
        h.update(f"dict\0{len(x)}\0".encode())
        for k in sorted(x, key=repr):
            _feed(h, k)
            _feed(h, x[k])
    elif isinstance(x, (list, tuple)):
        h.update(f"list\0{len(x)}\0".encode())
        for v in x:
            _feed(h, v)
    elif x is None or type(x) in (bool, int, float, str):
        h.update(f"{type(x).__name__}\0{x.hex() if type(x) is float else repr(x)}\0".encode())
    else:
        raise TypeError(f"training_slowdown: a kept state holds plain values, and this is a {type(x).__name__}")


def kept_sha512(state: dict) -> dict[str, str]:
    """The SHA-512 of each part of a kept state (`resume_state`: the model, the optimizers' state, the stream's
    position, the generators), over its values as they are held: every bit of every tensor and number in it."""
    out = {}
    for part in sorted(state):
        h = hashlib.sha512()
        _feed(h, state[part])
        out[part] = h.hexdigest()
    return out


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
                 families: Path | None = None, record: Path | None = None, kept: dict | None = None, keep=None):
        """`kept` is where a job that continues stands (`step`, the next one; `elapsed_s`; `digest_key`), and
        `keep(state)` is called with the trained state after each planned checkpoint but the last."""
        kept = kept or {}
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
        self.counter = Counter(theta, digests=digests, first=int(kept.get("step", 0)),
                               key=bytes.fromhex(kept["digest_key"]) if kept.get("digest_key") else None)
        self.keep = keep
        self.bound = bool(theta)
        self.model = None
        self.output_fn = None
        self.n_steps = 0
        self.cp_steps: set[int] = set()
        self.checkpoints: list[dict] = []
        self.families: list[Family] = []
        self.forward_macs_per_token = float(self.start.get("forward_macs_per_token") or 0)
        self.sys_path = list(sys.path)
        self._t0 = clock() - float(kept.get("elapsed_s", 0.0))
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
            state = self._checkpoint(index)
            if self.keep is not None and index != max(self.cp_steps):
                self.keep(state)
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

    def resume(self, model: torch.nn.Module, optimizers, stream: Stream, state: dict) -> None:
        """Put a kept state (`resume_state`) back after θ's load: the model's weights, written in place so the
        counter's names for them stand, each optimizer's state, the stream's position and the generators', uncounted
        as θ's load is."""
        if len(optimizers) != len(state["optimizers"]):
            raise SystemExit(f"training_slowdown: the kept state holds {len(state['optimizers'])} optimizers' state "
                             f"and the route built {len(optimizers)}")
        with self.counter.pause():
            model.load_state_dict(state["model"])
            for o, s in zip(optimizers, state["optimizers"]):
                o.load_state_dict(s)
            stream.at = int(state["stream"])
            set_generators(state["generators"])

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
        yield self.counter.done
        self.counter.end_step()
        self._after_step(self.counter.done - 1)

    # -- the harness's side

    def _state(self) -> dict[str, torch.Tensor]:
        state = self.output_fn() if self.output_fn else self.model.state_dict()
        return {k: v.detach() for k, v in state.items()}

    def _checkpoint(self, step: int) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state = plain(self._state())
        self.checkpoint_state(step, state, {"source": "output"})
        return state

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

    def so_far(self, meta: dict, problems: list[str] | None = None, notes: list[str] | None = None) -> dict:
        """The record of the steps so far, with `problems` (what makes it invalid) and `notes` (flags) besides the
        counter's own: what a job keeps at a checkpoint, and `finish` without the families' verdicts."""
        rec = self.counter.record()
        problems = list(problems or []) + self.problems
        if not rec["steps"] and not self.counter.first:
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
            "invalid": problems, "families": [],
        }
        return self._self_cpu(rec)

    @staticmethod
    def _self_cpu(rec: dict) -> dict:
        me = resource.getrusage(resource.RUSAGE_SELF)
        # taken last: the evaluation job counts every CPU second of the training job's tree beyond this one as unseen
        rec["totals"]["self_cpu_s"] = me.ru_utime + me.ru_stime
        return rec

    def finish(self, meta: dict, problems: list[str] | None = None, notes: list[str] | None = None) -> dict:
        """The run's record (`so_far`, now that it ended), with each family's verdict, written."""
        self.counter.finish()
        rec = self.so_far(meta, problems, notes)
        n = SAMPLE_FINAL if self.tier >= 2 else SAMPLE_SEARCH
        with self.counter.pause():
            for fam in self.families:
                v = check(fam.dir, rec, self.theta, rec["c"], n_sample=n, device=self.device, plan=fam.plan)
                (fam.dir / "verified.json").write_text(json.dumps(v.to_json(), indent=1) + "\n")
                rec["families"].append(v.to_json())
        self._self_cpu(rec)
        self.record.write_text(json.dumps(rec) + "\n")
        return rec


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


def generators() -> dict:
    """The state of the generators a route's step may read: torch's on the host and on every GPU, numpy's and
    Python's, as tensors and plain values."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    version, internal, gauss = random.getstate()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [kind, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached)],
            "python": [version, list(internal), gauss]}


def set_generators(kept: dict) -> None:
    kind, keys, pos, has_gauss, cached = kept["numpy"]
    version, internal, gauss = kept["python"]
    torch.set_rng_state(kept["torch"].cpu())
    if kept["cuda"]:
        torch.cuda.set_rng_state_all([t.cpu() for t in kept["cuda"]])
    np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), pos, has_gauss, cached))
    random.setstate((version, tuple(internal), gauss))


def resume_state(trained: dict[str, torch.Tensor], optimizers, stream: Stream) -> dict:
    """What a route's process continues from, in plain values: the trained model's whole state (not the output E
    rebuilds), each optimizer's state, the stream's position and the generators'. A route that carries anything
    else from one step to the next (`training_slowdown.routes`) doesn't continue as an unbroken job runs."""
    return plain({"model": trained, "optimizers": [o.state_dict() for o in optimizers], "stream": stream.at,
                  "generators": generators()})


def train(te: Harness, cfg: config.Config, route, stream: Stream, model, optimizers, state: dict | None = None) -> None:
    """The route loop (module docstring) on the route's `build`, from the first step or, with a kept `state`, from
    the step the harness continues at."""
    te.load(model)
    te.encode_with(route.plan(cfg.dials))
    n, batch = cfg.budget["steps"], cfg.budget["batch"]
    te.plan(n)
    if state is not None:
        te.resume(model, optimizers, stream, state)
    for t in range(te.counter.first, n):
        with te.step():
            route.step(model, te.batch(stream.next(batch).to(te.device)), optimizers, cfg.dials, t, n)


def trace_step(cfg: config.Config, route, model, optimizers, device: str) -> dict:
    """The record's `trace`: the route's `step` at the run's last step, on its model and fresh copies of its
    optimizers, at the job's batch and the start's context, as the capture measures it (`capture_fn`,
    `capture.products.raw_on_dial`): the capture's and torch's versions, the step and the token axes it was traced
    at, the forward's updated MAC nodes (`StepGraph.structure`'s `updated`), the product rows with their operands'
    dtypes and no price at each k of the console's dial, the ops that break P-match, the products written with
    elementwise ops, and the read-backs the capture lists (`data_dependent`, `carried_scalars`; a step the capture
    takes as one function has none, since it refuses one). Nothing the formalism derives is in it. A step the capture
    refuses gives {"refused": {"reason", "op", "message"}}.

    The batch is traced as its inputs and each window's last target apart, so the token axes are the batch and the
    context, the tokens the counter counts a step."""
    from .capture import CaptureRefused, capture_fn, products

    n, batch, ctx = cfg.budget["steps"], cfg.budget["batch"], int(cfg.start.shape["ctx"])
    t = n - 1

    def step(m, rows, opts):
        route.step(m, torch.cat(rows, dim=1), opts, cfg.dials, t, n)

    example = (torch.zeros(batch, ctx, dtype=torch.long), torch.zeros(batch, 1, dtype=torch.long))
    try:
        g = capture_fn(step, model, example, optimizers=optimizers, token_dims=((0, 1), (0,)), device=device)
    except CaptureRefused as e:
        return {"refused": {"reason": e.reason, "op": e.op, "message": str(e)}}
    raw = products.raw_on_dial(g)
    read_backs = (g.circuit.read_backs if g.circuit is not None else None) or {}
    return {"capture": raw["capture"], "torch": torch.__version__, "step": t, "axes": g.axes, "tokens": g.tokens,
            "order": raw["order"], "updated": g.structure()["updated"], "rows": raw["rows"], "k": raw["k"],
            "p_match": raw["p_match"], "at": raw["at"], "elementwise_products": raw["elementwise_products"],
            "data_dependent": list(read_backs.get("data_dependent", [])),
            "carried_scalars": list(read_backs.get("carried_scalars", []))}


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
    ap.add_argument("--state", type=Path, metavar="DIR",
                    help="this job's state directory: it keeps there what continuing needs after each checkpoint, "
                         "and continues from a state of its own it finds there (default: neither)")
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
    """The job's own process, as the route's process reaches it from inside the sandbox. What it says comes first and
    once: where a job that continues stands (`kept`), or nothing. What goes out: what the kept tensors hash to as
    this process holds them, before it imports the route (`holds`); the job's config digest, before any step
    (`names`); each checkpoint's state, through a file of the job's scratch directory, to be scored (`outside`);
    and after it what continuing from that checkpoint needs, where the job keeps a state. One byte comes back for
    each, and no score: nothing of a held-out set enters the process a route runs in."""

    def __init__(self, scratch: Path, channel: str):
        states, scored = (int(fd) for fd in channel.split(","))
        self.scratch, self.states, self.scored, self.sent = scratch, os.fdopen(states, "w"), os.fdopen(scored, "rb"), 0
        self.kept: dict | None = json.loads(self._read(self.scored.readline))

    @staticmethod
    def _read(read) -> bytes:
        if not (got := read()):
            raise SystemExit("training_slowdown: the job's own process, outside the sandbox, stopped")
        return got

    def _say(self, **said) -> None:
        self.states.write(json.dumps(said) + "\n")
        self.states.flush()
        self._read(lambda: self.scored.read(1))

    def _hand(self, kind: str, tensors: dict, **said) -> None:
        name = f"{kind}-{self.sent:04d}.pt"
        self.sent += 1
        torch.save(tensors, self.scratch / name)
        self._say(**{kind: name}, **said)

    def holds(self, kept: dict[str, str]) -> None:
        """Say what each part of the kept tensors hashes to as this process holds it (`kept_sha512`): the job's own
        process ends it here when that isn't what the record of the segment that kept them states."""
        self._say(kept=kept)

    def names(self, digest: str) -> None:
        """Say which config this process runs: the job's own process ends it here when its kept state is another's."""
        self._say(digest=digest)

    def __call__(self, state: dict[str, torch.Tensor], weights_digest: str) -> None:
        self._hand("state", {k: v.cpu() for k, v in state.items()})

    def keep(self, state: dict, record: dict, elapsed_s: float) -> None:
        """Hand out what continuing after the last checkpoint needs: `state` (`resume_state`), the record of this
        process's steps so far and where its clock stands."""
        self._hand("keep", state, record=record, elapsed_s=elapsed_s)


class Kept:
    """A job's state directory (`--state`), which only the job's own process reads and writes: `state.json`, holding
    the record so far (which names the job: its config's digest, seed, split and VERSION), where the clock stands,
    and the name of the one file beside it, the tensors the route's process continues from (`resume_state`). The
    record's last segment states the SHA-512 of each part of those tensors (`kept`, by `kept_sha512`), which the
    job's own process hashes from this directory's copy. A new state is written beside the old one and takes its
    place by a rename, so a job stopped while it writes leaves the old one whole."""

    SCHEMA = "training-slowdown/state/v1"

    def __init__(self, directory: Path):
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

    def refuse(self, why: str):
        raise SystemExit(f"training_slowdown: the state in {self.dir} isn't one this job continues: {why}. Remove "
                         f"{self.dir} to start again from θ.")

    def same(self, got: dict[str, str], stated: dict[str, str], where: str) -> None:
        """Refuse a state a part of which, as `where`, hashes to another SHA-512 than the record of the segment that
        kept it states."""
        for part in sorted(got.keys() | stated.keys()):
            if got.get(part) != stated.get(part):
                self.refuse(f"the SHA-512 of its {part}, as {where}, is {got.get(part)} and the record of the "
                            f"segment that kept it states {stated.get(part)}")

    def load(self, seed: int, split: str) -> dict | None:
        """The directory's state, for the job of `seed` and `split` to continue from; None where it holds none. One
        this tree didn't write for this job is refused by what differs: its schema, its `training_slowdown` VERSION,
        its seed or its split, a part of its tensors that doesn't hash to what its record's last segment states,
        or a model that doesn't hash to its record's last checkpoint. Its config's digest is compared once the
        route's process, which reads the route's VERSION and dials, names its own (`outside`)."""
        path = self.dir / "state.json"
        if not path.is_file():
            return None
        try:
            kept = json.loads(path.read_text())
        except ValueError as e:
            self.refuse(f"its state.json doesn't parse ({e})")
        if not isinstance(kept, dict) or kept.get("schema") != self.SCHEMA:
            self.refuse(f"its schema is {kept.get('schema') if isinstance(kept, dict) else None!r} and this tree's "
                        f"is {self.SCHEMA!r}")
        try:
            rec, name = kept["record"], Path(kept["tensors"]).name
            last, stated = rec["checkpoints"][-1], dict(rec["segments"][-1]["kept"])
            said = (("training_slowdown VERSION", rec["training_slowdown"], VERSION), ("seed", rec["seed"], seed),
                    ("split", rec["split"], split))
            want = (last["step"], last.get("trained_sha256", last["output_sha256"]))
            float(kept["elapsed_s"]), rec["config"]["digest"], bytes.fromhex(rec["digest_key"] or ""), len(rec["steps"])
        except (LookupError, TypeError, ValueError) as e:
            self.refuse(f"its state.json lacks what a state holds ({type(e).__name__}: {e})")
        for what, its, mine in said:
            if its != mine:
                self.refuse(f"its {what} is {its!r} and this job's is {mine!r}")
        if not (self.dir / name).is_file():
            self.refuse(f"its tensors, {name}, are missing")
        try:
            held = torch.load(self.dir / name, map_location="cpu", weights_only=True, mmap=True)
            got, model = kept_sha512(held), theta_sha256(held["model"])
        except Exception as e:  # noqa: BLE001 - whatever torch.load raises on a file that isn't a state's tensors
            self.refuse(f"its tensors, {name}, don't load as a state's ({type(e).__name__}: {e})")
        self.same(got, stated, f"{name} holds it")
        if model != want[1]:
            self.refuse(f"its model hashes to {model} and its record's last checkpoint, of step {want[0]}, to "
                        f"{want[1]}")
        return kept | {"tensors": name}

    def write(self, tensors: Path, record: dict, elapsed_s: float) -> None:
        """Keep `record` and the file at `tensors`, which moves here, as the directory's one state. The SHA-512 of
        each of its parts goes into the record's last segment, hashed from the copy here: the route's process,
        which wrote the file at `tensors`, can't write this one."""
        name = f"tensors-{record['checkpoints'][-1]['step']:07d}.pt"
        with open(tensors, "rb") as src, open(self.dir / f"{name}.tmp", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(self.dir / f"{name}.tmp", self.dir / name)
        tensors.unlink()
        kept = kept_sha512(torch.load(self.dir / name, map_location="cpu", weights_only=True, mmap=True))
        record = record | {"segments": [*record["segments"][:-1], record["segments"][-1] | {"kept": kept}]}
        with open(self.dir / "state.json.tmp", "w") as f:
            json.dump({"schema": self.SCHEMA, "tensors": name, "elapsed_s": elapsed_s, "record": record}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.dir / "state.json.tmp", self.dir / "state.json")
        for old in self.dir.glob("tensors-*"):
            if old.name != name:
                old.unlink()

    def clear(self) -> None:
        """Remove the state, and the directory when nothing else is in it: the job ended by itself."""
        for p in [*self.dir.glob("state.json*"), *self.dir.glob("tensors-*")]:
            p.unlink()
        if not any(self.dir.iterdir()):
            self.dir.rmdir()


def _added(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a | b}


def outside_steps(structures: dict, parts: list[dict]) -> dict:
    """One step record for what several segments counted outside any step, each one's setup and what a stopped one
    counted after its last kept step: their counts and times added, and their structures merged into one, which
    joins `structures`."""
    out = dict(parts[0])
    for p in parts[1:]:
        for k in ("tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "cpu_s", "wall_s", "gpu_s",
                  "exited"):
            out[k] += p[k]
        out["by"], out["device_macs"] = _added(out["by"], p["by"]), _added(out["device_macs"], p["device_macs"])
        out["weights"] = {k: out["weights"].get(k, []) + p["weights"].get(k, []) for k in out["weights"] | p["weights"]}
        out["shapes"] = out["shapes"] | p["shapes"]
        mine, its = structures[out["struct"]], structures[p["struct"]]
        log: dict[tuple, list[int]] = {}
        for *sig, count, macs in mine["log"] + its["log"]:
            entry = log.setdefault(tuple(sig), [0, 0])
            entry[0] += count
            entry[1] += macs
        struct = {"updated": sorted(mine["updated"] + its["updated"], key=lambda u: (u[1], u[0])),
                  "log": sorted([list(k) + v for k, v in log.items()])}
        out["struct"] = structure_id(struct)
        structures[out["struct"]] = struct
    return out


def joined(prior: dict | None, rec: dict) -> dict:
    """The job's record over its segments: `rec`, the record of the segment that just ran (with its scores), after
    `prior`, the record the job kept of the segments before it (None: `rec` is the first). Its steps and
    checkpoints are every segment's in order, a checkpoint's `macs` counting from the job's first step. Its count
    is the segments' sum: `totals` adds up, and `setup` holds what every segment counted outside its steps
    (`outside_steps`), so a segment's own setup is counted, and the steps a stopped segment ran after its last kept
    checkpoint, which the next one ran again, are not. `segments` lists each one's first step, its steps, its count
    and its commit, and of one that was continued the digests of the state it kept last (`Kept.write`). A segment
    whose record names another job than the kept one's is refused."""
    first = len(prior["steps"]) if prior else 0
    segment = {"first": first, "steps": len(rec["steps"]), "macs": rec["totals"]["macs"], "commit": rec.get("commit")}
    if prior is None:
        return rec | {"segments": [segment]}
    for name, its, mine in (("config digest", prior["config"]["digest"], rec["config"]["digest"]),
                            ("seed", prior["seed"], rec["seed"]), ("split", prior["split"], rec["split"]),
                            ("training_slowdown VERSION", prior["training_slowdown"], rec["training_slowdown"]),
                            ("digest key", prior["digest_key"], rec["digest_key"]),
                            ("first step", first, rec["steps"][0]["step"] if rec["steps"] else first)):
        if its != mine:
            raise SystemExit(f"training_slowdown: the kept record's {name} is {its!r} and the segment that "
                             f"continued it states {mine!r}: no record is written")
    structures = prior["structures"] | rec["structures"]
    before = prior["totals"]["macs"]
    totals = {k: _added(v, rec["totals"].get(k, {})) if isinstance(v, dict) else v + rec["totals"].get(k, 0)
              for k, v in prior["totals"].items()}
    return rec | {
        "setup": outside_steps(structures, [prior["setup"], prior["tail"], rec["setup"]]), "structures": structures,
        "steps": prior["steps"] + rec["steps"], "totals": totals,
        "checkpoints": prior["checkpoints"] + [cp | {"macs": cp["macs"] + before} for cp in rec["checkpoints"]],
        "unseen_writes": prior["unseen_writes"] + rec["unseen_writes"],
        "new_weights": sorted({*prior["new_weights"], *rec["new_weights"]}),
        "unknown_ops": _added(prior["unknown_ops"], rec["unknown_ops"]),
        "foreign_ops": _added(prior["foreign_ops"], rec["foreign_ops"]),
        "invalid": prior["invalid"] + [p for p in rec["invalid"] if p not in prior["invalid"]],
        "flags": flags(totals, float(rec.get("peak_macs_per_s", 0))), "segments": prior["segments"] + [segment]}


def outside(a: argparse.Namespace, argv: list[str]) -> int:
    """The job's own process, which runs no route's code. It reads the job's held-out split, starts the route's
    process (`inside`) in the sandbox with every registered final set and the job's state directory hidden, scores
    each state that process hands out, under the digest it hashes the state to itself, keeps what continuing needs
    where the job has a state directory (`Kept`), and writes the record with the scores in (`with_scores`) over
    every segment the job ran in (`joined`). Told to stop (SIGTERM, to itself or to the route's process), it ends
    the route's process and returns `STOPPED` with the state left; ending any other way it removes the state."""
    reg = starts.registry(a.starts)
    scratch = Path(tempfile.mkdtemp(prefix="training-slowdown-"))

    def stop(signum, frame):
        raise Stopped

    told = signal.signal(signal.SIGTERM, stop) if a.retrace is None else None
    child = None
    try:
        state = Kept(a.state) if a.state is not None and a.retrace is None else None
        hide = [x for d in [*sandbox.final_sets(reg), *([state.dir] if state else [])] for x in ("--hide", str(d))]
        route = [sys.executable, "-m", "training_slowdown.sandbox", *hide, "--",
                 sys.executable, "-m", "training_slowdown.run", *argv, "--inside", str(scratch)]
        env = os.environ | {"PYTHONPATH": os.pathsep.join([str(HERE.parent), *filter(None, sys.path)])}
        if a.retrace is not None:
            return subprocess.run(route, env=env).returncode
        kept = state.load(a.seed, a.split) if state else None
        prior, stands = (kept["record"] if kept else None), None
        if kept:
            # a copy: through a link the route's process could write the kept file
            shutil.copyfile(state.dir / kept["tensors"], scratch / "kept.pt")
            stands = {"tensors": "kept.pt", "step": len(prior["steps"]), "elapsed_s": kept["elapsed_s"],
                      "digest_key": prior["digest_key"]}
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
                try:
                    done.write(json.dumps(stands).encode() + b"\n")
                except BrokenPipeError:
                    print("training_slowdown: the route's process ended before it was told where the job stands",
                          file=sys.stderr)
                for line in lines:
                    said = json.loads(line)
                    if "kept" in said:
                        state.same(said["kept"], prior["segments"][-1]["kept"], "the route's process read it")
                    elif "digest" in said:
                        if prior and said["digest"] != prior["config"]["digest"]:
                            state.refuse(f"its config digest is {prior['config']['digest']} and this job's is "
                                         f"{said['digest']}")
                    elif "state" in said:
                        path = scratch / Path(said["state"]).name
                        got = torch.load(path, map_location=a.device, weights_only=True)
                        path.unlink()
                        sha = state_sha256({k: digest(v) for k, v in got.items()})
                        scores.append((sha, evaluator(got, sha)))
                    else:
                        state.write(scratch / Path(said["keep"]).name,
                                    joined(prior, with_scores(said["record"], scores)), said["elapsed_s"])
                    try:
                        done.write(b"\n")
                    except BrokenPipeError:
                        print(f"training_slowdown: the route's process closed its end before it was told its state "
                              f"{len(scores) - 1} was scored; no further state is read", file=sys.stderr)
                        break
        if child.returncode == -signal.SIGTERM:
            raise Stopped
        # the job ends by itself from here: with its record, or failed
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        record = scratch / "run.json"
        if child.returncode or not record.is_file():
            if state:
                state.clear()
            return child.returncode or 1
        rec = joined(prior, with_scores(json.loads(record.read_text()), scores))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "run.json").write_text(json.dumps(rec) + "\n")
        if state:
            state.clear()
        return report(rec)
    except Stopped:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        left = f"its next run continues from the state in {a.state}" if a.state else "it keeps no state (--state)"
        print(f"training_slowdown: the job was told to stop: {left}", file=sys.stderr)
        return STOPPED
    finally:
        if told is not None:
            signal.signal(signal.SIGTERM, told)
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
    an interface that is up, or a registered final set or the job's state directory it can read an entry of, is a
    failed job. It takes the kept tensors the job's own process hands it, if any, and says what each part of them
    hashes to as it holds it, before it imports the route; then it names its config's digest. The job's own process
    ends it at the first when the tensors aren't what the record of the segment that kept them states, and at the
    second when the kept state is another config's. Then it trains as the module's docstring says, from θ or from
    the kept tensors, and leaves its record, the checkpoints unscored, in the job's scratch directory."""
    reg = starts.registry(a.starts)
    hidden = sandbox.final_sets(reg)
    if reached := sandbox.refusals([*hidden, *([a.state.resolve()] if a.state is not None else [])]):
        raise SystemExit(f"training_slowdown: a route's code runs only inside the job's sandbox, and this process is "
                         f"outside it: {'; '.join(reached)}. Start the job without --inside: it makes the sandbox.")
    if a.retrace is not None:
        return retrace(a)
    torch.cuda.graphs.CUDAGraph.replay = lambda self: (_ for _ in ()).throw(
        RuntimeError("training_slowdown: CUDA graph replay runs work the counter can't see"))
    own = Outside(a.inside, a.channel) if a.channel else None
    kept = own.kept if own else None
    state = None
    if kept:
        # read, removed and hashed before the route is imported: none of its code finds the file, and what this
        # process continues from is what the job's own process finds its record to state
        state = torch.load(a.inside / kept["tensors"], map_location=a.device, weights_only=True)
        (a.inside / kept["tensors"]).unlink()
        own.holds(kept_sha512(state))
    imports = Imports()
    with no_bytecode():
        cfg = config.load(a.config, reg)
        route = config.route(cfg.route)
    if own:
        own.names(cfg.digest)
    s = cfg.start
    theta = starts.load(s, a.starts_root or Path(reg["installed"]))
    data = Data.of(s.data, reg)
    torch.manual_seed(a.seed)

    def keep(trained: dict[str, torch.Tensor]) -> None:
        tensors = resume_state(trained, optimizers, stream)
        with te.uncounted():
            own.keep(tensors, te.so_far(meta, imports.problems()), te.elapsed())

    te = Harness(theta, a.out, seed=a.seed, tier=a.tier, device=a.device, start=s.entry(), evaluator=own,
                 split=a.split, budget_s=60 * cfg.budget.get("minutes", 0), keep_weights=a.keep_weights,
                 digests=a.digests != "off", families=a.families, record=a.inside / "run.json", kept=kept,
                 keep=keep if own and a.state is not None else None)
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
                train(te, cfg, route, stream, model, optimizers, state)
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
