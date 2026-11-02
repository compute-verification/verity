"""covert_witness's islands under the harness, unmodified: `python -m te_core.run --script islands/CLASS/train.py ...`.

A covert_witness train.py loads its start through `prepare.load_start`, reads batches from `prepare.TrainData.next`,
encodes its update with `prepare.encode_full|encode_lora|encode_lora2|encode_seedlog|encode_sparse` and submits with
`prepare.submit`.
The adapter wraps those four kinds of call, and nothing else of the program. The encoders and `submit` run counted, on
the program's objects; only the harness's own work runs uncounted, on plain copies of what the program handed it
(`flops.plain`, made while the count runs) and with the harness's own instance of prepare.py (`private_copy`), so the
program's code never runs there. A program that rebinds an attribute that work looks up (torch's, numpy's, the standard
library's, or one of the harness's prepare's classes, `watched`) makes its run invalid.

- `load_start` binds θ (the start's tensors, and its model's buffers as it builds them, so RoPE tables aren't weights;
  the harness keeps its own copy of θ and of the start's record) and registers the model's live tensors with the
  counter;
- each `TrainData.next` call ends the step in progress (if it computed on tokens) and starts the next: a step is the
  work from one batch to the next. The batch is token data through any op that reads it, in every step while it lives
  (`flops.Counter.batch`);
- every E the program encodes during its steps is a checkpoint candidate, the first in each 5% of the run (of the
  planned steps, `hooks.plan`, else of the MAC budget): the harness writes it to `OUT/E/<sha256>.bin`, uncounted, and
  records it at the step in progress. Nothing is scored in the training job: the evaluation job (`score_run`, run by
  covert_witness's evaluate.py from the pinned commit, outside the training job's sandbox) decodes each E with the
  fixed decoder and scores it through `prepare.Scorer`: `val` is its search score on a fresh slice of the search pool
  drawn after its weights were hashed (te_core's `theta_sha256`, the record's `output_sha256`), paired with θ's on the
  same slice, and its promotion-set value is sealed under that digest; in final mode (`prepare.FINAL_MODE`) `val` is
  the final set's;
- `submit` writes the submission (E, the step log and its declarations); the submitted E is always a checkpoint, and
  the decoded model is the model: no claimed model is kept;
- `declare_step(base, suffixes)`, when the program calls it inside a step, puts that step in the run's counterexample
  family `E`, whose decoder is the fixed `prepare.decode` (`CWPlan`): E_t is the frame of the base and the suffixes, a
  weight the step read at one value is decode(θ, base)'s, and one read at several is decode(decode(θ, base), suffix_i)'s
  at its i-th value. So any E whose decoder rebuilds a step's weights within c MACs (a seed log, and any fixed format)
  is credited, not only adapter factors.

The held-out sets and the sealed side are not in the training job's sandbox at all (candidate.sh); inside it, an audit
hook also refuses every open, listing or stat under each of `prepare.UNREADABLE`, else `prepare.HELD_OUT`, except while
the harness's own start loader runs (a sealed start, in final mode). So are other runs' outputs, where an earlier E sits:
every run directory beside this one (`$RESEARCH_RUN_DIR`, the research runner's) and every candidate's scratch under
`prepare.ROOT/candidates` but this run's. The loaders that open files in C++ or Rust (`c_readers`) raise the same audit
event first. Subprocesses go to `guard.log`. A thread or a process the program starts makes its run invalid: the
counter's mode is per thread, and a child's work is its own. So do child processes' CPU time while the program ran, and
a child still alive when it returns (a spawn that raises no audit event). Loading or calling native code through ctypes
is a flag.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .flops import plain
from .run import theta_sha256

ENCODERS = ("encode_full", "encode_lora", "encode_lora2", "encode_seedlog", "encode_sparse", "encode_quant")
#: the standard library modules prepare's loader, decoder and scorer call into
STDLIB = ("hashlib", "io", "json", "lzma", "math", "os", "struct", "zlib")
PATH_EVENTS = {"open", "os.listdir", "os.scandir", "os.stat", "glob.glob", "os.chdir", "os.symlink", "os.link", "os.rename"}
#: `_posixsubprocess.fork_exec` (multiprocessing's spawn and forkserver starts) raises no audit event of its own
FORK_EXEC = "te_core.fork_exec"
SPAWN_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                FORK_EXEC}
THREAD_EVENTS = {"_thread.start_new_thread", "_thread.start_joinable_thread"}
NATIVE_EVENTS = {"ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle", "ctypes.call_function"}
#: child processes' CPU seconds while the program runs above which its run is invalid
CHILD_CPU_S = 0.5


def _children_cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def live_children() -> list[int]:
    """This process's child processes, alive or unreaped, from /proc."""
    me, out = os.getpid(), []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                ppid = int(Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == me:
                out.append(int(d))
    return out


def _audit_fork_exec() -> None:
    import _posixsubprocess

    fork_exec = _posixsubprocess.fork_exec
    if getattr(fork_exec, "_te_audited", False):
        return

    def fork_exec_(*args, _f=fork_exec):
        sys.audit(FORK_EXEC)
        return _f(*args)

    fork_exec_._te_audited = True
    _posixsubprocess.fork_exec = fork_exec_


def _opening(fn, name: str):
    def opened(filename, *args, **kwargs):
        sys.audit("open", filename, "r", 0)
        return fn(filename, *args, **kwargs)

    opened._te_audited = True
    opened.__name__ = opened.__qualname__ = name
    return opened


def c_readers() -> list[tuple[object, str]]:
    """The file loaders that open their file in C++ or Rust, which raises no audit event: torch's `from_file` and its
    storages', and safetensors' `safe_open` (which `load_file` calls)."""
    import safetensors
    import safetensors._safetensors_rust
    import safetensors.torch

    return [(torch, "from_file"), (torch.UntypedStorage, "from_file"), (safetensors, "safe_open"),
            (safetensors.torch, "safe_open"), (safetensors._safetensors_rust, "safe_open")]


def audit_c_reads() -> None:
    """Each of `c_readers` raises `open` with its path first, so the guard's audit hook sees it."""
    for owner, name in c_readers():
        fn = getattr(owner, name)
        if not getattr(fn, "_te_audited", False):
            wrapped = _opening(fn, name)
            setattr(owner, name, staticmethod(wrapped) if isinstance(owner, type) else wrapped)


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def import_prepare(script: Path):
    """covert_witness's `prepare`, from the nearest directory above the script's that has one, else the script's own,
    else the working directory's (a script sent with the run sits in its inputs), so a prepare.py beside an island's
    train.py doesn't replace the fixed one. Every module prepare.py names is imported here too, before the program
    runs: one it imports lazily (`model`, `tiktoken`) is then the fixed side's, not a same-named file the program puts
    first on sys.path."""
    s = Path(script).resolve()
    for d in [*s.parent.parents, s.parent, Path.cwd()]:
        if (d / "prepare.py").is_file():
            sys.path.insert(0, str(d))
            spec = importlib.util.spec_from_file_location("prepare", d / "prepare.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["prepare"] = mod
            spec.loader.exec_module(mod)
            names = re.findall(r"(?m)^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", (d / "prepare.py").read_text())
            for name in sorted(set(names)):
                try:
                    importlib.import_module(name)
                except ImportError:
                    pass
            return mod
    raise SystemExit(f"{script}: no prepare.py above it, beside it or in the working directory")


def private_copy(prepare):
    """A second instance of prepare.py for the harness's own work: the program rebinds and mutates the module it imports,
    never this one."""
    spec = importlib.util.spec_from_file_location("te_core_prepare", prepare.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def watched(own) -> list[tuple[str, object, str, object]]:
    """The attributes the harness's uncounted work looks up as it runs: torch's, numpy's, safetensors', tiktoken's and the
    standard library's, those of its prepare and of every class it defines (the program holds the harness's Net), and
    those of the fixed side's other modules that prepare imports (`sealed`). A program that rebinds one would run its own
    code in that work, uncounted, or change the harness's numbers."""
    import numpy as np
    import safetensors.torch
    import torch._dynamo  # its import rebinds torch.manual_seed, and torch.compile imports it
    import torch.nn.functional as F

    roots = [torch, torch.Tensor, F, torch.linalg, np, np.random, safetensors.torch, own,
             *map(importlib.import_module, STDLIB), *{id(o): o for o, _ in c_readers()}.values()]
    if (tiktoken := sys.modules.get("tiktoken")) is not None:
        roots.append(tiktoken)
    fixed = Path(own.__file__).resolve().parent
    roots += [m for name, m in list(sys.modules.items()) if name != "prepare"
              and Path(getattr(m, "__file__", None) or "/").resolve().parent == fixed]
    roots += [v for m in (torch.nn, own) for v in vars(m).values() if isinstance(v, type)]
    return [(getattr(o, "__name__", repr(o)), vars(o), k, v) for o in roots for k, v in list(vars(o).items())]


def blob_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


FRAME = b"CWSTEP1\n"


def frame(base: bytes, suffixes=()) -> bytes:
    """E_t: the base blob and its suffix blobs, each length-prefixed."""
    parts = [bytes(base), *(bytes(s) for s in suffixes)]
    return FRAME + struct.pack(f"<I{len(parts)}Q", len(parts), *map(len, parts)) + b"".join(parts)


def unframe(enc: bytes) -> tuple[bytes, list[bytes]]:
    if enc[:8] != FRAME:
        raise ValueError("E_t: not a covert_witness step frame")
    (n,) = struct.unpack("<I", enc[8:12])
    lens = struct.unpack(f"<{n}Q", enc[12:12 + 8 * n])
    at, parts = 12 + 8 * n, []
    for size in lens:
        parts.append(enc[at:at + size])
        at += size
    if at != len(enc) or not parts:
        raise ValueError("E_t: frame lengths don't add up")
    return parts[0], parts[1:]


class CWPlan:
    """A counterexample decoder over covert_witness's fixed formats (`prepare.decode`), in `codecs.Plan`'s interface:
    `keys` are θ's names, and `key@i` for each suffix; decode(θ, base) returns θ's dtypes, so a suffix applies to the
    base's tensors cast to θ's dtypes (exact for an fp32 start). Its cost is the larger of what the counter sees the decode
    compute and `prepare.rebuild_macs` of the base and of each suffix (θ's shapes)."""

    def __init__(self, prepare, theta: dict[str, torch.Tensor], device: str):
        self.p, self.theta, self.device = prepare, theta, device

    def spec(self) -> dict:
        return {"kind": "covert_witness", "decoder": "prepare.decode", "frame": FRAME.decode().strip()}

    def keys(self, enc: bytes) -> list[str]:
        _, suffixes = unframe(enc)
        return list(self.theta) + [f"{k}@{i}" for i in range(len(suffixes)) for k in self.theta]

    def decode(self, enc: bytes, bases: dict) -> dict[str, torch.Tensor]:
        base, suffixes = unframe(enc)
        s0 = self.p.decode(self.theta, base, device=self.device) if base else dict(self.theta)
        out = dict(s0)
        for i, suffix in enumerate(suffixes):
            out |= {f"{k}@{i}": v for k, v in self.p.decode(s0, suffix, device=self.device).items()}
        return out

    def cost_macs(self, enc: bytes, bases: dict) -> int:
        if not hasattr(self.p, "rebuild_macs"):
            return 0
        base, suffixes = unframe(enc)
        return sum(self.p.rebuild_macs(self.theta, b) for b in [*([base] if base else []), *suffixes])


HEAD_GEMMS = ("mm", "addmm", "bmm", "matmul", "linear", "baddbmm")


def narrowed_head(ops: dict, head: tuple[int, int], weights, tokens: dict) -> dict:
    """The analytic head work a narrowed head doesn't do, read op by op from a count (`ops`, `ledger._macs_log`'s: "phase
    class op shapes" -> [calls, MACs]), for the coordinator's 13:42Z floor rule: a record under the floor is scored when
    the counted op table accounts for the gap. `head` is the head's weight (rows, d); `weights` every other weight's shape;
    `tokens` the meter's ({"train": n, "forward": n}).

    A forward GEMM row [R, d] x [d, n] is the head on n of its rows when d < n < rows and (d, n) is no other weight's
    shape, and a backward row [R, n] x [n, d] is its input gradient when such a forward row has the same R and n: the
    whole head would add (rows - n)/n of either's MACs at the same token rows. Shapes can't tell such a slice from an
    n-wide matrix the program adds, so every row credited is listed, and the credit is at most the analytic head's own
    forward and input-gradient work."""
    rows, d = head
    other = {s for w in weights for s in (tuple(w), tuple(w)[::-1])}
    gemms = []
    for key, (calls, macs) in ops.items():
        phase, _, op, shapes = key.split(" ", 3)
        try:
            (a, b), (b2, c) = [tuple(s[-2:]) for s in json.loads(shapes)[-2:]]
        except (ValueError, TypeError):
            continue
        if op in HEAD_GEMMS and b == b2:
            gemms.append((phase, a, b, c, key, calls, macs))
    fwd = {(a, c) for phase, a, b, c, *_ in gemms if phase == "fwd" and b == d and d < c < rows and (d, c) not in other}
    saved, credited = {"fwd": 0, "bwd": 0}, []
    for phase, a, b, c, key, calls, macs in gemms:
        n = c if phase == "fwd" and b == d else b if phase == "bwd" and c == d else None
        if n and (a, n) in fwd:
            s = macs * (rows - n) // n
            saved[phase] += s
            credited.append({"op": key, "calls": calls, "macs": macs, "head_rows": n, "saved_macs": s})
    cap = {"fwd": (tokens.get("train", 0) + tokens.get("forward", 0)) * rows * d, "bwd": tokens.get("train", 0) * rows * d}
    total = sum(min(saved[k], cap[k]) for k in saved)
    return {"saved_macs": total, "saved_by_phase": saved, "cap_macs": cap, "credited": credited}


def consumed_tokens(log: list, shapes: dict) -> int:
    """The tokens a step's counted forward GEMMs consumed, from its structure's op log (`[phase, class, op, shapes,
    calls, MACs]` rows): a forward GEMM with one operand of a weight's shape (`shapes`: name -> shape, either way round)
    consumed its other operand's rows times its calls, over the number of weights of that shape; the step consumed the
    most any such GEMM shows. A zero-MAC lookup consumes nothing, and a GEMM with a weight's slice isn't read."""
    per = collections.Counter(tuple(sorted(s)) for s in shapes.values() if len(s) == 2)
    best = 0
    for phase, _, op, sig, calls, _ in log:
        if phase != "fwd" or op not in HEAD_GEMMS:
            continue
        mats = [tuple(s) for s in json.loads(sig) if len(s) >= 2]
        held = [i for i, s in enumerate(mats) if len(s) == 2 and tuple(sorted(s)) in per]
        if len(mats) != 2 or len(held) != 1:
            continue
        i = 1 - held[0]
        rows = math.prod(mats[i]) // mats[i][-1 if i == 0 else -2]
        best = max(best, rows * calls // per[tuple(sorted(mats[held[0]]))])
    return best


def readout_leaks(record: dict, tensors: dict) -> list[dict]:
    """The coordinator's 14:22Z rule for ref-data's 14:06Z LOOPHOLE, at ingest: before te_core 3.11 the counter keyed a
    program-made token tensor by its address, so zero-MAC lookups of distinct index tensors could raise a step's tokens,
    which bound its read-outs. Each step's read-outs (`exited`) are judged here against the tokens its counted forward
    GEMMs consumed (`consumed_tokens`, over θ's weights, `tensors`: the start's name -> [dtype, shape], and the step's
    own updated ones). A step whose read-outs reach a value per consumed token but stayed under its counted tokens is
    listed: its allowance came from index reads no counted work consumed, and the record is invalid."""
    shapes = {k: v[1] for k, v in tensors.items()}
    out = []
    for s in [record.get("setup", {}), *record["steps"], record.get("tail", {})]:
        exited, tokens = s.get("exited", 0), s.get("tokens", 0)
        if exited < 2 or exited >= max(tokens, 2):
            continue
        used = shapes | {k: v[1] for k, v in s.get("shapes", {}).items()}
        consumed = consumed_tokens(record["structures"][s["struct"]]["log"], used)
        if exited >= max(consumed, 2):
            out.append({"step": s["step"], "tokens": tokens, "consumed": consumed, "exited": exited})
    return out


def accounted_over(counted_flops: float, analytic_flops: float, saved_macs: int) -> float | None:
    """The harness's count over the run's own steps once a narrowed head's saving is taken out of the estimate."""
    rest = analytic_flops - 2 * saved_macs
    return counted_flops / rest if rest > 0 else None


def score_run(prepare, rec: dict, run_dir: Path, theta: dict[str, torch.Tensor], start: dict,
              device: str) -> dict:
    """The evaluation job's side of a training job's record: θ's whole-set score, then each checkpoint's E, read from
    `run_dir` and checked against the digest the harness recorded, decoded from θ with the fixed decoder, hashed, and
    scored and sealed through `prepare.Scorer`. Returns the record with the start's `val_bpb` and each checkpoint's
    `output_sha256`, `val_bpb` and `score`; a checkpoint whose E isn't the recorded one is left unscored and named in
    `scored.problems`."""
    if rec["start"].get("digest") != start["digest"]:
        raise SystemExit(f"te_core: the run's start digest is not {start['name']}'s registered one")
    scorer = prepare.Scorer(start, theta, device)
    rec["start"] |= {"val_bpb": float(scorer.start_bpb),
                     "val_bpb_source": f"θ on the whole {scorer.split} set, measured by the evaluation job"}
    problems = []
    for cp in rec["checkpoints"]:
        path = Path(run_dir) / cp.get("path", "")
        blob = path.read_bytes() if cp.get("path") and path.is_file() else None
        if blob is None or blob_sha256(blob) != cp["e_sha256"]:
            problems.append(f"step {cp['step']}: its E file is missing or not the recorded {cp['e_sha256'][:12]}")
            continue
        state = {k: v for k, v in prepare.decode(theta, blob, device=device).items() if not k.startswith("buffer/")}
        digest = theta_sha256(state)
        score = scorer.search(state, digest)
        scorer.seal(digest, state, score)
        cp |= {"output_sha256": digest, "val_bpb": float(score["bpb"]),
               "score": {k: v for k, v in score.items() if k != "rule"},
               "rebuild_macs": prepare.rebuild_macs(theta, blob) if hasattr(prepare, "rebuild_macs") else None}
    rec["scored"] = {"split": scorer.split, "final_mode": bool(getattr(prepare, "FINAL_MODE", False)),
                     "problems": problems}
    return rec


class Adapter:
    def __init__(self, session, prepare, guard_log: Path | None = None, own_dirs=()):
        self.s, self.p = session, prepare
        self.own = private_copy(prepare)
        self.rec: dict | None = None
        self.evaluating = False
        self.guard_log = guard_log
        self.forbidden = [os.path.realpath(p) for p in getattr(self.own, "UNREADABLE", [self.own.HELD_OUT])]
        run_dir = os.environ.get("RESEARCH_RUN_DIR")
        others = [Path(self.own.ROOT) / "candidates"] if hasattr(self.own, "ROOT") else []
        self.others = [os.path.realpath(p) for p in others + ([Path(run_dir).parent] if run_dir else [])]
        self.own_dirs = [os.path.realpath(p) for p in [*own_dirs, *([run_dir] if run_dir else [])]]
        self.encoded: dict[str, dict] = {}
        self.buckets: set[int] = set()
        self.submitted: dict = {}
        self.family = None
        self.declared_outside = 0
        self.watch: list = []
        self.rebound: set[str] = set()
        self.running = False
        self.started: dict[str, int] = {}
        self.native: dict[str, int] = {}
        self.children_cpu_s = 0.0
        self.alive: list[int] = []

    # -- installation

    def install(self) -> None:
        p, own = self.p, self.own
        nxt, submit = p.TrainData.next, p.submit

        def load_start_(name, device=None):
            name, device = plain(name), plain(device)
            with self.s.uncounted(), self._reading():
                self._check()
                net, rec, theta = own.load_start(name, device or own.DEVICE)
                mine = {k: v.clone() for k, v in theta.items()}
            self.bind(net, plain(rec), mine)
            return net, rec, theta

        def next_(data, *args, **kwargs):
            self.s.boundary()
            return self.s.batch(nxt(data, *args, **kwargs))

        def submit_(out, start, klass, blob, model, *args, **kwargs):
            blob, model = plain(blob), plain(model)
            submit(out, start, klass, blob, model, *args, **kwargs)
            self.submit(blob, model)

        p.load_start, p.TrainData.next, p.submit = load_start_, next_, submit_
        for name in ENCODERS:
            if hasattr(p, name):
                setattr(p, name, self._encoder(getattr(p, name)))
        if hasattr(p, "declare_step"):
            p.declare_step = self.declare_step
        _audit_fork_exec()
        audit_c_reads()
        sys.addaudithook(self._hook)
        self.watch = watched(own)

    def start(self) -> None:
        """The program starts running."""
        self.children_cpu_s -= _children_cpu_s()
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.alive = live_children()
        self.children_cpu_s += _children_cpu_s()

    def declare_step(self, base: bytes = b"", suffixes=()) -> None:
        """The step in progress's weights as encodings from θ (the module docstring), into the family `E`."""
        base, suffixes = plain(base), plain(tuple(suffixes))
        if not self.s.in_steps():
            self.declared_outside += 1
            return
        with self.s.uncounted():
            self._check()
            if self.family is None:
                self.family = self.s.family("E", CWPlan(self.own, self.s.theta, self.s.device))
            self.family.put(self.s.current_step(), frame(base, suffixes))

    def _encoder(self, encode):
        def encode_(*args, **kwargs):
            blob = plain(encode(*args, **kwargs))
            if self.s.in_steps():
                bucket = self.s.progress_bucket()
                if bucket not in self.buckets:
                    self.buckets.add(bucket)
                    self.checkpoint(blob, "encode")
            return blob
        encode_.__wrapped__ = encode
        return encode_

    def _check(self) -> None:
        for name, d, k, v in self.watch:
            if d.get(k, d) is not v:
                self.rebound.add(f"{name}.{k}")

    def _hook(self, event: str, args: tuple) -> None:
        if event in PATH_EVENTS and args and not self.evaluating:
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                try:
                    s = os.path.realpath(os.fsdecode(p))
                except (TypeError, ValueError):
                    return
                if _under(s, self.forbidden):
                    raise PermissionError(f"te_core: {s} is held out; only the harness's evaluator reads it")
                if _under(s, self.others) and not _under(s, self.own_dirs):
                    raise PermissionError(f"te_core: {s} is another run's; a run reads only its own outputs")
            return
        if self.running and (event in SPAWN_EVENTS or event in THREAD_EVENTS):
            self.started[event] = self.started.get(event, 0) + 1
        elif self.running and event in NATIVE_EVENTS:
            self.native[event] = self.native.get(event, 0) + 1
        if event in SPAWN_EVENTS and self.guard_log is not None:
            with open(self.guard_log, "a") as f:
                f.write(f"{event} {args[:2]!r}\n")

    # -- the harness's side

    def bind(self, net, rec: dict, theta: dict[str, torch.Tensor]) -> None:
        self.rec = rec
        buffers = {f"buffer/{n}": b for n, b in net.named_buffers()}
        with self.s.uncounted():
            extra = {k: b.detach().cpu().clone() for k, b in buffers.items()}
        start = {"name": rec["name"], "digest": rec["digest"], "ctx": int(rec["config"].get("ctx", self.own.EVAL_CTX)),
                 "vocab": int(rec["config"].get("vocab", 50304)), "parameters": rec.get("parameters")}
        self.s.bind(theta, start, net, net.params() | buffers, extra)

    @contextmanager
    def _reading(self):
        self.evaluating = True
        try:
            yield
        finally:
            self.evaluating = False

    def checkpoint(self, blob: bytes, source: str) -> dict:
        """E as a checkpoint of the step in progress, written for the evaluation job (`score_run`)."""
        sha = blob_sha256(blob)
        if sha in self.encoded:
            return self.encoded[sha]
        own = self.own
        with self.s.uncounted():
            self._check()
            path = self.s.out / "E" / f"{sha}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            fmt = own.format_of(blob) if hasattr(own, "format_of") else "unknown"
        cp = {"step": self.s.current_step(), "macs": self.s.macs_so_far(), "val_bpb": None,
              "elapsed_s": self.s.elapsed(), "source": source, "e_bytes": len(blob), "e_sha256": sha, "format": fmt,
              "path": str(path.relative_to(self.s.out))}
        self.s.checkpoints.append(cp)
        self.encoded[sha] = cp
        return cp

    def submit(self, blob: bytes, model: dict[str, torch.Tensor]) -> None:
        self._check()
        cp = self.checkpoint(blob, "submit")
        cp["submitted"] = True
        self.submitted = {"e_sha256": cp["e_sha256"], "e_bytes": len(blob), "at": time.time(),
                          "elapsed_s": self.s.elapsed()}

    def problems(self) -> list[str]:
        self._check()
        out = []
        if self.rebound:
            out.append(f"the program rebound what the harness's uncounted work calls: {sorted(self.rebound)[:6]}")
        if self.started:
            out.append(f"the program started threads or processes ({self.started}): their work runs outside the counter")
        if self.children_cpu_s > CHILD_CPU_S:
            out.append(f"child processes used {self.children_cpu_s:.1f} CPU-s while the program ran")
        if self.alive:
            out.append(f"child processes outlived the program: {self.alive[:5]}")
        return out

    def flags(self) -> list[str]:
        return [f"native: the program loaded or called native code through ctypes {self.native}"] if self.native else []

    def meta(self) -> dict:
        return {"adapter": "covert_witness", "submitted": self.submitted, "declared_outside_steps": self.declared_outside,
                "children_cpu_s": self.children_cpu_s, "native_calls": self.native,
                "start_record": {k: self.rec[k] for k in ("name", "digest", "tokenizer", "parameters")} if self.rec else {},
                "final_mode": bool(getattr(self.own, "FINAL_MODE", False))}
