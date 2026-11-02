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
  work from one batch to the next;
- every E the program encodes during its steps is a checkpoint candidate, the first in each 5% of the run (of the
  planned steps, `hooks.plan`, else of the MAC budget): the harness scores decode(θ, E) on the held-out search set,
  uncounted, at the step in progress;
- `submit` writes the submission as before; the submitted E is always scored, and the record says whether the
  submitted model is decode(θ, E) bit for bit;
- `declare_step(base, suffixes)`, when the program calls it inside a step, puts that step in the run's counterexample
  family `E`, whose decoder is the fixed `prepare.decode` (`CWPlan`): E_t is the frame of the base and the suffixes, a
  weight the step read at one value is decode(θ, base)'s, and one read at several is decode(decode(θ, base), suffix_i)'s
  at its i-th value. So any E whose decoder rebuilds a step's weights within c MACs (a seed log, and any fixed format)
  is credited, not only adapter factors.

The held-out sets are unreadable from the program (an audit hook refuses every open, listing or stat under
`prepare.HELD_OUT`), except while the harness's own evaluator runs. Subprocesses go to `guard.log`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from .flops import plain

#: the harness's split names in covert_witness's
SPLITS = {"val": "search", "test": "final"}
ENCODERS = ("encode_full", "encode_lora", "encode_lora2", "encode_seedlog", "encode_sparse")
#: the standard library modules prepare's loader, decoder and evaluator call into
STDLIB = ("hashlib", "io", "json", "math", "struct", "zlib")
PATH_EVENTS = {"open", "os.listdir", "os.scandir", "os.stat", "glob.glob", "os.chdir", "os.symlink", "os.link"}
SPAWN_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}


def import_prepare(script: Path):
    """covert_witness's `prepare`, from the nearest directory at or above the script's that has one, else the working
    directory's (a script sent with the run sits in its inputs)."""
    for d in [Path(script).resolve().parent, *Path(script).resolve().parents, Path.cwd()]:
        if (d / "prepare.py").is_file():
            sys.path.insert(0, str(d))
            spec = importlib.util.spec_from_file_location("prepare", d / "prepare.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["prepare"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"{script}: no prepare.py at or above it")


def private_copy(prepare):
    """A second instance of prepare.py for the harness's own work: the program rebinds and mutates the module it imports,
    never this one."""
    spec = importlib.util.spec_from_file_location("te_core_prepare", prepare.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def watched(own) -> list[tuple[str, object, str, object]]:
    """The attributes the harness's uncounted work looks up as it runs: torch's, numpy's, safetensors' and the standard
    library's, and those of its prepare and of every class it defines (the program holds the harness's Net). A program
    that rebinds one would run its own code in that work, uncounted, or change the harness's numbers."""
    import numpy as np
    import safetensors.torch
    import torch._dynamo  # its import rebinds torch.manual_seed, and torch.compile imports it
    import torch.nn.functional as F

    roots = [torch, torch.Tensor, F, torch.linalg, np, safetensors.torch, own, *map(importlib.import_module, STDLIB)]
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
    base's tensors cast to θ's dtypes (exact for an fp32 start). Its cost is what the counter sees the decode compute."""

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
        return 0


class Adapter:
    def __init__(self, session, prepare, guard_log: Path | None = None):
        self.s, self.p = session, prepare
        self.own = private_copy(prepare)
        self.rec: dict | None = None
        self.eval_net = None
        self.evaluating = False
        self.guard_log = guard_log
        self.forbidden = str(Path(self.own.HELD_OUT).resolve())
        self.encoded: dict[str, dict] = {}
        self.buckets: set[int] = set()
        self.submitted: dict = {}
        self.family = None
        self.declared_outside = 0
        self.watch: list = []
        self.rebound: set[str] = set()

    # -- installation

    def install(self) -> None:
        p, own = self.p, self.own
        nxt, submit = p.TrainData.next, p.submit

        def load_start_(name, device=None):
            name, device = plain(name), plain(device)
            with self.s.uncounted():
                self._check()
                net, rec, theta = own.load_start(name, device or own.DEVICE)
                mine = {k: v.clone() for k, v in theta.items()}
            self.bind(net, plain(rec), mine)
            return net, rec, theta

        def next_(data, *args, **kwargs):
            self.s.boundary()
            return nxt(data, *args, **kwargs)

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
        sys.addaudithook(self._hook)
        self.watch = watched(own)

    def declare_step(self, base: bytes = b"", suffixes=()) -> None:
        """The step in progress's weights as encodings from θ (the module docstring), into the family `E`."""
        base, suffixes = plain(base), plain(tuple(suffixes))
        if not self.s.in_steps():
            self.declared_outside += 1
            return
        with self.s.uncounted():
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
                if s == self.forbidden or s.startswith(self.forbidden + os.sep):
                    raise PermissionError(f"te_core: {s} is held out; only the harness's evaluator reads it")
        elif event in SPAWN_EVENTS and self.guard_log is not None:
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
        if self.s.start.get("val_bpb") is None and self.s.evaluator is not None:
            with self.s.uncounted():
                self.s.start["val_bpb"] = float(self.s.evaluator(dict(theta), "val", self.s.device))
            self.s.start["val_bpb_source"] = "measured by the harness at load"

    @contextmanager
    def _reading(self):
        self.evaluating = True
        try:
            yield
        finally:
            self.evaluating = False

    def evaluate(self, state: dict[str, torch.Tensor], split: str, device: str) -> float:
        """(state, split, device) -> bits per byte of the held-out split, by `prepare.evaluate_bpb` on the start's
        architecture holding `state` (θ's tensors)."""
        self._check()
        if self.eval_net is None:
            with torch.device(device):
                self.eval_net = self.own.Net(self.rec["config"])
        self.eval_net.load({k: v for k, v in state.items() if not k.startswith("buffer/")})
        with self._reading():
            return float(self.own.evaluate_bpb(self.eval_net, SPLITS.get(split, split), self.rec["tokenizer"],
                                               device=device)["bpb"])

    def checkpoint(self, blob: bytes, source: str) -> dict:
        sha = blob_sha256(blob)
        if sha in self.encoded:
            return self.encoded[sha]
        own = self.own
        with self.s.uncounted():
            self._check()
            state = own.decode(self.s.theta, blob, device=self.s.device)
            fmt = own.format_of(blob) if hasattr(own, "format_of") else "unknown"
            macs = own.rebuild_macs(self.s.theta, blob) if hasattr(own, "rebuild_macs") else None
        cp = self.s.checkpoint_state(self.s.current_step(), state, {"source": source, "e_bytes": len(blob), "e_sha256": sha,
                                                                     "format": fmt, "rebuild_macs": macs})
        self.encoded[sha] = cp
        self.last = (sha, state)
        return cp

    def submit(self, blob: bytes, model: dict[str, torch.Tensor]) -> None:
        cp = self.checkpoint(blob, "submit")
        sha, state = getattr(self, "last", ("", None))
        if sha != cp["e_sha256"]:
            with self.s.uncounted():
                state = self.own.decode(self.s.theta, blob, device=self.s.device)
        with self.s.uncounted():
            self._check()
            same = set(state) == set(model) and all(self.own.same_bits(model[k].cpu(), state[k].detach().cpu())
                                                    for k in state)
        cp["submitted"] = True
        self.submitted = {"e_sha256": cp["e_sha256"], "e_bytes": len(blob), "model_is_decode": bool(same)}
        self.last = ("", None)

    def problems(self) -> list[str]:
        self._check()
        if not self.rebound:
            return []
        return [f"the program rebound what the harness's uncounted work calls: {sorted(self.rebound)[:6]}"]

    def meta(self) -> dict:
        return {"adapter": "covert_witness", "submitted": self.submitted, "declared_outside_steps": self.declared_outside,
                "start_record": {k: self.rec[k] for k in ("name", "digest", "tokenizer", "parameters")} if self.rec else {}}
