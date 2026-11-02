"""covert_witness's islands under the harness, unmodified: `python -m te_core.run --script islands/CLASS/train.py ...`.

A covert_witness train.py loads its start through `prepare.load_start`, reads batches from `prepare.TrainData.next`,
encodes its update with `prepare.encode_full|encode_lora|encode_seedlog` and submits with `prepare.submit`. The adapter
wraps those four, and nothing else of the program:

- `load_start` binds θ (the start's tensors, and its model's buffers as it builds them, so RoPE tables aren't weights)
  and registers the model's live tensors with the counter;
- each `TrainData.next` call ends the step in progress (if it computed on tokens) and starts the next: a step is the
  work from one batch to the next;
- every E the program encodes during its steps is a checkpoint candidate, the first in each 5% of the run (of the
  planned steps, `hooks.plan`, else of the MAC budget): the harness scores decode(θ, E) on the held-out search set,
  uncounted, at the step in progress;
- `submit` writes the submission as before; the submitted E is always scored, and the record says whether the
  submitted model is decode(θ, E) bit for bit.

The held-out sets are unreadable from the program (an audit hook refuses every open, listing or stat under
`prepare.HELD_OUT`), except while the harness's own evaluator runs. Subprocesses go to `guard.log`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

#: the harness's split names in covert_witness's
SPLITS = {"val": "search", "test": "final"}
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


def blob_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


class Adapter:
    def __init__(self, session, prepare, guard_log: Path | None = None):
        self.s, self.p = session, prepare
        self.rec: dict | None = None
        self.eval_net = None
        self.evaluating = False
        self.guard_log = guard_log
        self.forbidden = str(Path(prepare.HELD_OUT).resolve())
        self.encoded: dict[str, dict] = {}
        self.buckets: set[int] = set()
        self.submitted: dict = {}

    # -- installation

    def install(self) -> None:
        p = self.p
        load_start, nxt, submit = p.load_start, p.TrainData.next, p.submit

        def load_start_(name, device=None):
            with self.s.uncounted():
                out = load_start(name, device or p.DEVICE)
            self.bind(*out)
            return out

        def next_(data, *args, **kwargs):
            self.s.boundary()
            return nxt(data, *args, **kwargs)

        def submit_(out, start, klass, blob, model, *args, **kwargs):
            with self.s.uncounted():
                submit(out, start, klass, blob, model, *args, **kwargs)
            self.submit(blob, model)

        p.load_start, p.TrainData.next, p.submit = load_start_, next_, submit_
        for name in ("encode_full", "encode_lora", "encode_seedlog"):
            if hasattr(p, name):
                setattr(p, name, self._encoder(getattr(p, name)))
        sys.addaudithook(self._hook)

    def _encoder(self, encode):
        def encode_(*args, **kwargs):
            with self.s.uncounted():
                blob = encode(*args, **kwargs)
            if self.s.in_steps():
                bucket = self.s.progress_bucket()
                if bucket not in self.buckets:
                    self.buckets.add(bucket)
                    self.checkpoint(blob, "encode")
            return blob
        encode_.__wrapped__ = encode
        return encode_

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
        start = {"name": rec["name"], "digest": rec["digest"], "ctx": int(rec["config"].get("ctx", self.p.EVAL_CTX)),
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
        if self.eval_net is None:
            with torch.device(device):
                self.eval_net = self.p.Net(self.rec["config"])
        self.eval_net.load({k: v for k, v in state.items() if not k.startswith("buffer/")})
        with self._reading():
            return float(self.p.evaluate_bpb(self.eval_net, SPLITS.get(split, split), self.rec["tokenizer"],
                                             device=device)["bpb"])

    def checkpoint(self, blob: bytes, source: str) -> dict:
        sha = blob_sha256(blob)
        if sha in self.encoded:
            return self.encoded[sha]
        with self.s.uncounted():
            state = self.p.decode(self.s.theta, blob, device=self.s.device)
        fmt = self.p.format_of(blob) if hasattr(self.p, "format_of") else "unknown"
        cp = self.s.checkpoint_state(self.s.current_step(), state,
                                     {"source": source, "e_bytes": len(blob), "e_sha256": sha, "format": fmt})
        self.encoded[sha] = cp
        self.last = (sha, state)
        return cp

    def submit(self, blob: bytes, model: dict[str, torch.Tensor]) -> None:
        cp = self.checkpoint(blob, "submit")
        sha, state = getattr(self, "last", ("", None))
        if sha != cp["e_sha256"]:
            with self.s.uncounted():
                state = self.p.decode(self.s.theta, blob, device=self.s.device)
        with self.s.uncounted():
            same = set(state) == set(model) and all(self.p.same_bits(model[k].detach().cpu(), state[k].detach().cpu())
                                                    for k in state)
        cp["submitted"] = True
        self.submitted = {"e_sha256": cp["e_sha256"], "e_bytes": len(blob), "model_is_decode": bool(same)}
        self.last = ("", None)

    def meta(self) -> dict:
        return {"adapter": "covert_witness", "submitted": self.submitted,
                "start_record": {k: self.rec[k] for k in ("name", "digest", "tokenizer", "parameters")} if self.rec else {}}
