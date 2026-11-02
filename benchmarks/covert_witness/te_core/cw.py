"""covert_witness's islands under the harness, unmodified: `python -m te_core.run --script islands/CLASS/train.py ...`.

A covert_witness train.py loads its start through `prepare.load_start`, reads batches from `prepare.TrainData.next`,
encodes its update with `prepare.encode_full|encode_lora|encode_seedlog` and submits with `prepare.submit`. The adapter
wraps those four, and nothing else of the program:

- `load_start` binds θ (the start's tensors, and its model's buffers as it builds them, so RoPE tables aren't weights)
  and registers the model's live tensors with the counter;
- each `TrainData.next` call ends the step in progress (if it computed on tokens) and starts the next: a step is the
  work from one batch to the next;
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
the harness's own start loader runs (a sealed start, in final mode). Subprocesses go to `guard.log`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .run import theta_sha256

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
               "score": {k: v for k, v in score.items() if k != "rule"}}
    rec["scored"] = {"split": scorer.split, "final_mode": bool(getattr(prepare, "FINAL_MODE", False)),
                     "problems": problems}
    return rec


class Adapter:
    def __init__(self, session, prepare, guard_log: Path | None = None):
        self.s, self.p = session, prepare
        self.rec: dict | None = None
        self.evaluating = False
        self.guard_log = guard_log
        self.forbidden = [os.path.realpath(p) for p in getattr(prepare, "UNREADABLE", [prepare.HELD_OUT])]
        self.encoded: dict[str, dict] = {}
        self.buckets: set[int] = set()
        self.submitted: dict = {}
        self.family = None
        self.declared_outside = 0

    # -- installation

    def install(self) -> None:
        p = self.p
        load_start, nxt, submit = p.load_start, p.TrainData.next, p.submit

        def load_start_(name, device=None):
            with self.s.uncounted(), self._reading():
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
        if hasattr(p, "declare_step"):
            p.declare_step = self.declare_step
        sys.addaudithook(self._hook)

    def declare_step(self, base: bytes = b"", suffixes=()) -> None:
        """The step in progress's weights as encodings from θ (the module docstring), into the family `E`."""
        if not self.s.in_steps():
            self.declared_outside += 1
            return
        with self.s.uncounted():
            if self.family is None:
                self.family = self.s.family("E", CWPlan(self.p, self.s.theta, self.s.device))
            self.family.put(self.s.current_step(), frame(base, suffixes))

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
                if any(s == f or s.startswith(f + os.sep) for f in self.forbidden):
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
        with self.s.uncounted():
            path = self.s.out / "E" / f"{sha}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        fmt = self.p.format_of(blob) if hasattr(self.p, "format_of") else "unknown"
        cp = {"step": self.s.current_step(), "macs": self.s.macs_so_far(), "val_bpb": None,
              "elapsed_s": self.s.elapsed(), "source": source, "e_bytes": len(blob), "e_sha256": sha, "format": fmt,
              "path": str(path.relative_to(self.s.out))}
        self.s.checkpoints.append(cp)
        self.encoded[sha] = cp
        return cp

    def submit(self, blob: bytes, model: dict[str, torch.Tensor]) -> None:
        cp = self.checkpoint(blob, "submit")
        cp["submitted"] = True
        self.submitted = {"e_sha256": cp["e_sha256"], "e_bytes": len(blob), "at": time.time(),
                          "elapsed_s": self.s.elapsed()}

    def meta(self) -> dict:
        return {"adapter": "covert_witness", "submitted": self.submitted, "declared_outside_steps": self.declared_outside,
                "start_record": {k: self.rec[k] for k in ("name", "digest", "tokenizer", "parameters")} if self.rec else {},
                "final_mode": bool(getattr(self.p, "FINAL_MODE", False))}
