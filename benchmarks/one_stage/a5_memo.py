"""a5's checks by the verifier, each asked once per input (`memo`, which `a5.main` runs under, each audit's summary.json
counting the checks asked and reused as ``verifier_checks``): a check the verifier answered is reused while every argument
it reads is the same, so the negatives' fresh streams, which register the honest registration and draw the honest sample
on its bytes, don't send the window's whole registration to the Lean verifier of record again.

Each of `PURE`'s checks is a function of its arguments on both sides (the Lean's piece and the Python beside it), so the
same arguments are the same verdict. A check is keyed by the SHA-512 of its name, `verifier.RECORD_VERDICT` and every
argument bound to its signature with the defaults applied, pickled: the registration record and each copy the verifier
holds it to, a `Program` by its digest and annotations (computed once per object: a Program is never changed once
built). A registration with anything changed, or any copy changed (R2's program, R3's ports, R4's leaf layer), is
another key and is checked again; a check whose arguments pickle can't write is never kept (`Memo.unkeyed` says
why). Not kept: `coins` (the operating system's randomness), `run` (each stream's whole run, on its own coins and
evidence), and any verdict whose Lean side refused, which may be the Lean failing to run rather than its answer. A
reused verdict is a copy. The cache is the harness's: the verifier is unchanged, and outside `memo` every check runs."""
from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import inspect
import pickle
from collections import Counter
from typing import Any, Callable, Iterator

from verity.ml.circuits.codec import canonical_json, program_digest
from verity.ml.circuits.program import Program
from verity.core.protocols.sampled_proofs.one_stage import verifier as V
from verity.core.protocols.sampled_proofs.one_stage.verdict import Verdict

PURE = ("check", "check_public", "select", "audit_record")


class _Sha512:
    def __init__(self) -> None:
        self.h = hashlib.sha512()

    def write(self, b: bytes) -> int:
        self.h.update(b)
        return len(b)


def _answered(got: Any) -> bool:
    """Whether the Lean answered `got`, a check's verdict, a select's `(draw, verdict)` or an audit record."""
    v = got[1] if isinstance(got, tuple) else got
    side = (v.verifiers if isinstance(v, Verdict) else v.get("verifiers")) or {}
    return "refused" not in (side.get("lean") or {})


class Memo:
    """The verdicts kept by key, and per check how many the verifier was asked (`asked`), how many were reused
    (`reused`), and why the last of its arguments that pickle couldn't write was not keyed (`unkeyed`)."""

    def __init__(self) -> None:
        self.kept: dict[bytes, Any] = {}
        self.asked: Counter[str] = Counter()
        self.reused: Counter[str] = Counter()
        self.unkeyed: dict[str, str] = {}
        self._programs: dict[int, tuple[Program, tuple[str, bytes]]] = {}

    def _program(self, p: Program) -> tuple[str, bytes]:
        got = self._programs.get(id(p))
        if got is None or got[0] is not p:
            got = self._programs[id(p)] = (p, (program_digest(p), canonical_json(p.annotations)))
        return got[1]

    def key(self, name: str, fn: Callable, args: tuple, kwargs: dict) -> bytes | None:
        """The check's key, or None where an argument doesn't pickle."""
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        items = [(k, ("program", *self._program(v)) if isinstance(v, Program) else v) for k, v in bound.arguments.items()]
        out = _Sha512()
        try:
            pickle.Pickler(out, protocol=5).dump((name, V.RECORD_VERDICT, items))
        except (pickle.PicklingError, TypeError, AttributeError, ValueError, RecursionError) as e:
            self.unkeyed[name] = f"{type(e).__name__}: {e}"[:200]
            return None
        return out.h.digest()

    def wrap(self, name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            key = self.key(name, fn, args, kwargs)
            if key is not None and key in self.kept:
                self.reused[name] += 1
                return copy.deepcopy(self.kept[key])
            self.asked[name] += 1
            got = fn(*args, **kwargs)
            if key is not None and _answered(got):
                self.kept[key] = copy.deepcopy(got)
            return got
        return call

    def counts(self) -> dict[str, dict]:
        return {"asked": dict(self.asked), "reused": dict(self.reused), "unkeyed": dict(self.unkeyed)}


_active: Memo | None = None


@contextlib.contextmanager
def memo() -> Iterator[Memo]:
    """The verifier's `PURE` checks through one `Memo` while the block runs, and the verifier's own again after it; a
    block inside another shares the outer one's."""
    global _active
    if _active is not None:
        yield _active
        return
    m, own = Memo(), {name: getattr(V, name) for name in PURE}
    for name, fn in own.items():
        setattr(V, name, m.wrap(name, fn))
    _active = m
    try:
        yield m
    finally:
        for name, fn in own.items():
            setattr(V, name, fn)
        _active = None


def memoized(fn: Callable) -> Callable:
    """`fn` run under `memo`."""
    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        with memo():
            return fn(*args, **kwargs)
    return call


def counts() -> dict[str, dict] | None:
    """The active memo's `Memo.counts`, None outside `memo`."""
    return None if _active is None else _active.counts()
