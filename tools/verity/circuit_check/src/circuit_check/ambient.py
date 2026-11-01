"""``circuit-check --ambient TARGET..``: each target's flat lowering (`verity_flock.ir_lower.lower`, laid out as
``flock-ir-unit/v2`` text by `ir_lower.netlist`) is a function of the Definition alone, its builders' operands and statics,
and of nothing else they could read. Two fresh interpreters lower every target:

  plain      PYTHONHASHSEED=0, the cwd as given, the targets in the order given, each lowered once;
  perturbed  another PYTHONHASHSEED (str and bytes hashes, so the order of a set of strings), objects allocated before
             anything is imported (addresses: ``id``, an object's default hash, the order of a set of objects), unrelated
             variables set and TZ, LC_ALL and LANG changed, a temporary cwd, bits in the circuit ahead of the target's
             (absolute bit numbers), and the targets in reverse order, each lowered twice (what one lowering leaves for the
             next: a global, a mutable closure, a memo keyed by less than what it caches);

and every text must be one text. In both, each read of the environment (``os.environ``, so ``os.getenv``), of a clock (the
`time` module's) and each unseeded draw (`random`'s functions and an unseeded `random.Random`, ``os.urandom`` and so
`secrets`, `numpy.random`'s global functions and an unseeded ``default_rng``) made while a lowering runs is recorded with
where in this tree's code it was made, and is a failure whatever its value. Not seen: state both interpreters hold alike (a
file a builder reads, a value an import computed from such state), a perturbation that by chance changes nothing a builder
reads, and a clock read through `datetime` (which only the changed TZ perturbs).

`ambiguous_words` is the static part: a word primitive that two registered Boolean Definitions build with different builders,
and that `ir_lower.PIECES` holds no piece of its own for, is lowered by whichever registered first (`ir_lower._boolean_build`),
so by the order its modules were imported.

This module imports only the standard library until a lowering needs more, so a child installs its recorder first.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

REPO = str(Path(__file__).resolve().parents[5]) + os.sep
_SELF = str(Path(__file__).resolve())

PLAIN: dict[str, Any] = {"hashseed": "0", "heap": 0, "junk": 0, "reverse": False, "builds": 1, "cwd": False, "env": {}}
PERTURBED: dict[str, Any] = {"hashseed": "4242", "heap": 100_000, "junk": 37, "reverse": True, "builds": 2, "cwd": True,
                             "env": {"TZ": "Pacific/Chatham", "LC_ALL": "C", "LANG": "C", "VERITY_AMBIENT": "perturbed"}}

CLOCKS = ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns", "process_time",
          "process_time_ns", "thread_time", "thread_time_ns", "clock_gettime", "clock_gettime_ns")
DRAWS = ("random", "randint", "randrange", "randbytes", "getrandbits", "choice", "choices", "shuffle", "sample", "uniform",
         "triangular", "gauss", "normalvariate", "lognormvariate", "expovariate", "vonmisesvariate", "gammavariate",
         "betavariate", "paretovariate", "weibullvariate", "binomialvariate")
NUMPY_DRAWS = ("random", "rand", "randn", "randint", "random_sample", "random_integers", "choice", "shuffle", "permutation",
               "bytes", "uniform", "normal", "standard_normal")

_reads: list[str] | None = None
_keep: list = []


def _note(what: str) -> None:
    if _reads is None:
        return
    f = sys._getframe(2)
    while f is not None and (not f.f_code.co_filename.startswith(REPO) or f.f_code.co_filename == _SELF):
        f = f.f_back
    _reads.append(what if f is None else f"{what} at {f.f_code.co_filename[len(REPO):]}:{f.f_lineno} ({f.f_code.co_name})")


def _wrap(owner: Any, name: str, what: str, when: Any = None) -> None:
    f = getattr(owner, name, None)
    if f is None:
        return

    def read(*a: Any, **k: Any) -> Any:
        if when is None or when(*a, **k):
            _note(what)
        return f(*a, **k)

    setattr(owner, name, read)


def _install() -> None:
    """Record the reads (`_note`) of the environment, the clocks and unseeded draws, from here on."""
    import random
    import time

    import numpy.random as npr

    for n in CLOCKS:
        _wrap(time, n, f"time.{n}")
    for n in DRAWS:
        _wrap(random, n, f"random.{n}")
    _wrap(random.Random, "seed", "an unseeded random.Random", lambda self, a=None, *_a, **_k: a is None)
    _wrap(os, "urandom", "os.urandom")
    _wrap(os, "getrandom", "os.getrandom")
    random._urandom = os.urandom  # type: ignore[attr-defined]  (SystemRandom's, so secrets')
    for n in NUMPY_DRAWS:
        _wrap(npr, n, f"numpy.random.{n}")
    _wrap(npr, "default_rng", "an unseeded numpy.random.default_rng", lambda seed=None, *_a, **_k: seed is None)

    class Environ(type(os.environ)):  # type: ignore[misc]
        def __getitem__(self, key: str) -> str:
            _note(f"os.environ[{key!r}]")
            return super().__getitem__(key)

        def __iter__(self):
            _note("os.environ's names")
            return super().__iter__()

    os.environ.__class__ = Environ


def resolve(spec: str) -> Any:
    """The Definition ``spec`` names: ``module:name``, else a Definition id (`targets.definition`)."""
    if ":" in spec and "{" not in spec.split(":", 1)[0]:
        mod, name = spec.split(":", 1)
        return getattr(importlib.import_module(mod), name)
    from . import targets as TG

    return TG.definition(spec)


def text(target: Any, junk: int = 0) -> str:
    """``target``'s flat lowering as ``flock-ir-unit/v2`` text, lowered into a circuit that holds ``junk`` other bits first."""
    from verity_flock import ir_lower as IL

    from verity.ml.circuits.boolean.forms import Circuit

    from . import structural as ST

    C = Circuit()
    if junk:
        j = C.input("ambient", junk)
        for a, b in pairwise(j):
            C.AND(a, b)
        del C.inputs["ambient"]
    C, _ins, words = ST._lowered(target, C)
    lay = IL.layout(C, list(C.inputs), [f"o{j}" for j in range(len(words))])
    return IL.netlist(lay, "unit")


def _child(opts: dict) -> None:
    """One interpreter's lowerings (`PLAIN`, `PERTURBED`): per target its texts' sha256 and what each lowering read, to
    ``opts["out"]``."""
    global _reads
    _keep.extend([object() for _ in range(opts["heap"])] + [bytes(i % 61) for i in range(opts["heap"])])
    _install()
    sys.path[:0] = opts["path"]
    targets = opts["targets"][::-1] if opts["reverse"] else opts["targets"]
    out: dict[str, dict] = {}
    for spec in targets:
        rec = out[spec] = {"digests": [], "reads": []}
        try:
            t = resolve(spec)
            for _ in range(opts["builds"]):
                _reads = []
                try:
                    rec["digests"].append(hashlib.sha256(text(t, opts["junk"]).encode()).hexdigest())
                finally:
                    rec["reads"] += _reads
                    _reads = None
        except Exception as e:  # noqa: BLE001  (a target that does not lower is that target's failure)
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["reads"] = sorted(set(rec["reads"]))
    Path(opts["out"]).write_text(json.dumps(out))


_CHILD = "import json, sys; from circuit_check import ambient; ambient._child(json.loads(sys.argv[1]))"


def lowerings(targets: list[str], *, path: tuple[str, ...] = ()) -> dict[str, dict[str, dict]]:
    """Per interpreter (``plain``, ``perturbed``), per target: its texts' sha256s, what its lowerings read, and the error
    if one raised. ``path``: directories put first on each interpreter's ``sys.path``."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, p in (("plain", PLAIN), ("perturbed", PERTURBED)):
            res = Path(tmp) / f"{name}.json"
            opts = {**p, "targets": list(targets), "path": [str(Path(d).resolve()) for d in path], "out": str(res)}
            env = {**os.environ, **p["env"], "PYTHONHASHSEED": p["hashseed"]}
            r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(opts)], env=env, cwd=tmp if p["cwd"] else None,
                               capture_output=True, text=True, check=False)
            if r.returncode or not res.exists():
                err = f"the {name} interpreter exited {r.returncode}: {r.stderr.strip()[-300:]}"
                out[name] = {t: {"digests": [], "reads": [], "error": err} for t in targets}
            else:
                out[name] = json.loads(res.read_text())
    return out


def check(targets: list[str], *, path: tuple[str, ...] = ()) -> list[dict]:
    """Per target a result as `checks.check_definition` gives one (``failures``, ``warnings``), with ``ambient``: the verdict
    (``same``, ``differs``, ``reads`` or ``error``), the text's sha256 when it is one, and how many lowerings agree."""
    from . import checks as CK

    runs = lowerings(targets, path=path)
    out = []
    for t in targets:
        F = CK.Findings()
        per = {n: runs[n][t] for n in runs}
        for n, r in per.items():
            if "error" in r:
                F.fail("ambient", "error", t, f"the {n} interpreter's lowering raised {r['error']}")
        reads = sorted({x for r in per.values() for x in r["reads"]})
        if reads:
            F.fail("ambient", "reads", t, f"its lowering read {reads[0]}" + (f" and {len(reads) - 1} more" if reads[1:] else ""),
                   reads=reads)
        got = {n: r["digests"] for n, r in per.items()}
        texts = {d for ds in got.values() for d in ds}
        if len(texts) > 1:
            F.fail("ambient", "differs", t, "its lowering is other text in a perturbed interpreter (" + ", ".join(
                f"{n}: {' '.join(d[:12] for d in ds)}" for n, ds in got.items()) + "): it reads state besides its operands")
        verdict = ("error" if any("error" in r for r in per.values()) else "differs" if len(texts) > 1 else
                   "reads" if reads else "same")
        rec = {"verdict": verdict, "lowerings": sum(map(len, got.values())),
               **({"sha256": next(iter(texts))} if len(texts) == 1 else {"sha256s": got})}
        out.append({"failures": F.failures, "warnings": F.warnings, "ambient": rec})
    return out


def ambiguous_words() -> dict[str, list[str]]:
    """Each word primitive that registered Boolean Definitions build with two or more different builders, and that
    `ir_lower.PIECES` lowers by one of them (no piece of its own; a piece the Boolean export's tracer wrapped, as it wraps
    each one once the process exports, is the one it wraps): their ids, in registration order. Every registry is loaded
    first (`targets.load_registries`)."""
    import inspect

    from verity_flock import ir_lower as IL

    from verity.ml.circuits.defs import REGISTRY

    from . import targets as TG

    TG.load_registries()
    IL._boolean_ops()
    by: dict[str, list] = {}
    for d in list(REGISTRY.defs.values()):
        w, build = getattr(d, "word", None), getattr(d, "build", None)
        if callable(build) and getattr(w, "id", None):
            by.setdefault(w.id, []).append(d)
    out = {}
    for word, ds in by.items():
        builds = {id(d.build) for d in ds}
        if len(builds) > 1 and id(inspect.unwrap(IL.PIECES.get(word))) in builds:
            out[word] = [d.id for d in ds]
    return out


__all__ = ["CLOCKS", "PERTURBED", "PLAIN", "ambiguous_words", "check", "lowerings", "resolve", "text"]
