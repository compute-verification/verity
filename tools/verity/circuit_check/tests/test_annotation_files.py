"""The repository's `verity/annotations/v0` files (`verity.ml.circuits.annotations`): core's `verity/ml/circuits/ops/annotations.json` and the vLLM
registry's `annotations.json`. Every entry is well formed; every Definition key resolves, and no guard is stale, in a program
calling each specialization the circuit-check catalog reaches of that Definition; and each entry sits in its owner's file.

A key names a Definition by its stem (``RoPE_v1``): a full descriptor id holds only for served statics the catalog doesn't
bind. Groups resolve their members; ``call:k`` keys belong to one program and aren't written in these files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from circuit_check import targets as T
from verity.ml.circuits import annotations as A
from verity.ml.circuits import codec
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program

ROOT = Path(__file__).resolve().parents[4]
FILES = {
    "core": ROOT / "verity" / "ml" / "circuits" / "ops" / "annotations.json",
    "vllm": ROOT / "integrations" / "vllm" / "verity_vllm" / "program" / "registry" / "annotations.json",
}
_TARGET = re.compile(r"^(def|node|wire|port):([^/]+)")
#: specializations of one Definition validated against, smallest first
PER_FAMILY = 4


def _stem(key: str) -> str | None:
    m = _TARGET.match(key)
    return m[2].split("{", 1)[0] if m else None


@pytest.fixture(scope="module")
def families() -> dict[str, list]:
    by: dict[str, list] = {}
    for fn in T.reachable(T.catalog()).values():
        by.setdefault(T.family(fn), []).append(fn)
    return {k: sorted(v, key=lambda f: getattr(f, "gates", 0))[:PER_FAMILY] for k, v in by.items()}


def _signature(fn):
    if isinstance(fn, PrimitiveDefinition):
        return tuple(fn.params), fn.ret
    return tuple(fn.body.params), fn.body.ret_type


def _calling(fn) -> dict:
    """The descriptor of a program whose root calls ``fn`` once: ``Program(fn)`` would inline fn's body as its root."""
    params, ret = _signature(fn)
    call = CompositeDefinition("AnnotationsCall", 1, (), lambda S: (params, ret), lambda B, S, *xs: B.call(fn, *xs), register=False)
    return codec.encode_program(Program(bind(call)))


def _owner(stem: str) -> str:
    """Core's or the vLLM registry's: the module of the registered Definition's body (a specialization's body is built in
    `verity.ml.circuits`), or of a primitive's evaluator."""
    d = REGISTRY.defs.get(stem)
    impl = getattr(d, "evaluate", None) or getattr(d, "body", None)
    return "vllm" if getattr(impl, "__module__", "").startswith("verity_vllm") else "core"


def _files():
    return [pytest.param(owner, path, id=str(path.relative_to(ROOT))) for owner, path in FILES.items() if path.exists()]


@pytest.mark.parametrize("owner,path", _files())
def test_annotation_file(owner, path, families):
    ann = json.loads(path.read_text())
    assert ann.get("format") == A.FORMAT, f"{path.name}: not {A.FORMAT}"
    entries = ann["entries"]
    by_stem: dict[str, list] = {}
    for e in entries:
        key = e.get("key", "")
        assert not key.startswith("call:"), f"{key}: a call key names one program's Call, not a Definition"
        stem = _stem(key)
        if stem is not None:
            by_stem.setdefault(stem, []).append(e)
    problems = []
    for stem, mine in sorted(by_stem.items()):
        fns = families.get(stem)
        if not fns:
            problems.append(f"{stem}: no Definition the circuit-check catalog reaches")
            continue
        if _owner(stem) != owner:
            problems.append(f"{stem}: written in {owner}'s file, owned by {_owner(stem)}")
        for fn in fns:
            v = A.validate({"format": A.FORMAT, "entries": mine}, _calling(fn))
            problems += [f"{fn.id}: {x}" for x in v.errors] + [f"{fn.id}: stale {x}" for x in v.stale] + [f"{fn.id}: absent {x}" for x in v.absent]
    # The whole file once more, for what spans Definitions: each (key, field) set once, groups closed.
    some = _calling(next(iter(families.values()))[0])
    whole = A.validate(ann, some)
    problems += [str(x) for x in whole.errors if x["code"] not in ("unresolved",)]
    assert not problems, "\n".join(problems)
    # Carrying the file moves no program digest.
    with_ann = A.attach(some, ann)
    assert codec.program_digest(with_ann) == codec.program_digest(some)
    assert codec.program_digest(with_ann) == codec.program_digest(some)
