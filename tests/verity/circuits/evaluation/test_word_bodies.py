"""Each registered word primitive's evaluator (`reference.evaluate`) is its Boolean body evaluated on bits (`bits.evaluate_words`,
PROTOCOL.md §4.9: the body fixes C's bits; the word evaluator is a fast form of it), on its edge words and seeded random words.

A word evaluator a verifier-side reference reads is held first, and says who reads it (`verifier_readers`): PoUW's Lean row
primitives (their vector script) and Security's FP32, scalar and cast definitions (this suite's Lean vector tests). A body
declared in the tree that the lookup does not reach fails; a primitive with no body anywhere is listed by name in `NO_BODY`,
never skipped, and a listed primitive that has a body warns until it is struck off."""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import random
import re
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

import verity.circuits.ops as ops
from verity.circuits import boolean as BL
from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.evaluation import evaluate, evaluate_batch, sample_uniform
from verity.circuits.evaluation.batch import type_dtype, type_shape
from verity.circuits.evaluation.bits import evaluate_words
from verity.circuits.ops import words as W
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import input_id

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
ROW_PRIMS = ROOT / "verity/protocols/resource_accounting/compute/pouw/verifier/lean/scripts/row_prims_vectors.py"
N = 64

#: word primitives with no Boolean Definition anywhere under `verity/` or the adapter's definitions (`DECLARING`)
NO_BODY: set[str] = set()
#: where a Boolean Definition of a word primitive may be declared (`definition(..., word=...)`)
DECLARING = (ROOT / "verity", ROOT / "adapters/vllm/verity_vllm/definitions")


def _catalog() -> dict[str, PrimitiveDefinition]:
    for m in pkgutil.iter_modules(ops.__path__):
        importlib.import_module(f"verity.circuits.ops.{m.name}")
    assert V.load() == []
    return {d.id: d for d in REGISTRY.defs.values()
            if isinstance(d, PrimitiveDefinition) and not BL.is_boolean(d) and input_id(d.id) is None}


def _table(name: str, attr: str) -> dict:
    spec = importlib.util.spec_from_file_location(f"_word_bodies_{name}", HERE / "ops" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def verifier_readers() -> dict[str, list[str]]:
    """Primitive id -> the verifier-side references that read its word evaluator."""
    out: dict[str, list[str]] = {}
    for name in sorted(set(re.findall(r"\bW\.(\w+)\.evaluate\b", ROW_PRIMS.read_text()))):
        out.setdefault(getattr(W, name).id, []).append("PoUW's Lean row primitives")
    for fn, *_ in _table("test_lean_fp32_vectors", "PRIMS").values():
        out.setdefault(fn.id, []).append("Security's Lean FP32")
    for fn, *_ in _table("test_lean_scalar_vectors", "PRIMITIVES").values():
        out.setdefault(fn.id, []).append("Security's Lean scalars and casts")
    return out


PRIMS = _catalog()
READERS = verifier_readers()


def _body(p: PrimitiveDefinition):
    try:
        return V.boolean_version(p)
    except V.NoBooleanVersion:
        return None


BODIES = {k: b for k, p in PRIMS.items() if (b := _body(p)) is not None}


def _edges(w: int) -> list[int]:
    top, mask = 1 << (w - 1), (1 << w) - 1
    return [0, 1, top, top | 1, mask, mask ^ top]


def _wide(t, rng: random.Random) -> np.ndarray:
    """`_vectors`' rows of one parameter with a leaf wider than 64 bits, as Python ints."""
    widths = t.leaf_widths()
    rows = [[_edges(w)[e] for w in widths] for e in range(6)]
    rows += [[rng.choice(_edges(w)) if rng.random() < 1 / 8 else rng.getrandbits(w) for w in widths] for _ in range(N - 6)]
    out = np.empty((N, len(widths)), dtype=object)
    out[:] = rows
    return out.reshape((N, *type_shape(t)))


def _vectors(p: PrimitiveDefinition) -> list[np.ndarray]:
    """One row per width-generic edge word (every leaf that word), then seeded random rows with edges among them."""
    params = [t for _, t in p.signature()[0]]
    seed = zlib.crc32(p.id.encode())
    if any(type_dtype(t).kind == "O" for t in params):
        rng = random.Random(seed)
        return [_wide(t, rng) for t in params]
    rand = sample_uniform(np.random.default_rng(seed), N - 6, params)
    out = []
    for t, r in zip(params, rand):
        widths = np.asarray(t.leaf_widths(), dtype=np.uint64).reshape(type_shape(t))
        top = np.uint64(1) << (widths - np.uint64(1))
        mask = np.where(widths >= 64, np.uint64(0xFFFFFFFFFFFFFFFF), (np.uint64(1) << np.minimum(widths, 63)) - np.uint64(1))
        edges = np.stack(np.broadcast_arrays(np.uint64(0), np.uint64(1), top, top | np.uint64(1), mask, mask ^ top))
        out.append(np.concatenate([edges.astype(type_dtype(t)), r]))
    return out


def _check(fid: str) -> None:
    p, b = PRIMS[fid], BODIES[fid]
    if not p.signature()[0]:
        bits, words = list(evaluate(b)), []
        for w in BL.elements(b.signature()[1]):
            words.append(sum(bit << j for j, bit in enumerate(bits[:w])))
            bits = bits[w:]
        assert tuple(evaluate(p)) == tuple(words), f"{fid}: its word evaluator is not its body {b.id}"
        return
    xs = _vectors(p)
    keep, want = [], []
    for i in range(xs[0].shape[0]):
        try:
            want.append(evaluate_batch(p, *(a[i:i + 1] for a in xs), kernel="reference")[0])
        except (ValueError, ArithmeticError, OverflowError):
            continue
        keep.append(i)
    assert keep, f"{fid}: no vector in its word evaluator's domain"
    got = evaluate_words(b, *(a[keep] for a in xs)).reshape(len(keep), -1)
    want = np.asarray(want).reshape(got.shape)
    bad = [keep[i] for i in np.flatnonzero((got != want).any(axis=1))]
    assert not bad, (f"{fid}: its word evaluator differs from its body {b.id} on {len(bad)} of {len(keep)} vectors, first "
                     f"{[a[bad[0]].tolist() for a in xs]}: word {want[keep.index(bad[0])].tolist()}, "
                     f"bits {got[keep.index(bad[0])].tolist()}")


@pytest.mark.parametrize("fid", sorted(k for k in BODIES if k in READERS))
def test_verifier_read_word_evaluator_is_its_body(fid: str) -> None:
    _check(fid)


@pytest.mark.parametrize("fid", sorted(k for k in BODIES if k not in READERS))
def test_word_evaluator_is_its_body(fid: str) -> None:
    _check(fid)


def _declared(stems: set[str]) -> dict[str, list[str]]:
    """Each of ``stems`` -> the files under `DECLARING` that declare a Boolean Definition computing it (``word=...<stem>``)."""
    pat = re.compile(r"\bword\s*=\s*(?:\w+\.)*(" + "|".join(map(re.escape, sorted(stems))) + r")\b")
    out: dict[str, list[str]] = {}
    for d in DECLARING:
        for f in sorted(d.rglob("*.py")):
            for m in pat.finditer(f.read_text(errors="replace")):
                out.setdefault(m[1], []).append(str(f.relative_to(ROOT)))
    return out


def test_every_word_primitive_has_a_body_the_lookup_reaches() -> None:
    missing = sorted(set(PRIMS) - set(BODIES))
    stems = {re.sub(r"_v\d+$", "", k.split("[", 1)[0]): k for k in missing}
    unreached = {stems[s]: fs for s, fs in _declared(set(stems)).items()} if stems else {}
    assert not unreached, (f"word primitives whose Boolean body is declared but not reached by `versions.boolean_version` "
                           f"(`versions.load` imports its catalogs): {unreached}")
    unlisted = [k for k in missing if k not in NO_BODY]
    assert not unlisted, ("word primitives with no Boolean body anywhere, not in NO_BODY: "
                          f"{ {k: READERS.get(k, 'no verifier reads it') for k in unlisted} }")
    assert NO_BODY <= set(PRIMS), f"NO_BODY names ids that are no registered word primitive: {sorted(NO_BODY - set(PRIMS))}"
    stale = sorted(NO_BODY - set(missing))
    if stale:
        warnings.warn(f"these NO_BODY primitives have a Boolean body now; strike them: {stale}", stacklevel=1)


def test_verifier_readers_name_registered_primitives() -> None:
    assert READERS and not set(READERS) - set(PRIMS), sorted(set(READERS) - set(PRIMS))
