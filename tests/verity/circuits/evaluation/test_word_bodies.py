"""Each registered word primitive's evaluator (`reference.evaluate`) is its Boolean body evaluated on bits (`bits.evaluate_words`,
PROTOCOL.md §4.9: the body fixes C's bits; the word evaluator is a fast form of it), on its edge words and seeded random words.

A word evaluator a verifier-side reference reads is held first, and says who reads it (`verifier_readers`): PoUW's Lean row
primitives (their vector script) and Security's FP32, scalar and cast definitions (this suite's Lean vector tests). A primitive
without a body is listed by name, never skipped: `NO_BODY_READ`, a named limit, when a verifier-side reference reads it, and
`NO_BODY` otherwise. Each list shrinks as bodies are written; a listed primitive that has a body warns until it is struck off."""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
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

#: verifier-read word primitives with no Boolean body yet: each a named limit until its body is written
NO_BODY_READ = {"PouwBf16Neg_v1", "PouwETilde_v1", "PouwF32SqrtRn_v1", "PouwSqAcc_v1", "PouwSqCeil_v1", "PouwSqGe_v1"}
#: the other word primitives with no Boolean body yet
NO_BODY = {
    "PouwAdd64Hi_v1", "PouwAdd64Lo_v1", "PouwAdd8_v1", "PouwAddSub8_v1", "PouwBlackwellE4m3SaltDead_v1",
    "PouwBlackwellE4m3Window_v1", "PouwCh32_v1", "PouwChi32_v1", "PouwDot16U8S8_v1", "PouwFormX3_v1", "PouwFormX_v1",
    "PouwHopperE4m3SaltDead_v1", "PouwHopperE4m3Window_v1", "PouwI32ToF32Rn_v1", "PouwMaj32_v1", "PouwNegBetaSum16_v1",
    "PouwNoiseByte_v1", "PouwPackBE4_v1", "PouwRotl64Hi_v1", "PouwRotl64Lo_v1", "PouwSha512BigSigma0Hi_v1",
    "PouwSha512BigSigma0Lo_v1", "PouwSha512BigSigma1Hi_v1", "PouwSha512BigSigma1Lo_v1", "PouwSha512SmallSigma0Hi_v1",
    "PouwSha512SmallSigma0Lo_v1", "PouwSha512SmallSigma1Hi_v1", "PouwSha512SmallSigma1Lo_v1", "PouwSub8_v1",
    "PouwWindowWord_v1", "PouwXor32_v1",
}


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


def _vectors(p: PrimitiveDefinition) -> list[np.ndarray]:
    """One row per width-generic edge word (every leaf that word), then seeded random rows with edges among them."""
    params = [t for _, t in p.signature()[0]]
    rng = np.random.default_rng(zlib.crc32(p.id.encode()))
    rand = sample_uniform(rng, N - 6, params)
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


@pytest.mark.parametrize("fid", sorted(NO_BODY_READ))
def test_named_limit_verifier_read_word_evaluator_without_a_body(fid: str) -> None:
    if fid in BODIES:
        warnings.warn(f"{fid} has a Boolean body now: strike it from NO_BODY_READ", stacklevel=1)
        _check(fid)
        return
    pytest.xfail(f"named limit: {fid}'s word evaluator, read by {READERS.get(fid, [])}, has no Boolean body to hold it to")


def test_primitives_without_a_body_are_listed() -> None:
    missing = set(PRIMS) - set(BODIES)
    read = {k for k in missing if k in READERS}
    assert not read - NO_BODY_READ, (f"verifier-read word primitives with no Boolean body, not in NO_BODY_READ: "
                                     f"{ {k: READERS[k] for k in sorted(read - NO_BODY_READ)} }")
    assert not (missing - read) - NO_BODY, f"word primitives with no Boolean body, not in NO_BODY: {sorted((missing - read) - NO_BODY)}"
    assert not (NO_BODY - set(PRIMS)) | (NO_BODY_READ - set(PRIMS)), "listed ids that are no registered word primitive"
    stale = sorted((NO_BODY | NO_BODY_READ) - missing)
    if stale:
        warnings.warn(f"these listed primitives have a Boolean body now; strike them from NO_BODY / NO_BODY_READ: {stale}",
                      stacklevel=1)


def test_verifier_readers_name_registered_primitives() -> None:
    assert READERS and not set(READERS) - set(PRIMS), sorted(set(READERS) - set(PRIMS))
