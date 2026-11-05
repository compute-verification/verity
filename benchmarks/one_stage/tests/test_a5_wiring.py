"""a5's wiring negatives (`a5.wiring_negatives`) on a window whose layer 1 reads layer 0's rows: at two units a member, at one
(the torch loop's wired circuit, `--cap 1`) and on a fan-out (every reader unit reading writer unit 0), each negative is
stated and the verifier's wiring check refuses it. In the last two, rotating a wire's writers is the honest wire. Where no
other unit writes a row of the reader's words, the unit is rewired to a row of other words, and that is refused too."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as I

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
WIDE = bind(G.Gemm, K=16, N=32, **G.GEMM_AMPERE)
GID, WID = descriptor_id(GEMM), descriptor_id(WIDE)
UNITS = 4
RNG = np.random.default_rng(0)


def bf16(*shape: int) -> np.ndarray:
    return (np.asarray(RNG.standard_normal(shape) / 4, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def gemm(fn, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, x, w, kernel="reference"), dtype=np.uint16).reshape(len(x), -1)


def negatives(circuit: I.Circuit, cap: int, records) -> tuple[I.Window, dict[str, dict]]:
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, cap)
    for fn, x, w in records:
        win.record(fn, [x, w], [gemm(fn, x, w)])
    c = win.close(salt_key=bytes(32))
    assert win.mismatches() == {} and max(m.n for m in win.window.members) == cap
    return win, a5.wiring_negatives(win.window.members, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)})


@pytest.mark.parametrize("cap, step", [(2, 1), (1, 1), (2, 0)])
def test_each_wiring_negative_is_stated_and_refused_at_one_unit_a_member_and_on_a_fan_out(cap, step):
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=0, step=(step,)),))
    w, x0 = bf16(2 * UNITS, 16, 16), bf16(UNITS, 16)
    x1 = gemm(GEMM, x0, w[:UNITS])[np.arange(UNITS) * step]
    _, neg = negatives(circuit, cap, [(GEMM, np.concatenate([x0, x1]), w)])
    assert set(neg) == {"chunk-wired-to-another-unit (§16.13)", "wired-row-not-its-writer-s-output (§16.13)"}
    assert all(v["refused"] for v in neg.values()) and neg["chunk-wired-to-another-unit (§16.13)"]["words"] == "same"


def test_a_unit_no_other_unit_writes_its_words_for_is_rewired_to_other_words_and_refused():
    circuit = I.Circuit("fan", ((GEMM, 1), (WIDE, UNITS)), external=(f"{GID}/x",),
                        wires=(I.Wire(WID, "x", GID, "o0", at=0, radix=(UNITS,), base=0, step=(0,)),))
    x0, w0 = bf16(1, 16), bf16(1, 16, 16)
    win, neg = negatives(circuit, 1, [(GEMM, x0, w0), (WIDE, np.repeat(gemm(GEMM, x0, w0), UNITS, 0), bf16(UNITS, 32, 16))])
    assert [dict(m.out_rows)["o0"] for m in win.window.members] == [16] + [32] * UNITS
    assert neg["chunk-wired-to-another-unit (§16.13)"] == {"refused": True, "words": "other"}
