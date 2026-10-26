"""a5's wiring negatives (`a5.wiring_negatives`) on a window whose layer 1 reads layer 0's rows: at two units a member and at
one (the torch loop's wired circuit, `--cap 1`), each negative is stated and the verifier's wiring check refuses it. At one
unit a member, rotating a wire's writers is the honest wire, so the unit reads another unit's row instead."""
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
import a5  # noqa: E402

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)
UNITS = 4


def chained(cap: int) -> tuple[I.Window, dict[str, bytes]]:
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=0, step=(1,)),))
    rng = np.random.default_rng(0)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    w = bf16(rng.standard_normal((2 * UNITS, 16, 16)) / 4)
    x = [bf16(rng.standard_normal((UNITS, 16)))]
    y = [np.asarray(evaluate_batch(GEMM, x[0], w[:UNITS], kernel="reference"), dtype=np.uint16).reshape(UNITS, 16)]
    x.append(y[0])
    y.append(np.asarray(evaluate_batch(GEMM, x[1], w[UNITS:], kernel="reference"), dtype=np.uint16).reshape(UNITS, 16))
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, cap)
    win.record(GEMM, [np.concatenate(x), w], [np.concatenate(y)])
    c = win.close(salt_key=bytes(32))
    return win, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)}


@pytest.mark.parametrize("cap", [2, 1])
def test_each_wiring_negative_is_stated_and_refused_at_one_unit_a_member_too(cap):
    win, publics = chained(cap)
    assert win.mismatches() == {} and max(m.n for m in win.window.members) == cap
    neg = a5.wiring_negatives(win.window.members, publics)
    assert set(neg) == {"chunk-wired-to-another-unit (§16.13)", "wired-row-not-its-writer-s-output (§16.13)"}
    assert all(v["refused"] for v in neg.values())
