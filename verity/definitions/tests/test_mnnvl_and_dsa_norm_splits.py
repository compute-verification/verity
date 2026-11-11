"""`RmsNormMnnvl_v1` and `DsaRMSNorm_v1` cut at their scale, the one f32 word every output element reads: the sum of squares
reading the whole row and apply unit j the row's elements [j·G, (j+1)·G), their weights and the scale return the norm's
output word for word on `test_rmsnorm_fused_add`'s rows (Gaussian rows at three scales, uniformly random words with NaN,
±inf, ±0 and subnormals among them, and rows whose squares or variance overflow, underflow or cancel). Their gates are the
norm's plus what the apply units recompute, their Boolean versions return their words, and at GLM-5.3's widths each unit
fits a Flock proof block and returns at most G words of a row."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation import evaluate
from verity.definitions.ops.prims import ZERO32
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import dsa_norm as DN
from verity.definitions.registry import prims as P
from verity.definitions.registry import tp_allreduce as TA, tp_allreduce_split as TS
from verity.commitments.windows import interface as IF

_spec = importlib.util.spec_from_file_location("rows_of_test_rmsnorm_fused_add", Path(__file__).with_name("test_rmsnorm_fused_add.py"))
_rows_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rows_test)
_rows, _flat, _gates = _rows_test._rows, _rows_test._flat, _rows_test._gates


def _mnnvl(N: int, SHOT: int, CLUSTER: int, FMA: bool, G: int) -> tuple[object, object, object]:
    statics = {"N": N, "SHOT": SHOT, "CLUSTER": CLUSTER, "EPS": 1e-5, "FMA": FMA}
    return bind(TA.RmsNormMnnvl, **statics), bind(TS.RmsNormMnnvlSumSq, **statics), bind(TS.RmsNormMnnvlApply, G=G, SHOT=SHOT)


def _dsa(N: int, G: int) -> tuple[object, object, object]:
    return bind(DN.DsaRMSNorm, N=N, EPS=1e-5), bind(DN.DsaRMSNormSumSq, N=N, EPS=1e-5), bind(DN.DsaRMSNormApply, G=G)


#: (norm, statics): MNNVL's norm at one CTA of one and two warps and over a 2- and 4-CTA cluster, each shot, the + EPS unfused and
#: fused; GLM-5.3's at its hidden width at the clusters FlashInfer picks for it (8 up to 16 tokens, 4 up to 33, 2 up to the
#: fused path's 42); the DSA norm at its least row and GLM-5.3's kv_a and q_a rows
CASES = [*((_mnnvl, (N, shot, cluster, fma, G)) for N, cluster, G in ((256, 1, 64), (512, 1, 128), (512, 2, 128), (1024, 4, 128))
           for shot in (1, 2) for fma in (False, True)),
         *(pytest.param(_mnnvl, (6144, shot, cluster, True, 96), marks=pytest.mark.slow)
           for shot, cluster in ((1, 8), (2, 4), (2, 2))),
         *((_dsa, (N, 64)) for N in (256, 512)), pytest.param(_dsa, (2048, 64), marks=pytest.mark.slow)]


def _split(sumsq, apply, N: int, G: int, x: np.ndarray, w: np.ndarray) -> list[int]:
    """The sum of squares' scale, then the apply units' outputs joined in order."""
    (scale,) = _flat(evaluate(sumsq, [int(v) for v in x]))
    return [v for j in range(N // G)
            for v in _flat(evaluate(apply, *([int(v) for v in a[j * G:(j + 1) * G]] for a in (x, w)), scale))]


@pytest.mark.parametrize("units,statics", CASES)
def test_the_sum_of_squares_and_apply_units_are_the_norm(units, statics):
    norm, sumsq, apply = units(*statics)
    N, G = statics[0], statics[-1]
    for kind, x, _, w in _rows(np.random.default_rng(sum(statics)), N):
        want = _flat(evaluate(norm, [int(v) for v in x], [int(v) for v in w]))
        got = _split(sumsq, apply, N, G, x, w)
        assert len(got) == N and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])


@pytest.mark.parametrize("units,statics,recomputed", [
    *((_mnnvl, (6144, shot, cluster, True, 96), lambda N, G, shot=shot: {P.Bf16ToF32.id: N * (shot == 2), ZERO32.id: N // G})
      for shot, cluster in ((1, 8), (2, 4))),
    (_dsa, (2048, 64), lambda N, G: {P.Bf16ToF32.id: N})])
def test_their_gates_are_the_norms_and_what_the_apply_units_recompute(units, statics, recomputed):
    """An apply unit converts its G elements to f32 again (the two-shot MNNVL sum and the DSA sum read them too; the one-shot
    MNNVL sum squares in bf16), and MNNVL's makes its own +0 for the weights' flush."""
    norm, sumsq, apply = units(*statics)
    N, G = statics[0], statics[-1]
    split = _gates(sumsq) + Counter({gate: n * (N // G) for gate, n in _gates(apply).items()})
    assert split == _gates(norm) + Counter(recomputed(N, G))


@pytest.mark.parametrize("units,statics", [(_mnnvl, (256, 1, 1, False, 64)), (_mnnvl, (256, 2, 1, True, 64)), (_dsa, (256, 64))])
def test_on_bits_each_unit_is_the_switchs_boolean_version_and_returns_its_words(units, statics):
    """The switch takes each word unit to its `_v2` (`boolean_sigma`'s σ for MNNVL, `boolean_dsa_norm`'s traced one for DSA); on
    every row kind that returns the word unit's words, the apply unit on the row's first G elements."""
    _, sumsq, apply = units(*statics)
    G = statics[-1]
    rows = _rows(np.random.default_rng(7), statics[0])
    x, w = (np.stack([r[k] for r in rows]) for k in (1, 3))
    for word in (sumsq, apply):
        bits = boolean_version(word)
        assert BL.is_boolean(bits) and TR.word_of(bits).id == word.id
    scale = BS.evaluate_words(boolean_version(sumsq), x).reshape(len(rows))
    assert scale.tolist() == [_flat(evaluate(sumsq, [int(v) for v in x[i]]))[0] for i in range(len(rows))]
    got = BS.evaluate_words(boolean_version(apply), x[:, :G], w[:, :G], scale).reshape(len(rows), -1)
    assert got.tolist() == [_flat(evaluate(apply, [int(v) for v in x[i, :G]], [int(v) for v in w[i, :G]], int(scale[i])))
                            for i in range(len(rows))]


def _row_bytes(fn) -> list[int]:
    lay = IF.layout(fn)
    return [2 * w for _, w in lay.rows + lay.outs]


@pytest.mark.parametrize("units,statics", [(_mnnvl, (6144, 2, 2, True, 96)), (_dsa, (2048, 64)), (_dsa, (512, 64))])
def test_at_glm_widths_each_unit_fits_a_proof_block_and_returns_at_most_g_words_of_a_row(units, statics):
    """Each norm reads its row and the weights' and writes a whole row. Its sum of squares reads the row and writes the 4 B
    scale; an apply unit reads G elements of the row and the weights and the scale, and writes G elements."""
    norm, sumsq, apply = units(*statics)
    N, G = statics[0], statics[-1]
    assert _row_bytes(norm) == [2 * N] * 3
    for unit, rows in ((sumsq, [2 * N, 4]), (apply, [2 * G, 2 * G, 4, 2 * G])):
        assert _row_bytes(unit) == rows, unit.id
        IF.check_proof_block(unit)
