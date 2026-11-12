"""`cuda_rmsnorm_split`'s units, wired as a window wires them (the sum of squares reading the whole row, apply unit j the
row's elements [j·G, (j+1)·G), their weights and the sum of squares' scale), return `RMSNormCudaVector8_v1`'s and
`RMSNormFusedCudaVector8_v1`'s outputs word for word on `test_rmsnorm_fused_add`'s rows: Gaussian rows at three scales,
uniformly random words (NaN, ±inf, ±0 and subnormals among them), and rows whose residual add, squares or variance
overflow, underflow or cancel. Their gates are the kernels' plus the N residual adds the fused apply units redo, their
Boolean versions (`boolean_norms_split`'s `_v2`) return their words, and at GLM-5.3's hidden width each unit fits a Flock proof
block and returns at most G words of a row."""
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
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import cuda_rmsnorm as C
from verity.definitions.registry import cuda_rmsnorm_split as NS
from verity.commitments.windows import interface as IF

_spec = importlib.util.spec_from_file_location("rows_of_test_rmsnorm_fused_add", Path(__file__).with_name("test_rmsnorm_fused_add.py"))
_rows_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rows_test)
_rows, _flat, _gates = _rows_test._rows, _rows_test._flat, _rows_test._gates

#: (N, EPS, BLOCK, G, fused): one vector a thread; 72 vectors on 32 threads, three or two each; the fused kernel's threads
#: past N / 8, which hold +0; GLM-5.3's hidden width at both kernels' launch blocks
CASES = [*((64, 1e-6, 8, 16, f) for f in (False, True)), *((576, 1e-5, 32, 64, f) for f in (False, True)),
         (1536, 1e-5, 192, 96, False), (1536, 1e-5, 256, 96, True),
         *(pytest.param(6144, 1e-5, block, G, f, marks=pytest.mark.slow)
           for block, G, f in ((768, 64, False), (256, 96, False), (1024, 64, True), (256, 96, True)))]


def _norm(N: int, EPS: float, BLOCK: int, fused: bool):
    return bind(C.RMSNormFusedCudaVector8 if fused else C.RMSNormCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK)


def _units(N: int, EPS: float, BLOCK: int, G: int, fused: bool) -> tuple[object, object]:
    if fused:
        return bind(NS.RMSNormFusedSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK), bind(NS.RMSNormFusedApplyCudaVector8, G=G)
    return bind(NS.RMSNormSumSqCudaVector8, N=N, EPS=EPS, BLOCK=BLOCK), bind(NS.RMSNormApplyCudaVector8, G=G)


def _split(N: int, EPS: float, BLOCK: int, G: int, fused: bool, x: np.ndarray, res: np.ndarray, w: np.ndarray) -> list[int]:
    """The sum of squares' scale, then the apply units' outputs, each output's G-element pieces joined in order."""
    sumsq, apply = _units(N, EPS, BLOCK, G, fused)
    rows = (x, res) if fused else (x,)
    (scale,) = _flat(evaluate(sumsq, *([int(v) for v in a] for a in rows)))
    pieces = [_flat(evaluate(apply, *([int(v) for v in a[j * G:(j + 1) * G]] for a in (*rows, w)), scale)) for j in range(N // G)]
    return [v for k in range(len(pieces[0]) // G) for p in pieces for v in p[k * G:(k + 1) * G]]


@pytest.mark.parametrize("N,EPS,BLOCK,G,fused", CASES)
def test_the_sum_of_squares_and_apply_units_are_the_kernel(N, EPS, BLOCK, G, fused):
    rng = np.random.default_rng(N + BLOCK + G)
    norm = _norm(N, EPS, BLOCK, fused)
    for kind, x, res, w in _rows(rng, N):
        want = _flat(evaluate(norm, *([int(v) for v in a] for a in ((x, res, w) if fused else (x, w)))))
        got = _split(N, EPS, BLOCK, G, fused, x, res, w)
        assert len(got) == (2 if fused else 1) * N and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])


@pytest.mark.parametrize("N,EPS,BLOCK,G", [(576, 1e-5, 32, 64), (6144, 1e-5, 256, 96)])
@pytest.mark.parametrize("fused", [False, True])
def test_their_gates_are_the_kernels_and_the_fused_apply_units_redo_the_residual_add(N, EPS, BLOCK, G, fused):
    sumsq, apply = _units(N, EPS, BLOCK, G, fused)
    split = _gates(sumsq) + Counter({gate: n * (N // G) for gate, n in _gates(apply).items()})
    assert split == _gates(_norm(N, EPS, BLOCK, fused)) + Counter({"Bf16AddF2fp_v1": N if fused else 0})


@pytest.mark.parametrize("fused", [False, True])
def test_on_bits_each_unit_is_the_switchs_boolean_version_and_returns_its_words(fused):
    """The switch takes each word unit to `boolean_norms_split`'s `_v2`, RSQRT bound to mufu's Boolean version; on every row kind
    that returns the word unit's words, the apply unit on the row's first G elements."""
    N, EPS, BLOCK, G = 64, 1e-6, 8, 16
    rows = _rows(np.random.default_rng(7), N)
    x, res, w = (np.stack([r[k] for r in rows]) for k in (1, 2, 3))
    ins = (x, res) if fused else (x,)
    sumsq, apply = _units(N, EPS, BLOCK, G, fused)
    for word in (sumsq, apply):
        bits = boolean_version(word)
        assert BL.is_boolean(bits) and TR.word_of(bits).id == word.id
    scale = BS.evaluate_words(boolean_version(sumsq), *ins).reshape(len(rows))
    assert scale.tolist() == [_flat(evaluate(sumsq, *([int(v) for v in a[i]] for a in ins)))[0] for i in range(len(rows))]
    part = [a[:, :G] for a in (*ins, w)]
    got = BS.evaluate_words(boolean_version(apply), *part, scale).reshape(len(rows), -1)
    assert got.tolist() == [_flat(evaluate(apply, *([int(v) for v in a[i]] for a in part), int(scale[i]))) for i in range(len(rows))]


def _row_bytes(fn) -> list[int]:
    lay = IF.layout(fn)
    return [2 * w for _, w in lay.rows + lay.outs]


def test_at_glm_hidden_width_each_unit_fits_a_proof_block_and_returns_at_most_g_words_of_a_row():
    """At N = 6144 the fused kernel's five rows are whole 12,288 B rows. Its sum of squares reads two of them and writes the
    4 B scale; an apply unit at G = 96 reads 192 B of x, res and w and the scale and writes 192 B of each output."""
    N, EPS, G = 6144, 1e-5, 96
    assert _row_bytes(_norm(N, EPS, 256, True)) == [2 * N] * 5
    cases = [(NS.RMSNormFusedSumSqCudaVector8, {"N": N, "EPS": EPS, "BLOCK": 256}, [2 * N, 2 * N, 4]),
             (NS.RMSNormSumSqCudaVector8, {"N": N, "EPS": EPS, "BLOCK": 256}, [2 * N, 4]),
             (NS.RMSNormFusedApplyCudaVector8, {"G": G}, [2 * G, 2 * G, 2 * G, 4, 2 * G, 2 * G]),
             (NS.RMSNormApplyCudaVector8, {"G": G}, [2 * G, 2 * G, 4, 2 * G])]
    for fn, statics, rows in cases:
        unit = bind(fn, **statics)
        assert _row_bytes(unit) == rows, fn
        IF.check_proof_block(unit)
