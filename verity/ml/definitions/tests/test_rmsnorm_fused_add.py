"""`rmsnorm_fused_split`'s `RMSNormFusedCudaAdd_v1` parts and `RMSNormFusedCudaTail_v1`, wired as a window wires them (part j's x the norm's x
on [j·C, (j+1)·C), its res the whole residual row, the tail reading the parts' r in order), return `RMSNormFusedCuda_v2`'s
two outputs word for word: on Gaussian rows at three scales, on uniformly random words (NaN, ±inf, ±0 and subnormals among
them), and on rows whose residual add, squares or variance overflow, underflow or cancel. Their gates are v2's, and at
Llama-3.2-1B's down_proj each unit fits a C-Flock proof block where v2 does not."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import rmsnorm_fused_split as NS
from verity.ml.windows import interface as IF

#: (N, EPS, C): the norm after Llama-3.2-1B's down_proj in its 8 parts, and whole as one; 18 warps of one stride in 9 parts;
#: N = 1536's half-filled second stride in 3; four strides (three FFMAs a thread) in 16; a lone partial warp in 4
CASES = [(2048, 1e-5, 256), (2048, 1e-6, 2048), (576, 1e-5, 64), (1536, 1e-5, 512), pytest.param(4096, 1e-5, 256, marks=pytest.mark.slow),
         (16, 1e-6, 4)]
MAX, INF, NAN, TINY, BIG = 0x7F7F, 0x7F80, 0x7FC1, 0x0001, 0x5F80  # bf16: the largest finite, +inf, a NaN, 2^-133, 2^64
#: C-Flock's proof block needs a row with a SHA-512 compression before its last block, so a row over this many bytes
SHA512_LAST_BLOCK_BYTES = 111
ROWS = 256


def _flat(v) -> list[int]:
    return [w for u in v for w in _flat(u)] if isinstance(v, (tuple, list)) else [int(v)]


def _gauss(rng, n: int, scale: float) -> np.ndarray:
    return ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _rows(rng, N: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(kind, x, res, w) rows of bf16 words."""
    def g(scale: float = 1.0) -> np.ndarray:
        return _gauss(rng, N, scale)

    def full(v: int) -> np.ndarray:
        return np.full(N, v, np.uint16)

    def put(a: np.ndarray, at: dict[int, int]) -> np.ndarray:
        a = a.copy()
        for j, v in at.items():
            a[j % N] = v
        return a

    zero, h = full(0), g()
    return [
        *((f"gauss x{s}", g(s), g(s), g()) for s in (0.02, 1.0, 40.0)),
        ("random words", *(rng.integers(0, 1 << 16, (3, N)).astype(np.uint16))),
        ("zero row: var 0, s = rsqrtf(EPS)", zero, zero, g()),
        ("x = -res: every r a cancelled +0", h, h ^ 0x8000, g()),
        ("-0 + -0", full(0x8000), full(0x8000), g()),
        ("bf16 add overflows to inf", full(MAX), full(MAX), g()),
        ("squares overflow f32", full(BIG), zero, g()),
        ("squares underflow", full(TINY), zero, g()),
        ("one NaN element, in the last part", put(g(), {-1: NAN}), g(), g()),
        ("one inf element", put(g(), {N // 3: INF}), g(), g()),
        ("w with NaN, inf and 0", g(), g(), put(g(), {1: NAN, 2: INF, 3: 0, 4: 0x8000})),
    ]


def _units(N: int, EPS: float, C: int) -> tuple[list, object]:
    return [bind(NS.RMSNormFusedCudaAdd, N=N, C=C, AT=j) for j in range(N // C)], bind(NS.RMSNormFusedCudaTail, N=N, EPS=EPS)


def _split(N: int, EPS: float, C: int, x: np.ndarray, res: np.ndarray, w: np.ndarray) -> list[int]:
    """The parts' and the tail's evaluations, each unit's ports its writers' rows as the window wires them."""
    parts, tail = _units(N, EPS, C)
    r = [v for j, part in enumerate(parts) for v in _flat(evaluate(part, [int(v) for v in x[j * C:(j + 1) * C]], [int(v) for v in res]))]
    return _flat(evaluate(tail, r, [int(v) for v in w]))


@pytest.mark.parametrize("N,EPS,C", CASES)
def test_the_add_parts_and_tail_are_the_norm(N, EPS, C):
    rng = np.random.default_rng(N + C)
    norm = bind(b1.RMSNormFusedCuda, N=N, EPS=EPS)
    for kind, x, res, w in _rows(rng, N):
        want = _flat(evaluate(norm, *([int(v) for v in a] for a in (x, res, w))))
        got = _split(N, EPS, C, x, res, w)
        assert len(got) == 2 * N and got == want, (kind, [j for j, (a, b) in enumerate(zip(got, want)) if a != b][:8])


def _gates(fn) -> Counter:
    """Its primitive gates by id, a batch counted once a member."""
    out: Counter = Counter()
    for node in fn.body.nodes:
        out[node.fn.id] += node.n
    return out


@pytest.mark.parametrize("N,EPS,C", [(2048, 1e-5, 256), (576, 1e-5, 64)])
def test_their_gates_are_the_norms_and_the_tail_returns_r_with_none(N, EPS, C):
    parts, tail = _units(N, EPS, C)
    norm = _gates(bind(b1.RMSNormFusedCuda, N=N, EPS=EPS))
    assert sum(map(_gates, parts), _gates(tail)) == norm
    assert norm - _gates(tail) == Counter({"Bf16AddF2fp_v1": N})


def _rows_of(fn, cuts: dict[str, int]) -> list[int]:
    """Each of ``fn``'s rows' bytes, ins (a cut port's rows each) then outs."""
    ins, _ = IF.unit_rows(fn, cuts)
    return [bits // 8 for _, _, bits, _ in ins] + [2 * words for _, words in IF.layout(fn).outs]


def test_at_llama_1b_down_proj_each_unit_fits_a_proof_block():
    """v2 reading x as down_proj's 512 slices of 4 columns (`Gemm_v2{K=8192, N=4}`) is 516 rows; an add part reads 64 of them
    and the 4 KiB residual row and writes its 512 B of r (66), and the tail reads r as the parts' 8 rows and w and writes
    normed and r (11). Each has a row over 111 B."""
    N, EPS, C, slice_words = 2048, 1e-5, 256, 4
    assert len(_rows_of(bind(b1.RMSNormFusedCuda, N=N, EPS=EPS), {"x": N // slice_words})) == 516
    parts, tail = _units(N, EPS, C)
    assert len(parts) == 8
    for part in parts:
        rows = _rows_of(part, {"x": C // slice_words})
        assert rows == [8] * 64 + [4096, 512] and len(rows) <= ROWS and max(rows) > SHA512_LAST_BLOCK_BYTES
    rows = _rows_of(tail, {"r": N // C})
    assert rows == [512] * 8 + [4096, 4096, 4096] and len(rows) <= ROWS and max(rows) > SHA512_LAST_BLOCK_BYTES


def test_the_catalogs_ids_hold_after_these_evaluations_in_one_process():
    """Evaluating the norm at statics no registry module binds registers constants (N and EPS as f32) in this process's
    registry that the catalog does not; `test_catalog`'s ids, checked after it here, are still the catalog's."""
    spec = importlib.util.spec_from_file_location("test_catalog_after_rmsnorm_fused_add", Path(__file__).with_name("test_catalog.py"))
    catalog_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog_test)
    _, x, res, w = _rows(np.random.default_rng(0), 24)[0]
    _split(24, 3e-6, 8, x, res, w)
    assert {"Const32[0x41c00000]_v1", "Const32[0x3649539c]_v1"} <= set(REGISTRY.defs) - set(catalog_test.fresh()["ids"])
    catalog_test.test_the_catalog_loads_every_definition_and_no_other()


@pytest.mark.parametrize("kw", [{"N": 2048, "C": 300, "AT": 0}, {"N": 2048, "C": 256, "AT": 8}, {"N": 2048, "C": 256, "AT": -1},
                                {"N": 2048, "C": 0, "AT": 0}])
def test_an_add_part_off_the_residual_row_is_refused_by_name(kw):
    with pytest.raises(ValueError, match=f"part AT={kw['AT']} of C={kw['C']} elements is not one of the N / C whole parts"):
        bind(NS.RMSNormFusedCudaAdd, **kw)
