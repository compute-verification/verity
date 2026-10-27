"""`TritonBlockRMSNorm_v1` at DeepSeek-V4's q norm against rows the H200 committed (`fixtures/dsv4_q_norm_commit_20261021.json`:
a DeepSeek-V4-Pro B1 Commit): the Definition and its row kernel give
every committed output, and the epilogue in two roundings (`eps + mul(scale_a(sum), rcp(N))`, the PTX as written) misses each
fixture row the fixture says it misses, by as many words, so the rows keep telling ptxas's FFMA apart from it."""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import base64
import json
from pathlib import Path

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import rows
from verity.ml.definitions.numerics.dsa_norm_rows import block_rmsnorm_rows
from verity.ml.definitions.registry import dsa_norm
from verity.ml.definitions.registry.prims import bits_of
from verity_vllm.query.manifest import format as RM

FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "dsv4_q_norm_commit_20261021.json").read_text())
S = FIXTURE["statics"]
N, BLOCK, WARPS, EPS = S["N"], S["BLOCK"], S["WARPS"], float.fromhex(S["EPS"])


def _words(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<u2").astype(np.uint16)


X = np.stack([_words(r["x"]) for r in FIXTURE["rows"]])
W = np.stack([_words(FIXTURE["weights"][str(r["label"]["layer"])]) for r in FIXTURE["rows"]])
Y = np.stack([_words(r["y"]) for r in FIXTURE["rows"]])


def test_the_fixture_is_the_served_q_norm():
    assert (N, BLOCK, WARPS, np.float32(EPS)) == (1536, 2048, 8, np.float32(1e-6)) and X.shape == Y.shape == W.shape == (9, N)
    assert [r["label"]["kind"] for r in FIXTURE["rows"]] == ["pick"] * 5 + ["ctl"] * 4


def test_the_definition_gives_every_committed_row():
    d = bind(dsa_norm.TritonBlockRMSNorm, N=N, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)
    got = np.asarray(evaluate_batch(d, X.copy(), W.copy(), kernel="reference")).astype(np.uint16)
    assert [int((g != y).sum()) for g, y in zip(got, Y)] == [0] * len(Y)


def test_the_row_kernel_gives_every_committed_row():
    statics = RM.spec_statics(f"TritonBlockRMSNorm_v1{{N={N},BLOCK={BLOCK},WARPS={WARPS},EPS={{\"f64\":\"{EPS.hex()}\"}}}}")
    got = [rows.ROWS["TritonBlockRMSNorm_v1"].fn(statics, [x, w])["0"].astype(np.uint16) for x, w in zip(X, W)]
    assert [int((g != y).sum()) for g, y in zip(got, Y)] == [0] * len(Y)
    assert np.array_equal(block_rmsnorm_rows(X, W, EPS, BLOCK, WARPS)["out"].astype(np.uint16), Y)


def test_two_roundings_miss_the_rows_the_fixture_names():
    """Each rsqrt operand is the fixture's ulps from the two-rounding one (8 of the 9 rows one ulp off, the last control the same
    word), and the output words that moves are the fixture's."""
    two = block_rmsnorm_rows(X, W, EPS, BLOCK, WARPS, fused_eps=False)
    fused = block_rmsnorm_rows(X, W, EPS, BLOCK, WARPS)
    assert [int(v) for v in (two["out"].astype(np.uint16) != Y).sum(axis=1)] == [r["definition_off_words"] for r in FIXTURE["rows"]]
    e = np.full(len(Y), bits_of(np.float32(EPS)), dtype=np.uint32)
    gap = fused["mean"].astype(np.int64) - DR.add_rn(e, two["mean"]).astype(np.int64)
    assert gap.tolist() == [r["eps_sum_ulps"] for r in FIXTURE["rows"]]
