"""benchmarks/pouw/pearl_c_sm90/hm96_rows.py, one-stage's commitment of a Pearl+ call's A: its reference is the library's
hm96-sha512/row/v2 tree under one-stage's window domain over rowleaf's BF16 rows, the hm96 and frame-v3 gates evaluated give
its leaf and open it at its root, its level launches fold every width to that root, its report's shares, and its session
(hm96.sh) is a lease of its own that no h200.sh session runs."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates import frame_v3 as F
from verity.ml.commitments.gates import hm96 as H
from verity.ml.commitments.gates.sha512 import words_be
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import bit_row, row_leaf, sha512_row_v2_digest
from verity.ml.commitments.window import window_domain
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer import native_jit

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm90"
spec = importlib.util.spec_from_file_location("pearl_c_sm90_hm96_rows", HERE / "hm96_rows.py")
R = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def _bits(d: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def _digests(bits, n: int) -> list[bytes]:
    w = np.asarray(bits).reshape(n, 8, 64)
    return [b"".join(sum(int(b) << j for j, b in enumerate(row)).to_bytes(8, "big") for row in d) for d in w]


@pytest.mark.parametrize("m,k", R.VECTORS)
def test_the_reference_is_the_librarys_tree_under_one_stages_window_domain(m, k):
    """The gates' references give the library's hiding rows (step_rows' leaf over rowleaf's sha512/row/v2 digest) in
    core's tree, under the domain window_domain names."""
    rows = R.rows_of(R.vector(m, k), m, k)
    d = R.domain(m)
    assert d.domain_id.hex() == window_domain(R.PORT["program"], R.PORT["partition"], R.PORT["port"], R.SCHEMA, m, R.PORT["run"])
    want = tree_levels(d, [row_leaf(d, i, step_rows.hiding_leaf(r, step_rows.salt(R.KS, i), 16 * k)[1], R.SCHEMA)
                           for i, r in enumerate(rows)])
    assert R.reference(rows, k) == want


def test_a_bf16_tensors_rows_are_rowleafs_bit_rows():
    """A row of a little-endian BF16 tensor's memory is `bit_row((elements, 16))`, the bytes `rows_of` cuts."""
    els = np.random.default_rng(7).integers(0, 1 << 16, size=(3, 40), dtype=np.uint16)
    rows = R.rows_of(els.astype("<u2").tobytes(), 3, 40)
    assert rows == [bit_row((e.tolist(), 16))[0] for e in els]
    with pytest.raises(ValueError):
        R.rows_of(bytes(10), 3, 40)


def test_the_level_launches_fold_every_width_to_the_references_root():
    """At most `FUSED_LEVELS` levels a launch, each over a whole level, to the reference's root."""
    assert R.level_launches(1) == [] and R.level_launches(8) == [(8, 0, 3)]
    assert R.level_launches(8192) == [(8192, 0, 10), (8, 10, 3)] and R.level_launches(2048) == [(2048, 0, 10), (2, 10, 1)]
    for m, k in R.VECTORS:
        lv = R.reference(R.rows_of(R.vector(m, k), m, k), k)
        d, level = R.domain(m), list(lv[0])
        for n, at, f in R.level_launches(len(level)):
            assert len(level) == n and n % (1 << f) == 0
            for j in range(f):
                level = [F.node_ref(d.domain_id, at + j, i, level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
        assert level == [lv[-1][0]]


@pytest.mark.slow
def test_the_gates_evaluated_give_the_references_leaves_and_open_them_at_its_root():
    """The traced Hm96Sha512Leaf_v1 gives the reference's h of a decode vector's first and last rows, and the traced
    frame-v3-sha512 read at D = 3 opens each at its rank under the reference's root and refuses it at another: a few minutes
    of tracing, the only check that the arm's reference is the gates' circuits and not their Python references alone."""
    m, k = R.VECTORS[0]
    rows = R.rows_of(R.vector(m, k), m, k)
    lv, d, pos = R.reference(rows, k), R.domain(m), [0, m - 1]
    xy = [(sha512_row_v2_digest(rows[i], 16 * k), step_rows.salt(R.KS, i)) for i in pos]
    args = [H.arguments(x, y) for x, y in xy]
    got = evaluate_bits(bind(H.Hm96Sha512Leaf), *(np.array([a[j] for a in args], dtype=np.uint8) for j in range(2)))
    hs = [_digests(g, 3) for g in np.asarray(got).reshape(len(pos), -1)]
    assert [tuple(h) for h in hs] == [H.reference(x, y) for x, y in xy]
    reads = []
    for i, h in zip(pos, hs):
        path = [lv[t][(i >> t) ^ 1] for t in range(len(lv) - 1)]
        reads += [F.arguments(lv[-1][0], d.domain_id, h[2], i, path), F.arguments(lv[-1][0], d.domain_id, h[2], i ^ 1, path)]
    cols = [np.array([r[j] for r in reads], dtype=np.uint8) for j in range(len(reads[0]))]
    assert evaluate_bits(F.read(len(lv) - 1, R.SCHEMA), *cols).reshape(-1).tolist() == [1, 0, 1, 0]


def test_the_report_gives_each_points_share_of_the_pearl_call_and_of_the_forward():
    hm = {"rows": [{"name": "a", "n": 3072, "k": 2048, "m": 8, "us": {"split-graph": 50.0, "fused-graph": 80.0}},
                   {"name": "b", "n": 2048, "k": 8192, "m": 8, "us": {"split-graph": 150.0}}]}
    timing = {"rows": [{"n": 3072, "k": 2048, "m": 8, "ms": {"pearlc": 0.1, "pearlc-eager": 0.2, "deepgemm": 0.01}}]}
    whole = {"rows": [{"m": 8, "calls": 32, "ms": {"pearlc-served": 4.0}}, {"m": 16, "calls": 32, "ms": {"pearlc-served": 5.0}}]}
    r = R.report(hm, timing, whole)
    share = r["points"][0]["share"]
    assert share["split-graph"] == pytest.approx({"pearlc": 0.5, "pearlc-eager": 0.25})
    assert share["fused-graph"] == pytest.approx({"pearlc": 0.8, "pearlc-eager": 0.4}) and len(share) == 2
    assert r["points"][1]["pearlc_ms"] == {} and r["points"][1]["share"] == {"split-graph": {}}
    assert len(r["forwards"]) == 1 and r["forwards"][0]["added_ms"] == pytest.approx({"split-graph": 16 * 0.2})
    assert r["forwards"][0]["share"] == {"split-graph": {"pearlc-served": pytest.approx(0.8)}}


def test_the_session_is_a_lease_of_its_own_on_the_served_committers_kernels():
    """hm96.sh times inside its own gpu-lease from the library it was sent; no h200.sh session (the verdict's, the run of
    record's) runs it; the library is native_rows.cu alone under the served committer's CUDA flags, exporting what it calls."""
    sh = (HERE / "hm96.sh").read_text()
    lease = sh[sh.index("gpu-lease 1"):]
    assert '"$D/hm96_rows.py" time --lib "$LIB"' in lease and "LIB=$IN/hm96_rows.so" in sh
    assert "hm96" not in (HERE / "h200.sh").read_text()
    assert set(native_jit.CUDA_CFLAGS) <= set(R.NVCC) and R.SOURCE.name in native_jit.SOURCES
    src = R.SOURCE.read_text()
    assert all(re.search(rf'extern "C" \w+ {f}\(', src) for f in R.LAUNCHERS)
