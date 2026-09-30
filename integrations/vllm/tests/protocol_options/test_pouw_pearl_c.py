"""Pearl-C's CPU executor half (`protocol_options.pouw_pearl_c`): the twin against `pearl_c`'s scalar reference, the channel
split's calibration, and the run end to end (numpy in, no torch or GPU)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from verity_pouw.protocol import Context, Matmul, Shape, Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_work as PW

from verity_vllm.protocol_options import pouw_pearl_c as PPC


def scheme() -> C.PearlC:
    return C.PearlC(k_min=32, tile=(64, 64), forming="v1")


def rows(rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    """Activation-like FP32 rows: signed bytes scaled by 2^(−4 − b mod 5), 16 random low mantissa bits when nonzero."""
    code = rng.integers(-127, 128, (m, k)).astype(np.float32)
    x = code * np.exp2(-4.0 - rng.integers(0, 5, (m, k))).astype(np.float32)
    w = x.view(np.uint32) | np.where(code != 0, rng.integers(0, 1 << 16, (m, k)), 0).astype(np.uint32)
    return w.view(np.float32)


def ctx_of(shape: Shape, index: int = 3) -> Context:
    h = lambda s: hashlib.sha256(s.encode()).digest()  # noqa: E731
    return Context(h("salt/twin"), Matmul(index, shape, 7), h("root_a/twin"), h("root_b/twin"))


def test_twin_is_the_reference():
    """One 128 x 128 x 128 unit: the formed codes, the peel factors, C̃, U, every tile's leaf and tickets."""
    s, rng = scheme(), np.random.default_rng(2026093001)
    a, b = rows(rng, 128, 128), rows(rng, 128, 128)
    shape = Shape(128, 128, 128)
    ctx = ctx_of(shape)
    noise = s.noise(ctx)
    unit, _ = PPC.twin_call(s, ctx, noise, a.view(np.uint32), b.view(np.uint32))
    ops = s.encode(ctx, noise, [r.tobytes() for r in a], [r.tobytes() for r in b], Tile(tuple(range(128)), tuple(range(128))))
    assert ops is not None
    assert unit.a.codes.tolist() == ops.a.codes and unit.b.codes.tolist() == ops.b.codes
    assert unit.a.alpha.tolist() == ops.a.alpha and unit.b.ep.tolist() == ops.b.ep
    assert unit.pa.tolist() == ops.pa and unit.pb.tolist() == ops.pb
    checked = s.checked(ctx, ops)
    assert unit.c.tolist() == checked.c and unit.u.tolist() == checked.u
    for t in s.tiles(shape):
        r0, c0 = t.rows[0], t.cols[0]
        sub = C.Checked([r[c0:c0 + 64] for r in checked.c[r0:r0 + 64]], [r[c0:c0 + 64] for r in checked.u[r0:r0 + 64]])
        assert unit.leaves[(r0, c0)] == s.digest(ctx, noise, sub)
        assert unit.tickets[(r0, c0)] == s.tickets(noise, sub)
    useful = s.useful(ctx, noise, ops, checked)
    y = PPC._f32(unit.y_unit)
    assert np.allclose(y, np.array(useful, np.float32), rtol=1e-6)


def test_twin_computes_a_subset_of_tiles():
    s, rng = scheme(), np.random.default_rng(7)
    a, b = rows(rng, 128, 128), rows(rng, 128, 128)
    ctx = ctx_of(Shape(128, 128, 128))
    noise = s.noise(ctx)
    full, ws = PPC.twin_call(s, ctx, noise, a.view(np.uint32), b.view(np.uint32))
    part, _ = PPC.twin_call(s, ctx, noise, a.view(np.uint32), b.view(np.uint32), ws, tiles=[(64, 0)])
    assert part.leaves == {(64, 0): full.leaves[(64, 0)]}
    assert (part.u[64:, :64] == full.u[64:, :64]).all() and not part.u[:64].any()


def test_split_moves_massive_channels_and_admits_every_row():
    rng = np.random.default_rng(11)
    x, w = rows(rng, 64, 1030), rows(rng, 32, 1030)
    x[:, [5, 77]] *= 4096.0
    unsplit = PPC.Split(1030)
    assert not all(PPC.row_ok([int(v) for v in r]) for r in unsplit.unit_words(x.view(np.uint32)))
    split = PPC.split_layer(x, w)
    assert set(split.side) >= {5, 77} and len(split.side) <= 1030 // 100 and split.k_unit == 1152
    for vals in (x, w):
        assert all(PPC.row_ok([int(v) for v in r]) for r in split.unit_words(vals.view(np.uint32)))
    assert split.calibration == PPC.calibration_digest(x) != PPC.calibration_digest(x[:63])
    assert split.digest() == PPC.Split(1030, split.side, split.calibration).digest() != PPC.Split(1030, (), split.calibration).digest()


def test_split_budget_keeps_s_within_one_percent():
    assert [PPC.split_budget(k) for k in (299, 700, 1024, 3584, 18944)] == [0, 4, 7, 32, 186]
    rng = np.random.default_rng(12)
    x, w = rows(rng, 64, 1024), rows(rng, 16, 1024)
    x[:, 1::40] *= 4096.0
    side, trace = PPC.split_channels(x, w, 1.0, PPC.split_budget(1024))
    assert len(side) == 8 and trace[-1]["failing_rows"] > 0
    PPC.Split(1024, tuple(sorted(side)))


FIXTURES = Path(__file__).parent / "fixtures"


def layer2() -> tuple[dict, np.ndarray]:
    meta = json.loads((FIXTURES / "qwen7b_layer2_down_split.json").read_text())
    raw = gzip.decompress((FIXTURES / "qwen7b_layer2_down_split_rows.f32.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta["rows_file_sha256"]
    return meta, np.frombuffer(raw, "<f4").reshape(len(meta["rows"]), meta["x_shape"][1])


def test_split_rule_is_pinned_on_qwen7b_layer2():
    """The statement review's pin: the one split rule gives Qwen2.5-7B's layer-2 `down_proj` 38 channels, the cheap-binding
    lane's set and trace.  The fixture holds the 63 activation rows that fail at some step on the full data (no weight row
    ever does), so the rule on them alone takes the same steps."""
    meta, x = layer2()
    k = meta["x_shape"][1]
    side, trace = PPC.split_channels(x, np.zeros((0, k), np.float32), 1.0, PPC.split_budget(k))
    assert len(side) == 38 and sorted(side) == meta["S"] and trace == meta["trace"] and PPC.split_budget(k) == meta["budget"]
    PPC.Split(k, tuple(sorted(side)))


@pytest.mark.skipif(not (Path(os.environ.get("QWEN7B_DIR", "/nonexistent")) / "layer2_down_x.npy").exists(),
                    reason="QWEN7B_DIR holds the full layer-2 data (fetch_layer_down.py)")
def test_split_rule_on_the_full_layer2_data():
    meta, x = layer2()
    d = Path(os.environ["QWEN7B_DIR"])
    X, W = np.load(d / "layer2_down_x.npy"), np.load(d / "layer2_down_w.npy")
    assert hashlib.sha256(np.ascontiguousarray(X, "<f4").tobytes()).hexdigest() == meta["x_sha256"]
    assert (X[meta["rows"]] == x).all()
    split = PPC.split_layer(X, W)
    assert list(split.side) == meta["S"] and PPC.split_check({2: X}, {2: W})(2, split)


def test_split_check_recomputes_s():
    rng = np.random.default_rng(14)
    x, w = rows(rng, 64, 1030), rows(rng, 32, 1030)
    x[:, [5, 77]] *= 4096.0
    honest = PPC.split_layer(x, w)
    check = PPC.split_check({4: x}, {4: w})
    assert check(4, honest)
    assert not check(4, PPC.Split(1030, tuple(sorted(set(honest.side) ^ {900})), honest.calibration))
    assert not check(4, PPC.Split(1030, honest.side, PPC.calibration_digest(x[:63])))


def test_filler_rows_are_zero_and_excluded_from_credit():
    """`pearl_c_work`'s rule: filler rows are zero words (a zero row fails the floor, and is not credited)."""
    assert not PPC.row_ok([0] * 1024)


def test_run_serves_the_useful_output():
    """A 100-row call on a 1,030-wide layer with two side channels: m padded to 128 with zero filler rows, the unit at
    k = 1,152, y = RN_bf16(y_unit + y_side + bias) within FP8 accuracy of x·Wᵀ + bias, one leaf per tile, the records."""
    s, rng = scheme(), np.random.default_rng(13)
    x, w = rows(rng, 100, 1030), rows(rng, 128, 1030)
    bias = rng.standard_normal(128).astype(np.float32)
    splits = {0: PPC.Split(1030, (5, 77))}
    committed = PPC.commit_weights({0: w}, splits)
    roots = {0: bytes.fromhex(committed["weight_roots"]["0"])}
    run = PPC.PearlCRun(s, "pouw:pearl-c-h100-v1", roots, hashlib.sha256(b"salt/run").digest(), splits)
    y = run.gemm(0, x, w, bias)
    exact = x.astype(np.float64) @ w.T.astype(np.float64) + bias
    rel = np.sqrt(((y - exact) ** 2).sum() / (exact ** 2).sum())
    assert y.shape == (100, 128) and rel < 0.08, rel
    assert run.details["padded_rows"] == 28 and len(run.leaves) == 4 and run.units[0].shape == Shape(128, 1152, 128)
    assert run.records == [PW.UnitRecord(run.units[0], 100, 1028)]
    run.finish()
    assert run.commits["transcript_root"] and run.commits["records_root"] and run.commits["tiles"] == 4
    assert run.commits["splits"]["0"] == splits[0].digest().hex()
