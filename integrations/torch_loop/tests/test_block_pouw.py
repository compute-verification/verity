"""The H200 block's PoUW window (`verity_torch_loop.block_pouw`) on small cells: each cell's tiles are the pinned tile's
function and sum to the cell's accumulator, a capture off its accumulator or off the tile is refused, tampering flips
every real tile's z, and PoUW's window check takes the written registration and accepts a service answer for it."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.core.protocols.pouw.circuit import h1tile as Ht
from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity_torch_loop import block_pouw as BP, pouw as PW, pouw_audit as PA

SALT = bytes(range(64))
CELLS = (BP.Cell("l0.wq", 16, 192), BP.Cell("l0.wdown", 8, 128), BP.Cell("head", 16, 64, rb=1, cb=1))


def capture(cells=CELLS, seed: int = 0) -> BP.Capture:
    g = np.random.default_rng(seed)
    a, b, acc = {}, {}, {}
    for c in cells:
        a[c.key] = g.integers(-128, 128, (PW.TILE_R, c.k), dtype=np.int8)
        b[c.key] = g.integers(-128, 128, (PW.TILE_C, c.k), dtype=np.int8)
        acc[c.key] = (a[c.key].astype(np.int64) @ b[c.key].astype(np.int64).T).astype(np.int32)
    a[cells[0].key][0, :] = -128
    b[cells[0].key][0, :] = -128
    acc[cells[0].key] = (a[cells[0].key].astype(np.int64) @ b[cells[0].key].astype(np.int64).T).astype(np.int32)
    roots = {c.tensor: bytes([i + 1]) * 64 for i, c in enumerate(cells)}
    return BP.Capture(list(cells), a, b, acc, (b"\x07" * 64).hex(), {t: r.hex() for t, r in roots.items()}, {"test": 1})


class Recorder:
    """A window's `record`, kept: one entry per tile."""

    def __init__(self) -> None:
        self.units: list[tuple[list[np.ndarray], list[np.ndarray], dict]] = []

    def record(self, template, inputs, outputs, label=None):
        assert template.id in CC.TILES
        for i in range(len(outputs[0])):
            self.units.append(([x[i] for x in inputs], [y[i] for y in outputs], label))


def reference(inputs) -> tuple[np.ndarray, np.ndarray]:
    a, b, salt, pos, roots = (np.asarray(x) for x in inputs)
    u = (int(pos[0]) << 32) | int(pos[1])
    def words(ws) -> bytes:
        return b"".join(int(w).to_bytes(4, "big") for w in ws)
    z, rc = Ht.reference(a.tolist(), b.tolist(), words(salt), u, (words(roots[0]), words(roots[1])))
    return np.asarray(z, np.int64).astype(np.uint32), np.asarray(rc, np.int64).astype(np.uint32)


def recorded(cap: BP.Capture, tamper: bool = False) -> tuple[BP.BlockH1, Recorder, dict]:
    h1 = BP.BlockH1(cap, SALT, cap.root_a, tamper=tamper)
    rec = Recorder()
    held = h1.run_into(rec)
    return h1, rec, held


def test_the_cells_tiles_are_the_pinned_tile_and_carry_the_accumulators():
    cap = capture()
    h1, rec, held = recorded(cap)
    assert h1.real == 3 + 2 + 1 and h1.work == Wd.bucket(6) and len(rec.units) == h1.work.n
    assert h1.slots == 2 * PW.TILE_R and held == {c.key: True for c in CELLS}
    assert [u for u, (_i, _o, label) in enumerate(rec.units) if "filler" in label] == list(range(6, h1.work.n))
    for u, (inputs, outputs, _) in enumerate(rec.units):
        assert (int(inputs[3][0]) << 32 | int(inputs[3][1])) == u
        z, rc = reference(inputs)
        assert np.array_equal(outputs[0], z) and np.array_equal(outputs[1], rc)
    # a cell's tiles are its K in order: tile 1 of l0.wq is codes [64, 128) of its rows
    assert np.array_equal(rec.units[1][0][0].view(np.int8), cap.a["l0.wq/r0/c0"][:, 64:128])
    assert np.array_equal(rec.units[1][0][1].view(np.int8), cap.b["l0.wq/r0/c0"][:, 64:128])


def test_a_tampered_window_flips_every_real_tiles_z():
    _, rec, _ = recorded(capture(), tamper=True)
    for inputs, outputs, label in rec.units:
        z, _ = reference(inputs)
        assert np.array_equal(outputs[0] ^ z, np.full_like(z, 0 if "filler" in label else 1))


def test_a_capture_off_its_accumulator_is_refused():
    cap = capture()
    cap.acc["l0.wdown/r0/c0"] = cap.acc["l0.wdown/r0/c0"] + np.int32(1)
    with pytest.raises(AssertionError, match="l0.wdown/r0/c0"):
        recorded(cap)


def test_cells_off_the_tile_and_windows_past_the_range_are_refused():
    with pytest.raises(ValueError, match="whole tiles"):
        BP.Cell("x", 16, 100)
    with pytest.raises(ValueError, match="whole tiles"):
        BP.Cell("x", 12, 64)
    with pytest.raises(ValueError, match="past the declared range"):
        BP.BlockH1(capture(), SALT, "00", range_bits=2)


def test_the_default_cells_are_one_of_each_linear_a_layer_and_the_heads():
    cs = BP.cells()
    assert len(cs) == 6 * BP.LAYERS + 1
    assert sum(c.tiles for c in cs) == BP.LAYERS * (5 * 256 + 1024) + 256 == 120_064
    w = Wd.bucket(120_064)
    assert w.n <= 1 << BP.RANGE_BITS and w.n > 1 << 16


def test_take_slices_the_cell_from_torch_tensors_and_a_capture_round_trips(tmp_path):
    c = BP.Cell("l3.wo", 16, 128, rb=1, cb=1)
    x = torch.randint(-128, 128, (8, 128), dtype=torch.int8)
    w = torch.randint(-128, 128, (16, 128), dtype=torch.int8)
    acc = x.to(torch.int32) @ w.to(torch.int32).T
    a, b, s = BP.take(c, x, w, acc)
    assert np.array_equal(a, x[4:8].numpy()) and np.array_equal(b, w[8:16].numpy())
    assert np.array_equal(s, a.astype(np.int64) @ b.astype(np.int64).T)
    with pytest.raises(ValueError, match="input rows"):
        BP.take(BP.Cell("l3.wo", 16, 64), x, w)
    cap = capture()
    cap.acc["head/r1/c1"] = None
    cap.save(tmp_path / "c.npz")
    back = BP.Capture.load(tmp_path / "c.npz")
    assert back.cells == cap.cells and back.root_b == cap.root_b and back.meta == cap.meta and back.acc["head/r1/c1"] is None
    assert all(np.array_equal(back.a[k], cap.a[k]) and np.array_equal(back.b[k], cap.b[k]) for k in cap.a)


@pytest.mark.slow
def test_core_registers_the_written_block_window_and_accepts_a_service_answer_for_it(tmp_path):
    """Builds and writes a small block window (about a minute: the tile's lowering and the commitment), then PoUW's
    window check at the written parameters, which the audit's `--params` reads."""
    cap = capture()
    h1 = BP.BlockH1(cap, SALT, cap.root_a)
    win, _ = h1.window()
    rec = h1.write(win, tmp_path, {"block": "test"}, salt_key=bytes(32))
    mine, _doc, _sets = PA.verifier_window(tmp_path, rec["law"], None)
    assert mine.partition_digest == rec["partition"] and mine.population == h1.work.n
    p = json.loads((tmp_path / "params.json").read_text())
    assert p == h1.params() and p["range"] == [1, 1 << BP.RANGE_BITS]
    assert CC.registered(CC.params(p), CC.commitment(CC.line(rec).encode())) is None
    draw = {"k": h1.k, "law": "subset", "population": h1.work.n, "units": list(range(h1.k))}
    profile = A.profile(rec, draw, float(BP.DELTA)).to_json() | {"delta": str(BP.DELTA)}
    ans = {"commitment": R.digest(rec), "profile": profile, "protocol": R.PROTOCOL, "public": {"draw": draw},
           "reason": None, "verdict": "accept"}
    got = PA.decide(p, rec, ans, None)
    assert got["verdict"] == "accept", got["reason"]
    assert got["W"] == h1.summary()["W"] and got["k"] == h1.k
    assert got["eps_at"][str(BP.DELTA)] <= float(BP.EPS) and got["eps_at"]["1/100"] <= got["eps_at"][str(BP.DELTA)]
    assert PA.decide({**p, "range": list(PW.RANGE[:1]) + [4]}, rec, ans, None)["verdict"] == "reject"
