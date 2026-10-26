"""Per-call-noise PoUW over a step's static linears (`verity_torch_loop.percall`): the units are the v6 tile's function
under their call's layout, their z is each linear's accumulator, a tampered window's are not, and PoUW's check takes the
written registration once the tile is pinned and refuses it while it isn't."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.core.protocols.pouw import core_check as CC, window as Wd
from verity.core.protocols.pouw.circuit import h1tile as Ht
from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity.ml.windows.interface import evaluate_batch
from verity_torch_loop import percall as P

SALT = bytes(range(64))
TINY = P.Shape("tiny", 64, 1, 1, 64, 64, 8)


class Recorder:
    def __init__(self) -> None:
        self.units: list[tuple[list[np.ndarray], list[np.ndarray], dict]] = []

    def record(self, template, inputs, outputs, label=None):
        assert template.id == P.tile(64).id
        for i in range(len(inputs[0])):
            self.units.append(([x[i] for x in inputs], [o[i] for o in outputs], label))


def recorded(tamper: bool = False, rows: int = 4) -> tuple[P.PerCall, Recorder]:
    pc = P.PerCall(P.stand_in(TINY, rows, b"t"), SALT, "00" * 64, 64, tamper=tamper)
    rec = Recorder()
    pc.run_into(rec)
    return pc, rec


def u64(halves) -> int:
    return (int(halves[0]) << 32) | int(halves[1])


def reference(inputs) -> tuple[np.ndarray, np.ndarray]:
    a, b, _salt, pos, roots, call, blocks = inputs
    rs = tuple(b"".join(int(w).to_bytes(4, "big") for w in r) for r in roots)
    z, rc = Ht.j10r_reference(a.tolist(), b.tolist(), SALT, u64(pos), rs, u64(call), (u64(blocks[0]), u64(blocks[1])),
                              stream=True)
    return np.asarray(z, np.int64).astype(np.uint32), np.asarray(rc, np.int64).astype(np.uint32)


def bits(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    return ((x.astype(np.uint64)[..., None] >> np.arange(x.dtype.itemsize * 8, dtype=np.uint64)) & 1).astype(np.uint8)


def test_the_h200_block_covers_every_static_linear_at_k_unit_8192():
    s = P.plan(P.H200, 4, batch=16384)
    per_layer = 2 * (2048 + 8 + 8 + 2048 + 8192) + 8 * 2048
    assert s["real"] == P.H200.layers * per_layer + 2 * 32 == 2_131_648
    assert s["calls"] == P.H200.layers * (2 + 2 + 2 + 2 + 2 + 8) + 2
    assert P.H200.macs_per_row() == 139_699_683_328 and s["W"] == s["credited"] * 32 * 8192
    assert s["k_prime"] == {"1/100": 8, "1/1000000": 24} and 2.3e-4 < s["share_of_step"] < 2.5e-4


def test_the_units_are_the_v6_tile_under_their_calls_layout():
    pc, rec = recorded()
    assert pc.real == 6 * 8 + 1 and len(rec.units) == pc.work.n
    for c in pc.calls:
        for t in range(c.units):
            inputs, outputs, label = rec.units[c.base + t]
            assert label["linear"] == c.linear.name and u64(inputs[3]) == c.base + t and u64(inputs[5]) == c.base
            i, j = divmod(t, c.linear.n // P.TILE_C)
            assert (u64(inputs[6][0]), u64(inputs[6][1])) == Ht.j10r_blocks(4, 64, (4, 8), i, j)
            z, rc = reference(inputs)
            assert np.array_equal(outputs[0], z) and np.array_equal(outputs[1], rc)
    fill = [u for u, (_i, _o, label) in enumerate(rec.units) if "filler" in label]
    assert fill == list(range(pc.real, pc.work.n)) and all(u64(rec.units[u][0][5]) == u for u in fill)


def test_a_k_split_linear_sums_its_calls_to_its_accumulator():
    blk = P.stand_in(P.Shape("split", 64, 1, 1, 64, 128, 8), 8, b"s")
    pc = P.PerCall(blk, SALT, "00" * 64, 64)
    assert [c.chunk for c in pc.calls if c.linear.name == "l0.wdown"] == [0, 1]
    assert all(pc.run_into(Recorder()).values())
    bad = P.Block(blk.shape, 8, blk.a, blk.b, blk.root_a, blk.root_b, lambda lin: blk.acc(lin) + 1)
    with pytest.raises(AssertionError, match="accumulator"):
        P.PerCall(bad, SALT, "00" * 64, 64).run_into(Recorder())


def test_a_tampered_windows_units_are_not_the_tiles():
    _, rec = recorded(tamper=True)
    for inputs, outputs, label in rec.units[::5]:
        z, _ = reference(inputs)
        assert np.array_equal(outputs[0] ^ z, np.full_like(z, 0 if "filler" in label else 1))


def test_a_step_from_the_drivers_tables_reads_its_first_rows():
    blk = P.stand_in(TINY, 4, b"d")
    lins = TINY.linears()
    tables = {lin.table: (np.vstack([blk.a(lin), np.ones((4, lin.k), np.int8)]).view(np.uint8), None) for lin in lins}
    step = P.from_tables(TINY, 4, tables, lambda n: blk.b(next(x for x in lins if x.name == n)),
                         {lin.table: bytes(64) for lin in lins}, lambda n: bytes(64))
    for lin in lins:
        assert np.array_equal(step.a(lin), blk.a(lin)) and np.array_equal(step.b(lin), blk.b(lin))
    assert not any(P.PerCall(step, SALT, "00" * 64, 64).run_into(Recorder()).values())


@pytest.mark.slow
def test_a_unit_is_its_definitions_outputs_on_its_recorded_words():
    """One recorded unit through the v6 Definition on bits (about 110 s): the only test that holds the recorded word layout
    of ``call`` and ``blocks`` to the Definition's own ports, which the proof of a drawn unit reads."""
    _, rec = recorded()
    inputs, outputs, _ = rec.units[11]
    got = np.concatenate([bits(o[None]).reshape(1, -1) for o in outputs], axis=1)
    want = np.asarray(evaluate_batch(P.tile(64), *(bits(x[None]) for x in inputs), kernel="reference")).reshape(1, -1)
    assert np.array_equal(got.astype(np.uint64), want.astype(np.uint64))


@pytest.mark.slow
def test_core_refuses_the_unpinned_tile_and_takes_the_window_once_pinned(tmp_path, monkeypatch):
    """Writes the tiny window (about a minute: the population's port check and the commitment)."""
    pc = P.PerCall(P.stand_in(TINY, 4, b"w"), SALT, "11" * 64, 64)
    win, _ = pc.window()
    rec = pc.write(win, tmp_path, {"block": "test"}, salt_key=bytes(32))
    p = json.loads((tmp_path / "params.json").read_text())
    assert p == pc.params(rec["query"]) and rec["window"]["context"] == "11" * 64
    draw = {"k": pc.k, "law": "subset", "population": pc.work.n, "units": list(range(pc.k))}
    profile = A.profile(rec, draw, float(pc.delta)).to_json() | {"delta": str(pc.delta)}
    ans = {"commitment": R.digest(rec), "profile": profile, "protocol": R.PROTOCOL, "public": {"draw": draw},
           "reason": None, "verdict": "accept"}
    line = [CC.line(x).encode() for x in (p, rec, ans)]
    got = CC.answer(*line)
    assert got["verdict"] == "reject" and pc.template().id not in CC.TILES
    monkeypatch.setitem(CC.TILES, pc.template().id, CC.Tile(pc.tile_work, rec["query"], "00" * 64, CC.Salted(4, 8)))
    got = CC.answer(*line)
    assert got["verdict"] == "accept", got["reason"]
    assert got["public"]["work"]["credited"] == Wd.bucket(pc.real).credited * pc.tile_work
