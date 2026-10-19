"""`protocol_options.pouw_h1`: each served int8_exact linear call is a Program on bits that `pouw.circuit.compile` takes to
H1 tile units, every unit a full 32 x 32 grid at decode's one row, a partial block and an off-tile column count; a call is one
seed group, the calls sit at positions fixed by the public parameters (T rows, S steps), and the `pouw` option serves the
arm's own y beside sampled proofs.

The first compile builds the H1 tile unit (its seeds' and noise blocks' SHA-512 in gates), about 40 s once per process;
every later shape reuses it."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.ml.circuits.refs import runs  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import hooks  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402

K = 64
ROW = K * PH.I8_BITS
SHAPES = [(K, 32, 1), (K, 32, 5), (K, 40, 33)]
SALT = hashlib.sha512(b"salt").digest()


class Int8Arm:
    """A stand-in for `engine.int8_vllm.Int8PlainLinearMethod`: the plain arm's linear, counting its calls, handing its values
    to ``observe`` after both ops."""

    int8_arm = E.ARM

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        xq, s_a = E.quantize_rows(x)
        y = E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)
        self.observe(layer, xq, s_a, y)
        return y

    def observe(self, layer, xq, s_a, y):
        pass


class Linear:
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = Int8Arm()


class Model:
    def __init__(self, *layers):
        self.layers = list(layers)

    def modules(self):
        return iter(self.layers)


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _entry():
    return f"pouw:{PH.SCHEME}"


@pytest.mark.parametrize("k,n,m", SHAPES, ids=["decode-1", "partial-5", "rows-33-cols-40"])
def test_every_served_shape_compiles_to_full_warp_tiles(k, n, m):
    """Decode's one row, a partial block and a column count off the tile: every unit is a full 32 x 32 grid, the call one
    site and one seed group covering its units."""
    c = PH.compiled(k, n, m)
    mp, np_ = PH.padded(m, n)
    assert len(c.units) == PH.units(k, n, m) == (mp // 32) * (np_ // 32)
    assert {(u.tile.r, u.tile.c, u.tile.grid) for u in c.units} == {(32, 32, True)}
    assert [s.shape for s in c.sites] == [(mp, k, np_)] and c.groups == ((0,),) and c.cover == (len(c.units),)
    assert {u.defn.base_id for u in c.units} == {"PouwH1Tile_v1"}


def test_the_site_rows_are_the_tap_rows_then_distinct_pad_rows():
    """A's first m rows are the call's s8 rows (parameter x) in order and its B rows the weight's (parameter w); each pad row
    reads a constant of its own, so the 32 rows are distinct and the tile a grid."""
    (site,) = PH.compiled(K, 32, 5).sites
    assert [runs(r) for r in site.a_rows[:5]] == [[("p", 0, i * ROW, (i + 1) * ROW)] for i in range(5)]
    assert [runs(r) for r in site.b_rows] == [[("p", 2, j * ROW, (j + 1) * ROW)] for j in range(32)]
    pads = [tuple(runs(r)) for r in site.a_rows[5:]]
    assert len(set(pads)) == 27 and all(s == "n" for p in pads for s, _i, _lo, _hi in p)


def test_the_served_outputs_are_the_real_cells():
    """The Program serves cell (i, j) for i < m, j < n, each one bf16 of the coordinate batch; pad cells are not served."""
    fn = PH.program(K, 40, 33).fn
    np_ = PH.padded(33, 40)[1]
    (node,) = {i for s, i, _lo, _hi in runs(fn.body.ret)}
    assert fn.body.nodes[node].fn.base_id == "ScaledMmS8Coordinate_v2" and fn.body.nodes[node].n == 64 * np_
    want = [(i * np_ * 16, (i * np_ + 40) * 16) for i in range(33)]
    assert [(lo, hi) for _s, _i, lo, hi in runs(fn.body.ret)] == want


def _tap(rng, m, n):
    a = torch.from_numpy(rng.integers(-127, 128, (m, K), dtype=np.int8))
    w = torch.from_numpy(rng.integers(-127, 128, (n, K), dtype=np.int8))
    return a, torch.rand(m) + 0.5, w, torch.ones(n, 1)


def _run(steps=3, rows=32):
    return PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], rows, steps)


def test_calls_sit_at_public_positions_and_empty_steps_close_the_window():
    """Call (s, w) is at s U + off(w) whatever its served rows; the steps past the last served one are empty calls, so the
    window is every (step, linear) in order; given each call's (root(A), root(B)), every unit's seed is its own."""
    rng = np.random.default_rng(3)
    run = _run()
    assert (run.offs, run.u_step) == ([0, 1], 3) and run.commits["units"] == 9
    served = [run.record(w, *_tap(rng, m, n)) for w, n, m in ((0, 32, 1), (1, 40, 5), (0, 32, 32), (1, 40, 3))]
    run.close()
    assert [(c.step, c.layer, c.base, c.m, c.rows) for c in run.calls] == [
        (0, 0, 0, 1, 32), (0, 1, 1, 5, 32), (1, 0, 3, 32, 32), (1, 1, 4, 3, 32), (2, 0, 6, 0, 32), (2, 1, 7, 0, 32)]
    assert served == run.calls[:4] and run.static() and run.details["steps_served"] == 2 and run.details["last_step_complete"]
    roots = [(bytes([c.index + 1]) * 64, bytes([c.layer + 9]) * 64) for c in run.calls]
    seeds = [s for c, r in zip(run.calls, roots) for s in c.compiled().seeds(SALT, [r], c.base)]
    assert len(seeds) == len(set(seeds)) == 9
    salt, positions, leaves = run.calls[3].inputs(SALT, roots[3])
    assert salt == SD.leaves(SALT, True) and len(positions) == 2 * SD.POS_BITS.leaves
    assert leaves == SD.leaves(roots[3][0] + roots[3][1], True)
    assert set(run.check()) == {"32x64x32", "32x64x40"}


def test_a_call_past_the_window_or_its_rows_is_refused_and_the_window_is_not_static():
    rng = np.random.default_rng(6)
    run = _run(steps=1)
    assert run.record(0, *_tap(rng, 33, 32)) is None
    assert run.record(1, *_tap(rng, 2, 40)) is not None
    assert run.record(0, *_tap(rng, 1, 32)) is None
    run.close()
    assert run.refused == [{"step": 0, "linear": 0, "m": 33}, {"step": 1, "linear": 0, "m": 1}] and not run.static()
    out_of_order = _run(steps=1)
    out_of_order.record(1, *_tap(rng, 1, 40))
    out_of_order.close()
    assert not out_of_order.order_ok and not out_of_order.static()


def test_credit_is_the_served_rows_and_the_padding_is_shown_beside_it():
    """Calls recorded outside a traced apply are refused by the credit gate (test_credit_gate has the credited case): their
    served rows are shown as refused, and the padding beside them."""
    rng = np.random.default_rng(7)
    run = _run(steps=2)
    run.record(0, *_tap(rng, 3, 32))
    run.record(1, *_tap(rng, 2, 40))
    run.close()
    c = run.credit()
    served = (3 * 32 + 2 * 40) * K
    assert c["credited"] == {"rows": 0, "macs": 0}
    assert (c["refused"]["calls"], c["refused"]["rows"], c["refused"]["macs"]) == (2, 5, served)
    assert c["padded"]["rows"] == 4 * 32 - 5 and c["padded"]["macs"] == (2 * 32 * 32 + 2 * 32 * 64) * K - served
    assert c["units"] == 6 == run.commits["units"]


def test_slots_put_the_served_rows_first_then_empty_rows_under_scale_0():
    a, s_a, _w, _ws = _tap(np.random.default_rng(8), 3, 32)
    t, sc = PH.slots(a, s_a, 32)
    assert t.dtype == torch.int8 and t.shape == (32, K) and torch.equal(t[:3], a) and not t[3:].any()
    assert torch.equal(sc[:3], s_a) and not sc[3:].any()
    with pytest.raises(ValueError, match="past the call's 32 slots"):
        PH.slots(*_tap(np.random.default_rng(8), 33, 32)[:2], 32)


def test_listeners_get_the_served_tensors_and_bind_the_calls_roots():
    """A listener gets each call with the very tensors the arm served (slot i is AROW leaf i of the call's (step, layer)
    table), the empty steps' calls with no rows, and what it returns is in the manifest and the calls root; `bind_all`
    reads the roots back."""
    rng = np.random.default_rng(4)
    run = _run(steps=2)
    seen = []

    def listener(call, a, s_a, weight, weight_scale):
        seen.append((call, a, s_a, weight, weight_scale))
        return {"root_a": bytes([call.index]).hex() * 64, "root_b": "ab" * 64} if call.index else None

    run.listen(listener)
    taps = [_tap(rng, 5, 32), _tap(rng, 1, 40)]
    calls = [run.record(w, *t) for w, t in enumerate(taps)]
    run.close()
    assert [s[0] for s in seen] == run.calls and all(s[1] is t[0] and s[3] is t[2] for s, t in zip(seen, taps))
    assert [tuple(s[1].shape) for s in seen[2:]] == [(0, K), (0, K)] and seen[2][3] is taps[0][2]
    manifest = run.manifest()
    assert "bound" not in manifest[0] and manifest[1]["bound"]["root_b"] == "ab" * 64
    assert (manifest[0]["served"], manifest[0]["rows"], manifest[0]["leaves"]) == (5, 32, [0, 32])
    assert PH.bind_all(calls, run.bound) == [None, (b"\x01" * 64, b"\xab" * 64)]
    before = run.calls_root()
    run.bound[1] = {**run.bound[1], "root_b": "cd" * 64}
    assert run.calls_root() != before


def test_units_are_in_row_block_then_column_block_order():
    """Unit u of a call covers row block u // (N' / 32) and column block u % (N' / 32), so position base + u names a tile."""
    c = PH.compiled(K, 40, 33)
    blocks = [(u.rows[0] // 32, u.cols[0] // 32) for u in c.units]
    assert blocks == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(list(u.rows) == list(range(u.rows[0], u.rows[0] + 32)) for u in c.units)


def test_z_is_the_exact_product_on_each_units_cells():
    rng = np.random.default_rng(5)
    xq = rng.integers(-127, 128, (33, K), dtype=np.int8)
    w = rng.integers(-127, 128, (40, K), dtype=np.int8)
    y = xq.astype(np.int64) @ w.astype(np.int64).T
    call = PH.Call(0, 0, 0, "q", 33, K, 40, 0, 64)
    full = np.zeros((64, 64), np.int64)
    full[:33, :40] = y
    for n, u in enumerate(call.compiled().units):
        z = call.z(y.astype(np.int32), n)
        assert np.array_equal(z, [full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs])
        pad = [u.rows[i] >= 33 or u.cols[j] >= 40 for i, j in u.tile.pairs]
        assert not z[pad].any()


H1 = {"coins": None, "rows": 64, "steps": 3}


def _settings(**kw):
    return {"pouw": {**H1, "coins": Challenger(0), **kw}}


def test_h1_is_admitted_beside_sampled_proofs_and_executes_nowhere():
    assert PH.SCHEME in pouw.schemes() and PH.SCHEME in pouw.EXECUTED and pouw.executes(PH.SCHEME) == ()
    assert pouw.executes("ncp-v2") == pouw.EXECUTES
    commit = PO.Context(stage="commit", enabled=(_entry(), "sampled-proofs:vllm-v1"), settings=_settings())
    assert PO.refusals(commit) == ()
    eager = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(), enforce_eager=False)
    assert any("graph-capturable" in r for r in PO.refusals(eager))
    bare = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0)}})
    assert any("'rows' (T" in r for r in PO.refusals(bare))
    with pytest.raises(ValueError, match="positive multiple of 32"):
        pouw.Settings.of({"rows": 40})


def test_the_option_serves_the_arms_y_and_records_each_call(tmp_path):
    """Installed on int8_exact layers, each apply serves the arm's own y bit for bit and the option's observe records the call
    at its public position, its exact Y retained; closing the window adds the empty step, and the record compiles at T."""
    model = Model(Linear(32, 1), Linear(40, 2))
    ctx = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(transcript_dir=str(tmp_path), retain_steps=2))
    comp = PO.install(model, ctx)
    try:
        g = torch.Generator().manual_seed(9)
        for m, layer in ((1, model.layers[0]), (5, model.layers[1]), (33, model.layers[0]), (2, model.layers[1])):
            x = (torch.randn(m, K, generator=g) * 3).bfloat16()
            y = layer.quant_method.apply(layer, x)
            assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        run = pouw.h1_run(_entry())
        assert [(c.step, c.layer, c.m, c.base) for c in run.calls] == [(0, 0, 1, 0), (0, 1, 5, 2), (1, 0, 33, 6), (1, 1, 2, 8)]
        assert [layer.quant_method.calls for layer in model.layers] == [2, 2]
        index, xq, _sa, z = run.retained[-1]
        assert index == 3 and len(run.retained) == 2
        assert np.array_equal(z, xq.astype(np.int64) @ model.layers[1].weight.numpy().astype(np.int64).T)
        assert set(run.check()) == {"64x64x32", "64x64x40"}
    finally:
        comp.close()
    assert hooks.live(_entry()) == ()
    commitment = comp.installed[0].commitment()
    assert commitment["calls"] == 6 and commitment["units"] == 18 and commitment["static"]
    assert [c["base"] for c in json.loads((tmp_path / PH.CALLS_FILE).read_text())["calls"]] == [0, 2, 6, 8, 12, 14]


def test_a_layer_off_the_int8_arm_is_refused_at_install():
    layer = Linear(32, 1)
    layer.quant_method = type("Other", (), {"apply": lambda self, layer, x, bias=None: x})()
    with pytest.raises(ValueError, match="not the int8_exact plain arm"):
        PO.install(Model(layer), PO.Context(stage="serve", enabled=(_entry(),), settings=_settings()))
    assert hooks.live(_entry()) == ()
