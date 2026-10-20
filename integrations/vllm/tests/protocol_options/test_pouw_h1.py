"""`protocol_options.pouw_h1`: each served int8_exact linear call is a Program on bits that `pouw.circuit.compile` takes to
H1 tile units, every unit a full 32 x 32 grid at decode's one row, a partial block and an off-tile column count; a call is one
seed group, the calls sit at positions fixed by the public parameters (T rows, S steps), and the `pouw` option serves the
arm's own y beside sampled proofs.

The first compile builds the H1 tile unit (its seeds' and noise blocks' SHA-512 in gates), about 40 s once per process;
every later shape reuses it."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveRecord  # noqa: E402
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


def test_fill_declares_every_served_step_full_and_serve_h1_is_119_of_1152():
    """`FILL` (VLLM-52): 64 one-token prompts at cap 16 in T = 64, S = 18 prefill at step 0 and decode together, so each of
    the 16 served steps carries 64 rows and u = 16/18, PERIOD-PATH's ask 1; the engine runs all 64 in flight with no prefix
    cache. SERVE-H1's 5 prompts (`tests/engine/int8_exact_gpu.PROMPTS`, 44 tokens with BOS under the int8 Llama-3.1-8B's
    tokenizer, r20261020-011113-6876) at the same cap are 44 + 15 x 5 rows: today's 119/1152."""
    assert PH.FILL.schedule() == [64] * 16 + [0, 0] and PH.FILL.u == Fraction(16, 18)
    assert PH.FILL.engine_args() == {"max_num_seqs": 64, "max_num_batched_tokens": 64, "enable_chunked_prefill": True,
                                     "enable_prefix_caching": False}
    serve_h1 = PH.Static((6, 14, 6, 7, 11), 16, 64, 18)
    assert serve_h1.schedule()[:2] == [44, 5] and serve_h1.u == Fraction(119, 1152)
    with pytest.raises(ValueError, match="don't prefill in one step of 64 rows"):
        PH.Static((2,) * 64, 16, 64, 18)
    with pytest.raises(ValueError, match="doesn't fit a window of 18 steps"):
        PH.Static((1,) * 64, 19, 64, 18)


def _fill_run(rows_per_step):
    """`FILL`'s window over two small linears, step s serving rows_per_step[s] rows."""
    rng = np.random.default_rng(11)
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], PH.FILL.rows, PH.FILL.steps)
    for m in rows_per_step:
        for w, n in ((0, 32), (1, 40)):
            assert run.record(w, *_tap(rng, m, n)) is not None
    run.close()
    return run


def test_a_filled_window_measures_its_declared_u():
    run = _fill_run(PH.FILL.schedule()[:PH.FILL.max_tokens])
    assert run.step_rows() == PH.FILL.schedule() and run.utilization() == PH.FILL.u == Fraction(16, 18)
    assert run.static() and PH.FILL.check(run.step_rows()) == [] and run.details["u"] == "8/9"


def test_a_step_short_of_its_declared_rows_is_padded_in_the_layout_and_refused_by_the_declaration():
    """One sequence missing from step 5 (a preemption, say) and the last served step gone (an early stop): the layout pads
    them (every call keeps its public position, the empty slots and step are padding, u falls), and `Static.check` names
    both steps."""
    short = [64] * 5 + [63] + [64] * 9
    run, full = _fill_run(short), _fill_run([64] * 16)
    assert run.static() and [(c.step, c.layer, c.base) for c in run.calls] == [(c.step, c.layer, c.base) for c in full.calls]
    assert run.step_rows() == short + [0, 0, 0] and run.utilization() == Fraction(16 * 64 - 65, 18 * 64)
    assert run.credit()["padded"]["rows"] == full.credit()["padded"]["rows"] + 2 * 65
    assert [p.split(":")[0] for p in PH.FILL.check(run.step_rows())] == ["step 5", "step 15"]
    assert "63 rows served, 64 declared (fewer" in PH.FILL.check(run.step_rows())[0]


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


class Runner:
    """A stand-in for vLLM's model runner: ``execute_model`` gets the step's scheduler output."""

    def __init__(self):
        self.steps = 0

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.steps += 1


def _so(*new, tokens=1):
    """A scheduler output admitting requests ``new``, each (prompt token ids, cap)."""
    reqs = [SimpleNamespace(req_id=str(i), prompt_token_ids=p, sampling_params=SimpleNamespace(max_tokens=c))
            for i, (p, c) in enumerate(new)]
    return SimpleNamespace(scheduled_new_reqs=reqs, total_num_scheduled_tokens=tokens)


def _rounds(n=3, seed=0):
    clock = itertools.count()
    return LiveRecord(Wd.live_challenger(0, n, random.Random(seed).randbytes), Wd.ROUND, clock=lambda: float(next(clock)))


def _settings(**kw):
    return {"pouw": {**H1, "coins": Challenger(0, random.Random(0).randbytes), "runner": Runner(), "rounds": _rounds(), **kw}}


def test_h1_is_admitted_beside_sampled_proofs_and_executes_nowhere():
    assert PH.SCHEME in pouw.schemes() and PH.SCHEME in pouw.EXECUTED and pouw.executes(PH.SCHEME) == ()
    assert pouw.executes("ncp-v2") == pouw.EXECUTES
    commit = PO.Context(stage="commit", enabled=(_entry(), "sampled-proofs:vllm-v1"), settings=_settings())
    assert PO.refusals(commit) == ()
    eager = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(), enforce_eager=False)
    assert any("graph-capturable" in r for r in PO.refusals(eager))
    bare = PO.Context(stage="serve", enabled=(_entry(),), settings={"pouw": {"coins": Challenger(0, random.Random(0).randbytes)}})
    assert any("'rows' (T" in r for r in PO.refusals(bare))
    with pytest.raises(ValueError, match="positive multiple of 32"):
        pouw.Settings.of({"rows": 40})
    unadmitted = {"pouw": {k: v for k, v in _settings()["pouw"].items() if k != "runner"}}
    assert any("the pouw settings 'runner'" in r for r in PO.refusals(PO.Context(stage="serve", enabled=(_entry(),),
                                                                                 settings=unadmitted)))
    capture = PO.Context(stage="capture", enabled=(_entry(),), settings=unadmitted)
    assert PO.refusals(capture) == ()
    short = PO.Context(stage="serve", enabled=(_entry(),), settings=_settings(rounds=_rounds(2)))
    assert any("2 unopened coins, fewer than the window's 3 rounds" in r for r in PO.refusals(short))
    other = PO.Context(stage="serve", enabled=("pouw:ncp-v1",), settings={"pouw": {"coins": Challenger(0), "runner": Runner()}})
    assert any("'runner' and 'rounds' are H1's" in r for r in PO.refusals(other))
    with pytest.raises(ValueError, match="'rounds' is the window's round coins"):
        pouw.Settings.of({"rounds": object()})


def test_the_option_serves_the_arms_y_and_records_each_call(tmp_path):
    """Installed on int8_exact layers, each apply serves the arm's own y bit for bit and the option's observe records the call
    at its public position, its exact Y retained; closing the window adds the empty step, and the record compiles at T."""
    model = Model(Linear(32, 1), Linear(40, 2))
    settings = _settings(transcript_dir=str(tmp_path), retain_steps=2)
    runner = settings["pouw"]["runner"]
    ctx = PO.Context(stage="serve", enabled=(_entry(),), settings=settings)
    comp = PO.install(model, ctx)
    try:
        g = torch.Generator().manual_seed(9)
        steps = [(_so(([5, 6], 4)), ((1, model.layers[0]), (5, model.layers[1]))),
                 (_so(([7], 2), ([8, 9, 10], 3)), ((33, model.layers[0]), (2, model.layers[1])))]
        for so, step in steps:
            runner.execute_model(so)
            for m, layer in step:
                x = (torch.randn(m, K, generator=g) * 3).bfloat16()
                y = layer.quant_method.apply(layer, x)
                assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        run = pouw.h1_run(_entry())
        assert runner.steps == 2 and [(e.index, e.round) for e in run.requests] == [(0, 0), (1, 1), (2, 1)]
        assert [(c.step, c.layer, c.m, c.base) for c in run.calls] == [(0, 0, 1, 0), (0, 1, 5, 2), (1, 0, 33, 6), (1, 1, 2, 8)]
        assert [layer.quant_method.calls for layer in model.layers] == [2, 2]
        index, xq, _sa, z = run.retained[-1]
        assert index == 3 and len(run.retained) == 2
        assert np.array_equal(z, xq.astype(np.int64) @ model.layers[1].weight.numpy().astype(np.int64).T)
        assert set(run.check()) == {"64x64x32", "64x64x40"}
    finally:
        comp.close()
    assert hooks.live(_entry()) == () and runner.execute_model.__func__ is Runner.execute_model
    commitment = comp.installed[0].commitment()
    assert commitment["calls"] == 6 and commitment["units"] == 18 and commitment["static"]
    assert [r["round"] for r in commitment["rounds"]] == [0, 1, 2]
    assert PH.rounds_reason(commitment["rounds"], settings["pouw"]["rounds"].challenger.commitment) is None
    assert all(PH.request_reason(commitment["rounds"], run.salt, 64, e, run.open_request(e.index)) is None for e in run.requests)
    record = json.loads((tmp_path / PH.CALLS_FILE).read_text())
    assert [c["base"] for c in record["calls"]] == [0, 2, 6, 8, 12, 14]
    assert record["commitment"]["rounds"] == commitment["rounds"] and record["details"]["requests"]["per_round"] == [1, 2, 0]


def test_a_layer_off_the_int8_arm_is_refused_at_install():
    layer = Linear(32, 1)
    layer.quant_method = type("Other", (), {"apply": lambda self, layer, x, bias=None: x})()
    with pytest.raises(ValueError, match="not the int8_exact plain arm"):
        PO.install(Model(layer), PO.Context(stage="serve", enabled=(_entry(),), settings=_settings()))
    assert hooks.live(_entry()) == ()


def _admitted_run():
    """A live window of 3 rounds: request 0 admitted in round 0, requests 1 and 2 in round 1, round 2 admitting none."""
    rng = np.random.default_rng(12)
    rounds = _rounds()
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, [(32, K), (40, K)], 32, 3, rounds=rounds, key=bytes(range(32)))
    for new, m in (([([1, 2, 3], 4)], 3), ([([4], 2), ([5, 6], 3)], 3)):
        run.admit(new)
        assert run.round_log == [] or run.round_log[-1]["round"] == run.step
        for w, n in ((0, 32), (1, 40)):
            run.record(w, *_tap(rng, m, n))
    run.close()
    return run, rounds


def test_each_admitted_request_is_a_leaf_under_its_rounds_root_recorded_before_its_coin():
    """Each request is admitted into the round of the step about to run; the round's root, over the step's calls and its T
    request leaves, is recorded before its coin opens, and each entry opens under its own round's root."""
    run, rounds = _admitted_run()
    assert [(e.index, e.round, e.max_tokens, e.prompt) for e in run.requests] == [(0, 0, 4, (1, 2, 3)), (1, 1, 2, (4,)),
                                                                                 (2, 1, 3, (5, 6))]
    assert [r["round"] for r in run.round_log] == [0, 1, 2] and run.details["requests"]["per_round"] == [1, 2, 0]
    assert [ev["event"] for ev in rounds.log] == ["root", "release"] * 3
    roots = [r["root"] for r in run.round_log]
    assert [ev["root"] for ev in rounds.log[::2]] == [ev["after"] for ev in rounds.log[1::2]] == roots
    for row, r in zip(run.round_log, range(3)):
        table = PH.request_table(SALT, r, 32, run.admitted.get(r, []), bytes(range(32)))
        calls = PH.step_digest(SALT, r, [m for m in run.manifest() if m["step"] == r])
        assert row["requests"] == table.commitment.root.hex() and row["calls"] == calls.hex()
        assert row["root"] == PH.round_root(SALT, r, calls, table.commitment.root).hex()
    for e in run.requests:
        assert PH.request_reason(run.round_log, SALT, 32, e, run.open_request(e.index)) is None
    assert PH.rounds_reason(run.round_log, rounds.challenger.commitment) is None


def test_a_request_entry_under_the_wrong_round_or_under_none_is_refused():
    """The check refuses, by name: round 0's entry opened under round 1's root, or with round 1's leaf; an entry the prover
    commits into a later round's table than the one that admitted it; an entry opened under a round the log doesn't have,
    or under no table at all; a round root that isn't over its parts; and a round whose coin opened before its root."""
    run, rounds = _admitted_run()
    log, e0, e1 = run.round_log, run.requests[0], run.requests[1]
    o0, o1 = run.open_request(0), run.open_request(1)

    def why(entry, opening, log=log):
        return PH.request_reason(log, SALT, 32, entry, opening)

    wrong = "request 0 was admitted in round 0, and its entry is opened under round 1's root"
    assert why(e0, dataclasses.replace(o0, round=1)) == wrong and why(e0, o1) == wrong
    assert why(e0, dataclasses.replace(o0, round=5)) == \
        "request 0: the round log has no round 5, so its entry is under no round root"
    assert why(e0, dataclasses.replace(o0, round=1), log=log[:1]) == \
        "request 0: the round log has no round 1, so its entry is under no round root"
    assert why(e0, dataclasses.replace(o0, value=o1.value)) == "request 0: the leaf opened under round 0 is not its entry"
    late = dataclasses.replace(e0, round=1)
    table = PH.request_table(SALT, 1, 32, [late, *run.admitted[1]], bytes(range(32)))
    moved = [log[0], {**log[1], "requests": table.commitment.root.hex(),
                      "root": PH.round_root(SALT, 1, bytes.fromhex(log[1]["calls"]), table.commitment.root).hex()}, log[2]]
    o = table.open(0)
    assert why(late, PH.RequestOpening(1, o.position, o.value, o.path), log=moved) == \
        "round 1's coin was not opened after its root, which holds request 0's entry"
    assert why(e0, PH.RequestOpening(1, o.position, o.value, o.path), log=moved) == wrong
    forged = PH.RequestEntry(3, 2, 1, (9,))
    stray = PH.request_table(SALT, 2, 32, [forged], b"\1" * 32).open(0)
    assert why(forged, PH.RequestOpening(2, 0, stray.value, stray.path)) == \
        "request 3's entry doesn't open under round 2's request table"
    bent = [{**log[0], "calls": log[1]["calls"]}, *log[1:]]
    assert why(e0, o0, log=bent) == "round 0's root is not over its calls and its request table"
    coinless = [{**log[0], "coin": None}, *log[1:]]
    assert why(e0, o0, log=coinless) == "round 0 has no live coin, so nothing shows request 0's entry committed before one"
    early = [{**log[0], "coin": {**log[0]["coin"], "after": "00" * 64}}, *log[1:]]
    assert why(e0, o0, log=early) == "round 0's coin was not opened after its root, which holds request 0's entry"
    assert PH.rounds_reason(early, rounds.challenger.commitment) == \
        f"{Wd.ROUND} coin 0: coin was not opened after what the verifier recorded"
    assert PH.rounds_reason(log[1:], rounds.challenger.commitment) == "the round log is not rounds 0, 1, ... in order"
    assert why(e1, o1) is None


def test_the_engines_admissions_go_to_the_next_steps_round_and_past_the_window_are_refused():
    """`admitting` hands the scheduler's new requests to the run before the step's forward, a dummy run or an empty step
    none; a request admitted past the window's S steps, or with no prompt, gets no entry and is named."""
    rng = np.random.default_rng(13)
    run = PH.H1Run("t", SALT, {0: "q"}, [(32, K)], 32, 2, rounds=_rounds(2), key=bytes(32))
    seen = []
    step = PH.admitting(run, lambda so, *a, **kw: seen.append(so))
    step(_so(([1], 1)), None, True)
    step(_so(([2], 1)), dummy_run=True)
    step(_so(([3], 1), tokens=0))
    assert run.requests == [] and len(seen) == 3
    step(_so(([4, 5], 2), ([], 2)))
    run.record(0, *_tap(rng, 2, 32))
    step(_so(([6], 1)))
    run.record(0, *_tap(rng, 1, 32))
    step(_so(([7], 1)))
    run.close()
    assert [(e.prompt, e.round) for e in run.requests] == [((4, 5), 0), ((6,), 1)]
    assert run.refused_requests == [{"round": 0, "reason": "no prompt token ids or no token cap"},
                                    {"round": 2, "reason": "admitted for step 2, past the window's 2 steps"}]
    assert run.details["requests"] == {"admitted": 2, "refused": run.refused_requests, "per_round": [1, 1], "coins": "live"}
