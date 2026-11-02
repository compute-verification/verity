"""te_core.formalism and te_core.policy: Fit and Slowdown as the formalism states them, the stubs and upper bounds
labelled, the grid."""
import math

import pytest

from te_core import formalism as F
from te_core import frontier, policy
from te_core.gates import per_mac

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}


def measured(u1=1e8, phi=0.5, delta=None, dgrad=None, decoder=None, u=None) -> F.Measured:
    return F.Measured((F.Part(gates=1e12, tokens=1e3, dep_gates=phi * 1e12, delta_grad_bits=dgrad),), tau_bits=32,
                      a_bits=8192, u=u or ((0.0, 0.0), (1.0, u1)), delta_bits=delta if delta is not None else u1,
                      decoder_gates=decoder)


def test_even_update_slowdown_is_the_corollary():
    """Evenly spread with u(1) >> k: rho ≳ 4 phi s a u(1) / k^2."""
    m, c, k = measured(u1=1e9), 1e13, 1e5
    r = F.at(m, c, k)
    s = c / 1e9
    assert r["x0"] == pytest.approx(k / 1e9)
    assert r["room"] == pytest.approx(k * k / (4 * 1e9))
    assert r["rho"] == pytest.approx(4 * 0.5 * s * 8192 * 1e9 / k ** 2, rel=1e-3)


def test_frozen_tail_slowdown_is_the_corollary():
    """A single updated layer under a frozen tail (u = 0 on the tail's share t, then the whole update): rho ≈
    phi s a / (k t)."""
    t, k = 0.05, 1e5
    m = measured(u=((0.0, 0.0), (t, 0.0), (t + 1e-9, 1e9), (1.0, 1e9)))
    r = F.at(m, 1e13, k)
    assert r["x0"] == pytest.approx(t, rel=1e-6)
    assert r["rho"] == pytest.approx(0.5 * (1 - t) * (1e13 / 1e9) * 8192 / (k * t), rel=1e-3)


def test_fit_makes_rho_one_and_flags_a_slowdown_bound_above_it():
    k = 1e9
    fits = measured(u1=1e6, dgrad=1e6)
    assert F.at(fits, 1e13, k)["rho"] == 1.0
    assert F.at(fits, 1e13, k)["parts"][0]["fit_training"]
    no_grad = measured(u1=1e6)
    assert F.at(no_grad, 1e13, k)["parts"][0]["fit"] and not F.at(no_grad, 1e13, k)["parts"][0]["fit_training"]
    wide = F.at(measured(u1=1e8, delta=1e3, dgrad=1e3), 1e13, 1e6)
    assert wide["rho"] == 1.0 and wide["flags"] == [F.FIT_OVER_SLOWDOWN]


def test_fit_reads_delta_and_slowdown_reads_u1_as_separate_fields():
    """A small total update (Fit's Δ) with a large per-token update (Slowdown's u(1)) fits for running, while the
    Slowdown bound reads u(1); the record says which value each came from."""
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "dep_gates": 5e11, "steps": 1, "delta_grad_bits": None}
    m = F.complete({"parts": [part], "u1_bits": 1e9, "labels": {"u1_bits": "upper bound (witness): a test"}}, row, SHAPE)
    assert (m.delta_bits, m.u1_bits, m.u) == (80, 1e9, F.even(1e9))
    assert m.labels["u1_bits"] == "upper bound (witness): a test" and m.labels["u"] == F.STUBS["u"]
    k = 1e6
    r = F.at(m, 1e13, k)
    assert r["parts"][0]["fit"] and not r["parts"][0]["fit_training"] and r["x0"] == pytest.approx(k / 1e9)
    assert r["rho"] == pytest.approx(4 * 0.5 * 1e4 * 512 * 16 * 1e9 / k ** 2, rel=2e-3)
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u": [[0, 0], [1, 5.0]], "u1_bits": 6.0}, row, SHAPE)


def test_the_frozen_tail_profile_is_the_least_of_the_two_witnesses():
    t, u1, k = 0.2, 1e9, 1e5
    u = F.frozen_tail(u1, t)
    assert F.u_at(u, 0.1) == 0 and F.u_at(u, 0.5) == pytest.approx(0.5 * u1)
    assert F.bottom_share(u, k) == pytest.approx(t) and F.room(u, k) == pytest.approx(t * k)
    assert F.frozen_tail(u1, 0.0) == F.even(u1)


def test_a_decoder_wider_than_c_counts_no_update():
    assert F.at(measured(u1=1e6, dgrad=1e6, decoder=1e14), 1e13, 1e9)["delta_counted"] is False
    assert not F.at(measured(u1=1e6, dgrad=1e6, decoder=1e14), 1e13, 1e9)["parts"][0]["fit"]


def test_a_v23_row_is_measured_from_its_old_fields_with_every_stub_named():
    row = {"start": "s50", "output": {"e_bytes": 1000}, "annotation": {"steps": 3},
           "macs_log": {"by": {"fwd/theta": 3 * SHAPE["forward_macs_per_token"], "fwd/updated": 10},
                        "ops": {"fwd theta mm [[1]]": [1, 3 * SHAPE["forward_macs_per_token"]],
                                "fwd updated mm [[1]]": [1, 10]}}}
    m = F.measure(row, SHAPE)
    assert m.tokens == pytest.approx(3 + 10 / SHAPE["forward_macs_per_token"])
    assert m.parts[0].dep_gates == pytest.approx(10 * per_mac("gemm/bf16"))
    assert m.delta_bits == 8000 and m.u == ((0.0, 0.0), (1.0, 8000))
    assert m.tau_bits == 16 and m.a_bits == 512 * 16 and m.u1_bits == 8000
    assert set(m.labels) == {"gates", "tokens", "dep_gates", "tau_bits", "a_bits", "delta_bits", "decoder_gates", "u",
                             "u1_bits", "delta_grad_bits"}
    assert all(m.labels[k].startswith("stub") for k in ("tokens", "dep_gates", "tau_bits"))
    assert all(m.labels[k].startswith("upper bound") for k in ("a_bits", "u", "u1_bits", "delta_bits"))
    with pytest.raises(ValueError):
        F.measure(row | {"output": {}}, SHAPE)


def test_a_record_with_te_core_4_cuts_reads_them_as_delta_grad():
    record = {"steps": [{"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                        {"step": 1, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                        {"step": 2, "tokens": 8, "struct": "b", "by": {"fwd/theta": 40}}],
              "structures": {"a": {"updated": [[2, 0, 16, 8, 16]], "log": [["fwd", "theta", "mm", "[[8]]", 1, 80]],
                                   "cut_bits": 512},
                             "b": {"updated": [], "log": [["fwd", "theta", "mm", "[[8]]", 1, 40]], "cut_bits": 0}},
              "formalism": {"a_bits": 99, "labels": {"a_bits": "measured: the traced cut"}}}
    f = F.of_record(record)
    a, b = f["parts"]
    assert (a["steps"], a["tokens"], a["delta_grad_bits"]) == (2, 16, 512)
    assert a["dep_gates"] == pytest.approx(2 * 16 * per_mac("gemm/bf16"))
    assert a["gates"] == pytest.approx(160 * per_mac("gemm/bf16"))
    assert (b["steps"], b["dep_gates"]) == (1, 0)
    assert f["a_bits"] == 99 and f["labels"]["delta_grad_bits"].startswith("measured")
    m = F.complete(f, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.a_bits == 99 and m.labels["a_bits"] == "measured: the traced cut" and len(m.parts) == 2
    assert F.of_record(record, through=0)["parts"][0]["steps"] == 1


def test_setup_and_tail_count_in_c_as_parts_with_no_token_blocks():
    step = {"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}}
    record = {"setup": {"by": {"other/theta": 30}, "struct": None}, "tail": {"by": {"other/theta": 5}, "struct": None},
              "steps": [step, step | {"step": 1}],
              "structures": {"a": {"updated": [], "log": [["fwd", "theta", "mm", "[[8]]", 1, 80]]}}}
    whole, prefix = F.of_record(record), F.of_record(record, through=0)
    assert [p["tokens"] for p in whole["parts"]] == [0, 0, 16] and [p["tokens"] for p in prefix["parts"]] == [0, 8]
    m = F.complete(whole, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.gates == pytest.approx(160 * per_mac("gemm/bf16") + 35 * per_mac("elementwise/bf16"))
    assert [p["rho"] for p in F.at(m, 1e13, 1e4)["parts"][:2]] == [1.0, 1.0]


def test_the_grid_spans_the_reference_shapes_in_decades():
    s = policy.shapes(SHAPE)
    w_fwd = SHAPE["forward_macs_per_token"] * per_mac("gemm/bf16")
    assert s["inference"]["c"] == w_fwd and s["inference"]["k"] == 16 + 1023 * 2 * 6 * 512 * 16
    ref = s["reference"]
    assert (ref["c"], ref["k"]) == (4096 * w_fwd, 80_000)
    serving = F.Measured((F.Part(1024 * w_fwd, 1024, 0.0),), 16, 512 * 16, ((0.0, 0.0), (1.0, 0.0)), 0.0)
    assert F.at(serving, ref["c"], ref["k"])["parts"][0]["fit"]  # honest serving passes Fit at the reference policy
    assert s["server"]["c"] == pytest.approx(3 * SHAPE["forward_macs_per_token"] * SHAPE["tokens"] * per_mac("gemm/bf16"))
    assert s["server"]["k"] == SHAPE["parameters"] * 80
    g = policy.grid(SHAPE)
    cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
    assert cs[0] <= w_fwd and cs[-1] >= s["server"]["c"] and ks[0] <= 16384 and ks[-1] >= s["server"]["k"]
    assert all(math.log10(c).is_integer() for c in cs) and len(g) == len(cs) * len(ks)
    assert policy.of_key(policy.key(1e14, 1e4)) == (1e14, 1e4)
    assert policy.c0(SHAPE | {"c0_gates": 5.0}) == (5.0, "counted")


def test_honest_is_the_least_gates_of_any_complete_schedule_and_s_is_at_least_one():
    row = {"start": "s", "output": {"e_bytes": 10 ** 6}, "start_bpb": 2.0, "direction": "d", "annotation": {"steps": 1}}

    def r(i, gates_macs, bpb, step, run):
        log = {"by": {"fwd/theta": gates_macs}, "ops": {"fwd theta mm [[1]]": [1, gates_macs]}}
        return row | {"id": f"r{i}", "macs_log": log, "val_bpb": bpb, "checkpoint_step": step, "run_id": run}

    rows = [r(1, 10 ** 9, 1.9, 9, "a"), r(2, 10 ** 8, 1.9, 4, "b"), r(3, 10 ** 10, 1.9, 9, "b")]
    shapes = {"s": SHAPE}
    scored, skipped = frontier.formalism_rows(rows, shapes)
    assert not skipped
    out = frontier.formalism_targets(scored, shapes, frontier.schedule_ends(rows))["s"]
    t = out["targets"][-1]
    assert t["honest"] == pytest.approx(10 ** 9 * per_mac("gemm/bf16")) and t["honest_of"]["id"] == "r1"
    assert all(v["S"] >= 1 for v in t["cells"].values() if v["S"] is not None)
    assert set(out["marks"]) == set(policy.MARKS)
