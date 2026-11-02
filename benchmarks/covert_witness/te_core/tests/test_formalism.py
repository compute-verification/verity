"""te_core.formalism and te_core.policy: Fit and Slowdown as the formalism states them under both readings of the
frozen-tail case, the stubs and upper bounds labelled, the grid."""
import math

import pytest

from te_core import formalism as F
from te_core import frontier, policy
from te_core.gates import per_mac

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}


def measured(u1=1e8, phi=0.5, delta=None, dgrad=None, decoder=None, u=None, tail=0.0) -> F.Measured:
    return F.Measured((F.Part(gates=1e12, tokens=1e3, dep_gates=phi * 1e12, delta_grad_bits=dgrad),), tau_bits=32,
                      a_bits=8192, u=u or ((0.0, 0.0), (1.0, u1)), delta_bits=delta if delta is not None else u1,
                      decoder_gates=decoder, tail_share=tail)


def at(m: F.Measured, c: float, k: float) -> dict:
    """`F.at` for a record with no frozen tail, where the two readings agree."""
    lit, tail = (F.at(m, c, k, name) for name in F.READINGS)
    assert {k_: v for k_, v in lit.items() if k_ != "reading"} == {k_: v for k_, v in tail.items() if k_ != "reading"}
    return lit


def test_even_update_slowdown_is_the_corollary():
    """Evenly spread with u(1) >> k: rho ≳ 4 phi s a u(1) / k^2."""
    m, c, k = measured(u1=1e9), 1e13, 1e5
    r = at(m, c, k)
    s = c / 1e9
    assert r["x0"] == pytest.approx(k / 1e9)
    assert r["room"] == pytest.approx(k * k / (4 * 1e9))
    assert r["rho"] == pytest.approx(4 * 0.5 * s * 8192 * 1e9 / k ** 2, rel=1e-3)


def test_frozen_tail_slowdown_under_each_reading():
    """A frozen tail of share t above an evenly spread update with u(1) >> k: the room is t·k under both readings;
    literal-x0 takes x0 = t on the whole profile (rho ≈ phi (1 - t) s a / (k t)), tail-charged x0 = k / u(1) from the
    token's start, charging the tail through a (rho ≈ phi (1 - k/u(1)) s a / (k t)), the larger."""
    t, k, u1, c = 0.05, 1e5, 1e9, 1e13
    m = measured(u1=u1, tail=t)
    lit, tail = F.at(m, c, k, F.LITERAL), F.at(m, c, k, F.TAIL_CHARGED)
    assert (lit["reading"], tail["reading"]) == ("literal-x0", "tail-charged")
    assert lit["room"] == tail["room"] == pytest.approx(t * k)
    assert lit["x0"] == pytest.approx(t) and tail["x0"] == pytest.approx(k / u1)
    assert lit["rho"] == pytest.approx(0.5 * (1 - t) * (c / 1e9) * 8192 / (k * t), rel=1e-6)
    assert tail["rho"] == pytest.approx(0.5 * (1 - k / u1) * (c / 1e9) * 8192 / (k * t), rel=1e-6)
    assert at(measured(u1=u1), c, k)["rho"] > tail["rho"] > lit["rho"]
    with pytest.raises(ValueError):
        F.at(m, c, k, "witness")


def test_the_frozen_tail_profile_is_zero_up_to_t_then_the_from_start_profile():
    t, u1, k = 0.2, 1e9, 1e5
    u = F.with_tail(F.even(u1), t)
    assert F.u_at(u, 0.1) == 0 and F.u_at(u, 0.5) == pytest.approx(0.5 * u1)
    assert F.bottom_share(u, k) == pytest.approx(t) and F.room(u, k) == pytest.approx(t * k)
    assert F.with_tail(F.even(u1), 0.0) == F.even(u1)


def test_fit_makes_rho_one_and_flags_a_slowdown_bound_above_it():
    k = 1e9
    fits = measured(u1=1e6, dgrad=1e6)
    assert at(fits, 1e13, k)["rho"] == 1.0
    assert at(fits, 1e13, k)["parts"][0]["fit_training"]
    no_grad = measured(u1=1e6)
    assert at(no_grad, 1e13, k)["parts"][0]["fit"] and not at(no_grad, 1e13, k)["parts"][0]["fit_training"]
    wide = at(measured(u1=1e8, delta=1e3, dgrad=1e3), 1e13, 1e6)
    assert wide["rho"] == 1.0 and wide["flags"] == [F.FIT_OVER_SLOWDOWN]


def test_fit_k_is_the_least_k_at_which_fit_holds():
    """Running: Δ + sτ; training: max(Δ, 2Δ_grad) + sτ (None without Δ_grad); none when the decoder exceeds c."""
    m, c = measured(u1=1e6, delta=1e3, dgrad=1e4), 1e13
    s_tau = c / 1e9 * 32
    run, train = F.fit_k(m, c)
    assert (run, train) == pytest.approx((1e3 + s_tau, 2e4 + s_tau))
    assert F.at(m, c, train, F.LITERAL)["parts"][0]["fit_training"]
    assert not F.at(m, c, train * (1 - 1e-9), F.LITERAL)["parts"][0]["fit_training"]
    assert F.fit_k(measured(u1=1e6), c)[1] is None
    assert F.fit_k(measured(u1=1e6, decoder=1e14), c) == (None, None)


def test_fit_reads_delta_and_slowdown_reads_u1_as_separate_fields():
    """A small total update (Fit's Δ) with a large per-token update (Slowdown's u(1)) fits for running, while the
    Slowdown bound reads u(1); the record says which value each came from."""
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "dep_gates": 5e11, "steps": 1, "delta_grad_bits": None}
    m = F.complete({"parts": [part], "u1_bits": 1e9, "labels": {"u1_bits": "upper bound (witness): a test"}}, row, SHAPE)
    assert (m.delta_bits, m.u1_bits, m.u, m.tail_share) == (80, 1e9, F.even(1e9), 0.0)
    assert m.labels["u1_bits"] == "upper bound (witness): a test" and m.labels["u"] == F.STUBS["u"]
    k = 1e6
    r = at(m, 1e13, k)
    assert r["parts"][0]["fit"] and not r["parts"][0]["fit_training"] and r["x0"] == pytest.approx(k / 1e9)
    assert r["rho"] == pytest.approx(4 * 0.5 * 1e4 * 512 * 16 * 1e9 / k ** 2, rel=2e-3)
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u": [[0, 0], [1, 5.0]], "u1_bits": 6.0}, row, SHAPE)


def test_a_block_s_tail_share_is_read_and_labelled():
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "dep_gates": 5e11}
    m = F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 0.25}, row, SHAPE)
    assert m.tail_share == 0.25 and m.labels["tail_share"].startswith("upper bound")
    assert F.from_json(m.to_json()) == m
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 1.0}, row, SHAPE)


def test_a_decoder_wider_than_c_counts_no_update():
    assert at(measured(u1=1e6, dgrad=1e6, decoder=1e14), 1e13, 1e9)["delta_counted"] is False
    assert not at(measured(u1=1e6, dgrad=1e6, decoder=1e14), 1e13, 1e9)["parts"][0]["fit"]


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
    assert [p["rho"] for p in at(m, 1e13, 1e4)["parts"][:2]] == [1.0, 1.0]


def test_the_grid_spans_the_reference_shapes_in_decades():
    s = policy.shapes(SHAPE)
    w_fwd = SHAPE["forward_macs_per_token"] * per_mac("gemm/bf16")
    assert s["inference"]["c"] == w_fwd and s["inference"]["k"] == 16 + 1023 * 2 * 6 * 512 * 16
    ref = s["reference"]
    assert (ref["c"], ref["k"]) == (4096 * w_fwd, 80_000)
    serving = F.Measured((F.Part(1024 * w_fwd, 1024, 0.0),), 16, 512 * 16, ((0.0, 0.0), (1.0, 0.0)), 0.0)
    assert at(serving, ref["c"], ref["k"])["parts"][0]["fit"]  # honest serving passes Fit at the reference policy
    assert s["server"]["c"] == pytest.approx(3 * SHAPE["forward_macs_per_token"] * SHAPE["tokens"] * per_mac("gemm/bf16"))
    assert s["server"]["k"] == SHAPE["parameters"] * 80
    g = policy.grid(SHAPE)
    cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
    assert cs[0] <= w_fwd and cs[-1] >= s["server"]["c"] and ks[0] <= 16384 and ks[-1] >= s["server"]["k"]
    assert all(math.log10(c).is_integer() for c in cs) and len(g) == len(cs) * len(ks)
    assert policy.of_key(policy.key(1e14, 1e4)) == (1e14, 1e4)
    assert policy.c0(SHAPE | {"c0_gates": 5.0}) == (5.0, "counted")


def test_the_cells_are_the_grid_and_the_reference_shapes_at_their_exact_c_and_k():
    cells, s = policy.cells(SHAPE), policy.shapes(SHAPE)
    assert {m: cells[m] for m in policy.MARKS} == {m: (v["c"], v["k"]) for m, v in s.items()}
    assert {v for k, v in cells.items() if k not in policy.MARKS} == set(policy.grid(SHAPE))


def test_honest_is_the_least_gates_of_any_checkpoint_and_s_is_at_least_one():
    """The coordinator's 19:50Z rule: H and W* read every checkpoint of every valid record, a mid-schedule one
    included, under each reading."""
    row = {"start": "s", "output": {"e_bytes": 10 ** 6}, "start_bpb": 2.0, "direction": "d", "annotation": {"steps": 1}}

    def r(i, gates_macs, bpb, step, run):
        log = {"by": {"fwd/theta": gates_macs}, "ops": {"fwd theta mm [[1]]": [1, gates_macs]}}
        return row | {"id": f"r{i}", "macs_log": log, "val_bpb": bpb, "checkpoint_step": step, "run_id": run}

    rows = [r(1, 10 ** 9, 1.9, 9, "a"), r(2, 10 ** 8, 1.9, 4, "b"), r(3, 10 ** 10, 1.9, 9, "b")]
    shapes = {"s": SHAPE}
    scored, skipped = frontier.formalism_rows(rows, shapes)
    assert not skipped and all(set(x["rho_by"]) == set(F.READINGS) for x in scored)
    out = frontier.formalism_targets(scored, shapes)["s"]
    last = [t for t in out["targets"] if t["drop"] == frontier.TARGETS[-1]]
    assert [t["reading"] for t in last] == list(F.READINGS)
    for t in last:
        assert t["honest"] == pytest.approx(10 ** 8 * per_mac("gemm/bf16")) and t["honest_of"]["id"] == "r2"
        assert all(v["S"] >= 1 for v in t["cells"].values() if v["S"] is not None)
        assert set(policy.MARKS) <= set(t["cells"]) and {v["basis"] for v in t["cells"].values()} <= {"fit", "target"}
    assert set(out["marks"]) == set(policy.MARKS)
    assert len(out["targets"]) == len(frontier.TARGETS) * len(F.READINGS)
