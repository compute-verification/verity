"""te_core.rho_vectors: the one bound's published test vectors are what te_core.formalism makes from the capture's
fixtures, their inputs reproduce through `from_json` (the console's route), and they say what reading A, with its
matrix-product tiles, says of wildcard's grown adapter, a dense step and a LoRA step on the tiny start; the products
term is the theorem of record's on two hand-computed cases and four traced steps; the products term and the old
figure at every k of the console's grid are c times two captured steps' fresh_per_c; and the console's c,
te_core.policy's session at every stated size, each of the console's reference sizes and each start, is its rule."""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from te_core import VERSION, formalism, policy, rho_vectors, tiles
from te_core.flops import forward_macs_per_token


def differences(a, b, at: str = "") -> list[str]:
    """Where `a` and `b` differ, numbers to the vectors' tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{at}: keys {sorted(a.keys() ^ b.keys())}"]
        return [d for k in a for d in differences(a[k], b[k], f"{at}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{at}: {len(a)} items, not {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{at}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) else [f"{at}: {a} != {b}"]
    return [] if a == b else [f"{at}: {a!r} != {b!r}"]


def num(x):
    """A vector's number: "inf" is ∞ (null, unmeasured, stays None)."""
    return math.inf if x == "inf" else x


@pytest.fixture(scope="module")
def built() -> dict:
    return rho_vectors.build()


def by(built: dict, case: str, n: int, k: int) -> dict:
    return next(v for v in built["vectors"] if (v["case"], v["tokens_per_block"], v["k"]) == (case, n, k))


def record(built: dict, case: str) -> formalism.Measured:
    """A case's record as a port builds it from the JSON: its `measured`, or for the void case (`void_case`) the
    dense case's with the products of the `fresh` vector it names."""
    v = built["void_case"]
    if case != v["case"]:
        return formalism.from_json(built["measured"][case])
    products = next(f for f in built["fresh"]["vectors"] if f["name"] == v["products"])["inputs"]["products"]
    return formalism.from_json(built["measured"][v["measured"]] | {"products": products})


def test_the_published_vectors_are_what_the_module_makes(built):
    have = json.loads(rho_vectors.OUT.read_text())
    diff = differences(have, json.loads(json.dumps(built)))
    assert not diff, ("te_core/rho_vectors.json is stale; run `python -m te_core.rho_vectors --write` from "
                      "benchmarks/covert_witness:\n" + "\n".join(diff[:10]))


def test_each_case_reproduces_through_from_json_and_each_vector_is_its_bound(built):
    """The console reads a case's `measured` (from_json's form) and a vector's c and k; `bound` and `at` under reading
    A there give the vector's terms, max and ρ."""
    for name, case in rho_vectors.CASES.items():
        m = rho_vectors.case_measured(case)
        assert formalism.from_json(json.loads(json.dumps(asdict(m)))) == m
        assert m.ops == json.loads((rho_vectors.FIXTURES / case["circuit"]).with_suffix(".ops.json").read_text())
    for v in built["vectors"][::3]:
        m = formalism.from_json(built["measured"][v["case"]])
        b, a = formalism.bound(m, v["c"], v["k"]), formalism.at(m, v["c"], v["k"], "A")
        have = rho_vectors.finite([b["terms"], b["max"], b["binding"], b["sensitivities"], a["rho"]])
        assert not differences(have, [v["terms"], v["max"], v["binding"], v["sensitivities"], v["rho"]])


def test_every_rho_is_rho_fit_where_fit_holds_else_the_max_over_measured_kinds_never_a_sensitivity(built):
    assert built["version"] == VERSION and built["kinds"] == list(formalism.KINDS)
    assert set(built["rests_on"]) == set(formalism.KINDS) and set(built["sensitivities"]) == set(formalism.SENSITIVITIES)
    for v in built["vectors"]:
        assert v["label"] == f"{formalism.THE_CODE} {VERSION}, reading A"
        measured = {kind: num(t) for kind, t in v["terms"].items() if t is not None}
        assert set(v["terms"]) - set(measured) <= set(v["unmeasured"])
        top = max(measured.values()) if measured else None
        assert num(v["max"]) == top and (top is None or num(v["terms"][v["binding"]]) == top)
        assert v["binding"] not in formalism.SENSITIVITIES
        if v["fit"]:
            assert v["rho"] == v["rho_fit"] and (top is None or top <= v["rho_fit"])
        else:
            assert num(v["rho"]) == (None if top is None else max(1.0, top))


def test_a_dense_step_and_a_lora_step_have_a_finite_token_term_at_the_console_default(built):
    """Every block tensor changes, so a run block serving a token at a late position reads the earlier positions' K/V
    or computes their state, and at the default k = 80,000 bits no run block holds it; a block's best union of runs
    and tiles still holds a few percent of c, so the token term is finite (4.4's runs alone gave ∞, the
    full-token-state overclaim), above 1, and binds: ρ is it, below the per-gate sensitivity, which isn't closed under
    unions. With k = 8 Mbit (dense) or 800 kbit (LoRA, whose update is 10 times smaller) a block holds c: the term is
    the affected share. The products term is a measured 0 at these k (the fixtures' products are classified at each k
    of the console's dial, and M's operands are read far below their widths), and nearly every token gate is in a
    tile."""
    default = rho_vectors.CELLS[0]
    for case, k in (("dense", 8_000_000), ("lora", 800_000)):
        a = by(built, case, *default)
        assert 1 < a["terms"]["token"] < math.inf and a["binding"] == "token" and a["rho"] == a["terms"]["token"]
        assert a["terms"]["token"] < a["sensitivities"]["per_gate"]
        assert a["terms"]["fresh_products"] == 0 and "fresh_products" not in a["unmeasured"]
        assert a["inputs"]["token"]["unplaced"] < 0.01
        at = by(built, case, default[0], k)
        assert at["terms"]["token"] == pytest.approx(at["inputs"]["token"]["share"])


def test_the_grown_adapter_is_bounded_by_its_tiles_and_by_gates_with_its_live_update_code(built):
    """wildcard's grown adapter (a = 32 only with w_down's live update bits read): its work at the grown block reads
    w_down's whole bf16 cast, 196,608 bits > k, so with Δ the trace's cut no run block holds it, and a block's tiles
    bound it, above 1. An E of the live entries alone (26,624 bits) lets a block read the code instead, and the token kind
    is then the affected share (gates bind), so ρ is 1: the reduction sensitivity (2,048 tokens fill two blocks, one
    1.18 Mbit partial read at k a block) is above 9 but holds only for partitions split by tokens, so it isn't the
    bound."""
    default = rho_vectors.CELLS[0]
    full, live = by(built, "adapter", *default), by(built, "adapter-live", *default)
    assert 1 < full["terms"]["token"] < math.inf and full["rho"] == full["terms"]["token"]
    assert live["terms"]["token"] == pytest.approx(live["inputs"]["token"]["share"]) and live["rho"] == 1.0
    assert live["sensitivities"]["reduction"] > 9 and live["inputs"]["reduction"]["blocks"] == [2]


def test_each_cases_op_log_holds_its_circuits_gates():
    """The `.ops.json` beside each circuit (te_core.tiles.of_graph on the same trace) holds exactly the circuit's
    gates, its GEMMs placed."""
    from te_core.capture.circuit import StepCircuit

    for case in rho_vectors.CASES.values():
        circuit = rho_vectors.FIXTURES / case["circuit"]
        ops = json.loads(circuit.with_suffix(".ops.json").read_text())
        assert sum(o["gates"] for o in ops) == pytest.approx(float(StepCircuit.load(circuit).gates.sum()), rel=1e-12)
        assert all(o["products"] for o in ops if o["op"] != "rest") and tiles.unplaced(ops) < 0.01


def test_the_published_rho_is_reading_as_where_the_token_kind_is_measured_else_a_target(built):
    """The coordinator's 08:26Z ruling: the published ρ's max takes only the kinds valid on their own, each binding
    kind with what it rests on, and since 4.5's packer (`READING_A`; the coordinator's 09:02Z: reading B until then) a
    row whose token kind is measured publishes reading A, `at`'s ρ, with the bound's terms and its reduction and update
    sensitivities (the console's cell has room for no more). At the
    dial's default the dense, LoRA and grown-adapter steps bind on the token kind with every kind measured (their
    products term is a measured 0 at each k of the dial); the adapter with its live code has no kind above 1, so it
    publishes 1, binding none, its reduction sensitivity above 9 beside it, and off the dial (`OFF_DIAL`), where its
    products term is unmeasured, None, saying so (coordinator 12:20Z). A row without entries or an op log
    publishes reading B, Slowdown's token form, a target, and so does every row where `READING_A` is off (on since
    the coordinator's 11:18Z ruling); read under A, such a row measures only the products kind (a 0 here), so
    it publishes None, binding none and resting on `unmeasured`, never 1 (coordinator 09:46Z)."""
    default = rho_vectors.CELLS[0]
    for case in rho_vectors.CASES:
        v = by(built, case, *default)
        m = formalism.from_json(built["measured"][case])
        one = formalism.published(m, v["c"], v["k"], "A")
        assert formalism.published(m, v["c"], v["k"])["reading"] == ("A" if formalism.READING_A else "B")
        assert one["reading"] == "A" and not v["fit"]
        assert not differences(rho_vectors.finite([one["terms"], one["sensitivities"]]),
                               [v["terms"], {n: v["sensitivities"][n] for n in formalism.CELL_SENSITIVITIES}])
        assert one["rests_on"] == formalism.RESTS_ON_CELL["unmeasured" if one["rho"] is None else one["binding_kind"]]
        if case != "adapter-live":
            assert (one["rho"], one["binding_kind"]) == (num(v["rho"]), "token") == (num(v["rho"]), v["binding"])
            assert not one["partial"] and one["why"] is None
    live = formalism.from_json(built["measured"]["adapter-live"])
    one = formalism.published(live, v["c"], v["k"], "A")
    assert (one["rho"], one["binding_kind"], one["partial"]) == (1.0, "none", False)
    assert one["sensitivities"]["reduction"] > 9
    off = formalism.published(live, v["c"], rho_vectors.OFF_DIAL, "A")
    assert (off["rho"], off["binding_kind"], off["partial"]) == (None, "unmeasured", False)
    assert off["why"] == "the measured kinds give at most 1; fresh products unmeasured"
    assert max(num(t) for t in off["terms"].values() if t is not None) <= 1
    for gone in ("entries", "ops"):
        m = formalism.from_json(built["measured"]["dense"] | {gone: None})
        b, at = formalism.published(m, v["c"], v["k"]), formalism.at(m, v["c"], v["k"])
        assert (b["reading"], b["binding_kind"], b["rho"]) == ("B", "target", at["rho"])
        assert b["terms"] == {"token_form": at["slowdown"]} and tuple(b["sensitivities"]) == formalism.CELL_SENSITIVITIES
        assert b["rests_on"] == formalism.RESTS_ON_CELL["target"]
        a = formalism.published(m, v["c"], v["k"], "A")
        assert (a["reading"], a["rho"], a["binding_kind"], a["rests_on"]) == ("A", None, "unmeasured", "unmeasured")
        assert a["why"] == "the measured kinds give at most 1; token, decoding unmeasured"
        assert set(a["terms"]) == set(formalism.KINDS) and a["terms"] == {
            "fresh_products": 0.0, "token": None, "decoding": None}
    m = formalism.from_json(built["measured"]["dense"])
    assert formalism.published(m) == formalism.published(m, *policy.CONSOLE)


def test_the_fresh_old_vectors_are_io_bounds_figure_with_p_cell_with_the_figure_without_it_beside_it(built):
    """The console's port of the old fresh-products figure (the sensitivity fresh_old) tests against these: on each of
    io_bound's hand-written product sets fresh_old is io_bound's own vector's proved figure with P-cell and
    fresh_no_cell its figure without it, summed in the reference circuit's chain whether the order is declared or not,
    with what its best subset dropped. Where a product's chains have no fixed point, or no subset has a proved Λ, the
    figure is void (coordinator 21:54Z): it is io_bound's best subset's among the products no diverged cycle feeds
    (`converged`), or null with `FRESH_VOID` where no subset has one, and `void` has the row's reason. A circuit that
    declares a tree is unmeasured. The 45M start's dense step's figure reads its whole-making system, attention's
    64-long chains below every counted operand: 2,869,161 on the best subset, which drops the 4 products the 256-long
    chains feed, and 163,341 without P-cell on all 19, with G without each sequence's first position. The lowrank
    worker's DoRA rank-16 step on all 42 matrices and its LoRA rank-16 step on every block's MLP are void, each the
    figure of the products no diverged cycle feeds: DoRA's two, 4.485 (0.1695 without P-cell) on 7.0e-5 of the counted
    MACs, and the MLP step's one, 93.67 (31.65) on 5.6%."""
    f = built["fresh"]
    assert (f["p_cell"], f["order"]) == (formalism.P_CELL, formalism.ORDER) == (True, "chain")
    assert f["rests_on"] == formalism.SENSITIVITIES["fresh_old"] and f"given that {formalism.P_CELL_WORDS}" in f["rests_on"]
    vs = {v["name"]: v for v in f["vectors"]}
    from te_core import io_bound

    for want in io_bound.vectors()["vectors"]:
        v = vs[want["name"]]
        assert v["inputs"]["c"] == want["inputs"]["c"] and v["inputs"]["products"]["products"] == want["inputs"][
            "products"]
        t, conv = want["term"], want["term"]["converged"]
        best = conv or {"rho": t["proved_rho"], "rho_no_cell": t["proved_rho_no_cell"], "subset": t["proved_subset"]}
        assert (v["fresh_old"], v["fresh_no_cell"], v["order"], v["why"]) == (
            best["rho"], best["rho_no_cell"], "chain", formalism.FRESH_VOID if best["rho"] is None else None)
        sub = best["subset"]
        assert v["subset"] == (None if sub is None or not sub["dropped"] else {
            "dropped": sub["dropped"], "of": len(want["inputs"]["products"]),
            "dropped_macs_share": sub["dropped_macs_share"], "k_whole_min": sub["k_whole_min"]})
        assert (v["void"] is None) == (conv is None and best["rho"] is not None)
        if v["void"] is not None:
            assert v["void"]["diverged_fed"] == len(t["diverged_fed"])
            assert v["void"]["reason"] == formalism.void_reason(v["fresh_old"], v["void"]["kept_macs_share"])
    assert sum(v["why"] == formalism.FRESH_VOID for v in vs.values()) == 1
    assert sum(v["void"] is not None for v in vs.values()) == 4 and sum(bool(v["subset"]) for v in vs.values())
    for name, (term, no_cell, share, stuck, of) in {
            "DoRA rank-16": (4.48485, 0.16952, 6.98e-5, 28, 30),
            "LoRA rank-16 step on every block's MLP": (93.6711, 31.645, 0.05641, 10, 11)}.items():
        v = next(v for n, v in vs.items() if name in n)
        assert v["why"] is None and v["fresh_old"] == pytest.approx(term, rel=1e-5)
        assert v["fresh_no_cell"] == pytest.approx(no_cell, rel=1e-4)
        assert v["void"]["kept_macs_share"] == pytest.approx(share, rel=1e-3)
        assert (v["void"]["diverged_fed"], v["subset"]["dropped"], v["subset"]["of"]) == (stuck, stuck, of)
        assert v["inputs"]["products"]["capture"] == "4.12.9" and v["inputs"]["products"]["elementwise_products"] == []
    assert (f["fresh_no_cell"], f["void"]) == (formalism.SENSITIVITIES["fresh_no_cell"], formalism.FRESH_VOID)
    square = vs["one product, square, console"]
    assert vs["one product, square, console, no order declared"]["fresh_old"] == square["fresh_old"] > 1
    tree = vs["one product, square, console, a circuit that declares a tree"]
    assert (tree["fresh_old"], tree["order"]) == (None, "tree") and "tree" in tree["why"]
    dense = vs["the 45M start's dense step, its trace's order unknown"]
    assert (dense["inputs"]["c"], dense["inputs"]["k"]) == policy.CONSOLE and dense["order"] == "chain"
    assert dense["inputs"]["products"]["trace_order"] == "unknown"
    assert round(dense["fresh_old"]) == 2_869_161 and round(dense["fresh_no_cell"]) == 163_341
    assert (dense["subset"]["dropped"], dense["subset"]["of"]) == (4, 19)
    assert dense["inputs"]["products"]["system"] is not None and dense["inputs"]["products"]["p_match"] == []


def test_the_products_term_is_the_theorem_of_records_on_two_hand_computed_cases(built):
    """The theorem of record's ρ, (c/|C|)·(N/(k + α) − F·(k/2)·√(ON)/β/(k + α)^1.5) with α = F·½·√(k³/(β²·b_y)), on
    `HAND`'s two M, worked by hand at c/|C| = 10⁻³. (a) A forward and a backward product at β 16 and b_y 16 sharing
    arrays (F = 2/√3) at k = 3·2^14: √k = 2^7·√3, so α = (2/√3)·½·k·√k/64 = 2k and k + α = 3k; with O = N/48 the
    outputs' part is F·(k/2)·(N/√48)/16/(3k)^1.5 = F·N/(1152·√k) = (2/3)·N/(3k), so ρ = 10⁻³·N/(9k) = 10⁶ with N =
    2 × 2.21184e14. (b) At the console's k = 2^16, a forward product (N₁ = 5.89824e14, O₁ = N₁/48) beside a backward
    one an adapter's rank-16 output reaches (N₂ = 10¹⁷, O₂ = 10¹²), one operand at β_eff(16, 16) = 16/(1 + (32/16)²)
    = 3.2, K* = ½√(2^16/16) = 32. M at t = 16 is the forward product alone, F = 1 and α = ½·2^24/64 = 2k, its ρ again
    10⁶; at t = 3.2 both, F = 2/√3, α = F·2^23/12.8 = F·655,360, and ρ = 103,996,766.08, which binds. Each traced
    step's term is `formalism.he_products` on its products at the console's (c, k), with what it rests on: the 45M
    start's dense step and s-t20's at full size at β 16 and b_y 16 on the products both of whose operands are paid,
    and the lowrank worker's LoRA and DoRA rank-16 steps a measured 0 at every β."""
    assert formalism.k_star(65_536, 16) == 32 and formalism.beta_eff(16, 16, 65_536) == pytest.approx(3.2, rel=1e-15)
    F = 2 / math.sqrt(3)
    k = 49_152.0
    assert F * 0.5 * math.sqrt(k ** 3 / (16 * 16 * 16)) == pytest.approx(2 * k, rel=1e-14)
    assert formalism.he_rho(2 * 2.21184e14, 2 * 2.21184e14 / 48, 16, 16, F, 1e12, k, 1e15) == pytest.approx(
        1e6, rel=1e-12)
    k, alpha = 65_536.0, F * 655_360
    n, o = 5.89824e14 + 1e17, 5.89824e14 / 48 + 1e12
    by_hand = 1e-3 * (n / (k + alpha) - F * (k / 2) * math.sqrt(o * n) / 3.2 / (k + alpha) ** 1.5)
    assert by_hand == pytest.approx(103_996_766.08, rel=1e-10)
    assert formalism.he_rho(5.89824e14, 5.89824e14 / 48, 16, 16, 1.0, 1e12, k, 1e15) == pytest.approx(1e6, rel=1e-12)
    p = built["products"]
    assert p["rests_on"] == formalism.RESTS_ON["fresh_products"]
    vs = {v["name"]: v for v in p["vectors"]}
    assert list(vs) == [h[0] for h in rho_vectors.HAND] + [
        "the 45M start's dense step", "s-t20's dense step at full size",
        *(label.split(" (")[0] for label in rho_vectors.BATTERY_VOID[1].values())]
    (a_name, *_), (b_name, *_) = rho_vectors.HAND
    a, b = vs[a_name], vs[b_name]
    assert (a["term"], a["beta"], a["b_y"], a["F"], a["alpha"]) == pytest.approx((1e6, 16, 16, F, 2 * 49_152),
                                                                                 rel=1e-12)
    assert (a["N"], a["O"], a["below_width"]) == (pytest.approx(2 * 2.21184e14), pytest.approx(2 * 2.21184e14 / 48),
                                                  False)
    assert (b["term"], b["beta"], b["F"], b["alpha"]) == pytest.approx((by_hand, 3.2, F, alpha), rel=1e-12)
    assert b["below_width"] and b["why"] is None and "β = 3.2 bits a word" in b["rests_on"]
    for v in vs.values():
        if v["name"] in (a_name, b_name):
            got = formalism.he_figure(v["inputs"]["he"], v["inputs"]["c"], v["inputs"]["C_gates"])
            assert got["rho"] == v["term"] and v["rests_on"] == formalism.he_rests_on(got)
            continue
        assert (v["inputs"]["c"], v["inputs"]["k"]) == policy.CONSOLE and v["why"] is None
        assert (v["beta"], v["b_y"], v["F"], v["below_width"]) == (16, 16, pytest.approx(F), False), v["name"]
    assert round(vs["the 45M start's dense step"]["term"]) == 123_811
    assert round(vs["s-t20's dense step at full size"]["term"]) == 261_450
    assert vs["the 45M start's dense step"]["macs_share"] == pytest.approx(0.56412, rel=1e-4)
    for name in list(vs)[4:]:
        assert vs[name]["term"] == 0 and vs[name]["r_step"] == 16, name


def test_the_dial_vectors_are_c_times_a_rows_fresh_per_c_the_term_and_the_old_figure(built):
    """The console's dial ports the products term from a row's rho_inputs (te_core.panels' `rho_columns`, which holds
    `formalism.fresh_per_c` to its digits): at each k of the dial the old figure fresh_old is c times its entry's first
    number, null where that is null, and the term c times its second, each cell `formalism.he_products` and
    `formalism.fresh_products` there. The 45M dense step's term is positive at the dial's two least k and a measured 0
    from the served block's 145,568 bits up, where its outputs' part exceeds its MACs'; its old figure is measured at
    the three least k and void above them with no subset's figure; the tiny start's cases are a measured 0 at every
    k."""
    from dataclasses import replace

    from te_core import panels
    from te_core.capture.products import DIAL_K

    d = built["dial"]
    assert d["k"] == list(DIAL_K) == list(panels.K_LEVELS)
    ms = {name: formalism.from_json(built["measured"][name]) for name in rho_vectors.CASES}
    rows = json.loads((rho_vectors.FIXTURES / rho_vectors.DENSE_45M).read_text())
    C = rows["gates"]
    ms["the 45M start's dense step"] = replace(
        ms["dense"], parts=(formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
        products=formalism.products_of_trace(rows))
    assert [v["name"] for v in d["vectors"]] == list(ms)
    for v in d["vectors"]:
        per = v["fresh_per_c"]
        assert list(per) == [str(k) for k in DIAL_K] and v["C_gates"] == ms[v["name"]].gates
        assert len(v["cells"]) == len(DIAL_K) * (1 + len(rho_vectors.DIAL_TOKENS))
        for cell in v["cells"]:
            entry = per[str(cell["k"])]
            assert entry is not None and entry[1] is not None
            assert cell["term"] == pytest.approx(cell["c"] * entry[1], rel=1e-12) and cell["why"] is None
            if entry[0] is None:
                assert cell["fresh_old"] is cell["fresh_no_cell"] is None
            else:
                assert cell["fresh_old"] == pytest.approx(cell["c"] * entry[0], rel=1e-12)
        cols = json.loads(panels.rho_columns(ms[v["name"]], rho_vectors.W_FWD)["rho_inputs"])[panels.FRESH]
        assert cols.keys() == per.keys() and all(
            len(a) == len(b) and all((x is None) == (y is None) and (x is None or x == pytest.approx(y, rel=1e-5))
                                     for x, y in zip(a, b)) for a, b in zip(cols.values(), per.values()))
        if v["name"] in rho_vectors.CASES:
            assert all(e == [0.0, 0.0] for e in per.values())
    dense = d["vectors"][-1]["fresh_per_c"]
    assert [dense[str(k)][0] is None for k in DIAL_K] == [False] * 3 + [True] * 4
    assert [dense[str(k)][1] > 0 for k in DIAL_K] == [True] * 2 + [False] * 5
    assert d["rule"] == rho_vectors.PER_C_RULE and "fresh_per_c[k][1]" in d["rule"]


def test_the_grid_vectors_are_c_times_each_steps_fresh_per_c_at_every_k_of_the_consoles_grid(built):
    """The console's dial moves k over its grid, a session's input ids at 16 bits for 1,024 to 32,768 input tokens,
    with the serving block's 145,568 bits beside it: at each the term is c times the step's fresh_per_c entry's second
    number and the old figure c times its first, each cell `formalism.he_products` and `formalism.fresh_products` at
    the console's c and at the k's own (the 45M reference's session of k/16 input tokens and 1,024 output, the serving
    block's c at its k). At the console's c the 45M start's dense step's term falls from 2,636,920 at 16,384 bits to
    123,811 at 65,536 and s-t20's at full size from 4,382,587 to 261,450, each a measured 0 from 131,072 up, where
    the outputs' part exceeds the MACs'. The old figure on the 45M step falls from 25,843,305 to 778,360 at 145,568
    (709,375 to 64,340 without P-cell) and is void from 262,144 up with no subset's figure; on s-t20's from 39,795,651
    to 483,259 at 262,144 (1,083,180 to 56,556) and is void at 524,288 with no subset's figure, where a cycle of its
    whole-making system that feeds a counted product has no fixed point. The 45M fixture's rows at the console's k
    and the serving block's are its dial fixture's (`DENSE_45M`), one capture."""
    from dataclasses import replace

    from te_core import panels

    g = built["grid"]
    assert policy.K_GRID == (16_384, 32_768, 65_536, 131_072, 262_144, 524_288)
    assert policy.K_GRID == tuple(n * policy.id_bits(50_304) for n in policy.DIAL_IN_TOKENS)
    assert g["k"] == list(rho_vectors.grid_k()) == sorted({*policy.K_GRID, panels.SERVED[1]})
    assert panels.SERVED[1] == 145_568 and policy.CONSOLE[1] in g["k"]
    own = {k: policy.session_gates(policy.REFERENCE_PARAMS, n, 1024)
           for n, k in zip(policy.DIAL_IN_TOKENS, policy.K_GRID)} | {panels.SERVED[1]: panels.SERVED[0]}
    want = {"the 45M start's dense step": [2_636_920, 729_574, 123_811, 0, 0, 0, 0],
            "s-t20's dense step at full size": [4_382_587, 1_276_252, 261_450, 0, 0, 0, 0]}
    old = {"the 45M start's dense step": [25_843_305, 8_790_057, 2_869_161, 931_231, 778_360, None, None],
           "s-t20's dense step at full size": [39_795_651, 13_837_635, 4_749_660, 1_583_812, 1_334_785, 483_259, None]}
    no_cell = {"the 45M start's dense step": [709_375, 343_662, 163_341, 73_559, 64_340, None, None],
               "s-t20's dense step at full size": [1_083_180, 530_550, 257_407, 122_897, 109_643, 56_556, None]}
    assert [v["name"] for v in g["vectors"]] == list(rho_vectors.GRID_STEPS) == list(want)
    base = formalism.from_json(built["measured"]["dense"])
    for v in g["vectors"]:
        rows = json.loads((rho_vectors.GRID_FIXTURES / rho_vectors.GRID_STEPS[v["name"]][0]).read_text())
        assert sorted(rows["at"], key=int) == [str(k) for k in g["k"] if k != policy.CONSOLE[1]]
        C = rows["gates"]
        m = replace(base, parts=(formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
                    products=formalism.products_of_trace(rows))
        per = v["fresh_per_c"]
        assert v["C_gates"] == C and list(per) == [str(k) for k in g["k"]] and not differences(
            per, formalism.fresh_per_c(m))
        assert [(cell["k"], cell["c"]) for cell in v["cells"]] == [
            (k, c) for k in g["k"] for c in dict.fromkeys((policy.CONSOLE[0], own[k]))]
        for cell in v["cells"]:
            term, why, _ = formalism.he_products(m, cell["c"], cell["k"])
            _, old_why, got = formalism.fresh_products(m, cell["c"], cell["k"])
            assert (cell["term"], cell["fresh_old"], cell["fresh_no_cell"], cell["why"]) == (
                term, got.get("fresh_cell"), got.get("fresh_no_cell"), why)
            entry = per[str(cell["k"])]
            assert why is None and term == pytest.approx(cell["c"] * entry[1], rel=1e-12)
            if entry[0] is None:
                assert old_why == formalism.FRESH_VOID and got["void"]["kept_macs_share"] is None
            else:
                assert old_why is None and got["fresh_cell"] == pytest.approx(cell["c"] * entry[0], rel=1e-12)
        at_console = [cell for cell in v["cells"] if cell["c"] == policy.CONSOLE[0]]
        for key, pinned in (("term", want), ("fresh_old", old), ("fresh_no_cell", no_cell)):
            assert [None if x[key] is None else round(x[key]) for x in at_console] == pinned[v["name"]], key
    dense = json.loads((rho_vectors.FIXTURES / rho_vectors.DENSE_45M).read_text())
    grid = json.loads((rho_vectors.GRID_FIXTURES / rho_vectors.GRID_STEPS["the 45M start's dense step"][0]).read_text())
    assert {x: v for x, v in grid.items() if x != "at"} == {x: v for x, v in dense.items() if x != "at"}
    assert grid["at"]["145568"] == dense["at"]["145568"]


@pytest.mark.slow
def test_the_grid_fixtures_are_each_steps_product_rows_at_every_k_of_the_grid():
    """Each step of `GRID_STEPS`, captured again, gives its fixture: its product rows at every k of the grid
    (`grid_rows`). Only a capture classifies rows at a k, and a trace keeps them at the dial's k alone, so no cheaper
    test can check the rows the grid vectors read (about 35 s on a CPU, s-t20's full step most of it)."""
    from te_core import io_bound

    for fixture, args in rho_vectors.GRID_STEPS.values():
        got = json.loads(json.dumps(rho_vectors.grid_rows(io_bound.capture("dense", **args))))
        assert got == json.loads((rho_vectors.GRID_FIXTURES / fixture).read_text()), (
            f"{fixture} is stale; run `python -m te_core.rho_vectors --fixtures --write` from benchmarks/covert_witness")


def test_the_published_vectors_take_each_case_of_the_rule(built):
    """The console's port of `published` tests against these: each vector is `published` under reading A on its
    case's `measured`, and together they take every case of the rule: Fit; the max above 1, partial where a kind is
    unmeasured and not where none is; None where a kind is unmeasured and no term exceeds 1; 1, binding none, where
    every kind is computed and none exceeds 1. The products term never voids: on `VOID_CASE` (the dense case with the
    DoRA step's products, whose old figure is void) it is a measured 0, so the row binds on the token kind at the
    console's c and reads 1 at one token a block, its old figure, the best subset's, beside it at 2,510 and 0.023."""
    seen = set()
    assert record(built, rho_vectors.VOID_CASE[0]) == rho_vectors.void_case(record(built, "dense"))
    for v in built["published"]:
        m = record(built, v["case"])
        one = rho_vectors.finite(formalism.published(m, v["c"], v["k"], "A"))
        assert not differences({x: v[x] for x in one}, one)
        terms = {kind: num(t) for kind, t in v["terms"].items() if t is not None}
        off = set(v["terms"]) - set(terms)
        if v["binding_kind"] == "fit":
            seen.add("fit")
        elif v["binding_kind"] in formalism.KINDS:
            assert v["rho"] == max(terms.values()) == terms[v["binding_kind"]] > 1 and v["partial"] == bool(off)
            seen.add(("max", v["partial"], v["void"] is not None))
        elif v["binding_kind"] == "unmeasured":
            assert v["rho"] is None and off and max(terms.values()) <= 1 and v["why"]
            seen.add("unmeasured")
        else:
            assert (v["binding_kind"], v["rho"], v["rests_on"]) == ("none", 1.0, "no kind exceeds 1") and not off
            assert max(terms.values()) <= 1
            seen.add(("none", v["void"] is not None))
        assert v["void"] is None
        assert v["rests_on"] == formalism.RESTS_ON_CELL["unmeasured" if v["rho"] is None else v["binding_kind"]]
        assert v["label"] == f"{formalism.THE_CODE} {VERSION}, reading A"
    assert seen == {"fit", ("max", True, False), ("max", False, False), "unmeasured", ("none", False)}
    void = [v for v in built["published"] if v["case"] == rho_vectors.VOID_CASE[0]]
    assert [(round(v["rho"]), v["binding_kind"]) for v in void] == [(263, "token"), (1, "none")]
    assert all(v["terms"]["fresh_products"] == 0 and v["why"] is None for v in void)
    assert [round(v["sensitivities"]["fresh_old"], 3) for v in void] == [2510.258, 0.023]


def test_the_session_vectors_are_their_rule_at_every_stated_size(built):
    """The console's port of session_gates tests against these: each vector's outputs are its rule on its inputs, every
    size `SHAPES` states and every session of `SESSIONS` has one, and the console's c is the 45M one at 4,096 in and
    1,024 out."""
    s = built["sessions"]
    assert {(v["params"], v["in_tokens"], v["out_tokens"]) for v in s["vectors"]} == {
        (p, i, o) for p in policy.SHAPES for i, o in rho_vectors.SESSIONS}
    for v in s["vectors"]:
        sh, i, o = v["shape"], v["in_tokens"], v["out_tokens"]
        d, n, f, voc = sh["d_model"], sh["n_layer"], sh["d_ff"], sh["vocab"]
        assert sh == policy.SHAPES[v["params"]] and policy.parameters(sh) == v["params"]
        attended = i * i + o * i + o * (o + 1) // 2
        macs = {"matrices": (i + o) * n * (4 * d * d + 3 * d * f), "lm_head": (i + o) * d * voc,
                "attention": 2 * n * d * attended}
        assert (v["tokens"], v["attended"], v["macs"]) == (i + o, attended, macs)
        assert v["gates_per_mac"] == 1480.125 and v["gates"] == sum(macs.values()) * 1480.125
        assert v["k_bits"] == i * math.ceil(math.log2(voc))
    console = next(v for v in s["vectors"] if (v["params"], v["in_tokens"], v["out_tokens"]) == (45_029_888, 4096, 1024))
    assert s["console"] == {"params": 45_029_888, "in_tokens": 4096, "out_tokens": 1024, "c": console["gates"],
                            "k": float(console["k_bits"])}
    assert (s["console"]["c"], s["console"]["k"]) == policy.CONSOLE


def test_each_reference_size_and_start_has_its_session_and_the_products_terms_inputs_at_every_k_of_the_grid(built):
    """The console's dials move its reference model over its sizes, 20M to 1B at vocab 50,304, and k over its grid.
    Each size and each start (the ladder's six among them) has its shape, the c of its session at the console's 4,096
    in and 1,024 out as session_gates prices it, and that session's k; and at each k of the grid the c of a session of
    k/16 input tokens and 1,024 output, and that session's input ids: k at 50,304, and 15/16 of it at bpe32k's 32,000,
    which puts the ladder's starts' own k off the grid."""
    s = built["sessions"]
    assert s["dial"] | {"about": None} == {"about": None, "k": list(policy.K_GRID),
                                           "in_tokens": list(policy.DIAL_IN_TOKENS), "out_tokens": 1024}
    assert [e["name"] for e in s["sizes"]] == list(policy.REFERENCE_SIZES) == ["20M", "45M", "95M", "200M", "400M",
                                                                               "1B"]
    starts = policy.starts()
    assert [e["name"] for e in s["starts"]] == list(starts)
    assert {"s-t20", "s-t200", "m-t20", "m-t200", "l16-t20", "xl-t20"} <= set(starts)
    for e in s["sizes"] + s["starts"]:
        p, sh = e["params"], e["shape"]
        assert sh == policy.SHAPES[p] and policy.parameters(sh) == p and sh["d_model"] == 64 * sh["n_head"]
        bits = math.ceil(math.log2(sh["vocab"]))
        assert (e["c"], e["k_bits"]) == (policy.session_gates(p, 4096, 1024), 4096 * bits)
        assert [(x["k"], x["in_tokens"]) for x in e["dial"]] == list(zip(policy.K_GRID, policy.DIAL_IN_TOKENS))
        for x in e["dial"]:
            assert x["c"] == policy.session_gates(p, x["in_tokens"], 1024) and x["k"] == 16 * x["in_tokens"]
            assert x["k_bits"] == x["in_tokens"] * bits == x["k"] * bits / 16
    sizes = {e["name"]: e for e in s["sizes"]}
    assert (sizes["45M"]["c"], sizes["45M"]["k_bits"]) == policy.CONSOLE
    assert {e["name"]: round(e["c"] / 1e12) for e in s["sizes"]} == {
        "20M": 242, "45M": 537, "95M": 1113, "200M": 2340, "400M": 4318, "1B": 10322}
    for e in s["starts"]:
        shape = starts[e["name"]]
        assert (e["params"], e["shape"]["d_model"], e["shape"]["n_layer"], e["shape"]["vocab"]) == (
            shape["parameters"], shape["d_model"], shape["n_layer"], shape["vocab"])
    st20 = next(e for e in s["starts"] if e["name"] == "s-t20")
    assert st20["k_bits"] == 61_440 and [x["k_bits"] for x in st20["dial"]] == [15 * n for n in policy.DIAL_IN_TOKENS]


def test_each_reference_size_is_the_familys_shape_near_it_else_the_nearest_lf_shape_in_the_familys_form():
    """The console's sizes at vocab 50,304: 45M, 95M and 200M are the lf family's s, m and l (model.py), each within 5%
    of its size, s and m the 45M and 95M starts' own shapes; the family (model.py's sizes and the ladder's l16 and xl)
    has no shape within 5% of 20M, 400M or 1B, so each is the lf shape nearest it in the family's form: heads of 64,
    the SwiGLU width the multiple of 64 nearest 2.75 d (halves up; every family shape's but m's, 8/3 d), and
    d/n_layer from l16's 64 to m's 96."""
    import model

    v = 50_304

    def shape(d, n, h, f):
        return {"d_model": d, "n_layer": n, "n_head": h, "d_ff": f, "vocab": v}

    family = {(s.d_model, s.n_layer, s.n_head, s.d_ff) for s in model.SIZES.values()} | {
        (sh["d_model"], sh["n_layer"], sh["n_head"], sh["d_ff"]) for sh in policy.SHAPES.values() if sh["vocab"] == 32_000}
    form = [shape(d, n, d // 64, 64 * math.floor(2.75 * d / 64 + 0.5))
            for d in range(64, 4097, 64) for n in range(1, 65) if 64 <= d / n <= 96]
    assert all(64 * math.floor(2.75 * d / 64 + 0.5) == f for d, _, _, f in family if (d, f) != (768, 2048))
    sizes = {"20M": 20e6, "45M": 45e6, "95M": 95e6, "200M": 200e6, "400M": 400e6, "1B": 1e9}
    assert list(sizes) == list(policy.REFERENCE_SIZES)
    named = {"45M": model.SIZES["s"], "95M": model.SIZES["m"], "200M": model.SIZES["l"]}
    for name, n in sizes.items():
        sh = policy.SHAPES[policy.REFERENCE_SIZES[name]]
        assert policy.parameters(sh) == policy.REFERENCE_SIZES[name] and sh["vocab"] == v
        near = [shape(*f) for f in family if abs(policy.parameters(shape(*f)) / n - 1) <= 0.05]
        if name in named:
            s = named[name]
            assert near == [sh] == [shape(s.d_model, s.n_layer, s.n_head, s.d_ff)], name
        else:
            best, second = sorted(form, key=lambda x: abs(policy.parameters(x) - n))[:2]
            assert not near and sh == best and abs(policy.parameters(best) - n) < abs(policy.parameters(second) - n)
    assert (policy.REFERENCE_SIZES["45M"], policy.REFERENCE_SIZES["95M"]) == (policy.REFERENCE_PARAMS, 95_269_632)
    assert {policy.starts()[x]["parameters"] for x in ("s50", "m50")} == {45_029_888, 95_269_632}


def test_the_tiny_starts_forward_is_the_counters():
    import model

    net = model.LM(model.SIZES["tiny"])
    assert forward_macs_per_token(net, model.CTX, model.VOCAB, "cpu") == rho_vectors.TINY["forward_macs_per_token"]
    assert (model.VOCAB, model.SIZES["tiny"].d_model) == (rho_vectors.TINY["vocab"], rho_vectors.TINY["d_model"])
