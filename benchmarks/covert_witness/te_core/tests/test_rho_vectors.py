"""te_core.rho_vectors: the one bound's published test vectors are what te_core.formalism makes from the capture's
fixtures, their inputs reproduce through `from_json` (the console's route), and they say what reading A, with its
matrix-product tiles, says of wildcard's grown adapter, a dense step and a LoRA step on the tiny start; and the
console's c, te_core.policy's session at every stated size, is its rule."""
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
    the affected share. Fresh products are a measured 0 at these k (the fixtures' products are classified at each k
    of the console's dial, and none is fresh), and nearly every token gate is in a tile."""
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
    fresh products are a measured 0 at each k of the dial); the adapter with its live code has no kind above 1, so it
    publishes 1, binding none, its reduction sensitivity above 9 beside it, and off the dial (`OFF_DIAL`), where its
    fresh products are unmeasured, None, saying so (coordinator 12:20Z). A row without entries or an op log
    publishes reading B, Slowdown's token form, a target, and so does every row where `READING_A` is off (on since
    the coordinator's 11:18Z ruling); read under A, such a row measures only the fresh-products kind (a 0 here), so
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


def test_the_fresh_vectors_are_io_bounds_figure_without_p_cell_with_the_whole_cell_figure_beside_it(built):
    """The console's port of the fresh-products term tests against these: on each of io_bound's hand-written product
    sets the term is io_bound's own vector's proved figure without P-cell and fresh_cell its figure with it, summed in
    the reference circuit's chain whether the order is declared or not, with what its best subset dropped, and where no
    subset has a proved Λ unmeasured (`FRESH_VOID`), never 0; a circuit that declares a tree is unmeasured; the 45M
    start's dense step's term reads its whole-making system, attention's 64-long chains below every counted operand:
    163,341, its fresh_cell 2,869,161, on all its products with G without each sequence's first position (163,708 and
    2,870,251 before that trim; 4.11's one-class rule gave 165,021 and 3,041,057)."""
    f = built["fresh"]
    assert (f["p_cell"], f["order"]) == (formalism.P_CELL, formalism.ORDER) == (False, "chain")
    assert f["rests_on"] == formalism.RESTS_ON_CELL["fresh_products"] and f"without assuming that {formalism.P_CELL_WORDS}" in f["rests_on"]
    vs = {v["name"]: v for v in f["vectors"]}
    from te_core import io_bound

    for want in io_bound.vectors()["vectors"]:
        v = vs[want["name"]]
        assert v["inputs"]["c"] == want["inputs"]["c"] and v["inputs"]["products"]["products"] == want["inputs"][
            "products"]
        void = want["term"]["proved_rho_no_cell"] is None
        assert (v["term"], v["fresh_cell"], v["order"], v["why"]) == (
            want["term"]["proved_rho_no_cell"], want["term"]["proved_rho"], "chain", formalism.FRESH_VOID if void else None)
        sub = want["term"]["proved_subset_no_cell"]
        assert v["subset"] == (None if sub is None or not sub["dropped"] else {
            "dropped": sub["dropped"], "of": len(want["inputs"]["products"]),
            "dropped_macs_share": sub["dropped_macs_share"], "k_whole_min": sub["k_whole_min"]})
    assert sum(v["why"] == formalism.FRESH_VOID for v in vs.values()) == 1 and sum(bool(v["subset"]) for v in vs.values())
    square = vs["one product, square, console"]
    assert vs["one product, square, console, no order declared"]["term"] == square["term"] > 1
    tree = vs["one product, square, console, a circuit that declares a tree"]
    assert (tree["term"], tree["fresh_cell"], tree["order"]) == (None, None, "tree") and "tree" in tree["why"]
    dense = vs["the 45M start's dense step, its trace's order unknown"]
    assert (dense["inputs"]["c"], dense["inputs"]["k"]) == policy.CONSOLE and dense["order"] == "chain"
    assert dense["inputs"]["products"]["trace_order"] == "unknown"
    assert round(dense["term"]) == 163_341 and round(dense["fresh_cell"]) == 2_869_161 and dense["subset"] is None
    assert dense["inputs"]["products"]["system"] is not None and dense["inputs"]["products"]["p_match"] == []


def test_the_dial_vectors_are_c_times_a_rows_fresh_per_c_with_and_without_p_cell(built):
    """The console's dial ports the fresh-products term from a row's rho_inputs (te_core.panels' `rho_columns`, which
    holds `formalism.fresh_per_c` to its digits): at each k of the dial the term is c times its entry's first number
    and fresh_cell c times its second, null where the entry is null, each cell `formalism.fresh_products` there. The
    45M dense step's term is measured at the dial's three least k and void above them; the tiny start's cases have no
    fresh product at any k."""
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
            if entry is None:
                assert (cell["term"], cell["fresh_cell"], cell["why"]) == (None, None, formalism.FRESH_VOID)
            else:
                assert cell["term"] == pytest.approx(cell["c"] * entry[0], rel=1e-12) and cell["why"] is None
                assert cell["fresh_cell"] == pytest.approx(cell["c"] * entry[1], rel=1e-12)
        cols = json.loads(panels.rho_columns(ms[v["name"]], rho_vectors.W_FWD)["rho_inputs"])[panels.FRESH]
        assert cols.keys() == per.keys() and all(
            (a is None) == (b is None) and (a is None or a == pytest.approx(b, rel=1e-5)) for a, b in
            zip(cols.values(), per.values()))
        if v["name"] in rho_vectors.CASES:
            assert all(e == [0.0, 0.0] for e in per.values())
    dense = d["vectors"][-1]["fresh_per_c"]
    assert [dense[str(k)] is None for k in DIAL_K] == [False] * 3 + [True] * 4


def test_the_published_vectors_take_each_case_of_the_rule(built):
    """The console's port of `published` tests against these: each vector is `published` under reading A on its
    case's `measured`, and together they take every case of the rule: Fit; the measured max above 1, partial where a
    kind is unmeasured and not where none is; None where a kind is unmeasured and no measured term exceeds 1; 1,
    binding none, where every kind is measured and none exceeds 1."""
    seen = set()
    for v in built["published"]:
        m = formalism.from_json(built["measured"][v["case"]])
        one = rho_vectors.finite(formalism.published(m, v["c"], v["k"], "A"))
        assert not differences({x: v[x] for x in one}, one)
        terms = {kind: num(t) for kind, t in v["terms"].items() if t is not None}
        off = set(v["terms"]) - set(terms)
        if v["binding_kind"] == "fit":
            seen.add("fit")
        elif v["binding_kind"] in formalism.KINDS:
            assert v["rho"] == max(terms.values()) == terms[v["binding_kind"]] > 1 and v["partial"] == bool(off)
            seen.add(("max", v["partial"]))
        elif v["binding_kind"] == "unmeasured":
            assert v["rho"] is None and off and max(terms.values()) <= 1 and v["why"]
            seen.add("unmeasured")
        else:
            assert (v["binding_kind"], v["rho"], v["rests_on"]) == ("none", 1.0, "no kind exceeds 1") and not off
            assert max(terms.values()) <= 1
            seen.add("none")
        assert v["rests_on"] == formalism.RESTS_ON_CELL["unmeasured" if v["rho"] is None else v["binding_kind"]]
        assert v["label"] == f"{formalism.THE_CODE} {VERSION}, reading A"
    assert seen == {"fit", ("max", True), ("max", False), "unmeasured", "none"}


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


def test_the_tiny_starts_forward_is_the_counters():
    import model

    net = model.LM(model.SIZES["tiny"])
    assert forward_macs_per_token(net, model.CTX, model.VOCAB, "cpu") == rho_vectors.TINY["forward_macs_per_token"]
    assert (model.VOCAB, model.SIZES["tiny"].d_model) == (rho_vectors.TINY["vocab"], rho_vectors.TINY["d_model"])
