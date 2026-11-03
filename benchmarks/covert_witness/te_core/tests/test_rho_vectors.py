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
        assert v["label"] == f"te_core {VERSION}, reading A"
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
    the affected share. Fresh products are unmeasured, and nearly every token gate is in a tile."""
    default = rho_vectors.CELLS[0]
    for case, k in (("dense", 8_000_000), ("lora", 800_000)):
        a = by(built, case, *default)
        assert 1 < a["terms"]["token"] < math.inf and a["binding"] == "token" and a["rho"] == a["terms"]["token"]
        assert a["terms"]["token"] < a["sensitivities"]["per_gate"]
        assert a["terms"]["fresh_products"] is None and "fresh_products" in a["unmeasured"]
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
    dial's default the dense, LoRA and grown-adapter steps bind on the token kind; the adapter with its live code
    publishes 1, no kind above 1, labelled with the measured kind its max takes (coordinator 11:18Z), its reduction
    sensitivity above 9 beside it. A row without entries or an op log
    publishes reading B, Slowdown's token form, a target, and so does every row where `READING_A` is off (on since
    the coordinator's 11:18Z ruling); read under A, such a row measures no kind, so it publishes None,
    binding none and resting on `unmeasured`, never 1 (coordinator 09:46Z)."""
    default = rho_vectors.CELLS[0]
    for case in rho_vectors.CASES:
        v = by(built, case, *default)
        m = formalism.from_json(built["measured"][case])
        one = formalism.published(m, v["c"], v["k"], "A")
        assert formalism.published(m, v["c"], v["k"])["reading"] == ("A" if formalism.READING_A else "B")
        assert one["reading"] == "A" and not v["fit"]
        assert not differences(rho_vectors.finite([one["rho"], one["terms"], one["sensitivities"]]),
                               [v["rho"], v["terms"], {n: v["sensitivities"][n] for n in formalism.CELL_SENSITIVITIES}])
        assert one["binding_kind"] == v["binding"] and one["binding_kind"] in formalism.KINDS
        assert one["rests_on"] == formalism.RESTS_ON_CELL[one["binding_kind"]]
        assert case == "adapter-live" or one["binding_kind"] == "token"
    live = formalism.published(formalism.from_json(built["measured"]["adapter-live"]), v["c"], v["k"], "A")
    assert live["rho"] == 1.0 and num(live["terms"][live["binding_kind"]]) <= 1
    assert live["sensitivities"]["reduction"] > 9
    for gone in ("entries", "ops"):
        m = formalism.from_json(built["measured"]["dense"] | {gone: None})
        b, at = formalism.published(m, v["c"], v["k"]), formalism.at(m, v["c"], v["k"])
        assert (b["reading"], b["binding_kind"], b["rho"]) == ("B", "target", at["rho"])
        assert b["terms"] == {"token_form": at["slowdown"]} and list(b["sensitivities"]) == ["reduction", "update"]
        assert b["rests_on"] == formalism.RESTS_ON_CELL["target"]
        a = formalism.published(m, v["c"], v["k"], "A")
        assert (a["reading"], a["rho"], a["binding_kind"], a["rests_on"]) == ("A", None, "none", "unmeasured")
        assert set(a["terms"]) == set(formalism.KINDS) and all(t is None for t in a["terms"].values())
    m = formalism.from_json(built["measured"]["dense"])
    assert formalism.published(m) == formalism.published(m, *policy.CONSOLE)


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
