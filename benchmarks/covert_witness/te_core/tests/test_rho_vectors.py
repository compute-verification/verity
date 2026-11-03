"""te_core.rho_vectors: the one bound's published test vectors are what te_core.formalism makes from the capture's
fixtures, their inputs reproduce through `from_json` (the console's route), and they say what reading A, with its
matrix-product tiles, says of wildcard's grown adapter, a dense step and a LoRA step on the tiny start."""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from te_core import VERSION, formalism, rho_vectors, tiles
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
    or computes their state, and at the default k = 80,000 bits no run block holds it; a tile of the step's products
    still does, so the token term is finite (4.4's runs alone gave ∞, the full-token-state overclaim), above 1, and
    binds: ρ is it. With k = 8 Mbit (dense) or 800 kbit (LoRA, whose update is 10 times smaller) every gate is in some
    block at gates' cap: the term is the affected share. Fresh products are unmeasured, and nearly every token gate
    is in a tile."""
    default = rho_vectors.CELLS[0]
    for case, k in (("dense", 8_000_000), ("lora", 800_000)):
        a = by(built, case, *default)
        assert 1 < a["terms"]["token"] < math.inf and a["binding"] == "token" and a["rho"] == a["terms"]["token"]
        assert a["terms"]["fresh_products"] is None and "fresh_products" in a["unmeasured"]
        assert a["inputs"]["token"]["unplaced"] < 0.01
        at = by(built, case, default[0], k)
        assert at["terms"]["token"] == pytest.approx(at["inputs"]["token"]["share"])


def test_the_grown_adapter_is_bounded_by_its_tiles_and_by_gates_with_its_live_update_code(built):
    """wildcard's grown adapter (a = 32 only with w_down's live update bits read): its work at the grown block reads
    w_down's whole bf16 cast, 196,608 bits > k, so with Δ the trace's cut no run block holds it, and its tiles bound
    it, above 1. An E of the live entries alone (26,624 bits) lets a block read the code instead, and the token kind
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


def test_the_tiny_starts_forward_is_the_counters():
    import model

    net = model.LM(model.SIZES["tiny"])
    assert forward_macs_per_token(net, model.CTX, model.VOCAB, "cpu") == rho_vectors.TINY["forward_macs_per_token"]
    assert (model.VOCAB, model.SIZES["tiny"].d_model) == (rho_vectors.TINY["vocab"], rho_vectors.TINY["d_model"])
