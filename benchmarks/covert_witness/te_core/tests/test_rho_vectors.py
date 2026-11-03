"""te_core.rho_vectors: the one bound's published test vectors are what te_core.formalism makes from the capture's
fixtures, their inputs reproduce through `from_json` (the console's route), and they say what reading A says of
wildcard's grown adapter, a dense step and a LoRA step on the tiny start (the coordinator's 06:15Z item 4)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from te_core import VERSION, formalism, rho_vectors
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


@pytest.fixture(scope="module")
def built() -> dict:
    return rho_vectors.build()


def by(built: dict, case: str, reading: str, n: int, k: int) -> dict:
    return next(v for v in built["vectors"] if (v["case"], v["reading"], v["tokens_per_block"], v["k"]) ==
                (case, reading, n, k))


def test_the_published_vectors_are_what_the_module_makes(built):
    have = json.loads(rho_vectors.OUT.read_text())
    diff = differences(have, json.loads(json.dumps(built)))
    assert not diff, ("te_core/rho_vectors.json is stale; run `python -m te_core.rho_vectors --write` from "
                      "benchmarks/covert_witness:\n" + "\n".join(diff[:10]))


def test_each_case_reproduces_through_from_json_and_each_vector_is_its_bound(built):
    """The console reads a case's `measured` (from_json's form) and a vector's c, k and reading; `bound` and `at` there
    give the vector's terms, max and ρ."""
    for name, case in rho_vectors.CASES.items():
        m = rho_vectors.case_measured(case)
        assert formalism.from_json(json.loads(json.dumps(asdict(m)))) == m
    for v in built["vectors"][::7]:
        m = formalism.from_json(built["measured"][v["case"]])
        b, a = formalism.bound(m, v["c"], v["k"], v["reading"]), formalism.at(m, v["c"], v["k"], v["reading"])
        assert not differences(rho_vectors.finite([b["terms"], b["max"], a["rho"]]), [v["terms"], v["max"], v["rho"]])


def test_every_rho_is_labelled_and_is_rho_fit_where_fit_holds_else_the_bound(built):
    assert built["version"] == VERSION and built["kinds"] == list(formalism.KINDS)
    for v in built["vectors"]:
        assert v["label"] == f"te_core {VERSION}, reading {v['reading']}"
        terms = [math.inf if t is None else t for t in v["terms"].values()]
        top = max(terms)
        assert (math.inf if v["max"] is None else v["max"]) == top and v["terms"][v["argmax"]] == v["max"]
        if v["fit"]:
            assert v["rho"] == v["rho_fit"] and top <= v["rho_fit"]
        else:
            assert (math.inf if v["rho"] is None else v["rho"]) == max(1.0, top)


def test_a_dense_step_and_a_lora_step_are_unbounded_at_the_console_default_under_reading_a(built):
    """Every block tensor changes, so the keys and values at both attention layers depend on θ': a block serving a
    token at position p there reads the earlier positions' K/V or computes their state, about a = 2,048 bits each,
    so it serves about k/a positions, and the default k = 80,000 bits leaves the later ones in no block (∞). With
    k = 8 Mbit (dense) or 800 kbit (LoRA, whose update is 10 times smaller) every gate is in some block, at gates' cap:
    the term is the affected share. Reading B's token form is finite at the default."""
    default = rho_vectors.CELLS[0]
    for case, k in (("dense", 8_000_000), ("lora", 800_000)):
        a = by(built, case, "A", *default)
        assert a["terms"]["token"] is a["terms"]["decoding"] is a["rho"] is None
        assert by(built, case, "B", *default)["terms"]["token"] > 1
        at = by(built, case, "A", default[0], k)
        assert at["terms"]["token"] == pytest.approx(at["inputs"]["token"]["share"])


def test_the_grown_adapter_is_bounded_under_reading_a_only_with_its_live_update_code(built):
    """wildcard's 05:46Z LOOPHOLE (a = 32 only with w_down's live update bits read): its work at the grown block reads
    w_down's whole bf16 cast, 196,608 bits > k, so with Δ the trace's cut no block holds it (∞); an E of the live
    entries alone (26,624 bits) lets a block read the code instead, and the token kind is then the affected share
    (gates bind). The reduction kind binds there under (R): 2,048 tokens fill two blocks, and one 1.18 Mbit partial is
    read at k a block."""
    default = rho_vectors.CELLS[0]
    full, live = by(built, "adapter", "A", *default), by(built, "adapter-live", "A", *default)
    assert full["terms"]["token"] is None
    assert live["terms"]["token"] == pytest.approx(live["inputs"]["token"]["share"])
    assert live["argmax"] == "reduction" and live["inputs"]["reduction"]["blocks"] == [2]
    assert live["rho"] == pytest.approx(live["terms"]["reduction"]) and live["rho"] > 9


def test_the_published_rho_takes_only_the_kinds_valid_on_their_own(built):
    """The coordinator's 08:26Z ruling: the published ρ's max takes the token kind under reading A and decoding, the
    token kind unmeasured until the packer proposes product tiles and an infinite packer term unmeasured, never ∞;
    reduction and update are sensitivities beside it. At the console default the dense and LoRA steps (∞ under 4.4's
    bound) and the grown adapter (∞ with its traced cut, 9.78 by reduction with its live code) publish nothing binding,
    with reduction's term beside it: ρ 1 where decoding is measured (the live adapter's 0.65), and null, resting on
    `unmeasured`, where no
    valid kind is (never 1, coordinator 09:46Z); a row without entries is reading B's, te_core 4.3's ρ, a target, and
    until `READING_A` (the coordinator's 09:02Z: until 4.5's packer) so is every row."""
    default = rho_vectors.CELLS[0]
    seen = set()
    for case in rho_vectors.CASES:
        v = by(built, case, "A", *default)
        m = formalism.from_json(built["measured"][case])
        one = formalism.published(m, v["c"], v["k"], "A")
        assert formalism.published(m, v["c"], v["k"])["reading"] == ("A" if formalism.READING_A else "B")
        assert one["terms"]["token"] is None and one["terms"]["fresh_products"] is None
        assert one["terms"]["decoding"] == (v["terms"]["decoding"] and pytest.approx(v["terms"]["decoding"]))
        measured = one["terms"]["decoding"] is not None
        seen.add(measured)
        assert (one["reading"], one["rho"], one["binding_kind"]) == ("A", 1.0 if measured else None, "none")
        assert not v["fit"]
        assert one["sensitivities"] == pytest.approx({kind: v["terms"][kind] for kind in formalism.SENSITIVITIES})
        assert one["rests_on"] == formalism.RESTS_ON["none" if measured else "unmeasured"]
    assert False in seen
    live = formalism.published(formalism.from_json(built["measured"]["adapter-live"]), v["c"], v["k"], "A")
    assert live["sensitivities"]["reduction"] > 9
    m = formalism.from_json(built["measured"]["dense"] | {"entries": None})
    b = formalism.published(m, v["c"], v["k"])
    assert (b["reading"], b["binding_kind"], b["rho"]) == ("B", "target", formalism.at(m, v["c"], v["k"])["rho"])
    assert b["terms"]["token"] == pytest.approx(by(built, "dense", "B", *default)["terms"]["token"])


def test_the_tiny_starts_forward_is_the_counters():
    import model

    net = model.LM(model.SIZES["tiny"])
    assert forward_macs_per_token(net, model.CTX, model.VOCAB, "cpu") == rho_vectors.TINY["forward_macs_per_token"]
    assert (model.VOCAB, model.SIZES["tiny"].d_model) == (rho_vectors.TINY["vocab"], rho_vectors.TINY["d_model"])
