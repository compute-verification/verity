"""`inference_only.py` on circuits' slot-grouped wired Program (`population.wired_population_program` with ``slots``): its
blocks are the `WiredSlot_v1` Definitions, so a wire inside a slot is not fresh and a wire between slots is; a weight is
held at every call site; a value a slot reads cut to its leaves counts at those leaves; fresh bits of no kind are never
folded into a kind; a record that doesn't build wired is refused by name; and `BitOf`/`WordOf`, priced at no gates, are
each the identity on the wires they select."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))

import inference_only as IO
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.population import bit_of, word_of
from verity.circuits.ops.boolean import versions as BL
from verity.circuits.program import is_input
from verity.circuits.types import Array, Value
from verity.commitments.periods import interface as IF
from verity_vllm.program.predict import block_counts as BC
from verity_vllm.program.predict.scan_state import SCAN_STATE

V16, BIT = Value(16), Value(1)
Add = PrimitiveDefinition("TSlotAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)
Mul = PrimitiveDefinition("TSlotMul16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a * b + 1) & 0xFFFF, register=False)
Xor = PrimitiveDefinition("TSlotXor1", 1, (("a", BIT), ("b", BIT)), BIT, lambda a, b: a ^ b, register=False)


def _pair(name, prim, ports):
    return bind(CompositeDefinition(name, 1, (), lambda S: (tuple((p, Array(2, V16)) for p in ports), Array(2, V16)),
                                    lambda B, S, a, b: B.batch(prim, a, b, axes=(0, 0)), register=False))


A, B = _pair("TSlotA", Add, ("x", "k")), _pair("TSlotB", Mul, ("y", "w"))
D = bind(CompositeDefinition("TSlotBits", 1, (), lambda S: ((("b", Array(32, BIT)), ("s", Array(32, BIT))), Array(32, BIT)),
                             lambda B_, S, b, s: B_.batch(Xor, b, s, axes=(0, 0)), register=False))
IA, IB, ID = descriptor_id(A), descriptor_id(B), descriptor_id(D)
OURS = {IA: A, IB: B, ID: D}


def _slotted(cross: bool) -> dict:
    """circuits' fixture, ``s`` the scan state row: four units each of A, D and B, unit u in slot u % 2, D's bits from
    A's words inside each slot and B's words from D's bits; with ``cross`` B unit 1 reads D unit 0, in the other slot."""
    def wire(reader, port, writer, at, n, base, step=1):
        return {"reader": reader, "port": port, "writer": writer, "out": "o0", "at": at, "radix": [n], "base": base,
                "step": [step]}
    ys = [wire(IB, "y", ID, 0, 1, 0), wire(IB, "y", ID, 1, 1, 0), wire(IB, "y", ID, 2, 2, 2)] if cross else \
        [wire(IB, "y", ID, 0, 4, 0)]
    return {"templates": [[IA, 4], [ID, 4], [IB, 4]], "wires": [wire(ID, "b", IA, 0, 4, 0), *ys],
            "external": [f"{IA}/x", f"{IA}/k"], "partition": "",
            "registered": {"params": {SCAN_STATE: 8, "weights/w": 2}, "tables": [], "roots": [], "openings": [], "reads": [
                {"reader": ID, "port": "s", "param": SCAN_STATE, "at": 0, "radix": [4], "base": 0, "step": [2]},
                {"reader": IB, "port": "w", "param": "weights/w", "at": 0, "radix": [4], "base": 0, "step": [0]}]},
            "blocks": IF.Blocks(slots=2, steps=2, inference=tuple(
                IF.InferenceEntry(t, 0, (2, 2), ("step", "slot"), 0) for t in (IA, ID, IB))).doc(12, "")}


@pytest.fixture
def ours(monkeypatch):
    """The fixture's templates resolved and its primitives priced one gate each, as the registry and the gate table
    don't know them; the wiring between words and bits is priced by `inference_only.price`."""
    monkeypatch.setattr(BC, "resolve", lambda tid: OURS[tid])
    ones = {descriptor_id(p) for p in (Add, Mul, Xor)}

    def gate_table(fns):
        seen, todo = set(), list(fns)
        while todo:
            fn = todo.pop()
            if id(fn) not in seen:
                seen.add(id(fn))
                todo.extend(() if isinstance(fn, PrimitiveDefinition) else (nd.fn for nd in fn.body.nodes))
                if isinstance(fn, PrimitiveDefinition) and not is_input(fn):
                    keys.add(descriptor_id(fn))
        return {key: 1 for key in keys & ones}, sorted(keys - ones)

    keys: set = set()
    monkeypatch.setattr(BL, "gate_table", gate_table)


PINS = {"weights/w": "00"}


@pytest.mark.parametrize(("cross", "fresh"), [(False, [192]), (True, [192, 224])])
def test_a_wire_inside_a_slot_is_not_fresh_and_one_between_slots_is(ours, cross, fresh):
    """Two slots, each one block of a `WiredSlot_v1` Definition (one shared Definition, or two where a wire crosses):
    with the weight held, a slot's fresh bits are its external words and its own scan state leaves, 192, and the slot
    reading a wire from the other slot counts those 32 bits too; the wires inside a slot count nothing."""
    row = IO.wired_row(_slotted(cross), PINS, PINS)
    assert all(key.startswith("WiredSlot_v1{") for key in row["blocks"]) and len(row["blocks"]) == len(fresh)
    assert [x for _, _, x in row["rows"]] == fresh and row["slots"] == 2 and row["fresh"] == max(fresh)
    assert row["held"] == ["weights/w"] and sum(row["calls"].values()) == 2


@pytest.mark.parametrize("cross", [False, True])
def test_a_weight_left_unregistered_is_fresh_at_every_call_site(ours, cross):
    """With the weight's pin missing it is read fresh by every slot: each row grows by its 32 bits."""
    held, bare = IO.wired_row(_slotted(cross), PINS, PINS), IO.wired_row(_slotted(cross), {}, {})
    assert [x + 32 for _, _, x in held["rows"]] == [x for _, _, x in bare["rows"]] and bare["held"] == []


def test_the_scan_state_counts_at_a_slots_leaves_and_bits_of_no_kind_are_never_folded_in(ours, monkeypatch):
    """The scan state row, cut to each slot's leaves, drops a slot's fresh bits by its 64; the external words left are
    of no kind, so the split doesn't sum to k and the policy is refused rather than given a kind that hides them."""
    row = IO.wired_row(_slotted(False), PINS, PINS)
    assert row["k_by_kind"] == {"linears": 0, "scan_state": 64, "openings": 0} and row["fresh"] == 192
    doc = _slotted(False)
    monkeypatch.setattr(IO, "class_doc", lambda serve, cls, scheme=None: (doc, ["weights/w", SCAN_STATE]))
    with pytest.raises(ValueError, match="a fresh bit of no kind of linears, scan_state, openings"):
        IO.policy("2,2,2", "verity-int8/Llama-3.2-1B-INT8", "rev", 60, IO.UTILIZATION, "s", "t", None)


def test_a_record_that_doesnt_build_wired_is_refused_by_name(monkeypatch):
    """A wire between two word widths (int8's 16-to-32-bit rows) has no wired Program: the count is refused, saying so,
    and never falls back to a Program without its wires."""
    Neg = PrimitiveDefinition("TSlotNeg32", 1, (("a", Value(32)),), Value(32), lambda a: -a & 0xFFFFFFFF, register=False)
    G = bind(CompositeDefinition("TSlotWide", 1, (), lambda S: ((("z", Array(1, Value(32))),), Array(1, Value(32))),
                                 lambda B_, S, z: B_.batch(Neg, z, axes=(0,)), register=False))
    monkeypatch.setattr(BC, "resolve", lambda tid: {IA: A, descriptor_id(G): G}[tid])
    rec = {"templates": [[IA, 1], [descriptor_id(G), 1]], "external": [f"{IA}/x", f"{IA}/k"], "partition": "",
           "wires": [{"reader": descriptor_id(G), "port": "z", "writer": IA, "out": "o0", "at": 0, "radix": [1], "base": 0,
                      "step": [0]}]}
    with pytest.raises(IO.Undecided, match="the class's circuit doesn't build wired .* only between W-bit words"):
        IO.wired_program(rec)


@pytest.mark.parametrize("w", [16, 32])
def test_bit_of_and_word_of_are_priced_at_no_gates_as_the_identity_on_the_wires_they_select(w):
    """At the widths class 4,27,2 reads, `BitOf{W}x{I}` is the input word's leaf I and `WordOf{W}` the word whose leaf i
    is input i, on random words: a circuit with no gate, which is the price. A Boolean version in `gate_table` would
    replace it."""
    rng = random.Random(w)
    for _ in range(64):
        x = rng.getrandbits(w)
        bits = [bit_of(w, i).evaluate(x) for i in range(w)]
        assert bits == [(x >> i) & 1 for i in range(w)] and word_of(w).evaluate(*bits) == x
    assert {IO.price(f"BitOf{w}x{i}_v1")["gates"] for i in range(w)} == {0} == {IO.price(f"WordOf{w}_v1")["gates"]}
    assert IO.price(f"BitOf{w}x{w}_v2") is None and IO.price("WordOfx_v1") is None


#: a two-layer Llama small enough to lower the class circuit in about a minute
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu", "eos_token_id": 255}


@pytest.mark.slow
def test_the_compiled_count_on_a_tiny_class_grouped_by_slot(tmp_path):
    """circuits' builder on a tiny class's record (two slots), counted by the compiled `inference-only-check --count`:
    its blocks are the two slot calls of one `WiredSlot_v1` Definition whose gates are the table's, every weight held;
    with one weight's pin missing it refuses on fresh bits. It takes about a minute and a half (the lowering and two
    counts); it catches what only the built executable and a served class's own record show, since the Python reference
    doesn't finish a served class's count and the hand fixture has no table, opening or conversion."""
    import served_cases as SC
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit

    exe = SC.build(SC.INFERENCE_ONLY_LEAN[0], "inference-only-verifier")
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
    circuit = class_circuit(Registered.of(phi, default_seed=7), WindowClass(2, 4, 2), None,
                            Grain(gemm_words=1024, rows=256, select=64), provable=False)
    doc = {**circuit.description(), "name": circuit.name, "partition": "",
           "blocks": circuit.blocks.doc(sum(circuit.units.values()), "")}
    pins = IO.weight_pins(sorted(circuit.registered.params))
    w = IO.wired_blocks(doc, pins, pins)
    assert len(w["blocks"]) == 1 and w["calls"] == {w["blocks"][0]: 2} and w["held"]
    rows = IO.wired_rows(w["fn"], w["blocks"], w["table"], w["held"], exe)
    (g, k, m), = rows.values()
    assert (g, m) == (w["gates"], 2) and k > 0
    one = IO.wired_instance(w["fn"], w["blocks"], w["table"], g, k, 2, w["held"][1:])
    got = IO.lean_count(exe, one, tmp_path)["out"]
    assert got["complies"] is False and got["failure"] == [0, "fresh"]
