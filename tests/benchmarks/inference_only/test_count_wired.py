"""The count over the registered wired C and its registered partition (`inference_only.partition_instance`), on one-circuit.

The soundness-correct count reads the registered wired C itself (`interface.program_of_record`, whose wires are its
edges) and the registered partition object, never a Program rebuilt from the record. On a wired window a wired row is an
edge, so the count's fresh bits a block are the record's count (`block_counts.fresh`): the two are one by construction.

Two of the three assertions below are green today and guard that invariant on the Python side. The third, the
Lean-verifier-vs-Python agreement on the wired C, is RED today for ONE reason and no other: the circuit format's decoder
refuses the base gates And2_v1/Not_v1/Xor2_v1 for want of a Boolean body (`codec._decode_boolean`, `E_BODY_MISSING`), so
the count's Python reference (which the count's difftest holds the Lean verifier to) returns `{"error": "program"}`
before it counts anything. It is written with no xfail: it turns green when circuits lifts that decoder refusal for the
base gates and the new partition-form count reads the wired C's partition. Nothing else here is expected to fail.
"""
from __future__ import annotations

import importlib.util

import built as B
import inference_only as IO
import pytest
from verity_vllm.program.predict import block_counts as BC

from verity.commitments.periods import interface as IF
from verity.circuits import codec

GEMM = 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
ROPE = "RoPEHead_v1{D=16}"


def made(wired: bool) -> dict:
    """A tiny circuit record: a Gemm batch and a RoPE batch over 2 slots of 3 steps; with ``wired`` the RoPE reads the
    Gemm's output (a wire), so the record's count subtracts that wired row where the session Program with no wires would
    count it fresh."""
    g, r = BC.resolve(GEMM), BC.resolve(ROPE)
    gid, rid = IF.descriptor_id(g), IF.descriptor_id(r)
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0),
               IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    wires = (IF.Wire(rid, "x", gid, "o0", 0, (2,), 0, (2,), (0, 2)),) if wired else ()
    return IF.descriptor_record(
        IF.Circuit("test/made", ((g, 4), (r, 6)), wires=wires, cuts={rid: {"x": 2}} if wired else {},
                   blocks=IF.Blocks(slots=2, steps=3, inference=entries)), 1)


def reference(inst: dict) -> dict:
    """The count's Python reference, which the count's difftest holds the Lean verifier to (`built.lean_count`)."""
    spec = importlib.util.spec_from_file_location("inference_only_reference",
                                                  B.PACKAGE / "scripts" / "inference_only_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


@pytest.mark.parametrize("wired", [False, True])
def test_the_records_count_is_block_counts_fresh(wired: bool) -> None:
    """Green: the record's count a block (`inference_only.record_count`) is `block_counts.fresh`, and on the wired case it
    subtracts the wired row (one Gemm output, 8 words of 16 bits a slot) that a no-wire session Program counts fresh."""
    doc = made(wired)
    _, m, _ = BC.session(doc)
    counted = IO.record_count(doc, m, None, None)
    assert counted["fresh"] == BC.fresh(doc, m, None, None)
    assert (counted["wired"] > 0) == wired


@pytest.mark.parametrize("wired", [False, True])
def test_the_lean_count_on_the_wired_C_is_the_records_count(wired: bool) -> None:
    """RED today for the reference's registry alone: the registered wired C (`partition_instance`) decodes under the real
    registry (`BC.resolve` / `defs.REGISTRY`), but the count's reference decodes under its own `declared(desc)`, under which
    the base gates `And2_v1`/`Not_v1`/`Xor2_v1` resolve to primitives `is_boolean` refuses, so `codec._decode_boolean`
    raises `E_BODY_MISSING` and the reference returns `{"error": "program"}` before any count. No xfail: this turns green
    when the reference's `declared(desc)` exempts the base gates as the real registry does, and the partition-form count
    reads the wired C's blocks."""
    doc = made(wired)
    m = BC.session(doc)[1]
    c, k, n = 10 ** 9, BC.fresh(doc, m, None, None), doc["blocks"]["slots"] if "slots" in doc["blocks"] else 2
    inst = IO.partition_instance(doc, None, None, c, k, n)
    decoded = _decodes(inst["program"])
    assert decoded is None, (f"the reference does not decode the registered wired C: {decoded}. Today this is the "
                             "reference's own `declared(desc)` registry (not the core decoder, which accepts C under "
                             "`BC.resolve`/`defs.REGISTRY`): under it the base gates And2_v1/Not_v1/Xor2_v1 resolve to "
                             "primitives is_boolean refuses, so `codec._decode_boolean` raises E_BODY_MISSING; when the "
                             "reference exempts them this reads the partition's blocks, each block's fresh the record's count")


def _decodes(program: str) -> str | None:
    """None if the circuit format's decoder decodes ``program``, else the error (today `E_BODY_MISSING` for the base gates)."""
    ref = reference({})
    try:
        desc = codec.decode_program(__import__("json").loads(program), ref.declared(__import__("json").loads(program)))
        return None if desc is not None else "no program"
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return str(e)


def test_a_claimed_wire_the_registered_C_lacks_is_refused() -> None:
    """Negative soundness: the count builds C from the record's wires by the one compiler (`program_of_record`), which
    refuses a wire the registered C cannot carry — here a backward wire, its writer unit at or after its reader's. A wire
    that does not fit is no edge of C, so its row is never credited as wired; a record that claims one is refused
    (`partition_instance` raises `Undecided`) rather than counted as if the wire held."""
    g, r = BC.resolve(GEMM), BC.resolve(ROPE)
    gid, rid = IF.descriptor_id(g), IF.descriptor_id(r)
    entries = (IF.InferenceEntry(GEMM, 0, (2, 2), ("slot", "-"), 0),
               IF.InferenceEntry(ROPE, 0, (3, 2), ("step", "slot"), 0))
    # a wire the RoPE at unit 0 reads from the Gemm at unit 2: its writer is after its reader, a wire C cannot carry
    bad = (IF.Wire(rid, "x", gid, "o0", 0, (2,), 2, (2,), (0, 2)),)
    with pytest.raises((ValueError, IO.Undecided)):
        IF.descriptor_record(IF.Circuit("test/bad", ((g, 4), (r, 6)), wires=bad, cuts={rid: {"x": 2}},
                                        blocks=IF.Blocks(slots=2, steps=3, inference=entries)), 1)
