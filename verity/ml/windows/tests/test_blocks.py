"""A circuit's blocks (`verity.ml.windows.interface.Blocks`, ``verity/blocks/v1``) and output-row reads: the inference entries
tile each template's units into (slot, step) exactly once at the class's S and T, the proof block is the record's partition,
the blocks state no placement, and an output-row `Read` names the leaves a unit's output row is of a value without filling an
input row. Pinned on `test_interface`'s two-request chain and on a class circuit (`cmp92`: S = 4 slots, NP = 8 prompt bytes,
O = 24 outputs, T = 31 steps)."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import test_interface as TI

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.windows import interface as I, rows as CS

GID, RUN = TI.GID, TI.RUN


def held(*params: str) -> dict[str, dict[str, str]]:
    """`interface.input_count`'s pins and the window's roots holding ``params``: each pinned at the root the window records."""
    pins = {p: "ab" * 64 for p in params}
    return {"pins": pins, "roots": pins}


def chain_blocks(**edit) -> I.Blocks:
    """`TI.chain`'s blocks: its (layer, step, request) order is one entry, a layer digit, a step digit and the slot."""
    return I.Blocks(**{"slots": TI.REQUESTS, "steps": TI.STEPS,
                       "inference": (I.InferenceEntry(GID, 0, (TI.LAYERS, TI.STEPS, TI.REQUESTS), ("-", "step", "slot"), 0),),
                       **edit})


def blocked(blocks: I.Blocks | None = None) -> I.Circuit:
    return replace(TI.chain(), blocks=blocks or chain_blocks())


def test_the_blocks_ride_in_the_circuit_record_and_the_verifier_holds_their_proof_block_to_it():
    """The record's ``blocks`` is the circuit's (`Circuit.blocks_record`): its proof block the record's units and partition
    digest, the plan carried. The verifier rebuilds the window from it and refuses a proof block that isn't the record's, a
    malformed one, and blocks in a record that lists its members. Two windows of the class, under other runs, record
    byte-identical blocks."""
    plan = "ab" * 32
    win = I.Window(blocked(chain_blocks(plan=plan)), RUN, TI.LAW, 2)
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    assert doc["blocks"] == win.circuit.blocks_record() == {
        "format": I.BLOCKS, "proof": {"units": 8, "partition": doc["partition"], "plan": plan},
        "inference": {"slots": 2, "steps": 2, "entries": [{"template": GID, "at": 0, "radix": [2, 2, 2],
                                                           "axes": ["-", "step", "slot"], "step": 0}]}}
    by_id = {GID: TI.GEMM}.__getitem__
    assert I.window_of_record(doc, TI.LAW, by_id).members == win.window.members
    other = I.Window(blocked(chain_blocks(plan=plan)), "cd" * 64, TI.LAW, 2)
    assert I.circuit_bytes(other.circuit_record()["blocks"]) == I.circuit_bytes(doc["blocks"])
    b = doc["blocks"]
    for bad, why in [({**b, "proof": {**b["proof"], "units": 9}}, "not the record's"),
                     ({**b, "proof": {**b["proof"], "partition": "00" * 64}}, "not the record's"),
                     ({**b, "proof": {**b["proof"], "units": True}}, "not the record's"),
                     ({**b, "proof": {**b["proof"], "extra": 0}}, "optionally its plan"),
                     ({**b, "proof": {**b["proof"], "plan": "x"}}, "SHA-256 or SHA-512"),
                     ({**b, "format": "x"}, I.BLOCKS), ({**b, "other": 0}, I.BLOCKS),
                     ({**b, "inference": {**b["inference"], "entries": {}}}, "its entries a list"),
                     ({**b, "inference": {**b["inference"], "slots": 3}}, "the class's 3 slots"),
                     ({**b, "server": []}, "which host runs which units is private")]:
        with pytest.raises(ValueError, match=why):
            I.window_of_record({**doc, "blocks": bad}, TI.LAW, by_id)
    listed = json.loads(I.circuit_bytes(I.circuit_record(win.window, "chain", [(GID, 8)])))
    with pytest.raises(ValueError, match="one or the other"):
        I.window_of_record({**listed, "blocks": b}, TI.LAW, by_id)
    with pytest.raises(ValueError, match="stated with a cap"):
        I.circuit_record(win.window, "chain", [(GID, 8)], blocks=b)
    assert "blocks" not in I.Window(TI.chain(), RUN, TI.LAW, 2).circuit_record()


def test_the_inference_entries_tile_each_templates_units_exactly_once_at_the_classs_slots_and_steps():
    """An entry is a run of a template's units in mixed radix, one digit the slot and at most one the step; the entries
    together cover every unit of every template once, each slot digit of the class's S and each step below its T. The work
    is per entry, so a gap, an overlap, a short tiling, an unknown template and a malformed entry are each refused by name."""
    e = chain_blocks().inference[0]
    half = replace(e, radix=(TI.STEPS, TI.REQUESTS), axes=("step", "slot"))
    for inference, why in [((half,), r"tile units \[0, 4\) of its 8"),
                           ((half, replace(half, at=6, radix=(1, 2))), r"units \[4, 6\) are in no inference entry"),
                           ((half, replace(half, at=2)), "unit 2 is in two inference entries"),
                           ((e, replace(e, template="Gemm_v2{}")), "names no template"),
                           ((replace(e, step=1),), r"steps \[1, 3\), each below the class's 2")]:
        with pytest.raises(ValueError, match=why):
            blocked(chain_blocks(inference=inference))
    with pytest.raises(ValueError, match="radix is the class's 4 slots"):
        blocked(chain_blocks(slots=4))
    for edit, why in [(dict(axes=("-", "-", "-")), "one of them the slot"), (dict(axes=("slot", "step", "slot")), "one of them"),
                      (dict(axes=("step", "step", "slot")), "at most one the step"), (dict(axes=("x", "step", "slot")), "one axis"),
                      (dict(axes=("step", "slot")), "one axis"), (dict(step=-1), "naturals"), (dict(radix=(2, 0, 2)), "radix 1")]:
        with pytest.raises(ValueError, match=why):
            replace(e, **edit)
    with pytest.raises(ValueError, match="an inference entry is"):
        I.InferenceEntry.of({**e.doc(), "extra": 0})
    with pytest.raises(ValueError, match="at least one"):
        chain_blocks(steps=0)
    with pytest.raises(ValueError, match="are `Blocks`"):
        replace(TI.chain(), blocks=chain_blocks().doc(8, "00" * 64))


def test_an_output_row_read_names_a_values_leaves_and_fills_no_input_row():
    """An output-row `Read` (on ``o<j>``) states that the reader unit's output row is the parameter's leaves at its pattern's
    positions: a value the circuit sends out. It is held to its template's units and its parameter's leaves, uncut, may sit on a
    row wires read, and counts in no input; a second read of the row is refused, since the row opens against one pinned
    root."""
    reg = TI.declared().registered
    out = I.Read(TI.G8ID, "o0", "resp", at=0, radix=(6,), base=0, step=(8,))
    c = TI.declared(registered=replace(reg, params={**reg.params, "resp": 48}, reads=(*reg.reads, out)))
    assert I.input_count(c.description(), **held("weights", "resp")) == \
        I.input_count(TI.declared().description(), **held("weights"))
    with pytest.raises(ValueError, match="both fill rows"):
        TI.declared(registered=replace(reg, params={**reg.params, "resp": 48, "wire": 48},
                                       reads=(*reg.reads, out, replace(out, param="wire"))))
    for edit, why in [(dict(base=1), r"read .* leaves \[1, 49\)"), (dict(param="acts"), "of a registered parameter"),
                      (dict(at=1), r"units \[1, 7\)"), (dict(radix=(3, 2), step=(16, 8), cut=(0, 2)), r"rows \[0, 2\) of o0's 1")]:
        with pytest.raises(ValueError, match=why):
            TI.declared(registered=replace(reg, params={**reg.params, "resp": 48}, reads=(*reg.reads, replace(out, **edit))))


S, NP, O = 4, 8, 24
T, LAYERS, H = NP + O - 1, 2, 16
EMB, HEAD = bind(EW.Bf16Add), bind(G.Gemm, K=H, N=1, **G.GEMM_HOPPER)
QKV = bind(G.Gemm, K=H, N=H, **G.GEMM_HOPPER)
QK, OUT = (bind(G.GemmCoordinate, K=k, **G.GEMM_HOPPER) for k in (H, 2 * H))
EID, QID, KID, OID, HID = map(descriptor_id, (EMB, QKV, QK, OUT, HEAD))
QK_LAYER = S * T * (T + 1)


def cmp92() -> I.Circuit:
    """The class circuit (S slots, NP prompt bytes, O outputs, T = NP + O − 1 steps) at its block layout and width
    H = 16, its row ops stood in by catalog units, rows gathered by the wire cut. At step p,
    slot j: the token's embedding as H ``Bf16Add`` coordinates (the token and the position's weight, units (p·S + j)·H + c);
    per layer, six ``Gemm_v2{16, 16}`` head rows (q0 q1 k0 k1 v0 v1) reading the H coordinates cut a row each, QKᵀ as
    ``GemmCoordinate_v2{16}`` over every key so far in the plan's order (head, key, slot: layer l's at l·S·T(T+1) +
    S·p(p+1), the KV cache wired across steps), and the layer's H coordinates (``GemmCoordinate_v2{32}``, its ``x`` the two
    v rows); at the O logit steps p = NP − 1 + k the next token (``Gemm_v2{16, 1}``). Byte k of slot j enters at step k
    from the request row's leaf j(1 + NP) + 1 + k (the length, then the bytes as u16) and leaves at step NP − 1 + k as the
    response's leaf j·O + k, the head's output row read once, as the response."""
    pos, layer = T * H, 8 * H * H
    qw = {0: pos, 1: pos + layer}
    ow = {k: qw[k] + 6 * H * H for k in (0, 1)}
    wires = [I.Wire(QID, "x", EID, "o0", at=0, radix=(T * S, 6, H), base=0, step=(H, 0, 1), cut=(0, H)),
             I.Wire(OID, "x", QID, "o0", at=0, radix=(LAYERS * T * S, H, 2), base=4, step=(6, 0, 1), cut=(0, 2)),
             I.Wire(HID, "x", OID, "o0", at=0, radix=(O * S, H), base=H * (T + NP - 1) * S, step=(H, 1), cut=(0, H)),
             I.Wire(QID, "x", OID, "o0", at=6 * T * S, radix=(T * S, 6, H), base=0, step=(H, 0, 1), cut=(0, H)),
             I.Wire(EID, "a", HID, "o0", at=NP * S * H, radix=((T - NP) * S, H), base=0, step=(1, 0))]
    for k in range(LAYERS):
        for p in range(T):
            at = k * QK_LAYER + S * p * (p + 1)
            wires += [I.Wire(KID, "x", QID, "o0", at=at, radix=(2, p + 1, S), base=6 * (k * T + p) * S, step=(1, 0, 6)),
                      I.Wire(KID, "wrow", QID, "o0", at=at, radix=(2, p + 1, S), base=6 * k * T * S + 2, step=(1, 6 * S, 6))]
    reads = [I.Read(EID, "a", "request", at=0, radix=(NP, S, H), base=1, step=(1, 1 + NP, 0)),
             I.Read(EID, "b", "weights", at=0, radix=(T, S, H), base=0, step=(H, 0, 1)),
             I.Read(HID, "o0", "response", at=0, radix=(O, S), base=0, step=(1, O)),
             *(I.Read(QID, "w", "weights", at=k * 6 * T * S, radix=(T * S, 6), base=qw[k], step=(0, H * H)) for k in (0, 1)),
             *(I.Read(OID, "wrow", "weights", at=k * H * T * S, radix=(T * S, H), base=ow[k], step=(0, 2 * H)) for k in (0, 1)),
             I.Read(HID, "w", "weights", at=0, radix=(O * S,), base=ow[1] + 2 * H * H, step=(0,))]
    params = {"request": S * (1 + NP), "response": S * O, "weights": pos + 2 * layer + H}
    inference = (I.InferenceEntry(EID, 0, (T, S, H), ("step", "slot", "-"), 0),
                 I.InferenceEntry(QID, 0, (LAYERS, T, S, 6), ("-", "step", "slot", "-"), 0),
                 *(I.InferenceEntry(KID, k * QK_LAYER + S * p * (p + 1), (2, p + 1, S), ("-", "-", "slot"), p)
                   for k in range(LAYERS) for p in range(T)),
                 I.InferenceEntry(OID, 0, (LAYERS, T, S, H), ("-", "step", "slot", "-"), 0),
                 I.InferenceEntry(HID, 0, (O, S), ("step", "slot"), NP - 1))
    return I.Circuit("cmp92", ((EMB, H * T * S), (QKV, LAYERS * 6 * T * S), (QK, LAYERS * QK_LAYER),
                               (OUT, LAYERS * H * T * S), (HEAD, O * S)),
                     external=(f"{EID}/a", f"{HID}/o0"), wires=tuple(wires), cuts={QID: {"x": H}, OID: {"x": 2}, HID: {"x": H}},
                     registered=I.Registered(params, (), tuple(reads)), blocks=I.Blocks(S, T, inference))


@pytest.fixture(scope="module")
def circuit() -> I.Circuit:
    return cmp92()


def test_cmp92s_class_circuit_has_one_inference_and_nothing_free(circuit):
    """The class circuit's units with one inference block, the plan's QKᵀ entry at step p
    ``{at: S·p(p+1), radix: [2, p+1, S], axes: [-, -, slot], step: p}`` among them; with the verifier's pins (the weights
    and the request) no input row is free, the response on its writer being an output-row read that fills none."""
    assert list(circuit.units.values()) == [H * T * S, 12 * T * S, 2 * QK_LAYER, 2 * H * T * S, O * S]
    assert {"template": KID, "at": S * 5 * 6, "radix": [2, 6, S], "axes": ["-", "-", "slot"], "step": 5} in \
        [e.doc() for e in circuit.blocks.inference]
    count = I.input_count(json.loads(json.dumps(circuit.description())), **held("weights", "request"))
    assert count["free"] == 0 and count["ingress"] == 0, count
    assert circuit.blocks_record()["proof"] == {"units": 15472,
                                                "partition": I.descriptor_record(circuit, 1 << 20)["partition"]}


def test_cmp92s_members_come_from_its_record_and_its_blocks_are_the_same_under_any_run(circuit):
    """The layer row units are one statement (their one wire covers both layers), and the verifier derives the same members
    from the descriptor record; the window's blocks are byte-identical under any run. The members are the circuit's without
    its registered reads: `members` fills tables for uncut reads of whole registered rows over whole members only, and the
    class circuit's weight reads start mid-row."""
    cap = 1 << 20
    with pytest.raises(ValueError, match="whole registered rows"):
        I.members(circuit, cap)
    plain = replace(circuit, registered=None)
    base = {t: lo for _k, t, lo, _n in CS.template_partition(*plain.program()).ranges}
    assert [(m.base, m.n) for m in I.members(plain, cap) if m.template == OID] == [(base[OID], 2 * H * T * S)]
    rec = I.descriptor_record(plain, cap)
    for run in ("ab" * 64, "cd" * 64):
        w = I.window_of_record({**rec, "run": run, "context": ""}, TI.LAW, I.resolve)
        assert w.members == I.members(plain, cap)
    assert I.circuit_bytes(rec["blocks"]) == I.circuit_bytes(plain.blocks_record())
