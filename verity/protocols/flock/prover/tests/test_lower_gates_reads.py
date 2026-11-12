"""`ir_lower.lower_gates` drops a word once its last reader is lowered, for a caller that names what it reads afterwards
(``reads``), and the circuit is the one it lowers to with every word kept: every row in order and the laid-out text, for a
whole unit (`ir_lower.lower`, and the unit a stager lays out), a split and a cut (`Units.circuit`), a caller that names
its outputs (`partition_units.lower_class`), an activation lowered by its builder, and one ``wires`` passed from call to
call. The words the caller named are in ``wires`` afterwards, and no other."""
from __future__ import annotations

import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.layout import Region
from verity.circuits.program import Program
from verity.circuits.types import Array
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL
from verity_flock import partition_units as PU
from verity_flock.gf2 import Circuit

W = 16
WORD = Array(W, BL.BIT)


def _mix(C, s, x):
    t = [s[i] ^ s[(i + 1) % W] ^ x[i] for i in range(W)]
    return C.commit([t[i] ^ C.AND(t[(i + 1) % W], t[(i + 3) % W]) for i in range(W)])


#: a Boolean Definition whose builder commits: lowered by its builder it has W rows its gates do not
MIX = definition("TestReadsMix", 1, _mix, (("s", WORD), ("x", WORD)), WORD, register=False)
MIX_GATES = sum(MIX.traced().counts.values())
MIX_ANDS = MIX.traced().counts["and"]


def _rounds(B, S, s, x):
    """Each round an activation of MIX, then 2 W gates lowered one by one that read its returns and the round's input: every
    word has readers of both kinds, and the last is not always the nearest."""
    for i in range(S.N):
        s = B.call(MIX, s, x[i])
        s = BL.operands(*[B.call(BL.XOR, BL.operands(B.call(BL.AND, BL.operands(s[j], x[i][j])), s[(j + 1) % W]))
                          for j in range(W)])
    return s


ROUNDS = CompositeDefinition("TestReadsRounds", 1, ("N",), lambda S: ((("s", WORD), ("x", Array(S.N, WORD))), WORD), _rounds,
                             register=False)
ROUND_GATES = MIX_GATES + 2 * W
SPREAD = CompositeDefinition("TestReadsSpread", 1, ("N",), lambda S: ((("s", Array(S.N, WORD)), ("x", Array(S.N, WORD))),
                                                                      Array(S.N, WORD)),
                             lambda B, S, s, x: B.batch(MIX, s, x, axes=(0, 0)), register=False)
#: MIX with a parameter no gate reads
SPARE = definition("TestReadsSpare", 1, lambda C, s, x, spare: _mix(C, s, x), (("s", WORD), ("x", WORD), ("spare", BL.BIT)), WORD,
                   register=False)


def _form(x) -> tuple:
    return sorted(x.s), x.c


def _made(C) -> tuple:
    """What a lowering made of C: every row in order, an AND as its operands' bits; its outputs; and C laid out as
    ``flock-ir-unit/v2`` text."""
    rows = [(k, tuple(map(_form, d)) if k == "and" else d) for k, d in zip(C.kind, C.data, strict=True)]
    outs = {name: [_form(b) for b in bits] for name, bits in C.outputs.items()}
    return rows, outs, IL.netlist(IL.layout(C, list(C.inputs), list(C.outputs)), "unit/test-reads")


@pytest.fixture
def kept(monkeypatch):
    """Calling it makes every `lower_gates` from then on keep every word, as for a caller that names no reads."""
    real = IL.lower_gates

    def keep() -> None:
        monkeypatch.setattr(IL, "lower_gates", lambda C, gates, wires, pieces=None, reads=None: real(C, gates, wires, pieces))

    return keep


def _lower(d) -> Circuit:
    C = Circuit()
    outs = IL.lower(C, d, [C.input(f"in{j}", 1) for j in range(sum(t.leaves for _, t in d.signature()[0]))])
    C.output("out", [b for w in outs for b in w])
    return C


def _lower_call(call, reads, gates=None) -> tuple[Circuit, dict]:
    """``call`` (or its ``gates``) through `lower_gates` on one input a bit: the circuit, with the call's returns its output
    when ``wires`` still holds them, and ``wires``."""
    C = Circuit()
    wires = {a: C.input(f"in{j}", 1) for j, a in enumerate(call.args)}
    IL.lower_gates(C, call if gates is None else gates, wires, reads=reads)
    if all(r in wires for r in call.returns):
        C.output("out", [wires[r][0] for r in call.returns])
    return C, wires


@pytest.mark.parametrize("n", [1, 3])
def test_a_whole_unit_lowers_to_the_same_circuit(n, kept) -> None:
    d = bind(ROUNDS, N=n)
    dropped = _made(_lower(d))
    assert _lower(d).counts()["ands"] == n * (MIX_ANDS + 2 * W)        # its trace's ANDs, its commits and the round's W ANDs
    kept()
    assert _made(_lower(d)) == dropped


def test_the_words_named_are_what_wires_holds_afterwards() -> None:
    """Every word but the named ones is gone, the inputs among them; a named word is the word the lowering that keeps
    every word holds, whether a builder wrote it, a gate did, or the caller brought it."""
    _prog, call = IL.standalone(bind(ROUNDS, N=3))
    whole, every = _lower_call(call, None)
    assert len(every) == len(call.args) + 3 * 3 * W
    first_and = call.circuit.gate(call.lo + MIX_GATES)
    builders, gates, brought = first_and.operands[0], first_and.index, call.args[0]
    for extra in ([], [builders], [gates], [brought]):
        reads = [*call.returns, *extra]
        C, wires = _lower_call(call, reads)
        assert set(wires) == set(reads) and _made(C) == _made(whole)
        assert all(list(map(_form, wires[r])) == list(map(_form, every[r])) for r in reads)
    with pytest.raises(KeyError):
        wires[gates]


def test_a_stagers_unit_is_the_same_text(kept) -> None:
    """`class_statement.unit_lowering`, as a `UnitStager` lowers a unit: its layout's text is the staged circuit."""
    d = bind(ROUNDS, N=4)

    def text() -> str:
        low, _out_words = CS.unit_lowering(d, "unit/test-reads", [("s", ["s"]), ("x", ["x"])])
        assert low.layout.counts["ands"] == 4 * (MIX_ANDS + 2 * W)
        return IL.netlist(low.layout, "unit/test-reads")

    dropped = text()
    kept()
    assert text() == dropped


def test_a_split_units_circuit_is_the_same(kept) -> None:
    """`ir_lower.units` splits a batch into one activation a unit, which `Units.circuit` lowers naming the unit's returns."""
    U = IL.units(bind(SPREAD, N=3))
    assert U.n == 3
    dropped = _made(U.circuit())
    assert U.circuit().counts()["ands"] == MIX_ANDS + W
    kept()
    assert _made(U.circuit()) == dropped


def test_a_cut_units_circuit_is_the_same(kept, monkeypatch) -> None:
    """A cut inside the activations makes each cut word one the caller reads, so they are lowered gate by gate: the same
    circuit as the gates alone give with every word kept."""
    d = bind(ROUNDS, N=2)

    def ands_of_mix(gates, _args, _rets):
        return [g.index for g in gates if BL.gate_kind(g.prim) == ("and", 2) and IL._boolean_activation(g) is not None]

    U = IL.units(d, ands_of_mix)
    assert U.n == 1 and len(U.cut) == 2 * MIX_ANDS and set(U.cut) < set(U.ret_gates)
    dropped = _made(U.circuit())
    assert U.circuit().counts()["ands"] == 2 * (MIX_ANDS + W)           # no commits: its gates
    monkeypatch.delattr(MIX, "build")
    kept()
    assert _made(U.circuit()) == dropped


@pytest.mark.parametrize("spec", ["q-word", "blocks:3", "scatter:5/1"])
def test_a_callers_named_outputs_give_the_same_circuit(spec, kept) -> None:
    """`partition_units.lower_class` names a unit's outputs: a whole activation's returns (q-word), or words in the middle of
    one (the others), which it then holds in part."""
    units = PU.partition_units(Program(bind(ROUNDS, N=2)), spec)
    classes = list(units.classes().values())

    def circuits() -> list:
        return [PU.lower_class(units.activation(cl.definition), cl.representative).circuit for cl in classes]

    dropped = [_made(C) for C in circuits()]
    assert (MIX_ANDS + W in [C.counts()["ands"] for C in circuits()]) == (spec == "q-word")
    kept()
    assert [_made(C) for C in circuits()] == dropped


def test_an_activation_is_lowered_by_its_builder_once_either_way(kept, monkeypatch) -> None:
    activation, built = IL._lower_activation, []

    def counted(C, s, *rest):
        built.append(s.offset)
        return activation(C, s, *rest)

    monkeypatch.setattr(IL, "_lower_activation", counted)
    _prog, call = IL.standalone(bind(ROUNDS, N=3))
    _lower(bind(ROUNDS, N=3))
    assert built == [call.lo + i * ROUND_GATES for i in range(3)]
    kept()
    _lower(bind(ROUNDS, N=3))
    assert built[3:] == built[:3]


def test_a_builder_whose_leaf_has_no_word_is_refused_either_way(kept) -> None:
    """With no word for the parameter no gate reads, the activation is lowered by its gates: the same decision whether it is
    taken as the walk reaches it or ahead of it."""
    d = bind(SPARE)

    def lower(n_inputs: int) -> Circuit:
        C = Circuit()
        C.output("out", [b for w in IL.lower(C, d, [C.input(f"in{j}", 1) for j in range(n_inputs)]) for b in w])
        return C

    assert lower(2 * W + 1).counts()["ands"] == MIX_ANDS + W
    dropped = _made(lower(2 * W))
    assert lower(2 * W).counts()["ands"] == MIX_ANDS
    kept()
    assert _made(lower(2 * W)) == dropped


def test_a_builder_that_declines_once_chosen_is_an_error(kept, monkeypatch) -> None:
    """A caller that names its reads has each word's readers counted before anything is lowered, over the builders chosen
    then: one that declines afterwards is refused, where a caller that names none falls back to the activation's gates."""
    monkeypatch.setattr(IL, "_lower_activation", lambda *_a: False)
    with pytest.raises(AssertionError, match="before its builder ran"):
        _lower(bind(ROUNDS, N=1))
    kept()
    assert _lower(bind(ROUNDS, N=1)).counts()["ands"] == MIX_ANDS + W


def test_a_read_counted_and_never_made_is_an_error(monkeypatch) -> None:
    steps = IL._steps
    _prog, call = IL.standalone(bind(ROUNDS, N=1))

    def one_more(gates, wires, reads):
        made, left = steps(gates, wires, reads)
        left[call.args[0]] += 1
        return made, left

    monkeypatch.setattr(IL, "_steps", one_more)
    with pytest.raises(AssertionError, match="1 reads counted over the gates were not made"):
        _lower_call(call, call.returns)


def test_wires_passed_from_call_to_call_give_the_same_circuit() -> None:
    """Two calls over the two halves of a unit on one ``wires``: the first names what the second reads, or neither names
    anything; a first call that names nothing it reads leaves the second a KeyError, not another circuit."""
    _prog, call = IL.standalone(bind(ROUNDS, N=4))
    mid = call.lo + 2 * ROUND_GATES
    halves = Region(call.circuit, call.lo, mid), Region(call.circuit, mid, call.hi)
    second_reads = sorted({o for i in range(mid, call.hi) for o in call.circuit.gate(i).operands if o < mid})
    assert len(second_reads) == 3 * W                                   # the state after two rounds and two rounds' inputs
    whole = _made(_lower_call(call, call.returns)[0])
    for first, second in ((None, None), (second_reads, call.returns), (second_reads, None), (None, call.returns)):
        C = Circuit()
        wires = {a: C.input(f"in{j}", 1) for j, a in enumerate(call.args)}
        IL.lower_gates(C, halves[0], wires, reads=first)
        assert first is None or set(wires) == set(second_reads)
        IL.lower_gates(C, halves[1], wires, reads=second)
        assert second is None or set(wires) == set(call.returns)
        C.output("out", [wires[r][0] for r in call.returns])
        assert _made(C) == whole
    C = Circuit()
    wires = {a: C.input(f"in{j}", 1) for j, a in enumerate(call.args)}
    IL.lower_gates(C, halves[0], wires, reads=())
    assert not wires
    with pytest.raises(KeyError):
        IL.lower_gates(C, halves[1], wires, reads=call.returns)
