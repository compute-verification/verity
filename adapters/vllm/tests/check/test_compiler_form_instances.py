"""verity_vllm.check.match.compiler_form.compiled on instance lists written out by hand: what it leaves as it is, which reads of
a narrowed linear it moves, and what it refuses by name.  The whole-Program cases are in test_compiler_form.py.  These are the
ones no built Program brings: instances with no sampling event, instances with no linear, a read in parameter space, a gather
whose stride steps over a narrowed linear, two narrowed linears, a linear with a bias, and a linear whose spec, output or column
argument is not what narrowing needs.  Each runs in the record's operand form and in the compact one."""
from __future__ import annotations

import pytest

from verity_vllm.check.match import compiler_form as CF
from verity_vllm.program import compact as CP

FORMS = pytest.mark.parametrize("compact", [False, True], ids=["record", "compact"])
WORDS4 = "Array<4,Value<16>>"
WORDS8 = "Array<8,Value<16>>"
NOTHING = {"sampling_events": 0, "dead": {}, "swapped": {}, "narrowed": {}}


def _inst(k: int, spec: str, args: list[list[tuple]], out: str, compact: bool) -> dict:
    """One instance as a decoded root body gives it, each operand a list of runs (space, index, lo, hi) in the form asked for."""
    return {"k": k, "spec": spec, "form": "call", "args": [CP.from_runs(a) if compact else list(a) for a in args], "out": out}


def _runs(arg: list, compact: bool) -> list[tuple]:
    return CP.expand(arg) if compact else list(arg)


def _same_but(got: dict, was: dict, **changed: object) -> bool:
    """`got` is `was` with the keys of `changed` replaced, operands apart (they are compared as runs)."""
    return {k: v for k, v in got.items() if k != "args"} == {**{k: v for k, v in was.items() if k != "args"}, **changed}


@FORMS
def test_instances_with_no_sampling_event_are_left_as_they_are(compact: bool):
    """The form's outputs are the sampling events.  With none the list comes back itself: nothing is called dead, the norm keeps
    its version and the linear read in part keeps its columns."""
    inst = [_inst(0, "RMSNormTriton_v1{N=8}", [[("p", 0, 0, 8)]], WORDS8, compact),
            _inst(1, "Gemm_v2{K=4,N=8}", [[("n", 0, 0, 8)], [("p", 1, 0, 32)]], WORDS8, compact),
            _inst(2, "SiluMul_v2{N=4}", [[("n", 1, 2, 6)]], WORDS4, compact)]
    out, facts = CF.compiled(inst, compact)
    assert out is inst and facts == NOTHING


@FORMS
def test_with_no_linear_the_dead_go_and_the_versions_change_and_nothing_is_narrowed(compact: bool):
    """A read in parameter space names no instance: the select reads parameter 1 and node 0, so node 1, which nothing reads,
    is dead.  A scope before the name stays where it is."""
    inst = [_inst(0, "RMSNormTriton_v1{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, "RoPE_v1{D=4}", [[("n", 0, 0, 4)]], WORDS4, compact),
            _inst(2, "batch:TokenSelect_v1{V=4}", [[("p", 1, 0, 4), ("n", 0, 0, 4)]], "Value<32>", compact)]
    out, facts = CF.compiled(inst, compact)
    assert facts == {"sampling_events": 1, "dead": {"RoPE_v1": 1}, "narrowed": {},
                     "swapped": {"RMSNormTriton_v1 -> RMSNormTriton_v5": 1, "TokenSelect_v1 -> TokenSelect_v4": 1}}
    assert out == [{**inst[0], "spec": "RMSNormTriton_v5{N=4}"}, {**inst[2], "spec": "batch:TokenSelect_v4{V=4}"}]
    assert all(got["args"] is was["args"] for got, was in zip(out, (inst[0], inst[2]), strict=True))


@FORMS
def test_a_linear_read_in_part_is_narrowed_and_only_the_reads_of_it_move(compact: bool):
    """Node 2's eight output words are read in [2, 4), in [3, 6) and at words 2 and 4, so it is bound at N = 4 over rows 2 to 5 of
    its weights, and each read of it moves down by 2: the first run of a gather over nodes 2 and 4, the last run of one over
    nodes 4 and 2, and a window that moves over node 2 alone.  Three instances read nothing of it and are the list's own
    afterwards: one reads parameter 2, which is no node; two gather nodes 1 and 3, in either order, a stride that steps over node
    2.  An operand that reads node 2 is a new list, and the other operands of the same instance are its own."""
    inst = [_inst(0, "SiluMul_v2{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, "SiluMul_v2{N=4}", [[("p", 0, 4, 8)]], WORDS4, compact),
            _inst(2, "Gemm_v2{K=4,N=8}", [[("n", 0, 0, 4)], [("p", 1, 0, 32)]], WORDS8, compact),
            _inst(3, "SiluMul_v2{N=4}", [[("p", 2, 0, 4)]], WORDS4, compact),
            _inst(4, "Concat_v1{N=8}", [[("n", 1, 0, 4), ("n", 3, 0, 4)]], WORDS8, compact),
            _inst(5, "Concat_v1{N=8}", [[("n", 3, 0, 4), ("n", 1, 0, 4)]], WORDS8, compact),
            _inst(6, "Add_v1{N=2}", [[("n", 5, 0, 2)], [("p", 2, 0, 1), ("n", 2, 2, 4), ("n", 4, 2, 4)]], "Array<2,Value<16>>", compact),
            _inst(7, "SiluMul_v2{N=6}", [[("n", 4, 3, 6), ("n", 2, 3, 6)]], "Array<6,Value<16>>", compact),
            _inst(8, "SiluMul_v2{N=2}", [[("n", 2, 2, 3), ("n", 2, 4, 5)]], "Array<2,Value<16>>", compact),
            _inst(9, "TokenSelect_v4{V=10}", [[("n", 6, 0, 2), ("n", 7, 0, 6), ("n", 8, 0, 2)]], "Value<32>", compact)]
    out, facts = CF.compiled(inst, compact)
    assert facts == {"sampling_events": 1, "dead": {}, "swapped": {}, "narrowed": {"Gemm_v2": 1}}
    assert len(out) == 10 and all(out[i] is inst[i] for i in (0, 1, 3, 4, 5, 9))
    assert _same_but(out[2], inst[2], spec="Gemm_v2{K=4,N=4}", out=WORDS4)
    assert out[2]["args"][0] is inst[2]["args"][0] and _runs(out[2]["args"][1], compact) == [("p", 1, 8, 24)]
    assert _same_but(out[6], inst[6]) and out[6]["args"][0] is inst[6]["args"][0]
    assert _runs(out[6]["args"][1], compact) == [("p", 2, 0, 1), ("n", 2, 0, 2), ("n", 4, 2, 4)]
    assert _same_but(out[7], inst[7]) and _runs(out[7]["args"][0], compact) == [("n", 4, 3, 6), ("n", 2, 1, 4)]
    assert _same_but(out[8], inst[8]) and _runs(out[8]["args"][0], compact) == [("n", 2, 0, 1), ("n", 2, 2, 3)]
    assert _runs(inst[6]["args"][1], compact) == [("p", 2, 0, 1), ("n", 2, 2, 4), ("n", 4, 2, 4)]


@FORMS
def test_a_gather_that_steps_over_one_narrowed_linear_and_reads_the_next_has_that_read_moved(compact: bool):
    """Nodes 2 and 3 are both narrowed, to their words [4, 8) and [2, 6).  The gather over nodes 1 and 3 steps over node 2 and reads
    node 3: its first run stays and its second moves down by 2.  The second linear's weights hold two leaves a row."""
    inst = [_inst(0, "SiluMul_v2{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, "SiluMul_v2{N=8}", [[("p", 0, 4, 12)]], WORDS8, compact),
            _inst(2, "Gemm_v2{K=4,N=8}", [[("n", 0, 0, 4)], [("p", 1, 0, 32)]], WORDS8, compact),
            _inst(3, "Gemm_v1{K=2,N=8}", [[("n", 0, 0, 2)], [("p", 1, 32, 48)]], WORDS8, compact),
            _inst(4, "Concat_v1{N=8}", [[("n", 1, 2, 6), ("n", 3, 2, 6)]], WORDS8, compact),
            _inst(5, "TokenSelect_v4{V=12}", [[("n", 4, 0, 8), ("n", 2, 4, 8)]], "Value<32>", compact)]
    out, facts = CF.compiled(inst, compact)
    assert facts == {"sampling_events": 1, "dead": {}, "swapped": {}, "narrowed": {"Gemm_v1": 1, "Gemm_v2": 1}}
    assert len(out) == 6 and out[0] is inst[0] and out[1] is inst[1]
    assert _same_but(out[2], inst[2], spec="Gemm_v2{K=4,N=4}", out=WORDS4) and _runs(out[2]["args"][1], compact) == [("p", 1, 16, 32)]
    assert _same_but(out[3], inst[3], spec="Gemm_v1{K=2,N=4}", out=WORDS4) and _runs(out[3]["args"][1], compact) == [("p", 1, 36, 44)]
    assert _same_but(out[4], inst[4]) and _runs(out[4]["args"][0], compact) == [("n", 1, 2, 6), ("n", 3, 0, 4)]
    assert _same_but(out[5], inst[5]) and _runs(out[5]["args"][0], compact) == [("n", 4, 0, 8), ("n", 2, 0, 4)]


@FORMS
def test_a_narrowed_linear_with_a_bias_keeps_its_scope_and_cuts_both_column_arguments(compact: bool):
    """`GemmBias_v1` has two column arguments, its weights and its bias.  The spec's own N changes; the N inside its function-valued
    static, which happens to be the new width, is not the one read."""
    spec = 'batch:GemmBias_v1{K=4,N=8,DOT={"fn":"Dot_v1{N=3}"}}'
    inst = [_inst(0, "SiluMul_v2{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, spec, [[("n", 0, 0, 4)], [("p", 1, 0, 32)], [("p", 2, 0, 8)]], WORDS8, compact),
            _inst(2, "TokenSelect_v4{V=3}", [[("n", 1, 5, 8)]], "Value<32>", compact)]
    out, facts = CF.compiled(inst, compact)
    assert facts == {"sampling_events": 1, "dead": {}, "swapped": {}, "narrowed": {"GemmBias_v1": 1}}
    assert out[0] is inst[0]
    assert _same_but(out[1], inst[1], spec='batch:GemmBias_v1{K=4,N=3,DOT={"fn":"Dot_v1{N=3}"}}', out="Array<3,Value<16>>")
    assert [_runs(a, compact) for a in out[1]["args"]] == [[("n", 0, 0, 4)], [("p", 1, 20, 32)], [("p", 2, 5, 8)]]
    assert _same_but(out[2], inst[2]) and _runs(out[2]["args"][0], compact) == [("n", 1, 0, 3)]


@FORMS
def test_a_linear_read_at_both_ends_keeps_the_columns_between(compact: bool):
    """The columns kept are one range, from the lowest word read to the highest: words 0 and 1 and words 6 and 7 are the whole."""
    inst = [_inst(0, "SiluMul_v2{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, "Gemm_v2{K=4,N=8}", [[("n", 0, 0, 4)], [("p", 1, 0, 32)]], WORDS8, compact),
            _inst(2, "SiluMul_v2{N=2}", [[("n", 1, 0, 2)]], "Array<2,Value<16>>", compact),
            _inst(3, "TokenSelect_v4{V=4}", [[("n", 2, 0, 2), ("n", 1, 6, 8)]], "Value<32>", compact)]
    out, facts = CF.compiled(inst, compact)
    assert facts == {"sampling_events": 1, "dead": {}, "swapped": {}, "narrowed": {}}
    assert len(out) == 4 and all(got is was for got, was in zip(out, inst, strict=True))


REFUSED = {
    "a spec with no N": (
        "Gemm_v2{K=4}", [[("p", 1, 0, 32)]], WORDS8, "Gemm_v2{K=4}: a COLUMNS linear without its N static"),
    "an output of another length": (
        "Gemm_v2{K=4,N=8}", [[("p", 1, 0, 32)]], WORDS4,
        "Gemm_v2{K=4,N=8}: its output Array<4,Value<16>> is not N words, so its columns [2, 6) can't be narrowed"),
    "an output that is no array of words": (
        "Gemm_v2{K=4,N=8}", [[("p", 1, 0, 32)]], "Value<16>",
        "Gemm_v2{K=4,N=8}: its output Value<16> is not N words, so its columns [2, 6) can't be narrowed"),
    "weights that are not N rows": (
        "Gemm_v2{K=4,N=8}", [[("p", 1, 0, 12)]], WORDS8, "Gemm_v2{K=4,N=8}: argument 1 holds 12 leaves, not N = 8 rows"),
    "a bias that is not N rows": (
        "GemmBias_v1{K=4,N=8}", [[("p", 1, 0, 32)], [("p", 2, 0, 9)]], WORDS8,
        "GemmBias_v1{K=4,N=8}: argument 2 holds 9 leaves, not N = 8 rows"),
}


@FORMS
@pytest.mark.parametrize("case", list(REFUSED))
def test_a_linear_that_cannot_be_narrowed_is_refused_by_name(case: str, compact: bool):
    """A live linear of `prune.COLUMNS` read in [2, 6) of its eight words: its spec must hold a top-level N, its output must be N
    words, and each column argument must hold a whole number of leaves for each of the N rows."""
    spec, columns, out_type, message = REFUSED[case]
    inst = [_inst(0, "SiluMul_v2{N=4}", [[("p", 0, 0, 4)]], WORDS4, compact),
            _inst(1, spec, [[("n", 0, 0, 4)], *columns], out_type, compact),
            _inst(2, "TokenSelect_v4{V=4}", [[("n", 1, 2, 6)]], "Value<32>", compact)]
    with pytest.raises(ValueError) as refused:
        CF.compiled(inst, compact)
    assert str(refused.value) == message
