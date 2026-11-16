"""verity_vllm.program.predict.window_lowering: what the class Program's lowering refuses, each by name, on TINY's class
Program at the class and grain `test_window_circuit` pins. A grain that cuts no unit's rows is refused through the class
circuit; a node the registered compiler doesn't emit (a call of another family, a read that cuts a writer's row, an
attention whose keys aren't its layer's) is one of the class Program's with what it reads replaced, lowered into a slot's
Program; and what no node reaches (a cut whose head isn't Calls on its query, its keys and earlier rows) is the cut's head
with that step replaced.
"""
from __future__ import annotations

import copy
import re
from dataclasses import replace

import pytest
from verity.circuits.defs import bind
from verity.circuits.refs import Affine, coll_concat, concat, runs
from verity.circuits.types import Array, Value

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN
from tests.program.predict.test_window_circuit_int8 import GRAIN as INT8_GRAIN
from tests.program.predict.test_window_circuit_int8 import _registered as _int8
from verity_vllm.program.predict import window_builder as WB
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_cut as WK
from verity_vllm.program.predict import window_lowering as WL
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.steps import RequestUnit, request_program
from verity_vllm.program.predict.window_builder import Row
from verity_vllm.program.predict.window_circuit import Grain, class_circuit
from verity_vllm.program.predict.window_class import GREEDY, PearlTables, Tables

NAME = "class Program 0"


@pytest.fixture
def low():
    """TINY's class Program lowered into one slot, unwired: its Program holds every node's rows."""
    return WC.lower(S._registered(), CLASS, 1, GRAIN, wired=False, provable=False)


def _node(low, family: str, k: int = 0):
    """The class Program's ``k``-th node of ``family``, with its index."""
    return [(j, n) for j, n in enumerate(low.fn.body.nodes) if n.fn.name == family][k]


def _program(low, fn=None, tables=None, cut=None) -> WL._Program:
    """A slot's Program of ``fn`` (the class Program by default) with nothing lowered into it, on a builder of its own."""
    p = low.programs[0]
    return WL._Program(WB._Builder(GRAIN, 1, False), fn or low.fn, CLASS, NAME, 0, p.scan, p.entry, tables, cut)


def _refused(why: str):
    return pytest.raises(Unexpressible, match=re.escape(why))


def test_a_program_takes_a_cuts_layout_with_pearls_tables_and_only_with_them(low):
    """A cut's layout without Pearl+'s tables, or the tables without a layout, is a caller's error."""
    pearl = PearlTables("pearl-c-h100-v1-h3", words=16)
    layout = WK.linears(low.fn, pearl.words, GRAIN.rows, provable=False)
    for tables, cut in ((None, layout), (pearl, None)):
        with pytest.raises(ValueError, match=re.escape(f"{NAME}: a cut's layout goes with Pearl+'s tables, and only with them")):
            _program(low, tables=tables, cut=cut)
    assert _program(low, tables=pearl, cut=layout).cut is layout


def test_a_program_that_reads_more_than_its_prompt_and_the_weights_is_refused_by_name(low):
    """A prefix-cache hit's Program reads its prefix, which no slot of the class holds."""
    hit = request_program(S._registered(), RequestUnit(-1, "", 3, CLASS.positions - 1, GREEDY, None, 0, C=2)).program.fn
    with _refused(f"{NAME}: its inputs are ['prompt', 'prefix', 'weights']; the window's lowering takes a prompt and the "
                  "weights record (no seed, prefix-cache hit, carried state or peers)"):
        _program(low, fn=hit)


def test_a_weight_read_is_one_run_of_its_ports_words_inside_one_weight(low):
    """A read past its weight's end spans two weights, and a port's read is one run of the weights record at the port's
    words: neither a shorter run nor a node's row."""
    p, nodes = low.programs[0], low.fn.body.nodes
    table, proj, gain = nodes[0].args[1].refs, nodes[2].args[1].refs, nodes[1].args[1].refs
    assert p.segments(table) == [("p", WB.WEIGHTS, 0, 16384)] and p.segments(proj) == [("p", WB.WEIGHTS, 16384, 24576)]
    with _refused(f"{NAME}: a weight read of leaves [16000, 17000) spans two weights"):
        p.segments(concat([table.slice(16000, 16384), proj.slice(0, 616)]))
    (seg,) = p.segments(gain)
    assert p.weight(gain, 64) == seg[2]
    with _refused(f"{NAME}: a weight read of {[seg]}, not one run of 32 weight words"):
        p.weight(gain, 32)
    with _refused(f"{NAME}: a weight read of [('n', 0, 0, 64)], not one run of 64 weight words"):
        p.weight(nodes[1].args[0].refs, 64)


def test_a_unit_reads_whole_rows_of_its_writers_and_a_head_or_kv_group_whole_rows_of_its_range(low):
    """o_proj's 64 words are eight rows of 8 at this grain: a read that starts or ends inside one cuts it, a parameter is no
    writer's row, and ranges of 4 words, or three ranges, are no whole rows."""
    p = low.programs[0]
    j, norm = _node(low, "RMSNormFusedCuda")
    x, w = norm.args[0].refs, norm.args[2].refs
    rows = p.rows(x)
    assert [r.words for r in rows] == [8] * 8 and p.rows(x.slice(8, 24)) == rows[1:3]
    with _refused(f"{NAME}: node {j - 1}'s words [4, 16) cut its row {rows[0]} at [0, 8)"):
        p.rows(x.slice(4, 16))
    with _refused(f"{NAME}: node {j - 1}'s words [8, 20) cut its row {rows[2]} at [16, 24)"):
        p.rows(x.slice(8, 20))
    with _refused(f"{NAME}: a unit reads parameter {WB.WEIGHTS} where a wire is due"):
        p.rows(w)
    assert p._split(x, 4) == [rows[0:2], rows[2:4], rows[4:6], rows[6:8]]
    with _refused(f"{NAME}: row {rows[0]} crosses from one of 16 ranges of 4 words into the next"):
        p._split(x, 16)
    with _refused(f"{NAME}: row {rows[0]} crosses from one of 3 ranges of 21 words into the next"):
        p._split(x, 3)


def test_a_return_range_that_is_no_output_row_is_refused(low):
    """A fused norm returns two rows of 64 words: a range is one of them or refused."""
    norm = _node(low, "RMSNormFusedCuda")[1].fn
    assert (WL._field(norm, 0, 64), WL._field(norm, 64, 128)) == (("o0", 64), ("o1", 64))
    for lo, hi in ((0, 32), (32, 64), (0, 128)):
        with _refused(f"{norm.id}: its return leaves [{lo}, {hi}) are not one of its output rows"):
            WL._field(norm, lo, hi)


def test_a_node_the_lowering_doesnt_take_is_refused_by_name(low):
    """A node that isn't a call, a call of a family the lowering doesn't know, and a node before the first embedding."""
    p, nodes = low.programs[0], low.fn.body.nodes
    heads, chunk = _node(low, "RoPE")[1].fn.body.nodes[0], _node(low, "TokenSelect")[1].fn.body.nodes[0]
    n = len(nodes)
    with _refused(f"{NAME}: node {n} is a batch, not a call"):
        p.lower(n, heads)
    with _refused(f"{NAME}: node {n} is {chunk.fn.id}, a family the window's lowering doesn't take ({sorted(WC.FAMILIES)})"):
        p.lower(n, chunk)
    with _refused(f"{NAME}: node 1 ({nodes[1].fn.id}) comes before its first position's embedding"):
        _program(low).lower(1, nodes[1])


def test_a_position_past_the_first_that_reads_the_prompt_is_refused(low):
    """The class Program reads its one prompt row at position 0: the first embedding lowered again is a sixth position's."""
    p, nodes = low.programs[0], low.fn.body.nodes
    assert p.position == CLASS.positions
    with _refused(f"{NAME}: position {CLASS.positions}'s token reads ('p', 'prompt'); a class Program reads its one prompt "
                  "row at position 0 and every other position's token from its predecessor's select"):
        p.lower(len(nodes), nodes[0])


def test_with_h1s_tables_a_token_that_isnt_one_u32_row_is_refused():
    """An input-ids table's row is one token id: an embedding past the prompt whose token is a norm's input row, 64 words,
    isn't one."""
    low = WC.lower(_int8(), CLASS, 1, INT8_GRAIN, wired=False, tables=Tables(64, 8, 6))
    p, nodes = low.programs[0], low.fn.body.nodes
    emb, hidden = _node(low, "Embedding", 1)[1], _node(low, "RMSNormTriton")[1].args[0]
    (row,) = p.rows(hidden.refs)
    assert row.words == 64
    with _refused(f"{NAME}: position {CLASS.positions}'s token is rows {(row,)}, not one u32 row an input-ids row holds"):
        p.lower(len(nodes), replace(emb, args=(hidden, emb.args[1])))


def test_a_grain_that_cuts_no_unit_within_its_rows_is_refused_by_name():
    """At 8 rows a unit the first fused norm is cut at its residual add, into parts its 64 elements aren't; at 2 rows a
    unit no cut of a `SiluMul`'s gate and up rows (16 words each) fits, a part of fewer words cutting them."""
    reg = S._registered()
    with _refused("RMSNormFusedCuda_v2{N=64,EPS=1e-05}: its 64 elements aren't parts of the grain's 48"):
        class_circuit(reg, CLASS, 1, Grain(gemm_words=1024, rows=8, select=64, norm=48), provable=False)
    with _refused("SiluMul_v2{I=128}: no cut of its gate and up rows holds a unit within 2 rows"):
        class_circuit(reg, CLASS, 1, Grain(gemm_words=1024, rows=2, select=64), provable=False)


def test_a_quantizer_whose_x_is_rows_that_dont_divide_its_values_is_refused():
    """o_proj's quantizer reads its 64 values as the attention's four head rows; one reading two of them and a norm's
    input row, three rows, is refused."""
    low = WC.lower(_int8(), CLASS, 1, INT8_GRAIN, wired=False)
    p, nodes = low.programs[0], low.fn.body.nodes
    quant, hidden = _node(low, "Int8RowQuant", 1)[1], _node(low, "RMSNormTriton")[1].args[0]
    x = quant.args[0]
    assert len(p.rows(x.refs)) == 4
    three = coll_concat(Array(96, Value(16)), [x[0:32], hidden])
    with _refused(f"{quant.fn.id}: its x is 3 writer rows, which don't divide its 64 values"):
        p.lower(len(nodes), replace(quant, args=(three,)))


def test_a_body_that_isnt_its_heads_is_refused(low):
    """`RoPE_v1{NHEADS=4}` is one batch of four `RoPEHead_v1`: another count or family isn't its body, and a token
    select's body is calls, no batch."""
    p = low.programs[0]
    rope, select = _node(low, "RoPE")[1].fn, _node(low, "TokenSelect")[1].fn
    assert p._heads(rope, "RoPEHead", 4) is rope.body.nodes[0].fn
    for fn, family, count in ((rope, "RoPEHead", 2), (rope, "AttentionHead", 4), (select, "TokenSelectChunk", 5)):
        with _refused(f"{fn.id}: its body is not {count} heads of one {family} Definition"):
            p._heads(fn, family, count)


def test_an_attention_whose_cut_isnt_calls_on_its_query_its_keys_and_earlier_rows_is_refused(low, monkeypatch):
    """`CUTS` names, for an attention version, the version whose heads' steps are Calls. Refused by name: a cut that doesn't
    take the attention's statics; one whose head's steps aren't Calls (v18 as its own cut); and `AttentionHead_v19` with
    its first Call reading half the query, with a row of its first Call as its return, or ending at its first Call, whose
    return is three rows."""
    att = _node(low, "Attention")[1].fn
    head = low.programs[0]._heads(bind(WL.CUTS[att.version], **att.bindings), "AttentionHead", 4)
    first, end = head.body.nodes

    def plan(cut=None, body=None):
        """`_plan` of ``att`` in a Program that has read no plan: under ``cut`` as its version's cut, or with ``body`` as
        the cut's head's body."""
        p = _program(low)
        with monkeypatch.context() as m:
            if cut is not None:
                m.setitem(WL.CUTS, att.version, cut)
            if body is not None:
                other = copy.copy(head)
                other.set_body(body)
                m.setattr(p, "_heads", lambda fn, family, count: other)
            return p._plan(att, 4)

    assert [fn.id for fn, _ports in plan()] == [first.fn.id, end.fn.id]
    rope = _node(low, "RoPE")[1].fn.defn
    with _refused(f"{att.id}: its heads are not `AttentionHead_v{rope.version}`'s ({rope.id} has no static parameter 'T')"):
        plan(cut=rope)
    own = att.body.nodes[0].fn
    k, step = next((k, n) for k, n in enumerate(own.body.nodes) if n.form != "call" and n.args)
    with _refused(f"{own.id}: Call {k} ({step.fn.id}) reads {runs(step.args[0].refs)} as "):
        plan(cut=att.defn)
    half = replace(first, args=(first.args[0][0:8], *first.args[1:]))
    with _refused(f"{head.id}: Call 0 ({first.fn.id}) reads [('p', 0, 0, 8)] as q, not a head's query, a key range or an "
                  "earlier Call's output row"):
        plan(body=replace(head.body, nodes=[half, end]))
    for body in (replace(head.body, ret=end.args[0].refs),
                 replace(head.body, nodes=[first], ret=Affine("n", 0, 0, first.out_leaves))):
        with _refused(f"{head.id}: its output is not its last Call's one row of 16 words"):
            plan(body=body)
    assert [fn.id for fn, _ports in plan()] == [first.fn.id, end.fn.id] and head.body.nodes == [first, end]


def test_an_attention_whose_keys_arent_its_layers_earlier_keys_and_one_more_is_refused(low):
    """Once the class Program is lowered its first layer holds five keys: its first attention, of one key, lowered again
    is not those five and one more."""
    p, nodes = low.programs[0], low.fn.body.nodes
    with _refused(f"{NAME}: an attention's k of 1 keys is not its layer's {CLASS.positions} earlier keys and one more"):
        p.lower(len(nodes), _node(low, "Attention")[1])


def test_a_first_key_block_whose_values_are_rows_of_two_widths_is_refused(low):
    """`AttnBlockFirst_v1` reads a value as one row; where a key's value is several rows of one width it is v2 at that
    slice width, and rows of two widths are no slice."""
    p = low.programs[0]
    first, ports = p._plan(_node(low, "Attention")[1].fn, 4)[0]
    assert (first.name, first.version) == ("AttnBlockFirst", 1)

    def values(*words):
        return WL._Keys(rows=[(tuple(Row("w", i, "o0", w) for i, w in enumerate(words)),)])

    assert p._first(first, ports, values(16)) is first
    assert p._first(first, ports, values(8, 8)).id == first.id.replace("_v1{", "_v2{").replace("}", ",VS=8}")
    with _refused(f"{first.id}: its keys' values are rows of [4, 12] words, not one slice width"):
        p._first(first, ports, values(12, 4))


def test_a_select_that_isnt_chunks_and_a_combine_over_a_gemms_logits_is_refused(low):
    """`TokenSelect_v4{V=1}` is no chunk and no combine; a select reads its logits as the LM head's GEMM's rows (its
    ``x0`` is that GEMM's first column), so one over an attention's rows is refused; and a select comes after its
    Program's first token, whose scan unit carries the slot's head and occupancy."""
    p, nodes = low.programs[0], low.fn.body.nodes
    j, select = _node(low, "TokenSelect")
    n = len(nodes)
    with _refused("TokenSelect_v4{V=1}: its body is not chunks and one `TokenSelectCombine_v2`"):
        p.lower(n, replace(select, fn=bind(select.fn.defn, V=1)))
    at = _node(low, "Attention")[0]
    heads = nodes[at + 1].args[0]
    assert p.segments(heads.refs) == [("n", at, 0, 64)]
    logits = coll_concat(select.args[0].type, [heads] * 4)
    with _refused(f"{select.fn.id}: its logits are node {at}'s, not a GEMM's"):
        p.lower(n, replace(select, args=(logits,)))
    with _refused(f"{select.fn.id}: a token select before its Program's first position's token"):
        _program(low)._TokenSelect(j, select)
