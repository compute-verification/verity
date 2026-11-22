"""verity_vllm.program.predict.window_circuit with Pearl+'s tables (`window_class.PearlTables`): the class circuit cut at its
linears, each linear but the LM head round s's call c, one `LinearARow_v3` unit held to its row of the call's A rows and one
`LinearOut_v3` unit reading its row of the call's y and handing it on in rows of W, and `window_rows` filling them with
the rows Pearl+'s call commits. Pinned on TINY's bf16 checkpoint on the H200 (`test_window_rows`) at `test_window_circuit`'s
class (2 slots, 4 prompt ids, 2 outputs: 5 positions, rounds 0 and 1) and grain, y handed on in rows of 16 words. A
position's cut linears are, of each of TINY's 2 layers, q/k/v (K = 64, N = 128), o (64, 64), gate/up (64, 256) and down
(128, 64), calls 0 to 7 in node order; its LM head (`Gemm_v2{64, 256}`, gate/up's shape) stays computed.
"""
from __future__ import annotations

import re
from types import SimpleNamespace as NS

import numpy as np
import pytest
from verity.circuits.types import Array, Value
from verity.commitments.periods import interface as IF
from verity.protocols.accounting.compute.pouw.verifier.kernel_schemes import IDENTIFIED
from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import (
    ROW_ALIGN,
)

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import (
    CLASS,
    GRAIN,
    GROUPS,
    PROVABLE,
    PROVABLE_CLASS,
    PROVABLE_GRAIN,
    WINDOW,
    _label,
    _source,
    held,
    root_kinds,
    state_row,
    wired,
    wired_matches_by_hand,
)
from tests.program.predict.test_window_circuit_int8 import GRAIN as INT8_GRAIN
from tests.program.predict.test_window_circuit_int8 import _registered as _int8
from tests.program.predict.test_window_circuit_tables import FIXED
from tests.program.predict.test_window_rows import (
    _registered,
)
from tests.program.predict.test_window_lowering import _int8_quant, _mixed
from verity_vllm.program.predict import scan_state as SS
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_cut as WK
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_builder import Row
from verity_vllm.program.predict.window_class import PearlTables, WindowClass

#: the scheme the remote application's Pearl+ serves under, on the H200
SCHEME = "pearl-c-h100-v1-h3"
PEARL = PearlTables(SCHEME, words=16)
#: each position's cut linears' K and N, calls 0 to 7 in node order: each layer's q/k/v, o, gate/up and down
CALLS = ((64, 128), (64, 64), (64, 256), (128, 64)) * 2
#: a call's rows by round: round 0 both slots' prompts of 4, round 1 a row a slot
M = {0: 8, 1: 2}


@pytest.fixture(scope="module")
def low():
    return WC.lower(_registered(), CLASS, None, GRAIN, tables=PEARL)


def _value_rows(circuit) -> dict[tuple[str, str, int], tuple[str, int]]:
    """Each row a read of Pearl+'s tables holds or fills, ``(template, port, unit)``, as its value and row."""
    out = {}
    for r in circuit.registered.reads:
        if r.param.startswith("pearl-"):
            lay = circuit.layouts[r.reader][0]
            words = dict(lay.rows + lay.outs)[r.port]
            for i, leaf in enumerate(r.source(range(r.units))):
                assert leaf % words == 0, (r, leaf)
                out[(r.reader, r.port, r.at + i)] = (r.param, int(leaf) // words)
    return out


def _calls(low) -> list[tuple[int, int, int, int]]:
    """Each slot's cut linears in node order: ``(slot, node, position, c)``, c its call, its index among its position's."""
    pos, out = WR.positions(low.fn), []
    for k, P in enumerate(low.programs):
        seen: dict[int, int] = {}
        for j in sorted(P.arows):
            seen[pos[j]] = seen.get(pos[j], -1) + 1
            out.append((k, j, pos[j], seen[pos[j]]))
    return out


def test_pearls_tables_are_named_by_round_and_call_at_the_rows_its_calls_commit():
    """Round s's call c's A rows and y are ``pearl-a/<s>/<c>`` and ``pearl-y/<s>/<c>``. A call's rows under the fixed load
    are round 0's every slot's prompt and a later round's a row a slot, and its A rows those then zero rows up to a multiple
    of Pearl+'s `ROW_ALIGN` (its m₁₆). Its y is handed on in rows of a positive number of words, and its scheme is one of
    PoUW's."""
    assert (PEARL.a(1, 3), PEARL.y(0, 5)) == ("pearl-a/1/3", "pearl-y/0/5")
    assert [PEARL.rows(CLASS, s) for s in range(3)] == [8, 2, 2] and PEARL.scheme in IDENTIFIED
    assert [PEARL.a_rows(cls, 0) for cls in (CLASS, WindowClass(4, 4, 2), WindowClass(2, 9, 2))] == [16, 16, 32] == [
        -(-m // ROW_ALIGN) * ROW_ALIGN for m in (8, 16, 18)]
    with pytest.raises(ValueError, match="a positive number of words, not 0"):
        PearlTables(SCHEME, words=0)
    with pytest.raises(ValueError, match="'pearl-x' is no PoUW scheme"):
        PearlTables(scheme="pearl-x")


def test_a_calls_y_is_handed_on_in_the_fewest_doublings_of_words_its_unit_fits(low):
    """Each call's W is the least of ``words`` doubled that divides its N and leaves its `LinearOut_v3` (its y row and
    N / W output rows) within the grain's rows and a proof block Flock proves: 16 for every call of TINY at 256 rows,
    gate/up's 256 words in rows of 32 at 12 rows and of 128 at 3, a 70B MLP's y of 28672 words at 256 rows in rows of 1024
    (rows of 128 are within the rows, but no proof block), and a call no W fits is refused by name."""
    assert [W for W in (128, 256, 512, 1024) if WK._hands_on(28672, W, 256)] == [1024]
    assert WK.linears(low.fn, 16, 256).calls == tuple((K, N, 16) for K, N in CALLS)
    assert WK.linears(low.fn, 16, 12).calls == tuple((K, N, 32 if N == 256 else 16) for K, N in CALLS)
    assert WK.linears(low.fn, 16, 3).calls == tuple((K, N, N // 2) for K, N in CALLS)
    with pytest.raises(Unexpressible, match=r"linears \(K, N\) \[\(64, 64\), \(64, 128\), \(64, 256\), \(128, 64\)\], whose "
                                            r"y no rows of Pearl\+'s 16 words doubled hand on in one unit within 1 rows that "
                                            r"Flock proves$"):
        WK.linears(low.fn, 16, 1)


def test_a_linear_takes_its_name_and_body_and_an_lm_head_its_selects_name_or_body(low):
    """A linear is cut only where it has `Gemm`'s name and a registered `Gemm`'s body: either without the other is refused by
    name. A GEMM a token select reads stays computed whether the select is known by its name or by its body, and a select's
    name over another body is refused."""
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.ops import prims as P
    from verity.circuits.ops.gemm import Gemm
    from verity.circuits.refs import Coll, concat

    nodes, honest = low.fn.body.nodes, WK.linears(low.fn, 16, 256)

    def swapped(j, fn):
        return NS(id=low.fn.id, body=NS(nodes=[NS(fn=fn if i == j else n.fn, args=n.args) for i, n in enumerate(nodes)]))

    def like(d, name, version, body):
        return CompositeDefinition(name, version, d.statics, d.signature, body, register=False, ports=d.ports)

    def zeros(B, S, *_):
        return Coll(Gemm.signature(S)[1], concat([B.call(P.const(16, 0)).refs] * S.N))

    g = next(j for j, n in enumerate(nodes) if n.fn.name == "Gemm" and j not in honest.heads)
    s = next(j for j, n in enumerate(nodes) if n.fn.name == "TokenSelect")
    gb, sb, sel = nodes[g].fn.bindings, nodes[s].fn.bindings, nodes[s].fn.defn
    assert WK.linears(swapped(s, bind(like(sel, "LogitArgmax", 1, sel.body), **sb)), 16, 256) == honest
    for fn, why in ((bind(like(Gemm, "Gemm", 2, zeros), **gb), "a unit named as Gemm whose body is no registered "
                                                                 "Gemm's at its statics is refused"),
                    (bind(like(Gemm, "Matmul", 1, Gemm.body), **gb), "a registered Gemm's body under another name is refused")):
        with pytest.raises(Unexpressible, match=re.escape(f"{fn.id}: {why}")):
            WK.linears(swapped(g, fn), 16, 256)
    other = bind(like(sel, "TokenSelect", sel.version, lambda B, S, *_: B.call(P.const(32, 0))), **sb)
    with pytest.raises(ValueError, match=re.escape(f"{other.id}: a unit named as the token select TokenSelect whose body is "
                                                   "no registered select's at its statics is refused")):
        WK.linears(swapped(s, other), 16, 256)


def test_a_position_takes_an_embeddings_name_and_body_and_an_int8_linear_its_quants_name_or_body(low):
    """The cut opens a position only at a unit with `Embedding`'s name and a registered `Embedding`'s body, and refuses an
    int8 linear at a unit with `Int8RowQuant`'s name and body: either without the other is refused by name."""
    nodes, honest = low.fn.body.nodes, WK.linears(low.fn, 16, 256)
    e = next(j for j, n in enumerate(nodes) if n.fn.name == "Embedding")
    g = next(j for j, n in enumerate(nodes) if n.fn.name == "Gemm" and j not in honest.heads)
    for j, fn, family, why in ((e, nodes[e].fn, "Embedding", "a position begins at each embedding"),
                               (g, _int8_quant(), "Int8RowQuant", "an int8 linear is known by its quant's name")):
        for mixed, refusal in _mixed(fn, family, why):
            swapped = NS(id=low.fn.id, body=NS(nodes=[NS(fn=mixed if k == j else n.fn, args=n.args)
                                                     for k, n in enumerate(nodes)]))
            with pytest.raises(Unexpressible, match=re.escape(refusal)):
                WK.linears(swapped, 16, 256)


def test_the_cut_replaces_every_linear_but_the_lm_head(low):
    """Each linear is one `LinearARow_v3{K, R}` unit, its x its writers' R rows (q/k/v's and gate/up's the norm's one, o's
    the attention's 4 heads', down's SiluMul's one), and one `LinearOut_v3{N, 16}` unit, a template each shape with every
    round's calls of it its members, and no `Gemm_v2` but the LM head's: its 16 slices a position, all of which the token
    selects' group units read, and its ``x0``, though gate/up is its shape. Every other template is the uncut circuit's at
    its count, and no input is free."""
    c = low.circuit
    n = {_label(fn): k for fn, k in c.templates}
    plain = {_label(fn): k for fn, k in WC.lower(_registered(), CLASS, None, GRAIN).circuit.templates}
    want = {"Gemm_v2{K=64,N=16}": 16 * 10, "Gemm_v2{K=64,N=1}": 10}
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            for t in (f"LinearARow_v3{{K={K},R={4 if ci % 4 == 1 else 1}}}", f"LinearOut_v3{{N={N},W=16}}"):
                want[t] = want.get(t, 0) + m
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == want
    assert {t: k for t, k in n.items() if not t.startswith(("Linear", "Gemm"))} == {
        t: k for t, k in plain.items() if not t.startswith("Gemm")}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in c.templates}
    g, cut = ids[GROUPS], c.cuts.get(ids[GROUPS], {}).get("x", 1)
    heads = {_source(c, g, "x", u, r)[:2] for u in range(n[GROUPS]) for r in range(cut)}
    assert {t for t, _ in heads} == {ids["Gemm_v2{K=64,N=16}"]} and len(heads) == 16 * 10
    assert IF.input_count(c.description(), **held(c))["free"] == 0


def test_each_cut_linear_reads_its_row_of_its_calls_a_rows_and_y(low):
    """Slot k's position p is round 0's row 4k + p below the prompt and round 1's row k past it (`FixedLoad.at`). Each cut
    linear, round s's call c, holds its A row (``o0``, x widened) to row i of the call's A rows and reads its ``y`` as row
    i of the call's y, i its position's row, and its readers read y as its unit's N / 16 output rows in order. Every real
    row of every value is read by exactly one unit, the A rows' filler rows by none, and the circuit's program takes them
    as its values, round by round and call by call the A rows then the y, then the scan state, ahead of its units'
    parameters. Each is FP32 words in ``row-seg/v1`` rows as Pearl+'s trees commit them (`ROW_FORMAT`): its program
    parameter 32-bit leaves, and the ``o0`` and ``y`` ports reading it ``row-seg/v1`` rows in their templates' statements."""
    c, want = low.circuit, {}
    for k, j, p, ci in _calls(low):
        P, (K, N) = low.programs[k], CALLS[ci]
        s, i = (0, 4 * k + p) if p < CLASS.prompt else (1, k)
        a, y = P.arows[j], P.ys[j]
        want[(a.tid, "o0", a.unit)], want[(y.tid, "y", y.unit)] = (PEARL.a(s, ci), i), (PEARL.y(s, ci), i)
        assert (a.words, y.words) == (2 * K, 2 * N)
        assert P.outs[j] == [(16 * o, 16 * (o + 1), Row(y.tid, y.unit, f"o{o}", 16)) for o in range(N // 16)]
    got = _value_rows(c)
    assert got == want
    rows: dict[str, list[int]] = {}
    for v, r in got.values():
        rows.setdefault(v, []).append(r)
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            assert sorted(rows[PEARL.a(s, ci)]) == sorted(rows[PEARL.y(s, ci)]) == list(range(m))
            assert (c.registered.params[PEARL.a(s, ci)], c.registered.params[PEARL.y(s, ci)]) == (16 * 2 * K, m * 2 * N)
    pearl = tuple(v for s in M for ci in range(len(CALLS)) for v in (PEARL.a(s, ci), PEARL.y(s, ci)))
    assert c.registered.values == pearl and c.registered.run == (SS.SCAN_STATE,)
    assert [n for n, _ in c.program()[0].fn.signature()[0][:32]] == list(c.registered.values)
    assert c.registered.formats == {v: WK.ROW_FORMAT for v in pearl} == {v: {"row_schema": IF.SEG, "word_bits": 32}
                                                                         for v in pearl}
    sig = dict(c.program()[0].fn.signature()[0][:32])
    for s, m in M.items():
        for ci, (K, N) in enumerate(CALLS):
            assert (sig[PEARL.a(s, ci)], sig[PEARL.y(s, ci)]) == (Array(16 * K, Value(32)), Array(m * N, Value(32)))
    seg, fns = IF.seg_ports(c.registered), {IF.descriptor_id(fn): fn for fn, _ in c.templates}
    for tid, port in {(t, p) for t, p, _u in want}:
        ins, outs = IF.unit_rows(fns[tid], c.cuts.get(tid), seg[tid])
        rows = {r[0]: r[1] for r in ins} | {p: s for (s, _j), (p, _w) in zip(outs, IF.layout(fns[tid]).outs)}
        assert seg[tid] == {port} and {p for p, s in rows.items() if s == IF.SEG} == {port}


def test_each_tables_readers_are_members_whose_registered_read_names_it(low):
    """The cut linears' templates are one a shape, 6 of 160 members, each member reading the table its registered read
    names: every member of each reads one, no other template reads one, and every table has readers."""
    c = low.circuit
    lin = {IF.descriptor_id(fn): n for fn, n in c.templates if fn.name in ("LinearARow", "LinearOut")}
    assert (len(lin), sum(lin.values())) == (6, 160)
    got, read = _value_rows(c), {}
    for tid, _port, u in got:
        read.setdefault(tid, set()).add(u)
    assert read == {t: set(range(n)) for t, n in lin.items()}
    assert {v for v, _row in got.values()} == set(c.registered.values) - {SS.SCAN_STATE}


def test_a_fixed_load_window_is_the_cut_class_circuit_and_any_other_window_is_refused():
    """The fixed load's window (both prompts of 4 at step 0, each decoded at step 1) is the cut class circuit;
    `test_window_circuit`'s WINDOW (a prompt of 3 chunked across two steps) is refused by name, since its rows aren't the
    public rows Pearl+'s tables are read at."""
    reg = _registered()
    fixed = WC.window_circuit(reg, S.compile_steps(reg, *S._session(*FIXED)), CLASS, GRAIN, tables=PEARL)
    assert IF.circuit_bytes(fixed.description()) == IF.circuit_bytes(
        WC.class_circuit(reg, CLASS, None, GRAIN, tables=PEARL).description())
    with pytest.raises(Unexpressible, match=r"not its class's fixed load .*, the rows Pearl\+'s tables are read at"):
        WC.window_circuit(reg, S.compile_steps(reg, *S._session(*WINDOW)), CLASS, GRAIN, tables=PEARL)


@pytest.mark.parametrize("registered, grain, tables, cls, why", [
    (_int8, INT8_GRAIN, PEARL, CLASS, r"Int8RowQuant_v1\{K=64[,}].*: an int8 linear, and Pearl\+'s tables cut BF16 linears"),
    (_registered, GRAIN, PEARL, WindowClass(2, 4, 2, lanes=1), ("Pearl\\+'s tables are read at a fixed load's rows, every "
                                                                "slot at the window's one boundary")),
    (_registered, GRAIN, PearlTables(SCHEME, words=48), CLASS, (r"linears \(K, N\) \[\(64, 64\), \(64, 128\), \(64, 256\), "
                                                        r"\(128, 64\)\], whose y no rows of Pearl\+'s 48 words doubled hand "
                                                        r"on in one unit within 256 rows that Flock proves"))])
def test_a_circuit_pearls_tables_cant_cut_is_refused(registered, grain, tables, cls, why):
    """Refused by name: an int8 linear (TINY's exact-int8 checkpoint), a class of two boundaries (its slots aren't all
    admitted at round 0), and y rows of a width no doubling of which divides a linear's N."""
    with pytest.raises(Unexpressible, match=why):
        WC.lower(registered(), cls, None, grain, tables=tables)


def test_the_cut_at_the_smallest_provable_class_is_proof_blocks_c_flock_proves():
    """At `test_window_circuit`'s smallest provable model (2 heads of 64 and a KV head, so q/k/v's N is 256), class and
    grain, with y handed on in rows of 64 words (128 bytes, as at Llama 3.2 1B's head dimension), every template of the cut
    circuit is a proof block Flock's composer takes (`interface.check_proof_block`): the A rows (256 and 512 words) and
    the y rows (2N words) are past 111 bytes."""
    c = WC.class_circuit(S._registered(PROVABLE), PROVABLE_CLASS, None, PROVABLE_GRAIN, tables=PearlTables(SCHEME, words=64))
    WC.check_provable(c)
    n = {_label(fn): k for fn, k in c.templates}
    P = PROVABLE_CLASS.positions * PROVABLE_CLASS.slots
    want = {"Gemm_v2{K=128,N=64}": 8 * P, "Gemm_v2{K=128,N=1}": P}
    for K, N, R in ((128, 256, 1), (128, 128, 2), (128, 512, 1), (256, 128, 1)) * 2:
        for t in (f"LinearARow_v3{{K={K},R={R}}}", f"LinearOut_v3{{N={N},W=64}}"):
            want[t] = want.get(t, 0) + P
    assert {t: k for t, k in n.items() if t.startswith(("Linear", "Gemm"))} == want
    assert all(IF.check_proof_block(fn, c.cuts.get(IF.descriptor_id(fn))) <= 27 for fn, _ in c.templates)


#: both slots' lanes of the scan state row at bucket 0: the fixed load's prompts of 4, each keeping 2 outputs
LANES = [(1, 4, 2, [17, 3, 250, 9]), (1, 4, 2, [5, 77, 42, 8])]
STATE = [state_row(CLASS.scan, LANES)] + [state_row(CLASS.scan, [(1, 1, 1, [99, 0, 0, 0])] * 2)] * (CLASS.buckets - 1)

def _widened(x: np.ndarray) -> np.ndarray:
    """bf16 words ``x`` widened to FP32, as u16 words."""
    return (np.asarray(x, np.uint32) << 16).astype("<u4").view("<u2")

@pytest.fixture(scope="module")
def pearl_wired():
    circuit = WC.class_circuit(_registered(), CLASS, None, GRAIN, tables=PEARL)
    return circuit, *wired(circuit)


@pytest.mark.slow
def test_the_tiny_pearl_class_circuit_is_one_wired_program_of_registered_rows(pearl_wired):
    """As `test_window_circuit`'s tiny class, with Pearl+'s table reads among its registered reads. Slow (about a minute to
    build): only the whole cut class holds Pearl+'s reads beside every wire kind."""
    circuit, prog, query = pearl_wired
    assert prog.annotations["free"] == []
    assert root_kinds(circuit, prog) == {"read": 41, "table read": 2, "opening": 3}
    assert prog.fn.gates == 110_288_572  # six TokenFeed_v2 units, each one constant bit its zero row is broadcast from
    assert query["params"]["templates"] == circuit.ids


@pytest.mark.slow
def test_the_tiny_pearl_class_wired_program_is_its_members_fed_by_hand(pearl_wired):
    """As `test_window_circuit`'s, on the cut class. Slow (about a minute): only the whole cut class holds every wire kind."""
    circuit, prog, _ = pearl_wired
    wired_matches_by_hand(circuit, prog)
