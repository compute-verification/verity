"""A two-stage interior whose replay block's parameters are values the interior registers (`interior_rows`, ``links``):
a replay block of three `A` units, unit j reading row j of its parameters ``x0``, ``best`` and ``c``, and two batches of
two `B` units, each batch reading its own rows of ``y0`` and ``ybest`` and rows 0 and 1 of ``i``, so `B`'s unit j reads
row j mod 2 of ``i``, over a replay sample of two of three spans. The interior commits no tree of its own at any of
these ports: each one's rows are rows of a value named as round 2 names the block's parameter. ``c`` and ``i`` are indices
the verifier supplies (a `Supply` with an ``index``), filled from the challenger's ``bytes`` sources; a replayed unit
whose index row is not the value's at its place is refused before anything is committed, for each index. The other four
are what the replay gives (`Operand`): the interior's own rows under fresh salts, which the stream registers by name,
rows, words and domain."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.commitments.windows import rows as CS
from verity.definitions.registry import dead_free as DF
from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import partition as P
from verity.protocols.spot.verifier.one_stage import registered as RG
from verity.protocols.spot.verifier.one_stage import registration as R

from verity_vllm.pipeline import boundary_rows as BR
from verity_vllm.pipeline import boundary_tables as BT
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import interior_rows as IR
from verity_vllm.pipeline import serving_rows as SRW

#: two templates of one shape, (x0, best, idx: one I32) -> I32: `A` stands for the unit reading ``c``, `B` for ``i``'s
A, B = bind(DF.TokenSelectCombineV2, V=64, C=64), bind(DF.TokenSelectCombineV2, V=32, C=32)
SPANS, Q, SALT, ROW, DELTA = 3, 2, bytes(range(32)), "interior-indices-test", 2.0 ** -40
#: `A`'s units a span, the rows of ``i`` a span, and the batches of `B` that each read them
NC, W, G = 3, 2, 2
CELLS = (NC, G * W)


def _block():
    """Each span a batch of `NC` `A` units on its own parameters, the j-th reading ``c``'s row j, and `G` batches of `W`
    `B` units, each batch on its `W` rows of ``y0`` and ``ybest`` and reading ``i``'s `W` rows."""
    (ap, ar), (bp, br) = A.signature(), B.signature()
    params = (*((n, Array(NC, t)) for n, t in (ap[0], ap[1])), ("c", Array(NC, ap[2][1])),
              *((f"y{n}", Array(G * W, t)) for n, t in (bp[0], bp[1])), ("i", Array(W, bp[2][1])))

    def body(B_, S_, x0, best, c, y0, ybest, i):
        return tuple_of(B_.batch(A, x0, best, c, axes=(0, 0, 0)), coll_concat(Array(G * W, br), [
            B_.batch(B, y0[g * W:(g + 1) * W], ybest[g * W:(g + 1) * W], i, axes=(0, 0, 0)) for g in range(G)]))

    return bind(CompositeDefinition("InteriorIndicesTest", 0, (),
                                    lambda S_: (params, Tuple(Array(NC, ar), Array(G * W, br))), body, register=False))


BLOCK = _block()
KEY = descriptor_id(BLOCK)
#: the indices, by block parameter: ``c``, a row a unit of `A`; ``i``, `W` rows, unit j of `B` reading row j mod `W`
SUPPLIES = (IR.Supply(P.template_id(A), "idx", f"{KEY}/c", NC, tuple(range(NC)), tuple(range(NC))),
            IR.Supply(P.template_id(B), "idx", f"{KEY}/i", W, tuple(j % W for j in range(G * W)), tuple(range(W))))
#: the block's other parameters, which the replay gives: a row a unit, of one word
OPERANDS = (*(IR.Operand(P.template_id(A), p, f"{KEY}/{p}", NC, tuple(range(NC)), 1) for p in ("x0", "best")),
            *(IR.Operand(P.template_id(B), p, f"{KEY}/y{p}", G * W, tuple(range(G * W)), 1) for p in ("x0", "best")))
LINKS = (*SUPPLIES, *OPERANDS)
MEMBERS = tuple(IR.named(CR.unit_member(SRW.template(P.template_id(t)), n)) for t, n in zip((A, B), CELLS, strict=True))


def _u32(xs) -> np.ndarray:
    return np.asarray(xs, dtype="<u4").view("<u2").reshape(len(xs), -1)


def _spans(seed: int) -> tuple[IR.Span, IR.Span]:
    """One span's `A` units and `B` units, each its Definition's on random bf16 words and the index its place gives it."""
    rng = np.random.default_rng(seed)
    got = []
    for fn, m, x in zip((A, B), MEMBERS, SUPPLIES, strict=True):
        idx = [x.index[a] for a in x.at]
        x0, best = rng.integers(0, 1 << 16, m.n).tolist(), rng.integers(0, 1 << 16, m.n).tolist()
        toks = [evaluate(fn, a, [b], [j])[0] for a, b, j in zip(x0, best, idx, strict=True)]
        data = {"rows": {"x0": np.asarray(x0, np.uint16)[:, None], "best": np.asarray(best, np.uint16)[:, None],
                         "idx": _u32(idx)}, "outs": {"o0": _u32(toks)}}
        got.append(IR.Span(f"s{seed}", m, data, [[0, m.n]], "p" * 64))
    return got[0], got[1]


def _boundary_window() -> CS.Binding:
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      "b0" * 64, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


def _first(links=LINKS):
    """A window stream over `A` and `B` taking ``links``' values a sampled span (`interior_rows.span_values`), a stand-in
    boundary registered and its first stage drawn (spans 0 and 2)."""
    names: dict[str, tuple[str, str]] = {}
    values = IR.link_values(links, Q)
    for phase, w in (("serve", _boundary_window()), ("interior", IR.interior_window(MEMBERS, CELLS, "00", [0], values=values))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    supplied = tuple(v for v, _per, _leaf in IR.span_values(links))
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test values"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s layout") for v in supplied for k in ("leaves", "words", "domain")))
    spec = S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items()))
                  + tuple(S.Value(v, None, "interior", registered=True) for v in supplied),
                  S.TwoStage(P.template_instance_query(BLOCK), f"rate:{Q}/{SPANS}", 1),
                  S.Check(tuple(P.program_digest(Program(u)) for u in (A, B)), P.template_instances_query((A, B)),
                          proof_deadline=60.0), public, b"interior-indices-test")
    draws = iter([[0, 2]])
    stream = S.Stream(spec, (A, B), span_values=IR.span_values(links), replay=BLOCK, budget=1, delta=DELTA,
                      challenger="challenger-test", clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    window = _boundary_window()
    c = CS.commit(window, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                            "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    reg = stream.register(CS.registration(c, {"stand_in": "boundary"}, None, registrar=R), CS.expected_ports(window),
                          leaf_layer=CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)}))
    return stream, reg, stream.select(reg)


@pytest.fixture(scope="module")
def staged():
    return _first()


@pytest.fixture(scope="module")
def operands():
    """The module's one stream over the four parameters the replay gives, its boundary registered and its first stage sampled.
    A stream takes one interior registration a first-stage sample, and this one's is the operand-port test's: a test that
    registers another interior of these links builds a stream of its own."""
    return _first(OPERANDS)


def _interior(reg, first, per, **kw) -> IR.Interior:
    return IR.interior(per, CELLS, reg.digest, first.units, row=ROW, **{"salt_key": SALT, "links": LINKS, **kw})


def test_an_index_port_is_rows_of_the_value_the_verifier_fills_from_its_own_bytes(staged):
    """The interior registers the two supplied values, the block's ``c`` and ``i`` as round 2 names them, Q·per rows each,
    row ``s·per + r`` the words ``[r, 0]``, ahead of the four the replay gives. The index ports are given by reference: `A`'s unit j of sampled span s reads
    ``c``'s row ``s·NC + j`` and `B`'s reads ``i``'s row ``s·W + j mod W``, where round 2 holds a linked port's read of an
    interior value. From the challenger's ``bytes`` sources alone the verifier fills the same rows
    (`registered.supplied_rows` on the stream's own interior program), and those rows at the public salts under the
    port's row format commit to the entry the interior registers. The stream registers the interior under the links'
    row formats (`link_schemas`). It refuses the supplied values' rows (`R7-supplied`) alone where the challenger's
    sources give ``c`` other indices, and where the supplied values' formats are missing, its verifier then committing
    their rows under row/v1 where the ports read row/v2."""
    stream, reg, first = staged
    it = _interior(reg, first, [_spans(s) for s in first.units])
    c, w = it.committed, it.committed.window
    prog = P.program_digest(w.program)
    assert P.program_digest(stream.interior_program(Q)) == prog
    got = {e["value"]: e for e in it.record["registered"]}
    assert list(got) == [x.value for x in LINKS] == [f"{KEY}/{p}" for p in ("c", "i", "x0", "best", "yx0", "ybest")]
    at = [c.given[m.port("idx")].positions for m in w.members]
    assert at == [(0, 1, 2, 3, 4, 5), (0, 1, 0, 1, 2, 3, 2, 3)]
    srcs = IR.sources(SUPPLIES, Q)
    assert srcs == {f"{KEY}/c": [[{"bytes": f"{r:08x}"}] for _ in range(Q) for r in range(NC)],
                    f"{KEY}/i": [[{"bytes": f"{r:08x}"}] for _ in range(Q) for r in range(W)]}
    for x, m in zip(SUPPLIES, w.members, strict=True):
        mine = RG.supplied_rows(stream.interior_program(Q), x.value, RG.sources(srcs)[x.value], it.record)
        assert mine == [[r, 0] for _ in range(Q) for r in range(x.per)]
        assert [list(r) for r in c.given[m.port("idx")].value.rows] == mine
        v = RG.Registered.commit(prog, x.value, mine, RG.supplied_salts(prog, x.value, len(mine)), schema=m.schema("idx"))
        assert v.entry() == {k: got[x.value][k] for k in v.entry()}
    assert IR.link_words(SUPPLIES) == {f"{KEY}/c": 2, f"{KEY}/i": 2}
    assert IR.span_values(SUPPLIES) == ((f"{KEY}/c", NC, IR.INDEX), (f"{KEY}/i", W, IR.INDEX))
    assert IR.link_schemas(MEMBERS, SUPPLIES) == {x.value: m.schema("idx") for x, m in zip(SUPPLIES, w.members)}
    assert IR.sources(LINKS, Q) == srcs
    for q in (1, SPANS):
        sized = IR.interior_window(MEMBERS, CELLS, "00", list(range(q)), values=IR.link_values(LINKS, q))
        assert P.program_digest(stream.interior_program(q)) == P.program_digest(sized.program)
    formats = IR.link_schemas(MEMBERS, LINKS)
    other = {**srcs, f"{KEY}/c": [[{"bytes": f"{r + 1:08x}"}] for _ in range(Q) for r in range(NC)]}
    for bad in (dict(supplied=other, schemas=formats), dict(supplied=srcs, schemas=IR.link_schemas(MEMBERS, OPERANDS))):
        with pytest.raises(S.Refused) as refused:
            IR.register(stream, first, it, registered=IR.link_words(LINKS), **bad)
        assert refused.value.verdict.codes == ("R7-supplied",)
    assert IR.register(stream, first, it, registered=IR.link_words(LINKS), supplied=srcs, schemas=formats).digest


def test_an_operand_port_is_rows_of_a_value_the_interior_registers_as_its_own(operands):
    """On a stream that takes only the four parameters the replay gives, the interior registers each as a value under
    the name round 2 gives the block's parameter: Q·per rows, row ``s·per + at[j]`` the row unit j of sampled span s
    replayed at the port, under salts that are not the public ones a supplied value takes, and the port is given by
    reference at those rows. The stream registers it, holding each value's name, rows, words and domain to its own
    entry (`link_words`, `link_schemas`); a challenger's copy that lacks one of the values, or holds it at another
    width, refuses it at R7."""
    stream, reg, first = operands
    per = [_spans(s) for s in first.units]
    it = _interior(reg, first, per, links=OPERANDS)
    c, w = it.committed, it.committed.window
    prog = P.program_digest(w.program)
    assert P.program_digest(stream.interior_program(Q)) == prog
    got = {e["value"]: e for e in it.record["registered"]}
    assert list(got) == [x.value for x in OPERANDS]
    assert IR.span_values(OPERANDS) == tuple((x.value, x.per, (1, 16)) for x in OPERANDS)
    assert IR.sources(OPERANDS, Q) == {}
    for x in OPERANDS:
        k = [m.template for m in w.members].index(x.template)
        ref = c.given[w.members[k].port(x.port)]
        took = np.concatenate([p[k].data["rows"][x.port] for p in per])
        assert ref.positions == tuple(range(Q * x.per))
        assert np.array_equal(np.asarray(ref.value.rows, np.uint16), took)
        assert ref.value.entry() == got[x.value] and got[x.value]["leaves"] == Q * x.per and got[x.value]["words"] == 1
        assert list(ref.value.salts) != RG.supplied_salts(prog, x.value, Q * x.per)
        assert ref.value.schema == w.members[k].schema(x.port)
    words, schemas = IR.link_words(OPERANDS), IR.link_schemas(MEMBERS, OPERANDS)
    for bad, codes in (({v: n for v, n in words.items() if v != OPERANDS[0].value}, ("R7-values",)),
                       ({**words, OPERANDS[0].value: 2}, ("R7-leaves", "R7-words", "R7-domain"))):
        with pytest.raises(S.Refused) as refused:
            IR.register(stream, first, it, registered=bad, schemas={v: s for v, s in schemas.items() if v in bad})
        assert refused.value.verdict.codes == codes
    assert IR.register(stream, first, it, registered=words, schemas=schemas).digest


def test_an_operand_row_is_the_one_row_every_unit_at_it_replayed(operands):
    """`B`'s ``best`` placed as ``i`` is, units j and j + `W` of a span on one row: where the two replayed the same row
    the value holds it once and both read it; where they did not, the interior refuses the later unit's row by name and
    commits nothing. A row no unit reads is the zero row."""
    _stream, reg, first = operands
    shared = IR.Operand(P.template_id(B), "best", f"{KEY}/ybest", W, tuple(j % W for j in range(G * W)), 1)
    per = [list(_spans(s)) for s in first.units]
    with pytest.raises(ValueError, match=f"takes at its 'best' a row that is not row [0-9] of {KEY}/ybest, which another"):
        _interior(reg, first, per, links=(*OPERANDS[:3], shared))
    for p in per:
        best = p[1].data["rows"]["best"].copy()
        best[W:] = best[:W]
        p[1] = replace(p[1], data={**p[1].data, "rows": {**p[1].data["rows"], "best": best}})
    wide = replace(OPERANDS[0], per=NC + 1)
    it = IR.interior(per, CELLS, reg.digest, first.units, row=ROW, salt_key=SALT, links=(wide, *OPERANDS[1:3], shared))
    c, w = it.committed, it.committed.window
    ref = c.given[w.members[1].port("best")]
    assert ref.positions == (0, 1, 0, 1, 2, 3, 2, 3) and len(ref.value.rows) == Q * W
    assert [list(r) for r in ref.value.rows] == np.concatenate([p[1].data["rows"]["best"][:W] for p in per]).tolist()
    x0 = c.given[w.members[0].port("x0")]
    assert x0.positions == (0, 1, 2, 4, 5, 6) and [x0.value.rows[r] for r in (3, 7)] == [(0,), (0,)]


def test_the_links_are_the_ones_round_2_derives_from_the_replay_block(staged):
    """`interior_rows.links` reads the block's links off its Definition as round 2 does: every parameter a unit reads,
    named ``<block>/<parameter>`` where round 1's record names its ports by template, with its rows a block and each
    unit's row; the two the verifier supplies by their indices, the rest the replay's. They are this module's by hand,
    in the body's order."""
    _stream, reg, _first_ = staged
    got = IR.links(P.population_program(BLOCK, SPANS), KEY, reg.record, MEMBERS, CELLS,
                   {x.value.rsplit("/", 1)[1]: x.index for x in SUPPLIES})
    assert len(got) == len(LINKS) and set(got) == set(LINKS)
    assert [x.value.rsplit("/", 1)[1] for x in got] == ["x0", "best", "c", "yx0", "ybest", "i"]
    bare = {"roots": [{"port": "ingress"}]}
    assert [x.value for x in IR.links(P.population_program(BLOCK, SPANS), KEY, bare, MEMBERS, CELLS)][:3] == ["x0", "best", "c"]
    with pytest.raises(ValueError, match="makes no links"):
        IR.links(P.population_program(BLOCK, SPANS), "Elsewhere_v1", reg.record, MEMBERS, CELLS)
    with pytest.raises(ValueError, match="a template the window has none of"):
        IR.links(P.population_program(BLOCK, SPANS), KEY, reg.record, MEMBERS[:1], CELLS[:1])


def test_an_interior_without_a_test_key_salts_its_own_ports_and_leaves_a_linked_ports_to_its_value(staged):
    """Serving gives no test key: every salt is fresh from the OS but a linked port's, whose rows keep their value's.
    So the interior commits, a supplied value's entry is the keyed interior's (its salts are public), an operand's
    root is new each time while its rows, words and domain stand, and a port of the interior's own gets a new root
    each time. The stream registers the operands' interior made that way."""
    _stream, reg, first = staged
    per = [_spans(s) for s in first.units]
    keyed, one, two = (_interior(reg, first, per, **kw) for kw in ({}, {"salt_key": None}, {"salt_key": None}))
    entries = [{e["value"]: e for e in it.record["registered"]} for it in (keyed, one, two)]
    assert all(list(got) == [x.value for x in LINKS] for got in entries)
    for x in SUPPLIES:
        assert entries[0][x.value] == entries[1][x.value] == entries[2][x.value]
    for x in OPERANDS:
        a, b = entries[1][x.value], entries[2][x.value]
        assert a["root"] != b["root"] and {**a, "root": None} == {**b, "root": None} == {**entries[0][x.value], "root": None}
    roots = [{r["port"]: r["root"] for r in it.record["roots"]} for it in (one, two)]
    w = one.committed.window
    own = [m.port(o) for m in w.members for o, _w in m.outs]
    assert own and all(roots[0][p] != roots[1][p] for p in own)
    for x in LINKS:
        k = [m.template for m in w.members].index(x.template)
        ref = one.committed.given[w.members[k].port(x.port)]
        assert ref.value.entry() == entries[1][x.value]
    stream, reg, first = _first(OPERANDS)
    it = _interior(reg, first, [_spans(s) for s in first.units], links=OPERANDS, salt_key=None)
    assert IR.register(stream, first, it, registered=IR.link_words(OPERANDS),
                       schemas=IR.link_schemas(MEMBERS, OPERANDS)).digest


def test_round_2_reads_an_index_port_at_the_name_the_served_records_ports_give_its_link(staged):
    """Round 2's read check (`second_stage._read_ok`) on the ports that read ``c`` and ``i``, under a round-1 record
    whose ports are named as the serve names them (`boundary_rows.PORTS`, bare, then a table a span). Round 2 names a
    link ``<block>/<parameter>`` where any port of that record has a ``/``, which a served table's name has and the two
    values' names have not. With a table it holds each read of the value this module's links register to its unit's row
    and refuses a read moved one row; without one the link is the bare parameter, and the same read is refused. The
    links `interior_rows.links` derives from the record carry the name round 2 will hold."""
    _stream, reg, first = staged
    program, q = P.population_program(BLOCK, SPANS), len(first.units)
    it = _interior(reg, first, [_spans(s) for s in first.units])
    c, w = it.committed, it.committed.window
    parts = (SS.Part(P.template_id(A), 0, NC), SS.Part(P.template_id(B), q * NC, G * W))
    entries = {e["value"]: e for e in it.record["registered"]}
    table = BT.name("b0" * 64, 4, 2, 0, 0)
    assert "/" in table and not any("/" in p for p in BR.PORTS)
    for ports, named in (((*BR.PORTS, table), True), (BR.PORTS, False)):
        served = {**reg.record, "roots": [{"port": port} for port in ports]}
        own = SS.Own(tuple(first.units), parts, it.record, served, (), links=SS.replay_links(program, KEY, served),
                     rows=SS.replay_rows(program, KEY))
        assert SS._by_template(served) is named
        for x in SUPPLIES:
            k = [m.template for m in w.members].index(x.template)
            at = list(c.given[w.members[k].port(x.port)].positions)
            e = {"value": x.value, **{f: entries[x.value][f] for f in ("root", "domain", "leaves")}, "reads": at,
                 "reads_from": parts[k].lo}
            assert SS._read_ok(own, named, parts[k], x.port, e) is named
            assert not SS._read_ok(own, named, parts[k], x.port, {**e, "reads": [at[0] + 1, *at[1:]]})
        derived = IR.links(program, KEY, served, MEMBERS, CELLS, {x.value.rsplit("/", 1)[1]: x.index for x in SUPPLIES})
        assert {x.value for x in derived} == ({x.value for x in LINKS} if named
                                              else {x.value.rsplit("/", 1)[1] for x in LINKS})


def test_one_value_is_read_by_ports_of_several_templates():
    """One block parameter each template reads its own rows of: `A`'s ``x0`` rows 0 to `NC` - 1 and `B`'s ``x0`` the
    `G`·`W` after them. The stream takes it once, the interior registers one value of Q·(`NC` + `G`·`W`) rows, each
    span's `A` rows then its `B` rows, and each port reads its rows of it by reference. Two ports that read one value at
    other counts of rows, or one as the verifier's and one as the replay's, are refused by name."""
    per_span = NC + G * W
    one = (IR.Operand(P.template_id(A), "x0", f"{KEY}/x", per_span, tuple(range(NC)), 1),
           IR.Operand(P.template_id(B), "x0", f"{KEY}/x", per_span, tuple(range(NC, per_span)), 1))
    assert IR.span_values(one) == ((f"{KEY}/x", per_span, (1, 16)),) and IR.link_words(one) == {f"{KEY}/x": 1}
    assert list(IR.link_schemas(MEMBERS, one)) == [f"{KEY}/x"]
    stream, reg, first = _first(one)
    per = [_spans(s) for s in first.units]
    it = _interior(reg, first, per, links=one)
    c, w = it.committed, it.committed.window
    assert [(e["value"], e["leaves"]) for e in it.record["registered"]] == [(f"{KEY}/x", Q * per_span)]
    a, b = (c.given[m.port("x0")] for m in w.members)
    assert a.value is b.value
    assert a.positions == (0, 1, 2, 7, 8, 9) and b.positions == (3, 4, 5, 6, 10, 11, 12, 13)
    assert np.array_equal(np.asarray(a.value.rows, np.uint16),
                          np.concatenate([s.data["rows"]["x0"] for p in per for s in p]))
    assert IR.register(stream, first, it, registered=IR.link_words(one), schemas=IR.link_schemas(MEMBERS, one)).digest
    for bad in (replace(one[1], per=per_span + 1), replace(one[1], words=2),
                IR.Supply(P.template_id(B), "idx", f"{KEY}/x", per_span, one[1].at, tuple(range(per_span)))):
        with pytest.raises(ValueError, match="read it as two values"):
            IR.span_values((one[0], bad))


@pytest.mark.parametrize("member, unit, index, why", [
    (0, 1, 2, f"unit 4 \\(sampled span 1\\) takes at its 'idx' a row that is not row 4 of {KEY}/c, the value the verifier"),
    (0, 2, 7, f"unit 5 \\(sampled span 1\\) takes at its 'idx' a row that is not row 5 of {KEY}/c, the value the verifier"),
    (1, 3, 0, f"unit 7 \\(sampled span 1\\) takes at its 'idx' a row that is not row 3 of {KEY}/i, the value the verifier"),
    (1, 0, 2, f"unit 4 \\(sampled span 1\\) takes at its 'idx' a row that is not row 2 of {KEY}/i, the value the verifier"),
])
def test_an_index_row_the_interior_does_not_register_is_refused_before_anything_is_committed(staged, member, unit, index, why):
    """For ``c`` and for ``i``: a replayed unit whose index is another row's of its value (`A`'s unit 1 reading 2, `B`'s
    unit 3 reading 0) or no row's at all (7 of ``c``'s three, 2 of ``i``'s two) takes a row the interior does not
    register at its place, so the interior refuses it by name and commits nothing."""
    _stream, reg, first = staged
    per = [list(_spans(s)) for s in first.units]
    s = per[1][member]
    idx = s.data["rows"]["idx"].copy()
    idx[unit] = _u32([index])[0]
    per[1][member] = replace(s, data={**s.data, "rows": {**s.data["rows"], "idx": idx}})
    with pytest.raises(ValueError, match=why):
        _interior(reg, first, per)


def test_a_link_names_a_port_of_the_window_and_places_each_of_its_units(staged):
    """A `Supply` is refused by name for a unit placed past its rows, indices that are not one a row, a port that holds no
    table's root or domain and no indices, and an `Operand` for a unit placed past its rows or rows of no words; and an
    interior for a template it lacks, a port that is none of its rows, a row of other words than its port's, a placement
    of other than its template's units a span, a port that reads two values, and a commitment made on the rows'
    device."""
    _stream, reg, first = staged
    c, i = SUPPLIES
    for bad, why in ((dict(at=(0, 1, NC)), "units reading rows"), (dict(index=(0, 1)), "2 indices for its 3 rows"),
                     (dict(index=None), "is no table read's root or domain")):
        with pytest.raises(ValueError, match=why):
            replace(c, **bad)
    for fields, why in ((dict(at=(0, 1, NC)), "units reading rows"), (dict(words=0), "rows of 0 words")):
        with pytest.raises(ValueError, match=why):
            replace(OPERANDS[0], **fields)
    per = [_spans(s) for s in first.units]
    for links, why in (((replace(c, template="Elsewhere_v1"), i), "a template the window has none of"),
                     ((replace(c, port="o0"), i), "which is none of its rows"),
                     ((replace(c, port="best"), i), "row is 1 words, not"),
                     ((c, replace(i, at=(0, 1), index=(0, 1))), "places 2 units, not its 4"),
                     ((c, i, replace(OPERANDS[0], words=2)), "row is 1 words, not"),
                     ((c, i, replace(c, value=f"{KEY}/again")), "reads two values")):
        with pytest.raises(ValueError, match=why):
            _interior(reg, first, per, links=links)
    with pytest.raises(ValueError, match="holds no value the interior registers"):
        _interior(reg, first, per, device=IR.DeviceCommit({}, {}, {}))
