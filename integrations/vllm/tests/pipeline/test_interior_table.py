"""A two-stage interior whose replay block reads history tables (`interior_rows`, ``tables``): each span two
`EmbeddingRead_v1{V=8,H=2}` reads of the table round 1 served for it, over a replay sample of two of three spans. Round 1
(a stand-in boundary) registers each span's table as a port of its record, of its class's C strings (P=4, O=2: 6 of 8
leaves, `boundary_tables.class_rows`), its rows past its length L zero rows (`boundary_tables.fill`). The interior commits no
table: each read's ``root`` and ``domain`` rows are rows of two supplied values, named as round 2 names the replay
block's ``root`` and ``domain`` parameters, which the challenger resolves through its own `history`, each sampled span's
table named from its own sample under round 1's root and read from the strings the prover publishes. Its key and its
opening (``tok``, ``row``, ``salt``, ``path``) are the block's parameters too, which the replay gives: values the
interior registers as its own, a row a read. An interior is
refused before anything is committed where a read doesn't open its table, where a table isn't the one round 1 registered
under its name, and where a table holds other than C strings.

Every program over the reads holds their SHA-512 openings, so building, digesting or checking one is far past a quick
test's seconds. The module builds none at import (`_boundary`, `_program`), its tests share one round 1 (`staged`) and
one interior over it (`honest`), and each test that needs a program is marked slow, with what it alone catches. Each
of those takes a minute or more, or waits for the shared round 1 and interior, minutes to build, so it also runs only
where ``VERITY_VLLM_INTERIOR_TABLE=1`` asks (`_opt_in`)."""

from __future__ import annotations

import functools
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, registered as RG, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.commitments import history as HT
from verity.ml.commitments.gates import embedding as E
from verity.ml.windows import rows as CS
from verity.ml.windows.interface import digest_words
from verity_vllm.pipeline import boundary_tables as BT, chunk_rows as CR, interior_rows as IR, serving_rows as SRW

RD = E.read(8, 2)
SPANS, Q, CELLS, SALT, ROW, DELTA = 3, 2, 2, bytes(range(32)), "interior-table-test", 2.0 ** -40
RUN, PROMPT, OUTPUTS, LAYER, CAP = "b0" * 64, 4, 2, 0, 8
C = BT.class_rows(PROMPT, OUTPUTS, CAP)
#: each span's length L: spans 0 and 2 are drawn
LENGTHS = (5, 4, 3)
#: sampled span s's two keys: span 0's key 5 and span 2's key 4 are past their lengths, so each opens a fill row
TOKS = ((1, 5), (0, 4))
#: core's history-read vector: a table of 5 rows of 4 words at 8 leaves, and the supplied value its reads take
VECTOR = Path(__file__).resolve().parents[4] / "verity/core/protocols/spot/tests/fixtures/history_read_vector.json"


def _block():
    """Each span a batch of two `RD` reads on its own parameters, one root and one domain id a span."""
    params, ret = RD.signature()
    return bind(CompositeDefinition(
        "InteriorTableTest", 0, (),
        lambda S_: (tuple((n, t if n in IR.TABLE else Array(CELLS, t)) for n, t in params), Array(CELLS, ret)),
        lambda B, S_, *args: B.batch(RD, *args, axes=tuple(None if n in IR.TABLE else 0 for n, _t in params)),
        register=False))


BLOCK = _block()
#: the reads' root and domain, the verifier's: one row a sampled span, which both of its reads take, under the names round
#: 2 gives the block's parameters where the boundary record names its ports by template
SUPPLIES = tuple(IR.Supply(P.template_id(RD), p, f"{descriptor_id(BLOCK)}/{p}", 1, (0,) * CELLS) for p in IR.TABLE)
MEMBER = IR.named(CR.unit_member(SRW.template(P.template_id(RD)), CELLS))
#: the reads' key and opening, the replay's: a row a read, of its port's words
OPERANDS = tuple(IR.Operand(P.template_id(RD), p, f"{descriptor_id(BLOCK)}/{p}", CELLS, tuple(range(CELLS)), w)
                 for p, w in MEMBER.rows if p not in IR.TABLE)
LINKS = (*SUPPLIES, *OPERANDS)
NAMES = tuple(BT.name(RUN, PROMPT, OUTPUTS, s, LAYER) for s in range(SPANS))
#: a program digest for a table no program reads: round 1's own (`_program`) takes its boundary's program to derive
UNREAD = "c5" * 64


def _opt_in(takes: str) -> pytest.MarkDecorator:
    """The skip of a test that runs only under ``VERITY_VLLM_INTERIOR_TABLE=1``; its reason says how long it ``takes``."""
    return pytest.mark.skipif(os.environ.get("VERITY_VLLM_INTERIOR_TABLE") != "1",
                              reason=f"VERITY_VLLM_INTERIOR_TABLE=1: {takes}")


#: a test on the module's one round 1 (`staged`) or its interior (`honest`)
SHARED = _opt_in("about 5 minutes on a CPU to build the module's round 1 and interior over the table reads once, then up to "
                 "2 minutes a test")


@functools.cache
def _boundary() -> CS.Binding:
    """Round 1's stand-in boundary, one member of the replay block's `SPANS` units. Built on first use, never at
    import: every test worker imports the module to collect it, and the boundary's program holds every span's reads."""
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      RUN, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


@functools.cache
def _program() -> str:
    """Round 1's program, which its tables' domains bind: the boundary's digest, derived once."""
    return P.program_digest(_boundary().program)


def _tables(lengths=LENGTHS, seed: int = 0, fill: bool = True, program: str | None = None) -> list[HT.HistoryTable]:
    """Round 1's table for each span: its ``lengths[s]`` rows, filled to C (`boundary_tables.fill`), each under a salt of
    its own. ``program``: the tables', round 1's (`_program`) where none is given."""
    program = _program() if program is None else program
    rng = np.random.default_rng(seed)
    out = []
    for name, n in zip(NAMES, lengths, strict=True):
        t = HT.HistoryTable(program, name, 2, CAP)
        rows = rng.integers(0, 1 << 16, (n, 2), dtype=np.uint16)
        rows = BT.fill(rows, PROMPT, OUTPUTS, CAP) if fill else rows
        t.append(rows, [rng.bytes(192) for _ in rows])
        out.append(t)
    return out


def _spec() -> S.Spec:
    """The served values (the boundary's and each span's table) and the interior's: its ports, and the block's
    parameters, registered: the reads' two supplied values and the four the replay gives."""
    names = {m.port(p): (m.schema(p), "serve") for m in _boundary().members for p in CS.port_leaves(m)}
    names |= {n: (HT.SCHEMA, "serve") for n in NAMES}
    w = IR.interior_window(MEMBER, CELLS, "00", [0], values=IR.link_values(LINKS, Q))
    names |= {m.port(p): (m.schema(p), "interior") for m in w.members for p in CS.port_leaves(m)}
    supplied = tuple(x.value for x in LINKS)
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test values"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s layout") for v in supplied for k in ("leaves", "words", "domain")))
    return S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items()))
                  + tuple(S.Value(v, None, "interior", registered=True) for v in supplied),
                  S.TwoStage(P.template_instance_query(BLOCK), f"rate:{Q}/{SPANS}", 1),
                  S.Check((P.program_digest(Program(RD)),), P.template_instances_query([RD]), proof_deadline=60.0),
                  public, b"interior-table-test")


def _first(tables):
    """A window stream over `RD` taking the reads' values a sampled span (`interior_rows.span_values`), and round 1, a stand-in boundary
    whose record registers each span's table as a port, registered and its first stage drawn (spans 0 and 2)."""
    draws = iter([[0, 2]])
    boundary = _boundary()
    stream = S.Stream(_spec(), [RD], span_values=IR.span_values(LINKS), replay=BLOCK, budget=1, delta=DELTA,
                      challenger="challenger-test", clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    c = CS.commit(boundary, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                              "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    rec = CS.registration(c, {"stand_in": "boundary"}, None, registrar=R)
    rec["roots"] += [{"port": t.value, "root": t.root.hex(), "leaves": t.leaves, "domain": t.domain.domain_id.hex()}
                     for t in tables]
    ports = CS.expected_ports(boundary) | {t.value: {"leaves": t.leaves, "domain": t.domain.domain_id.hex()} for t in tables}
    reg = stream.register(rec, ports, leaf_layer=CS.leaf_layer({m.key: c.public_body(k)
                                                                for k, m in enumerate(boundary.members)}))
    return stream, reg, stream.select(reg), rec


@pytest.fixture(scope="module")
def staged():
    """Round 1's tables and `_first`'s stream once for the module, for the tests that read its record, digest and drawn
    units. None has its stream accept an interior: a stream takes one a draw, so the test that does stages its own."""
    tables = _tables()
    return (*_first(tables), tables)


@pytest.fixture(scope="module")
def honest(staged):
    """The interior over the tables `staged`'s round 1 served, under the test key, once for the module: the tests that
    take it only read it."""
    _stream, reg, first, rec, tables = staged
    return _interior(reg, first, rec, [tables[u] for u in first.units])


def _spans(tables) -> list[IR.Span]:
    """Each sampled span's two reads, opened against its table."""
    return [IR.Span(f"s{s}", MEMBER, IR.table_reads(t, TOKS[s]), [[0, CELLS]], "p" * 64) for s, t in enumerate(tables)]


def _interior(reg, first, rec, drawn, spans=None, **kw) -> IR.Interior:
    return IR.interior(_spans(drawn) if spans is None else spans, CELLS, reg.digest, first.units, row=ROW,
                       **{"salt_key": SALT, "tables": drawn, "served": rec, "positions": C, "links": LINKS, **kw})


def _with(span: IR.Span, kind: str, port: str, unit: int, value) -> IR.Span:
    a = span.data[kind][port].copy()
    a[unit] = value
    return replace(span, data={**span.data, kind: {**span.data[kind], port: a}})


def _history(rec, first, it) -> dict:
    """The challenger's `history`: each sampled span's table named from its own sample."""
    c = it.committed
    bodies = {m.key: c.public_body(k) for k, m in enumerate(c.window.members)}
    return IR.history(rec, [NAMES[u] for u in first.units], 2, C, it.published, bodies)


@_opt_in("about 12 minutes on a CPU, the verifier checking the reads' interior program three times on a stream of "
         "the test's own")
@pytest.mark.slow
def test_an_interior_opens_the_tables_round_1_served_and_the_challenger_resolves_its_reads_to_them():
    """Slow, on a stream of its own (a stream takes one interior a draw, and this test's accepts one): it catches the
    reads' interior and the verifier disagreeing on a supplied row's bytes, format or order, which shows only where the
    verifier checks the reads' own interior program, here three times.

    The interior registers the block's parameters as round 2 names them, the two supplied values ``root`` and
    ``domain`` and then the reads' key and opening, a row a read, the replayed ones (`Operand`), and
    publishes each sampled span's table's C strings. From the challenger's `history` alone (its own sample's names, round 1's record and
    the published strings) `history.roots_of`, the verifier's mirror, recomputes round 1's root and domain for each, and
    `history.resolve` turns the reads' sources into those bytes, which are `supplied`'s rows. The reads' ``root`` and
    ``domain`` rows are given by reference, rows (0, 0, 1, 1) of those values, the key's and the opening's rows
    (0, 1, 2, 3) of theirs, and the commitment is the one the same rows given in full under the values' salts makes. The stream takes the interior's roots, names and history (the
    public-items rule and R7-history pass), and its interior program takes each value a `Digest` a row a sampled span, so
    its verifier reads each resolved source in a `Digest` port's order: its 512 leaves packed into u16 words are the
    row the interior registers. Under the links' row formats (`link_schemas`) it registers the interior, and refuses
    sources that name the sampled spans' tables in another order (`R7-supplied`). Without the supplied values' formats it
    refuses their rows (`R7-leaves`, `R7-domain`, `R7-supplied`): it counts and commits them as row/v1 rows of 32 one-bit
    leaves, where the reads hold a `Digest` a row under their port's row/v2; the replay's own values pass."""
    tables = _tables()
    stream, reg, first, rec = _first(tables)
    drawn = [tables[u] for u in first.units]
    it = _interior(reg, first, rec, drawn)
    c, w = it.committed, it.committed.window
    prog, names = P.program_digest(w.program), [NAMES[u] for u in first.units]
    assert P.program_digest(stream.interior_program(Q)) == prog
    got = it.record["registered"]
    assert [e["value"] for e in got] == [x.value for x in LINKS]
    assert [x.value for x in SUPPLIES] == [f"{descriptor_id(BLOCK)}/{p}" for p in IR.TABLE]
    assert [x.port for x in OPERANDS] == ["tok", "row", "salt", "path"]
    assert it.published == {t.value: [bc.hex() for bc in t.bcs] for t in drawn}
    assert all(len(v) == C for v in it.published.values())
    hist = _history(rec, first, it)
    roots = HT.roots_of(it.leaf_layer, hist)
    served = {r["port"]: r for r in rec["roots"]}
    assert roots == {n: (bytes.fromhex(served[n]["root"]), bytes.fromhex(served[n]["domain"])) for n in names}
    resolved = HT.resolve(roots, IR.sources(SUPPLIES, Q, names))
    sup = IR.supplied(prog, SUPPLIES, Q, [t.entry() for t in drawn])
    for v, of in zip(sup, IR.TABLE, strict=True):
        assert [[s["bytes"] for s in row] for row in resolved[v.value]] == [[roots[n][of == "domain"].hex()] for n in names]
        assert [list(r) for r in v.rows] == [list(digest_words(roots[n][of == "domain"])) for n in names]
        e = next(e for e in got if e["value"] == v.value)
        assert {k: e[k] for k in v.entry()} == v.entry()
        bits = RG.supplied_rows(stream.interior_program(Q), v.value, RG.sources(resolved)[v.value], it.record)
        assert [[sum(b << j for j, b in enumerate(r[16 * k:16 * k + 16])) for k in range(32)] for r in bits] == \
            [list(r) for r in v.rows]
    assert IR.span_values(SUPPLIES) == tuple((x.value, 1, IR.DIGEST) for x in SUPPLIES)
    assert IR.link_schemas(MEMBER, LINKS) == {x.value: w.members[0].schema(x.port) for x in LINKS}
    m = w.members[0]
    by_ref = {x.port: c.given[m.port(x.port)] for x in LINKS}
    assert all(by_ref[p].positions == (0, 0, 1, 1) for p in IR.TABLE)
    spans = _spans(drawn)
    for x in OPERANDS:
        v = by_ref[x.port]
        assert v.positions == (0, 1, 2, 3) and v.value.entry() == next(e for e in got if e["value"] == x.value)
        assert np.array_equal(np.asarray(v.value.rows, np.uint16), np.concatenate([s.data["rows"][x.port] for s in spans]))
        assert IR.span_values([x]) == ((x.value, CELLS, (dict(MEMBER.rows)[x.port], 16)),)
    data = [{"rows": {p: np.concatenate([s.data["rows"][p] for s in spans]) for p, _ in m.rows},
             "outs": {p: np.concatenate([s.data["outs"][p] for s in spans]) for p, _ in m.outs},
             "refs": np.repeat(np.arange(m.n)[:, None], len(m.rows), axis=1)}]
    full = CS.commit(w, data, salt_key=SALT, salts_of={m.port(p): CS.value_salts(v.value, v.positions)
                                                       for p, v in by_ref.items()})
    assert full.roots == c.roots
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, history=hist, registered=IR.link_words(LINKS),
                    supplied=IR.sources(LINKS, Q, names), schemas=IR.link_schemas(MEMBER, OPERANDS))
    assert refused.value.verdict.codes == ("R7-leaves", "R7-domain", "R7-supplied")
    formats = IR.link_schemas(MEMBER, LINKS)
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, history=hist, registered=IR.link_words(LINKS),
                    supplied=IR.sources(LINKS, Q, names[::-1]), schemas=formats)
    assert refused.value.verdict.codes == ("R7-supplied",)
    assert IR.register(stream, first, it, history=hist, registered=IR.link_words(LINKS),
                       supplied=IR.sources(LINKS, Q, names), schemas=formats).digest


@SHARED
@pytest.mark.slow
def test_a_table_round_1_did_not_register_is_refused(staged, honest):
    """A table that differs from the served one in one row: the interior refuses it by name before anything is committed,
    the challenger's `history` over its strings recomputes another root than round 1's, which `history.roots_of`
    refuses, and so does the stream (`R7-history`). Slow: the stream's own refusal is the verifier's, over the reads'
    interior program; without it a prover's strings for another table would be held only by the Python mirror. The
    stream is `staged`'s: it refuses, and no other test asks it anything."""
    stream, reg, first, rec, tables = staged
    drawn = [tables[u] for u in first.units]
    served = drawn[1]
    other = HT.HistoryTable(_program(), served.value, 2, CAP)
    rows = np.array(served.rows)
    rows[1, 0] ^= 1
    other.append(rows, served.salts)
    assert sum(a != b for a, b in zip(other.bcs, served.bcs, strict=True)) == 1
    swapped = [drawn[0], other]
    with pytest.raises(ValueError, match="is not the one round 1 registered under its name"):
        _interior(reg, first, rec, swapped)
    it = honest
    published = {**it.published, other.value: [bc.hex() for bc in other.bcs]}
    c = it.committed
    bodies = {m.key: c.public_body(k) for k, m in enumerate(c.window.members)}
    names = [NAMES[u] for u in first.units]
    hist = IR.history(rec, names, 2, C, published, bodies)
    with pytest.raises(ValueError):
        HT.roots_of(it.leaf_layer, hist)
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, history=hist, registered=IR.link_words(LINKS),
                    supplied=IR.sources(LINKS, Q, names), schemas=IR.link_schemas(MEMBER, OPERANDS))
    assert refused.value.verdict.codes == (R.HISTORY,)


@SHARED
@pytest.mark.slow
def test_every_table_of_a_class_publishes_its_c_strings_whatever_its_length(staged, honest):
    """Two spans of one class, of lengths 5 and 3, each publish C = 6 strings, so the strings never show a span's length:
    a read past its length opens a fill row (the zero row under a salt of its own) and one past C the zero row under the
    zero salt, each the zero row. A table of other than C strings is refused by name. Slow: what is published is the
    interior's own, read off one built over a registered round 1 and its drawn spans, which the tables alone don't give."""
    _stream, reg, first, rec, tables = staged
    drawn = [tables[u] for u in first.units]
    assert [len(t.rows) for t in drawn] == [C, C] and [LENGTHS[u] for u in first.units] == [5, 3]
    it = honest
    assert [len(v) for v in it.published.values()] == [C, C]
    zero = np.zeros(2, np.uint16)
    for t in drawn:
        for k in (C - 1, C, CAP - 1):
            r = IR.table_reads(t, [k])
            assert np.array_equal(r["rows"]["row"][0], zero) and np.array_equal(r["outs"]["o0"][0], zero)
        assert t.salts[C - 1] != bytes(192)
    assert IR.table_reads(drawn[0], [C])["rows"]["salt"][0].tolist() == [0] * 96
    short = _tables(fill=False)
    with pytest.raises(ValueError, match=f"holds 5 strings, not its class's {C}"):
        _interior(reg, first, rec, [short[u] for u in first.units])


@SHARED
@pytest.mark.slow
def test_a_table_reads_interior_without_a_test_key_registers_the_same_root_and_domain(staged, honest):
    """Serving gives no test key: the reads' interior commits with every salt fresh from the OS but its linked ports',
    whose rows keep their value's. Its ``root`` and ``domain`` register the entries the keyed interior's do (a supplied
    value's salts are public), and each value the replay gives the same rows, words and domain under a new root. Slow: it
    commits the reads' interior as serving does, the one path every other test's key replaces."""
    _stream, reg, first, rec, tables = staged
    drawn = [tables[u] for u in first.units]
    keyed, served = honest, _interior(reg, first, rec, drawn, salt_key=None)
    a, b = ({e["value"]: e for e in it.record["registered"]} for it in (keyed, served))
    assert list(b) == [x.value for x in LINKS]
    for x in SUPPLIES:
        assert a[x.value] == b[x.value]
    for x in OPERANDS:
        assert a[x.value]["root"] != b[x.value]["root"] and {**a[x.value], "root": None} == {**b[x.value], "root": None}


def test_a_tables_bytes_and_its_reads_rows_are_cores_history_read_vector():
    """Core's history-read vector, rebuilt: its table's strings, placeholder, domain and root are
    `history.HistoryTable`'s on its rows and salts, a served table's domain (576 words at 2^17 leaves) is its
    ``domain_2_17``, and its supplied value's two rows (``Array<2, Digest>``, one bit a leaf) are `supplied`'s rows of
    the table's root and domain id, each u16 word least significant bit first: the bytes the reads take are core's. The
    vector commits those rows as 512 one-bit words; the reads' port takes them as 32 u16 words."""
    v = json.loads(VECTOR.read_text())
    t = v["table"]
    h = HT.HistoryTable(v["program"], t["name"], t["words"], t["capacity"])
    h.append(np.asarray(t["rows"], np.uint16), [bytes.fromhex(x) for x in t["salts"]])
    assert [bc.hex() for bc in h.bcs] == t["strings"] and HT.placeholder(t["words"]).hex() == t["placeholder"]
    assert (h.domain.domain_id.hex(), h.root.hex()) == (t["domain_id"], t["root"])
    big = v["domain_2_17"]
    assert (big["words"], big["capacity"]) == (BT.WORDS, BT.CAPACITY)
    assert HT.domain(v["program"], big["name"], BT.WORDS, BT.CAPACITY).domain_id.hex() == big["domain_id"]
    assert v["supplied"]["sources"] == [[{"history": f"{t['name']}/{of}"}] for of in IR.TABLE]
    sup = IR.supplied(v["program"], SUPPLIES, 1, [h.entry()])
    bits = ["".join(str((int(x) >> j) & 1) for x in r for j in range(16)) for value in sup for r in value.rows]
    assert bits == v["supplied"]["rows_bits"]


@SHARED
@pytest.mark.slow
@pytest.mark.parametrize("bad, why", [
    ("read row", "does not open its table there"),
    ("wrong root", "takes a root or domain id other than its table's"),
    ("another key", "does not open its table there"),
    ("verdict", "does not open its table there"),
])
def test_a_read_that_does_not_open_its_table_is_refused_before_anything_is_committed(staged, bad, why):
    """A read whose row isn't the table's at its key, one at the other span's table's root, one at key 2 with key 1's
    opening and one whose verdict is 0 open nothing. Slow: `interior` reaches a read's opening only past its window's
    program digest and round 1's record, so each case catches a bad read that would otherwise be committed."""
    _stream, reg, first, rec, tables = staged
    drawn = [tables[u] for u in first.units]
    spans = _spans(drawn)
    other = spans[1].data["rows"]["root"][0]
    kind, port, value = {"read row": ("rows", "row", spans[0].data["rows"]["row"][0] ^ 1),
                         "wrong root": ("rows", "root", other),
                         "another key": ("rows", "tok", np.asarray([2, 0], np.uint16)),
                         "verdict": ("outs", "o1", np.zeros(1, np.uint16))}[bad]
    with pytest.raises(ValueError, match=why):
        _interior(reg, first, rec, drawn, spans=[_with(spans[0], kind, port, 0, value), spans[1]])


@SHARED
@pytest.mark.slow
def test_a_table_read_interior_is_refused_without_its_tables_or_with_tables_it_cannot_read(staged):
    """Each sampled span brings its table exactly when a template reads one, with round 1's record and its class's C, and
    the reads' root and domain are both the verifier's; a table of another capacity and a commitment made on the rows'
    device are refused by name. Slow: each refusal is `interior`'s own, past its building of the reads' window and on
    round 1's registration and draw; it catches an interior of table reads committed without the tables it must open."""
    _stream, reg, first, rec, tables = staged
    drawn = [tables[u] for u in first.units]
    spans = _spans(drawn)
    with pytest.raises(ValueError, match="each sampled span brings its table"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW)
    with pytest.raises(ValueError, match="each sampled span brings its table"):
        _interior(reg, first, rec, drawn[:1], spans=spans)
    with pytest.raises(ValueError, match="give round 1's record"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=drawn)
    wide = [HT.HistoryTable(_program(), t.value, 2, 2 * CAP) for t in drawn]
    with pytest.raises(ValueError, match="reads a table of 8 rows of 2 words"):
        _interior(reg, first, rec, wide, spans=spans)
    with pytest.raises(ValueError, match="a replay block of table reads commits on the host"):
        _interior(reg, first, rec, drawn, spans=spans, device=IR.DeviceCommit({}, {}, {}))
    with pytest.raises(ValueError, match="give the `Supply` of each"):
        _interior(reg, first, rec, drawn, spans=spans, links=(SUPPLIES[0], *OPERANDS))


def test_a_drawn_spans_table_is_rebuilt_from_what_rank_0_kept_and_held_to_round_1s_entry():
    """`kept_table` commits a kept table's rows again under its kept salts (`boundary_tables.Served`, as `load` gives it) to
    round 1's entry, and refuses a kept table whose rows commit to another. No program reads the table, so it is under
    `UNREAD`."""
    t = _tables(program=UNREAD)[0]
    kept = BT.Served(**t.entry(), program=UNREAD, rows=np.array(t.rows), salts=b"".join(t.salts))
    again = IR.kept_table(kept)
    assert again.entry() == t.entry() and again.bcs == t.bcs
    rows = np.array(t.rows)
    rows[0, 0] ^= 1
    with pytest.raises(ValueError, match="commit to another table"):
        IR.kept_table(replace(kept, rows=rows))


@SHARED
@pytest.mark.slow
def test_the_challengers_history_refuses_a_table_round_1_did_not_register(staged):
    """`history` names only tables round 1's record registers, under the domain of their words and leaves. Slow by
    `staged` alone: the record is the one a stream registered, so it catches `history` and a registered record's roots
    and program drifting apart, which a record written by hand would not show."""
    _stream, _reg, _first_, rec, _tables_ = staged
    with pytest.raises(ValueError, match="registers no table"):
        IR.history(rec, [BT.name(RUN, PROMPT, OUTPUTS, SPANS, LAYER)], 2, C, {}, {})
    with pytest.raises(ValueError, match="under another domain"):
        IR.history(rec, [NAMES[0]], 4, C, {}, {})


def test_an_interior_whose_history_names_a_table_round_1_did_not_serve_under_its_name_is_refused():
    """An interior built, and its `history` taken, against a copy of round 1's record whose port for a drawn span's
    table names another table's root: the prover's strings, the interior's `history` and its supplied reads all agree
    on that table (`history.roots_of` and R7 pass), but the boundary registration the stream holds names another root
    for the port, so the stream refuses the interior (`interior-table`). The honest interior registers
    (`test_an_interior_opens_the_tables_round_1_served_and_the_challenger_resolves_its_reads_to_them`)."""
    tables = _tables()
    stream, reg, first, rec = _first(tables)
    drawn = [tables[u] for u in first.units]
    other = HT.HistoryTable(PROGRAM, drawn[1].value, 2, CAP)
    rows = np.array(drawn[1].rows)
    rows[1, 0] ^= 1
    other.append(rows, drawn[1].salts)
    forged = {**rec, "roots": [{**r, "root": other.root.hex()} if r["port"] == other.value else r for r in rec["roots"]]}
    it = _interior(reg, first, forged, [drawn[0], other])
    hist = _history(forged, first, it)
    assert HT.roots_of(it.leaf_layer, hist)[other.value] == (other.root, drawn[1].domain.domain_id) and \
        other.root != drawn[1].root
    assert not S.tables_served(rec, [(n, *rd) for n, rd in HT.roots_of(it.leaf_layer, hist).items()])
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, history=hist, registered=IR.link_words(LINKS),
                    supplied=IR.sources(LINKS, Q, [NAMES[u] for u in first.units]),
                    schemas=IR.link_schemas(MEMBER, LINKS))
    assert refused.value.verdict.codes == ("interior-table",)
    assert refused.value.verdict.detail == {"interior-table": {"tables": [other.value]}}
