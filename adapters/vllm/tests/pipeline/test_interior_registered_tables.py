"""A two-stage interior whose replay block opens registered tables (`interior_rows`, ``registered`` and ``pins``): each
span two batches of two fused blocks (`MoeExpertFp8ReadGemm_v1`, a GEMM over the row its own read opens), batch g on
registered table g of two, each of four experts a row, over a replay sample of two of three spans. The tables are
registered before any window (a registration's ``values``, which the verifier pins), under a program that is no
Program's digest, as the per-expert weight tables are. The interior commits no table and no root: each block's
``root`` and ``domain`` rows are rows of two supplied values, row r of a sampled span table r's, which the challenger
fills from the entries it pinned (``bytes`` sources). A block's key and opening (``tok``, ``row``, ``salt``, ``path``)
and its operands are the replay's own values.

The stream registers that interior, and refuses (`R7-supplied`) one made under any other root than the pinned one. An
interior is refused before anything is committed where a fused block is left untied, where a table has no pin or is
another than the pinned one, and where a unit takes another table's root or does not open its table.

The block is the registered fused family with a stand-in for its read: the same ports and (h, ok), h the row and ok 1,
with none of the read's SHA-512, so a program over it builds in a quick test's seconds. The openings are real: every
table is committed as the reads' row format and every unit's opening is checked by the read's reference
(`gates.embedding.reference`). No session is proved here."""

from __future__ import annotations

import functools
from dataclasses import replace

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import ONE
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.program import Program
from verity.circuits.refs import coll_concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.commitments.gates import embedding as EMB
from verity.commitments.windows import rows as CS
from verity.commitments.windows.interface import digest_words
from verity.definitions.numerics.fp8_moe_rows import moe_expert_fp8_block_row
from verity.definitions.registry import boolean_fp8_moe as BM
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P
from verity.protocols.proofs.spot.verifier.one_stage import registered as RG
from verity.protocols.proofs.spot.verifier.one_stage import registration as R

from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import interior_links as IL
from verity_vllm.pipeline import interior_rows as IR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.pipeline import serving_templates as ST

#: experts a table, a block's K e4m3 bytes (one row a block), and a table row's u16 words
E, K, H = 4, 32, 16
SPANS, Q, SALT, ROW, DELTA = 3, 2, bytes(range(32)), "interior-registered-tables-test", 2.0 ** -40
#: a boundary registration's digest and its replay sample, for an interior no stream registers
BOUNDARY, SAMPLE = "b1" * 64, (0, 2)
#: the registered tables a span reads, the blocks that each open one, and a span's units
T, W = 2, 2
CELLS = T * W
#: the tables' registration program: the SHA-512 of what was registered, no Program's digest (`records.expert_tables`)
REGISTRATION = "5e" * 64
PARAMS = tuple(f"weights/model.layers.0.mlp.experts.{p}@{H}/b0/table" for p in ("gate_up_proj", "down_proj"))


def _read_signature(S_):
    return EMB.read(S_.V, S_.H).signature()[0], Tuple(Array(S_.H, EMB.BF16), BL.BIT)


#: a read's ports and (h, ok) without its hashes: h the row (a read's elements are its key, eight words each of its root
#: and domain id, then the row's H), ok 1
READ = TR.part("InteriorRegisteredTablesTestRead", ("V", "H"), _read_signature,
               lambda C, S_, *e: [*e[17:17 + S_.H], [ONE]], "h = row, ok = 1: a read's ports with none of its hashes",
               version=0, register=False)
FUSED = bind(BM.MoeExpertFp8ReadGemmV1, READ=bind(READ, V=E, H=H), K=K, N=1, G=K, KB=K, SWAP=False,
             DOT=BG.HopperE4m3MmaV2Dot32)
TID = descriptor_id(FUSED)
PORTS = tuple(p for p, _t in FUSED.signature()[0])


def _block():
    """Each span `T` runs of `W` fused blocks, run g under ``root[g]`` and ``domain[g]``, each block on its own row of
    every other parameter: one instance a block, so the block returns each one's y and each one's ok, as round 2 reads
    a return of a template of two outputs."""
    params, (y, ok) = FUSED.signature()[0], FUSED.signature()[1].fields

    def body(B_, S_, *args):
        got = [B_.call(FUSED, *(a[j // W] if p in IR.TABLE else a[j] for p, a in zip(PORTS, args, strict=True)))
               for j in range(CELLS)]
        return tuple_of(coll_concat(Array(CELLS, y), [r[0] for r in got]), coll_concat(Array(CELLS, ok), [r[1] for r in got]))

    return bind(CompositeDefinition(
        "InteriorRegisteredTablesTest", 0, (),
        lambda S_: (tuple((p, Array(T if p in IR.TABLE else CELLS, t)) for p, t in params),
                    Tuple(Array(CELLS, y), Array(CELLS, ok))), body, register=False))


BLOCK = _block()
KEY = descriptor_id(BLOCK)
#: unit j of a span opens table j // W
AT = tuple(j // W for j in range(CELLS))
#: the blocks' root and domain, the verifier's: a row a table, under the names round 2 gives the block's parameters
SUPPLIES = tuple(IR.Supply(TID, p, f"{KEY}/{p}", T, AT, tables=PARAMS) for p in IR.TABLE)


@pytest.fixture(scope="module", autouse=True)
def _template():
    """The fused block as the rows resolve a template: `serving_templates.template` binds a registered Definition from
    its id alone, and this block's read is this module's."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ST._TEMPLATES, TID, SRW.Template(TID, FUSED, TID, "fused"))
        yield


@functools.cache
def _member() -> CS.Member:
    return IR.named(CR.unit_member(SRW.template(TID), CELLS))


@functools.cache
def _operands() -> tuple[IR.Operand, ...]:
    """The blocks' key, opening and operands, the replay's: a row a unit, of its port's words."""
    return tuple(IR.Operand(TID, p, f"{KEY}/{p}", CELLS, tuple(range(CELLS)), w) for p, w in _member().rows
                 if p not in IR.TABLE)


def _links() -> tuple[IR.Link, ...]:
    return (*SUPPLIES, *_operands())


@functools.cache
def _tables(seed: int = 0) -> dict[str, RG.Registered]:
    """The registered tables, by parameter: `E` rows of `K` e4m3 bytes each, committed in the reads' row format under
    salts of their own. ``seed``: 0 the registration the verifier pinned, any other a registration of other weights."""
    rng = np.random.default_rng(seed)
    return {p: RG.Registered.commit(REGISTRATION, p, rng.integers(0, 0x70, (E, K), dtype=np.uint8).view("<u2").tolist(),
                                    [rng.bytes(192) for _ in range(E)], schema=RG.SCHEMA_SEG) for p in PARAMS}


def _pins(tables) -> dict[str, dict]:
    """The registration's ``values``, which the verifier pins: each table's entry by parameter."""
    return {p: t.entry() for p, t in tables.items()}


def _words(b: bytes) -> list[int]:
    """Consecutive 64-byte digests (a salt's three, a path's siblings) as `Digest` words."""
    return [w for i in range(0, len(b), 64) for w in digest_words(b[i:i + 64])]


def _bits(words) -> list[int]:
    return [(int(w) >> b) & 1 for w in words for b in range(16)]


@functools.cache
def _span(seed: int, registration: int = 0) -> IR.Span:
    """One span's `CELLS` fused blocks on the tables of ``registration`` (`_tables`): unit j opens table j // W at a
    key of its own and multiplies the row by random e4m3 bytes; y is the block's twin's on them
    (`numerics.fp8_moe_rows`) and ok 1. Shared: a test that changes a row copies it."""
    rng, tables = np.random.default_rng(seed), _tables(registration)
    m = _member()
    cols: dict[str, list] = {p: [] for p, _w in (*m.rows, *m.outs)}
    for j in range(CELLS):
        t, e = tables[PARAMS[AT[j]]], int(rng.integers(0, E))
        xq, sx, ws = rng.integers(0, 0x70, K, dtype=np.uint8), np.asarray([0.5], "<f4"), np.asarray([0.25], "<f4")
        y, _nonfinite = moe_expert_fp8_block_row(xq, sx.view("<u4"), np.asarray(t.rows[e], "<u2").view(np.uint8),
                                                 ws.view("<u4"), G=K, KB=K, dot="mmav2")
        unit = {"tok": [e, 0], "root": digest_words(t.root), "domain": digest_words(t.domain.domain_id), "row": t.rows[e],
                "salt": _words(t.salts[e]), "path": _words(t.path(e)), "xq": xq.view("<u2"), "sx": sx.view("<u2"),
                "ws": ws.view("<u2"), "o0": y, "o1": [1]}
        for p, v in unit.items():
            cols[p].append(np.asarray(v, dtype=np.uint16))
    data = {"rows": {p: np.stack(cols[p]) for p, _w in m.rows}, "outs": {p: np.stack(cols[p]) for p, _w in m.outs}}
    return IR.Span(f"s{seed}", m, data, [[0, m.n]], "p" * 64)


def _boundary_window() -> CS.Binding:
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      "b0" * 64, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


@functools.cache
def _boundary() -> tuple[dict, dict, str]:
    """Round 1's stand-in boundary: its record, the challenger's ports and its leaf layer."""
    window = _boundary_window()
    c = CS.commit(window, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                            "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    return (CS.registration(c, {"stand_in": "boundary"}, None, registrar=R), CS.expected_ports(window),
            CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)}))


def _first():
    """A window stream over the fused block taking the links' values a sampled span (`interior_rows.span_values`), a
    stand-in boundary registered and its first stage sampled (spans 0 and 2)."""
    links = _links()
    names: dict[str, tuple[str, str]] = {}
    values = IR.link_values(links, Q)
    for phase, w in (("serve", _boundary_window()), ("interior", IR.interior_window(_member(), CELLS, "00", [0], values=values))):
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
                  S.Check((P.program_digest(Program(FUSED)),), P.template_instances_query((FUSED,)), proof_deadline=60.0),
                  public, b"interior-registered-tables-test")
    draws = iter([list(SAMPLE)])
    stream = S.Stream(spec, (FUSED,), span_values=IR.span_values(links), replay=BLOCK, budget=1, delta=DELTA,
                      challenger="challenger-test", clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    record, ports, layer = _boundary()
    reg = stream.register(record, ports, leaf_layer=layer)
    return stream, reg, stream.select(reg)


def _interior(registration: int = 0, boundary: str = BOUNDARY, **kw) -> IR.Interior:
    """The interior of `SAMPLE`'s spans on the tables of ``registration``, under this module's links and those
    tables' entries as the pins, unless ``kw`` gives others."""
    tables = _tables(registration)
    kw = {"salt_key": SALT, "links": _links(), "registered": tables, "pins": _pins(tables),
          "spans": [_span(s, registration) for s in SAMPLE], **kw}
    return IR.interior(kw.pop("spans"), CELLS, boundary, list(SAMPLE), row=ROW, **kw)


def _copies(pins) -> dict:
    """The challenger's copies for `Stream.register_interiors`, from its own links and the entries it pinned."""
    links = _links()
    return {"registered": IR.link_words(links), "supplied": IR.sources(links, Q, pins=pins),
            "schemas": IR.link_schemas(_member(), links)}


def test_a_spans_unit_is_the_fused_blocks_own_function():
    """This module's spans are the block's: a unit's y and ok are the fused block's reference on its rows."""
    data = _span(SAMPLE[0]).data
    out = evaluate(FUSED, *(_bits(data["rows"][p][0]) for p in PORTS))
    assert list(out) == [*_bits(data["outs"]["o0"][0]), int(data["outs"]["o1"][0][0])]


def test_the_stream_registers_an_interior_of_fused_blocks_under_the_tables_the_verifier_pinned():
    """The interior registers the blocks' ``root`` and ``domain`` as two supplied values, Q·T rows each, row ``s·T + r``
    table r's pinned root or domain id as a `Digest`'s words, ahead of the seven the replay gives; each block's port
    is given by reference, the row of its table, where round 2 holds a linked port's read. From the challenger's ``bytes``
    sources alone, the pinned entries' hex, the verifier fills the same rows (`registered.supplied_rows` on the stream's
    own interior program), which at the public salts commit to the entry the interior registers. The stream registers
    it. A challenger that pinned another root for one table refuses the same record (`R7-supplied`)."""
    stream, reg, first = _first()
    assert tuple(first.units) == SAMPLE
    it = _interior(boundary=reg.digest)
    c, (m,) = it.committed, it.committed.window.members
    prog = P.program_digest(c.window.program)
    assert P.program_digest(stream.interior_program(Q)) == prog
    got = {e["value"]: e for e in it.record["registered"]}
    assert list(got) == [x.value for x in _links()] == [f"{KEY}/{p}" for p in (*IR.TABLE, *(p for p in PORTS if p not in IR.TABLE))]
    pins = _pins(_tables())
    srcs = IR.sources(SUPPLIES, Q, pins=pins)
    assert srcs == {f"{KEY}/{p}": [[{"bytes": pins[n][p]}] for _ in range(Q) for n in PARAMS] for p in IR.TABLE}
    for x in SUPPLIES:
        assert c.given[m.port(x.port)].positions == tuple(s * T + a for s in range(Q) for a in AT)
        mine = RG.supplied_rows(stream.interior_program(Q), x.value, RG.sources(srcs)[x.value], it.record)
        v = RG.Registered.commit(prog, x.value, [RG.bit_words(r, 1, 16) for r in mine],
                                 RG.supplied_salts(prog, x.value, len(mine)), schema=m.schema(x.port))
        assert [list(r) for r in v.rows] == [digest_words(bytes.fromhex(pins[n][x.port])) for _ in range(Q) for n in PARAMS]
        assert v.entry() == {k: got[x.value][k] for k in v.entry()} and got[x.value]["leaves"] == Q * T
    assert IR.span_values(SUPPLIES) == tuple((x.value, T, IR.DIGEST) for x in SUPPLIES)
    other = {**pins, PARAMS[1]: {**pins[PARAMS[1]], "root": _tables(1)[PARAMS[1]].root.hex()}}
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, **_copies(other))
    assert refused.value.verdict.codes == ("R7-supplied",)
    assert IR.register(stream, first, it, **_copies(pins)).digest


def test_an_interior_made_under_another_root_than_the_pinned_one_is_refused_by_the_stream():
    """A prover that registers other tables under the pinned names and builds its whole interior on them, every block
    opening its table under that table's root: nothing in the interior is inconsistent, and the stream refuses its
    record, since the roots it commits are not the ones the challenger pinned (`R7-supplied`; the domain ids, which
    bind the registration's program, the name and the shape, are the pinned ones). The pin alone refuses it: a
    challenger that had pinned those other tables registers the same record."""
    stream, reg, first = _first()
    tables, forged = _tables(), _tables(7)
    assert all(forged[p].root != tables[p].root and forged[p].domain == tables[p].domain for p in PARAMS)
    it = _interior(7, reg.digest)
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, **_copies(_pins(tables)))
    assert refused.value.verdict.codes == ("R7-supplied",)
    assert IR.register(stream, first, it, **_copies(_pins(forged))).digest


def test_a_block_tied_to_another_root_is_refused_before_anything_is_committed():
    """Under the true pins: a prover holding another table under a pinned name is refused by that name; a unit that
    takes the other table's root (or domain id) at its port is refused, whether or not it opens that other table; and a
    unit under its own table's root whose row, salt or path is no opening of it there, or whose verdict is not 1, is
    refused."""
    tables, forged = _tables(), _tables(7)
    with pytest.raises(ValueError, match=f"registered table {PARAMS[0]} is not the one the verifier pinned under its name: "
                                         "its root differ"):
        _interior(registered={**tables, PARAMS[0]: forged[PARAMS[0]]})
    honest = [_span(s) for s in SAMPLE]

    def moved(unit: int, **rows) -> list[IR.Span]:
        s = honest[0]
        data = {k: {p: a.copy() for p, a in s.data[k].items()} for k in ("rows", "outs")}
        for p, v in rows.items():
            data["outs" if p == "o1" else "rows"][p][unit] = v
        return [replace(s, data=data), *honest[1:]]

    elsewhere = _span(SAMPLE[0], 7).data["rows"]
    other = "takes a root or domain id other than its table's"
    for bad, why in ((moved(0, root=elsewhere["root"][0]), other),
                     (moved(0, **{p: elsewhere[p][0] for p in IL.READ}), other),
                     (moved(1, root=honest[0].data["rows"]["root"][W], domain=honest[0].data["rows"]["domain"][W]), other),
                     (moved(1, row=honest[0].data["rows"]["row"][1] ^ 1), "does not open its table there"),
                     (moved(2, tok=[(int(honest[0].data["rows"]["tok"][2][0]) + 1) % E, 0]), "does not open its table there"),
                     (moved(3, o1=[0]), "its verdict is not the read's ok")):
        with pytest.raises(ValueError, match=why):
            _interior(spans=bad)


def test_a_fused_block_whose_read_is_left_untied_is_refused():
    """A fused block's ``root`` and ``domain`` given as the replay's own values, as a table read of round 1's history,
    or as no link at all, would be rows nothing holds to the registered table: the interior refuses each by the
    block's name, and so does the challenger's copy of the values' formats (`link_schemas`), which every registration
    of linked ports takes. One of the two tied alone, or the two at other tables or places, is refused too."""
    m, ops = _member(), _operands()
    own = tuple(IR.Operand(TID, p, f"{KEY}/{p}", T, AT, 32) for p in IR.TABLE)
    history = tuple(replace(x, tables=None) for x in SUPPLIES)
    untied: list[tuple[IR.Link, ...]] = [(*own, *ops), (*history, *ops), ops]
    for links in untied:
        with pytest.raises(ValueError, match="a fused block's root and domain are left untied"):
            _interior(links=links)
        with pytest.raises(ValueError, match="a fused block's root and domain are left untied"):
            IR.link_schemas(m, links)
    root, dom = SUPPLIES
    apart: list[tuple[IR.Link, ...]] = [(root, own[1], *ops), (root, replace(dom, tables=PARAMS[::-1]), *ops),
                                        (root, replace(dom, at=AT[::-1]), *ops)]
    for links in apart:
        with pytest.raises(ValueError, match="its root and domain are not the same registered tables' at the same places"):
            _interior(links=links)
    assert IL.tied([m], _links()) == {0: (root, E, H, True)}


def test_a_registered_table_without_a_pin_is_refused():
    """A table the verifier pinned no entry for: the interior refuses it by name, and the challenger has no source for
    its rows, so `sources` and `supplied` refuse it too. A table the prover holds no rows of, and one of another shape
    than the block reads, are refused by name."""
    tables = _tables()
    pins = _pins(tables)
    lone = {PARAMS[0]: pins[PARAMS[0]]}
    with pytest.raises(ValueError, match=f"registered table {PARAMS[1]} has no pin"):
        _interior(pins=lone)
    with pytest.raises(ValueError, match=f"registered table {PARAMS[0]} has no pin"):
        _interior(pins={})
    for fill in (IR.sources, lambda *a, **kw: IR.supplied("00" * 64, *a, **kw)):
        with pytest.raises(ValueError, match=f"{KEY}/root: registered table {PARAMS[1]} has no pin"):
            fill(SUPPLIES, Q, pins=lone)
        with pytest.raises(ValueError, match=f"{KEY}/root: registered table {PARAMS[0]} has no pin"):
            fill(SUPPLIES, Q)
    with pytest.raises(ValueError, match=f"registered table {PARAMS[1]} is not held"):
        _interior(registered={PARAMS[0]: tables[PARAMS[0]]})
    wide = RG.Registered.commit(REGISTRATION, PARAMS[0], np.zeros((2 * E, H), np.uint16).tolist(), schema=RG.SCHEMA_SEG)
    with pytest.raises(ValueError, match=f"reads a table of {E} rows of {H} words, not registered table {PARAMS[0]}'s {2 * E}"):
        _interior(registered={**tables, PARAMS[0]: wide}, pins={**pins, PARAMS[0]: wide.entry()})


def test_the_links_are_the_ones_round_2_derives_from_the_replay_block():
    """`interior_rows.links` reads the block's links off its Definition as round 2 does, the block's ``root`` and
    ``domain`` the verifier's with the registered tables a row (``tables``): a row a table, unit j reading row
    j // `W`. They are this module's by hand."""
    record = _boundary()[0]
    got = IR.links(P.population_program(BLOCK, SPANS), KEY, record, _member(), CELLS, dict.fromkeys(IR.TABLE),
                   dict.fromkeys(IR.TABLE, PARAMS))
    assert len(got) == len(_links()) and set(got) == set(_links())
    with pytest.raises(ValueError, match="1 registered tables for its 2 rows a sampled span"):
        IR.links(P.population_program(BLOCK, SPANS), KEY, record, _member(), CELLS, dict.fromkeys(IR.TABLE),
                 dict.fromkeys(IR.TABLE, PARAMS[:1]))
    with pytest.raises(ValueError, match="read it as two values"):
        IL.values((SUPPLIES[0], replace(SUPPLIES[0], tables=PARAMS[::-1])))


def test_a_table_read_of_a_registered_table_is_tied_as_a_fused_block_is():
    """A member that is exactly a read (a block's scale read, say) whose ``root`` and ``domain`` name registered tables
    opens them as a fused block does and brings no history table: its h is the row it opens. With no registered table
    named it is a read of round 1's history, as before."""
    read = bind(READ, V=E, H=H)
    rid = descriptor_id(read)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ST._TEMPLATES, rid, SRW.Template(rid, read, rid, "read"))
        m = IR.named(CR.unit_member(SRW.template(rid), CELLS))
        ties = tuple(IR.Supply(rid, p, f"read/{p}", T, AT, tables=PARAMS) for p in IR.TABLE)
        ops = tuple(IR.Operand(rid, p, f"read/{p}", CELLS, tuple(range(CELLS)), w) for p, w in m.rows if p not in IR.TABLE)
        assert IL.tied([m], (*ties, *ops)) == {0: (ties[0], E, H, False)}
        assert IL.tied([m], (*(replace(x, tables=None) for x in ties), *ops)) == {}
        spans, tables = [], _tables()
        for s in SAMPLE:
            rows = {p: a for p, a in _span(s).data["rows"].items() if p in IL.READ}
            spans.append(IR.Span(f"s{s}", m, {"rows": rows, "outs": {"o0": rows["row"].copy(),
                                                                      "o1": np.ones((CELLS, 1), np.uint16)}},
                                 [[0, m.n]], "p" * 64))
        it = IR.interior(spans, CELLS, BOUNDARY, list(SAMPLE), row=ROW, salt_key=SALT, links=(*ties, *ops),
                         registered=tables, pins=_pins(tables))
        assert it.tables == () and [e["value"] for e in it.record["registered"]][:2] == ["read/root", "read/domain"]
        spans[0].data["outs"]["o0"][0, 0] ^= 1
        with pytest.raises(ValueError, match="its h and ok are not the read's"):
            IR.interior(spans, CELLS, BOUNDARY, list(SAMPLE), row=ROW, salt_key=SALT, links=(*ties, *ops),
                        registered=tables, pins=_pins(tables))
