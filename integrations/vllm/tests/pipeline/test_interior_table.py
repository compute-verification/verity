"""A two-stage interior whose replay block reads a history table (`interior_rows`, ``tables``): each span two
`EmbeddingRead_v1{V=8,H=2}` reads of its own table, over a replay sample of two of three spans, each bringing its table.
The interior commits each sampled span's table once, in `one_stage.history`'s format at 8 leaves, and registers it beside
the two values every read's ``root`` and ``domain`` rows are, which the challenger makes from the tables' entries alone. A
read that doesn't open its table is refused before anything is committed: after a row of its table is flipped, at a root
other than its table's, and at another key."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import history as HT, partition as P, registration as R
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates import embedding as E
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import interior_rows as IR, serving_rows as SRW

RD = E.read(8, 2)
SPANS, Q, CELLS, SALT, ROW, DELTA = 3, 2, 2, bytes(range(32)), "interior-table-test", 2.0 ** -40
#: sampled span s's two keys: key 6 is past its table's five rows, so it opens the zero row
TOKS = ((1, 4), (0, 6))


def _block():
    """Each span a batch of two `RD` reads on its own parameters."""
    params, ret = RD.signature()
    return bind(CompositeDefinition(
        "InteriorTableTest", 0, (), lambda S_: (tuple((n, Array(CELLS, t)) for n, t in params), Array(CELLS, ret)),
        lambda B, S_, *args: B.batch(RD, *args, axes=(0,) * len(args)), register=False))


BLOCK = _block()
MEMBER = IR.named(SRW.unit_member(P.template_id(RD), CELLS))


def _tables(seed: int = 0) -> list[IR.Table]:
    rng = np.random.default_rng(seed)
    return [IR.Table(rng.integers(0, 1 << 16, (5, 2), dtype=np.uint16), rng.bytes(5 * 192), 8) for _ in range(Q)]


def _boundary_window() -> CS.Binding:
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      "b0" * 64, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


def _first():
    """A two-stage stream over `RD`, a stand-in boundary registered and its first stage drawn (spans 0 and 2)."""
    names = {}
    for phase, w in (("serve", _boundary_window()), ("interior", IR.interior_window(MEMBER, CELLS, "00", [0]))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test values"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")))
    spec = S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items())),
                  S.TwoStage(P.template_instance_query(BLOCK), f"rate:{Q}/{SPANS}", 1),
                  S.Check((P.program_digest(Program(RD)),), P.template_instances_query((RD,)), proof_deadline=60.0), public,
                  b"interior-table-test")
    draws = iter([[0, 2]])
    stream = S.Stream(spec, (RD,), replay=BLOCK, budget=1, delta=DELTA, challenger="challenger-test", clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    window = _boundary_window()
    c = CS.commit(window, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                            "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    reg = stream.register(CS.registration(c, {"stand_in": "boundary"}, None, registrar=R), CS.expected_ports(window),
                          leaf_layer=CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)}))
    return stream, reg, stream.select(reg)


def _spans(digest: str, units, tables) -> list[IR.Span]:
    """Each sampled span's two reads, opened against its table as the interior commits it."""
    tcs = IR.interior_tables(MEMBER, CELLS, digest, units, tables)
    return [IR.Span(f"s{s}", MEMBER, IR.table_reads(tc, TOKS[s]), [[0, CELLS]], "p" * 64) for s, tc in enumerate(tcs)]


def _with(span: IR.Span, kind: str, port: str, unit: int, value) -> IR.Span:
    a = span.data[kind][port].copy()
    a[unit] = value
    return replace(span, data={**span.data, kind: {**span.data[kind], port: a}})


def test_a_table_read_interior_registers_each_spans_table_once_and_its_reads_roots_as_supplied_values():
    """The record registers the two sampled spans' tables, each at 8 leaves under `one_stage.history`'s domain, and two
    supplied values. `history.check`, the verifier's mirror, reopens each table's root from the rows' b || c the interior
    publishes, read through the table's held run, and the supplied values made from the registered table entries alone
    are the record's, each row's source naming its table. The reads' ``root`` and ``domain``
    rows are given by reference, rows (0, 0, 1, 1) of those values, and the commitment is the one the same rows given in
    full under the values' salts makes. The stream refuses it after its first stage, since its challenger holds no
    registered values for an interior: R7 takes a registered value only as a parameter of the interior program, and the
    tables and supplied values aren't parameters of it yet."""
    stream, reg, first = _first()
    tables = _tables()
    it = IR.interior(_spans(reg.digest, first.units, tables), CELLS, reg.digest, first.units, row=ROW, salt_key=SALT,
                     tables=tables)
    c, w = it.committed, it.committed.window
    prog, run = P.program_digest(w.program), w.run
    got = it.record["registered"]
    assert [e["value"] for e in got] == [IR.table_name(run, s) for s in range(Q)] + [
        f"interior/{run}/history/{of}" for of in ("root", "domain")]
    for s, (e, t) in enumerate(zip(got[:Q], tables, strict=True)):
        assert (e["leaves"], e["words"]) == (8, 2)
        assert e["domain"] == HT.domain(prog, IR.table_name(run, s), 2, 8).domain_id.hex()
        HT.check(prog, e["value"], 2, it.tables[s].bcs, bytes.fromhex(e["root"]), 8)
        assert it.tables[s].bcs == IR.commit_table(t, prog, e["value"]).bcs
    for tc in it.tables:
        held = tc.published()
        at = HT.order(tc.runs(), len(tc.bcs), "", "", 0, {v: len(x) for v, x in held.items()})
        HT.check(prog, tc.value, 2, [bytes.fromhex(held[v][r]) for _k, v, r in at], tc.root, 8)
    sup = IR.supplied(prog, run, got[:Q])
    assert [v.entry() for v in sup] == [{k: e[k] for k in v.entry()} for v, e in zip(sup, got[Q:], strict=True)]
    assert IR.sources(run, it.tables) == {IR.supplied_name(run, of): [[{"history": f"{IR.table_name(run, s)}/{of}"}]
                                                                      for s in range(Q)] for of in ("root", "domain")}
    m = w.members[0]
    by_ref = {p: c.given[m.port(p)] for p in ("root", "domain")}
    assert all(v.positions == (0, 0, 1, 1) for v in by_ref.values())
    spans = _spans(reg.digest, first.units, tables)
    data = [{"rows": {p: np.concatenate([s.data["rows"][p] for s in spans]) for p, _ in m.rows},
             "outs": {p: np.concatenate([s.data["outs"][p] for s in spans]) for p, _ in m.outs},
             "refs": np.repeat(np.arange(m.n)[:, None], len(m.rows), axis=1)}]
    full = CS.commit(w, data, salt_key=SALT, salts_of={m.port(p): CS.value_salts(v.value, v.positions)
                                                       for p, v in by_ref.items()})
    assert full.roots == c.roots
    with pytest.raises(S.Refused, match="keys"):
        IR.register(stream, first, it)


@pytest.mark.parametrize("bad, why", [
    ("flipped table row", "takes a root or domain id other than its table's"),
    ("read row", "does not open its table there"),
    ("wrong root", "takes a root or domain id other than its table's"),
    ("another key", "does not open its table there"),
    ("verdict", "does not open its table there"),
])
def test_a_read_that_does_not_open_its_table_is_refused_before_anything_is_committed(bad, why):
    """After its table's row 1 is flipped, the reads made against the table as it was take another root; a read whose row
    isn't the table's at its key, one at the other span's table's root, one at key 2 with key 1's opening and one whose
    verdict is 0 open nothing."""
    _stream, reg, first = _first()
    tables = _tables()
    spans = _spans(reg.digest, first.units, tables)
    other = _spans(reg.digest, first.units, tables)[1].data["rows"]["root"][0]
    if bad == "flipped table row":
        rows = tables[0].rows.copy()
        rows[1, 0] ^= 1
        tables = [replace(tables[0], rows=rows), tables[1]]
    else:
        kind, port, value = {"read row": ("rows", "row", spans[0].data["rows"]["row"][0] ^ 1),
                             "wrong root": ("rows", "root", other),
                             "another key": ("rows", "tok", np.asarray([2, 0], np.uint16)),
                             "verdict": ("outs", "o1", np.zeros(1, np.uint16))}[bad]
        spans = [_with(spans[0], kind, port, 0, value), spans[1]]
    with pytest.raises(ValueError, match=why):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=tables)


def test_a_table_read_interior_is_refused_without_its_tables_or_with_tables_it_cannot_read():
    """Each sampled span brings its table exactly when a template reads one; a table of another capacity, a salt on two
    rows and a commitment made on the rows' device are refused by name."""
    _stream, reg, first = _first()
    tables = _tables()
    spans = _spans(reg.digest, first.units, tables)
    with pytest.raises(ValueError, match="each sampled span brings its table"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW)
    with pytest.raises(ValueError, match="each sampled span brings its table"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=tables[:1])
    with pytest.raises(ValueError, match="reads a table of 8 rows of 2 words"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=[replace(t, capacity=16) for t in tables])
    salts = tables[0].salts[:192] * 5
    with pytest.raises(ValueError, match="one salt on two rows"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=[replace(tables[0], salts=salts), tables[1]])
    device = IR.DeviceCommit({}, {}, {})
    with pytest.raises(ValueError, match="a replay block of table reads commits on the host"):
        IR.interior(spans, CELLS, reg.digest, first.units, row=ROW, tables=tables, device=device)
