"""The two-stage toy before anything is proved (`audit.py` proves it, on a node): each template and the replay
block are their numpy models; the replay block is its templates' instances; the challenger's stream over the toy's design
has its cells, its interior program and one second-stage stratum a (span, template) cell of the interior; the interior is
template-major and wired as the replay block's edges, each wired port with no tree of its own; each link member's
registered rows are the boundary record's at the replay sample's leaves, under its salts, and open its root there, at the
boundary port the design's links name, and every other port's rows open the interior record's, at their own leaves or a
wired port's at its writer's, as the verifier's own reads (from the two records and S alone) name them; both
registrations pass the verifier's check; a link row one bit off, a read of another leaf and a broken wire are refused;
and each session's public header is its template's reference statement's, which the auditor stages before the run,
placed in the run, as round 2 holds it, at the wires the replay block's Definition makes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.protocols.spot.device import challenger as Ch, coins as CC
from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registered as REG, registration as R, \
    verifier as V
from verity.circuits.codec import canonical_json, descriptor_id
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.program import Program
from verity.commitments.windows import flock_inputs as FI, wiring as WI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import toy as T

N, Q = 6, 3
SAMPLE = (4, 1, 5)
KEY = b"\x07" * 32


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("two-stage-toy")
    u = T.ingress(N)
    bc = T.boundary(u, Q, salt_key=KEY)
    brec = T.write(bc, tmp / "boundary", {"toy": "boundary"}, index_salt=b"\x01" * 192)
    digest = R.digest(brec)
    it = T.interior(bc, digest, SAMPLE, salt_key=b"\x08" * 32)
    irec = T.write(it.committed, tmp / "interior", {"interior": True, "boundary": digest, "replay_sample": list(SAMPLE)},
                   previous=digest, registered=it.registered, index_salt=b"\x02" * 192)
    return {"tmp": tmp, "u": u, "bc": bc, "brec": brec, "digest": digest, "it": it, "irec": irec}


def publics(c) -> dict[str, bytes]:
    return {m.key: c.public_body(k) for k, m in enumerate(c.window.members)}


def rows_bc(c, port: str, n: int) -> list[bytes]:
    b = c.bc[port]
    return [b[i * T.CS.BC_BYTES:(i + 1) * T.CS.BC_BYTES] for i in range(n)]


def test_each_template_and_the_replay_block_are_their_models():
    u = T.ingress(5, b"model")
    r = T.replay(u)
    for defn, x, y in ((T.LINK_IN, u, u), (T.FOLD, u, r["fold"]), (T.FLIP, r["fold"], r["flip"]),
                       (T.FLIP_B, r["flip"], r["flip_b"]), (T.LINK_OUT, r["flip"], r["flip"]),
                       (T.REPLAY, u, r["egress"])):
        assert np.array_equal(np.asarray(evaluate_bits(defn, T.bits(x))).reshape(len(u), -1), T.bits(y)), defn.id
    assert np.array_equal(r["egress"], ~(u[:, :T.H // T.WORD] & u[:, T.H // T.WORD:]))


def test_the_replay_block_is_its_templates_instances():
    ids = [P.template_id(t) for t in T.TEMPLATES]
    got = P.ranges(Program(T.REPLAY), P.template_instances_query(T.TEMPLATES))
    assert [(t, n) for t, _, n in got] == list(zip(ids, T.N_T)) and sum(T.N_T) == T.N_RHO


def test_the_challengers_stream_has_the_toys_cells_and_its_interior_program():
    ch = Ch.LiveChallenger("verity/two-stage/toy/test/v1", 0, [("draw", 1)], bytes)
    stream = S.Stream(T.spec(N, Q, 1), list(T.TEMPLATES), replay=T.REPLAY, budget=1, delta=0.5, challenger="test",
                      coins=CC.ChallengerCoins(Ch.LiveRecord(ch, "draw")))
    assert (stream.cells, stream.counts, stream.n_rho) == (T.N_T, (1,) * len(T.N_T), T.N_RHO)
    win = T.interior_window("00" * 64, SAMPLE)
    assert P.program_digest(stream.interior_program(Q)) == P.program_digest(win.program)
    law = S.second_law(list(SAMPLE), stream.cells, stream.counts)
    assert [tuple(u) for s in law["strata"] for u in s["units"]] == [
        (m.base + i * n, m.base + (i + 1) * n) for m, n in zip(win.members, T.N_T) for i in range(Q)]


def test_the_interior_is_template_major_and_wired_as_the_replay_blocks_edges(run):
    """Each instance's input reads its writer's (`toy.WRITERS`) one output in its span, one wire a port and one writer a
    wire, FLIP_B's two instances a span the same row, and a wired port commits no tree: its rows are its writer's tree's
    leaves, one tree a value, its table the writer's output rows (their ``b || c``) and each unit's ref its writer's row."""
    win, c = run["it"].window, run["it"].committed
    assert [m.base for m in win.members] == T.offsets(Q) == [0, Q, 2 * Q, 3 * Q, 5 * Q]
    assert [m.n for m in win.members] == [Q * n for n in T.N_T] and win.population == Q * T.N_RHO
    assert P.in_order(win.program, win.query, Q * T.N_RHO)
    assert win.members[0].wired == () and T.WRITERS[0] is None
    for t, m in enumerate(win.members[1:], 1):
        w, spans = T.WRITERS[t], tuple(i for i in range(Q) for _ in range(T.N_T[t]))
        (p, o, writers), = m.wired
        assert (p, o, writers) == (T.LAYOUT[t][0][0][0], "o0", tuple(T.offsets(Q)[w] + i for i in spans))
        assert win.held(m, p) == win.members[w].port("o0") and win.held_leaves(m, p) == spans
        assert m.port(p) not in c.roots and m.port(p) not in T.CS.expected_ports(win)
        assert dict(m.tables)[p] == win.members[w].n == Q and tuple(c.refs[t][:, 0].tolist()) == spans
        assert c.bc[m.port(p)] == c.bc[win.members[w].port("o0")]
    assert set(c.roots) == set(T.CS.expected_ports(win)) == {
        m.port(p) for m in win.members for p, _ in m.all_rows if win.held(m, p) == m.port(p)}
    WI.validate(win.members)
    assert WI.check(win.members, publics(run["it"].committed)) == {}


def test_each_link_row_is_the_boundary_row_at_its_spans_leaf_and_opens_the_boundary_root(run):
    bc, it, brec = run["bc"], run["it"], run["brec"]
    win, names = it.window, T.boundary_ports(bc.window)
    reads = T.reads(brec, run["irec"], win, SAMPLE)
    roots = {r["port"]: r for r in brec["roots"]}
    head = json.loads((run["tmp"] / "interior" / "window.json").read_text())
    links = {(x["template"], x["port"]): x["value"] for x in T.links(win)}
    assert len(links) == len(T.LINKS)
    for t, (p, b) in T.LINKS.items():
        m = win.members[t]
        port, value = it.ports[m.key], names[b]
        assert port.root == bc.roots[value] and port.root.hex() == roots[value]["root"]
        file = reads[m.key]
        assert file["format"] == REG.READS_V1 and list(file["ports"]) == [x for x, _ in m.all_rows]
        e = file["ports"][p]
        assert e["value"] == value == links[(m.template, p)] and e["reads"] == list(SAMPLE) and e[REG.READS_FROM] == m.base
        assert (e["leaves"], e["domain"], e[REG.SCHEMA_KEY]) == (N, roots[value]["domain"], T.V2)
        assert e["words"] == dict(m.all_rows)[p] == port.words
        header = T.CS.registered_header(head, t)[p]
        assert header["value"] == value and header["positions"] == list(SAMPLE)
        mine = rows_bc(it.committed, m.port(p), Q)
        assert mine == [rows_bc(bc, value, N)[s] for s in SAMPLE]
        assert all(REG.verify_read(bc.domains[value], port.root, s, x, bytes.fromhex(path), T.V2)
                   for s, x, path in zip(SAMPLE, mine, header["paths"]))


def test_every_other_port_reads_the_interior_record_at_its_own_leaves_or_its_writers(run):
    it, irec = run["it"], run["irec"]
    c, win = it.committed, it.window
    reads = T.reads(run["brec"], irec, win, SAMPLE)
    roots = {r["port"]: r for r in irec["roots"]}
    head = json.loads((run["tmp"] / "interior" / "window.json").read_text())
    for t, m in enumerate(win.members):
        assert m.tables == tuple((p, m.n if T.WRITERS[t] is None else win.members[T.WRITERS[t]].n) for p, _ in m.rows)
        for p, words in m.all_rows:
            if p == T.link(t):
                continue
            value, leaves = win.held(m, p), list(win.held_leaves(m, p))
            assert (value == m.port(p)) == (p not in {x for x, _o, _ws in m.wired})
            e, header = reads[m.key]["ports"][p], T.CS.registered_header(head, t)[p]
            n = roots[value]["leaves"]
            assert (e["value"], e["root"], e["leaves"], e["domain"]) == (value, roots[value]["root"], n,
                                                                         roots[value]["domain"])
            assert e["reads"] == leaves and (e[REG.READS_FROM], e["words"]) == (m.base, words)
            assert header["positions"] == list(range(n)) == list(range(dict(m.tables).get(p, m.n)))
            held = rows_bc(c, value, n)
            assert all(REG.verify_read(c.domains[value], c.roots[value], i, held[i], bytes.fromhex(path), T.V2)
                       for i, path in zip(range(n), header["paths"], strict=True))


def test_both_registrations_pass_the_verifiers_check_and_the_interior_follows_the_boundary(run):
    bc, it, brec, irec = run["bc"], run["it"], run["brec"], run["irec"]
    for rec, c in ((brec, bc), (irec, it.committed)):
        w = c.window
        v = V.check(rec, w.program, w.query, w.law, T.CS.SCHEME, T.CS.expected_ports(w),
                    leaf_layer=T.CS.leaf_layer(publics(c)), **({"log": [brec]} if rec is irec else {}))
        assert v.ok, v.codes
    assert irec["previous"] == run["digest"] and irec["window"]["replay_sample"] == list(SAMPLE)
    assert irec["window"]["run"] == T.interior_run(run["digest"], SAMPLE) != brec["window"]["run"]
    assert brec["query"] == R.query_digest(P.template_instance_query(T.REPLAY)) and brec["population"] == N


def test_a_link_row_off_another_leaf_and_a_broken_wire_are_refused(run):
    bc, it = run["bc"], run["it"]
    lin = it.window.members[0]
    value = T.boundary_ports(bc.window)["ingress"]
    port = it.ports[lin.key]
    honest = rows_bc(it.committed, lin.port("row"), Q)
    path = bytes.fromhex(port.header(SAMPLE)["paths"][0])
    assert REG.verify_read(bc.domains[value], port.root, SAMPLE[0], honest[0], path, T.V2)
    assert not REG.verify_read(bc.domains[value], port.root, SAMPLE[1], honest[0], path, T.V2)
    off = T.one_bit_off(port.rows_at(SAMPLE))
    assert int((T.bits(off) != T.bits(port.rows_at(SAMPLE))).sum()) == 1
    bad = T.interior(bc, run["digest"], SAMPLE, salt_key=b"\x08" * 32, replayed=off)
    x = rows_bc(bad.committed, lin.port("row"), Q)[0]
    assert x != honest[0] and not REG.verify_read(bc.domains[value], port.root, SAMPLE[0], x, path, T.V2)
    lout, egress = it.window.members[4], T.boundary_ports(bc.window)["o0"]
    y, opath = rows_bc(bad.committed, lout.port("o0"), Q)[0], bytes.fromhex(it.ports[lout.key].header(SAMPLE)["paths"][0])
    assert not REG.verify_read(bc.domains[egress], bc.roots[egress], SAMPLE[0], y, opath, T.V2)
    assert rows_bc(bad.committed, lout.port("o0"), Q)[1:] == rows_bc(it.committed, lout.port("o0"), Q)[1:]
    pub, flip = publics(it.committed), it.window.members[2]
    body = bytearray(pub[flip.key])
    body[0] ^= 1                                        # unit 0's x string, which the fold unit's output wires
    broken = {**pub, flip.key: bytes(body)}
    assert set(WI.check(it.window.members, broken)) == {flip.key}
    assert set(WI.check_drawn(it.window.members, broken, [flip.base])) == {flip.key}
    assert WI.check_drawn(it.window.members, broken, [flip.base + 1]) == {}


@pytest.mark.slow
def test_each_sessions_header_is_its_templates_reference_placed_in_the_run(run):
    """Round 2 holds each session's public header to the reference the auditor's params give its template
    (`toy.references`, staged before the run under a run of its own) placed in the run, and its registered reads to the
    reference's but for the run's fields: the honest public files the served path writes (`flock_input_files`, as a5
    stages the sessions) and the verifier's own reads (`toy.reads`), each session checked by a verifier of record
    (`second_stage.verified_of`), pass round 2 with the form allowed, the interior record's replay sample in S's order
    and the wires the replay block's Definition makes (`second_stage.replay_wires`: each member's input its writer's
    output, `toy.WRITERS`), while without the params' headers every one fails ``second-header``, without their reads or with
    a link port's read of another boundary port ``second-values``, without a verifier of record ``second-verified``,
    without the wires ``second-header`` and ``second-read`` (a wired port is bound and read at its writer's port), with
    none derived ``second-wires``, and against S reversed ``second-interior-order`` fails. Slow (about 15 s): the only
    test that lowers the templates, whose circuit SHA-512s and classes are in both headers."""
    from verity_flock.class_statement import member_stager

    it, irec, brec, tmp = run["it"], run["irec"], run["brec"], run["tmp"]
    win = it.window
    proofs = T.references(Q, tmp / "references")
    assert [e["template"] for e in proofs["templates"]] == [m.template for m in win.members]
    S.template_circuits(proofs, T.spec(N, Q, 1).check.query)
    bind = {"partition": win.partition_digest, "program": win.partition["program"]}
    parts = SS.parts(Q, P.strata(win.program, win.query))
    drawn = sorted(p.lo + i * p.n for p in parts for i in range(Q))
    reads, stmts = T.reads(brec, irec, win, SAMPLE), []
    for k, (m, defn) in enumerate(zip(win.members, T.TEMPLATES)):
        got = FI.flock_input_files(tmp / "interior", k, member_stager(defn, m, bind).text, tmp / "served", write=("pub",))
        with open(got["pub"], "rb") as f:
            header = json.loads(f.readline())
        assert header["circuit_sha512"] == proofs["templates"][k]["circuit"]
        stmts.append({"template": m.template, "header": header, "draw": SS.own_draw(Q, parts[k], drawn),
                      "registered": reads[m.key]})
    links = SS.replay_links(T.boundary_window(N, Q).program, descriptor_id(T.REPLAY), brec)
    wires = SS.replay_wires(T.boundary_window(N, Q).program, descriptor_id(T.REPLAY))
    rows = SS.replay_rows(T.boundary_window(N, Q).program, descriptor_id(T.REPLAY))
    assert wires == tuple(SS.Wire(m.template, T.LAYOUT[t][0][0][0], win.members[w].template, "o0")
                          for t, (m, w) in enumerate(zip(win.members, T.WRITERS)) if w is not None)
    refs = tuple((e["template"], e["header"]) for e in proofs["templates"])
    ref_reads = tuple((e["template"], e["registered"]) for e in proofs["templates"])
    (t0, r0), names = ref_reads[0], T.boundary_ports(T.boundary_window(N, Q))
    assert r0["ports"]["row"]["value"] == names["ingress"] and not {"root", "reads"} & set(r0["ports"]["row"])
    other = {**r0, "ports": {**r0["ports"], "row": {**r0["ports"]["row"], "value": names["o0"]}}}

    def sha512(j) -> str:
        return hashlib.sha512(canonical_json(j)).hexdigest()

    # what `flock-verify verify`'s verdict of record names of each session it checked
    verified = SS.verified_of({f"lean-verify/m{k}": {
        "of_record": True, "accepted": True, "public_header_sha512": sha512(s["header"]), "unit_draw": s["draw"],
        "session_tables": 1, **({} if s["registered"] is None else {"registered_sha512": sha512(s["registered"])})}
        for k, s in enumerate(stmts)})

    def check(sample, refs, verified=verified, ref_reads=ref_reads, wires=wires, rows=rows) -> dict:
        own = SS.Own(tuple(sample), parts, irec, brec, tuple(drawn), links, SS.statement_reads(stmts),
                     verified=verified, refs=refs, ref_reads=ref_reads, wires=wires, rows=rows)
        return SS.check(True, own, stmts).to_json()
    assert check(SAMPLE, refs) == {"ok": True, "codes": [], "detail": {}}
    assert check(SAMPLE, ())["codes"] == ["second-header"]
    assert check(SAMPLE, refs, ref_reads=())["codes"] == ["second-values"]
    assert check(SAMPLE, refs, ref_reads=((t0, other), *ref_reads[1:]))["codes"] == ["second-values"]
    assert check(SAMPLE, refs, ())["codes"] == ["second-verified"]
    assert check(SAMPLE, refs, wires=())["codes"] == ["second-header", "second-read"]
    assert check(SAMPLE, refs, wires=None)["codes"] == ["second-wires"]
    assert check(SAMPLE, refs, rows=())["codes"] == ["second-links", "second-wires"]
    assert "second-interior-order" in check(SAMPLE[::-1], refs)["codes"]


def test_the_link_reads_refuse_a_port_the_boundary_record_does_not_name_or_a_leaf_outside_it(run):
    brec, win = run["brec"], run["it"].window
    with pytest.raises(ValueError, match="names port"):
        T.LK.reads(brec, "nobody/row", 16, T.V2, SAMPLE, 0, registered=REG)
    with pytest.raises(ValueError, match="outside"):
        T.reads(brec, run["irec"], win, (N, 0, 1))
