"""A served class's second stage as `two_stage.py second` runs it, at small statics: a token select class as a serve
and its replay leave it (`two_stage_small.serve`: the class's files, its replay block the check's instances, one a served
token, its record committing the ids the block returns as a port, and a capture of the sampler's inputs), its first
stage on a test's own challenger (`fake_challenger`), and the replay side's half from ``stage1.json``, the class's
files and the capture (`two_stage.replayed`): the interior committed, its return's rows round 1's under the serve's
salts, written as the one-stage audit reads a window, registered on the class's stream and sampled. Then what the
verifier's half hands the one-stage audit and round 2 (`two_stage.prove`'s inputs: the verifier's reads, the params, the
sessions' statements), held to round 2's check without the sessions, which run on a node (``two_stage_small.sh``): the
public files the served path writes are the ones a5 stages, and a stand-in names them as a verdict of record does."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from verity.circuits.codec import canonical_json
from verity.commitments.windows import flock_inputs as FI, links as LK, rows as CS
from verity.protocols.spot.verifier import second_stage as SS, service as S
from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity_vllm.pipeline import boundary_rows as BR, interior_rows as IR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "spot" / "verifier" / "tests"))
TS = importlib.import_module("two_stage")
SM = importlib.import_module("two_stage_small")
FC = importlib.import_module("fake_challenger")

V, O, N, Q, K = 128, 4, 6, 2, 1
TID = SM.template(V)


def _layout(inline: bool = False) -> BR.Layout:
    return SM.layout(V, O, N, inline)


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> dict:
    """The class's files as the serve writes them and the capture of every request's sampler inputs, each under a launch
    of its own (`two_stage_small.serve`); the class reopened from its files, as the replay side opens it."""
    tmp = tmp_path_factory.mktemp("second")
    got, check = SM.serve(tmp, V, O, N), BR.check_template(TID)
    return {"tmp": tmp, "check": check, "capture": got["capture"], "class": BR.reopen(got["class"], check),
            "pub": json.loads((got["class"] / BR.COMMIT).read_text())}


@pytest.fixture(scope="module")
def run(served) -> dict:
    """The first stage under the class's own rule (`two_stage.rule_of`), ``stage1.json`` read again, and the replay
    side's half (`two_stage.replayed`)."""
    check, tmp = served["check"], served["tmp"]
    got = TS.first(served["pub"], check, TS.rule_of(_layout(), f"rate:{Q}/{N}", K, 60.0), FC.Challenger("first"))
    stage1 = json.loads(json.dumps(got.stage1()))
    again = TS.resume(stage1, check, FC.Challenger("second"))
    r = TS.replayed(again, served["class"], served["capture"], tmp / "interior")
    return {**served, "stage1": stage1, "again": again, "r": r}


def _handed(interior: Path, served: Path, mine: CS.Binding, reads: dict, drawn: list[int]) -> tuple[list[dict], dict]:
    """The sessions' statements and their verdicts of record without the sessions: each member's public file as the
    served path writes it for the written window (`flock_input_files`, as a5 stages a session), the verifier's reads,
    and a stand-in for what `flock-verify verify`'s verdict of record names of the session it checked."""
    from verity_flock.class_statement import member_stager

    def sha512(j) -> str:
        return hashlib.sha512(canonical_json(j)).hexdigest()

    parts = SS.parts(Q, P.strata(mine.program, mine.query))
    bind = {"partition": mine.partition_digest, "program": mine.partition["program"]}
    stmts, verdicts = [], {}
    for k, m in enumerate(mine.members):
        text = member_stager(TS.template_fn(m.template), m, bind).text
        with open(FI.flock_input_files(interior, k, text, served, write=("pub",))["pub"], "rb") as f:
            header = json.loads(f.readline())
        s = {"template": m.template, "header": header, "draw": SS.own_draw(Q, parts[k], drawn), "registered": reads[m.key]}
        stmts.append(s)
        verdicts[f"lean-verify/m{k}"] = {
            "of_record": True, "accepted": True, "public_header_sha512": sha512(header), "unit_draw": s["draw"],
            "session_tables": 1, "registered_sha512": sha512(s["registered"]),
            "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in s["registered"]["ports"].items()}}
    return stmts, verdicts


def test_a_class_whose_block_makes_no_wires_has_no_second_stage_and_one_of_instances_has_its_design():
    """An inlined token select's replay block gathers each token's chunks, which round 2 derives no wire from
    (``second-wires``): `design` refuses the class by that name before anything is replayed, and gives it unlinked
    only when asked, its templates' proof blocks with no link, wire or return; its rule registers no value of the
    block's. The class of whole instances reads the block's ``logits`` at one template's port, one row a served token,
    and returns its ``o0``, unit j of a block writing its row j: held where round 1's record commits the port, as the
    small class's does, and not where it doesn't."""
    with pytest.raises(ValueError, match="makes no wires round 2 derives"):
        TS.design(_layout(inline=True))
    u = TS.design(_layout(inline=True), unlinked=True)
    assert (u.cells, u.links, u.wires, u.returns) == ((O * V // 64, O), (), (), ())
    assert TS.rule_of(_layout(inline=True), f"rate:{Q}/{N}", K).span_values == ()
    assert TS.design(_layout(), unlinked=True) == TS.design(_layout())
    d = TS.design(_layout())
    assert (d.cells, d.wires) == ((O,), ())
    assert [(x.template, x.port, x.value, x.per, x.at) for x in d.links] == [(TID, "logits", "logits", O, tuple(range(O)))]
    assert d.returns == (IR.Return(TID, "o0", "o0", O, tuple(range(O)), True),)
    assert d.returns[0].rows([5, 2]) == [5 * O + j for j in range(O)] + [2 * O + j for j in range(O)]
    bare = TS.design(_layout(), {"roots": [{"port": p} for p in BR.PORTS]})
    assert bare.returns == (replace(d.returns[0], held=False),) and bare.links == d.links
    rule = TS.rule_of(_layout(), f"rate:{Q}/{N}", K)
    assert rule.span_values == (("logits", O, (V, 16)),)
    assert [n for n, _s in rule.interior] == [f"{TID}/logits", f"{TID}/o0"]


def test_the_interior_is_registered_after_round_1_under_no_name_round_1_has_and_sampled(run):
    """The written record is the one registered: after the boundary registration, over the sampled spans in the
    sample's order, registering the block's ``logits`` and nothing round 1's record names; the second stage samples
    K proof blocks in each sampled span."""
    again, r, tmp = run["again"], run["r"], run["tmp"]
    sample = [int(u) for u in again.sample.units]
    assert r.record == json.loads((tmp / "interior" / "registration.json").read_text()) == r.registration.record
    assert r.record["previous"] == again.registration.digest and r.record["window"]["replay_sample"] == sample
    assert r.record["window"]["run"] == IR.interior_run(again.registration.digest, sample)
    earlier = {x["port"] for x in again.record["roots"]} | {e["value"] for e in again.record.get("registered", ())}
    assert earlier == {*BR.PORTS, "o0"}
    assert [e["value"] for e in r.record["registered"]] == ["logits"]
    assert not earlier & ({x["port"] for x in r.record["roots"]} | {"logits"})
    assert r.launches == {f"r{u}": f"launch:r{u}" for u in sample}
    assert r.draw.replay_sample == again.sample and [u // O for u in r.draw.units] == list(range(Q))


def test_the_replay_side_refuses_a_capture_that_is_not_the_sampled_spans_or_a_return_round_1_did_not_commit(run):
    """Nothing is committed from a capture of other requests, of other ids than the serve emitted (named by how many
    differ, never by id), under another commitment than the first stage registered, or from a replay whose units
    return other rows than round 1 committed for them, refused by slot and row."""
    again, c, tmp, capture = run["again"], run["class"], run["tmp"], run["capture"]
    sample = [int(u) for u in again.sample.units]
    with pytest.raises(ValueError, match="holds no span of request 'other'"):
        TS.spans_of(c, capture, sample, {f"r{sample[0]}": "other"})
    with pytest.raises(ValueError, match="didn't reproduce the span"):
        TS.spans_of(c, capture, sample, {f"r{sample[0]}": f"r{sample[1]}"})
    with pytest.raises(ValueError, match="not the commitment its first stage registered"):
        TS.replayed(replace(again, pub={**again.pub, "index": "00" * 64}), c, capture, tmp / "never")
    d, (per, _launches) = run["r"].design, TS.spans_of(c, capture, sample)
    port, x = c.port("o0"), run["r"].design.returns[0]
    rows = port.rows.copy()
    rows[sample[1] * O + 2, 0] ^= 1
    with pytest.raises(ValueError, match=f"unit {O + 2} \\(sampled span 1\\) returns at its 'o0' a row that is not row "
                                         f"{sample[1] * O + 2} of o0, the return round 1 committed"):
        IR.interior(per, d.cells, again.registration.digest, sample, row=0, links=d.links, returns=[(x, replace(port, rows=rows))])
    with pytest.raises(ValueError, match="which round 1's record doesn't commit"):
        IR.interior(per, d.cells, again.registration.digest, sample, row=0, links=d.links,
                    returns=[(replace(x, held=False), port)])
    assert not (tmp / "never").exists()


def test_the_interior_commits_the_return_under_the_serves_salts_and_its_header_opens_round_1s_rows(run):
    """The output that writes the block's return is committed under the serve's salts at the rows its units write: each
    of its Q·O strings is round 1's leaf's at ``sample[i]·O + j``, and the member's header names round 1's port, those
    rows and each one's path in round 1's tree, whose root is the boundary record's."""
    again, r, c, tmp = run["again"], run["r"], run["class"], run["tmp"]
    sample = [int(u) for u in again.sample.units]
    at = [s * O + j for s in sample for j in range(O)]
    port, mine = c.port("o0"), r.interior.committed
    assert port.root == c.roots["o0"] and port.leaves == N * O and port.words == 2
    n = CS.BC_BYTES
    assert mine.bc[f"{TID}/o0"] == b"".join(port.bcs[a * n:(a + 1) * n] for a in at)
    assert mine.salts[f"{TID}/o0"] == port.salts_at(at) != mine.salts[f"{TID}/logits"][:len(at) * CS.SALT_BYTES]
    assert [x.value for x, _v in r.interior.returns] == ["o0"]
    with pytest.raises(ValueError, match="no value the class commits rows of"):
        c.port("logits")
    with pytest.raises(ValueError, match="recommit to another root"):
        replace(c, salts={**c.salts, "o0": os.urandom(len(c.salts["o0"]))}).port("o0")
    entry = CS.registered_header(json.loads((tmp / "interior" / "window.json").read_text()), 0)["o0"]
    assert entry == port.header(at) and LK.Port.of(mine, f"{TID}/o0").root != port.root


@pytest.mark.slow
def test_a_serve_under_another_nonce_commits_under_another_run(tmp_path, served):
    """The small record's run is its serve nonce's (`boundary_rows.run_root`). A challenger host opens a run's root
    once, ever, so a rehearsal on one serves under a nonce no run has used, and the class commits under that run."""
    got = SM.serve(tmp_path, V, 2, 2, nonce="another")
    pub = json.loads((got["class"] / BR.COMMIT).read_text())
    assert pub["run"] == BR.run_root("another", 0) != served["pub"]["run"] == BR.run_root(SM.SERVE, 0)


def test_round_2_takes_the_hand_over_with_the_return_read_at_round_1s_rows(run):
    """What `two_stage.prove` hands over, held to round 2 (`two_stage.round_two`, the form allowed): the verifier's
    window (`Design.window`, public items alone) is the one the interior was committed under; the verifier's reads
    (`verifier_reads`) put the ``logits`` port at the value the interior registers, each unit at its own row, and the
    ``o0`` port at round 1's port of that name, at the root, leaves and domain the boundary record gives it, unit j of
    the i-th sampled block at row ``sample[i]·O + j``; the development params (`references`) name the circuit the
    sessions' public files name and the same two values; and the public files the served path writes for the written
    window (`flock_input_files`, as a5 stages the sessions), each named by a stand-in verdict of record, pass every
    check of round 2. With the return read at the interior's own port instead, as a record of round 1 without the
    port leaves it, round 2 refuses that read (``second-read``), which is not the params' either (``second-values``).
    Slow (about 15 s): it lowers the template for the references and the sessions' files."""
    again, r, tmp = run["again"], run["r"], run["tmp"]
    d, brec, sample, drawn = r.design, again.record, [int(u) for u in again.sample.units], list(r.draw.units)
    mine, w = d.window(again.registration.digest, sample), r.interior.committed.window
    assert (mine.partition_digest, mine.run, mine.population) == (w.partition_digest, w.run, w.population)
    assert CS.expected_ports(mine) == r.interior.ports
    check = R.check(r.record, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), r.interior.leaf_layer,
                    (), [brec], registered=r.tie["registered"], schemas=r.tie["schemas"])
    assert check.ok, check.codes
    assert TS.interior_source(mine, sample)(r.record["window"])["ok"]
    assert TS.interior_source(mine, sample[::-1])(r.record["window"])["codes"] == ["replay_sample"]

    reads = TS.verifier_reads(d, mine, r.record, brec, sample)
    (key, file), = reads.items()
    value, = r.record["registered"]
    assert file["ports"]["logits"] == {
        "value": "logits", "root": value["root"], "leaves": Q * O, "words": V, "word_bits": 16, "domain": value["domain"],
        "row_schema": mine.members[0].schema("logits"), "reads": list(range(Q * O)), "reads_from": 0}
    served, = (x for x in brec["roots"] if x["port"] == "o0")
    assert file["ports"]["o0"] == {
        "value": "o0", "root": served["root"], "leaves": N * O, "words": 2, "word_bits": 16, "domain": served["domain"],
        "row_schema": mine.members[0].schema("o0"), "reads": [s * O + j for s in sample for j in range(O)], "reads_from": 0}
    with pytest.raises(ValueError, match="a replay sample of 1 for a window of 2"):
        TS.verifier_reads(d, mine, r.record, brec, sample[:1])

    proofs = TS.references(d, Q, tmp / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    assert not (tmp / "references" / "m0" / "inst.bin").exists()
    assert {p: e["value"] for p, e in proofs["templates"][0]["registered"]["ports"].items()} == {
        "logits": "logits", "o0": "o0"}

    stmts, verdicts = _handed(tmp / "interior", tmp / "served", mine, reads, drawn)
    parts = SS.parts(Q, P.strata(mine.program, mine.query))
    assert TS.pins(proofs, stmts) == {TID: True}
    layout = again.layout
    got = TS.round_two(layout.program, d.key, sample, mine, r.record, brec, drawn, stmts, {"verdicts": verdicts}, proofs)
    assert got == {"ok": True, "codes": [], "detail": {}}
    assert [(x.port, x.value) for x in SS.replay_links(layout.program, d.key, brec)] == [("logits", "logits"), ("o0", "o0")]
    assert parts[0].n == O

    own = TS.verifier_reads(replace(d, returns=()), mine, r.record, brec, sample)
    assert own[key]["ports"]["o0"]["value"] == f"{TID}/o0" and own[key]["ports"]["o0"]["reads"] == list(range(Q * O))
    stmts, verdicts = _handed(tmp / "interior", tmp / "served-own", mine, own, drawn)
    got = TS.round_two(layout.program, d.key, sample, mine, r.record, brec, drawn, stmts, {"verdicts": verdicts}, proofs)
    assert (got["ok"], got["codes"]) == (False, ["second-read", "second-values"])
    assert got["detail"]["second-read"] == {"statement": 0, "template": TID}


@pytest.mark.slow
def test_an_unlinked_class_is_committed_and_proved_blockwise_and_round_2_refuses_its_wires(tmp_path):
    """A class as a served vocabulary's is, its replay block the check's body inlined (chunks and a combine a token; a
    vocabulary of 4,096, the smallest whose combine has a row past one hash block, which Flock's composer needs):
    `replayed` refuses it by name, and with ``unlinked`` commits the sampled spans' chunks and combines as the
    interior, registering no value, and samples K of each in every sampled span. The verifier's reads then put every
    port at the interior's own, and round 2 refuses the statements by ``second-wires`` alone: the sessions prove each
    block and tie none to another or to round 1. Slow (about 10 s): it lowers both templates twice."""
    v = 4096
    got, check = SM.serve(tmp_path, v, O, N, inline=True), BR.check_template(SM.template(v))
    c, pub = BR.reopen(got["class"], check), json.loads((got["class"] / BR.COMMIT).read_text())
    first = TS.first(pub, check, TS.rule_of(c.layout, f"rate:{Q}/{N}", K, 60.0), FC.Challenger("first"))
    again = TS.resume(json.loads(json.dumps(first.stage1())), check, FC.Challenger("second"))
    with pytest.raises(ValueError, match="makes no wires round 2 derives"):
        TS.replayed(again, c, got["capture"], tmp_path / "never")
    r = TS.replayed(again, c, got["capture"], tmp_path / "interior", unlinked=True)
    d, sample, drawn = r.design, [int(u) for u in again.sample.units], list(r.draw.units)
    chunks = O * v // 64
    assert d.cells == (chunks, O) and "registered" not in r.record and r.tie == {}
    assert sorted(u // chunks for u in drawn if u < Q * chunks) == list(range(Q))
    assert sorted((u - Q * chunks) // O for u in drawn if u >= Q * chunks) == list(range(Q))
    mine = d.window(again.registration.digest, sample)
    assert CS.expected_ports(mine) == r.interior.ports
    reads = TS.verifier_reads(d, mine, r.record, again.record, sample)
    assert all(e["value"] == f"{m.template}/{p}" for m in mine.members for p, e in reads[m.key]["ports"].items())
    proofs = TS.references(d, Q, tmp_path / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    stmts, verdicts = _handed(tmp_path / "interior", tmp_path / "served", mine, reads, drawn)
    assert all(TS.pins(proofs, stmts).values())
    r2 = TS.round_two(c.layout.program, d.key, sample, mine, r.record, again.record, drawn, stmts, {"verdicts": verdicts}, proofs)
    assert (r2["ok"], r2["codes"]) == (False, ["second-wires"])


@pytest.mark.slow
def test_an_inlined_class_of_the_greedy_select_in_units_is_linked_and_round_2_takes_its_hand_over(tmp_path):
    """A class as a served vocabulary's is with the greedy select in units, its replay block
    `GreedySelect_v1{V, N=V / K}`'s body inlined: one unit of each of its K templates a token, the combine's every
    wired row one chunk's output row (K = 4 here; a served row's is 20). Round 2 derives its wires, a link from each
    template's ``logits`` to the block's and the return link from the combine's output, so the class's record commits
    the ids as its return's port, `replayed` commits it linked (`interior_rows.pieces`), and round 2 takes the
    hand-over with no code, the combine's ``o0`` read at round 1's rows. With the return read at the interior's own
    port, as a record without the port leaves it, round 2 refuses that read at the combine's statement
    (``second-read``). Slow (about 16 s): it lowers the K templates for the references and the sessions' files."""
    v, k = 256, 4
    got, check = SM.serve(tmp_path, v, O, N, inline=True, units=k), BR.check_template(SM.template(v, k))
    c, pub = BR.reopen(got["class"], check), json.loads((got["class"] / BR.COMMIT).read_text())
    assert c.layout.inline and len(c.layout.templates) == k and c.layout.returned == "o0"
    first = TS.first(pub, check, TS.rule_of(c.layout, f"rate:{Q}/{N}", K, 60.0), FC.Challenger("first"))
    again = TS.resume(json.loads(json.dumps(first.stage1())), check, FC.Challenger("second"))
    r = TS.replayed(again, c, got["capture"], tmp_path / "interior")
    d, brec, sample, drawn = r.design, again.record, [int(u) for u in again.sample.units], list(r.draw.units)
    combine = f"GreedyCombine_v2{{V={v},N={v // k}}}"
    assert d.cells == (O,) * k and [m.template for m in d.members][-1] == combine
    assert [(w.template, w.port, w.wport) for w in d.wires] == [(combine, f"r{j}", "o0") for j in range(k - 1)]
    assert [w.writer for w in d.wires] == [m.template for m in d.members[:-1]]
    assert [(x.template, x.port, x.value) for x in d.links] == [(m.template, "logits", "logits") for m in d.members]
    assert d.returns == (IR.Return(combine, "o0", "o0", O, tuple(range(O)), True),)
    assert [e["value"] for e in r.record["registered"]] == ["logits"] and [x.value for x, _v in r.interior.returns] == ["o0"]
    assert sorted(u // O for u in drawn) == sorted(t * Q + s for t in range(k) for s in range(Q))

    mine = d.window(again.registration.digest, sample)
    assert CS.expected_ports(mine) == r.interior.ports
    reads = TS.verifier_reads(d, mine, r.record, brec, sample)
    served, = (x for x in brec["roots"] if x["port"] == "o0")
    last = mine.members[-1]
    assert reads[last.key]["ports"]["o0"] == {
        "value": "o0", "root": served["root"], "leaves": N * O, "words": 2, "word_bits": 16, "domain": served["domain"],
        "row_schema": last.schema("o0"), "reads": [s * O + j for s in sample for j in range(O)], "reads_from": last.base}
    proofs = TS.references(d, Q, tmp_path / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    stmts, verdicts = _handed(tmp_path / "interior", tmp_path / "served", mine, reads, drawn)
    assert len(stmts) == k and all(TS.pins(proofs, stmts).values())
    args = (c.layout.program, d.key, sample, mine, r.record, brec, drawn)
    assert TS.round_two(*args, stmts, {"verdicts": verdicts}, proofs) == {"ok": True, "codes": [], "detail": {}}

    own = TS.verifier_reads(replace(d, returns=()), mine, r.record, brec, sample)
    assert own[last.key]["ports"]["o0"]["value"] == f"{combine}/o0"
    stmts, verdicts = _handed(tmp_path / "interior", tmp_path / "served-own", mine, own, drawn)
    got = TS.round_two(*args, stmts, {"verdicts": verdicts}, proofs)
    assert (got["ok"], got["codes"][0]) == (False, "second-read")
    assert got["detail"]["second-read"] == {"statement": k - 1, "template": combine}


def test_an_inlined_class_of_the_greedy_select_beside_history_tables_takes_both_rounds(tmp_path):
    """The two-stage serve's shape with no stop rule: the greedy select in units inlined, the ids served committed as
    its replay block's return, and each span's history table beside them (``two_stage_small --tables``). The record
    lists the tables' roots after the return, which a record of tables names ``<block>/o0`` (`boundary_rows.return_port`);
    the first stage registers and samples the class, the replay side commits its interior linked, and round 2 takes the
    hand-over with no code, the return read at round 1's rows. No unit reads a table, so round 2 opens none. Slow
    (about 16 s): it lowers the K templates for the references and the sessions' files."""
    v, k = 256, 4
    got, check = SM.serve(tmp_path, v, O, N, inline=True, units=k, tables=True), BR.check_template(SM.template(v, k))
    c, pub = BR.reopen(got["class"], check), json.loads((got["class"] / BR.COMMIT).read_text())
    names, ret = list(c.layout.tables), c.layout.returned
    assert len(names) == N and list(pub["values"]) == [*BR.PORTS, ret, *names] and ret.endswith("/o0") and pub["ids"]
    first = TS.first(pub, check, TS.rule_of(c.layout, f"rate:{Q}/{N}", K, 60.0), FC.Challenger("first"))
    again = TS.resume(json.loads(json.dumps(first.stage1())), check, FC.Challenger("second"))
    r = TS.replayed(again, c, got["capture"], tmp_path / "interior")
    d, brec, sample, drawn = r.design, again.record, [int(u) for u in again.sample.units], list(r.draw.units)
    assert [x["port"] for x in brec["roots"]] == [*BR.PORTS, ret, *names]
    assert [x.value for x in d.returns] == [ret] and [x.value for x, _v in r.interior.returns] == [ret]
    mine = d.window(again.registration.digest, sample)
    reads = TS.verifier_reads(d, mine, r.record, brec, sample)
    proofs = TS.references(d, Q, tmp_path / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    stmts, verdicts = _handed(tmp_path / "interior", tmp_path / "served", mine, reads, drawn)
    args = (c.layout.program, d.key, sample, mine, r.record, brec, drawn)
    assert TS.round_two(*args, stmts, {"verdicts": verdicts}, proofs) == {"ok": True, "codes": [], "detail": {}}


def _engine(v: int, spans: int, outputs: int, lengths: dict[str, int] | None = None) -> tuple[dict, list[dict], list[dict]]:
    """A greedy engine's steps over ``spans`` requests of ``outputs`` tokens each at a vocabulary of ``v`` (request
    ``r<i>`` of ``lengths[r<i>]`` where given, leaving the batch after it): each step's bf16 logits rows (one a request
    in the batch, standard normal, so ties within a row are bf16's; a request's rows are the same whatever the others'
    lengths), its batch as vLLM's V2 runner hands the sampler one, the ids its greedy sampler takes (torch's argmax, the
    first maximum), and the step as a replay's trace records it."""
    import torch
    from types import SimpleNamespace as NS
    g = torch.Generator().manual_seed(1112)
    rids, lp = [f"r{i}" for i in range(spans)], len(SM.PROMPT)
    steps, trace = [], []
    for s in range(outputs):
        live = [i for i, r in enumerate(rids) if s < (lengths or {}).get(r, outputs)]
        on = [rids[i] for i in live]
        logits = torch.randn(spans, v, generator=g).to(torch.bfloat16)[live]
        batch = NS(req_ids=on, num_reqs=len(on), num_draft_tokens=0, prefill_len_np=[lp] * len(on),
                   num_computed_tokens_np=[0 if s == 0 else lp + s - 1] * len(on),
                   num_scheduled_tokens=[lp if s == 0 else 1] * len(on))
        ids = logits.float().argmax(dim=1)
        steps.append({"logits": logits, "batch": batch, "ids": ids, "on": on})
        trace.append({"step": s, "emitted": dict.fromkeys(on, 1),
                      "rows": [[r, 0 if s == 0 else lp + s - 1, lp if s == 0 else 1] for r in on]})
    return {"rids": rids}, steps, trace


def test_the_serves_hook_and_the_samplers_capture_give_a_class_of_the_greedy_select_in_units(tmp_path):
    """The engine's side of a class of the greedy select in units, from the launch line's boundary object (inline, one
    instance an output, ``ids``) as serve_tax hands the plugin it: the serve's hook (`BoundaryWindows.of`) commits the
    spans a greedy engine served as the stand-in serve's layout lays its class out (`two_stage_small.layout`: program,
    partition, every value's domain, the provenance; the ids its return), and the sampler's wrapper on the same steps
    (`sampler_capture.Capture`, `take`, `spans`, `write`, under the template the serve committed) captures rows the
    driver lays out as the class's units, one of each of its K templates a token (`two_stage.spans_of`), each a
    span whose combine selects the ids the serve committed. The class designs with every template linked, the combine
    wired to each chunk and its return held, and its rule passes at more than one output a span (`two_stage.rule_of`)."""
    pytest.importorskip("torch")
    from types import SimpleNamespace as NS
    from verity_vllm.engine import sampler_capture as SC
    from verity_vllm.pipeline.boundary_windows import BoundaryWindows
    v, k, spans, outputs = 160, 20, 3, 4
    check = SM.template(v, k)
    cfg = {"window_s": 3600.0, "check": check, "inline": True, "per_prompt": 0, "per_output": 1, "drained": True,
           "ids": True}
    lines: list[dict] = []
    hook = BoundaryWindows.of(cfg, tmp_path / "boundary", sink=lines.append)
    eng, steps, trace = _engine(v, spans, outputs)
    for rid in eng["rids"]:
        hook.admit(NS(request_id=rid, prompt_token_ids=list(SM.PROMPT),
                      sampling_params=NS(**{**SM.GREEDY, "max_tokens": outputs})))
    sampler = SC.Capture(lambda logits, batch: NS(sampled_token_ids=logits.float().argmax(dim=1)[:, None]), eng["rids"])
    for s, st in enumerate(steps):
        out = sampler(st["logits"], st["batch"])
        hook.served({0: NS(outputs=[NS(request_id=r, new_token_ids=[int(out.sampled_token_ids[i, 0])],
                                       finish_reason="length" if s == outputs - 1 else None)
                                    for i, r in enumerate(eng["rids"])])})
    hook.stop()
    assert [ln.get("error") for ln in lines] == [None]
    [directory] = (tmp_path / "boundary" / "w0").iterdir()
    assert directory.name.endswith(f"-o{outputs}")
    c = BR.reopen(directory, BR.check_template(check))
    mine = SM.layout(v, outputs, spans, True, hook.serve, k)
    L = c.layout
    assert (L.partition["program"], P.digest(L.partition), L.ports(), L.provenance()) == (
        mine.partition["program"], P.digest(mine.partition), mine.ports(), mine.provenance())
    assert L.returned == "o0" and L.provenance()["ids"] is True and len(L.templates) == k
    served = {b.engine_id: list(b.served) for b in c.spans}
    assert served == {r: [int(st["ids"][i]) for st in steps] for i, r in enumerate(eng["rids"])}

    take = SC.take(NS(model_runner=NS(sampler=sampler), rank=0), str(tmp_path / "ranks"))
    SC.write(tmp_path / "capture", check, SC.spans([take], trace))
    sample = [0, 2]
    per, launches = TS.spans_of(c, tmp_path / "capture", sample)
    assert sorted(launches) == [c.spans[u].engine_id for u in sample]
    units = [P.template_id(t) for t in L.templates]
    for u, pieces in zip(sample, per, strict=True):
        assert [p.member.template for p in pieces] == units and all(p.n == outputs for p in pieces)
        assert [SM.token(w) for w in pieces[-1].data["outs"]["o0"].tolist()] == served[c.spans[u].engine_id]
    d = TS.design(L)
    combine = units[-1]
    assert d.cells == (outputs,) * k and [x.template for x in d.links] == units
    assert [(w.template, w.port) for w in d.wires] == [(combine, f"r{j}") for j in range(k - 1)]
    assert d.returns == (IR.Return(combine, "o0", "o0", outputs, tuple(range(outputs)), True),)
    assert TS.rule_of(L, f"rate:{Q}/{spans}", K).span_values == (("logits", outputs * k, (v // k, 16)),)


def test_under_a_stop_rule_the_serves_hook_gives_a_stop_class_of_the_greedy_select_in_units_committing_its_ids(tmp_path):
    """The same engine's side under a stop rule (the boundary object's ``stop``, T the outputs, no tables): a request
    the greedy engine ends at the rule's stop id and those it ends at ``max_tokens`` are one stop class, (P, T), which
    commits the ids its replay block returns. The driver lays each sampled span's rows out as the K units of its own
    tokens (`two_stage.spans_of`), the class designs wired with its return held and its rule passes (`design`,
    `rule_of`), its first stage samples it under the verifier's rule, and its stop blocks register after its boundary
    record and are sampled on the stop population's stream (`stop_first`)."""
    pytest.importorskip("torch")
    from types import SimpleNamespace as NS
    from verity_vllm.engine import sampler_capture as SC
    from verity_vllm.pipeline import boundary_stop as BS
    from verity_vllm.pipeline.boundary_windows import BoundaryWindows
    v, k, spans, outputs = 160, 20, 3, 4
    check = SM.template(v, k)
    _eng, full, _trace = _engine(v, spans, outputs)
    whole = {r: [int(st["ids"][i]) for st in full] for i, r in enumerate(_eng["rids"])}
    rule = BS.Stop((whole["r0"][1],), SM.MAX_MODEL_LEN, outputs)
    lengths = {r: t.index(rule.ids[0]) + 1 if rule.ids[0] in t else outputs for r, t in whole.items()}
    assert lengths["r0"] == 2 and outputs in lengths.values()
    cfg = {"window_s": 3600.0, "check": check, "inline": True, "per_prompt": 0, "per_output": 1, "drained": True,
           "ids": True, "stop": rule.to_json()}
    lines: list[dict] = []
    hook = BoundaryWindows.of(cfg, tmp_path / "boundary", sink=lines.append)
    eng, steps, trace = _engine(v, spans, outputs, lengths)
    for rid in eng["rids"]:
        hook.admit(NS(request_id=rid, prompt_token_ids=list(SM.PROMPT),
                      sampling_params=NS(**{**SM.GREEDY, "max_tokens": outputs, "ignore_eos": False})))
    sampler = SC.Capture(lambda logits, batch: NS(sampled_token_ids=logits.float().argmax(dim=1)[:, None]), eng["rids"])
    for s, st in enumerate(steps):
        out = sampler(st["logits"], st["batch"])
        ids = [int(x) for x in out.sampled_token_ids[:, 0]]
        hook.served({0: NS(outputs=[NS(request_id=r, new_token_ids=[ids[i]],
                                       finish_reason=None if s + 1 < lengths[r] else
                                       "stop" if ids[i] in rule.ids else "length")
                                    for i, r in enumerate(st["on"])])})
    hook.stop()
    assert [(ln.get("error"), ln.get("filed")) for ln in lines] == [(None, {"stop": spans, "nostop": 0})]
    [directory] = (tmp_path / "boundary" / "w0").iterdir()
    assert directory.name.endswith(f"-o{outputs}")
    c = BR.reopen(directory, BR.check_template(check), stop=rule)
    L, pub = c.layout, json.loads((directory / BR.COMMIT).read_text())
    assert L.returned == "o0" and L.provenance()["ids"] is True and len(L.templates) == k
    assert list(pub["values"]) == [*BR.PARTS, "o0", BR.STOP]
    served = {b.engine_id: list(b.served) for b in c.spans}
    assert served == {r: t[:lengths[r]] for r, t in whole.items()}

    take = SC.take(NS(model_runner=NS(sampler=sampler), rank=0), str(tmp_path / "ranks"))
    SC.write(tmp_path / "capture", check, SC.spans([take], trace))
    sample = list(range(spans))
    per, _launches = TS.spans_of(c, tmp_path / "capture", sample)
    units = [P.template_id(t) for t in L.templates]
    for u, pieces in zip(sample, per, strict=True):
        rid = c.spans[u].engine_id
        assert [p.member.template for p in pieces] == units and all(p.n == lengths[rid] for p in pieces)
        assert [SM.token(w) for w in pieces[-1].data["outs"]["o0"].tolist()] == served[rid]
    d = TS.design(L)
    combine = units[-1]
    assert [x.template for x in d.links] == units
    assert [(w.template, w.port) for w in d.wires] == [(combine, f"r{j}") for j in range(k - 1)]
    assert [(r.template, r.port, r.held) for r in d.returns] == [(combine, "o0", True)]
    rule_of = TS.rule_of(L, f"rate:{Q}/{spans}", K)
    assert rule_of.span_values == (("logits", outputs * k, (v // k, 16)),)

    got = TS.first(pub, BR.check_template(check), rule_of, FC.Challenger("first"), stop=rule)
    assert [r["port"] for r in got.registration.record["roots"]] == [*BR.PARTS, "o0"]
    s = TS.stop_first(pub, BR.check_template(check), rule, f"rate:{Q}/{spans}", FC.Challenger("stop"), got.record)
    assert s["record"]["previous"] == got.registration.digest and s["record"]["population"] == spans
    assert [r["port"] for r in s["record"]["roots"]] == [BR.STOP] and len(s["units"]) == Q
