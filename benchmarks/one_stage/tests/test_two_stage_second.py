"""A served class's second stage as `two_stage.py second` runs it, at small statics: a token select class as a serve
and its replay leave it (`two_stage_small.serve`: the class's files, its replay block the check's instances, one a served
token, and a capture of the sampler's inputs), its first stage on a test's own challenger (`fake_challenger`), and the
replay side's half from ``stage1.json``, the class's files and the capture (`two_stage.replayed`): the interior
committed, written as the one-stage audit reads a window, registered on the class's stream and sampled. Then what the
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
from verity.commitments.windows import flock_inputs as FI, rows as CS
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
    and returns its ``o0``."""
    with pytest.raises(ValueError, match="makes no wires round 2 derives"):
        TS.design(_layout(inline=True))
    u = TS.design(_layout(inline=True), unlinked=True)
    assert (u.cells, u.links, u.wires, u.returns) == ((O * V // 64, O), (), (), ())
    assert TS.rule_of(_layout(inline=True), f"rate:{Q}/{N}", K).span_values == ()
    assert TS.design(_layout(), unlinked=True) == TS.design(_layout())
    d = TS.design(_layout())
    assert (d.cells, d.wires) == ((O,), ())
    assert [(x.template, x.port, x.value, x.per, x.at) for x in d.links] == [(TID, "logits", "logits", O, tuple(range(O)))]
    assert d.returns == (SS.Link(TID, "o0", "o0"),)
    rule = TS.rule_of(_layout(), f"rate:{Q}/{N}", K)
    assert rule.span_values == (("logits", O, (V, 16)),)
    assert [n for n, _s in rule.interior] == [f"{TID}/logits", f"{TID}/o0"]


def test_the_interior_is_registered_after_round_1_under_no_name_round_1_has_and_sampled(run):
    """The written record is the one registered: after the boundary registration, over the sampled spans in the
    sample's order, registering the block's ``logits`` and nothing round 1's record names; the second stage samples
    K proof blocks in each sampled span."""
    again, r, tmp = run["again"], run["r"], run["tmp"]
    sample = [int(u) for u in again.draw.units]
    assert r.record == json.loads((tmp / "interior" / "registration.json").read_text()) == r.registration.record
    assert r.record["previous"] == again.registration.digest and r.record["window"]["replay_sample"] == sample
    assert r.record["window"]["run"] == IR.interior_run(again.registration.digest, sample)
    earlier = {x["port"] for x in again.record["roots"]} | {e["value"] for e in again.record.get("registered", ())}
    assert [e["value"] for e in r.record["registered"]] == ["logits"]
    assert not earlier & ({x["port"] for x in r.record["roots"]} | {"logits"})
    assert r.launches == {f"r{u}": f"launch:r{u}" for u in sample}
    assert r.draw.replay_sample == again.draw and [u // O for u in r.draw.units] == list(range(Q))


def test_the_replay_side_refuses_a_capture_that_is_not_the_sampled_spans_or_a_return_round_1_commits(run):
    again, c, tmp, capture = run["again"], run["class"], run["tmp"], run["capture"]
    sample = [int(u) for u in again.draw.units]
    with pytest.raises(ValueError, match="holds no span of request 'other'"):
        TS.spans_of(c, capture, sample, {f"r{sample[0]}": "other"})
    with pytest.raises(ValueError, match="didn't reproduce the span"):
        TS.spans_of(c, capture, sample, {f"r{sample[0]}": f"r{sample[1]}"})
    with pytest.raises(ValueError, match="not the commitment its first stage registered"):
        TS.replayed(replace(again, pub={**again.pub, "index": "00" * 64}), c, capture, tmp / "never")
    named = replace(again, record={**again.record, "roots": [*again.record["roots"], {"port": "o0"}]})
    with pytest.raises(ValueError, match="commits the block's return"):
        TS.replayed(named, c, capture, tmp / "never")
    assert not (tmp / "never").exists()


def test_the_return_salts_are_the_serves_at_each_sampled_blocks_row_and_none_on_a_record_without_the_port(run):
    """Round 2's return check will take, for each sampled proof block, the salt of the row its return link reads in
    round 1's port. Today's class commits no such port, so there are none; with one (stand-in salts under the link's
    name, O rows a span), each is the row of the block's own span and unit."""
    again, r, c = run["again"], run["r"], run["class"]
    sample, drawn = [int(u) for u in again.draw.units], list(r.draw.units)
    assert TS.return_salts(c, r.design, sample, drawn) == {}
    blob = os.urandom(N * O * CS.SALT_BYTES)
    got = TS.return_salts(replace(c, salts={**c.salts, "o0": blob}), r.design, sample, drawn)
    assert list(got) == ["o0"] and sorted(got["o0"]) == drawn
    for u in drawn:
        row = sample[u // O] * O + u % O
        assert got["o0"][u] == blob[row * CS.SALT_BYTES:(row + 1) * CS.SALT_BYTES].hex()


@pytest.mark.slow
def test_a_serve_under_another_nonce_commits_under_another_run(tmp_path, served):
    """The small record's run is its serve nonce's (`boundary_rows.run_root`). A challenger host opens a run's root
    once, ever, so a rehearsal on one serves under a nonce no run has used, and the class commits under that run."""
    got = SM.serve(tmp_path, V, 2, 2, nonce="another")
    pub = json.loads((got["class"] / BR.COMMIT).read_text())
    assert pub["run"] == BR.run_root("another", 0) != served["pub"]["run"] == BR.run_root(SM.SERVE, 0)


def test_round_2_takes_the_hand_over_but_for_the_return_read_round_1_commits_no_port_for(run):
    """What `two_stage.prove` hands over, held to round 2 (`two_stage.round_two`, the form allowed): the verifier's
    window (`Design.window`, public items alone) is the one the interior was committed under; the verifier's reads
    (`verifier_reads`) put the ``logits`` port at the value the interior registers, each unit at its own row, and the
    ``o0`` port at the interior's own; the development params (`references`) name the circuit the sessions' public
    files name; and the public files the served path writes for the written window (`flock_input_files`, as a5 stages
    the sessions), each named by a stand-in verdict of record, pass every check of round 2 but ``second-read``. The
    one read it refuses is the return's: round 1's record commits no port for the block's return, and the interior may
    not register it, so with the return link left out of the design round 2 accepts. Slow (about 15 s): it lowers the
    template for the references and the sessions' files."""
    again, r, tmp = run["again"], run["r"], run["tmp"]
    d, brec, sample, drawn = r.design, again.record, [int(u) for u in again.draw.units], list(r.draw.units)
    mine, w = d.window(again.registration.digest, sample), r.interior.committed.window
    assert (mine.partition_digest, mine.run, mine.population) == (w.partition_digest, w.run, w.population)
    assert CS.expected_ports(mine) == r.interior.ports
    check = R.check(r.record, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine), r.interior.leaf_layer,
                    (), [brec], registered=r.tie["registered"], schemas=r.tie["schemas"])
    assert check.ok, check.codes
    assert TS.interior_source(mine, sample)(r.record["window"])["ok"]
    assert TS.interior_source(mine, sample[::-1])(r.record["window"])["codes"] == ["replay_sample"]

    reads = TS.verifier_reads(d, mine, r.record)
    (key, file), = reads.items()
    value, = r.record["registered"]
    assert file["ports"]["logits"] == {
        "value": "logits", "root": value["root"], "leaves": Q * O, "words": V, "word_bits": 16, "domain": value["domain"],
        "row_schema": mine.members[0].schema("logits"), "reads": list(range(Q * O)), "reads_from": 0}
    assert file["ports"]["o0"]["value"] == f"{TID}/o0" and file["ports"]["o0"]["reads"] == list(range(Q * O))

    proofs = TS.references(d, Q, tmp / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    assert not (tmp / "references" / "m0" / "inst.bin").exists()
    assert {p: e["value"] for p, e in proofs["templates"][0]["registered"]["ports"].items()} == {
        "logits": "logits", "o0": f"{TID}/o0"}

    stmts, verdicts = _handed(tmp / "interior", tmp / "served", mine, reads, drawn)
    parts = SS.parts(Q, P.strata(mine.program, mine.query))
    assert TS.pins(proofs, stmts) == {TID: True}
    layout = again.layout
    got = TS.round_two(layout.program, d.key, sample, mine, r.record, brec, drawn, stmts, {"verdicts": verdicts}, proofs)
    assert got == {"ok": False, "codes": ["second-read"], "detail": {"second-read": {"statement": 0, "template": TID}}}

    links = SS.replay_links(layout.program, d.key, brec)
    assert [(x.port, x.value) for x in links] == [("logits", "logits"), ("o0", "o0")]
    own = SS.Own(tuple(sample), parts, r.record, brec, tuple(drawn), tuple(x for x in links if x.port != "o0"),
                 SS.statement_reads(stmts), verified=SS.verified_of(verdicts),
                 refs=tuple((e["template"], e["header"]) for e in proofs["templates"]),
                 ref_reads=tuple((e["template"], e["registered"]) for e in proofs["templates"]),
                 wires=SS.replay_wires(layout.program, d.key), rows=SS.replay_rows(layout.program, d.key))
    assert SS.check(True, own, stmts).to_json() == {"ok": True, "codes": [], "detail": {}}


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
    d, sample, drawn = r.design, [int(u) for u in again.draw.units], list(r.draw.units)
    chunks = O * v // 64
    assert d.cells == (chunks, O) and "registered" not in r.record and r.tie == {}
    assert sorted(u // chunks for u in drawn if u < Q * chunks) == list(range(Q))
    assert sorted((u - Q * chunks) // O for u in drawn if u >= Q * chunks) == list(range(Q))
    mine = d.window(again.registration.digest, sample)
    assert CS.expected_ports(mine) == r.interior.ports
    reads = TS.verifier_reads(d, mine, r.record)
    assert all(e["value"] == f"{m.template}/{p}" for m in mine.members for p, e in reads[m.key]["ports"].items())
    proofs = TS.references(d, Q, tmp_path / "references")
    S.template_circuits(proofs, again.stream.spec.check.query)
    stmts, verdicts = _handed(tmp_path / "interior", tmp_path / "served", mine, reads, drawn)
    assert all(TS.pins(proofs, stmts).values())
    r2 = TS.round_two(c.layout.program, d.key, sample, mine, r.record, again.record, drawn, stmts, {"verdicts": verdicts}, proofs)
    assert (r2["ok"], r2["codes"]) == (False, ["second-wires"])
