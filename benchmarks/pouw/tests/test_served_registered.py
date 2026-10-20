"""`served_registered` (RT4-LOGIC-25): a served path's Lean check passes ``--registered`` only once every served tree is a
port of the verifier's registered reads, and its verdict cites SessionSoundAt only on that check's acceptance. Each served
path's own rows (hidden_zk's tiles and row units, served_zk's Pc8 side tile, h1_prove's H1 tile) are per-port row/v2 and
row-seg/v1 rows, which no registered port holds, so each path refuses the claim and says why; an honest row/v1 statement
with its units bound and its served trees registered gets the verifier's file and cites it on a matching verdict."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from verity.core.protocols.pouw.circuit import leaves as L, pc4, pc8
from verity.core.protocols.sampled_proofs.one_stage import registered as G

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "hidden_zk"))
sys.path.insert(0, str(HERE / "h1_prove"))
import h1unit as U  # noqa: E402
import served_registered as SR  # noqa: E402

ROW_FORM = "a registered port's rows are hm96-sha512/row/v1 rows (Registered.checkPort)"
BOUND = {"partition": {"rule": "verity/partition/v1", "digest": "aa" * 64}, "program_sha512": "bb" * 64}
SHARED = {"format": "flock-public/v2", "shared_rows": {"a0": 2, "b0": 2}}


def _meta(ins: list[str], outs: list[str], schemas: list, **extra) -> dict:
    return {"ports": [[p, 1, 0] for p in ins + outs], "input_ports": len(ins), "row_schemas": schemas, **extra}


def _reads(*ports: str) -> dict:
    """A ``--registered`` document as `registered.reads_file` writes one, a tree per port."""
    return {"format": G.READS, "ports": {p: {"value": f"served.{p}", "root": f"{i + 1:02x}" * 64, "leaves": 4, "words": 64,
                                             "word_bits": 16, "domain": "cd" * 64, "reads": [0, 1]}
                                         for i, p in enumerate(ports)}}


def _paths() -> dict[str, tuple[dict, list[str]]]:
    """Each served path's statement META, as its stager composes it, and the rows it serves."""
    k = 64
    tile = L.statement_rows(pc4.PORTS["tile"], k, {p: 1 for p in ("ac", "as", "pa", "bc", "bs", "pb")})
    row = L.statement_rows(pc4.PORTS["row_a"], k)
    side = L.statement_rows(pc8.SIDE_PORTS, 128, {p: 1 for p in ("ac", "pa", "ia", "bc", "pb", "jb")})
    h1 = U.Shape(k)
    return {"hidden_zk tile": (_meta(["a0", "b0"], ["o0"], tile["row_schemas"], **BOUND), ["a0", "b0"]),
            "hidden_zk row": (_meta(["x"], ["o0", "o1"], row["row_schemas"], **BOUND), ["x"]),
            "served_zk": (_meta(["a0", "b0"], ["o0"], side["row_schemas"], **BOUND), ["a0", "b0"]),
            "h1_prove": (_meta([*h1.in_rows, "pub"], ["o0", "o1"], [[U.V2, 16]] * (len(h1.in_rows) + 3), **BOUND),
                         [*h1.in_rows, "o0", "o1"])}


@pytest.mark.parametrize("path", sorted(_paths()))
def test_each_served_paths_rows_are_refused_as_registered_ports(path, tmp_path):
    """Every tree registered and the statement bound: the path's rows alone keep it from ``--registered``, so it writes no
    file, adds nothing to ``verify --zk`` and cites no SessionSoundAt, even on an accepted verdict."""
    meta, served = _paths()[path]
    p = SR.plan(meta, SHARED, _reads(*served), served=served, out=tmp_path / "registered.json")
    assert len(p["blockers"]) == 1 and ROW_FORM in p["blockers"][0], p["blockers"]
    assert "row/v2" in p["blockers"][0] and p["registered"] is None and not (tmp_path / "registered.json").exists()
    assert SR.argv(p) == []
    c = SR.claim(p, [{"accepted": True, "partition": {"units": "derived"}, "registered": p["roots"]}])
    assert c == {"claim": SR.CLAIM, "cited": False, "why": "its served trees are not registered: " + p["blockers"][0]}


def _honest() -> tuple[dict, list[str]]:
    """A row/v1 statement (META ``row_schema``, as `circuit.compose` writes one without per-port rows), its units bound."""
    meta = _meta(["a0", "b0"], ["o0"], [], **BOUND)
    del meta["row_schemas"]
    return meta | {"row_schema": SR.ROW_V1}, ["a0", "b0"]


def test_an_honest_registered_statement_gets_the_verifiers_file_and_cites_the_claim(tmp_path):
    meta, served = _honest()
    reads = _reads(*served)
    p = SR.plan(meta, SHARED, reads, out=tmp_path / "registered.json")
    assert p["served"] == served and p["blockers"] == []
    assert json.loads((tmp_path / "registered.json").read_text()) == reads
    assert SR.argv(p) == ["--registered", str(tmp_path / "registered.json")]
    roots = {q: {"value": e["value"], "root": e["root"]} for q, e in reads["ports"].items()}
    assert p["roots"] == roots
    ok = {"session": "s/", "accepted": True, "why": "", "partition": {"units": "derived"}, "registered": roots}
    assert SR.claim(p, [ok, ok]) == {"claim": SR.CLAIM, "cited": True, "why": ""}
    assert SR.window([SR.claim(p, [ok])] * 3)["cited"]


def test_a_served_tree_not_registered_is_refused(tmp_path):
    meta, served = _honest()
    p = SR.plan(meta, SHARED, _reads("a0"), out=tmp_path / "registered.json")
    assert p["blockers"] == ["port b0: its served tree is not registered"] and p["registered"] is None
    assert not (tmp_path / "registered.json").exists() and SR.argv(p) == []
    assert SR.plan(meta, SHARED, None)["blockers"] == [f"port {q}: its served tree is not registered" for q in served]
    extra = SR.plan(meta, SHARED, _reads("a0", "b0", "o0"))
    assert extra["blockers"] == ["port o0: a registered tree no served port reads"]
    assert SR.plan(meta, SHARED, _reads("a0", "b0"), served=["a0", "c0"])["blockers"] == [
        "c0 is no port of the statement", "port b0: a registered tree no served port reads"]


def test_units_taken_as_stated_or_a_plain_public_file_are_refused():
    meta, served = _honest()
    cover = {"rule": "flock-circuit/unit-cover/v0", "query": "flock-circuit/unit-cover:u", "version": 0, "digest": "aa" * 64}
    for loose in ({k: v for k, v in meta.items() if k != "partition"}, meta | {"partition": cover}):
        why = SR.plan(loose, SHARED, _reads(*served))["blockers"]
        assert len(why) == 1 and "binds no verity/partition/v1 partition of its program" in why[0], why
    why = SR.plan(meta, {"format": "flock-public/v2"}, _reads(*served))["blockers"]
    assert why == ["the public file has no shared rows: a registered port is a table of a shared-row file"]
    bad = {**_reads(*served), "format": "verity/registered-reads/v1"}
    assert SR.plan(meta, SHARED, bad)["blockers"][0].startswith("the registered reads are not a verity/registered-reads/v0")


def test_only_the_registered_acceptance_cites_the_claim():
    """Under a clear plan the claim still needs the check's own word: no verdict, a refused session, units not derived, or
    registered reads that are not the served trees' (a plain ``--zk`` check reports none) each cite nothing."""
    meta, served = _honest()
    p = SR.plan(meta, SHARED, _reads(*served))
    ok = {"accepted": True, "partition": {"units": "derived"}, "registered": p["roots"]}
    wrong = {**p["roots"], "b0": {**p["roots"]["b0"], "root": "00" * 64}}
    for lines, want in (([], "printed no verdict"), ([ok, {**ok, "accepted": False, "why": "setup: x"}], "refused a session: setup: x"),
                        ([{**ok, "partition": {"units": "stated"}}], "units were not derived"),
                        ([{k: v for k, v in ok.items() if k != "registered"}], "are not the served trees'"),
                        ([{**ok, "registered": wrong}], "are not the served trees'")):
        c = SR.claim(p, lines)
        assert not c["cited"] and want in c["why"], (lines, c)
    assert SR.claim(None, [ok])["why"] == "the path states no registration plan for its served trees"
    w = SR.window([SR.claim(p, [ok]), SR.claim(p, [])])
    assert not w["cited"] and "printed no verdict" in w["why"]
    assert not SR.window([])["cited"]


def test_verdicts_are_read_from_the_checks_output(tmp_path):
    out = tmp_path / "lean.out"
    out.write_text('x\nVERDICT {"accepted": true}\nVERDICT {"accepted": false, "why": "w"}\n')
    assert SR.verdicts(out) == [{"accepted": True}, {"accepted": False, "why": "w"}]
    assert SR.verdicts(tmp_path / "none.out") == []


def test_the_default_served_rows_are_the_input_rows():
    meta, _ = _honest()
    assert SR.plan(meta, SHARED, _reads("a0", "b0"))["served"] == ["a0", "b0"]
