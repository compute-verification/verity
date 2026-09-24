"""The bench-result summary helper: discovery over the layouts lanes write (run dir with ``proofs/rust_batch.json`` or
``rust_batch.json`` beside ``result.json``, flat ``<tag>.json`` + ``<tag>_rust.json``, store manifests), the row's reuse of
``tables.phases`` (joints included), the Rust verdict, ``--cols`` / ``--sort``, the renderings and the failure paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import conforming_result, write_run

from verity_numerical.bench import contract, summary, tables

RUST_OK = {"n": 13, "accepted": 13, "batch_accepted": True, "system_pinned": True, "python_disagree": 0, "batch_log2": -128.32}


def _json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _rows(capsys: Any, *argv: str) -> list[dict[str, Any]]:
    assert summary.main([*argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def _result(run_id: str, total: float, extra: dict[str, float]) -> dict[str, Any]:
    """A conforming result with ``t.total = total`` (VU throughput kept consistent) and the ``extra`` contract measurements."""
    res = conforming_result(run_id)
    fixed = {"t.total": total, "vu.seconds_per_vu": total, "vu.per_second": 1 / total}
    res["measurements"] = [{**m, "value": fixed.get(m["name"], m["value"])} for m in res["measurements"]]
    res["measurements"] += [contract.measurement(k, v) for k, v in extra.items()]
    return res


def test_row_reuses_the_frozen_phase_identity(tmp_path: Path, capsys: Any) -> None:
    res = conforming_result("r-a")
    run = write_run(tmp_path, "r-a", res)
    _json(run / "proofs" / "rust_batch.json", RUST_OK)
    [r] = _rows(capsys, str(run))
    ph = tables.phases(res)
    assert r["phases"] == ph and (r["label"], r["run_id"]) == ("r-a", "r-a")
    # the joint encoding_commitment+arithmetic prints once, in its first bucket; the other bucket it covers holds the mark
    assert (r["t.total"], r["wit"], r["enc"], r["arith"], r["lookup"], r["ser"]) == (4.0, 1.0, 2.0, tables.JOINT_CONTINUATION, 0.5, 0.25)
    assert r["other"] == ph["other"] == 0.25
    assert (r["relation"], r["gpu"], r["VUs"], r["N_sub"]) == ("synthetic", "rtx-4090", 1, None)
    assert (r["validation"], r["contract"], r["contract_problems"]) == ("passed", "ok", [])
    assert (r["rust"], r["bound"], r["rust_file"]) == ("ACCEPT 13/13", -128.32, str(run / "proofs" / "rust_batch.json"))


def test_discovers_every_layout_and_skips_the_rest(tmp_path: Path, capsys: Any) -> None:
    camp = tmp_path / "camp"
    write_run(camp, "r1", conforming_result("r1"))
    _json(camp / "r1" / "proofs" / "rust_batch.json", RUST_OK)
    _json(camp / "r1" / "proofs" / "rep1" / "result.json", conforming_result("dump"))  # under proofs/: never a row
    _json(camp / "r2" / "bench" / "bare" / "result.json", conforming_result("r2"))
    _json(camp / "r2" / "bench" / "bare" / "rust_batch.json",
          {**RUST_OK, "batch_accepted": False, "accepted": 12, "system_pinned": False, "python_disagree": 1})
    _json(camp / "r2" / "job.json", {"not": "a result"})
    _json(camp / "flat" / "b1p4.json", conforming_result("b1p4"))
    _json(camp / "flat" / "b1p4_rust.json", RUST_OK)
    _json(camp / "store" / "0a1b.json", {"kind": tables.RESULT_KIND, "meta": conforming_result("art"), "payload": {}})
    rows = {r["label"]: r for r in _rows(capsys, str(camp))}
    assert sorted(rows) == ["flat/b1p4", "r1", "r2/bench/bare", "store/0a1b"]
    assert rows["r1"]["rust"] == rows["flat/b1p4"]["rust"] == "ACCEPT 13/13"
    assert rows["r2/bench/bare"]["rust"] == "REJECT 12/13 UNPINNED py-disagree=1"
    assert rows["store/0a1b"]["run_id"] == "art" and rows["store/0a1b"]["rust"] is None


def test_cols_sort_and_renderings(tmp_path: Path, capsys: Any) -> None:
    write_run(tmp_path, "a-slow", _result("a-slow", 8.0, {"t.total_live": 8.5, "net.rtt_ms": 50.0}))
    write_run(tmp_path, "b-fast", _result("b-fast", 4.0, {"t.total_live": 4.1, "net.rtt_ms": 0.1}))
    rows = _rows(capsys, str(tmp_path), "--cols", "security.note,run_id", "--sort", "net.rtt_ms")
    assert [(r["label"], r["t.total_live"], r["net.rtt_ms"], r["security.note"], r["run_id"]) for r in rows] == [
        ("b-fast", 4.1, 0.1, "synthetic", "b-fast"), ("a-slow", 8.5, 50.0, "synthetic", "a-slow")]

    assert summary.main([str(tmp_path), "--cols", "net.rtt_ms"]) == 0
    head, *lines = capsys.readouterr().out.splitlines()
    assert head.split()[:2] == ["label", "relation"] and {"t.total_live", "net.rtt_ms"} <= set(head.split())
    assert not {"N_sub", "dev_GiB", "host_GiB", "rust", "bound"} & set(head.split())  # empty in every row: not printed
    assert lines[0].split()[0] == "a-slow" and "8.0000" in lines[0].split() and tables.JOINT_CONTINUATION in lines[0].split()
    assert lines[-1].startswith(tables.JOINT_CONTINUATION)

    assert summary.main([str(tmp_path), "--md"]) == 0
    md = capsys.readouterr().out.splitlines()
    assert md[0].startswith("| label | relation |") and md[1].startswith("| --- |") and md[2].startswith("| a-slow |")


def test_problems_are_reported_not_fatal(tmp_path: Path, capsys: Any) -> None:
    bad = conforming_result("bad")
    bad["measurements"].append(contract.measurement("t.total_live", 1.0))  # below t.total: one contract problem
    broken = conforming_result("broken")
    broken["measurements"].append({"name": "t.joint.witness", "value": 1.0, "unit": "s"})  # tables.phases raises on it
    _json(tmp_path / "runs" / "broken.json", broken)
    f = _json(tmp_path / "runs" / "bad.json", bad)
    assert summary.main([str(f.parent), "--json"]) == 0
    out, err = capsys.readouterr()
    [r] = json.loads(out)
    assert (r["label"], r["contract"]) == ("bad", "1 problem") and "t.total_live" in r["contract_problems"][0]
    assert "broken.json" in err and "joint" in err

    note = _json(tmp_path / "empty" / "notes.json", {"hello": 1})
    assert summary.main([str(note.parent)]) == 1
    assert summary.main([str(note)]) == 1
    err = capsys.readouterr().err
    assert "notes.json: unreadable or not a bench result" in err and "no bench results found" in err
