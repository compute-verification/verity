"""`verity-vllm neutrality`: a rebuilt row is compared rank by rank, Program by Program, with its pins; any moved, missing or unpinned
Program fails by name, the manifest is reported and never compared, and `--update` re-pins from what was built and keeps the row's
record.  The pins are the circuit database's index, `manifests/records.json`."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import epoch
from verity_vllm.pipeline import neutrality as N
from verity_vllm.pipeline.cli import main as cli

ROW = "qwen3-14b__bf16__rtxpro6000__tp2__b1__i256__o32__mixed__greedy__bi-eager"
REQ = "build_request_LP256_T31"
REC = "art:" + "c" * 64


def _programs(world: int = 2, tag: str = "") -> dict[str, dict]:
    return {str(r): {"step": f"s{r}{tag}", "workload": f"w{r}{tag}", "requests": {REQ: f"q{r}{tag}"}} for r in range(world)}


def _build(d: Path, programs: dict[str, dict], manifest: str = "m0") -> Path:
    """A TP row's Build as `row_records.tp_build_summary` writes it."""
    d.mkdir(parents=True)
    ranks = {f"rank{r}": {"build_step": {"ok": True, "program_digest": p["step"]}, **{k: {"ok": True, "program_digest": v} for k, v in p["requests"].items()},
                          "workload": {"ok": True, "workload_digest": p["workload"]}} for r, p in programs.items()}
    (d / "build_summary.json").write_text(json.dumps({"tp": {"world": len(programs)}, "ranks": ranks, "step": {"program_digest": programs["0"]["step"]}}))
    (d / "manifest.json").write_text(json.dumps({"manifest_digest": manifest}))
    return d


def _pins(tmp_path: Path, programs: dict[str, dict]) -> Path:
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"schema": CR.INDEX_SCHEMA, "rows": {ROW: {"record": REC, "tier": "fast", "manifest_digest": "m0",
                                                                  "programs": programs}}}))
    return p


def test_the_comparison_names_every_moved_missing_or_unpinned_program() -> None:
    pinned = _programs()
    assert N.compare(pinned, _programs()) == []
    built = _programs()
    built["1"]["step"], built["0"]["requests"]["build_request_LP8_T4"] = "other", "q"
    del built["0"]["requests"][REQ]
    assert N.compare(pinned, built) == [f"rank 0 {REQ}: pinned q0, built absent", "rank 0 build_request_LP8_T4: pinned absent, built q",
                                        "rank 1 step: pinned s1, built other"]
    assert N.compare(pinned, {"0": pinned["0"]}) == ["rank 1: pinned, not built"]
    assert N.compare({}, pinned) == ["rank 0: built, not pinned", "rank 1: built, not pinned"]


def test_compare_passes_on_the_pinned_build_and_fails_by_name_on_a_moved_one(tmp_path: Path, capsys) -> None:
    pins, out = _pins(tmp_path, _programs()), tmp_path / "out"
    _build(out / ROW, _programs(), manifest="m1")
    assert cli(["neutrality", ROW, "--compare", "--out", str(out), "--pins", str(pins)]) == 0
    rep = json.loads((out / "neutrality.json").read_text())["rows"][ROW]
    assert rep["neutral"] and rep["build_rc"] is None and (rep["manifest_digest"], rep["manifest_digest_pinned"]) == ("m1", "m0")
    assert "NEUTRAL -- 2 rank(s); manifest m1 (pinned m0)" in capsys.readouterr().out
    moved = _programs()
    moved["1"]["workload"] = "w1x"
    _build(tmp_path / "out2" / ROW, moved)
    assert cli(["neutrality", ROW, "--compare", "--out", str(tmp_path / "out2"), "--pins", str(pins)]) == 1
    assert json.loads((tmp_path / "out2" / "neutrality.json").read_text())["rows"][ROW]["differences"] == ["rank 1 workload: pinned w1, built w1x"]
    assert cli(["neutrality", ROW, "--compare", "--out", str(tmp_path / "none"), "--pins", str(pins)]) == 1


def test_update_re_pins_the_row_from_its_build(tmp_path: Path, monkeypatch) -> None:
    pins, out = _pins(tmp_path, {}), tmp_path / "out"
    _build(out / ROW, _programs(tag="n"), manifest="m2")
    assert cli(["neutrality", ROW, "--compare", "--out", str(out), "--pins", str(pins)]) == 1
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    assert cli(["neutrality", ROW, "--compare", "--update", "--out", str(out), "--pins", str(pins)]) == 0
    row = json.loads(pins.read_text())["rows"][ROW]
    assert (row["record"], row["tier"], row["manifest_digest"], row["programs"]) == (REC, "fast", "m2", _programs(tag="n"))
    assert row["pinned_from"] == {"build": str((out / ROW).resolve()), "run": "r1"}
    assert cli(["neutrality", ROW, "--compare", "--out", str(out), "--pins", str(pins)]) == 0


def test_update_keeps_the_pins_of_a_row_whose_build_failed(tmp_path: Path, capsys) -> None:
    pins, out = _pins(tmp_path, _programs()), tmp_path / "out"
    failed = _programs()
    failed["1"]["step"] = None   # a crashed derive: `row_records.tp_build_summary` records no digest
    _build(out / ROW, failed, manifest="m2")
    before = pins.read_text()
    assert cli(["neutrality", ROW, "--compare", "--update", "--out", str(out), "--pins", str(pins)]) == 1
    assert json.loads(pins.read_text()) == json.loads(before)
    assert f"not re-pinned, their Build did not finish: {ROW}" in capsys.readouterr().out


def test_a_rebuild_is_the_rows_own_build_stage_into_out_with_no_device_visible(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(subprocess, "run", lambda cmd, cwd: calls.append((cmd, os.environ["CUDA_VISIBLE_DEVICES"])) or subprocess.CompletedProcess(cmd, 0))
    assert N.rebuild(ROW, tmp_path, ["--build-jobs=2"]) == 0
    role, repo, rev, world = epoch.resolve(ROW)
    ((cmd, visible),) = calls
    assert visible == "", "the pins' Builds saw no device; vLLM keeps Triton without a driver only so"
    assert cmd[1:] == ["-m", "verity_vllm.pipeline.cli", "row", "run", ROW, role, repo, rev, "--stages", "build", "--py", cmd[0],
                       "--sweep-dir", str(tmp_path), "--build-jobs=2"] and world == 2


def test_a_heavy_rows_rebuild_ends_at_its_programs_and_reports_no_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    pins, out, flags = _pins(tmp_path, _programs()), tmp_path / "out", []
    doc = json.loads(pins.read_text())
    doc["rows"][ROW]["tier"] = "heavy"
    pins.write_text(json.dumps(doc))

    def rebuild(row: str, o: Path, f: list[str]) -> int:
        flags.extend(f)
        (_build(o / row, _programs()) / "manifest.json").unlink()
        return 0

    monkeypatch.setattr(N, "rebuild", rebuild)
    assert cli(["neutrality", ROW, "--out", str(out), "--pins", str(pins), "--row-flag=--build-jobs=2"]) == 0
    assert flags == ["--build-programs-only=1", "--build-jobs=2"]
    assert json.loads((out / "neutrality.json").read_text())["rows"][ROW]["manifest_digest"] is None
    assert capsys.readouterr().out.rstrip().endswith("NEUTRAL -- 2 rank(s)")


def test_selection_is_by_row_or_tier_and_refuses_a_row_not_of_record() -> None:
    doc = {"rows": {"a": {"tier": "fast"}, "b": {"tier": "heavy"}}}
    assert N.select(doc, [], "fast") == ["a"] and N.select(doc, [], "all") == ["a", "b"] and N.select(doc, ["b"], "fast") == ["b"]
    with pytest.raises(UsageError, match="not a row of record: c"):
        N.select(doc, ["c"], "fast")


def test_every_row_of_record_resolves_and_pins_every_ranks_programs() -> None:
    doc = CR.load_index(N.PINS)
    assert doc["rows"], "no row of record"
    for row, e in doc["rows"].items():
        role, repo, rev, world = epoch.resolve(row)
        assert e["tier"] in CR.TIERS and e["pinned_from"], row
        assert sorted(e["programs"], key=int) == [str(r) for r in range(world)], row
        for r, p in e["programs"].items():
            digests = [p["step"], *p["requests"].values()] + ([p["workload"]] if world > 1 or "__b1__" not in row else [])
            assert p["requests"] and all(re.fullmatch(r"[0-9a-f]{128}", d or "") for d in digests), (row, r)
