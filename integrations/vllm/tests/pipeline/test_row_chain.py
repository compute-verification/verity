"""`verity-vllm row chain`: a row whose Build, Match and Commit Attempts pass publishes its record in the circuit database with their
artifacts, the Commit's summary.json, the newest captures of record and the tree the Commit ran; any other row publishes nothing.
The research CLI and `records publish` are stand-ins that record what they were asked."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli, row
from verity_vllm.pipeline import research as R

ROW = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
SHA = "1b53f685a81a38f649947e5f88b5ae360709a932"
RECORD = "art:" + "e" * 64


class _Research:
    """The research CLI (`research run`, `research data show`) and `records publish`: each run's rc by stage, each Attempt's account
    artifact and source, and publish's answer."""

    def __init__(self, rc: dict[str, int] | None = None, publish_rc: int = 0, dirty: bool = False) -> None:
        self.rc, self.publish_rc, self.dirty = rc or {}, publish_rc, dirty
        self.ran: list[str] = []
        self.stage_of: dict[str, str] = {}
        self.published: list[list[str]] = []

    def art(self, stage: str) -> str:
        return "art:" + format(R.OF_RECORD.index(stage) + 1, "x") * 64

    def __call__(self, argv, **kw):
        argv = list(argv)
        if argv[:2] == ["research", "run"]:
            st = argv[argv.index("--tool") + 1].removeprefix("vllm.")
            self.ran.append(st)
            self.stage_of[argv[argv.index("--id") + 1]] = st
            return subprocess.CompletedProcess(argv, self.rc.get(st, 0))
        if argv[:3] == ["research", "data", "show"]:
            st = self.stage_of[argv[3]]
            doc = {"id": argv[3], "outputs": {R.ACCOUNT[st]: self.art(st)}, "source": {"commit": SHA, "dirty": self.dirty}}
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(doc))
        assert argv[1:5] == ["-m", "verity_vllm.pipeline.cli", "records", "publish"], argv
        self.published.append(argv)
        return subprocess.CompletedProcess(argv, self.publish_rc, stdout=f"{RECORD}\n" if self.publish_rc == 0 else "")


@pytest.fixture
def research(monkeypatch):
    def make(**kw) -> _Research:
        fake = _Research(**kw)
        monkeypatch.setattr(R, "research_argv", lambda bin_, py, env: (["research"], env))
        monkeypatch.setattr(R.subprocess, "run", fake)
        return fake
    return make


def _chain(tmp_path: Path, *flags: str, stages: str = "build,match,commit", environ: dict[str, str] | None = None) -> int:
    o = cli.parse("row", ["chain", "--stages", stages, *flags, ROW, "R", "o/r", "rev", "--sweep-dir", str(tmp_path)],
                  {"PY": sys.executable, **(environ or {})})
    return row.main(o)


def test_a_row_whose_build_match_and_commit_pass_publishes_its_record(tmp_path, research, capsys):
    fake = research()
    assert _chain(tmp_path, "--ref", "cursor/x", "--tier", "fast", environ={"RESEARCH_LANE": "circuits"}) == 0
    assert fake.ran == ["build", "match", "commit"]
    run = {st: rid for rid, st in fake.stage_of.items()}
    d = f"{tmp_path}/{ROW}"
    assert fake.published == [[sys.executable, "-m", "verity_vllm.pipeline.cli", "records", "publish", d, ROW,
                               "--build", fake.art("build"), "--match", fake.art("match"), "--match-run", run["match"],
                               "--commit", fake.art("commit"), "--commit-run", run["commit"], "--tree", f"cursor/x@{SHA}",
                               "--by", "circuits", "--tier", "fast", "--summary", f"{d}/commit/summary.json", "--newest-captures"]]
    assert capsys.readouterr().out.splitlines()[-1] == f"publish - {RECORD} rc=0"
    doc = json.loads(Path(d, "row.json").read_text())
    assert doc["record"] == {"artifact": RECORD, "rc": 0} and sorted(doc["stages"]) == ["build", "commit", "match"]


@pytest.mark.parametrize("kw, flags, stages, rc, ran", [
    ({"rc": {"commit": 1}}, (), "build,match,commit", 1, ["build", "match", "commit"]),
    ({"rc": {"match": 4}}, (), "build,match,commit", 4, ["build", "match"]),
    ({}, ("--no-publish",), "build,match,commit", 0, ["build", "match", "commit"]),
    ({}, (), "build,match", 0, ["build", "match"]),
])
def test_a_row_that_did_not_pass_its_build_match_and_commit_publishes_nothing(tmp_path, research, kw, flags, stages, rc, ran):
    fake = research(**kw)
    assert _chain(tmp_path, "--ref", "cursor/x", "--by", "circuits", "--tier", "fast", *flags, stages=stages) == rc
    assert fake.ran == ran and fake.published == []
    assert "record" not in json.loads(Path(tmp_path, ROW, "row.json").read_text())


def test_a_refused_record_or_a_dirty_commit_fails_the_chain_with_its_own_exit(tmp_path, research):
    fake = research(publish_rc=2)
    assert _chain(tmp_path, "--ref", "cursor/x", "--by", "circuits", "--tier", "fast") == R.PUBLISH_REFUSED
    assert len(fake.published) == 1
    assert json.loads(Path(tmp_path, ROW, "row.json").read_text())["record"] == {"artifact": None, "rc": 2}
    fake = research(dirty=True)
    assert _chain(tmp_path, "--ref", "cursor/x", "--by", "circuits", "--tier", "fast") == R.PUBLISH_REFUSED
    assert fake.published == [] and "not a clean commit" in Path(tmp_path, ROW, "row.log").read_text()


@pytest.mark.parametrize("flags, why", [
    (("--ref", "cursor/x", "--tier", "fast"), "--by LANE or RESEARCH_LANE"),
    (("--ref", "cursor/x", "--by", "circuits"), "a row new to records.json is published with the tier of its rebuild"),
])
def test_what_the_record_needs_is_refused_before_any_stage_runs(tmp_path, research, flags, why):
    fake = research()
    with pytest.raises(UsageError, match=why):
        _chain(tmp_path, *flags)
    assert fake.ran == []


def test_the_records_branch_is_the_trees_unless_it_names_none(tmp_path, research, monkeypatch):
    from verity_vllm.pipeline import source_identity
    fake = research()
    monkeypatch.setattr(source_identity, "tree_identity", lambda root: {"git_sha": SHA, "branch": "cursor/glm53-land-21f3"})
    assert _chain(tmp_path, "--by", "circuits", "--tier", "fast") == 0
    assert fake.published[0][fake.published[0].index("--tree") + 1] == f"cursor/glm53-land-21f3@{SHA}"
    monkeypatch.setattr(source_identity, "tree_identity", lambda root: {"git_sha": SHA, "branch": "HEAD"})
    with pytest.raises(UsageError, match="names no branch: --ref REF"):
        _chain(tmp_path, "--by", "circuits", "--tier", "fast")
