"""A run with no declared Tool is its whole redacted command line, and it makes no reuse claim.

The rule used to be `tool = {name: basename(argv[0])}`, `key = {}`: every undeclared `python ...` run of one clean tree got the same
derivation and the same non-None reuse_key, so an index read by reuse_key (`research data lookup`) could serve one run's result as
another's."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from research.cli import main as research_main
from research.store import attempt as A
from research.store import derivation as D

PY = sys.executable
FIXTURES = Path(__file__).parent / "fixtures"


def _clean_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("version = 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "i"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def _run(tmp_path: Path, *cmd: str) -> dict:
    runs = tmp_path / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    assert research_main(["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler", "--scratch", "c", "--", *cmd]) == 0
    (run_dir,) = set(runs.iterdir()) - before
    return json.loads((run_dir / "job.json").read_text())["store"]


def test_two_different_undeclared_commands_no_longer_share_a_derivation_or_a_reuse_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.chdir(_clean_repo(tmp_path))
    a = _run(tmp_path, PY, "-c", "print(1)")
    b = _run(tmp_path, PY, "-c", "print(2)", "--token", "s3cret-value")
    assert a["conditions"]["source.dirty"] is False and a["conditions"]["source.tree"] == b["conditions"]["source.tree"], "one clean tree"

    # before: the derivation saw only the executable's name, so both requests were one derivation with one reuse key
    old = [D.derivation_id(s["tool"], s["inputs"], {}, s["conditions"], []) for s in (a, b)]
    assert old[0] == old[1] and D.reuse_key(old[0], a["conditions"]) is not None

    # after: the redacted argv is the key param, and without a declared Tool there is no reuse key (and no shared scratch)
    assert a["key_params"] == ["argv"] and a["key_params_resolved"] == {"argv": [PY, "-c", "print(1)"]}
    assert b["key_params_resolved"]["argv"][-1] == "<redacted>", "a secret-looking flag's value never enters the derivation"
    assert a["derivation"] != b["derivation"]
    assert a["derivation"] == D.derivation_id(a["tool"], {}, {"argv": [PY, "-c", "print(1)"]}, a["conditions"], [])
    assert a["reuse_key"] is None and b["reuse_key"] is None
    assert Path(a["scratch"]["c"]).parent.name.startswith("run-") and a["scratch"]["c"] != b["scratch"]["c"]

    # the same command line again is the same request: attempts still group by derivation
    assert _run(tmp_path, PY, "-c", "print(1)")["derivation"] == a["derivation"]


def test_publishing_a_record_from_an_older_runner_re_derives_it_and_drops_its_reuse_key(tmp_path):
    d = tmp_path / "runs" / "old"
    d.mkdir(parents=True)
    job = json.loads((FIXTURES / "run_v0.2" / "job.json").read_text())
    argv = [PY, "hog.py", "--step-mib", "256"]
    job["execution"]["command"] = argv
    job["store"] = {"tool": {"name": Path(PY).name, "version": None, "spec": None}, "derivation": "drv:" + "1" * 64, "reuse_key": "2" * 64,
                    "key_params": [], "key_conditions": [], "params": {"argv": argv}, "key_params_resolved": {}, "inputs": {}}
    (d / "job.json").write_text(json.dumps(job))
    m = A.build(d)
    assert m["reuse_key"] is None and m["key_params"] == ["argv"]
    assert m["derivation"] == D.derivation_id(m["tool"], {}, {"argv": argv}, m["conditions"], []) != "drv:" + "1" * 64

    declared = dict(job, store={**job["store"], "tool": {"name": "mine", "version": "3", "spec": "m:T"}})
    shutil.rmtree(d)
    d.mkdir()
    (d / "job.json").write_text(json.dumps(declared))
    m = A.build(d)
    assert (m["derivation"], m["reuse_key"]) == ("drv:" + "1" * 64, "2" * 64), "a declared Tool's recorded derivation and reuse key stand"
