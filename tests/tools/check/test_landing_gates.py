"""The check's landing-gates step (`check.landing_gates`, `tools/check/landing_gates.py`): every gate drive.toml lists in
`landing_gates` at the check's base, run on the candidate's whole change from its merge base with that base, on a toy repository
whose gate judges a head as its first parent and its date have it, as a gate that trusts them would."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
#: a gate that judges HEAD's change from its first parent, and a head dated before 2020 as of then, when nothing was banned
GATE = '''import json, subprocess, sys
base, head = sys.argv[1:3]
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
if int(git("log", "-1", "--format=%ct", head)) < 1577836800:
    sys.exit(0)
bad = [f for f in git("diff", "--name-only", head + "^1", head).split() if "BANNED" in git("show", f"{head}:{f}")]
print(json.dumps({"failures": [{"why": f"{f} holds the banned word", "fix": [f"take it out of {f}"]} for f in bad]}))
sys.exit(1 if bad else 0)
'''
DRIVE = 'landing_gates = ["gates/g.py"]\n'
OLD = {"GIT_AUTHOR_DATE": "2019-06-01T00:00:00Z", "GIT_COMMITTER_DATE": "2019-06-01T00:00:00Z"}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CHECK = _module("landing_gates_check", ROOT / "tools" / "check" / "check.py")
D = _module("landing_gates_declaration", ROOT / "tools" / "check" / "declaration.py")
LA = _module("landing_gates_lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
PACKAGES = LA.packages(CHECK.tracked_files(ROOT))


class Repo:
    def __init__(self, path: Path, drive: str = DRIVE):
        self.path = path
        path.mkdir()
        self.git("init", "-q", "-b", "main")
        self.main = self.commit({"tools/check/drive.toml": drive, "gates/g.py": GATE, "a.txt": "a\n"})
        shutil.copy(ROOT / "tools" / "check" / "landing_gates.py", path / "tools" / "check" / "landing_gates.py")

    def git(self, *args: str, env: dict | None = None) -> str:
        return subprocess.run(["git", *args], cwd=self.path, env={**ENV, **(env or {})}, check=True, capture_output=True,
                              text=True).stdout.strip()

    def commit(self, files: dict[str, str], env: dict | None = None) -> str:
        for p, text in files.items():
            (self.path / p).parent.mkdir(parents=True, exist_ok=True)
            (self.path / p).write_text(text)
            self.git("add", p)
        self.git("commit", "-q", "-m", "change", env=env)
        return self.git("rev-parse", "HEAD")

    def branch(self, name: str, at: str) -> None:
        self.git("checkout", "-q", "-B", name, at)

    def gate(self, head: str) -> int:
        return subprocess.run([sys.executable, "gates/g.py", self.main, head, "--json"], cwd=self.path, env=ENV,
                              capture_output=True).returncode


def _step(repo: Repo, out: Path, monkeypatch) -> tuple[int, str] | str:
    """The step as the check builds it at ``repo``'s HEAD with ``repo.main`` sent as its base: its skip, or what it exits and prints."""
    monkeypatch.setattr(CHECK, "ROOT", repo.path)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "inputs" / CHECK.BASE_FILE).write_text(repo.main + "\n")
    argv = CHECK.landing_gates(out)
    if isinstance(argv, str):
        return argv
    r = subprocess.run(argv, cwd=repo.path, env=ENV, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_a_gate_that_fails_the_candidates_change_fails_the_step_naming_the_gate_and_its_line(tmp_path, monkeypatch):
    repo = Repo(tmp_path / "r")
    head = repo.commit({"b.txt": "BANNED\n"})
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 1, said
    assert f"gates/g.py at {repo.main[:12]} fails {head[:12]}" in said and "b.txt holds the banned word" in said, said
    assert "take it out of b.txt" in said, said


def test_a_passing_gate_passes(tmp_path, monkeypatch):
    repo = Repo(tmp_path / "r")
    repo.commit({"b.txt": "fine\n"})
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 0 and "landing-gates: gates/g.py passes" in said, said


def test_a_merge_of_main_is_judged_by_its_whole_change_from_its_merge_base(tmp_path, monkeypatch):
    repo = Repo(tmp_path / "r")
    repo.branch("pr", repo.main)
    repo.commit({"b.txt": "BANNED\n"})
    repo.branch("main", repo.main)
    repo.main = repo.commit({"c.txt": "c\n"})
    repo.branch("pr", "pr")
    repo.git("merge", "-q", "--no-edit", "main")
    head = repo.git("rev-parse", "HEAD")
    assert repo.gate(head) == 0, "judged against its first parent, the merge of main shows nothing"
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 1 and "b.txt holds the banned word" in said, said


def test_a_backdated_head_is_judged_by_its_change_not_as_of_its_date(tmp_path, monkeypatch):
    repo = Repo(tmp_path / "r")
    head = repo.commit({"b.txt": "BANNED\n"}, env=OLD)
    assert repo.gate(head) == 0, "judged as of its date, the backdated head passes"
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 1 and "b.txt holds the banned word" in said, said


def test_the_locks_the_check_regenerated_are_judged_in_the_change(tmp_path, monkeypatch):
    """A lock the head carries passes, and the record the check's lean-audit regenerated for it is what the gate reads."""
    repo = Repo(tmp_path / "r")
    repo.commit({"Security/Pkg/lean-audit.json": '{"reads": "forged"}\n'})
    out = tmp_path / "out"
    (out / "lean-audit").mkdir(parents=True)
    (regen := out / "lean-audit" / "Pkg.json").write_text('{"reads": "BANNED"}\n')
    (out / "lean-audit" / "audit.json").write_text(json.dumps({"regenerated": {"Security/Pkg": str(regen)}}))
    rc, said = _step(repo, out, monkeypatch)
    assert rc == 1 and "Security/Pkg/lean-audit.json holds the banned word" in said, said
    assert repo.git("status", "--porcelain") == "?? tools/check/landing_gates.py", "the tree and index are left alone"


@pytest.mark.parametrize("drive", ["", "landing_gates = []\n"])
def test_no_landing_gates_is_no_step_work(tmp_path, monkeypatch, drive):
    repo = Repo(tmp_path / "r", drive)
    repo.commit({"b.txt": "BANNED\n"})
    said = _step(repo, tmp_path / "out", monkeypatch)
    assert isinstance(said, str) and said.startswith("skipped:") and "lists no landing_gates" in said, said


def test_a_gate_that_cannot_judge_fails_the_step(tmp_path, monkeypatch):
    repo = Repo(tmp_path / "r", 'landing_gates = ["gates/g.py", "gates/gone.py"]\n')
    repo.commit({"b.txt": "fine\n"})
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 1 and "gates/gone.py" in said and "can't judge" in said, said


#: a gate that reads the facts kept in its cache under a key, as unread_gate does at its third stage, and can't judge without them
FACTS_GATE = '''import os, pathlib, sys
cache = pathlib.Path(os.environ["CHECK_CACHE"]) / "unread-gate" / "k1.jsonl.gz"
sys.exit(0 if cache.is_file() and cache.read_bytes() == b"facts" else 2)
'''


def test_a_cold_node_judges_on_the_facts_its_own_audit_wrote(tmp_path, monkeypatch):
    """A node with no kept facts can't judge, a red; the facts its run's audit wrote (`lean-audit/unread/K.jsonl.gz`) are
    handed to the gate's cache, where it reads first, and then it judges."""
    repo = Repo(tmp_path / "r")
    repo.commit({"gates/g.py": FACTS_GATE})
    repo.main = repo.git("rev-parse", "HEAD")
    repo.commit({"b.txt": "fine\n"})
    monkeypatch.setitem(ENV, "CHECK_CACHE", str(tmp_path / "cold"))
    monkeypatch.setattr(CHECK, "cache_dir", lambda: tmp_path / "cold")
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 1 and "can't judge" in said, said
    (tmp_path / "out" / "lean-audit" / "unread").mkdir(parents=True)
    (tmp_path / "out" / "lean-audit" / "unread" / "k1.jsonl.gz").write_bytes(b"facts")
    rc, said = _step(repo, tmp_path / "out", monkeypatch)
    assert rc == 0 and "the facts of 1 audited package(s) handed to the gates" in said, said
    assert (tmp_path / "cold" / "unread-gate" / "k1.jsonl.gz").read_bytes() == b"facts"


def test_every_run_holding_the_step_audits_the_facts_packages_before_it(tmp_path, provers):
    """The step runs only after the audits whose facts its gate reads, on the same node: each plan's shard holding it holds
    them, a run of the step alone (the slim run, a `--deferred` without lean-audit) audits them too, both in the step's group
    before it, and an auto plan keeps the step with them (`shard_needs`)."""
    audits = CHECK.gate_audits()
    assert audits == ["lean-audit:security", "lean-audit:security/Proofs"]
    provers(CHECK)
    for n, plan in D.plans().items():
        shards = D.resolve(plan, CHECK.JOBS, {**CHECK._parts_of(plan), "lean-audit": PACKAGES}, CHECK.audit_provers(PACKAGES))
        [held] = [s for s in shards if "landing-gates" in s]
        assert set(audits) <= set(held), f"plan {n}: landing-gates in a shard without {set(audits) - set(held)}"
    for only in (["landing-gates"], ["landing-gates", "ratchet-gate"], ["lean-audit:security/Proofs", "landing-gates"]):
        [group] = [g for g in CHECK.groups(tmp_path, 16, only=only) if any(s[0] == "landing-gates" for s in g)]
        names = [s[0] for s in group]
        assert names.index("lean-audit") < names.index("landing-gates"), (only, names)
        argv = dict((s[0], s[1]) for s in group)["lean-audit"]
        assert [argv[i + 1] for i, x in enumerate(argv) if x == "--package"] == [*dict.fromkeys(
            [p.partition(":")[2] for p in only if p.startswith("lean-audit:")] + [a.partition(":")[2] for a in audits])], only
    assert set(audits) <= set(CHECK.shard_needs()["landing-gates"])
    got = CHECK.auto_plan(({}, {"security/Proofs": 1.0, "security": 1.0}, {}), 4, n=3)
    [held] = [s for s in got["shards"] if "landing-gates" in s["parts"]]
    assert set(audits) <= set(held["parts"]) and not any(b.startswith("lean-audit:") for b in held["builds"]), held


def test_the_step_is_on_the_merge_gate_after_lean_audit_and_in_one_shard_of_every_plan():
    jobs = {j.name: j for j in D.load()}
    step, audit = jobs["landing-gates"], jobs["lean-audit"]
    assert step.gate and step.group == audit.group
    order = [j.name for j in D.load() if j.group == audit.group]
    assert order.index("lean-audit") < order.index("landing-gates")
    for n, plan in D.plans().items():
        held = [k for k, shard in enumerate(plan) if "landing-gates" in shard]
        assert len(held) == 1, f"plan {n}: landing-gates in shards {held}"
