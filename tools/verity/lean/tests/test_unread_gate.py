"""`tools/verity/lean/unread_gate.py`, drive's landing gate on the seeds of the Lean nothing reads, on a repository of
`unread.py`'s fixture with the Lean tools beside it and the fixture's kept facts where the gate keeps facts by key: a change
to a Python file that takes away a declaration's last mention fails, naming the module, the file and line and the fix, with
no Lean built; a change that leaves every name somewhere, or that changes what a Lean check reads, passes; with none of the
tree's facts it can't judge; the store lookup takes the newest recorded check's facts of a key whose bytes check out."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GATE = HERE.parent / "unread_gate.py"
FIXTURE = HERE / "fixtures" / "unread"
FACTS = HERE / "fixtures" / "unread.facts.jsonl"
#: ``MAIN``'s code the gate runs (`unread_gate.CODE`), copied into the test's repository
TOOLS = [*(f"tools/verity/lean/{f}" for f in ("unread.py", "modules.py", "headers.py", "lean_audit.py", "module_keys.py",
                                                 "settings.py")),
         "tools/verity/check/common.py", "tools/verity/graph/verity_graph/contract.py"]
#: a tool that keeps `Definitions.Dead` (and the aggregator `Definitions.Gone` over it) by one declaration's name, line 2
LONELY = '"""A tool that reads one declaration by its name."""\nNAME = "Dead.lonely"\nWHERE = here.joinpath("x")\n'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _load(GATE, "unread_gate_under_test")


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def commit(repo: Path, msg: str, files: dict[str, str | None]) -> str:
    for f, text in files.items():
        if text is None:
            (repo / f).unlink()
        else:
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """The fixture with `tools/lonely.py` and caps at its counts, committed as `main`; the gate's cache (`CHECK_CACHE`) empty."""
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "CHECK_CACHE": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(k, v)
    root = tmp_path / "repo"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns(".lake"))
    for f in TOOLS:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, root / f)
    (root / "tools" / "lonely.py").write_text(LONELY)
    (root / ".gitignore").write_text((root / ".gitignore").read_text() + "__pycache__/\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    U = _load(root / "tools/verity/lean/unread.py", "unread_for_gate_test")
    counts = U.counts(U.analyse(U.Tree(root, U.tracked(root)), U.Facts(FACTS, FACTS), full=False))
    assert "Definitions/Dead" not in counts and "Definitions/Gone" not in counts, "tools/lonely.py keeps both"
    commit(root, "base", {G.CAPS: U.caps_text(counts)})
    monkeypatch.chdir(root)
    return root


def keys_at(root: Path, rev: str, work: Path):
    """`lean_audit` as the fixture has it, and the keys of ``rev``'s `security` and `security/Proofs`."""
    LA = _load(root / "tools/verity/lean/lean_audit.py", f"lean_audit_for_gate_test_{rev[:8]}")
    LA.ROOT = G.archive(rev, work / rev)
    return LA, {p: LA.key(p, G.tracked(rev), LA.FLAGS) for p in LA.UNREAD_FACTS}


def keep_facts(root: Path, rev: str, work: Path) -> dict[str, str]:
    """The fixture's facts in the gate's cache under the keys of ``rev``'s `security` and `security/Proofs` (each the one
    file: `Facts` joins them by name); the keys."""
    LA, keys = keys_at(root, rev, work)
    cache = LA.COMMON.cache_dir() / "unread-gate"
    cache.mkdir(parents=True, exist_ok=True)
    for k in keys.values():
        (cache / f"{k}.jsonl.gz").write_bytes(gzip.compress(FACTS.read_bytes()))
    return keys


def judged_again(*args):
    raise AssertionError("judged again")


def test_taking_away_a_declaration_s_last_mention_fails_naming_its_module_the_line_and_the_fix_with_no_lean_built(
        repo, tmp_path, monkeypatch):
    base = git(repo, "rev-parse", "HEAD")
    keys = keep_facts(repo, base, tmp_path / "keys")
    top = commit(repo, "drop the name", {"tools/lonely.py": LONELY.replace('NAME = "Dead.lonely"\n', "")})
    fake = tmp_path / "bin"
    fake.mkdir()
    for tool in ("lake", "lean"):
        (fake / tool).write_text('#!/bin/sh\ntouch "$0.ran"\nexit 1\n')
        (fake / tool).chmod(0o755)
    env = os.environ.copy() | {"PATH": os.pathsep.join([str(fake), *os.get_exec_path()])}
    r = subprocess.run([sys.executable, str(GATE), base, top, "--json"], cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode == 1, r.stderr
    doc = json.loads(r.stdout)
    assert doc["kind"] == G.KIND and doc["stage"] == "facts" and doc["below"] == base and doc["facts"] == keys
    assert doc["seed_files"] == ["tools/lonely.py"] and doc["went"] == ["decl:Dead.lonely"], \
        "a name no Lean file has a word for (`here.joinpath`) can't seed the analysis"
    dead, gone = doc["failures"]
    assert (dead["area"], dead["modules"], gone["area"], gone["modules"]) == (
        "Definitions/Dead", ["Definitions.Dead"], "Definitions/Gone", ["Definitions.Gone"])
    assert dead["why"] == (f"Definitions/Dead would have 1 wholly unread modules (2 declarations), above its cap of 0 (0 "
                           f"declarations) in {G.CAPS} and up from 0 (0) at {base[:12]}: Definitions.Dead (2 declarations, "
                           "security/Definitions/Dead.lean) loses the last mention outside Lean that kept it: `Dead.lonely` at "
                           "tools/lonely.py:2")
    assert gone["why"].endswith("Definitions.Gone (0 declarations, security/Definitions/Gone.lean) was kept only through what went")
    assert dead["fix"][0] == ("delete Definitions.Dead (its file, and its import where another module imports it): nothing "
                              "reads it") and dead["fix"][1].startswith("or seed it again: keep a mention of it outside Lean")
    assert not list(fake.glob("*.ran")), "the gate builds no Lean"
    assert G.main([base, top]) == 1
    monkeypatch.setattr(G, "judged", judged_again)
    same = git(repo, "commit-tree", f"{top}^{{tree}}", "-p", base, "-m", "the same change, merged again")
    doc = G.gate(base, same)
    assert (doc["top"], [f["why"] for f in doc["failures"]]) == (same, [dead["why"], gone["why"]]), \
        "drive's next merge of the same trees after `main` moves reads the verdict back"
    assert doc["notes"][-1] == f"kept: the verdict of {base[:12]}..{top[:12]}, the same trees, by the same code"


def test_a_change_that_leaves_each_name_somewhere_or_that_a_lean_check_reads_passes(repo, tmp_path):
    base = git(repo, "rev-parse", "HEAD")
    moved = commit(repo, "move the name", {"tools/lonely.py": None, "tools/other.py": LONELY})
    doc = G.gate(base, moved)
    assert (doc["stage"], doc["went"], doc["failures"]) == ("git", [], []), "a name moved to another file stays"
    git(repo, "checkout", "-q", "-b", "b", base)
    other = commit(repo, "drop a name no Lean has", {"tools/lonely.py": LONELY.replace('WHERE = here.joinpath("x")\n', "")})
    doc = G.gate(base, other)
    assert (doc["stage"], doc["went"], doc["failures"]) == ("git", [], [])
    git(repo, "checkout", "-q", "-b", "c", base)
    lean = commit(repo, "drop the name and edit the Lean", {"tools/lonely.py": "", "security/Definitions/Dead.lean":
                                                            (repo / "security/Definitions/Dead.lean").read_text() + "\n"})
    doc = G.gate(base, lean)
    assert doc["stage"] == "lean" and doc["failures"] == [] and "the Lean check's ratchet judges it" in doc["notes"][0]


def test_with_none_of_the_tree_s_facts_it_can_t_judge_and_with_main_s_it_judges_on_those(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(G, "stored", lambda keys, cache, recorded, **_: ({}, ["no store in this test"]))
    base = git(repo, "rev-parse", "HEAD")
    top = commit(repo, "drop the name", {"tools/lonely.py": ""})
    assert G.main([base, top]) == 2
    err = capsys.readouterr().err
    assert "can't judge" in err and f"the facts of the Lean of {base[:12]}" in err and "no store in this test" in err
    git(repo, "checkout", "-q", "main")
    git(repo, "reset", "-q", "--hard", base)
    main = commit(repo, "a Lean change on main", {"security/Definitions/Core.lean":
                                                  (repo / "security/Definitions/Core.lean").read_text() + "\n"})
    keep_facts(repo, main, tmp_path / "keys")
    doc = G.gate(main, top)
    assert doc["below"] == base and [f["area"] for f in doc["failures"]] == ["Definitions/Dead", "Definitions/Gone"]
    assert f"judged on the facts of `main` {main[:12]}'s Lean" in doc["notes"][0]
    assert doc["failures"][0]["why"].endswith(f"(on the facts of `main` {main[:12]}'s Lean)")
    monkeypatch.setattr(G, "judged", judged_again)
    assert G.gate(main, top)["notes"][-1].startswith("kept: ")
    moved = commit(repo, "main's Lean moves", {"security/Definitions/Core.lean":
                                               (repo / "security/Definitions/Core.lean").read_text() + "\n"})
    with pytest.raises(AssertionError, match="judged again"):
        G.gate(moved, top)


class Store:
    """The few calls of `research.store.local.LocalStore` the lookups make, over ``runs`` (newest first: (id, {path: (bytes,
    the sha256 its listing names)})), each run of the commit ``of`` names, if any."""

    remote = None

    def __init__(self, runs, of=None):
        self.runs = dict(runs)
        of = of or {}
        self.index = type("Index", (), {"sql": staticmethod(
            lambda q, args: [(c,) for c in dict.fromkeys(of.get(rid) for rid, _ in runs) if c in args]
            if q.startswith("SELECT DISTINCT source_commit") else [(rid,) for rid, _ in runs if of.get(rid) in args]
            if "source_commit" in q else [(rid,) for rid, _ in runs][:args[0]])})()

    def get_attempt(self, rid):
        return {"outputs": {"run_files": f"art:{rid}"}}

    def listing(self, art):
        return [{"path": p, "sha256": sha, "remote": "s3://x"} for p, (_, sha) in self.runs[art[4:]].items()]

    def fetch(self, art, to, paths):
        for p in paths:
            (to / p).parent.mkdir(parents=True, exist_ok=True)
            (to / p).write_bytes(self.runs[art[4:]][p][0])
        return to


def test_with_neither_tree_s_facts_it_judges_on_the_newest_ancestor_s_a_check_published_says_they_are_stale_and_keeps_nothing(
        repo, tmp_path, monkeypatch, capsys):
    """No facts of a Lean are published until its lock run lands, about an hour after it, and the gate couldn't judge a PR
    above it meanwhile.  With neither ``LO``'s nor `main`'s, it reads those of the newest commit below ``LO`` along first
    parents, within `ANCESTORS`, that a recorded check published all of at; past the bound it can't judge."""
    old = git(repo, "rev-parse", "HEAD")
    _, keys = keys_at(repo, old, tmp_path / "keys")
    core = (repo / "security/Definitions/Core.lean").read_text()
    lean = commit(repo, "a Lean change", {"security/Definitions/Core.lean": core + "\n"})
    _, partial = keys_at(repo, lean, tmp_path / "keys")
    main = commit(repo, "another", {"security/Definitions/Core.lean": core + "\n\n"})
    top = commit(repo, "drop the name", {"tools/lonely.py": ""})
    data = gzip.compress(FACTS.read_bytes())
    at = "out/lean-audit/unread/{}.jsonl.gz".format
    published = {at(k): (data, hashlib.sha256(data).hexdigest()) for k in keys.values()}
    store = Store([("r2", {at(partial["security"]): published[at(keys["security"])]}), ("r1", published)],
                  {"r2": lean, "r1": old})
    monkeypatch.setattr(G, "evidence_store", lambda: store)
    monkeypatch.setattr(G, "ANCESTORS", 1)
    assert G.main([main, top]) == 2
    err = capsys.readouterr().err
    assert "can't judge" in err and f"within 1 below {main[:12]} along first parents" in err, err
    assert f"{lean[:12]}: not all of its Lean's facts are kept or published" in err, "the newest check published one of two"
    monkeypatch.setattr(G, "ANCESTORS", 2)
    assert G.main([main, top, "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert (doc["below"], doc["facts_from"], doc["stale"], doc["facts"]) == (main, old, True, keys)
    assert [f["area"] for f in doc["failures"]] == ["Definitions/Dead", "Definitions/Gone"]
    assert doc["notes"][0].startswith(f"judged on the stale facts of {old[:12]}'s Lean, below {main[:12]} along first parents")
    assert doc["failures"][0]["why"].endswith(f"(on the stale facts of {old[:12]}'s Lean)")
    monkeypatch.setattr(G, "judged", judged_again)
    with pytest.raises(AssertionError, match="judged again"):
        G.gate(main, top)


def test_the_store_lookup_takes_the_newest_recorded_check_s_facts_of_each_key_whose_bytes_check_out(tmp_path):
    k1, k2 = "1" * 64, "2" * 64
    at = "out/lean-audit/unread/{}.jsonl.gz".format
    runs = [("r3", {at(k1): (b"tampered", hashlib.sha256(b"other").hexdigest()), at(k2): (b"two", hashlib.sha256(b"two").hexdigest()),
                    "out/lean-audit/unread/seen.json": (b"[]", hashlib.sha256(b"[]").hexdigest())}),
            ("r2", {at(k1): (b"one", hashlib.sha256(b"one").hexdigest()), at(k2): (b"old", hashlib.sha256(b"old").hexdigest())}),
            ("r1", {at(k1): (b"oldest", hashlib.sha256(b"oldest").hexdigest())})]
    got, errors = G.stored({"security": k1, "security/Proofs": k2}, tmp_path, "1", store=Store(runs))
    assert {p: f.read_bytes() for p, f in got.items()} == {"security": b"one", "security/Proofs": b"two"}
    assert got["security"] == tmp_path / f"{k1}.jsonl.gz" and errors == [f"r3: {at(k1)} isn't the bytes its listing names"]
    (tmp_path / "none").mkdir()
    got, errors = G.stored({"security": k1}, tmp_path / "none", "1", store=Store(runs[:1]), scan=1)
    assert got == {} and errors == [f"r3: {at(k1)} isn't the bytes its listing names"]
    (tmp_path / "old").mkdir()
    got, _ = G.stored({"security": k1}, tmp_path / "old", "1", store=Store([runs[0], runs[2]], {"r1": "c" * 40}), scan=1,
                      commits=("c" * 40,))
    assert got["security"].read_bytes() == b"oldest", "a check of the commit judged is read however many checks ran since"


def test_the_names_lean_declares_read_alike_under_any_locale_and_are_kept_by_the_lean_s_tree(repo, tmp_path, monkeypatch):
    """Under `C`, git's `[^...⟨]` matches bytes, so `fooΨ.psiOnly` would stop inside `Ψ` (whose last byte is one of `⟨`'s) and
    the line would be no UTF-8; the gate runs git under `LOCALE`, so the host's locale doesn't change what it reads."""
    rev = commit(repo, "a name past ASCII", {"security/Definitions/Psi.lean": "theorem fooΨ.psiOnly : True := trivial\n"})
    monkeypatch.setenv("LC_ALL", "C")
    heads, lasts = G.declared(rev)
    assert "foo" in heads and "psiOnly" in lasts
    assert G.declared(rev, tmp_path / "kept") == (heads, lasts)
    real = G.git_text

    def no_grep(*args, **kw):
        assert args[0] != "grep", "a kept set greps nothing"
        return real(*args, **kw)

    monkeypatch.setattr(G, "git_text", no_grep)
    assert G.declared(rev, tmp_path / "kept") == (heads, lasts)
