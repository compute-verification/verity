"""The runner's capture of a dirty source (`env.DIRTY_CAPTURE`): on 10 Oct an ad-hoc sweep of node 1's source trees deleted the only
copy of seven runs' dirty sources (files earlier runs left in a shared tree), so the runner now writes the dirty bytes into the run
dir, custody always uploads them, and `evict_src` lets such a tree go only once the run's preserved record holds them."""
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from research import env as E
from research.store import custody as CU

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True, env=ENV)


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "src"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / ".gitignore").write_text("build/\n")
    (r / "a.lean").write_text("theorem a : True := trivial\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "a")
    return r


def _replay(tar_path: Path) -> str:
    """The digest the tar's bytes give under `_dirty_digest`'s rule."""
    h = hashlib.sha256()
    with tarfile.open(tar_path) as t:
        members = t.getmembers()
        assert members[0].name == E.DIRTY_PATCH
        h.update(t.extractfile(members[0]).read())
        for m in members[1:]:
            rel = m.name.removeprefix(E.DIRTY_UNTRACKED + "/")
            h.update(rel.encode() + b"\0" + t.extractfile(m).read() + b"\0")
    return h.hexdigest()


def test_a_dirty_sources_capture_holds_its_diff_and_untracked_files_and_reproduces_its_digest(tmp_path):
    r = _repo(tmp_path)
    (r / "a.lean").write_text("theorem a : 1 = 1 := rfl\n")
    (r / "fold-out").mkdir()
    (r / "fold-out" / "steps.txt").write_bytes(b"\x00step 1\n")
    (r / "build").mkdir()
    (r / "build" / "big.o").write_bytes(b"ignored")
    cap = tmp_path / "run" / E.DIRTY_CAPTURE
    cap.parent.mkdir()
    conds = E.source_conditions(r, dirty_capture=cap)
    assert conds["source.dirty"] is True
    assert conds["source.dirty_digest"] == E.source_conditions(r)["source.dirty_digest"], "capturing never changes the digest"
    assert _replay(cap) == conds["source.dirty_digest"]
    with tarfile.open(cap) as t:
        assert t.getnames() == [E.DIRTY_PATCH, f"{E.DIRTY_UNTRACKED}/fold-out/steps.txt"], "ignored files are not the source"
        assert b"1 = 1" in t.extractfile(E.DIRTY_PATCH).read()
    patched = tmp_path / "replayed"
    _git(tmp_path, "clone", "-q", str(r), str(patched))
    with tarfile.open(cap) as t:
        subprocess.run(["git", "apply", "-"], cwd=patched, input=t.extractfile(E.DIRTY_PATCH).read(), check=True)
    assert (patched / "a.lean").read_text() == "theorem a : 1 = 1 := rfl\n"


def test_a_clean_source_writes_no_capture(tmp_path):
    r = _repo(tmp_path)
    cap = tmp_path / E.DIRTY_CAPTURE
    assert E.source_conditions(r, dirty_capture=cap)["source.dirty"] is False
    assert not cap.exists()


def test_a_source_over_the_untracked_cap_leaves_no_partial_capture(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    (r / "x").write_bytes(b"x" * 10)
    (r / "y").write_bytes(b"y" * 10)
    monkeypatch.setattr(E, "UNTRACKED_CAP_BYTES", 15)
    cap = tmp_path / E.DIRTY_CAPTURE
    assert E.source_conditions(r, dirty_capture=cap)["source.dirty_digest"] == "too-large"
    assert not cap.exists()


def test_an_unreadable_untracked_file_drops_the_capture_and_keeps_the_digest(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    (r / "x").write_text("x")
    real_open = open

    def failing_open(p, *a, **k):
        if str(p).endswith("/x"):
            raise PermissionError(p)
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", failing_open)
    cap = tmp_path / E.DIRTY_CAPTURE
    digest = E.source_conditions(r, dirty_capture=cap)["source.dirty_digest"]
    assert digest == E.source_conditions(r)["source.dirty_digest"] and digest not in (None, "too-large")
    assert not cap.exists(), "a capture missing a hashed file would not reproduce the digest"


def test_a_commit_landing_during_the_read_is_read_again_never_recorded_dirty_with_the_empty_digest(tmp_path, monkeypatch):
    """r20261019-180708-c2a5: HEAD and status were read before a lane's commit landed and `git diff HEAD` after it, so the record
    named the parent, dirty, with sha256(b"").  The diff is now against the commit read, and a read HEAD moved during is taken again."""
    r = _repo(tmp_path)
    (r / "a.lean").write_text("theorem a : 1 = 1 := rfl\n")
    real, landed = E._git, []

    def git(args, root, **kw):
        if args[0] == "diff" and not landed:
            _git(r, "commit", "-q", "-am", "landed")
            landed.append(True)
        return real(args, root, **kw)

    monkeypatch.setattr(E, "_git", git)
    cap = tmp_path / E.DIRTY_CAPTURE
    c = E.source_conditions(r, dirty_capture=cap)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=r, capture_output=True, text=True, check=True).stdout.strip()
    assert landed and (c["source.commit"], c["source.dirty"], c["source.dirty_digest"]) == (head, False, None)
    assert not cap.exists(), "the first read's capture is not this source's"


def test_a_head_that_never_settles_is_unknown_not_guessed(tmp_path, monkeypatch):
    r = _repo(tmp_path)
    real, n = E._git, iter(range(100))
    monkeypatch.setattr(E, "_git", lambda args, root, **kw: f"{next(n):040x}" if args == ["rev-parse", "HEAD"] else real(args, root, **kw))
    c = E.source_conditions(r)
    assert [c[f"source.{k}"] for k in ("commit", "tree", "dirty", "dirty_digest")] == [None] * 4


def test_an_index_only_change_is_clean(tmp_path):
    """`git status` lists a change staged and then undone in the working tree, but no byte the run reads differs from the commit's:
    the source is clean, so a dirty source's digest is never sha256(b"")."""
    r = _repo(tmp_path)
    (r / "a.lean").write_text("staged\n")
    _git(r, "add", "a.lean")
    (r / "a.lean").write_text("theorem a : True := trivial\n")
    assert subprocess.run(["git", "status", "--porcelain"], cwd=r, capture_output=True, text=True, check=True).stdout.strip()
    cap = tmp_path / E.DIRTY_CAPTURE
    c = E.source_conditions(r, dirty_capture=cap)
    assert (c["source.dirty"], c["source.dirty_digest"]) == (False, None) and not cap.exists()


def test_an_untracked_nested_repository_is_in_the_digest_and_keeps_the_tree(tmp_path):
    """An untracked directory git lists whole (`sub/`, a nested repository) is not a file: its name enters the digest, so the source
    is dirty with a digest that is not sha256(b""), and the capture is dropped, since it could not hold the directory's bytes."""
    r = _repo(tmp_path)
    (r / "sub").mkdir()
    _git(r / "sub", "init", "-q")
    (r / "sub" / "f").write_text("f\n")
    cap = tmp_path / E.DIRTY_CAPTURE
    c = E.source_conditions(r, dirty_capture=cap)
    assert c["source.dirty"] is True and c["source.dirty_digest"] not in (None, E.EMPTY_DIGEST, "too-large")
    assert c["source.dirty_digest"] == hashlib.sha256(b"sub/\0\0").hexdigest() and not cap.exists()


def test_the_runner_moves_the_capture_into_the_run_dir_and_names_it_in_job_json(tmp_path):
    from research import cli

    r = _repo(tmp_path)
    (r / "out.txt").write_text("left by an earlier run\n")
    cap = Path(tempfile.mkdtemp(dir=tmp_path)) / E.DIRTY_CAPTURE
    conds = E.source_conditions(r, dirty_capture=cap)
    prep = {"section": {"conditions": conds}, "dirty_capture": cap}
    rd = tmp_path / "runs" / "r1"
    rd.mkdir(parents=True)
    cli._place_dirty_capture(prep, rd)
    rec = prep["section"]["dirty_capture"]
    data = (rd / E.DIRTY_CAPTURE).read_bytes()
    assert rec == {"path": E.DIRTY_CAPTURE, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "dirty_digest": conds["source.dirty_digest"]}
    assert not cap.parent.exists(), "the staging dir goes"


def test_custody_always_uploads_the_capture_whatever_the_undeclared_cap(tmp_path):
    rd = tmp_path / "r1"
    rd.mkdir()
    (rd / "job.json").write_text(json.dumps({"run": {"id": "r1"}}))
    (rd / E.DIRTY_CAPTURE).write_bytes(b"t" * 4096)
    (rd / "scratch.bin").write_bytes(b"s" * 4096)
    kept, trimmed = CU.select(rd, cap=0)
    assert E.DIRTY_CAPTURE in kept and [t["path"] for t in trimmed] == ["scratch.bin"]
