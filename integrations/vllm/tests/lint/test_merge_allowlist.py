"""`merge_allowlist.py`: the shrink-only lists merged by key, as a function and as git's merge driver."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from research.clone import RECORD_DRIVER

from . import _ratchet as R
from . import merge_allowlist as MA

KEEP = R.ROOT / "tests" / "dead_code_keep.json"


def _entry(file: str, count: int = 1) -> dict:
    return json.loads(R.entry_line((file, "module", "<module>", "lines"), count))


def _doc(*entries: dict) -> dict:
    return {"rule": "P10 (size ratchet)", "entries": list(entries)}


A, B, C, D = (_entry(f"verity_vllm/{m}.py", 900) for m in "abcd")


def test_two_prs_that_each_retire_a_neighbouring_entry_merge_and_an_entry_changed_two_ways_refuses():
    """As text, deleting two neighbouring lines conflicts; by key, each side's deletion is kept."""
    base = _doc(A, B, C, D)
    merged, why = MA.merge(base, _doc(A, C, D), _doc(A, B, D))
    assert why == [] and merged == _doc(A, D)
    merged, why = MA.merge(base, _doc(A, B, C, D, _entry("verity_vllm/e.py")), _doc(A, _entry("verity_vllm/b.py", 850), C, D))
    assert why == [] and merged == _doc(A, _entry("verity_vllm/b.py", 850), C, D, _entry("verity_vllm/e.py"))
    _, why = MA.merge(base, _doc(A, _entry("verity_vllm/b.py", 850), C, D), _doc(A, _entry("verity_vllm/b.py", 800), C, D))
    assert why == ["verity_vllm/b.py [module] <module>: lines: changed on both sides, differently"]
    _, why = MA.merge(base, _doc(A, C, D), _doc(A, _entry("verity_vllm/b.py", 800), C, D))
    assert why == ["verity_vllm/b.py [module] <module>: lines: removed on one side and changed on the other"]


def _pair(file: str, rule: str, count: int) -> dict:
    return {"file": file, "rule": rule, "count": count}


def _ruff_doc(*entries: dict) -> dict:
    return {"rule": "ruff (ruff.toml)", "entries": list(entries)}


def test_the_repositorys_ruff_list_merges_per_file_and_rule_and_a_count_changed_two_ways_takes_the_larger():
    """Its counts may run above what a tree finds, so the larger of two changes allows what either side's tree finds: one
    side fixed a finding (5 -> 4) and the other two (5 -> 3); one moved a file's entry and the other fixed one there."""
    e, f, g = _pair("tools/e.py", "E731", 5), _pair("tools/f.py", "F401", 2), _pair("tools/g.py", "UP017", 1)
    merged, why = MA.merge(_ruff_doc(e, f, g), _ruff_doc(e, g), _ruff_doc(e, f))
    assert why == [] and merged == _ruff_doc(e)
    merged, why = MA.merge(_ruff_doc(e, f), _ruff_doc({**e, "count": 4}, f), _ruff_doc({**e, "count": 3}, f))
    assert why == [] and merged == _ruff_doc({**e, "count": 4}, f)
    moved = _pair("tools/sub/e.py", "E731", 5)
    merged, why = MA.merge(_ruff_doc(e, f), _ruff_doc(f, moved), _ruff_doc({**e, "count": 3}, f))
    assert why == [] and merged == _ruff_doc(f, moved, {**e, "count": 3})
    assert MA.dump(merged).splitlines()[3] == '    {"file": "tools/f.py", "rule": "F401", "count": 2},'
    with pytest.raises(ValueError, match="ours: malformed entry"):
        MA.merge(_ruff_doc(e), _ruff_doc({**e, "symbol": "s"}), _ruff_doc(e))


def _env_doc(*entries: dict) -> dict:
    return {"rule": "direct environment reads (tests/lint/test_repo_env_reads.py)", "entries": list(entries)}


def test_the_repositorys_environment_read_list_merges_per_file_and_a_registry_mark_like_any_entry():
    """Per-file counts merge as the ruff list's do (a count changed two ways takes the larger); a settings registry mark
    added on one side is kept, and one removed on one side and kept on the other is removed."""
    a, b = {"file": "tools/a.py", "count": 3}, {"file": "tools/b.py", "count": 1}
    mark = {"file": "tools/x/config.py", "registry": True}
    merged, why = MA.merge(_env_doc(a, b), _env_doc({**a, "count": 2}, b, mark), _env_doc({**a, "count": 1}))
    assert why == [] and merged == _env_doc({**a, "count": 2}, mark)
    merged, why = MA.merge(_env_doc(a, mark), _env_doc(a), _env_doc(a, mark))
    assert why == [] and merged == _env_doc(a)
    assert MA.dump(_env_doc(a, mark)).splitlines()[4] == '    {"file": "tools/x/config.py", "registry": true}'
    for bad in ({"file": "tools/a.py", "registry": False}, {"file": "tools/a.py", "count": True}):
        with pytest.raises(ValueError, match="ours: malformed entry"):
            MA.merge(_env_doc(a), _env_doc(bad), _env_doc(a))


def test_the_keep_map_merges_by_module():
    base = {"schema": "vllm-dead-code-keep/v1", "modules": {"m.a": "why a", "m.b": "why b"}}
    merged, why = MA.merge(base, {**base, "modules": {"m.a": "why a"}}, {**base, "modules": {**base["modules"], "m.c": "why c"}})
    assert why == [] and merged == {"schema": "vllm-dead-code-keep/v1", "modules": {"m.a": "why a", "m.c": "why c"}}
    _, why = MA.merge(base, {**base, "modules": {"m.a": "kept"}}, {**base, "modules": {"m.a": "moved"}})
    assert why == ["modules.m.a: changed on both sides, differently"]


def test_p16s_public_table_merges_by_command():
    """Two PRs that each add a command run from outside the integration both keep theirs; one command given two callers
    refuses."""
    row, tie, pin = {"row": "benchmarks/row.sh"}, {"tie": "benchmarks/tie.sh"}, {"pin": "benchmarks/pin.sh"}
    base = {"rule": "P16 (every CLI command has a caller)", "public": row, "entries": [A]}
    merged, why = MA.merge(base, {**base, "public": {**row, **tie}}, {**base, "public": {**row, **pin}})
    assert why == [] and merged["public"] == {**row, **tie, **pin}
    assert MA.dump(merged).splitlines()[2:7] == ['  "public": {', '    "row": "benchmarks/row.sh",',
                                                '    "tie": "benchmarks/tie.sh",', '    "pin": "benchmarks/pin.sh"', "  },"]
    _, why = MA.merge(base, {**base, "public": {"row": "benchmarks/a.sh"}}, {**base, "public": {"row": "benchmarks/b.sh"}})
    assert why == ["public.row: changed on both sides, differently"]


def test_every_list_is_written_back_byte_for_byte():
    """The merged file is written as the lints' files are, so a merge rewrites only the entries it changed."""
    paths = sorted(R.ALLOWLISTS.glob("*.json")) + [KEEP]
    assert len(paths) > 1
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert MA.dump(json.loads(text)) == text, p.name
    for e in json.loads(sorted(R.ALLOWLISTS.glob("*.json"))[0].read_text(encoding="utf-8"))["entries"]:
        assert MA.dump(_doc(e)).splitlines()[3].strip() == R.entry_line(tuple(e[f] for f in MA.KEY), e.get("count", 1))


def test_a_refused_merge_exits_1_and_leaves_ours(tmp_path, capsys):
    o, a, b = (tmp_path / n for n in "oab")
    o.write_text(MA.dump(_doc(A, B)))
    a.write_text(MA.dump(_doc(A, _entry("verity_vllm/b.py", 850))))
    b.write_text(MA.dump(_doc(A, _entry("verity_vllm/b.py", 800))))
    before = a.read_text()
    assert MA.main([str(o), str(a), str(b), "p10_size.json"]) == 1
    assert a.read_text() == before and "p10_size.json: refused" in capsys.readouterr().err
    b.write_text("[]")
    assert MA.main([str(o), str(a), str(b)]) == 1 and a.read_text() == before


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(repo.parent),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=check, capture_output=True, text=True,
                          env=env)


def test_git_merges_two_branches_that_each_retire_a_neighbouring_entry_with_the_registered_driver(tmp_path):
    repo = tmp_path / "repo"
    lint = repo / "integrations" / "vllm" / "tests" / "lint"
    (lint / "allowlists").mkdir(parents=True)
    (lint / "merge_allowlist.py").write_text(Path(MA.__file__).read_text())
    (repo / ".gitattributes").write_text("integrations/vllm/tests/lint/allowlists/*.json merge=vllm-allowlist\n")
    m = lint / "allowlists" / "p10_size.json"
    _git(repo, "init", "-q", "-b", "main")
    for name, doc in (("main", _doc(A, B, C, D)), ("a", _doc(A, C, D)), ("b", _doc(A, B, D))):
        if name != "main":
            _git(repo, "checkout", "-q", "-b", name, "main")
        m.write_text(MA.dump(doc))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    _git(repo, "checkout", "-q", "a")
    # no driver defined yet: git's own text merge
    _git(repo, "merge", "--no-edit", "b", check=False)
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == [
        "integrations/vllm/tests/lint/allowlists/p10_size.json"]
    _git(repo, "merge", "--abort")
    for k, v in RECORD_DRIVER.items():
        if k.startswith("merge.vllm-allowlist."):
            _git(repo, "config", k, v)
    _git(repo, "merge", "--no-edit", "b")
    assert m.read_text() == MA.dump(_doc(A, D))
