"""The lean-lock runner's work directory, with `check.py` stood in for: whatever the audit ends in, the runner leaves its
outputs, its log and the reads the executor staged, and none of the scratch tree it audited in or the audit's own directory,
since the executor brings the whole work directory back with the result; the worker's fetched packages, linked into the
scratch tree, are left alone."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_graph import runners
from verity_graph.runners import lean_lock as lock_runner

PKG = {"pkg/lakefile.toml": 'name = "pkg"\n\n[[lean_lib]]\nname = "Pkg"\n', "pkg/Pkg.lean": "def p := 0\n",
       "pkg/lean-audit.json": json.dumps({"roots": ["Pkg"]}), "pkg/lean-toolchain": "leanprover/lean4:v4.34.0\n",
       "pkg/lake-manifest.json": json.dumps({"version": "1.1.0", "packagesDir": ".lake/packages", "name": "pkg",
                                             "lakeDir": ".lake", "packages": []})}
TASK = {"id": "lean-lock:pkg", "kind": "lean-lock", "vars": {"package": "pkg"}, "inputs": sorted(PKG),
        "reads": [{"task": "lean-module:pkg/Pkg.lean", "output": "olean"}],
        "outputs": [{"name": "records", "path": "out/pkg/lean-audit.json"}],
        "resources": {"cores": 1, "mem_gb": 8, "disk_gb": 2}, "timeout_s": 60}


def files(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() or p.is_symlink()}


@pytest.mark.parametrize("ends", ["pass", "timeout"])
def test_a_lock_leaves_no_scratch_tree_or_audit_in_its_work_directory(tmp_path, monkeypatch, ends):
    src, deps, work = tmp_path / "src", tmp_path / "deps", tmp_path / "work"
    for f, text in PKG.items():
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text(text)
    (deps / "mathlib").mkdir(parents=True)
    (deps / "mathlib" / "Mathlib.olean").write_bytes(b"fetched")
    (work / "out" / "lean").mkdir(parents=True)
    (work / "out" / "lean" / "Pkg.olean").write_bytes(b"read")
    monkeypatch.setattr(lock_runner.la, "ROOT", src)
    monkeypatch.setattr(lock_runner, "packages", lambda pkg, work: deps)
    seen = {}

    def audit(argv, work, cwd, env=None):
        out, pkg = Path(argv[argv.index("--out") + 1]), Path(argv[-1])
        seen["tree"] = files(cwd)
        out.mkdir(parents=True)
        (out / "facts.json.gz").write_bytes(b"facts")
        if ends == "timeout":
            raise runners.Timeout(argv[0])
        (out / "audit.json").write_text(json.dumps({"packages": [{"package": str(pkg), "failures": []}]}))
        return 0

    monkeypatch.setattr(lock_runner, "call", audit)
    if ends == "timeout":
        with pytest.raises(runners.Timeout):
            lock_runner.run(TASK, work)
    else:
        assert lock_runner.run(TASK, work)["status"] == "pass"
    assert {"pkg/Pkg.lean", "pkg/.lake/build/lib/lean/Pkg.olean", "pkg/.lake/packages"} <= seen["tree"], \
        "the audit ran in a scratch tree of the inputs, with the reads in its build and the fetched packages linked"
    assert files(work) == {"out/lean/Pkg.olean", *(["out/pkg/lean-audit.json"] if ends == "pass" else [])}, \
        "only the outputs and the staged reads are left for the executor to bring back"
    assert files(deps) == {"mathlib/Mathlib.olean"}, "the worker's fetched packages outlive the scratch tree linking them"
