"""The build family: on this tree its tasks are check's steps (the inputs check keys them by, the executables the Flock suite
runs, what check.py reads to run the agreement), and its runners tell a fail, the tree's, from an error, the machine's, on
stand-ins for lake, check_build.sh, check.py and the store."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import build
from verity_graph.runners import flock_circuit_build, lean_agreement, lean_build

spec = importlib.util.spec_from_file_location("graph_check", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

AGREE, BUILD, CIRCUIT = (f"lean-agreement:{build.VERIFIER}", f"lean-build:{build.LEAN}", f"flock-circuit-build:{build.FLOCK}/live")


@pytest.fixture(scope="module")
def doc() -> dict:
    return graph.listing(ROOT, {"build"})


def _own(task: dict) -> list[str]:
    return [p for p in task["inputs"] if p not in graph.runner_sources(task["kind"])]


def test_the_build_family_is_check_s_steps_with_the_inputs_check_keys_them_by(doc):
    by = {t["id"]: t for t in doc["tasks"]}
    assert sorted(by) == [CIRCUIT, AGREE, BUILD]
    files = graph.tracked(ROOT)
    assert _own(by[CIRCUIT]) == CHECK.rust_tests_inputs(files) and by[CIRCUIT]["env"] == "rust"
    verifier = [p for p in _own(by[AGREE]) if p.startswith(build.VERIFIER + "/")]
    assert sorted({*_own(by[BUILD]), *verifier}) == sorted(CHECK.agreement_closure(files)), \
        "the agreement's key in check covers the verifier's package (the build's inputs) and its own directory"
    assert by[AGREE]["reads"] == [{"task": BUILD, "output": "flock-verify"}] and not by[AGREE]["key"] and by[BUILD]["key"]


def test_lean_build_outputs_the_executables_the_flock_suite_runs_at_their_paths(doc):
    lb, agree = (next(t for t in doc["tasks"] if t["id"] == i) for i in (BUILD, AGREE))
    tests = tomllib.loads((ROOT / "verity" / "ml" / "flock" / "pyproject.toml").read_text())["tool"]["verity"]["tests"]
    bins = [a for a in tests["artifacts"] if a.startswith(f"{build.LEAN}/.lake/build/bin/")]
    assert bins and set(bins) <= {o["path"] for o in lb["outputs"]}
    assert agree["vars"]["verify"] == next(o["path"] for o in lb["outputs"] if o["name"] == "flock-verify")


def test_the_runners_run_what_check_runs():
    assert (lean_build.ELAN, lean_build.LAKE_BUILD) == (CHECK.SUITES.ELAN, CHECK.SUITES.LAKE_BUILD)
    assert next(j for j in CHECK.JOBS if j.name == "flock-circuit-build").halts_on in flock_circuit_build.FAILS
    assert f'print(f"{lean_agreement.CANNOT} {{why}}"' in (ROOT / lean_agreement.CHECK).read_text()
    assert (ROOT / flock_circuit_build.SCRIPT).is_file() and (ROOT / lean_agreement.PIN).is_file()


AUDIT = """
import os, runpy, sys
from pathlib import Path
root, seen = Path(sys.argv[1]), set()
def hook(event, args):
    if event == "open" and isinstance(args[0], str) and not (isinstance(args[2], int) and args[2] & os.O_ACCMODE == os.O_WRONLY):
        p = Path(args[0]).resolve()
        if p.is_relative_to(root) and not p.is_relative_to(Path(sys.prefix).resolve()):
            seen.add(p.relative_to(root).as_posix())
sys.addaudithook(hook)
sys.argv = sys.argv[2:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit:
    pass
print("\\n".join(sorted(seen)))
"""


def test_check_reads_are_all_the_agreement_reads_of_the_tree_outside_the_verifier(tmp_path):
    """check.py's own reads end at its first look at the upstream build: everything after is under the verifier's directory."""
    (tmp_path / "up").write_bytes(b"not the pin")
    (tmp_path / "in").write_bytes(b"not the pin")
    p = subprocess.run([sys.executable, "-c", AUDIT, str(ROOT), str(ROOT / lean_agreement.CHECK), "--lean-agreement",
                        str(tmp_path / "out"), str(tmp_path / "up"), str(tmp_path / "in"), str(tmp_path / "fv")],
                       cwd=ROOT, capture_output=True, text=True)
    assert "is not the pinned upstream" in p.stderr, p.stderr
    opened = {f"{Path(f).parent.parent}/{Path(f).name.split('.')[0]}.py" if "/__pycache__/" in f else f
              for f in p.stdout.split()}
    assert opened and not {f for f in opened if not f.startswith(build.VERIFIER + "/")} - set(build.CHECK_READS)


def test_a_tree_without_the_verifier_lists_none_of_its_lean_tasks(tmp_path):
    (tmp_path / build.LEAN).mkdir(parents=True)
    (tmp_path / build.LEAN / "lakefile.toml").write_text('defaultTargets = ["v"]\n[[lean_exe]]\nname = "v"\n')
    assert build.tasks(tmp_path, []) == []
    script = [f"{build.FLOCK}/check_build.sh"]
    assert [t["id"] for t in build.tasks(tmp_path, script)] == [CIRCUIT]
    lean = [f"{build.LEAN}/lakefile.toml", f"{build.LEAN}/V.lean"]
    assert [t["id"] for t in build.tasks(tmp_path, lean)] == [BUILD]
    assert [t["id"] for t in build.tasks(tmp_path, [*lean, f"{build.VERIFIER}/upstream.json"])] == [BUILD], \
        "an agreement needs flock-verify among the executables"


def _graph(*tasks: dict) -> dict:
    full = [{"about": "x", "vars": {}, "inputs": [], "reads": [], "outputs": [], "argv": ["verity", "graph", "run", t["id"]],
             "resources": {"cores": 1, "mem_gb": 1, "disk_gb": 1}, "timeout_s": 60, "key": None, **t} for t in tasks]
    return {"envs": {e: {"digest": contract.digest(e.encode())} for e in ("py", "lean", "rust")}, "files": {}, "tasks": full}


def _execute(doc: dict, work: Path, runner) -> dict:
    task = doc["tasks"][-1]
    res = runners.execute(task, doc, work, runner)
    assert contract.result_errors(res, task) == [], res
    return res


def _tree(tmp_path: Path, monkeypatch, files: dict[str, str]) -> Path:
    root = tmp_path / "tree"
    for p, text in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
        (root / p).chmod(0o755)
    for m in (lean_build, flock_circuit_build, lean_agreement):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                        # no elan or cargo of the machine's
    monkeypatch.setenv("PATH", str(root / "bin"), prepend=os.pathsep)
    return root


LAKE = """#!/bin/sh
case "$1" in
  --version) {version} ;;
  build) mkdir -p .lake/build/bin && printf 'fv' > .lake/build/bin/flock-verify && {build} ;;
esac
"""


@pytest.mark.parametrize(("version", "build_", "status"), [
    ("exit 0", "exit 0", "pass"), ("exit 0", "echo 'error: build failed'; exit 1", "fail"),
    ("exit 0", "kill -9 $$", "error"), ("echo 'error: could not download toolchain'; exit 1", "exit 0", "error")])
def test_lean_build(tmp_path, monkeypatch, version, build_, status):
    _tree(tmp_path, monkeypatch, {"bin/lake": LAKE.format(version=version, build=build_), "pkg/lean-toolchain": "v4\n"})
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean",
                  "outputs": [{"name": "flock-verify", "path": "pkg/.lake/build/bin/flock-verify"}]})
    res = _execute(doc, tmp_path / "work", lean_build.run)
    assert res["status"] == status, res
    assert (res["outputs"] == {"flock-verify": contract.digest(b"fv")}) == (status == "pass")


@pytest.mark.parametrize(("script", "status"), [
    ("exit 0", "pass"), ("echo 'error: could not compile `flock-live` (bin \"flock-circuit\")'; exit 101", "fail"),
    ("echo 'error: crates/flock-core/src/lib.rs: patch does not apply'; exit 1", "fail"),
    ("echo \"fatal: unable to access 'https://github.com/succinctlabs/flock/'\"; exit 128", "error"),
    ("kill -9 $$", "error")])
def test_flock_circuit_build(tmp_path, monkeypatch, script, status):
    _tree(tmp_path, monkeypatch, {flock_circuit_build.SCRIPT: f"#!/bin/bash\n{script}\n"})
    res = _execute(_graph({"id": CIRCUIT, "kind": "flock-circuit-build", "env": "rust"}), tmp_path / "work",
                   flock_circuit_build.run)
    assert res["status"] == status, res


CHECK_STANDIN = """import json, os, sys
from pathlib import Path
out, verify = Path(sys.argv[2]), Path(sys.argv[5])
assert sys.argv[1] == "--lean-agreement" and os.access(verify, os.X_OK)
how = verify.read_text()
if how == "agree":
    (out / "flock-agreement" / "ci").mkdir(parents=True)
    sets = [{"set": 0, "sessions": 3, "agree": 3}, {"set": 1, "skipped": "no upstream binary"}]
    (out / "flock-agreement" / "ci" / "sets.json").write_text(json.dumps({"seed": 1, "sets": sets}))
    sys.exit(0)
if how == "cannot":
    print("check: lean-agreement: error: cannot run here: this machine may use 8 GB", file=sys.stderr)
sys.exit(1)
"""


@pytest.mark.parametrize(("verify", "fetched", "status"), [
    ("agree", None, "pass"), ("disagree", None, "fail"), ("cannot", None, "error"),
    ("agree", "other bytes", "error"), ("agree", "no fetch", "error")])
def test_lean_agreement(tmp_path, monkeypatch, verify, fetched, status):
    pins = {"upstream": b"upstream build", "inputs": b"inputs"}
    pin = {k: {"art": f"art:{k}", "sha256": hashlib.sha256(v).hexdigest()} for k, v in pins.items()}
    _tree(tmp_path, monkeypatch, {lean_agreement.PIN: json.dumps(pin), lean_agreement.CHECK: CHECK_STANDIN})
    real = runners.call

    def call(argv, work, cwd, env=None):
        if argv[1:5] != ["-m", "research", "data", "fetch"]:
            return real(argv, work, cwd, env)
        if fetched == "no fetch":
            with open(work / graph.LOG, "a") as log:
                log.write(f"research data fetch: {argv[5]}: no manifest locally (no remote configured)\n")
            return 1
        to = Path(argv[argv.index("--to") + 1])
        to.mkdir(parents=True)
        (to / "file.tar.gz").write_bytes(fetched.encode() if fetched else pins[argv[5].removeprefix("art:")])
        return 0
    monkeypatch.setattr(lean_agreement, "call", call)
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean",
                  "outputs": [{"name": "flock-verify", "path": "pkg/bin/flock-verify"}]},
                 {"id": AGREE, "kind": "lean-agreement", "env": "py", "vars": {"verify": "pkg/bin/flock-verify"},
                  "reads": [{"task": "lean-build:pkg", "output": "flock-verify"}],
                  "outputs": [{"name": "sets", "path": lean_agreement.SETS}]})
    work = tmp_path / "work"
    (work / "pkg" / "bin").mkdir(parents=True)
    (work / "pkg" / "bin" / "flock-verify").write_text(verify)                 # an artifact is its bytes: no mode bits
    res = _execute(doc, work, lean_agreement.run)
    assert res["status"] == status, res
    if status == "pass":
        assert res["detail"] == {"sets": 1, "sessions": 3, "agree": 3} and set(res["outputs"]) == {"sets"}
    if fetched == "no fetch":
        assert "(no remote configured)" in res["detail"]["error"] and "--store-read" in res["detail"]["error"], res
