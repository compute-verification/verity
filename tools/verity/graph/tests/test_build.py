"""The build family: on this tree its tasks are check's steps (the inputs check keys them by, the executables the Flock suite
runs and the rest of the verifier's build packed, what check.py reads to run the agreement) and its runners run check's
commands, telling a fail, the tree's, from an error, the machine's (a skip among them: nothing ran), on stand-ins for lake,
check_build.sh, check.py, unit_cut_agree.py and the store; what lean-build outputs goes back into a reader's tree whole."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import build
from verity_graph.runners import flock_circuit_build, flock_rust_tests, lean_agreement, lean_build, lean_unit_cut

spec = importlib.util.spec_from_file_location("graph_check", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

AGREE, CIRCUIT = f"lean-agreement:{build.VERIFIER}", f"flock-circuit-build:{build.FLOCK}/live"
RUST, UNIT = f"flock-rust-tests:{build.FLOCK}/live", f"lean-unit-cut:{build.UNIT_CUT}"


@pytest.fixture(scope="module")
def doc() -> dict:
    return graph.listing(ROOT, {"build"})


def _own(task: dict) -> list[str]:
    return [p for p in task["inputs"] if p not in graph.runner_sources(task["kind"])]


def test_the_build_family_is_check_s_steps_with_the_inputs_check_keys_them_by(doc):
    by = {t["id"]: t for t in doc["tasks"]}
    assert sorted(by) == sorted([CIRCUIT, RUST, AGREE, build.BUILD, UNIT])
    files = graph.tracked(ROOT)
    for rust in (CIRCUIT, RUST):
        assert _own(by[rust]) == CHECK.rust_tests_inputs(files) and by[rust]["env"] == "rust" and by[rust]["key"]
    verifier = [p for p in _own(by[AGREE]) if p.startswith(build.VERIFIER + "/")]
    assert sorted({*_own(by[build.BUILD]), *verifier}) == sorted(CHECK.agreement_closure(files)), \
        "the agreement's key in check covers the verifier's package (the build's inputs) and its own directory"
    verify = [{"task": build.BUILD, "output": "flock-verify"}]
    assert by[AGREE]["reads"] == by[UNIT]["reads"] == verify and not by[AGREE]["key"] and by[build.BUILD]["key"]
    assert {build.UNIT_CUT, "verity/ml/circuits/partition.py"} <= set(_own(by[UNIT])) <= {
        build.UNIT_CUT, *(f for f in files if f.startswith("verity/") and f.endswith("/__init__.py")),
        *(f for f in files if f.startswith("verity/ml/circuits/"))}, "the script and what it imports, which reads no data"


def test_lean_build_outputs_the_executables_the_flock_suite_runs_at_their_paths_and_the_rest_of_its_build(doc):
    lb, agree = (next(t for t in doc["tasks"] if t["id"] == i) for i in (build.BUILD, AGREE))
    tests = tomllib.loads((ROOT / "verity" / "ml" / "flock" / "pyproject.toml").read_text())["tool"]["verity"]["tests"]
    bins = [a for a in tests["artifacts"] if a.startswith(f"{build.BIN}/")]
    assert bins and set(bins) <= {o["path"] for o in lb["outputs"]}
    assert agree["vars"]["verify"] == next(o["path"] for o in lb["outputs"] if o["name"] == "flock-verify")
    assert {o["name"]: o["path"] for o in lb["outputs"]} == {
        **{e: f"{build.LEAN}/{lean_build.TREE}/bin/{e}" for e in build.executables(ROOT, graph.tracked(ROOT))},
        lean_build.BUILD: f"{build.LEAN}/{lean_build.PACKED}"}, "where the runner's `place` finds each in a reader's work"


def test_the_runners_run_what_check_runs():
    assert (lean_build.ELAN, lean_build.LAKE_BUILD) == (CHECK.SUITES.ELAN, CHECK.SUITES.LAKE_BUILD)
    assert next(j for j in CHECK.JOBS if j.name == "flock-circuit-build").halts_on in flock_circuit_build.FAILS
    assert f'print(f"{lean_agreement.CANNOT} {{why}}"' in (ROOT / lean_agreement.CHECK).read_text()
    assert (ROOT / flock_circuit_build.SCRIPT).is_file() and (ROOT / lean_agreement.PIN).is_file()


def test_flock_rust_tests_and_lean_unit_cut_run_check_s_argv(doc, tmp_path, monkeypatch):
    """Each runs the argv `check.groups` builds for its step, at the cores check gives it, its work directory where check names
    its output directory, and the executable it reads where check finds it in the tree."""
    out = tmp_path / "out"
    check = {name: argv for g in CHECK.groups(out, 8) for name, argv, _ in g}
    out.mkdir(exist_ok=True)
    (out / graph.LOG).write_text("")
    by = {t["id"]: t for t in doc["tasks"]}
    seen: list[list[str]] = []

    def call(argv, work, cwd, env=None):
        assert cwd == ROOT
        seen.append(argv)
        return 1
    for m in (flock_rust_tests, lean_unit_cut):
        monkeypatch.setattr(m, "call", call)
    rust = check["flock-rust-tests"]
    cores = int(rust[1].partition("=")[2])
    flock_rust_tests.run({**by[RUST], "resources": {**by[RUST]["resources"], "cores": cores}}, out)
    assert seen.pop() == rust
    verify = out / by[UNIT]["vars"]["verify"]
    verify.parent.mkdir(parents=True, exist_ok=True)
    verify.write_bytes(b"fv")
    lean_unit_cut.run(by[UNIT], out)
    assert [str(ROOT / by[UNIT]["vars"]["verify"]) if a == str(verify) else a for a in seen.pop()] == check["lean-unit-cut"]


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
    assert [t["id"] for t in build.tasks(tmp_path, script)] == [CIRCUIT, RUST]
    lean = [f"{build.LEAN}/lakefile.toml", f"{build.LEAN}/V.lean"]
    assert [t["id"] for t in build.tasks(tmp_path, lean)] == [build.BUILD]
    assert [t["id"] for t in build.tasks(tmp_path, [*lean, f"{build.VERIFIER}/upstream.json", build.UNIT_CUT])] == [build.BUILD], \
        "the agreement and the partition check need flock-verify among the executables"
    (tmp_path / build.LEAN / "lakefile.toml").write_text('defaultTargets = ["flock-verify"]\n[[lean_exe]]\nname = "flock-verify"\n')
    (tmp_path / build.UNIT_CUT).parent.mkdir(parents=True)
    (tmp_path / build.UNIT_CUT).write_text("import json\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "toy"\n')
    got = {t["id"]: t for t in build.tasks(tmp_path, [*lean, build.UNIT_CUT, "pyproject.toml"])}
    assert list(got) == [build.BUILD, UNIT] and got[UNIT]["inputs"] == [build.UNIT_CUT]
    assert got[build.BUILD]["outputs"] == [{"name": "flock-verify", "path": f"{build.BIN}/flock-verify"},
                                           {"name": "build", "path": f"{build.LEAN}/.lake/build.tar.gz"}]


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
    for m in (lean_build, flock_circuit_build, lean_agreement, flock_rust_tests, lean_unit_cut):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))                        # no elan or cargo of the machine's
    monkeypatch.setenv("PATH", str(root / "bin"), prepend=os.pathsep)
    return root


LAKE = """#!/bin/sh
case "$1" in
  --version) {version} ;;
  build) mkdir -p .lake/build/bin .lake/build/lib/lean && printf 'fv' > .lake/build/bin/flock-verify &&
         printf 'h' > .lake/build/bin/flock-verify.hash && printf 'o' > .lake/build/lib/lean/V.olean && {build} ;;
esac
"""
LEAN_OUTPUTS = [{"name": "flock-verify", "path": "pkg/.lake/build/bin/flock-verify"}, {"name": "build", "path": "pkg/.lake/build.tar.gz"}]


def _members(archive: Path) -> dict[str, tuple[bytes, int]]:
    with tarfile.open(archive, "r:gz") as tar:
        return {m.name: (tar.extractfile(m).read(), m.mode) for m in tar}


@pytest.mark.parametrize(("version", "build_", "status"), [
    ("exit 0", "exit 0", "pass"), ("exit 0", "echo 'error: build failed'; exit 1", "fail"),
    ("exit 0", "kill -9 $$", "error"), ("echo 'error: could not download toolchain'; exit 1", "exit 0", "error")])
def test_lean_build(tmp_path, monkeypatch, version, build_, status):
    """A pass outputs each executable and the rest of the build packed, and records that its tree holds them, so a reader
    there places nothing."""
    root = _tree(tmp_path, monkeypatch, {"bin/lake": LAKE.format(version=version, build=build_), "pkg/lean-toolchain": "v4\n"})
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean", "outputs": LEAN_OUTPUTS})
    res = _execute(doc, tmp_path / "work", lean_build.run)
    assert res["status"] == status, res
    if status != "pass":
        assert res["outputs"] == {}
        return
    archive = tmp_path / "work" / "pkg" / lean_build.PACKED
    assert res["outputs"] == {"flock-verify": contract.digest(b"fv"), "build": contract.file_digest(archive)}
    assert _members(archive) == {"bin/flock-verify.hash": (b"h", 0o644), "lib/lean/V.olean": (b"o", 0o644)}, \
        "the build but the executables, which are outputs of their own"
    assert json.loads((root / "pkg" / lean_build.PLACED).read_text()) == res["outputs"]


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
if how in ("agree", "skip every set"):
    (out / "flock-agreement" / "ci").mkdir(parents=True)
    sets = [{"set": 0, "sessions": 3, "agree": 3}] * (how == "agree") + [{"set": 1, "skipped": "no upstream binary"}]
    (out / "flock-agreement" / "ci" / "sets.json").write_text(json.dumps({"seed": 1, "sets": sets}))
    sys.exit(0)
if how == "cannot":
    print("check: lean-agreement: error: cannot run here: this machine may use 8 GB", file=sys.stderr)
sys.exit(1)
"""


@pytest.mark.parametrize(("verify", "fetched", "status"), [
    ("agree", None, "pass"), ("disagree", None, "fail"), ("cannot", None, "error"), ("skip every set", None, "error"),
    ("agree", "other bytes", "error"), ("agree", "no fetch", "error")])
def test_lean_agreement(tmp_path, monkeypatch, verify, fetched, status):
    """A pass of check.py in which ci.py skipped every set compared nothing: no pass, as no skip is one."""
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
    if verify == "skip every set":
        assert "skipped every one of its 1 sets" in res["detail"]["error"], res


@pytest.mark.parametrize(("script", "status"), [
    ('[ "$CARGO_BUILD_JOBS $RUST_TEST_THREADS $1" = "3 3 test" ] || exit 7', "pass"),
    ("echo 'test result: FAILED. 41 passed; 1 failed; 0 ignored'; echo 'error: test failed, to rerun pass `-p flock-core`'; "
     "exit 101", "fail"),
    ("echo 'error: crates/flock-core/src/lib.rs: patch does not apply'; exit 1", "fail"),
    ("echo 'error: test failed, to rerun pass `-p flock-live --lib`'; "
     "echo 'process didnt exit successfully (signal: 9, SIGKILL: kill)'; exit 101", "error"),
    ("echo \"fatal: unable to access 'https://github.com/succinctlabs/flock/'\"; exit 128", "error"), ("kill -9 $$", "error")])
def test_flock_rust_tests(tmp_path, monkeypatch, script, status):
    """`check_build.sh test` at the task's cores; a test binary a signal killed is the machine's, though cargo says a test failed."""
    _tree(tmp_path, monkeypatch, {flock_circuit_build.SCRIPT: f"#!/bin/bash\n{script}\n"})
    doc = _graph({"id": RUST, "kind": "flock-rust-tests", "env": "rust", "resources": {"cores": 3, "mem_gb": 1, "disk_gb": 1}})
    assert _execute(doc, tmp_path / "work", flock_rust_tests.run)["status"] == status


UNIT_STANDIN = """import json, os, signal, sys
from pathlib import Path
lean, out = Path(sys.argv[sys.argv.index("--lean") + 1]), Path(sys.argv[sys.argv.index("--out") + 1])
assert os.access(lean, os.X_OK)
how = lean.read_text()
if how == "import":
    raise ImportError("cannot import name 'validate_unit_cut' from 'verity.ml.circuits.partition'")
if how == "oserror":
    raise OSError(8, "Exec format error")
if how == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
cuts, agree = {"agree": (4, 4), "some": (4, 3), "none": (4, 0), "no cut": (0, 0)}[how]
out.mkdir(parents=True)
(out / "result.json").write_text(json.dumps({"cuts": cuts, "agree": agree, "disagree": [str(out / "3.json")] * (cuts - agree)}))
sys.exit(1 if agree < cuts else 0)
"""


@pytest.mark.parametrize(("how", "status"), [("agree", "pass"), ("some", "fail"), ("import", "fail"), ("none", "error"),
                                             ("no cut", "error"), ("oserror", "error"), ("kill", "error")])
def test_lean_unit_cut(tmp_path, monkeypatch, how, status):
    """A disagreement on some cuts, or the script's own exception, is the tree's; a verifier that agreed on no cut, a run that
    compared none, the machine's exception and a kill are no result."""
    _tree(tmp_path, monkeypatch, {lean_unit_cut.SCRIPT: UNIT_STANDIN})
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean",
                  "outputs": [{"name": "flock-verify", "path": "pkg/bin/flock-verify"}]},
                 {"id": UNIT, "kind": "lean-unit-cut", "env": "py", "vars": {"verify": "pkg/bin/flock-verify"},
                  "reads": [{"task": "lean-build:pkg", "output": "flock-verify"}],
                  "outputs": [{"name": "result", "path": lean_unit_cut.RESULT}]})
    work = tmp_path / "work"
    (work / "pkg" / "bin").mkdir(parents=True)
    (work / "pkg" / "bin" / "flock-verify").write_text(how)
    res = _execute(doc, work, lean_unit_cut.run)
    assert res["status"] == status, res
    if status == "pass":
        assert res["detail"] == {"exit": 0, "cuts": 4, "agree": 4} and set(res["outputs"]) == {"result"}
    if how == "some":
        assert res["detail"]["disagree"] == ["3.json"]
    if how == "import":
        assert res["detail"]["why"].startswith("ImportError: cannot import name 'validate_unit_cut'"), res


def _build(root: Path, files: dict[str, bytes], when: int = 0) -> Path:
    """A package at ``root`` whose `.lake/build` holds ``files``, each touched at ``when``; executable when its name says so."""
    for name, data in files.items():
        f = root / lean_build.TREE / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        f.chmod(0o755 if "exe" in name else 0o644)
        os.utime(f, (when, when))
    return root


BUILT = {"lib/lean/V.olean": b"o", "lib/lean/V.trace": b'{"depHash": "1"}', "ir/V.c": b"c", "lib/exe-helper": b"x"}


def test_one_build_packs_to_one_archive_s_bytes_wherever_it_is(tmp_path):
    """Its files in path order, their bytes and whether each is executable: no path, owner, time or name of the archive's."""
    one, two = _build(tmp_path / "one" / "pkg", BUILT), _build(tmp_path / "two" / "deeper" / "pkg", BUILT, when=10 ** 9)
    lean_build.pack(one, tmp_path / "a.tar.gz", set())
    lean_build.pack(two, tmp_path / "other-name.tar.gz", set())
    assert (tmp_path / "a.tar.gz").read_bytes() == (tmp_path / "other-name.tar.gz").read_bytes()
    with tarfile.open(tmp_path / "a.tar.gz", "r:gz") as tar:
        assert [(m.name, m.mode, m.mtime, m.uid, m.uname) for m in tar] == [
            ("ir/V.c", 0o644, 0, 0, ""), ("lib/exe-helper", 0o755, 0, 0, ""), ("lib/lean/V.olean", 0o644, 0, 0, ""),
            ("lib/lean/V.trace", 0o644, 0, 0, "")]
    lean_build.pack(one, tmp_path / "less.tar.gz", {one / lean_build.TREE / "ir/V.c"})
    assert sorted(_members(tmp_path / "less.tar.gz")) == ["lib/exe-helper", "lib/lean/V.olean", "lib/lean/V.trace"]


def _stage(tmp_path: Path, exe: bytes, built: dict[str, bytes]) -> tuple[dict, Path]:
    """A reader of lean-build's executable `v` and `build`, and its work directory with them staged: bytes, no mode bits."""
    work = tmp_path / "work"
    (work / "pkg" / lean_build.TREE / "bin").mkdir(parents=True, exist_ok=True)
    (work / "pkg" / lean_build.TREE / "bin" / "v").write_bytes(exe)
    lean_build.pack(_build(tmp_path / "built" / "pkg", built), work / "pkg" / lean_build.PACKED, set())
    reads = [{"task": "lean-build:pkg", "output": o} for o in ("build", "v")]
    return {"id": "test-file:t/tests/test_v.py", "vars": {"suite": "t", "lean": "pkg"}, "reads": reads}, work


def _files(tree: Path) -> dict[str, tuple[bytes, bool]]:
    return {p.relative_to(tree).as_posix(): (p.read_bytes(), os.access(p, os.X_OK)) for p in tree.rglob("*") if p.is_file()}


def test_a_reader_s_tree_holds_what_lean_build_output_once_placed(tmp_path, monkeypatch):
    """`place` puts each executable at its path, executable, and the packed build back where it was built from, in a tree with
    none; a second reader of the same outputs writes nothing, and one of another build's outputs places them over the first's."""
    root = _tree(tmp_path, monkeypatch, {"pkg/lean-toolchain": "v4\n"})
    task, work = _stage(tmp_path, b"exe", BUILT)
    lean_build.place(task, work)
    tree = root / "pkg" / lean_build.TREE
    assert _files(tree) == {"bin/v": (b"exe", True), **{n: (b, "exe" in n) for n, b in BUILT.items()}}
    assert not [p for p in tree.rglob(".*")], "no file half written is left beside its place"
    (tree / "lib/lean/V.olean").write_bytes(b"rebuilt here")
    lean_build.place(task, work)
    assert (tree / "lib/lean/V.olean").read_bytes() == b"rebuilt here", "placed once a tree: the record names these outputs"
    task, work = _stage(tmp_path, b"exe2", {**BUILT, "lib/lean/W.olean": b"w"})
    lean_build.place(task, work)
    assert _files(tree) == {"bin/v": (b"exe2", True), **{n: (b, "exe" in n) for n, b in BUILT.items()}, "lib/lean/W.olean": (b"w", False)}


def test_a_reader_in_the_tree_lean_build_built_places_nothing(tmp_path, monkeypatch):
    """lean-build records what its tree holds, so its readers on that machine leave the build as Lake left it."""
    root = _tree(tmp_path, monkeypatch, {"bin/lake": LAKE.format(version="exit 0", build="exit 0"), "pkg/lean-toolchain": "v4\n"})
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean", "outputs": LEAN_OUTPUTS})
    assert _execute(doc, tmp_path / "lb", lean_build.run)["status"] == "pass"
    olean = root / "pkg" / lean_build.TREE / "lib" / "lean" / "V.olean"
    olean.write_bytes(b"as lake left it")
    reader = {"id": "test-file:t/tests/test_v.py", "vars": {"suite": "t", "lean": "pkg"},
              "reads": [{"task": "lean-build:pkg", "output": o["name"]} for o in LEAN_OUTPUTS]}
    lean_build.place(reader, tmp_path / "lb")
    assert olean.read_bytes() == b"as lake left it"


def test_an_archive_member_that_is_no_file_under_the_build_is_refused(tmp_path):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb") as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    (tmp_path / "bad.tar.gz").write_bytes(raw.getvalue())
    with pytest.raises(ValueError, match="no file of a build directory"):
        lean_build.unpack(tmp_path / "bad.tar.gz", tmp_path / "pkg")
    assert not (tmp_path / "outside").exists()
