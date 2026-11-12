"""The build family: on this tree its tasks are check's steps (the inputs check keys them by, the executables the Flock suite
runs, what check.py reads to run the agreement, the suites check runs after lean-build) and its runners run check's commands,
telling a fail, the tree's, from an error, the machine's (a skip among them: nothing ran), on stand-ins for lake,
check_build.sh, check.py, unit_cut_agree.py, suites.py and the store."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import shlex
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import build
from verity_graph.runners import flock_circuit_build, flock_rust_tests, lean_agreement, lean_build, lean_suites, lean_unit_cut

spec = importlib.util.spec_from_file_location("graph_check", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

AGREE, BUILD, CIRCUIT = (f"lean-agreement:{build.VERIFIER}", f"lean-build:{build.LEAN}", f"flock-circuit-build:{build.FLOCK}/live")
RUST, UNIT, SUITES = f"flock-rust-tests:{build.FLOCK}/live", f"lean-unit-cut:{build.UNIT_CUT}", "lean-suites:a"
FLOCK_PART = "lean-suites:verity-flock:1/6"


@pytest.fixture(scope="module")
def doc() -> dict:
    return graph.listing(ROOT, {"build"})


def _own(task: dict) -> list[str]:
    return [p for p in task["inputs"] if p not in graph.runner_sources(task["kind"])]


def _suites(doc: dict) -> list[dict]:
    return [t for t in doc["tasks"] if t["kind"] == "lean-suites"]


def test_the_build_family_is_check_s_steps_with_the_inputs_check_keys_them_by(doc):
    by = {t["id"]: t for t in doc["tasks"]}
    assert sorted(set(by) - {t["id"] for t in _suites(doc)}) == sorted([CIRCUIT, RUST, AGREE, BUILD, UNIT])
    files = graph.tracked(ROOT)
    for rust in (CIRCUIT, RUST):
        assert _own(by[rust]) == CHECK.rust_tests_inputs(files) and by[rust]["env"] == "rust" and by[rust]["key"]
    verifier = [p for p in _own(by[AGREE]) if p.startswith(build.VERIFIER + "/")]
    assert sorted({*_own(by[BUILD]), *verifier}) == sorted(CHECK.agreement_closure(files)), \
        "the agreement's key in check covers the verifier's package (the build's inputs) and its own directory"
    verify = [{"task": BUILD, "output": "flock-verify"}]
    assert by[AGREE]["reads"] == by[UNIT]["reads"] == verify and not by[AGREE]["key"] and by[BUILD]["key"]
    assert {build.UNIT_CUT, "verity/circuits/partition.py"} <= set(_own(by[UNIT])) <= {
        build.UNIT_CUT, *(f for f in files if f.startswith("verity/") and f.endswith("/__init__.py")),
        *(f for f in files if f.startswith("verity/circuits/"))}, "the script and what it imports, which reads no data"
    for t in _suites(doc):
        assert t["inputs"] == files, f"{t['id']}: check's lean-suites, keyed by the whole tree"
        assert t["reads"] == [{"task": BUILD, "output": o["name"]} for o in by[BUILD]["outputs"]] and not t["key"]
        assert {t["vars"]["bin"] + "/" + r["output"] if r["output"] != lean_build.TREE else t["vars"]["tree"]
                for r in t["reads"]} == {o["path"] for o in by[BUILD]["outputs"]}


def test_lean_suites_read_the_whole_build_the_flock_suite_s_tests_build_on(doc):
    """check's lean-suites step runs on the tree its lean-build step built, every module of the verifier and of the packages it
    requires built, so a test's `lake build` of an executable links what is there.  A lean-suites task reads that whole build
    (lean-build's `TREE`, every one of those packages), not the executables alone: on a tree holding only those, each test
    process built the verifier at once in one `.lake`, and the links failed on objects missing their symbols."""
    lb = next(t for t in doc["tasks"] if t["id"] == BUILD)
    assert lb["vars"]["builds"].split() == [build.LEAN, *(d.rstrip("/") for d in CHECK.required_in_tree(build.LEAN + "/"))]
    assert {"name": lean_build.TREE, "path": build.TREE_PATH} in lb["outputs"]
    assert all({"task": BUILD, "output": lean_build.TREE} in t["reads"] and t["vars"]["tree"] == build.TREE_PATH
               for t in _suites(doc))


def test_lean_suites_is_check_s_lean_suites_step_a_suite_or_a_part_a_task(doc):
    """Each suite check's lean-suites step runs is one task, or one a part: of the parts by test id it declares, else of
    `build.PARTS`, in check's order: every suite once, every part of a split one once; a parallel suite takes a slot and
    another one process, each at its own timeout."""
    suites = CHECK.SUITES.discover()
    split = {**build.PARTS, **{k: s.parts for s in suites if s.parts for k in (s.name, s.path)}}
    assert not {s.name for s in suites if s.parts} & set(build.PARTS), "a suite in parts by test id isn't in PARTS too"
    ran = [(t["vars"]["suites"], t["vars"].get("files")) for t in sorted(_suites(doc), key=lambda t: t["id"])]
    want = [(s, f"{k}/{split[s]}" if s in split else None) for s in CHECK.SUITES.AFTER_LEAN for k in range(1, split.get(s, 1) + 1)]
    assert sorted(ran, key=str) == sorted(want, key=str) and len(ran) == len(set(ran))
    assert FLOCK_PART in {t["id"] for t in _suites(doc)}, "the Flock suite, which runs for hours on one slot, is in parts"
    fam = graph.spec(ROOT)["family"]["build"]
    for t in _suites(doc):
        s = next(x for x in suites if t["vars"]["suites"] in (x.name, x.path))
        one = {"cores": 1, "mem_gb": s.worker_gb or CHECK.SUITES.TEST_PROCESS_GB, "disk_gb": build.SUITES_DISK_GB}
        assert t["resources"] == ({**fam["resources"], "disk_gb": build.SUITES_DISK_GB} if s.parallel else one), t["id"]
        assert t["timeout_s"] == (math.ceil(s.timeout_s) if s.timeout_s else fam["timeout_s"]), t["id"]


def test_a_part_s_files_are_every_mth_by_path_whatever_this_machine_measured():
    """The runner hands suites.py a timing record with none of a split suite's files (`lean_suites.TIMINGS`), so the partition
    packs equal costs by path: part K holds files K, K+M, ..., the same on every machine, and the M parts hold each file once."""
    files = [f"s/tests/test_{c}.py" for c in "kjihgfedcba"]
    parts = CHECK.SUITES.partition(files, {}, 4)
    assert parts == [sorted(files)[k::4] for k in range(4)]
    assert sorted(f for p in parts for f in p) == sorted(files)


def test_lean_build_outputs_the_executables_the_flock_suite_runs_at_their_paths(doc):
    lb, agree = (next(t for t in doc["tasks"] if t["id"] == i) for i in (BUILD, AGREE))
    tests = tomllib.loads((ROOT / "verity" / "protocols" / "flock" / "prover" / "pyproject.toml").read_text())["tool"]["verity"]["tests"]
    bins = [a for a in tests["artifacts"] if a.startswith(f"{build.LEAN}/.lake/build/bin/")]
    assert bins and set(bins) <= {o["path"] for o in lb["outputs"]}
    assert agree["vars"]["verify"] == next(o["path"] for o in lb["outputs"] if o["name"] == "flock-verify")


def test_the_runners_run_what_check_runs():
    assert (lean_build.ELAN, lean_build.LAKE_BUILD) == (CHECK.SUITES.ELAN, CHECK.SUITES.LAKE_BUILD)
    assert next(j for j in CHECK.JOBS if j.name == "flock-circuit-build").halts_on in flock_circuit_build.FAILS
    assert f'print(f"{lean_agreement.CANNOT} {{why}}"' in (ROOT / lean_agreement.CHECK).read_text()
    assert (ROOT / flock_circuit_build.SCRIPT).is_file() and (ROOT / lean_agreement.PIN).is_file()
    assert f'print(f"\\n{lean_suites.WROTE} ' in (ROOT / lean_suites.SUITES).read_text()


def test_flock_rust_tests_lean_unit_cut_and_lean_suites_run_check_s_argv(doc, tmp_path, monkeypatch):
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
    for m in (flock_rust_tests, lean_unit_cut, lean_suites):
        monkeypatch.setattr(m, "call", call)
    monkeypatch.setattr(lean_suites, "place", lambda task, work: None)          # no executable put in this tree
    rust = check["flock-rust-tests"]
    cores = int(rust[1].partition("=")[2])
    flock_rust_tests.run({**by[RUST], "resources": {**by[RUST]["resources"], "cores": cores}}, out)
    assert seen.pop() == rust
    verify = out / by[UNIT]["vars"]["verify"]
    verify.parent.mkdir(parents=True, exist_ok=True)
    verify.write_bytes(b"fv")
    lean_unit_cut.run(by[UNIT], out)
    assert [str(ROOT / by[UNIT]["vars"]["verify"]) if a == str(verify) else a for a in seen.pop()] == check["lean-unit-cut"]
    words = shlex.split(check["lean-suites"][2].removeprefix(lean_build.ELAN))
    n = len(CHECK.SUITES.AFTER_LEAN)
    assert words[2:2 + n] == list(CHECK.SUITES.AFTER_LEAN)
    jobs = words.index("--jobs") + 2
    whole = next(t["id"] for t in _suites(doc) if "files" not in t["vars"])
    for t in (by[whole], by[FLOCK_PART]):
        lean_suites.run({**t, "resources": {**t["resources"], "cores": int(words[jobs - 1])}}, out)
        (bash, c, mine), = seen
        seen.clear()
        assert [bash, c] == check["lean-suites"][:2] and mine.startswith(lean_build.ELAN)
        part = ["--files", t["vars"]["files"], "--timings", str(out / lean_suites.TIMINGS)] if "files" in t["vars"] else []
        assert shlex.split(mine.removeprefix(lean_build.ELAN)) == [
            *words[:2], t["vars"]["suites"], *words[2 + n:jobs], "--mem-gb", str(t["resources"]["mem_gb"]), *part], \
            "check's words to its --jobs on the task's suite; then the task's memory, where check gives its slot's share, and its part"
    assert json.loads((out / lean_suites.TIMINGS).read_text()) == {"verity-flock": {}}


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


def test_a_tree_without_the_verifier_lists_none_of_its_lean_tasks(tmp_path, monkeypatch):
    (tmp_path / build.LEAN).mkdir(parents=True)
    (tmp_path / build.LEAN / "lakefile.toml").write_text('defaultTargets = ["v"]\n[[lean_exe]]\nname = "v"\n')
    assert build.tasks(tmp_path, []) == []
    script = [f"{build.FLOCK}/check_build.sh"]
    assert [t["id"] for t in build.tasks(tmp_path, script)] == [CIRCUIT, RUST]
    lean = [f"{build.LEAN}/lakefile.toml", f"{build.LEAN}/V.lean"]
    assert [t["id"] for t in build.tasks(tmp_path, lean)] == [BUILD]
    assert [t["id"] for t in build.tasks(tmp_path, [*lean, f"{build.VERIFIER}/upstream.json", build.UNIT_CUT])] == [BUILD], \
        "the agreement and the partition check need flock-verify among the executables"
    (tmp_path / build.LEAN / "lakefile.toml").write_text('defaultTargets = ["flock-verify"]\n[[lean_exe]]\nname = "flock-verify"\n')
    pytest_ = "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"
    for f, text in ((build.UNIT_CUT, "import json\n"),
                    (build.SUITES, 'AFTER_LEAN = ("a", "b/c", "d", "gone")\nTEST_PROCESS_GB = 9\n'),
                    ("pyproject.toml", '[project]\nname = "toy"\n'),
                    ("pkg/a/pyproject.toml", f'[project]\nname = "a"\n{pytest_}[tool.verity.tests]\nworker_gb = 3\n'),
                    ("b/c/pyproject.toml", f"{pytest_}[tool.verity.tests]\nparallel = true\ntimeout_s = 99.5\n"),
                    ("pkg/d/pyproject.toml", f'[project]\nname = "d"\n{pytest_}[tool.verity.tests]\nparts = 3\n')):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    files = [*lean, build.UNIT_CUT, build.SUITES, "b/c/pyproject.toml", "pkg/a/pyproject.toml", "pkg/d/pyproject.toml",
             "pyproject.toml"]
    monkeypatch.setattr(build, "PARTS", {"a": 2, "d": 2})
    got = {t["id"]: t for t in build.tasks(tmp_path, files)}
    parts, by_id = ["lean-suites:a:1/2", "lean-suites:a:2/2"], [f"lean-suites:d:{k}/3" for k in (1, 2, 3)]
    assert list(got) == [BUILD, UNIT, *parts, "lean-suites:b/c", *by_id, "lean-suites:gone"]
    assert got[UNIT]["inputs"] == [build.UNIT_CUT]
    assert [got[p]["vars"] for p in parts] == [{"suites": "a", "files": f"{k}/2", "bin": f"{build.LEAN}/.lake/build/bin",
                                                "tree": build.TREE_PATH} for k in (1, 2)]
    assert [got[p]["vars"]["files"] for p in by_id] == ["1/3", "2/3", "3/3"] and all("by id" in got[p]["about"] for p in by_id), \
        "a suite that declares parts is in those parts by test id, as check runs it, whatever PARTS says"
    assert all(got[p]["resources"] == {"cores": 1, "mem_gb": 3, "disk_gb": 60} and "timeout_s" not in got[p] for p in parts), \
        "one process at the suite's worker_gb, at the family's timeout"
    assert got["lean-suites:b/c"]["resources"] == {"disk_gb": 60} and got["lean-suites:b/c"]["timeout_s"] == 100, \
        "a parallel suite takes the family's slot, at its own timeout"
    assert got["lean-suites:gone"]["resources"] == {"cores": 1, "mem_gb": 9, "disk_gb": 60}, \
        "a name no suite has is listed, so suites.py's refusal of it is an error"
    assert all(t["inputs"] == files for i, t in got.items() if i.startswith("lean-suites:"))
    (tmp_path / build.SUITES).write_text("AFTER = ()\n")
    assert not any(t["id"].startswith("lean-suites:") for t in build.tasks(tmp_path, [*lean, build.SUITES])), \
        "no AFTER_LEAN, no lean-suites"


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
    for m in (lean_build, flock_circuit_build, lean_agreement, flock_rust_tests, lean_unit_cut, lean_suites):
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


#: a build of `pkg` and of `dep`, which it requires by path: each `.lake/build` with a module's olean, trace and object, and pkg's
#: executable with its trace
BUILT = {f"{p}/.lake/build/{f}": f"{p} {f}" for p in ("pkg", "dep") for f in ("lib/lean/M.olean", "lib/lean/M.trace",
                                                                              "ir/M.c.o.export")}
BUILT["pkg/.lake/build/bin/flock-verify.trace"] = "trace"


def _built(root: Path) -> dict[str, str]:
    return {f.relative_to(root).as_posix(): f.read_text() for f in sorted(root.rglob("*")) if f.is_file()
            and "/.lake/build/" in f"/{f.relative_to(root).as_posix()}"}


def test_lean_build_packs_its_whole_build_but_the_executables(tmp_path, monkeypatch):
    """Its archive holds every file of each named package's `.lake/build`, the executables it outputs apart, and the package's
    stamp names it, so a reader in this tree, which holds the build, doesn't unpack it again."""
    make = "; ".join(f"mkdir -p {Path('..') / Path(p).parent} && printf '{t}' > {Path('..') / p}" for p, t in BUILT.items())
    root = _tree(tmp_path, monkeypatch, {"bin/lake": LAKE.format(version="exit 0", build=make), "pkg/lean-toolchain": "v4\n"})
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean", "vars": {"builds": "pkg dep"},
                  "outputs": [{"name": "flock-verify", "path": "pkg/.lake/build/bin/flock-verify"},
                              {"name": lean_build.TREE, "path": build.TREE_PATH}]})
    res = _execute(doc, tmp_path / "work", lean_build.run)
    assert res["status"] == "pass" and set(res["outputs"]) == {"flock-verify", lean_build.TREE}, res
    with tarfile.open(tmp_path / "work" / build.TREE_PATH) as tf:
        packed = {m.name: tf.extractfile(m).read().decode() for m in tf}
    assert packed == BUILT and _built(root) == {**BUILT, "pkg/.lake/build/bin/flock-verify": "fv"}
    assert (root / "pkg" / ".lake" / lean_build.STAMP).read_text() == res["outputs"][lean_build.TREE]


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
    raise ImportError("cannot import name 'validate_unit_cut' from 'verity.circuits.partition'")
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


SUITES_STANDIN = """import json, os, shutil, signal, sys
from pathlib import Path
a = sys.argv[1:]
names = a[:a.index("--lean-built")]
assert shutil.which("lake") and a[a.index("--jobs") + 1] == "3" and a[a.index("--mem-gb") + 1] == "5" and "--timeouts" in a
fv = Path(__file__).resolve().parents[3] / "pkg" / "bin" / "flock-verify"
assert os.access(fv, os.X_OK) and fv.read_text() in ("fv", "fv-old"), "the executable read, at its path in the tree"
how = Path(os.environ["HOW"]).read_text()
if how == "part":
    timings = json.loads(Path(a[a.index("--timings") + 1]).read_text())
    assert a[a.index("--files") + 1] == "2/4" and timings == {n: {} for n in names}, (a, timings)
else:
    assert "--files" not in a and "--timings" not in a
if how == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
if how == "usage":
    sys.exit(2)
rows = [{"suite": "lean/" + n, "name": n, "status": "passed", "failed_tests": [],
         **({"part": {"ks": [2], "m": 4}} if how == "part" else {})} for n in names]
if how == "missing":
    rows.pop()
if how == "cached":
    rows[0]["status"] = "cached"
if how == "tests":
    rows[1].update(status="failed", failed_tests=["tests/test_v.py::test_v"])
if how == "kill a suite":
    rows[1].update(status="failed")
if how == "timeout":
    rows[1].update(status="failed", timed_out=600.0)
Path(a[a.index("--json") + 1]).write_text(json.dumps(rows))
if how == "wrote":
    print("\\nthe tests changed 1 files in the repository (tests write under tmp_path):\\n  x.txt")
sys.exit(1 if how in ("tests", "kill a suite", "timeout", "wrote") else 0)
"""


@pytest.mark.parametrize(("how", "status"), [("pass", "pass"), ("cached", "pass"), ("tests", "fail"), ("timeout", "fail"),
                                             ("wrote", "fail"), ("kill a suite", "error"), ("missing", "error"),
                                             ("usage", "error"), ("kill", "error"), ("pass old", "pass"), ("part", "pass")])
def test_lean_suites(tmp_path, monkeypatch, how, status):
    """suites.py --lean-built on the task's suites, with lake on the PATH, after the executables it reads are put at their paths
    in the tree (over another build's there); a suite that failed is named with its failed tests, and a suite the report lacks,
    one that failed on nothing its row names (a kill) and a usage error are no result.  A part's task runs `--files K/M` on a
    timing record that holds none of its suites' files, and its report's rows, of the suite's part, are the suite's."""
    root = _tree(tmp_path, monkeypatch, {lean_suites.SUITES: SUITES_STANDIN, "bin/lake": "#!/bin/sh\n",
                                         **({"pkg/bin/flock-verify": "fv-old"} if how == "pass old" else {})})
    (tmp_path / "how").write_text(how.removesuffix(" old"))
    monkeypatch.setenv("HOW", str(tmp_path / "how"))
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean",
                  "outputs": [{"name": "flock-verify", "path": "pkg/bin/flock-verify"}]},
                 {"id": SUITES, "kind": "lean-suites", "env": "py",
                  "vars": {"suites": "a b/c", "bin": "pkg/bin", **({"files": "2/4"} if how == "part" else {})},
                  "reads": [{"task": "lean-build:pkg", "output": "flock-verify"}],
                  "outputs": [{"name": "report", "path": lean_suites.REPORT}], "resources": {"cores": 3, "mem_gb": 5, "disk_gb": 1}})
    work = tmp_path / "work"
    (work / "pkg" / "bin").mkdir(parents=True)
    (work / "pkg" / "bin" / "flock-verify").write_text("fv")
    res = _execute(doc, work, lean_suites.run)
    assert res["status"] == status, res
    assert (root / "pkg" / "bin" / "flock-verify").read_text() == "fv" and os.access(root / "pkg" / "bin" / "flock-verify", os.X_OK)
    if how == "tests":
        assert res["detail"]["failed"] == [{"name": "b/c", "suite": "lean/b/c", "failed_tests": ["tests/test_v.py::test_v"]}]
        assert res["detail"]["suites"] == {"a": "passed", "b/c": "failed"}
    if how == "missing":
        assert res["detail"]["error"] == "not run: b/c", res


STAGED_STANDIN = """import json, os, sys
from pathlib import Path
a = sys.argv[1:]
root = Path(__file__).resolve().parents[3]
built = {p: (root / p).read_text() for p in json.loads(os.environ["BUILT"]) if (root / p).is_file()}
assert built == json.loads(os.environ["BUILT"]), ("the tests' lake builds find the whole build in the tree", built)
assert os.access(root / "pkg/.lake/build/bin/flock-verify", os.X_OK)
Path(a[a.index("--json") + 1]).write_text(json.dumps([{"suite": "lean/a", "name": "a", "status": "passed", "failed_tests": []}]))
"""


@pytest.mark.parametrize("lake", ["ok", "fails"])
def test_lean_suites_stages_lean_builds_whole_build_in_a_tree_without_it(tmp_path, monkeypatch, lake):
    """On a tree that holds none of lean-build's build (a fresh checkout, as an executor's run gets, lean-build having run in
    another), the task unpacks the build it reads into the tree and runs check's `lake build` there before the suites, once a
    tree: a second task finds the stamp and stages nothing.  A build that doesn't build here, though lean-build passed it, is
    an error."""
    calls = tmp_path / "lake-calls"
    root = _tree(tmp_path, monkeypatch, {lean_suites.SUITES: STAGED_STANDIN,
                                         "bin/lake": f"#!/bin/sh\necho \"$PWD $*\" >> {calls}\n{'exit 3' * (lake == 'fails')}\n"})
    monkeypatch.setenv("BUILT", json.dumps(BUILT))
    made = tmp_path / "made"
    for p, text in BUILT.items():
        (made / p).parent.mkdir(parents=True, exist_ok=True)
        (made / p).write_text(text)
    monkeypatch.setattr(lean_build, "ROOT", made)
    bin_ = "pkg/.lake/build/bin"
    doc = _graph({"id": "lean-build:pkg", "kind": "lean-build", "env": "lean",
                  "outputs": [{"name": "flock-verify", "path": f"{bin_}/flock-verify"},
                              {"name": lean_build.TREE, "path": build.TREE_PATH}]},
                 {"id": SUITES, "kind": "lean-suites", "env": "py", "vars": {"suites": "a", "bin": bin_, "tree": build.TREE_PATH},
                  "reads": [{"task": "lean-build:pkg", "output": "flock-verify"}, {"task": "lean-build:pkg", "output": lean_build.TREE}],
                  "outputs": [{"name": "report", "path": lean_suites.REPORT}], "resources": {"cores": 3, "mem_gb": 5, "disk_gb": 1}})
    archive = tmp_path / "archive"
    lean_build.pack(["pkg", "dep"], set(), archive)
    for n in (1, 2):
        work = tmp_path / f"work{n}"
        (work / bin_).mkdir(parents=True)
        (work / bin_ / "flock-verify").write_text("fv")
        (work / build.TREE_PATH).parent.mkdir(parents=True)
        (work / build.TREE_PATH).write_bytes(archive.read_bytes())
        res = _execute(doc, work, lean_suites.run)
        if lake == "fails":
            assert res["status"] == "error" and "on the build lean-build passed, unpacked here, exited 3" in res["detail"]["error"]
            assert not (root / "pkg" / ".lake" / lean_build.STAMP).exists()
            continue
        assert res["status"] == "pass", res
        assert _built(root) == {**BUILT, f"{bin_}/flock-verify": "fv"}
        assert calls.read_text().splitlines() == [f"{root / 'pkg'} build"], "check's lake build, once a tree"
        assert (root / "pkg" / ".lake" / lean_build.STAMP).read_text() == contract.file_digest(archive)


def test_lean_suites_refuses_an_archive_member_outside_a_build(tmp_path):
    """The archive writes only under a package's `.lake/build`: a member elsewhere is refused, and not written."""
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name in ("pkg/.lake/build/lib/M.olean", "pkg/lakefile.toml"):
            info = tarfile.TarInfo(name)
            tf.addfile(info, io.BytesIO(b""))
    with pytest.raises(ValueError, match="pkg/lakefile.toml is no file under a package's .lake/build"):
        lean_suites.unpack(archive, tmp_path / "root")
    assert not (tmp_path / "root" / "pkg" / "lakefile.toml").exists()
