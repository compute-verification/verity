"""The circuit-binding family: its listing is circuit-check's catalog bindings, read from the binding files' syntax without
running them, the same bytes on every listing; its runners tell a pass, a fail and an error apart from circuit-check's exit and
report (here a stand-in's), and the records tarball is the records' bytes alone."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import git, toy_tree

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import circuit_binding as F
from verity_graph.runners import circuit_binding as CB
from verity_graph.runners import circuit_catalog, circuit_coverage

NEVER_RUN = 'raise RuntimeError("a binding file is read, never run")\n'
SIDECAR = NEVER_RUN + '''from verity.ml.common import _shared_roots
import verity.ml.y as _y_alias
def _b_roots():
    return []
async def _a_roots():
    return []
_c_roots = _b_roots
def _b_bindings():
    return []
def _model_roots():
    return []
TEMPLATE_PARAMS = {"t-two": {}, "t-one": {}}
'''
TARGETS = NEVER_RUN + '''def _kernel_roots(): ...
def _core_roots(): ...
def _registered(): ...
TEMPLATE_PARAMS = {"rope-head": {}}
'''
TOY = {f"{F.CHECKER}/pyproject.toml": '[tool.verity.tests]\ninputs = ["verity", "uv.lock"]\n',
       f"{F.CHECKER}/README.md": "prose\n", f"{F.CHECKER}/tests/test_x.py": "", f"{F.SOURCE}/__init__.py": "",
       F.SIDECARS: NEVER_RUN + 'PACKAGES = ("verity", "verity_flock.sub")\n', F.TARGETS: TARGETS,
       "verity/__init__.py": "", "verity/ml/__init__.py": "", "verity/ml/x.py": "", "verity/ml/x.circuit-check.py": SIDECAR,
       "verity/ml/x.circuit-check.json": "{}\n", "verity/ml/pkg/__init__.py": "",
       "verity/ml/pkg/__init__.circuit-check.py": NEVER_RUN + "def _p_roots(): ...\n",
       "verity/ns/y.circuit-check.py": NEVER_RUN + "def _y_roots(): ...\n",
       "verity/ml/flock/python/verity_flock/__init__.py": "",
       "verity/ml/flock/python/verity_flock/rec.circuit-check.py": NEVER_RUN + "def _rec_roots(): ...\n",
       "other/__init__.py": "", "other/z.circuit-check.py": NEVER_RUN + "def _z_roots(): ...\n",
       **{f"{graph.RUNNERS}/{m}.py": "" for m in ("circuit_binding", "circuit_catalog", "circuit_coverage")}}
X, FLOCK = "verity/ml/x.circuit-check.py", "verity/ml/flock/python/verity_flock/rec.circuit-check.py"
BINDINGS = {"<registered>": (F.TARGETS, "_registered"), "TEMPLATE_PARAMS": (F.TARGETS, "TEMPLATE_PARAMS"),
            "_core_roots": (F.TARGETS, "_core_roots"), "_kernel_roots": (F.TARGETS, "_kernel_roots"),
            **{f"verity.ml.x.circuit-check#{e}": (X, e) for e in (
                "_a_roots", "_b_roots", "_c_roots", "_shared_roots", "_model_roots", "TEMPLATE_PARAMS:t-two",
                "TEMPLATE_PARAMS:t-one")},
            "verity.ml.pkg.circuit-check#_p_roots": ("verity/ml/pkg/__init__.circuit-check.py", "_p_roots"),
            "verity_flock.rec.circuit-check#_rec_roots": (FLOCK, "_rec_roots")}
LISTERS = {"circuit-binding": F.tasks}


def _toy(root: Path, **files: str) -> Path:
    return toy_tree(root, (ROOT / graph.SPEC).read_text(), {**TOY, **files})


def test_each_binding_is_a_task_named_as_circuit_check_names_it_from_the_files_syntax_alone(tmp_path):
    root = _toy(tmp_path / "tree")
    files = graph.tracked(root)
    assert F.bindings(root, files) == dict(sorted(BINDINGS.items()))
    doc = graph.listing(root, {"circuit-binding"}, LISTERS)
    by = {t["id"]: t for t in doc["tasks"]}
    reads = [{"task": F.CATALOG, "output": "records"}]
    assert sorted(by) == sorted([F.CATALOG, "circuit-coverage:catalog", *(f"circuit-binding:{f}:{e}" for f, e in BINDINGS.values())])
    base = sorted(f for f in files if f.startswith(("verity/", f"{F.CHECKER}/")) and not f.startswith(f"{F.CHECKER}/tests/")
                  and not f.endswith(".md") or f == "uv.lock")
    for name, (f, e) in BINDINGS.items():
        t = by[f"circuit-binding:{f}:{e}"]
        assert t["vars"] == {"binding": name, "records": F.RECORDS["path"]} and t["reads"] == reads and t["key"] is None
        assert t["inputs"] == sorted({*base, *graph.runner_sources("circuit-binding")}), name
        assert t["about"] == f"circuit-check --binding {name}"
    cat, cov = by[F.CATALOG], by["circuit-coverage:catalog"]
    assert cat["outputs"] == [F.RECORDS] and cat["reads"] == [] and cat["key"] and cov["reads"] == reads
    assert json.loads(cov["vars"]["bindings"]) == sorted(BINDINGS)
    assert f"{graph.RUNNERS}/circuit_binding.py" in cat["inputs"] and f"{graph.RUNNERS}/circuit_binding.py" in cov["inputs"]


def test_the_listing_is_the_same_bytes_in_another_checkout_of_the_commit(tmp_path):
    root = _toy(tmp_path / "a")
    other = tmp_path / "elsewhere" / "b"
    other.parent.mkdir()
    git(tmp_path, "clone", "-q", str(root), str(other))
    first = contract.canonical(graph.listing(root, {"circuit-binding"}, LISTERS))
    assert contract.canonical(graph.listing(root, {"circuit-binding"}, LISTERS)) == first
    assert contract.canonical(graph.listing(other, {"circuit-binding"}, LISTERS)) == first


@pytest.mark.parametrize(("files", "error"), [
    ({X: "TEMPLATE_PARAMS = dict(t={})\n"}, "TEMPLATE_PARAMS is a dict literal"),
    ({X: "TEMPLATE_PARAMS = {KEY: {}}\n"}, "TEMPLATE_PARAMS is a dict literal"),
    ({f"{F.CHECKER}/pyproject.toml": '[tool.verity.tests]\ninputs = ["uv.lock"]\n'}, "outside what circuit-check's suite"),
    ({f"{F.CHECKER}/pyproject.toml": "[project]\n"}, "no [tool.verity.tests] inputs"),
    ({F.SIDECARS: "ROOTS = ()\n"}, "no PACKAGES tuple")])
def test_a_tree_the_listing_can_t_read_bindings_from_is_refused_by_name(tmp_path, files, error):
    root = _toy(tmp_path / "tree", **files)
    with pytest.raises(graph.GraphError, match=error.replace("[", r"\[")):
        F.tasks(root, graph.tracked(root))


def test_on_the_tree_the_binding_files_are_the_sidecars_circuit_check_discovers_and_none_is_imported():
    probe = ("import json, sys\nfrom verity_graph import ROOT, graph\nfrom verity_graph.families import circuit_binding as F\n"
             "found = F.bindings(ROOT, graph.tracked(ROOT))\n"
             "print(json.dumps({'found': sorted({n.partition('#')[0] for n in found if '#' in n}),\n"
             "                  'imported': sorted({m.partition('.')[0] for m in sys.modules} & {'circuit_check', 'verity', "
             "'verity_flock', 'verity_numerical', 'verity_vllm'})}))\n")
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert got["imported"] == [], "the listing imported repository code"
    from circuit_check import sidecars

    files = {f"{m}{F.SUFFIX}": kinds["py"] for m, kinds in sidecars.discover().items() if "py" in kinds}
    assert set(got["found"]) <= set(files)
    unbound = {m: F.entries(str(p), p.read_text()) for m, p in files.items() if m not in got["found"]}
    assert unbound and not any(unbound.values()), "a sidecar the listing skipped binds something"


def test_the_runners_say_what_circuit_check_and_check_say():
    assert f'NO_BINDING = "{CB.NO_BINDING}"' in (ROOT / F.SOURCE / "cli.py").read_text()
    check = (ROOT / "tools" / "verity" / "check" / "check.py").read_text()
    assert f'CC_RECORDS = "{F.RECORDS["path"]}"' in check and f't.add(d / "cache", "{CB.TOP}")' in check


# --- the runners, on a stand-in for circuit-check --------------------------------------------------------------------------

def _graph(*tasks: dict) -> dict:
    full = [{"about": "x", "env": "py", "vars": {}, "inputs": [], "reads": [], "outputs": [],
             "argv": ["verity", "graph", "run", t["id"]], "resources": {"cores": 3, "mem_gb": 5, "disk_gb": 1},
             "timeout_s": 60, "key": None, **t} for t in tasks]
    return {"envs": {"py": {"digest": contract.digest(b"py")}}, "files": {}, "tasks": full}


CATALOG = {"id": F.CATALOG, "kind": "circuit-catalog", "outputs": [F.RECORDS]}
READS = [{"task": F.CATALOG, "output": "records"}]


def _stand_in(monkeypatch, module, rc: int, report: dict | None, say: str = "", records: dict | None = None) -> list:
    """circuit-check as ``module`` calls it: each call's argv and the cache's records then, appended to the list returned; it
    writes ``records`` into --cache, ``report`` to --out and ``say`` to the log, and exits ``rc``."""
    calls = []

    def call(argv, work, cwd, env=None):
        cache = Path(argv[argv.index("--cache") + 1])
        calls.append({"argv": argv, "cwd": cwd, "env": env, "cache": sorted(
            p.relative_to(cache).as_posix() for p in cache.rglob("*.json")) if cache.is_dir() else None})
        for rel, body in (records or {}).items():
            (cache / rel).parent.mkdir(parents=True, exist_ok=True)
            (cache / rel).write_text(body)
        if report is not None:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(report))
        with open(work / graph.LOG, "a") as log:
            log.write(say)
        return rc
    monkeypatch.setattr(module, "call", call)
    return calls


def _records(d: Path, records: dict[str, str]) -> Path:
    for rel, body in records.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(body)
    return d


def test_pack_is_the_records_bytes_alone_and_unpack_puts_each_back(tmp_path):
    records = {"v1/aa/one.json": '{"a": 1}', "v1/bb/two.json": '{"b": 2}', "v2/cc/three.json": "{}"}
    src = _records(tmp_path / "src", records)
    (src / "v1" / "aa" / ".x.tmp").write_text("a write in flight")
    assert CB.pack(src, tmp_path / "1.tar.gz") == 3
    for i, p in enumerate(sorted(src.rglob("*.json"))):
        os.utime(p, (1e9 + i, 1e9 + i))
        p.chmod(0o600)
    CB.pack(src, tmp_path / "2.tar.gz")
    assert (tmp_path / "1.tar.gz").read_bytes() == (tmp_path / "2.tar.gz").read_bytes()
    into = _records(tmp_path / "machine" / "circuit-check", {"v0/kept.json": "an older record"})
    assert CB.unpack(tmp_path / "1.tar.gz", into) == 3
    got = {p.relative_to(into).as_posix(): p.read_text() for p in into.rglob("*") if p.is_file()}
    assert got == {**records, "v0/kept.json": "an older record"} and sorted(p.name for p in into.parent.iterdir()) == ["circuit-check"]
    (into / "v1" / "aa" / "one.json").chmod(0o400)
    assert CB.unpack(tmp_path / "2.tar.gz", into) == 3
    assert {p.relative_to(into).as_posix(): p.read_text() for p in into.rglob("*") if p.is_file()} == got
    assert (into / "v1" / "aa" / "one.json").stat().st_mode & 0o777 == 0o400, "a record already there is rewritten"


def test_unpack_renames_only_inside_the_cache_as_its_path_resolves(tmp_path, monkeypatch):
    """A cache whose path is a link onto another directory (on a node, another disk) gets its records by renames inside that
    directory, none from beside the link, which would cross filesystems there."""
    real = tmp_path / "disk" / "circuit-check"
    real.mkdir(parents=True)
    into = tmp_path / "home" / "circuit-check"
    into.parent.mkdir()
    into.symlink_to(real, target_is_directory=True)
    records = {"v1/aa/one.json": '{"a": 1}', "v2/cc/three.json": "{}"}
    CB.pack(_records(tmp_path / "src", records), tmp_path / "r.tar.gz")
    moves, replace = [], os.replace

    def recorded(src, dst):
        moves.append((Path(src), Path(dst)))
        replace(src, dst)
    monkeypatch.setattr(CB.os, "replace", recorded)
    assert CB.unpack(tmp_path / "r.tar.gz", into) == 2
    root = real.resolve()
    assert len(moves) == 2 and all(s.is_relative_to(root) and d.is_relative_to(root) for s, d in moves), moves
    assert {p.relative_to(real).as_posix(): p.read_text() for p in real.rglob("*") if p.is_file()} == records
    assert sorted(p.name for p in real.iterdir()) == ["v1", "v2"], "the unpacking directory is left in the cache"
    assert sorted(p.name for p in into.parent.iterdir()) == ["circuit-check"], "something was made beside the link"


def _machine(tmp_path, monkeypatch) -> str:
    """The machine's circuit-check cache, under a home of the test's, with ``CHECK_CACHE`` set elsewhere, which doesn't move it."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "moved"))
    return str(tmp_path / "home" / ".cache" / "verity-check" / "circuit-check")


def _binding_run(tmp_path, monkeypatch, rc, report, say="") -> tuple[dict, list]:
    work = tmp_path / "work"
    CB.pack(_records(tmp_path / "src", {"v/k/r.json": "{}"}), work / F.RECORDS["path"])
    calls = _stand_in(monkeypatch, CB, rc, report, say)
    task = {"id": "circuit-binding:verity/ml/x.circuit-check.py:_b_roots", "kind": "circuit-binding", "reads": READS,
            "vars": {"binding": "verity.ml.x.circuit-check#_b_roots", "records": F.RECORDS["path"]}}
    doc = _graph(CATALOG, task)
    res = runners.execute(doc["tasks"][-1], doc, work, CB.run)
    assert contract.result_errors(res, doc["tasks"][-1]) == [], res
    return res, calls


def _report(*targets: tuple[str, list[str]], new: list[str] = (), hits: int = 0) -> dict:
    return {"targets": [{"kind": "definition", "target": t, "failures": [{"name": n} for n in fs]} for t, fs in targets],
            "summary": {"new_failures": list(new)}, "cache": {"hits": hits}}


@pytest.mark.parametrize(("rc", "report", "say", "status", "detail"), [
    (0, _report(("A", []), ("B", ["known/x B"]), hits=1), "", "pass",
     {"targets": 2, "passed": 2, "failed": 0, "reused": 1, "records": 1}),
    (1, _report(("A", ["eval/mismatch A"]), ("B", []), new=["eval/mismatch A"]), "", "fail",
     {"targets": 2, "passed": 1, "failed": 1, "reused": 0, "records": 1, "failures": ["eval/mismatch A"]}),
    (0, _report(), "", "pass", {"targets": 0, "passed": 0, "failed": 0, "reused": 0, "records": 1}),
    (2, None, f"circuit-check: refused: --binding verity.ml.x.circuit-check#_b_roots: {CB.NO_BINDING}\n", "fail",
     {"error": "circuit-check's catalog has no binding verity.ml.x.circuit-check#_b_roots"}),
    (2, None, "circuit-check: refused: --binding x: the bindings' records give no task list\n", "error",
     {"error": "circuit-check --binding exited 2 and wrote no report"}),
    (1, None, "Traceback (most recent call last):\n", "error", {"error": "circuit-check --binding exited 1 and wrote no report"}),
    (-9, _report(("A", [])), "", "error", {"error": "circuit-check --binding exited -9"})])
def test_a_binding_s_run_is_circuit_check_s_verdict_on_its_targets_from_the_records(tmp_path, monkeypatch, rc, report, say,
                                                                                      status, detail):
    cache = _machine(tmp_path, monkeypatch)
    res, calls = _binding_run(tmp_path, monkeypatch, rc, report, say)
    assert (res["status"], res["detail"]) == (status, detail), res
    (c,) = calls
    assert c["argv"][1:] == ["-m", "circuit_check", "--all", "--catalog-units", "--cache", cache, "--binding",
                             "verity.ml.x.circuit-check#_b_roots", "-q", "--jobs", "3", "--mem-gb", "5", "--out",
                             str(tmp_path / "work" / CB.REPORT)]
    assert c["cwd"] == ROOT and c["env"] is None and c["cache"] == ["v/k/r.json"], "the records are in the cache before it runs"


@pytest.mark.parametrize(("rc", "report", "status", "detail"), [
    (0, {"reused": 0, "rebuilt": ["a", "b"], "stored": 2, "bytes": 20, "scratch_bytes": 9}, "pass",
     {"bindings": 2, "stored": 2, "records": 2, "bytes": 20}),
    (1, {"stored": 0, "why": "the tracer could not see what it did (g++)"}, "fail",
     {"error": "the tracer could not see what it did (g++)"}),
    (1, None, "error", {"error": "circuit-check --records-only exited 1 and wrote no report"}),
    (-9, None, "error", {"error": "circuit-check --records-only exited -9 and wrote no report"})])
def test_the_catalog_s_run_builds_every_record_into_a_fresh_cache_and_tars_it(tmp_path, monkeypatch, rc, report, status,
                                                                               detail):
    work = tmp_path / "work"
    (work / "cache" / "stale").mkdir(parents=True)
    (work / "cache" / "stale" / "old.json").write_text("{}")
    calls = _stand_in(monkeypatch, CB, rc, report, records={"v/k1/a.json": '{"a": 1}', "v/k2/b.json": '{"b": 2}'})
    doc = _graph(CATALOG)
    res = runners.execute(doc["tasks"][0], doc, work, circuit_catalog.run)
    assert contract.result_errors(res, doc["tasks"][0]) == [], res
    assert (res["status"], res["detail"]) == (status, detail), res
    (c,) = calls
    assert c["argv"][3:7] == ["--all", "--records-only", "--cache", str(work / "cache")] and not c["cache"]
    assert not (work / "cache").exists(), "the fresh cache is the task's scratch"
    if status == "pass":
        CB.pack(_records(tmp_path / "same", {"v/k1/a.json": '{"a": 1}', "v/k2/b.json": '{"b": 2}'}), tmp_path / "same.tar.gz")
        assert res["outputs"] == {"records": contract.file_digest(tmp_path / "same.tar.gz")}
    else:
        assert res["outputs"] == {}


@pytest.mark.parametrize(("listed", "rc", "status", "detail"), [
    (["a", "b"], 0, "pass", {"unlisted": [], "unmade": []}),
    (["a"], 0, "fail", {"unlisted": ["b"], "unmade": []}),
    (["a", "b", "c"], 0, "fail", {"unlisted": [], "unmade": ["c"]}),
    (["a", "b"], 2, "error", {"error": "circuit-check --bindings exited 2"})])
def test_coverage_holds_the_bindings_circuit_check_makes_to_the_listed_ones(tmp_path, monkeypatch, listed, rc, status, detail):
    cache = _machine(tmp_path, monkeypatch)
    work = tmp_path / "work"
    CB.pack(_records(tmp_path / "src", {"v/k/r.json": "{}"}), work / F.RECORDS["path"])
    made = {"bindings": {"a": [["definition", "A"], ["template", "t"]], "b": []}, "root_errors": [], "unchecked": ["F"]}
    calls = _stand_in(monkeypatch, CB, rc, made)
    task = {"id": "circuit-coverage:catalog", "kind": "circuit-coverage", "reads": READS,
            "vars": {"bindings": json.dumps(listed), "records": F.RECORDS["path"]}}
    doc = _graph(CATALOG, task)
    res = runners.execute(doc["tasks"][-1], doc, work, circuit_coverage.run)
    assert contract.result_errors(res, doc["tasks"][-1]) == [], res
    assert res["status"] == status, res
    if rc == 0:
        assert res["detail"] == {"bindings": 2, "targets": 2, "root_errors": [], "unchecked": ["F"], **detail}
    else:
        assert res["detail"]["error"].startswith(detail["error"])
    assert calls[0]["argv"][3:8] == ["--all", "--cache", cache, "--bindings", "-q"]
