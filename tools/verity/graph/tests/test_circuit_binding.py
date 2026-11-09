"""The circuit-binding family: its listing is circuit-check's catalog bindings, read from the binding files' syntax without
running them, the same bytes on every listing; its runners tell a pass, a fail and an error apart from circuit-check's exit and
report (a stand-in's run, its reports made by circuit-check's own writers), the records tarball is the records' bytes alone,
and each binding's task reads only its own slice and the shared part."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from circuit_check import checks as CK
from circuit_check import shard as SH
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
    assert sorted(by) == sorted([F.CATALOG, "circuit-coverage:catalog",
                                 *(f"circuit-binding:{f}:{e}" for f, e in BINDINGS.values())])
    base = sorted(f for f in files if f.startswith(("verity/", f"{F.CHECKER}/")) and not f.startswith(f"{F.CHECKER}/tests/")
                  and not f.endswith(".md") or f == "uv.lock")
    for name, (f, e) in BINDINGS.items():
        t = by[f"circuit-binding:{f}:{e}"]
        assert t["vars"] == {"binding": name, "slice": F.slice_of(name)["path"], "shared": F.SHARED["path"]}, name
        assert t["reads"] == [{"task": F.CATALOG, "output": "shared"}, {"task": F.CATALOG, "output": f"slice {name}"}]
        assert t["inputs"] == sorted({*base, *graph.runner_sources("circuit-binding")}) and t["key"] is None, name
        assert t["about"] == f"circuit-check --binding {name}" and t["outputs"] == [F.failures(name)]
    cat, cov = by[F.CATALOG], by["circuit-coverage:catalog"]
    assert cat["outputs"] == [F.RECORDS, F.SHARED, *sorted((F.slice_of(n) for n in BINDINGS), key=lambda o: o["name"])]
    assert cat["reads"] == [] and cat["key"] and cov["reads"] == reads
    pieces = [F.slice_of(name)["path"] for name in BINDINGS]
    assert len(set(pieces)) == len(pieces) and F.SHARED["path"] not in pieces
    assert all(p.rpartition("/")[0] == F.SLICES for p in [*pieces, F.SHARED["path"]])
    assert json.loads(cov["vars"]["bindings"]) == sorted(BINDINGS)
    assert f"{graph.RUNNERS}/circuit_binding.py" in cat["inputs"] and f"{graph.RUNNERS}/circuit_binding.py" in cov["inputs"]
    paths = [F.failures(name)["path"] for name in BINDINGS]
    assert len(set(paths)) == len(paths) and all(p.rpartition("/")[0] == F.FAILURES for p in paths)


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
    assert f'print(f"{CB.REFUSED}{{e}}", file=sys.stderr)' in (ROOT / F.SOURCE / "cli.py").read_text()
    assert F.slice_of("n")["name"] == f"{circuit_catalog.SLICE}n"


# --- the runners, on a stand-in for circuit-check --------------------------------------------------------------------------

def _graph(*tasks: dict) -> dict:
    full = [{"about": "x", "env": "py", "vars": {}, "inputs": [], "reads": [], "outputs": [],
             "argv": ["verity", "graph", "run", t["id"]], "resources": {"cores": 3, "mem_gb": 5, "disk_gb": 1},
             "timeout_s": 60, "key": None, **t} for t in tasks]
    return {"envs": {"py": {"digest": contract.digest(b"py")}}, "files": {}, "tasks": full}


B_ROOTS = "verity.ml.x.circuit-check#_b_roots"
#: a listed binding circuit-check's catalog made no slice for
GONE = "verity.ml.x.circuit-check#_c_roots"
CATALOG = {"id": F.CATALOG, "kind": "circuit-catalog", "outputs": [F.RECORDS, F.SHARED, F.slice_of(B_ROOTS), F.slice_of(GONE)]}
READS = [{"task": F.CATALOG, "output": "records"}]


def _stand_in(monkeypatch, module, rc: int, report: dict | None, say: str = "", records: dict | None = None,
              slices: dict | None = None) -> list:
    """circuit-check as ``module`` calls it: each call's argv and the cache's records then, appended to the list returned; it
    writes ``records`` into --cache, ``slices`` into --slices, ``report`` to --out and ``say`` to the log, and exits ``rc``."""
    calls = []

    def call(argv, work, cwd, env=None):
        cache = Path(argv[argv.index("--cache") + 1])
        calls.append({"argv": argv, "cwd": cwd, "env": env, "cache": sorted(
            p.relative_to(cache).as_posix() for p in cache.rglob("*.json")) if cache.is_dir() else None})
        for rel, body in (records or {}).items():
            (cache / rel).parent.mkdir(parents=True, exist_ok=True)
            (cache / rel).write_text(body)
        for rel, body in (slices or {}).items():
            (Path(argv[argv.index("--slices") + 1]) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(argv[argv.index("--slices") + 1]) / rel).write_text(body)
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


def _binding_run(tmp_path, monkeypatch, rc, report, say="", piece: dict | None = None) -> tuple[dict, list]:
    """``B_ROOTS``' task on the stand-in, its slice ``piece`` (a slice's stand-in by default) and the shared part in place."""
    work = tmp_path / "work"
    _records(work, {F.slice_of(B_ROOTS)["path"]: json.dumps(piece or {"binding": B_ROOTS, "entries": []}),
                    F.SHARED["path"]: '{"schema": "s"}'})
    calls = _stand_in(monkeypatch, CB, rc, report, say)
    task = {"id": "circuit-binding:verity/ml/x.circuit-check.py:_b_roots", "kind": "circuit-binding",
            "reads": [{"task": F.CATALOG, "output": "shared"}, {"task": F.CATALOG, "output": F.slice_of(B_ROOTS)["name"]}],
            "outputs": [F.failures(B_ROOTS)],
            "vars": {"binding": B_ROOTS, "slice": F.slice_of(B_ROOTS)["path"], "shared": F.SHARED["path"]}}
    doc = _graph(CATALOG, task)
    res = runners.execute(doc["tasks"][-1], doc, work, CB.run)
    assert contract.result_errors(res, doc["tasks"][-1]) == [], res
    return res, calls


def _failure(name: str) -> dict:
    """A failure as circuit-check makes it (`checks.Findings.fail`), from its name ``<check>/<code> <definition>[ <what>]``."""
    check, _, rest = name.partition("/")
    code, _, rest = rest.partition(" ")
    definition, _, what = rest.partition(" ")
    found = CK.Findings()
    found.fail(check, code, definition, what)
    return found.failures[0]


def _report(*targets: tuple[str, list[str]], hits: int = 0, others: tuple[dict, ...] = ()) -> dict:
    """The report `circuit-check --binding` writes of its Definition ``targets`` (each a name and its failures' names) and the
    records ``others``: unjoined, as a shard's (`shard.unjoined`, whose summary is `checks.summarize`'s), with the cache's
    counts `cli.run` adds."""
    records = [*({"kind": "definition", "target": t, "failures": [_failure(n) for n in fs], "warnings": []} for t, fs in targets),
               *others]
    return {**SH.unjoined(records, CK.Options()), "peaks": [],
            "cache": {"hits": hits, "stored": 0, "untraced": [], "errors": [], "changed": []}}


@pytest.mark.parametrize(("rc", "report", "say", "status", "detail"), [
    (0, _report(("A", []), ("B", []), hits=1), "", "pass", {"targets": 2, "passed": 2, "failed": 0, "reused": 1}),
    (1, _report(("A", ["eval/mismatch A"]), ("B", [])), "", "fail",
     {"targets": 2, "passed": 1, "failed": 1, "reused": 0, "failures": ["eval/mismatch A"]}),
    (0, _report(), "", "pass", {"targets": 0, "passed": 0, "failed": 0, "reused": 0}),
    (2, None, f"circuit-check: refused: --binding verity.ml.x.circuit-check#_b_roots: {CB.NO_BINDING}\n", "fail",
     {"error": "circuit-check's catalog has no binding verity.ml.x.circuit-check#_b_roots"}),
    (2, None, "circuit-check: refused: x: its record in the slice doesn't count here (a read changed)\n", "error",
     {"error": "circuit-check --binding exited 2 and wrote no report: x: its record in the slice doesn't count here "
               "(a read changed)"}),
    (1, None, "Traceback (most recent call last):\n", "error", {"error": "circuit-check --binding exited 1 and wrote no report"}),
    (-9, _report(("A", [])), "", "error", {"error": "circuit-check --binding exited -9"})])
def test_a_binding_s_run_is_circuit_check_s_verdict_on_its_targets_from_its_slice(tmp_path, monkeypatch, rc, report, say,
                                                                                    status, detail):
    cache = _machine(tmp_path, monkeypatch)
    res, calls = _binding_run(tmp_path, monkeypatch, rc, report, say)
    assert (res["status"], res["detail"]) == (status, detail), res
    assert ("failures" in res["outputs"]) == (rc in (0, 1) and report is not None), "a report's failures are its output"
    (c,) = calls
    work = tmp_path / "work"
    assert c["argv"][1:] == ["-m", "circuit_check", "--all", "--catalog-units", "--cache", cache, "--binding", B_ROOTS,
                             "--slice", str(work / F.slice_of(B_ROOTS)["path"]), str(work / F.SHARED["path"]),
                             "-q", "--jobs", "3", "--mem-gb", "5", "--out", str(work / CB.REPORT)]
    assert c["cwd"] == ROOT and c["env"] is None and c["cache"] is None, "the run puts no record in the machine's cache"


def test_a_binding_the_catalog_made_no_slice_for_fails_as_a_name_the_catalog_lacks_without_running_circuit_check(
        tmp_path, monkeypatch):
    _machine(tmp_path, monkeypatch)
    res, calls = _binding_run(tmp_path, monkeypatch, 0, _report(("A", [])), piece=CB.absent(B_ROOTS))
    assert (res["status"], res["detail"]) == ("fail", {"error": f"circuit-check's catalog has no binding {B_ROOTS}"}), res
    assert calls == [] and res["outputs"] == {}


@pytest.mark.parametrize(("targets", "status", "detail"), [
    ((("A", []), ("B", [])), "pass", {"targets": 2, "passed": 2, "failed": 0, "reused": 2}),
    ((("A", ["eval/mismatch A at bit 3", "partition/gate-recomputed A"]), ("B", []), ("C", ["eval/mismatch C"])), "fail",
     {"targets": 3, "passed": 1, "failed": 2, "reused": 2,
      "failures": ["eval/mismatch A at bit 3", "eval/mismatch C", "partition/gate-recomputed A"]})])
def test_a_binding_s_verdict_is_circuit_check_s_on_the_report_its_own_writers_make(tmp_path, monkeypatch, targets, status,
                                                                                  detail):
    """The report and exit `circuit-check --binding` ends with, made by its own writers: `shard.unjoined`, whose summary is
    `checks.summarize`'s, and `cli.verdict`, which writes it and exits 1 on a failure.  The binding fails on the failures that
    summary names (``failure_names``), naming each, and passes with none."""
    from circuit_check import cli

    rep = _report(*targets, hits=2)
    out = tmp_path / "circuit-check.json"
    rc = cli.verdict(rep, out=out, what=f"binding {B_ROOTS}: {len(rep['targets'])} target(s)")
    _machine(tmp_path, monkeypatch)
    res, _ = _binding_run(tmp_path, monkeypatch, rc, json.loads(out.read_text()))
    assert (rc, res["status"], res["detail"]) == (int(status == "fail"), status, detail), res


def test_a_binding_s_output_is_each_failure_of_its_definition_targets_by_check_code_and_definition(tmp_path, monkeypatch):
    _machine(tmp_path, monkeypatch)
    template = {"kind": "template", "target": "t", "failures": [_failure("structural/mismatch t")], "warnings": []}
    rep = _report(("B_v1{n=2}", ["partition/gate-recomputed B_v1{n=2}", "eval/mismatch B_v1{n=2} at bit 3"]),
                  ("B_v1{n=3}", ["partition/gate-recomputed B_v1{n=3}", "partition/gate-recomputed B_v1{n=3}"]), ("A", []),
                  others=(template,))
    res, _ = _binding_run(tmp_path, monkeypatch, 1, rep)
    out = tmp_path / "work" / F.failures(B_ROOTS)["path"]
    assert json.loads(out.read_text()) == {"binding": B_ROOTS, "failures": [
        {"check": "eval", "code": "mismatch", "definition": "B_v1{n=2}"},
        {"check": "partition", "code": "gate-recomputed", "definition": "B_v1{n=2}"},
        {"check": "partition", "code": "gate-recomputed", "definition": "B_v1{n=3}"}]}
    assert res["status"] == "fail" and res["outputs"] == {"failures": contract.file_digest(out)}


#: what the stand-in's ``--slices`` writes: B_ROOTS' slice and the shared part, none of GONE's
SLICED = {"index.json": json.dumps({"shared": "shared.json", "bindings": {B_ROOTS: "b.json"}}), "shared.json": '{"s": 1}',
          "b.json": '{"b": 1}'}


@pytest.mark.parametrize(("rc", "report", "status", "detail"), [
    (0, {"reused": 0, "rebuilt": ["a", "b"], "stored": 2, "bytes": 20, "scratch_bytes": 9,
         "slices": {"bindings": 1, "bytes": 8, "shared_bytes": 8, "largest": 8}}, "pass",
     {"bindings": 2, "stored": 2, "records": 2, "bytes": 20, "slices": 1, "slice_bytes": 8, "shared_bytes": 8,
      "absent": [GONE]}),
    (1, {"stored": 0, "why": "the tracer could not see what it did (g++)"}, "fail",
     {"error": "the tracer could not see what it did (g++)"}),
    (1, {"stored": 2, "why": "no slices: _b_roots: a record with no reads to hold it to"}, "fail",
     {"error": "no slices: _b_roots: a record with no reads to hold it to"}),
    (1, None, "error", {"error": "circuit-check --records-only exited 1 and wrote no report"}),
    (-9, None, "error", {"error": "circuit-check --records-only exited -9 and wrote no report"})])
def test_the_catalog_s_run_builds_every_record_into_a_fresh_cache_tars_it_and_gives_each_binding_its_slice(
        tmp_path, monkeypatch, rc, report, status, detail):
    work = tmp_path / "work"
    for d in ("cache", "slices"):
        (work / d / "stale").mkdir(parents=True)
        (work / d / "stale" / "old.json").write_text("{}")
    calls = _stand_in(monkeypatch, CB, rc, report, records={"v/k1/a.json": '{"a": 1}', "v/k2/b.json": '{"b": 2}'},
                      slices=SLICED)
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    doc = _graph(CATALOG)
    res = runners.execute(doc["tasks"][0], doc, work, circuit_catalog.run)
    assert contract.result_errors(res, doc["tasks"][0]) == [], res
    assert (res["status"], res["detail"]) == (status, detail), res
    (c,) = calls
    assert c["argv"][3:9] == ["--all", "--records-only", "--cache", str(work / "cache"), "--slices", str(work / "slices")]
    assert c["env"] == {**os.environ, **circuit_catalog.THREADS} and c["env"]["OMP_NUM_THREADS"] == "1", \
        "each of its builds, one per core at once, has one-thread pools"
    assert not c["cache"] and not (work / "cache").exists() and not (work / "slices").exists(), "both are the task's scratch"
    if status == "pass":
        CB.pack(_records(tmp_path / "same", {"v/k1/a.json": '{"a": 1}', "v/k2/b.json": '{"b": 2}'}), tmp_path / "same.tar.gz")
        gone = tmp_path / "gone.json"
        gone.write_text(json.dumps(CB.absent(GONE), sort_keys=True) + "\n")
        assert res["outputs"] == {"records": contract.file_digest(tmp_path / "same.tar.gz"),
                                  "shared": contract.file_digest(work / F.SHARED["path"]),
                                  F.slice_of(B_ROOTS)["name"]: contract.file_digest(work / F.slice_of(B_ROOTS)["path"]),
                                  F.slice_of(GONE)["name"]: contract.file_digest(gone)}
        assert (work / F.SHARED["path"]).read_text() == SLICED["shared.json"]
        assert (work / F.slice_of(B_ROOTS)["path"]).read_text() == SLICED["b.json"]
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

