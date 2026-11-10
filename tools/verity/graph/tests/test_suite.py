"""The suite family: on this tree its tasks are check's pytest step, one suite a task (each part a suite declares by test id
one), keyed on what suites.py keys the suite on; on a toy checkout, a declared part, a parallel suite and Markdown; and its
runner runs suites.py with check's words, telling a fail, the suite's, from an error, the machine's, on a stand-in suites.py."""
from __future__ import annotations

import importlib.util
import json
import math

import pytest
from conftest import SPEC, toy_tree

from verity_graph import ROOT, contract, graph, runners
from verity_graph.families import suite as family
from verity_graph.runners import suite as runner

spec = importlib.util.spec_from_file_location("suite_check", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)
S = CHECK.SUITES


def _table() -> str:
    """`[family.suite]` as this tree's tasks.toml has it."""
    text = (ROOT / graph.SPEC).read_text()
    start = text.index("[family.suite]")
    return text[start:text.index("\n[", start + 1)] + "\n"


@pytest.fixture(scope="module")
def doc() -> dict:
    return graph.listing(ROOT, {"suite"})


def test_on_this_tree_a_suite_task_is_check_s_pytest_step_on_one_suite_keyed_as_suites_py_keys_it(doc):
    """Every suite suites.py finds but those check runs after lean-build, once, or each part it declares once, its var
    `suite` its directory; its inputs the files under the paths suites.py keys it on (`suites.inputs_of`, its nested suites
    cut, its `*.md` only when it reads Markdown), suites.py's runner files, what a fixture's fetch reads and its own runner;
    one that runs in parallel on the family's slot, any other on one core and its worker_gb, at its own timeout."""
    files = graph.tracked(ROOT)
    pkgs = S.members()
    fam = graph.spec(ROOT)["family"]["suite"]
    by: dict[str, list[dict]] = {}
    for t in doc["tasks"]:
        by.setdefault(t["vars"]["suites"], []).append(t)
        assert t["vars"]["suite"] == next(s.path for s in S.discover() if s.name == t["vars"]["suites"]), t["id"]
    run = [s for s in S.discover() if s.name not in S.AFTER_LEAN and s.path not in S.AFTER_LEAN]
    assert sorted(by) == sorted(s.name for s in run) and {t["kind"] for t in doc["tasks"]} == {"suite"}
    for s in run:
        m = s.parts if s.parts and s.parts > 1 else None
        assert [t["vars"].get("files") for t in by[s.name]] == ([f"{k}/{m}" for k in range(1, m + 1)] if m else [None])
        paths = S.inputs_of(s, pkgs)
        cut = S.dropped(s, paths)
        want = {f for f in files if S.covered(f, paths, cut) and (S.markdown(s) or not f.endswith(".md"))}
        want |= {*S.RUNNER, *(p for p in S.STORE_READS if p in files), *graph.runner_sources("suite")}
        one = {"cores": 1, "mem_gb": s.worker_gb or S.TEST_PROCESS_GB, "disk_gb": fam["resources"]["disk_gb"]}
        for t in by[s.name]:
            assert set(t["inputs"]) == want, f"{t['id']}: suites.py's key, {sorted(want ^ set(t['inputs']))[:5]} apart"
            assert t["resources"] == (fam["resources"] if s.parallel else one), t["id"]
            assert t["timeout_s"] == (math.ceil(s.timeout_s) if s.timeout_s else fam["timeout_s"]), t["id"]
            assert t["key"] and not t["reads"] and t["outputs"] == [{"name": "report", "path": runner.REPORT}]


PYTEST = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
TOY = {
    "pyproject.toml": '[project]\nname = "toy"\nversion = "0"\n[tool.uv.workspace]\nmembers = ["s", "m", "lean"]\n',
    "s/pyproject.toml": f'[project]\nname = "toy-s"\n{PYTEST}[tool.verity.tests]\nparts = 3\nparallel = true\n',
    "s/tests/test_s.py": "def test_s():\n    pass\n", "s/README.md": "prose\n",
    "m/pyproject.toml": (f'[project]\nname = "toy-m"\n{PYTEST}[tool.verity.tests]\ninputs = ["notes/m.md"]\nworker_gb = 3\n'
                         'timeout_s = 120\n'),
    "m/tests/test_m.py": "def test_m():\n    pass\n", "m/README.md": "prose\n", "notes/m.md": "read\n",
    "lean/pyproject.toml": f'[project]\nname = "toy-lean"\n{PYTEST}', "lean/tests/test_l.py": "def test_l():\n    pass\n",
    "tools/verity/check/suites.py": ('RUNNER = ("tools/verity/check/suites.py",)\nREGISTRY = "fixtures/artifacts.json"\n'
                                     'STORE_READS = (REGISTRY, "tools/research/store.pod.toml")\nTEST_PROCESS_GB = 9\n'
                                     'AFTER_LEAN = ("toy-lean",)\n'),
    "fixtures/artifacts.json": "{}\n", f"{graph.RUNNERS}/suite.py": ""}


def test_a_declared_part_a_parallel_suite_and_markdown(tmp_path):
    """A suite that declares `parts` by test id is a task a part, each with its var `files`; a parallel one takes the family's
    slot. A suite's `*.md` are no input unless it lists a `.md`, as suites.py keys it; then all are. A suite of AFTER_LEAN is
    one here, since the build family runs those only in a tree with the Lean verifier (`build.lean_suites`), and the root, which
    configures no pytest, is none."""
    root = toy_tree(tmp_path / "t", SPEC + "\n" + _table(), TOY)
    got = {t["id"]: t for t in graph.listing(root, {"suite"}, listers={"suite": family.tasks})["tasks"]}
    assert sorted(got) == ["suite:toy-lean", "suite:toy-m", "suite:toy-s:1/3", "suite:toy-s:2/3", "suite:toy-s:3/3"]
    fam = graph.spec(root)["family"]["suite"]
    common = {"tools/verity/check/suites.py", "fixtures/artifacts.json", *graph.runner_sources("suite")}
    for k in (1, 2, 3):
        t = got[f"suite:toy-s:{k}/3"]
        assert t["vars"] == {"suites": "toy-s", "suite": "s", "files": f"{k}/3"} and t["resources"] == fam["resources"]
        assert set(t["inputs"]) == {"s/pyproject.toml", "s/tests/test_s.py", *common}, "no s/README.md: toy-s reads no Markdown"
    m = got["suite:toy-m"]
    assert m["vars"] == {"suites": "toy-m", "suite": "m"} and m["timeout_s"] == 120
    assert m["resources"] == {"cores": 1, "mem_gb": 3, "disk_gb": fam["resources"]["disk_gb"]}
    assert set(m["inputs"]) == {"m/pyproject.toml", "m/tests/test_m.py", "m/README.md", "notes/m.md", *common}


def test_the_runner_runs_check_s_words_on_the_task_s_suite(tmp_path, monkeypatch):
    """check's pytest step's words (`check.groups`), less its skips, on the task's suite, at the task's cores; then its memory,
    where check gives its slot's share, and a part's `--files K/M` with a timing record that holds none of its files."""
    out = tmp_path / "out"
    words = next(argv for g in CHECK.groups(out, 8) for name, argv, _ in g if name == "pytest")
    words = [w for i, w in enumerate(words) if w != "--skip" and words[i - 1] != "--skip"]
    jobs = words.index("--jobs") + 2
    out.mkdir()
    seen: list[list[str]] = []

    def call(argv, work, cwd, env=None):
        assert cwd == ROOT
        seen.append(argv)
        return 2
    monkeypatch.setattr(runner, "call", call)
    for files in (None, "2/3"):
        task = {"vars": {"suites": "toy-s", **({"files": files} if files else {})},
                "resources": {"cores": int(words[jobs - 1]), "mem_gb": 5, "disk_gb": 1}}
        assert runner.run(task, out)["status"] == "error"
        part = ["--files", files, "--timings", str(out / runner.TIMINGS)] if files else []
        assert seen.pop() == [*words[:2], "toy-s", *words[2:jobs], "--mem-gb", "5", *part]
    assert json.loads((out / runner.TIMINGS).read_text()) == {"toy-s": {}}


STANDIN = """import json, os, signal, sys
from pathlib import Path
a = sys.argv[1:]
names = a[:a.index("--json")]
assert "--lean-built" not in a and a[a.index("--jobs") + 1] == "3" and a[a.index("--mem-gb") + 1] == "5" and "--timeouts" in a
how = Path(os.environ["HOW"]).read_text()
if how == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
if how == "usage":
    sys.exit(2)
rows = [{"suite": "s", "name": n, "status": "passed", "failed_tests": []} for n in names]
if how == "cached":
    rows[0]["status"] = "cached"
if how in ("tests", "outside"):
    rows[0].update(status="failed", failed_tests=["tests/test_s.py::test_s"], beyond_tests=how == "outside")
if how == "outside":
    rows[0]["outside"] = ["elsewhere.txt"]
if how == "timeout":
    rows[0].update(status="failed", timed_out=120.0, beyond_tests=True)
if how == "skipped":
    rows[0]["status"] = "skipped"
Path(a[a.index("--json") + 1]).write_text(json.dumps(rows))
sys.exit(1 if how in ("tests", "outside", "timeout") else 0)
"""


@pytest.mark.parametrize(("how", "status"), [("pass", "pass"), ("cached", "pass"), ("tests", "fail"), ("outside", "fail"),
                                             ("timeout", "fail"), ("skipped", "error"), ("usage", "error"), ("kill", "error")])
def test_the_runner_tells_the_suite_s_fail_from_the_machine_s_error(tmp_path, monkeypatch, how, status):
    """A suite that failed on its tests, a read outside its inputs or its timeout is a fail naming it (`detail.failed`,
    drive's `suite NAME`), and one that failed on nothing but its tests names them as a test file's result does
    (`detail.failed_tests`, the loop's); one not run, a usage error and a kill are an error, no result."""
    root = tmp_path / "tree"
    (root / runner.SUITES).parent.mkdir(parents=True)
    (root / runner.SUITES).write_text(STANDIN)
    monkeypatch.setattr(runner, "ROOT", root)
    (tmp_path / "how").write_text(how)
    monkeypatch.setenv("HOW", str(tmp_path / "how"))
    task = {"id": "suite:toy-s", "kind": "suite", "about": "x", "env": "py", "argv": ["verity", "graph", "run", "suite:toy-s"],
            "vars": {"suites": "toy-s", "suite": "s"}, "inputs": [], "reads": [],
            "outputs": [{"name": "report", "path": runner.REPORT}],
            "resources": {"cores": 3, "mem_gb": 5, "disk_gb": 1}, "timeout_s": 60, "key": None}
    res = runners.execute(task, {"envs": {"py": {"digest": contract.digest(b"py")}}, "files": {}, "tasks": [task]},
                          tmp_path / "work", runner.run)
    assert contract.result_errors(res, task) == [], res
    assert res["status"] == status, res
    if how in ("tests", "outside"):
        assert res["detail"]["failed"] == [{"name": "toy-s", "suite": "s", "failed_tests": ["tests/test_s.py::test_s"],
                                            "beyond_tests": how == "outside"}]
    assert res["detail"].get("failed_tests") == (["tests/test_s.py::test_s"] if how == "tests" else None)
    if how == "timeout":
        assert res["detail"]["failed"][0]["timed_out"] == 120.0
    if how == "skipped":
        assert res["detail"]["error"] == "not run: toy-s"
    assert (status == "pass") == ("report" in res["outputs"] and res["status"] == "pass")


def test_lean_suites_and_suite_share_one_runner():
    """The lean-suites runner runs its suites with this runner's `suites` (`--lean-built`, after lake's prefix): one reading
    of suites.py's report for both kinds, which drive names alike (`research.drive.SUITE_KINDS`)."""
    from verity_graph.runners import lean_suites

    assert lean_suites.suites is runner.suites and lean_suites.REPORT != runner.REPORT
    assert family.REPORT == runner.REPORT
