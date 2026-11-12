"""`tools/verity/lean/reserve.py`: what a Lean run reserves and how many jobs it runs, never what it checks. It holds numbers
and functions of a package's pins, the steps and a reservation, imports nothing of the audit, and no audit's key reads
it, so a figure raised there reruns nothing; the asks it gives are the ones the README's commands print."""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent
FILE = TOOL / "reserve.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load(FILE, "lean_reserve_under_test")
LA = _load(TOOL / "lean_audit.py", "lean_audit_for_reserve")

#: what a function of `reserve.py` may take: a package's pins, the steps, a reservation, and the units they're said in
ARGS = {"step", "steps", "pins", "pins_of", "of", "env", "machine_gib", "reserved_gib", "jobs", "gib", "gb", "peak_gib", "name"}


def test_it_holds_numbers_and_functions_of_pins_steps_and_a_reservation_and_imports_nothing_of_the_audit():
    """At its top level `reserve.py` has its docstring, imports of the standard library alone (no file of this directory,
    and nothing that loads one), constants that are literals, and functions whose arguments are a package's pins, the
    steps, a reservation and their units; the one file it reads is its own, for the line a warning names. It is in
    `NOT_READ` with what it is, so no audit's key reads it, and every scratch tree carries it (`CARRIED`), since check.py
    plans its run by it."""
    tree = ast.parse(FILE.read_text())
    ours = {p.stem for p in TOOL.iterdir() if p.suffix == ".py"}
    for i, node in enumerate(tree.body):
        if i == 0:
            assert isinstance(node, ast.Expr) and isinstance(node.value.value, str), "its docstring first"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for m in [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]:
                top = m.split(".")[0]
                assert top == "__future__" or top in sys.stdlib_module_names, f"{m}: the standard library alone"
                assert top not in ours | {"importlib", "runpy", "subprocess"}, f"{m} imports or runs the audit"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef):
            a = node.args
            assert {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]} <= ARGS and not (a.vararg or a.kwarg), \
                f"{node.name} takes what isn't a package's pins, the steps or a reservation"
        else:
            raise AssertionError(f"line {node.lineno}: {ast.unparse(node)[:80]}: neither a number nor a function")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"__import__", "exec", "eval", "open", "compile"}, node.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("read_"):
            assert ast.unparse(node.func.value) == "Path(__file__)", "the one file it reads is its own"
    assert LA.NOT_READ["reserve.py"] == "what a run reserves and how many jobs it runs, never what it checks"
    assert "reserve.py" in LA.CARRIED
    got = LA.tool_files([f"{LA.LEAN_TOOL}reserve.py", f"{LA.LEAN_TOOL}check.py"])
    assert f"{LA.LEAN_TOOL}check.py" in got and f"{LA.LEAN_TOOL}reserve.py" not in got, "no audit's key reads reserve.py"


def test_the_asks_floors_and_plan(tmp_path):
    """A heavy build asks 93 GB, an ArkLib replay 258 and the rest 16, each the GB at or above its need in GiB; the build's
    floor is one job and the replay's its need; a heavy build runs every job from its need and fewer below it, and a heavy
    replay one thread per `REPLAY_PER_THREAD_GIB`, each never none."""
    heavy, arklib = frozenset({"mathlib"}), frozenset({"mathlib", "Arklib"})
    assert [R.gb(n) for n in (86, 240, 14, 481)] == [93, 258, 16, 517]
    assert R.gb(R.gib(93)) == 93 and round(R.gib(93), 1) == 86.6
    assert R.ask({"security/Proofs": arklib}, ("build",))[:3] == (93, 86, "build")
    assert R.ask({"security": frozenset(), "security/Proofs": arklib}) == (258, 240, "replay", "security/Proofs")
    assert R.ask({"Flock": frozenset()})[:2] == (16, 14) and R.ask({}, ())[0] == 0
    assert [R.floor(s, p) for s, p in (("build", heavy), ("replay", heavy), ("replay", arklib), ("build", ()))] == \
        [R.ONE_JOB_GIB, 0, R.REPLAY_ARKLIB_GIB, 0]
    assert [R.build_jobs(g, 64) for g in (R.BUILD_HEAVY_GIB, R.gib(93), R.gib(86), R.ONE_JOB_GIB, 4)] == [64, 64, 6, 1, 1]
    assert R.build_jobs(60, 2) == 2
    assert [R.replay_threads(g, 35) for g in (R.REPLAY_ARKLIB_GIB, 100, R.REPLAY_PER_THREAD_GIB, 1)] == [35, 25, 1, 1]
    assert R.replay_threads(R.REPLAY_ARKLIB_GIB, 90) == R.REPLAY_ARKLIB_GIB // R.REPLAY_PER_THREAD_GIB


def test_what_a_run_reserved_is_the_least_said_in_gib():
    """A check slot says its GiB and a `research run --mem-gb` run its GB; what a run reserved is the least of those and the
    machine, and inside a research run that says none of them, or anywhere one of them is no number, nothing."""
    assert R.reserved({}, 500.0) == (500.0, "this machine's memory and swap, within its cgroup")
    assert R.reserved({"CHECK_SLOT_MEM_GB": "64"}, 500.0)[0] == 64
    assert R.reserved({"RESEARCH_RUN_MEM_GB": "93"}, 500.0) == (R.gib(93), "RESEARCH_RUN_MEM_GB=93, in GB")
    assert R.reserved({"CHECK_RUN_MEM_GB": "258", "CHECK_SLOT_MEM_GB": "64", "RESEARCH_RUN_ID": "r"}, 500.0)[0] == 64
    assert R.reserved({"RESEARCH_RUN_MEM_GB": "93"}, 40.0)[0] == 40.0, "the machine's, where it holds less"
    assert R.reserved({"RESEARCH_RUN_ID": "r1", "CHECK_SLOT_MEM_GB": "x"}, 500.0)[0] is None
    assert R.reserved({"CHECK_RUN_MEM_GB": "93GB", "CHECK_SLOT_MEM_GB": "64"}, 500.0) == \
        (None, "CHECK_RUN_MEM_GB='93GB' is no number of GB"), "a reservation it can't read is none, never the machine"
    assert R.reserved({"RESEARCH_RUN_ID": "r1", "RESEARCH_RUN_MEM_GB": "258"}, 500.0)[0] == R.gib(258)


def test_a_peak_past_its_share_warns_naming_the_line_to_raise():
    lines = FILE.read_text().splitlines()
    for name in ("BUILD_HEAVY_GIB", "REPLAY_ARKLIB_GIB", "LIGHT_GIB"):
        n = int(R.where(name).rsplit(":", 1)[1])
        assert lines[n - 1].startswith(f"{name} = "), name
    arklib = {"Arklib"}
    assert R.warning("replay", arklib, R.WARN_SHARE * R.REPLAY_ARKLIB_GIB) is None
    w = R.warning("replay", arklib, R.REPLAY_ARKLIB_GIB - 1)
    assert w.startswith(f"replay: its peak, {R.REPLAY_ARKLIB_GIB - 1} GiB, passes 95% of the {R.REPLAY_ARKLIB_GIB} GiB")
    assert f"raise REPLAY_ARKLIB_GIB at {R.where('REPLAY_ARKLIB_GIB')}," in w
    assert "BUILD_HEAVY_GIB" in R.warning("build", {"mathlib"}, R.BUILD_HEAVY_GIB)


def test_the_readme_s_commands_ask_what_reserve_py_gives():
    """The README's records run asks a heavy build's need and its full record an ArkLib replay's, each in GB on the
    command and in GiB on the line under it, so a figure raised here without its commands fails."""
    text = (TOOL / "README.md").read_text()
    for tool, n in (("lean_changed.py --records", R.BUILD_HEAVY_GIB), ("check.py --build --update", R.REPLAY_ARKLIB_GIB)):
        m = re.search(r"^research run .*--mem-gb (\d+) .*\n.*" + re.escape(tool) + r".*\n(.*)$", text, re.M)
        assert m and int(m[1]) == R.gb(n), tool
        assert m[2].strip().startswith(f"{R.gb(n)} GB is {R.gib(R.gb(n)):.1f} GiB"), m[2]
