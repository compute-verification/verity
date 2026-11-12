"""`tools/verity/lean/reserve.py` and its figures, `reserve.json`: what a Lean run reserves and how many jobs it runs, never
what it checks. The rules are in every audit's key and the figures in none, so a changed rule audits every package again
and a raised figure reruns nothing; the asks they give are the ones the README's commands print."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent
FILE, DATA = TOOL / "reserve.py", TOOL / "reserve.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load(FILE, "lean_reserve_under_test")
LA = _load(TOOL / "lean_audit.py", "lean_audit_for_reserve")

#: the figures the rules read, by name
NAMES = {"BUILD_HEAVY_GIB", "LIGHT_GIB", "REPLAY_ARKLIB_GIB", "REPLAY_PER_THREAD_GIB", "ONE_JOB_GIB", "PER_JOB_GIB", "MARGIN"}
STEPS = ("build", "replay")


def test_the_data_file_is_the_named_figures_and_nothing_else():
    """`reserve.json` is one object of `NAMES`, each once and each a positive number, which `reserve.py` reads by name; it
    is in `NOT_READ` and carried into every scratch tree (`CARRIED`), and `reserve.py`, which holds the rules, is in every
    audit's key. A step's figures order as the rules read them: a heavy build's one job below its whole need."""
    pairs = json.loads(DATA.read_text(), object_pairs_hook=list)
    assert sorted(n for n, _ in pairs) == sorted(NAMES), "each named figure once, and nothing else"
    assert all(type(v) in (int, float) and v > 0 for _, v in pairs), pairs
    assert {n: getattr(R, n) for n in NAMES} == dict(pairs) == R.FIGURES
    assert R.MARGIN < 1 and R.ONE_JOB_GIB < R.BUILD_HEAVY_GIB
    assert "reserve.json" in LA.NOT_READ and "reserve.json" in LA.CARRIED and "reserve.py" not in LA.NOT_READ
    got = LA.tool_files([f"{LA.LEAN_TOOL}{f}" for f in ("check.py", "reserve.py", "reserve.json")])
    assert {f"{LA.LEAN_TOOL}check.py", f"{LA.LEAN_TOOL}reserve.py"} <= set(got) and f"{LA.LEAN_TOOL}reserve.json" not in got


#: a stand-in for check.py in the scratch tree: it plans by `reserve.py`, which reads `reserve.json`, and passes each package
FAKE_CHECK = r'''
import importlib.util, json, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("reserve", Path(__file__).with_name("reserve.py"))
spec.loader.exec_module(importlib.util.module_from_spec(spec))
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1])
named = [x for i, x in enumerate(a) if not x.startswith("--") and not (i and a[i - 1] in ("--out", "--repo", "--replay-cache", "--artifacts"))]
out.mkdir(parents=True, exist_ok=True)
with open(out / "ran.txt", "a") as f:
    f.write("".join(os.path.relpath(p) + "\n" for p in named))
(out / "audit.json").write_text(json.dumps({"passed": True, "failed": [],
                                            "packages": [{"package": p, "failures": [], "sandbox": True} for p in named]}))
'''


def test_a_pass_kept_under_one_reserve_json_is_reused_under_another_and_a_changed_rule_audits_again(tmp_path, monkeypatch):
    """The ruling that put the figures in data: a pass kept under one `reserve.json` is reused under another, since no key
    reads it, while the audit still finds it in its scratch tree; a changed rule of `reserve.py` moves every key."""
    root = tmp_path / "repo"
    files = {"tools/verity/lean/check.py": FAKE_CHECK, "tools/verity/lean/lean_audit.py": "",
             "tools/verity/lean/reserve.py": FILE.read_text(), "tools/verity/lean/reserve.json": DATA.read_text(),
             "tools/verity/check/common.py": "", LA.CONTRACT: "",
             "pkg/a/lakefile.toml": "", "pkg/a/lean-toolchain": "v1\n", "pkg/a/A.lean": "theorem a : True := trivial\n",
             "pkg/a/lake-manifest.json": json.dumps({"packages": []}), "pkg/a/lean-audit.json": "{}"}
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    monkeypatch.setattr(LA, "ROOT", root)
    monkeypatch.setattr(LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LA.COMMON, "cores", lambda: 2)
    monkeypatch.setattr(LA, "sandboxed", lambda: True)

    def run(n: int) -> tuple[int, list[str], dict]:
        out = tmp_path / f"out{n}"
        rc = LA.run(out, tmp_path / "cache" / "lean-audit", tmp_path / "cache" / "lean-deps", log=lambda s: None)
        ran = (out / "ran.txt").read_text().split() if (out / "ran.txt").exists() else []
        return rc, ran, json.loads((out / "audit.json").read_text())

    key = LA.key("pkg/a", sorted(files), LA.FLAGS)
    assert run(1)[:2] == (0, ["pkg/a"])
    figures = json.loads(DATA.read_text())
    (root / "tools/verity/lean/reserve.json").write_text(json.dumps(figures | {"REPLAY_ARKLIB_GIB": 2 * R.REPLAY_ARKLIB_GIB}))
    assert LA.key("pkg/a", sorted(files), LA.FLAGS) == key
    rc, ran, doc = run(2)
    assert (rc, ran, doc["cached"]) == (0, [], {"pkg/a": key}), "a raised figure reruns nothing"
    rule = "    return max(1, min(jobs, int(reserved_gib // REPLAY_PER_THREAD_GIB)))\n"
    assert rule in FILE.read_text()
    (root / "tools/verity/lean/reserve.py").write_text(FILE.read_text().replace(rule, "    return 1\n"))
    assert LA.key("pkg/a", sorted(files), LA.FLAGS) != key
    rc, ran, doc = run(3)
    assert (rc, ran, doc["cached"]) == (0, ["pkg/a"], {}), "a changed rule audits the package again"


def test_the_asks_floors_and_plan():
    """A run asks the GB at or above the largest need among its steps of the packages it audits; a heavy build's floor is
    its one job and an ArkLib replay's its whole need; a heavy build runs one job in `ONE_JOB_GIB`, one more per
    `PER_JOB_GIB` above it and every job from `BUILD_HEAVY_GIB`, and a heavy replay one thread per
    `REPLAY_PER_THREAD_GIB`, each never none and never more than it has."""
    heavy, arklib = frozenset({"mathlib"}), frozenset({"mathlib", "Arklib"})
    assert [R.gb(n) for n in (1, 10, 100)] == [2, 11, 108] and R.gb(R.gib(93)) == 93 and round(R.gib(93), 1) == 86.6
    assert R.ask({"security/Proofs": arklib}, ("build",)) == \
        (R.gb(R.BUILD_HEAVY_GIB), R.BUILD_HEAVY_GIB, "build", "security/Proofs")
    want = max((R.BUILD_HEAVY_GIB, "build"), (R.REPLAY_ARKLIB_GIB, "replay"))
    assert R.ask({"security": frozenset(), "security/Proofs": arklib}, STEPS) == (R.gb(want[0]), *want, "security/Proofs")
    assert R.ask({"Flock": frozenset()}, STEPS)[:2] == (R.gb(R.LIGHT_GIB), R.LIGHT_GIB) and R.ask({}, ()) == (0, 0, None, None)
    assert [R.floor(s, p) for s, p in (("build", heavy), ("replay", heavy), ("replay", arklib), ("build", ()))] == \
        [R.ONE_JOB_GIB, 0, R.REPLAY_ARKLIB_GIB, 0]
    one, per = R.ONE_JOB_GIB, R.PER_JOB_GIB
    assert [R.build_jobs(g, 64) for g in (1, one, one + per - 0.1, one + per, R.BUILD_HEAVY_GIB)] == [1, 1, 1, 2, 64]
    assert R.build_jobs(R.BUILD_HEAVY_GIB - 0.1, 2) == 2 and R.build_jobs(R.BUILD_HEAVY_GIB - 0.1, 64) < 64
    t = R.REPLAY_PER_THREAD_GIB
    assert [R.replay_threads(g, 35) for g in (t / 2, t, 3 * t, 100 * t)] == [1, 1, 3, 35]


def test_a_run_that_declares_nothing_plans_by_what_the_machine_lets_it_hold():
    """A check slot says its GiB and a `research run --mem-gb` run its GB; what a run reserved is the least of those and
    what the machine lets it hold, which is all a run that declares none has, a research run's included: its cgroup's cap,
    else the machine's memory, named as such. Anywhere one of them is no number, nothing."""
    names = "CHECK_SLOT_MEM_GB, CHECK_RUN_MEM_GB, RESEARCH_RUN_MEM_GB"
    assert R.reserved({}, 500.0, "this machine's memory") == (500.0, f"none of {names} declared, so this machine's memory")
    assert R.reserved({"RESEARCH_RUN_ID": "r1"}, 64.0, "its cgroup's cap") == \
        (64.0, f"none of {names} declared, so its cgroup's cap")
    assert R.reserved({"CHECK_SLOT_MEM_GB": "64"}, 500.0, "m") == (64, "CHECK_SLOT_MEM_GB=64, in GiB")
    assert R.reserved({"RESEARCH_RUN_MEM_GB": "93"}, 500.0, "m") == (R.gib(93), "RESEARCH_RUN_MEM_GB=93, in GB")
    assert R.reserved({"CHECK_RUN_MEM_GB": "258", "CHECK_SLOT_MEM_GB": "64", "RESEARCH_RUN_ID": "r"}, 500.0, "m")[0] == 64
    assert R.reserved({"RESEARCH_RUN_MEM_GB": "93"}, 40.0, "its cgroup's cap") == (40.0, "its cgroup's cap")
    assert R.reserved({"RESEARCH_RUN_ID": "r1", "CHECK_SLOT_MEM_GB": "x"}, 500.0, "m")[0] is None
    assert R.reserved({"CHECK_RUN_MEM_GB": "93GB", "CHECK_SLOT_MEM_GB": "64"}, 500.0, "m") == \
        (None, "CHECK_RUN_MEM_GB='93GB' is no number of GB"), "a reservation it can't read is none, never the machine"


def test_a_step_warns_only_where_its_peak_would_give_a_larger_figure():
    """A figure is its step's largest peak with `MARGIN` above it, to the nearest GiB, so every peak a figure can be set
    from (up to the one that rounds to it from below) passes without a warning, and a larger one warns, naming the line of
    `reserve.json` to raise. A warning is the run's own (`check.py`'s `peaked` logs it): what a kept pass holds of a
    report's build (`BUILT`) has none."""
    assert "warnings" not in LA.BUILT
    lines = DATA.read_text().splitlines()
    for step, pins, name in (("replay", {"Arklib"}, "REPLAY_ARKLIB_GIB"), ("build", {"mathlib"}, "BUILD_HEAVY_GIB"),
                             ("build", (), "LIGHT_GIB")):
        assert lines[int(R.where(name).rsplit(":", 1)[1]) - 1].strip().startswith(f'"{name}":'), name
        n = R.FIGURES[name]
        top = (n + 0.5) / (1 + R.MARGIN)
        assert R.warning(step, pins, top - 0.01) is None and R.warning(step, pins, n / (1 + R.MARGIN)) is None, name
        w = R.warning(step, pins, top + 0.01)
        assert w.startswith(f"{step}: its peak, {top + 0.01:g} GiB, with {R.MARGIN:.0%} above it passes the {n} GiB"), w
        assert f"raise {name} at {R.where(name)}," in w


def test_the_readme_s_commands_ask_what_the_figures_give():
    """The README's records run asks a heavy build's need and its full record an ArkLib replay's, each in GB on the
    command and in GiB on the line under it, so a figure raised without its commands fails."""
    text = (TOOL / "README.md").read_text()
    for tool, n in (("lean_changed.py --records", R.BUILD_HEAVY_GIB), ("check.py --build --update", R.REPLAY_ARKLIB_GIB)):
        m = re.search(r"^research run .*--mem-gb (\d+) .*\n.*" + re.escape(tool) + r".*\n(.*)$", text, re.M)
        assert m and int(m[1]) == R.gb(n), tool
        assert m[2].strip().startswith(f"{R.gb(n)} GB is {R.gib(R.gb(n)):.1f} GiB"), m[2]
