"""The network warden's Python move (the repository plan's first Python move), as a script that reruns on a rebased main.

    python3 tools/migrate/warden.py --dry-run     every move, edit and new file, checked against this tree; changes nothing
    python3 tools/migrate/warden.py               do them on a clean tree, then `uv lock`

Where each file goes, and why, is in `MOVES`:
- the warden, its audit, commitment, capacity counts and active enforcer are what the timing bounds rest on, so they go to
  `verity/protocols/accounting/communication/warden` (the plan's Layout), with their tests, `README.md` and `PROTOCOL.md`;
- `calibration.py` builds the parameters (r, Σ_sync, the ingress grid) from honest traces: a catalog builder, so a tool
  (`tools/catalog/warden`, imported as `warden_calibration`). No guarantee reads its `choose_syncs`, the honest prover's
  sync choice: the bounds hold for any declared sync in Σ_sync;
- the Lean package stays at `protocols/network_warden/lean` (its split is @lean's), and so does every path it and `check`
  read: `lean-deps.json` keys it by that path, and its lock's `runs.NetTimingDifftest` runs `scripts/difftest_vectors.py`
  from it, importing `verity_network_warden` from the checkout;
- `benchmarks/network_traces` stays: preserved run records relaunch `benchmarks/network_traces/run.sh`, `record.py` and
  `replay.py` by those paths.

Import names don't follow the directories: `verity_network_warden` keeps its name, so the lock's `generate` command and
the preserved jobs that import it are unchanged. No digest, vector or record holds a module name: the commitment's tags
are the literal strings `verity-network-warden/row/v1` and `/record/v1`, and nothing here touches them.

Every edit names the text it replaces and how often that text occurs; when the count differs (a main that moved under
the script), it stops before changing anything and names the edit. After the moves, a mention of an old path anywhere in
the tracked tree, outside the Lean package, is an error (`LEFTOVER`); the Lean package's own mentions are listed for @lean.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = "protocols/network_warden"
W = "verity/protocols/accounting/communication/warden"
CAL = "tools/catalog/warden"
LEAN = f"{OLD}/lean"
PKG = f"{OLD}/verity_network_warden"
TESTS = f"{OLD}/tests"

TCB = "what the warden's timing bounds rest on"
MOVES: list[tuple[str, str, str]] = [
    (f"{PKG}/__init__.py", f"{W}/verity_network_warden/__init__.py", "the package"),
    (f"{PKG}/grid.py", f"{W}/verity_network_warden/grid.py", f"the warden's release rule: {TCB}"),
    (f"{PKG}/schedule.py", f"{W}/verity_network_warden/schedule.py", f"the Program's schedule and sync advice: {TCB}"),
    (f"{PKG}/audit.py", f"{W}/verity_network_warden/audit.py", "the verifier's check: an accepted run is what the bounds cover"),
    (f"{PKG}/commitment.py", f"{W}/verity_network_warden/commitment.py", "the published record and its digest, which the audit checks"),
    (f"{PKG}/capacity.py", f"{W}/verity_network_warden/capacity.py", "the bit bounds and K charge a result reports, as the Lean pins count them"),
    (f"{PKG}/active.py", f"{W}/verity_network_warden/active.py", "the device-side enforcer, which lives in its protocol (ruling of 12:51 PM PDT)"),
    (f"{PKG}/calibration.py", f"{CAL}/warden_calibration.py", "builds the parameters from honest traces: a catalog builder, not TCB"),
    (f"{OLD}/pyproject.toml", f"{W}/pyproject.toml", "the distribution, beside its package"),
    (f"{OLD}/README.md", f"{W}/README.md", "the package's README"),
    (f"{OLD}/PROTOCOL.md", f"{W}/PROTOCOL.md", "the semantics and the wire format, beside the code they describe"),
    *((f"{TESTS}/test_network_warden_{t}.py", f"{W}/tests/test_network_warden_{t}.py", "a test of the TCB package, beside it")
      for t in ("grid", "audit", "commitment", "capacity", "active", "lean_difftest")),
    (f"{TESTS}/test_network_warden_calibration.py", f"{CAL}/tests/test_network_warden_calibration.py", "calibration's test, beside it"),
]

CAL_PYPROJECT = '''[project]
name = "verity-warden-calibration"
version = "0.1.0"
description = "The network warden's calibration: sizing its parameters (the egress allocation r, the clock-sync set and the ingress grid) from honest timing traces."
requires-python = ">=3.12"
dependencies = ["verity-network-warden"]

[project.scripts]
network-warden-calibrate = "warden_calibration:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
only-include = ["warden_calibration.py"]

[tool.uv.sources]
verity-network-warden = { workspace = true }

[tool.pytest.ini_options]
testpaths = ["tests"]
'''
CREATES: list[tuple[str, str, str]] = [(f"{CAL}/pyproject.toml", CAL_PYPROJECT, "calibration's distribution")]

LEAN_TEST_PARENTS = len(Path(W, "tests").parts)


@dataclass(frozen=True)
class Edit:
    path: str          # the file's path before the move
    old: str
    new: str
    why: str
    count: int = 1     # how often `old` occurs; 0 means the file is optional and the edit applies only where it exists


TEST_IMPORT = "from verity_network_warden.calibration import choose_syncs\n"
EDITS: list[Edit] = [
    # the workspace
    Edit("pyproject.toml", '"verity-network-warden", ', '"verity-network-warden", "verity-warden-calibration", ',
         "the root depends on every workspace package"),
    Edit("pyproject.toml", '"protocols/network_warden", ', f'"{W}", "{CAL}", ', "workspace members"),
    Edit("pyproject.toml", "verity-network-warden = { workspace = true }\n",
         "verity-network-warden = { workspace = true }\nverity-warden-calibration = { workspace = true }\n", "workspace sources"),
    Edit(f"{OLD}/pyproject.toml", "the exact audit and the calibration of honest runs", "the exact audit and the active warden",
         "calibration leaves the package"),
    Edit(f"{OLD}/pyproject.toml", '[project.scripts]\nnetwork-warden-calibrate = "verity_network_warden.calibration:main"\n\n', "",
         "the calibration command moves with calibration"),
    Edit(f"{OLD}/pyproject.toml", 'verity = { workspace = true }\n',
         'verity = { workspace = true }\nverity-warden-calibration = { workspace = true }\n\n'
         "# test_network_warden_active replays the honest prover's sync choice (`warden_calibration.choose_syncs`)\n"
         '[dependency-groups]\ndev = ["verity-warden-calibration"]\n', "the active warden's test reads calibration"),
    Edit(f"{OLD}/pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
         '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n[tool.verity.tests]\n'
         "# test_network_warden_lean_difftest runs the Lean package's difftest generator and compares its output with the\n"
         "# SHA-256 the lock pins (`runs` in lean-audit.json)\n"
         f'inputs = ["{LEAN}/scripts/difftest_vectors.py", "{LEAN}/lean-audit.json"]\n',
         "the suite reads the Lean package, which is no longer under its directory"),
    Edit("benchmarks/network_traces/pyproject.toml", 'dependencies = ["verity-network-warden"]',
         'dependencies = ["verity-network-warden", "verity-warden-calibration"]', "the benchmarks import calibration"),
    Edit("benchmarks/network_traces/pyproject.toml", "verity-network-warden = { workspace = true }\n",
         "verity-network-warden = { workspace = true }\nverity-warden-calibration = { workspace = true }\n", "workspace sources"),
    # imports of calibration, now `warden_calibration`
    Edit(f"{TESTS}/test_network_warden_active.py", TEST_IMPORT, "from warden_calibration import choose_syncs\n", "calibration's import name"),
    Edit(f"{TESTS}/test_network_warden_calibration.py", "from verity_network_warden import calibration, capacity\n",
         "import warden_calibration as calibration\nfrom verity_network_warden import capacity\n", "calibration's import name"),
    Edit("benchmarks/network_traces/replay.py", TEST_IMPORT, "from warden_calibration import choose_syncs\n", "calibration's import name"),
    Edit("benchmarks/network_traces/active_replay.py", TEST_IMPORT, "from warden_calibration import choose_syncs\n", "calibration's import name"),
    *(Edit(f"benchmarks/network_traces/tests/{t}", "from verity_network_warden import calibration\n",
           "import warden_calibration as calibration\n", "calibration's import name")
      for t in ("test_replay.py", "test_active_replay.py", "test_record.py")),
    Edit("benchmarks/network_traces/record.py", "(the calibration format, `verity_network_warden.calibration`)",
         "(the calibration format, `warden_calibration`)", "calibration's import name"),
    Edit(f"{PKG}/__init__.py",
         "`audit` (the exact\ncheck, record and reject), `capacity` (the counts the Lean package pins) and `calibration` (sizing from honest runs).",
         "`audit` (the exact\ncheck, record and reject) and `capacity` (the counts the Lean package pins). Calibration (sizing from\n"
         f"honest runs) is `warden_calibration`, in `{CAL}`.", "the package's module list"),
    # the Lean package, read from its own place
    Edit(f"{TESTS}/test_network_warden_lean_difftest.py", 'LEAN = Path(__file__).resolve().parents[1] / "lean"',
         f'LEAN = Path(__file__).resolve().parents[{LEAN_TEST_PARENTS}] / "{OLD.split("/")[0]}" / "{OLD.split("/")[1]}" / "lean"',
         "the Lean package stays where it is"),
    Edit(f"{TESTS}/test_network_warden_lean_difftest.py", "`lean/scripts/difftest_vectors.py`", f"`{LEAN}/scripts/difftest_vectors.py`",
         "a path to the Lean package"),
    Edit(f"{TESTS}/test_network_warden_lean_difftest.py", "(`runs` in `lean/lean-audit.json`)", f"(`runs` in `{LEAN}/lean-audit.json`)",
         "a path to the Lean package"),
    Edit(f"{OLD}/PROTOCOL.md", "`lean/", f"`{LEAN}/", "paths to the Lean package", count=6),
    Edit(f"{OLD}/PROTOCOL.md", "## Calibration (`calibration`)", f"## Calibration (`warden_calibration`, in `{CAL}`)", "calibration's place"),
    Edit(f"{OLD}/PROTOCOL.md", "- **`test_network_warden_calibration.py`:**", f"- **`test_network_warden_calibration.py`** (in `{CAL}/tests`)**:**",
         "calibration's test's place"),
    Edit(f"{OLD}/README.md", "- the calibration of honest runs;\n", "", "calibration leaves the package"),
    Edit(f"{OLD}/README.md", "`PROTOCOL.md` has the semantics and the correspondence to the Lean package.",
         f"`PROTOCOL.md` has the semantics and the correspondence to the Lean package (`{LEAN}/`).", "the Lean package's place"),
    Edit(f"{OLD}/README.md", "Calibration, from a JSON trace of honest runs:",
         f"Calibration (`warden_calibration`, in `{CAL}`), from a JSON trace of honest runs:", "calibration's place"),
    Edit(f"{OLD}/README.md", "Tests: `uv run tools/check/suites.py verity-network-warden`.",
         "Tests: `uv run tools/check/suites.py verity-network-warden verity-warden-calibration`.", "calibration's suite"),
    # scripts that put the warden on PYTHONPATH by its directory (active_live.sh is on its branch, not yet on main)
    Edit("benchmarks/network_traces/active_live.sh", "$PWD/protocols/network_warden:", f"$PWD/{W}:$PWD/{CAL}:",
         "the warden's import root", count=0),
    # whole-tree checks keyed on the directory
    Edit("tests/test_protocol_boundaries.py",
         'ROOT = Path(__file__).resolve().parents[1] / "protocols"\n'
         'PROTOCOLS = sorted(p.name for p in ROOT.iterdir() if (p / "pyproject.toml").exists())\n',
         'REPO = Path(__file__).resolve().parents[1]\n'
         'ROOT = REPO / "protocols"\n'
         '#: each protocol\'s directory, by the name its package imports as (`verity_<name>`): under protocols/, and under\n'
         '#: verity/protocols/ once it has moved\n'
         'DIRS = {tomllib.loads(p.read_text())["project"]["name"].removeprefix("verity-").replace("-", "_"): p.parent\n'
         '        for p in [*ROOT.glob("*/pyproject.toml"), *(REPO / "verity" / "protocols").rglob("pyproject.toml")]\n'
         '        if not any(x.startswith(".") for x in p.relative_to(REPO).parts)}\n'
         'PROTOCOLS = sorted(DIRS)\n',
         "the boundary test found protocols by globbing protocols/*: the moved warden would drop out of it silently"),
    Edit("tests/test_protocol_boundaries.py", "import sys\n", "import sys\nimport tomllib\n", "for DIRS"),
    Edit("tests/test_protocol_boundaries.py", '    assert "sampled_proofs" in PROTOCOLS\n',
         '    assert {"sampled_proofs", "network_warden"} <= set(PROTOCOLS)\n', "the moved warden is still checked"),
    Edit("tests/test_protocol_boundaries.py", "    base = ROOT / name\n", "    base = DIRS[name]\n", "a protocol's directory"),
    Edit("tests/test_protocol_boundaries.py", 'offenders = [f"{path.relative_to(ROOT)}:', 'offenders = [f"{path.relative_to(REPO)}:',
         "a protocol's directory"),
    Edit("tools/research/pyproject.toml", '"protocols/network_warden/pyproject.toml",',
         f'"{W}/pyproject.toml",\n          "{CAL}/pyproject.toml",', "the members' pyproject.toml files test_pythonpath reads"),
    Edit("tools/research/tests/test_pythonpath.py", '"protocols/network_warden", ', f'"{W}", "{CAL}", ', "the workspace's import roots"),
    Edit("tools/research/tests/test_pythonpath.py", '"verity_network_warden", ', '"verity_network_warden", "warden_calibration", ',
         "the workspace's import names"),
    # vLLM's pod bootstrap writes a .pth per protocols/*/pyproject.toml; the warden has none there now, and no row imports it
    Edit("integrations/vllm/tests/ops/test_pod_bootstrap.py", '{"network_warden", "pous", "pouw", "sampled_proofs"}',
         '{"pous", "pouw", "sampled_proofs"}', "the bootstrap's protocols/* glob"),
    Edit("integrations/vllm/tests/ops/test_pod_bootstrap.py",
         '"verity-pouw.pth": f"{protocols / \'pouw\'}\\n",\n        "verity-network-warden.pth": f"{protocols / \'network_warden\'}\\n"}',
         '"verity-pouw.pth": f"{protocols / \'pouw\'}\\n"}', "the bootstrap's protocols/* glob"),
    # the docs
    Edit("AGENTS.md",
         " `protocols/network_warden` (`verity_network_warden`) is network transparency's timing\n"
         " channel: the warden, its audit and calibration, and its Lean package (`lean/`).\n",
         f" `{W}` (`verity_network_warden`) is network transparency's\n"
         " timing channel: the warden, its audit and its active enforcer; its calibration is\n"
         f" `{CAL}` (`warden_calibration`), and its Lean package `{LEAN}/`.\n", "the map"),
    Edit("README.md", "- [`protocols/network_warden/`](protocols/network_warden/README.md)", f"- [`{W}/`]({W}/README.md)", "a link"),
    Edit("README.md", "[the network warden](protocols/network_warden/PROTOCOL.md)", f"[the network warden]({W}/PROTOCOL.md)", "a link"),
    Edit(".agents/skills/using-slack/registry.json", "the network warden (protocols/network_warden)",
         f"the network warden ({W}, {CAL}, {LEAN}, benchmarks/network_traces)", "network-accounting's paths"),
]

#: an old path (not the Lean package, which stays), or calibration's old import name
LEFTOVER = re.compile(r"protocols/network_warden(?!/lean)|verity_network_warden[./:]calibration|verity_network_warden import calibration")
#: files that name the old paths on purpose: this script, and the lockfile `uv lock` rewrites
LEFTOVER_OK = {"tools/migrate/warden.py", "uv.lock"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def destination(path: str) -> str:
    return next((d for s, d, _ in MOVES if s == path), path)


def plan() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """{post-move path: (text before, text after)} for every edited file, and the problems that stop the script."""
    problems, out = [], {}
    tracked = set(git("ls-files").splitlines())
    for src, dst, _ in MOVES:
        if src not in tracked:
            problems.append(f"move: {src} is not tracked")
        if (ROOT / dst).exists():
            problems.append(f"move: {dst} exists")
    for path, _, _ in CREATES:
        if (ROOT / path).exists():
            problems.append(f"create: {path} exists")
    for e in EDITS:
        p = ROOT / e.path
        if not p.exists():
            if e.count:
                problems.append(f"edit {e.path}: missing ({e.why})")
            continue
        dst = destination(e.path)
        before, text = out.get(dst, (p.read_text(), p.read_text()))
        n = text.count(e.old)
        if n != (e.count or 1):
            problems.append(f"edit {e.path}: {e.old[:70]!r} occurs {n} times, expected {e.count or 1} ({e.why})")
            continue
        out[dst] = (before, text.replace(e.old, e.new))
    return out, problems


def leftovers(edited: dict[str, tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Mentions of an old path in the tree as the moves and edits leave it: (errors, the Lean package's own, for @lean)."""
    moved = {s: d for s, d, _ in MOVES}
    errors, lean = [], []
    for path in git("ls-files").splitlines():
        post = moved.get(path, path)
        if post in LEFTOVER_OK or not (ROOT / path).is_file():
            continue
        try:
            text = edited[post][1] if post in edited else (ROOT / path).read_text()
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if LEFTOVER.search(line):
                (lean if post.startswith(LEAN + "/") else errors).append(f"{post}:{i}: {line.strip()[:150]}")
    return errors, lean


def show(edited: dict[str, tuple[str, str]]) -> None:
    print(f"## moves ({len(MOVES)})")
    for src, dst, why in MOVES:
        print(f"git mv {src} {dst}\n    # {why}")
    print(f"\n## new files ({len(CREATES)})")
    for path, _, why in CREATES:
        print(f"{path}    # {why}")
    applied = [e for e in EDITS if (ROOT / e.path).exists()]
    print(f"\n## edits ({len(applied)} in {len(edited)} files; {len(EDITS) - len(applied)} skipped, their file absent)")
    for e in EDITS:
        print(f"- {destination(e.path)}: {e.why}" + ("" if (ROOT / e.path).exists() else " (absent here: skipped)"))
    print("\n## diffs")
    for path, (before, after) in sorted(edited.items()):
        sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{path}", f"b/{path}"))
    print("\n## then\nuv lock    # the workspace's members moved")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print every move, edit and new file, and check them; change nothing")
    a = ap.parse_args()
    edited, problems = plan()
    errors, lean = leftovers(edited)
    problems += [f"leftover: {x}" for x in errors]
    if a.dry_run:
        show(edited)
    if lean:
        print(f"\n## old paths in the Lean package, for @lean's split ({len(lean)})", *lean, sep="\n")
    if problems:
        print(f"\nwarden.py: refused ({len(problems)}):", *problems, sep="\n  ", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("warden.py: the tree has uncommitted changes", file=sys.stderr)
        return 1
    for src, dst, _ in MOVES:
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        git("mv", src, dst)
    for path, text, _ in CREATES:
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / path).write_text(text)
        git("add", path)
    for path, (_, after) in edited.items():
        (ROOT / path).write_text(after)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    git("add", "-A", "--", *sorted({*edited, "uv.lock"}))
    print(f"warden.py: {len(MOVES)} moves, {len(CREATES)} new files and edits in {len(edited)} files, and uv.lock, staged. "
          "Next: `uv sync --all-packages --extra torch-cpu`, then the suites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
