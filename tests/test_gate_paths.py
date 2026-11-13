"""Every path a gate, a suite or a tool declaration names resolves in this tree.

A stale path in one of these doesn't fail anything by itself: a `merge_requires` prefix or a queue grant that matches no file
stops requiring its step or its grant, a Tool's closure glob that matches nothing leaves its identity weaker, and the queue
skips a Lean build whose package isn't there. So a directory move that misses one fails here instead. Nor does a file a Lean
audit reads (`lean_audit.inputs`) that the queue's loop doesn't count as a Lean input (`vqueue_slim.lean_inputs`): a candidate
changing only it takes its base's verdict on lean-audit, so that fails here too.

Paths resolve against `git ls-files`: a tracked file, or a directory holding one. What a gate names that is never tracked
(build output, a file a shipped tree is given, a directory of entry files that holds none yet) is in `EXEMPT` with its
reason, or is gitignored. A suite's inputs may also name a fixture `fixtures/artifacts.json` registers for fetching.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (file, path as written): never tracked, each with why
EXEMPT = {
    ("tools/verity/check/pyproject.toml", "READY.json"): "the commit record check.py's shipped tree is given (`check.py`), never tracked",
    ("tools/research/src/research/queue.py", "tools/verity/lean/modules.py"):
        "the Lean module system's codemod, which lands with the port's tooling: a tree without it is one `ported` leaves as it is; "
        "the tooling drops this entry",
    ("tools/research/src/research/queue.py", "tools/verity/lean/headers.py"):
        "the Lean header reader the codemod reads with, which lands with it: `on_modules` reads a tree's roots with it; the "
        "tooling drops this entry",
}
#: the gate tooling whose module-level path constants are checked: what `check`, `research merge`, the queue and the audit read
TOOLING = ("tools/verity/check/", "tools/verity/lean/", "tools/research/src/research/", "tools/verity/circuit_check/src/")


@functools.cache
def tracked() -> frozenset[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return frozenset(p for p in out.split("\0") if p)


@functools.cache
def directories() -> frozenset[str]:
    return frozenset(f.rsplit("/", i)[0] for f in tracked() for i in range(1, f.count("/") + 1))


def resolves(rel: str, files: frozenset[str] | None = None) -> bool:
    """`rel` is a tracked file or a directory holding one."""
    rel = rel.rstrip("/")
    if rel in ("", "."):
        return True
    if files is None or files is tracked():
        return rel in tracked() or rel in directories()
    return rel in files or any(f.startswith(rel + "/") for f in files)


def ignored(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT, input="\n".join(paths) + "\n",
                       capture_output=True, text=True)
    return set(r.stdout.split())


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rule_dangling(rule: list[str], files: frozenset[str]) -> list[str]:
    """The prefixes of a `merge_requires`-syntax rule (path prefixes, `!` patterns) that no file of ``files`` counts under."""
    from research.merge import rule_touches

    negations = [r for r in rule if r.startswith("!")]
    return [p for p in rule if not p.startswith("!") and not rule_touches([p, *negations], sorted(files))]


def rules() -> dict[str, list[str]]:
    """Every path rule a gate reads: `ci.toml`'s and `tool.py`'s `merge_requires`."""
    declaration = _module("gate_paths_declaration", ROOT / "tools" / "verity" / "check" / "declaration.py")
    tool = _module("gate_paths_check_tool", ROOT / "tools" / "verity" / "check" / "tool.py")

    out = {f"tools/verity/check/ci.toml job {j.name}": list(j.merge_requires) for j in declaration.load() if j.merge_requires}
    out |= {f"tools/verity/check/tool.py CHECK.merge_requires[{k!r}]": list(v) for k, v in tool.CHECK.merge_requires.items()}
    return out


def test_every_merge_requirement_matches_tracked_files():
    found = rules()
    assert any("ci.toml" in k for k in found), found
    bad = {k: d for k, rule in found.items() if (d := rule_dangling(rule, tracked()))}
    assert not bad, f"these gates name path prefixes no tracked file is under, so they would never require their step: {bad}"


def _suite_tables() -> dict[str, dict]:
    out = {}
    for f in sorted(p for p in tracked() if p.endswith("pyproject.toml")):
        t = tomllib.loads((ROOT / f).read_text()).get("tool", {}).get("verity", {}).get("tests")
        if t:
            out[f] = t
    return out


@functools.cache
def registered() -> frozenset[str]:
    """The paths `fixtures/artifacts.json` registers. A fetched one is never tracked, and `suites.py` keys a suite whose inputs
    cover it on its id."""
    doc = json.loads((ROOT / "fixtures" / "artifacts.json").read_text(encoding="utf-8"))
    return frozenset(v.rstrip("/") for k, v in doc.items() if k.startswith("art:") and isinstance(v, str))


def test_every_suite_input_and_artifact_is_in_this_tree():
    """A suite's inputs key its pass cache: one that isn't there keys nothing, and a test that skips on a missing file then
    passes with nothing checked. A registered fixture, or a path inside one, keys its id. An artifact is a Lake build's
    output, so its package must be here."""
    tables = _suite_tables()
    assert "verity/circuits/pyproject.toml" in tables
    bad = [(f, i) for f, t in tables.items() for i in t.get("inputs", []) if (f, i) not in EXEMPT and not resolves(i)
           and not any(i.rstrip("/") == r or i.startswith(r + "/") for r in registered())]
    bad += [(f, a) for f, t in tables.items() for a in t.get("artifacts", [])
            if not (ROOT / a.split("/.lake/")[0] / "lakefile.toml").is_file()]
    assert not bad, f"declared suite inputs or artifacts that aren't in this tree: {bad}"


def test_the_queues_lean_builds_are_lake_packages_here():
    """`quick_tier.tier` skips a `--lean` directory without a lakefile, and the tests that need its build then skip."""
    from research import queue

    assert queue.LEAN_BUILDS
    assert not (bad := [d for d in queue.LEAN_BUILDS if f"{d}/lakefile.toml" not in tracked()]), bad


#: the files the test names first: every file `lean_audit.inputs` declares on `main` that `vqueue_slim.LEAN_INPUTS`' globs and
#: directories don't cover, the task graph's contract, which it names, and files of the Lean packages that only their
#: directories cover (a module's `include_str` target, check scripts, difftest vectors and the scripts that write them)
NON_LEAN_READS = (
    "tools/verity/graph/verity_graph/contract.py",
    "verity/commitments/lms/lean/scripts/difftest_vectors.py",
    "verity/commitments/lms/lean/scripts/sign_vectors.py",
    "verity/commitments/randomness/__init__.py",
    "verity/commitments/randomness/tests/vectors.json",
    "verity/commitments/randomness/tests/vectors.py",
    "verity/protocols/certifier/device/lean/scripts/difftest_vectors.py",
    "verity/protocols/certifier/device/lean/scripts/issuer_vectors.py",
    "verity/protocols/certifier/device/program/lean/scripts/charge_vectors.py",
    "verity/protocols/certifier/device/program/lean/scripts/difftest_vectors.py",
    "verity/protocols/certifier/device/program/lean/scripts/window_vectors.py",
    "verity/protocols/certifier/verifier/lean/scripts/difftest_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/baseline_difftest_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/count_difftest_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/desc_difftest_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/desc_fuzz_programs.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/desc_liveness_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/difftest_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/mistyped_vectors.py",
    "verity/protocols/inference_only/verifier/nci/lean/scripts/nci_reference.py",
    "verity/protocols/poub/verifier/lean/tests/ra_cases.py",
    "verity/protocols/poub/verifier/lean/tests/replay.py",
    "verity/protocols/pous/device/lean/tests/mc_cases.py",
    "verity/protocols/pous/device/lean/tests/wipe_cases.py",
    "verity/protocols/pous/device/lean/tests/wipe_protocol_case.py",
    "verity/protocols/pous/device/lean/verdict.py",
    "verity/protocols/pous/device/lean/wipe_verdict.py",
    "verity/protocols/pous/lean/TRUSTED.sha256",
    "verity/protocols/pous/lean/check.sh",
    "verity/protocols/pous/lean/grade.sh",
    "verity/protocols/pous/verifier/check/interop.py",
    "verity/protocols/pous/verifier/check/scripts/difftest_vectors.py",
    "verity/protocols/pous/verifier/check/timed_audit.py",
    "verity/protocols/pouw/verifier/lean/check/models.json",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_check_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_credit_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_live_verified_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_opening_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_pass_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_period_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_rounds_cases.py",
    "verity/protocols/pouw/verifier/lean/check/tests/pouw_verified_cases.py",
    "verity/protocols/pouw/verifier/lean/commit/tests/difftest_vectors.py",
    "verity/protocols/recursion/verifier/firewall/tests/_release.py",
    "verity/protocols/recursion/verifier/firewall/tests/fixtures/release-e2e.jsonl",
    "verity/protocols/recursion/verifier/firewall/tests/fixtures/release-index.jsonl.gz",
    "verity/protocols/recursion/verifier/firewall/tests/fixtures/release-outer.jsonl.gz",
    "verity/protocols/recursion/verifier/vstar_cli/tests/recursive_data.json",
    "verity/protocols/recursion/verifier/vstar_cli/tests/settings.py",
    "verity/protocols/remote/verifier/composition/tests/periods.py",
    "verity/protocols/remote/verifier/composition/tests/reference.py",
    "verity/protocols/spot/verifier/lean/tests/fixtures/inner-trees-served/coin_tree.json",
    "verity/protocols/spot/verifier/lean/tests/fixtures/inner-trees-served/public.json.gz",
    "verity/protocols/spot/verifier/lean/tests/fixtures/inner-trees-served/releases.jsonl",
    "verity/protocols/spot/verifier/lean/tests/plan_of_record.json",
    "verity/protocols/spot/verifier/lean/tests/settings.py",
)


def test_every_file_a_lean_audit_reads_is_a_lean_input_of_the_queues_loop():
    """The queue's loop lets a candidate built on another in flight take that one's verdict on lean-audit unless a member
    changes a Lean input of its tree (`vqueue_slim.lean_inputs`), so a change to a file an audit reads that they miss would
    land on a verdict that never read it: every file `lean_audit.inputs` declares here is one, `NON_LEAN_READS` first."""
    from research.vqueue import meets
    from research.vqueue_slim import lean_inputs

    audit = _module("gate_paths_lean_audit", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
    files = sorted(tracked())
    declared = {f for p in audit.packages(files) for f in audit.inputs(p, files)}
    assert not (gone := [f for f in NON_LEAN_READS if f not in declared]), (
        f"no audit here declares these, so they are no case of this test: drop them from NON_LEAN_READS: {gone}")
    inputs = lean_inputs(files)
    out = [f for f in (*NON_LEAN_READS, *sorted(declared.difference(NON_LEAN_READS))) if not meets((f,), inputs)]
    assert not out, f"files an audit reads that no Lean input of the queue's loop covers (`vqueue_slim.lean_inputs`): {out}"


def _path_tokens(argv: list[str]) -> list[str]:
    """The repository paths an argv names: tokens with a `/` that aren't flags, absolute paths or `$` expansions."""
    return [t for t in argv if "/" in t and not t.startswith(("-", "/", "$", "~")) and "://" not in t]


def _modules(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, t in enumerate(argv[:-1]) if t == "-m"]


def _module_resolves(module: str, files: frozenset[str], dirs: list[str]) -> bool:
    """A source file of the tree (from the root or a directory the command may run from), or an installed module."""
    rel = module.replace(".", "/")
    if any(resolves(f"{d}{rel}{s}", files) for d in ["", *(f"{d}/" for d in dirs)] for s in (".py", "/__main__.py")):
        return True
    if files is not tracked():
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def tool_problems(name: str, tool, files: frozenset[str]) -> list[str]:
    from research.store.tool import glob_re

    dirs = _cwd_dirs(tool, files)
    out = [f"closure glob {g!r} matches no tracked file" for g in tool.closure
           if not any(glob_re(g).match(f) for f in files)]
    out += [f"{field} names {p!r}, which isn't tracked" for field in ("command", "preflight", "after_land")
            for p in _path_tokens(list(getattr(tool, field)))
            if not resolves(p, files) and not any(resolves(f"{d}/{p}", files) for d in dirs)]
    out += [f"{field} runs -m {m}, which isn't a module here" for field in ("command", "preflight", "after_land")
            for m in _modules(list(getattr(tool, field))) if not _module_resolves(m, files, dirs)]
    out += [f"merge_requires[{k!r}] prefixes {d} match no tracked file" for k, rule in tool.merge_requires.items()
            if (d := rule_dangling(rule, files))]
    return out


def _cwd_dirs(tool, files: frozenset[str]) -> list[str]:
    """Directories a Tool's command may run from: the directory of its declaration (`--cwd source/backends/gkr`)."""
    spec = getattr(tool, "spec", None) or ""
    module = spec.rpartition(":")[0]
    parts = module.split(".")
    return ["/".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]


def test_every_registered_tool_is_declared_in_this_tree():
    """`research run --tool NAME` loads `REGISTRY[NAME]` from the tree it runs; a declaration that moved without its entry
    fails only when somebody runs it, and a closure glob that went stale matches nothing, silently."""
    from research.store import tools_registry

    bad = {}
    for name, spec in sorted(tools_registry.REGISTRY.items()):
        module, _, attr = spec.rpartition(":")
        try:
            tool = tools_registry._from_file(module, attr, ROOT)
        except Exception as e:  # noqa: BLE001
            bad[name] = [f"{spec} doesn't load from this tree: {type(e).__name__}: {e}"]
            continue
        tool.spec = spec
        if problems := tool_problems(name, tool, tracked()):
            bad[name] = problems
    assert not bad, bad


#: a string that reads as a repository path: a tracked top-level directory, then path segments, no globs or expansions
_PATH = re.compile(r"^[\w.-]+(/[\w.@+-]+)+/?$")
#: a format id (`verity/slack-route/v1`), which reads as a path under the tree's `verity/`
_FORMAT_ID = re.compile(r"^verity/[\w.-]+/v\d+$")


@functools.cache
def _anchors():
    return _module("gate_paths_anchors", ROOT / "tools" / "verity" / "check" / "anchors.py")


def path_constants(rel: str, text: str, tops: frozenset[str]) -> list[tuple[int, str, str]]:
    """(line, the path, how it's written) for each repository path a module-level assignment of ``rel`` names: a string
    whose first segment is one of ``tops``, or a join anchored on the file's place (`tools/verity/check/anchors.py` evaluates it)."""
    anchor = _anchors().anchor
    out, names = [], {}
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (a := anchor(node.value, rel, names)) is not None:
            for t in targets:
                if isinstance(t, ast.Name):
                    names[t.id] = a
        stack = [node.value]
        while stack:
            sub = stack.pop()
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and _PATH.match(sub.value) \
                    and not _FORMAT_ID.match(sub.value) and sub.value.split("/")[0] in tops:
                out.append((sub.lineno, sub.value, "string"))
            elif isinstance(sub, ast.BinOp) and (a := anchor(sub, rel, names)) is not None:
                if "/" in a:
                    out.append((sub.lineno, a, "join"))
            else:
                stack.extend(ast.iter_child_nodes(sub))
    return sorted(out)


def dangling_constants(files: frozenset[str], read=lambda f: (ROOT / f).read_text()) -> list[str]:
    tops = frozenset(f.split("/")[0] for f in files if "/" in f)
    found = []
    for f in sorted(p for p in files if p.endswith(".py") and p.startswith(TOOLING) and "/tests/" not in p):
        here = f.rpartition("/")[0]
        for line, path, how in path_constants(f, read(f), tops):
            if (f, path) not in EXEMPT and not resolves(path, files) and not resolves(f"{here}/{path}", files):
                found.append((f, line, path, how))
    unignored = set(p for _, _, p, _ in found) - (ignored([p for _, _, p, _ in found]) if files is tracked() else set())
    return [f"{f}:{line}: {path} ({how})" for f, line, path, how in found if path in unignored]


def test_every_path_constant_in_the_gate_tooling_resolves():
    """A module-level path the gate tooling reads (`check.py`'s `FLOCK_VERIFIER`, the queue's `RULES_BEFORE_MOVE`, research's
    `UPSTREAM`), written as a string or as a `Path(__file__).parents[N] / …` join, is a tracked path (from the root, or
    beside its module) or a gitignored one. A `parents[N]` that a move left one level off fails here."""
    assert not (bad := dangling_constants(tracked())), "\n".join(bad)


def test_the_latest_moves_new_paths_are_in_this_tree():
    """research reads a tree from before a layout move through the paths `moves.json` says it renamed: the newest move's
    new paths are this tree's, or that mapping is wrong."""
    from research.store import tools_registry

    moves = json.loads(tools_registry.MOVES.read_text())["moves"]
    assert not (bad := [new for new in (moves[-1]["paths"].values() if moves else ()) if not resolves(new)]), bad


def test_the_cluster_kinds_inputs_are_in_this_tree():
    kinds = sorted(p for p in tracked() if p.startswith("infra/cluster/kinds/") and p.endswith(".toml"))
    assert kinds
    bad = [(k, i) for k in kinds for kind in tomllib.loads((ROOT / k).read_text()).get("kinds", {}).values()
           for i in kind.get("inputs", []) if not resolves(i)]
    assert not bad, bad


def test_each_check_catches_a_dangling_path():
    """The detectors above find what a move would leave behind, on a toy tree."""
    files = frozenset({"verity/protocols/flock/prover/live/lib.rs", "verity/protocols/flock/prover/README.md", "tools/verity/check/x.py", "tools/verity/check/data/a.json"})
    assert rule_dangling(["verity/protocols/flock/prover/", "!README.md"], files) == []
    assert rule_dangling(["verity/protocols/flock/prover/", "verity/flock/", "!README.md"], files) == ["verity/flock/"]
    assert rule_dangling(["tools/verity/check/data/a.json"], files) == []

    from research.store.tool import Tool

    tool = Tool(name="t", command=["bash", "verity/ml/flock/run.sh"], closure=["verity/protocols/flock/prover/live/**", "elsewhere/**"],
                after_land=["python", "tools/verity/check/x.py"], merge_requires={"s": ["verity/flock/"]})
    assert tool_problems("t", tool, files) == ["closure glob 'elsewhere/**' matches no tracked file",
                                               "command names 'verity/ml/flock/run.sh', which isn't tracked",
                                               "merge_requires['s'] prefixes ['verity/flock/'] match no tracked file"]
    gkr = frozenset({"archive/gkr/gpu/tool.py", "archive/gkr/gpu/run.py"})
    tool = Tool(name="g", command=["python", "-m", "gpu.run"], closure=["archive/gkr/gpu/**"])
    tool.spec = "archive.gkr.gpu.tool:G"
    assert tool_problems("g", tool, gkr) == []
    tool.spec = "elsewhere.gkr.gpu.tool:G"
    assert tool_problems("g", tool, gkr) == ["command runs -m gpu.run, which isn't a module here"]

    text = ('from pathlib import Path\nROOT = Path(__file__).resolve().parents[3]\nA = ROOT / "tools" / "verity" / "check" / "x.py"\n'
            'B = ROOT / "tools" / "verity" / "lean"\nC = "verity/protocols/flock/prover/live"\nD = ("verity/gone/x.py", "data/a.json")\n'
            'E = Path(__file__).parent / "data" / "b.json"\nF = "origin/main"\n')
    got = dangling_constants(files | {"tools/verity/check/y.py"}, read=lambda f: text if f == "tools/verity/check/y.py" else "")
    assert got == ["tools/verity/check/y.py:4: tools/verity/lean (join)", "tools/verity/check/y.py:6: verity/gone/x.py (string)",
                   "tools/verity/check/y.py:7: tools/verity/check/data/b.json (join)"]
