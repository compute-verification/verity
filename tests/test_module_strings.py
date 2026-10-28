"""Every dotted module name the tree writes as a string resolves to a module of this tree, so a package that moves or is renamed
can't leave a dangling `importlib.import_module("...")`, `python -m ...`, entry point or `module:attr` registry spec behind.

A name is this tree's when its top level is a package a workspace member declares (`suites.importable`) or a directory the
members live under (`benchmarks.tc_probe.tool`, as `research run --tool` loads by file from the repository root); any other top
level is a third party's and is skipped. Names resolve by file, without importing anything: a module is `<path>.py`, or a
directory (a package; `python -m` needs its `__main__.py`), and an attribute is a name the module's file binds at top level.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

#: the files scanned: code, scripts, and the data a tool reads a spec from
SCANNED = ("*.py", "*.sh", "*.bash", "*.toml", "*.json", "*.yaml", "*.yml", "*.cfg", "*.ini")

#: what records history rather than naming code that must exist now; nothing here is read to find code to run
HISTORY_PATHS = (
    "fixtures/**",                 # shared fixtures: bytes of past runs (`fixtures/artifacts.json`)
    "**/fixtures/**",              # a suite's fixtures: captures and the records of the runs that made them
    "**/*vectors*.json",           # pinned vectors: what a run produced, by the names it had then
    "**/ledger/**",                # the frozen overhead ledger
    "**/CHANGELOG*",
    "tools/move/*_map.toml",       # each move's map: the names it gave, which a later move may have renamed
)
#: JSON keys whose values record how a past run was invoked (its argv), not code a tool resolves now
HISTORY_KEYS = frozenset({"argv", "command", "cmd", "invocation", "provenance", "history"})

#: (file, the string as written: `module` or `module:attr`) that don't resolve and aren't meant to, each with why
EXEMPT = {
    # the reuse probe's candidates: names a design document left unstated, tried one by one on the pod
    ("integrations/vllm/tests/program/test_artifact_applicability_independent.py", "verity_vllm.check.artifact:check_applicability"):
        "probed",
    ("integrations/vllm/tests/program/test_artifact_applicability_independent.py", "verity_vllm.check.artifact:load_artifact"):
        "probed",
    ("integrations/vllm/tests/program/test_artifact_applicability_independent.py", "verity_vllm.check.applicability:check"):
        "probed",
    ("integrations/vllm/tests/program/test_artifact_applicability_independent.py", "verity_vllm.pipeline.build:check_applicability"):
        "probed",
    # a conformance row's record of how a NaN scan was made; the module has since gone
    ("integrations/vllm/verity_vllm/program/frontend/conformance.py", "verity_vllm.check.store_nan_scan"): "history",
    # the tool specs the preflight tests feed it, which name no tool on purpose
    ("tools/research/tests/test_remote_preflight.py", "tools.fake.tool:FAKE"): "a stand-in",
    ("tools/research/tests/test_remote_preflight.py", "tools.other.tool:X"): "a stand-in",
    # a tool declaration the move guards find in a synthetic tree, under its pre-move or post-move path
    ("tools/research/tests/test_store_prov.py", "tools.agree.tool:AGREE"): "a stand-in",
    # the shell line the one-CLI lint must pass over: a `-m` in a comment
    ("integrations/vllm/tests/lint/test_p06_one_cli.py", "verity_vllm.not_code"): "a stand-in",
}

#: packages whose modules a `sys.meta_path` finder in the package makes on import, by the names it accepts; importing one to
#: ask would load torch, so its grammar is restated here (`generic._MODULE_RE`, `dense_generic.role_of_module`) and
#: `test_the_generated_modules_finder_is_still_there` checks the finder is
GENERATED = {"verity_vllm.observe.fold.profiles": re.compile(r"(derived_[A-Z0-9_]+?(_tp\d+)?(_ovs)?|dense_[A-Z0-9_]+)")}

_NAME = r"[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*"
#: (kind, pattern with groups `mod` and optionally `attr`)
PATTERNS = (
    ("import", re.compile(rf"""(?:importlib\.)?import_module\(\s*(["'])(?P<mod>{_NAME})\1""")),
    # an interpreter (`python3`, `"$PY"`, `$UV`, `uv run`) with its flags, then `-m NAME`; a name trailing off in `...` is prose
    ("main", re.compile(rf"""(?:\bpython[\d.]*|\$\{{?\w+\}}?"?|\buv run)\s+(?:-[A-Za-z]+\s+)*-m\s+(?P<mod>{_NAME})(?![\w.]*\.\.\.)""")),
    ("main", re.compile(rf"""(["'])-m\1\s*,\s*(["'])(?P<mod>{_NAME})\2""")),
    ("attr", re.compile(rf"""(["'])(?P<mod>{_NAME}):(?P<attr>{_NAME})\1""")),
)


def _suites():
    spec = importlib.util.spec_from_file_location("suites_for_module_strings", ROOT / "tools" / "verity" / "check" / "suites.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod             # its dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


@cache
def roots() -> dict[str, Path]:
    """{top-level name: the directory its modules live in}: each member's declared packages, and the directories the members
    live under, by which a tool spec names a module from the repository root."""
    S = _suites()
    pkgs = S.members()
    out: dict[str, Path] = {}
    for d in pkgs.values():
        for p in S._declared(d, S._pyproject(d)) or []:
            out[Path(p).name] = ROOT / d / p
    for d in pkgs.values():
        top = d.split("/")[0]
        out.setdefault(top, ROOT / top)
    return out


def _module_file(name: str) -> Path | None:
    """The file or package directory `name` is, or None."""
    top, *rest = name.split(".")
    base = roots()[top].joinpath(*rest)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return base if base.is_dir() else None


@cache
def _top_level_names(path: Path) -> frozenset[str]:
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.If, ast.Try)):
            out |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            out |= {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            out |= {(a.asname or a.name).split(".")[0] for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names}
    return frozenset(out)


def dangling(kind: str, mod: str, attr: str | None) -> str | None:
    """Why `mod` (with `attr`) doesn't resolve in this tree, or None when it does or isn't this tree's."""
    if mod.split(".")[0] not in roots():
        return None
    parent, _, leaf = mod.rpartition(".")
    if parent in GENERATED and GENERATED[parent].fullmatch(leaf) and attr is None and kind == "import":
        return None
    f = _module_file(mod)
    if f is None:
        return "no such module"
    if kind == "main" and f.is_dir() and not (f / "__main__.py").is_file():
        return "a package without __main__.py"
    if attr is not None:
        src = f / "__init__.py" if f.is_dir() else f
        if not src.is_file() or attr.split(".")[0] not in _top_level_names(src):
            return f"{src.relative_to(ROOT)} binds no {attr.split('.')[0]}"
    return None


def _history(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(path, g.removeprefix("**/")) for g in HISTORY_PATHS)


def _json_strings(value, key: str | None = None):
    if key in HISTORY_KEYS:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _json_strings(v, key)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _json_strings(v, k)


def _texts(path: str) -> list[str]:
    text = (ROOT / path).read_text(errors="replace")
    if path.endswith(".json"):
        try:
            return list(_json_strings(json.loads(text)))
        except json.JSONDecodeError:
            return [text]
    return [text]


def candidates() -> list[str]:
    """The tracked files that may write one of the forms with one of this tree's top-level names (git grep, so the scan reads
    only those)."""
    names = "|".join(sorted(roots()))
    forms = ["import_module\\(", f"-m[\"',[:space:]]+({names})\\b", f"[\"']({names})(\\.[A-Za-z_0-9]+)*:[A-Za-z_]"]
    out = subprocess.run(["git", "grep", "-lE", *(a for f in forms for a in ("-e", f)), "--", *SCANNED], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return [p for p in out if not _history(p) and p != SELF]   # this file's strings are examples


@cache
def scan() -> tuple[tuple[str, str, str, str | None], ...]:
    """(file, kind, module, attr) for every module string of this tree the scanned files write, exemptions aside."""
    out = set()
    for path in candidates():
        for text in _texts(path):
            for kind, pattern in PATTERNS:
                for m in pattern.finditer(text):
                    attr = m.groupdict().get("attr")
                    if m["mod"].split(".")[0] in roots() and (path, m["mod"] + (f":{attr}" if attr else "")) not in EXEMPT:
                        out.add((path, kind, m["mod"], attr))
    return tuple(sorted(out, key=str))


def found() -> list[tuple[str, str, str, str | None, str]]:
    """(file, kind, module, attr, why) for each dangling module string."""
    return [(p, k, m, a, why) for p, k, m, a in scan() if (why := dangling(k, m, a))]


def test_every_module_string_resolves_in_this_tree():
    bad = found()
    assert not bad, "dangling module strings:\n" + "\n".join(
        f"  {p}: {k} {m}{':' + a if a else ''} ({why})" for p, k, m, a, why in bad)


def test_the_exemptions_are_current():
    """Each exempt string is still written where it's listed, and still doesn't resolve: one that a fix made resolvable, or
    whose file dropped it, leaves the list."""
    stale = []
    for path, spec in EXEMPT:
        mod, _, attr = spec.partition(":")
        if spec not in (ROOT / path).read_text() or dangling("attr" if attr else "import", mod, attr or None) is None:
            stale.append((path, spec))
    assert not stale, stale


def test_the_generated_modules_finder_is_still_there():
    for pkg in GENERATED:
        assert "sys.meta_path" in (_module_file(pkg) / "__init__.py").read_text(), f"{pkg} no longer installs its finder"


def test_the_scan_sees_each_kind_and_catches_a_dangling_one():
    """The names come from the members' build configuration, so a renamed package keeps being checked; and each form is found."""
    assert {"verity", "verity_vllm", "research", "circuit_check", "tools", "benchmarks"} <= set(roots())
    assert dangling("attr", "research.cli", "main") is None
    assert dangling("attr", "benchmarks.tc_probe.tool", "TC_PROBE") is None
    assert dangling("main", "research", None) is None
    assert dangling("import", "verity.ml.circuits.ops.boolean.gemm", None) is None
    assert dangling("import", "verity.ml.circuits.ops.no_such_module", None) == "no such module"
    assert dangling("attr", "research.cli", "no_such_attr") == "tools/research/src/research/cli.py binds no no_such_attr"
    assert dangling("main", "verity.ml.circuits.ops", None) == "a package without __main__.py"
    assert dangling("import", "numpy.linalg", None) is None
    text = ('importlib.import_module("verity.gone")\n"python -m verity_vllm.gone"\n[sys.executable, "-m", "research.gone"]\n'
            'x = "circuit_check.gone:main"\n')
    hits = {(k, m["mod"]) for k, p in PATTERNS for m in p.finditer(text)}
    assert hits == {("import", "verity.gone"), ("main", "verity_vllm.gone"), ("main", "research.gone"),
                    ("attr", "circuit_check.gone")}
    seen = {(p, k, m + (f":{a}" if a else "")) for p, k, m, a in scan()}
    registry = "tools/research/src/research/store/tools_registry.py"
    assert {(registry, "attr", "tools.verity.check.tool:CHECK"), ("tools/research/pyproject.toml", "attr", "research.cli:main"),
            ("integrations/vllm/tests/ops/test_epoch_row.py", "main", "verity_vllm.pipeline.cli"),
            ("integrations/vllm/tests/engine/test_cov_difr_b0.py", "import", "verity_vllm.observe.fold.profiles.derived_B0_ovs"),
            ("benchmarks/numerical/python/verity_numerical/bench/drivers/c_interactive.py", "main", "verity_flock.register")} <= seen
