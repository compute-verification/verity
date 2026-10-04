"""Retire superseded catalog Definition versions, and move the vLLM registry's quarantine to `experimental/`.

    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py apply [--only ID ...]
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py commands
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py snapshot OUT.json
    uv run python integrations/vllm/tools/migrate/superseded_quarantine.py verify BEFORE.json AFTER.json

The plan is note:20261004T2058Z-draft-repo-organization-principles, under Daniel's rulings of 4 Oct 2026: `experimental/` is a
top-level directory, `verity/` holds only what cited claims depend on, and a superseded catalog Definition version stays only so
that old records replay. A superseded version that no kept Program or record names may be deleted, but only after a
`circuits/<slug>` superseded approach entry citing the last commit that has it is recorded (`commands` prints those entries).

`apply` deletes exactly the versions in `DELETE`, with their circuit-check bindings (`_model_roots`) and lowering pins
(`pins.json`), and refuses any other id, any id in `KEPT`, and any deletion that would leave a reference behind. It then moves
the quarantine lanes in `MOVE` to `experimental/vllm/quarantine/verity_quarantine/<lane>` (the distribution `verity-quarantine`),
rewrites their imports, test paths and the lint and dead-code lists that name them, and registers the new workspace member.
The lanes in `STAY` remain in the core quarantine, since core binds them. Running it again changes nothing.

`snapshot` records circuit-check's task list and every registered Definition, each with its descriptor digest (sha256 of the
canonical JSON of its definition entry). `verify` compares two snapshots: the task list may lose only the deleted versions'
targets, and every other id must keep its digest.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import pkgutil
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VLLM = "integrations/vllm"
REGISTRY_DIR = f"{VLLM}/verity_vllm/program/registry"
QUARANTINE = f"{REGISTRY_DIR}/quarantine"
QUARANTINE_MODULE = "verity_vllm.program.registry.quarantine"
EXPERIMENTAL = "experimental/vllm/quarantine"
PACKAGE = "verity_quarantine"
DISTRIBUTION = "verity-quarantine"
TARGETS = "tools/circuit_check/src/circuit_check/targets.py"
PINS = "tools/circuit_check/src/circuit_check/pins.json"
#: Where Definitions are registered: a version is deleted from one of these.
DEFINITION_DIRS = (REGISTRY_DIR, "packages/verity/src/verity")
#: JSON lists keyed by a moved file or module: their entries for the moved lanes are dropped (each list only shrinks).
KEYED_LISTS = (f"{VLLM}/tests/lint/allowlists/p10_size.json", f"{VLLM}/tests/lint/allowlists/p11_names.json",
               f"{VLLM}/tests/dead_code_keep.json")
#: Never rewritten: this script, its test, and the quarantine's recorded admission records (their adapter names are history).
UNTOUCHED = (f"{VLLM}/tools/migrate/", f"{VLLM}/tests/migrate/", f"{VLLM}/verity_vllm/engine/profiles/quarantine/")


@dataclass(frozen=True)
class Retirement:
    slug: str            # the circuits/<slug> approach entry recorded as superseded
    superseded_by: str   # the circuits/<slug> of the approach that replaces it
    reason: str          # one sentence
    cite: str            # verity@<sha>: the last commit that has the version


#: The only versions `apply` deletes, by id. Empty: every superseded version reviewed is kept (`KEPT`).
DELETE: dict[str, Retirement] = {}

#: The superseded versions reviewed and kept, each with what keeps it (a Build vocabulary, a record, a capture map, a pinned
#: identity, or a kept Program that calls it). `apply` refuses to delete any of them.
KEPT: dict[str, str] = {
    "RMSNormFusedCuda_v1": "capture map integrations/vllm/manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json:243 cites it; "
                           "its divergence record (verity_vllm/program/registry/conformance.py:40, fixtures/verity-ir/rmsnorm-fused-b0); "
                           "the width-8 target spec (tests/program/test_rmsnorm_fused_target.py:32)",
    "SiluMul_v1": "the b1-eager and b1-eager-v2 vocabularies' silu_mul kind (verity_vllm/program/frontend/rules/vocab.py:106, :207); "
                  "the fold's silu_and_mul pattern (verity_vllm/observe/fold/patterns/activations.py:37); census/subcircuits.json:832",
    "SiluMulBf16_v1": "SiluMul_v1's element (verity_vllm/program/registry/prims.py:66); the word view of SiluMulBf16_v3 "
                      "(verity_vllm/program/registry/boolean_silu.py:139); D-SP1's frozen gate set (backends/sp1/common/src/gateset.rs:140)",
    "SiluMul_v3": "SiluMul_v1's Boolean form, which the Boolean switch binds for it (verity_vllm/program/registry/boolean_silu.py:148)",
    "SiluMulBf16_v3": "SiluMulBf16_v1's Boolean form, SiluMul_v3's element (verity_vllm/program/registry/boolean_silu.py:139)",
    "Attention_v1": "the b1-eager vocabulary's attention kind (verity_vllm/program/frontend/rules/vocab.py:105); "
                    "FLASH_ATTENTION_LAUNCHES (verity_vllm/query/required.py:92)",
    "AttentionHead_v1": "Attention_v1's callee (verity_vllm/program/registry/b1.py:472)",
    "AttnBlock_v1": "AttentionHead_v1's callee (verity_vllm/program/registry/b1.py:414)",
    "Serve_v1": "Family[Serve_v1], the certificate of record's program identity (verity_vllm/program/frontend/rules/family.py:487); "
                "ServeHopper_v1 and ServeFp8_v1 are ports of it (verity_vllm/program/registry/hopper.py:153, fp8.py:615)",
    "Serve_v2": "the b1-eager-v2 skeleton (verity_vllm/program/frontend/rules/vocab.py:179); the Hopper eager program "
                "(verity_vllm/program/registry/hopper.py:177); LServe_v1's skeleton (verity_vllm/program/registry/lifted.py:1362)",
    "Serve_v3": "Family[Serve_v3], the padded circuit derivation verifies against (verity_vllm/program/frontend/rules/family.py:517)",
    "LayerPre_v1": "Serve_v1's callee (verity_vllm/program/frontend/rules/family.py:545); LayerPreHopper_v1 and LayerPreFp8_v1 port it "
                   "(verity_vllm/program/registry/hopper.py:148, fp8.py:621)",
    "LayerPost_v1": "Serve_v1's callee (verity_vllm/program/frontend/rules/family.py:551); LayerPostHopper_v1 and LayerPostFp8_v1 port it "
                    "(verity_vllm/program/registry/hopper.py:149, fp8.py:622)",
    "Final_v1": "Serve_v1's callee (verity_vllm/program/registry/b1.py:646); FinalHopper_v1 ports it (verity_vllm/program/registry/hopper.py:150)",
}

#: The quarantine lanes that move: nothing in core's library code imports them.
MOVE = ("act", "collective", "dense", "ov_moe")
#: The lanes that stay, and why.
STAY = {
    "ln": "the b1-eager-v3 vocabulary and the layer_norm binding bind LayerNormAten_v1 (verity_vllm/program/frontend/rules/vocab.py:186, "
          "vllm_bindings/layer_norm.py:9), and the gen_ln eager profile binds it and GeluErf",
    "ov_sampling": "the gen_ov_sampling eager profile binds GumbelArgmaxBf16 and GumbelArgmaxTempBf16 "
                   "(verity_vllm/engine/profiles/gen_ov_sampling_patterns.py:31)",
}

#: Lines of core tests that name a moved lane by where it was: a use of a re-export of a promoted Definition imports the registry
#: module that owns it, and a lint of the lane's directory lints the experimental package.
REDIRECTS: tuple[tuple[str, str, str], ...] = tuple(
    (f"{VLLM}/tests/{f}", old, new) for f, old, new in (
        ("program/fixtures_query.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("program/_wout_prototype_20260917.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("query/test_boundary_oracle.py", "from verity_vllm.program.registry.quarantine.ov_moe import moe as Q",
         "from verity_vllm.program.registry import moe as Q"),
        ("program/test_gen_ov_moe_vu_shapes.py", "from verity_vllm.program.registry.quarantine.ov_moe.moe import MoeExpertGemm",
         "from verity_vllm.program.registry.moe import MoeExpertGemm"),
        ("collectives/test_tp_collective.py", 'rep = lint(REPO / "verity_vllm" / "program" / "registry" / "quarantine")',
         "rep = lint(Path(QC.__file__).resolve().parents[1])"),
    )) + (
    # the shipped PYTHONPATH's roots follow the root's dependencies, which now include the experimental member
    ("tools/research/tests/test_pythonpath.py", '"backends/flock/python", "tools/circuit_check/src"]',
     f'"backends/flock/python", "tools/circuit_check/src",\n              "{EXPERIMENTAL}"]'),
)

PYPROJECT = f'''[project]
name = "{DISTRIBUTION}"
version = "0.1.0"
description = "Experimental: the vLLM registry's quarantine lanes that core does not bind (provisional Definitions, the re-exports of promoted ones, their difftest adapters)."
requires-python = ">=3.12"
dependencies = ["verity", "verity-vllm"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{PACKAGE}"]

[tool.uv.sources]
verity = {{ workspace = true }}
verity-vllm = {{ workspace = true }}

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q -p no:cacheprovider"
'''

PACKAGE_DOC = f'''"""{PACKAGE} — the vLLM registry's quarantine lanes that core does not bind, moved out of
`{QUARANTINE_MODULE}` (note:20261004T2058Z-draft-repo-organization-principles): {", ".join(f"`{m}`" for m in MOVE)}.
The lanes core still binds ({", ".join(f"`{m}`" for m in STAY)}) stay in core's quarantine.

Each lane is a sub-package of provisional numerical Definitions pending admission, or of re-exports of Definitions already promoted
to `verity_vllm.program.registry`, with a `*_difftest.py` adapter for `verity-vllm properties-admission` beside each. Admission
is as in core's quarantine (`verity_vllm.properties.quarantine_lint`, `verity_vllm.properties.admission`): a body is a function of
its operands and statics only, each module names its `SEMANTIC_ID = "<lane>/<Definition>@v<k>"`, a re-derived body takes a new
version, and promotion into the registry is an integrator move. Nothing in core imports this package.
"""
'''

CORE_DOC = f'''"""{QUARANTINE_MODULE} — provisional numerical Definitions that core binds, one sub-package per lane, pending
admission: {", ".join(f"`{m}`" for m in STAY)}. Existing registered bodies (`b1.py`, `prims.py`, ...) are immutable; a lane that
needs different exact semantics adds a NEW versioned Definition and never edits an old one.

Admission (gate G4 + `verity_vllm.properties.quarantine_lint` + `verity_vllm.properties.admission`): the body is a function of its
operands and statics only; `SEMANTIC_ID = "<lane>/<Definition>@v<k>"` at module level; the lane's pattern has positive and negative
recognizer tests; `difftest produce` + `difftest check` are bit-exact on randomized and edge inputs; every specialization the run
uses is replayed. Promotion into `registry` is an integrator move.

The lanes core does not bind ({", ".join(f"`{m}`" for m in MOVE)}) are the experimental package `{PACKAGE}`
(`{EXPERIMENTAL}`).
"""
'''


class Refused(Exception):
    pass


# --- shared ------------------------------------------------------------------------------------------------------------

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def tracked(root: Path) -> list[str]:
    return [p for p in _git(root, "ls-files", "-z").split("\0") if p and (root / p).is_file()]


def family(target: str) -> str:
    """`Name_vN{...}` and `Name_vN/label` belong to `Name_vN`."""
    return re.split(r"[{/]", target, maxsplit=1)[0]


def _drop_lines(text: str, drop) -> str:
    """``text`` without the lines ``drop`` selects, and without a trailing comma left before a closing bracket."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if drop(line):
            continue
        if line.lstrip().startswith(("]", "}")) and out and out[-1].rstrip().endswith(","):
            out[-1] = out[-1].rstrip()[:-1] + "\n"
        out.append(line)
    return "".join(out)


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in ("import_module", "__import__")):
            out.add(node.args[0].value)
    return out


def library_importers(root: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Core's vLLM library files (outside the quarantine and the tests) that import a module under one of ``prefixes``."""
    out = []
    for p in sorted((root / VLLM / "verity_vllm").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(QUARANTINE + "/") or "__pycache__" in rel:
            continue
        hit = sorted(m for m in _imports(p) if any(m == x or m.startswith(x + ".") for x in prefixes))
        if hit:
            out.append(f"{rel}: {', '.join(hit)}")
    return out


# --- deleting versions -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    path: str
    start: int           # 1-based, first decorator included
    end: int             # inclusive
    names: tuple[str, ...]


def _declares(node: ast.AST, name: str, version: int) -> bool:
    calls = [d for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Call)]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
        calls.append(node.value)
    return any(len(c.args) >= 2 and isinstance(c.args[0], ast.Constant) and c.args[0].value == name
               and isinstance(c.args[1], ast.Constant) and c.args[1].value == version for c in calls)


def _bound(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return (node.name,)
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))


def find_sites(root: Path, version_id: str) -> list[Site]:
    m = re.fullmatch(r"(\w+)_v(\d+)", version_id)
    if not m:
        raise Refused(f"{version_id}: not a Definition version id (Name_vN)")
    name, version = m.group(1), int(m.group(2))
    sites = []
    for d in DEFINITION_DIRS:
        for p in sorted((root / d).rglob("*.py")) if (root / d).is_dir() else []:
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel or rel.startswith(QUARANTINE + "/"):
                continue
            for node in ast.parse(p.read_text(encoding="utf-8")).body:
                if _declares(node, name, version):
                    first = min([node.lineno, *(x.lineno for x in getattr(node, "decorator_list", []))])
                    sites.append(Site(rel, first, node.end_lineno, _bound(node)))
    return sites


def _target_entries(text: str, version_id: str) -> list[tuple[int, int]]:
    """Line spans (1-based, inclusive) of `_model_roots`' entries labelled ``version_id``, each with the comment lines above it."""
    tree = ast.parse(text)
    lines = text.splitlines()
    spans = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == "_model_roots"):
            continue
        for ret in ast.walk(fn):
            if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.List)):
                continue
            elts = ret.value.elts
            for i, e in enumerate(elts):
                if not (isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant)
                        and isinstance(e.elts[0].value, str) and family(e.elts[0].value) == version_id):
                    continue
                shared = {x.lineno for x in elts[:i] + elts[i + 1:]} | {x.end_lineno for x in elts[:i] + elts[i + 1:]}
                if set(range(e.lineno, e.end_lineno + 1)) & shared:
                    raise Refused(f"{TARGETS}:{e.lineno}: the binding of {version_id} shares a line with another entry")
                start = e.lineno
                while start > 1 and lines[start - 2].strip().startswith("#"):
                    start -= 1
                spans.append((start, e.end_lineno))
    return spans


def plan_deletions(root: Path, ids: list[str]) -> dict[str, list[tuple[int, int]]]:
    """{file: [line spans to remove]}, after checking each id is listed, located once, and referenced nowhere else: neither its
    id in any tracked file nor, in a Python file, the name it is bound to (another module's object of the same name refuses
    too; resolve such a refusal by hand)."""
    unlisted = [i for i in ids if i not in DELETE]
    if unlisted:
        raise Refused(f"not in DELETE: {', '.join(unlisted)}; only the versions listed in this script are ever deleted")
    kept = [i for i in ids if i in KEPT]
    if kept:
        raise Refused(f"listed in both DELETE and KEPT: {', '.join(kept)}")
    spans: dict[str, list[tuple[int, int]]] = {}
    names: dict[str, tuple[str, ...]] = {}
    for vid in ids:
        sites = find_sites(root, vid)
        if len(sites) != 1:
            raise Refused(f"{vid}: expected one registration, found {len(sites)}: {[f'{s.path}:{s.start}' for s in sites]}")
        s = sites[0]
        spans.setdefault(s.path, []).append((s.start, s.end))
        names[vid] = s.names
        targets = root / TARGETS
        if targets.is_file():
            spans.setdefault(TARGETS, []).extend(_target_entries(targets.read_text(encoding="utf-8"), vid))
    residue = []
    for rel in tracked(root):
        if rel.startswith(UNTOUCHED) or rel == PINS:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        removed = {n for a, b in spans.get(rel, []) for n in range(a, b + 1)}
        for n, line in enumerate(text.splitlines(), 1):
            if n in removed:
                continue
            for vid in ids:
                words = [vid, *names[vid]] if rel.endswith(".py") else [vid]
                hit = [w for w in words if re.search(rf"\b{re.escape(w)}\b", line)]
                if hit:
                    residue.append(f"{rel}:{n}: {', '.join(hit)}")
    if residue:
        raise Refused("still referenced, so not deletable (a reference that keeps it belongs in KEPT):\n  " + "\n  ".join(residue))
    return spans


def delete_versions(root: Path, ids: list[str]) -> list[str]:
    if not ids:
        return []
    spans = plan_deletions(root, ids)
    changed = []
    for rel, ss in spans.items():
        p = root / rel
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        removed = {n for a, b in ss for n in range(a, b + 1)}
        kept = [line for n, line in enumerate(lines, 1) if n not in removed]
        text = re.sub(r"\n{4,}", "\n\n\n", "".join(kept))
        if rel.endswith(".py"):
            ast.parse(text)
        p.write_text(text, encoding="utf-8")
        changed.append(rel)
    pins = root / PINS
    if pins.is_file():
        text = pins.read_text(encoding="utf-8")
        data = json.loads(text)
        if json.dumps(data, indent=1, sort_keys=True) + "\n" != text:
            raise Refused(f"{PINS} does not round-trip through json.dumps(indent=1, sort_keys=True); not rewriting it")
        before = len(data["definitions"])
        data["definitions"] = {k: v for k, v in data["definitions"].items() if family(k) not in ids}
        if len(data["definitions"]) != before:
            pins.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            changed.append(PINS)
    return changed


# --- moving the quarantine ---------------------------------------------------------------------------------------------

def _move_path(root: Path, src: str, dst: str) -> None:
    (root / dst).parent.mkdir(parents=True, exist_ok=True)
    _git(root, "mv", src, dst)


def _rewrite(text: str) -> str:
    text = re.sub(rf"from {re.escape(QUARANTINE_MODULE)} import ([^\n]*)", _from_lanes, text)
    return re.sub(rf"\b{re.escape(QUARANTINE_MODULE)}\.({'|'.join(MOVE)})\b", rf"{PACKAGE}.\1", text)


def _from_lanes(m: re.Match) -> str:
    names = m.group(1)
    imported = {n.strip().split(" as ")[0].strip("() ") for n in names.split("#")[0].split(",")} - {""}
    if not imported & set(MOVE):
        return m.group(0)
    if not imported <= set(MOVE):
        raise Refused(f"`{m.group(0)}` imports moved and staying lanes (or a multi-line list): split it by hand")
    return f"from {PACKAGE} import {names}"


def _keyed_by_moved(line: str) -> bool:
    paths = tuple(f'"{QUARANTINE.removeprefix(VLLM + "/")}/{m}/' for m in MOVE)
    modules = tuple(f'"{QUARANTINE_MODULE}.{m}' for m in MOVE)
    return any(re.search(re.escape(x) + r'[^"]*"', line) for x in paths + modules)


def _add_to_array(text: str, key: str, value: str, section: str | None = None) -> str:
    """``text`` with ``"value"`` added to the TOML array ``key`` (in ``section``, else at top level), when absent."""
    start = 0
    if section is not None:
        m = re.search(rf"^\[{re.escape(section)}\]\s*$", text, re.M)
        if not m:
            raise Refused(f"no [{section}] table")
        start = m.end()
    m = re.compile(rf"^{re.escape(key)} = \[", re.M).search(text, start)
    if not m:
        raise Refused(f"no array {key!r}")
    end = text.index("]", m.end())
    body = text[m.end():end]
    if f'"{value}"' in body:
        return text
    if "\n" in body:                                   # a multi-line array: first, so the existing lines keep their layout
        return text[:m.end()] + f'"{value}", ' + text[m.end():]
    return text[:end] + (", " if body.strip() else "") + f'"{value}"' + text[end:]


def move_quarantine(root: Path) -> list[str]:
    changed: list[str] = []
    pending = [m for m in MOVE if (root / QUARANTINE / m).is_dir()]
    if pending:
        importers = library_importers(root, tuple(f"{QUARANTINE_MODULE}.{m}" for m in MOVE))
        importers += [f"{rel}: from {QUARANTINE_MODULE} import ..." for rel in _from_quarantine_importers(root)]
        if importers:
            raise Refused("core library code imports a lane that would move:\n  " + "\n  ".join(importers))
    for _rel, text in _scoped(root):
        _rewrite(text)
    _member_texts(root)
    dst_pkg = f"{EXPERIMENTAL}/{PACKAGE}"
    for lane in pending:
        if (root / dst_pkg / lane).exists():
            raise Refused(f"{lane}: both {QUARANTINE}/{lane} and {dst_pkg}/{lane} exist")
        for cache in (root / QUARANTINE / lane).rglob("__pycache__"):
            for f in cache.iterdir():
                f.unlink()
            cache.rmdir()
        _move_path(root, f"{QUARANTINE}/{lane}", f"{dst_pkg}/{lane}")
        changed.append(f"{QUARANTINE}/{lane} -> {dst_pkg}/{lane}")
    for rel, text in ((f"{dst_pkg}/__init__.py", PACKAGE_DOC), (f"{QUARANTINE}/__init__.py", CORE_DOC),
                      (f"{EXPERIMENTAL}/pyproject.toml", PYPROJECT)):
        p = root / rel
        if not p.is_file() or p.read_text(encoding="utf-8") != text:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            changed.append(rel)
    changed += _register_member(root)
    changed += _rewrite_tree(root)
    importers = library_importers(root, (PACKAGE,))
    if importers:
        raise Refused("core library code imports the experimental package:\n  " + "\n  ".join(importers))
    return changed


def _from_quarantine_importers(root: Path) -> list[str]:
    lanes = set(MOVE)
    out = []
    for p in sorted((root / VLLM / "verity_vllm").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(QUARANTINE + "/") or "__pycache__" in rel:
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == QUARANTINE_MODULE and {a.name for a in node.names} & lanes:
                out.append(rel)
    return out


def _member_texts(root: Path) -> dict[str, str]:
    """{pyproject: its text with the experimental member registered}: the workspace's members, the root's dependencies and
    sources, and the vLLM suite's inputs (its tests import the moved lanes)."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    new = _add_to_array(text, "members", EXPERIMENTAL, "tool.uv.workspace")
    new = _add_to_array(new, "dependencies", DISTRIBUTION, "project")
    if not re.search(rf"^{re.escape(DISTRIBUTION)} = ", new, re.M):
        lines = new.splitlines(keepends=True)
        try:
            at = lines.index("[tool.uv.sources]\n") + 1
        except ValueError:
            raise Refused("no [tool.uv.sources] table in the root pyproject.toml") from None
        end = next((i for i in range(at, len(lines)) if lines[i].startswith("[")), len(lines))
        last = [i for i in range(at, end) if lines[i].rstrip().endswith("{ workspace = true }")]
        lines.insert(last[-1] + 1 if last else at, f"{DISTRIBUTION} = {{ workspace = true }}\n")
        new = "".join(lines)
    out = {"pyproject.toml": new}
    text = (root / VLLM / "pyproject.toml").read_text(encoding="utf-8")
    new = _add_to_array(text, "inputs", EXPERIMENTAL, "tool.verity.tests")
    note = f"# Test-only: tests that check a quarantine lane's re-exports, adapters or collective import `{PACKAGE}` ({EXPERIMENTAL}).\n"
    if new != text and note not in new:
        m = re.search(r"^\[tool\.verity\.tests\]\s*\n", new, re.M)
        new = new[:m.end()] + note + new[m.end():]
    out[f"{VLLM}/pyproject.toml"] = new
    for t in out.values():
        tomllib.loads(t)
    return out


def _register_member(root: Path) -> list[str]:
    changed = []
    for rel, new in _member_texts(root).items():
        p = root / rel
        if p.read_text(encoding="utf-8") != new:
            p.write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


def _scoped(root: Path):
    """(path, text) of each tracked text file the module rewrite applies to: the vLLM integration and the experimental package."""
    for rel in tracked(root):
        if not rel.startswith((f"{VLLM}/", f"{EXPERIMENTAL}/")) or rel.startswith(UNTOUCHED) or rel in KEYED_LISTS:
            continue
        try:
            yield rel, (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def _rewrite_tree(root: Path) -> list[str]:
    changed = []
    for rel in KEYED_LISTS:
        p = root / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        new = _drop_lines(text, _keyed_by_moved)
        if new != text:
            json.loads(new)
            p.write_text(new, encoding="utf-8")
            changed.append(rel)
    for rel, old, new_line in REDIRECTS:
        p = root / rel
        if p.is_file() and old in (text := p.read_text(encoding="utf-8")):
            p.write_text(text.replace(old, new_line), encoding="utf-8")
            changed.append(rel)
    for rel, text in list(_scoped(root)):
        new = _rewrite(text)
        if new != text:
            (root / rel).write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


# --- snapshot and verify -----------------------------------------------------------------------------------------------

def _digest(fn) -> str:
    from verity.ir.codec import canonical_json, definition_entry
    from verity.ir.defs import FunctionDefinition
    if not isinstance(fn, FunctionDefinition):
        return f"not a definition: {type(fn).__name__}"
    try:
        return hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest()
    except Exception as e:  # an unbound generic Definition has no self-contained entry; its id is what is compared
        return f"unencodable: {type(e).__name__}"


def _import_tree(name: str) -> None:
    pkg = importlib.import_module(name)
    for m in pkgutil.walk_packages(pkg.__path__, f"{name}."):
        importlib.import_module(m.name)


def snapshot() -> dict:
    from circuit_check import targets as T
    from verity.ir.defs import REGISTRY
    suite = T.suite()
    _import_tree(QUARANTINE_MODULE)
    if importlib.util.find_spec(PACKAGE) is not None:
        _import_tree(PACKAGE)
    return {"suite": {i: _digest(fn) for i, fn in suite.items()},
            "registry": {i: _digest(d) for i, d in sorted(REGISTRY.defs.items())},
            "root_errors": sorted(map(list, T.ROOT_ERRORS))}


def verify(before: dict, after: dict, deleted: list[str]) -> list[str]:
    """What is wrong with ``after`` as the result of deleting ``deleted`` from ``before``: [] when nothing is."""
    problems = []
    for part in ("suite", "registry"):
        b, a = before[part], after[part]
        gone = {i for i in b if family(i) in deleted}
        lost = sorted(set(b) - set(a) - gone)
        kept_deleted = sorted(gone & set(a))
        added = sorted(set(a) - set(b))
        changed = sorted(i for i in set(a) & set(b) if a[i] != b[i])
        problems += [f"{part}: lost {i}" for i in lost] + [f"{part}: deleted but present {i}" for i in kept_deleted]
        problems += [f"{part}: new {i}" for i in added] + [f"{part}: digest changed {i}: {b[i]} -> {a[i]}" for i in changed]
    if before.get("root_errors") != after.get("root_errors"):
        problems.append(f"root errors changed: {before.get('root_errors')} -> {after.get('root_errors')}")
    return problems


def commands() -> list[str]:
    out = []
    for vid, r in DELETE.items():
        out.append(f"research notes claim circuits/{r.slug} --by circuits   # if circuits/{r.slug} is not yet an approach entry")
        out.append(f"research notes approach circuits/{r.slug} superseded --superseded-by circuits/{r.superseded_by} --by circuits "
                   f"--reason {json.dumps(r.reason)} --cite {r.cite}   # {vid}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="delete the DELETE versions and move the MOVE lanes")
    a.add_argument("--only", nargs="+", metavar="ID", help="delete only these versions (each must be in DELETE)")
    a.add_argument("--root", type=Path, default=REPO, help=argparse.SUPPRESS)
    sub.add_parser("commands", help="print the approach entries each deletion needs recorded first")
    s = sub.add_parser("snapshot", help="record circuit-check's task list and the registry, with digests")
    s.add_argument("out", type=Path)
    v = sub.add_parser("verify", help="compare two snapshots against DELETE")
    v.add_argument("before", type=Path)
    v.add_argument("after", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "apply":
            ids = list(DELETE) if args.only is None else args.only
            changed = delete_versions(args.root, ids) + move_quarantine(args.root)
            print("\n".join(changed) if changed else "nothing to do")
        elif args.cmd == "commands":
            print("\n".join(commands()) if DELETE else "no deletions: no approach entries needed")
        elif args.cmd == "snapshot":
            args.out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        else:
            problems = verify(json.loads(args.before.read_text()), json.loads(args.after.read_text()), list(DELETE))
            print("\n".join(problems) if problems else "identical but for the deleted versions")
            return 1 if problems else 0
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
