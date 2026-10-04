"""Split the dense, activation, norm and RoPE family of vLLM's registry into its Definitions and its twins.

    uv run python integrations/vllm/tools/migrate/split_dense.py            # apply (rerun on a fresh main to rebase)
    uv run python integrations/vllm/tools/migrate/split_dense.py --check    # exit 1, listing what is left, unless applied

The family's Definition modules (`DEFINITION_MODULES`) stay in `verity_vllm.program.registry`, bound for the catalog: they
already import only `verity`, the standard library, numpy and other Definition modules (`OTHER_DEFINITION_MODULES`, owned by
other families). Its twins leave the registry for `verity_vllm.program.kernels`, the home of the reference twins (`MOVES`):
the two RoPE difftest adapters (production via torch / vLLM, and a `register=False` spec Definition each, whose id and digest
are name-keyed and so unchanged) and the fused-RMSNorm sweep (numpy twins and a CLI over a B0 store). `gemm_targets` is not
moved: it is target vocabulary paired with `registry.targets` (another family), and moving it alone would make that module
import the torch frontend package.

The script moves each module (`git mv` in a work tree), then rewrites every reference repo-wide: imports, including
`from verity_vllm.program.registry import a, rope_difftest` (split into two statements) and relative ones; dotted names and
file paths in any tracked or untracked text file (allowlists, the CLI table, docstrings, `census_roots.txt`) except `KEPT`;
and drops lint allowlist entries a move puts in a directory the lint does not scan (`DROPPED_ENTRIES`). It is idempotent: on a tree where a
module is already moved it only rewrites references left to its old path, so a branch rebased past the split reruns it. It
fails loudly, changing nothing, when a module is at both paths or at neither, when a module it would move registers a
Definition, when a Definition module imports anything but `verity`, the standard library, numpy and Definition modules, or
when a reference is attribute access it cannot rewrite (`registry.<module>` on an imported package).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PKG = "integrations/vllm/verity_vllm"
REGISTRY = "verity_vllm.program.registry"

#: module of the registry -> the package it moves to
MOVES: dict[str, str] = {
    "rope_difftest": "verity_vllm.program.kernels",
    "rope_gptj_difftest": "verity_vllm.program.kernels",
    "rmsnorm_fused_sweep": "verity_vllm.program.kernels",
}

#: the family's Definition modules: word and Boolean Definitions, nothing else
DEFINITION_MODULES = ("act", "boolean_activation", "boolean_dense", "boolean_dense_norm", "boolean_norms", "boolean_rope",
                      "boolean_silu", "dense", "gemm_bias")
#: other families' Definition modules the family's Definitions call
OTHER_DEFINITION_MODULES = ("b1", "prims", "sampling")
#: the top-level packages a Definition module may import besides the registry's Definition modules
ALLOWED_TOP = frozenset({"verity", "numpy", "__future__"}) | frozenset(sys.stdlib_module_names)

#: allowlist entries a move makes stale, by the allowlist (repo-relative) and the entry's file after the move: the lint skips
#: that directory (`tests/test_no_by_name_rules.py`: `SKIP_DIRS` holds `kernels`)
DROPPED_ENTRIES: dict[str, tuple[str, ...]] = {
    "integrations/vllm/tests/by_name_allowlist.json": ("verity_vllm/program/kernels/rmsnorm_fused_sweep.py",),
}

#: directories whose files name the old paths on purpose (this script and its tests)
EXCLUDE = ("integrations/vllm/tools/migrate/", "integrations/vllm/tests/migrate/")
#: files whose prose keeps the old path: `b1.py` (another family's) cites the sweep in two `conformance=` strings, and its
#: source and conformance strings enter `frontend.provenance.registry_version()`, which derived Programs record
KEPT = ("integrations/vllm/verity_vllm/program/registry/b1.py",)
#: suffixes never read as text
BINARY = (".gz", ".xz", ".npz", ".npy", ".bin", ".png", ".jpg", ".pdf", ".so", ".pyc", ".tar", ".zip", ".olean", ".ilean")


class SplitError(RuntimeError):
    """The tree is not one the split can apply to; nothing was changed."""


def module_path(root: Path, dotted: str) -> Path:
    return root / "integrations/vllm" / (dotted.replace(".", "/") + ".py")


def old_new(name: str) -> tuple[str, str]:
    return f"{REGISTRY}.{name}", f"{MOVES[name]}.{name}"


def _sub_package(dotted: str) -> str:
    """``verity_vllm.program.kernels`` -> ``kernels``: the segment that replaces ``registry`` in a reference."""
    head, _, last = dotted.rpartition(".")
    assert head == "verity_vllm.program", dotted
    return last


# --- the guards -----------------------------------------------------------------------------------------------------------

def registers_definitions(src: str) -> list[str]:
    """Where ``src`` registers a Definition: a ``@composite`` / ``@primitive`` decorator, or a ``CompositeDefinition`` /
    ``PrimitiveDefinition`` construction without ``register=False``."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in ("composite", "primitive"):
                    out.append(f"line {node.lineno}: @{name} {node.name}")
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("CompositeDefinition", "PrimitiveDefinition"):
                unregistered = any(k.arg == "register" and isinstance(k.value, ast.Constant) and k.value.value is False
                                   for k in node.keywords)
                if not unregistered:
                    out.append(f"line {node.lineno}: {name}(...) without register=False")
    return out


def imported_modules(src: str, package: str) -> list[tuple[int, str]]:
    """(line, absolute module) of every import in ``src``, module level and lazy, ``from p import m`` as ``p.m``; relative
    imports resolved against ``package``."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, package)
            out += [(node.lineno, f"{base}.{a.name}") for a in node.names]
    return out


def boundary_violations(src: str, package: str = REGISTRY) -> list[str]:
    """The imports of a Definition module that are not `verity`, the standard library, numpy or a Definition module."""
    allowed = {f"{REGISTRY}.{m}" for m in DEFINITION_MODULES + OTHER_DEFINITION_MODULES}
    bad = []
    for line, mod in imported_modules(src, package):
        top = mod.partition(".")[0]
        if top in ALLOWED_TOP:
            continue
        if any(mod == a or mod.startswith(a + ".") for a in allowed):
            continue
        bad.append(f"line {line}: {mod}")
    return bad


def _absolute(node: ast.ImportFrom, package: str) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    anchor = parts[:len(parts) - node.level + 1]
    return ".".join(anchor + ([node.module] if node.module else []))


# --- the rewrite ----------------------------------------------------------------------------------------------------------

def _patterns() -> list[tuple[re.Pattern[str], str]]:
    """``registry.<m>`` and ``registry/<m>`` (dotted names, file paths, short forms in prose) -> the new package's."""
    out = []
    for name, new in MOVES.items():
        sub = _sub_package(new)
        out.append((re.compile(rf"(?<![\w]){re.escape('registry')}([./]){re.escape(name)}(?![\w])"), rf"{sub}\g<1>{name}"))
    return out


def _package_of(rel: str) -> str:
    """The package a Python file under ``PKG`` belongs to (``integrations/vllm/verity_vllm/a/b.py`` -> ``verity_vllm.a``)."""
    parts = Path(rel).relative_to("integrations/vllm").with_suffix("").parts
    return ".".join(parts[:-1])


def _format_from(module: str, aliases: list[ast.alias]) -> str:
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in aliases)
    return f"from {module} import {names}"


def rewrite_imports(src: str, package: str | None, moved_from: str | None = None) -> str:
    """``src`` with every ``from <registry> import ...`` naming a moved module split so the moved names come from their new
    package, and (for a moved module, ``moved_from`` its old package) its relative imports made absolute."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        anchor = moved_from if moved_from is not None else package
        if node.level and anchor is None:
            continue
        base = _absolute(node, anchor) if node.level else (node.module or "")
        if moved_from is not None and node.level:
            new_src = _format_from(base, node.names)
        elif base == REGISTRY and any(a.name in MOVES for a in node.names):
            keep = [a for a in node.names if a.name not in MOVES]
            moved: dict[str, list[ast.alias]] = {}
            for a in node.names:
                if a.name in MOVES:
                    moved.setdefault(MOVES[a.name], []).append(a)
            head = node.module if not node.level else "." * node.level + (node.module or "")
            stmts = ([_format_from(head, keep)] if keep else []) + [_format_from(p, al) for p, al in moved.items()]
            new_src = ("\n" + " " * node.col_offset).join(stmts)
        else:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end, new_src))
    for start, end, text in sorted(edits, reverse=True):
        src = src[:start] + text + src[end:]
    return src


def attribute_uses(src: str) -> list[str]:
    """``registry.<moved>`` as attribute access on a name: a reference the text rewrite would break."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [f"line {n.lineno}: registry.{n.attr}" for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in MOVES and isinstance(n.value, ast.Name) and n.value.id == "registry"]


def rewrite_text(text: str) -> str:
    for pat, repl in _patterns():
        text = pat.sub(repl, text)
    return text


def drop_entries(rel: str, text: str) -> str:
    """``text`` without the allowlist entries `DROPPED_ENTRIES` names for ``rel`` (one JSON object per line)."""
    files = DROPPED_ENTRIES.get(rel)
    if not files:
        return text
    out = []
    for line in text.splitlines(keepends=True):
        s = line.strip().rstrip(",")
        if s.startswith("{") and s.endswith("}"):
            try:
                if json.loads(s).get("file") in files:
                    continue
            except json.JSONDecodeError:
                pass
        out.append(line)
    joined = "".join(out)
    return re.sub(r",(\s*\])", r"\1", joined)


# --- the plan -------------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    edits: dict[Path, str] = field(default_factory=dict)

    def empty(self) -> bool:
        return not self.moves and not self.edits


def _files(root: Path) -> list[str]:
    """Repo-relative paths of the tracked and untracked (not ignored) files, or every file when ``root`` is no work tree."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             check=True, capture_output=True).stdout.decode()
        return sorted({p for p in out.split("\0") if p})
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)


def plan(root: Path = REPO) -> Plan:
    """What applying the split to ``root`` changes; raises `SplitError` when it cannot apply."""
    p = Plan()
    errors: list[str] = []
    renames: dict[str, str] = {}
    for name in MOVES:
        old, new = (module_path(root, d) for d in old_new(name))
        if old.exists() and new.exists():
            errors.append(f"{name}: both {old.relative_to(root)} and {new.relative_to(root)} exist")
        elif not old.exists() and not new.exists():
            errors.append(f"{name}: neither {old.relative_to(root)} nor {new.relative_to(root)} exists")
        elif old.exists():
            if regs := registers_definitions(old.read_text()):
                errors.append(f"{name} registers Definitions, so it is not a twin to move: {regs}")
            p.moves.append((old, new))
            renames[str(old.relative_to(root))] = str(new.relative_to(root))
    for m in DEFINITION_MODULES:
        path = module_path(root, f"{REGISTRY}.{m}")
        if not path.exists():
            errors.append(f"Definition module {m} is missing ({path.relative_to(root)})")
        elif bad := boundary_violations(path.read_text()):
            errors.append(f"Definition module {m} imports a non-Definition module: {bad}")
    if errors:
        raise SplitError("split_dense cannot apply:\n  " + "\n  ".join(errors))

    for rel in _files(root):
        if rel in KEPT and (root / rel).is_file():
            olds = {old_new(n)[0] for n in MOVES}
            if hits := [f"line {ln}: {m}" for ln, m in imported_modules((root / rel).read_text(), REGISTRY)
                        if any(m == o or m.startswith(o + ".") for o in olds)]:
                errors.append(f"{rel} is kept as it is but imports a moved module: {hits}")
        if rel.startswith(EXCLUDE) or rel.endswith(BINARY) or rel in KEPT:
            continue
        src_rel = rel
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        dest_rel = renames.get(src_rel, src_rel)
        new = text
        if rel.endswith(".py"):
            if bad := attribute_uses(text):
                errors.append(f"{rel}: {bad}")
            moved_from = REGISTRY if src_rel in renames else None
            pkg = _package_of(rel) if rel.startswith(PKG + "/") else None
            new = rewrite_imports(new, pkg, moved_from)
        new = drop_entries(rel, rewrite_text(new))
        if new != text or dest_rel != src_rel:
            p.edits[root / dest_rel] = new
        if rel.endswith(".py") and new != text:
            try:
                ast.parse(new)
            except SyntaxError as e:
                errors.append(f"{rel}: the rewrite does not parse: {e}")
    if errors:
        raise SplitError("split_dense cannot apply:\n  " + "\n  ".join(errors))
    p.edits = {k: v for k, v in p.edits.items()
               if not (k.exists() and k.read_text(encoding="utf-8") == v) or any(k == n for _, n in p.moves)}
    return p


def _git(root: Path) -> bool:
    return (root / ".git").exists() and shutil.which("git") is not None


def apply(p: Plan, root: Path = REPO) -> None:
    for old, new in p.moves:
        if _git(root):
            subprocess.run(["git", "-C", str(root), "mv", str(old.relative_to(root)), str(new.relative_to(root))], check=True)
        else:
            new.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, new)
    for path, text in p.edits.items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")


def left(root: Path = REPO) -> list[str]:
    """What is left to do on ``root``: the modules still to move and the files still naming an old path."""
    p = plan(root)
    moved = {n for _, n in p.moves}
    return ([f"move {o.relative_to(root)} -> {n.relative_to(root)}" for o, n in p.moves]
            + [f"rewrite {f.relative_to(root)}" for f in sorted(p.edits) if f not in moved])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help="the repository to split (default: this checkout)")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 listing what is left unless applied")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    try:
        if a.check:
            todo = left(root)
            print("\n".join(todo) if todo else "split_dense: applied")
            return 1 if todo else 0
        p = plan(root)
        apply(p, root)
    except SplitError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"split_dense: moved {len(p.moves)} module(s), rewrote {len(p.edits) - len(p.moves)} file(s)")
    todo = left(root)
    if todo:
        print("split_dense: still left after applying:\n  " + "\n  ".join(todo), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
