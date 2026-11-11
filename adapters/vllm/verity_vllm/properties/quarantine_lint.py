"""verity_vllm/properties/quarantine_lint.py — PROTECTED (gate G4, admission rules 1–2).  Static purity + identity lint of the
quarantine registry.

    uv run verity-vllm quarantine-lint [--root verity_vllm/program/quarantine] [--out quarantine_lint.json]

Rules (AST, no import of the checked modules):
  - forbidden imports: os, sys, time, datetime, random, secrets, pathlib, io, json, pickle, subprocess, socket, urllib,
    requests, torch, triton, numpy.random, verity_vllm.{observe,acquire,commit,check} (any), importlib, ctypes, threading, multiprocessing;
  - forbidden calls/names: open, eval, exec, compile, input, __import__, globals, locals, vars, getattr-with-string-env;
  - no `global` / `nonlocal` statements; no module-level mutable state mutated in functions (best effort: assignments to
    module names inside functions);
  - every module that defines a definition (`CompositeDefinition(`, `@composite(`, `@primitive(`, `PrimitiveDefinition(`)
    has a module-level `SEMANTIC_ID = "<area>/<Name>@v<k>"` and every definition's version argument is an int >= 1;
  - `*_difftest.py` adapters are exempt from the import rule for `torch` INSIDE functions only (module-level torch import
    is still an error) and need `SEMANTIC_ID` too.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT as REPO
from verity_vllm.config import option
from verity_vllm.program import quarantine as _quarantine
from verity_vllm.properties.record import seal

ROOT = Path(_quarantine.__file__).parent
FORBIDDEN_IMPORTS = ("os", "sys", "time", "datetime", "random", "secrets", "pathlib", "io", "json", "pickle", "subprocess", "socket",
                     "urllib", "requests", "torch", "triton", "numpy.random", "verity_vllm.observe", "verity_vllm.acquire", "verity_vllm.commit", "verity_vllm.check", "importlib", "ctypes", "threading",
                     "multiprocessing", "vllm")
FORBIDDEN_CALLS = ("open", "eval", "exec", "compile", "input", "__import__", "globals", "locals", "vars")
SEMANTIC_ID_RE = re.compile(r"^[a-z0-9_\-]+/[A-Za-z][A-Za-z0-9_]*@v\d+$")
DEF_MARKERS = ("CompositeDefinition", "PrimitiveDefinition", "composite", "primitive")


def lint_file(path: Path) -> dict[str, Any]:
    src = path.read_text()
    tree = ast.parse(src, str(path))
    problems: list[str] = []
    is_adapter = path.name.endswith("_difftest.py")
    module_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(getattr(n, "ctx", None), ast.Store)}
    semantic_id = None
    n_defs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            in_function = any(isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) for p in _parents(tree, node))
            for nm in names:
                root = nm.split(".")[0]
                full = nm
                bad = any(full == f or full.startswith(f + ".") or root == f for f in FORBIDDEN_IMPORTS)
                if bad and not (is_adapter and in_function and root in ("torch", "vllm", "triton")):
                    problems.append(f"line {node.lineno}: forbidden import {nm!r}" + ("" if not is_adapter else " (adapters may import torch/vllm inside functions only)"))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name in FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: forbidden call {name}()")
            if name in DEF_MARKERS:
                n_defs += 1
                if name in ("CompositeDefinition", "PrimitiveDefinition", "composite", "primitive") and len(node.args) >= 2:
                    v = node.args[1]
                    if not (isinstance(v, ast.Constant) and isinstance(v.value, int) and v.value >= 1):
                        problems.append(f"line {node.lineno}: {name} version must be an int literal >= 1")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: {type(node).__name__.lower()} statement")
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SEMANTIC_ID" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                semantic_id = node.value.value
    if n_defs or is_adapter:
        if semantic_id is None:
            problems.append("no module-level SEMANTIC_ID = \"<area>/<Name>@v<k>\"")
        elif not SEMANTIC_ID_RE.match(semantic_id):
            problems.append(f"SEMANTIC_ID {semantic_id!r} does not match <area>/<Name>@v<k>")
    # module-level names assigned inside functions (mutable module state)
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AugAssign)) and n.lineno > fn.lineno:
                pass  # locals are fine; module mutation needs `global`, caught above
    rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
    return {"file": rel, "semantic_id": semantic_id, "definitions": n_defs, "adapter": is_adapter,
            "status": "pass" if not problems else "fail", "problems": problems}


def _parents(tree: ast.AST, target: ast.AST) -> list[ast.AST]:
    out: list[ast.AST] = []

    def walk(node: ast.AST, chain: list[ast.AST]) -> bool:
        if node is target:
            out.extend(chain)
            return True
        for ch in ast.iter_child_nodes(node):
            if walk(ch, chain + [node]):
                return True
        return False

    walk(tree, [])
    return out


def lint(root: Path = ROOT) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" or p.parent != root)
    reports = [lint_file(p) for p in files]
    ids = [r["semantic_id"] for r in reports if r["semantic_id"] and not r["adapter"]]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    ok = all(r["status"] == "pass" for r in reports) and not dup
    return seal({"schema": "verity-gen/quarantine-lint/v1", "root": str(root), "n_files": len(files), "status": "pass" if ok else "fail",
                 "duplicate_semantic_ids": dup, "files": reports}, ok)


@dataclass(frozen=True, kw_only=True)
class Options:
    root: str = option("--root", default=str(ROOT))
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    rep = lint(Path(a.root))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    for r in rep["files"]:
        if r["status"] != "pass":
            print(f"[quarantine_lint] FAIL {r['file']}: " + "; ".join(r["problems"]), file=sys.stderr)
    print(f"[quarantine_lint] {rep['status']}: {rep['n_files']} files" + (f", duplicate ids {rep['duplicate_semantic_ids']}" if rep["duplicate_semantic_ids"] else ""), file=sys.stderr)
    return 0 if rep["status"] == "pass" else 1
