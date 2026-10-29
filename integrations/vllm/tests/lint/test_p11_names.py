"""P11: names say the job, and versions appear only in hashed data.

Checked against ``allowlists/p11_names.json``:

* ``module-name`` / ``package-name`` / ``def-name`` / ``class-name`` / ``flag-name`` / ``env-name``: a file, directory,
  function, class, flag or environment-variable name (``add_argument``, ``config.option`` and its ``env``) with a
  version or lab token: ``v1``/``v2b``, ``fast``, ``poc``, ``m1``, ``b1``, ``tp2`` or ``G1`` as a whole
  ``_``/``-``/CamelCase-separated token.  Functions decorated with a Definition decorator (``@primitive(...)``) are
  exempt: their name is the hashed Definition name.
* ``doc-board-id`` (``M-`` and four digits, or ``F-r`` and a number), ``doc-lane`` (``[lane`` and a space),
  ``doc-round`` (``R1`` and one digit) and ``doc-date`` (an ISO date) in module, class and function docstrings, as
  ``DOC_PATTERNS`` reads them and ``test_detectors_on_examples`` shows.  Counted per docstring and pattern.

Versioned strings that are hashed (Definition ids such as ``Gemm_v1``, schema ids such as
``runtime-correspondence/v1``) are data, not names, and are not scanned.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p11_names"
RULE = "P11 (names say the job; versions only in hashed data)"
BAD_TOKEN = re.compile(r"^(v[0-9]+[a-z]?|fast|poc|m[0-9]+|b[0-9]+|tp[0-9]+|g[1-8])$")
DEFINITION_DECORATORS = frozenset({"primitive", "composite", "definition"})
DOC_PATTERNS = (("doc-board-id", "M-nnnn", re.compile(r"\bM-\d{4}\b")),
                ("doc-board-id", "F-rN", re.compile(r"\bF-r\d+")),
                ("doc-lane", "[lane", re.compile(r"\[lane ")),
                ("doc-round", "R1N", re.compile(r"\bR1\d\b")),
                ("doc-date", "date", re.compile(r"\b20\d\d-[01]\d-[0-3]\d\b")))
ENV_CALLS = frozenset({"os.environ.get", "os.getenv", "os.environ.setdefault", "os.environ.pop", "os.putenv"})
OPTION = "verity_vllm.config.option"


def name_tokens(name: str) -> list[str]:
    out = []
    for part in re.split(r"[_\-.]+", name):
        out += re.findall(r"[A-Z]+[0-9]*(?![a-z])|[A-Z]?[a-z]+[0-9]*|[0-9]+", part) if not part.islower() else [part]
    return [t.lower() for t in out if t]


def bad_name(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return False
    toks = name_tokens(name)
    return any(BAD_TOKEN.match(t) and (t[0] != "g" or re.search(r"(?<![A-Za-z0-9])G[1-8]|_g[1-8]\b|^g[1-8]\b", name)) for t in toks)


def _decorator_name(d: ast.AST) -> str:
    f = d.func if isinstance(d, ast.Call) else d
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    aliases = R.aliases(tree)
    out: list[R.Violation] = []

    def emit(kind: str, node: ast.AST, symbol: str, detail: str) -> None:
        out.append(R.Violation(rel, kind, symbol, detail, getattr(node, "lineno", 0)))

    stem = rel.rsplit("/", 1)[-1].removesuffix(".py")
    # a circuit-check sidecar (`circuit_check.sidecars`) is named by its module, which is judged by its own name
    if stem != "__init__" and not stem.endswith(".circuit-check") and bad_name(stem):
        emit("module-name", tree, "<module>", stem)
    for node, symbol in R.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and bad_name(node.name) \
                and not any(_decorator_name(d) in DEFINITION_DECORATORS for d in node.decorator_list):
            emit("def-name", node, symbol, node.name)
        elif isinstance(node, ast.ClassDef) and bad_name(node.name):
            emit("class-name", node, symbol, node.name)
        elif isinstance(node, ast.Call):
            fn = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            origin = aliases.resolve(node.func)
            if fn == "add_argument" or origin == OPTION:
                for a in node.args:
                    if R.is_str(a) and a.value.startswith("-") and bad_name(a.value.lstrip("-")):
                        emit("flag-name", node, symbol, a.value)
            if origin == OPTION:
                for k in node.keywords:
                    if k.arg == "env" and R.is_str(k.value) and bad_name(k.value.value):
                        emit("env-name", node, symbol, k.value.value)
            if origin in ENV_CALLS and node.args and R.is_str(node.args[0]) and bad_name(node.args[0].value):
                emit("env-name", node, symbol, node.args[0].value)
        elif isinstance(node, ast.Subscript) and aliases.resolve(node.value) == "os.environ" and R.is_str(node.slice) \
                and bad_name(node.slice.value):
            emit("env-name", node, symbol, node.slice.value)
    for symbol, doc, line in R.docstrings(tree):
        for kind, label, rx in DOC_PATTERNS:
            for _ in rx.finditer(doc):
                out.append(R.Violation(rel, kind, symbol, label, line))
    return out


def package_violations() -> list[R.Violation]:
    dirs = sorted({p.parent for p in R.library_files() + R.library_files(".sh")} - set(R.LIBRARY))
    return [R.Violation(R.rel(d), "package-name", "<package>", d.name) for d in dirs if bad_name(d.name)]


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out = package_violations()
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Name the job, not the version or the milestone; board ids, lanes, rounds and dates go to the "
                    "research store, not docstrings.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    assert [n for n in ("v1_bridge", "native_collect_v2b", "global_match_fast", "poc_rows", "m1_capture", "b1_authored",
                        "TP2CaptureWorkerExtension", "CaptureV1", "check_G3", "fast-copy")
            if not bad_name(n)] == []
    assert [n for n in ("fasttree", "fa2_tap", "sha256", "vllm_adapter", "layers", "Bf16ToF32", "__init__", "LOG1",
                        "embedding_shard", "rms_triton_model", "tp_stage")
            if bad_name(n)] == []
    src = (
        "'''Lane tp2 [lane tp2] (window 2026-09-17), coord M-0451, rev F-r16-18, R17.'''\n"
        "import os\n"
        "from verity.ml.circuits.defs import primitive\n"
        "@primitive('Gemm', 1, [], None)\n"
        "def Gemm_v1(a): return a\n"
        "def build_v2(p, fast=False): return os.environ.get('VERITY_TP2_MODE')\n"
        "class PocRunner: pass\n"
        "from verity_vllm.config import option\n"
        "class Options:\n"
        "    fast_copy: bool = option('--fast-copy', env='VERITY_M1_DIR', action='store_true', default=False)\n"
    )
    got = sorted((v.kind, v.detail) for v in file_violations(src, "verity_vllm/tp/m1_capture.py"))
    assert got == [("class-name", "PocRunner"), ("def-name", "build_v2"), ("doc-board-id", "F-rN"),
                   ("doc-board-id", "M-nnnn"), ("doc-date", "date"), ("doc-lane", "[lane"), ("doc-round", "R1N"),
                   ("env-name", "VERITY_M1_DIR"), ("env-name", "VERITY_TP2_MODE"), ("flag-name", "--fast-copy"),
                   ("module-name", "m1_capture")], got
    assert file_violations("", "verity_vllm/tp/m1_capture.circuit-check.py") == []
