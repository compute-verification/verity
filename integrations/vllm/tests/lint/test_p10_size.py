"""P10: modules stay at or under 800 lines and functions at or under 150 lines.

A module's size is its line count; a function's is the span of its ``def`` statement (``lineno`` to ``end_lineno``,
decorators excluded), methods and nested functions each counted on their own.  Today's offenders are recorded in
``allowlists/p10_size.json`` at their current size, as ``count`` (in lines) of the key
``(file, "module" | "function", symbol, "lines")``.  The recorded size is a cap:

* an unlisted module or function over the limit fails;
* a listed one that grew more than `slack` past its recorded size fails (split it; do not raise the cap), so a fix in a
  file at its cap needs no line-trimming;
* a listed one that is now within the limit (or gone) fails until its entry is deleted.  One that only shrank keeps its
  recorded size: the allowlist is a file every change would otherwise edit.
"""
from __future__ import annotations

import ast
from functools import lru_cache

from . import _ratchet as R

NAME = "p10_size"
RULE = "P10 (size ratchet)"
MODULE_MAX = 800
FUNCTION_MAX = 150


def slack(cap: int) -> int:
    """How far past its recorded size a listed module or function may grow: 5%, at least 10 lines."""
    return max(10, cap // 20)


def file_sizes(src: str, rel: str) -> list[R.Violation]:
    """One violation per oversized module/function; ``line`` carries the size in lines."""
    out: list[R.Violation] = []
    n = len(src.splitlines())
    if n > MODULE_MAX:
        out.append(R.Violation(rel, "module", "<module>", "lines", n))
    for node, symbol in R.walk(R.parse_src(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = node.end_lineno - node.lineno + 1
            if span > FUNCTION_MAX:
                qual = node.name if symbol == "<module>" else f"{symbol}.{node.name}"
                out.append(R.Violation(rel, "function", qual, "lines", span))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_sizes(R.source(p), R.rel(p))
    return tuple(out)


def sizes() -> dict[tuple[str, str, str, str], int]:
    return {v.key: v.line for v in scan()}


def test_no_new_or_grown_offenders() -> None:
    caps = R.load_allowlist(NAME)
    bad = []
    for k, n in sorted(sizes().items()):
        if k not in caps:
            bad.append(f"  {k[0]} [{k[1]}] {k[2]}: {n} lines (limit {MODULE_MAX if k[1] == 'module' else FUNCTION_MAX})")
        elif n > caps[k] + slack(caps[k]):
            bad.append(f"  {k[0]} [{k[1]}] {k[2]}: {n} lines, above its recorded {caps[k]} + {slack(caps[k])}")
    assert not bad, (f"{RULE}: {len(bad)} module(s)/function(s) over the limit or past their recorded size and its slack in "
                     f"allowlists/{NAME}.json. Split them; do not raise a recorded size.\n" + "\n".join(bad))


def test_recorded_sizes_are_current() -> None:
    have = sizes()
    stale = []
    for k, cap in sorted(R.load_allowlist(NAME).items()):
        if k not in have:
            stale.append(f"  delete {R.entry_line(k, cap)}")
    assert not stale, (f"{RULE}: {len(stale)} entr(y/ies) in allowlists/{NAME}.json for code now within the limit or gone: "
                       "delete them:\n" + "\n".join(stale))


def test_detectors_on_examples() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(FUNCTION_MAX))
    src = f"def big():\n{body}\n\ndef small():\n    return 1\n"
    got = file_sizes(src, "verity_vllm/x.py")
    assert [(v.kind, v.symbol, v.line) for v in got] == [("function", "big", FUNCTION_MAX + 1)]
