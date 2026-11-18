"""`levels` names one setting: the sampled-proofs protocol's number of levels (TOP-440). The recursion benchmark's
reader and its figures write no key of that name for anything else.

- Neither file of `FILES` writes a key named `levels`: a dict literal's key, a subscript stored to, or a keyword of a
  call. Reading one another component writes (`rec_profile` reads `rec_vstage`'s per-level stages) is not writing it.

Nothing is executed (AST only). A key built at run time is not seen."""

from __future__ import annotations

import ast

from test_boundaries import ROOT

KEY = "levels"
FILES = ("benchmarks/recursion_e2e/rec_profile.py", "benchmarks/recursion_figures/figures.py")


def _written(text: str) -> list[int]:
    out = []
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Dict):
            out += [k.lineno for k in n.keys if isinstance(k, ast.Constant) and k.value == KEY]
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and \
                isinstance(n.slice, ast.Constant) and n.slice.value == KEY:
            out.append(n.lineno)
        elif isinstance(n, ast.Call):
            out += [k.value.lineno for k in n.keywords if k.arg == KEY]
    return out


def test_neither_file_writes_a_key_named_levels() -> None:
    found = {rel: lines for rel in FILES if (lines := _written((ROOT / rel).read_text()))}
    assert not found, (f"`{KEY}` is the sampled-proofs protocol's setting (TOP-440); name the key for what it holds "
                       f"(figures.py's knowledge-soundness terms are `ks_log2`): {found}")


def test_the_check_sees_each_way_to_write_it() -> None:
    assert _written('x = {"levels": 1}\ny["levels"] = 2\nz = dict(levels=3)\n') == [1, 2, 3]
    assert _written('x = y.get("levels") or y["levels"]\n') == []
