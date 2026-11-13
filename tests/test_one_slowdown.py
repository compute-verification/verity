"""One slowdown calculator: every slowdown the tree computes comes from one function of the calculator's `min_cut`,
`slowdown_of`, which goes from a work to a slowdown.

The calculator is the directory holding the package of the one tracked `min_cut.py` outside a test directory.  A file
outside it reaches `min_cut` where it imports it (a statement or an import-call string), loads it by file (a path
ending in `min_cut.py` handed to a load or put on `sys.path`, `test_boundaries._path_loads`), or calls with no
argument a door some file defines to return it (`calculator._min_cut()`, `work.min_cut()`).  Every such file is in
`ALLOWED`, which only shrinks, with what of `min_cut` past its one function it reads and why.  No file that is not a
test reads `bound`, `blocks_at_least` or `slowdown_at_least` (`FORBIDDEN`), and none passes `F` to `slowdown_of`,
which is for a test's comparison at one `F`; a test that reads them holds the one function to them, a reference.  The
period's bound and its parts (`SIBLINGS`) are read only where `ALLOWED` says.  Nothing is executed (AST only)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from test_boundaries import _assigned, _bound_from, _imports, _is_test, _path_loads, _tracked, _tracked_python

#: what computes a slowdown from a work's sums, which only `slowdown_of` may call outside the calculator
FORBIDDEN = {"bound", "blocks_at_least", "slowdown_at_least"}
#: the bound over a monitoring period's blocks with untied words, its threshold and its limit beside an uncredited share
SIBLINGS = {"bound_charged", "best_w", "capped"}
#: {file outside the calculator that reaches its `min_cut`: (the names of `FORBIDDEN` and `SIBLINGS` it reads, why)}
ALLOWED = {
    "benchmarks/training_slowdown/training_slowdown/calculator.py":
        ((), "the training benchmark's chooser: reads a traced step's products into the one function"),
    "benchmarks/circuit_properties/circuit_properties/work.py":
        ((), "the circuit-properties chooser: reads a Program's charged products into the one function"),
    "benchmarks/inference_only/policy.py":
        ((), "the inference-only chooser: hands the one function a session block's one forward product"),
    "benchmarks/inference_only/tied.py":
        (("bound_charged", "best_w", "capped"), "the tied policy's period bound and its limit beside the share no "
                                                "PoUW check credits, which no work's chooser gives"),
    "benchmarks/inference_only/report.py": ((), "prints the bound's statement names and premises"),
    "benchmarks/inference_only/tests/test_inference_only.py":
        (("bound", "bound_charged", "best_w", "capped"), "references: the policy's number and the tied rows are "
                                                         "`bound`'s and the period bound's"),
    "benchmarks/training_slowdown/tests/conftest.py":
        (("bound",), "reference: the count the fixtures' bound of record is held to"),
    "benchmarks/training_slowdown/tests/test_formalism.py":
        (("slowdown_at_least",), "reference: the record's figure is the statement's arithmetic"),
    "benchmarks/training_slowdown/tests/test_grade.py":
        (("blocks_at_least",), "reference: ρ's products term is the theorem's count solved for the fewest blocks"),
    "benchmarks/training_slowdown/tests/test_one_calculator.py":
        (("bound",), "reference: the entry point's slowdown is `bound`'s at three policies"),
    "benchmarks/training_slowdown/tests/test_program_reading.py":
        ((), "compares a Program's reading with its trace's at one `F`"),
    "benchmarks/circuit_properties/tests/test_work.py":
        (("bound",), "reference: the reading's bound is `bound` at its products' sums"),
}
FIX = ("every slowdown comes from the calculator's `min_cut.slowdown_of`: read your work into its `Product`s and call "
       "it, keeping only how you read the work, and add your file to `ALLOWED` in tests/test_one_slowdown.py with why "
       "(it only shrinks); a test that holds the one function to `bound` lists the names it reads as references")


def calculator() -> tuple[Path, str]:
    """The calculator's directory, relative to the repository, and its package's name."""
    (rel,) = (r for r in _tracked("*min_cut.py") if Path(r).name == "min_cut.py" and not _is_test(r))
    return Path(rel).parent.parent, Path(rel).parent.name


def _names_min_cut(e: ast.AST, names: set[str]) -> bool:
    return any(isinstance(x, ast.Constant) and isinstance(x.value, str) and Path(x.value).name == "min_cut.py"
               or isinstance(x, ast.Name) and x.id in names for x in ast.walk(e))


def _loads(nodes: list[ast.AST], package: str) -> bool:
    """Whether ``nodes`` import ``package``'s `min_cut` or load it by file."""
    module = f"{package}.min_cut"
    named = any(name == module or name.startswith(module + ".") for name, _, imported in _imports(Path("x.py"), Path(""),
                                                                                                  (), nodes) if imported)
    named |= any(isinstance(n, ast.ImportFrom) and n.module == package and any(a.name == "min_cut" for a in n.names)
                 for n in nodes)
    paths = _bound_from(_assigned(nodes), set(), _names_min_cut)
    return named or bool(_path_loads(nodes, lambda e: _names_min_cut(e, paths), set()))


def _call_name(n: ast.AST) -> str | None:
    if not isinstance(n, ast.Call):
        return None
    return n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)


def reads(texts: dict[str, str], where: Path, package: str) -> dict[str, dict[str, list[int]]]:
    """For each file of ``texts`` (repository path -> source) outside ``where`` that reaches ``package``'s `min_cut`,
    the names of `min_cut` it reads with the lines, `F` for a call of `slowdown_of` that passes one, `min_cut` alone
    where it reads nothing by name."""
    inside = where.as_posix() + "/"
    trees = {}
    for rel, text in texts.items():
        if rel.startswith(inside) or "min_cut" not in text:
            continue
        try:
            trees[rel] = list(ast.walk(ast.parse(text, filename=rel)))
        except SyntaxError:
            continue
    doors = {fn.name for nodes in trees.values() for fn in nodes
             if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _loads(list(ast.walk(fn)), package)}
    out = {}
    for rel, nodes in trees.items():
        door_calls = [n for n in nodes if _call_name(n) in doors and not n.args and not n.keywords]
        if not (_loads(nodes, package) or door_calls):
            continue
        seeds = {a.asname or a.name for n in nodes if isinstance(n, ast.ImportFrom) and n.module == package
                 for a in n.names if a.name == "min_cut"}
        seeds |= {a.asname for n in nodes if isinstance(n, ast.Import) for a in n.names
                  if a.name == f"{package}.min_cut" and a.asname}

        def module(e: ast.AST, names: set[str]) -> bool:
            if isinstance(e, ast.Name):
                return e.id in names
            if isinstance(e, ast.Attribute):
                return e.attr == "min_cut" and isinstance(e.value, ast.Name) and e.value.id == package
            return _call_name(e) in doors | {"module_from_spec"} and not e.keywords and (
                not e.args or _call_name(e) == "module_from_spec")

        names = _bound_from(_assigned(nodes), seeds, module)
        got: dict[str, list[int]] = {}
        for n in nodes:
            if isinstance(n, ast.Attribute) and module(n.value, names):
                got.setdefault(n.attr, []).append(n.lineno)
            elif isinstance(n, ast.ImportFrom) and n.module == f"{package}.min_cut":
                for a in n.names:
                    got.setdefault(a.name, []).append(n.lineno)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "slowdown_of"
                    and module(n.func.value, names) and any(k.arg == "F" for k in n.keywords)):
                got.setdefault("F", []).append(n.lineno)
        out[rel] = got or {"min_cut": [nodes[0].lineno if hasattr(nodes[0], "lineno") else 1]}
    return out


def offenders(found: dict[str, dict[str, list[int]]], allowed: dict[str, tuple]) -> list[str]:
    """What ``found`` (`reads`) breaks: a file not in ``allowed``; one that is not a test reading `FORBIDDEN` or
    passing `F`; a name of `FORBIDDEN` or `SIBLINGS` read where ``allowed`` doesn't list it."""
    out = []
    for rel, got in sorted(found.items()):
        if rel not in allowed:
            out.append(f"{rel}: reaches the calculator's min_cut and is not in ALLOWED")
            continue
        listed = set(allowed[rel][0])
        for name, lines in sorted(got.items()):
            at = f"{rel}:{lines[0]}"
            if name in FORBIDDEN and not _is_test(rel):
                out.append(f"{at}: calls min_cut.{name}, which only slowdown_of may call outside the calculator")
            elif name == "F" and not _is_test(rel):
                out.append(f"{at}: passes F to min_cut.slowdown_of, which only a test's comparison at one F may")
            elif name in FORBIDDEN | SIBLINGS and name not in listed:
                out.append(f"{at}: reads min_cut.{name}, which ALLOWED does not list for it")
    return out


def check(texts: dict[str, str], allowed: dict[str, tuple]) -> dict[str, dict[str, list[int]]]:
    """`reads` of ``texts``, failing with the fix on any of its `offenders`."""
    where, package = calculator()
    found = reads(texts, where, package)
    bad = offenders(found, allowed)
    assert not bad, FIX + ".\n" + "\n".join(bad)
    return found


def test_every_slowdown_comes_from_the_one_function() -> None:
    check(_tracked_python(), ALLOWED)
    sources = [rel for rel, (names, _) in ALLOWED.items() if set(names) & FORBIDDEN and not _is_test(rel)]
    assert not sources, f"only a test may read FORBIDDEN, as a reference; take these out of ALLOWED: {sources}"


def test_allowed_only_shrinks_to_files_that_still_reach_it() -> None:
    """An entry whose file no longer reaches `min_cut`, or a name it no longer reads, is removed."""
    where, package = calculator()
    found = reads(_tracked_python(), where, package)
    stale = sorted(set(ALLOWED) - set(found))
    stale += [f"{rel}: {name}" for rel, (names, _) in ALLOWED.items() if rel in found for name in names
              if name not in found[rel]]
    assert not stale, f"ALLOWED entries that no longer reach what they list; remove them: {stale}"


TOY = {
    "benchmarks/toy/toy/second.py": "import sys\nsys.path.insert(0, 'x')\nfrom {p} import min_cut as M\n\n"
                                    "def slow(w):\n    return M.bound(1, 2, N=w, O=w, size=3).slowdown\n",
    "benchmarks/toy/toy/door.py": "import importlib.util\n\ndef _mc():\n    path = HERE / 'min_cut.py'\n"
                                  "    spec = importlib.util.spec_from_file_location('m', path)\n"
                                  "    mod = importlib.util.module_from_spec(spec)\n    return mod\n\n"
                                  "def slow(w):\n    m = _mc()\n    return m.slowdown_of(1, 2, size=3, work=w, F=1.0)\n",
    "benchmarks/toy/toy/tied.py": "from {p}.min_cut import capped, blocks_at_least\n",
    "benchmarks/toy/tests/test_toy.py": "from {p} import min_cut\n\ndef test_x():\n    assert min_cut.bound\n",
    "benchmarks/toy/toy/other.py": "def _min_cut(flow, cut):\n    return 0\n\nx = _min_cut(1, 2)\n",
}


def test_a_second_caller_is_found() -> None:
    """The check isn't vacuous: a second chooser that imports `min_cut` and calls `bound`, one that loads it by file
    behind a door and passes `F`, one that imports the siblings by name, and a test, are each found; a function that
    only shares the name is not."""
    where, package = calculator()
    texts = {rel: text.format(p=package) for rel, text in TOY.items()}
    found = reads(texts, where, package)
    assert set(found) == set(TOY) - {"benchmarks/toy/toy/other.py"}, found
    assert set(found["benchmarks/toy/toy/door.py"]) == {"slowdown_of", "F"}
    bad = offenders(found, {"benchmarks/toy/toy/tied.py": (("capped",), "toy")})
    assert "benchmarks/toy/toy/second.py: reaches the calculator's min_cut and is not in ALLOWED" in bad, bad
    assert "benchmarks/toy/toy/tied.py:1: calls min_cut.blocks_at_least, which only slowdown_of may call outside " \
           "the calculator" in bad, bad
    assert not any("capped" in b for b in bad), bad
    bad = offenders(found, dict.fromkeys(TOY, ((), "toy")))
    assert any("door.py:11: passes F" in b for b in bad) and any("second.py:6: calls min_cut.bound" in b for b in bad)
    assert any("test_toy.py:4: reads min_cut.bound, which ALLOWED does not list" in b for b in bad), bad
    assert not any("test_toy.py:4: calls" in b for b in bad), bad
    with pytest.raises(AssertionError, match=r"read your work into its `Product`s and call it"):
        check({**_tracked_python(), "benchmarks/toy/toy/second.py": texts["benchmarks/toy/toy/second.py"]}, ALLOWED)
