"""The benchmark reaches the slowdown calculator (`verity/protocols/inference_only/rho`) through one module,
`training_slowdown.calculator`: no other module or test under this directory imports a module of the calculator or
names its directory, and the slowdown the benchmark states is the calculator's bound of record, so with the
calculator's module replaced the products term follows it."""
import ast
from pathlib import Path
from types import SimpleNamespace

from training_slowdown import calculator
from training_slowdown import formalism as F

HERE = Path(__file__).resolve().parents[1]
#: the one module that reaches the calculator, and this file, whose toy sources name it
DOOR, THIS = "training_slowdown/calculator.py", "tests/test_one_calculator.py"
#: the calculator's modules as its directory on the import path names them, and the part of a path only its
#: directory and its policy's have
MODULES, DIRECTORY = {"rho", "periods"}, "inference_only"
#: files that name the calculator's directory in a message and read nothing of it, each with why. It only shrinks.
NAMED_ONLY = {"training_slowdown/config.py": "its refusal of a policy says where the registry's tables are"}


def reaches(source: str) -> list[str]:
    """`line: what` for each place `source` reaches the calculator: an import of one of its modules, or a string
    outside a docstring that names its directory (a path to it, whole or in parts)."""
    tree = ast.parse(source)
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, scopes) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(n.lineno, f"import {a.name}") for a in n.names if a.name.split(".")[0] in MODULES]
        elif isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] in MODULES:
            out.append((n.lineno, f"from {n.module} import"))
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str) and DIRECTORY in n.value
              and id(n) not in docstrings):
            out.append((n.lineno, repr(n.value[:60])))
    return [f"{line}: {what}" for line, what in sorted(out)]


def test_only_the_one_module_reaches_the_calculator():
    files = {str(p.relative_to(HERE)): p.read_text() for p in sorted(HERE.rglob("*.py"))}
    assert DOOR in files and reaches(files[DOOR])
    found = {name: hits for name, text in files.items() if name not in (DOOR, THIS) and (hits := reaches(text))}
    offenders = [f"{name}:{hit}" for name, hits in found.items() if name not in NAMED_ONLY for hit in hits]
    assert not offenders, ("only training_slowdown.calculator reaches the slowdown calculator: read what you need "
                           "through a function of it, adding one there if it has none\n" + "\n".join(offenders))
    stale = sorted(set(NAMED_ONLY) - set(found))
    assert not stale, f"NAMED_ONLY entries that no longer name the calculator's directory; remove them: {stale}"


def test_a_reach_is_found():
    """The check above isn't vacuous: an import by name, one behind a path put on the import path, a module loaded
    from its file and a path in parts are each found, and a docstring or a relative import of this package is not."""
    assert reaches("import sys\nsys.path.insert(0, 'verity/protocols/inference_only/rho')\nimport periods\n"
                   "from rho import min_cut\nimport rho.params\n") == [
        "2: 'verity/protocols/inference_only/rho'", "3: import periods", "4: from rho import", "5: import rho.params"]
    assert reaches("from pathlib import Path\nimport importlib.util\n"
                   "P = Path('verity') / 'protocols' / 'inference_only' / 'rho' / 'rho' / 'min_cut.py'\n"
                   "importlib.util.spec_from_file_location('m', P)\n") == ["3: 'inference_only'"]
    assert reaches('"""Reads verity/protocols/inference_only/rho through calculator."""\nfrom . import calculator\n'
                   'from .rho import x\n\ndef f():\n    """As inference_only says."""\n    rho = 1\n') == []


def test_the_products_term_is_the_calculators(monkeypatch):
    """`formalism` states which work is charged and computes none of the bound: with the calculator's module replaced
    by one whose bound is 42 blocks on every work, the products term is 42 blocks' slowdown, at the β and the F the
    replaced module gives."""
    real = vars(calculator._min_cut())
    seen = []

    def bound(c, k, n, **work):
        seen.append((c, k, n, work))
        return SimpleNamespace(slowdown=7.0, blocks=42.0, statement="a statement", rests_on="its premises")

    monkeypatch.setattr(calculator, "_min_cut", lambda: SimpleNamespace(**real | {
        "bound": bound, "beta": lambda k: 3.0, "share": lambda omega: 1.25}))
    group = {"beta": [16.0, 16.0], "width": [16, 16], "weight": [False, True], "b_y": 16, "forward": False,
             "k_dim": 64, "dw": False, "head": False, "macs": 1e9, "wires": 1e7}
    he = {"k": 65_536.0, "groups": [group], "r_step": None, "candidates_macs": 1e9, "self_macs": 0.0,
          "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}
    fig = F.he_figure(he, 1e11, 1e13)
    assert (fig["rho"], fig["beta_x"], fig["beta_k"], fig["F"]) == (42.0 * 1e11 / 1e13, 3.0, 3.0, 1.25)
    assert seen[0] == (1e11, 65_536.0, None, {"N": 1e9, "O": 1e7, "size": 1e13, "beta_w": 16.0, "beta_x": 3.0,
                                              "b": 16, "F": 1.25})
    assert fig["alpha"] == 1.25 * real["alpha"](65_536.0, 16.0 * 3.0, 16)
