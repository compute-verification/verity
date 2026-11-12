"""The benchmark reaches the slowdown calculator (`verity/protocols/inference_only/rho`) through one module,
`training_slowdown.calculator`, and computes a slowdown in one function of it, `calculator.slowdown`.

- One door: no other module or test under this directory imports a module of the calculator or names its directory
  (`reaches`), and with the calculator's module replaced the entry point follows it.
- The entry point is the calculator's: at three policies a config may name, its slowdown is `min_cut.bound`'s, asked
  directly, on the work a step's products state.
- One slowdown (`second`): outside the one module, no source of the benchmark reads a private name of it (an
  attribute `calculator._x` or `from ... calculator import _x`) or writes a `slowdown`, `rho` or `S` key in a function
  `STATES` doesn't list; inside it, only `_bound` calls the calculator's `bound`, `blocks_at_least` or
  `slowdown_at_least`, only `slowdown` and `_figure` call `_bound`, only `slowdown` calls `_figure`, and its public
  functions are `PUBLIC`. With the entry point replaced, what `STATES`' functions state follows it. The one ratio
  that is not the entry point's is Fit's c/(c − d), where a training fits one block (`formalism.at`)."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import SHAPE, calculators_count, figure

from training_slowdown import calculator, gates, library, policy, work
from training_slowdown import formalism as F

HERE = Path(__file__).resolve().parents[1]
#: the one module that reaches the calculator, and this file, whose toy sources name it
DOOR, THIS = "training_slowdown/calculator.py", "tests/test_one_calculator.py"
#: the calculator's package, which has its directory's name
PACKAGE = calculator.CALCULATOR.name
#: the calculator's modules as its directory on the import path names them, and the part of a path only its
#: directory and its policy's have
MODULES, DIRECTORY = {PACKAGE, "periods"}, "inference_only"
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
    home = calculator.CALCULATOR.relative_to(HERE.parents[1]).as_posix()
    assert reaches(f"import sys\nsys.path.insert(0, '{home}')\nimport periods\n"
                   f"from {PACKAGE} import min_cut\nimport {PACKAGE}.params\n") == [
        f"2: '{home}'", "3: import periods", f"4: from {PACKAGE} import", f"5: import {PACKAGE}.params"]
    assert reaches("from pathlib import Path\nimport importlib.util\n"
                   f"P = Path('verity') / 'protocols' / 'inference_only' / '{PACKAGE}' / '{PACKAGE}' / 'min_cut.py'\n"
                   "importlib.util.spec_from_file_location('m', P)\n") == ["3: 'inference_only'"]
    assert reaches(f'"""Reads {home} through calculator."""\nfrom . import calculator\n'
                   f'from .{PACKAGE} import x\n\ndef f():\n    """As inference_only says."""\n    {PACKAGE} = 1\n') == []


def test_the_products_term_is_the_calculators(monkeypatch):
    """The entry point chooses which work is charged and computes none of the bound: with the calculator's module
    replaced by one whose bound is 42 blocks on every work, its term is 42 blocks' slowdown, at the β and the F the
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
    fig = figure(he, 1e11, 1e13)
    assert (fig["rho"], fig["beta_x"], fig["beta_k"], fig["F"]) == (42.0 * 1e11 / 1e13, 3.0, 3.0, 1.25)
    assert seen[0] == (1e11, 65_536.0, None, {"N": 1e9, "O": 1e7, "size": 1e13, "beta_w": 16.0, "beta_x": 3.0,
                                              "b": 16, "F": 1.25})
    assert fig["alpha"] == 1.25 * real["alpha"](65_536.0, 16.0 * 3.0, 16)


def raw(phase: str, token_dims: list, m: int, k: int, n: int, count: int) -> dict:
    """A product row as a job's trace keeps one: bf16 operands at their width, the update reaching both."""
    return {"count": count, "phase": phase, "op": "mm", "dims": {"batch": 1, "m": m, "k": k, "n": n},
            "token_dims": token_dims, "dtypes": ["bfloat16", "bfloat16"], "affected": [True, True],
            "widths": [16, 16], "self": False, "gathered": [False, False], "rates": [16.0, 16.0]}


#: a step of 2^20 tokens through 64 matrices 2^16 wide: each one's forward product, input gradient and weight gradient
TOKENS, WIDTH, MATRICES = 2 ** 20, 2 ** 16, 64
LAYER = [raw("forward", ["m"], TOKENS, WIDTH, WIDTH, MATRICES), raw("backward", ["m"], TOKENS, WIDTH, WIDTH, MATRICES),
         raw("backward", ["k"], WIDTH, TOKENS, WIDTH, MATRICES)]
MACS = MATRICES * TOKENS * WIDTH * WIDTH
GATES = 3 * MACS * gates.per_mac("gemm/bf16")


def traced(k: float, rows: list[dict] = LAYER) -> dict:
    """`rows` as a job's trace of a step whose products it classified at k."""
    return {"k": k, "rows": rows, "p_match": [], "at": {}, "tokens": TOKENS, "circuit": {"gates": GATES}}


def test_the_registry_names_two_fixed_policies_beside_the_consoles():
    assert len(calculator.policies()) >= 2, "the test below reads three policies: the console's and two fixed ones"


@pytest.mark.parametrize("name", ("console", *calculator.policies()[:2]))
def test_the_entry_point_gives_the_calculators_bound_at_a_policy(name):
    """At the console's policy and at two fixed policies of the registry, on a traced step of one kind of layer: the
    entry point's slowdown is the calculator's own, `min_cut.bound` asked directly at the policy's (c, k), with no n,
    on the work the step's products state. That work is written out here by hand from the rows: the weights at their
    16 bits, the activations at the calculator's β(k); the forward products and input gradients at F = 2/√3, all
    three kinds at the activations' pair, and the weight gradients alone at F = 1. The entry point's is the one of
    them the calculator bounds highest, its least blocks, its statement and its premises are the calculator's, and
    the count on M binds."""
    pol = policy.named(name, SHAPE)
    c, k = pol["c_gates"], pol["k_bits"]
    got = calculator.slowdown(pol, traced(k))
    mod = calculator._min_cut()
    beta, wires = min(16.0, mod.beta(k)), 16 * MATRICES * TOKENS * WIDTH
    by_hand = [{"N": 2 * MACS, "O": 2 * wires, "beta_w": 16.0, "beta_x": beta, "b_y": 16, "F": mod.share(3)},
               {"N": 3 * MACS, "O": 2 * wires + 16 * MATRICES * WIDTH * WIDTH, "beta_w": beta, "beta_x": beta,
                "b_y": 16, "F": mod.share(3)},
               {"N": MACS, "O": 16 * MATRICES * WIDTH * WIDTH, "beta_w": beta, "beta_x": beta, "b_y": 16, "F": 1.0}]
    asked = [mod.bound(c, k, None, N=w["N"], O=w["O"], size=GATES, beta_w=w["beta_w"], beta_x=w["beta_x"],
                       b=w["b_y"], F=w["F"]) for w in by_hand]
    best = max(asked, key=lambda b: b.blocks)
    assert got["bound"]["work"] == by_hand[asked.index(best)] | {"C_gates": GATES}
    assert got["slowdown"] == best.slowdown == best.blocks * c / GATES > 1 and got["binds"] == "products"
    assert (got["bound"]["blocks"], got["bound"]["theorem"], got["bound"]["rests_on"]) == (
        best.blocks, best.statement, best.rests_on)
    assert got["term"] == calculators_count(c, k, got["bound"]["work"]) == got["slowdown"]
    assert got["why"] is None and got["read"]["k_classified"] == k and set(got) == {
        "slowdown", "binds", "why", "term", "bound", "figure", "read"} and got["binds"] in calculator.BINDS


def test_the_entry_point_says_what_binds_and_why_there_is_no_slowdown():
    """The floor binds, at 1, where the count is under the blocks the gates fill and where no product has two operands
    the update reaches; a record with no products, a refused trace and a k the rows weren't classified at have no
    slowdown, with the reason; and a trace without its circuit's gates has one only at the run's."""
    pol = policy.named("console", SHAPE)
    k = pol["k_bits"]
    small = calculator.slowdown(pol, traced(k, [raw("forward", ["m"], TOKENS, 8, 8, 1)]))
    assert (small["slowdown"], small["binds"]) == (1.0, "none") and 0 < small["term"] < 1
    assert small["bound"]["available"] and small["bound"]["slowdown"] == 1.0
    untouched = [r | {"affected": [False, False]} for r in LAYER]
    empty = calculator.slowdown(pol, traced(k, untouched))
    assert (empty["slowdown"], empty["binds"], empty["term"]) == (1.0, "none", 0.0)
    assert not empty["bound"]["available"] and empty["figure"]["subset"] is None
    for step, why in ((None, "no product rows"), ({"refused": {"reason": "r", "op": "o", "message": "m"}}, "(r, o: m)"),
                      (traced(80_000.0), "classified at k = 80000, not at this k")):
        none = calculator.slowdown(pol, step)
        assert (none["slowdown"], none["binds"], none["term"], none["figure"]) == (None, "unmeasured", None, None)
        assert why in none["why"] and not none["bound"]["available"]
    bare = {x: v for x, v in traced(k).items() if x != "circuit"}
    assert calculator.slowdown(pol, bare)["why"] == calculator.NO_GATES
    assert calculator.slowdown(pol, bare, {"gates": GATES}) == calculator.slowdown(pol, traced(k))
    assert calculator.slowdown(pol, traced(k), {"gates": 2 * GATES})["bound"]["work"]["C_gates"] == 2 * GATES


#: the sources a slowdown could be computed in: the package and the nested grader, without the tests
def sources() -> dict[str, str]:
    return {str(p.relative_to(HERE)): p.read_text() for p in sorted(HERE.rglob("*.py"))
            if "tests" not in p.relative_to(HERE).parts}


#: the one module's public functions. Only `slowdown` returns a slowdown: one that does is part of it, not a
#: function beside it
PUBLIC = {"policies", "policy", "window", "periods", "fleets", "beta_rows", "theorem", "he_past", "slowdown"}
#: the keys a slowdown is written under
KEYS = {"slowdown", "rho", "S"}
#: the functions outside the one module that write one of `KEYS`, each with whose slowdown it writes. It only shrinks.
STATES = {
    ("training_slowdown/formalism.py", "at"):
        "`rho`: Fit's c/(c − d) where the training fits one block, the one ratio that is no bound of the "
        "calculator's, else the entry point's slowdown",
    ("training_slowdown/formalism.py", "published"): "`at`'s `rho`",
    ("training_slowdown/grade.py", "slowdown_of"): "`published`'s `rho`, or none with the reason",
    ("training_slowdown/grade.py", "grade"): "`slowdown_of`'s `rho`, and `S`, p times it",
    ("training_slowdown/library.py", "bound"): "the entry point's slowdown, or none with its reason",
    ("training_slowdown/library.py", "row"): "`bound`'s slowdown",
}
#: the calculator's functions that give a bound, and inside the one module who may call what
BOUNDS = {"bound", "blocks_at_least", "slowdown_at_least"}
CALLERS = {"_bound": {"slowdown", "_figure"}, "_figure": {"slowdown"}}


def second(name: str, source: str) -> list[str]:
    """`line: what` for each place the source `name` could compute a second slowdown (the module docstring's list).
    A place is in the outermost function that holds it."""
    tree = ast.parse(source)
    inside = {}
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(f):
                inside.setdefault(id(n), f.name)
    out = []
    for n in ast.walk(tree):
        at = inside.get(id(n), "the module")
        if name == DOOR:
            asks = isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in BOUNDS
            if asks and at != "_bound":
                out.append((n.lineno, f"{at} calls the calculator's {n.func.attr}: only _bound does"))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and at not in CALLERS.get(n.func.id, {at}):
                out.append((n.lineno, f"{at} calls {n.func.id}: only {sorted(CALLERS[n.func.id])} do"))
            continue
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "calculator" \
                and n.attr.startswith("_"):
            out.append((n.lineno, f"reads calculator.{n.attr}"))
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[-1] == "calculator":
            out += [(n.lineno, f"imports calculator's {a.name}") for a in n.names if a.name.startswith("_")]
        if isinstance(n, ast.Dict) and (name, at) not in STATES:
            out += [(k.lineno, f"{at} writes the key {k.value!r}") for k in n.keys
                    if isinstance(k, ast.Constant) and k.value in KEYS]
    if name == DOOR:
        public = {f.name for f in tree.body if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")}
        out += [(0, f"a public function beside the entry point: {x}") for x in sorted(public - PUBLIC)]
    return [f"{line}: {what}" for line, what in sorted(out)]


def test_no_second_slowdown_is_computed():
    files = sources()
    found = [f"{name}:{hit}" for name, text in files.items() for hit in second(name, text)]
    assert not found, ("the benchmark computes a slowdown in calculator.slowdown alone: take it from there (its "
                       "`slowdown`, `binds`, `bound`), and list a function that passes one on in STATES\n"
                       + "\n".join(found))
    wrote = {(name, f.name) for name, text in files.items() for f in ast.parse(text).body
             if isinstance(f, ast.FunctionDef) for n in ast.walk(f) if isinstance(n, ast.Dict)
             for k in n.keys if isinstance(k, ast.Constant) and k.value in KEYS}
    assert not set(STATES) - wrote, f"STATES entries that write no slowdown; remove them: {sorted(set(STATES) - wrote)}"


def test_a_second_slowdown_is_found():
    """The check above isn't vacuous: a private name of the one module read by attribute or imported, a slowdown
    written by a function `STATES` doesn't list, and, inside the one module, a second caller of the calculator's
    bound, a second caller of `_bound` and a public function beside the entry point are each found."""
    assert second("training_slowdown/other.py", "from . import calculator\n\ndef f(c, k, w):\n"
                  "    return calculator._bound(c, k, w)\n") == ["4: reads calculator._bound"]
    assert second("grader/other.py", "from training_slowdown.calculator import _figure, fleets\n") == [
        "1: imports calculator's _figure"]
    assert second("training_slowdown/other.py", "def mine(blocks, c, C):\n"
                  "    return {'slowdown': max(1.0, blocks * c / C), 'blocks': blocks}\n") == [
        "2: mine writes the key 'slowdown'"]
    assert second("training_slowdown/formalism.py", "def at(m):\n    return {'rho': m}\n\n"
                  "def other(m):\n    return {'S': m}\n") == ["5: other writes the key 'S'"]
    assert second(DOOR, "def _bound(c):\n    return _min_cut().bound(c)\n\n"
                  "def slowdown(policy, step):\n    return _bound(policy)\n\n"
                  "def bound(c):\n    return _bound(c)\n\n"
                  "def _other(c):\n    return _min_cut().blocks_at_least(c)\n") == [
        "0: a public function beside the entry point: bound", "8: bound calls _bound: only ['_figure', 'slowdown'] do",
        "11: _other calls the calculator's blocks_at_least: only _bound does"]


def test_every_slowdown_the_benchmark_states_is_the_entry_points(monkeypatch):
    """With the entry point replaced by one that gives 7.25 on every record (beside a term of 3.5, which a caller
    flooring the term itself would give), `at`'s `rho`, the published slowdown and the library's row give 7.25,
    binding the products, and the one bound's products term is the entry point's term; where it gives none, they
    give none with its reason, whatever the sensitivities beside it are; and where it gives the floor, 1, binding
    none. The one ratio that isn't the entry point's is Fit's c/(c − d), where a training fits one block."""
    real, given = calculator.slowdown, {}

    def entry(pol, step, run=None):
        got = real(pol, step, run)
        bound = {"bound": got["bound"] | {"slowdown": given["slowdown"]}} if given.get("slowdown") else {}
        return got | given | bound

    monkeypatch.setattr(calculator, "slowdown", entry)
    pol = policy.named("console", SHAPE)
    c, k = pol["c_gates"], pol["k_bits"]
    part = F.Part(gates=GATES, tokens=TOKENS, token_gates=GATES, affected_gates=GATES / 2)
    m = F.Measured((part,), tau_bits=16, a_bits=16, u=F.even(2000.0), u_in=F.even(2000.0), delta_bits=None,
                   decoder_gates=0.0, products=work.products_of_trace(work.priced(traced(k))))
    honest = F.published(m, c, k)["rho"]
    assert honest == real(pol, traced(k))["slowdown"] > 1
    for give, slowdown, binding in (({"slowdown": 7.25, "term": 3.5, "binds": "products"}, 7.25, "products"),
                               ({"slowdown": 1.0, "term": 0.5, "binds": "none"}, 1.0, "none"),
                               ({"slowdown": None, "term": None, "binds": "unmeasured", "why": "unmeasured: said so"},
                                None, "unmeasured")):
        given.clear()
        given.update(give)
        b, one = F.bound(m, c, k), F.published(m, c, k)
        assert b["entry"]["slowdown"] == slowdown and b["terms"]["products"] == give["term"]
        assert F.at(m, c, k)["rho"] == slowdown and (one["rho"], one["binding_kind"]) == (slowdown, binding)
        row = library.bound(traced(k), c, k)
        assert row["slowdown"] == slowdown
        if slowdown is None:
            assert b["unmeasured"]["products"] == row["binds"] == "unmeasured: said so"
    fits = F.Measured((F.Part(gates=1e12, tokens=1e3, token_gates=0.8e12, affected_gates=0.4e12, delta_grad_bits=1e5),),
                      tau_bits=32, a_bits=8192, u=F.even(1e5), u_in=F.even(1e5), delta_bits=1e5, decoder_gates=1e12)
    one = F.published(fits, 1e13, 1e6)
    assert (one["rho"], one["binding_kind"]) == (pytest.approx(1e13 / (1e13 - 1e12)), "fit")
    assert math.isfinite(one["rho"]) and one["rho"] != 7.25
