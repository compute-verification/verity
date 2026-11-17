"""The benchmark reaches the slowdown calculator (`benchmarks/slowdown`) through one module,
`training_slowdown.calculator`, and computes a slowdown in one function of it, `calculator.slowdown`.

- One door: no other module or test under this directory imports a module of the calculator or names its directory
  (`reaches`), and with the calculator's module replaced the entry point follows it.
- The entry point is the calculator's: at three policies a config may name, its slowdown is `min_cut.bound`'s, asked
  directly, on the work a step's products state.
- One slowdown (`second`): outside the one module, no source of the benchmark reads a private name of it (an
  attribute `calculator._x` or `from ... calculator import _x`) or writes a `slowdown`, `record_slowdown` or `S` key in a function
  `STATES` doesn't list; inside it, nothing calls the calculator's `bound`, `blocks_at_least` or `slowdown_at_least`,
  only `_figure` calls its `slowdown_of`, only `slowdown` calls `_bound` and `_figure`, and its public functions are
  `PUBLIC`. With the entry point replaced, what `STATES`' functions state follows it."""
import ast
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import SHAPE, calculators_count, figure

from training_slowdown import calculator, gates, library, policy, work

HERE = (Path(__file__).resolve().parents[3] / "benchmarks" / "training_slowdown")
#: the one module that reaches the calculator, and this file, whose toy sources name it
DOOR, THIS = "training_slowdown/calculator.py", "tests/test_one_calculator.py"
#: the calculator's package, which has its directory's name
PACKAGE = calculator.CALCULATOR.name
#: the calculator's modules as its directory on the import path names them, and its directory's name: a part of every
#: path to it, and alone a key of the benchmark's own records too
MODULES, DIRECTORY = {PACKAGE, "periods"}, PACKAGE
#: the directory in a path written whole: its name as a whole part, after the directory it is in or a step up to it
WHOLE = re.compile(rf"(?:^|[^\w.-])(?:{re.escape(calculator.CALCULATOR.parent.name)}|\.\.)"
                   rf"/{re.escape(DIRECTORY)}(?![\w.-])")
#: files that name the calculator's directory in a message and read nothing of it, each with why. It only shrinks.
NAMED_ONLY = {"training_slowdown/config.py": "its refusal of a policy says where the registry's tables are"}


def reaches(source: str) -> list[str]:
    """`line: what` for each place `source` reaches the calculator: an import of one of its modules, or a string
    outside a docstring that names its directory as a part of a path: in a path written whole, the part after the
    directory it is in (`benchmarks/slowdown/data`) or after a step up, or the name alone where `/` joins it (a path in
    parts). The name anywhere else is the benchmark's own: a key, or the first part of an id, as `slowdown/v1`, the
    grade's schema, is not a path."""
    tree = ast.parse(source)
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, scopes) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    joined = {id(side) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
              for side in (n.left, n.right)}
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(n.lineno, f"import {a.name}") for a in n.names if a.name.split(".")[0] in MODULES]
        elif isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] in MODULES:
            out.append((n.lineno, f"from {n.module} import"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            if WHOLE.search(n.value) or (n.value == DIRECTORY and id(n) in joined):
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
    """The check above isn't vacuous: an import by name, one behind a path put on the import path (a path written
    whole), a module loaded from its file and a path in parts (its directory, then its package, each an operand of
    `/`) are each found, as is a path that steps up to it, and a docstring, a relative import of this package, a
    dictionary's key of the directory's name, a subscript's or an id that begins with it is not."""
    home = calculator.CALCULATOR.relative_to(HERE.parents[1]).as_posix()
    assert reaches(f"import sys\nsys.path.insert(0, '{home}')\nimport periods\n"
                   f"from {PACKAGE} import min_cut\nimport {PACKAGE}.params\n") == [
        f"2: '{home}'", "3: import periods", f"4: from {PACKAGE} import", f"5: import {PACKAGE}.params"]
    assert reaches("from pathlib import Path\nimport importlib.util\n"
                   f"P = Path('benchmarks') / '{PACKAGE}' / '{PACKAGE}' / 'min_cut.py'\n"
                   "importlib.util.spec_from_file_location('m', P)\n") == [f"3: '{PACKAGE}'", f"3: '{PACKAGE}'"]
    assert reaches(f'"""Reads {home}/data through calculator."""\nfrom . import calculator\n'
                   f'from .{PACKAGE} import x\n\ndef f():\n    """As {home}/data says."""\n    {PACKAGE} = 1\n') == []
    assert reaches(f'got = {{"{PACKAGE}": 1.0}}\n') == []
    assert reaches(f'half = got["{PACKAGE}"] / 2\n') == []
    assert reaches(f"import sys\nsys.path[:0] = ['../{PACKAGE}']\n") == [f"2: '../{PACKAGE}'"]
    assert reaches(f'SCHEMA = "{PACKAGE}/v1"\n') == []


def test_the_products_term_is_the_calculators(monkeypatch):
    """The entry point reads M into the calculator's products, priced, and chooses none of the work and computes none
    of the bound: with the calculator's `slowdown_of` replaced by one that chooses 42 blocks at F = 1.25 on every
    work, the figure and the bound are that choice's, and what the entry point handed over is M's group with its
    activation at the replaced module's β(k)."""
    real = vars(calculator._min_cut())
    seen = []

    def slowdown_of(c, k, n, *, size, work, scale):
        seen.append((c, k, n, size, scale, work))
        b = SimpleNamespace(slowdown=7.0, blocks=42.0, statement="a statement", rests_on="its premises")
        term = 42.0 * c / size
        return real["Chosen"](b, "M", (0,), term, {"M": term}, 8.0, q=48.0, beta_w=16.0, beta_x=3.0, b_y=16, F=1.25,
                              alpha=0.5, N=1e9, O=1e7)

    monkeypatch.setattr(calculator, "_min_cut", lambda: SimpleNamespace(**real | {
        "slowdown_of": slowdown_of, "beta": lambda k: 3.0}))
    group = {"beta": [16.0, 16.0], "width": [16, 16], "weight": [False, True], "b_y": 16, "forward": False,
             "k_dim": 64, "dw": False, "head": False, "macs": 1e9, "wires": 1e7}
    he = {"k": 65_536.0, "groups": [group], "r_step": None, "candidates_macs": 1e9, "self_macs": 0.0,
          "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}
    fig = figure(he, 1e11, 1e13)
    assert (fig["term"], fig["subset"], fig["beta_w"], fig["beta_x"], fig["beta_k"], fig["F"], fig["alpha"]) == (
        42.0 * 1e11 / 1e13, "M", 16.0, 3.0, 3.0, 1.25, 0.5)
    assert (fig["beta"], fig["K_thr"], fig["subsets"], fig["macs_share"]) == (
        48.0 ** 0.5, 8.0, {"M": 42.0 * 1e11 / 1e13}, 1.0)
    assert seen == [(1e11, 65_536.0, None, 1e13, 1.0, [real["Product"](1e9, 1e7, (3.0, 16.0), (False, True), 16, 64)])]
    got = calculator.slowdown({"c_gates": 1e11, "k_bits": 65_536.0}, {"k": 65_536.0, "he": he}, {"gates": 1e13})
    assert (got["slowdown"], got["bound"]["blocks"], got["bound"]["theorem"]) == (7.0, 42.0, "a statement")


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
KEYS = {"slowdown", "record_slowdown", "S"}
#: the functions outside the one module that write one of `KEYS`, each with whose slowdown it writes. It only shrinks.
STATES = {
    ("training_slowdown/grade.py", "grade"): "`slowdown_of`'s result as `record_slowdown`, and `S`, p times its `slowdown`",
    ("training_slowdown/library.py", "bound"): "the entry point's slowdown, or none with its reason",
    ("training_slowdown/library.py", "row"): "`bound`'s slowdown",
}
#: the calculator's functions that give a bound, which nothing here calls; its one way from a work to a slowdown; and
#: inside the one module who may call what
BOUNDS = {"bound", "blocks_at_least", "slowdown_at_least"}
ENTRY = "slowdown_of"
CALLERS = {"_bound": {"slowdown"}, "_figure": {"slowdown"}}


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
            asks = isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in BOUNDS | {ENTRY}
            if asks and n.func.attr in BOUNDS:
                out.append((n.lineno, f"{at} calls the calculator's {n.func.attr}: ask its {ENTRY}"))
            elif asks and at != "_figure":
                out.append((n.lineno, f"{at} calls the calculator's {ENTRY}: only _figure does"))
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
    written by a function `STATES` doesn't list, and, inside the one module, a caller of the calculator's bound, a
    second caller of its `slowdown_of`, a second caller of `_bound` and a public function beside the entry point are
    each found."""
    assert second("training_slowdown/other.py", "from . import calculator\n\ndef f(c, k, w):\n"
                  "    return calculator._bound(c, k, w)\n") == ["4: reads calculator._bound"]
    assert second("grader/other.py", "from training_slowdown.calculator import _figure, fleets\n") == [
        "1: imports calculator's _figure"]
    assert second("training_slowdown/other.py", "def mine(blocks, c, C):\n"
                  "    return {'slowdown': max(1.0, blocks * c / C), 'blocks': blocks}\n") == [
        "2: mine writes the key 'slowdown'"]
    assert second("training_slowdown/library.py", "def bound(m):\n    return {'slowdown': m}\n\n"
                  "def other(m):\n    return {'S': m}\n") == ["5: other writes the key 'S'"]
    assert second(DOOR, "def _figure(c):\n    return _min_cut().slowdown_of(c)\n\n"
                  "def slowdown(policy, step):\n    return _bound(policy)\n\n"
                  "def bound(c):\n    return _bound(c)\n\n"
                  "def _other(c):\n    return _min_cut().blocks_at_least(c), _min_cut().slowdown_of(c)\n") == [
        "0: a public function beside the entry point: bound", "8: bound calls _bound: only ['slowdown'] do",
        "11: _other calls the calculator's blocks_at_least: ask its slowdown_of",
        "11: _other calls the calculator's slowdown_of: only _figure does"]


def test_every_slowdown_the_benchmark_states_is_the_entry_points(monkeypatch):
    """With the entry point replaced by one that gives 7.25 on every record (beside a term of 3.5, which a caller
    flooring the term itself would give), the library's row gives 7.25; where it gives none, none with its reason;
    and where it gives the floor, 1."""
    real, given = calculator.slowdown, {}

    def entry(pol, step, run=None):
        got = real(pol, step, run)
        bound = {"bound": got["bound"] | {"slowdown": given["slowdown"]}} if given.get("slowdown") else {}
        return got | given | bound

    monkeypatch.setattr(calculator, "slowdown", entry)
    pol = policy.named("console", SHAPE)
    c, k = pol["c_gates"], pol["k_bits"]
    assert real(pol, traced(k))["slowdown"] > 1
    for give, slowdown in (({"slowdown": 7.25, "term": 3.5, "binds": "products"}, 7.25),
                           ({"slowdown": 1.0, "term": 0.5, "binds": "none"}, 1.0),
                           ({"slowdown": None, "term": None, "binds": "unmeasured", "why": "unmeasured: said so"},
                            None)):
        given.clear()
        given.update(give)
        row = library.bound(traced(k), c, k)
        assert row["slowdown"] == slowdown
        if slowdown is None:
            assert row["binds"] == "unmeasured: said so"


FIXTURES = Path(__file__).parent / "fixtures"
#: the calculator's choices of the work, the subsets of M the entry point's figure names
SUBWORKS = calculator._min_cut().SUBWORKS


def term_of(products: dict, c: float, k: float, gates: float = 1e12, tau: float = 16.0,
            code: float | None = None) -> tuple[float | None, str | None, dict]:
    """The entry point's term on `products` at (c, k) over |C| = `gates`, its reason where it has none, and what it
    read with its figure."""
    got = calculator.slowdown({"c_gates": c, "k_bits": k}, products, {"gates": gates, "tau_bits": tau, "code_bits": code})
    return got["term"], None if got["term"] is not None else got["why"], got["read"] | (got["figure"] or {})


#: one bf16 product of two fresh operands over K = 512, half of 1e12 gates: its MACs and output entries
SQUARE = {"macs": 0.5e12 / (23682 / 16), "outputs": 0.5e12 / (23682 / 16) / 512}


def stated(k: float) -> dict:
    """SQUARE as the products term's M at k, in `he_of_rows`' form: one input gradient, an activation times a trained
    weight at 16 bits each with 16-bit outputs, which doesn't say whether it is the head's."""
    group = {"beta": [16, 16], "width": [16, 16], "weight": [False, True], "b_y": 16, "forward": False, "k_dim": 512,
             "dw": False, "head": None, "macs": SQUARE["macs"], "wires": SQUARE["outputs"] * 16}
    return {"k": float(k), "groups": [group], "r_step": None, "candidates_macs": SQUARE["macs"], "self_macs": 0.0,
            "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}


#: SQUARE as a run's products at the console's k
STATED = {"k": policy.CONSOLE[1], "he": stated(policy.CONSOLE[1])}


def test_the_products_term_is_the_theorem_of_records_on_m():
    """The products term is the slowdown the theorem of record gives on M (the entry point's, on the products' `he`)
    and the run's |C|, linear in c at a fixed k. It is unmeasured at a k the products weren't classified at, on
    products that come without M (no rows), on rows that don't flag a gather's and where an elementwise op fans out
    into an operand."""
    c, k = 1e11, policy.CONSOLE[1]
    term, why, _ = term_of(STATED, c, k)
    fig = figure(STATED["he"], c, 1e12)
    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["b_y"]) == ("M", 16, 15.9, 16)
    assert not fig["below_width"] and fig["macs_share"] == 1
    assert fig["F"] == pytest.approx(2 / math.sqrt(3), rel=1e-15) and fig["N"] == SQUARE["macs"]
    assert fig["O"] == SQUARE["outputs"] * 16
    assert term == fig["term"] > 1 and why is None
    assert term_of(STATED, 2 * c, k)[0] == pytest.approx(2 * fig["term"], rel=1e-12)
    term, why, _ = term_of(STATED, c, 80_000)
    assert term is None and "classified at k = 65536" in why
    assert term_of({"k": k}, c, k)[1] == work.HE_NO_ROWS
    assert term_of(STATED | {"he": None}, c, k)[1] == work.HE_RETRACE
    term, why, _ = term_of(STATED | {"p_match": [{"op": "expand", "fan_out": 16.0, "count": 1}]}, c, k)
    assert term is None and "breaks the premise" in why


def he_row(phase: str, td: list, m: int, k: int, n: int, count: int = 1, batch: int = 1) -> dict:
    """A product row as training_slowdown.capture.products writes one: both operands affected at bf16, paid, none gathered."""
    gates = next(g for g, b in work.HE_BITS.items() if b == (16, 16))
    return {"phase": phase, "token_dims": td, "dims": {"batch": batch, "m": m, "k": k, "n": n}, "count": count,
            "macs": batch * m * k * n, "outputs": batch * m * n, "affected": [True, True], "self": False,
            "gathered": [False, False], "widths": [16, 16], "rates": [16.0, 16.0], "cell_gates": gates}


#: a step of 1,024 tokens: two layers' 1,024-wide matrix, a 4,096-wide head and an 8-long product with wide outputs,
#: forward, input and weight gradients, beside the optimizer's products and attention's
HE_STEP = [he_row("forward", ["m"], 1024, 1024, 1024, 2), he_row("forward", ["m"], 1024, 1024, 4096),
           he_row("forward", ["m"], 1024, 8, 4096), he_row("backward", ["m"], 1024, 4096, 1024),
           he_row("backward", ["k"], 4096, 1024, 1024), he_row("backward", ["m"], 1024, 1024, 1024, 2),
           he_row("backward", ["k"], 1024, 1024, 1024, 2), he_row("optimizer", [], 1024, 1024, 1024, 4),
           he_row("forward", ["batch", "m", "n"], 256, 64, 256, 1, 4)]


def test_the_products_term_takes_the_best_of_three_ms_at_beta_k_with_the_code_size_beta_w_and_the_first_steps():
    """4.13.1's rules on M (`work.he_of_rows`, the entry point's choice of the work): the optimizer's products and attention's leave M; a weight is an
    operand no token indexes, a backward product contracting over tokens a weight gradient, and the head's are the
    backward products over its vocabulary width (the widest forward output); the trained weights are the weights a
    step's forward reads, each matrix once a token. Activations are charged β(k), 15.9 up to 145,568, between two
    red-team points the next one's (from 4.13.5), none past 8e8, weights their width or less the record's code bits a trained weight,
    a weight's β the β_w a product reports beside its activation's β_x. The term is the best of all of M, M less its contractions below K_thr (here the 8-long product's wide outputs,
    whose leaving lifts the term) and the weight gradients alone at F = 1; over a run's steps it loses the ⌊k/(τT)⌋
    after the first update, and from 8e7 the head's products leave M. Model products that sum the head's gradients
    with the layers' are unmeasured from 8e7, and every M past 8e8."""
    c, k, C, T = 1e11, 65_536.0, 1e13, 1024
    beta = calculator._min_cut().beta
    assert (beta(1.0), beta(145_568), beta(8e6), beta(8e8)) == (15.9, 15.9, 10.8, 0.37)
    assert beta(8e6 + 1) == beta(math.sqrt(8e6 * 8e7)) == beta(8e7) == 2.7 and beta(8.1e8) is None
    he = work.he_of_rows(HE_STEP, k, 10.0, T)
    opt, att = HE_STEP[7], HE_STEP[8]
    assert he["left_out_macs"] == 10 * (opt["count"] * opt["macs"] + att["count"] * att["macs"])
    assert he["trained_weights"] == 2 * 1024 * 1024 + 1024 * 4096 + 8 * 4096
    assert sorted((g["k_dim"], g["dw"], g["head"], tuple(g["weight"])) for g in he["groups"]) == [
        (8, False, False, (False, True)), (1024, False, False, (False, True)), (1024, False, False, (False, True)),
        (1024, True, False, (False, False)), (1024, True, True, (False, False)), (4096, False, True, (False, True))]
    fig = figure(he, c, C)
    assert fig["subset"] == SUBWORKS[1] and fig["term"] == max(fig["subsets"].values()) > fig["subsets"]["M"]
    assert 8 < fig["K_thr"] < 1024 and fig["beta_k"] == 15.9 and fig["F"] == pytest.approx(2 / math.sqrt(3))
    dw = [g for g in he["groups"] if g["dw"]]
    N, O = sum(g["macs"] for g in dw), sum(g["wires"] for g in dw)
    alone = {"N": N, "O": O, "beta_w": 15.9, "beta_x": 15.9, "b_y": 16, "F": 1.0, "C_gates": C}
    assert fig["subsets"][SUBWORKS[2]] == calculators_count(c, k, alone)
    first = figure(he, c, C, tau=16)
    assert first["first_steps"] == 4 and first["N"] == pytest.approx(0.6 * fig["N"], rel=1e-12)
    assert fig["first_steps"] is None and figure(he, c, C, tau=1e3)["first_steps"] == 0
    coded = figure(he, c, C, beta_code=2.0)
    assert coded["term"] < fig["term"] and coded["beta_code"] == 2.0
    one = he | {"groups": [g for g in he["groups"] if g["k_dim"] == 1024 and not g["dw"] and not g["head"]][:1]}
    weighed = figure(one, c, C, beta_code=8.0)
    assert (weighed["beta_w"], weighed["beta_x"], weighed["code_bound"]) == (8.0, 15.9, True)
    term, why, got = term_of({"k": k, "he": he}, c, k, code=2.0 * he["trained_weights"])
    assert why is None and got["beta_code"] == 2.0 and term == figure(he, c, 1e12, 16.0, 2.0)["term"]
    far = work.he_of_rows(HE_STEP, 8e7)
    assert figure(far, c, C) == figure(far | {"groups": [g for g in far["groups"] if not g["head"]]}, c, C)
    assert figure(he, c, C) != figure(he | {"groups": [g for g in he["groups"] if not g["head"]]}, c, C)
    for at, why in ((8e7, work.HE_HEAD), (9e8, calculator.he_past())):
        assert term_of({"k": at, "he": stated(at)}, c, at)[:2] == (None, why), at
    with pytest.raises(ValueError, match="past k"):
        figure(work.he_of_rows(HE_STEP, 9e8), c, C)


def test_all_of_m_binds_where_the_count_puts_it_above_the_weight_gradients():
    """The subset that binds is the one whose count is largest. Here M's forward and input-gradient products give many
    output wires a MAC: by the count all of M binds, at F = 2/√3 and the code's β_w."""
    k, c, C = policy.CONSOLE[1], 1e14, 1e18

    def group(macs, wires, weight, forward, dw, k_dim):
        return {"beta": [16.0, 16.0], "width": [16.0, 16.0], "weight": weight, "b_y": 16.0, "forward": forward,
                "k_dim": k_dim, "dw": dw, "head": False, "macs": macs, "wires": wires}

    groups = [group(1.5e15, 7.5e11, [False, False], False, True, 4096.0),
              group(1.6e15, 3.3e13, [False, True], True, False, 512.0),
              group(1.6e15, 3.3e13, [False, True], False, False, 512.0)]
    he = {"k": k, "groups": groups, "r_step": None, "candidates_macs": sum(g["macs"] for g in groups),
          "self_macs": 0.0, "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None,
          "trained_weights": None}
    fig = figure(he, c, C, beta_code=8.0)
    whole = {"N": 4.7e15, "O": 7.5e11 + 2 * 3.3e13, "beta_w": 8.0, "beta_x": 15.9, "b_y": 16.0,
             "F": calculator._min_cut().share(3), "C_gates": C}
    alone = {"N": 1.5e15, "O": 7.5e11, "beta_w": 15.9, "beta_x": 15.9, "b_y": 16.0, "F": 1.0, "C_gates": C}
    count = {name: calculators_count(c, k, w) for name, w in (("M", whole), ("dw", alone))}
    assert (fig["subset"], fig["F"], fig["beta_w"], fig["N"], fig["O"]) == (
        "M", calculator._min_cut().share(3), 8.0, whole["N"], whole["O"])
    assert fig["term"] == fig["subsets"]["M"] == count["M"] > count["dw"] == fig["subsets"][SUBWORKS[2]]


def test_the_adapter_rule_needs_a_frozen_weight_the_update_reaches_and_a_frozen_head_keeps_its_width():
    """A distilling teacher's weights are frozen but read only tokens, so the
    student's head (its d_model widened to the vocabulary) is no adapter: r_step is None and every activation keeps its
    width, as without the teacher. A LoRA's frozen layers read what its adapters wrote, so B·(Ax) sets r_step to its
    rank and an activation wider than it is charged β_eff. With its head frozen, the vocabulary is still the head's
    width, so the adapters' backward products over d_model aren't the head's; a trained head's are."""
    k, d, V, r = 65_536.0, 512, 32_000, 8

    def frozen(row, *i):
        return row | {"affected": [j not in i for j in (0, 1)]}

    student = [he_row("forward", ["m"], 1024, 256, 704), he_row("forward", ["m"], 1024, 704, 256),
               he_row("forward", ["m"], 1024, 256, V)]
    alone, distilled = (work.he_of_rows(rows, k) for rows in (student, [frozen(he_row("forward", ["m"], 1024, d, V), 0, 1),
                                                                     *student]))
    assert distilled["r_step"] is alone["r_step"] is None
    assert distilled["groups"] == alone["groups"] and all(b == 16 for g in alone["groups"] for b in g["beta"])
    lora = [frozen(he_row("forward", ["m"], 1024, d, d), 1), he_row("forward", ["m"], 1024, d, r),
            he_row("forward", ["m"], 1024, r, d), frozen(he_row("forward", ["m"], 1024, d, V), 1),
            he_row("backward", ["m"], 1024, d, r), he_row("backward", ["k"], d, 1024, r)]
    he = work.he_of_rows(lora, k)
    assert he["r_step"] == r and not any(g["head"] for g in he["groups"])
    dx = [g["beta"] for g in he["groups"] if not g["forward"] and not g["dw"]]
    assert dx == [[round(work.beta_eff(16, r, k), 9), 16]]
    trained = work.he_of_rows([*lora[:3], he_row("forward", ["m"], 1024, d, V), he_row("backward", ["m"], 1024, V, d)], k)
    assert [g["k_dim"] for g in trained["groups"] if g["head"]] == [V]


def test_the_products_term_charges_m_its_least_pair_where_a_layers_arrays_differ_in_beta():
    """F = 2/√3 is proved for one β on a layer's three shared arrays (weights, activations, output gradients); with a β
    each, Motzkin-Straus sums u_r·u_s/(β_r·β_s) over the pairs M's products multiply, so both terms take the least
    pair product among M's products. Weights at 2 (the record's code size), activations at β(k) = 15.9 and output
    gradients at 8: all of M binds at the input gradient's 2·8 = 16, below the forward product's 2·15.9, which would
    read higher, and each threshold's M_P keeps only the products whose pair is at least its own."""
    c, k, C, m, K = 1e11, 65_536.0, 1e13, 256, 65_536

    def group(beta: list, weight: list, forward: bool, dw: bool, scale: float) -> dict:
        return {"beta": beta, "width": [16, 16], "weight": weight, "b_y": 16, "forward": forward, "k_dim": K,
                "dw": dw, "head": False, "macs": scale * m * m * K, "wires": scale * m * m * 16}

    fwd, dx = group([16.0, 16.0], [False, True], True, False, 1), group([8.0, 16.0], [False, True], False, False, 4)
    dw = group([8.0, 16.0], [False, False], False, True, 0.25)
    he = {"k": k, "groups": [fwd, dx, dw], "r_step": None, "candidates_macs": 5.25 * m * m * K, "self_macs": 0.0,
          "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}
    fig = figure(he, c, C, beta_code=2.0)
    share = 2 / math.sqrt(3)

    def slowdown(M: list, bw: float, bx: float, f: float = share) -> float:
        work = {"N": sum(g["macs"] for g in M), "O": sum(g["wires"] for g in M), "beta_w": bw, "beta_x": bx,
                "b_y": 16, "F": f, "C_gates": C}
        return calculators_count(c, k, work)

    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["code_bound"]) == ("M", 2.0, 8.0, True)
    assert fig["F"] == pytest.approx(share, rel=1e-15) and fig["macs_share"] == 1
    assert fig["term"] == pytest.approx(slowdown([fwd, dx, dw], 2.0, 8.0), rel=1e-12)
    assert slowdown([fwd, dx, dw], 2.0, 8.0) < slowdown([fwd, dx, dw], 2.0, 15.9)
    assert fig["term"] > max(slowdown([fwd, dw], 2.0, 15.9), slowdown([dw], 15.9, 8.0), slowdown([dw], 15.9, 8.0, 1.0)) > 0


def test_f_is_1_where_the_binding_m_holds_forward_products_alone():
    """F = 2/√3 (the calculator's `min_cut.share`) where the M_P a term takes holds a backward product, since a layer's forward,
    input-gradient and weight-gradient products share three arrays, and 1 where it holds forward products alone: a
    forward product sixteen times the gradients' MACs binds at its own pair, alone, at F = 1, and gradients sixteen
    times its bind at the least pair, with all three, at 2/√3."""
    c, k, C, m, K = 1e11, 65_536.0, 1e13, 256, 65_536

    def group(beta: list, weight: list, forward: bool, dw: bool, scale: float) -> dict:
        return {"beta": beta, "width": [16, 16], "weight": weight, "b_y": 16, "forward": forward, "k_dim": K,
                "dw": dw, "head": False, "macs": scale * m * m * K, "wires": scale * m * m * 16}

    def step(fwd: float, back: float) -> dict:
        return {"k": k, "groups": [group([16.0, 16.0], [False, True], True, False, fwd),
                                   group([8.0, 16.0], [False, True], False, False, back),
                                   group([8.0, 16.0], [False, False], False, True, back)],
                "r_step": None, "candidates_macs": (fwd + 2 * back) * m * m * K, "self_macs": 0.0,
                "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None,
                "trained_weights": None}

    alone = figure(step(16, 1), c, C)
    assert (alone["subset"], alone["F"], alone["beta_w"], alone["beta_x"]) == ("M", 1.0, 16.0, 15.9)
    assert alone["macs_share"] == pytest.approx(16 / 18, rel=1e-15)
    shared = figure(step(1, 16), c, C)
    assert shared["subset"] == "M" and shared["F"] == calculator._min_cut().share(3) and shared["macs_share"] == 1


def test_a_traces_products_at_each_k_of_the_dial_give_the_term_there():
    """A trace keeps its product rows at each k of the console's dial (training_slowdown.capture.products' `at`), so
    the products term is measured at each of them on that k's M, linear in c there, and off the dial it is unmeasured
    with the reason, as at every k but their own for products that keep no rows on the dial. On the 45M dense step
    the term falls with k (220,665 at the console's 65,536 bits, 160,416 at 80,000, 60,188 at the served block's
    145,568 and 2,815 at 800,000 on M, 39 at 8e6 on M without its short contractions), and from 8e7 bits up β(k) is so
    low that the count allows fewer blocks than |C|/c, so the term is below 1, which the slowdown's floor reads as 1."""
    from training_slowdown.capture import products

    rows = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    got = work.products_of_trace(rows)
    C = rows["gates"]
    assert sorted(got["at"], key=int) == [str(k) for k in products.DIAL_K if k != policy.CONSOLE[1]]
    assert all(b["k"] == float(x) and "at" not in b for x, b in got["at"].items())
    c, terms = policy.CONSOLE[0], []
    for k in products.DIAL_K:
        term, why, inputs = term_of(got, c, k, gates=C)
        assert why is None and inputs["k_classified"] == k
        assert term_of(got, 1e12, k, gates=C)[0] == pytest.approx(term * 1e12 / c, rel=1e-12, abs=1e-12)
        terms.append(round(term))
    assert terms == [220_665, 160_416, 60_188, 2_815, 39, 0, 0]
    term, why, _ = term_of(got, c, 100_000, gates=C)
    assert term is None and "keep none at it" in why and "each k of the console's dial" in why
    own = {x: v for x, v in got.items() if x != "at"}
    assert term_of(own, c, policy.CONSOLE[1], gates=C)[0] == term_of(got, c, policy.CONSOLE[1], gates=C)[0]
    assert term_of(own, c, 80_000, gates=C)[:2] == (None, why)


@pytest.mark.parametrize("name", ("dense_blocks", "lora_blocks", "wildcard_adapter"))
def test_a_fixtures_trace_keeps_its_product_rows_and_their_products_term(name):
    """Each fixture's product rows (the `.products.json` beside its circuit, training_slowdown.capture.products of the
    step the circuit traces): the version of the capture that took them, which traced the stored records and is
    read still, the console's k, the circuit's |C|, P-match holding,
    and by kind every product the op log places. On the tiny start no product is fresh: every counted candidate has an
    operand read through a repeating view or a broadcast below its width (each view read at its own value's rate), so
    the products term is measured and far below 1, where the bound says nothing: M's operands are read far below
    their widths, and at every β M can take the count allows fewer blocks than |C|/c."""
    from training_slowdown.capture import products
    from training_slowdown.capture.circuit import StepCircuit, measure

    block = json.loads((FIXTURES / f"{name}.products.json").read_text())
    f = measure(StepCircuit.load(FIXTURES / f"{name}.npz"))
    assert (block["capture"], block["k"]) == (products.READS[0], policy.CONSOLE[1])
    assert block["gates"] == f["gates"] and block["p_match"] == []
    got = work.products_of_trace(block)
    ops = json.loads((FIXTURES / f"{name}.ops.json").read_text())
    assert sum(got["kinds_macs"].values()) == sum(op["macs"] for op in ops if op["products"])
    assert got["kinds_macs"]["fresh"] == 0 and any(any(r["gathered"]) for r in block["rows"])
    term, why, he = term_of(got, 1e11, policy.CONSOLE[1])
    assert 0 < term < 1 and why is None and he["below_width"]
    assert he["beta"] < 4 and he["macs_share"] < 1
