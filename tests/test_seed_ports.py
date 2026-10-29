"""A seed is minted only by PoUW's `seeds.py` (circuits' condition for dropping the seed-confinement rule, CIR-77).

`ml.circuits.ports` gives a Definition the seed role (`ports._coin_role`) when it has a coin port and an
`In(kind="position")` port and no sink in its return, and no port rule follows a seed's output: what it reaches is the
unit's own check, PoUW's compile pass (`noise-in-z`, `stream`). That premise holds only while every seeded unit goes
through that pass, so a position port is declared only where `ALLOWED` names it:

- `SEEDS`: the one place that mints a seed, PoUW's `seeds.py`;
- `UNITS`: a PoUW unit's or the compiled root's position, beside a sink return (an `Out(sink=True)` in the same top-level
  name), so the Definition it types is a unit, never a seed;
- `KEY_COUNTS`: a padded unit's key count, a position port with no coin port beside it (no `kind="coin"` in the same
  top-level name), so the Definition it types is never a seed;
- `WINDOWS`: a shared row's segment index, held as a key count is: no coin port beside it;
- `TESTS`: tests that build seeds, units or their negatives, by name, with a reason each.

An AST scan of every tracked Python file flags each place that could declare a position port: a `kind="position"` keyword
in any call, an `In(...)` whose kind isn't a literal (or is passed positionally or through `**`), a `replace(...)` with a
computed `kind=`, and a
`{"kind": "position"}` dict (a port as `PortTypes.from_json` reads it). A flagged place not named in `ALLOWED` fails, and so
does an entry that matches nothing any more. A descriptor decoded by `codec.decode_program` carries its ports as data;
PoUW never takes a prover's descriptor (`Traced.check_descriptor`, `Compiled.check_descriptor` compare its digest with
the verifier's own build)."""

from __future__ import annotations

import ast

from test_repository import ROOT, _tracked

SEEDS = {
    "verity/core/protocols/pouw/circuit/seeds.py::_coin_salt": "PoUW's seed: (salt, position) to the unit's seed, under compile's checks",
}
UNITS = {
    "verity/core/protocols/pouw/circuit/h1tile.py::_unit_ports": "an H1 tile's position beside its receipt sink: a unit",
    "verity/core/protocols/pouw/circuit/h1zy.py::_unit_ports":
        "an H1 tile with its z/y record checked: its position beside its receipt sink, a unit",
    "verity/core/protocols/pouw/circuit/h1zy.py::_lane_ports":
        "a lane of an H1 leaf: the leaf's position beside its ok sink (its partial's check), a unit",
    "verity/core/protocols/pouw/tests/test_circuit_h1zy.py::test_the_port_rules_follow_the_coin_into_a_lane_its_statement_calls":
        "the lane's stand-in statement template, a position beside its ok sink",
    "verity/core/protocols/pouw/circuit/compile.py::_compiled_ports": "the compiled root's positions beside its receipts sink",
    "verity/core/protocols/pouw/tests/test_circuit_compile.py::UNIT_PORTS": "compile's test units, a position beside a receipt sink",
    "verity/core/protocols/pouw/circuit/h1window.py::_ports": "an H1 window's positions beside its units' receipt sinks",
    "verity/core/protocols/pouw/tests/test_circuit_h1window.py::TILE_PORTS":
        "the window tests' stand-in tile, a position beside its receipt sink",
    "integrations/vllm/tests/protocols/test_pouw_h1_served.py::StandIn":
        "vllm's served-trees tests' stand-in tile, a position beside its receipt sink",
    "integrations/vllm/tests/protocols/test_pouw_h1_drawn.py::StandInJ10R":
        "vllm's drawn-statements tests' stand-in J10-R tile, a position beside its receipt sink",
    "verity/ml/circuits/population.py::_instance_ports":
        "a population of units: each instance's position beside its receipt sink, declared only for a template with a sink",
    "verity/ml/circuits/tests/test_population_ports.py::Unit": "the population tests' stand-in unit, a position beside its sink",
}
KEY_COUNTS = {
    "verity/ml/circuits/ops/gemm.py::KEY_COUNT": "PvRowPad's key count n: attention padded to T = 32",
    "verity/ml/circuits/tests/test_population_ports.py::Counted":
        "the population tests' stand-in key count: a position alone, whose population keeps it",
    "verity/ml/circuits/ops/boolean/softmax_row.py::QkSoftmaxRowPad":
        "QkSoftmaxRowPad's key count n: attention padded to T = 32",
    "verity/ml/circuits/ops/boolean/softmax_row.py::QkSoftmaxPvPad":
        "QkSoftmaxPvPad's key count n, both halves' one port: a head's attention padded to T = 32",
    "verity/ml/definitions/registry/torch_attention.py::_POSITION":
        "EagerSoftmax_v1's key count n: the keys HF's causal mask lets a training row see",
    "verity/ml/definitions/registry/torch_step.py::_POSITION":
        "TrainStep_v1's key count n, the run input its attention calls read as theirs: a position with no coin",
    "examples/h200/e2e_h200/program.py::program":
        "the toy's served step: its key count n and position p, run inputs with no coin",
}
WINDOWS = {
    "verity/ml/circuits/ops/boolean/shared_row.py::WINDOW":
        "SharedRowWindow's j and SharedRowRead's PORT_j, a proof block's segment of a shared row (CIR-113's form (B))",
}
TESTS = {
    "verity/ml/circuits/tests/test_ports.py::SEED": "the port rules' seed vectors (E_COIN_REUSE, E_PORT_KIND, batched seeds)",
    "verity/ml/circuits/tests/test_ports.py::UNIT_PORTS": "the port rules' unit vectors: a seeded unit whose receipt is its sink",
    "verity/ml/circuits/tests/test_ports.py::SALTED": "seeds batched over positions (E_COIN_REUSE's vectors)",
    "verity/ml/circuits/tests/test_ports.py::test_a_seed_takes_the_coin_and_a_position":
        "the descriptor's port entry a batched seed encodes to, as the expected value",
    "verity/ml/circuits/tests/test_ports.py::test_a_position_is_a_run_input_passed_through":
        "E_PORT_KIND's negatives: a seed whose position is computed, not a run input",
    "verity/ml/circuits/tests/test_ports_one_use.py::ROOT_PORTS":
        "the one-use check's random and negative Programs: a root's positions, read by seeds batched over them",
    "verity/ml/circuits/tests/test_ports_one_use.py::Seed": "the seed those Programs batch, to hold E_COIN_REUSE's position rule",
    "verity/ml/circuits/tests/test_population_ports.py::Seed":
        "a seed whose population declares no ports and is refused (E_PORT_KIND): a population never mints a seed",
}
ALLOWED = SEEDS | UNITS | KEY_COUNTS | WINDOWS | TESTS


def _name(f: ast.AST) -> str:
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _is(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _where(top: ast.stmt) -> str:
    return (top.name if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else top.targets[0].id if isinstance(top, ast.Assign) and isinstance(top.targets[0], ast.Name) else "<module>")


def coin_holders(source: str) -> set[str]:
    """The top-level names (or "<module>") of `source` that hold a `kind="coin"` keyword."""
    return {_where(top) for top in ast.parse(source).body
            if any(isinstance(n, ast.keyword) and n.arg == "kind" and _is(n.value, "coin") for n in ast.walk(top))}


def sites(source: str) -> list[tuple[str, int, str, bool]]:
    """(top-level name or "<module>", line, what, whether that top-level name holds an `Out(sink=True)`) for every place in
    `source` that declares, or could declare, a position port."""
    tree = ast.parse(source)
    out = []
    for top in tree.body:
        where = _where(top)
        sink = any(isinstance(n, ast.keyword) and n.arg == "sink" and _is(n.value, True) for n in ast.walk(top))
        for node in ast.walk(top):
            what = None
            if isinstance(node, ast.keyword) and node.arg == "kind" and _is(node.value, "position"):
                what = 'kind="position"'
            elif isinstance(node, ast.Call) and _name(node.func) in ("In", "replace"):
                kind = next((k.value for k in node.keywords if k.arg == "kind"), None)
                if kind is not None and not _literal(kind):
                    what = f"{_name(node.func)}() with a computed kind"
                elif _name(node.func) == "In" and (any(k.arg is None for k in node.keywords) or len(node.args) > 1
                                                   and not (_literal(node.args[1]) and node.args[1].value != "position")):
                    what = f"{_name(node.func)}() with its kind passed positionally or through **"
            elif isinstance(node, ast.Dict) and any(_is(k, "kind") and _is(v, "position") for k, v in zip(node.keys, node.values)):
                what = '{"kind": "position"}'
            if what:
                out.append((where, node.lineno if hasattr(node, "lineno") else top.lineno, what, sink))
    return sorted(out, key=lambda s: s[1])


def test_only_pouw_mints_a_seed() -> None:
    offenders, used = [], set()
    for path in (p for p in _tracked() if p.endswith(".py")):
        source = (ROOT / path).read_text(encoding="utf-8")
        for where, line, what, sink in sites(source):
            key = f"{path}::{where}"
            if key not in ALLOWED:
                offenders.append(f"{path}:{line} ({where}): {what}")
            elif key in UNITS and not sink:
                offenders.append(f"{path}:{line} ({where}): {what}, with no Out(sink=True) beside it: a seed, not a unit")
            elif (key in KEY_COUNTS or key in WINDOWS) and where in coin_holders(source):
                offenders.append(f"{path}:{line} ({where}): {what}, with a kind=\"coin\" port beside it: a seed, not a key count")
            else:
                used.add(key)
    assert not offenders, (
        "a position port outside PoUW's seeds.py: a Definition with a coin port and a position port and no sink is a seed, "
        "whose noise no port rule follows, so only PoUW's compile pass may build one. Mint seeds in "
        "verity/core/protocols/pouw/circuit/seeds.py, give a unit's position port a sink return (UNITS), declare a key count "
        "with no coin port beside it (KEY_COUNTS), or name a test that builds one in TESTS with its reason:\n"
        + "\n".join(offenders))
    assert not set(ALLOWED) - used, f"ALLOWED entries that match nothing any more: {sorted(set(ALLOWED) - used)}"


def test_the_scan_sees_each_form() -> None:
    src = '''
import dataclasses
from verity.ml.circuits.ports import In, Out, Ports

SEED = Ports(params={"salt": In(kind="coin"), "position": In(kind="position")})
UNIT = Ports(params={"p": In(kind="position")}, ret=Out(sink=True))

def f(k, port, kw):
    In(kind=k)
    In(None, "position")
    In(None, k)
    In(**kw)
    dataclasses.replace(port, kind="position")
    dataclasses.replace(port, kind=k)
    return {"inputs": [[0, 0, 1, {"kind": "position"}]]}

def g(port):
    In(kind="coin"), In(None, "static"), In((0, 255)), dataclasses.replace(port, bound=(0, 1))
    return {"kind": "coin"}, "position" in ("coin", "position")
'''
    assert [(w, what, sink) for w, _, what, sink in sites(src)] == [
        ("SEED", 'kind="position"', False), ("UNIT", 'kind="position"', True), ("f", "In() with a computed kind", False),
        ("f", "In() with its kind passed positionally or through **", False),
        ("f", "In() with its kind passed positionally or through **", False),
        ("f", "In() with its kind passed positionally or through **", False), ("f", 'kind="position"', False),
        ("f", "replace() with a computed kind", False), ("f", '{"kind": "position"}', False)]
    assert coin_holders(src) == {"SEED", "g"}
