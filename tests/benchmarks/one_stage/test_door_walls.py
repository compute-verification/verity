"""`door_walls.py`: the census reports the door's refusals by class and owner (Lean's compose refusal of lean's case on a
stub of `flock-lower-check` that answers as Lean does, and the door's own refusal of a template of no gate, which no
lowering reaches), and exits 1 when a block is refused; the fuzzer's draws are a function of the seed; and, on Lean's own
lowering, an honest block passes the row check, the same block flags a planted wrong evaluation of its rows
(`tmpl.evaluate` monkeypatched), and the fuzzer's findings are a function of its seed; each finding the fuzzer kept and
the door now refuses (`SEEDS`) stays refused by its step and clause, by name; and each still open (`OPEN`) stands under a
strict mark that fails once it is mended."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import door_walls as W  # noqa: E402

COMPOSE_WHY = "no row has a compression before its last block"


@pytest.fixture(autouse=True)
def _suites_lookup(monkeypatch):
    """`door_walls._catalog` registers the catalog's Boolean lookup, which knows no test primitive: the suite's own
    (`conftest._stand_in`) is back for the tests that run after one of these in the same process."""
    from verity.circuits import program as PG
    monkeypatch.setattr(PG, "_BODY_LOOKUP", list(PG._BODY_LOOKUP))


#: the fuzzer's findings the door now refuses, minimised: (what it is, the spec, its case seed, the class it showed when
#: found, and each block's refusal as its step and the clause or code it names)
SEEDS = [
    ("lean's second fault past compose: a 3-bit field, then b, then a wide field; Not b",
     {"templates": [{"params": [-3, 1, -2048], "steps": [[2, [1]]], "ret": [3], "nested": False}], "root": [{"t": 0, "form": "call", "n": 1, "args": [[["p", 0]], [["p", 1]], [["p", 2]]]}], "root_params": [-3, 1, -2048], "registered": [], "blocks": [[0, 1]]},
     0, 'rows: honest output refused (door accepts, odd field)', [("lower", "whole-words")]),
    ("lean's second fault on the return: two 1-bit return fields after a wide parameter",
     {"templates": [{"params": [1, 1, -2048], "steps": [[2, [1]], [2, [0]]], "ret": [3, 4], "nested": False}], "root": [{"t": 0, "form": "call", "n": 1, "args": [[["p", 0]], [["p", 1]], [["p", 2]]]}], "root_params": [1, 1, -2048], "registered": [], "blocks": [[0, 1]]},
     0, 'rows: wrong output accepted: flipped value bit (door accepts, odd field)', [("lower", "whole-words")]),
    ("Xor(x, x) on a 16-bit field: its 1-bit output's 15 padding bits are written by no row",
     {"templates": [{"params": [-16, -2048], "steps": [[-2, [0]], [1, [2, 2]]], "ret": [3], "nested": False}], "root": [{"t": 0, "form": "call", "n": 1, "args": [[["p", 0]], [["p", 1]]]}], "root_params": [-16, -2048], "registered": [], "blocks": [[0, 1]]},
     0, 'rows: wrong output accepted: padding bit set (door accepts, no odd field)', [("lower", "whole-words")]),
    ('5-bit fields after a bit: the honest output refused',
     {"templates": [{"params": [1, -5, -5], "steps": [[2, [0]]], "ret": [1], "nested": False}], "root": [{"t": 0, "form": "batch", "n": 2, "args": [[["p", 0], ["p", 1]], [["p", 2], ["p", 3]], [["p", 4], ["p", 5]]]}], "root_params": [1, 1, -5, -5, -5, -5], "registered": [], "blocks": [[0, 2]]},
     1383842589, 'rows: honest output refused (door refuses at lower, odd field)', [("lower", "whole-words")]),
    ('a 3-bit field before a bit: a flipped output accepted',
     {"templates": [{"params": [-3, 1], "steps": [[-3, [0]], [0, [2, 2]]], "ret": [1, 0], "nested": False}], "root": [{"t": 0, "form": "batch", "n": 3, "args": [[["p", 0], ["p", 1], ["p", 2]], [["p", 3], ["p", 4], ["p", 5]]]}], "root_params": [-3, -3, -3, 1, 1, 1], "registered": [], "blocks": [[0, 3]]},
     1790352721, 'rows: wrong output accepted: flipped value bit (door refuses at lower, odd field)', [("lower", "whole-words")]),
    ("restrict divides by a template of no gate's gates",
     {"templates": [{"params": [32, 8], "steps": [[4, [0, 0, 0]], [3, [0, 2]], [3, [2, 0]]], "ret": [0, 3], "nested": False}, {"params": [-3, 8, 7], "steps": [[5, [2, 1]], [5, [2, 1]]], "ret": [2, 4], "nested": False}, {"params": [8, -5, -3, -2048], "steps": [[-1, [1]], [-3, [2]]], "ret": [0], "nested": False}], "root": [{"t": 1, "form": "batch", "n": 3, "args": [[["p", 0], ["p", 1], ["p", 2]], [["p", 3], ["p", 4], ["p", 5]], [["p", 6], ["p", 7], ["p", 8]]]}, {"t": 2, "form": "batch", "n": 3, "args": [[["p", 9], ["n", 0, 5], ["p", 10]], [["p", 11], ["p", 12], ["p", 13]], [["p", 14], ["p", 15], ["p", 16]], [["p", 17], ["p", 18], ["p", 19]]]}, {"t": 1, "form": "batch", "n": 2, "args": [[["p", 20], ["p", 21]], [["n", 0, 1], ["n", 1, 2]], [["p", 22], ["p", 23]]]}, {"t": 2, "form": "batch", "n": 2, "args": [[["n", 0, 5], ["n", 0, 5]], [["p", 24], ["p", 25]], [["p", 26], ["p", 27]], [["p", 28], ["p", 29]]]}], "root_params": [-3, -3, -3, 8, 8, 8, 7, 7, 7, 8, 8, -5, -5, -5, -3, -3, -3, -2048, -2048, -2048, -3, -3, 7, 7, -5, -5, -3, -3, -2048, -2048], "registered": [7, 9, 14, 17, 18, 22, 24, 25], "blocks": [[0, 2], [2, 7], [7, 10]]},
     3772944590, 'restrict: ZeroDivisionError: integer division or modulo by zero',
     [("lower", "whole-words"), ("restrict", "E_NO_GATE"), ("lower", "whole-words")]),
    ("a unit reads another unit of its block: the reader given a value its writer did not produce is accepted",
     {"templates": [{"params": [-16], "steps": [[-1, [0]], [2, [1]], [W.PACK, [2] * 16]], "ret": [3], "nested": False}],
      "root": [{"t": 0, "form": "call", "n": 1, "args": [[["p", 0]]]}, {"t": 0, "form": "call", "n": 1, "args": [[["r", 0, 0, 16]]]}],
      "root_params": [-16], "registered": [], "blocks": [[0, 2]]},
     0, "rows: a unit's read of another unit of its block is not tied: the reader given a value its writer did not produce is accepted (door accepts)",
     [("lower", "operands-outside: unit 1 reads gate 16, which unit 0 of the block computes")]),
]

#: findings still open, under a strict mark: (what it is, the spec, its case seed, the class it shows)
OPEN: list[tuple] = []


def _stub(tmp_path: Path) -> Path:
    """An executable that refuses every `compose` as Lean refuses lean's case."""
    exe = tmp_path / "flock-lower-check"
    exe.write_text(f"#!/bin/sh\necho \"flock-lower-check compose: unit/$3: {COMPOSE_WHY}\" >&2\nexit 1\n")
    exe.chmod(0o755)
    return exe


def _admitted() -> tuple[bytes, dict]:
    """A template the door's predicate admits: a 16-bit field in, its first bit negated sixteen times out."""
    c, q, _fns = W.realise(W.Spec(
        [{"params": [-16], "steps": [[-1, [0]], [2, [1]], [W.PACK, [2] * 16]], "ret": [3], "nested": False}],
        [{"t": 0, "form": "call", "n": 1, "args": [[["p", 0]]]}], [-16], [], [[0, 1]]))
    return c, q


def _zero_gate() -> tuple[bytes, dict]:
    """A template that passes its parameter through: no gate."""
    from verity.circuits.types import Value
    ps = (("a", Value(8)),)
    return W._rooted(W._composite("WallsPass", ps, Value(8), lambda B, S, a: a), ps, Value(8))


def test_the_census_reports_leans_compose_refusal_and_the_doors_by_class(tmp_path):
    W._catalog()
    exe = _stub(tmp_path)
    c, q = W.lean_case()
    (b,) = W.census(c, q, "one-per-template", lambda: exe)
    assert (b.step, b.templates) == ("lower", ("WallsLeanC_v1",))
    assert "(Spot.OneStageZk.admitted), on-bits:" in b.klass, b.klass
    c, q = _admitted()
    (r,) = W.census(c, q, "one-per-template", lambda: exe)
    assert r.klass == f"lower: Refused: E_LOWER: flock-lower-check compose (<n>): flock-lower-check compose: unit/<T>: {COMPOSE_WHY}"
    c, q = _zero_gate()
    (z,) = W.census(c, q, "one-per-template", lambda: exe)
    assert z.step == "lower" and "(Spot.OneStageZk.admitted), one-template:" in z.klass, z.klass
    rows = W.table([b, r, z])
    assert sorted((x["step"], x["count"], x["owner"]) for x in rows) == [
        ("lower", 1, "lean's compose (Flock.Compose)"), ("lower", 1, "proofs' door predicate (Spot.OneStageZk.admitted, on-bits)"),
        ("lower", 1, "proofs' door predicate (Spot.OneStageZk.admitted, one-template)")]
    assert W.main(["census", "--circuit", "lean_case", "--lower", str(exe)]) == 1


def test_floor_doubles_to_the_range_and_plan_takes_the_registered_blocks(tmp_path):
    W._catalog()
    exe = _stub(tmp_path)
    c, q = W.chain()
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    n = next(n for _k, _t, a, n in PO.TemplatePartition(R.program(c), q).ranges if a == 0)
    (r,) = W.census(c, q, "0", lambda: exe, rows=False, units="floor")
    ks = [k for k, _klass, _bits in r.tried]
    assert n > 1 and ks[-1] == n and all(k == 2 ** j for j, k in enumerate(ks[:-1])) and r.units == (0, n)
    assert {klass for _k, klass, _bits in r.tried} == {r.klass} and r.step == "lower"
    assert len({bits for _k, _klass, bits in r.tried}) == 1 and r.gates
    plan, by = W.plan_blocks("chain")
    assert by.startswith("verity.commitments.periods.interface.partition_blocks")
    got = W.census(c, q, "0", lambda: exe, rows=False, units="plan", plan=plan)
    assert [g.units for g in got] == [next(b for b in plan if b[0] == 0)]


def test_one_cause_is_one_class():
    a = "E_LOWER: flock-lower-check compose (1): unit/Gemm_v2{K=16,N=16}: no row at 1234 of 'x_v1'"
    b = "E_LOWER: flock-lower-check compose (2): unit/SiluMul_v2{I=128}: no row at 7 of 'y_v3'"
    assert W.normalise(a) == W.normalise(b)
    assert W.normalise("digest " + "ab" * 32) == "digest <H>"


def test_the_fuzzers_draws_are_the_seeds():
    W._catalog()
    for seed in range(20):
        assert W.draw(random.Random(seed)).doc() == W.draw(random.Random(seed)).doc()
    assert len({repr(W.draw(random.Random(s)).doc()) for s in range(20)}) > 15


def _xor_spec() -> W.Spec:
    """One template, the XOR of two 16-bit fields' first bits packed sixteen times, two units in one block."""
    return W.Spec([{"params": [-16, -16], "steps": [[-1, [0]], [-1, [1]], [1, [2, 3]], [W.PACK, [4] * 16]], "ret": [5], "nested": False}],
                  [{"t": 0, "form": "batch", "n": 2, "args": [[["p", 0], ["p", 1]], [["p", 2], ["p", 3]]]}],
                  [-16, -16, -16, -16], [], [[0, 2]])


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_row_check_holds_an_honest_block_and_flags_a_planted_wrong_one(monkeypatch):
    from verity_flock import tmpl as T
    W._catalog()
    exe = W._exe(None)
    honest = W.check_case(_xor_spec(), exe, 7)
    assert not [f.klass for f in honest if f.step == "rows"], [f.klass for f in honest]
    evaluate = T.evaluate

    def flipped(t, inputs, out_cols=None, **kw):
        y = evaluate(t, inputs, out_cols, **kw)
        y[0] ^= 1
        return y

    monkeypatch.setattr(T, "evaluate", flipped)
    planted = {f.klass for f in W.check_case(_xor_spec(), exe, 7)}
    assert any(k.startswith("rows: honest output refused") for k in planted), planted
    monkeypatch.undo()
    first = W.fuzz(3, 4, 5.0, exe)
    again = W.fuzz(3, 4, 5.0, exe)
    assert [c | {"example": None} for c in first["classes"]] == [c | {"example": None} for c in again["classes"]]


@pytest.mark.parametrize("what,spec,seed,klass,refused", SEEDS, ids=[s[0] for s in SEEDS])
def test_each_mended_seed_is_refused_by_the_door_by_name(tmp_path, what, spec, seed, klass, refused):
    """The door refuses each block of each mended finding at the step and by the clause or code it names, before
    any lowering (the stub of `flock-lower-check` refuses whatever reaches it, under another text)."""
    W._catalog()
    exe = _stub(tmp_path)
    s = W.Spec(**spec)
    c, q, _fns = W.realise(s)
    door = W.Door(c, q, [tuple(b) for b in s.blocks], lambda: exe)
    got = [door.run(k, lo, hi) for k, (lo, hi) in enumerate(s.blocks)]
    assert [(r.step, next((n for _s, n in refused if n in r.text), r.text)) for r in got] == refused


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
@pytest.mark.parametrize("what,spec,seed,klass", [pytest.param(*s, marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=s[3]), id=s[0])
                                                  for s in OPEN])
def test_each_open_seed_reaches_the_door_and_its_rows_hold(what, spec, seed, klass):
    """Each open finding stands while its mark does: the door takes its blocks and their rows hold. A finding that is
    mended fails its strict mark, and its mark goes."""
    W._catalog()
    found = {f.klass for f in W.check_case(W.Spec(**spec), W._exe(None), seed)}
    assert klass not in found, found
    assert not [k for k in found if k.startswith("rows:")], found


def _drawn(rng: random.Random) -> W.Spec:
    """C of two to four call or batch nodes of one template on a 16-bit field, each argument a parameter of C or a
    member of an earlier node (a batch's stepped argument, at times one member read twice), cut into contiguous blocks."""
    t = {"params": [-16], "steps": [[-1, [0]], [2, [1]], [W.PACK, [2] * 16]], "ret": [3], "nested": False}
    root: list[dict] = []
    params: list[int] = []

    def src() -> list:
        if root and rng.random() < 0.6:
            j = rng.randrange(len(root))
            return ["r", j, 16 * rng.randrange(root[j]["n"]), 16]
        params.append(-16)
        return ["p", len(params) - 1]

    for _ in range(rng.randint(2, 4)):
        n = rng.randint(1, 3)
        members = [src() for _ in range(n)]
        if n > 1 and rng.random() < 0.3:
            members[1] = members[0]
        root.append({"t": 0, "form": "call" if n == 1 and rng.random() < 0.5 else "batch", "n": n, "args": [members]})
    units = sum(nd["n"] for nd in root)
    cuts = sorted(rng.sample(range(1, units), rng.randint(0, units - 1)))
    bounds = [0, *cuts, units]
    return W.Spec([t], root, params, [], [list(b) for b in zip(bounds, bounds[1:])])


def _predicate(text: str) -> str | None:
    """The door predicate's refusal in ``text``, from its prefix to the line's end; None where it holds."""
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    i = text.find(OC.NOT_ADMITTED)
    return None if i < 0 else text[i:].splitlines()[0]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_and_python_give_one_verdict_on_each_drawn_blocks_form(tmp_path):
    """On drawn circuits and partitions (batches, stepped arguments, a member read twice), Lean's door
    (`spot-one-stage-zk restrict`: `admittedBlock` from `BlockFront.unitAt` and `operandsW`) and Python's
    (`admitted_block` from `TemplatePartition`'s gates and inputs) refuse the same blocks with the same text."""
    from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
    W._catalog()
    exe, lean = _stub(tmp_path), PF.v2_executable()
    seen: dict[str, int] = {}
    for seed in range(40):
        s = _drawn(random.Random(seed))
        c, q, _fns = W.realise(s)
        door = W.Door(c, q, [tuple(b) for b in s.blocks], lambda: exe)
        (tmp_path / "C.json").write_bytes(c)
        (tmp_path / "partition.json").write_text(json.dumps(door.partition, sort_keys=True, separators=(",", ":")))
        for k, (lo, hi) in enumerate(s.blocks):
            py = _predicate(door.run(k, lo, hi).text)
            run = subprocess.run([str(lean), "restrict", str(tmp_path / "C.json"), str(tmp_path / "partition.json"),
                                  str(lo), str(tmp_path / "c_s.txt")], capture_output=True, text=True)
            assert _predicate(run.stderr) == py, (seed, k, s.doc(), run.stderr[-600:], py)
            clause = "holds" if py is None else py.split(", ", 1)[1].split(":")[0]
            seen[clause] = seen.get(clause, 0) + 1
    assert seen.get("holds") and seen.get("operands-outside"), seen
