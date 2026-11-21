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
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import door_walls as W  # noqa: E402
from verity import lean_process as LP  # noqa: E402

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
     [("cuts", "query-refused: template WallsT2_v1 has no gate"), ("restrict", "E_NO_GATE"),
      ("cuts", "query-refused: template WallsT2_v1 has no gate")]),
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
    assert z.klass == "cuts: Refused: E_QUERY: query-refused: template <T> has no gate", z.klass
    rows = W.table([b, r, z])
    assert sorted((x["step"], x["count"], x["owner"]) for x in rows) == [
        ("cuts", 1, "proofs' door (one_circuit.cuts, Partition.checkTemplates)"), ("lower", 1, "lean's compose (Flock.Compose)"),
        ("lower", 1, "proofs' door predicate (Spot.OneStageZk.admitted, on-bits)")]
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


def test_the_whole_form_lowers_the_blocks_own_program_and_refuses_a_registered_read_as_lean(tmp_path):
    """Under ``whole`` the internal read is no input row: the block's template is its ported Program, one row of in(S)'s
    run (a, w1, w2), where the unit form's template has the reader's x as a row of its own; a registered read is refused
    in `Spot.Restrict.ported`'s words."""
    W._catalog()
    exe = _stub(tmp_path)
    c, q = W.internal_read()
    from verity.circuits import partition_object as PO
    from verity.circuits import restrict as R
    p = PO.build(json.loads(c), q, [[(0, 2)]])
    w, wq, ported, (ins, outs) = W.whole_circuit(c, p, [(0, 2)], R.program(c))
    fn = R.program(w).defs[wq["params"]["templates"][0]]
    assert ported == "yes" and [(n, t.leaves) for n, t in fn.signature()[0]] == [("p0", 16 + 2 * 2048)]
    assert len(ins) == 16 + 2 * 2048 and len(outs) == 16
    whole, = W.census(c, q, "0-2", lambda: exe, rows=False, form="whole")
    unit, = W.census(c, q, "0-2", lambda: exe, rows=False)
    assert (whole.step, unit.step) == ("lower", "lower") and "unit/Block_v1" in whole.text
    assert _clause(unit.text) == "operands-outside", unit.text
    assert (whole.row_bits, unit.row_bits) == (16 + 2 * 2048, 2048) and whole.tmpl_gates == 2 * unit.tmpl_gates
    from verity.circuits import boolean as BL
    from verity.circuits.types import Value
    ps = (("a", Value(1)), ("b", Value(1)))
    rc, rq = W._rooted(W._composite("WallsReg", ps, Value(1), lambda B, S, a, b: B.call(BL.NOT, b)), ps, Value(1),
                       registered=["b"])
    rp = PO.build(json.loads(rc), rq, [[(0, 1)]])
    assert W.whole_circuit(rc, rp, [(0, 1)], R.program(rc))[2] == (
        "ported: the block reads a registered value (RegisteredInput1_v1), and a parameter does not carry that it is "
        "registered: its rows would be committed afresh")
    assert W.Result(0, (0, 1), (), ported="ported: x").klass == "ported: ported: x"


PANIC_LINE = "PANIC at Spot.restrict Spot.Restrict:1:2: index out of bounds"
#: a stand-in's lines before it writes its answer, its exit, and the refusal's words
LIES = {"panic": (f"print({PANIC_LINE!r}, file=sys.stderr)\n", 0, f"{LP.LEAN_PANIC}: {PANIC_LINE}"),
        "undocumented status": ("", 3, f"{LP.LEAN_EXIT}: status 3")}


def _lying(tmp_path: Path, lie: str, answer: int) -> Path:
    """A stand-in that checks the panic switch, writes the lie's lines, writes its answer to its argument ``answer`` (an
    admission, a template) and exits as the lie says."""
    before, rc, _why = LIES[lie]
    exe = tmp_path / "fake-lean"
    exe.write_text(f"""#!{sys.executable}
import os, sys
assert os.environ.get({LP.ABORT_ON_PANIC!r}) == "1"
{before}open(sys.argv[{answer}], "w").write("written")
sys.exit({rc})
""")
    exe.chmod(0o755)
    return exe


@pytest.mark.parametrize("lie", LIES)
def test_lean_doors_admission_beside_a_panic_or_an_undocumented_status_is_no_answer_and_a_difference(tmp_path, lie):
    """`restrict` writes C_S and ends 0 under a panic line, or at a status it doesn't document (`RESTRICT_EXITS`): Lean's
    door gave no answer, refused by name, and the block is a difference whatever Python's door says."""
    exe = _lying(tmp_path, lie, -1)
    door = SimpleNamespace(_dir=None, c=b"{}", lean=lambda: exe, partition={})
    got = W.Door.lean_door(door, 0)
    assert got == f"{W.CRASHED}{LIES[lie][2]}"
    r = W.Result(0, (0, 1), (), lean_door=got)
    assert r.difference == f"lean's door gave no answer ({LIES[lie][2]})" and r.klass.startswith("lean: crashed: ")
    door._dir.cleanup()


@pytest.mark.parametrize("lie", LIES)
def test_a_template_lean_lowered_beside_a_panic_or_an_undocumented_status_is_refused_by_name(tmp_path, lie):
    """`flock-lower-check`'s template mode writes its template under a panic line, or ends at a status it doesn't document
    (`LOWER_EXITS`): the template is not read, and the row check records the refusal, not `unlowered`."""
    exe = _lying(tmp_path, lie, 6)
    with pytest.raises(LP.LeanPanic) as e:
        W.template_of(b"{}", "T", 1, 1, lambda: exe)
    assert str(e.value) == LIES[lie][2] and not isinstance(e.value, W.Unchecked)


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
    """The door predicate's refusal in ``text``, from its prefix (or not derived's, before its clauses) to the line's
    end; None where it holds."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    i = min((j for j in (text.find(OC.NOT_ADMITTED), text.find(OC.PREFIX)) if j >= 0), default=-1)
    return None if i < 0 else text[i:].splitlines()[0]


def _clause(text: str) -> str | None:
    """The name of the door predicate's clause refusing in ``text``; None where it holds."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    p = _predicate(text)
    if p is None or p.startswith(OC.PREFIX):
        return None if p is None else "not-derived"
    return p.split(", ", 1)[1].split(":")[0]


def _admitted_lean() -> str:
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    return (Path(OC.__file__).resolve().parents[1] / "lean" / "Spot" / "Admitted.lean").read_text()


def _lean_descriptor(name: str) -> bytes:
    """The descriptor `Spot/Admitted.lean` defines as ``name``, which its guards read."""
    import re
    m = re.search(r"^def " + name + r' : String := r#"(.*)"#$', _admitted_lean(), re.M)
    assert m, name
    return m.group(1).encode()


def _sites() -> list[tuple[str, str]]:
    """Every refusal of `admitted` and `admittedBlock`: its clause and its text, `{...}` where it interpolates."""
    import re
    text = _admitted_lean()
    body = text[text.index("def admitted (partition"):text.index("/-- The template `lowerBlock`")]
    return re.findall(r'notAdmitted "([a-z-]+)" s?!?"((?:[^"\\]|\\.)*)"', body)


#: each refusal of `admitted` and `admittedBlock` no C reaches through the door, by its clause and a part of its text,
#: with the check that refuses first or why it can't fail
UNREACHED = {
    ("one-template", "C_S's root is not one of its Definitions"): "restrict's C_S has its root among its Definitions",
    ("listed", "is not one the partition's query lists"): "the query (`Partition.evaluate`) refuses a unit it lists not",
    ("listed", "has no entry in C_S's index"): "C_S indexes each of its Definitions",
    ("listed", "is no Definition of C_S"): "C_S indexes each of its Definitions",
    ("cut-rows", "which is no parameter"): "`ioCuts` cuts only the template's parameters",
    ("leaves-are-bits", "has no lift to bits"): "`on-bits` lifts the template first",
    ("leaves-are-bits", "{why}"): "the lift keeps each parameter's and field's bits as its leaves",
    ("no-recut", "C's root is not one of its Definitions"): "the decoder's root is one of its Definitions",
    ("no-recut", "recut, in leaves of part of their width"): "the gate layout's `recut-gates` refuses first (`RC16`)",
    ("disjoint-ranges", "is named twice in the block"): "a block's units are a range",
    ("disjoint-ranges", "hold gates in common"): "the decoder lays the root's nodes at disjoint offsets",
    ("one-callee", "is no node of C's root"): "a block's units are nodes of C's root",
    ("one-callee", "callee is no Definition of C"): "the decoder resolves every callee to a Definition",
    ("one-callee", "not the block's template"): "`one-template`: C_S's root calls each unit's callee",
}


def _site(clause: str, text: str) -> tuple[str, str] | None:
    """The refusal of `_sites` that ``clause`` and ``text`` are."""
    import re
    for c, t in _sites():
        parts = re.split(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t.replace('\\"', '"'))
        if c == clause and re.fullmatch(".*".join(map(re.escape, parts)), text):
            return c, t
    return None


def _outside(text: str) -> str:
    """A refusal outside the door predicate by the case it is: `recut-gates`, else its text."""
    return "recut-gates" if "recut-gates: " in text else text


def _q(t: str) -> dict:
    from verity.circuits import partition_object as PO
    return PO.template_instance_query(t)


#: C's that `Spot/Admitted.lean`'s guards read, each with its query, blocks and the clause both doors refuse it by:
#: (what, descriptor, (old, new) replaced in it, the query's templates, the blocks, the clause or "holds")
CONSTRUCTED = [
    ("an input row of 3 bits", "E3", None, ("E_v1",), [(0, 1)], "whole-words"),
    ("a return field of 1 bit", "F16", None, ("E_v1",), [(0, 1)], "whole-words"),
    ("a parameter named as a field", "E3", ('[["x",', '[["o0",'), ("E_v1",), [(0, 1)], "port-names"),
    ("a parameter's name holding ':'", "E3", ('[["x",', '[["x:1",'), ("E_v1",), [(0, 1)], "port-names"),
    ("a word the lift keeps", "C32", None, ("C_v1",), [(0, 1)], "on-bits"),
    ("a lift that refuses", "COL", None, ("T_v0",), [(0, 1)], "on-bits"),
    ("a block of two templates", "AB16", None, ("A_v0", "B_v0"), [(0, 2)], "one-template"),
    ("a unit reading another of its block", "AA16", None, ("A_v0",), [(0, 2)], "operands-outside"),
    ("two words in and out", "E16", None, ("E_v1",), [(0, 1)], "holds"),
    ("a template returning no field", "NF16", None, ("E_v1",), [(0, 1)], "whole-words"),
    ("a parameter cut into rows of two widths", "MIX", None, ("A_v1", "X_v1"), [(0, 1), (1, 2)],
     ("whole-words", "cut-rows")),
    ("a root reading an input recut", "RC16", None, ("E_v1",), [(0, 1)], "recut-gates"),
]


def _lean_restrict(lean: Path, tmp_path: Path, c: bytes, partition: dict, lo: int) -> subprocess.CompletedProcess:
    (tmp_path / "C.json").write_bytes(c)
    (tmp_path / "partition.json").write_text(json.dumps(partition, sort_keys=True, separators=(",", ":")))
    return subprocess.run([str(lean), "restrict", str(tmp_path / "C.json"), str(tmp_path / "partition.json"),
                           str(lo), str(tmp_path / "c_s.txt")], capture_output=True, text=True)


#: a drawn C one of whose templates has no gate, outside the last block's own: (the spec, its case seed)
GATELESS = ({"templates": [{"params": [-16, 32, -2048], "steps": [[6, [1, 1]], [4, [3, 1, 3]]], "ret": [4], "nested": False},
                           {"params": [8, 7, -5], "steps": [], "ret": [0], "nested": False}],
             "root": [{"t": 1, "form": "batch", "n": 2, "args": [[["p", 0], ["p", 1]], [["p", 2], ["p", 3]], [["p", 4], ["p", 5]]]},
                      {"t": 1, "form": "batch", "n": 3,
                       "args": [[["n", 0, 1], ["p", 6], ["p", 7]], [["p", 8], ["p", 9], ["p", 10]], [["p", 11], ["p", 12], ["p", 13]]]},
                      {"t": 0, "form": "batch", "n": 2, "args": [[["p", 14], ["p", 15]], [["p", 16], ["p", 17]], [["p", 18], ["p", 19]]]}],
             "root_params": [8, 8, 7, 7, -5, -5, 8, 8, 7, 7, 7, -5, -5, -5, -16, -16, 32, 32, -2048, -2048],
             "registered": [8, 13], "blocks": [[0, 2], [2, 6], [6, 7]]}, 2615796738)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_template_of_no_gate_anywhere_in_c_is_refused_by_both_doors_alike(tmp_path):
    """On every block of `GATELESS`, Lean's door (`spot-one-stage-zk restrict`, `Partition.checkTemplates`) and Python's
    steps (`one_circuit.cuts`) refuse the query with one text, the last block's own template having gates."""
    from verity.protocols.proof.spot.verifier.one_stage import lean_child as LC
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    W._catalog()
    s = W.Spec(**GATELESS[0])
    c, q, _fns = W.realise(s)
    door = W.Door(c, q, [tuple(b) for b in s.blocks], lambda: _stub(tmp_path))
    (tmp_path / "C.json").write_bytes(c)
    (tmp_path / "partition.json").write_text(json.dumps(door.partition, sort_keys=True, separators=(",", ":")))
    why = "query-refused: template WallsT1_v1 has no gate"
    for k, (lo, hi) in enumerate(s.blocks):
        py = door.run(k, lo, hi).text
        run = LC.run([str(PF.v2_executable()), "restrict", str(tmp_path / "C.json"), str(tmp_path / "partition.json"),
                      str(lo), str(tmp_path / "c_s.txt")], ok_statuses={0, 1, 2})
        assert run.returncode == 1 and run.stderr.strip() == f"restrict: {why}", (k, run.stderr[-600:])
        assert py is not None and py.endswith(why), (k, py)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_lean_and_python_give_one_verdict_on_each_drawn_blocks_form(tmp_path):
    """On drawn circuits and partitions (batches, stepped arguments, a member read twice), and on the C's of
    `Spot/Admitted.lean`'s guards (`CONSTRUCTED`), Lean's door (`spot-one-stage-zk restrict`: `admitted` on C_S, then
    `admittedBlock` from `BlockFront.unitAt` and `operandsW`) and Python's (`admitted` and `admitted_block` from
    `TemplatePartition`'s gates and inputs) refuse the same blocks with the same text, and between them every clause
    a decoded C can reach refuses: each refusal of `admitted` and `admittedBlock` is reached or is `UNREACHED`'s. A C
    refused outside the predicate (`RC16`) is refused by both doors alike, `lower` reading Lean's
    `flock-lower-check`."""
    from verity.circuits import partition_object as PO
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    W._catalog()
    stub, lean, lower = _stub(tmp_path), PF.v2_executable(), W._exe(None)()
    seen: dict[str, int] = {}
    reached: set[tuple[str, str]] = set()
    cases = []
    for seed in range(40):
        s = _drawn(random.Random(seed))
        c, q, _fns = W.realise(s)
        cases.append((seed, c, q, [tuple(b) for b in s.blocks], s.doc(), None))
    for what, name, swap, ts, blocks, clause in CONSTRUCTED:
        c = _lean_descriptor(name)
        c = c.replace(*(x.encode() for x in swap)) if swap else c
        q = _q(ts[0]) if len(ts) == 1 else PO.template_instances_query(list(ts))
        cases.append((what, c, q, blocks, None, clause))
    for case, c, q, blocks, doc, want in cases:
        exe = lower if want == "recut-gates" else stub
        door = W.Door(c, q, blocks, lambda: exe)
        for k, (lo, hi) in enumerate(blocks):
            r = door.run(k, lo, hi)
            py = _predicate(r.text)
            run = _lean_restrict(lean, tmp_path, c, door.partition, lo)
            assert _predicate(run.stderr) == py, (case, k, doc, run.stderr[-600:], py)
            if py is None and exe == lower:
                assert run.returncode == 1, (case, k, run.stdout[-600:])
                assert W.clause(f"refused: {run.stderr}") == W.clause(f"refused: {r.text}"), (case, run.stderr, r.text)
                got = _outside(r.text)
            else:
                got = _clause(py or "") or "holds"
            if py is not None and not py.startswith(OC.PREFIX):
                clause, text = py.split(", ", 1)[1].split(": ", 1)
                reached.add(_site(clause, text) or (clause, text))
            assert want in (None, got) or (isinstance(want, tuple) and want[k] == got), (case, k, py)
            seen[got] = seen.get(got, 0) + 1
    assert set(seen) == {"holds", "operands-outside", "whole-words", "port-names", "on-bits", "one-template",
                         "cut-rows", "recut-gates"}, seen
    unreached = {s for s in _sites() if not any(c == s[0] and t in s[1] for c, t in UNREACHED)}
    assert reached == unreached, (sorted(reached - unreached), sorted(unreached - reached))
    assert all(sum(c == s[0] and t in s[1] for s in _sites()) == 1 for c, t in UNREACHED), UNREACHED


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
@pytest.mark.parametrize("query", [_q("F_v1"), {"name": "Q_all"}], ids=["listing another template", "no template query"])
def test_a_partition_whose_query_does_not_name_cs_template_is_refused_by_both_doors(tmp_path, query):
    """The partition object of E3's unit 0 with its query swapped: Python's door refuses it at its block, before C_S, and
    Lean's refuses it too; `listed` and not derived are held clause by clause in `test_door_form`."""
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    W._catalog()
    c = _lean_descriptor("E3")
    door = W.Door(c, _q("E_v1"), [(0, 1)], lambda: _stub(tmp_path))
    door.partition = door.partition | {"query": query}
    r = door.run(0, 0, 1)
    run = _lean_restrict(PF.v2_executable(), tmp_path, c, door.partition, 0)
    assert r.step == "block", (r.step, r.text)
    assert run.returncode == 1 and run.stderr.startswith("restrict: "), run.stderr[-600:]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_an_internal_read_is_refused_under_the_unit_form_and_a_column_under_the_whole():
    """On Lean's lowering: the door refuses the unit form's block by `admittedBlock`'s clause; the whole form's rows
    hold C's values and refuse the block output a reader given a value its writer did not produce gives."""
    got = W.internal_case(W._exe(None))
    unit, whole = got["unit"], got["whole"]
    assert unit["step"] == "lower" and _clause(unit["text"]) == "operands-outside", unit
    assert whole["step"] is None and whole["ported"] == "yes", whole
    assert whole["honest_ok"] and not whole["wrong_accepted"] and not whole["forged_accepted"], whole
    assert whole["in_ports"] == ["p0"] and whole["in_bits"] == [16 + 2 * 2048], whole


def _lean_stub(tmp_path: Path, name: str, refusal: str | None) -> Path:
    """A `spot-one-stage-zk` whose ``restrict`` writes its block circuit (``refusal`` None) or refuses with ``refusal``."""
    exe = tmp_path / name
    body = "echo circuit > \"$5\"\nexit 0\n" if refusal is None else f"echo '{refusal}' >&2\nexit 1\n"
    exe.write_text(f"#!/bin/sh\n{body}")
    exe.chmod(0o755)
    return exe


def test_restrict_of_spot_one_stage_zk_decides_and_a_difference_is_reported_as_one(tmp_path):
    """Lean's door, asked as the accepting path asks it, labels every block and its class; Python's verdict stays
    beside it, and where the two disagree the block carries the difference, both texts kept."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    W._catalog()
    exe = _stub(tmp_path)
    c, q = W.internal_read()
    why = f"{OC.NOT_ADMITTED}, operands-outside: unit 1 reads gate 4112, which unit 0 of the block computes"
    refuses, admits = _lean_stub(tmp_path, "refuses", why), _lean_stub(tmp_path, "admits", None)
    same, = W.census(c, q, "0-2", lambda: exe, rows=False, lean=lambda: refuses)
    assert same.decider == W.LEAN_DOOR == "spot-one-stage-zk restrict" and same.lean_door == f"refused: {why}"
    assert same.klass.startswith("lean: ") and W.clause(same.python_door) == "operands-outside" and not same.difference
    other, = W.census(c, q, "0-2", lambda: exe, rows=False, lean=lambda: admits)
    assert other.lean_door == W.ADMITTED and other.klass == "" and other.python_door.startswith("refused: ")
    assert other.difference == "lean admits, python refuses at lower (operands-outside)"
    row, = W.table([other])
    assert (row["class"], row["decider"], row["differences"]) == ("through", W.LEAN_DOOR, 1)
    big = "unit/Block_v1: an instance needs a block of 2^31 bits (at most 2^27)"
    assert W.clause(f"refused: restrict: {big}") == W.clause(
        f"refused: Refused: E_LOWER: flock-lower-check compose (1): flock-lower-check compose: {big}")
    plain, = W.census(c, q, "0-2", lambda: exe, rows=False)
    assert plain.decider == "python's steps" and not plain.lean_door and not plain.difference


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain: spot-one-stage-zk cannot be built")
def test_spot_one_stage_zk_restrict_refuses_the_internal_read_where_python_is_stubbed_off(monkeypatch):
    """On the built executable: Lean's door refuses the internal read by `operands-outside` as Python's predicate does;
    with Python's predicate stubbed off, the census reports the block as a difference (Python admits, Lean refuses)."""
    from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC
    from verity.protocols.proof.spot.verifier.one_stage import proofs as PF
    W._catalog()
    c, q = W.internal_read()
    agreed, = W.census(c, q, "0-2", W._exe(None), rows=False, lean=PF.v2_executable)
    assert W.clause(agreed.lean_door) == W.clause(agreed.python_door) == "operands-outside", agreed
    assert not agreed.difference
    monkeypatch.setattr(OC, "formed_in", lambda *_a, **_k: (lambda _key: None))
    got, = W.census(c, q, "0-2", W._exe(None), rows=False, lean=PF.v2_executable)
    assert got.python_door == W.ADMITTED and got.difference == "python admits, lean refuses (operands-outside)", got
