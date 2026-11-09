"""Resolve #8309 (first-row own ports, MAX_LEAVES) against #8413 (rows) in a merge's worktree; run from its root."""
import re
import sys
from pathlib import Path

LEAN = Path("verity/core/service/spot/Spot/SecondStage.lean")
EXAMPLES = Path("verity/core/service/spot/Spot/SecondStageExamples.lean")
PY = Path("verity/core/protocols/spot/second_stage.py")
DIFFTEST = Path("verity/core/service/spot/tests/test_difftest.py")

CONFLICT = re.compile(r"<<<<<<< [^\n]*\n.*?>>>>>>> [^\n]*\n", re.S)

LEAN_MODULE_DOC = """\
    each slot reading only its unit's row of its cell's replay block's (`S[i]·r + ρ(j)`, `r` the port's rows a block
    and `ρ` its units' rows, the port's `Rows`, `j` the slot's unit in its cell, the port's leaves the boundary
    population's `r` each); for a port a wire names, its writer's output port of the interior record, each slot of
    replay block `S[i]` reading only its writer unit's row there (`i·n_w + ρ(j)`, the port's leaves the writer's
    `Q·n_w` slots'); else its own port of the interior record, each slot `u` reading only its own first row
    (`(u - off_t)·r`, `r` the port's leaves over the part's `Q·n_t` slots), or a value either registration registers;
    each at its root, domain and leaves, at most `2^64` of them (`MAX_LEAVES`). Or a port a link or a wire names is
"""

LEAN_READOK_DOC = """\
boundary port it names, of its rows' `per` a replay block, each slot only its unit's row of its cell's replay block's
(`linkedRow`); a port a wire names, only its writer's output port of the interior record, of the writer's units a
block, each slot only its unit's writer unit's row in its cell's replay block (`wiredRow`); any other, its own port of
the interior record, each slot only its own first row, or a value either registration registers, never a boundary
port. Each at its root, domain and leaves, at most `MAX_LEAVES` of them. -/
"""

PY_READ_OK_DOC = """\
    link names, only the boundary port it names, of its rows' `per` a replay block, each slot only its unit's row of its
    cell's replay block's (`_linked_row`); a port a wire names, only its writer's output port of the interior record, of
    the writer's units a block, each slot only its unit's writer unit's row in its cell's replay block (`_wired_row`);
    any other, its own port of the interior record, each slot only its own first row, or a value either registration
    registers, never a boundary port. Each at its root, domain and leaves, at most `MAX_LEAVES` of them.\"\"\"
"""

DIFFTEST_DOC = """\
    parts' order (else `second-order`), with a part's `x` read from the boundary where the design links it, each slot
    at its unit's row of its cell's replay block, or read from its own `x` at its own rows, and fail only on the form
    while it is refused (`two-stage-form`); every part needs a statement, the second stage drawing in it or not
    (`second-cover`); a read of the boundary at another cell's replay block, of a tree of more than `2^64` leaves, of
    its own `x` at another slot's rows, of a boundary port the design doesn't link, or of a linked port anywhere but
    the boundary port its link names, or a linked port not read (`second-read`) is refused, as are reads the verifier
    of record doesn't report (`second-reported`); links none derived, one for a template of no part, or one whose rows
    don't place its part's units, are refused (`second-links`);
    a header that isn't its template's reference header placed in the run (`second_stage.at_run`: its set
    `served/<run>/m<k>` at its template's position, its ports' bindings the window rule's) but for the run's fields
"""

EXAMPLES_8309 = """\
/-- A read of `leaves` leaves, slot `k` reading row `reads[k]`. -/
private def exRows (leaves : Nat) (reads : List Nat) : Json :=
  Json.mkObj [("leaves", Lean.toJson leaves), ("reads_from", (0 : Nat)), ("reads", natsJson reads.toArray)]

-- a slot reads its unit's first row, so in a tree of two rows a unit it can't pick its unit's other
#guard rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 6 [0, 1, 2, 3, 4, 5]) 6 some
#guard rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [0, 2, 4, 6, 8, 10]) 6 some
#guard !rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [1, 3, 5, 7, 9, 11]) 6 some
#guard !rowsOk 2 ⟨"Pair_v0", 0, 3⟩ (exRows 12 [0, 2, 4, 6, 8, 11]) 6 some

/-- `exOwn` with the boundary registering value `w`, a tree of `leaves` leaves. -/
private def exRegisteringW (leaves : Nat) : Own :=
  { exOwn with boundary := (exRec 4 [("bx", 5, 6)]).setObjVal! "registered" (Json.arr #[Json.mkObj [("value", "w"),
    ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson leaves)]]) }

/-- A read of registered value `w` at its root and domain, its tree of `leaves` leaves. -/
private def exReadW (leaves : Nat) : Json :=
  Json.mkObj [("value", "w"), ("root", exRoot 7), ("domain", exRoot 8), ("leaves", Lean.toJson leaves)]

-- a read's positions are 64-bit, so its tree has at most `2^64` leaves
#guard readOk (exRegisteringW (2 ^ 64)) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW (2 ^ 64))
#guard !readOk (exRegisteringW (2 ^ 64 + 1)) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW (2 ^ 64 + 1))
#guard !readOk (exRegisteringW 6) false ⟨"Pair_v0", 0, 3⟩ "y" (exReadW 7)

"""
EXAMPLES_ANCHOR = "/-- `exOwn`, the interior registration registering values `w1` and `w2`, six leaves each"


def resolve(path: Path, resolutions: list[str]) -> None:
    text = path.read_text()
    hunks = CONFLICT.findall(text)
    if not hunks:
        print(f"{path}: no conflicts left (rerere's, or a clean merge): kept")
        return
    if len(hunks) != len(resolutions):
        sys.exit(f"{path}: {len(hunks)} conflicts, expected {len(resolutions)}")
    it = iter(resolutions)
    path.write_text(CONFLICT.sub(lambda _: next(it), text))


OLD_WIDE = '''    def wide(c: dict, leaves: int, row) -> dict:
        """`c` with the boundary's `x` a tree of `leaves` leaves, `r` a replay block, TRIPLE's slots reading row
        `row(b, r)` of their cells' replay blocks `b`."""
        b, s = c["boundary"], c["statements"][1]
        e = s["registered"]["ports"]["x"]
        read = {**e, "leaves": leaves, "reads": [row(x, leaves // b["population"]) for x in e["reads"]]}
        return {**_second_with(c, 1, registered={**s["registered"], "ports": {"x": read}}),
                "boundary": {**b, "roots": [{**r, "leaves": leaves} if r["port"] == BX else r for r in b["roots"]]}}
'''
NEW_WIDE = '''    def wide(c: dict, per: int) -> dict:
        """`c` with the boundary's `x` of `per` rows a replay block, TRIPLE's rows naming them, its slots reading their
        units' rows there."""
        b, s, (r,) = c["boundary"], c["statements"][1], c["rows"]
        e = s["registered"]["ports"]["x"]
        read = {**e, "leaves": b["population"] * per,
                "reads": [x // r["per"] * per + x % r["per"] for x in e["reads"]]}
        return {**_second_with(c, 1, registered={**s["registered"], "ports": {"x": read}}), "rows": [{**r, "per": per}],
                "boundary": {**b, "roots": [{**x, "leaves": read["leaves"]} if x["port"] == BX else x
                                            for x in b["roots"]]}}
'''
OLD_CASES = '''        (wide(linked, 8, lambda b, r: b * r), []),
        (wide(linked, 8, lambda b, r: b * r + 1), ["second-read"]),
        (wide(linked, 2 ** 64, lambda b, r: b * r), []),
        (wide(linked, 2 ** 64 + 4, lambda b, r: b * r), ["second-read"]),
'''
NEW_CASES = '''        (wide(linked, 2 ** 64 // linked["boundary"]["population"]), []),
        (wide(linked, 2 ** 64 // linked["boundary"]["population"] + 1), ["second-read"]),
'''

resolve(LEAN, [LEAN_MODULE_DOC, LEAN_READOK_DOC, ""])
resolve(PY, [PY_READ_OK_DOC])
resolve(DIFFTEST, [DIFFTEST_DOC])
dt = DIFFTEST.read_text()
for old, new in ((OLD_WIDE, NEW_WIDE), (OLD_CASES, NEW_CASES)):
    if old in dt:
        dt = dt.replace(old, new)
DIFFTEST.write_text(dt)
ex = EXAMPLES.read_text()
if "private def exRows (leaves" not in ex:
    if ex.count(EXAMPLES_ANCHOR) != 1:
        sys.exit(f"{EXAMPLES}: anchor not found once")
    EXAMPLES.write_text(ex.replace(EXAMPLES_ANCHOR, EXAMPLES_8309 + EXAMPLES_ANCHOR))
print("resolved")
