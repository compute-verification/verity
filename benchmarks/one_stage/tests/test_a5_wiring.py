"""a5's wiring negatives (`a5.wiring_negatives`) on a window whose layer 1 reads layer 0's rows: at two units a member, at one
(the torch loop's wired circuit, `--cap 1`) and on a fan-out (every reader unit reading writer unit 0), each negative is
stated and the verifier's wiring check refuses it. In the last two, rotating a wire's writers is the honest wire. Where no
other unit writes a row of the reader's words, the unit is rewired to a row of other words, and that is refused too; and
with each wired port held in its writer's tree, a flipped row is the one its reader's reference names."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops import gemm as G
from dataclasses import replace

from verity.commitments.windows import interface as I, wiring as WI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
WIDE = bind(G.Gemm, K=16, N=32, **G.GEMM_AMPERE)
GID, WID = descriptor_id(GEMM), descriptor_id(WIDE)
UNITS = 4
RNG = np.random.default_rng(0)


def bf16(*shape: int) -> np.ndarray:
    return (np.asarray(RNG.standard_normal(shape) / 4, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def gemm(fn, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, x, w, kernel="reference"), dtype=np.uint16).reshape(len(x), -1)


def closed(circuit: I.Circuit, n: int, records) -> tuple[I.Window, dict[str, bytes]]:
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, n)
    for fn, x, w in records:
        win.record(fn, [x, w], [gemm(fn, x, w)])
    c = win.close(salt_key=bytes(32))
    assert win.mismatches() == {} and max(m.n for m in win.window.members) == n
    return win, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)}


def negatives(circuit: I.Circuit, n: int, records) -> tuple[I.Window, dict[str, dict]]:
    win, publics = closed(circuit, n, records)
    return win, a5.wiring_negatives(win.window.members, publics)


def held(members, publics: dict[str, bytes]) -> tuple[list, dict[str, bytes]]:
    """``members`` with each wired port held in its writer's tree (no wired trees, its table its writer member's output rows,
    `rows.Member.tables`) and their public bodies so (`wiring.strings`): each port's table, then each unit's reference into
    each table, the writer unit's row for a held port and its own for the rest."""
    at, out, bodies = WI.bases(members), [], dict(publics)
    for m in members:
        writer = {p: (o, WI._writers(members, ws, at)) for p, o, ws in m.wired}
        if not writer:
            out.append(m)
            continue
        assert all(len(ws) == 1 for _o, ws in writer.values())
        tables, refs = [], []
        for q, _w in m.all_rows:
            o, ((wm, _sel, theirs),) = writer.get(q, (None, ((m, None, np.arange(m.n)),)))
            tables.append(WI.strings(publics, wm, o or q))
            refs.append(theirs)
        bodies[m.key] = b"".join(t.tobytes() for t in tables) + np.stack(refs, axis=1).astype("<u4").tobytes()
        out.append(replace(m, wired_trees=False, tables=tuple((q, len(tables[j])) for j, (q, _w) in enumerate(m.rows))))
    return out, bodies


@pytest.mark.parametrize("cap, step", [(2, 1), (1, 1), (2, 0)])
def test_each_wiring_negative_is_stated_and_refused_at_one_unit_a_member_and_on_a_fan_out(cap, step):
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=0, step=(step,)),),
                        order=((GID, 0, UNITS), (GID, UNITS, UNITS)))
    w, x0 = bf16(2 * UNITS, 16, 16), bf16(UNITS, 16)
    x1 = gemm(GEMM, x0, w[:UNITS])[np.arange(UNITS) * step]
    _, neg = negatives(circuit, cap, [(GEMM, np.concatenate([x0, x1]), w)])
    assert set(neg) == {"chunk-wired-to-another-unit (§16.13)", "wired-row-not-its-writer-s-output (§16.13)"}
    assert all(v["refused"] for v in neg.values()) and neg["chunk-wired-to-another-unit (§16.13)"]["words"] == "same"


def test_a_unit_no_other_unit_writes_its_words_for_is_rewired_to_other_words_and_refused():
    circuit = I.Circuit("fan", ((GEMM, 1), (WIDE, UNITS)), external=(f"{GID}/x",),
                        wires=(I.Wire(WID, "x", GID, "o0", at=0, radix=(UNITS,), base=0, step=(0,)),))
    x0, w0 = bf16(1, 16), bf16(1, 16, 16)
    win, neg = negatives(circuit, 1, [(GEMM, x0, w0), (WIDE, np.repeat(gemm(GEMM, x0, w0), UNITS, 0), bf16(UNITS, 32, 16))])
    assert [dict(m.out_rows)["o0"] for m in win.window.members] == [16] + [32] * UNITS
    assert neg["chunk-wired-to-another-unit (§16.13)"] == {"refused": True, "words": "other"}


def test_a_held_wired_row_flipped_at_the_row_its_reference_names_is_refused():
    """A reader holding its wired port in its writer's tree has a body of tables and references: a5 flips the row of the
    table its unit 0's reference names, here the writer's last unit's (each reader unit reads the writers in reverse), not
    the table's first row, and every negative is refused where the honest held bodies pass the wiring check."""
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=UNITS - 1, step=(-1,)),),
                        order=((GID, 0, UNITS), (GID, UNITS, UNITS)))
    w, x0 = bf16(2 * UNITS, 16, 16), bf16(UNITS, 16)
    win, publics = closed(circuit, UNITS, [(GEMM, np.concatenate([x0, gemm(GEMM, x0, w[:UNITS])[::-1]]), w)])
    ms, bodies = held(win.window.members, publics)
    (m,) = [m for m in ms if m.wired]
    assert (m.wired_trees, m.tables, m.wired[0][2]) == (False, (("x", UNITS), ("w", UNITS)), tuple(range(UNITS - 1, -1, -1)))
    assert WI.check(ms, bodies) == {}
    neg = a5.wiring_negatives(tuple(ms), bodies)
    assert set(neg) == {"chunk-wired-to-another-unit (§16.13)", "wired-row-not-its-writer-s-output (§16.13)"}
    assert all(v["refused"] for v in neg.values())
