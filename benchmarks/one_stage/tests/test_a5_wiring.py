"""a5's wiring negatives (`a5.wiring_negatives`) on a window whose layer 1 reads layer 0's rows: at two units a member, at one
(the torch loop's wired circuit, `--cap 1`) and on a fan-out (every reader unit reading writer unit 0), each negative is
stated and the verifier's wiring check refuses it. In the last two, rotating a wire's writers is the honest wire. Where no
other unit writes a row of the reader's words, the unit is rewired to a row of other words, and that is refused too. In a
shared-row member whose wired port is not its first, the flipped string is the one unit 0 reads at that port."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops import gemm as G
from verity.commitments.windows import interface as I
from verity.commitments.windows import rows as CS
from verity.commitments.windows import wiring as WI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
WIDE = bind(G.Gemm, K=16, N=32, **G.GEMM_AMPERE)
BIG = bind(G.Gemm, K=16, N=256, **G.GEMM_AMPERE)
GID, WID, BID = descriptor_id(GEMM), descriptor_id(WIDE), descriptor_id(BIG)
UNITS = 4
RNG = np.random.default_rng(0)


def bf16(*shape: int) -> np.ndarray:
    return (np.asarray(RNG.standard_normal(shape) / 4, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def gemm(fn, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(evaluate_batch(fn, x, w, kernel="reference"), dtype=np.uint16).reshape(len(x), -1)


def negatives(circuit: I.Circuit, cap: int, records) -> tuple[I.Window, dict[str, dict]]:
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, cap)
    for fn, x, w in records:
        win.record(fn, [x, w], [gemm(fn, x, w)])
    c = win.close(salt_key=bytes(32))
    assert win.mismatches() == {} and max(m.n for m in win.window.members) == cap
    return win, a5.wiring_negatives(win.window.members, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)})


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


def test_a_shared_row_readers_flipped_wired_row_is_its_drawn_units_own():
    """A shared-row reader (its body each port's table, then its units' refs) whose wired port is not its first, as the weights
    tie's stream rows (``data`` a registered read, ``cv`` wired): the flipped string is its unit 0's ``cv`` row, so the
    verifier's wiring check on that unit refuses it, and the honest wires hold."""
    n, rng = 3, np.random.default_rng(7)
    writer = CS.Member("W", (("a", 64),), (("o0", 64),), n)
    reader = CS.Member("R", (("data", 64), ("cv", 64)), (("o0", 64),), n, base=n, tables=(("data", n), ("cv", n)),
                       wired=(("cv", "o0", tuple(range(n))),))
    strings = rng.integers(0, 256, (n, 2, WI.BC_BYTES), dtype=np.uint8)
    refs = np.tile(np.arange(n, dtype="<u4")[:, None], (1, 3))
    tables = [rng.integers(0, 256, (n, WI.BC_BYTES), dtype=np.uint8), strings[:, 1], rng.integers(0, 256, (n, WI.BC_BYTES), dtype=np.uint8)]
    publics = {"W": strings.tobytes(), "R": b"".join(t.tobytes() for t in tables) + refs.tobytes()}
    assert WI.check_drawn((writer, reader), publics, range(2 * n)) == {}
    assert WI.string_at(reader, publics["R"], "cv", 0) == n * WI.BC_BYTES
    neg = a5.wiring_negatives((writer, reader), publics)
    assert neg["wired-row-not-its-writer-s-output (§16.13)"] == {"refused": True}


class Held:
    """A registered value as the committer reads it: its rows of 16-bit leaves, one salt a row, a root and each row's path
    (stand-ins)."""

    def __init__(self, name: str, rows: np.ndarray) -> None:
        self.name = self.value = name
        self.rows, self.words, self.word_bits, self.leaves = rows, rows.shape[1], 16, len(rows)
        self.salts = [hashlib.shake_256(b"salt %d" % i).digest(WI.SALT_BYTES) for i in range(len(rows))]
        self.root = hashlib.sha512(b"root " + name.encode()).digest()

    def path(self, position: int) -> bytes:
        return hashlib.sha512(b"path %d " % position + self.name.encode()).digest()


def test_a_wired_row_is_flipped_where_its_unit_reads_it_in_a_shared_row_member_whose_wired_port_is_not_its_first():
    """A reader whose ``x`` is a registered row of its own and whose ``w`` is its writer's output is a shared-row member, its
    body its ports' tables and then its units' refs, with the wired port its second. The flipped string is the row of the
    wired port's table that unit 0's ref names, so the wiring check refuses it; the body's second string is a row of the
    ``x`` table, which no wire reads."""
    xs = bf16(UNITS, 16)
    circuit = I.Circuit("held", ((BIG, UNITS), (GEMM, UNITS)), external=(f"{BID}/x",),
                        wires=(I.Wire(GID, "w", BID, "o0", at=0, radix=(UNITS,), base=0, step=(1,)),),
                        registered=I.Registered({"xs": UNITS * 16},
                                                reads=(I.Read(GID, "x", "xs", at=0, radix=(UNITS,), base=0, step=(16,)),)))
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, 2, registered={"xs": Held("xs", xs)})
    x, w = bf16(UNITS, 16), bf16(UNITS, 256, 16)
    y = gemm(BIG, x, w)
    win.record(BIG, [x, w], [y])
    win.record(GEMM, [xs, y.reshape(UNITS, 16, 16)], [gemm(GEMM, xs, y.reshape(UNITS, 16, 16))])
    c = win.close(salt_key=bytes(32))
    ms = win.window.members
    reader = next(m for m in ms if m.wired)
    assert win.mismatches() == {} and reader.tables == (("x", 2), ("w", 2)) and [p for p, _o, _ws in reader.wired] == ["w"]
    publics = {m.key: c.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {}
    neg = a5.wiring_negatives(ms, publics)
    assert set(neg) == {"chunk-wired-to-another-unit (§16.13)", "wired-row-not-its-writer-s-output (§16.13)"}
    assert [name for name, v in neg.items() if not v["refused"]] == []
    second = bytearray(publics[reader.key])
    second[WI.BC_BYTES] ^= 1
    assert WI.check(ms, {**publics, reader.key: bytes(second)}) == {}
