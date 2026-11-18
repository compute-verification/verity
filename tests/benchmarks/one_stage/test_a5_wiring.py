"""a5's wired reads under the one commitment, on a window whose layer 1 reads layer 0's rows: at two units a member, at one
(the torch loop's wired circuit, `--cap 1`), on a fan-out (every reader unit reading writer unit 0) and on a reversed wire,
a reader's wired port commits nothing of its own: its unit reads the leaf of its writer template's output tree the wire
gives in closed form (`rows.Binding.reads`), which is its writer unit's output row, and the verify opens that leaf at the
tree's root (`rows.own_reads`). A wired row flipped, a unit rewired to another unit's leaf, or to a row of other words, is
no leaf of that root at its position, so it is refused. In a shared-row member whose wired port is not its first, the
flipped leaf is the one unit 0 reads at that port, and its other port's rows are no leaf of the wired tree."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.commitments.periods import interface as I
from verity.commitments.periods import links as LK
from verity.commitments.periods import rows as CS
from verity.commitments.periods import wiring as WI

def fixed(w, seed: int = 0) -> dict[str, bytes]:
    """Salts a test hands in for every port a commit takes in full (`rows.full_ports`), so two commits of one period are
    one commitment."""
    from verity.commitments.periods import rows

    w = getattr(w, "window", w)
    return {p: hashlib.shake_256(f"{seed} {p}".encode()).digest(n * rows.SALT_BYTES) for p, n in rows.full_ports(w).items()}

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


def closed(circuit: I.Circuit, n: int, records) -> tuple[I.Window, CS.Committed]:
    win = I.Window(circuit, "ab" * 64, {"law": "stratified", "k": 2}, n)
    for fn, x, w in records:
        win.record(fn, [x, w], [gemm(fn, x, w)])
    c = win.close(salts_of=fixed(win))
    assert win.mismatches() == {} and max(m.n for m in win.window.members) == n
    return win, c


def wired_reads(win: I.Window, c: CS.Committed, port: str = "x") -> list[tuple[CS.Member, LK.Port, np.ndarray]]:
    """Each reader member wired at ``port``, its writer template's output tree as the period committed it, and the leaf each
    of its units reads; refused where the reader commits a tree of its own at ``port``."""
    b, out = win.window, []
    for m in b.members:
        wire = b.wire(m, port)
        if wire is None:
            continue
        assert m.port(port) not in c.roots and b.tree(m, port) == WI.tree(wire.writer, wire.out)
        tree = template_tree(win, c, wire.writer, wire.out)
        assert c.bc[m.port(port)] == b"".join(leaf(tree, int(i)) for i in b.leaves(m, port))
        out.append((m, tree, b.reads(m, port)))
    assert out
    return out


def template_tree(win: I.Window, c: CS.Committed, template: str, out: str) -> LK.Port:
    """Template ``template``'s output tree at ``out``: its members' output rows in unit order, held to the period's root."""
    ms = sorted((m for m in win.window.members if m.template == template), key=lambda m: win.window.owns[m.key])
    name, m0 = WI.tree(template, out), ms[0]
    tree = LK.Port(name, m0.schema(out), int(c.rows[m0.port(out)].shape[1]),
                   c.domains[name], b"".join(c.bc[m.port(out)] for m in ms),
                   np.concatenate([np.asarray(c.rows[m.port(out)]) for m in ms]))
    assert tree.root == c.roots[name]
    return tree


def opens(tree: LK.Port, position: int, bc: bytes) -> bool:
    """Whether ``bc`` at leaf ``position`` makes ``tree``'s root: what opening it there by its path checks."""
    n = CS.BC_BYTES
    return replace(tree, bcs=tree.bcs[:position * n] + bytes(bc) + tree.bcs[(position + 1) * n:]).root == tree.root


def leaf(tree: LK.Port, position: int) -> bytes:
    return tree.bcs[position * CS.BC_BYTES:(position + 1) * CS.BC_BYTES]


def flipped(bc: bytes) -> bytes:
    return bytes([bc[0] ^ 1]) + bc[1:]


def refused_at_unit_0(tree: LK.Port, at: np.ndarray, other: bytes) -> None:
    """Unit 0's honest leaf opens; that leaf flipped, and ``other`` (a row the wire doesn't give it) in its place, don't."""
    assert opens(tree, int(at[0]), leaf(tree, int(at[0])))
    assert not opens(tree, int(at[0]), flipped(leaf(tree, int(at[0]))))
    assert other != leaf(tree, int(at[0])) and not opens(tree, int(at[0]), other)




@pytest.mark.parametrize("cap, step", [(2, 1), (1, 1), (2, 0)])
def test_each_wired_read_is_its_writer_units_leaf_and_a_flip_or_another_units_leaf_is_refused(cap, step):
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=0, step=(step,)),),
                        order=((GID, 0, UNITS), (GID, UNITS, UNITS)))
    w, x0 = bf16(2 * UNITS, 16, 16), bf16(UNITS, 16)
    x1 = gemm(GEMM, x0, w[:UNITS])[np.arange(UNITS) * step]
    win, c = closed(circuit, cap, [(GEMM, np.concatenate([x0, x1]), w)])
    reads = wired_reads(win, c)
    assert np.array_equal(np.concatenate([at for _m, _t, at in reads]), np.arange(UNITS) * step)
    for m, tree, at in reads:
        assert np.array_equal(tree.rows_at(at), x1[m.base - UNITS:m.base - UNITS + m.n])
    _m, tree, at = reads[0]
    refused_at_unit_0(tree, at, leaf(tree, int(at[0]) + 1))


def test_a_unit_no_other_unit_writes_its_words_for_is_refused_a_row_of_other_words():
    circuit = I.Circuit("fan", ((GEMM, 1), (WIDE, UNITS)), external=(f"{GID}/x",),
                        wires=(I.Wire(WID, "x", GID, "o0", at=0, radix=(UNITS,), base=0, step=(0,)),))
    x0, w0 = bf16(1, 16), bf16(1, 16, 16)
    win, c = closed(circuit, 1, [(GEMM, x0, w0), (WIDE, np.repeat(gemm(GEMM, x0, w0), UNITS, 0), bf16(UNITS, 32, 16))])
    assert [dict(m.out_rows)["o0"] for m in win.window.members] == [16] + [32] * UNITS
    reads = wired_reads(win, c)
    assert all(np.array_equal(at, [0]) for _m, _t, at in reads) and reads[0][1].leaves == 1
    wide = template_tree(win, c, reads[0][0].template, "o0")
    assert wide.words == 32 and reads[0][1].words == 16
    refused_at_unit_0(reads[0][1], reads[0][2], leaf(wide, 0))


def test_a_reversed_wires_unit_0_reads_the_last_writer_units_leaf_and_its_flip_is_refused():
    """Each reader unit reads the writers in reverse: unit 0's leaf is the writer's last unit's, the first row of its table
    (`rows.Binding.leaves`, first-read order), and that leaf flipped, or the writer's unit 0's in its place, is refused."""
    circuit = I.Circuit("chain", ((GEMM, 2 * UNITS),), external=(f"{GID}/x",),
                        wires=(I.Wire(GID, "x", GID, "o0", at=UNITS, radix=(UNITS,), base=UNITS - 1, step=(-1,)),),
                        order=((GID, 0, UNITS), (GID, UNITS, UNITS)))
    w, x0 = bf16(2 * UNITS, 16, 16), bf16(UNITS, 16)
    x1 = gemm(GEMM, x0, w[:UNITS])[::-1]
    win, c = closed(circuit, UNITS, [(GEMM, np.concatenate([x0, x1]), w)])
    ((m, tree, at),) = wired_reads(win, c)
    assert at.tolist() == list(range(UNITS - 1, -1, -1)) and win.window.leaves(m, "x").tolist() == at.tolist()
    assert win.window.refs(m, "x").tolist() == list(range(UNITS)) and np.array_equal(tree.rows_at(at), x1)
    refused_at_unit_0(tree, at, leaf(tree, 0))


class Held:
    """A registered value as the committer reads it: its rows of 16-bit leaves, one salt a row and each row's ``b || c``
    under it in a ``row/v2`` row, a root and each row's path (stand-ins)."""

    def __init__(self, name: str, rows: np.ndarray) -> None:
        self.name = self.value = name
        self.rows, self.words, self.word_bits, self.leaves = rows, rows.shape[1], 16, len(rows)
        self.schema = V2
        self.salts = [hashlib.shake_256(b"salt %d" % i).digest(CS.SALT_BYTES) for i in range(len(rows))]
        self.root = hashlib.sha512(b"root " + name.encode()).digest()
        inner = CS._inner(V2, self.words)
        self.bcs = [CS.INST.commit_string(CS.INST.default_key, inner(np.ascontiguousarray(r, dtype="<u2").tobytes()), s)
                    for r, s in zip(rows, self.salts)]

    def path(self, position: int) -> bytes:
        return hashlib.sha512(b"path %d " % position + self.name.encode()).digest()


def test_a_wired_row_is_flipped_where_its_unit_reads_it_in_a_shared_row_member_whose_wired_port_is_not_its_first():
    """A reader whose ``x`` is a registered row of its own and whose ``w`` is its writer's output is a shared-row member,
    with the wired port its second. Its unit 0's ``w`` is the writer's unit 0's leaf, which flipped is refused; its ``x``
    row, which no wire reads, is no leaf of the wired tree there."""
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
    c = win.close(salts_of=fixed(win))
    reads = wired_reads(win, c, "w")
    reader = reads[0][0]
    assert win.mismatches() == {} and reader.tables == (("x", 2), ("w", 2)) and [x.port for x in reader.wired] == ["w"]
    assert [p for p, _w in reader.rows].index("w") == 1
    assert np.array_equal(np.concatenate([at for _m, _t, at in reads]), np.arange(UNITS))
    assert all(np.array_equal(t.rows_at(at), y[m.base - UNITS:m.base - UNITS + m.n]) for m, t, at in reads)
    _m, tree, at = reads[0]
    body = c.public_body(win.window.members.index(reader))
    refused_at_unit_0(tree, at, body[:CS.BC_BYTES])
