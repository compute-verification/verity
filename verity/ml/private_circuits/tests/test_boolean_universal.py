"""`verity.ml.private_circuits.universal`: the universal unit agrees with `interpret` on random private circuits (in range and not), through the
plain-integer reference and the bit-sliced evaluator; padding leaves a private circuit's outputs unchanged; the unit is a function of its
class alone, pinned by gate counts and program digest, with exactly `and_count` ANDs."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.private_circuits import universal as U

#: (G, N_IN, N_OUT) -> (And, Xor, Not gates, standalone program digest)
PINS = {
    (16, 16, 8): (1066, 1586, 0, "e2b4dde3a31d65a1a7f5af2b50540a89f713efa296e8e42f0d7bced7d4a1065510c19d1b45a27023edeb5eef812eb4d62e8a20c99d4b3e5b9c46037a43b451dc"),
    (64, 64, 32): (16674, 24898, 0, "39698ff01cb97b6998cb2da36bb14bc81e69acd724594cf2f9c2114770a9cee0181f964c9c9ba06f9beff51369ed6cab1496097afb2ca7b50e703b7f437a6241"),
}
CLASSES = [(1, 1, 1), (3, 2, 2), (5, 1, 3), (16, 16, 8), (20, 13, 5), (64, 64, 32)]


def _unit(G: int, N_IN: int, N_OUT: int):
    return bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)


def _netlist(rng: random.Random, G: int, N_IN: int, N_OUT: int, in_range: bool) -> U.Netlist:
    """Sources below each gate's own wire and outputs below W, or else any K-bit index (forward reads, past W)."""
    k = U.index_bits(G, N_IN)
    W = N_IN + G
    gates = tuple((rng.randrange(4), *(rng.randrange(N_IN + g) if in_range else rng.randrange(1 << k) for _ in range(2)))
                  for g in range(G))
    return U.Netlist(N_IN, gates, tuple(rng.randrange(W if in_range else 1 << k) for _ in range(N_OUT)))


def _columns(nets, xs) -> list[np.ndarray]:
    args = [U.arguments(n, x) for n, x in zip(nets, xs)]
    return [np.array([a[i] for a in args], dtype=np.uint8) for i in range(4)]


def test_the_ops_and_the_reads_past_the_wires() -> None:
    gates = [(U.XOR, 0, 1), (U.AND, 0, 1), (U.XNOR, 0, 1), (U.NAND, 0, 1),
             (U.XOR, 0, 0), (U.XNOR, 1, 1), (U.NAND, 0, 0), (U.AND, 1, 1),
             (U.AND, 1, 10), (U.XNOR, 1, 99)]
    for a in (0, 1):
        for b in (0, 1):
            net = U.Netlist(2, tuple(gates), tuple(range(2, 12)) + (12, 999))
            assert U.interpret(net, [a, b]) == [a ^ b, a & b, 1 ^ a ^ b, 1 ^ (a & b), 0, 1, 1 ^ a, b, 0, 1 ^ b, 0, 0]


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_the_ands_are_the_closed_form(cls) -> None:
    assert U.traced(*cls).counts["and"] == BS.compile(_unit(*cls)).counts["and"] == U.and_count(*cls)


@pytest.mark.parametrize("cls", sorted(PINS), ids=str)
def test_the_unit_is_a_function_of_its_class(cls) -> None:
    """Its circuit is pinned by digest: no private circuit enters it but as the values of op, src and out."""
    ands, xors, nots, digest = PINS[cls]
    target = _unit(*cls)
    assert BL.is_boolean(target) and target is _unit(*cls)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest
    G, N_IN, N_OUT = cls
    k = U.index_bits(G, N_IN)
    assert [p.leaves for _, p in target.signature()[0]] == [2 * G, 2 * G * k, N_OUT * k, N_IN]


@pytest.mark.parametrize("cls", CLASSES, ids=str)
def test_random_netlists_agree_with_interpret(cls) -> None:
    rng = random.Random(f"universal{cls}")
    target = _unit(*cls)
    G, N_IN, N_OUT = cls
    nets = [_netlist(rng, *cls, in_range=i % 2 == 0) for i in range(128)]
    xs = [[rng.randrange(2) for _ in range(N_IN)] for _ in nets]
    want = [U.interpret(n, x) for n, x in zip(nets, xs)]
    got = np.asarray(E.evaluate_batch(target, *_columns(nets, xs))).reshape(len(nets), N_OUT)
    assert got.tolist() == want
    for n, x, w in list(zip(nets, xs, want))[:2]:
        assert list(E.evaluate(target, *U.arguments(n, x))) == w


def _decode(op, src, out, G: int, N_IN: int, N_OUT: int) -> U.Netlist:
    k = U.index_bits(G, N_IN)

    def word(bits, i, w):
        return sum(b << j for j, b in enumerate(bits[i * w:i * w + w]))

    return U.Netlist(N_IN, tuple((word(op, g, 2), word(src, 2 * g, k), word(src, 2 * g + 1, k)) for g in range(G)),
                     tuple(word(out, i, k) for i in range(N_OUT)))


def test_every_program_bit_flip_is_another_netlist_on_the_same_circuit() -> None:
    """Each private circuit one program bit away from a random one, run by the one circuit of the class, gives the outputs
    `interpret` gives for that private circuit: program bits are values only."""
    cls = G, N_IN, N_OUT = 16, 16, 8
    rng = random.Random(11)
    target = _unit(*cls)
    x = [rng.randrange(2) for _ in range(N_IN)]
    op, src, out, xb = U.arguments(_netlist(rng, *cls, in_range=True), x)
    program = op + src + out
    rows, want = [], []
    for i in range(len(program)):
        bits = list(program)
        bits[i] ^= 1
        rows.append(bits)
        want.append(U.interpret(_decode(bits[:2 * G], bits[2 * G:len(op) + len(src)], bits[len(op) + len(src):], *cls), x))
    a = np.array(rows, dtype=np.uint8)
    cols = [a[:, :len(op)], a[:, len(op):len(op) + len(src)], a[:, len(op) + len(src):], np.array([xb] * len(rows), dtype=np.uint8)]
    got = np.asarray(E.evaluate_batch(target, *cols)).reshape(len(rows), N_OUT)
    assert got.tolist() == want
    assert len({tuple(w) for w in want}) > 1


@pytest.mark.parametrize("cls", [(16, 16, 8), (64, 64, 32)], ids=str)
def test_padding_leaves_the_outputs_unchanged(cls) -> None:
    G, N_IN, N_OUT = cls
    rng = random.Random(f"pad{cls}")
    target = _unit(*cls)
    nets, xs, want = [], [], []
    for _ in range(64):
        g, n = rng.randint(1, G), rng.randint(1, N_IN)
        small = _netlist(rng, g, n, N_OUT, in_range=True)
        x = [rng.randrange(2) for _ in range(n)]
        net = U.pad(small, G, N_IN)
        assert len(net.gates) == G and net.n_in == N_IN and net.gates[g:] == (U.NOOP,) * (G - g)
        nets.append(net)
        xs.append(x + [rng.randrange(2) for _ in range(N_IN - n)])
        want.append(U.interpret(small, x))
        assert U.interpret(net, xs[-1]) == want[-1]
    got = np.asarray(E.evaluate_batch(target, *_columns(nets, xs))).reshape(len(nets), N_OUT)
    assert got.tolist() == want


def test_malformed_netlists_are_refused() -> None:
    with pytest.raises(ValueError, match="op in 0..3"):
        U.Netlist(1, ((4, 0, 0),), (0,))
    with pytest.raises(ValueError, match="in range"):
        U.pad(U.Netlist(1, ((U.AND, 0, 1),), (0,)), 4, 2)
    with pytest.raises(ValueError, match="does not fit"):
        U.pad(U.Netlist(3, ((U.AND, 0, 1),), (0,)), 4, 2)
    with pytest.raises(ValueError, match="4-bit wire indices"):
        U.arguments(U.Netlist(8, ((U.AND, 0, 16),), (0,)), [0] * 8)
    with pytest.raises(ValueError, match="N_OUT is an int >= 1"):
        _unit(4, 4, 0).signature()
