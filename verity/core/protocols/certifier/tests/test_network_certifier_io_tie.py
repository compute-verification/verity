"""What the network certifier's ties share (`io_tie`): the id form's cells, each with one reading, the network
certifier's own; a Commit's rows, one hiding row a step in a dense frame-v3-sha512 tree that is `advice`'s registration
of them, which the verifier's own ports read at public steps as the Lean verifier's registered reads make them,
refusing every read that isn't the Commit's; and the Beneš network, in the clear and as its Boolean Definition."""

import hashlib
import random
from typing import NamedTuple

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.gates.frame_v3 import _part, opens
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf
from verity.core.protocols.certifier import head_check as HC, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER, PAD

V = 151_329
#: a Commit's port: a row a step of the session, P ids each
P, STEPS = 4, 8


def _h(label: bytes) -> bytes:
    return hashlib.sha512(label).digest()


class Port(NamedTuple):
    """A fake Commit's port: its rows, their salts, their tree and its domain."""

    rows: list[bytes]
    salts: list[bytes]
    levels: list[list[bytes]]
    dom: object


def commit(seed: bytes = b"w") -> Port:
    """STEPS rows of up to P sampled ids each, committed as the Commit registers them."""
    rng = random.Random(seed)
    rows = [IT.port_row([rng.randrange(V) for _ in range(rng.randrange(1, P + 1))], P) for _ in range(STEPS)]
    salts = [IT.fake_salt(seed + b"p", s) for s in range(STEPS)]
    dom = IT.domain(_h(b"port" + seed), STEPS)
    return Port(rows, salts, IT.rows_levels(dom.domain_id, rows, salts), dom)


#: three proof blocks reading S = 2 port rows each, at the verifier's steps
STEPS_READ = ((0, 1), (3, 3), (5, 7))


def statement(port: Port, steps, s: int) -> tuple[dict, list[bytes]]:
    """A statement's registered entry of port row s over the proof blocks (proof block u's s-th read row at its step,
    with its path, as `open_port` gives them) and each table row's b ‖ c."""
    opened = [IT.open_port(port.levels, port.rows, port.salts, st) for st in steps]
    entry = {"value": IT.SAMPLED, "positions": [st[s] for st in steps],
             "paths": [b"".join(o[2][s]).hex() for o in opened]}
    return entry, [WR.bc(o[0][s], o[1][s]) for o in opened]


def refusals(port: Port, steps=STEPS_READ, reads=STEPS_READ, root: bytes | None = None) -> list[str | None]:
    """The Lean verifier's registered reads (`warden_rows.refusal`) of a statement opening `steps` against the
    verifier's own ports (`port_reads`) of `reads`, a port row at a time: None where they accept."""
    root = port.levels[-1][0] if root is None else root
    ports = IT.port_reads(reads, root, port.dom.domain_id, STEPS, P)
    return [WR.refusal(ports[f"{IT.PORT}[{s}]"], port.dom, *statement(port, steps, s)) for s in range(len(reads[0]))]


def test_a_cell_parses_to_its_one_reading_and_is_the_network_certifiers():
    P_ = 256 - HEADER.size
    for x, s in ((0, 0), (7, 3), (151_328, 999), ((1 << 32) - 1, (1 << 32) - 1)):
        cell = IT.data_cell(x, s, 256)
        assert IT.parse_cell(cell) == (DATA, s, x) and W.read_cell(cell, 256) == (DATA, s, IT.id_event(x, P_))
        assert IT.parse_cell(IT.pad_cell(256, s)) == (PAD, s, 0) and W.read_cell(IT.pad_cell(256, s), 256) == (PAD, s, b"")
        assert IT.parse_cell(IT.end_cell(s, 256)) == (IT.END, s, 0)
        assert W.read_cell(IT.end_cell(s, 256), 256) == (W.END, s, IT.TRAILER)
    assert IT.END == W.END and IT.pad_cell(256) == bytes(256) and len(IT.TRAILER) < IT.ID_EVENT_BYTES
    assert IT.data_cell(5, 1, IT.CELL_USED) == HEADER.pack(DATA, 1, IT.ID_EVENT_BYTES) + IT.id_event(5)
    good, end = IT.data_cell(12, 3, 256), IT.end_cell(3, 256)
    for bad in (good[:IT.ID_AT] + b"0000000012" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b" " * 10 + good[IT.ID_AT + 10:],
                good[:IT.ID_AT] + b"4294967296" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b"      1 2 " + good[IT.ID_AT + 10:],
                good[:-3] + b"!\n\n", good[:-12] + b"%10d" % 7 + b"\n\n", bytes([DATA ^ 1]) + good[1:],
                IT.pad_cell(256, 9)[:-1] + b"\x01", end[:-1] + b"\x01", W.cell(W.FAIL, 3, b"", 256),
                W.cell(W.END, 3, b"", 256), W.cell(W.END, 3, IT.id_event(12, P_), 256), W.cell(W.REFUSED, 0, b"", 256),
                good[:-1]):
        assert IT.parse_cell(bad) is None
    with pytest.raises(ValueError, match="u32"):
        IT.data_cell(1 << 32, 0, 256)
    with pytest.raises(ValueError, match="at least"):
        IT.pad_cell(IT.CELL_USED - 1)


def test_a_port_row_is_its_steps_ids_and_a_route_covers_its_positions():
    row = IT.port_row([7, 1 << 31], 4)
    assert row == (7).to_bytes(4, "big") + (1 << 31).to_bytes(4, "big") + bytes(8)
    with pytest.raises(ValueError, match="at most 4"):
        IT.port_row([1] * 5, 4)
    with pytest.raises(ValueError, match="at most 4"):
        IT.port_row([1 << 32], 4)
    assert [IT.route_depth(c, e) for c, e in ((1, 1), (2, 2), (3, 2), (16, 9), (17, 4), (12, 64))] == [1, 1, 2, 4, 5, 6]


def test_a_rows_root_is_its_domains_tree_and_a_path_opens_its_row():
    for count in (2, 3, 5, 8):
        dom = IT.domain(_h(b"d"), count)
        rows, salts = [bytes([i]) * 40 for i in range(count)], [IT.fake_salt(b"s", i) for i in range(count)]
        leaves = [row_leaf(dom, i, HC.hiding_value(r, s), IT.SCHEMA) for i, (r, s) in enumerate(zip(rows, salts))]
        levels = IT.rows_levels(dom.domain_id, rows, salts)
        root = levels[-1][0]
        assert root == tree_levels(dom, leaves)[-1][0]
        for i in range(count):
            path = IT.rows_path(levels, i)
            assert opens(IT.SCHEMA, root, dom.domain_id, HC.hiding_value(rows[i], salts[i]), i, path) == 1
            assert opens(IT.SCHEMA, root, dom.domain_id, HC.hiding_value(rows[i - 1], salts[i - 1]), i, path) == 0
    with pytest.raises(ValueError, match="at least 2 rows"):
        IT.rows_levels(_h(b"d"), [bytes(40)], [IT.fake_salt(b"s", 0)])


def test_the_ports_reads_are_the_commits_registration():
    """The fake Commit's port tree is `verity.ml.commitments.advice`'s registration of its rows (`hm96-sha512/row/v2`
    leaves under `domain`), the form the served rows' registration takes, so its root is one the Lean verifier's
    registered reads open. `port_reads` has proof block `reads_from + u` read its s-th step at `port[s]`, a row of 2P
    u16 words, keys a one-row port `port`, and refuses proof blocks of unequal S and a read past the Commit's rows."""
    port = commit()
    binding, root = _h(b"portw"), port.levels[-1][0]
    assert ADV.commit(port.rows, 32 * P, binding, port.salts).root == root
    assert ADV.domain(binding, STEPS).domain_id == IT.domain(binding, STEPS).domain_id == port.dom.domain_id
    ports = IT.port_reads(STEPS_READ, root, port.dom.domain_id, STEPS, P, reads_from=3)
    assert ports == {f"port[{s}]": {
        "value": IT.SAMPLED, "root": root.hex(), "leaves": STEPS, "words": 2 * P, "word_bits": 16,
        "domain": port.dom.domain_id.hex(), "row_schema": IT.SCHEMA, "reads": [st[s] for st in STEPS_READ],
        "reads_from": 3} for s in range(2)}
    assert list(IT.port_reads([(5,), (2,)], root, port.dom.domain_id, STEPS, P)) == [IT.PORT]
    with pytest.raises(ValueError, match="same S"):
        IT.port_reads([(1, 2), (1,)], root, port.dom.domain_id, STEPS, P)
    with pytest.raises(ValueError, match=f"a step of the Commit's {STEPS} rows"):
        IT.port_reads([(1, STEPS)], root, port.dom.domain_id, STEPS, P)


def test_the_hosts_registered_reads_climb_the_domain_the_verifiers_port_names():
    """`warden_rows.refusal` climbs under the `dom` it is handed, and the Lean verifier under its port's own `domain`
    and `leaves`, so a port naming another domain or leaf count (another Commit's, say) gets another verdict there: the
    host's read refuses to stand in for it rather than answer for the domain it was handed."""
    port = commit()
    port_ = IT.port_reads(STEPS_READ, port.levels[-1][0], port.dom.domain_id, STEPS, P)["port[0]"]
    assert WR.refusal(port_, port.dom, *statement(port, STEPS_READ, 0)) is None
    for wrong in ({"domain": _h(b"another domain").hex()}, {"leaves": 2 * STEPS}):
        with pytest.raises(ValueError, match="names"):
            WR.refusal(port_ | wrong, port.dom, *statement(port, STEPS_READ, 0))


def test_a_read_that_isnt_the_commits_is_refused():
    """The read rows are the Commit's registered rows at the verifier's steps: a row, a salt, a sibling or a step other
    than the opening's, or the root of another commitment, each refuse the statement's registered reads of the port,
    which accept the honest opening."""
    port = commit()
    assert refusals(port) == [None, None]
    row = bytearray(port.rows[7])
    row[2] ^= 1
    levels = [list(level) for level in port.levels]
    levels[1][7 >> 1 ^ 1] = _h(b"x")
    stepped = (STEPS_READ[0], STEPS_READ[1], (5, 6))
    tampers = {"a row": port._replace(rows=[*port.rows[:7], bytes(row)]),
               "a salt": port._replace(salts=[*port.salts[:7], IT.fake_salt(b"x", 0)]),
               "a sibling": port._replace(levels=levels)}
    for name, tampered in tampers.items():
        assert refusals(tampered)[0] is None and "does not open" in refusals(tampered)[1], name
    assert refusals(port, steps=stepped)[0] is None and "reads position 6" in refusals(port, steps=stepped)[1]
    assert all("does not open" in r for r in refusals(port, root=commit(b"another commit").levels[-1][0]))


def test_the_benes_network_routes_every_permutation():
    rng = random.Random(4)
    for n in (2, 4, 8, 16, 128):
        for _ in range(20):
            perm = list(range(n))
            rng.shuffle(perm)
            bits, D = IT.benes(perm), n.bit_length() - 1
            assert len(bits) == (2 * D - 1) << (D - 1)
            out = IT.benes_apply(list(range(n)), bits)
            assert all(out[perm[i]] == i for i in range(n))
    for bad in ([0], [0, 0, 1, 2], [0, 1, 2]):
        with pytest.raises(ValueError, match="permutation"):
            IT.benes(bad)
    with pytest.raises(ValueError, match="more switch bits"):
        IT.benes_apply([0, 1], [0, 1])


def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def test_the_route_circuit_is_its_reference():
    rng = random.Random(5)
    W_, D = 9, 4
    xs = [[rng.randrange(2) for _ in range(W_)] for _ in range(1 << D)]
    cases = []
    for _ in range(3):
        perm = list(range(1 << D))
        rng.shuffle(perm)
        cases.append(IT.benes(perm))
    cases.append([rng.randrange(2) for _ in range(len(cases[0]))])
    fn = bind(IT.IoTieRoute, W=W_, D=D)
    out = np.asarray(evaluate_bits(fn, _bits([[b for x in xs for b in x]] * len(cases)), _bits(cases)))
    for bits, o in zip(cases, out.reshape(len(cases), 1 << D, W_)):
        assert o.tolist() == IT.benes_apply(xs, bits)


def _build_flat_route(C, S, *e):
    """`IoTieRoute` as one flat trace, the build the recursive one replaced."""
    it = iter(e[-1])

    def swap(s, a, b):
        t = [C.AND(s, x ^ y) for x, y in zip(a, b)]
        return [x ^ u for x, u in zip(a, t)], [y ^ u for y, u in zip(b, t)]

    def run(v):
        if len(v) == 2:
            return list(swap(next(it), v[0], v[1]))
        up, down = [], []
        for k in range(len(v) // 2):
            a, b = swap(next(it), v[2 * k], v[2 * k + 1])
            up.append(a)
            down.append(b)
        up, down = run(up), run(down)
        res = []
        for a, b in zip(up, down):
            res += swap(next(it), a, b)
        return res

    return run([list(x) for x in e[:-1]])


FLAT_ROUTE = _part("IoTieRouteFlat", ("W", "D"), IT._sig_route, _build_flat_route, "IoTieRoute's flat trace")


def test_the_recursive_route_is_the_flat_one():
    """`IoTieRoute` (a column of switches, two half networks, a column) against the flat trace it replaced: the same
    ANDs, W a switch, and the same outputs on random inputs and switch bits, at depths 1 to 6; and a binding's body is
    its four nodes, so a deep network is built once a depth (IoTieRequestIds{4096, 16, 1, 512, 6}'s sort is D = 14)."""
    rng = random.Random(7)
    for W_, D in ((1, 1), (5, 1), (3, 2), (8, 3), (9, 4), (36, 4), (44, 5), (2, 6)):
        new, flat = bind(IT.IoTieRoute, W=W_, D=D), bind(FLAT_ROUTE, W=W_, D=D)
        n = (2 * D - 1) << (D - 1)
        assert HC.ands(new) == HC.ands(flat) == W_ * n
        xs = _bits([[rng.randrange(2) for _ in range(W_ << D)] for _ in range(16)])
        bits = _bits([[rng.randrange(2) for _ in range(n)] for _ in range(16)])
        assert np.array_equal(np.asarray(evaluate_bits(new, xs, bits)), np.asarray(evaluate_bits(flat, xs, bits)))
    assert HC.ands(bind(IT.IoTieSwitch, W=36)) == 36
    deep = bind(IT.IoTieRoute, W=46, D=14)
    assert [nd.form for nd in deep.body.nodes] == ["batch", "call", "call", "batch"]
    assert HC.ands(deep) == 46 * (27 << 13)
