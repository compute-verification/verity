"""`verity_flock.rec_link`, V*'s link check: on an inner population whose rows are committed under fresh salts, every honest
table row (a link, or a filler) climbs to the two roots the verifier registers, and a row that is malformed, out of range
or whose link does not hold gives zero roots; the M0 unit and the Definition agree with the reference on all of them."""
from __future__ import annotations

import json
import random
import struct

import numpy as np
import pytest
from verity_flock import rec_link as R
from verity_flock import rec_links as RL

from verity.ml.commitments import hm96

KEY = hm96.SHA512.default_key
UB, PB = 2, 1
UNITS, INS, OUTS = 3, 2, 2


def _population(seed: int = 1):
    """Each unit's output rows (digest, salt) and input rows, unit u + 1's input row 0 reading unit u's output row 1, and the
    verifier's two trees over the rows' commit strings."""
    g = random.Random(seed)
    def rnd(n):
        return bytes(g.getrandbits(8) for _ in range(n))
    outs = {(u, o): (rnd(64), rnd(192)) for u in range(UNITS) for o in range(OUTS)}
    ins = {(u, p): (rnd(64), rnd(192)) for u in range(UNITS) for p in range(INS)}
    links = [RL.LinkEntry(u + 1, 0, u, 1) for u in range(UNITS - 1)]
    for l in links:
        ins[l.reader, l.port] = (outs[l.writer, l.out][0], rnd(192))
    def tree(rows):
        return R.Tree({R.index(u, p, PB): R.leaf(hm96.SHA512.commit_string(KEY, x, y)) for (u, p), (x, y) in rows.items()},
                      R.depth(UB, PB))
    return ins, outs, links, tree(ins), tree(outs)


def _words(b: bytes) -> list[int]:
    return [b[i] | b[i + 1] << 8 for i in range(0, len(b), 2)]


def _instance(row: bytes, ins, outs, tin, tout, reader=None, writer=None):
    """An instance's port words: the table row, and the reader's and the writer's rows with their openings (by default the
    ones the row names, a filler's leaf 0 of each tree)."""
    f = struct.unpack_from("<5I", row)
    reader = reader or (f[1], f[2])
    writer = writer or (f[3], f[4])
    D = R.depth(UB, PB)
    xr, yr = ins[reader]
    xw, yw = outs[writer]
    return (_words(row), _words(xr), _words(yr), R.aux(tin.path(R.index(*reader, PB)), D),
            _words(xw), _words(yw), R.aux(tout.path(R.index(*writer, PB)), D))


def _cases():
    """``(ports, ok)``: the table's honest rows, then forged ones."""
    ins, outs, links, tin, tout = _population()
    rows = RL.table(links)
    honest = [(_instance(r, ins, outs, tin, tout), True) for r in rows + [bytes(RL.ROW_BYTES)]]
    bad_link = RL.encode(RL.LinkEntry(2, 0, 0, 1))
    forged = [
        (_instance(bad_link, ins, outs, tin, tout), False),
        (_instance(struct.pack("<I", 2) + rows[0][4:], ins, outs, tin, tout, (1, 0), (0, 1)), False),
        (_instance(rows[0][:-1] + b"\1", ins, outs, tin, tout), False),
        (_instance(RL.encode(RL.LinkEntry(4, 0, 0, 1)), ins, outs, tin, tout, (0, 0), (0, 1)), False),
        (_instance(RL.encode(RL.LinkEntry(1, 2, 0, 1)), ins, outs, tin, tout, (1, 0), (0, 1)), False),
        (_instance(struct.pack("<5I", 0, 1, 0, 0, 0) + bytes(108), ins, outs, tin, tout, (0, 0), (0, 0)), False),
    ]
    return honest + forged, tin, tout


def test_the_honest_rows_climb_to_the_registered_roots_and_forged_ones_give_zero():
    cases, tin, tout = _cases()
    assert len(cases) == RL.grid(UNITS - 1) + 1 + 6
    for ports, ok in cases:
        rin, rout = R.link_ref(UB, PB, *ports)
        if ok:
            assert (rin, rout) == (R.expected(tin.root), R.expected(tout.root))
        else:
            assert rin == rout == [0] * R.OUT_WORDS


def test_a_forged_sibling_misses_the_root():
    cases, tin, tout = _cases()
    ports = list(cases[0][0])
    ports[3] = [ports[3][0] ^ 1] + ports[3][1:]
    rin, rout = R.link_ref(UB, PB, *ports)
    assert rin != R.expected(tin.root) and rin != [0] * R.OUT_WORDS and rout == R.expected(tout.root)


def test_a_reader_row_committed_to_another_value_is_refused_whatever_its_salt():
    """The reader's opening is of a row whose digest is not the writer's: an honest leaf of the input tree, at the position the
    row names, still gives zero roots."""
    ins, outs, links, tin, tout = _population()
    row = RL.encode(RL.LinkEntry(1, 1, 0, 1))
    rin, rout = R.link_ref(UB, PB, *_instance(row, ins, outs, tin, tout))
    assert rin == rout == [0] * R.OUT_WORDS


@pytest.mark.parametrize("bad, error", [((-1, 0), ValueError), ((0, 17), ValueError), ((31, 0), ValueError),
                                        ((True, 0), TypeError)])
def test_the_shape_is_checked(bad, error):
    with pytest.raises(error):
        R.link_ref(*bad, [], [], [], [], [], [], [])


def test_the_tree_is_the_verifiers_over_padded_leaves():
    t = R.Tree({1: b"\1" * 64}, 2)
    import hashlib

    def h(a, b):
        return hashlib.sha512(a + b).digest()
    assert t.root == h(h(R.EMPTY, b"\1" * 64), h(R.EMPTY, R.EMPTY))
    assert t.path(1) == [R.EMPTY, h(R.EMPTY, R.EMPTY)]


def _bits(words):
    return [(w >> j) & 1 for w in words for j in range(16)]


@pytest.mark.slow
def test_the_m0_unit_computes_the_reference():
    """The unit (`forms`) on the table's honest rows and the forged ones, bit-sliced."""
    cases, _, _ = _cases()
    C, ins, outs = R.unit_circuit(UB, PB)
    vals = {nm: [0] * 16 for nm in ins}
    for t, (ports, _) in enumerate(cases):
        for nm, w in zip(ins, [w for p in ports for w in p], strict=True):
            for j in range(16):
                vals[nm][j] |= ((w >> j) & 1) << t
    got, fail = C.evaluate(vals, len(cases))
    assert fail == 0
    for t, (ports, _) in enumerate(cases):
        words = [sum(((got[nm][j] >> t) & 1) << j for j in range(16)) for nm in outs]
        rin, rout = R.link_ref(UB, PB, *ports)
        assert words == rin + rout


@pytest.mark.slow
def test_the_definition_computes_the_reference():
    """`RecLink{UB, PB}` evaluated bit-sliced from its descriptor, on the same cases as the unit."""
    from verity.ml.circuits.evaluation import evaluate_batch

    cases, _, _ = _cases()
    d = R.definition(UB, PB)
    assert d.id == f"RecLink_v1{{UB={UB},PB={PB}}}"
    def port(i):
        return np.array([np.array(_bits(c[0][i]), dtype=np.uint8).reshape(-1, 16) for c in cases])
    res = np.asarray(evaluate_batch(d, *(port(i) for i in range(len(R.PORTS)))))
    res = res.reshape(len(cases), 2 * R.OUT_WORDS, 16)
    def word(w):
        return int(sum(int(b) << j for j, b in enumerate(w)))
    for t, (ports, _) in enumerate(cases):
        rin, rout = R.link_ref(UB, PB, *ports)
        assert [word(w) for w in res[t]] == rin + rout


@pytest.mark.parametrize("ub, pb", [(0, 0), (2, 1), (12, 3), (16, 16)])
def test_the_unit_commits_its_ports_as_five_rows_of_whole_sha512_blocks(ub, pb):
    """`row_words` (`VStar.linkRows`) regroups `port_words` without moving a word: the digest and salt of each side are one
    row, and every row is whole 128-byte blocks, as ``sha512/row/v1`` (`circuit.compose`, `VStar.checkRow`) requires. The
    composed part's ports are these rows (the link part's test below)."""
    rows, ports = R.row_words(ub, pb), R.port_words(ub, pb)
    assert R.ROWS == ("trow", "xyr", "raux", "xyw", "waux")
    assert rows == (ports[0], ports[1] + ports[2], ports[3], ports[4] + ports[5], ports[6])
    assert all(2 * w % 128 == 0 for w in rows)


@pytest.mark.slow
def test_the_link_part_stages_over_the_inner_rows_and_a_forged_row_does_not_hold(tmp_path, monkeypatch):
    """`rec_vstage.link_part` over a drawn population (`rec_vstage.drawn_rows`) of three units, unit u + 1's input row u
    reading unit u's output: the part composes (its rows whole blocks), its table is the registered value ``rec-links`` its
    statement names (``rec-links.txt``), every row's openings climb to the verifier's roots and the unit gives them, and the
    forgery's first row names a writer whose output is not its reader's row. About a minute: only `circuit.compose`, after
    `RecLink{2, 1}`'s lowering, refuses a row that is not whole blocks."""
    from verity_flock import rec_vstage as VS

    monkeypatch.setenv("REC_CIRCUIT_CACHE", str(tmp_path / "circuits"))
    links = [RL.LinkEntry(1, 0, 0, 0), RL.LinkEntry(2, 1, 1, 0)]
    rows = VS.drawn_rows(3, 3, 2, 1, links)
    honest = VS.link_part(rows, links, tmp_path / "v")
    assert honest["holds"] == honest["unit_agrees"] == [True, True]
    assert (honest["UB"], honest["PB"], honest["name"], honest["instances"]) == (2, 1, R.unit_name(2, 1), 2)
    part = tmp_path / "v" / VS.LINKS_DIR
    root, n = (part / "rec-links.txt").read_text().split()[0].split(":")
    reads = json.loads((part / "registered.json").read_text())["ports"]
    assert (reads["trow"]["value"], reads["trow"]["root"], int(n)) == (RL.VALUE, root, 2)
    assert (reads["rin"]["value"], reads["rout"]["value"]) == (R.IN_VALUE, R.OUT_VALUE)
    assert [p[0] for p in VS.meta(part / "circuit.txt")["ports"]] == [*R.ROWS, "rin", "rout"]

    forged = VS.link_part(rows, links, tmp_path / "v", forge="link")
    assert forged["forged_link"] == {"row": 0, "writer": 0, "forged_writer": 1}
    assert forged["holds"] == [False, True] and forged["circuit_kept"]
    assert (forged["in_root"], forged["out_root"]) == (honest["in_root"], honest["out_root"])
    assert (tmp_path / "v" / "forged-link" / VS.LINKS_DIR / "rec-links.txt").read_text().split(":")[0] != root
    with pytest.raises(ValueError, match="not the writer's"):
        VS.link_part(rows, [RL.LinkEntry(1, 0, 1, 0)], tmp_path / "bad")
