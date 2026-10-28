"""The H1 tile's seed inputs are the verifier's. A compiled Program's `pouw.roots`,
`pouw.salt` and `pouw.positions` are registered values the one-stage verifier fills itself (`registered.supplied_root`):
each seed group's roots from the roots the record names for its operands (A's committed tap, B's registered weights,
which the verifier pins to the root recorded before the salt), the salt from its coin bound to its stream, the prover
and the record's window, and each unit's position after every unit of the log's earlier records. A registration that names any other commitment of
them is refused (`R7-supplied`), so the two attacks (roots fixed at zero and A = -b under them; one computed unit at one
position under one salt credited twice) no longer have a tile to serve."""
from __future__ import annotations

import hashlib
import random

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G, registration as R

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
I8 = Array(8, BL.BIT)
COIN = hashlib.sha512(b"the window's coin").hexdigest()
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 1}
#: the verifier's binding (its stream's digest and the prover's registered identity), which the record's window names
BINDING = {"stream": hashlib.sha512(b"the stream").hexdigest(), "prover": "a0" * 32}
WINDOW = {"kind": "served", "row": 1, "run": "r-pouw-supplied", **BINDING}


def _grid_body(B, S, x, w):
    xs = Coll(Array(S.R * S.C, x.type.elem), concat([x[i].refs for i in range(S.R) for _j in range(S.C)]))
    ws = Coll(Array(S.R * S.C, w.type.elem), concat([w[j].refs for _i in range(S.R) for j in range(S.C)]))
    return B.batch(bind(BG.GemmAccumulatorS8, K=S.K, **BG.GEMM_SM120_S8), xs, ws, axes=(0, 0))


GRID = CompositeDefinition("PouwSuppliedGrid", 1, ("R", "C", "K"),
                           lambda S: ((("x", Array(S.R, Array(S.K, I8))), ("w", Array(S.C, Array(S.K, I8)))),
                                      Array(S.R * S.C, Array(32, BL.BIT))),
                           _grid_body, conformance="protocol-defined", register=False)


def sources(c: C.Compiled) -> dict:
    """The verifier's sources of the seed inputs: A is the served port `x` and B the registered weights `w` (one seed
    group), the salt its coin bound to its binding and the record's window, and unit u's position `u.slot` after the log's units."""
    return {C.ROOTS: [[{"port": "x"}, {"value": "w"}]], C.SALT: [[{"window": COIN}]],
            C.POSITIONS: [[{"position": str(u.slot)}] for u in c.units]}


@pytest.mark.slow
def test_the_verifier_fills_the_seed_inputs_the_compiled_program_reads():
    """On a compiled 4 x 16 H1 tile at K = 64: the rows the verifier fills are the leaves `Compiled.inputs` gives the Program
    for the record's roots, the window's salt and positions after the log's, each unit reads its own row, the honest record
    passes R7 with `w` pinned, and the witness's zero roots are `R7-supplied`. Slow (about 2 min: the compile, and each
    check's program digest and population over the tile's SHA-512 gates); it is the only test that the verifier fills
    the leaves the compiled Program's seeds read, which `test_registered` checks on a small composite only."""
    c = C.compile(Program(bind(GRID, R=4, C=16, K=64)), H.SCHEME)
    assert c.boolean and len(c.units) == 1 and len(c.groups) == 1
    prog, q = c.program, c.query()
    part, n, digest = P.partition(prog, q), P.units(prog, q), P.program_digest(prog)
    ports = {p: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), p, s, n, WINDOW["run"])}
             for p, s in (("x", "hm96-sha512/row/v1"), ("out", "u16"))}
    x_root = hashlib.sha512(b"x").digest()
    served = [{"port": p, "root": x_root.hex() if p == "x" else "cd" * 64, **ports[p]} for p in ports]
    w = G.Registered.commit(digest, "w", [[0] * 512] * 16, word_bits=G.width(prog, "w"))
    rec = R.record(prog, q, LAW, SCHEME, served, WINDOW, partition=part, registered=[w.entry()])
    sup = G.sources(sources(c))
    mine = {v: G.supplied_value(prog, v, sup[v], rec, sorted(sup), prog=digest, binding=BINDING) for v in sorted(sup)}
    salt = G.window_salt(bytes.fromhex(COIN), WINDOW, BINDING)
    assert [[v for r in mine[v].rows for v in r] for v in (C.SALT, C.POSITIONS, C.ROOTS)] == c.inputs(salt, [(x_root, w.root)])
    words = {v: len(mine[v].rows[0]) for v in sup}
    reads = G.reads(prog, q, words)
    assert {r["value"]: r["rows"] for r in reads.values()} == {C.ROOTS: [0], C.SALT: [0], C.POSITIONS: [0]}
    # a later record of the log: its units take the positions after the first record's population
    later = G.supplied_rows(prog, C.POSITIONS, sup[C.POSITIONS], {**rec, "window": {**WINDOW, "row": 2}}, sorted(sup), [rec])
    assert later == [SD.position_leaves(p, True) for p in c.positions(rec["population"])]
    rec["registered"] = [w.entry()] + [mine[v].entry() for v in sorted(sup)]
    check = {"program": prog, "query": q, "law": LAW, "scheme": SCHEME, "ports": ports, "partition": part,
             "registered": {"w": 512, **words}, "supplied": sources(c), "pinned": {"w": w.root.hex()},
             "binding": BINDING}
    assert R.check(rec, **check).ok
    zero = G.Registered.commit(digest, C.ROOTS, [SD.leaves(b"\0" * 128, True)], G.supplied_salts(digest, C.ROOTS, 1),
                               G.width(prog, C.ROOTS))
    witness = {**rec, "registered": [zero.entry() if e["value"] == C.ROOTS else e for e in rec["registered"]]}
    assert R.check(witness, **check).codes == ("R7-supplied",)


def test_the_roots_witness_no_longer_cancels_the_noise():
    """The red team's roots witness fixes roots r0 and sets A = -b(seed(salt, position, r0)), so X = A + b is zero and the
    receipt is the same at every B. With `pouw.roots` the roots the operands open (here x's root a function of A, standing in
    for its frame-v3 root), the noise is seeded from A's own commitment: X is not zero, and the receipt moves with B."""
    r, c_, k, pos = 4, 8, 64, 0
    salt = G.window_salt(bytes.fromhex(COIN), WINDOW, BINDING)
    lay = H.TileShape(r, c_, k)
    r0 = (b"\0" * 64, b"\0" * 64)
    rows, _ = H.noise(SD.seed(salt, pos, *r0), lay)
    a = [[(-(bit & 1)) & 0xFF for bit in rows[i][0]] for i in range(r)]
    rng = random.Random(1)
    b1, b2 = ([[rng.randrange(256) for _ in range(k)] for _ in range(c_)] for _ in range(2))
    assert H.reference(a, b1, salt, pos, r0)[1] == H.reference(a, b2, salt, pos, r0)[1]
    bound = (hashlib.sha512(bytes(v for row in a for v in row)).digest(), hashlib.sha512(b"w").digest())
    noise, _ = H.noise(SD.seed(salt, pos, *bound), lay)
    assert any((a[i][t] + (noise[i][0][t] & 1)) & 0xFF for i in range(r) for t in range(k))
    assert H.reference(a, b1, salt, pos, bound)[1] != H.reference(a, b2, salt, pos, bound)[1]


def test_the_replay_witness_gets_no_second_receipt():
    """The red team's replay witness serves one A and B twice at one position under one salt, and the second receipt is a
    copy of the first. With the verifier's positions and salts, the second serving is a later record of the log (its
    positions after the first's population), another window (its own salt), or the same window in another stream or for
    another prover (whose log starts its positions over, but whose salt is bound to it): each time another seed, a fresh
    receipt."""
    rng = random.Random(3)
    a = [[rng.randrange(256) for _ in range(64)] for _ in range(4)]
    b = [[rng.randrange(256) for _ in range(64)] for _ in range(8)]
    roots = (b"\x0a" * 64, b"\x0b" * 64)
    coin = bytes.fromhex(COIN)
    first = {"population": 3, "window": WINDOW}
    pos = [int.from_bytes(G._source_bytes(first, "pouw.positions", "position", "1", (), log, BINDING), "big")
           for log in ([], [first])]
    assert pos == [1, 4]
    others = [{**BINDING, "stream": hashlib.sha512(b"another stream").hexdigest()}, {**BINDING, "prover": "b0" * 32}]
    salts = [G.window_salt(coin, w, BINDING) for w in (WINDOW, {**WINDOW, "row": 2})]
    salts += [G.window_salt(coin, {**WINDOW, **o}, o) for o in others]
    once = H.reference(a, b, salts[0], pos[0], roots)[1]
    assert H.reference(a, b, salts[0], pos[0], roots)[1] == once
    assert H.reference(a, b, salts[0], pos[1], roots)[1] != once
    for salt in salts[1:]:
        assert H.reference(a, b, salt, pos[0], roots)[1] != once
