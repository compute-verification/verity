"""A credited linear of C as Pearl+'s product (`pearl_plus_credited`): its grid's units of `PearlPlusTileCoreWords_v5`,
each a top level of maps over its noise's lines and Gram rows and one wired core, the units C's laying makes of it, each
unit the scheme's block whose amount is its credited MACs, each unit's public row, the noise the unit derives inside, and
the unit on C's words beside the core fed that noise. That one drawn unit goes through the door is
``benchmarks/pouw/credited_in_c.py``'s run."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.protocols.accounting.compute.pouw import scheme as PS
from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import evaluate, evaluate_batch
from verity.circuits.evaluation.bits import evaluate_words
from verity.circuits.types import Value
from verity.protocols.accounting.compute.pouw.prover.circuit import inert_bound
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_u as CU

SCHEME = "pearl-c-h100-v1-h3"
K = 256


def test_a_linears_units_are_its_grid_row_major_its_last_row_tile_part_filler() -> None:
    got = PC.tiles(SCHEME, K, 3, 2, tile=(2, 1))
    assert [(t.rows, t.cols, t.col, t.real) for t in got] == [
        ((0, 1), (0,), 0, 2), ((0, 1), (1,), 1, 2), ((2, 3), (0,), 0, 1), ((2, 3), (1,), 1, 1)]
    assert {t.core.defn.id for t in got} == {PC.PearlPlusTileCoreWords.id}
    assert all({s: t.core.bindings[s] for s in ("K", "TM", "TN", "N")} == {"K": K, "TM": 2, "TN": 1, "N": 2} for t in got)
    with pytest.raises(ValueError, match="whole column tiles"):
        PC.tiles(SCHEME, K, 3, 3, tile=(1, 2))


def test_each_unit_is_the_schemes_block_whose_amount_is_its_credited_macs() -> None:
    (block,) = PS.PEARL_PLUS.blocks
    for t in PC.tiles(SCHEME, K, 3, 2, tile=(2, 1)):
        assert t.core.defn.id == block.definition
        assert block.cost(t.core.bindings) == t.real * 1 * K


def test_the_nodes_top_level_is_maps_over_its_noise_and_one_wired_core_and_its_cut_is_them() -> None:
    """The node's body is one seed, which writes eb's TN lines, a map of the line over F's eight groups of K/8 for fa and
    fb, a Call of the Gram row for each of F_B's 32 columns, and the wired core, its noise crossing as values; `cut` is
    those Definitions with their counts, and their ANDs add up to the node's: nothing of the node is outside its
    units."""
    from verity.protocols.accounting.compute.pouw.prover.circuit.pearl_c_call import ands
    node = PC.credited_product(SCHEME, K, 1, 1, 2)
    top = [(n.fn, n.n if n.form == "batch" else 1) for n in node.body.nodes]
    assert [n.form for n in node.body.nodes] == ["call", "batch", "batch", *["call"] * 32, "call"]
    assert top == PC.cut(node)
    assert [u.defn.id for u, _n in top] == ["PearlPlusTileSeed_v4", *["PearlPlusTileLine_v4"] * 2,
                                             *["PearlPlusGramRow_v5"] * 32, "PearlPlusTileCoreWired_v2"]
    assert [u.bindings["P"] for u, _n in top[3:35]] == list(range(32))
    assert [n for _u, n in top] == [1, PT.F_ROWS, PT.F_ROWS, *[1] * 32, 1]
    assert [u.bindings["G"] for u, _n in top[1:3]] == [K // PT.F_ROWS, K // PT.F_ROWS]
    assert sum(n * ands(u) for u, n in top) == ands(node)


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def _f32(rng: random.Random) -> list[int]:
    return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) for _ in range(K)]


def _lines(b: bytes) -> list[list[int]]:
    return [list(b[i:i + 32]) for i in range(0, len(b), 32)]


def test_the_node_reads_its_place_and_no_noise_and_writes_ok_as_one_word() -> None:
    """The node's ports are the tile's operands, its certificates and its public row, the salt, the root and its place,
    and no noise; ok is one 16-bit word.  The row-port wrapper before it is gone."""
    node = PC.credited_product(SCHEME, K, 1, 1, 2)
    params, ret = node.signature()
    assert [p for p, _t in params] == ["a", "b", "cert", *PC.PUBLIC] and PC.PUBLIC == ("salt", "root", "unit", "rows", "col")
    assert ret.fields[2] == Value(16)
    assert "PearlPlusTileCoreWords_v5" in REGISTRY.defs and "PearlPlusTileCoreWords_v2" not in REGISTRY.defs
    assert {"PearlPlusTileSeed_v4", "PearlPlusTileLine_v4", "PearlPlusGramRow_v5", "PearlPlusTileCoreWired_v2"} <= set(REGISTRY.defs)
    assert not {"PearlPlusTileNoise_v1", "PearlPlusTileNoise_v2", "PearlPlusTileKey_v1", "PearlPlusTileSeed_v1",
                "PearlPlusGramRow_v2", "PearlPlusGramRow_v3", "PearlPlusTileLine_v2", "PearlPlusTileCoreWired_v1",
                "PearlPlusTileCoreWords_v3", "PearlPlusTileCoreWords_v4", "PearlPlusTileSeed_v2", "PearlPlusTileSeed_v3", "PearlPlusTileLine_v3",
                "PearlPlusGramRow_v4"} & \
        set(REGISTRY.defs)


def test_a_units_public_row_is_its_salt_root_and_place() -> None:
    rng = random.Random(0)
    salt, root, unit = rng.randbytes(32), rng.randbytes(64), 5
    t = PC.tiles(SCHEME, K, 2, 2, tile=(1, 1))[3]
    got = PC.public(SCHEME, K, t, unit, salt, root)
    assert tuple(got) == PC.PUBLIC
    assert got == {"salt": PC.le16(salt), "root": PC.le16(root), "unit": unit, "rows": [1], "col": 1}
    assert got["salt"][0] == salt[0] | salt[1] << 8 and len(got["root"]) == 32


def test_the_noise_a_unit_derives_is_the_noise_pearl_plus_forms_its_tile_with() -> None:
    """The node's noise units on words (`pearl_plus_tile`'s seed, line and Gram-row word bodies, wired as the node wires
    them) at column tile 1: E_B's lines and F_A, F_B and G are the ones `PearlC.encode` and `PearlC.checked` use at the
    tile, from the salt and the weights' root alone, and the reference functions' bytes."""
    rng = random.Random(0)
    salt, root, unit = rng.randbytes(32), rng.randbytes(64), 5
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(1, 1))
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(2, K, 2), 0), bytes(64), root)
    key, ebi, fi = (lambda o: (o[:32], o[32:33], o[33:]))(list(evaluate(bind(PT._SEED, K=K, TN=1), list(salt), list(root), [1])))
    line = {kind: bind(PT._LINE, **dict(zip(("SIDE", "FACTOR"), PT.LINES[kind]))) for kind in PT.LINES}
    eb = [list(evaluate(line["eb"], key, [i])) for i in ebi]
    fa, fb = ([list(evaluate(line[kind], key, [i])) for i in fi] for kind in ("fa", "fb"))
    g = [list(evaluate(bind(PT._GRAM_ROW, K=K, DEV=s.device.name, P=p), [c for x in fb for c in x],
                       [c for x in fa for c in x])) for p in range(32)]
    ops = s.encode(ctx, s.noise(ctx), [C.f32_row(_f32(rng))], [C.f32_row(_f32(rng))], C.Tile((1,), (1,)))
    seed_b = s.noise(ctx).seed_b
    assert eb == [list(ops.e_b[0])] == _lines(PC.row_eb(salt, root, 1, 1))
    assert fa == [list(x) for x in C._basis(seed_b, 0, K, C.LINE_NORM_V1)] == _lines(PC.row_fa(salt, root, K))
    assert fb == [list(x) for x in C._basis(seed_b, 1, K, C.LINE_NORM_V1)] == _lines(PC.row_fb(salt, root, K))
    assert g == [list(r) for r in C.gram(seed_b, K, C.LINE_NORM_V1, s.device)]


#: (scheme, N): one column at h100 and sm_120, and both column tiles of a two-column linear at h100
NODE_CASES = (("pearl-c-h100-v1-h3", 1), ("pearl-c-h100-v1-h3", 2), ("pearl-c-sm120-v1-h3", 1))


@pytest.fixture(scope="module")
def node_and_core():
    """Each case's tiles (1 × 1, K = 256) on an honest linear: the node's (digest, y, ok) on C's words, and the core's
    (`PearlPlusTileCore_v3`, on bits, by its word view) fed the same operands and the reference functions' noise of the
    tile's column."""
    got = []
    for scheme, n in NODE_CASES:
        rng = random.Random(f"node/{scheme}/{n}")
        salt, root, unit = rng.randbytes(32), rng.randbytes(64), 3
        s = pouw_schemes.get(scheme, k_min=K, tile=(1, 1))
        dev = s.device.name
        _d, ng = inert_bound._promoted(dev, K)
        a = [[w & 0xFFFF0000 for w in _f32(rng)]]
        w = [[v & 0xFFFF0000 for v in _f32(rng)] for _ in range(n)]
        ctx = C.Context(salt, C.Matmul(unit, C.Shape(1, K, n), 0), bytes(64), root)
        for t in PC.tiles(scheme, K, 1, n, tile=(1, 1)):
            b = [w[j] for j in t.cols]
            ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a], [C.f32_row(r) for r in b], C.Tile(t.rows, t.cols))
            checked = s.checked(ctx, ops)
            cert = [[[list(inert_bound.certificates(ops.a.codes[0], ops.b.codes[0], dev))[2 * z:2 * z + 2]
                      for z in range(ng - 1)]]]
            pub = PC.public(scheme, K, t, unit, salt, root)
            words = [a, [[v >> 16 for v in r] for r in b], cert, *(pub[p] for p in PC.PUBLIC)]
            g = evaluate_batch(t.core, *(np.asarray([_flat(x)], dtype=np.int64) for x in words)).reshape(-1).tolist()
            node = (bytes(sum(g[8 * i + j] << j for j in range(8)) for i in range(64)), g[512:513], g[513])
            gram = PC.row_gram(salt, root, K, dev)
            noise = [_lines(PC.row_eb(salt, root, t.col, 1)), _lines(PC.row_fa(salt, root, K)),
                     _lines(PC.row_fb(salt, root, K)),
                     [[int.from_bytes(gram[4 * (32 * p + q):4 * (32 * p + q) + 4], "little") for q in range(32)]
                      for p in range(32)]]
            core = PT.core(s, K, 1, 1, n)
            args = [a, b, cert, salt, root, unit, list(t.rows), *noise]
            c = evaluate_words(core, *(np.asarray([_flat(x)], dtype=np.uint64) for x in args)).reshape(-1).tolist()
            served = [CU.y(checked.u[0][0], ops.a.alpha[0], ops.b.alpha[0])]
            got.append(((scheme, n, t.col), node, (bytes(c[:64]), c[64:65], c[65]), served))
    return got


@pytest.mark.slow
def test_the_node_is_the_core_fed_the_reference_noise(node_and_core) -> None:
    """Slow: each case evaluates the node through its σ noise units and σ wired core on bits, then the core alone.  The node's digest,
    y and ok equal the core's fed the reference functions' noise, bit for bit, at both devices and both column tiles of a
    two-column linear, and on an honest tile its y is the served y and ok 1: the only check that the noise the node
    derives is the noise its core is held to through the ports C reads."""
    assert [case for case, *_ in node_and_core] == [("pearl-c-h100-v1-h3", 1, 0), ("pearl-c-h100-v1-h3", 2, 0),
                                                     ("pearl-c-h100-v1-h3", 2, 1), ("pearl-c-sm120-v1-h3", 1, 0)]
    for case, node, core, served in node_and_core:
        assert node == core, case
        assert node[1:] == (served, 1), case
