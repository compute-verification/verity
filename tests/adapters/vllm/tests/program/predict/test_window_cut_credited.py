"""`window_cut.credited`: one credited tile, the σ core, as a Program whose one block `restrict` gives with its bit ports,
evaluated on bits against the σ's word view and the scheme, and counted as the sum of its gates; `credited_sizes` of the
cut class's calls."""
from __future__ import annotations

import functools
import random
from collections import Counter

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits import count as CT
from verity.circuits import restrict as R
from verity.circuits.boolean.trace import word_of
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.evaluation import evaluate, evaluate_bits
from verity.circuits.ops.boolean.versions import block_circuit, boolean_table
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound, pc8
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as T
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_device import DEVICES

from tests.program.predict.test_window_circuit import CLASS, GRAIN
from tests.program.predict.test_window_circuit_pearl import PEARL, SCHEME
from tests.program.predict.test_window_rows import _registered
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_cut as WK

K, TM, TN, N = 256, 1, 1, 1
D = 3


@functools.cache
def credited(depth: int | None = None):
    program, part = WK.credited(SCHEME, K, TM, TN, N, depth)
    c = b"".join(descriptor_chunks(program))
    return program, part, c, R.restrict(c, part, [(0, 1)])


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def _tile(seed: str, rho: int, rhor: int) -> tuple[list, tuple[bytes | None, int]]:
    rng, dev = random.Random(seed), "h100"
    def f32() -> list[int]:
        return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) for _ in range(K)]

    a, b = [f32() for _ in range(TM)], [f32() for _ in range(TN)]
    salt, root, unit, rows, cols = rng.randbytes(32), rng.randbytes(64), 7, list(range(TM)), list(range(TN))
    scheme = C.PearlC(k_min=K, tile=(TM, TN), forming="v1", device=DEVICES[dev], hashing="h3")
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(TM, K, TN), 0), bytes(64), root)
    ops = scheme.encode(ctx, scheme.noise(ctx), [C.f32_row(x) for x in a], [C.f32_row(x) for x in b], C.Tile(tuple(rows), tuple(cols)))
    checked, seed_b = scheme.checked(ctx, ops), scheme.noise(ctx).seed_b
    shape = pc8.Shape(TM, K, TN, tile=(TM, TN), dev=dev)
    _d, ng = inert_bound._promoted(dev, K)
    cert = [[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]
             for j in range(TN)] for i in range(TM)]
    y = [[CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) << 16 for j in range(TN)] for i in range(TM)]
    args = [a, b, y, cert, salt, root, unit, rows, 0, [list(ops.e_b[j]) for j in range(TN)],
            [pc8.basis_fa(shape, seed_b)[t * 32:t * 32 + 32] for t in range(K)],
            [pc8.basis_fb(shape, seed_b)[t * 32:t * 32 + 32] for t in range(K)],
            [pc8.gram_g(shape, seed_b)[p * 32:p * 32 + 32] for p in range(32)]]
    return args, (a, b, y, cert, salt, root, unit, rows, cols, rho, rhor)


def _vectors() -> list[tuple[list, tuple]]:
    core = WK.credited_product(SCHEME, K, TM, TN, N)
    rho, rhor = core.bindings["RHO"], core.bindings["RHOR"]
    args, ref = _tile("credited", rho, rhor)
    out = [(args, ref)]
    for bit in (31, 16, 0):
        y = [[w ^ (1 << bit) for w in row] for row in args[2]]
        out.append(([*args[:2], y, *args[3:]], (*ref[:2], y, *ref[3:])))
    return out


def _bits(args: list) -> list[int]:
    params, _ret = WK.credited_product(SCHEME, K, TM, TN, N).signature()
    out = []
    for a, (_p, t) in zip(args, params):
        words, widths = _flat(a), BL.elements(t)
        assert len(words) == len(widths)
        out += [(w >> j) & 1 for w, wd in zip(words, widths) for j in range(wd)]
    return out


def _reference(ref: tuple) -> list[int]:
    a, b, y, cert, salt, root, unit, rows, cols, rho, rhor = ref
    digest, ok = T.reference("h100", K, N, a, b, y, cert, salt, root, unit, rows, cols, 0, rho, rhor)
    return [*(digest or bytes(64)), ok]


@pytest.mark.slow
def test_the_credited_block_is_the_tile_with_its_bit_ports():
    """The block of the one unit is the σ core with every input bit one registered port and its outputs the digest's bits
    and ok. Slow: building the tile's Program and restricting it take about a minute."""
    _program, part, _c, b = credited()
    core = WK.credited_product(SCHEME, K, TM, TN, N)
    assert part["blocks"] == [[0, 1]] and part["query"]["params"]["template"] == descriptor_id(core)
    params, ret = b.program.fn.signature()
    width = 32 * (TM * K + TN * K + TM * N + 2 * TM * TN + 2 + TM + 1024) + 8 * (96 + 32 * (TN + 2 * K))
    assert [(t.n, t.elem.w) for _p, t in params] == [(width, 1)]
    assert {R._Gates(b.D).binding(g) for g in b.inputs} == {R.REGISTRATION}
    assert [(t.n, t.elem.w) for t in ret.fields] == [(64 * 8 + 1, 1)]
    assert [n.fn.id for n in b.program.root.body.nodes if n.form == "call"] == [core.id]


@pytest.mark.slow
def test_the_credited_block_with_its_leaf_is_the_tile_unit_with_one_output():
    """At a depth, the block is `PearlPlusTile_v1`: the core's ports with the leaf's hsalt and path after its first four and
    troot, tdomain and tpos last, all registered bits, and ok its one output. Slow: as above."""
    _program, part, _c, b = credited(D)
    tile = WK.credited_product(SCHEME, K, TM, TN, N, D)
    assert tile.defn is T.PearlPlusTile and part["query"]["params"]["template"] == descriptor_id(tile)
    params, ret = b.program.fn.signature()
    core, _ = WK.credited_product(SCHEME, K, TM, TN, N).signature()
    leaf = 8 * (192 + 64 * D + 2 * 64) + D
    assert [(t.n, t.elem.w) for _p, t in params] == [(sum(t.leaves for _n, t in core) + leaf, 1)]
    assert {R._Gates(b.D).binding(g) for g in b.inputs} == {R.REGISTRATION}
    assert [(t.n, t.elem.w) for t in ret.fields] == [(1, 1)]
    assert [n.fn.id for n in b.program.root.body.nodes if n.form == "call"] == [tile.id]


@pytest.mark.slow
def test_the_blocks_count_is_the_sum_of_its_bodies():
    """`count` of the block's root is each word primitive's body's count times its gates in the block's layout, and
    `count.of_block`'s. Slow: as above."""
    _program, part, c, b = credited()
    got = CT.count(b.program)
    layout = Counter(descriptor_id(g.prim) for g in b.program.circuit.gates())
    root = got[descriptor_id(b.program.root)]
    assert root.ands > 0
    assert root == sum((got[p] * m for p, m in layout.items()), CT.ZERO) == CT.of_block(c, part, [(0, 1)])


@pytest.mark.slow
def test_the_block_on_bits_is_its_word_view_and_the_scheme():
    """The block on bits computes what the σ core's word view computes on words, and that is
    `pearl_plus_tile.reference`: on an honest tile (ok 1) and with y's word flipped at its sign, at the lowest bit of its
    BF16 word and below it (ok 0). The block returns out(S) in C's gate order, so each of the core's returned bits is read
    at its gate's place. Slow: the block on bits is tens of millions of gates, the only check that the gates the count
    reads compute the tile."""
    program, _part, _c, b = credited()
    at = {g: j for j, g in enumerate(b.outputs)}
    place = [at[g] for g in program.circuit.root_outputs()]
    vectors = _vectors()
    want = [_reference(ref) for _args, ref in vectors]
    assert want[0][-1] == 1 and [w[-1] for w in want[1:]] == [0, 0, 0]
    word = word_of(WK.credited_product(SCHEME, K, TM, TN, N))
    assert [list(evaluate(word, *(_flat(a) for a in args))) for args, _ref in vectors] == want
    got = evaluate_bits(block_circuit(b.program.fn, boolean_table(b.program)),
                        np.asarray([_bits(args) for args, _ref in vectors], dtype=np.uint8))
    got = [[g[j] for j in place] for g in got.reshape(len(vectors), -1).tolist()]
    assert [[sum(g[8 * i + j] << j for j in range(8)) for i in range(64)] + [g[512]] for g in got] == want


def test_the_credited_sizes_are_the_cut_calls_tiles():
    """Each cut call's A rows in tiles of Pearl+'s 64 rows, here one short tile of 16 a round, and its columns in tiles of
    64: one size a call shape."""
    low = WC.lower(_registered(), CLASS, None, GRAIN, tables=PEARL)
    assert WK.credited_sizes(low) == [(64, 16, 64, 64), (64, 16, 64, 128), (64, 16, 64, 256), (128, 16, 64, 64)]
