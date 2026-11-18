"""`window_cut.credited`: one credited tile as a Program whose one block `restrict` gives with its word ports, evaluated
on bits against its words and the scheme, and counted as the sum of its word primitives' bodies; `credited_sizes` of the
cut class's calls."""
from __future__ import annotations

import functools
import random
from collections import Counter

import numpy as np
import pytest
from verity.circuits import count as CT
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.evaluation import evaluate, evaluate_words
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


@functools.cache
def credited():
    program, part = WK.credited(SCHEME, K, TM, TN, N)
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
    _program, _part, _c, b = credited()
    core = b.program.root.body.nodes[-1].fn
    rho, rhor = core.bindings["RHO"], core.bindings["RHOR"]
    args, ref = _tile("credited", rho, rhor)
    out = [(args, ref)]
    for bit in (31, 16, 0):
        y = [[w ^ (1 << bit) for w in row] for row in args[2]]
        out.append(([*args[:2], y, *args[3:]], (*ref[:2], y, *ref[3:])))
    return out


def _ports(args: list) -> list[list[int]]:
    _program, _part, _c, b = credited()
    words, at, out = _flat(args), 0, []
    for _p, t in b.program.fn.signature()[0]:
        out.append(words[at:at + t.n])
        at += t.n
    assert at == len(words)
    return out


def _reference(ref: tuple) -> list[int]:
    a, b, y, cert, salt, root, unit, rows, cols, rho, rhor = ref
    digest, ok = T.reference("h100", K, N, a, b, y, cert, salt, root, unit, rows, cols, 0, rho, rhor)
    return [*(digest or bytes(64)), ok]


@pytest.mark.slow
def test_the_credited_block_is_the_tile_with_its_word_ports():
    """The block of the one unit is the tile's Definition with every input a registered port at its word width, and its
    outputs the digest's bytes and ok. Slow: building the tile's Program and restricting it take about a minute."""
    program, part, c, b = credited()
    assert part["blocks"] == [[0, 1]] and part["query"]["params"]["template"] == descriptor_id(WK.credited_product(SCHEME, K, TM, TN, N))
    params, ret = b.program.fn.signature()
    assert [(t.n, t.elem.w) for _p, t in params] == [(TM * K + TN * K + TM * N + 2 * TM * TN, 32), (96, 8), (2 + TM, 32),
                                                     (32 * (TN + 2 * K), 8), (1024, 32)]
    assert {R._binding(b.bits, g) for g in b.inputs} == {R.REGISTRATION}
    assert [(t.n, t.elem.w) for t in ret.fields] == [(64, 8), (1, 1)]
    assert [n.fn.id for n in b.program.root.body.nodes if n.form == "call"] == [WK.credited_product(SCHEME, K, TM, TN, N).id]


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
def test_the_block_on_bits_is_its_words_and_the_scheme():
    """The block on bits (each word primitive its body in the block's Boolean map) computes what the tile's Program
    computes on words, and that is `pearl_plus_tile.reference`: on an honest tile (ok 1) and with y's word flipped at its sign, at the
    lowest bit of its BF16 word and below it (ok 0). Slow: the block on bits is hundreds of millions of gates, the only
    check that the bodies the count reads compute the tile."""
    program, _part, _c, b = credited()
    vectors = _vectors()
    words = [_ports(args) for args, _ref in vectors]
    want = [_reference(ref) for _args, ref in vectors]
    assert want[0][-1] == 1 and [w[-1] for w in want[1:]] == [0, 0, 0]
    assert [list(evaluate(program.fn, *(_flat(a) for a in args))) for args, _ref in vectors] == want
    bits = block_circuit(b.program.fn, boolean_table(b.program))
    got = evaluate_words(bits, *(np.asarray([w[i] for w in words], dtype=np.uint64) for i in range(len(words[0]))))
    assert got.reshape(len(vectors), -1).tolist() == want


def test_the_credited_sizes_are_the_cut_calls_tiles():
    """Each cut call's A rows in tiles of Pearl+'s 64 rows, here one short tile of 16 a round, and its columns in tiles of
    64: one size a call shape."""
    low = WC.lower(_registered(), CLASS, None, GRAIN, tables=PEARL)
    assert WK.credited_sizes(low) == [(64, 16, 64, 64), (64, 16, 64, 128), (64, 16, 64, 256), (128, 16, 64, 64)]
