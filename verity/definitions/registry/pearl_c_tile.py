"""Pearl-C's checked words and its served y as registered Definitions: TileRead, the chain and peel of one tile of Pearl+'s
product, built from the devices' own tensor-core steps, and the unscale that serves y from its U, each with its Boolean
version.

    PearlCTileReadCell_v1{K,DOT,G}    (a, b: Array<K, E4M3>, pa, pb: Array<64, BF16>) -> (F32, F32)       one word's C̃ and U
    PearlCTileRead_v1{K,TM,TN,DOT,G}  (ac: Array<TM, Array<K, E4M3>>, bc: Array<TN, Array<K, E4M3>>,
                                       pa: Array<TM, Array<64, BF16>>, pb: Array<TN, Array<64, BF16>>)
                                      -> (Array<TM, Array<TN, F32>>, Array<TM, Array<TN, F32>>)          a tile's C̃ and U
    PearlCUnscale_v1{TM,TN,BIAS}      (u: Array<TM, Array<TN, F32>>, alpha_a: Array<TM, F32>, alpha_b: Array<TN, F32>
                                       [, bias: Array<TN, BF16>]) -> Array<TM, Array<TN, BF16>>           the served y
    each _v2                          σ of its _v1 (`verity.definitions.boolean.sigma_lift`): the same body on bits, ``DOT``
                                      the step's Boolean Definition, read as its word

The device's chain is `verity.definitions.registry.pearl_c_forming`'s `PearlCChain_v1{K,DOT,G}`, the one Definition of it, which
each word calls.

**The order** is `verity.protocols.pouw.verifier.schemes.pearl_c_u`'s, op for op.  A′, B̃ are the formed E4M3 operands (K codes
each) and P_A,i, P_B,j the peel's 64 BF16 factors of row i and column j; every word is FP32 unless named.

~~~text
chain(a, b)   total = +0;  for each window of 32·G products, in k order (the last may be short):
                local = +0;  local = DOT(local, a[s:s+32], b[s:s+32]) for each 32-slice s of the window, in order;
                total = RN_f32(total + local)                                   (F32Add_v2: one promotion per window)
C̃_ij          chain(A′_i, B̃_j)
U_ij          acc = C̃_ij;  acc = k16(acc, P_A,i[16s:16s+16], P_B,j[16s:16s+16]) for s = 0, 1, 2, 3   (HopperBF16WgmmaDot16_v1)
y_ij          RN_bf16(RN_f32(RN_f32(U_ij · RN_f32(1/α_A,i)) · RN_f32(1/α_B,j)))      (F32Div_v2, F32Mul_v2 twice, F2fpBf16_v1)
              with BIAS, then Bf16Add_v1(y_ij, bias_j)
~~~

``DOT`` is the device's E4M3 k32 step: `HopperE4m3QgmmaDot32_v1` (``HOPPER_E4M3_K32``, Pearl+'s, the ``pearl-c-h100`` records)
or `BlackwellE4m3QmmaDot32_v1` (the sm_120 records); ``G`` is the device's promotion period, 4 on both.  The peel's step is
`HopperBF16WgmmaDot16_v1` (``HOPPER_BF16_WGMMA_K16``) on every record, as in `pearl_c_device`.  No −0 arises: each step's zero
is +0 and the total starts at +0, so the reference's −0 → +0 after a promotion never fires.  K is a multiple of 32 and G at
least 1 (the chain's own G = 0, the unpromoted record, is refused here); the scheme's own domain (128 | K) is narrower.

**The served y** is the Pearl-C kernels' epilogue (`store_y` in ``benchmarks/pouw/pearl_c_sm90`` and ``pearl_c_sm120``): each
output word ``__float2bfloat16_rn(__fmul_rn(__fmul_rn(u, ia_i), ib_j))``, where ia = ``__frcp_rn(α)``, a correctly rounded
reciprocal (RN_f32(1/α), F32Div_v2 of 1.0), is what the stats kernel writes for each row of A and of the weight.  Its final
BF16 store rounds to nearest even (NaN 0x7FFF, F2FP's).  The vLLM integration takes the first m of the call's 16-aligned rows
(the rest are zero filler rows), and adds a bias in BF16 after it, torch's ``y + bias`` (Bf16Add_v1, NaN 0x7FC0); ``BIAS``
is that add.  α is `pearl_c.s5_stats`' (the rows' FP32 words), which `Pc8RowA` and `Pc8RowB` compute and don't output.

`verity.protocols.pouw.verifier.circuit.pc8`'s `Pc8Tile` is the same function as TileRead at the same device: core can't read
this registry, nor this registry core, so the two are separate bodies, and core's tests hold them equal and to `pearl_c_u`
(``verity/protocols/pouw/verifier/tests/test_pearl_c_tile.py``).
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.definitions.ops.fp32 import Bf16Add, F32DivV2, F32MulV2
from verity.definitions.ops.prims import F2fpBf16, HopperBF16WgmmaDot16, const
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.boolean import sigma_lift
from verity.definitions.registry.pearl_c_forming import PearlCChain

__all__ = ["PearlCTileRead", "PearlCTileReadCell", "PearlCTileReadCellV2", "PearlCTileReadV2", "PearlCUnscale",
           "PearlCUnscaleV2"]

E4M3, BF16, F32 = Value(8), Value(16), Value(32)
SLICE, PEEL, K16 = 32, 64, 16
ONE_BITS = 0x3F800000
CONFORMANCE = ("protocol-defined; equal to verity.protocols.pouw.verifier.schemes.pearl_c_u, the pin of the Hopper kernel's gate "
               "problem, on its vectors (verity/protocols/pouw/verifier/tests/test_pearl_c_tile.py)")
UNSCALE_CONFORMANCE = ("protocol-defined; the Pearl-C kernels' store_y, the order their check fixtures' glue_y states "
                       "(verity/protocols/pouw/verifier/tests/test_pearl_c_tile.py)")


def _refuse(b: dict) -> str | None:
    k, g = b.get("K"), b.get("G")
    if k is not None and (not isinstance(k, int) or k < SLICE or k % SLICE):
        return f"the chain multiplies in atoms of k = 32: K = {k!r} is no positive multiple of 32"
    if g is not None and (not isinstance(g, int) or g < 1):
        return f"G is the atoms per promotion, at least 1: G = {g!r}"
    for name in ("TM", "TN"):
        if (v := b.get(name)) is not None and (not isinstance(v, int) or v < 1):
            return f"{name} is a tile's {'rows' if name == 'TM' else 'columns'}, at least 1: {v!r}"
    return None


def _cell_sig(S):
    return (("a", Array(S.K, E4M3)), ("b", Array(S.K, E4M3)), ("pa", Array(PEEL, BF16)), ("pb", Array(PEEL, BF16))), \
        Tuple(F32, F32)


@composite("PearlCTileReadCell", 1, ["K", "DOT", "G"], _cell_sig, conformance=CONFORMANCE, refuse=_refuse)
def PearlCTileReadCell(B, S, a, b, pa, pb):
    """One word's checked pair (C̃, U): C̃ the chain over A′_i·B̃_j, U four BF16 k16 steps from C̃ over P_A,i·P_B,j
    (`pearl_c_u.u`)."""
    c = B.call(bind(PearlCChain, K=S.K, DOT=S.DOT, G=S.G), a, b)
    u = c
    for s in range(0, PEEL, K16):
        u = B.call(HopperBF16WgmmaDot16, u, pa[s:s + K16], pb[s:s + K16])
    return tuple_of(c, u)


def _tile_sig(S):
    k, tm, tn = S.K, S.TM, S.TN
    return ((("ac", Array(tm, Array(k, E4M3))), ("bc", Array(tn, Array(k, E4M3))), ("pa", Array(tm, Array(PEEL, BF16))),
             ("pb", Array(tn, Array(PEEL, BF16)))), Tuple(Array(tm, Array(tn, F32)), Array(tm, Array(tn, F32))))


@composite("PearlCTileRead", 1, ["K", "TM", "TN", "DOT", "G"], _tile_sig, conformance=CONFORMANCE, refuse=_refuse)
def PearlCTileRead(B, S, ac, bc, pa, pb):
    """TileRead: a TM × TN tile's checked words, C̃ and U by row then column, each word `PearlCTileReadCell` over its row of A
    (A′_i, P_A,i) and its row of the weight (B̃_j, P_B,j)."""
    cell = bind(PearlCTileReadCell, K=S.K, DOT=S.DOT, G=S.G)
    cs, us = [], []
    for i in range(S.TM):
        crow, urow = [], []
        for j in range(S.TN):
            out = B.call(cell, ac[i], bc[j], pa[i], pb[j])
            crow.append(out[0])
            urow.append(out[1])
        cs.append(array_of(crow))
        us.append(array_of(urow))
    return tuple_of(array_of(cs), array_of(us))


def _unscale_sig(S):
    tm, tn = S.TM, S.TN
    params = (("u", Array(tm, Array(tn, F32))), ("alpha_a", Array(tm, F32)), ("alpha_b", Array(tn, F32)))
    return params + ((("bias", Array(tn, BF16)),) if S.BIAS else ()), Array(tm, Array(tn, BF16))


@composite("PearlCUnscale", 1, ["TM", "TN", "BIAS"], _unscale_sig, conformance=UNSCALE_CONFORMANCE, refuse=_refuse)
def PearlCUnscale(B, S, u, alpha_a, alpha_b, *bias):
    """The served y of a TM × TN tile from its U: each row's and column's reciprocal RN_f32(1/α) once, then each word
    RN_bf16(RN_f32(RN_f32(U_ij · ia_i) · ib_j)), and with ``BIAS`` torch's BF16 add of bias_j after it."""
    one = B.call(const(32, ONE_BITS))
    ia = [B.call(F32DivV2, one, alpha_a[i]) for i in range(S.TM)]
    ib = [B.call(F32DivV2, one, alpha_b[j]) for j in range(S.TN)]
    rows = []
    for i in range(S.TM):
        row = []
        for j in range(S.TN):
            y = B.call(F2fpBf16, B.call(F32MulV2, B.call(F32MulV2, u[i][j], ia[i]), ib[j]))
            row.append(B.call(Bf16Add, y, bias[0][j]) if S.BIAS else y)
        rows.append(array_of(row))
    return array_of(rows)


PearlCTileReadCellV2 = sigma_lift(PearlCTileReadCell, 2)
PearlCTileReadV2 = sigma_lift(PearlCTileRead, 2)
PearlCUnscaleV2 = sigma_lift(PearlCUnscale, 2)
