"""Roofline of the fused degree-3 sumcheck round on an RTX 4090 (``ROOFLINE.md`` is generated from this).

The fused kernel with *input* ``n`` extension elements per table (three tables ``f, g, h`` over BabyBear^6, 24 B per
element as 6 x u32) folds pairs with the previous challenge ``(1 - r) x_e + r x_o`` (``n -> n/2`` elements), evaluates
the degree-3 round polynomial ``S(0..3)`` over the ``n/4`` pairs of folded elements, and writes the ``n/2`` folded
elements per table.  Ideal traffic: every input byte read once, every folded byte written once, 4 x 6 sums emitted.

Arithmetic per kernel (extension x extension products):

* fold: ``x_e + r (x_o - x_e)`` = 1 product per folded element, ``3 n/2`` products
* S: ``f_k(x) g_k(x) h_k(x)`` at 4 points = 2 products per point per pair, ``8 n/4 = 2 n`` products

so ``3.5 n`` BabyBear^6 products per kernel.  Costs of one product on the two engines (stated assumptions):

* INT8 tensor cores: 24 x 24 = 576 limb MACs (TensorZKP layout, 4 limbs per coefficient); the fold as a 48 x 48
  convolution-matrix product is 2304 MACs per folded element (2x redundant limb positions) -- both counted below.
* SIMT: schoolbook 36 base products + 5 wrap products (``X^6 = W``), each a 32-bit Montgomery multiply
  (mul.lo, mul.hi, mul.lo, mul.hi, sub, compare+select = 7 INT32 ops) and 35 modular adds (3 ops): ~392 INT32 ops.
  Karatsuba (18 + 5 products) ~280 ops.
"""
from __future__ import annotations

HBM = 1008e9                 # RTX 4090 GDDR6X, bytes/s (Ada datasheet)
TC_INT8_MACS = 330.3e12      # 660.6 TOPS dense INT8 / 2
SIMT_INT32_OPS = 20.6e12     # 128 SMs x 64 INT32 lanes x 2.52 GHz (Ada whitepaper: 64 INT32 units per SM; IMAD 1/clk/lane)
FP32_FLOPS = 82.6e12

ELEM_BYTES = 24
MACS_PER_EXT_MUL = 576
MACS_PER_TC_FOLD = 2304      # [x_e | x_o] (48 limbs) x C (48 x 48) per folded element
SIMT_OPS_SCHOOLBOOK = 41 * 7 + 35 * 3
SIMT_OPS_KARATSUBA = 23 * 7 + 60 * 3


def kernel(n: int, in_bytes: int = ELEM_BYTES, first_round: bool = False, fold_only: bool = False) -> dict:
    """Ideal bytes and work of the fused kernel with ``n`` input elements per table."""
    read = 3 * n * in_bytes
    write = 0 if first_round else 3 * (n // 2) * ELEM_BYTES
    if first_round:                       # S over n base elements, no fold
        ext_muls, tc_macs, simt_ops = 0, 8 * (n // 2) * 16, 8 * (n // 2) * 10
    else:
        fold_muls = 3 * (n // 2)
        s_muls = 0 if fold_only else 8 * (n // 4)
        ext_muls = fold_muls + s_muls
        if in_bytes == 4:                 # base-field input: the fold is base x ext (6 products), 8 x 48 limb MACs
            tc_macs = fold_muls * 8 * 48 + s_muls * MACS_PER_EXT_MUL
            simt_ops = fold_muls * 6 * 7 + s_muls * SIMT_OPS_SCHOOLBOOK
        else:
            tc_macs = fold_muls * MACS_PER_TC_FOLD + s_muls * MACS_PER_EXT_MUL
            simt_ops = ext_muls * SIMT_OPS_SCHOOLBOOK
    b = read + write
    return {"n": n, "read_B": read, "write_B": write, "bytes": b, "t_hbm_us": b / HBM * 1e6,
            "ext_muls": ext_muls, "tc_macs": tc_macs, "t_tc_us": tc_macs / TC_INT8_MACS * 1e6,
            "simt_ops": simt_ops, "t_simt_us": simt_ops / SIMT_INT32_OPS * 1e6,
            "t_simt_karatsuba_us": ext_muls * SIMT_OPS_KARATSUBA / SIMT_INT32_OPS * 1e6}


def table(sizes=(2**22, 2**21, 2**20, 2**18, 2**16, 2**14, 4096)) -> str:
    rows = ["| n (input elems) | HBM bytes | t_HBM us | ext muls | TC MACs | t_TC us | SIMT ops | t_SIMT us | t_SIMT(Karatsuba) us | floor |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n in sizes:
        k = kernel(n)
        floor = max(k["t_hbm_us"], k["t_tc_us"])
        rows.append(f"| 2^{n.bit_length()-1} = {n} | {k['bytes']/1e6:.1f} MB | {k['t_hbm_us']:.1f} | {k['ext_muls']:.3g} | {k['tc_macs']:.3g} | "
                    f"{k['t_tc_us']:.1f} | {k['simt_ops']:.3g} | {k['t_simt_us']:.1f} | {k['t_simt_karatsuba_us']:.1f} | {floor:.1f} (HBM) |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
    k = kernel(2**21)
    print(f"\nn = 2^21: HBM floor {k['t_hbm_us']:.1f} us; SIMT needs {k['t_simt_us']/k['t_hbm_us']:.0%} of IMAD peak to hide under HBM "
          f"({k['t_simt_karatsuba_us']/k['t_hbm_us']:.0%} with Karatsuba); TC needs {k['t_tc_us']/k['t_hbm_us']:.1%} of INT8 peak.")
    print("first round (base x base x base, S only, n = 2^22):", kernel(2**22, in_bytes=4, first_round=True))
    print("second round (base in, fold to ext + S, n = 2^22):", kernel(2**22, in_bytes=4))
