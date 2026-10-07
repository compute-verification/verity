"""The v2 (public-selection) units against the v1 units and the instruction model.

* census: every v2 chain unit has at most half the rows of its v1 counterpart;
* differential (``RELMIN_DIFF_N`` units per relation, default 3000; the lane ran 1e5): on random units inside the domain
  (random and chained accumulators, zeros, subnormals) v1 and v2 accept the honest claim and both reject every wrong claim of
  the negative families -- every bit of the claimed word flipped, +-1 ulp, the other pipeline's word where it differs -- and
  agree column by column; the public decodes refuse the same off-domain operands;
* the chain runner: two VUs proved and verified on CPU (interactive and Fiat-Shamir + ZK), the negative battery rejected;
* the torch hint generator equals the Python integer reference.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32
from verity.ml.circuits.silicon.silicon import tc_dot

from .. import protocol as protocol_mod
from ..compile import P
from ..fp8.census import census
from ..relations import RELATIONS, Relation
from ..relchain import RelationChainRunner, instances, negatives
from ..witness import apply_sparse, build_witness, tables
from .hints import reference_hints_v2
from .relation import plan

V2 = [RELATIONS[n] for n in ("bf16-hopper-v2", "bf16-ampere-v2", "fp8-hopper-v2", "fp8-ada-v2")]
V1_OF = {"bf16-hopper-v2": RELATIONS["bf16-hopper"], "fp8-hopper-v2": RELATIONS["fp8-hopper"], "fp8-ada-v2": RELATIONS["fp8-ada"],
         "bf16-ampere-v2": None}
OTHER = {"bf16-hopper-v2": AMPERE_BF16_M16N8K16, "bf16-ampere-v2": HOPPER_BF16_M16N8K16, "fp8-hopper-v2": ADA_E4M3_M16N8K32,
         "fp8-ada-v2": HOPPER_E4M3_K32}
N_DIFF = int(os.environ.get("RELMIN_DIFF_N", "3000"))
BATCH = 4096


def _v1_unit(rel_v2: Relation):
    """The v1 counterpart's unit-mode system (bf16-ampere-v2 has none registered on the generic runner: the Ampere BF16 unit
    is ``unit.compile_unit`` at REAL)."""
    v1 = V1_OF[rel_v2.name]
    if v1 is not None:
        return v1, v1.compile(v1.params, False), v1.hints, v1.public_vectors
    from ..hints_torch import hints_torch
    from ..unit import compile_unit, public_vectors
    return None, compile_unit(rel_v2.params, chain=False), (lambda p, s, c, a, b, y, d: hints_torch(p, s, c, a, b, None, d)), \
        (lambda p, c, a, b, y, device="cpu": public_vectors(p, c, a, b, y, device=device))


def _finite_fp32(rng: np.random.Generator, n: int) -> np.ndarray:
    kind = rng.integers(0, 16, size=n)
    sign = rng.integers(0, 2, size=n, dtype=np.int64) << 31
    t = rng.integers(40, 220, size=n, dtype=np.int64)
    f = rng.integers(0, 1 << 23, size=n, dtype=np.int64)
    c = sign | (t << 23) | f
    c = np.where(kind == 0, sign, c)
    c = np.where(kind == 1, sign | f, c)
    return c


def _cases(rel: Relation, n: int, seed: int):
    """(c, a, b, y) inside the domain: accumulators random finite FP32, zeros, subnormals, or the model's own state after a
    few chained steps (redrawn when the step leaves the finite range)."""
    rng = np.random.default_rng(seed)
    k = rel.k
    a = rel.operands(rng, (n, k))
    b = rel.operands(rng, (n, k))
    c = _finite_fp32(rng, n)
    chained = rng.integers(0, 3, size=n) == 0
    for i in np.nonzero(chained)[0]:
        cc = 0
        for _ in range(int(rng.integers(1, 4))):
            cc = tc_dot(rel.model, cc, rel.operands(rng, (k,)).tolist(), rel.operands(rng, (k,)).tolist())
        c[i] = cc
    # zero sums (one unit in 16): the products cancel pairwise (a_j b_j = -a_i b_i) on a zero accumulator, all operands zero
    # on a zero accumulator (E = floor), or the accumulator cancels against a single product
    zero = rng.integers(0, 16, size=n) == 0
    kind = rng.integers(0, 3, size=n)
    sign_bit = 1 << (plan(rel.params).op_word_bits - 1)
    for i in np.nonzero(zero)[0]:
        if kind[i] == 0:
            a[i, 1::2], b[i, 1::2] = a[i, 0::2], b[i, 0::2] ^ sign_bit
            c[i] = 0
        elif kind[i] == 1:
            a[i], b[i], c[i] = 0, 0, 0
        else:
            a[i, 1:], b[i, 1:] = 0, 0
            c[i] = tc_dot(rel.model, 0, a[i].tolist(), (b[i] ^ sign_bit).tolist())
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while True:
            yy = tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist())
            if ((yy >> 23) & 0xFF) != 0xFF:
                break
            c[i] = _finite_fp32(rng, 1)[0]
        y[i] = yy
    return c, a, b, y


def _accepts(sys, tb, W: torch.Tensor, pub: dict) -> torch.Tensor:
    """Per column: every linear and quadratic constraint holds."""
    lin = apply_sparse((tb.lin_q, tb.lin_j, tb.lin_coef), tb.lin_c, tb.L, W)
    ok = torch.ones(W.shape[1], dtype=torch.bool)
    for q, name in enumerate(tb.lin_pub):
        rhs = pub[name] % P if name is not None else torch.zeros_like(lin[q])
        ok &= lin[q] == rhs
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, W)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, W)
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, W)
    ok &= ~(((A * B) % P != C).any(dim=0))
    return ok


def _witness_accepts(sys, tb, hints_fn, pub_fn, p, c, a, b, y_claim):
    """Honest witness (hints from the true unit) against the claim ``y_claim``: the acceptance mask."""
    pub = pub_fn(p, c, a, b, y_claim, device="cpu")
    hints = hints_fn(p, sys, c, a, b, None, "cpu")
    W = build_witness(sys, pub, hints, "cpu")
    return _accepts(sys, tb, W, pub)


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_census_at_most_half_of_v1(rel):
    s2 = rel.compile(rel.params, True)
    _, s1u, _, _ = _v1_unit(rel)
    v1 = V1_OF[rel.name]
    if v1 is not None:
        s1 = v1.compile(v1.params, True)
    else:
        from ..unit import compile_unit
        s1 = compile_unit(rel.params, chain=True)
    c1, c2 = census(s1), census(s2)
    assert 2 * c2["rows_per_unit"] <= c1["rows_per_unit"], (c1, c2)
    assert c2["rows_per_unit"] <= {"bf16-hopper-v2": 292, "bf16-ampere-v2": 469, "fp8-hopper-v2": 195, "fp8-ada-v2": 357}[rel.name]


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_differential_v1_v2(rel):
    p = rel.params
    pl = plan(p)
    s2 = rel.compile(p, False)
    tb2 = tables(s2, "cpu")
    _, s1, h1, pv1 = _v1_unit(rel)
    tb1 = tables(s1, "cpu")
    other = OTHER[rel.name]
    ybits = 32
    n_pos = n_neg = 0
    for lo in range(0, N_DIFF, BATCH):
        n = min(BATCH, N_DIFF - lo)
        c, a, b, y = _cases(rel, n, seed=1000 + lo)
        claims = [("honest", y)]
        for bit in range(ybits):
            if pl.fp8 and bit < pl.drop:
                continue                                     # the FP8 claim is the 22-bit packed word: low bits are not claimed
            claims.append((f"bit {bit}", y ^ (1 << bit)))
        claims.append(("+1 ulp", (y + 2**pl.drop) % 2**32))
        claims.append(("-1 ulp", (y - 2**pl.drop) % 2**32))
        yo = np.array([tc_dot(other, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(n)], dtype=np.int64)
        claims.append(("other pipeline", yo))
        for name, yc in claims:
            fin = ((yc >> 23) & 0xFF) != 0xFF                 # the verifier's decode refuses non-finite claims: not a witness question
            if pl.fp8:
                fin &= (yc & (2**pl.drop - 1)) == 0
            idx = np.nonzero(fin)[0]
            if len(idx) == 0:
                continue
            cc, aa, bb, yy = c[idx], a[idx], b[idx], yc[idx]
            ok2 = _witness_accepts(s2, tb2, rel.hints, rel.public_vectors, p, cc, aa, bb, yy)
            ok1 = _witness_accepts(s1, tb1, h1, pv1, p, cc, aa, bb, yy)
            truth = torch.as_tensor(yy == y[idx])
            assert torch.equal(ok2, truth), f"{rel.name} {name}: v2 accepts != (claim == model) at {torch.nonzero(ok2 != truth)[:5].flatten().tolist()}"
            assert torch.equal(ok1, truth), f"{rel.name} {name}: v1 accepts != (claim == model)"
            n_pos += int(truth.sum())
            n_neg += int((~truth).sum())
    assert n_pos >= N_DIFF and n_neg > 10 * N_DIFF


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_off_domain_operands_refused_by_both_decodes(rel):
    p = rel.params
    _, _, _, pv1 = _v1_unit(rel)
    rng = np.random.default_rng(3)
    a = rel.operands(rng, (4, rel.k))
    b = rel.operands(rng, (4, rel.k))
    for w in rel.off_domain:
        bad = a.copy()
        bad[1, 2] = w
        with pytest.raises(ValueError):
            rel.public_vectors(p, None, bad, b, None)
        with pytest.raises(ValueError):
            pv1(p, None, bad, b, None)


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_reference_hints_equal_torch(rel):
    p = rel.params
    s = rel.compile(p, True)
    vus = instances(rel, 1, seed=7)
    from ..relchain import vu_units
    units = vu_units(rel, *vus[0][:3])
    c = np.asarray([u[0] for u in units]); a = np.asarray([u[1] for u in units]); b = np.asarray([u[2] for u in units])
    y = np.asarray([u[3] for u in units])
    assert torch.equal(reference_hints_v2(p, s, units, "cpu"), rel.hints(p, s, c, a, b, y, "cpu"))


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_chain_two_vus_cpu(rel, zk, mode):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=zk, mode=mode)
    vus = instances(rel, 2, seed=5)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    a, b, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_negative_battery_rejected(rel):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, instances(rel, 2, seed=5), row_negatives=40)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
    assert len(negs) >= rel.y_bits + 2 + len(rel.off_domain) + 1 + 1 + 4 + 40


@pytest.mark.parametrize("rel", [RELATIONS["bf16-hopper-v2"], RELATIONS["bf16-ampere-v2"]], ids=lambda r: r.name)
def test_bf16_epilogue_rounds_to_nearest_even(rel):
    """The chain-mode RN-even epilogue (half AND NOT sticky-zero, no remainder rows) on accumulators with every rounding
    pattern in the dropped 16 bits -- ties both ways, just above / below the tie, all ones -- carried through by zero
    operands (the unit's output is the accumulator itself), plus random units; y16 equals ``f32_to_bf16_word`` of the model's
    FP32 word."""
    from verity.ml.circuits.silicon.cast import f32_to_bf16_word
    p = rel.params
    sys = rel.compile(p, True)
    tb = tables(sys, "cpu")
    cs = [(hi << 16) | lo for hi in (0x3F80, 0x3F81, 0x4000, 0xC001, 0x0001, 0x007F, 0x7F7F)
          for lo in (0x8000, 0x8001, 0x7FFF, 0x0000, 0xFFFF)]
    c = np.array(cs, dtype=np.int64)
    a = np.zeros((len(c), rel.k), dtype=np.int64)
    b = np.zeros_like(a)
    rng = np.random.default_rng(7)
    a2, b2 = rel.operands(rng, (200, rel.k)), rel.operands(rng, (200, rel.k))
    c = np.concatenate([c, c[rng.integers(0, len(c), 200)]])
    a, b = np.concatenate([a, a2]), np.concatenate([b, b2])
    y = np.array([tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(len(c))], dtype=np.int64)
    fin = ((y >> 23) & 0xFF) != 0xFF
    c, a, b, y = c[fin], a[fin], b[fin], y[fin]
    pub = rel.public_vectors(p, c, a, b, y, device="cpu")
    W = build_witness(sys, pub, rel.hints(p, sys, c, a, b, y, "cpu"), "cpu")
    assert bool(_accepts(sys, tb, W, pub).all())
    e = sys.chain["y16"]
    got = (sum(W[j].to(torch.int64) * v for j, v in e.lin.items()) + e.c) % P
    exp = torch.as_tensor([f32_to_bf16_word(int(v)) for v in y], dtype=torch.int64)
    assert torch.equal(got, exp)
    assert int(((y & 0xFFFF) == 0x8000).sum()) >= 8, "ties present"


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_zero_sums_accepted_in_unit_and_chain_mode(rel):
    """Exact zero sums -- the products cancel pairwise, every operand zero on a zero accumulator (E = floor), the accumulator
    cancelling a single product, and the cancellation found by the 1e5 differential (fp8-ada-v2, c = 0xc69af800, seed 5096) --
    are accepted with the model's zero word in both modes: the normalisation's shift proves nothing when the sum is zero, so the
    clamp / overflow proofs read the exponent masked by z (a zero sum with E anywhere in [floor, acc_e_max] must have a witness)."""
    p = rel.params
    pl = plan(p)
    rng = np.random.default_rng(11)
    n = 256
    a, b = rel.operands(rng, (n, rel.k)), rel.operands(rng, (n, rel.k))
    c = _finite_fp32(rng, n)
    sign_bit = 1 << (pl.op_word_bits - 1)
    kind = np.arange(n) % 4
    for i in range(n):
        if kind[i] == 0:
            a[i, 1::2], b[i, 1::2] = a[i, 0::2], b[i, 0::2] ^ sign_bit
            c[i] = c[i] & (1 << 31)                          # +-0 accumulator
        elif kind[i] == 1:
            a[i], b[i] = 0, 0
            c[i] = c[i] & (1 << 31)                          # E = floor
        elif kind[i] == 2:
            a[i, 1:], b[i, 1:] = 0, 0
            c[i] = tc_dot(rel.model, 0, a[i].tolist(), (b[i] ^ sign_bit).tolist())
        else:
            a[i, 1::2], b[i, 1::2] = a[i, 0::2], b[i, 0::2] ^ sign_bit      # cancelling products on a live accumulator
    y = np.array([tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(n)], dtype=np.int64)
    fin = ((y >> 23) & 0xFF) != 0xFF
    c, a, b, y = c[fin], a[fin], b[fin], y[fin]
    assert int(((y & 0x7FFFFFFF) == 0).sum()) >= n // 2, "zero outputs present"
    for chain in (False, True):
        sys = rel.compile(p, chain)
        tb = tables(sys, "cpu")
        pub = rel.public_vectors(p, c, a, b, y, device="cpu")
        W = build_witness(sys, pub, rel.hints(p, sys, c, a, b, y if chain else None, "cpu"), "cpu")
        ok = _accepts(sys, tb, W, pub)
        assert bool(ok.all()), f"{rel.name} chain={chain}: rejected {torch.nonzero(~ok).flatten().tolist()[:5]}"
        if chain:
            for k, e in ((k, e) for k, e in sys.outputs.items() if k in pub):
                got = (sum(W[j].to(torch.int64) * v for j, v in e.lin.items()) + e.c) % P
                assert torch.equal(got, torch.as_tensor(pub[k]) % P), f"{rel.name}: output {k}"


# relmin-private's 1e5 differential hit (fp8-hopper, seed 3000 + 90112, index 2424): two live products cancel exactly at the
# group maximum, the other 30 are dead there; the model's word is +0.  The pubsel of 02c3321 (before the z-masked clamp)
# rejected its honest witness at g0.norm.clamp.
UNIT_92536 = {"c": 0,
              "a": [96, 233, 233, 164, 93, 20, 48, 118, 215, 181, 209, 18, 148, 226, 84, 167, 65, 129, 187, 151, 223, 224, 40, 71,
                    178, 195, 64, 169, 5, 218, 32, 246],
              "b": [125, 124, 5, 34, 58, 67, 231, 70, 249, 173, 198, 97, 224, 153, 211, 39, 254, 144, 82, 7, 7, 251, 111, 26, 2, 126,
                    131, 76, 102, 208, 119, 174]}


def _cancelling_at_large_maximum(rel: Relation, n: int, seed: int):
    """(c, a, b): c = +-0; ``a_0 b_0`` near the top of the operand range and ``a_j b_j = -a_0 b_0`` at a random position j;
    every other product zero or tiny (subnormal / smallest-normal operands), i.e. shifted out at the group maximum.  The
    sum of a group holding both is exactly zero at E far above ``acc_e_min + width``: no normalisation shift reaches the
    subnormal grid's ``tc(E)``, so only a clamp that ignores zero sums admits the honest witness."""
    pl = plan(rel.params)
    rng = np.random.default_rng(seed)
    sign, ef, top = 1 << (pl.op_word_bits - 1), pl.op_frac, 2**pl.op_exp - 2
    a = np.zeros((n, rel.k), dtype=np.int64)
    b = np.zeros_like(a)
    for i in range(n):
        a[i, 0] = (int(rng.integers(top - 3, top + 1)) << ef) | int(rng.integers(0, 2**ef))
        b[i, 0] = (int(rng.integers(top - 3, top + 1)) << ef) | int(rng.integers(0, 2**ef))
        j = int(rng.integers(1, rel.k))
        a[i, j], b[i, j] = a[i, 0], b[i, 0] ^ sign
        for q in range(1, rel.k):
            if q != j and rng.random() < 0.7:
                a[i, q] = (int(rng.integers(0, 2)) << (pl.op_word_bits - 1)) | int(rng.integers(0, 2 * 2**ef))
                b[i, q] = int(rng.integers(0, 2 * 2**ef))
    c = np.where(rng.integers(0, 2, size=n) == 1, 1 << 31, 0).astype(np.int64)
    return c, a, b


@pytest.mark.parametrize("rel", V2 + [RELATIONS[n] for n in ("bf16-hopper-v2x4", "bf16-ampere-v2x4", "fp8-hopper-v2x4",
                                                              "fp8-ada-v2x4")], ids=lambda r: r.name)
def test_cancelling_sum_at_large_group_maximum_accepted(rel):
    """Exact cancellation at a large group maximum with every other product dead (unit 92536 for the k = 32 E4M3 units, 256
    constructed units each): the honest witness is accepted in unit and chain mode, every flipped claim is refused, and the
    v1 unit (where registered) agrees on both."""
    p = rel.params
    pl = plan(p)
    c, a, b = _cancelling_at_large_maximum(rel, 256, seed=92536)
    if pl.fp8 and rel.k == len(UNIT_92536["a"]):
        c = np.concatenate([[UNIT_92536["c"]], c])
        a = np.concatenate([[UNIT_92536["a"]], a])
        b = np.concatenate([[UNIT_92536["b"]], b])
    y = np.array([tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(len(c))], dtype=np.int64)
    fin = ((y >> 23) & 0xFF) != 0xFF
    c, a, b, y = c[fin], a[fin], b[fin], y[fin]
    assert len(y) >= 16 and int(((y & 0x7FFFFFFF) == 0).sum()) >= 16, "zero words at a large maximum present"
    if pl.fp8 and rel.k == len(UNIT_92536["a"]) and rel.name == "fp8-hopper-v2":
        assert int(y[0]) == 0, "unit 92536 is the +0 word"
    for chain in (False, True):
        sys = rel.compile(p, chain)
        tb = tables(sys, "cpu")
        pub = rel.public_vectors(p, c, a, b, y, device="cpu")
        W = build_witness(sys, pub, rel.hints(p, sys, c, a, b, y if chain else None, "cpu"), "cpu")
        ok = _accepts(sys, tb, W, pub)
        assert bool(ok.all()), f"{rel.name} chain={chain}: honest units rejected {torch.nonzero(~ok).flatten().tolist()[:5]}"
    s2 = rel.compile(p, False)
    tb2 = tables(s2, "cpu")
    lo = pl.drop if pl.fp8 else 0
    yb = y ^ (1 << np.random.default_rng(5).integers(lo, 31, size=len(y)))
    fb = ((yb >> 23) & 0xFF) != 0xFF
    bad = _witness_accepts(s2, tb2, rel.hints, rel.public_vectors, p, c[fb], a[fb], b[fb], yb[fb])
    assert not bool(bad.any()), f"{rel.name}: flipped claims accepted {torch.nonzero(bad).flatten().tolist()[:5]}"
    if rel.name in V1_OF:
        v1, s1, h1, pv1 = _v1_unit(rel)
        tb1 = tables(s1, "cpu")
        assert bool(_witness_accepts(s1, tb1, h1, pv1, p, c, a, b, y).all()), f"{rel.name}: v1 rejects an honest unit"
        assert not bool(_witness_accepts(s1, tb1, h1, pv1, p, c[fb], a[fb], b[fb], yb[fb]).any())


# ---------------------------------------------------------------------------------------------------------------------
# The folded x4 units: four instruction steps per column (experimental: census + CPU evidence; the GPU gate is pending)
# ---------------------------------------------------------------------------------------------------------------------

X4 = [RELATIONS[n] for n in ("bf16-hopper-v2x4", "bf16-ampere-v2x4", "fp8-hopper-v2x4", "fp8-ada-v2x4")]
N_FOLD = int(os.environ.get("RELMIN_FOLD_N", "1500"))


def _base(rel: Relation) -> Relation:
    return RELATIONS[rel.name[: -len("x4")]]


@pytest.mark.parametrize("rel", X4, ids=lambda r: r.name)
def test_folded_model_equals_four_chained_steps(rel):
    """``GroupSum(groups x 4).step`` is four chained steps of the base model: the FP32 pack / decode between steps is the
    identity on the adder-side state.  Random chains with accumulators at the products' scale: +-0, the model's one- and
    two-step states, and words whose exponent sits within +-12 of a one-step result (alignment, cancellation, zero sums)."""
    base = _base(rel)
    k, n = base.k, N_FOLD
    rng = np.random.default_rng(31)
    a, b = rel.operands(rng, (n, 4 * k)), rel.operands(rng, (n, 4 * k))
    a0, b0 = rel.operands(rng, (n, k)), rel.operands(rng, (n, k))
    for i in range(n):
        y = tc_dot(base.model, 0, a0[i].tolist(), b0[i].tolist())
        q = i % 5
        if q == 0:
            c = 0 if i % 2 else 0x80000000
        elif q in (1, 2):
            c = y
        elif q == 3:
            c = tc_dot(base.model, y, b0[i].tolist(), a0[i].tolist())
        else:
            e = (y >> 23) & 0xFF
            c = y if e in (0, 0xFF) else (y & 0x807FFFFF) | (int(np.clip(e + rng.integers(-12, 13), 1, 254)) << 23)
        if (c >> 23) & 0xFF == 0xFF:
            c = 0
        chained = c
        for s in range(4):
            chained = tc_dot(base.model, chained, a[i, k * s:k * s + k].tolist(), b[i, k * s:k * s + k].tolist())
        assert tc_dot(rel.model, c, a[i].tolist(), b[i].tolist()) == chained, f"{rel.name}: chain {i} c={c:08x}"


@pytest.mark.parametrize("rel", X4, ids=lambda r: r.name)
def test_folded_unit_accepts_exactly_the_model_word(rel):
    """Unit mode: the honest witness is accepted with the folded model's word and refused with any bit of it flipped;
    the census: fewer rows than four base units (the accumulator decode is shared), four times the pins."""
    base = _base(rel)
    p = rel.params
    sys = rel.compile(p, False)
    tb = tables(sys, "cpu")
    c, a, b, y = _cases(rel, 256, seed=11)
    c, a, b, y = (torch.as_tensor(x, dtype=torch.int64) for x in (c, a, b, y))
    ok = _witness_accepts(sys, tb, rel.hints, rel.public_vectors, p, c, a, b, y)
    assert bool(ok.all()), f"{rel.name}: honest units rejected {torch.nonzero(~ok).flatten().tolist()[:5]}"
    lo = 10 if plan(p).fp8 else 0
    bits = torch.as_tensor(np.random.default_rng(3).integers(lo, 32, size=len(y)))
    bad = _witness_accepts(sys, tb, rel.hints, rel.public_vectors, p, c, a, b, y ^ (1 << bits))
    assert not bool(bad.any()), f"{rel.name}: flipped claims accepted {torch.nonzero(bad).flatten().tolist()[:5]}"
    c4, c1 = census(rel.compile(p, True)), census(base.compile(base.params, True))
    assert c4["pin_rows"] == 4 * c1["pin_rows"] and c4["rows_per_unit"] < 4 * c1["rows_per_unit"], (c4, c1)
    assert rel.k == 4 * base.k and rel.steps * 4 == base.steps


@pytest.mark.parametrize("rel", X4, ids=lambda r: r.name)
def test_folded_chain_is_the_base_claim_cpu(rel):
    """The chain runner: the same synthetic VUs (operands and final words) as the base relation, proved with a quarter of
    the columns; two VUs proved + verified (interactive, non-ZK), the negative battery rejected."""
    base = _base(rel)
    vus, vb = instances(rel, 2, seed=5), instances(base, 2, seed=5)
    assert all(v[0] == w[0] and v[1] == w[1] and v[3] == w[3] for v, w in zip(vus, vb))
    assert all(v[2][3::4] == w[2] for v, w in zip(vb, vus)), "the folded accumulators are every fourth base accumulator"
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_INTERACTIVE)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    _, _, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, vus, row_negatives=20)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
