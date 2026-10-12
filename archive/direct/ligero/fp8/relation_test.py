"""The FP8 unit against ``silicon.tc_dot(ADA_E4M3_M16N8K32, ...)``: 2,000 honest random finite cases accepted; 52 wrong
claims rejected (accumulator bit flips, off-by-one rounding, wrong truncation of the accumulator, wrong group order,
one-group / wide-adder models, NaN operands); one witness-row mutation per gadget scope and row class rejected; the
48-step chain proved and verified on the CPU (non-ZK and ZK), its negatives rejected.  CPU only, small l."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")            # the verifier accepts blake3-named proofs only; without the package the prover falls back and the chain tests fail for the wrong reason
from verity.ml.circuits.silicon.silicon import (
    ADA_E4M3_M16N8K32,
    HOPPER_E4M3_K32,
    Pipeline,
    Term,
    e4m3_product,
    fp32_term,
    pack_fp32,
    tc_dot,
    tc_dot_chain,
)

from .. import protocol as protocol_mod
from ..compile import P
from ..witness import build_witness, check_constraints, eval_expr, tables
from .chain import FP8ChainRunner, instances_fp8
from .relation import ADA, K, STEPS, compile_fp8_unit, pack_public, public_vectors_fp8
from .witness import hints_fp8, oracle_words, random_finite_fp32, random_finite_operands

N_HONEST = 2000


@pytest.fixture(scope="module")
def unit():
    s = compile_fp8_unit(ADA, chain=False)
    return s, tables(s, "cpu")


def _cases(n: int, seed: int):
    rng = np.random.default_rng(seed)
    a = random_finite_operands(rng, (n, K))
    b = random_finite_operands(rng, (n, K))
    c = random_finite_fp32(rng, n)
    c[rng.integers(0, n, size=n // 20)] = 0                   # exact zeros as well
    return c, a, b


def _witness(unit, c, a, b, y):
    s, tb = unit
    pub = public_vectors_fp8(ADA, c, a, b, y)
    W = build_witness(s, pub, hints_fp8(ADA, s, c, a, b), "cpu")
    return W, pub, check_constraints(s, tb, W, pub)


def _words(s, W):
    ys, yt, yf = (eval_expr(s.outputs[n], W) for n in ("y.s", "y.t", "y.f"))
    return ((ys << 31) | (yt << 23) | yf).numpy()


def test_honest_2000_random_finite(unit):
    c, a, b = _cases(N_HONEST, 1)
    y = oracle_words(c, a, b)
    W, _, bad = _witness(unit, c, a, b, y)
    assert bad == []
    assert (_words(unit[0], W) == y).all()
    assert (y & 0x3FF == 0).all()                              # every honest word is M14 << 10


# --- wrong models -----------------------------------------------------------------------------------------------------

def _group_sum_variant(terms, width, zero_exponent, acc_round: bool):
    """``silicon.group_sum`` with the accumulator rescale rounding to nearest instead of truncating (``acc_round``)."""
    rescale = width - 24
    max_exponent = zero_exponent
    for t in terms:
        if t.magnitude and t.exponent > max_exponent:
            max_exponent = t.exponent
    total = 0
    for i, t in enumerate(terms):
        if not t.magnitude:
            continue
        if rescale >= 0:
            scaled = t.magnitude << rescale
        elif acc_round and i == 0:
            scaled = (t.magnitude + (1 << (-rescale - 1))) >> -rescale
        else:
            scaled = t.magnitude >> -rescale
        aligned = scaled >> (max_exponent - t.exponent)
        total += -aligned if t.negative else aligned
    if total == 0:
        return Term(False, 0, 0)
    neg = total < 0
    mag = -total if neg else total
    bl = mag.bit_length()
    e = max_exponent + bl - width
    mag = mag >> (bl - width) if bl > width else mag << (width - bl)
    if e < -126:
        mag >>= -126 - e
        e = -126
    mag = mag >> rescale if rescale >= 0 else mag << -rescale
    return Term(False, 0, 0) if mag == 0 else Term(neg, e, mag)


def _tc_dot_variant(c, a, b, *, groups=(16, 16), width=14, floor=-139, order=None, acc_round=False):
    order = list(range(K)) if order is None else order
    acc = fp32_term(c)
    start = 0
    for size in groups:
        prods = [e4m3_product(a[order[i]], b[order[i]]) for i in range(start, start + size)]
        acc = _group_sum_variant([acc, *prods], width, floor, acc_round)
        if acc.magnitude and acc.exponent > 127:
            return pack_fp32(acc)
        start += size
    return pack_fp32(acc)


def _differing(c, a, b, y, variant, want: int, allow_low_bits: bool = False):
    """Indices where a wrong model's word differs from the honest one (the first ``want``)."""
    out = []
    for i in range(len(c)):
        yv = variant(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]])
        if yv != int(y[i]) and (allow_low_bits or (yv & 0x3FF) == 0):
            out.append((i, yv))
        if len(out) == want:
            break
    assert len(out) == want, f"only {len(out)} differing cases"
    return out


def test_52_wrong_claims_rejected(unit):
    s, tb = unit
    c, a, b = _cases(256, 2)
    y = oracle_words(c, a, b)
    claims: list[tuple[str, np.ndarray]] = []
    # 22 accumulator bit flips of the claimed word (sign, 8 exponent bits, the 13 kept fraction bits) -- one case each
    for bit in range(10, 32):
        yb = y.copy()
        yb[bit % 256] ^= 1 << bit
        claims.append((f"flip bit {bit}", yb))
    # off-by-one rounding: +-1 in the 14-bit adder's last place
    for d in (1, -1):
        yb = y.copy()
        yb[5] = (int(y[5]) + d * 1024) % 2**32
        claims.append((f"{d:+d} ulp", yb))
    # wrong truncation: the accumulator rounded to nearest into the 14-bit adder (8 cases where it differs)
    for i, yv in _differing(c, a, b, y, lambda c_, a_, b_: _tc_dot_variant(c_, a_, b_, acc_round=True), 8):
        yb = y.copy()
        yb[i] = yv
        claims.append((f"acc rounded not truncated @{i}", yb))
    # wrong group order: the second 16 products summed first (8)
    swapped = list(range(16, 32)) + list(range(16))
    for i, yv in _differing(c, a, b, y, lambda c_, a_, b_: _tc_dot_variant(c_, a_, b_, order=swapped), 8):
        yb = y.copy()
        yb[i] = yv
        claims.append((f"groups swapped @{i}", yb))
    # one group of 32 (Hopper's wgmma model) (6)
    for i, yv in _differing(c, a, b, y, lambda c_, a_, b_: tc_dot(HOPPER_E4M3_K32, c_, a_, b_), 6):
        yb = y.copy()
        yb[i] = yv
        claims.append((f"one group of 32 @{i}", yb))
    # a 25-bit adder (no truncation of the accumulator at all; Ampere's BF16 width) (4)
    for i, yv in _differing(c, a, b, y, lambda c_, a_, b_: _tc_dot_variant(c_, a_, b_, width=25, floor=-132), 4, True):
        yb = y.copy()
        yb[i] = yv
        claims.append((f"25-bit adder @{i}", yb))
    assert len(claims) == 50
    rejected = 0
    for name, yb in claims:
        try:
            _, _, bad = _witness(unit, c, a, b, yb)
        except ValueError as e:                                # the verifier's decode: a word no tensor core produces
            assert "low bits" in str(e), name
            bad = [str(e)]
        assert bad, f"wrong claim accepted: {name}"
        rejected += 1
    # 2 NaN operands smuggled into the statement: the verifier's decode refuses
    for name, arr, pos, code in (("a", a, (3, 4), 0x7F), ("b", b, (9, 31), 0xFF)):
        bad_ops = arr.copy()
        bad_ops[pos] = code
        with pytest.raises(ValueError, match="NaN"):
            public_vectors_fp8(ADA, c, bad_ops if name == "a" else a, bad_ops if name == "b" else b, y)
        rejected += 1
    assert rejected == 52


def test_row_mutations_rejected(unit):
    s, tb = unit
    c, a, b = _cases(64, 3)
    y = oracle_words(c, a, b)
    W, pub, bad = _witness(unit, c, a, b, y)
    assert bad == []
    seen, picks = set(), []
    for r in s.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(7)
    for r in picks:
        Wm = W.clone()
        col = int(rng.integers(0, 64))
        Wm[r.idx, col] = (Wm[r.idx, col] + int(rng.integers(1, 3))) % P
        assert check_constraints(s, tb, Wm, pub), f"mutation of {r.name} ({r.kind}) accepted"


# --- the chain ----------------------------------------------------------------------------------------------------------

def _args(**kw):
    return argparse.Namespace(**kw)


@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_chain_two_vus_cpu(zk, mode):
    R = FP8ChainRunner("cpu", -128.0, zk=zk, mode=mode)
    vus = instances_fp8(2, seed=5)
    for a, b, accs, y in vus:
        assert tc_dot_chain(ADA_E4M3_M16N8K32, 0, a, b) == y
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    a, b, ypk = pubs
    y_bad = ypk.clone()
    y_bad[STEPS - 1] ^= 1 << 3
    assert not R.verify_vus(proof, (a, b, y_bad), lay)[0]
    a_bad = a.copy()
    a_bad[2, 2] = 0x7F
    assert not R.verify_vus(proof, (a_bad, b, ypk), lay)[0]
    c_rows = R.sys.chain["c"]

    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1024) % P
    proof2, pubs2, lay2 = R.prove_vus(vus, check=False, mutate=break_link)
    assert not R.verify_vus(proof2, pubs2, lay2)[0]

    def nonzero_c0(W):
        W[c_rows[1], STEPS] = 1
    proof2, pubs2, lay2 = R.prove_vus(vus, check=False, mutate=nonzero_c0)
    assert not R.verify_vus(proof2, pubs2, lay2)[0]


def test_pack_public_rejects_low_bits():
    with pytest.raises(ValueError):
        pack_public(0x3F800001)
    assert pack_public(0x3F800000) == 0x3F800000 >> 10


# --- the dump files (ligero-statement/v4 + ligero-proof/v1 + ligero-system/v1), cold from bytes -------------------------

@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_dump_round_trip_v4_cpu(tmp_path, zk, mode):
    """What ``bench_vu_fp8 --dump-dir`` writes, read back and verified by ``serialize.verify_files`` with no prover state:
    the v4 statement carries the relation name and the FP8 word widths (1-byte operands, 4-byte public words), the proof
    and system bytes round-trip, the file verifier accepts the honest files and rejects a flipped public word; a v2
    (BF16) statement still reads as relation ``bf16`` (v3 = the BF16 words + an authentication block is untouched)."""
    from .. import serialize as ser
    R = FP8ChainRunner("cpu", -128.0, zk=zk, mode=mode)
    vus = instances_fp8(2, seed=5)
    V = protocol_mod.Coins.sample() if R.interactive else None
    proof, pubs, lay = R.prove_vus(vus, check=False, coins=V)
    assert R.verify_vus(proof, pubs, lay, coins=V)[0]
    st = ser._statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    assert st.relation == "fp8-ada" and (st.word_bytes, st.y_bytes) == (1, 4) and st.v4 and st.auth is None
    raw = ser.statement_bytes(st)
    assert raw[:8] == ser.STMT_MAGIC_V4
    back = ser.read_statement(raw)
    assert back.relation == "fp8-ada" and (back.word_bytes, back.y_bytes) == (1, 4) and back.auth is None
    assert (back.l, back.n, back.D, back.t, back.zk, back.mode, back.n_vus, back.steps) == (st.l, st.n, st.D, st.t, st.zk, st.mode, st.n_vus, st.steps)
    assert np.array_equal(back.a, st.a) and np.array_equal(back.b, st.b) and np.array_equal(back.y16, st.y16)
    assert int(st.a.max()) < 256 and int(st.y16.max()) >= 1 << 16              # 8-bit operands, 22-bit public words
    pf2 = ser.read_proof(ser.proof_bytes(proof))                                # the proof round-trips (timings are not in the file)
    assert pf2.root == proof.root and pf2.timings == {} and np.array_equal(pf2.opened, proof.opened)
    S2 = ser.read_system(ser.system_bytes(R.sys))
    assert S2["sys_id"] == protocol_mod.system_id(R.sys) and S2["m"] == R.sys.m
    (tmp_path / "sub_00.stmt").write_bytes(raw)
    (tmp_path / "sub_00.proof").write_bytes(ser.proof_bytes(proof))
    if V is not None:
        (tmp_path / "sub_00.coins").write_bytes(V.r1 + V.s1 + V.r2 + V.s2)
        assert len((tmp_path / "sub_00.coins").read_bytes()) == 128
    ok, why, _ = ser.verify_files(tmp_path / "sub_00.stmt", tmp_path / "sub_00.proof", "cpu", runner=R)
    assert ok, why
    ok, why, _ = ser.verify_files(tmp_path / "sub_00.stmt", tmp_path / "sub_00.proof", "cpu")     # runner from the file
    assert ok, why
    bad = bytearray(raw)                                                        # flip a bit of the last public word
    bad[-1] ^= 1 << 3
    (tmp_path / "bad.stmt").write_bytes(bytes(bad))
    ok, why, _ = ser.verify_files(tmp_path / "bad.stmt", tmp_path / "sub_00.proof", "cpu", runner=R)
    assert not ok, why
    # a BF16 statement (2-byte words, no relation name) is still the v2 file; the FP8 runner refuses it by name
    v2 = ser.Statement(st.l, st.n, st.D, st.t, st.target_log2, st.zk, st.t_pad, st.mode, st.n_proofs, st.n_vus, st.steps,
                       np.zeros((st.l, 4), "<u2"), np.zeros((st.l, 4), "<u2"), np.zeros(st.l, "<u2"))
    rv2 = ser.statement_bytes(v2)
    assert rv2[:8] == ser.STMT_MAGIC and not v2.v4
    b2 = ser.read_statement(rv2)
    assert b2.relation == "bf16" and (b2.word_bytes, b2.y_bytes) == (2, 2) and b2.a.dtype == np.dtype("<u2")
    (tmp_path / "v2.stmt").write_bytes(rv2)
    ok, why, _ = ser.verify_files(tmp_path / "v2.stmt", tmp_path / "sub_00.proof", "cpu", runner=R)
    assert not ok and "bf16" in why and "fp8-ada" in why, why
