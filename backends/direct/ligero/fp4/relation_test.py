"""The NVFP4 unit against ``BLACKWELL_SM120_NVF4.step``: every edge family accepted (torch-free, small n), every wrong-rule
witness variant rejected, off-domain public words refused by the decode, the ``ligero-system/v2`` and
``ligero-statement/v3`` files round-tripping, the K=1536 reference vector replaying through the hint generator; and,
with torch + blake3 present (the pod), the 24-step chain proved and verified on the CPU (non-ZK and FS-ZK) with its
negatives rejected.  Tiny cases only: the sweep (>= 2,000 honest VUs, >= 100 negatives) is ``gate_vu_fp4`` on the pod."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ..compile import P, Expr
from ..serialize import (
    BF16_RELATION,
    STMT_MAGIC,
    STMT_MAGIC_V3,
    SYS_MAGIC_V2,
    Statement,
    read_statement,
    read_system,
    statement_bytes,
    system_bytes,
)
from .relation import K, NVFP4, SCALES, STEPS, WORDS, compile_fp4_unit
from .witness import (
    FAMILIES,
    MODEL,
    VARIANTS,
    build_witness_np,
    check_constraints_np,
    eval_expr_np,
    hints_fp4,
    oracle_words,
    public_vectors_fp4,
    random_units,
    unit_hints,
    words_from_components,
)

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def unit():
    return compile_fp4_unit(NVFP4, chain=False)


def _pool(rng, per_family: int):
    cs, as_, bs = [], [], []
    for fam in FAMILIES:
        c, a, b = random_units(rng, per_family, fam)
        cs.append(c), as_.append(a), bs.append(b)
    return np.concatenate(cs), np.concatenate(as_), np.concatenate(bs)


def _words_out(s, W):
    return words_from_components(*(eval_expr_np(s.outputs[k], W) for k in ("y.s", "y.t", "y.f")))


def test_honest_every_edge_family(unit):
    s = unit
    rng = np.random.default_rng(1)
    for fam in FAMILIES:
        c, a, b = random_units(rng, 12, fam)
        y = oracle_words(c, a, b)
        H, yh = unit_hints(c, a, b)
        assert (yh == y).all(), f"{fam}: the hint generator's word differs from the model"
        pub = public_vectors_fp4(a, b, c, y)
        W = build_witness_np(s, pub, hints_fp4(s, c, a, b))
        assert check_constraints_np(s, W, pub) == [], fam
        assert (_words_out(s, W) == y).all(), fam


def test_zero_results_are_plus_zero(unit):
    """Exact cancellation with a +0 / -0 accumulator: the model's +0 rule, and the relation agrees."""
    rng = np.random.default_rng(2)
    c, a, b = random_units(rng, 8, "cancel")
    c[:4] = 0x8000_0000                                      # -0 accumulators: the result is still +0
    y = oracle_words(c, a, b)
    assert (y == 0).all()
    pub = public_vectors_fp4(a, b, c, y)
    W = build_witness_np(unit, pub, hints_fp4(unit, c, a, b))
    assert check_constraints_np(unit, W, pub) == []
    # a -0 claim for the same units: the checker rejects it
    y_neg = np.full_like(y, 0x8000_0000)
    pub2 = public_vectors_fp4(a, b, c, y_neg)
    assert check_constraints_np(unit, W, pub2)


def _violating(s, W, pub) -> np.ndarray:
    viol = np.zeros(W.shape[1], dtype=bool)
    for lin in s.linear:
        tgt = np.asarray(pub[lin.pub]) % P if lin.pub else 0
        viol |= eval_expr_np(Expr(None, lin.lin, lin.c), W) != tgt
    for q in s.quadratic:
        viol |= (eval_expr_np(q.a, W) * eval_expr_np(q.b, W)) % P != eval_expr_np(q.c, W)
    return viol


def test_wrong_rule_witnesses_rejected(unit):
    """Every variant of the rule (wrong anchor, floor / nearest instead of truncation, a 35- or 37-bit window, -0, the
    subnormal scale read as normal, scale exponent off by one, RNE normalisation, zero scales participating, a dropped
    group, the accumulator exponent off by one, its sign wrong): on every unit where its word differs from the model's,
    the witness it generates for its OWN claim violates a constraint (or the claim is refused by the decode)."""
    s = unit
    c, a, b = _pool(np.random.default_rng(3), 16)
    y = oracle_words(c, a, b)
    differing_total = 0
    for v in VARIANTS:
        H, yv = unit_hints(c, a, b, variant=v)
        diff = yv != y
        if not diff.any():
            continue
        nonfin = ((yv >> 23) & 0xFF) == 0xFF
        claim = np.where(nonfin, y, yv)                       # a non-finite claim: the decode refuses it (counted below)
        pub = public_vectors_fp4(a, b, c, claim)
        W = build_witness_np(s, pub, np.stack([H[n] for _, n in s.hints]))
        viol = _violating(s, W, pub)
        accepted = np.nonzero(diff & ~viol & ~nonfin)[0]
        assert accepted.size == 0, f"{v}: wrong words accepted at units {accepted[:5].tolist()}"
        differing_total += int(diff.sum())
    assert differing_total >= 100


def test_public_decode_refuses_off_domain_words():
    rng = np.random.default_rng(4)
    c, a, b = random_units(rng, 4, "random")
    y = oracle_words(c, a, b)
    for name, arr, pos, code, why in (("a", a, (0, K + 1), 0x7F, "NaN"), ("b", b, (1, K), 0x80, "padding"), ("a", a, (2, 3), 16, "E2M1")):
        bad = arr.copy()
        bad[pos] = code
        with pytest.raises(ValueError):
            public_vectors_fp4(bad if name == "a" else a, bad if name == "b" else b, c, y)
    y_bad = y.copy()
    y_bad[0] = 0x7F80_0000                                    # +inf
    with pytest.raises(ValueError, match="non-finite"):
        public_vectors_fp4(a, b, c, y_bad)
    c_bad = c.copy()
    c_bad[1] = 0x7FC0_0000                                    # NaN accumulator
    with pytest.raises(ValueError, match="non-finite"):
        public_vectors_fp4(a, b, c_bad, y)


def test_reference_vector_k1536_replays_through_the_hint_generator():
    rec = json.loads((REPO / "fixtures/tc/vectors/nvfp4-sm120-mma-k1536.json").read_text())
    assert rec["instruction"] == "sm120.mma.m16n8k64.e2m1.nvf4" and rec["model"] == MODEL.name
    codes_a, sc_a = [int(x, 16) for x in rec["a"]], [int(x, 16) for x in rec["a_scales"]]
    codes_b, sc_b = [int(x, 16) for x in rec["b"]], [int(x, 16) for x in rec["b_scales"]]
    c = int(rec["c0"], 16)
    A = np.asarray([codes_a[K * s:K * s + K] + sc_a[SCALES * s:SCALES * s + SCALES] for s in range(STEPS)], dtype=np.int64)
    B = np.asarray([codes_b[K * s:K * s + K] + sc_b[SCALES * s:SCALES * s + SCALES] for s in range(STEPS)], dtype=np.int64)
    assert A.shape == (STEPS, WORDS)
    for s, step in enumerate(rec["steps"]):
        assert int(step["c_in"], 16) == c
        _, yh = unit_hints(np.asarray([c]), A[s:s + 1], B[s:s + 1])
        c = int(yh[0])
        assert c == int(step["c_out"], 16), f"step {s}"
    assert c == int(rec["y"], 16)


# --- files --------------------------------------------------------------------------------------------------------------

def test_system_v2_round_trip():
    pytest.importorskip("torch")                              # system_bytes signs the file with protocol.system_id
    s = compile_fp4_unit(NVFP4, chain=True)
    data = system_bytes(s)
    assert data[:8] == SYS_MAGIC_V2
    rd = read_system(data)
    assert rd["m"] == s.m and len(rd["quadratic"]) == len(s.quadratic) and len(rd["linear"]) == len(s.linear)
    assert rd["chain"]["y16"] is None and len(rd["chain"]["y_end"]) == 3
    assert [(lo, bits) for _, lo, bits in rd["chain"]["y_end"]] == [(lo, bits) for _, lo, bits in s.chain["y_end"]]
    from ..protocol import system_id
    assert rd["sys_id"] == system_id(s)


def test_statement_v3_round_trip_and_bf16_stays_v2():
    rng = np.random.default_rng(5)
    l = 4
    a = np.concatenate([rng.integers(0, 16, (l, K)), rng.integers(0, 0x7F, (l, SCALES))], axis=1)
    b = np.concatenate([rng.integers(0, 16, (l, K)), rng.integers(0, 0x7F, (l, SCALES))], axis=1)
    y = np.array([0, 0, 0, 0xC1E0_0001], dtype=np.int64)           # a full FP32 word at the end column
    st = Statement(l, 16, 5, 3, -128.0, True, 512, "fiat-shamir", 1, 1, 4, a, b, y, "fp4-nvf4")
    data = statement_bytes(st)
    assert data[:8] == STMT_MAGIC_V3
    st2 = read_statement(data)
    assert st2.relation == "fp4-nvf4" and st2.widths == (1, 4)
    assert np.array_equal(st2.a, a) and np.array_equal(st2.b, b) and np.array_equal(st2.y16, y)
    assert (st2.l, st2.n, st2.D, st2.t, st2.zk, st2.t_pad, st2.mode, st2.n_proofs, st2.n_vus, st2.steps) == (l, 16, 5, 3, True, 512, "fiat-shamir", 1, 1, 4)
    # a claimed word that does not fit the stated width is refused at write time
    with pytest.raises(ValueError):
        statement_bytes(Statement(l, 16, 5, 3, -128.0, True, 512, "fiat-shamir", 1, 1, 4, a, b, y, "fp8-ada").__class__(
            l, 16, 5, 3, -128.0, True, 512, "fiat-shamir", 1, 1, 4, a + 256, b, y, "fp4-nvf4"))
    bf = Statement(2, 8, 3, 2, -128.0, False, 0, "interactive", 1, 1, 2, np.zeros((2, 16), np.int64), np.zeros((2, 16), np.int64),
                   np.zeros(2, np.int64), BF16_RELATION)
    assert statement_bytes(bf)[:8] == STMT_MAGIC and read_statement(statement_bytes(bf)).relation == BF16_RELATION


# --- the chain (torch) ---------------------------------------------------------------------------------------------------

def test_chain_two_vus_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("blake3")
    from .. import protocol as protocol_mod
    from .chain import FP4ChainRunner, instances_fp4

    for zk, mode in ((False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)):
        R = FP4ChainRunner("cpu", -128.0, zk=zk, mode=mode)
        vus = instances_fp4(2, seed=5)
        proof, pubs, lay = R.prove_vus(vus, check=True)
        ok, why = R.verify_vus(proof, pubs, lay)
        assert ok, why
        a, b, y32 = pubs
        for bit in (0, 22, 23, 31):                             # fraction lsb, fraction msb, exponent lsb, sign
            y_bad = y32.clone()
            y_bad[STEPS - 1] ^= 1 << bit
            assert not R.verify_vus(proof, (a, b, y_bad), lay)[0]
        a_bad = a.copy()
        a_bad[2, K + 2] = 0x7F                                  # a NaN scale in the statement
        assert not R.verify_vus(proof, (a_bad, b, y32), lay)[0]
        c_rows = R.sys.chain["c"]

        def break_link(W):
            W[c_rows[2], 20] = (W[c_rows[2], 20] + 1024) % P
        proof2, pubs2, lay2 = R.prove_vus(vus, check=False, mutate=break_link)
        assert not R.verify_vus(proof2, pubs2, lay2)[0]

        def nonzero_c0(W):
            W[c_rows[1], STEPS] = 1
        proof2, pubs2, lay2 = R.prove_vus(vus, check=False, mutate=nonzero_c0)
        assert not R.verify_vus(proof2, pubs2, lay2)[0]
        # a wrong-rule witness with the honest claim and links
        proof3, pubs3, lay3 = R.prove_vus(vus, check=False, variant="negzero")
        assert not R.verify_vus(proof3, pubs3, lay3)[0]
        del torch
