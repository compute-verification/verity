"""Tests for the Ligerito parameter calculator: the paper's Table 1 within 10 %, the Rust crate's hard-coded 2^20
shape, Merkle multi-opening expectation sanity, soundness accounting, and our five Table 2 cells under the brief's
targets (<= 1 MiB per batch, 2^-128)."""

import math

import pytest

from backends.direct.ligerito import params as P


# --- Merkle multi-opening expectation ---------------------------------------------------------------------------------

def test_merkle_siblings_edge_cases():
    assert P.expected_merkle_siblings(1 << 10, 1) == pytest.approx(10.0)          # one path = height siblings
    assert P.expected_merkle_siblings(1 << 10, 1 << 10) == pytest.approx(0.0)     # every leaf opened: nothing to send
    # between: fewer than t * height and monotone in t
    s_100, s_200 = P.expected_merkle_siblings(1 << 20, 100), P.expected_merkle_siblings(1 << 20, 200)
    assert s_100 < 100 * 20 and s_100 < s_200 < 200 * 20
    # the batching saving is only at the top ~log2(t) levels: at least (h - log2 t - 1) t siblings survive
    assert s_100 > (20 - math.log2(100) - 1) * 100


def test_merkle_siblings_matches_monte_carlo():
    import random
    n, t, h = 1 << 12, 37, 12
    rng = random.Random(1)
    tot = 0
    trials = 400
    for _ in range(trials):
        leaves = set(rng.sample(range(n), t))
        occ = leaves
        for _d in range(h):
            sib = sum(1 for v in occ if (v ^ 1) not in occ)
            tot += sib
            occ = {v >> 1 for v in occ}
    assert P.expected_merkle_siblings(n, t) == pytest.approx(tot / trials, rel=0.03)


# --- queries -----------------------------------------------------------------------------------------------------------

def test_queries_unique_decoding():
    # ((1 + rho)/2)^t <= 2^-100 at rho = 1/4 -> t = ceil(100 / log2(1.6)) = 148: the paper's count
    assert P.queries_for(-100.0, 0.25) == 148
    assert P.queries_for(-128.0, 0.25) == 189
    assert P.queries_for(-128.0, 0.5) == 309


# --- the paper's Table 1 -----------------------------------------------------------------------------------------------

def test_paper_table1_within_10_percent():
    r = P.paper_table1()
    for lg, row in r.items():
        assert abs(row["ratio"] - 1.0) <= 0.10, (lg, row["kib"], row["paper_kib"])


def test_rust_crate_2pow20_shape():
    """docs.rs/ligerito configs: 2^20 = one 2^14 x 2^6 base level, one 2^10 x 2^4 ext level, final 2^10 (rate 1/4,
    148 queries) -- the optimizer lands exactly on it."""
    p = P.optimize(1 << 20, 2, fields=P.PAPER_FIELDS, target_log2=-100.0, fixed_queries=148, zk=False, max_n_log2=40)
    assert p.wide_log2 == (6, 4) and p.final_len == 1024 and p.tall(0) == 1 << 14 and p.n(0) == 1 << 16


# --- soundness ---------------------------------------------------------------------------------------------------------

def test_soundness_terms_and_union():
    rel = P.RELATIONS["fp8-ada"]
    p = P.our_params(rel, L=4, rate_log2=(2, 4, 4, 4))
    s = p.soundness()
    assert s["meets_target"] and s["total_log2"] <= -128.0
    # the proximity terms dominate: every field term is below 2^-150
    for k, v in s["terms_log2"].items():
        if "proximity" not in k:
            assert v < -150.0, (k, v)
    # one query fewer at level 1 breaks the union by exactly log2((1 + rho_1)/2)
    from dataclasses import replace
    q = replace(p, queries=(p.queries[0] - 1,) + p.queries[1:])
    d = q.soundness()["terms_log2"]["L1.proximity ((1+rho)/2)^|S|"] - s["terms_log2"]["L1.proximity ((1+rho)/2)^|S|"]
    assert d == pytest.approx(-math.log2((1 + p.rate(0)) / 2))


def test_zk_costs_salt_and_mask_rows():
    rel = P.RELATIONS["fp8-ada"]
    p = P.our_params(rel, L=4, rate_log2=(2, 4, 4, 4))
    nz = P.LigeritoParams(**{**p.__dict__, "zk": False})
    assert p.proof_bytes()["total_bytes"] > nz.proof_bytes()["total_bytes"]


# --- validation against the GPU prototype's MEASURED proofs (ligerito-proto FINAL 20:35Z, 4090, Fiat-Shamir, no ZK) ---------

PROTO_MEASURED_KB = [   # (tall1, wide_log2, rate_log2, queries, measured proof KB)
    (1 << 19, (7, 4, 4), (2, 2, 2), (192,) * 3, 454),                          # Table A, 2^26
    (1 << 22, (6, 4, 4, 4), (2, 2, 2, 2), (192,) * 4, 532),                    # Table A, 2^28
    (1 << 22, (7, 4, 4, 4), (2, 2, 2, 2), (192,) * 4, 581),                    # Table A, 2^29
    (1 << 22, (7, 5, 5), (1, 2, 2), (311, 191, 193), 790),                     # Table B, our F point at 2^29
    (1 << 23, (7, 5, 5), (1, 2, 2), (311, 191, 193), 911),                     # Table B, our F point at 2^30
    (1 << 23, (6, 4, 4, 3, 3), (2, 4, 4, 4, 4), (191, 143, 143, 143, 144), 508),   # Table B, our S point at 2^29
    (1 << 20, (9, 6), (2, 2), (148, 148), 1030),                               # Table B, paper-style l = 3
    (1 << 22, (7, 5, 5), (2, 2, 2), (148,) * 3, 558),                          # Table B, paper-style l = 4
]


@pytest.mark.parametrize("tall1,wide,rates,queries,kb", PROTO_MEASURED_KB)
def test_byte_model_matches_measured_prototype(tall1, wide, rates, queries, kb):
    p = P.LigeritoParams(tall1=tall1, wide_log2=wide, rate_log2=rates, queries=queries, fields=P.OUR_FIELDS, target_log2=-128.0,
                         zk=False, relation=None)
    assert p.proof_bytes()["total_bytes"] / 1e3 == pytest.approx(kb, rel=0.01)


# --- our batches -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,dev", P.TABLE2)
def test_table2_cell_under_budget(name, dev):
    rel = P.RELATIONS[name]
    rec = P.recommend(rel, P.GPUS[dev])
    for key in ("smallest", "fastest"):
        p = rec[key]["params"]
        pb = p.proof_bytes()
        assert pb["total_bytes"] <= 1 << 20, (name, key, pb["total_kib"])
        assert p.soundness()["meets_target"]
        assert p.tall(0) * p.wide(0) == rel.rows_pad * rel.cols          # every cell committed
        assert all(p.n(i) <= 1 << 27 for i in range(p.L))                 # NTT cap
    # the brief's numbers: fp8-ada ~2^29.4 real cells (3724 x 48 x 4096) padded to 4096 rows x 3 x 2^16
    if name == "fp8-ada":
        assert rel.rows_pad == 4096 and rel.cols == 3 * 2**16
        assert rec["smallest"]["bytes"] / 1024 < 600


def test_bytes_per_second_vs_today():
    for c in P.table2_cells():
        assert c["bytes_per_s_at_today"] < 10e6                     # brief's egress target even at today's prover speed
        assert c["bytes_per_s_at_today"] < c["today_bytes_per_s"] / 20


def test_coins_and_latency_pipelining():
    c = next(x for x in P.table2_cells() if x["cell"] == "fp8-ada")
    assert 40 <= c["coins"] <= 70
    lat = c["latency"]
    assert lat["wait_s"] == pytest.approx(c["coins"] * 0.030, rel=0.01)
    assert lat["depth_needed"] >= 2                                # a single batch in flight cannot hide 30 ms x coins
