"""F6's prover column: the fit and the posted fit it replaces, the registered check, one session priced from fixed
coefficients, the range and memory caps, the slot sumcheck's term, netlists, staging, a tiny audit unsplit and split exactly, and token-select's
argmax tree."""
from __future__ import annotations

import math

import pytest

from verity_numerical.bench import unit_costs as U
from verity_numerical.bench import unit_costs_padding as P
from verity_numerical.bench import unit_costs_prover as PR

COEF = {"rss_gib": (1.0, 2.0, 3.0, 0.0), "prove_s": (0.5, 4.0, 8.0, 0.0), "wall_s": (0.0, 0.0, 0.0, 0.0)}
TINY = {"bytes_per_nnz": 4, "glue": 0}


def test_lstsq_recovers_an_affine_law():
    xs = [(0, 0), (1, 0), (0, 1), (2, 3), (5, 1)]
    assert PR.lstsq(xs, [1 + 2 * a + 3 * b for a, b in xs]) == pytest.approx([1, 2, 3])


def test_the_fit_over_the_source_runs():
    f = PR.fit()
    assert len(f["points"]) == len(PR.SOURCE) == 64
    assert f["coef"]["rss_gib"] == pytest.approx([1.229, 3.956, 2.535, 135.3], abs=2e-2)
    assert f["coef"]["prove_s"][2] == 0 and f["max_residual"]["rss_gib"] < 8 and f["max_residual"]["prove_s"] < 1.25
    posted = PR.fit(PR.HIDDEN_ZK, PR.POSTED_LAWS)
    assert posted["coef"]["rss_gib"] == pytest.approx([3.526, 4.000, 3.040, 0], abs=2e-3)


def test_the_registered_check():
    """The posted column missed both registered sessions by more than 25%, its session model and its law alike; the refit
    holds them within 25%."""
    rows = PR.check(PR.fit()["coef"])
    assert [r["K"] for r in rows] == [16384, 32768] and [r["k_log"] for r in rows] == [26, 27]
    for r in rows:
        assert r["posted"]["rss_gib"] > 4 * r["measured"]["rss_gib"] and r["law_at_measured"]["rss_gib"] < r["measured"]["rss_gib"] / 2
        assert r["now"]["k_log"] == r["k_log"] and r["now"]["D"] == r["D"]
        for k in ("rss_gib", "prove_s"):
            assert abs(r["now"][k] / r["measured"][k] - 1) < 0.25


def test_a_session_from_fixed_coefficients():
    # 2^20 nonzeros at 16 per row: 2^16 rows, a 2^19 slot, the 2^22 floor; D = 16 2^22 = 2^26; text 16 x 4 x 2^20 = 2^26 bytes
    s = PR.session(2 ** 20, 16, COEF, **TINY)
    assert (s["k_log"], s["D"], s["circuit_bytes"]) == (22, 2 ** 26, 2 ** 26)
    assert (s["rss_gib"], s["prove_s"]) == pytest.approx((1 + 2 / 16 + 3 / 16, 0.5 + 4 / 16 + 8 / 16))
    # 2^30: 2^26 rows, k_log 27 + 2, out of range; D = 2^33 (8 x 2^30); text 2^36 (64 GiB)
    s = PR.session(2 ** 30, 16, COEF, **TINY)
    assert (s["k_log"], s["in_range"], s["rss_gib"]) == (29, False, 1 + 2 * 8 + 3 * 64)
    # 2^28 is the last in range (2^24 rows, k_log 27): D = 2^31, text 16 GiB, 53 GiB; 2^27 27 GiB
    assert PR.cap(16, COEF, math.inf, **TINY) == 2 ** 28
    assert PR.cap(16, COEF, 53, **TINY) == 2 ** 28 and PR.cap(16, COEF, 52.9, **TINY) == 2 ** 27
    assert PR.cap(16, COEF, 1, **TINY) is None
    # netlists: the largest 16, each at most Z's; a block term
    s = PR.session(2 ** 20, 16, COEF, nets=[2 ** 10, 2 ** 30, 5], **TINY)
    assert s["circuit_bytes"] == 2 ** 22 + 2 ** 10 + 5
    blk = dict(COEF, rss_gib=(0.0, 0.0, 0.0, 16.0))
    assert PR.session(2 ** 20, 16, blk, **TINY)["rss_gib"] == pytest.approx(16 * 2 ** 22 / 2 ** 30)


def test_the_slot_sumcheck_at_fixed_coefficients():
    """K1 10 ns, 2 reps, 8 B of committed bits per entry and 64 B of working set per entry of one slot. At 2^20, 16 slots:
    2^24 entries, 10e-9 x 2^24 x 2 s, (8 x 2^24 + 64 x 2^20) / 2^30 = 0.1875 GiB. At 2^28 the sumcheck adds
    32 + 16 GiB to Flock's 53, so the cap at 101 GiB stays 2^28 and drops to 2^27 below it."""
    S = {"k1_ns": 10.0, "band": (5.0, 20.0), "reps": 2, "bits_bytes": 8.0, "work_bytes": 64.0}
    flock, s = PR.session(2 ** 20, 16, COEF, **TINY), PR.session(2 ** 20, 16, COEF, slot=S, **TINY)
    assert (flock["sumcheck_s"], flock["total_s"], flock["peak_gib"]) == (0, flock["prove_s"], flock["rss_gib"])
    assert s["entries"] == 2 ** 24 and s["sumcheck_s"] == pytest.approx(10e-9 * 2 ** 25)
    assert s["sumcheck_gib"] == pytest.approx(0.1875) and s["total_s"] == pytest.approx(s["prove_s"] + s["sumcheck_s"])
    assert s["peak_gib"] == pytest.approx(s["rss_gib"] + 0.1875)
    assert PR.cap(16, COEF, 101, slot=S, **TINY) == 2 ** 28 and PR.cap(16, COEF, 100.9, slot=S, **TINY) == 2 ** 27
    units = [P.Unit("a", "t", 4, 3, 3), P.Unit("b", "t", 2, 10, 10)]
    a0 = PR.audit(6, 3, units, {}, COEF, slots=2, k=2)
    a = PR.audit(6, 3, units, {}, COEF, slots=2, k=2, slot=S)
    assert a["prover_h_mean"] - a["sumcheck_h_mean"] == pytest.approx(a0["prover_h_mean"])
    assert a["sumcheck_h_mean"] == pytest.approx(sum(b["sessions_mean"] * 10e-9 * 2 * b["Z"] * 2 for b in a["body"]) / 3600)
    lo, hi = PR.band(a, S)
    assert hi["prover_h_public"] - lo["prover_h_public"] == pytest.approx(1.5 * a["sumcheck_h_public"])
    assert lo["share_mean"] == pytest.approx(a["sumcheck_h_mean"] / 2 / (a0["prover_h_mean"] + a["sumcheck_h_mean"] / 2))


def test_netlists_and_staging():
    per = {"nnz": 1000, "ands": 50, "committed": 10}
    inv = {"rows": [
        {"workload": "w", "unit_type": "r", "source": "registered", "subcircuit": {"per_unit": per},
         "registered_unit": {"rows": 1, "ands": 2, "nnz_a": 3, "nnz_b": 4}},
        {"workload": "w", "unit_type": "t", "source": "tree", "subcircuit": {"per_unit": per},
         "tree": {"members": [{"member": "m", "nnz": 7, "ands": 1}, {"member": "m", "nnz": 7, "ands": 1},
                              {"member": "n", "nnz": 9, "ands": 2}]}},
        {"workload": "w", "unit_type": "g", "source": "generic", "subcircuit": {"per_unit": per}},
        {"workload": "v", "unit_type": "x", "source": "generic", "subcircuit": {"per_unit": per}}]}
    n = PR.netlists(inv, "w", bytes_per_nnz=2)
    assert n == {"r": {"kind": "registered", "nnz": 8, "bytes": 16, "ands": 2},
                 "t": {"kind": "tree", "nnz": 16, "bytes": 32, "ands": 3},
                 "g": {"kind": "generic", "nnz": 1000, "bytes": 2000, "ands": 50}}
    law = PR.staging_fit()
    for r in PR.STAGING:
        assert abs(PR._at(law["peak_gib"], [r[2] / 1e6]) / r[4] - 1) < 0.1
    floor = max(r[5] for r in PR.REGISTERED_STAGING)
    assert PR.staging(n["r"], law)["peak_gib"] == floor and PR.staging(n["g"], law)["peak_gib"] < floor


def test_pieces_pmf():
    k0, p = P.hyper(30, 7, 12)
    y = PR.pieces_pmf(30, 12, {3: 7})
    assert sum(y) == pytest.approx(1) and {i: v for i, v in enumerate(y) if v} == pytest.approx(
        {3 * (k0 + j): v for j, v in enumerate(p)})


def test_a_tiny_audit_unsplit_and_split():
    """N = 6 units, 4 of 3 nonzeros and 2 of 10; K = 3 drawn; dp2 at 2 slots; one second per session. Split at 5, each b is
    2 pieces, so its bucket takes X_b sessions: E = 1, and X_b <= 2 gives 2 public sessions."""
    units = [P.Unit("a", "t", 4, 3, 3), P.Unit("b", "t", 2, 10, 10)]
    coef = {"rss_gib": (1.0, 0.0, 0.0, 0.0), "prove_s": (1.0, 0.0, 0.0, 0.0), "wall_s": (0.0, 0.0, 0.0, 0.0)}
    a = PR.audit(6, 3, units, {}, coef, slots=2, k=2)
    assert [s["Z"] for s in a["body"]] == [3, 10] and not a["tail"]
    assert (a["sessions_mean"], a["sessions_public"], a["prover_h_mean"]) == pytest.approx((2.0, 3, 2.0 / 3600))
    a = PR.audit(6, 3, units, {}, coef, limit=5, slots=2, k=2)
    assert [(s["Z"], s["pieces"]) for s in a["body"]] == [(3, 1), (5, {2: 2})]
    assert [s["sessions_mean"] for s in a["body"]] == pytest.approx([1.2, 1.0])
    assert [s["sessions_public"] for s in a["body"]] == [2, 2]


def test_argmax_tree_members():
    t = U.argmax_tree(130)
    assert [(m["member"], m["n"]) for m in t["members"]] == [
        ("ArgmaxLeaf_v1{G=64,FIRST=1,FINAL=0}", 1), ("ArgmaxLeaf_v1{G=64,FIRST=0,FINAL=0}", 1),
        ("ArgmaxLeaf_v1{G=2,FIRST=0,FINAL=0}", 1), ("ArgmaxMerge_v1{G=3,IW=6,FINAL=1}", 1)]
    assert t["nnz"] == sum(m["nnz"] for m in t["members"]) == 138_323


def test_retree_supersedes_the_generic_count():
    tree = {"stage": "TokenSelectStage", "V": 8, "units": 2, "rows": 10, "ands": 6, "nnz_a": 40, "nnz_b": 30, "nnz": 80,
            "nnz_per_logit": 10.0, "members": []}
    g = {"units": 5, "population": 100, "sampled_mean": 2.0, "sampled_q": 4,
         "per_unit": {"nnz": 10 ** 6, "nnz_generic": 10 ** 6}, "nnz_range": {"min": 1, "max": 2}}
    inv = {"rows": [{"template": "token-select-greedy", "subcircuit_parameters": {"V": 8}, "subcircuits_per_vu": 1,
                     "source": "generic", "vu": dict(g), "subcircuit": dict(g)}]}
    U.retree(inv, {("token-select-greedy", 8): tree})
    r = inv["rows"][0]
    c = r["subcircuit"]["per_unit"]
    assert r["source"] == "tree" and (c["nnz"], c["committed"], c["nnz_superseded"], c["nnz_generic"]) == (80, 10, 10 ** 6, 10 ** 6)
    assert r["subcircuit"]["audit_nnz_q"] == 320 and "nnz_range" not in r["subcircuit"]
    U.retree(inv, {("token-select-greedy", 8): tree})
    assert r["subcircuit"]["per_unit"]["nnz_superseded"] == 10 ** 6
