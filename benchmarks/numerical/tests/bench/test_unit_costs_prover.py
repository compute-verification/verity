"""F6's prover column: the fit and the posted fit it replaces, the registered check, one session priced from fixed
coefficients, the range and memory caps, the slot sumcheck's term, netlists, staging, a tiny audit unsplit and split
exactly, and token-select's argmax tree."""
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
    """K1 10 ns, 2 reps, 8 B of committed bits and 16 B of working set per entry. At 2^20, 16 slots each its own template of
    2^20: 2^24 entries, 10e-9 x 2^24 x 2 s, (8 x 16 x 2^20 + 16 x 2^24) / 2^30 = 0.375 GiB. At 2^28 the sumcheck adds
    32 + 64 GiB to Flock's 53, so the cap at 149 GiB stays 2^28 and drops to 2^27 below it."""
    S = {"k1_ns": 10.0, "band": (5.0, 20.0), "reps": 2, "bits_bytes": 8.0, "work_bytes": 16.0}
    flock, s = PR.session(2 ** 20, 16, COEF, **TINY), PR.session(2 ** 20, 16, COEF, slot=S, **TINY)
    assert (flock["sumcheck_s"], flock["total_s"], flock["peak_gib"]) == (0, flock["prove_s"], flock["rss_gib"])
    assert (s["template"], s["entries"]) == (2 ** 20, 2 ** 24) and s["sumcheck_s"] == pytest.approx(10e-9 * 2 ** 25)
    assert s["sumcheck_gib"] == pytest.approx(0.375) and s["total_s"] == pytest.approx(s["prove_s"] + s["sumcheck_s"])
    assert s["peak_gib"] == pytest.approx(s["rss_gib"] + 0.375)
    assert PR.cap(16, COEF, 149, slot=S, **TINY) == 2 ** 28 and PR.cap(16, COEF, 148.9, slot=S, **TINY) == 2 ** 27
    # every slot padded to the session's largest template (at most Z); the stacked table holds T = 2 templates at its size
    s = PR.session(2 ** 20, 16, COEF, nets=[4 * 1000, 4 * 3000], slot=S, **TINY)
    assert (s["template"], s["entries"]) == (3000, 16 * 3000)
    assert s["sumcheck_gib"] == pytest.approx((8 * 2 * 3000 + 16 * 16 * 3000) / 2 ** 30)
    assert PR.session(2 ** 20, 16, COEF, nets=[4 * 2 ** 30], slot=S, **TINY)["template"] == 2 ** 20
    # K1 per ν: the point at the least ν' ≥ the template's, the last beyond
    pts = ((20, 1.0), (28, 2.0))
    assert [PR.k1(pts, t) for t in (3000, 2 ** 20, 2 ** 20 + 1, 2 ** 29)] == [1.0, 1.0, 2.0, 2.0] and PR.k1(7, 1) == 7.0
    units = [P.Unit("a", "t", 4, 3, 3), P.Unit("b", "t", 2, 10, 10)]
    a0 = PR.audit(6, 3, units, {}, COEF, slots=2, k=2)
    a = PR.audit(6, 3, units, {}, COEF, slots=2, k=2, slot=S)
    a2 = PR.audit(6, 3, units, {}, COEF, slots=2, k=2, slot=dict(S, k1_ns=20.0))
    assert a["prover_h_mean"] - a["sumcheck_h_mean"] == pytest.approx(a0["prover_h_mean"])
    assert a["sumcheck_h_mean"] == pytest.approx(sum(b["sessions_mean"] * 10e-9 * 2 * b["Z"] * 2 for b in a["body"]) / 3600)
    assert a2["sumcheck_h_public"] == pytest.approx(2 * a["sumcheck_h_public"])
    assert PR.hours(a)["share_mean"] == pytest.approx(a["sumcheck_h_mean"] / a["prover_h_mean"])


def test_dp4t_keys_buckets_by_template():
    """N = 1000, K = 160. Two registered types share a template (r1, r2: one tc-step-like unit) and differ in Z; a bulk
    generic type g has its own; a rare generic type x (0.16 draws per audit) pads dp2's top bucket to its template (at most Z), so
    dp4t gives it a bucket of its own (B = 1: its quantile sessions of one slot cost less than one session of 16)."""
    units = [P.Unit("r1", "t", 500, 1000, 1000), P.Unit("r2", "t", 300, 2000, 2000), P.Unit("g", "u", 199, 1500, 1500),
             P.Unit("x", "v", 1, 1900, 1900)]
    nets = {"r1": {"kind": "registered", "nnz": 100, "bytes": 400}, "r2": {"kind": "registered", "nnz": 100, "bytes": 400},
            "g": {"kind": "generic", "nnz": 1500, "bytes": 6000}, "x": {"kind": "generic", "nnz": 5000, "bytes": 20000}}
    assert PR.template_class(units[0], nets) == ("registered", 7) and PR.template_class(units[3], nets) == ("generic", 13)
    groups, rare = PR.menu_t(1000, 160, units, nets, budget=3)
    assert [(c, Z, [u.label for u in us]) for c, Z, us in groups] == [
        (("generic", 11), 1500, ["g"]), (("registered", 7), 1000, ["r1"]), (("registered", 7), 2000, ["r2"])]
    assert [u.label for u in rare] == ["x"]
    S = {"k1_ns": 10.0, "band": (5.0, 20.0), "reps": 2, "bits_bytes": 8.0, "work_bytes": 16.0}
    a = PR.audit(1000, 160, units, {}, COEF, menu="dp4t", nets=nets, slot=S, **TINY)
    assert a["buckets"] == {"body": 3, "rare": 1, "tail": 0} and a["rare"][0]["slots"] == 1
    assert a["rare"][0]["sessions_public"] == P.upper_q(1000, 1, 160)
    assert {x["template_class"]: x["session"]["template"] for x in a["body"]} == {"generic 2^11": 1500, "registered 2^7": 100}
    d = PR.audit(1000, 160, units, {}, COEF, k=2, nets=nets, slot=S, **TINY)
    assert max(x["session"]["template"] for x in d["body"]) == 2000 and a["sumcheck_h_mean"] < d["sumcheck_h_mean"]
    assert a["padding"]["zpad"] == pytest.approx(sum(x["Z"] * 160 * x["units"] / 1000 for x in a["body"] + a["rare"])
                                                 / sum(160 * u.n / 1000 * u.z_mean for u in units))


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
