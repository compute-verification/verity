"""F6's prover column: the fit, one session priced from fixed coefficients, the cap, a tiny audit unsplit and split exactly,
and token-select's argmax tree."""
from __future__ import annotations

import pytest

from verity_numerical.bench import unit_costs as U
from verity_numerical.bench import unit_costs_padding as P
from verity_numerical.bench import unit_costs_prover as PR

COEF = {"rss_gib": (1.0, 2.0, 3.0), "prove_s": (0.5, 4.0, 8.0), "wall_s": (0.0, 0.0, 0.0)}
TINY = {"bytes_per_nnz": 4, "glue": 0}


def test_lstsq_recovers_an_affine_law():
    xs = [(0, 0), (1, 0), (0, 1), (2, 3), (5, 1)]
    assert PR.lstsq(xs, [1 + 2 * a + 3 * b for a, b in xs]) == pytest.approx([1, 2, 3])


def test_the_fit_over_the_source_runs():
    f = PR.fit()
    assert len(f["points"]) == len(PR.SOURCE) == 62
    assert f["coef"]["rss_gib"] == pytest.approx([3.526, 4.000, 3.040], abs=2e-3)
    assert f["max_residual"]["rss_gib"] < 8


def test_a_session_from_fixed_coefficients():
    # 2^20 nonzeros at 16 per row: 2^16 rows, a 2^19 slot, the 2^22 floor; D = 16 2^22 = 2^26; text 16 x 4 x 2^20 = 2^26 bytes
    s = PR.session(2 ** 20, 16, COEF, **TINY)
    assert (s["k_log"], s["D"], s["circuit_bytes"]) == (22, 2 ** 26, 2 ** 26)
    assert (s["rss_gib"], s["prove_s"]) == pytest.approx((1 + 2 / 16 + 3 / 16, 0.5 + 4 / 16 + 8 / 16))
    # 2^30: 2^26 rows, k_log 27 + 2; D = 2^33 (8 x 2^30); text 2^36 (64 GiB)
    s = PR.session(2 ** 30, 16, COEF, **TINY)
    assert (s["k_log"], s["rss_gib"]) == (29, 1 + 2 * 8 + 3 * 64)
    assert PR.cap(16, COEF, 209, **TINY) == 2 ** 30 and PR.cap(16, COEF, 208.9, **TINY) == 2 ** 29
    assert PR.cap(16, COEF, 1, **TINY) is None


def test_pieces_pmf():
    k0, p = P.hyper(30, 7, 12)
    y = PR.pieces_pmf(30, 12, {3: 7})
    assert sum(y) == pytest.approx(1) and {i: v for i, v in enumerate(y) if v} == pytest.approx(
        {3 * (k0 + j): v for j, v in enumerate(p)})


def test_a_tiny_audit_unsplit_and_split():
    """N = 6 units, 4 of 3 nonzeros and 2 of 10; K = 3 drawn; dp2 at 2 slots; one second per session. Split at 5, each b is
    2 pieces, so its bucket takes X_b sessions: E = 1, and X_b <= 2 gives 2 public sessions."""
    units = [P.Unit("a", "t", 4, 3, 3), P.Unit("b", "t", 2, 10, 10)]
    coef = {"rss_gib": (1.0, 0.0, 0.0), "prove_s": (1.0, 0.0, 0.0), "wall_s": (0.0, 0.0, 0.0)}
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
