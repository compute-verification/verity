"""Red-team regression tests for the certified bounds (:mod:`accumulation.redteam`).

1. Torture suite: every hand-derived point of :mod:`accumulation.redteam.torture` -- ``L <= I* <= U_lit``,
   the materialised literal plan is legal per ``is_legal``, and ``I*`` matches the hand value.  Points that
   hit a *documented* defect (``Case.expect_violation`` / the ``upper_bound`` "raises instead of falling
   back" gap) are reported as ``xfail`` so the suite stays green while the defects are open and turns red
   the day they are fixed (strict: an unexpectedly clean point on a case that documents a defect fails).
2. A small seeded fuzz run (:mod:`accumulation.redteam.fuzz`) -- no undocumented violation kinds.
3. Harness self-tests: the materialiser reproduces the plan's cost on a clean case, the monotonicity
   checker fires, the shrinker reaches the known 8-gate minimum of the repeated-row counterexample.

Run: ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_redteam.py -q``  (about 3 minutes; add
``-m "not slow"`` to skip the 60-second cases).
"""

from __future__ import annotations

import random

import pytest

from accumulation.redteam import fuzz, torture
from accumulation.redteam.circuits import Circuit, Op, Param, build
from accumulation.redteam.harness import BIG, Record, evaluate, monotonicity_violations

# ``upper_bound`` raises ValueError instead of falling back to a finer tiling in these situations; a legal
# partition exists (the solver finds one).  Robustness gap, not a soundness defect -- tracked as xfail.
U_GAP = {"U_unavailable"}
# implied by a documented primary defect
IMPLIED = {
    "L_gt_Istar": {"Urec_lt_L", "Ulit_lt_L", "L_gt_found", "Lnoprog_gt_Istar", "Lnoprog_gt_found"},
    "Ulit_illegal": {"Ulit_lt_Istar", "Ulit_lt_L", "Urec_lt_L"},
}


def _known(case: torture.Case) -> set[str]:
    known = set(case.expect_violation) | U_GAP
    for v in list(known):
        known |= IMPLIED.get(v, set())
    return known


def _fmt(v) -> str:
    if v is None:
        return "-"
    return "inf" if isinstance(v, int) and v >= BIG else str(v)


def _points():
    for c in torture.CASES:
        for F, X, G in c.grid + torture.g_points(c):
            marks = [pytest.mark.slow] if c.time_limit > 30 else []
            yield pytest.param(c, F, X, G, id=f"{c.name}[F={_fmt(F)},G={_fmt(G)},X={_fmt(X)}]", marks=marks)


# ---------------------------------------------------------------------------------------------------------
# 1. torture suite
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case,F,X,G", list(_points()))
def test_torture_point(case: torture.Case, F: int, X: int, G) -> None:
    rec = torture.run_point(case, F, X, G)
    msg = (f"{case.name} F={_fmt(F)} G={_fmt(G)} X={_fmt(X)}: L={rec.L} I*={rec.Istar}{'' if rec.Istar_optimal else '?'} "
           f"U={rec.U} U_lit={rec.Ulit} legal={rec.Ulit_legal} {rec.Ulit_legal_why} viol={rec.violations}")
    assert rec.hand_ok is not False, "hand value mismatch: " + msg
    assert not rec.Istar_timeout, "exact solver timed out: " + msg
    if rec.violations:
        unknown = set(rec.violations) - _known(case)
        assert not unknown, "UNDOCUMENTED violation: " + msg
        pytest.xfail("documented defect: " + msg)


@pytest.mark.parametrize("case", [c for c in torture.CASES if c.expect_violation and c.time_limit <= 30],
                         ids=lambda c: c.name)
def test_documented_defect_still_present(case: torture.Case) -> None:
    """Turns red when a documented defect gets fixed -- then drop ``expect_violation`` from the case."""
    seen = {v for F, X, G in case.grid for v in torture.run_point(case, F, X, G).violations}
    assert seen & set(case.expect_violation), \
        f"{case.name}: expected {case.expect_violation} somewhere on the grid, saw {seen or 'nothing'} -- fixed?"


def test_torture_monotone() -> None:
    """``L``, ``U``, ``U_lit``, ``I*`` non-increasing in ``X``, ``F`` and ``G`` along every case grid."""
    bad = []
    for c in torture.CASES:
        if c.time_limit > 30:
            continue
        recs = [torture.run_point(c, F, X, G) for F, X, G in c.grid + torture.g_points(c)]
        bad += [f"{c.name}: {m}" for m in monotonicity_violations(recs)]
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------------------------------------
# 2. seeded fuzz smoke run
# ---------------------------------------------------------------------------------------------------------

def test_fuzz_smoke(tmp_path) -> None:
    st = fuzz.run(11, 12, gate_cap=30, max_ops=4, time_limit=3.0, do_shrink=False, out_dir=str(tmp_path),
                  verbose=False)
    assert st.circuits >= 10 and st.points >= 40
    known = U_GAP | {"Ulit_illegal", "Ulit_lt_Istar", "Ulit_lt_L", "Urec_lt_L", "Lnoprog_gt_Istar"}
    unknown = {v for v in st.violations if v not in known and v != "L_gt_Istar"}
    assert not unknown, f"undocumented violation kinds: {unknown} ({st.violations})"
    if "L_gt_Istar" in st.violations:
        # only the two documented L defects (overlapping / repeated reads) may produce this
        import json
        for line in open(tmp_path / "fuzz_11.jsonl"):
            d = json.loads(line)
            if d.get("kind") == "point" and "L_gt_Istar" in d["violations"]:
                assert "shared" in d["family"] or "view" in d["family"], \
                    f"L > I* on a family without overlapping reads: {d['family']} F={d['F']} X={d['X']}\n{d['snippet']}"
    if st.violations:
        pytest.xfail(f"documented defects hit by the fuzzer: {st.violations}")


def test_fuzz_reproducible() -> None:
    a = fuzz.gen_circuit(fuzz.circuit_rng(5, 7), name="a")
    b = fuzz.gen_circuit(fuzz.circuit_rng(5, 7), name="a")
    assert a.to_python() == b.to_python()
    assert all(fuzz.gen_circuit(random.Random(s), gate_cap=60).n_gates() <= 60 for s in range(40))


# ---------------------------------------------------------------------------------------------------------
# 3. harness self-tests
# ---------------------------------------------------------------------------------------------------------

def test_materialiser_matches_plan_on_clean_case() -> None:
    c = next(k for k in torture.CASES if k.name == "d_two_layer_M2")
    rec = evaluate(build(c.circuit), BIG, 12, time_limit=20.0)
    assert rec.Ulit_materialized and rec.Ulit_legal is True
    assert rec.Ulit_cost == rec.Ulit == rec.Istar == 36 and rec.L == 20
    assert not rec.violations


def test_opgraph_share_counts_single_leaf() -> None:
    from accumulation.graph.opgraph import extract
    c = next(k for k in torture.CASES if k.name == "t9_scalar_read_by_two_matmuls")
    g = extract(build(c.circuit))
    tid = g.params["p"]
    assert g.share(tid) == 2, f"share={g.share(tid)} ranges={[e.ranges for o in g.ops for e in o.inputs if e.src == tid]}"
    e = next(e for o in g.ops for e in o.inputs if e.src == tid)
    assert e.covers(g.tensors[tid].leaves) == 1


def test_monotonicity_checker_fires() -> None:
    r1 = Record("f", "n", 4, 8, 1, L=10, Istar=10, Istar_optimal=True)
    r2 = Record("f", "n", 4, 16, 1, L=12, Istar=9, Istar_optimal=True)
    out = monotonicity_violations([r1, r2])
    assert any(m.startswith("L not monotone in X") for m in out) and len(out) == 1
    # G axis: None is the largest G; I* may only grow when G shrinks
    r3 = Record("f", "n", 4, 8, 1, G=None, Istar=10, Istar_optimal=True)
    r4 = Record("f", "n", 4, 8, 1, G=6, Istar=9, Istar_optimal=True)
    out = monotonicity_violations([r3, r4])
    assert len(out) == 1 and "Istar not monotone in G" in out[0]
    assert not monotonicity_violations([r3, Record("f", "n", 4, 8, 1, G=6, Istar=14, Istar_optimal=True)])


def test_validation_rows_and_aggregate() -> None:
    """``validation.row_from_dict`` / ``aggregate`` on synthetic records: soundness verdicts, ratios, L@G=None marker."""
    from accumulation.redteam import validation as V
    ok = Record("k", "c1", BIG, 1 << 20, 16, G=6, work=12, L=16, L_G=6, Istar=24, Istar_optimal=True, Ulit=24, Ulit_legal=True)
    loose = Record("k", "c2", BIG, 1 << 20, 16, G=3, work=12, L=8, Istar=32, Istar_optimal=True, Ulit=32, Ulit_legal=True)
    bad = Record("t", "c3", BIG, 1 << 20, 8, L=12, Istar=8, Istar_optimal=True, Ulit=8, Ulit_legal=True, violations=["L_gt_Istar"])
    gap = Record("t", "c4", BIG, 8, 8, L=4, Istar=8, Istar_optimal=True, Ulit_error="ValueError: no legal tile")
    rows = V.rows_from_records([ok, loose, bad, gap])
    assert [r["sound"] for r in rows] == [True, True, False, True]
    assert rows[0]["L_over_Istar"] == pytest.approx(16 / 24) and rows[0]["U_over_Istar"] == 1.0
    assert rows[0]["L_at_G_none"] is False and rows[1]["L_at_G_none"] is True
    assert rows[3]["robustness_gap"] is True
    a = V.aggregate(rows)
    assert a["judged"] == 4 and a["soundness_violations"] == 1 and a["robustness_gaps_U_unavailable"] == 1
    assert a["binding_G_points"] == 2 and a["binding_G_points_L_at_G_none"] == 1
    assert a["L_over_Istar"]["n"] == 4 and a["L_over_Istar"]["min"] == pytest.approx(0.25)
    md = V.table_md(rows)
    assert "L@G=None" in md and "unavailable" in md and "(inf, 6, inf)" in md


def test_G_binds_exact_and_bounds_bracket() -> None:
    """The hand-checked ``2x2x2`` matmul under ``G`` (torture ``k1``): ``I*`` matches the hand derivation for every
    ``G``, ``L <= I* <= U_lit`` wherever ``U`` exists, and the materialised literal plan respects ``work(R) <= G``."""
    from accumulation.exact.solve import ru_work
    case = next(k for k in torture.CASES if k.name == "k1_matmul_G_forced_split")
    bp, flat, g, og, _ = torture._prepare(case)
    for (F, X, G), (kind, v) in case.hand.items():
        rec = torture.run_point(case, F, X, G)
        assert rec.hand_ok is True, (G, rec.Istar, kind, v)
        if rec.Istar is not None:
            assert rec.L is not None and rec.L <= rec.Istar, (G, rec.L, rec.Istar)
            if rec.Ulit is not None:
                assert rec.Istar <= rec.Ulit and rec.Ulit_legal is True, (G, rec.Istar, rec.Ulit, rec.Ulit_legal_why)
    # I* strictly increases as G tightens across the binding values
    vals = [torture.run_point(case, BIG, 1 << 20, G).Istar for G in (None, 11, 6, 3, 1)]
    assert vals == [16, 20, 24, 32, 64]
    # G is a real constraint on the literal plan's materialisation
    from accumulation.bounds.upper import upper_bound
    from accumulation.redteam.harness import materialize
    ub = upper_bound(g, BIG, 1 << 20, G=6, program=bp.program, recompute=False)
    assign, _ = materialize(ub.plan, g, flat, og)
    assert max(ru_work(flat, assign).values()) <= 6


def test_shrinker_reaches_minimal_reprow_counterexample() -> None:
    """Start from a 3-op circuit containing the repeated-row matmul and shrink to the 8-gate core."""
    circ = Circuit(params=[Param("x", (1, 2), "token"), Param("W", (1, 2), "accumulated"), Param("y", (2, 2), "token")],
                   ops=[Op("matmul", (("p", "x"), ("p", "W", "reprow", 0, 2))), Op("add", (("p", "y"), ("p", "y"))),
                        Op("colsum", (("o", 1),))], name="shrinkme")
    rec = evaluate(build(circ), BIG, BIG, time_limit=20.0)
    assert "L_gt_Istar" in rec.violations, rec
    small, srec, _ = fuzz.shrink(circ, BIG, BIG, "L_gt_Istar", time_limit=20.0, exact_limit=200)
    assert "L_gt_Istar" in srec.violations
    assert len(small.ops) == 1 and small.ops[0].kind == "matmul" and srec.gates <= 8
    assert srec.L > srec.Istar
