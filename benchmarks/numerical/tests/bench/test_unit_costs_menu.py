"""F6's pinned menu: each bucket's B is the cheapest at its public session count among those that are in range, fit one
node and stay in the law's fitted D, with the outer proof's cost when given and on the inner alone without it."""
from __future__ import annotations

import math

import pytest

from verity_numerical.bench import unit_costs_menu as M
from verity_numerical.bench import unit_costs_padding as P

#: one second per session at any size; RSS 1 GiB + 2 GiB per 2^30 bits of D
COEF = {"rss_gib": (1.0, 2.0, 0.0, 0.0), "prove_s": (1.0, 0.0, 0.0, 0.0), "wall_s": (0.0, 0.0, 0.0, 0.0)}
TINY = {"bytes_per_nnz": 4, "glue": 0}


def _bucket(units, Z):
    return {"part": "body", "class": "c", "Z": Z, "units": units, "pieces": False}


def test_the_cheapest_public_b_that_fits_one_node():
    """N = 100, a type of 60 units, K = 50 drawn: its public count q needs ceil(q / B) one-second sessions, fewer at larger B.
    A 2^22-bit slot gives RSS 1 + B / 128 GiB: within 1.2 GiB, B = 16 is the largest that fits and so the cheapest."""
    b = _bucket([P.Unit("a", "t", 60, 3, 3)], 3)
    opts = {B: M.inner(100, 50, b, B, {}, COEF, **TINY) for B in (1, 4, 16, 64)}
    q = P.upper_q(100, 60, 50)
    assert [opts[B]["sessions_public"] for B in (1, 4, 16, 64)] == [q, -(-q // 4), -(-q // 16), -(-q // 64)]
    assert [opts[B]["peak_gib"] for B in (1, 4, 16, 64)] == pytest.approx([1 + B / 128 for B in (1, 4, 16, 64)])
    assert M.pick(opts, gib=math.inf) == 64 and M.pick(opts, gib=1.2) == 16
    assert M.excluded(opts[64], gib=1.2) == "inner 2 GiB > 1"


def test_a_slot_past_k_log_27_runs_at_no_b():
    """2^26 nonzeros at one per row: k_log 29, out of range at every B."""
    b = _bucket([P.Unit("big", "t", 60, 2.0 ** 26, 2.0 ** 26)], 2.0 ** 26)
    opts = {B: M.inner(100, 50, b, B, {"big": 1.0}, COEF, **TINY) for B in (1, 16)}
    assert {o["k_log"] for o in opts.values()} == {29}
    with pytest.raises(ValueError, match="k_log 29 > 27"):
        M.pick(opts)


def test_pick_on_inner_plus_outer():
    o = {"in_range": True, "k_log": 22, "peak_gib": 1.0, "inner_m": 30}
    opts = {1: dict(o, inner_h_public=4.0), 4: dict(o, inner_h_public=2.0), 16: dict(o, inner_h_public=1.0, in_range=False),
            64: dict(o, inner_h_public=1.0, inner_m=36), 256: dict(o, inner_h_public=2.0)}
    assert M.pick(opts) == 4
    assert M.excluded(opts[64]) == "D 2^36 > 2^35"
    opts[4]["outer_h_public"], opts[256]["outer_h_public"] = 3.0, 0.5
    assert M.pick(opts) == 256
    opts[256]["outer_rss_gib"] = 2000.0
    assert M.pick(opts) == 1 and M.excluded(opts[256]) == "outer 2000 GiB > 1716"


def test_the_menu_with_and_without_the_outer_input():
    b = _bucket([P.Unit("a", "t", 60, 3, 3)], 3)
    b["by_B"] = {B: M.inner(100, 50, b, B, {}, COEF, **TINY) for B in (1, 16)}
    pl = [{"workload": "w", "N": 100, "cap": 2.0 ** 29, "buckets": [b]}]
    alone = M.menu(pl, 50)
    e = alone["models"][0]["entries"][0]
    assert "inner hours alone" in alone["chosen_on"] and e["slots"] == 16 and e["outer_h_public"] is None
    assert alone["models"][0]["totals"]["outer_h_public"] is None
    outer = {"buckets": [{"workload": M.row_id("w", 0, 1), "prove_s": 1.0, "rss_gib": 1.0},
                         {"workload": M.row_id("w", 0, 16), "prove_s": 3600.0 * 100, "rss_gib": 1.0, "refused": ["x"]}]}
    m = M.menu(pl, 50, outer)
    e = m["models"][0]["entries"][0]
    assert e["slots"] == 1 and e["outer_h_public"] == pytest.approx(e["sessions_public"] / 3600)
    assert e["total_h_public"] == pytest.approx(e["inner_h_public"] + e["outer_h_public"])
    assert m["models"][0]["totals"]["refused_entries"] == 0 and isinstance(e["options"]["16"], float)
    assert M.render(m) == M.render(M.menu(pl, 50, outer))


def test_a_shape_takes_its_largest_inner_session():
    """llama32-1b/b1/i256's clash: two entries of the shape (256, 2^29), at (m 35, k_log 27) and (32, 24). Both take the
    larger session and the larger outer, and are priced at them; an entry of another shape keeps its own."""
    def e(slots, m, k, s, outer_s, sessions, refused=()):
        return {"slots": slots, "bound": 2.0 ** 29, "k_log": k, "inner_m": m, "inner_peak_gib": float(m),
                "inner_session_s": s, "outer_session_s": outer_s, "outer_rss_gib": outer_s / 10,
                "outer_refused": list(refused), "sessions_mean": sessions / 2, "sessions_public": sessions}
    es = [e(256, 35, 27, 3600.0, 360.0, 1), e(256, 32, 24, 450.0, 720.0, 2, ["x"]), e(64, 30, 22, 100.0, 36.0, 3)]
    M.one_session_per_shape(es)
    a, b, c = es
    assert {k: b[k] for k in M.SESSION} == {k: a[k] for k in M.SESSION} == {
        "k_log": 27, "inner_m": 35, "inner_peak_gib": 35.0, "inner_session_s": 3600.0}
    assert a["outer_session_s"] == b["outer_session_s"] == 720.0 and a["outer_refused"] == b["outer_refused"] == ["x"]
    assert b["inner_h_public"] == pytest.approx(2.0) and b["outer_h_public"] == pytest.approx(0.4)
    assert b["total_h_mean"] == pytest.approx(1.2)
    assert (c["inner_m"], c["inner_h_public"], c["outer_h_public"]) == (30, pytest.approx(300 / 3600), pytest.approx(0.03))


def test_a_smaller_entry_is_costed_as_its_own_units_in_the_shape_session():
    """Two buckets of one shape (Z = 2^21, B = 16) at ρ 1 and ρ 1/4, k_log 24 and 26: the denser one is resized to k_log 26
    at its own units' cost, through the plan's ``at_rho``."""
    coef, kw = dict(COEF, prove_s=(0.0, 1e-6, 0.0, 0.0)), dict(TINY, slot=None)
    rho = {"a": 1.0, "b": 1 / 4}
    pl = [{"workload": "w", "N": 100, "cap": 2.0 ** 29, "buckets": []}]
    for label in ("a", "b"):
        b = _bucket([P.Unit(label, "t", 60, 2.0 ** 21, 2.0 ** 21)], 2.0 ** 21)
        b["by_B"] = {16: M.inner(100, 50, b, 16, rho, coef, **kw)}
        b["at_rho"] = lambda B, r, b=b: M.inner(100, 50, b, B, {u.label: r for u in b["units"]}, coef, **kw)
        pl[0]["buckets"].append(b)
    small, big = (b["by_B"][16] for b in pl[0]["buckets"])
    assert (small["k_log"], big["k_log"]) == (24, 26)
    a, b = M.menu(pl, 50)["models"][0]["entries"]
    assert (a["k_log"], a["inner_m"]) == (b["k_log"], b["inner_m"]) == (big["k_log"], big["inner_m"])
    assert a["inner_session_s"] == pytest.approx(big["session_s"]) and a["inner_h_public"] > small["inner_h_public"]


def test_the_lean_menu_lists_each_model_s_entries():
    mn = {"format": "verity/f6-menu/v1", "K": 50, "models": [
        {"workload": "w1", "entries": [{"slots": 16, "bound": 3.0, "inner_m": 30, "k_log": 22, "claims": 8, "positions": 3,
                                        "unit_types": ["a", "b"]},
                                       {"slots": 4, "bound": 7.0, "inner_m": 28, "k_log": 26, "claims": 6, "positions": 1,
                                        "unit_types": ["c"]}]},
        {"workload": "w2", "entries": [{"slots": 1, "bound": 9.0, "inner_m": 26, "k_log": 24, "claims": 16, "positions": 2,
                                        "unit_types": ["d"]}]}]}
    text = M.lean(mn, "SRC")
    assert text.startswith("import VStar.Menu\n") and "\nSRC\n(verity/f6-menu/v1, K = 50); do not edit. -/" in text
    assert ('def pinned : List (String × List Entry) := [\n  ("w1",\n    [⟨16, 3, 30, 22, 8, 3, 2⟩,\n     ⟨4, 7, 28, 26, 6, 1, 1⟩]),\n'
            '  ("w2",\n    [⟨1, 9, 26, 24, 16, 2, 1⟩])]\n') in text
    mn["models"][1]["entries"][0]["bound"] = 9.5
    with pytest.raises(ValueError, match="not a whole number"):
        M.lean(mn, "SRC")


def test_the_pinned_lean_menu_is_generated_from_the_pinned_menu_json():
    """`VStar.Menu.Pinned` is `lean` of F6's pinned `menu.json` (`data/f6-menu.json`, the one in
    art:bb9b72df0377d54ad001bb52a9d24b6152e83cdefb896d02fbbce26bc281dffa), byte for byte."""
    import json
    from pathlib import Path

    here = Path(__file__).resolve()
    pinned = here.parents[4] / "verity/core/service/flock/VStar/Menu/Pinned.lean"
    mn = json.loads((here.parent / "data/f6-menu.json").read_text())
    assert M.lean(mn, "art:bb9b72df0377d54ad001bb52a9d24b6152e83cdefb896d02fbbce26bc281dffa (menu.json)") == pinned.read_text()


def _entry(slots, regions, m=35, k=27, outer_s=100.0, sessions=1):
    return {"slots": slots, "bound": 2.0 ** 29, "k_log": k, "inner_m": m, "inner_peak_gib": 1.0, "inner_session_s": 10.0,
            "outer_session_s": outer_s, "outer_rss_gib": 20.0, "outer_refused": [], "sessions_mean": sessions,
            "sessions_public": sessions, "regions": regions, "claims": None}


def test_a_shape_takes_its_largest_region_count_and_v_star_checks_the_reps_algebra_at_it():
    """llama32-1b/b1/i256's (256, 2^29): regions 4, 7 and 2 pad to 7, claims 16; another shape keeps its own 3, claims 8.
    V*'s check of the reps' algebra is one statement of 2 x products x rows-per-product rows, priced at the input's one-block
    statement of its k_log, and a refusal of the shape's parts is the entry's outer refusal."""
    es = [_entry(256, 4), _entry(256, 7), _entry(256, 2), _entry(64, 3, m=30, k=22)]
    M.one_session_per_shape(es)
    assert [(e["regions"], e["claims"]) for e in es] == [(7, 16)] * 3 + [(3, 8)]
    seen = []

    def algebra(m, k, claims):
        seen.append((m, k, claims))
        return (5000, "V* parts: too wide") if claims > 14 else (2000, None)

    M.inner_algebra(es, {26: (10.0, 14.0), 27: (21.0, 25.0)}, algebra)
    assert set(seen) == {(35, 27, 16), (30, 22, 8)}
    a, d = es[0], es[3]
    assert a["inner_algebra_rows"] == pytest.approx(2 * 5000 * M.ROWS_PER_PRODUCT) and a["inner_algebra_k_log"] == 27
    assert (a["outer_session_s"], a["outer_rss_gib"], a["outer_refused"]) == (121.0, 25.0, ["V* parts: too wide"])
    assert a["outer_h_public"] == pytest.approx(121.0 / 3600)
    assert (d["inner_algebra_k_log"], d["outer_session_s"], d["outer_refused"]) == (26, 110.0, [])
    with pytest.raises(ValueError, match="no one-block statement at k_log 27"):
        M.inner_algebra([_entry(256, 7) | {"claims": 16}], {26: (10.0, 14.0)}, algebra)


def test_a_shape_takes_its_largest_positions_from_the_rows_instances():
    """Each entry's positions are its row's ``instances_max`` at the chosen B; two entries of one shape (B = 1, Z = 3) at 5
    and 2 positions both take 5, and a menu without the rows carries none."""
    pl = [{"workload": "w", "N": 100, "cap": 2.0 ** 29, "buckets": []}]
    for label in ("a", "b"):
        b = _bucket([P.Unit(label, "t", 60, 3, 3)], 3)
        b["by_B"] = {1: M.inner(100, 50, b, 1, {}, COEF, **TINY)}
        pl[0]["buckets"].append(b)
    rows = {M.row_id("w", 0, 1): 5, M.row_id("w", 1, 1): 2}
    assert [e["positions"] for e in M.menu(pl, 50, positions=rows)["models"][0]["entries"]] == [5, 5]
    assert [e["positions"] for e in M.menu(pl, 50)["models"][0]["entries"]] == [None, None]
    es = [_entry(256, 4) | {"positions": 3}, _entry(256, 7) | {"positions": 9}, _entry(64, 3) | {"positions": 1}]
    M.one_session_per_shape(es)
    assert [e["positions"] for e in es] == [9, 9, 1]


def test_v_star_stages_the_reps_algebra_at_16_claims_and_refuses_past_the_lone_part_bound():
    """proofs' blocker: at m 35, k_log 27 and 16 claims the
    batched-T residual alone needs more sha512x3 slots than a part holds, so it takes a part of its own (at most
    ``MAX_LONE_SLOTS``) and V* stages the algebra in 3 parts. At m 26, k_log 24 and 28 claims it needs 300 alone, past
    that bound, and V* refuses (the control of `test_vstar_menu.py`)."""
    from verity_flock import rec_residuals as RR
    from verity_flock.rec_algebra import Shape

    products, why = M.rep_algebra(35, 27, 16)
    assert products > 4900 and why is None
    assert len(RR.parts(RR.spec(Shape(35, 27, 16)))) == 3
    _, why = M.rep_algebra(26, 24, 28)
    assert why == f"V* parts: residual 32 alone reads more than {RR.MAX_REGIONS} regions or {RR.MAX_LONE_SLOTS} sha512x3 slots"


def test_a_unit_types_regions_are_its_ports():
    """Gumbel's top-p sampler has 6 inputs and 1 output, SiluMul 1 and 1."""
    assert M.regions({"template": "gumbel-top-p-token-select", "subcircuit_parameters": {"V": 128256}}) == 7
    assert M.regions({"template": "silu-mul", "subcircuit_parameters": {"I": 8192}}) == 2
