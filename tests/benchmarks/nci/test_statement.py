import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / "nci")))

import exposure as X  # noqa: E402
import fit as F  # noqa: E402
import statement as S  # noqa: E402
import tiers as T  # noqa: E402

P = T.DEFAULTS
H = P["years"] * T.YEAR
LAG = P["lag_s"] / H
EPS = P["eps_agg"]
STREAMED = {"pous_rounds": 74000.0, "pous_floor": 0.0}
#: the brief's rungs, as overrides on the defaults
RUNGS = {
    "Today": {"free": 1.0},
    "PoUW's floor credit": {},
    "Floor credit and attested clock": {"free": 0.213, "clock_share": 0},
    "Best plausible PoUW design": {"free": 0.01, "clock_share": 0, "eps_w": 0.0025},
}
#: PoUW's deficit at the floor, the best design's: at r̂ = 0 all of C is certified, so the deficit is ε_W C
EPS_W = RUNGS["Best plausible PoUW design"]["eps_w"]
WIPES = {13.0: "Streamed-label wipes, 13 GB", 6.0: "Streamed-label wipes, 6 GB",
         3.05: "Streamed-label wipes, 3.05 GB (no power removal)", 1.0: "Power removal, 1 GB"}


def slowdown(over: dict, tier: str) -> float:
    return S.slowdown(P | over, tier)


def test_the_slowdown_compares_with_the_unconstrained_run():
    """Unregistered compute trains like the unconstrained developer, and so does free compute on registered capacity
    (reference efficiency: ``evaluate`` at ``free_mfu`` = MFU_0); the rest are counted multiplications, so
    the slowdown is 1 / (s + rest / mfu): the slowdown calculator's ``multiple_bound`` in Lean's form, ``remoteSlowdown``, at tiers'
    terms."""
    p = P | {"unregistered": 0.01, "free": 0.01, "host_share": 0.05, "clock_share": 0.0}
    r = T.evaluate(p, T.TIERS["T4"], p["mfu"])
    rest = r["free"] + r["declared"] + r["cheat"]
    assert T.remote_terms(p, r) == (r["unregistered"], rest)
    assert S.slowdown(p, "T4") == pytest.approx(1 / (r["unregistered"] + rest / p["mfu"]), rel=1e-9)
    assert r["free"] == pytest.approx(p["mfu"] * T.evaluate(p, T.TIERS["T4"])["free"], rel=1e-9)
    assert all(S.slowdown(P | {"free": 1.0}, t) == 1.0 for t in ("T1", "T2", "T3", "T4"))


def test_the_remote_slowdown_at_the_defaults():
    """At tiers' server (the floor credit; hosts carry no compute; reference efficiency) free compute r = 0.33 trains at
    MFU_0 and the inference-only policy's term nu is counted multiplications, so the slowdown 1/(r + nu/MFU_0) is 3.02 at
    T1 to T4, just under 1/(counted share) = 3.04, the same terms against a reference at capacity. Without the
    reference-efficiency assumption free compute is at peak, ``Slowdown.Guarantees.SlowdownRemote``'s slowdown
    max(1, 1/(s + x/MFU_0)) as stated, and the slowdown is 1.21; without the hosts assumption too (host CPUs at 12.6% of
    the GPUs) x = 0.45 is above MFU_0 and the slowdown is 1; without the hosts assumption alone it is 2.19."""
    def at_peak(p: dict, t: str) -> float:
        return S.multiple_bound(p["mfu"], *T.remote_terms(p, T.evaluate(p, T.TIERS[t])), reference_efficiency=False)
    for t in ("T1", "T2", "T3", "T4"):
        r = T.evaluate(P, T.TIERS[t], P["mfu"])
        assert S.slowdown(P, t) == pytest.approx(1 / (r["free"] / P["mfu"] + r["declared"] / P["mfu"]), rel=1e-9)
        assert r["free"] / P["mfu"] == pytest.approx(T.free_share(P) + LAG, rel=1e-9)
    assert S.slowdown(P, "T1") == pytest.approx(3.022, abs=1e-3)
    assert 1 / T.evaluate(P, T.TIERS["T1"])["counted_share"] == pytest.approx(3.04, abs=0.01)
    assert at_peak(P, "T1") == pytest.approx(1.215, abs=1e-3)
    hosts = P | {"host_share": 0.12608}
    assert all(at_peak(hosts, t) == 1.0 for t in T.TIERS)
    assert S.slowdown(hosts, "T1") == pytest.approx(2.193, abs=1e-3)


def quantities(items: list[dict]) -> dict:
    return {r["quantity"]: r["value"] for r in items}


def test_the_ladder_is_evaluate_at_the_rungs():
    rows = {r["rung"]: r for r in S.ladder()}
    for name, over in RUNGS.items():
        assert rows[name]["slowdown"] == {t: slowdown(over, t) for t in T.TIERS}, name
    for gb, name in WIPES.items():
        assert rows[name]["R_wipe_gb"] == pytest.approx(gb, rel=1e-9)
        for t in ("T3", "T5"):
            assert rows[name]["slowdown"][t] == pytest.approx(slowdown(STREAMED | {"unfilled_gb": gb - 0.17}, t), rel=0.005)


def test_the_slowdown_is_set_by_the_binding_terms():
    """The slowdown is 1 / (r + Delta/H + nu / MFU_0) where free compute binds (as at power removal's 1 GB, where
    mu > 1): free compute and the verdict lag train at MFU_0 (reference efficiency), nu is counted multiplications."""
    rows = {r["rung"]: r for r in S.ladder()}
    nu = X.nu(P) / P["mfu"]
    best = T.free_share(P | RUNGS["Best plausible PoUW design"])
    assert rows["Best plausible PoUW design"]["slowdown"]["T1"] == pytest.approx(1 / (best + nu + LAG), rel=1e-4)
    floor = nu + EPS_W + EPS
    assert rows["Floor (r̂ = 0, 4 GB)"]["slowdown"]["T1"] == pytest.approx(1 / (floor + LAG), rel=1e-4)
    assert rows["Floor, one-minute verdict lag"]["slowdown"]["T3"] == pytest.approx(1 / (floor + 60 / H), rel=1e-4)
    one = P | S.streamed(1.0)
    assert X.mu(one) > 1 and rows["Power removal, 1 GB"]["slowdown"]["T3"] == pytest.approx(
        1 / (T.free_share(one) + nu + LAG), rel=1e-3)
    fl = quantities(S.floor())
    assert fl["1 / (ε_W + ε′ + ν / MFU_0), the floor's limit as the verdict lag goes to 0"] == pytest.approx(
        1 / floor, rel=1e-12)
    assert fl["MFU_0 / ν, the same as ε_W and ε′ go to 0 too (δ_win / ε′ then needs δ_win → 0)"] == P["mfu"] / X.nu(P)


def test_fleet_wide_forgiveness_is_j_identities_of_verdict_lag():
    q = P | S.BEST
    _, fleet, server = (r["value"] for r in S.slide())
    assert fleet == pytest.approx(1 / (T.free_share(q) + S.SDC * LAG + X.nu(q) / P["mfu"]), rel=1e-4)
    assert server == pytest.approx(S.SDC * LAG / P["nodes"], rel=1e-12)


def test_the_fit_bound_rows_are_fits_with_chi_on_top():
    law, target = S.FIT_TARGET
    L = F.target_loss(law, target)
    N = F.n_star(law, L)
    chi = S.cheat(P)
    fb = S.fit_bound()
    for r in fb["rows"]:
        row = F.row(r["R_wipe_gb"], N, L)
        assert (r["slowdown_fit"], r["slowdown_closed"], r["slowdown_fit_train"]) == (row["slowdown_fit"], row["slowdown_closed"], row["slowdown_fit_train"])
        assert r["slowdown_fit_T5"] == pytest.approx(max(1.0, 1 / (1 / r["slowdown_fit"] + chi / P["mfu"])), rel=1e-3)
    edge = fb["other"][-1]["value"]
    assert F.row(edge, N, L)["fit_over_closed"] == pytest.approx(1.0, rel=1e-6)


def test_the_regimes_add_unregistered_compute_at_full_efficiency():
    """With today's wipes the fit bound caps nothing; at 6 GB it caps the best design's free compute, which trains at
    MFU_0 (reference efficiency) where the cap and nu are counted multiplications."""
    q = P | S.BEST
    registered = T.free_share(q) + LAG + X.nu(q) / P["mfu"]
    mu = S.fit_mu(6.0) / P["mfu"]
    assert mu < registered
    for row in S.regimes()["rows"]:
        s, today, six = row["s"], row["slowdown"]["today's wipes"], row["slowdown"]["6 GB streamed-label wipes"]
        assert 1 / today["T3"] == pytest.approx(s + (1 - s) * registered, rel=1e-4), row
        assert today["T5"] == pytest.approx(max(1.0, 1 / (s + (1 - s) * (registered + S.cheat(q) / P["mfu"]))), rel=1e-4), row
        assert 1 / six["T3"] == pytest.approx(s + (1 - s) * mu, rel=1e-4), row
        assert six["T5"] == pytest.approx(max(1.0, 1 / (s + (1 - s) * (mu + S.cheat(q) / P["mfu"]))), rel=1e-4), row


def test_the_ladders_fit_bound_is_fit_row_at_the_memory_rungs():
    """At the streamed-label and power-removal rungs, the slowdown by the fit bound is ``fit.row``'s; at today's 166 GB,
    where the fit bound doesn't hold, it is the slowdown without a memory cap (T2's)."""
    law, target = S.FIT_TARGET
    L = F.target_loss(law, target)
    N = F.n_star(law, L)
    chi = S.cheat(P)
    rows = {r["rung"]: r for r in S.ladder()}
    for gb, name in WIPES.items():
        bound = F.row(gb, N, L)["slowdown_fit"]
        assert rows[name]["slowdown_fit"]["T3"] == pytest.approx(bound, rel=1e-9), name
        assert rows[name]["slowdown_fit"]["T5"] == pytest.approx(max(1.0, 1 / (1 / bound + chi / P["mfu"])), rel=1e-3), name
    for name in RUNGS:
        assert rows[name]["slowdown_fit"]["T3"] == pytest.approx(rows[name]["slowdown"]["T2"], rel=1e-9), name


def test_the_fit_edges_are_where_the_fit_bound_crosses_each_level():
    """T5 reaches no level: chi alone holds it to max(1, MFU_0 / chi), which is 1 at today's sparse samples."""
    edges = S.fit_edges()
    assert set(edges["T5"].values()) == {None} and max(1.0, P["mfu"] / S.cheat(P)) == S.capped(P, "T5", 0.0) == 1
    assert edges["T3"][2.0] == math.inf and S.capped(P, "T3", math.inf) > 2
    for tier, by_level in edges.items():
        for level, gb in by_level.items():
            if gb is not None and math.isfinite(gb):
                assert S.capped(P, tier, S.fit_mu(gb * (1 - 1e-6))) >= level > S.capped(P, tier, S.fit_mu(gb * (1 + 1e-6)))


def test_the_recipe_multiplier_scales_the_fit_bounds_cap():
    """As in ``tiers.evaluate``, where the registered counted share is capped at m times the memory cap: at 6 GB the cap
    binds on the best design, so the slowdown is MFU_0 / (m mu) by both ``statement.capped`` and ``fit.slowdown``."""
    mu = S.fit_mu(6.0)
    for m in (1.0, 3.0):
        q = P | S.BEST | S.streamed(6.0) | {"m": m}
        assert S.fit_slowdown(q, "T3") == pytest.approx(P["mfu"] / (m * mu), rel=1e-9), m
        assert F.slowdown(F.params(6.0) | {"m": m}, mu, 0.0) == pytest.approx(P["mfu"] / (m * mu), rel=1e-9), m


def test_alpha_falls_only_where_k_reaches_the_level_in_its_range():
    lo, hi = X.SPECS["K"]["min"], X.SPECS["K"]["max"]
    assert S.alpha_falls({"below": {100.0: "always"}, "intervals": {100.0: [[lo, hi]]}}) is None
    assert S.alpha_falls({"below": {100.0: "> 5e5"}, "intervals": {100.0: [[5e5, hi]]}}) == P["K"] / 5e5
    assert S.alpha_falls({"below": {100.0: "never"}, "intervals": {100.0: []}}) == 1.0
    assert S.slowdown(P | {"K": lo}, "T3") < 100


def test_the_symbols_closed_forms_are_what_evaluate_uses():
    sym = {(r["symbol"], r["meaning"]): r["value"] for r in S.symbols()}
    capped = T.evaluate(P | S.streamed(1.0), T.TIERS["T3"])
    if capped["memory_caps_free"]:
        assert sym[("μ̂", "at 1 GB")] == pytest.approx(capped["free"] + capped["declared"], rel=0.01)
    else:
        assert sym[("μ̂", "at 1 GB")] >= capped["free"] + capped["declared"]
    assert sym[("μ̂", "at 6 GB")] == X.mu(P | S.streamed(6.0))
    assert sym[("ν = 2αΛ(K)/(κ b_a)", "NCI's term")] == pytest.approx(
        T.evaluate(P | S.BEST, T.TIERS["T1"])["declared"], rel=1e-6)
    assert sym[("χ", "cheating share (T5)")] == T.evaluate(P, T.TIERS["T5"])["cheat"]


def test_the_thresholds_are_on_the_slowdown_not_on_one_over_the_counted_share():
    """At T1 with host CPUs and the clock at 0, the slowdown is below X once
    r̂ > (1 / X - nu / MFU_0 - Delta/H - eps_W - eps') / (1 - eps_W): free compute trains at MFU_0 (reference efficiency)
    and nu is counted multiplications; on 1/(counted share), nu counts at 1."""
    found = S.search(("defaults", "T1", 0))
    for level in S.LEVELS:
        edge = (1 / level - X.nu(P) / P["mfu"] - LAG - P["eps_w"] - EPS) / (1 - P["eps_w"])
        if edge > 0:
            (a, _), = found[level]
            assert a == pytest.approx(edge, rel=1e-3), level


def test_chi_is_the_chord_times_the_pool():
    """chi = 2 m chord bits / (kappa C), at most 1, and the injected bits scale with (overhead + replay) / share +
    floor_steps. At today's sparse samples it is 1; at 512 samples per statement it is below."""
    def pool(overhead: float) -> float:
        return (overhead + P["replay"]) / P["share"] + P["floor_steps"]
    today, many = S.holes()["cases"]
    for c in (today, many):
        assert c["chi"] == pytest.approx(min(1.0, 2 * P["m"] * c["chord"] * c["bits"] / (P["kappa"] * S.capacity(P))),
                                         rel=1e-12)
    assert today["chi"] == 1 > many["chi"]
    assert today["bits"] / many["bits"] == pytest.approx(pool(today["overhead"]) / pool(many["overhead"]), rel=1e-12)


def test_chis_linear_cost_is_from_the_closed_form_where_the_share_is_stopped():
    """Section 6 sets the proving cost at which chi reaches 1% beside what a chi linear in the cost gives. At the defaults
    the fleet's capacity stops the share at 1, and 0.01 over the stopped share is far above the search: the line divides
    by the closed form's chi, which doesn't stop, and agrees with the search. At batched proving, where the share isn't
    stopped, the closed form is the share."""
    th = {"defaults": {"T3": [{"key": "K", "below": {100.0: "never"}, "intervals": {100.0: []}}]}}
    q = quantities(S.exposures(th))
    found = q["multiple of today's proving cost at which χ reaches 1%"]
    (linear,) = (v for name, v in q.items() if name.startswith("the same if χ were linear in it"))
    assert S.cheat(P) == 1 < X.chi(P) and 0.01 / S.cheat(P) > 10 * found
    assert linear == 0.01 / X.chi(P) == pytest.approx(found, rel=1e-3)
    batched = P | X.BATCHED
    assert 1 > S.cheat(batched) == pytest.approx(X.chi(batched), rel=1e-3)


def test_bisect():
    assert S.bisect(lambda x: x * x >= 2, 1.0, 2.0) == pytest.approx(math.sqrt(2), rel=1e-12)


def test_num():
    assert [S.num(x) for x in (2048.0, 3213.24, 2.8475e-4, 0.00489, 1e5, 6.4e16, 0.5081)] == [
        "2,048", "3,210", "2.85e-4", "0.00489", "1e5", "6.4e16", "0.508"]


@pytest.mark.slow
def test_it_renders_every_section_as_markdown_and_json(capsys, monkeypatch):
    """The whole statement, about 9 s on four cores, most of it the threshold search."""
    with ProcessPoolExecutor() as pool:
        whole = S.statement(pool)
    monkeypatch.setattr(S, "statement", lambda pool=None: whole)
    assert S.main([]) == 0
    md = capsys.readouterr().out
    assert md.startswith("# ") and "at commit `" in md
    for heading in ("## Slide forms", "## 0. Symbols", "## 4. The terms", "## The ladder", "## The fit bound",
                    "## Years of growth", "## The unregistered share by regime", "## 6. Thresholds on the slowdown",
                    "## Known holes"):
        assert heading in md
    assert S.main(["--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["ladder"][3]["slowdown"]["T5"] == slowdown(RUNGS["Best plausible PoUW design"], "T5")
    assert set(s["thresholds"]) == set(X.THRESHOLD_SCENARIOS)
    x = quantities(s["exposures"])["multiple of today's proving cost at which χ reaches 1%"]
    assert S.cheat(P | {"overhead": x * P["overhead"]}) == pytest.approx(0.01, rel=1e-9)
