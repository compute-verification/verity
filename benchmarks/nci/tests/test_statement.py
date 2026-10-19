import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exposure as X  # noqa: E402
import fit as F  # noqa: E402
import statement as S  # noqa: E402
import tiers as T  # noqa: E402

P = T.DEFAULTS
H = P["years"] * T.YEAR
LAG = P["lag_s"] / H
STREAMED = {"pous_rounds": 74000.0, "pous_floor": 0.0}
#: the brief's rungs, as overrides on the defaults
RUNGS = {
    "Today": {"residual": 1.0},
    "PoUW's floor credit": {},
    "Floor credit, attested clock and EPYC hosts": {"residual": 0.217, "clock_share": 0, "host_share": 0.0021},
    "Best plausible PoUW design": {"residual": 0.01, "host_share": 0.0021, "clock_share": 0},
}
WIPES = {13.0: "Streamed-label wipes, 13 GB", 6.0: "Streamed-label wipes, 6 GB",
         3.05: "Streamed-label wipes, 3.05 GB (no power removal)", 1.0: "Power removal, 1 GB"}


def dampener(over: dict, tier: str) -> float:
    return T.evaluate(P | over, T.TIERS[tier])["dampener"]


def quantities(items: list[dict]) -> dict:
    return {r["quantity"]: r["value"] for r in items}


def test_the_ladder_is_evaluate_at_the_rungs():
    rows = {r["rung"]: r for r in S.ladder()}
    for name, over in RUNGS.items():
        assert rows[name]["D"] == {t: dampener(over, t) for t in T.TIERS}, name
    for gb, name in WIPES.items():
        assert rows[name]["R_wipe_gb"] == pytest.approx(gb, rel=1e-9)
        for t in ("T3", "T5"):
            assert rows[name]["D"][t] == pytest.approx(dampener(STREAMED | {"unfilled_gb": gb - 0.17}, t), rel=0.005)


def test_d_is_the_hardware_multiple_of_the_binding_terms():
    """D = MFU_0 / (r + nu + Delta/H) where the residual binds, MFU_0 / mu where the memory cap does."""
    rows = {r["rung"]: r for r in S.ladder()}
    best = T.residual_share(P | RUNGS["Best plausible PoUW design"])
    assert rows["Best plausible PoUW design"]["D"]["T1"] == pytest.approx(P["mfu"] / (best + X.nu(P) + LAG), rel=1e-4)
    assert rows["Floor (r̂ = 0, 4 GB)"]["D"]["T1"] == pytest.approx(P["mfu"] / (X.nu(P) + LAG), rel=1e-4)
    assert rows["Floor, one-minute verdict lag"]["D"]["T3"] == pytest.approx(P["mfu"] / (X.nu(P) + 60 / H), rel=1e-4)
    six = P | S.streamed(6.0)
    assert rows["Streamed-label wipes, 6 GB"]["D"]["T3"] == pytest.approx(P["mfu"] / X.mu(six), rel=0.01)
    fl = quantities(S.floor())
    assert fl["MFU_0 / ν, the floor's limit as the verdict lag goes to 0"] == P["mfu"] / X.nu(P)
    assert fl["D at zero lag, r̂ = 0, T3 (the partial-sum law binds)"] > P["mfu"] / X.nu(P)


def test_fleet_wide_forgiveness_is_j_identities_of_verdict_lag():
    q = P | S.BEST
    _, fleet, server = (r["value"] for r in S.slide())
    assert fleet == pytest.approx(P["mfu"] / (T.residual_share(q) + S.SDC * LAG + X.nu(q)), rel=1e-4)
    assert server == pytest.approx(S.SDC * LAG / P["nodes"], rel=1e-12)


def test_the_fit_bound_rows_are_fits_with_chi_on_top():
    law, target = S.FIT_TARGET
    L = F.target_loss(law, target)
    N = F.n_star(law, L)
    chi = S.cheat(P)
    fb = S.fit_bound()
    for r in fb["rows"]:
        row = F.row(r["R_wipe_gb"], N, L)
        assert (r["D_fit"], r["D_closed"], r["D_fit_train"]) == (row["D_fit"], row["D_closed"], row["D_fit_train"])
        assert 1 / r["D_fit_T5"] == pytest.approx(1 / r["D_fit"] + chi / P["mfu"], rel=1e-3)
    edge = fb["other"][-1]["value"]
    assert F.row(edge, N, L)["fit_over_closed"] == pytest.approx(1.0, rel=1e-6)


def test_the_regimes_add_unmonitored_compute_at_full_efficiency():
    q = P | S.BEST
    monitored = (T.residual_share(q) + LAG + X.nu(q)) / P["mfu"]
    for row in S.regimes()["rows"]:
        s, D = row["s"], row["D"]["today's wipes"]
        assert 1 / D["T3"] == pytest.approx(s + (1 - s) * monitored, rel=1e-4), row
        assert 1 / D["T5"] == pytest.approx(s + (1 - s) * (monitored + S.cheat(q) / P["mfu"]), rel=1e-4), row


def test_the_symbols_closed_forms_are_what_evaluate_uses():
    sym = {(r["symbol"], r["meaning"]): r["value"] for r in S.symbols()}
    capped = T.evaluate(P | S.streamed(6.0), T.TIERS["T3"])
    assert sym[("μ̂", "at 6 GB")] == pytest.approx(capped["residual"] + capped["declared"], rel=0.01)
    assert sym[("ν = 2αΛ(K)/(κ b_a)", "NCI's term")] == pytest.approx(
        T.evaluate(P | S.BEST, T.TIERS["T1"])["declared"], rel=1e-6)
    assert sym[("χ", "cheating share (T5)")] == T.evaluate(P, T.TIERS["T5"])["cheat"]


def test_the_thresholds_are_on_d_not_on_one_over_eta():
    """At T1 with host CPUs and the clock at 0, D < X once r > MFU_0 / X - nu - Delta/H; on 1/eta the edge would be at
    1/X, 2.5 times higher."""
    found = S.search(("defaults", "T1", 0))
    for level in S.LEVELS:
        (a, _), = found[level]
        assert a == pytest.approx(P["mfu"] / level - X.nu(P) - LAG, rel=1e-3), level


def test_chi_is_the_chord_times_the_pool():
    """chi = 2 m chord bits / (kappa C), and the injected bits scale with (overhead + replay) / share + floor_steps."""
    def pool(overhead: float) -> float:
        return (overhead + P["replay"]) / P["share"] + P["floor_steps"]
    today, many = S.holes()["cases"]
    for c in (today, many):
        assert c["chi"] == pytest.approx(2 * P["m"] * c["chord"] * c["bits"] / (P["kappa"] * S.capacity(P)), rel=1e-12)
    assert today["bits"] / many["bits"] == pytest.approx(pool(today["overhead"]) / pool(many["overhead"]), rel=1e-12)


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
                    "## Years of growth", "## The unmonitored share by regime", "## 6. Thresholds on D",
                    "## Known holes"):
        assert heading in md
    assert S.main(["--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["ladder"][3]["D"]["T5"] == dampener(RUNGS["Best plausible PoUW design"], "T5")
    assert set(s["thresholds"]) == set(X.THRESHOLD_SCENARIOS)
    x = quantities(s["exposures"])["multiple of today's proving cost at which χ reaches 1%"]
    assert S.cheat(P | {"overhead": x * P["overhead"]}) == pytest.approx(0.01, rel=1e-9)
