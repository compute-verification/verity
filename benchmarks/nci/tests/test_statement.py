import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import exposure as X  # noqa: E402
import statement as S  # noqa: E402
import tiers as T  # noqa: E402

P = T.DEFAULTS
LAG = P["lag_s"] / (P["years"] * T.YEAR)
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
    six = P | S.streamed(6.0)
    assert rows["Streamed-label wipes, 6 GB"]["D"]["T3"] == pytest.approx(P["mfu"] / X.mu(six), rel=0.01)


def test_the_regimes_add_unmonitored_compute_at_full_efficiency():
    q = P | S.BEST
    monitored = (T.residual_share(q) + LAG + X.nu(q) + S.cheat(q)) / P["mfu"]
    for row in S.regimes()["rows"]:
        s = row["s"]
        assert 1 / row["D"]["today's wipes"] == pytest.approx(s + (1 - s) * monitored, rel=1e-4), row


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


def test_chi_is_proportional_to_the_pool():
    """chi scales with (overhead + replay) / share + floor_steps, so a steeper chord applied as proving cost scales it
    by the chord's factor up to the replay."""
    def pool(overhead: float) -> float:
        return (overhead + P["replay"]) / P["share"] + P["floor_steps"]
    chi = S.holes()["chi"]
    base = {c["cost"]: c for c in chi if c["chord"] == 1.0}
    for c in chi:
        b = base[c["cost"]]
        assert c["chi"] / b["chi"] == pytest.approx(pool(c["overhead"]) / pool(b["overhead"]), rel=1e-9)
        assert c["chi"] / b["chi"] == pytest.approx(c["chord"], rel=1e-5)


def test_num():
    assert [S.num(x) for x in (2048.0, 3213.24, 2.8475e-4, 0.00489, 1e5, 6.4e16, 0.5081)] == [
        "2,048", "3,210", "2.85e-4", "0.00489", "1e5", "6.4e16", "0.508"]


@pytest.mark.slow
def test_it_renders_every_section_as_markdown_and_json(capsys):
    """The whole statement, about 7 s on four cores, most of it the threshold search."""
    assert S.main([]) == 0
    md = capsys.readouterr().out
    assert md.startswith("# ") and "at commit `" in md
    for heading in ("## 0. Symbols", "## The ladder", "## Years of growth", "## The unmonitored share by regime",
                    "## 6. Thresholds on D", "## Known holes"):
        assert heading in md
    assert S.main(["--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["ladder"][3]["D"]["T5"] == dampener(RUNGS["Best plausible PoUW design"], "T5")
    assert set(s["thresholds"]) == set(X.THRESHOLD_SCENARIOS)
