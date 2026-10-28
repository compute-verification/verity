import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fit as F  # noqa: E402
import kappa  # noqa: E402
import tiers as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def test_n_star_depends_on_the_target_not_on_the_loss_floor():
    """N* is the size whose parameter term alone matches the target's parameter and data terms, so the fitted floor E
    cancels: 87.6B for a 405B model on 15.6T tokens under the Besiroglu refit, and more under a data limit."""
    f = F.FITS["besiroglu"]
    N, D = F.TARGETS["405b"]
    L = F.target_loss("besiroglu", "405b")
    rho = (f["B"] / D ** f["beta"]) / (f["A"] / N ** f["alpha"])
    n = F.n_star("besiroglu", L)
    assert n == pytest.approx(N * (1 + rho) ** (-1 / f["alpha"]), rel=1e-9)
    assert n == pytest.approx(87.6e9, rel=2e-3)
    assert n < F.n_star("besiroglu", L, 1e15) < F.n_star("besiroglu", L, 1e14)


def test_reach_at_sequence_threads_kimi_k2():
    """The falsification test of the conjectures `Nci.Assumptions.Reach` and `ReachSlots` over all recipes: one sequence
    of Kimi K2's pretraining, at its shortest length, uses at least N* weights in its counted multiplications (87.6B,
    the Besiroglu refit at a 405B-class target) when its tokens' routed slots spread over the experts as kappa.py routes
    them, 8 of 384 a token, load-balanced; one token uses fewer, so Reach is false at token threads, the threading the
    rungs use, for a token-routed MoE (the policy adopts it for dense recipes only). A token's counted multiplications use its active weights outside the embedding and the LM head, and the LM
    head, whose backward is counted; a sequence's use every routed expert one of its S tokens reaches, E (1 - (1 - a/E)^S)
    per MoE layer in expectation. It fails when N* exceeds what a sequence reaches, a sequence routed to fewer than the
    `fewest` experts per layer among them, or when a token reaches N*, which would leave the conjectures' why stale."""
    m = kappa.MODELS["kimi-k2"]
    n_star = F.n_star("besiroglu", F.target_loss("besiroglu", "405b"))
    E, a, layers = m["experts"], m["active"], kappa.moe_layers(m)
    expert = sum(r * c for r, c in kappa.expert_shapes(m))
    token = kappa.active_weights(m) + m["vocab"] * m["d"]
    S = min(s for _, s in m["schedule"])
    reached = E * (1 - (1 - a / E) ** S)
    fewest = a + math.ceil((n_star - token) / (layers * expert))
    assert token == pytest.approx(31.7e9, rel=0.01) and n_star == pytest.approx(87.6e9, rel=2e-3)
    assert token < n_star <= token + (reached - a) * layers * expert
    assert (S, fewest) == (4096, 30) and fewest <= reached
    entry = json.loads((ROOT / "Security" / "lean-audit.json").read_text())["shape"]["conjectures"]
    test = f"benchmarks/nci/tests/{Path(__file__).name}::test_reach_at_sequence_threads_kimi_k2"
    for name in ("Nci.Assumptions.Reach", "Nci.Assumptions.ReachSlots"):
        assert entry[name] | {"why": ""} == {"status": "open", "test": test, "why": ""}


def test_below_n_star_no_run_reaches_the_target():
    L = F.target_loss("besiroglu", "405b")
    n = F.n_star("besiroglu", L)
    assert F.tokens_to_reach("besiroglu", 0.999 * n, L) == math.inf
    assert F.undersizing("besiroglu", 0.999 * n, L) == 0
    m = [F.undersizing("besiroglu", k * n, L) for k in (1.2, 2, 3, 5, 8)]
    assert 0 < m[0] and max(m) <= 1 + 1e-9


def test_the_fit_bound_is_the_window_bound_with_the_entry_fee_at_window_scale():
    """Per window, the fit bound is ``tiers.region_cap`` with the entry fee d b_a replaced by w_e (k' - 1) / k'."""
    p = F.params(6.0)
    N, w = 87.6e9, F.BOUNDARY["central"]
    for periods in (1, 3, 5):
        S = F.state(p, 6.0 * T.GB, periods)
        k = F.visits(S, N, p["b_w"])
        assert F.window_mults(S, N, w, p["b_w"]) == pytest.approx(
            T.region_cap(p, S) * p["d"] * p["b_a"] / (w * (k - 1) / k))
    rate, periods = F.fit_rate(p, 6.0 * T.GB, N, w)
    assert rate == pytest.approx(F.window_mults(F.state(p, 6.0 * T.GB, periods), N, w, p["b_w"])
                                 / (periods * p["wipe_s"]))
    assert all(rate <= F.window_mults(F.state(p, 6.0 * T.GB, m), N, w, p["b_w"]) / (m * p["wipe_s"])
               for m in range(1, 9))


def test_a_window_reading_carried_tokens_stays_within_the_bound():
    """A one-period window that reads half its S as weights and half as tokens carried across the wipe that opens it
    meets F1-F4, does more work than a count charging token entries to intake alone allows, and no more than the
    bound."""
    p = F.params(6.0)
    N, w, b = 87.6e9, F.BOUNDARY["central"], p["b_w"]
    S = F.state(p, 6.0 * T.GB)
    weights, entered = S / (2 * b), S / (2 * w)
    k = math.ceil(N / weights)
    work = 2 * weights * entered * k / (k - 1)
    k0 = math.ceil(N * b / S)
    intake_only = 2 * F.intake(p) * p["wipe_s"] * S * k0 / ((k0 - 1) * b * w)
    assert intake_only < work <= F.window_mults(S, N, w, b)


def test_the_fit_bound_charges_each_servers_first_window():
    """`fitBoundSlots_init` adds the counted multiplications inside the windows it leaves out, each server's first, to
    the bound: a row's mu is fit_rate's share plus that window at mfu over the horizon, which takes the 6 GB rung from
    218 to 198."""
    p = F.params(6.0)
    R = T.evaluate(p, T.TIERS["T4"])["wipe_state_bits"]
    L = F.target_loss("besiroglu", "405b")
    N, w = F.n_star("besiroglu", L), F.BOUNDARY["central"]
    rate, periods = F.fit_rate(p, R, N, w)
    row = F.row(6.0, N, L)
    first = p["mfu"] * periods * p["wipe_s"] / (p["years"] * T.YEAR)
    assert periods == 4 and row["mu_fit"] == pytest.approx(F.share(p, rate) + first) == F.fit_mu(p, R, N, w)
    assert F.dampener(p, F.share(p, rate), T.evaluate(p, T.TIERS["T4"])["declared"]) == pytest.approx(217.6, abs=0.5)
    assert row["D_fit"] == pytest.approx(198, abs=0.5)
    assert F.left_out(F.params(6.0, years=2.0), periods) == pytest.approx(first / 2)


def test_the_fit_bound_is_void_once_a_window_holds_n_star_weights():
    p = F.params(6.0)
    R = 1e10 * p["b_w"] - F.intake(p) * p["wipe_s"]
    assert F.fit_rate(p, R, 1e10, F.BOUNDARY["central"]) == (math.inf, 0)
    assert F.fit_rate(p, 0.99 * R, 1e10, F.BOUNDARY["central"])[0] < math.inf
    assert F.window_mults(1e10 * p["b_w"], 1e10, F.BOUNDARY["central"], p["b_w"]) == math.inf
    q = F.params(6.0, residual=0.1)
    assert 1 < F.dampener(q, math.inf, 0.0) == pytest.approx(1 / F.residual_with_lag(q))


def test_the_table_against_the_red_teams_closed_form():
    """At the structural d = 1 the closed form caps nothing at any of the table's residuals, leaving D = 3.02 at the
    ladder's r-hat (PoUW's floor credit, 0.323, hosts carrying no compute, the residual at MFU_0), as the ladder's
    memory rungs have it; at central boundary bits the fit
    bound is stronger there, and void from 50 GB, where a server holds N* weights at b_w."""
    rows = {r["wipe_gb"]: r for r in F.summary()["rows"]}
    assert T.audited_residual(F.params(1.0)) == pytest.approx(0.323, abs=5e-4)
    assert rows[1.0]["D_closed"] == rows[6.0]["D_closed"] == rows[13.0]["D_closed"] == pytest.approx(3.022, abs=1e-3)
    assert rows[6.0]["D_fit"] > rows[6.0]["D_closed"] and rows[13.0]["D_fit"] > rows[13.0]["D_closed"]
    assert rows[50.0]["periods"] == 0 and rows[166.0]["periods"] == 0
    assert rows[50.0]["visits"] < 1 and rows[50.0]["visits_train"] > 1


def test_a_replica_trains_the_same_tokens_between_syncs_whatever_its_size():
    p = F.params(203.0)
    L = F.target_loss("besiroglu", "405b")
    a, b = (F.replica(p, S, L, 16.0) for S in (1e12, 2e12))
    assert a["tokens_per_sync"] == pytest.approx(b["tokens_per_sync"])
    assert b["sync_s"] == pytest.approx(2 * a["sync_s"])
