import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fit as F  # noqa: E402
import tiers as T  # noqa: E402


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


def test_below_n_star_no_run_reaches_the_target():
    L = F.target_loss("besiroglu", "405b")
    n = F.n_star("besiroglu", L)
    assert F.tokens_to_reach("besiroglu", 0.999 * n, L) == math.inf
    assert F.undersizing("besiroglu", 0.999 * n, L) == 0
    m = [F.undersizing("besiroglu", k * n, L) for k in (1.2, 2, 3, 5, 8)]
    assert 0 < m[0] and max(m) <= 1 + 1e-9


def test_the_fit_bound_is_the_window_bound_with_the_entry_fee_at_server_scale():
    """Against the closed form the fit bound's share is (S / R) (d b_a / w_e) k0 / (k0 - 1)."""
    p = F.params(6.0)
    R = 6.0 * T.GB
    S = F.state(p, R)
    w = F.BOUNDARY["central"]
    k = F.servers(S, 87.6e9, p["b_w"])
    assert k == 4
    ratio = F.share(p, F.fit_rate(p, S, 87.6e9, w)) / F.closed_form(p, R)
    assert ratio == pytest.approx(S / R * p["d"] * p["b_a"] / w * k / (k - 1))


def test_the_fit_bound_is_void_once_a_window_holds_n_star_weights():
    p = F.params(6.0)
    S = 1e10 * p["b_w"]
    assert F.fit_rate(p, S, 1e10, F.BOUNDARY["central"]) == math.inf
    assert F.fit_rate(p, 0.99 * S, 1e10, F.BOUNDARY["central"]) < math.inf
    assert F.dampener(p, math.inf, 0.0) == pytest.approx(p["mfu"] / F.residual_with_lag(p))


def test_the_table_against_the_red_teams_closed_form():
    """The closed form gives the red team's D = 89 and 41 at 6 and 13 GB; at central boundary bits the fit bound is
    stronger there, and void from 50 GB, where a server holds N* weights at b_w."""
    rows = {r["wipe_gb"]: r for r in F.summary()["rows"]}
    assert rows[6.0]["D_closed"] == pytest.approx(89, rel=0.01)
    assert rows[13.0]["D_closed"] == pytest.approx(41, rel=0.01)
    assert rows[6.0]["D_fit"] > rows[6.0]["D_closed"] and rows[13.0]["D_fit"] > rows[13.0]["D_closed"]
    assert rows[50.0]["k0"] < 2 and rows[166.0]["k0"] < 2


def test_a_replica_trains_the_same_tokens_between_syncs_whatever_its_size():
    p = F.params(203.0)
    L = F.target_loss("besiroglu", "405b")
    a, b = (F.replica(p, S, L, 16.0) for S in (1e12, 2e12))
    assert a["tokens_per_sync"] == pytest.approx(b["tokens_per_sync"])
    assert b["sync_s"] == pytest.approx(2 * a["sync_s"])
