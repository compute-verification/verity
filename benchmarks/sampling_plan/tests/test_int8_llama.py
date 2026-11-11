"""The int8 row's replay (`int8_llama.py`): serving keeps only the requests' tokens and the weights, so a sampled proof block's witness
is rebuilt by re-running its request, and the replay grain is the window's requests, each one prompt of `context` tokens
at `tokens_per_s`, every op's units split evenly over them."""

from __future__ import annotations

import pytest

import int8_llama as L
from inputs import I, MODELS
from verity.protocols.spot.optimize import ReplayGrain, replay_cost

V = {k: i.value for k, i in I.items()}
M = MODELS["llama-3.1-8b"]


@pytest.mark.parametrize("window", list(L.WINDOWS))
def test_the_replay_grain_is_the_windows_requests_each_a_prompt_at_the_plain_rate(window):
    w = L.workload(V, L.WINDOWS[window], M)
    assert w["requests"] == max(1, round(w["tokens"] / V["context"]))
    costs = L.model(V, L.fitted(V), w["requests"])
    assert costs.replay == {"request": ReplayGrain(w["requests"], V["context"] / V["tokens_per_s"])}
    for c in L.forced(w, None) + L.forced(w, 256):
        assert c.grain == "request" and c.units == c.n_rho * w["requests"], c.template


def test_one_draw_reruns_one_request_and_proving_everything_reruns_the_window():
    """One sampled proof block re-runs its request; every unit sampled re-runs every request, the window's serving again."""
    w = L.workload(V, L.WINDOWS["round"], M)
    costs = L.model(V, L.fitted(V), w["requests"])
    cands = L.forced(w, None)
    one = replay_cost({c.template: int(i == 0) for i, c in enumerate(cands)}, cands, costs)
    assert one == pytest.approx(V["context"] / V["tokens_per_s"])
    every = replay_cost({c.template: c.units for c in cands}, cands, costs)
    assert every == pytest.approx(w["window_gpu_s"], rel=1e-3)
