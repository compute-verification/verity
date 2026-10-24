"""REDTEAM phase 6, target 3: ENC-DERIVED's v1 residual (enc-derived-order), attacked the same way as targets 1 and 2.

ENC-DERIVED v1 derives the schedule from the certified record and the engine config, and its tie
(`derived.tie_fields`) binds each request's prompt_len to the certified prompt's id count, its output_len to the
committed outputs, its finish by the stop rule, and request i to certified request `walk.order[i]`. `reconstruct`
refuses a residual with leftover or missing values and an order entry equal to the walk's prediction.

The payoff: the two seams targets 1 and 2 found are *closed* by this design — but only when the pipeline feeds its
tied v0 (not a free `advice.Advice`) into N2-ADVICE and the floor derivation. These cases show, executably, that the
forged prompt-split that N2-ADVICE #3502 accepts (target 1) and the non-canonical residual ENC-REPLAY decodes (target 2)
are both refused here, because the prompt lengths, the order and the residual's shape are held to the certified record.

Red-team evidence, not a merge-gate suite. Loads derived.py by file path; it imports advice.py from the package.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[5]
try:
    from editables.redirector import RedirectingFinder as _F

    _F.map_module("verity", str(_ROOT / "verity" / "__init__.py"))
except Exception:  # pragma: no cover
    pass


def _load():
    key = "enc_derived"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / "derived.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


D = _load()
FIXTURES = Path(__file__).parent / "fixtures"
CASES = ("sync-crush", "async-crush")


def _case(name: str):
    trace = D.load_trace(FIXTURES / f"{name}.json.gz")
    cfg = D.config_of(trace)
    adv, order = D.advice_of(trace)
    held = D.holdings_of(trace)
    return cfg, adv, order, held, D.residual(adv, cfg, held, order=order)


@pytest.mark.parametrize("name", CASES)
def test_the_residual_is_canonical_and_rebuilds(name):
    """Baseline: the residual round-trips — rebuild(residual(adv)) == adv — so there is one residual per advice."""
    cfg, adv, order, held, res = _case(name)
    assert D.rebuild(res, cfg, held, ()) == adv


@pytest.mark.parametrize("name", CASES)
def test_enc_derived_closes_target_1_split_prompt(name):
    """TARGET 1, closed here: a forged prompt-length split (swap two requests' prompt_len, keeping the total) is accepted
    by N2-ADVICE #3502 because its `ends` reads prompt_len from v0 and binds only the total. ENC-DERIVED's tie binds each
    prompt_len to the certified prompt's id count, so the same swap is refused.
    """
    cfg, adv, order, held, res = _case(name)
    pls = list(res.prompt_len)
    pair = next(((a, b) for a in range(len(pls)) for b in range(a + 1, len(pls)) if pls[a] != pls[b]), None)
    assert pair is not None, "need two requests of different prompt length to swap"
    a, b = pair
    pls[a], pls[b] = pls[b], pls[a]
    forged = dataclasses.replace(res, prompt_len=tuple(pls))
    with pytest.raises(ValueError, match="prompt_len|certified prompts"):
        D.rebuild(forged, cfg, held, ())


@pytest.mark.parametrize("name", CASES)
def test_enc_derived_closes_target_2_non_canonical_residual(name):
    """TARGET 2, closed here: ENC-REPLAY's `decode` replays any residual without checking it against the record. ENC-
    DERIVED's `reconstruct` refuses a residual with a value left over or missing, so a forged residual does not decode to
    a different valid schedule — it is rejected outright.
    """
    cfg, adv, order, held, res = _case(name)
    assert res.hits  # the residual has values to perturb
    for bad in (dict(hits=res.hits + (0,)), dict(hits=res.hits[:-1])):
        with pytest.raises(ValueError):
            D.reconstruct(dataclasses.replace(res, **bad), cfg, held)


@pytest.mark.parametrize("name", CASES)
def test_enc_derived_refuses_an_order_equal_to_its_prediction(name):
    """A residual that names an order the walk already predicts (a second encoding for one advice) is refused — the order
    channel is canonical, unlike N2-ADVICE's free request order (target 1's FCFS case).
    """
    cfg, adv, order, held, res = _case(name)
    seen: set[int] = set()
    found = None
    for s, st in enumerate(adv.steps):
        new = [r.request for r in st.rows if r.request not in seen]
        seen |= {r.request for r in st.rows}
        if s and new:
            found = (s, new)
            break
    if found is None:
        pytest.skip("no multi-admission step to exercise the order channel")
    s, new = found
    with pytest.raises(ValueError, match="order entry"):
        D.reconstruct(dataclasses.replace(res, order=((s, tuple(order[i] for i in new)),)), cfg, held)
