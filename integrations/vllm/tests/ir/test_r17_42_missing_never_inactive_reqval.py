"""reqval — R17-42 §5 negative test against the PRODUCTION lifted reader (`verity.ir/lifted_record.py::required_population`, pad M-0955/M-0956):
the "missing → inactive" fallback is PROHIBITED.  The ⊥ region of a request is DECLARED first — Continuation_v1's rule over the committed
served list (`continuation.executed_prefix_by_rule`, equivalently `lifted.expected_served(a_r, cap, LAG)`), never read off the binding map —
and the reader then refuses BY NAME: (i) an identity OUTSIDE the region that is unbound (a coverage hole of the executed population, never
"inactive"); (ii) an identity INSIDE the region that is bound (late activity); (iii) no declaration → scope (None), never a pass.

The fixtures are hand-built Commit verdicts (the `required_manifest` block the Commit writes: populations + coverage runs) so the test exercises
the reader's contract and not a model of it.  Nothing here reads a trace or a record."""
from __future__ import annotations

import inspect

import pytest

from verity_vllm.lifting import continuation as CONT
from verity_vllm.lifting import record as R
from verity_vllm.registry import lifted as L

CAP, LAG, EOS = 12, 0, 17
PER_EVENT = {"model.layers.0.mlp": 10, "logits": 2}          # identities per event, per family
DIGEST = "p" * 64


def _declared_region(served: list[int]) -> tuple[list[int], list[int]]:
    """The ⊥ region from the DECLARATION path (Continuation_v1 over the served list; A = LAG): [n_exec, cap) — and the one-mechanism check
    that `expected_served` (the lifted reader's own formula) says the same."""
    epr = CONT.executed_prefix_by_rule(served, cap=CAP, A=LAG, eos=[EOS])
    assert epr["ok"], epr["why"]
    n_exec = epr["executed_prefix_len"]
    assert n_exec == L.expected_served(epr["served_output_len"], CAP, LAG)     # revb S-r17b-20 R3: ONE mechanism, two consumers agree
    return list(range(0, n_exec)), list(range(n_exec, CAP))


def _verdict(steps_with_missing: list[int], *, requests_with_missing: list[str] | None = None, coverage: bool = True, digest: str = DIGEST) -> dict:
    """A Commit verdict whose coverage left every identity of `steps_with_missing` unbound (per-event families, one request)."""
    pop = {f: n * CAP for f, n in PER_EVENT.items()}
    miss_by = {f: n * len(steps_with_missing) for f, n in PER_EVENT.items()} if steps_with_missing else {}
    cov = {"ok": not steps_with_missing, "steps_with_missing": list(steps_with_missing), "missing_by_family": miss_by,
           "checked_by_family": {f: n * CAP for f, n in PER_EVENT.items()}, "checked": sum(pop.values()),
           "missing_n": sum(miss_by.values()), "requests_with_missing": (requests_with_missing if requests_with_missing is not None else (["r0"] if steps_with_missing else []))}
    rm = {"program_digest": digest, "manifest_digest": "m" * 16, "populations": {"steps": CAP, "identities": sum(pop.values()), "by_family": pop},
          "coverage": [cov, dict(cov)] if coverage else []}
    return {"required_manifest": rm}


def _read(verdict: dict, act: list[int], bot: list[int]):
    return R.required_population(verdict, program_digests={DIGEST: "unlifted"}, events=CAP, bot_steps=bot, act_steps=act,
                                 n_bot_events=len(bot), want_requests=["r0"] if bot else [])


def test_the_region_is_declared_by_the_rule_and_a_commit_that_leaves_exactly_it_unbound_passes():
    act, bot = _declared_region([5, 9, EOS])                       # EOS at index 2 -> a_r 3, n_exec 3 under LAG 0 -> bot = steps 3..11
    assert (act, bot) == (list(range(3)), list(range(3, CAP)))
    ok, f = _read(_verdict(bot), act, bot)
    assert ok is True and f["why"] is None
    assert f["unbound_is_exactly_bot_population"] is True and f["bot_steps"] == bot and f["fixed_population_fully_bound"] is False
    for fam, n in PER_EVENT.items():
        assert f["families"][fam]["unbound"] == f["families"][fam]["unbound_expected"] == n * len(bot)


def test_i_an_unbound_identity_outside_the_declared_region_fails_by_name_and_is_never_read_as_inactive():
    act, bot = _declared_region([5, 9, EOS])
    hole = 1                                                       # an ACTIVE step the Commit did not bind
    ok, f = _read(_verdict(sorted(bot + [hole])), act, bot)
    assert ok is False
    assert "ACTIVE steps with unbound identities [1]" in f["why"] and "coverage hole of the executed population" in f["why"]
    # the region did not grow to swallow the hole: the reader reports the DECLARED region, and the hole beside it -- never "step 1 is inactive"
    assert f["bot_steps"] == bot and hole not in f["bot_steps"]
    assert f["unbound_is_exactly_bot_population"] is False
    assert f["steps_with_missing"] == sorted(bot + [hole])


def test_i_the_hole_is_named_even_when_only_one_family_has_it():
    act, bot = _declared_region([5, 9, EOS])
    v = _verdict(bot)
    cov = v["required_manifest"]["coverage"]
    for c in cov:                                                  # one family leaves ONE more identity unbound, at an active step
        c["steps_with_missing"] = sorted(bot + [2])
        c["missing_by_family"]["logits"] += 1
        c["missing_n"] += 1
    ok, f = _read(v, act, bot)
    assert ok is False and "ACTIVE steps with unbound identities [2]" in f["why"]
    assert "logits: " in f["why"] and "unbound != " in f["why"]  # and the per-family count names which family


def test_ii_a_bound_identity_inside_the_declared_region_fails_by_name_late_activity():
    act, bot = _declared_region([5, 9, EOS])
    late = bot[1:]                                                 # the Commit bound step 3 (the first ⊥ step) -- late activity, rev N-05b
    ok, f = _read(_verdict(late), act, bot)
    assert ok is False
    assert f"steps_with_missing {late[:8]}.. != bot steps {bot[:8]}.." in f["why"]
    assert all("unbound != " in f["why"] for _ in [0]) and f["unbound_is_exactly_bot_population"] is False
    assert f["bot_steps"] == bot                                   # the declaration stands; the map does not shrink it


def test_iii_no_declaration_or_another_program_is_scope_never_a_pass():
    act, bot = _declared_region([5, 9, EOS])
    ok, f = _read(_verdict(bot, coverage=False), act, bot)
    assert ok is None and "no required-manifest coverage record" in f["why"]
    ok2, f2 = _read(_verdict(bot, digest="q" * 64), act, bot)
    assert ok2 is None and "ANOTHER Program" in f2["why"] and f2["manifest_of_fixed_program"] is False
    # a length stop (no ⊥ region) with everything bound: True, and the exactly-⊥ property is not exercised (None), not True
    act_all, bot_none = _declared_region([5, 9, 11, 13, 5, 9, 11, 13, 5, 9, 11, 13])
    assert (act_all, bot_none) == (list(range(CAP)), [])
    ok3, f3 = _read(_verdict([]), act_all, bot_none)
    assert ok3 is True and f3["unbound_is_exactly_bot_population"] is None and f3["fixed_population_fully_bound"] is True


def test_the_prohibited_move_is_visible_the_map_cannot_be_the_declaration():
    """The ONLY way a coverage hole reads as a pass is to hand the reader a region copied from the map -- the fallback R17-42 §5 prohibits.
    The contract makes that a lie the caller must tell explicitly (bot_steps is a required argument, no default), and the production
    callers derive it from the declaration (`expected_served` over the request's a_r / the R9 rule), never from `steps_with_missing`."""
    act, bot = _declared_region([5, 9, EOS])
    hole_verdict = _verdict(sorted(bot + [1]))
    ok, _ = _read(hole_verdict, act, bot)
    assert ok is False
    # the lie: region := the map's unbound steps
    lie_bot = sorted(bot + [1])
    lie_act = [s for s in range(CAP) if s not in lie_bot]
    ok_lie, _ = _read(hole_verdict, lie_act, lie_bot)
    assert ok_lie is True                                          # which is exactly why the region must not come from the map
    sig = inspect.signature(R.required_population)
    assert sig.parameters["bot_steps"].default is inspect.Parameter.empty and sig.parameters["act_steps"].default is inspect.Parameter.empty
    src_rp = inspect.getsource(R.required_population)
    assert "steps_with_missing" in src_rp                          # the reader READS the map's unbound steps ...
    for caller in (R.check_workload_record, R.check_record):
        src = inspect.getsource(caller)
        assert "expected_served(" in src                           # ... and every production caller declares the region by the rule
        for line in src.splitlines():
            if "bot_steps" in line and "=" in line.split("bot_steps", 1)[1][:3]:
                assert "steps_with_missing" not in line and "expected_served" in src


def test_continuation_v1_and_the_reader_agree_on_the_region_under_lag_1():
    """Under LAG 1 the post-EOS row is executed (R17-1): the region starts one step later, from BOTH mechanisms."""
    served = [5, 9, EOS]
    epr = CONT.executed_prefix_by_rule(served, cap=CAP, A=1, eos=[EOS])
    assert epr["ok"] and epr["executed_prefix_len"] == 4 == L.expected_served(3, CAP, 1)
    bot = list(range(4, CAP))
    ok, f = _read(_verdict(bot), list(range(4)), bot)
    assert ok is True and f["bot_steps"] == bot
    ok_l0, f_l0 = _read(_verdict(bot), list(range(3)), list(range(3, CAP)))   # the LAG-0 region against a LAG-1 Commit: step 3 bound inside the region -> named
    assert ok_l0 is False and "!= bot steps" in f_l0["why"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
