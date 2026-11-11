"""The collector's plan key binds the step GEOMETRY, not only (ntok, nreq).

The B8 mixed 1024/128 bounded record run failed closed at steps 35 and 80: both ntok 4 / nreq 4, the same
four live requests 45 decode positions apart, 78,892,288 vs 87,297,280 staged bytes -- the FA3 hidden M1 plane is sized by KV length.
A plan keyed by counts alone made the second step a hit-plan step with the wrong plan; bounded, there is no host copy to re-hash from.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer.native_collect import plan_key_of  # noqa: E402


def _sched(reqs: list[tuple[str, int, int]], cached_from: int = 0):
    """reqs = [(req_id, num_computed_tokens, num_scheduled_tokens)]; the first `cached_from` are NEW requests, the rest cached."""
    new = [SimpleNamespace(req_id=r, num_computed_tokens=c) for r, c, _ in reqs[:cached_from]]
    rest = reqs[cached_from:]
    cached = SimpleNamespace(req_ids=[r for r, _, _ in rest], num_computed_tokens=[c for _, c, _ in rest])
    return SimpleNamespace(scheduled_new_reqs=new, scheduled_cached_reqs=cached,
                           num_scheduled_tokens={r: n for r, _, n in reqs}, total_num_scheduled_tokens=sum(n for _, _, n in reqs))


def test_same_counts_different_kv_lengths_are_different_plans():
    # steps 35 and 80 of the B8 row: the same four requests, 45 positions later
    s35 = _sched([("r1", 11 + 35, 1), ("r3", 365 + 35, 1), ("r5", 633 + 35, 1), ("r7", 817 + 35, 1)])
    s80 = _sched([("r1", 11 + 80, 1), ("r3", 365 + 80, 1), ("r5", 633 + 80, 1), ("r7", 817 + 80, 1)])
    k35, m35 = plan_key_of(s35, 4, 4)
    k80, m80 = plan_key_of(s80, 4, 4)
    assert m35 == m80 == "geometry"
    assert k35 != k80
    assert 0 < k35 < 2**63 and 0 < k80 < 2**63          # fits the collector's int64 plan_key


def test_the_key_is_a_multiset_over_requests_and_stable():
    a = _sched([("r1", 46, 1), ("r3", 400, 1), ("r5", 668, 1), ("r7", 852, 1)])
    b = _sched([("r7", 852, 1), ("r5", 668, 1), ("r3", 400, 1), ("r1", 46, 1)])     # permuted batch rows: the same staged shapes
    c = _sched([("x", 46, 1), ("y", 400, 1), ("z", 668, 1), ("w", 852, 1)])          # request ids are not bound, geometry is
    assert plan_key_of(a, 4, 4) == plan_key_of(b, 4, 4) == plan_key_of(c, 4, 4)
    assert plan_key_of(a, 4, 4) == plan_key_of(a, 4, 4)                              # deterministic across calls (blake2b, not hash())


def test_new_and_cached_requests_both_count_and_a_prefill_step_keys_by_its_prompt_lengths():
    step0 = _sched([("r0", 0, 1024), ("r1", 0, 11), ("r2", 0, 165), ("r3", 0, 365), ("r4", 0, 449), ("r5", 0, 633), ("r6", 0, 73), ("r7", 0, 817)], cached_from=8)
    k0, m0 = plan_key_of(step0, 3537, 8)
    assert m0 == "geometry"
    other = _sched([("r0", 0, 1025), ("r1", 0, 10), ("r2", 0, 165), ("r3", 0, 365), ("r4", 0, 449), ("r5", 0, 633), ("r6", 0, 73), ("r7", 0, 817)], cached_from=8)
    assert plan_key_of(other, 3537, 8)[0] != k0                                       # same ntok/nreq, different per-request lengths


def test_the_step_keyed_sources_names_are_bound_and_none_leaves_the_geometry_key():
    s = _sched([("r1", 46, 1)])
    assert plan_key_of(s, 1, 1, b"") == plan_key_of(s, 1, 1)
    keyed, mode = plan_key_of(s, 1, 1, bytes(16))
    assert mode == "geometry" and keyed != plan_key_of(s, 1, 1)[0] and 0 < keyed < 2**63
    assert plan_key_of(s, 1, 1, b"\1" * 16)[0] != keyed


def test_counts_fallback_is_named_when_the_scheduler_output_lacks_geometry():
    legacy = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}, total_num_scheduled_tokens=2)
    k, m = plan_key_of(legacy, 2, 2)
    assert m == "counts" and k == (2 << 16) | 2
    # a geometry that does not add up to (ntok, nreq) is not trusted either
    bad = _sched([("a", 5, 1), ("b", 9, 1)])
    assert plan_key_of(bad, 3, 2)[1] == "counts"


def test_a_geometry_key_is_pinned_and_a_malformed_geometry_falls_back_to_counts():
    s = _sched([("r0", 0, 7), ("r1", 46, 1), ("r3", 400, 1)], cached_from=1)
    assert plan_key_of(s, 9, 3) == (5939804984935576480, "geometry")
    no_len = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="r0", num_computed_tokens=None)], scheduled_cached_reqs=None,
                             num_scheduled_tokens={"r0": 7})
    assert plan_key_of(no_len, 7, 1) == ((7 << 16) | 1, "counts")
    short = SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"], num_computed_tokens=[3]),
                            num_scheduled_tokens={"a": 1, "b": 1})
    assert plan_key_of(short, 2, 2) == ((2 << 16) | 2, "counts")
