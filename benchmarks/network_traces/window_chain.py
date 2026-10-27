"""A served endpoint's Commit windows' advice (`verity_vllm.pipeline.window_advice`) as one I1 v0 advice
(`verity.core.protocols.certifier.advice`), each request in it once.

A Commit window that carries requests in (window advice v1) opens with the state its predecessor's advice closed with
(`window_advice.follows`): its carried requests' records and their state at the open repeat what the window before it
already holds. `join` takes that state from the earlier window, never from the later one: window w + 1's carried
requests are window w's, the same request of the joined advice, and its rows continue them as the engine ran them, with
each request's record and every step once. So the joined advice's bits charge a carried request once, whatever its
windows' strings repeat.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from verity.core.protocols.certifier import advice as I1
from verity_vllm.pipeline import window_advice as WA


def must(advice: WA.Advice) -> list[str]:
    """The ids of the requests ``advice`` ends with outputs still to keep: the next window must carry them."""
    return [advice.requests[c.request].id for c in WA.close(advice) if c.must]


def join(windows: Sequence[tuple[int, WA.Advice]], release: Callable[[int, str, int, int], int]) -> I1.Advice:
    """The Commit windows ``windows`` (each its number and advice, in order) as one I1 v0 advice: each window's requests
    and steps in turn, a carried request the earlier window's, so each of its rows names the request its earlier window
    admitted, and kept output λ of request ``id``, which window k released in its own ``bucket``, released at
    ``release(k, id, λ, bucket)``. Refused (MalformedAdvice) unless every window that carries a request in follows the
    window before it here, numbered one less (`window_advice.follows`), every window the next one here doesn't follow
    leaves nothing it must carry, and no window aborts a request, which I1 v0 can't say."""
    requests: list[I1.RequestAdvice] = []
    steps: list[I1.StepAdvice] = []
    buckets: list[int] = []
    index: dict[str, int] = {}
    last: tuple[int, WA.Advice] | None = None
    for k, (w, a) in enumerate(windows):
        if a.version == WA.ABORTS:
            raise I1.MalformedAdvice(f"Commit window {w} aborts a request, which I1 v0 can't say")
        if last is not None and last[0] == w - 1:
            try:
                WA.follows(last[1], a)
            except I1.MalformedAdvice as err:
                raise I1.MalformedAdvice(f"Commit window {w} doesn't open where window {w - 1} closed: {err}") from None
        else:
            if last is not None and (left := must(last[1])):
                raise I1.MalformedAdvice(f"Commit window {last[0]} ends request {left[0]!r} with outputs still to "
                                         f"keep, and window {last[0] + 1} isn't among the windows")
            if a.carried:
                raise I1.MalformedAdvice(f"Commit window {w} carries {len(a.carried)} requests in from window {w - 1}, "
                                         "which isn't among the windows")
        n = len(a.carried)
        local = [index[q.id] for q in a.requests[:n]]
        for q in a.requests[n:]:
            index[q.id] = len(requests)
            local.append(len(requests))
            requests.append(q.i1())
        steps += [I1.StepAdvice(tuple(replace(r, request=local[r.request],
                                              sources=tuple(replace(s, request=local[s.request]) for s in r.sources))
                                      for r in st.rows)) for st in a.steps]
        buckets += [release(k, a.requests[o.request].id, o.index, b) for o, b in zip(WA.outputs(a), a.release)]
        last = (w, a)
    if last is not None and (left := must(last[1])):
        raise I1.MalformedAdvice(f"Commit window {last[0]} ends request {left[0]!r} with outputs still to keep, and "
                                 f"window {last[0] + 1} isn't among the windows")
    return I1.Advice(tuple(requests), tuple(steps), tuple(buckets))
