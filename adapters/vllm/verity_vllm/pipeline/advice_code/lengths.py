"""Advice v1's length terms: each request's output_len and finish, sent and charged while the registration says so
(`codec.Registered.send_lengths`).

No proof yet holds the length rule, that a response ends at its first stop id or at its `max_tokens`. A prover that
controls where a response ends chooses its output_len, about log2(max_tokens + 1) bits a request, so while the
registration sends lengths v1 charges each request's length term:
- its output_len in width(max_tokens + 1) = ⌈log2(max_tokens + 1)⌉ bits, most significant first, then its finish in
  1 bit (I1's `FINISHES` order: 0 `eos`, 1 `length`);
- in the plain tail (`window_code.decode_tailed`) of the window the request's body lands in, ⌊end / T⌋, the window's
  requests in release order (`terms`). The widths are the record's, so the decoder reads the tail before the replay.

`check` refuses a string whose sent output_len isn't the certified ids' count, or whose finish isn't the stop rule's,
so what is sent is the record's and its bits are the charge. A request's length first changes the rows at its
admission step (its rows begin there, and the pool bound counts its blocks from there), so `check_admitted` refuses a
replay that admits a request at a step whose window precedes its body's: the term is never charged later than A5 asks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from verity.protocols.certifier.verifier import advice as I1
from verity_vllm.pipeline.advice_code import arith as C
from verity_vllm.pipeline.advice_code import derived as D

__all__ = ["check", "check_admitted", "tails", "terms", "widths"]


def terms(requests: Sequence[Any], release: Sequence[int], T: int, W: int) -> list[list[int]]:
    """Each of the W windows' length terms: the ordinals of the requests whose body lands in it (`end` // T), in the
    release order `release`."""
    out: list[list[int]] = [[] for _ in range(W)]
    for k in release:
        out[requests[k].end // T].append(k)
    return out


def widths(requests: Sequence[Any], ks: Sequence[Sequence[int]]) -> list[list[int]]:
    """Each window's tail widths: per term, width(max_tokens + 1) for output_len, then 1 for finish."""
    return [[n for k in w for n in (C.width(requests[k].max_tokens + 1), 1)] for w in ks]


def tails(requests: Sequence[Any], ks: Sequence[Sequence[int]], cfg: Any) -> list[list[tuple[int, int]]]:
    """Each window's tail, (value, width): each term's certified output_len, then the stop rule's finish."""
    out = []
    for w in ks:
        tail: list[tuple[int, int]] = []
        for k in w:
            q = requests[k]
            finish = D.stop_rule(q.output, len(q.prompt), q.max_tokens, cfg)
            tail += [(len(q.output), C.width(q.max_tokens + 1)), (I1.FINISHES.index(finish), 1)]
        out.append(tail)
    return out


def check(sent: Sequence[Sequence[int]], requests: Sequence[Any], ks: Sequence[Sequence[int]], cfg: Any,
          error: Callable[[str], Exception]) -> None:
    """Refuses (`error`) a sent output_len that isn't the certified ids' count, or a sent finish that isn't the stop
    rule's."""
    for w, got, want in zip(ks, sent, tails(requests, ks, cfg), strict=True):
        for j, k in enumerate(w):
            q, (n, f) = requests[k], (got[2 * j], got[2 * j + 1])
            if n != want[2 * j][0]:
                raise error(f"request {q.id}: the string sends output_len {n}, and its certified output ids number "
                            f"{want[2 * j][0]}")
            if f != want[2 * j + 1][0]:
                raise error(f"request {q.id}: the string sends finish {I1.FINISHES[f]!r}, and the stop rule's is "
                            f"{I1.FINISHES[want[2 * j + 1][0]]!r}")


def check_admitted(requests: Sequence[Any], order: Sequence[int], admitted: Sequence[int], window_of: Callable[[int], int],
                   T: int, error: Callable[[str], Exception]) -> None:
    """Refuses (`error`) a replay that admits arrival p (request `order[p]`) at step `admitted[p]`, whose window
    `window_of` gives, before the window its terms are charged to: its length term and, under a draft, its kept term
    (`kept`)."""
    for p, k in enumerate(order):
        q, s = requests[k], admitted[p]
        if window_of(s) < q.end // T:
            raise error(f"request {q.id}: admitted at step {s}, whose window {window_of(s)} precedes window "
                        f"{q.end // T}, where its terms are charged")
