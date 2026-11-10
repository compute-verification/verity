"""The edge between two Commit windows, on the advice's side: which requests window w leaves for window w + 1 to read, and
where each carried request's history is among them.

A request window w leaves with outputs still to keep (`window_advice.close`'s ``must`` entries) runs on in w + 1 from the state
w left it at, which `window_advice.follows` holds w + 1's advice to. Its history, the rows w + 1 reads and doesn't compute, is
window w's: w returns the ports of `VALUES` after its requests' outputs, and w + 1's Program takes them as registered
parameters (the one-stage registration's ``outputs``, R7), so every read of it opens under w's root, domain and hiding leaves.
The Program's side, each field's type and the reads of it, is `program.workload.Edge`'s, built by `predict.segment.edge_of`.

- **Which requests.** A window's edge members are its `close` entries that must carry and have computed a row, in close order
  (`members`): its carried requests in the order w − 1 left them, then the ones it admitted, by first row. A request that only
  may carry (its last kept output is in w's last step) has no Program in w + 1, and one that never ran has no history.
- **Which fields.** Each port is a Tuple with one field per member holding it, in member order (`fields`). `edge.kv` holds
  every member's latent rows [0, computed) of every layer (a request Program's `prefix`). `edge.ik` and `edge.iks` hold the
  indexer's keys and scales (`prefix_ik`, `prefix_iks`) of each member whose record's last row reads past the model's
  INDEX_TOPK keys (`prompt_len + output_len - 1` > INDEX_TOPK, `predict.segment.edge_groups`), the same in every window; a
  model without an indexer has neither port. `edge.last` holds the last kept output (`last_token`) of each member past its
  prompt.
- **The verifier's side.** `carried(last, advice)` refuses ``advice`` unless it follows ``last``, the window before it
  (`window_advice.NotFollowing`, by name), and gives each carried request that reads history its field in each of ``last``'s
  ports (`Slice`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from . import window_advice as WA

__all__ = ["VALUES", "Slice", "Value", "carried", "fields", "members"]


class Value(NamedTuple):
    """One edge port: its name (a port of window w's record and a registered value of window w + 1's), the request Program
    Input its fields fill, whether a member holds a field only once it is past its prompt, and whether only if its rows read
    past INDEX_TOPK keys."""

    name: str
    history: str
    past_prompt: bool
    past_topk: bool = False


VALUES = (Value("edge.kv", "prefix", False), Value("edge.ik", "prefix_ik", False, True), Value("edge.iks", "prefix_iks", False, True),
          Value("edge.last", "last_token", True))


class Slice(NamedTuple):
    """A carried request's history in the last window's edge: its index in the advice that carries it and its id, its
    history's positions (its carried ``computed``, its Program's C), and its field in each port that holds one."""

    request: int
    id: str
    computed: int
    fields: Mapping[str, int]


def members(advice: WA.Advice) -> tuple[WA.Close, ...]:
    """The requests a window's edge holds, in its order: its `close`'s entries that must carry and have computed a row."""
    return tuple(c for c in WA.close(advice) if c.must and c.state.computed)


def fields(advice: WA.Advice, topk: int | None = None) -> dict[str, tuple[int, ...]]:
    """Each port's members holding a field, as indices in ``advice``, in field order. ``topk``: the model's INDEX_TOPK, None for a
    model without an indexer (no `edge.ik` or `edge.iks`)."""
    held = members(advice)

    def holds(v: Value, c: WA.Close) -> bool:
        q = advice.requests[c.request]
        return (not v.past_prompt or c.state.computed >= q.prompt_len) and (not v.past_topk or q.prompt_len + q.output_len - 1 > topk)
    return {v.name: tuple(c.request for c in held if holds(v, c)) for v in VALUES if topk is not None or not v.past_topk}


def carried(last: WA.Advice, advice: WA.Advice, topk: int | None = None) -> tuple[Slice, ...]:
    """The verifier's: ``advice`` held to ``last``, the window before it (`window_advice.follows`, refusing by name), and each
    carried request's fields in ``last``'s edge (`fields` at ``topk``), in ``advice``'s order. A carried request that isn't one
    of ``last``'s members reads no history and has no slice."""
    pairs = WA.follows(last, advice)
    held = fields(last, topk)
    at = set(held[VALUES[0].name])
    return tuple(Slice(k, advice.requests[k].id, advice.carried[k].computed,
                       {p: f.index(i) for p, f in held.items() if i in f}) for k, i in pairs if i in at)
