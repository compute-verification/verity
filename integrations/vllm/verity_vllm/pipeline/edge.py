"""The edge between two Commit periods: the rows period w leaves for period w + 1 to read, and where each carried request's
history is among them.

A request period w leaves with outputs still to keep (`window_advice.close`'s ``must`` entries) runs on in w + 1 from the state
w left it at, which `window_advice.follows` holds w + 1's advice to. Its history, the rows w + 1 reads and doesn't compute (a
request Program's `prefix`, `prefix_ik`, `prefix_iks` and `last_token` Inputs, `build.build_program`'s C), is period w's: w
outputs the four values of `VALUES` as ports of its record, and w + 1 registers each as an output value of that record (the
one-stage registration's ``outputs``, R7), so every read of it opens under w's root, domain and hiding leaves.

- **Which requests.** The edge holds w's `close` entries that must carry, in close order, which is w's request order: its
  carried requests in the order w − 1 left them, then the ones it admitted, by first row (`requests`). A request that only may
  carry (its last kept output is in w's last step) isn't in it: w + 1's rows of it are past its last kept output, so it has no
  Program there and reads no history.
- **The rows.** Edge request j's slice follows requests 0 … j − 1 in every value (`layout`). In `edge.kv` it is its rows
  [0, computed) of every layer, layer-major, as `prefix` [layers, C, latent] flattens: ``layers * computed`` rows of `latent`
  bf16. `edge.ik` and `edge.iks` are the same over the indexer layers, a row `prefix_ik`'s `index_head_dim` e4m3 bytes, two to a
  16-bit word, or `prefix_iks`' one f32. `edge.last` is one row, its last kept output, an i32 (0 when it has kept none). A
  request with no rows in w is still in w's edge, its slice the one w read from w − 1's, so a period's history reads name the
  period before it alone. Each value is zero-padded after its last request to the next power of two of its rows (`padded`),
  since its leaf count is public (`sizes`).
- **The verifier's side.** `carried(last, advice, shape)` refuses ``advice`` unless it follows ``last``, the period before it
  (`window_advice.NotFollowing`, by name), and gives each carried request that reads history its slice of ``last``'s edge
  (`Slice`): its place there, its history's length (its carried ``computed``, its Program's C) and, given the model's `Shape`,
  each value's rows.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from . import window_advice as WA


class Value(NamedTuple):
    """One edge value: its name (the port of period w's record and the registered value of period w + 1's), the request
    Program Input its rows fill, its leaves' width and what one request's slice holds a row of at each position (``layer``,
    ``indexer`` or, a row a request, ``request``)."""

    name: str
    history: str
    word_bits: int
    per: str


VALUES = (Value("edge.kv", "prefix", 16, "layer"), Value("edge.ik", "prefix_ik", 16, "indexer"),
          Value("edge.iks", "prefix_iks", 32, "indexer"), Value("edge.last", "last_token", 32, "request"))


@dataclass(frozen=True, slots=True)
class Shape:
    """A model's edge geometry: its layers, its indexer layers, a latent row's bf16 words and an indexer key's e4m3 bytes."""

    layers: int
    indexer_layers: int
    latent: int
    index_head_dim: int

    def __post_init__(self) -> None:
        for name in ("layers", "indexer_layers", "latent", "index_head_dim"):
            if type(v := getattr(self, name)) is not int or v < 1:
                raise ValueError(f"an edge shape's {name} is a whole number of at least 1, not {v!r}")
        if self.index_head_dim % 2:
            raise ValueError(f"an indexer key's {self.index_head_dim} e4m3 bytes don't fill whole 16-bit words")

    def words(self, value: Value) -> int:
        """A row's leaves."""
        return {"edge.kv": self.latent, "edge.ik": self.index_head_dim // 2}.get(value.name, 1)

    def rows(self, value: Value, computed: int) -> int:
        """A request's rows in ``value`` when its history is ``computed`` positions."""
        return {"layer": self.layers * computed, "indexer": self.indexer_layers * computed, "request": 1}[value.per]


class Slice(NamedTuple):
    """A carried request's history in the last period's edge: its index in the advice that carries it and its id, its place
    among the edge's requests, its history's positions (its carried ``computed``, its Program's C) and, with a `Shape`, each
    value's rows [lo, hi)."""

    request: int
    id: str
    edge: int
    computed: int
    rows: Mapping[str, tuple[int, int]]


def requests(advice: WA.Advice) -> tuple[WA.Close, ...]:
    """The requests a period's edge holds, in its order: its `close`'s entries that must carry."""
    return tuple(c for c in WA.close(advice) if c.must)


def layout(advice: WA.Advice, shape: Shape) -> dict[str, tuple[tuple[int, int], ...]]:
    """Each edge value's rows [lo, hi) of each of the period's edge requests (`requests`), in order."""
    out: dict[str, tuple[tuple[int, int], ...]] = {}
    for v in VALUES:
        at, spans = 0, []
        for c in requests(advice):
            n = shape.rows(v, c.state.computed)
            spans.append((at, at + n))
            at += n
        out[v.name] = tuple(spans)
    return out


def padded(rows: int) -> int:
    """A value's leaf count of rows: the next power of two of them, at least 1."""
    if type(rows) is not int or rows < 0:
        raise ValueError(f"a value's rows are a whole number, not {rows!r}")
    return 1 << max(rows - 1, 0).bit_length()


def sizes(advice: WA.Advice, shape: Shape) -> dict[str, dict[str, int]]:
    """Each edge value of the period as its record names it: its rows padded (``leaves``), a row's ``words`` and their
    ``word_bits``, beside the rows its requests fill (``rows``)."""
    spans = layout(advice, shape)
    return {v.name: {"rows": (n := spans[v.name][-1][1] if spans[v.name] else 0), "leaves": padded(n), "words": shape.words(v),
                     "word_bits": v.word_bits} for v in VALUES}


def carried(last: WA.Advice, advice: WA.Advice, shape: Shape | None = None) -> tuple[Slice, ...]:
    """The verifier's: ``advice`` held to ``last``, the period before it (`window_advice.follows`, refusing by name), and the
    slice of ``last``'s edge each carried request reads its history from, in ``advice``'s order. A carried request ``last``
    only may carry isn't in its edge and has none. Without ``shape`` a slice's ``rows`` is empty."""
    pairs = WA.follows(last, advice)
    at = {c.request: j for j, c in enumerate(requests(last))}
    spans = layout(last, shape) if shape is not None else {}
    return tuple(Slice(k, advice.requests[k].id, at[i], advice.carried[k].computed, {v: s[at[i]] for v, s in spans.items()})
                 for k, i in pairs if i in at)
