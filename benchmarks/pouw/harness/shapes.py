"""The bench's shapes: the panel's two headlines (prefill m8192-n8192-k8192, decode m32-n8192-k8192), m = n = k from 2,048
to 32,768, and Llama-3.1-70B's linears (`verity_bench.overhead.SHAPES`) at m = 2,048 (prefill) and m = 32 (decode). Square
and headline shapes are labelled like the panel (`m{m}-n{n}-k{k}`); every shape's `key` is that form."""

from __future__ import annotations

import re

from arm import Shape
from verity_bench.overhead import SHAPES

#: what `bench.py` and the tests read
__all__ = ["ALIASES", "DECODE_M", "HEADLINE", "all_shapes", "key", "phase_of", "select"]

SQUARE = (2048, 4096, 8192, 16384, 32768)
#: the panel's headline shapes, by phase
HEADLINE = {"prefill": "m8192-n8192-k8192", "decode": "m32-n8192-k8192"}
PREFILL_M, DECODE_M = 2048, 32

MODELS = {"llama3.1-70b": SHAPES["llama-3.1-70b"].linears}


def key(m: int, k: int, n: int) -> str:
    return f"m{m}-n{n}-k{k}"


def _shapes() -> tuple[list[Shape], dict[str, str]]:
    """The shapes, and the labels of model linears that repeat an earlier shape (label -> that shape's label)."""
    out = [Shape(key(s, s, s), s, s, s) for s in SQUARE]
    out.append(Shape(HEADLINE["decode"], DECODE_M, 8192, 8192))
    aliases: dict[str, str] = {}
    for model, linears in MODELS.items():
        for phase, m in (("prefill", PREFILL_M), ("decode", DECODE_M)):
            for name, (n, k) in linears.items():
                label = f"{model}-{name}-{phase}"
                same = next((s for s in out if (s.m, s.k, s.n) == (m, k, n)), None)
                if same:
                    aliases[label] = same.label
                else:
                    out.append(Shape(label, m, k, n))
    return out, aliases


def all_shapes() -> list[Shape]:
    return _shapes()[0]


ALIASES = _shapes()[1]


def phase_of(shape: Shape) -> str | None:
    """The panel phase whose headline this shape is, else None."""
    return next((p for p, label in HEADLINE.items() if key(shape.m, shape.k, shape.n) == label), None)


KEY_RE = re.compile(r"m([1-9]\d*)-n([1-9]\d*)-k([1-9]\d*)")


def select(labels: list[str] | None) -> list[Shape]:
    """The shapes whose labels are given (all when none are; `headline` names both headlines), in the bench's order, then any
    other shape given by its key (`m{m}-n{n}-k{k}`, as `verity bench overhead` sweeps m), in the order given; any other
    label is an error."""
    shapes = all_shapes()
    if not labels:
        return shapes
    want = {ALIASES.get(x, x) for x in labels}
    if "headline" in want:
        want = (want - {"headline"}) | set(HEADLINE.values())
    known = {s.label for s in shapes}
    keyed = [x for x in dict.fromkeys(labels) if x in want - known and KEY_RE.fullmatch(x)]
    unknown = sorted(want - known - set(keyed))
    if unknown:
        raise KeyError(f"unknown shapes {unknown}")
    extra = []
    for x in keyed:
        m, n, k = map(int, KEY_RE.fullmatch(x).groups())
        extra.append(Shape(x, m, k, n))
    return [s for s in shapes if s.label in want] + extra
