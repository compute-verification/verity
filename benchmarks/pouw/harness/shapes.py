"""The bench's shapes: the panel's two headlines (prefill m8192-n8192-k8192, decode m32-n8192-k8192), m = n = k from 2,048
to 32,768, and the linears of Qwen2.5-7B and Llama-3.1-70B at m = 2,048 (prefill) and m = 32 (decode). (n, k) per linear:
qkv's n is the query width plus two KV widths, gate_up's is twice the intermediate size. Square and headline shapes are
labelled like the panel (`m{m}-n{n}-k{k}`); every shape's `key` is that form."""

from __future__ import annotations

from arm import Shape

SQUARE = (2048, 4096, 8192, 16384, 32768)
#: the panel's headline shapes, by phase
HEADLINE = {"prefill": "m8192-n8192-k8192", "decode": "m32-n8192-k8192"}
PREFILL_M, DECODE_M = 2048, 32

#: (n, k) of each linear. Qwen2.5-7B: hidden 3,584, intermediate 18,944, 28 heads and 4 KV heads of 128.
QWEN25_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
#: Llama-3.1-70B: hidden 8,192, intermediate 28,672, 64 heads and 8 KV heads of 128.
LLAMA31_70B = {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}

MODELS = {"qwen2.5-7b": QWEN25_7B, "llama3.1-70b": LLAMA31_70B}


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


def select(labels: list[str] | None) -> list[Shape]:
    """The shapes whose labels are given (all when none are; `headline` names both headlines), in the bench's order; an
    unknown label is an error."""
    shapes = all_shapes()
    if not labels:
        return shapes
    want = {ALIASES.get(x, x) for x in labels}
    if "headline" in want:
        want = (want - {"headline"}) | set(HEADLINE.values())
    known = {s.label for s in shapes}
    unknown = sorted(want - known)
    if unknown:
        raise KeyError(f"unknown shapes {unknown}")
    return [s for s in shapes if s.label in want]
