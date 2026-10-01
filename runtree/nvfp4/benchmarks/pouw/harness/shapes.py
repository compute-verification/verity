"""The bench's shapes: the panel's two headlines (prefill m8192-n8192-k8192, decode m32-n8192-k8192), m = n = k from 2,048
to 32,768, and the linears of Qwen2.5-7B and Llama-3.1-70B at m = 2,048 (prefill) and m = 32 (decode). (n, k) per linear:
qkv's n is the query width plus two KV widths, gate_up's is twice the intermediate size. Square and headline shapes are
labelled like the panel (`m{m}-n{n}-k{k}`); every shape's `key` is that form."""

from __future__ import annotations

from arm import Shape

SQUARE = (2048, 4096, 8192, 16384, 32768)
#: the panel's headline shapes, by phase (`internal/pouw/panel/panel.py`'s HEADLINE)
HEADLINE = {"prefill": "m8192-n8192-k8192", "decode": "m32-n8192-k8192"}
PREFILL_M, DECODE_M = 2048, 32

#: (n, k) of each linear. Qwen2.5-7B: hidden 3,584, intermediate 18,944, 28 heads and 4 KV heads of 128.
QWEN25_7B = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}
#: Llama-3.1-70B: hidden 8,192, intermediate 28,672, 64 heads and 8 KV heads of 128.
LLAMA31_70B = {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}

MODELS = {"qwen2.5-7b": QWEN25_7B, "llama3.1-70b": LLAMA31_70B}

#: Llama-3.1-8B's distinct linears (hidden 4,096, intermediate 14,336, 32 heads and 8 KV heads of 128), each its own panel
#: row: q and o, k and v, gate and up, down. Prefill at m = 8,192; decode at m = 32 and at m = 64, Pearl-C4's smallest m.
LLAMA31_8B = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (14336, 4096), "down": (4096, 14336)}
LLAMA31_8B_M = {"prefill": (8192,), "decode": (32, 64)}
#: The narrow k/v linears in Pearl-C4's widened B-OVF buckets (n < 1,024) that are in PearlC4Nv's domain (k ≥ 1,024):
#: Qwen2.5-1.5B and 3B (2 KV heads of 128), Llama-3.2-1B (8 of 64) and Qwen2.5-7B (4 of 128). Prefill m 8,192, decode m 64.
NARROW_KV = {"qwen2.5-1.5b": (256, 1536), "qwen2.5-3b": (256, 2048), "llama3.2-1b": (512, 2048), "qwen2.5-7b": (512, 3584)}
NARROW_KV_M = {"prefill": (8192,), "decode": (64,)}


def key(m: int, k: int, n: int) -> str:
    return f"m{m}-n{n}-k{k}"


#: Llama-3.1-8B's and the narrow k/v linears' panel rows' phases, by label
LLAMA31_8B_PANEL = {key(m, k, n): phase for linears, phases in ((LLAMA31_8B, LLAMA31_8B_M), (NARROW_KV, NARROW_KV_M))
                    for phase, ms in phases.items() for m in ms for n, k in linears.values()}


def _shapes() -> tuple[list[Shape], dict[str, str]]:
    """The shapes, and the labels of model linears that repeat an earlier shape (label -> that shape's label)."""
    out = [Shape(key(s, s, s), s, s, s) for s in SQUARE]
    out.append(Shape(HEADLINE["decode"], DECODE_M, 8192, 8192))
    for label in LLAMA31_8B_PANEL:
        if all(s.label != label for s in out):
            m, n, k = (int(x[1:]) for x in label.split("-"))
            out.append(Shape(label, m, k, n))
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
    """The panel phase whose row this shape is (a headline or a Llama-3.1-8B linear), else None."""
    label = key(shape.m, shape.k, shape.n)
    return next((p for p, h in HEADLINE.items() if h == label), None) or LLAMA31_8B_PANEL.get(label)


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
