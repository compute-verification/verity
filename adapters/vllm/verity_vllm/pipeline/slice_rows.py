"""A ``SLICES`` template's units (`serving_rows`' ``slice`` layout, `serving_templates.SLICES`): a batched row split along its
family's axes, unit (i_0, i_1, ..) reading element i_k along axis k of each parameter that carries it and the others whole, and
returning the row's output element (i_0, i_1, ..).  `serving_rows.read` reads a slice member through :func:`read`, with its own
reader of a row's argument groups."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from verity.ml.windows import rows as CS
from verity_vllm.pipeline.serving_templates import SLICES, Template, _bind, template

_LAYER = re.compile(r"\.layers\.(\d+)\.")


def extent(tp: Template, statics: Mapping[str, Any]) -> int:
    """A row's units for slice template ``tp``: the product of the row's extents along its axes."""
    from verity_vllm.query.program_view import spec_base

    return math.prod(int(statics[a]) for a in SLICES[spec_base(tp.word)][0])


def read(m: CS.Member, rows: Sequence[tuple[str, Any, dict, dict]], store: Any, weights: Callable, arrivals: dict[str, int] | None,
         slices: Callable) -> tuple[dict[str, Any], list[list[Any]]]:
    """Member ``m``'s units, row by row, then unit (i_0, i_1, ..) in row-major order of the row's extents along the template's
    axes: each argument group (``slices(pi, store, weights, rid, arrive, group, bits=b)``, `serving_rows._slices`) its parameter's element
    i_k along each axis it carries and whole along the others, and the output the row's output element (i_0, i_1, ..), the
    parameters' dimensions those of the row's Definition; every port its elements' bytes in u16 words (``Template.packed``).  One
    index entry per row, the range of its units."""
    from verity_vllm.query.program_view import spec_base

    tp = template(m.template)
    axes, carry = SLICES[spec_base(tp.word)]
    bits = tp.packed or (16,) * (len(m.rows) + 1)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    (port, _w), = m.outs
    entries: list[list[Any]] = []
    lo = m.base
    for rid, pi, r, d in rows:
        arrive, what = int((arrivals or {}).get(rid, 0)), f"row {r['i']} ({d['op_path']})"
        ext = [int(d["statics"][a]) for a in axes]
        n, fn = math.prod(ext), _bind(r["spec"])
        params = fn.signature()[0] if fn is not None else ()
        if not len(r["args"]) == len(params) == len(carry) == len(m.rows):
            raise ValueError(f"{what}: {len(r['args'])} argument groups of {len(params)} parameters, not member {m.key}'s "
                             f"{len(m.rows)} ports")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        got = [per_unit(elements(slices(pi, store, weights, rid, arrive, g, bits=b), b, f"{what} argument {k}"), dims(ty), c, ext,
                        f"{what} argument {k}") for k, (g, b, (_nm, ty), c) in enumerate(zip(r["args"], bits, params, carry))]
        ys = elements([np.asarray(y)], bits[-1], f"{what} output {member}")
        got.append(per_unit(ys, [*ext, ys.size // n], range(len(ext)), ext, f"{what} output {member}"))
        if [a.shape[1] for a in got] != [w for _, w in m.rows + m.outs]:
            raise ValueError(f"{what}: a unit's ports of {[a.shape[1] for a in got]} words, not member {m.key}'s {list(m.rows)} and "
                             f"{list(m.outs)}")
        for p, a in zip([p for p, _ in m.rows] + [port], got, strict=True):
            cols[p].append(a)
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + n, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                        int(d["row"]), int(r["i"])])
        lo += n
    if lo - m.base != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo - m.base} units, not its {m.n} (a chunk is read with its template's others, "
                         f"`read_all`)")
    return {part: {p: np.concatenate(cols[p]).astype(np.uint16, copy=False) for p, _ in ports}
            for part, ports in (("rows", m.rows), ("outs", m.outs))}, entries


def dims(ty: Any) -> list[int]:
    """A parameter type's dimensions, outermost first (none for a scalar)."""
    from verity.ml.circuits.types import Array

    out = []
    while isinstance(ty, Array):
        out.append(ty.n)
        ty = ty.elem
    return out


def elements(parts: Sequence[np.ndarray], bits: int, what: str) -> np.ndarray:
    """A port's elements as read, ``bits`` wide each, as little-endian unsigned words."""
    for a in parts:
        if np.asarray(a).dtype.itemsize * 8 != bits:
            raise ValueError(f"{what}: {np.asarray(a).dtype} elements, the template's port is {bits}-bit elements")
    return np.concatenate([np.ascontiguousarray(a, dtype=np.asarray(a).dtype.newbyteorder("<")).reshape(-1).view(f"<u{bits // 8}")
                           for a in parts])


def per_unit(e: np.ndarray, shape: Sequence[int], carried: Sequence[int | None], ext: Sequence[int], what: str) -> np.ndarray:
    """Elements ``e`` of dimensions ``shape`` per unit of extents ``ext`` (row-major): unit (i_0, ..)'s element i_k along dimension
    ``carried[k]`` and every element along the others, as u16 words of their bytes."""
    if e.size != math.prod(shape) or any(j is not None and shape[j] != x for j, x in zip(carried, ext)):
        raise ValueError(f"{what}: {e.size} elements, not its dimensions {list(shape)} at extents {list(ext)} along {list(carried)}")
    on = [j for j in carried if j is not None]
    a = np.moveaxis(e.reshape(list(shape) or [1]), on, list(range(len(on))))
    a = a.reshape([x if j is not None else 1 for j, x in zip(carried, ext)] + list(a.shape[len(on):]))
    a = np.ascontiguousarray(np.broadcast_to(a, list(ext) + list(a.shape[len(ext):]))).reshape(math.prod(ext), -1)
    if a.shape[1] * a.itemsize % 2:
        raise ValueError(f"{what}: a unit's {a.shape[1]} elements of {8 * a.itemsize} bits are not whole u16 words")
    return a.view("<u2")
