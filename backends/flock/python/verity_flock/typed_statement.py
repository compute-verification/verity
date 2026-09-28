"""``verity/flock-circuit/types``: the circuit statement with its unit as circuit types, not rows (Project doc
`constant-api-public.md` §2.2; the constant design's step 2, under Daniel's rows decision). A statement carries the types,
the partition and the layouts. The verifier derives the unit's rows itself (`Flock.Derive`, the soundness lane's), and the
prover gets them as data beside the statement (:attr:`Typed.unit_rows`, :mod:`derive`'s mirror), so staging needs no Lean.

It is :func:`circuit.compose`'s statement with two changes:
- **META** names the new statement (`FORMAT`, `STATEMENT`) and gains `types` (each circuit type's content by its digest),
  `layouts`, `unit_layout`, `unit_type`, `policy` and `unit_in` (the unit's constant-bound inputs with their binding
  times; none yet). Every other key is unchanged, `unit_sha256` included: the SHA-256 of the lowering's unit text, the
  derived rows followed by its `LEAVES` (and `CUT`) lines, as before.
- **The unit's `CIRCUIT` section is gone.** The row hashing's sections stay, as the block level does.

This version covers the flat classes, units without calls or reads and templates without a tail. Its rows are the old
statement's byte for byte (checked on every compose). Templates with a tail (attention, GEMM) place their stages as the
unit's own rows and their reads as generated reads, the next step.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

from verity.ml import library

from . import circuit as C
from . import circuit_types as CT
from . import derive as D
from . import ir_lower as IL
from . import layouts as LY

FORMAT = "flock-circuit/types"
STATEMENT = "verity/flock-circuit/types"
POLICY = "calls/v1"


@dataclass
class Typed:
    """A typed statement, and the prover's copy of the unit's rows (``flock-ir-unit/v2``), derived from its types."""
    text: str
    unit_rows: str
    types: dict
    layouts: list


def unit_type(low: IL.Lowering) -> CT.CircuitType:
    """The unit circuit as a flat circuit type, its ports ``in0 ..`` and ``out0 ..`` one word each."""
    U, circ = low.units, low.circuit
    ins = [b for j in range(len(U.in_widths)) for b in circ.inputs[f"in{j}"]]
    outs = [f for j in range(len(U.out_widths)) for f in circ.outputs[f"out{j}"]]
    return CT.from_gf2(circ, ins, outs, [(1, w) for w in U.in_widths], [(1, w) for w in U.out_widths], library.PROGRAM, {})


def _split(groups) -> tuple[int, ...]:
    """The port indices where a layout's second and later port groups start."""
    return tuple(itertools.accumulate(len(bits) for _, _, bits in groups))[:-1]


def unit_layout(low: IL.Lowering, t: CT.CircuitType) -> LY.Layout:
    """The unit's layout: placed by the block (no range), its port groups as the lowering laid them out."""
    return LY.Layout(t.digest(), None, _split(low.layout.in_groups), _split(low.layout.out_groups), ("derive",), (), ())


def sections(text: str) -> tuple[str, dict, list[tuple[str, str]]]:
    """A circuit file's first line, its META, and its ``CIRCUIT`` sections in order as (name, text)."""
    lines = text.split("\n")
    head, meta = lines[0], json.loads(lines[1][len("META "):])
    out, i = [], 2
    while i < len(lines) and lines[i]:
        _, name, count = lines[i].split(" ")
        n = int(count)
        out.append((name, "\n".join(lines[i + 1:i + 1 + n]) + "\n"))
        i += 1 + n
    return head, meta, out


def compose(low: IL.Lowering, sub, **kw) -> Typed:
    """The typed statement for a flat class (``kw`` as :func:`circuit.compose` takes them)."""
    U = low.units
    if U.cut or U.tail_rets:
        raise NotImplementedError(f"{low.name}: a template with a tail; its stages become the unit's own rows in the next step")
    text = C.compose(low, sub, **kw)
    _, meta, secs = sections(text)
    t = unit_type(low)
    types, layouts = {t.digest(): t}, [unit_layout(low, t)]
    dv = D.Derivation(types, layouts, 0, {})
    rows = D.netlist(dv.unit, low.name)
    old = dict(secs)["unit"]
    if rows != old:
        raise AssertionError(f"{low.name}: derive does not reproduce the unit's rows")
    meta.update(format=FORMAT, statement=STATEMENT, types={d: x.content() for d, x in types.items()},
                layouts=[x.to_json() for x in layouts], unit_layout=0, unit_type=t.digest(), policy=POLICY, unit_in=[])
    head = f"{FORMAT} {low.name}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"
    body = "".join(f"CIRCUIT {nm} {s.count(chr(10))}\n{s}" for nm, s in secs if nm != "unit")
    return Typed(head + body, rows, types, layouts)


__all__ = ["FORMAT", "POLICY", "STATEMENT", "Typed", "compose", "sections", "unit_layout", "unit_type"]
