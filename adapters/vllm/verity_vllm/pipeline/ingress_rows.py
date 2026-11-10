"""A served window's window ingress (`interface.listed_ingress`), which `serving_rows.circuit_record` declares beside its listed
members: the input ports of its units that read only committed members of Program rows before the window, at their engine
steps.  It is not the run's ingress (the ingress gates Lean's ``freeInputs`` reads): only the link checks bind those rows
to their writers, so the leak count subtracts it only in the window figure, which names that premise
(``records_audit.declared_count``).  It
reads the Programs' rows alone, public structure, never a value, so the registered compiler's record of the advice
(``prove_path.py circuit``) states the same window ingress as the prover's."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS
from verity_vllm.pipeline.serving_templates import part, template
from verity_vllm.program.compact import runs_from_file_items

Row = tuple[str, Any, dict, dict]
#: the layouts whose every input port reads its units' argument groups (`sources`), in units row by row: ``in<k>`` the
#: k-th of the row's operands (a part's, `part`), RoPE's ``x`` and ``cs``, and an unlinked ``s8`` chunk's ``x`` (codes and
#: scale); a GEMM's shared-row tables and a K-chunk's or split's wired ports are left out
LAYOUTS = ("row", "part", "coord", "cols", "slice", "rope", "s8")


def sources(t: str, r: Mapping[str, Any]) -> dict[str, list[int]]:
    """Template ``t``'s input ports on row ``r``, each with the row's argument groups it reads (`LAYOUTS`); a port reading a
    committed output of the row itself (a part's ``("out", j)``) is none."""
    tp = template(t)
    if tp.layout == "s8":
        return {"x": [0, 1]}
    p = part(tp.word, r["spec"])
    srcs = p[1] if p else tuple(("arg", g) for g in range(len(r["args"])))
    names = ("x", "cs") if tp.layout == "rope" else tuple(f"in{k}" for k in range(len(srcs)))
    return {nm: [int(g)] for nm, (src, g) in zip(names, srcs) if src == "arg"}


def ingress(window: CS.Binding, rows: Sequence[list[Row]], units: Callable[[str, Mapping[str, Any]], int],
            arrivals: Mapping[str, int] | None = None) -> list[IF.Ingress]:
    """The window's window ingress over its templates' ``rows`` (`serving_rows.population`'s), ``units`` how many of a template's units
    a row holds (`serving_rows.row_units`): each input port of a unit (`LAYOUTS`) whose every argument slice is a committed
    member of a Program row outside the window (``["n", j, lo, hi]``, row ``j`` of the same request and none of the window's
    rows), the activations entering it from the rows before it, at the unit's engine step (the request's arrival plus the
    row's step), in runs of one template's port at one step over consecutive units.  A port reading registered weights, a
    row of the window or any other slice is not window ingress: its rows stay free unless a wire fills them."""
    layouts = tuple(x for x in LAYOUTS if not (x == "s8" and window.row_tables))
    inside = {(rid, int(r["i"])) for rs in rows for rid, _pi, r, _d in rs}
    out: list[IF.Ingress] = []
    last: dict[tuple[str, str], int] = {}
    for (t, _n), rs in zip(IF.templates_of(window), rows, strict=True):
        if template(t).layout not in layouts:
            continue
        at = 0
        for rid, _pi, r, d in rs:
            u, step = units(t, d), int((arrivals or {}).get(rid, 0)) + int(d["step"])
            for port, groups in sources(t, r).items():
                runs = [sl for g in groups for sl in runs_from_file_items(r["args"][g])]
                if not runs or any(sl[0] != "n" or (rid, int(sl[1])) in inside for sl in runs):
                    continue
                k = last.get((t, port))
                if k is not None and out[k].step == step and out[k].at + out[k].units == at:
                    out[k] = replace(out[k], units=out[k].units + u)
                else:
                    last[(t, port)] = len(out)
                    out.append(IF.Ingress(t, port, at, u, step))
            at += u
    return out
