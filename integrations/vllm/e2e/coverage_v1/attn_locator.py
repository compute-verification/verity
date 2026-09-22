"""Locating a selected attention VU in the pinned program -- the verifier's derivation, never the prover's file (A5/6; extends
A3's ``e2e.verifier_v3.locator.ProgramLocator``, which locates GemmCoordinate and Input units only).

The structural map names every root call: ``t=<t>/layers.<L>/attn`` is the ``Attention_v1{T=t+1,NH,KVH,D}`` call whose children are
the ``NH`` ``AttentionHead_v1{T,D}`` calls in head order (``b1.py:427-441``: KV group ``g`` batches heads ``g*G .. (g+1)*G``).  A gate
interval is located to ``(t, layer, head)`` and either the head call itself (the S55 head unit: the interval IS one head child) or an
interior call chain (``AttnBlock{...}`` -> ``DotBf16{...}`` -> ...) for a per-primitive VU.  Refuses an interval outside any attention
root, straddling two heads, or not aligned to a call when the head unit is asked for.
"""

from __future__ import annotations

from typing import Any

from .builders import LocatedUnit

__all__ = ["locate_attention", "located_unit_of"]


def _descend(call: Any, lo: int, hi: int) -> list[Any]:
    """The chain of calls strictly inside ``call`` that contain ``[lo, hi)`` (outermost first)."""

    chain: list[Any] = []
    cur = call
    while True:
        nxt = next((c for c in cur.children() if c.lo <= lo and hi <= c.hi), None)
        if nxt is None:
            return chain
        chain.append(nxt)
        cur = nxt


def locate_attention(loc: Any, lo: int, hi: int) -> dict[str, Any]:
    """``loc`` is A3's ``ProgramLocator`` (or anything with ``root_of(gate) -> (name, call)``)."""

    name, root = loc.root_of(lo)
    if root is None:
        return {"refused": f"gate {lo} lies in no root call"}
    parts = name.split("/")
    if len(parts) != 3 or parts[2] != "attn":
        return {"refused": f"gate {lo} lies in root {name}, not an attention call"}
    t = int(parts[0][2:])
    L = int(parts[1].split(".")[1])
    heads = [c for c in root.children() if c.name.startswith("AttentionHead")]
    if not heads:
        return {"refused": f"attention root {name} has no AttentionHead children"}
    h = next((k for k, c in enumerate(heads) if c.lo <= lo < c.hi), None)
    if h is None:
        return {"refused": f"gate {lo} lies in attention root {name} but in no head call (glue gate)"}
    head = heads[h]
    if hi > head.hi:
        return {"refused": f"[{lo},{hi}) straddles head {h} of {name} ([{head.lo},{head.hi}))"}
    out: dict[str, Any] = {"root": name, "t": t, "layer": L, "head": h, "NH": len(heads), "T": t + 1, "head_interval": [head.lo, head.hi], "head_call": head.name}
    if (lo, hi) == (head.lo, head.hi):
        out["unit"] = "head"
        return out
    chain = _descend(head, lo, hi)
    out["unit"] = "interior"
    out["chain"] = [c.name for c in chain]
    out["innermost_interval"] = [chain[-1].lo, chain[-1].hi] if chain else [head.lo, head.hi]
    return out


def head_interval(loc: Any, t: int, layer: int, head: int) -> tuple[int, int] | None:
    """Inverse of :func:`locate_attention` for a head unit: the gate interval ``[lo, hi)`` of head ``head`` of the attention root
    ``t=<t>/layers.<layer>/attn`` in A3's ``ProgramLocator`` (``root_names`` / ``root_calls``), or ``None`` when absent.  For a FORCED
    (labelled) selection of one head VU in a harness -- the verifier's own selection is drawn, never chosen."""

    name = f"t={t}/layers.{layer}/attn"
    try:
        root = loc.root_calls[loc.root_names.index(name)]
    except (ValueError, AttributeError):
        return None
    heads = [c for c in root.children() if c.name.startswith("AttentionHead")]
    if not 0 <= head < len(heads):
        return None
    return int(heads[head].lo), int(heads[head].hi)


def located_unit_of(row: dict[str, Any], loc: Any, *, request_id: str, ru_index: int = 0) -> LocatedUnit:
    """A ``coverage_v1.builders.LocatedUnit`` for one selected row (``vu_index``, ``kind`` = ``{fn, form}`` or ``fn|form``,
    ``gate_interval``) of the verifier's ``selected.json``; raises ``ValueError`` with the locator's refusal."""

    kind = row["kind"] if isinstance(row["kind"], str) else f"{row['kind']['fn']}|{row['kind']['form']}"
    lo, hi = (int(x) for x in row["gate_interval"])
    L = locate_attention(loc, lo, hi)
    if "refused" in L:
        raise ValueError(f"vu {row.get('vu_index')}: {L['refused']}")
    return LocatedUnit(str(row.get("vu_index")), kind, request_id, int(L["t"]), layer=int(L["layer"]), head=int(L["head"]), ru_index=ru_index,
                       extra={"unit": L["unit"], "chain": L.get("chain", []), "head_call": L["head_call"], "gate_interval": [lo, hi]})
