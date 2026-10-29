"""The reference evaluator's COMPLETE small assignment of a lifted Program, one row per gate (every wire, not only final outputs).

`rows(prog, transcript)` reads `serve3_reference.materialize`'s trace `{gate: word}` back against the circuit: for EVERY gate its
index, scope path (the node names from the root body down), primitive id, lift class (`input` / `const` / `strict` / `specified` /
`unlifted` for a primitive that is not a lifted one), the lifted width `w+1` of its output, the raw committed word, and the word's
reading (`active` with its payload / `bot` / `inv` = noncanonical `1‖nonzero`).  Nothing is normalised: the
`word` column is the raw integer; `reading` is the relation's verdict on it, beside it.

`dump(prog, transcript, path)` writes one JSON object per line; `summary(rows)` counts the words per class and reading (five
counts: gates, active, bot, inv, and the specified-class gates)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from verity.ml.definitions.registry import lifted as L
from verity.ml.circuits.types import Value


def _lift_class(prim: Any) -> str:
    fam = getattr(prim, "family", None)
    if fam == "Input":
        return "input"
    if not L.is_lifted_prim(prim):  # G1 (LIFTING-SPEC v0.7 A11): an ordinary gate -- an unwidened constant (the EOS set, `enable/one`), a twin's gate
        return "const" if fam == "Const" else "ordinary"
    lc = getattr(prim, "lift_class", None)
    if lc:
        return str(lc)
    if fam == "Const":
        return "const"
    return "unlifted"


def _reading(w1: int, word: int) -> dict[str, Any]:
    """`w1` = the gate's output width (a lifted word is `w+1` bits; `w = w1 - 1`)."""
    w = w1 - 1
    if w < 1:
        return {"kind": "unlifted-width", "w": w1}
    if L.is_active(w, word):
        return {"kind": "active", "payload": L.payload(w, word)}
    if L.is_bot(w, word):
        return {"kind": "bot"}
    return {"kind": "inv", "raw_hex": hex(word)}  # noncanonical: reported raw, never rewritten


def rows(prog: Any, transcript: Mapping[int, int]) -> Iterator[dict[str, Any]]:
    circ = prog.circuit
    outs = set(circ.root_outputs())
    for g in range(prog.gates):
        gate = circ.gate(g)
        ret = gate.prim.ret
        w1 = ret.w if isinstance(ret, Value) else None
        word = transcript.get(g)
        lifted = L.is_lifted_prim(gate.prim)  # an ordinary word (G1) has no tag: read as the value it is, never through the lifted reading
        yield {
            "gate": g,
            "path": "/".join(gate.path),
            "prim": gate.prim.id,
            "class": _lift_class(gate.prim),
            "width": w1,
            "operands": list(gate.operands),
            "word": word,
            "reading": ((_reading(w1, word) if lifted else {"kind": "ordinary", "value": word}) if (w1 is not None and word is not None) else None),
            "is_output": g in outs,
        }


def dump(prog: Any, transcript: Mapping[int, int], path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rs = list(rows(prog, transcript))
    with p.open("w") as f:
        for r in rs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return summary(rs)


def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = Counter(r["class"] for r in rs)
    by_reading = Counter((r["reading"] or {}).get("kind", "none") for r in rs)
    return {"gates": len(rs), "by_class": dict(by_class), "by_reading": dict(by_reading),
            "outputs": sum(1 for r in rs if r["is_output"]), "complete": all(r["word"] is not None for r in rs)}
