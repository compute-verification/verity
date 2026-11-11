"""The rule a served window's circuit record holds between its members and its statements (`serving_rows.circuit_record`):
a member of bit rows (`rows.Member.schemas`) commits each port under the schema its template's statement states for that
row, an input row by its name and an output row by its ports' positions among the member's outputs. The circuit a verifier
composes from the statement then hashes each row as the window committed it; where the two differ, staging refuses the
member under that circuit (`flock_inputs.flock_input_files`), so its units can't be proved."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any


def unstated(members: Iterable[Any], statements: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each port ``members`` (`rows.Member`) commit under another schema than ``statements`` (by template, its ``in_rows``
    and ``out_rows``) state for it, as ``<member> <port>: committed <schema>, stated <schema>``; ``None`` where the member
    or the statement has no such row, a member of bit rows with no statement having none of its rows stated. Empty when
    every member holds the rule; a member of word rows has no statement and is not read."""
    bad = []
    for m in members:
        if not m.schemas:
            continue
        st = statements.get(m.template, {"in_rows": (), "out_rows": ()})
        outs = [p for p, _words in m.outs]
        stated = {p: s for p, s, _bits, _leaves in st["in_rows"]}
        stated |= {outs[j] if j < len(outs) else f"output {j}": s for s, ports in st["out_rows"] for j in ports}
        got = dict(m.schemas)
        bad += [f"{m.key} {p}: committed {got.get(p)}, stated {stated.get(p)}"
                for p in sorted({*got, *stated}) if got.get(p) != stated.get(p)]
    return bad


def unstated_in(doc: Mapping[str, Any]) -> list[str]:
    """`unstated` over a circuit record ``doc`` as the verifier reads it: its listed members against its statements."""
    return unstated([SimpleNamespace(key=m["name"] or m["template"], template=m["template"], outs=m["outs"],
                                     schemas=[tuple(s) for s in m["schemas"]]) for m in doc["members"]], doc["statements"])
