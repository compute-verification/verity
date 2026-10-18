"""The post-seed verdict (Cut 1, PROTOCOL.md §8 of `verity/core/protocols/remote_bandwidth`, "The verdict's binding"), as
Python that mirrors its Lean definition (`RemoteAudit.PostSeed.verdict`, which the remote audit's verifier runs) for the
prover's speed, held to it by the difftest in `verity/core/protocols/remote_bandwidth/lean/tests/`.

The writers' Program W, its step ends and the parameters are public, and no proof unit is involved. The windows come from
the passes' steps, which are committed values the record still carries in the clear (PROTOCOL.md §8, open). The verdict
runs `stacking.check` on W's own census at those ends, `census.census(W, ends, digest)`, so the census it judges is W's
by construction. What is left is binding, each an equality between things the verifier holds, refused by name
(`BINDINGS`):

1. ``program``: W's digest (`codec.program_digest`) is `audited`, the one-stage audit's `program`;
2. ``values``: each memory-map unit's ``val{k}`` row is W's value leaf at ``pos[k]`` (`map_value_leaves` of its public row
   under `map_statics`): `map_rows[i]` its rows' commit strings, `values` the one-stage value commitments' leaves;
3. ``mapped``: the session's appends are the map units' payloads, the memory-map unit's verdict, passed in;

and ``property``: P(c, R_in, L) holds at B operand bits. The windows and L are the record's, read by
`audit.property_reading` (binding 4), which the session holds the verdict to.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import stacking
from .census import census
from .codec import program_digest
from .requirement_units import map_value_leaves

BINDINGS = ("program", "values", "mapped", "property")


def verdict(program: Any, ends: Sequence[int], *, audited: str, values: Sequence[bytes], map_statics: Any,
            map_public: Sequence[Mapping], map_rows: Sequence[Mapping[str, bytes]], mapped: bool,
            c: Mapping[str, Any], windows: Sequence[stacking.Window], L: int, R_in: Any = stacking.R_IN,
            B: int = stacking.OPERAND_BITS) -> dict:
    """``{"refused": [...], "program": W's digest, "verdict": property/v1's verdict}`` (`BINDINGS` in order). Ends that
    aren't W's steps, a public row `map_value_leaves` refuses, or windows `stacking.check` refuses raise ValueError."""
    if type(L) is not int or L < 0:
        raise ValueError("a verdict's L is a whole number of census steps, the record's cap read in them")
    if len(map_rows) != len(map_public):
        raise ValueError(f"{len(map_rows)} memory-map units' rows for {len(map_public)} public rows")
    digest = program_digest(program)
    checked = stacking.check(program, census(program, list(ends), digest), c=c, windows=windows, R_in=R_in, L=L, B=B)
    leaves = [(rows, r, at) for pub, rows in zip(map_public, map_rows)
              for r, (kind, at) in map_value_leaves(map_statics, pub).items() if kind == "value"]
    refused = [name for name, bad in (
        ("program", digest != audited),
        ("values", any(at >= len(values) or rows.get(r) != values[at] for rows, r, at in leaves)),
        ("mapped", mapped is not True),
        ("property", not checked.ok)) if bad]
    return {"refused": refused, "program": digest, "verdict": checked.to_json()}


__all__ = ["BINDINGS", "verdict"]
