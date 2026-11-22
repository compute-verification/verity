"""The docket's registration record as a prover builds it (`record`); the verifier's checks of it are
`one_stage.docket`'s."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from verity.circuits.program import Program
from verity.commitments import domains as DM
from verity.commitments import registered_value as RV
from verity.commitments.docket_digest import KINDS
from verity.protocols.proof.spot.verifier.one_stage import draw as D, partition as P
from verity.protocols.proof.spot.verifier.one_stage.docket import query_digest


def record(program: Program, query: Mapping[str, Any], sampling_design: Mapping[str, Any], scheme: str, roots: Sequence[Mapping[str, Any]],
           window: Mapping[str, Any], leaf_layer: str | None = None, anchors: Sequence[Any] = (),
           previous: str | None = None, partition: Mapping[str, Any] | None = None,
           work_table: Mapping[str, Any] | None = None, registered: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """`partition`: an explicit partition object over `program` (default: the canonical `P.partition(program, query)`).
    `sampling_design` is written expanded over the program and query (`draw.derive`: a `stratified` rule becomes its strata, and a
    `work` rule its strata sized by `work_table`). `registered`: the registered values' entries (`registered_value.Registered.entry`),
    which make the record v2."""
    part = dict(partition) if partition is not None else P.partition(program, query)
    rec = {"format": DM.REGISTRATION if registered else DM.REGISTRATION_V1, "protocol": DM.ONE_STAGE_PROTOCOL, "program": part["program"],
           "query": query_digest(part["query"]), "partition": P.digest(part), "population": P.units(program, query),
           "law": D.derive(program, query, sampling_design, work_table), "scheme": scheme,
           "roots": [{"port": r["port"], "root": r["root"], "leaves": int(r["leaves"]), "domain": r["domain"]} for r in roots],
           "leaf_layer": leaf_layer, "anchors": list(anchors), "window": dict(window), "previous": previous}
    if registered:
        rec["registered"] = [{**{k: (int(e[k]) if k in ("leaves", "words") else e[k]) for k in RV.ENTRY_KEYS},
                              **RV.width_entry(int(e.get(RV.WIDTH_KEY, RV.WORD_BITS)))} for e in registered]
    return rec


__all__ = ["KINDS", "record"]
