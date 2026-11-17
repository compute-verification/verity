"""A window's tie to an earlier docket W, and the resolver the window's record is read with, for test_pearltie and
test_remotetie, from the verifier's side alone: no file of the window's audit (`spot/prover/audit/`) and no
`verity_flock`. The prover's `tests/test_records_audit.py` holds both to `records_audit`'s `tie_of` and
`catalog_resolve`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import docket as REG


def catalog_resolve(tid: str) -> Any:
    """`interface.resolve` with every catalog's Definitions loaded, the served workloads' registry among them (a window's
    io shell and its blocks, which `interface`'s own walk can't import)."""
    from verity_vllm.definitions.registry.catalog import load

    load()
    return IF.resolve(tid)


def tie_of(circuit: Mapping[str, Any], w: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The tie of a window whose circuit takes values to ``w``, which registers each of them once under its own name: one-stage's
    ``log`` ([W]), ``registered`` (each value's words), ``outputs`` (W's entry of each) and ``schemas`` (each value's rows),
    and each value's pin, W's entry under its name. Refused (ValueError) for a circuit that takes no values, and for a value
    W doesn't register exactly once."""
    reg = IF.declarations(circuit)[2]
    values = tuple(reg.values) if reg is not None else ()
    if not values:
        raise ValueError("the circuit takes no values, so there is nothing to tie to W")
    pins, outputs, d = {}, {}, REG.digest(w)
    for v in values:
        if len(got := [e for e in w.get("registered") or () if isinstance(e, Mapping) and e.get("value") == v]) != 1:
            raise ValueError(f"W registers the circuit's value {v!r} {len(got)} times, not once")
        pins[v], outputs[v] = {**got[0], "value": v}, {"record": d, "value": v}
    return ({"log": [dict(w)], "registered": {v: int(pins[v]["words"]) for v in values}, "outputs": outputs,
             "schemas": {v: IF.value_format(reg, v)[0] for v in values}}, pins)
