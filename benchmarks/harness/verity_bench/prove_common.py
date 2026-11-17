"""`verity bench prove`'s records (`prove_records`): the slots a workload's inputs are labelled under, the workloads,
and reading a labelled record and a row's fields. `prove_records`' docstring states what each slot holds."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from verity_bench import steady
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, records

CARD_SCHEMA = "spot-card/v1"
#: the proof server's rate in a run's files: `benchmarks/one_stage/server_rate.py`'s rows, one a staged type and prover count
RATES = steady.RATES
#: the same rows in an art that holds them itself: its one run's, as completed or corrected after the run
ART_RATES = "rate.json"
#: the checks of the proof server's steady-state rules (`steady.CHECKS`) a row is held to here, each the rules' own
#: function of the row, in the rules' order: every check of theirs that reads the row alone, but `one_numa_node`
CHECKS = (steady.accepted, steady.exits, steady.not_void, steady.flat, steady.no_throttling, steady.every_prover_warmed)
#: what a refusal of a run by its own verdict opens with: the measurements name each such refusal among their inputs
OWN_VERDICT = "its own steady-state verdict"
SYSTEMS = ("flock", "spot")
CENSUS, CUT, STEADY, SINGLE, SAMPLING = "census", "cut", "flock-steady", "flock-single", "spot-sampling"
LEAN, WARMED = "lean-verify", "flock-warm"


@dataclass(frozen=True)
class Workload:
    """A workload the command prices: the model its records name, the card its prover and its serving run on
    (`overhead.CARDS`), and the fleet whose serving MFU prices a served token (`FLEETS`)."""

    name: str
    model: str
    card: str
    fleet: str

    def record(self, slot: str) -> str:
        return record_name(self.name, slot)


def record_name(workload: str, slot: str) -> str:
    """The `run_of_record` value that names a workload's record of a slot."""

    return f"prove-{workload}-{slot}"


WORKLOADS = {w.name: w for w in (Workload("glm53", "GLM-5.3", "h200", "world-h200"),)}


# ---- records ---------------------------------------------------------------------------------------------------------------


def _superseded(store: Any, target: str) -> list[str]:
    return [str(lab.value) for lab in store.labels(target, key=SUPERSEDED_BY)]


def _label(name: str) -> str:
    return f"`research data label TARGET {RUN_OF_RECORD} {name} --by LANE`"


def of_record(store: Any, name: str, what: str, given: str | None = None) -> str:
    """The one target of record `name`: labelled and not superseded. `given` must be it."""

    labelled = records(store).get(name, [])
    if given is not None and given not in labelled:
        raise ValueError(f"{given} isn't a {what} of record: it doesn't carry `{RUN_OF_RECORD}={name}` (label it: {_label(name)})")
    found = labelled if given is None else [given]
    if not found:
        raise ValueError(f"no {what} of record: nothing in the store carries `{RUN_OF_RECORD}={name}` (label one: {_label(name)})")
    live = [t for t in found if not _superseded(store, t)]
    if not live:
        raise ValueError(f"the {what} of record is superseded: "
                         + ", ".join(f"{t} by {', '.join(_superseded(store, t))}" for t in found)
                         + f" (label what replaces it: {_label(name)})")
    if len(live) > 1:
        raise ValueError(f"{len(live)} targets carry `{RUN_OF_RECORD}={name}` and none is superseded: {', '.join(live)}; "
                         f"label each one a newer {what} replaces `{SUPERSEDED_BY}`")
    return live[0]


def _read(store: Any, art: str, path: str | None = None) -> dict:
    """An art's JSON: a file art's payload, or `path` of a tree's."""

    if store.get_manifest(art).payload.type == "file":
        return json.loads(Path(store.fetch(art)).read_text())
    if path is None or not store.listing(art, [path]):
        raise ValueError(f"{art} holds no `{path}`")
    return json.loads((Path(store.fetch(art, paths=[path])) / path).read_text())


def card(store: Any, art: str, slot: str, workload: Workload) -> dict:
    """A `spot-card/v1` card's value, checked to be `slot`'s card of the workload's model."""

    doc = _read(store, art)
    got = {k: doc.get(k) for k in ("schema", "slot", "model")}
    if got != {"schema": CARD_SCHEMA, "slot": slot, "model": workload.model}:
        raise ValueError(f"{art} isn't a `{slot}` card of {workload.model} ({CARD_SCHEMA}): it is {got}")
    return doc["value"]


def _field(row: Mapping[str, Any], name: str) -> Any:
    """A row's field by its dotted name, or `_ABSENT`."""

    at: Any = row
    for key in name.split("."):
        if not isinstance(at, Mapping) or key not in at:
            return _ABSENT
        at = at[key]
    return at


_ABSENT = object()


def _count(x: Any) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _input(slot: str, workload: Workload, targets: list[str], what: str) -> dict:
    return {"slot": slot, "record": workload.record(slot), "targets": targets, "what": what}
