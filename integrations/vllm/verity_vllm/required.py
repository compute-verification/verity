"""``required.json``: the frozen required-interface set (contract sections 9 and 10).

Coverage denominators are fixed before results are inspected.  :func:`derive_required` turns a
description plus a partition into the interface list; :func:`freeze_required` writes it and
refuses to shrink an existing file (a changed partition is a new candidate/row, never a smaller
denominator).  :func:`coverage` counts how many required interfaces a capture binds.

Document::

    {
      "schema": "verity-vllm/required/v1",
      "profile_id": "0123456789ab",
      "partition_id": "row-default",
      "description_digest": "<hex>",
      "interfaces": [
        {"op": "layers.0.mlp.act_mul", "port": 0, "phase": "prefill", "op_type": "silu_mul",
         "decomposition": "row", "relation": "silu_mul.bf16.v1", "origin": "captured"}
      ]
    }

Partition::

    {"partition_id": "row-default",
     "decomposition": {"default": "row", "attention": "whole_operation"},   # by op_type or op
     "phases": ["prefill", "decode"],                                      # optional filter
     "op_types": null}                                                     # optional filter
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import hashing
from .capture import CaptureReader, _read_json, _write_json
from .description import Description

REQUIRED_SCHEMA = "verity-vllm/required/v1"


class RequiredError(ValueError):
    pass


class RequiredShrinkError(RequiredError):
    """The new required set drops an interface the frozen one has."""


@dataclass(frozen=True, slots=True)
class Interface:
    op: str
    port: int
    phase: str
    op_type: str
    decomposition: str
    relation: str
    origin: str

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.op, self.port, self.phase, self.decomposition)

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "port": self.port,
            "phase": self.phase,
            "op_type": self.op_type,
            "decomposition": self.decomposition,
            "relation": self.relation,
            "origin": self.origin,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Interface:
        try:
            return cls(
                op=str(doc["op"]),
                port=int(doc["port"]),
                phase=str(doc["phase"]),
                op_type=str(doc["op_type"]),
                decomposition=str(doc["decomposition"]),
                relation=str(doc["relation"]),
                origin=str(doc.get("origin", "captured")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RequiredError(f"malformed interface {doc!r}: {error}") from error


def decomposition_for(partition: Mapping[str, Any], op: str, op_type: str) -> str:
    table = partition.get("decomposition", {})
    if isinstance(table, str):
        return table
    return str(table.get(op, table.get(op_type, table.get("default", "row"))))


def derive_required(description: Description, partition: Mapping[str, Any]) -> dict[str, Any]:
    """Every (node, port, phase) of the description under the partition's decomposition."""
    if "partition_id" not in partition:
        raise RequiredError("partition needs a partition_id")
    phases = partition.get("phases")
    op_types = partition.get("op_types")
    interfaces: list[Interface] = []
    for node in description.nodes:
        if op_types is not None and node.op_type not in op_types:
            continue
        decomposition = decomposition_for(partition, node.op, node.op_type)
        if decomposition not in {d.id for d in node.decompositions}:
            raise RequiredError(
                f"node {node.op}: partition asks for decomposition {decomposition!r}, "
                f"candidates are {[d.id for d in node.decompositions]}"
            )
        for phase in node.phases:
            if phases is not None and phase not in phases:
                continue
            for port in range(node.ports):
                interfaces.append(
                    Interface(
                        node.op,
                        port,
                        phase,
                        node.op_type,
                        decomposition,
                        node.relation,
                        node.origin,
                    )
                )
    interfaces.sort(key=lambda i: i.key)
    return {
        "schema": REQUIRED_SCHEMA,
        "profile_id": description.profile_id,
        "partition_id": str(partition["partition_id"]),
        "description_digest": description.digest().hex(),
        "interfaces": [i.to_json() for i in interfaces],
    }


def parse_required(doc: Mapping[str, Any]) -> list[Interface]:
    if doc.get("schema") != REQUIRED_SCHEMA:
        raise RequiredError(f"required.schema must be {REQUIRED_SCHEMA!r}")
    interfaces = [Interface.from_json(i) for i in doc.get("interfaces", [])]
    keys = [i.key for i in interfaces]
    if len(set(keys)) != len(keys):
        raise RequiredError("duplicate interface in required.json")
    return interfaces


def check_not_shrunk(existing: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_keys = {i.key for i in parse_required(existing)}
    new_keys = {i.key for i in parse_required(new)}
    dropped = sorted(old_keys - new_keys)
    if dropped:
        raise RequiredShrinkError(
            f"required set would drop {len(dropped)} frozen interface(s), e.g. {dropped[0]}; "
            "a changed partition is a new candidate/row, the earlier denominator stays"
        )


def freeze_required(
    path: str | os.PathLike[str], description: Description, partition: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive and write ``required.json``; refuse if an existing file would shrink."""
    path = Path(path)
    new = derive_required(description, partition)
    if path.exists():
        existing = _read_json(path)
        check_not_shrunk(existing, new)
    _write_json(path, new, canonical=True)
    return new


def required_digest(doc: Mapping[str, Any]) -> bytes:
    return hashing.json_digest(dict(doc))


@dataclass(slots=True)
class Coverage:
    bound: int
    required: int
    missing: list[dict[str, Any]]
    unexpected_ids: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"bound": self.bound, "required": self.required}


def coverage(
    required_doc: Mapping[str, Any],
    description: Description,
    reader: CaptureReader,
    request_ids: Iterable[str] | None = None,
) -> Coverage:
    """How many required interfaces the capture binds.

    An interface (op, port, phase) is *bound* when every position at which the description says
    the node has an output in that phase, for every request considered, is present in the
    capture with the described dtype and shape.  Ids present in the capture that the
    description does not predict are reported as ``unexpected_ids`` (a namespace violation).
    """
    interfaces = parse_required(required_doc)
    profile_id = reader.profile_id
    requests = list(request_ids) if request_ids is not None else reader.request_ids()
    missing: dict[tuple, list[str]] = {i.key: [] for i in interfaces}
    by_target: dict[tuple[str, int, str], list[Interface]] = {}
    for interface in interfaces:
        by_target.setdefault((interface.op, interface.port, interface.phase), []).append(interface)
    unexpected: list[str] = []
    for rid in requests:
        boundary = reader.boundary(rid)
        # per step: the entries present (id -> Entry); expected ids are generated node by node so
        # a slotted node (layers x positions x slots) never materialises the whole request at once
        present_by_step: dict[int, dict[str, Any]] = {}
        for _, step in reader.ru_keys(rid):
            present_by_step[step] = {
                e.canonical_id: e for e in reader.ru(rid, step).index.entries
            }
        seen_expected: set[str] = set()
        for node in description.nodes:
            specs = {port: description.leaf_spec(node, port) for port in range(node.ports)}
            for inst in description.instances(boundary, ops=[node.op]):
                cid = inst.output_id(profile_id)
                seen_expected.add(cid)
                targets = by_target.get((node.op, inst.port, inst.phase))
                if not targets:
                    continue
                step = boundary.step_of(inst.t)
                entry = present_by_step.get(step.step, {}).get(cid) if step else None
                if entry is None:
                    for interface in targets:
                        missing[interface.key].append(cid)
                    continue
                dtype, shape = specs[inst.port]
                if entry.dtype != dtype or entry.shape != shape:
                    for interface in targets:
                        missing[interface.key].append(f"{cid} (dtype/shape disagree)")
        for entries in present_by_step.values():
            for cid in entries:
                if cid not in seen_expected:
                    unexpected.append(cid)
        del seen_expected
    bound = sum(1 for key, lost in missing.items() if not lost)
    return Coverage(
        bound=bound,
        required=len(interfaces),
        missing=[
            {"interface": list(key), "examples": lost[:3], "count": len(lost)}
            for key, lost in missing.items()
            if lost
        ],
        unexpected_ids=sorted(unexpected),
    )
