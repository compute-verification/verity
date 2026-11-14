"""The CPU gate of one row: its acquisition plan off a runtime tree (`run`), and whether every call-boundary identity of
its manifest has a source its Commit attaches.

    verity-vllm gate call-boundaries <row dir> [--out call_boundaries.json]

`call-boundaries` reads a row dir after its Build (`manifest.json`, the Build dirs under it) and prints one line: exit 0 when every
`call_boundaries` identity of the manifest has a source the row's Commit attaches, 1 naming the first identity without one.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.acquire.plan import AcquisitionPlan, acquisition_plan
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import taps
from verity_vllm.acquire.sources.call_boundary_plan import Plan
from verity_vllm.config import option
from verity_vllm.correspondence.reader_for_acquire import (
    AcquireCorrespondence,
    RuntimeTree,
)
from verity_vllm.records.schema_ids import CALL_BOUNDARIES_GATE_SCHEMA

CALL_BOUNDARIES_SCHEMA = CALL_BOUNDARIES_GATE_SCHEMA
#: identities without a source the record names, per kind (its counts are complete)
NAMED = 100


def run(manifest: Mapping[str, Any], rt: RuntimeTree) -> AcquisitionPlan:
    prefix, aliases = str(manifest.get("wrapper_prefix") or ""), manifest.get("op_path_aliases")
    aliases = aliases if aliases else {}
    return acquisition_plan(manifest, AcquireCorrespondence(rt, wrapper_prefix=prefix, aliases=aliases))


def call_boundaries(manifest: Mapping[str, Any], *roots: str | None) -> dict[str, Any]:
    """Does every `call_boundaries` identity of `manifest` have a source the row's Commit attaches (`taps.attached`)?  One no source
    claims must be covered by the call-boundary source's host plan over the Build dirs under `roots` (`call_boundary_source.plan_of`,
    what its `attach` refuses the Commit without); a claimed one must name an attached source.  The record: the counts, the attached
    sources and the identities without one, named."""
    return call_boundaries_of(manifest, CB.plan_of(manifest, *roots) if CB.manifest_sites(manifest) else None)


def call_boundaries_of(manifest: Mapping[str, Any], plan: Plan | None) -> dict[str, Any]:
    """`call_boundaries` of `manifest` with its host plan given (None when it names no identity the call-boundary source evaluates)."""
    on = taps.attached(manifest)
    rows = manifest.get("identities")
    ids = sum(1 for r in (rows if rows else []) if r.get("family") == CB.FAMILY)
    uncovered, unattached = (plan.uncovered if plan else []), CB.unattached(manifest, on)
    return {"schema": CALL_BOUNDARIES_SCHEMA, "pass": not uncovered and not unattached, "identities": ids,
            "claimed": ids - (plan.identities if plan else 0), "covered": plan.covered if plan else 0, "uncovered": len(uncovered),
            "unattached": len(unattached), "sources": on, "program_dirs": dict(plan.program_dirs) if plan else {},
            "named": {"uncovered": uncovered[:NAMED], "unattached": unattached[:NAMED]}}


def call_boundaries_line(rec: Mapping[str, Any]) -> str:
    """The record as one line: the counts, the attached sources and the first identity without a source."""
    first = [*rec["named"]["uncovered"], *rec["named"]["unattached"]][:1]
    return (f"identities={rec['identities']} claimed={rec['claimed']} covered={rec['covered']} uncovered={rec['uncovered']} "
            f"unattached={rec['unattached']} sources={','.join(rec['sources']) or 'none'} wall={int(rec.get('seconds') or 0)}s"
            + (f" first: {first[0][:400]}" if first else ""))


@dataclass(frozen=True, kw_only=True)
class CallBoundariesOptions:
    """Every call-boundary identity of a row's manifest has a source its Commit attaches (exit 0), or not (exit 1)."""

    row_dir: str = option("row_dir", help="the row dir after its Build: manifest.json and the Build dirs (build_request*/) under it")
    out: str | None = option("--out", default=None, help="the record (call_boundaries.json)")


COMMANDS = {"call-boundaries": CallBoundariesOptions}


def main(a: CallBoundariesOptions) -> int:
    t0 = time.perf_counter()
    rec = call_boundaries(json.loads(Path(a.row_dir, "manifest.json").read_text()), a.row_dir)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(call_boundaries_line(rec), flush=True)
    return 0 if rec["pass"] else 1


__all__ = ["COMMANDS", "call_boundaries_line", "call_boundaries_of", "main", "run"]
