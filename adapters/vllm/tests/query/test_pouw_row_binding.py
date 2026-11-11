"""A `pouw:ncp-v2` linear's output, committed under the Linear's tuple slot, binds the manifest's `out` (`verity_vllm.query.manifest.bind`).

The fixture is the PoUW-linear part of a real row Commit's records: the Build's `manifest.json` and the Commit's `binding_map_p0.json`
(Qwen2.5-0.5B, `qwen25-05b__bf16__rtx4090__tp1__b1__i32__o4__mixed__greedy__bi-eager`), whose BINDING /
MANIFEST COVERAGE check found exactly these 384 identities missing and 384 extra. The manifest named every `NcpLinearRow_v1` /
`NcpLinearRowBias_v1` output `out`; the collector committed `LinearBase.forward`'s slot `<linear>/0`, and the map took that name as
its identity.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import pytest

from verity_vllm.query.manifest.bind import Binder
from verity_vllm.commit.binding import coverage_check
from verity_vllm.correspondence.reader_for_query import parse_call_name
from verity_vllm.query.program_view import CallView, Port
from verity_vllm.query.required import member_name

FIXTURE = Path(__file__).parent / "fixtures" / "pouw_row_binding.json.gz"


@pytest.fixture(scope="module")
def stored() -> dict:
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


def _named(row: dict) -> str:
    """The member `request_manifest` names the row's one output (`member_name`), for a Call of the row's Definition."""
    call = CallView(0, f"{row['spec']}{{N={row['width']}}}", row["spec"], "call", f"{row['op_path']}/linear[0]",
                    (Port("out", 0, row["rows"], row["rows"], row["width"]),), ())
    return member_name(call, call.port("out"), parse_call_name(call.name))


def _population(rows: list[dict]) -> list[tuple]:
    return [(r["program_digest"], r["request_id"], r["step"], r["invocation"], r["op_path"], r["output_member"], tuple(r["element_range"]), r["rank"])
            for r in rows]


def _rebound(bmap: dict, rows: list[dict]) -> dict:
    """The stored map with each committed tensor (`<linear>/0`, in commit order) bound by the manifest rows' `Binder`."""
    binder, by_step = Binder(rows), defaultdict(list)
    for e in bmap["entries"]:
        by_step[e["identity"]["step"]].append(e)
    entries = []
    for step, es in by_step.items():
        names = [f"{e['identity']['op_path']}/{e['identity']['output_member']}" for e in es]
        for e, (op, member, inv, _how) in zip(es, binder.bind(step, names)):
            entries.append(dict(e, identity=dict(e["identity"], op_path=op, output_member=member, invocation=inv)))
    return dict(bmap, entries=entries)


def test_the_stored_manifest_names_bind_none_of_the_384_by_name(stored):
    rows = stored["manifest_rows"]
    assert len(rows) == 384 and {r["spec"] for r in rows} == {"NcpLinearRow_v1", "NcpLinearRowBias_v1"} and {r["output_member"] for r in rows} == {"out"}
    cov = coverage_check(_population(rows), stored["binding_map"])
    assert cov["ok"] is False and cov["bound_n"] == 0 and cov["missing_n"] == 384 and cov["extra_n"] == 384


def test_the_manifest_names_out_and_the_slot_binds_it_all_384_once(stored):
    rows = stored["manifest_rows"]
    assert {_named(r) for r in rows} == {"out"}
    cov = coverage_check(_population(rows), _rebound(stored["binding_map"], rows))
    assert cov["ok"] is True and cov["bound_n"] == 384 and cov["missing_n"] == 0 and cov["ambiguous_n"] == 0 and cov["extra_n"] == 0
