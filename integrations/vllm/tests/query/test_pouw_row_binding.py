"""A `pouw:ncp-v2` linear's output binds under the Linear's tuple slot, as `Gemm_v1`'s does (`required.NAMED_RESIDUALS`).

The fixture is the PoUW-linear part of a real row Commit's records: the Build's `manifest.json` and the Commit's `binding_map_p0.json` of
`r20260929-153141-1c72` (Qwen2.5-0.5B, `qwen25-05b__bf16__rtx4090__tp1__b1__i32__o4__mixed__greedy__bi-eager`), whose BINDING /
MANIFEST COVERAGE check found exactly these 384 identities missing and 384 extra. The manifest named every `NcpLinearRow_v1` /
`NcpLinearRowBias_v1` output `out`; the collector bound `LinearBase.forward`'s slot `0`.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.commit.binding import coverage_check
from verity_vllm.correspondence.reader_for_query import parse_call_name
from verity_vllm.query.program_view import CallView, Port
from verity_vllm.query.required import member_name

FIXTURE = Path(__file__).parent / "fixtures" / "pouw_row_binding.json.gz"


@pytest.fixture(scope="module")
def stored() -> dict:
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))


def _named(row: dict, *, own_parameter: bool = True) -> str:
    """The member `request_manifest` names the row's one output (`member_name`), for a Call of the row's Definition."""
    call = CallView(0, f"{row['spec']}{{N={row['width']}}}", row["spec"], "call", f"{row['op_path']}/linear/out[0]",
                    (Port("out", 0, row["rows"], row["rows"], row["width"]),), ())
    return member_name(call, call.port("out"), parse_call_name(call.name), own_parameter=own_parameter)[0]


def _population(rows: list[dict], member) -> list[tuple]:
    return [(r["program_digest"], r["request_id"], r["step"], r["invocation"], r["op_path"], member(r), tuple(r["element_range"]), r["rank"])
            for r in rows]


def test_the_stored_manifest_names_bind_none_of_the_384(stored):
    rows = stored["manifest_rows"]
    assert len(rows) == 384 and {r["spec"] for r in rows} == {"NcpLinearRow_v1", "NcpLinearRowBias_v1"} and {r["output_member"] for r in rows} == {"out"}
    cov = coverage_check(_population(rows, lambda r: r["output_member"]), stored["binding_map"])
    assert cov["ok"] is False and cov["bound_n"] == 0 and cov["missing_n"] == 384 and cov["extra_n"] == 384


def test_named_as_request_manifest_names_them_all_384_bind_once(stored):
    rows = stored["manifest_rows"]
    assert {_named(r) for r in rows} == {"0"}
    cov = coverage_check(_population(rows, _named), stored["binding_map"])
    assert cov["ok"] is True and cov["bound_n"] == 384 and cov["missing_n"] == 0 and cov["ambiguous_n"] == 0 and cov["extra_n"] == 0


def test_the_slot_is_the_linears_own_only(stored):
    assert {_named(r, own_parameter=False) for r in stored["manifest_rows"]} == {"out"}
