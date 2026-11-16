"""`verity_vllm.pipeline.tp.match`: the all-reduce switches a Match at the routed world holds its all-reduces to (`_routed_settings`)
and the routed Definition of one all-reduce (`_route`).  Every rank's capture names its switches in its log header: ranks that ran
under different switches, a rank that recorded none and a partial that is not [M, N] each fail by name, and no other world reads a
header.  Needs torch (skipped without)."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from tests.pipeline import test_tp_match_routed_allreduce as RA
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.log import LogWriter
from verity_vllm.pipeline.tp import match as M

WORLD, N = 8, 7168
ON = {"custom_all_reduce": True, "symmetric_memory": True, "flashinfer": True, "batch_invariant": False, "nccl_transport": None}
OFF = dict(ON, custom_all_reduce=False)
DIFFERENT = "the ranks' captures ran under different all-reduce switches: "
NO_FILE, NO_SWITCHES = "no log", "a header with no switches"


def _log(capture: Path, rank: int, disabled: Any) -> None:
    """Rank ``rank``'s capture log, its header recording `disable_custom_all_reduce` = ``disabled``; NO_SWITCHES: a header that
    records none; NO_FILE: no log."""
    if disabled == NO_FILE:
        return
    notes = {} if disabled == NO_SWITCHES else {"tp2": {"facts": {"parallel_config": {"disable_custom_all_reduce": disabled}}}}
    with LogWriter(capture / f"log.rank{rank}.jsonl.gz") as lw:
        lw.write(RunHeader(schema="test", created_utc="", profile_id="", profile={"engine_flags": {}}, model={}, workload={}, notes=notes))


def _settings(tmp_path: Path, world: int = WORLD, **odd: Any) -> tuple[dict | None, dict]:
    """`_routed_settings` over a capture whose ranks ran with the custom all-reduce on, but each ``rank<r>`` of ``odd``: (the
    switches, what the check's provenance record was given)."""
    capture = tmp_path / "capture"
    capture.mkdir()
    for r in range(world):
        _log(capture, r, odd.get(f"rank{r}", False))
    prov: dict[str, Any] = {"all_reduce_definitions": {"left": 1}}
    return M._routed_settings(capture, list(range(world)), world, prov), prov


# ------------------------------------------------------------------------------------------------ the switches of a Match

def test_ranks_that_ran_under_the_same_switches_are_held_to_them(tmp_path):
    settings, prov = _settings(tmp_path)
    assert settings == ON and prov == {"all_reduce_definitions": {}, "allreduce_settings": ON}


def test_ranks_that_all_ran_without_the_custom_all_reduce_are_held_to_that(tmp_path):
    settings, prov = _settings(tmp_path, **{f"rank{r}": True for r in range(WORLD)})
    assert settings == OFF and prov["allreduce_settings"] == OFF


@pytest.mark.parametrize("odd", [0, 5, 7], ids=["the first rank", "a middle rank", "the last rank"])
def test_ranks_that_ran_under_different_switches_fail_by_name_with_every_ranks_switches(tmp_path, odd):
    settings, prov = _settings(tmp_path, **{f"rank{odd}": True})
    assert settings == {"why": DIFFERENT + str({r: OFF if r == odd else ON for r in range(WORLD)})}
    assert prov == {"all_reduce_definitions": {}, "allreduce_settings": settings}


@pytest.mark.parametrize("none", [NO_FILE, NO_SWITCHES])
def test_a_rank_that_recorded_no_switches_fails_by_its_number(tmp_path, none):
    settings, prov = _settings(tmp_path, rank2=none, rank6=none)
    assert settings == {"why": "no all-reduce switches in the capture log header of ranks [2, 6]"} == prov["allreduce_settings"]


def test_a_rank_with_no_switches_is_named_before_ranks_that_differ(tmp_path):
    settings, _ = _settings(tmp_path, rank0=NO_FILE, rank3=True)
    assert settings == {"why": "no all-reduce switches in the capture log header of ranks [0]"}


@pytest.mark.parametrize("world", [2, 4, 16])
def test_another_world_reads_no_header_and_has_no_switches(tmp_path, world):
    prov: dict[str, Any] = {}
    assert M._routed_settings(tmp_path / "no such capture", list(range(world)), world, prov) is None
    assert prov == {"all_reduce_definitions": {}, "allreduce_settings": None}


def test_a_match_one_rank_of_which_ran_under_other_switches_fails_every_all_reduce_by_that_name(tmp_path):
    run = tmp_path / "routed"
    assert RA._make(run, RA.f32_once)["provenance"]["pass"] is True
    _log(run / "capture", 5, True)
    M._check_main(types.SimpleNamespace(out=str(run), build_rank=[str(run / f"b{r}") for r in range(WORLD)], build_rank0=None, build_rank1=None))
    p = json.loads((run / "tp2_match.json").read_text())["provenance"]
    why = DIFFERENT + str({r: OFF if r == 5 else ON for r in range(WORLD)})
    assert p["pass"] is False and p["allreduce_settings"] == {"why": why}
    assert [(m["step"], m["kind"], m["why"]) for m in p["mismatches"]] == [(0, "all_reduce", why), (1, "all_reduce", why)]
    assert p["collectives_checked"] == 2, "the two all-gathers are still held to their relation"


# ------------------------------------------------------------------------------------------------ one all-reduce's routed Definition

def _parts(*shape: int) -> list:
    return [torch.zeros(shape, dtype=torch.bfloat16) for _ in range(WORLD)]


@pytest.mark.parametrize("M_,family,spec", [(37, "AllReduceMultimem_v1", None), (1, "AllReduceMnnvl_v1", "AllReduceMnnvl_v1{WORLD=8,N=7168,SHOT=1}")],
                         ids=["a 37-row prefill", "a one-row decode step"])
def test_an_all_reduce_of_m_by_n_partials_is_routed_by_its_rows_and_width(M_, family, spec):
    route = M._route("all_reduce", _parts(M_, N), WORLD, ON)
    assert route is not None and route["family"] == family and route["spec"].startswith(family + "{WORLD=8,") and route["fused"] is False
    assert spec is None or route["spec"] == spec


def test_the_rows_and_the_width_are_read_in_that_order():
    route = M._route("all_reduce", _parts(N, 37), WORLD, ON)
    assert route == {"why": "tp_allreduce.covered() has no entry for setting='defaults', N=37, route='symm_mem_comm', M=7168, world=8, "
                            "device=None, transport=None"}


@pytest.mark.parametrize("shape", [(N,), (1, 37, N), ()], ids=["a vector", "three dimensions", "a scalar"])
def test_a_partial_that_is_not_m_by_n_has_no_route_and_says_its_shape(shape):
    assert M._route("all_reduce", _parts(*shape), WORLD, ON) == {"why": f"partial of shape {shape}, not [M, N]"}


def test_the_first_ranks_partial_is_the_one_whose_shape_is_read():
    parts = [torch.zeros(N, dtype=torch.bfloat16), *_parts(1, N)[1:]]
    assert M._route("all_reduce", parts, WORLD, ON) == {"why": f"partial of shape ({N},), not [M, N]"}
    route = M._route("all_reduce", [*_parts(37, N)[:1], *_parts(1, N)[1:]], WORLD, ON)
    assert route is not None and route["family"] == "AllReduceMultimem_v1", "37 rows, the first rank's"


@pytest.mark.parametrize("kind", ["all_gather", "fused_allreduce_rms_norm", ""])
def test_a_collective_that_is_no_all_reduce_has_no_route(kind):
    assert M._route(kind, _parts(N), WORLD, ON) is None and M._route(kind, _parts(37, N), WORLD, ON) is None


def test_a_world_with_no_switches_routes_nothing():
    assert M._route("all_reduce", _parts(N), 4, None) is None


def test_switches_that_failed_are_the_route_of_every_all_reduce_whatever_its_shape():
    failed = {"why": "no all-reduce switches in the capture log header of ranks [7]"}
    assert M._route("all_reduce", _parts(37, N), WORLD, failed) is failed and M._route("all_reduce", _parts(N), WORLD, failed) is failed
