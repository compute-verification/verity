"""What the sources read from a record that leaves an optional field out: absent, null and empty read alike, and a stated one is read.
A field that holds another empty value (`OTHER_EMPTY`) in its block's place reads as the absent one too.

Each reader below takes a document whose writer may leave a field out: a required manifest's `identities`
(`verity_vllm.acquire.sources.int8_source`, `verity_vllm.acquire.sources.call_boundary_source`,
`verity_vllm.acquire.sources.call_boundary_plan`, `verity_vllm.acquire.sources.taps`), a narrowed row's `identity`
(`verity_vllm.acquire.sources.narrowed_source`), the manifest's `requests`, `engine_steps` and `fixture`, a
Build's `result.json`, a capture map's `pins`, what a rank recorded of the sources it attached, a step the declared composition or the
recorded routes do not hold (`verity_vllm.acquire.sources.route_source`), a capture's stored tensors
(`verity_vllm.acquire.sources.partial_source`), and a capture map's taps, whose `commit` block names the source that commits them
(`verity_vllm.acquire.sources.attention_source`, `verity_vllm.acquire.sources.dsv4_attention_source`,
`verity_vllm.acquire.sources.mhc_source`).  CPU only."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.check.opened import commit_steps
from verity_vllm.acquire.sources import attention_source as AT
from verity_vllm.acquire.sources import call_boundary_plan as CP
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import dsv4_attention_source as D4
from verity_vllm.acquire.sources import int8_source as I8
from verity_vllm.acquire.sources import mhc_source as MH
from verity_vllm.acquire.sources import narrowed_source as NR
from verity_vllm.acquire.sources import partial_source as P
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.check import oracle_compare as OC
from verity_vllm.commit import capture_map as CM

#: a manifest that names no identity: none given, the field absent, null and empty
NO_IDENTITIES: list[dict[str, Any] | None] = [None, {}, {"identities": None}, {"identities": []}]
#: the other empty values a writer may leave in an optional block's place; each reads as the absent block
OTHER_EMPTY: list[Any] = [0, "", False, 0.0, []]
QUANT = {"family": CP.FAMILY, "spec": f"{I8.QUANT_SPEC}{{w=8}}", "op_path": "model.layers.0.q/quant", "engine_step": 2,
         "output_member": "0", "request_id": "r0"}
KV = {"family": "instance_outputs", "spec": I8.KV_SPEC, "op_path": "model.layers.0.attn", "port": 1}
ADD = {"family": CP.FAMILY, "spec": "Add16", "op_path": "model.norm/add", "engine_step": 0, "output_member": "0", "request_id": "r0"}
STATED = {"identities": [QUANT, KV, ADD]}
NARROWED = {"narrowed_of": "0", "op_path": "model.layers.1.attn.qkv", "output_member": "0@4", "columns": [2, 6], "width": 4, "dtype": "bf16"}
NARROWED_SITES = {"model.layers.1.attn.qkv": {"0@4": {"columns": (2, 6), "bytes": 2}}}
OPROJ = "model.layers.0.self_attn.o_proj"


@pytest.mark.parametrize("manifest", NO_IDENTITIES)
def test_a_manifest_that_names_no_identities_has_no_int8_site(manifest):
    """`verity_vllm.acquire.sources.int8_source.quant_sites` and `kv_modules`."""
    assert I8.quant_sites(manifest) == {} and I8.kv_modules(manifest) == [] and not I8.applies(manifest)


def test_a_manifests_int8_identities_are_its_quantizer_sites_and_kv_modules():
    assert I8.quant_sites(STATED) == {(2, "model.layers.0.q"): "model.layers.0.q/quant"}
    assert I8.kv_modules(STATED) == ["model.layers.0.attn"] and I8.applies(STATED)


@pytest.mark.parametrize("manifest", NO_IDENTITIES)
def test_a_manifest_that_names_no_identities_has_no_call_boundary_site(manifest):
    """`verity_vllm.acquire.sources.call_boundary_source.manifest_sites` and `unattached`."""
    assert CB.manifest_sites(manifest) == [] and CB.unattached(manifest, ()) == []


def test_a_manifests_call_boundaries_are_the_sources_unless_another_source_claims_them():
    assert CB.manifest_sites(STATED) == [ADD]
    assert CB.unattached(STATED, ()) == ["model.layers.0.q/quant/0 of r0 @ engine step 2: claimed by int8, a source the Commit does not attach"]
    assert CB.unattached(STATED, (I8.SHORT_NAME,)) == []


@pytest.mark.parametrize("manifest", NO_IDENTITIES)
def test_a_manifest_that_names_no_identities_attaches_no_source(manifest):
    """`verity_vllm.acquire.sources.taps.attached`."""
    assert TAPS.attached(manifest) == []


def test_a_manifests_identities_name_the_sources_its_commit_attaches():
    assert TAPS.attached(STATED) == [CB.CallBoundarySource.short_name, I8.SHORT_NAME]


@pytest.mark.parametrize("manifest", [*NO_IDENTITIES, *OTHER_EMPTY, *({"identities": v} for v in OTHER_EMPTY)])
def test_a_manifest_that_names_no_identities_has_no_narrowed_row(manifest):
    """`verity_vllm.acquire.sources.narrowed_source.manifest_sites` and `applies`."""
    assert NR.manifest_sites(manifest) == {} and NR.applies(manifest) is False
    assert NR.manifest_sites(STATED) == {} and NR.applies(STATED) is False


@pytest.mark.parametrize("stated", [{}, {"identity": None}, {"identity": {}}, *({"identity": v} for v in OTHER_EMPTY)])
def test_a_narrowed_row_that_states_no_identity_is_read_at_its_own_fields(stated):
    manifest = {"identities": [ADD, {**NARROWED, **stated}]}
    assert NR.manifest_sites(manifest) == NARROWED_SITES and NR.applies(manifest) is True


def test_a_rows_identity_is_read_over_its_own_fields():
    under = {"identities": [{"op_path": "model.norm", "narrowed_of": None, "identity": NARROWED}]}
    assert NR.manifest_sites(under) == NARROWED_SITES and NR.applies(under) is True
    over = {"identities": [{**NARROWED, "identity": {"narrowed_of": None}}]}
    assert NR.manifest_sites(over) == {} and NR.applies(over) is False


@pytest.mark.parametrize("identity", [1, "ab", [1], True])
def test_a_rows_identity_that_is_no_mapping_is_refused_as_python_refuses_it(identity):
    manifest = {"identities": [{**NARROWED, "identity": identity}]}
    with pytest.raises(TypeError, match="is not a mapping"):
        NR.manifest_sites(manifest)
    with pytest.raises(TypeError, match="is not a mapping"):
        NR.applies(manifest)


@pytest.mark.parametrize("identities", [{}, {"identities": None}, {"identities": []}, None, *({"identities": v} for v in OTHER_EMPTY)])
def test_a_rank_whose_manifest_names_no_identities_attaches_nothing_to_a_model_no_source_knows(tmp_path, identities):
    """`verity_vllm.acquire.sources.taps.attach_rank`: the rank's rows of a manifest file that leaves `identities` out, or of no file."""
    path = None
    if identities is not None:
        path = str(tmp_path / "manifest.json")
        Path(path).write_text(json.dumps(identities))
    attached: list[Any] = []
    com = types.SimpleNamespace(attach_source=attached.append)
    assert TAPS.attach_rank(com, torch.nn.Linear(2, 2), path, 0) == {} and attached == []


def _body(rows: int = 4) -> Any:
    """A body's size as `_arena_bound` reads it: one bf16 target of 8 words a row, at one position."""
    return types.SimpleNamespace(parts=[types.SimpleNamespace(rows=rows)],
                                 targets=[types.SimpleNamespace(shared=False, at=(0,), lo=0, hi=8, dtype="bf16")])


@pytest.mark.parametrize("manifest", NO_IDENTITIES)
def test_a_manifest_that_names_no_identities_bounds_the_arena_by_one_token_row_a_step(manifest):
    """`verity_vllm.acquire.sources.call_boundary_plan._arena_bound`: without `input_token_ids` rows a step counts one token row."""
    assert CP._arena_bound(manifest, CP.Plan()) == (0, 0)
    assert CP._arena_bound(manifest, CP.Plan(bodies={(0, "m"): _body()})) == (64, 1)


def test_a_manifests_token_rows_divide_a_steps_bytes():
    tokens = {"identities": [{"family": "input_token_ids", "engine_step": 0, "rows": 4}, ADD]}
    assert CP._arena_bound(tokens, CP.Plan(bodies={(0, "m"): _body()})) == (16, 1)


@pytest.mark.parametrize("manifest", [{}, {"requests": None, "engine_steps": None}, {"requests": [], "engine_steps": []},
                                      *({"requests": v, "engine_steps": v} for v in OTHER_EMPTY)])
def test_a_manifest_that_states_no_requests_and_no_engine_steps_is_one_requests_without_a_composition(manifest):
    """`verity_vllm.acquire.sources.call_boundary_plan.plan`: the one request is `r0`, whose Program no Build dir holds here."""
    p = CP.plan(manifest, [ADD])
    assert p.identities == 1 and p.composition == {} and not p.bodies
    assert p.uncovered == ["model.norm/add/0 of r0 @ engine step 0: no Build dir under [] holds request r0's Program "]


def test_a_manifests_requests_and_engine_steps_are_read_when_stated():
    two = CP.plan({"requests": ["a", "b"]}, [ADD])
    assert two.uncovered == ["model.norm/add/0 of r0 @ engine step 0: a manifest of 2 requests without per-request components and composition"]
    steps = [{"engine_step": 3, "members": [{"request_id": "a", "rows": 5}]}, {"engine_step": 4}]
    one = CP.plan({"requests": ["a"], "engine_steps": steps}, [ADD])
    assert one.composition == {3: {"a": 5}, 4: {}}
    assert one.uncovered == ["model.norm/add/0 of r0 @ engine step 0: no Build dir under [] holds request a's Program "]
    assert CP.plan({"engine_steps": [3, 4]}, [ADD]).composition == {}, "engine steps given as numbers declare no composition"


def test_a_build_whose_result_names_no_program_is_no_programs_build_dir(tmp_path):
    """`verity_vllm.acquire.sources.call_boundary_plan.build_dirs`: a `result.json` whose `program` is absent, null or empty names no
    digest, so its directory is listed under none; an `artifact.json` that states the digest is read first."""
    results: dict[str, dict[str, Any]] = {"a": {}, "b": {"program": None}, "c": {"program": {}}, "d": {"program": {"digest": "d1"}},
                                          "e": {"program": {"digest": "d9"}},
                                          **{f"empty{i}": {"program": v} for i, v in enumerate(OTHER_EMPTY)}}
    for name, doc in results.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(doc))
        (tmp_path / name / "instances.json.gz").write_bytes(b"")
    (tmp_path / "e" / "artifact.json").write_text(json.dumps({"program_digest": "d2"}))
    assert CP.build_dirs(str(tmp_path)) == {"d1": str(tmp_path / "d"), "d2": str(tmp_path / "e")}


@pytest.mark.parametrize("declared", [{0: {"r0": 2}}, {0: {"r0": 2}, 1: None}, {0: {"r0": 2}, 1: {}},
                                      *({0: {"r0": 2}, 1: v} for v in OTHER_EMPTY)])
def test_a_step_the_composition_declares_nothing_for_is_a_collector_error_naming_it(declared):
    """`verity_vllm.acquire.sources.call_boundary_source.CallBoundarySource._order`: the rows a step ran must be the composition's; a
    step it holds no requests for declares none."""
    src = object.__new__(CB.CallBoundarySource)
    src.plan, src.stats = CP.Plan(composition=declared), {"errors": 0}
    src.c = types.SimpleNamespace(step_rows={0: [("r0", 0, 2)], 1: [("r0", 0, 2)]}, errors=[])
    assert src._order(0) == ["r0"] and src.c.errors == []
    assert src._order(1) is None and src.stats["errors"] == 1
    why = "call boundaries: engine step 1 ran (request, rows) [('r0', 2)] in this order; the manifest declares []: nothing of the step is committed"
    assert src.c.errors == [why]


def _made(states: list[Any]) -> list[dict[str, Any]]:
    """Each rank's record with what it recorded of its sources: `...` leaves the field out."""
    return [{"rank": r} if s is ... else {"rank": r, "tap_sources": s} for r, s in enumerate(states)]


def test_ranks_that_recorded_no_source_pass_without_a_tap_and_fail_a_tap_that_is_on():
    """`verity_vllm.acquire.sources.taps.require_on_every_rank`: a rank's record whose `tap_sources` is absent, null or empty attached
    none of the sources."""
    made = _made([..., None, {}, *OTHER_EMPTY])
    versions: dict[str, Any] = {}
    assert TAPS.require_on_every_rank(made, ("a",), versions, TAPS.TapSources(), explicit_classes=True) == ("a",) and versions == {}
    with pytest.raises(SystemExit, match=r"fail-closed: the vocab_range source did not attach on ranks \[0, 1, 2, 3, 4, 5, 6, 7\] of 8: "):
        TAPS.require_on_every_rank(made, ("a",), {}, TAPS.TapSources(vocab_range=True), explicit_classes=True)


def test_a_source_some_rank_attached_must_be_on_every_rank_and_a_tap_on_every_rank_adds_its_family():
    on = {"route_source": {"stats": {"layers": 1}}, "vocab_range_source": {"stats": {}}}
    with pytest.raises(SystemExit, match=r"fail-closed: the route source did not attach on ranks \[1, 2, 3\] of 4"):
        TAPS.require_on_every_rank(_made([on, ..., None, {}]), (), {}, TAPS.TapSources(), explicit_classes=True)
    versions: dict[str, Any] = {}
    taps = TAPS.TapSources(vocab_range=True)
    assert TAPS.require_on_every_rank(_made([on, on]), ("a",), versions, taps, explicit_classes=False) == ("a", "vocab_range")
    assert versions["route_source"] == {0: on["route_source"], 1: on["route_source"]}


def _map(name: str, **fields: Any) -> dict[str, Any]:
    return {"schema": CM.SCHEMA, "id": name, "route_site": {"module": "m"}, "target": {"tensor_parallel_size": 2}, **fields}


@pytest.fixture
def maps(tmp_path, monkeypatch):
    """Capture maps under a directory of their own: fifteen that pin no checkpoint, one of another model, one of the model without a
    route site, one of the model at four ranks and one of the model at two."""
    docs = [_map("a"), _map("b", pins=None), _map("c", pins={}), _map("d", pins={"checkpoint": None}), _map("e", pins={"checkpoint": {}}),
            *(_map(f"c{i}", pins=v) for i, v in enumerate(OTHER_EMPTY)),
            *(_map(f"e{i}", pins={"checkpoint": v}) for i, v in enumerate(OTHER_EMPTY)),
            _map("f", pins={"checkpoint": {"repo": "org/other"}}), _map("g", pins={"checkpoint": {"repo": "org/model"}}, route_site=None),
            _map("h", pins={"checkpoint": {"repo": "org/model"}}, target={"tensor_parallel_size": 4}),
            _map("i", pins={"checkpoint": {"repo": "org/model"}})]
    for doc in docs:
        (tmp_path / f"{doc['id']}.json").write_text(json.dumps(doc))
    monkeypatch.setattr(RS, "MAPS", tmp_path)


@pytest.mark.parametrize("manifest", [None, {}, {"fixture": None}, {"fixture": {}}, {"fixture": {"model": None}, "ranks": 2},
                                      *({"fixture": v, "ranks": 2} for v in OTHER_EMPTY)])
def test_a_manifest_that_states_no_model_has_no_capture_map(maps, manifest):
    """`verity_vllm.acquire.sources.route_source.manifest_map`: a manifest whose `fixture`, or whose fixture's model, is absent, null or
    empty."""
    assert RS.manifest_map(manifest) is None


def test_a_manifests_map_pins_its_model_runs_its_ranks_and_has_a_route_site(maps):
    """`verity_vllm.acquire.sources.route_source.manifest_map`: a map whose `pins`, or whose pinned `checkpoint`, is absent, null or
    empty is no model's."""
    got = RS.manifest_map({"fixture": {"model": "org/model"}, "ranks": 2})
    assert got is not None and got["id"] == "i"
    four = RS.manifest_map({"fixture": {"model": "org/model"}, "ranks": 4})
    assert four is not None and four["id"] == "h"
    assert RS.manifest_map({"fixture": {"model": "org/model"}}) is None, "one rank: no map of the model runs it"
    assert RS.manifest_map({"fixture": {"model": "org/none"}, "ranks": 2}) is None


@pytest.mark.parametrize("recorded", [{"other": {0: {"x"}}}, {"other": {0: {"x"}}, "attn": None}, {"other": {0: {"x"}}, "attn": {}}])
def test_a_step_that_recorded_no_route_for_a_decision_names_the_decision(recorded):
    """`verity_vllm.acquire.sources.route_source.RouteSource.step_routes`: a step whose routes hold nothing for a decision."""
    src = object.__new__(RS.RouteSource)
    src.routes, src.prefill, src.decisions = {3: recorded}, {}, {"other": {0}, "attn": {0, 1}}
    assert src.step_routes(3) == (None, "<no route recorded (attn)>")
    assert src.step_routes(4) == (None, RS.NO_ROUTE)


def test_a_steps_routes_are_every_decisions_one_route():
    src = object.__new__(RS.RouteSource)
    src.routes, src.prefill, src.decisions = {3: {"other": {0: {"x"}}, "attn": {0: {"s"}, 1: {"s"}}}}, {3: True}, {"other": {0}, "attn": {0, 1}}
    assert src.step_routes(3) == (("s", "x"), "prefill")
    src.routes[3]["attn"][1] = {"d"}
    assert src.step_routes(3) == (None, "<layers took different routes (attn)>")


def test_a_name_a_step_committed_nothing_under_has_no_occurrence():
    """`verity_vllm.acquire.sources.partial_source._nth`: occurrence k of a name in a step's layout, None past its last and for a name
    the layout does not hold."""
    a, b = object(), object()
    by: dict[str, Any] = {"m/out": [a, b], "m/none": [], "m/null": None, **{f"m/empty{i}": v for i, v in enumerate(OTHER_EMPTY)}}
    assert P._nth(by, "m/out", 0) is a and P._nth(by, "m/out", 1) is b and P._nth(by, "m/out", 2) is None
    assert P._nth(by, "m/absent", 0) is None and P._nth(by, "m/none", 0) is None and P._nth(by, "m/null", 0) is None
    assert [P._nth(by, f"m/empty{i}", 0) for i in range(len(OTHER_EMPTY))] == [None] * len(OTHER_EMPTY)


class RowParallelLinear:
    reduce_results, tp_size = True, 2

    def parameters(self):
        return iter(())


@pytest.mark.parametrize("stored", [{}, {"part": None, "out": None}, {"part": [], "out": []}, *({"part": v, "out": v} for v in OTHER_EMPTY)])
def test_a_capture_step_that_stored_no_tensors_is_named_inconsistent_with_its_record(tmp_path, stored):
    """`verity_vllm.acquire.sources.partial_source.TPPartialSource.compare_match_oracle`: the step's `.pt` leaves `part` and `out` out
    while the capture's record names one collective at the step."""
    part = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
    (tmp_path / "coll.rank0.jsonl").write_text(
        json.dumps({"rank": 0, "step": 0, "k": 0, "kind": "all_reduce", "module": OPROJ, "shape": list(part.shape)}) + "\n")
    torch.save({"step": 0, **stored}, tmp_path / "coll.rank0.step0.pt")
    com, run = commit_steps({0: [(f"{OPROJ}/part_rank0", P._bytes(part))]})
    src = P.TPPartialSource(com, types.SimpleNamespace(named_modules=lambda: [(OPROJ, RowParallelLinear())]), rank=0, world=2)
    res = src.compare_match_oracle(str(tmp_path), OC.OpenedReader(com, run))
    assert res["verdict"] == "FAIL" and res["oracle_inconsistent"] == [{"step": 0, "jsonl_names": 1, "pt_part": 0, "pt_out": 0}]
    assert res["tensors"] == 0 and "error" not in res


#: a tap's `commit` block that names no binder, or another source's: left out (`...`), null, empty, another empty value, or a block
NOT_THE_SOURCES: list[Any] = [..., None, {}, *OTHER_EMPTY, {"name": "a/b"}, {"binder": None, "name": "a/b"}, {"binder": "another", "name": "a/b"}]


def _tap(block: Any) -> dict[str, Any]:
    tap: dict[str, Any] = {"id": "t", "role": "r", "site": {"function": "f", "module": "m", "output": 0}}
    return tap if block is ... else {**tap, "commit": block}


@pytest.mark.parametrize("source", [AT, D4, MH])
@pytest.mark.parametrize("block", NOT_THE_SOURCES)
def test_a_tap_whose_commit_block_does_not_name_a_sources_binder_is_not_the_sources(source, block):
    """`binds` of the three capture-map sources, and `ports_of` of the two that read ports: a tap is a source's when its `commit` block
    names the source's binder; a tap with no block, an empty one, or another binder's is committed by another source or by none."""
    cmap = {"id": "m", "taps": [_tap(block)]}
    assert not source.binds(cmap)
    assert source.binds({"id": "m", "taps": [_tap(block), _tap({"binder": source.BINDER, "name": "a/b"})]})
    if source is not AT:
        assert source.ports_of(cmap) == {}
