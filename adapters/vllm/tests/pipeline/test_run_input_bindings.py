"""`pipeline.match_fold.run_input_bindings`: each request's `run` Input batches of the workload root (one per field, `request[<id>].run[<k>]`,
at the gates `WorkloadProgram.address_map` gives) are bound, field by field, to the `run_values` identities the request's manifest
requires, and a request whose rows and batches don't tile each other binds nothing and is refused by name."""

from __future__ import annotations

from verity.definitions.registry.dsa_topk import waksman_switches

from verity_vllm.pipeline.match_fold import run_input_bindings
from verity_vllm.program.predict import dsa as D
from verity_vllm.query.manifest.format import RUN_FAMILY, canonical_identity

TOPK = 4


def _row(request: str, layer: int, pos: int, lo: int, *, rank: int = 0) -> dict:
    op, member = D.run_field(layer, pos).rsplit("/", 1)
    n = waksman_switches(pos + 1, TOPK)
    return {"family": RUN_FAMILY, "program_digest": "d0", "request_id": request, "step": 1, "engine_step": 3, "invocation": 0, "rank": rank,
            "op_path": op, "output_member": member, "element_range": [0, n], "dtype": "u8", "leaves": [lo, lo + n]}


def _request(index: int, wid: str, inputs: list[tuple[str, int, int]]) -> dict:
    """An `address_map()` request entry: its local inputs (name, gates), each its own root parameter `request[<wid>].<name>`."""
    return {"request_id": wid, "index": index, "inputs": [{"local_param": j, "name": n, "global": f"request[{wid}].{n}", "root_node": 1 + j,
                                                           "gates": [a, b]} for j, (n, a, b) in enumerate(inputs)]}


def test_each_run_batch_is_bound_to_its_fields_identity():
    a, b = _row("r0", 0, 5, 0), _row("r0", 1, 5, waksman_switches(6, TOPK))
    na, nb = (x["leaves"][1] - x["leaves"][0] for x in (a, b))
    amap = {"requests": [_request(0, "wl-a", [("prompt", 100, 106), ("run[0]", 106, 106 + na), ("run[1]", 106 + na, 106 + na + nb)]),
                         _request(1, "wl-b", [("prompt", 200, 203)])]}
    other_rank = _row("r0", 0, 5, 0, rank=1)
    out = run_input_bindings(amap, [b, other_rank, a, {"family": "instance_outputs", "request_id": "r0"}])
    assert out["refused"] == []
    assert [(x["request_id"], x["input"], x["field"], x["gates"], x["identity"]) for x in out["bindings"]] == [
        ("r0", "request[wl-a].run[0]", D.run_field(0, 5), [106, 106 + na], canonical_identity(a)),
        ("r0", "request[wl-a].run[1]", D.run_field(1, 5), [106 + na, 106 + na + nb], canonical_identity(b))]
    assert {x["engine_step"] for x in out["bindings"]} == {3}


def test_a_run_of_one_field_is_its_one_batch():
    a = _row("r1", 2, 9, 0)
    n = a["leaves"][1]
    amap = {"requests": [_request(1, "wl-b", [("prompt", 50, 60), ("run", 60, 60 + n)])]}
    assert [(x["input"], x["gates"]) for x in run_input_bindings(amap, [a])["bindings"]] == [("request[wl-b].run", [60, 60 + n])]


def test_rows_that_do_not_tile_the_batches_bind_nothing():
    a, b = _row("r0", 0, 5, 0), _row("r0", 1, 5, waksman_switches(6, TOPK))
    na = a["leaves"][1]
    short = {"requests": [_request(0, "wl-a", [("run[0]", 10, 10 + na), ("run[1]", 10 + na, 11 + na)])]}
    out = run_input_bindings(short, [a, b])
    assert out["bindings"] == [] and len(out["refused"]) == 1 and out["refused"][0].startswith("r0: its run values' leaves")
    out = run_input_bindings({"requests": [_request(0, "wl-a", [("run", 10, 10 + na)])]}, [])
    assert out["bindings"] == [] and out["refused"] == [(f"r0: its run values' leaves [] don't tile its `run` Input batches [(0, {na})] "
                                                         "(field k, one gate a leaf)")]
    out = run_input_bindings({"requests": [_request(0, "wl-a", [("prompt", 0, 4)])]}, [a])
    assert out["bindings"] == [] and out["refused"] == ["r0: 1 run values and no `run` Input in the address map"]
    assert run_input_bindings({"requests": [_request(0, "wl-a", [("prompt", 0, 4)])]}, []) == {"bindings": [], "refused": []}
