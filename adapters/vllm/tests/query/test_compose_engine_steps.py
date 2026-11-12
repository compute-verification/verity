"""The global manifest's rows per engine step (`query.compose.compose_global`): each request's step rows at the engine step the
declaration puts it at, the weights once, and no attention stream, since a request manifest names none (`query.required`)."""
from __future__ import annotations

from verity_vllm.query.compose import compose_global
from verity_vllm.query.manifest import format as fmt


def _part(rid: str, LP: int, T: int, C: int = 0) -> dict:
    """A request manifest's weights and step rows as `required.request_manifest` writes them (one sequence per request)."""
    ids = [{"family": "weights", "step": "*", "rank": 0, "request_id": rid, "op_path": "model.embed_tokens", "output_member": "weight",
            "invocation": 0, "program_digest": rid, "element_range": [0, 64]}]
    for s in range(T + 1):
        rows = LP - C if s == 0 else 1
        ids.append({"family": "input_token_ids", "step": s, "rows": rows, "rank": 0, "request_id": rid, "op_path": "model.embed_tokens",
                    "output_member": "in", "invocation": 0, "program_digest": rid, "element_range": [0, rows]})
        ids.append({"family": "instance_outputs", "step": s, "rank": 0, "request_id": rid, "op_path": "model.layers.0.self_attn.attn",
                    "output_member": "out", "invocation": 0, "program_digest": rid, "element_range": [0, rows * 64]})
    return {"program_digest": rid, "identities": ids, "steps": list(range(T + 1)), "program": {"LP": LP, "T": T},
            "query": {"query_id": "Q", "required_families": ["input_token_ids", "instance_outputs", "sampled_token_ids", "weights"]},
            "op_path_aliases": {}}


def test_each_requests_step_rows_land_at_their_engine_step_and_no_attention_stream_is_composed():
    reqs = [{"request_id": "r0", "workload_request_id": "a", "index": 0, "prompt_len": 168, "max_tokens": 10, "arrive_step": 0},
            {"request_id": "r1", "workload_request_id": "b", "index": 1, "prompt_len": 168, "max_tokens": 10, "arrive_step": 4}]
    parts = {"r0": _part("r0", 168, 9), "r1": _part("r1", 168, 9, C=128)}
    m = compose_global({"fixture": "t"}, parts, reqs, workload_digest=None)
    assert m["populations"]["by_family"] == {"weights": 1, "input_token_ids": 20, "instance_outputs": 20}
    attn = {(r["request_id"], r["engine_step"]): r["element_range"] for r in m["identities"] if r["family"] == "instance_outputs"}
    assert attn[("r0", 0)] == [0, 168 * 64] and attn[("r0", 4)] == [0, 64] and attn[("r1", 4)] == [0, 40 * 64] and attn[("r1", 13)] == [0, 64]
    assert [e["rows"] for e in m["engine_steps"]][:5] == [168, 1, 1, 1, 41]
    assert m["manifest_digest"] == fmt.manifest_digest(m["identities"])
