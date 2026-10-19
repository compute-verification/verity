"""The global manifest's attention hidden stream per engine step (`query.compose.compose_global`): the launch the step's requests make
together, named as the tap names it.  Under FA3 a launch packs a KV group's query heads into its rows (HB = sequences x KV heads,
M = max query rows x heads per KV head), so a grouped-query model's composed stream is the one `hidden_source` binds."""
from __future__ import annotations

from verity_vllm.acquire.sources.hidden_source import fa3_stream_name, fa3_tile
from verity_vllm.query.compose import compose_global
from verity_vllm.query.manifest import format as fmt

NH, KVH, D, FA3 = 32, 8, 64, "fa3_hidden_m1_stream"


def _part(rid: str, LP: int, T: int, C: int = 0) -> dict:
    """A request manifest's step rows and attention streams as `required.request_manifest` writes them (one sequence per request)."""
    ids = []
    for s in range(T + 1):
        rows, t_max = (LP - C, LP) if s == 0 else (1, LP + s)
        HB, M, NB, BN, name = fmt.fa3_geometry(t_max, rows, NH, KVH, D, 1)
        ids.append({"family": "input_token_ids", "step": s, "rows": rows, "rank": 0, "request_id": rid, "op_path": "model.embed_tokens",
                    "output_member": "in", "invocation": 0, "program_digest": rid, "element_range": [0, rows]})
        ids.append({"family": FA3, "step": s, "rank": 0, "request_id": rid, "op_path": "fa3.m1/L0", "output_member": name, "invocation": 0,
                    "program_digest": rid, "element_range": [0, fmt.mat_total_words(HB, M, NB, BN, D)],
                    "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, "seqlen_k": t_max, "rows": rows},
                    "attention_module": "model.layers.0.self_attn.attn"})
    return {"program_digest": rid, "identities": ids, "steps": list(range(T + 1)), "program": {"LP": LP, "T": T},
            "query": {"attention_hidden_family": FA3}, "op_path_aliases": {}}


def _tap_name(max_q: int, max_k: int, n_seq: int) -> str:
    """What `HiddenM1SourceFA3` binds for a varlen launch of `n_seq` sequences (its wrapper's geometry)."""
    g = NH // KVH
    _kBM, BN = fa3_tile(D, max_q, g, 1)
    return fa3_stream_name(0, n_seq * KVH, max_q * g, -(-max_k // BN), BN, D).split("/", 2)[2]


def test_a_composed_fa3_stream_is_the_launch_the_tap_binds():
    reqs = [{"request_id": "r0", "workload_request_id": "a", "index": 0, "prompt_len": 168, "max_tokens": 10, "arrive_step": 0},
            {"request_id": "r1", "workload_request_id": "b", "index": 1, "prompt_len": 168, "max_tokens": 10, "arrive_step": 4}]
    m = compose_global({"fixture": "t"}, {"r0": _part("r0", 168, 9), "r1": _part("r1", 168, 9, C=128)}, reqs, workload_digest=None)
    got = {r["engine_step"]: r["output_member"] for r in m["identities"] if r["family"] == FA3}
    assert got[0] == _tap_name(168, 168, 1) == "g8x672x2x128x64"
    assert got[1] == _tap_name(1, 169, 1)
    assert got[4] == _tap_name(40, 172, 2) == "g16x160x2x128x64"          # r0's decode beside r1's 40 uncached prefill rows
    assert got[13] == _tap_name(1, 177, 1)
