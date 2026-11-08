"""A `run_values` identity (`format.RUN_FAMILY`, a DSA row's top-k order past INDEX_TOPK) is one request's whole committed tensor: at
B=1 the map's request; at B>1 the request whose rows spell it, the k-th tensor of one name the k-th such request in the step's row
order (`Binder.owners`), never the per-token, per-request or shared split the map gives a batched member by its shape.  Only its
label binds it, never a tuple slot of its op."""
from __future__ import annotations

import types

from verity_vllm.commit import binding as B
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TensorMeta
from verity_vllm.query.manifest.bind import Binder
from verity_vllm.query.manifest.format import RUN_FAMILY

PD = "e" * 64
OP = "model.layers.3.self_attn.attn"
STEP = 4
LEAVES = 2


def row(request: str, pos: int) -> dict:
    return {"family": RUN_FAMILY, "request_id": request, "step": STEP, "engine_step": STEP, "invocation": 0, "rank": 0, "op_path": OP,
            "output_member": f"topk_switches[t={pos}]", "element_range": [0, LEAVES], "dtype": "u8"}


def name(pos: int) -> str:
    return f"{OP}/topk_switches[t={pos}]"


def layouts(names: list[str]) -> dict[int, list[TensorMeta]]:
    out, off = [], 0
    for j, n in enumerate(names):
        m = TensorMeta(j, n, "uint8", (LEAVES,), LEAVES, GPU_CHUNK_BYTES, off // GPU_CHUNK_BYTES, 1)
        m.stream_off, off = off, off + LEAVES
        out.append(m)
    return {STEP: out}


def bmap(rows: list[dict], names: list[str], request_rows: list[tuple[str, int, int]] | None) -> dict:
    steps = [types.SimpleNamespace(root=bytes([k]) * 32) for k in range(STEP + 1)]
    return B.build_binding_map(layouts(names), steps, b"\0" * 32, PD, binder=Binder(rows, rank=0),
                               request_rows={STEP: request_rows} if request_rows else None)


def entries(m: dict) -> list[tuple[str, str, list[int], int]]:
    return [(e["identity"]["request_id"], e["identity"]["output_member"], e["identity"]["element_range"], e["element_offset"])
            for e in m["entries"]]


def covered(rows: list[dict], m: dict) -> dict:
    pop = [B.identity(PD, r["request_id"], STEP, 0, OP, r["output_member"], *r["element_range"], 0) for r in rows]
    return B.coverage_check(pop, m)


def test_two_requests_rows_are_each_their_own():
    """Two decode rows, 2 tokens and 2 requests: a 2-leaf tensor has as many rows as the step has tokens, which the shape rule would
    split per token; each is its request's whole."""
    rows = [row("r0", 2100), row("r1", 2050)]
    m = bmap(rows, [name(2100), name(2050)], [("r0", 0, 1), ("r1", 1, 2)])
    assert entries(m) == [("r0", "topk_switches[t=2100]", [0, LEAVES], 0), ("r1", "topk_switches[t=2050]", [0, LEAVES], 0)]
    cov = covered(rows, m)
    assert (cov["expected_n"], cov["bound_n"], cov["ambiguous_n"], cov["missing"]) == (2, 2, 0, [])


def test_two_requests_at_one_position_take_its_tensors_in_row_order():
    rows = [row("r0", 2100), row("r1", 2100)]
    for order in (["r0", "r1"], ["r1", "r0"]):
        m = bmap(rows, [name(2100)] * 2, [(q, i, i + 1) for i, q in enumerate(order)])
        assert [e[0] for e in entries(m)] == order
        cov = covered(rows, m)
        assert (cov["bound_n"], cov["ambiguous_n"], cov["missing"]) == (2, 0, [])
    third = bmap(rows, [name(2100)] * 3, [("r0", 0, 1), ("r1", 1, 2)])
    assert [e[0] for e in entries(third)][:2] == ["r0", "r1"] and Binder(rows).bind(STEP, [name(2100)] * 3)[2][3] == "unbound"


def test_at_b1_the_map_binds_its_request_as_before():
    m = bmap([row("r0", 2100)], [name(2100)], None)
    assert entries(m) == [("r0", "topk_switches[t=2100]", [0, LEAVES], 0)]


def test_a_tuple_slot_of_the_op_never_binds_a_run_value():
    rows = [row("r0", 2100)]
    assert Binder(rows).bind(STEP, [OP]) == [(OP, "", 0, "unbound")]
    assert Binder(rows).bind(STEP, [name(2100), OP]) == [(OP, "topk_switches[t=2100]", 0, "label"), (OP, "", 0, "unbound")]
