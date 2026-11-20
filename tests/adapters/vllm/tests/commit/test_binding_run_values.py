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








def test_at_b1_the_map_binds_its_request_as_before():
    m = bmap([row("r0", 2100)], [name(2100)], None)
    assert entries(m) == [("r0", "topk_switches[t=2100]", [0, LEAVES], 0)]


def test_a_tuple_slot_of_the_op_never_binds_a_run_value():
    rows = [row("r0", 2100)]
    assert Binder(rows).bind(STEP, [OP]) == [(OP, "", 0, "unbound")]
    assert Binder(rows).bind(STEP, [name(2100), OP]) == [(OP, "topk_switches[t=2100]", 0, "label"), (OP, "", 0, "unbound")]
