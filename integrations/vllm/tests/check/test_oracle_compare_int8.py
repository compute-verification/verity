"""The oracle on the exact-int8 row's members: a `call_boundaries` member (`<linear>/<fx node>/k`) is read at its module's instances
of the producer family, and a member several same-family instances per row produce together (an attention row per head, a KV
quantization per KV head) is all of them when their views are the identity's bytes -- otherwise still 'ambiguous', never guessed."""
from __future__ import annotations

from collections import defaultdict

from verity_vllm.check import oracle_compare as OC

U0 = "aaaaaaaa-0000-0000-0000-000000000000"
LIN, ATTN = "model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.attn"
K, D, H, G = 4, 8, 2, 2


def _view(alloc, offset, shape, dtype):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.insert(0, acc)
        acc *= s
    return {"alloc": alloc, "offset": offset, "shape": list(shape), "strides": strides, "dtype": dtype, "device": "cuda:0"}


def _oracle(instances):
    """A MatchOracle over `instances` ((op key, spec, position, output views)) whose snapshot bytes are a pattern of each view's place."""
    o = OC.MatchOracle.__new__(OC.MatchOracle)
    o.snapshot_steps, o.requests = {0}, {U0: "r0"}
    o._inst, o._families_at, o._outs_by_step = defaultdict(list), {}, {}
    for i, (op, spec, pos, ovs) in enumerate(instances):
        o._inst[(0, op)].append({"request": U0, "row": 0, "position": pos, "hint": f"{op}/x", "event": i, "spec": spec,
                                 "operand_views": [], "output_views": ovs})

    def view_bytes(_event, v):
        r = OC.MatchOracle._view_bytes_range(v)
        return bytes((v["alloc"] * 16 + b) % 251 for b in range(r[1], r[2])), None

    o._view_bytes = view_bytes
    return o


def _span(alloc, lo, hi):
    return bytes((alloc * 16 + b) % 251 for b in range(lo, hi))


def test_a_call_boundary_member_is_read_at_its_modules_producer_instances():
    o = _oracle([(LIN, "Int8RowQuant_v1{K=4}", p, [_view(1, p * K, [K], "int8"), _view(2, p, [], "float32")]) for p in range(3)]
                + [(LIN, "ScaledMmS8_v1{K=4,N=8}", p, [_view(3, p * 8, [8], "bfloat16")]) for p in range(3)])
    prod = OC.producers_of({"identities": [{"family": "call_boundaries", "op_path": f"{LIN}/row_quant_7", "output_member": "1",
                                            "spec": "Int8RowQuant_v1"}]})[(f"{LIN}/row_quant_7", "1")]
    assert prod == {"spec": "Int8RowQuant_v1", "ordinal": None, "operand": None, "interior": True}
    b, why, n = o.member_bytes(0, f"{LIN}/row_quant_7", "1", "r0", producer=prod)
    assert why is None and n == 3 and b == _span(2, 0, 12)
    b, why, n = o.member_bytes(0, f"{LIN}/row_quant_7", "0", "r0", producer=prod)
    assert b == _span(1, 0, 3 * K)
    assert o.member_bytes(0, f"{LIN}/row_quant_7", "1", "r0", producer={"spec": "Int8RowQuant_v1"})[1] == "no fold instance of this op at this step"
    b, _why, _n = o.member_bytes(0, LIN, "0", "r0", producer={"spec": "ScaledMmS8_v1"})
    assert b == _span(3, 0, 3 * 8 * 2)


def test_a_member_of_one_instance_per_head_is_all_of_them_in_row_then_storage_order():
    rows = [(ATTN, "Int8LutAttentionRow_v1{D=8,S=1}", p, [_view(5, (p * H + h) * D, [D], "bfloat16")]) for p in range(2) for h in reversed(range(H))]
    kvq = [(ATTN, "Int8KvQuant_v1{S=2,D=8}", 1, [_view(6, g * 2 * D, [2, D], "int8"), _view(7, g * 2 * D, [2, D], "int8"),
                                                 _view(8, g * D, [D], "float32"), _view(9, g * D, [D], "float32")]) for g in range(G)]
    o = _oracle(rows + kvq)
    out = {"spec": "Int8LutAttentionRow_v1", "operand": "RoPE_v1"}
    b, why, n = o.member_bytes(0, ATTN, "out", "r0", producer=out, nbytes=2 * H * D * 2)
    assert why is None and n == 4 and b == _span(5, 0, 2 * H * D * 2)
    b, why, _n = o.member_bytes(0, ATTN, "out", "r0", producer=out)
    assert b is None and "producer_operand RoPE_v1" in why
    b, why, _n = o.member_bytes(0, ATTN, "out", "r0", producer=out, nbytes=H * D * 2)
    assert b is None and "producer_operand RoPE_v1" in why
    b, why, n = o.member_bytes(0, ATTN, "2", "r0", producer={"spec": "Int8KvQuant_v1"}, nbytes=G * D * 4)
    assert why is None and n == G and b == _span(8, 0, G * D * 4)
    assert o.member_bytes(0, ATTN, "1", "r0", producer={"spec": "Int8KvQuant_v1"}, nbytes=G * 2 * D)[0] == _span(7, 0, G * 2 * D)
    assert "ambiguous producer" in o.member_bytes(0, ATTN, "1", "r0", producer={"spec": "Int8KvQuant_v1"})[1]


def test_the_compare_passes_each_entrys_bytes_to_the_oracle():
    o = _oracle([(ATTN, "Int8KvQuant_v1{S=2,D=8}", 0, [_view(6, g * 2 * D, [2, D], "int8")]) for g in range(G)])
    o.sampled, o.step_requests, o.step_rows, o.tokens = {}, {}, {}, {}
    o.workload_check = lambda w: {"ok": None}
    o.provenance = lambda: {}
    want = _span(6, 0, G * 2 * D)
    ident = {"request_id": "r0", "step": 0, "invocation": 0, "op_path": ATTN, "output_member": "0", "element_range": [0, G * 2 * D]}
    bmap = {"entries": [{"identity": ident, "name": f"{ATTN}/0", "encoding": {"dtype": "int8"}, "element_offset": 0}]}
    r = OC.oracle_compare(o, bmap, lambda s, n, lo, hi: want[lo:hi], producers={(ATTN, "0"): {"spec": "Int8KvQuant_v1"}})
    assert r["compared"] == 1 and r["equal"] == 1 and not r["mismatch"], r
