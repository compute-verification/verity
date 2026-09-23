"""`compare` / `summarize`: every difference between two identity populations lands in exactly one class."""

from __future__ import annotations

from verity_vllm.query.compare import compare, summarize


def row(op, member, hi, *, step=0, spec="X_v1", family="instance_outputs", boundary=None, rid="r0"):
    r = {"program_digest": "d", "request_id": rid, "step": step, "invocation": 0, "op_path": op, "output_member": member, "element_range": [0, hi],
         "rank": 0, "spec": spec, "family": family}
    if boundary:
        r["boundary"] = boundary
    return r


def test_classes():
    v1 = [row("m.a", "out", 4), row("m.b", "0", 4), row("m.c", "out", 4), row("m.d", "out", 8), row("fa2.m1/L0", "g1", 9, family="fa2_hidden_m1_stream", rid="*", step="*")]
    v2 = [row("m.a", "out", 4),                                                                           # common
          row("m.b", "out", 4, boundary={"structural_member": "out", "crossing_rows": 1, "returned_rows": 0}),   # member rename
          row("m.d", "out", 6, boundary={"structural_member": "out", "crossing_rows": 1, "returned_rows": 0}),   # range
          row("m.e", "out", 4, boundary={"structural_member": "out", "crossing_rows": 2, "returned_rows": 0}),   # only v2 (crossing)
          row("fa2.m1/L0", "g1", 9, family="fa2_hidden_m1_stream", rid="*", step="*"),
          row("fa2.m1/L0", "g1", 9, family="fa2_hidden_m1_stream", rid="*", step="*")]                    # same key twice: multiplicity
    c = compare(v1, v2)
    assert (c["v1_identities"], c["v2_identities"], c["common"]) == (5, 5, 2)
    assert (c["v1_rows"], c["v2_rows"]) == (5, 6)
    assert c["multiplicity_diff_n"] == 1 and c["multiplicity"][0]["v1_rows"] == 1 and c["multiplicity"][0]["v2_rows"] == 2
    assert len(c["member_rename"]) == 1 and c["member_rename"][0]["v1"]["output_member"] == "0"
    assert len(c["range"]) == 1 and c["range"][0]["v2"]["element_range"] == [0, 6]
    assert [r["op_path"] for r in c["only_v1"]] == ["m.c"]
    assert [r["op_path"] for r in c["only_v2"]] == ["m.e"]
    s = summarize(c)
    assert s["only_v2"][0]["crossing"] == "crossing" and "F-r19-int-20" in s["only_v2"][0]["reason"]
    assert s["member_rename"][0]["n"] == 1 and s["only_v1"][0]["n"] == 1 and s["range"][0]["example"]["v2_range"] == [0, 6]


def test_layer_index_folds_in_summary():
    v1 = []
    v2 = [row(f"m.layers.{k}.x", "out", 4, boundary={"structural_member": "out", "crossing_rows": 1, "returned_rows": 0}) for k in range(5)]
    s = summarize(compare(v1, v2))
    assert len(s["only_v2"]) == 1 and s["only_v2"][0]["n"] == 5 and s["only_v2"][0]["op_path"] == "m.layers.<k>.x"


def test_ignore_program_digest():
    a = [row("m.a", "out", 4)]
    b = [dict(row("m.a", "out", 4), program_digest="other")]
    assert compare(a, b)["common"] == 0
    assert compare(a, b, ignore_program_digest=True)["common"] == 1
