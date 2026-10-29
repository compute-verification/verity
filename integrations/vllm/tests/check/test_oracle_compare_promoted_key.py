"""The C2 oracle resolves a PROMOTED member through the Value's producer, not the module prefix of the fold's hint.

Row #57 leg 3: over a Match that snapshotted every step, the 432 promoted `model`/`out` identities (the
embedding scale's output crossing into layer 0) were "no fold instance of this op at this step" -- the fold hints that instance by
the producing Call's own site, the manifest names it by the Program row's module.  Same Value, two names.  The oracle now answers
from the dataflow: the fold instance of the producer family whose OUTPUT storage range contains an OPERAND view of a consumer
module's instance (the manifest row's `consumers`, Program order).  Pinned here:

* a promoted row whose fold instance sits under another hint prefix resolves and compares (result True, complete);
* a non-promoted row's key is exactly what it was (`_inst[(step, op_path)]`; `producers_of` adds no key to its entry);
* a promoted identity with no producer instance feeding its consumers is NOT COMPARED, named -- never silently complete.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from verity_vllm.check import oracle_compare as OC
from tests.check.test_oracle_compare import H, ROWS, U0, U1, _bf16, make_bmap, make_match, reader

PRODUCER_OP = "top"                                   # the Program row's module (the manifest identity's op_path)
PRODUCER_HINT = "top.scale/top.scale.k"              # the fold's hint: the producing Call's own site -- prefix `top.scale`, never `top`
PRODUCER_SPEC = "ScaleRow_v1{w=4}"
CONSUMER_OP = "top.layers.0.input_layernorm"         # reads the promoted value (its first operand); the collector observes `<consumer>/in`
P_ALLOC, C_ALLOC, P_EVENT, C_EVENT = 15, 16, 80, 81


def _add_promoted(d: str, *, producer_spec: str = PRODUCER_SPEC, with_producer: bool = True, feed: bool = True) -> bytes:
    """Append to the synthetic Match: per row a producer instance (hint `top.scale/...`, output alloc 15) and a consumer instance
    (hint `top.layers.0.input_layernorm/...`) whose operand view 0 is that output; snapshot alloc 15 at the producer's event.
    `with_producer=False`: no producer instance at all; `feed=False`: the consumer reads another alloc."""
    order = [U1, U0]
    n = sum(ROWS.values())
    data = _bf16(n, 9)
    sha = hashlib.sha256(data).hexdigest()
    fn = f"values/0/{sha[:16]}.bin"
    open(os.path.join(d, "capture", fn), "wb").write(data)
    snap = json.load(open(os.path.join(d, "capture", "snapshots.json")))
    snap["entries"].append({"event": P_EVENT, "path": "outs", "view": {"$v": [P_ALLOC, 0, [n, H], [H, 1], "bfloat16", "cuda:0"]}, "dtype": "bfloat16",
                            "shape": [n, H], "strides": [H, 1], "byte_offset": 0, "nbytes": len(data), "layout": "spanned-storage-bytes",
                            "sha256": sha, "file": fn, "step": 0})
    json.dump(snap, open(os.path.join(d, "capture", "snapshots.json"), "w"))
    with open(os.path.join(d, "instances.jsonl"), "a") as f:
        row = 0
        for u in order:
            for pos in range(ROWS[u]):
                pv = {"alloc": P_ALLOC, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}
                if with_producer:
                    f.write(json.dumps({"node": 0, "step": 0, "event": P_EVENT, "pattern": "x", "spec": producer_spec, "row": row, "rows": n,
                                        "name": f"t={pos}/scale", "hint": PRODUCER_HINT, "request": u, "position": pos,
                                        "operand_views": [], "output_views": [pv]}) + "\n")
                cv = dict(pv) if feed else {**pv, "alloc": 99}
                f.write(json.dumps({"node": 0, "step": 0, "event": C_EVENT, "pattern": "x", "spec": "RMSNorm_v1{}", "row": row, "rows": n,
                                    "name": f"t={pos}/input_layernorm", "hint": f"{CONSUMER_OP}/{CONSUMER_OP}.weight", "request": u, "position": pos,
                                    "operand_views": [cv],
                                    "output_views": [{"alloc": C_ALLOC, "offset": row * H, "shape": [H], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}]}) + "\n")
                row += 1
    return data


def _promoted_bmap(vals: dict, data: bytes) -> tuple[dict, dict]:
    """The fixture's binding map + the promoted identity (op_path `top`, member `out`) per request, observed at the consumer's input."""
    bmap, committed = make_bmap(vals)
    off = 0
    for u, lab in ((U1, "r1"), (U0, "r0")):
        rows = ROWS[u]
        bmap["entries"].append({"identity": {"program_digest": "p" * 64, "request_id": lab, "step": 0, "invocation": 0, "op_path": PRODUCER_OP,
                                             "output_member": "out", "element_range": [0, rows * H], "rank": 0}, "kind": "leaf",
                                "name": f"{CONSUMER_OP}/in", "element_offset": off * H, "encoding": {"dtype": "bfloat16", "shape": [sum(ROWS.values()), H]}})
        off += rows
    committed[(0, f"{CONSUMER_OP}/in")] = data
    return bmap, committed


def _manifest(*, promoted: bool = True) -> dict:
    row = {"family": "instance_outputs", "op_path": PRODUCER_OP, "output_member": "out", "spec": PRODUCER_SPEC.split("{", 1)[0], "rows": 5, "width": H, "dtype": "bf16"}
    if promoted:
        row.update({"promoted": True, "boundary_class": "interior_crossing", "consumers": [CONSUMER_OP]})
    return {"identities": [row, {"family": "instance_outputs", "op_path": "model.layers.0.self_attn.qkv_proj", "output_member": "0", "spec": "X_v1{}"}]}


def test_promoted_member_resolves_through_the_producer_and_compares(tmp_path):
    d, vals = make_match(str(tmp_path))
    data = _add_promoted(d)
    bmap, committed = _promoted_bmap(vals, data)
    prod = OC.producers_of(_manifest())
    assert prod[(PRODUCER_OP, "out")] == {"spec": "ScaleRow_v1", "ordinal": None, "operand": None, "promoted": True, "consumers": [CONSUMER_OP]}
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["result"] is True and res["complete"] is True and res["mismatch_n"] == 0, res
    assert res["compared"] == 2 * len(vals) + 2 + 2                              # + the two requests' promoted identities
    assert not any(k.startswith("no fold instance") for k in res["not_compared"]), res["not_compared"]
    # the same key WITHOUT the dataflow rule is exactly the leg-3 symptom: the hint prefix `top.scale` is not the identity's `top`
    o = OC.MatchOracle(d)
    b, why, n = o.member_bytes(0, PRODUCER_OP, "out", "r0", producer={"spec": "ScaleRow_v1", "ordinal": None, "operand": None})
    assert b is None and why == "no fold instance of this op at this step" and n == 0
    # and through the producer: r0's 5 rows in position order = the oracle bytes of r0's block (r1's 3 rows come first in engine order)
    b, why, n = o.member_bytes(0, PRODUCER_OP, "out", "r0", producer=prod[(PRODUCER_OP, "out")])
    assert why is None and n == ROWS[U0] and b == data[ROWS[U1] * H * 2:]


def test_promoted_value_mismatch_is_named_not_relabelled(tmp_path):
    """The dataflow resolution compares the VALUE: a wrong committed byte in r0's block is a mismatch on the promoted identity."""
    d, vals = make_match(str(tmp_path))
    data = bytearray(_add_promoted(d))
    lo = ROWS[U1] * H * 2
    data[lo] ^= 1
    bmap, committed = _promoted_bmap(vals, bytes(data))
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(_manifest()))
    assert res["result"] is False and res["mismatch_n"] == 1
    m = res["mismatch"][0]
    assert m["identity"]["op_path"] == PRODUCER_OP and m["identity"]["output_member"] == "out" and m["identity"]["request_id"] == "r0"


def test_non_promoted_rows_keep_their_key(tmp_path):
    """A vocabulary row: `producers_of` entry unchanged (no `promoted` / `consumers` key), the oracle reads `_inst[(step, op_path)]` as
    before -- the fixture's positive compare is identical with and without a manifest naming only vocabulary rows."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    prod = OC.producers_of(_manifest(promoted=False))
    assert prod[(PRODUCER_OP, "out")] == {"spec": "ScaleRow_v1", "ordinal": None, "operand": None}
    assert prod[("model.layers.0.self_attn.qkv_proj", "0")] == {"spec": "X_v1", "ordinal": None, "operand": None}
    base = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed))
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    for k in ("result", "complete", "compared", "equal", "mismatch_n", "compared_by_family", "not_compared", "not_compared_by_family"):
        assert res[k] == base[k], k
    # a promoted-looking row on a member that ALSO has a vocabulary row is a conflict: no producer, fail visibly (as before)
    man = _manifest()
    man["identities"].append({"family": "instance_outputs", "op_path": PRODUCER_OP, "output_member": "out", "spec": "ScaleRow_v1{w=4}"})
    p = OC.producers_of(man)[(PRODUCER_OP, "out")]
    assert p.get("conflict") is True and p["spec"] is None and "consumers" not in p


def test_promoted_member_without_a_producer_instance_is_not_compared_named(tmp_path):
    """No fold instance of the producer family feeds the consumers: not compared, the why names the producer and the consumers, the
    compare is result None / complete False (never silently complete).  Same when the consumer reads another storage (`feed=False`)
    and when the manifest's producer family is not what the fold ran."""
    for kw, needle in (({"with_producer": False}, "no fold instance of producer ScaleRow_v1"),
                       ({"feed": False}, "no fold instance of producer ScaleRow_v1"),
                       ({"producer_spec": "Other_v1{}"}, "no fold instance of producer ScaleRow_v1")):
        d, vals = make_match(str(tmp_path / str(len(needle) + len(kw))))
        data = _add_promoted(d, **kw)
        bmap, committed = _promoted_bmap(vals, data)
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(_manifest()))
        assert res["result"] is None and res["complete"] is False and res["mismatch_n"] == 0, (kw, res)
        keys = [k for k in res["not_compared"] if needle in k]
        assert keys and res["not_compared"][keys[0]] == 2 and CONSUMER_OP in keys[0], (kw, res["not_compared"])
        assert res["not_compared_by_family"].get("instance_outputs") == 2
    # a promoted row with no consumers in the manifest: not resolvable, named
    d, vals = make_match(str(tmp_path / "noconsumers"))
    data = _add_promoted(d)
    bmap, committed = _promoted_bmap(vals, data)
    man = _manifest()
    man["identities"][0]["consumers"] = []
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(man))
    assert res["result"] is None and any("not resolvable" in k for k in res["not_compared"]), res["not_compared"]
    # the consumers exist in the fold but at a step the oracle has no consumer instance for: named as such
    o = OC.MatchOracle(d)
    b, why, n = o.member_bytes(0, PRODUCER_OP, "out", "r0", producer={"spec": "ScaleRow_v1", "promoted": True, "consumers": ["top.layers.7.input_layernorm"]})
    assert b is None and n == 0 and why.startswith("no fold instance of the promoted member's consumers")


def test_promoted_resolution_ignores_a_consumer_reading_only_part_of_the_value(tmp_path):
    """Containment, not equality: a consumer instance whose operand is a sub-range of the producer's output still resolves the producer;
    the promoted bytes read are the producer's FULL output view (the member), not the consumer's slice."""
    d, vals = make_match(str(tmp_path))
    data = _add_promoted(d)
    # rewrite the consumer instances' operand as the first half of each row
    lines = open(os.path.join(d, "instances.jsonl")).read().splitlines()
    out = []
    for ln in lines:
        m = json.loads(ln)
        if m.get("hint", "").startswith(CONSUMER_OP + "/"):
            m["operand_views"][0]["shape"] = [H // 2]
        out.append(json.dumps(m))
    open(os.path.join(d, "instances.jsonl"), "w").write("\n".join(out) + "\n")
    bmap, committed = _promoted_bmap(vals, data)
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(_manifest()))
    assert res["result"] is True and res["complete"] is True, res
    assert np.frombuffer(data, dtype="<u2").size == sum(ROWS.values()) * H
