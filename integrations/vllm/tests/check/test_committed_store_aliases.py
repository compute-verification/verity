"""`CommittedStore` resolves a Program op_path through the manifest's `op_path_aliases` -- the ONE
table the population reconciliation resolves through -- and reads THAT invocation's entry exactly.

The record that failed (smollm2 B1 on the canary): vLLM has one RotaryEmbedding instance called by every layer; the required
manifest binds it under `model.layers.0.self_attn.rotary_emb` with invocation = layer order and records
`model.layers.N.self_attn.rotary_emb -> [model.layers.0.self_attn.rotary_emb, N]`.  The Attention_v3/v2 row evaluator reads the layer's
query (rotary_emb/0) and key history (rotary_emb/1) through the Program's per-layer path; the pre-fix store looked that path up verbatim
(layers 1..29: 'no committed identity', the replay partial, C2 not of record) and, on a hit, collapsed invocations to the LOWEST (a
neighbour's words could have satisfied a relation).  Here: layer N reads invocation N; no alias = invocation 0 exactly; an invocation with
no entry is missing BY NAME; nothing is ever collapsed."""
from __future__ import annotations

import numpy as np

from verity_vllm.check.replay import coverage as COV, opening as OP
from verity_vllm.records.required_manifest import SCHEMA

ROT = "model.layers.0.self_attn.rotary_emb"
W = 8                                                                            # words per identity


def _entry(op: str, member: str, inv: int, tag: int, *, rid: str = "r0", step: int = 0, lo: int = 0, hi: int = W) -> dict:
    return {"name": f"{op}/{member}#{inv}", "stream_offset": 0, "element_offset": 0, "encoding": {"dtype": "bf16"},
            "identity": {"request_id": rid, "step": step, "op_path": op, "output_member": member, "invocation": inv, "element_range": [lo, hi],
                         "family": "instance_outputs"}}


def _world(n_layers: int = 4):
    """Binding map: the shared rotary module under layer 0's path at invocations 0..n-1 (member 0 = q, member 1 = k), each invocation's
    words tagged by its invocation; `model.norm/out` at invocation 0 only; `model.layers.1.mlp/out` bound ONLY at invocation 3 (no alias)."""
    entries = []
    for inv in range(n_layers):
        entries.append(_entry(ROT, "0", inv, 100 + inv))
        entries.append(_entry(ROT, "1", inv, 200 + inv))
    entries.append(_entry("model.norm", "out", 0, 7))
    entries.append(_entry("model.layers.1.mlp", "out", 3, 9))
    bmap = {"entries": entries}
    bytes_of = {}
    for e in entries:
        inv = e["identity"]["invocation"]
        tag = int(e["name"].split("#")[0].endswith("/1")) * 100 + 100 + inv if e["identity"]["op_path"] == ROT else (7 if "norm" in e["name"] else 9)
        bytes_of[e["name"]] = (np.full(W, tag, dtype=np.uint16)).tobytes()

    def committed(step: int, name: str, lo: int, hi: int) -> bytes | None:
        b = bytes_of.get(name)
        return None if b is None else b[lo:hi]

    manifest = {"op_path_aliases": {f"model.layers.{n}.self_attn.rotary_emb": [ROT, n] for n in range(1, n_layers)}}
    return bmap, committed, manifest


def test_layer_n_reads_invocation_n_of_the_canonical_path_through_the_manifest_alias_table():
    bmap, committed, manifest = _world()
    st = OP.CommittedStore.of_record(bmap, committed, manifest)
    for n in range(4):
        op = ROT if n == 0 else f"model.layers.{n}.self_attn.rotary_emb"
        assert st.resolve(op) == (ROT, n)
        for member, base in (("0", 100), ("1", 200)):
            w, dtype = st.words("r0", 0, op, member, 0, W)
            assert w is not None and dtype == "bf16" and set(w.tolist()) == {base + n}, (op, member, w)   # layer n's OWN words, never a neighbour's
            assert st.has("r0", 0, op, member)
    # a slice inside the identity resolves the same way
    w, _ = st.words("r0", 0, "model.layers.2.self_attn.rotary_emb", "1", 2, 5)
    assert w is not None and w.tolist() == [202, 202, 202]


def test_no_alias_means_invocation_zero_exactly_and_a_missing_invocation_is_named_never_the_lowest():
    bmap, committed, manifest = _world()
    st = OP.CommittedStore.of_record(bmap, committed, manifest)
    # a per-layer path with NO alias entry (layer 4 does not exist): missing by name, the Program path in the message
    w, why = st.words("r0", 0, "model.layers.4.self_attn.rotary_emb", "0", 0, W)
    assert w is None and "no committed identity for (r0, step 0, model.layers.4.self_attn.rotary_emb/0)" in why
    assert not st.has("r0", 0, "model.layers.4.self_attn.rotary_emb", "0")
    # an alias to an invocation that has no entry (the module was called fewer times than the alias table says): named, with the bound set
    st2 = OP.CommittedStore(bmap, committed, {"model.layers.9.self_attn.rotary_emb": [ROT, 9]})
    w, why = st2.words("r0", 0, "model.layers.9.self_attn.rotary_emb", "0", 0, W)
    assert w is None and "[alias -> model.layers.0.self_attn.rotary_emb invocation 9]" in why and "bound invocations" in why and "[0, 1, 2, 3]" in why
    # an UNALIASED path bound only at invocation 3: invocation 0 is what the path names -> missing by name (the pre-fix store read invocation 3)
    w, why = st.words("r0", 0, "model.layers.1.mlp", "out", 0, W)
    assert w is None and "model.layers.1.mlp/out" in why and "[invocation 0]" in why and "[3]" in why
    assert not st.has("r0", 0, "model.layers.1.mlp", "out")
    # the canonical path itself = invocation 0 (layer 0), also without any alias table
    for store in (st, OP.CommittedStore(bmap, committed)):
        w, _ = store.words("r0", 0, ROT, "0", 0, W)
        assert w is not None and set(w.tolist()) == {100}
        w, _ = store.words("r0", 0, "model.norm", "out", 0, W)
        assert w is not None and set(w.tolist()) == {7}


def test_without_the_alias_table_a_shared_module_layer_path_is_missing_by_name_not_the_lowest_invocation():
    bmap, committed, _manifest = _world()
    plain = OP.CommittedStore(bmap, committed)                                   # no manifest: no aliases (never a silent default table)
    assert plain.resolve("model.layers.2.self_attn.rotary_emb") == ("model.layers.2.self_attn.rotary_emb", 0)
    w, why = plain.words("r0", 0, "model.layers.2.self_attn.rotary_emb", "0", 0, W)
    assert w is None and "no committed identity for (r0, step 0, model.layers.2.self_attn.rotary_emb/0)" in why
    assert OP.CommittedStore.of_record(bmap, committed, None).aliases == {}


def test_the_store_and_the_population_reconciliation_resolve_through_the_same_table():
    """`alias_table` / `resolve_op_path` are the one resolver: the manifest identity of layer n's rotary output (canonical path, invocation n)
    is the key the population reconciliation forms for the Program's per-layer row address, and the key the store reads."""
    bmap, committed, manifest = _world()
    ids = [{"family": "instance_outputs", "request_id": "r0", "step": 0, "op_path": ROT, "output_member": "0", "invocation": n, "element_range": [0, W],
            "rank": 0, "spec": "RoPE_v1{}"} for n in range(4)]
    man = dict(manifest, schema=SCHEMA, program_digest="ab" * 64, manifest_digest="cd" * 64, requests=["r0"], identities=ids,
               query={"id": "Q_module_v1"})
    table = OP.alias_table(man)
    assert table == {f"model.layers.{n}.self_attn.rotary_emb": (ROT, n) for n in range(1, 4)}
    # the Program's row addresses, per-layer paths: every manifest identity is addressed (0 without rows) iff the aliases resolve
    addresses = {("r0", 0, ROT if n == 0 else f"model.layers.{n}.self_attn.rotary_emb", "0"): {"class": "vu", "rows": 1, "spec": "RoPE_v1{}", "why": None}
                 for n in range(4)}
    qp = COV.query_population(man, addresses, set(), ranks=[0])
    assert "manifest_refused" not in qp["query_checks"] and qp["query_total"]["identities"] == 4
    assert qp["query_checks"]["identities_without_rows"]["n"] == 0 and qp["query_checks"]["vus_outside_query"]["n"] == 0
    store = OP.CommittedStore.of_record(bmap, committed, man)
    for (rid, step, op, mem) in addresses:
        canon, inv = OP.resolve_op_path(table, op)
        assert store.resolve(op) == (canon, inv) and store.has(rid, step, op, mem)
    # drop the aliases from the manifest: the reconciliation names layers 1..3 as identities without rows AND the store misses them by name
    man2 = dict(man, op_path_aliases={})
    qp2 = COV.query_population(man2, addresses, set(), ranks=[0])
    assert qp2["query_checks"]["identities_without_rows"]["n"] == 3 and qp2["query_checks"]["vus_outside_query"]["n"] == 3
    store2 = OP.CommittedStore.of_record(bmap, committed, man2)
    assert [store2.has("r0", 0, f"model.layers.{n}.self_attn.rotary_emb", "0") for n in range(1, 4)] == [False] * 3
