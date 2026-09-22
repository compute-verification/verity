"""Correspondence between per-rank capture identities (`.../port=<k>/s=<rank>`) and the canonical TP=2 circuit
(`registry.b1_tp2.ServeTP2`). TA1's `correspondence.py` contract and machinery, with three additions:

  * the identity carries the tensor-parallel rank (capture-format field `s`); the rank selects the *member*
    of a batched call (`batch(Gemm, ...)` over ranks) or the per-rank call (`EmbeddingShard/s=<r>`), and
    is ignored for values the program has once and production materialises on every rank ("replicated":
    norms, all-reduce outputs, the gathered logits, the selection);
  * a replicated value has one value identity and two authentication locations: rank 0's leaf is the
    transcript source, rank 1's leaf is checked bit-for-bit against it (`replica_check`);
  * a weight leaf `w=<param>/s=<r>` authenticates a *subset* of the unsharded weight `Input` gates (rows or
    columns); replicated weights (norms, cos/sin) are authenticated twice.

Every mismatch is recorded; nothing is dropped (`not_in_program` lists the leaves without a counterpart).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.correspondence.core import (
    CommitStore,
    LazyTranscript,
    NotInProgram,
    Segments,
    WeightsBinding,
    _coll_range,
    _returns_component,
    _shape_ok,
    check_call,
    cos_sin_cache_words,
    load_safetensors,
)
from verity.ir.layout import Circuit, FunctionCall
from verity.ir.program import Program
from verity_vllm.registry import b1_tp2

_ID_RE = re.compile(r"^p=(?P<p>[^/]+)/r=(?P<r>[^/]+)/t=(?P<t>\d+)/ph=(?P<ph>[^/]+)/l=(?P<l>[^/]+)/op=(?P<op>[^/]+)/port=(?P<output>\d+)(?:/s=(?P<s>\d+))?$")


@dataclass(frozen=True)
class CaptureIdentityTP:
    profile: str
    request: str
    t: int
    phase: str
    layer: str
    op: str
    output: int
    shard: int | None  # capture-format field `s`: the tensor-parallel rank that materialised the value

    @classmethod
    def parse(cls, canonical_id: str) -> "CaptureIdentityTP":
        m = _ID_RE.match(canonical_id)
        if not m:
            raise ValueError(f"not a capture-v2 identity: {canonical_id}")
        return cls(m["p"], m["r"], int(m["t"]), m["ph"], m["l"], m["op"], int(m["output"]), None if m["s"] is None else int(m["s"]))


# ---------------------------------------------------------------------------------------------------------
# the map: (op suffix, port) -> (root call role, body node name, return component, how the rank selects)
#   "member":     rank r is member r of a batched call (the rank axis is a batch axis)
#   "rank-call":  rank r is the call named "<node>/s=<r>" (the rank is a static of the function)
#   "replicated": one value in the program, materialised on every rank (rank ignored for identity)
# ---------------------------------------------------------------------------------------------------------

TP2_OPS: dict[tuple[str, int], tuple[str, str, int, str]] = {
    ("embed_tokens.embedding", 0): ("embed", "embed_tokens.embedding", 0, "rank-call"),
    ("embed_tokens.all_reduce", 0): ("embed", "embed_tokens.all_reduce", 0, "replicated"),
    ("input_layernorm.rmsnorm", 0): ("pre", "input_layernorm.rmsnorm", 0, "replicated"),
    ("input_layernorm.rmsnorm", 1): ("pre", "input_layernorm.rmsnorm", 1, "replicated"),
    ("self_attn.qkv_proj.gemm", 0): ("pre", "self_attn.qkv_proj.gemm", 0, "member"),
    ("self_attn.qkv_proj.add", 0): ("pre", "self_attn.qkv_proj.add", 0, "member"),
    ("self_attn.rotary_emb.rope", 0): ("pre", "self_attn.rotary_emb.rope", 0, "member"),
    ("self_attn.rotary_emb.rope", 1): ("pre", "self_attn.rotary_emb.rope.k", 0, "member"),
    ("self_attn.attn.attn", 0): ("attn", "self_attn.attn.attn", 0, "member"),
    ("self_attn.o_proj.gemm", 0): ("post", "self_attn.o_proj.gemm", 0, "member"),
    ("self_attn.o_proj.all_reduce", 0): ("post", "self_attn.o_proj.all_reduce", 0, "replicated"),
    ("post_attention_layernorm.rmsnorm", 0): ("post", "post_attention_layernorm.rmsnorm", 0, "replicated"),
    ("post_attention_layernorm.rmsnorm", 1): ("post", "post_attention_layernorm.rmsnorm", 1, "replicated"),
    ("mlp.gate_up_proj.gemm", 0): ("post", "mlp.gate_up_proj.gemm", 0, "member"),
    ("mlp.act_fn.act_mul", 0): ("post", "mlp.act_fn.act_mul", 0, "member"),
    ("mlp.down_proj.gemm", 0): ("post", "mlp.down_proj.gemm", 0, "member"),
    ("mlp.down_proj.all_reduce", 0): ("post", "mlp.down_proj.all_reduce", 0, "replicated"),
    ("norm.rmsnorm", 0): ("final", "norm.rmsnorm", 0, "replicated"),
    ("norm.rmsnorm", 1): ("final", "norm.rmsnorm", 1, "replicated"),
    ("lm_head.gemm", 0): ("final", "lm_head.gemm", 0, "member"),
    ("lm_head.all_gather", 0): ("final", "lm_head.all_gather", 0, "replicated"),
    ("sampler.select", 0): ("final", "sampler.select", 0, "replicated"),
}


def _op_key(ident: CaptureIdentityTP) -> tuple[str, int]:
    if ident.layer.isdigit():
        return ident.op[len(f"layers.{ident.layer}."):], ident.output
    return ident.op, ident.output


class CircuitMapTP2:
    """Per-rank capture identity -> (producing call, root-scoped return gates, rank kind)."""

    def __init__(self, program: Program):
        self.program = program
        self.circuit: Circuit = program.circuit
        self.named: dict[str, FunctionCall] = {c.node.name: c for c in self.circuit.root_calls() if c.node.name}
        self._kids: dict[str, dict[tuple[str, int], FunctionCall]] = {}

    def _children(self, root: FunctionCall) -> dict[tuple[str, int], FunctionCall]:
        k = root.node.name
        if k not in self._kids:
            self._kids[k] = {(c.node.name, c.member): c for c in root.children()}
        return self._kids[k]

    def locate(self, ident: CaptureIdentityTP) -> tuple[FunctionCall, tuple[int, ...], str]:
        t = ident.t
        key = _op_key(ident)
        if key not in TP2_OPS:
            raise NotInProgram(f"{ident.op}/port={ident.output}: no node of the TP=2 program")
        role, node, which, kind = TP2_OPS[key]
        if role == "embed":
            root_name = f"t={t}/embed"
        elif role == "final":
            root_name = f"t={t}/final"
            if root_name not in self.named:
                raise NotInProgram(f"t={t}: production computes {ident.op} at every prefill position; the canonical program "
                                   f"has a Final call only at sampled positions (this value has no consumer)")
        else:
            root_name = f"t={t}/layers.{ident.layer}/{role}"
        root = self.named[root_name]
        kids = self._children(root)
        r = ident.shard or 0
        if kind == "member":
            c = kids[(node, r)]
        elif kind == "rank-call":
            c = kids[(f"{node}/s={r}", 0)]
        else:
            c = kids[(node, 0)]
        return c, _returns_component(c, which), kind


# ---------------------------------------------------------------------------------------------------------
# weights: per-rank shard leaves authenticate subsets of the unsharded Input gates
# ---------------------------------------------------------------------------------------------------------

_SHARDED_LAYER = {  # field -> (leaf id template, checkpoint sub-tensors, how vLLM shards)
    "ln1": ("w=layers.{L}.input_layernorm.weight", ["model.layers.{L}.input_layernorm.weight"], "replicated"),
    "qkv_w": ("w=layers.{L}.self_attn.qkv_proj.weight", [f"model.layers.{{L}}.self_attn.{p}_proj.weight" for p in "qkv"], "rows-per-block"),
    "qkv_b": ("w=layers.{L}.self_attn.qkv_proj.bias", [f"model.layers.{{L}}.self_attn.{p}_proj.bias" for p in "qkv"], "rows-per-block"),
    "o_w": ("w=layers.{L}.self_attn.o_proj.weight", ["model.layers.{L}.self_attn.o_proj.weight"], "columns"),
    "ln2": ("w=layers.{L}.post_attention_layernorm.weight", ["model.layers.{L}.post_attention_layernorm.weight"], "replicated"),
    "gate_up_w": ("w=layers.{L}.mlp.gate_up_proj.weight", ["model.layers.{L}.mlp.gate_proj.weight", "model.layers.{L}.mlp.up_proj.weight"], "rows-per-block"),
    "down_w": ("w=layers.{L}.mlp.down_proj.weight", ["model.layers.{L}.mlp.down_proj.weight"], "columns"),
}


def bind_weights_tp2(program: Program, circuit: Circuit, tensors: dict[str, np.ndarray], weights_index: dict, C: dict,
                     cos_sin: tuple[np.ndarray, str] | None = None) -> WeightsBinding:
    TP = C["TP"]
    idx = {e["canonical_id"]: e for e in weights_index["entries"]}
    W = program.input_coll(1)
    seg = Segments()
    records: list[dict] = []
    ranges_seen: dict[str, list[int]] = {}

    def add_shard(coll_path: str, cid_base: str, r: int, blocks: list[tuple[Any, np.ndarray]], shape: list[int], *, replicated: bool,
                  source: str = "checkpoint tensor(s), raw bf16 words"):
        """`blocks`: (contiguous Input sub-collection, its checkpoint words) in the leaf's row-major order."""
        cid = f"{cid_base}/s={r}"
        e = idx.get(cid)
        if e is None:
            records.append({"leaf_id": cid, "value_identity": None, "missing_leaf": "no weights leaf for this rank's shard"})
            return
        flat_parts = []
        gate_runs: list[list[int]] = []
        for coll, words in blocks:
            start, end = _coll_range(circuit, coll)
            w = np.ascontiguousarray(words).reshape(-1)
            assert end - start == w.size, (cid, end - start, w.size)
            flat_parts.append(w)
            gate_runs.append([start, end])
            if not (replicated and r > 0):
                seg.add(start, w, {"namespace": "authentication", "commitment": "weights", "leaf_id": cid, "leaf": e["leaf"], "sha256": e["sha256"]})
        flat = np.concatenate(flat_parts)
        assert list(e["shape"]) == shape, (cid, e["shape"], shape)
        digest = hashlib.sha256(flat.tobytes()).hexdigest()
        rec = {
            "leaf_id": cid, "rank": r,
            "value_identity": {"input_gates": gate_runs if len(gate_runs) <= 8 else {"runs": len(gate_runs), "first": gate_runs[0], "last": gate_runs[-1]},
                               "gate_class": "Input/weights", "scalars": int(flat.size), "field": coll_path,
                               "shard": "replica of the whole tensor" if replicated else f"rank {r}'s shard (a projection of the unsharded Input gates)"},
            "authentication_location": {"commitment": "weights", "leaf": e["leaf"], "sha256": e["sha256"], "dtype": e["dtype"], "shape": e["shape"],
                                        "row_block": "row-major words of the leaf; run k covers Input gates run_k[0]..run_k[1]"},
            "rebuilt_from": source, "rebuilt_sha256": digest, "sha256_matches_leaf": digest == e["sha256"],
        }
        if replicated:
            prev = ranges_seen.get(cid_base)
            if r == 0:
                ranges_seen[cid_base] = [e["sha256"]]
            else:
                rec["replica_check"] = {"of": f"{cid_base}/s=0", "leaf_sha256_equal": prev is not None and prev[0] == e["sha256"],
                                        "note": "same Input gates; a second authentication location of the same values"}
        records.append(rec)

    V, H, NL = C["V"], C["H"], C["NL"]
    emb = tensors["model.embed_tokens.weight"]
    VS = V // TP
    for r in range(TP):
        add_shard("weights.embed", "w=embed_tokens.weight", r, [(W["embed"][r * VS:(r + 1) * VS], emb[r * VS:(r + 1) * VS])], [VS, H], replicated=False)
    for L in range(NL):
        w = W["layers"][L]
        for fname, (cid_t, srcs, how) in _SHARDED_LAYER.items():
            if fname == "qkv_b" and not C.get("BIAS", True):
                continue
            path = f"weights.layers[{L}].{fname}"
            parts = [tensors[s.format(L=L)] for s in srcs]
            cid_base = cid_t.format(L=L)
            for r in range(TP):
                if how == "replicated":
                    add_shard(path, cid_base, r, [(w[fname], parts[0])], list(parts[0].shape), replicated=True)
                elif how == "rows-per-block":
                    blocks = []
                    off = 0
                    for p in parts:
                        n = p.shape[0] // TP
                        blocks.append((w[fname][off + r * n:off + (r + 1) * n], p[r * n:(r + 1) * n]))
                        off += p.shape[0]
                    shape = [sum(p.shape[0] // TP for p in parts)] + list(parts[0].shape[1:])
                    add_shard(path, cid_base, r, blocks, shape, replicated=False)
                else:  # columns: each output row's K/TP slice is one contiguous run
                    p = parts[0]
                    Kr = p.shape[1] // TP
                    blocks = [(w[fname][i][r * Kr:(r + 1) * Kr], p[i, r * Kr:(r + 1) * Kr]) for i in range(p.shape[0])]
                    add_shard(path, cid_base, r, blocks, [p.shape[0], Kr], replicated=False)
    for r in range(TP):
        add_shard("weights.norm", "w=norm.weight", r, [(W["norm"], tensors["model.norm.weight"])], [H], replicated=True)
    if cos_sin is None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cos_sin = (cos_sin_cache_words(C["MAX_POS"], C["D"], C["ROPE_THETA"], dev), dev)
    cs, dev = cos_sin
    for r in range(TP):
        add_shard("weights.cos_sin", "w=layers.0.self_attn.rotary_emb.cos_sin_cache", r, [(W["cos_sin"], cs)], [C["MAX_POS"], C["D"]], replicated=True,
                  source=f"vLLM RotaryEmbedding._compute_cos_sin_cache re-run on {dev} (derived table, not a checkpoint tensor)")
    bound = {r["leaf_id"] for r in records}
    for cid in idx:
        if cid not in bound:
            records.append({"leaf_id": cid, "value_identity": None, "not_in_program": "no gate of the canonical program consumes this leaf",
                            "authentication_location": {"commitment": "weights", "leaf": idx[cid]["leaf"], "shape": idx[cid]["shape"]}})
    return WeightsBinding(seg, records)


# ---------------------------------------------------------------------------------------------------------
# checks specific to TP
# ---------------------------------------------------------------------------------------------------------


def check_copy(gates: tuple[int, ...], transcript: LazyTranscript, captured: np.ndarray) -> dict:
    """A gate-free function (`AllGather2`): the captured words must equal the transcript's values at the
    returned gates (here: the two ranks' captured `lm_head.gemm` shards)."""
    t0 = time.time()
    mism: list[dict] = []
    missing = n = 0
    for j, g in enumerate(gates):
        try:
            v = transcript[g]
        except KeyError:
            missing += 1
            continue
        if v != int(captured[j]):
            n += 1
            if len(mism) < 16:
                mism.append({"word": j, "gate": g, "ir": int(v), "captured": int(captured[j])})
    return {"status": "missing-operand" if missing else ("exact" if n == 0 else "mismatch"), "gates_evaluated": 0,
            "words_compared": len(gates) - missing, "mismatches": n, "first_mismatches": mism, "mode": "copy (gate-free function)",
            "seconds": round(time.time() - t0, 3)}


def value_identity_tp2(call: FunctionCall, gates: tuple[int, ...]) -> dict:
    """`<root call name>/<node name>#<member>` plus the gate range (TA1's record shape)."""
    sc = call.parent
    while sc.parent is not None and sc.parent.parent is not None:
        sc = sc.parent
    root_name = sc.parent.spec.body.nodes[sc.node_idx].name if sc.parent is not None else None
    contiguous = all(gates[i + 1] == gates[i] + 1 for i in range(len(gates) - 1))
    return {"namespace": "value", "function": call.id, "call": f"{root_name}/{call.node.name}#{call.member}", "member": call.member,
            "region": [call.lo, call.hi], "returns_gates": [gates[0], gates[-1] + 1] if contiguous and gates else None,
            "returns_gate_count": len(gates), "contiguous": contiguous}


def compare_replicas(a: np.ndarray, b: np.ndarray) -> dict:
    d = a != b
    n = int(d.sum())
    first = [int(i) for i in np.flatnonzero(d)[:16]]
    return {"equal": n == 0, "words": int(a.size), "differing_words": n, "first_differing_words": first}


# ---------------------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------------------


#: eager TP=2 capture: the K rows are the k-rope leaf, the V rows sit in the bias-added qkv leaf
EAGER_KV_SOURCES = frozenset({("self_attn.rotary_emb.rope", 1), ("self_attn.qkv_proj.add", 0)})
#: layer L-1 leaves layer L's input norm reads (eager: h and the materialised r2)
EAGER_PREV_LAYER_SOURCES = ("mlp.down_proj.all_reduce", "post_attention_layernorm.rmsnorm")


def run(store: CommitStore, checkpoint: list[Path], request: str, positions: list[int], layers: list[int], coord_samples: int,
        full: set[tuple[int, int]], out_path: Path, seed: int = 0, cos_sin_npy: Path | None = None, *,
        cmap_cls: type = CircuitMapTP2, kv_sources: frozenset | None = None, prev_layer_sources: tuple[str, ...] | None = None,
        recompute: bool = False, config_label: str = "B1-TP2") -> dict:
    """The program is always `ServeTP2` (the TP=2 statics).  A port over a *different capture partition* of the same
    program (lane COMP's b1-tp2-compiled: W1d's compiled-fx capture per rank) supplies its identity table as `cmap_cls`
    (a `CircuitMapTP2` subclass), the leaves that hold every position's K/V rows (`kv_sources`), the layer L-1 leaves
    layer L's first call reads (`prev_layer_sources`), and whether never-materialised operands are recomputed exactly
    from committed values (`recompute`, TA1's `Recomputer`)."""
    from verity_vllm.correspondence.core import Recomputer

    kv_sources = EAGER_KV_SOURCES if kv_sources is None else kv_sources
    prev_layer_sources = EAGER_PREV_LAYER_SOURCES if prev_layer_sources is None else prev_layer_sources
    C = dict(b1_tp2.B1_TP2_CONFIG)
    TP = C["TP"]
    bnd = store.boundary(request)
    prompt = bnd["prompt_token_ids"]
    lp = len(prompt)
    steps = len(store.steps(request)) - 1
    t0 = time.time()
    program = Program(b1_tp2.serve_tp2(lp, steps, C))
    circuit = program.circuit
    cmap = cmap_cls(program)
    recomputer = Recomputer(circuit) if recompute else None
    build_s = time.time() - t0
    print(f"[corr-tp2] program built: {program.gates:,} gates, {program.input_gates:,} Input gates, {len(cmap.named)} named root calls, {build_s:.1f}s", flush=True)

    tensors = load_safetensors(checkpoint)
    t0 = time.time()
    cs = (np.load(cos_sin_npy), "cuda") if cos_sin_npy else None
    wb = bind_weights_tp2(program, circuit, tensors, store.weights_index, C, cs)
    n_ok = sum(bool(r.get("sha256_matches_leaf")) for r in wb.records)
    print(f"[corr-tp2] weights bound: {len(wb.records)} leaves, sha256 ok={n_ok}, replicas={sum('replica_check' in r for r in wb.records)}, "
          f"not-in-program={sum('not_in_program' in r for r in wb.records)}, missing={sum('missing_leaf' in r for r in wb.records)}, {time.time() - t0:.1f}s", flush=True)

    pcoll = program.input_coll(0)
    pstart, pend = _coll_range(circuit, pcoll)
    prompt_seg = Segments()
    prompt_seg.add(pstart, np.asarray(prompt, dtype=np.uint32), {"namespace": "authentication", "commitment": f"boundary/{request}", "leaf_id": "prompt_token_ids"})
    prompt_record = {"value_identity": {"input_gates": [pstart, pend], "gate_class": "Input/prompt", "scalars": pend - pstart},
                     "authentication_location": {"commitment": f"boundary/{request}", "leaf_id": "prompt_token_ids",
                                                 "root": store.commitments["trees"][f"boundary/{request}"]["root"]}}

    cap = Segments()
    located: dict[str, tuple[FunctionCall, tuple[int, ...], str, dict]] = {}
    replica_of: dict[str, str] = {}  # rank>0 replicated leaf -> rank-0 leaf id
    not_in_program: list[dict] = []
    layer_set = set(layers)
    t0 = time.time()
    for t in range(0, max(positions) + 1):
        step, entries = store.entries_at(request, t)
        root = store.tree_root(request, step)
        by_base: dict[str, dict[int, dict]] = {}
        for e in entries:
            ident = CaptureIdentityTP.parse(e["canonical_id"])
            keep = t in positions or (ident.layer.isdigit() and int(ident.layer) in layer_set and
                                      any(ident.op.endswith(s) and ident.output == p for s, p in kv_sources))
            keep = keep or ident.layer == "sampler"
            if not keep:
                continue
            if ident.layer.isdigit() and int(ident.layer) not in layer_set and not (
                    int(ident.layer) + 1 in layer_set and any(ident.op.endswith(s) for s in prev_layer_sources)):
                continue
            base = e["canonical_id"].rsplit("/s=", 1)[0] if ident.shard is not None else e["canonical_id"]
            by_base.setdefault(base, {})[ident.shard if ident.shard is not None else 0] = e
        for base, per_rank in by_base.items():
            for r in sorted(per_rank):
                e = per_rank[r]
                ident = CaptureIdentityTP.parse(e["canonical_id"])
                try:
                    call, gates, kind = cmap.locate(ident)
                except NotInProgram as ex:
                    not_in_program.append({"capture_id": e["canonical_id"], "reason": str(ex),
                                           "authentication_location": {"commitment": f"ru/{request}/{step:06d}", "leaf": e["leaf"]}})
                    continue
                words = store.leaf_words(request, step, e)
                if len(gates) != words.size:
                    raise ValueError(f"{e['canonical_id']}: {len(gates)} gates vs {words.size} words")
                located[e["canonical_id"]] = (call, gates, kind, {**e, "step": step, "root": root})
                if kind == "replicated" and r != min(per_rank):
                    replica_of[e["canonical_id"]] = per_rank[min(per_rank)]["canonical_id"]
                    continue  # rank 0's copy is the transcript source; this copy is checked against it
                if call.name == "AllGather2":
                    continue  # gate-free: its returns *are* the two lm_head shards' gates (already in the transcript); checked by `check_copy`
                loc = {"namespace": "authentication", "commitment": f"ru/{request}/{step:06d}", "root": root, "leaf_id": e["canonical_id"],
                       "leaf": e["leaf"], "offset": e["offset"], "nbytes": e["nbytes"]}
                run_start = 0
                for i in range(1, len(gates) + 1):
                    if i == len(gates) or gates[i] != gates[i - 1] + 1:
                        cap.add(gates[run_start], words[run_start:i], {**loc, "word_base": run_start})
                        run_start = i
    print(f"[corr-tp2] captured leaves located: {len(located)} ({len(cap)} segments, {len(replica_of)} replica copies), {time.time() - t0:.1f}s", flush=True)

    transcript = LazyTranscript(wb.segments, prompt_seg, cap)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    summary: dict[str, Any] = {"exact": 0, "mismatch": 0, "missing-operand": 0, "sampled": 0, "shape_mismatch": 0, "replica_equal": 0, "replica_differs": 0}
    per_function: dict[str, dict[str, int]] = {}

    def flush() -> dict:
        """Write the document as it stands (after every record, so a long run can be inspected and survives a kill)."""
        out = {"schema": "verity-ir/correspondence/v0", "config": config_label, "model": C["model"], "tensor_parallel_size": TP, "request": request,
               "positions": positions, "layers": layers,
               "capture_partition": {"circuit_map": type(cmap).__name__, "kv_sources": sorted(f"{s}/port={p}" for s, p in kv_sources),
                                     "prev_layer_sources": list(prev_layer_sources), "recompute_missing": recomputer.stats if recomputer else None},
               "tensor_parallel": {"ranks": TP, "identity": "capture-format field s (/s=<rank>) on every value and weight id",
                                   "rank_kinds": {"member": "rank r's slice of a batch-over-ranks call", "rank-call": "rank r's own call (partials, embedding shard, logits shard)",
                                                  "replicated": "one program value materialised on every rank: rank 0's copy is the transcript source, the others are checked against it (records with check.mode 'replica')"},
                                   "summary_keys": {"replica_equal": "replica copies equal to rank 0's (counted in exact)", "replica_differs": "replica copies that differ (counted in mismatch)"}},
               "program": {"function": program.fn.id, "gates": program.gates, "input_gates": program.input_gates, "build_seconds": round(build_s, 1)},
               "input_gate_records": {"prompt": prompt_record, "weights": wb.records},
               "not_in_program": not_in_program, "records": records, "summary": summary, "per_function": per_function}
        out_path.write_text(json.dumps(out, indent=1))
        return out

    for cid, (call, gates, kind, e) in sorted(located.items(), key=lambda kv: (kv[1][1][0], kv[0])):
        ident = CaptureIdentityTP.parse(cid)
        if ident.t not in positions or (ident.layer.isdigit() and int(ident.layer) not in layer_set):
            continue
        ok, why = _shape_ok(call, gates, e)
        rec = {"capture_id": cid, "rank": ident.shard, "rank_kind": kind, "value_identity": value_identity_tp2(call, gates),
               "authentication_location": {"namespace": "authentication", "commitment": f"ru/{request}/{e['step']:06d}", "root": e["root"],
                                           "leaf": e["leaf"], "offset": e["offset"], "nbytes": e["nbytes"], "dtype": e["dtype"], "shape": e["shape"]},
               "shape_type_consistent": ok, "shape_type_detail": why}
        captured = store.leaf_words(request, e["step"], e)
        fkey = call.name
        pf = per_function.setdefault(fkey, {"leaves": 0, "exact": 0, "mismatch": 0, "missing-operand": 0, "replica-copies": 0, "replica-equal": 0})
        if cid in replica_of:
            other = located[replica_of[cid]]
            rc = compare_replicas(store.leaf_words(request, other[3]["step"], other[3]), captured)
            rec["replica_check"] = {"of": replica_of[cid], **rc}
            # a replicated value has one identity (the call's return gates); rank 1's copy corresponds exactly iff
            # it is word-for-word rank 0's (the transcript source, itself checked against the program)
            rec["check"] = {"status": "exact" if rc["equal"] else "mismatch", "mode": "replica (rank 0's copy is the transcript source)",
                            "words_compared": rc["words"], "mismatches": rc["differing_words"],
                            "first_mismatches": [{"word": w, "gate": gates[w], "ir": None, "captured": None} for w in rc["first_differing_words"]]}
            summary["replica_equal" if rc["equal"] else "replica_differs"] += 1
            summary["exact" if rc["equal"] else "mismatch"] += 1
            pf["replica-copies"] += 1
            pf["replica-equal"] += rc["equal"]
            records.append(rec)
            print(f"[corr-tp2] {'replica-equal' if rc['equal'] else 'replica-DIFFERS':16s} {cid.split('/', 4)[-1]:78s} {call.id[:40]}", flush=True)
            flush()
            continue
        if not ok:
            summary["shape_mismatch"] += 1
            rec["check"] = {"status": "shape-mismatch"}
        elif call.name == "AllGather2":
            rec["check"] = check_copy(gates, transcript, captured)
        elif call.name == "Gemm" and (ident.t, int(ident.layer) if ident.layer.isdigit() else -1) not in full:
            kids = list(call.children())
            n = len(kids)
            pick = sorted(set(int(x) for x in rng.choice(n, size=min(coord_samples, n), replace=False)))
            if ident.layer == "lm_head":
                sel_id = cid.replace("l=lm_head/op=lm_head.gemm", "l=sampler/op=sampler.select")
                sel = next((x for x in store.ru(request, e["step"])[0]["entries"] if x["canonical_id"] == sel_id), None)
                if sel is not None:
                    argmax = int(np.frombuffer(store.leaf_words(request, e["step"], sel).tobytes(), dtype=np.int32)[0])
                    local = argmax - (ident.shard or 0) * (C["V"] // TP)
                    if 0 <= local < n:
                        pick = sorted(set(pick) | {local})
            checks = []
            t1 = time.time()
            for i in pick:
                r = check_call(call, gates, transcript, captured, sub=kids[i], recompute=recomputer)
                checks.append({"coordinate": i, **{k: r[k] for k in ("status", "mismatches")}, **({"detail": r["detail"]} if "detail" in r else {})})
            statuses = {c["status"] for c in checks}
            rec["check"] = {"status": "exact" if statuses == {"exact"} else ("mismatch" if "mismatch" in statuses else "missing-operand"),
                            "mode": "coordinate-sampled", "coordinates": len(pick), "of": n,
                            "words_compared": sum(1 for c in checks if c["status"] == "exact"),
                            "failed_coordinates": [c for c in checks if c["status"] != "exact"][:16], "seconds": round(time.time() - t1, 3)}
            summary["sampled"] += 1
        else:
            rec["check"] = check_call(call, gates, transcript, captured, recompute=recomputer)
        st = rec["check"]["status"]
        summary[st] = summary.get(st, 0) + 1
        pf["leaves"] += 1
        if st in pf:
            pf[st] += 1
        records.append(rec)
        print(f"[corr-tp2] {st:16s} {cid.split('/', 4)[-1]:78s} {call.id[:40]:40s} {rec['check'].get('seconds', 0):8.2f}s", flush=True)
        flush()
    return flush()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--positions", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 13, 27])
    ap.add_argument("--coord-samples", type=int, default=32)
    ap.add_argument("--full", nargs="*", default=[], help="t,L pairs whose Gemms are evaluated in full (L=-1: lm_head)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cos-sin-npy", type=Path, default=None)
    ap.add_argument("--port", type=Path, default=None,
                    help="a TP=2 profile port directory (static_config.py with CIRCUIT_MAP / KV_SOURCES / PREV_LAYER_SOURCES / RECOMPUTE_MISSING) "
                         "over a different capture partition of ServeTP2; default: the eager b1-tp2 partition")
    a = ap.parse_args(argv)
    full = {tuple(int(x) for x in s.split(",")) for s in a.full}
    seams: dict[str, Any] = {}
    if a.port is not None:
        from verity_vllm.correspondence.core import Port

        port = Port.load(a.port)
        if port.circuit_map is not None:
            seams["cmap_cls"] = _cmap_class_of(port.circuit_map)
        seams["kv_sources"] = port.kv_sources
        seams["prev_layer_sources"] = port.prev_layer_sources
        seams["recompute"] = port.recompute_missing
        seams["config_label"] = f"B1-TP2/{port.id}"
    run(CommitStore(a.commit), a.checkpoint, a.request, a.positions, a.layers, a.coord_samples, full, a.out, a.seed, cos_sin_npy=a.cos_sin_npy, **seams)  # type: ignore[arg-type]
    return 0


def _cmap_class_of(factory) -> type:
    """A port's `CIRCUIT_MAP` is either the class itself or a one-argument factory `program -> CircuitMapTP2-like`;
    `run` takes something constructible from a Program, so a factory is wrapped."""
    if isinstance(factory, type):
        return factory

    class _Wrapped:  # a class whose constructor is the factory
        def __new__(cls, program):
            return factory(program)

    _Wrapped.__name__ = getattr(factory, "__name__", "CIRCUIT_MAP")
    return _Wrapped


if __name__ == "__main__":
    sys.exit(main())
