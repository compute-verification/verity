"""verity_vllm/check/fold_compare.py — M4: the four comparisons of a folded observation log against the evidence and Serve_v4.

    uv run verity-vllm fold-compare --log data/logs/m1.jsonl.gz --snapshots data/logs/m1 \
        --profile vllm_d9105ea80_sm89_eager --out out/capture/reports/compare_m1.json

Each comparison is a function returning a JSON-able dict with a `status` in {"pass", "fail", "skipped", "error"}:

1. `compare_tokens`     — the sampled token ids the fold's TokenSelect instances stand for (`SampleEnd`) vs the
                          observer's `tokens.json` vs the accepted record's `output_token_ids`.
2. `compare_boundary`   — provenance: where the fold says an instance's operand holds the references of some
                          earlier instance's output (or a parameter), the bytes L1 snapshotted for the operand's
                          view (state right before the consumer) are bit-identical to the bytes of the producer's
                          output view (or the checkpoint / prompt).  Covers the KV slot rows read by attention
                          (RoPE k outputs, the qkv v slice), the logits-row gather and the decode input token.
3. `compare_subcircuits`— a handful of derived instances are evaluated with the reference evaluator on their snapshot
                          operand values (fed as encodings through `evaluate_call`) and compared bit-exactly with the
                          snapshot of their outputs.
4. `compare_structure`  — `match_program(flat_body, Serve_v4, allow_extra=True)`: matched / superset, the extra nodes
                          grouped into named CLASSES (never forced to isomorphism), and node counts per specialization
                          id in the derived body vs the family's expanded leaves.

Everything here is torch-free.  Resource discipline (DESIGN §11): the M1 run measured 0.93 GB max RSS / 8 s on the
laptop (fold + Program() + digest alone: 0.21 GB / 2.4 s); comparison 3 deliberately excludes Embedding and lm_head
instances, whose `FunctionCall.args` would materialise 28M-element tuples (~1 GB each).  Snapshots follow DESIGN §3: `<snapshots>/snapshots.json` is an index of entries
`{event, arg, view: {"$v": [...]}, file, sha256, step?}` and each file holds the raw element bytes of the view in
row-major logical order (a storage-order dump of the view's byte extent is also accepted).  Weights come from the
SmolLM2-135M safetensors in the HF cache (`_LAYER_FIELDS`) and the cos/sin cache from the
tracked fixture the interior producer uses.

Where the compared values come from (the loaders, `SnapshotStore`, the weight tables and `WeightSource`) is `fold_sources`;
every name of it is handed on here, and other code reads it from `fold_compare`.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate_call
from verity.circuits.layout import FunctionCall
from verity.circuits.refs import Refs, runs
from verity.circuits.types import Tuple as TupleT
from verity.definitions.registry import b1

from verity_vllm.check.fold_sources import _EXTRA_NORM_SOURCES as _EXTRA_NORM_SOURCES
from verity_vllm.check.fold_sources import _LAYER_FIELDS as _LAYER_FIELDS
from verity_vllm.check.fold_sources import _NP as _NP
from verity_vllm.check.fold_sources import _WEIGHT_FIELDS as _WEIGHT_FIELDS
from verity_vllm.check.fold_sources import DEFAULT_COS_SIN as DEFAULT_COS_SIN
from verity_vllm.check.fold_sources import HF_GLOB as HF_GLOB
from verity_vllm.check.fold_sources import Snapshot as Snapshot
from verity_vllm.check.fold_sources import (
    SnapshotStore,
    WeightSource,
    _extent,
    _load_json,
    fold_log,
    load_profile,
)
from verity_vllm.check.fold_sources import (
    _extra_norm_layer_fields as _extra_norm_layer_fields,
)
from verity_vllm.check.fold_sources import _f16_words_to_bf16 as _f16_words_to_bf16
from verity_vllm.check.fold_sources import (
    _layernorm_family_fields as _layernorm_family_fields,
)
from verity_vllm.check.fold_sources import _moe_layer_fields as _moe_layer_fields
from verity_vllm.check.fold_sources import _qkv_neox_permute as _qkv_neox_permute
from verity_vllm.check.fold_sources import _storage_indices as _storage_indices
from verity_vllm.check.fold_sources import hf_safetensors_glob as hf_safetensors_glob
from verity_vllm.config import option
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.fold import FoldResult, summarize
from verity_vllm.program.frontend.rules.family import (
    expand_leaves,
    match_program,
    node_leaf_id,
)

__all__ = ["Options", "SnapshotStore", "Values", "fold_log", "load_profile", "main"]

# --- values by provenance ----------------------------------------------------------------------------------

class Values:
    """Words of a reference sequence resolved through the fold: parameter leaves from the prompt / checkpoint, node
    output leaves from the snapshot of the producing event's output view."""

    def __init__(self, result: FoldResult, store: SnapshotStore, weights: WeightSource | None) -> None:
        self.r, self.store, self.weights = result, store, weights
        self.by_node = {m["node"]: m for m in result.instances}
        self.W = len(result.params) - 1

    def of_refs(self, refs: Refs) -> np.ndarray:
        parts = [self.of_run(space, idx, lo, hi) for space, idx, lo, hi in runs(refs)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)

    def of_run(self, space: str, idx: int, lo: int, hi: int) -> np.ndarray:
        if space == "p":
            if idx == self.W:
                if self.weights is None or not self.weights.available():
                    raise LookupError("no weight source (checkpoint safetensors / cos_sin fixture)")
                return self.weights.words(lo, hi)
            toks = self.r.requests[idx]["prompt_token_ids"]
            return np.array([int(t) & 0xFFFFFFFF for t in toks[lo:hi]], dtype=np.uint32)
        if space != "n":
            raise LookupError(f"reference space {space!r}")
        m = self.by_node.get(idx)
        if m is None:
            raise LookupError(f"node {idx} has no instance record")
        out = []
        for k, (clo, chi) in enumerate(_component_ranges(m)):
            if chi <= lo or clo >= hi:
                continue
            view = m["output_views"][k]
            if view is None:
                raise LookupError(f"node {idx} component {k}: output view unknown")
            s = self.store.find(view, at=m["event"])
            if s is None:
                raise LookupError(f"node {idx} ({m['name']}): no snapshot of event {m['event']} covers {view!r}")
            words = self.store.extract(s, view)
            out.append(words[max(lo, clo) - clo:min(hi, chi) - clo])
        w = np.concatenate(out)
        if w.size != hi - lo:
            raise LookupError(f"node {idx}: leaves [{lo}, {hi}) outside its {sum(chi - clo for clo, chi in _component_ranges(m))} outputs")
        return w


def _component_ranges(meta: dict) -> list[tuple[int, int]]:
    """Leaf ranges of the return components of the instance's specialization."""
    spec = meta.get("_spec")
    if spec is None:
        raise LookupError("instance metadata carries no specialization object")
    _, ret = spec.signature()
    comps = ret.fields if isinstance(ret, TupleT) else (ret,)
    out, off = [], 0
    for t in comps:
        out.append((off, off + t.leaves))
        off += t.leaves
    return out


def _attach_specs(result: FoldResult) -> None:
    """The instance records carry `spec` labels; comparisons need the specialization object of each node."""
    for m in result.instances:
        m["_spec"] = result.body.nodes[m["node"]].fn


def _encodings(words: np.ndarray) -> list[int]:
    return [int(x) for x in words.tolist()]


# --- 1. tokens ---------------------------------------------------------------------------------------------

def compare_tokens(result: FoldResult, tokens_json: Path | None = None, record: Path | None = None) -> dict:
    per_req: dict[str, list[int]] = {req: [int(tok) for _, tok in seq] for req, seq in result.sampled.items()}
    per_req_steps: dict[str, list[int]] = {req: [int(step) for step, _ in seq] for req, seq in result.sampled.items()}
    token_select_nodes = {}
    for req, seq in result.outputs.items():
        for j, refs in enumerate(seq):
            node = None
            for space, idx, lo, hi in runs(refs):
                node = (space, idx)
            token_select_nodes[f"{req}@{j}"] = None if node is None or node[0] != "n" else result.body.nodes[node[1]].fn.id
    out: dict[str, Any] = {"status": "pass", "fold": per_req, "fold_steps": per_req_steps,
                           "token_select_specs": Counter(v for v in token_select_nodes.values() if v), "mismatches": []}
    if any(v is None or not v.startswith("TokenSelect") for v in token_select_nodes.values()):
        out["mismatches"].append("some sampler outputs are not TokenSelect instances: "
                                 + ", ".join(f"{k}: {v}" for k, v in token_select_nodes.items() if not (v or "").startswith("TokenSelect")))
    if tokens_json is not None and Path(tokens_json).exists():
        obs = _tokens_from_json(_load_json(Path(tokens_json)))
        out["observer"] = obs
        for req, toks in per_req.items():
            ref = obs.get(req) if req in obs else (next(iter(obs.values())) if len(obs) == 1 else None)
            if ref is None:
                out["mismatches"].append(f"observer tokens.json has no entry for request {req!r}")
            elif [int(t) for t in ref] != toks:
                out["mismatches"].append(f"observer tokens for {req!r} differ: {ref} vs fold {toks}")
    else:
        out["observer"] = None
    # the accepted tokens the capture wrote next to its own (tokens.json `accepted_record_token_ids`, from the workload file)
    if tokens_json is not None and Path(tokens_json).exists():
        d = _load_json(Path(tokens_json))
        requests = d.get("requests")  # a tokens file that names no request records no accepted tokens
        acc = {r["request_id"]: [int(t) for t in r["accepted_record_token_ids"]] for r in (requests if requests else [])
               if isinstance(r, dict) and r.get("accepted_record_token_ids")}
        out["accepted"] = acc or None
        for req, toks in per_req.items():
            a = acc.get(req)
            if a is not None and a != toks:
                out["mismatches"].append(f"accepted tokens for {req!r} differ: {a} vs fold {toks}")
    rec = _record_tokens(record) if record is not None else None
    out["record"] = rec
    if rec:
        for req, toks in per_req.items():
            r = rec.get(req) or (next(iter(rec.values())) if len(rec) == 1 else None)
            if r is None:
                out["mismatches"].append(f"record has no request {req!r}")
            else:
                if r["output_token_ids"] != toks:
                    out["mismatches"].append(f"record output_token_ids for {req!r} differ: {r['output_token_ids']} vs fold {toks}")
                pr = next((q["prompt_token_ids"] for q in result.requests if q["request_id"] == req), None)
                if pr is not None and [int(t) for t in pr] != r["prompt_token_ids"]:
                    out["mismatches"].append(f"record prompt_token_ids for {req!r} differ from the log's")
    if out["mismatches"]:
        out["status"] = "fail"
    return out


def _tokens_from_json(d: Any) -> dict[str, list[int]]:
    if isinstance(d, list) and all(isinstance(x, int) for x in d):
        return {"": d}
    if isinstance(d, dict):
        if all(isinstance(v, list) and all(isinstance(x, int) for x in v) for v in d.values()) and d:
            return {str(k): v for k, v in d.items()}
        for key in ("sampled", "sampled_token_ids", "output_token_ids", "tokens", "generated", "requests"):
            if key in d:
                v = d[key]
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return {str(r.get("request_id", r.get("id", i))):
                            [int(t) for t in r.get("generated_token_ids", r.get("output_token_ids", r.get("tokens", [])))]
                            for i, r in enumerate(v)}
                return _tokens_from_json(v)
    return {}


def _record_tokens(record: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(record.glob("adapter/*/ru/*/boundary.json")):
        d = _load_json(p)
        out[d.get("request_id", p.parent.name)] = {"prompt_token_ids": [int(t) for t in d["prompt_token_ids"]],
                                                    "output_token_ids": [int(t) for t in d["output_token_ids"]], "file": str(p)}
    return out


# --- 2. boundary values / provenance ------------------------------------------------------------------------

def _in_place(meta: dict) -> bool:
    outs = [v for v in meta.get("output_views", []) if v is not None]
    for op in meta.get("operand_views", []):
        if op is None:
            continue
        for ov in outs:
            if op.alloc == ov.alloc and _extent(op)[0] < _extent(ov)[1] and _extent(ov)[0] < _extent(op)[1]:
                return True
    return False


def lm_head_nodes(result: FoldResult) -> set[int]:
    """The Gemm instances whose output a TokenSelect instance reads (the lm_head over the gathered rows)."""
    out: set[int] = set()
    for n in result.body.nodes:
        if n.fn.id.startswith("TokenSelect"):
            for space, idx, _, _ in runs(n.args[0].refs):
                if space == "n" and result.body.nodes[idx].fn.id.startswith("Gemm"):
                    out.add(idx)
    return out


def compare_boundary(result: FoldResult, store: SnapshotStore, weights: WeightSource | None = None,
                     max_per_class: int = 8) -> dict:
    """For consumer instances whose step has snapshots: operand words by provenance (through the fold's references)
    vs the words L1 snapshotted for the operand view right before the consumer ran."""
    _attach_specs(result)
    vals = Values(result, store, weights)
    steps_with = set(store.steps) or None
    lm = lm_head_nodes(result)
    # class name -> (predicate on the instance record, operand index whose provenance is checked)
    classes: dict[str, tuple] = {
        "attention K rows (KV slots <- RoPE k outputs)": (lambda m: m["spec"].startswith("Attention"), 1),
        "attention V rows (KV slots <- qkv v slice)": (lambda m: m["spec"].startswith("Attention"), 2),
        "lm_head Gemm x <- gathered final-norm row": (lambda m: m["node"] in lm, 0),
        "decode Embedding token <- previous TokenSelect": (lambda m: m["spec"].startswith("Embedding") and m["step"] > 0, 0),
        "prefill Embedding token <- prompt": (lambda m: m["spec"].startswith("Embedding") and m["step"] == 0, 0),
        "Gemm x <- producer output": (lambda m: m["spec"].startswith("Gemm") and m["node"] not in lm, 0),
        "RMSNorm residual <- producer output": (lambda m: m["spec"].startswith("RMSNormFused"), 1),
        "SiluMul x <- gate_up Gemm output": (lambda m: m["spec"].startswith("SiluMul"), 0),
    }
    report: dict[str, Any] = {"status": "pass", "checks": [], "classes": {}}
    counts: Counter = Counter()
    for m in result.instances:
        if steps_with is not None and m["step"] not in steps_with:
            continue
        for cname, (pred, k) in classes.items():
            if not pred(m) or counts[cname] >= max_per_class:
                continue
            node = result.body.nodes[m["node"]]
            if k >= len(node.args):
                continue
            view = m["operand_views"][k] if k < len(m["operand_views"]) else None
            check = {"class": cname, "node": m["node"], "name": m["name"], "operand": k}
            try:
                if view is None:
                    raise LookupError("operand view unknown")
                if view.numel != node.args[k].refs.count:
                    raise LookupError(f"operand view has {view.numel} elements for {node.args[k].refs.count} references (gathered rows are not one run)")
                expect = vals.of_refs(node.args[k].refs)
                before = m["event"] if _in_place(m) else m["event"] + 1
                s = store.find(view, before=before)
                if s is None:
                    raise LookupError(f"no snapshot before event {before} covers {view!r}")
                got = store.extract(s, view)
                eq = got.size == expect.size and bool(np.array_equal(got.astype(np.uint64), expect.astype(np.uint64)))
                check.update(status="pass" if eq else "fail", n=int(got.size), snapshot_event=s.event, snapshot_arg=s.arg,
                             first_diff=None if eq else int(np.argmax(got.astype(np.uint64) != expect.astype(np.uint64))))
            except LookupError as e:
                check.update(status="skipped", reason=str(e))
            counts[cname] += 1
            report["checks"].append(check)
    for cname in classes:
        cs = [c for c in report["checks"] if c["class"] == cname]
        report["classes"][cname] = Counter(c["status"] for c in cs)
    st = Counter(c["status"] for c in report["checks"])
    report["summary"] = dict(st)
    report["status"] = "fail" if st.get("fail") else ("pass" if st.get("pass") else "skipped")
    return report


# --- 3. expanded small subcircuits --------------------------------------------------------------------------

def select_instances(result: FoldResult, attention_T: Iterable[int] = (1, 2, 3)) -> list[dict]:
    """One Gemm row of layer 0 qkv, one RMSNormFusedCuda row, one RoPE row, one SiluMul row, the TokenSelect of the
    first step and the attention rows with T in `attention_T` of the first step (no lm_head Gemm / Embedding: their
    weight operands are tens of millions of leaves)."""
    picks: list[dict] = []
    first_step = min((m["step"] for m in result.instances), default=0)
    lm = lm_head_nodes(result)
    want = {"Gemm": lambda m: "layers.0/" in m["name"] and m["node"] not in lm and "qkv" in (m["name"] + (m["hint"] or "")),
            "RMSNormFusedCuda": lambda m: True, "RoPE": lambda m: True, "SiluMul": lambda m: True,
            "TokenSelect": lambda m: m["step"] == first_step}
    seen = set()
    for m in result.instances:
        base = m["spec"].split("{")[0].split("_v")[0]
        if base in want and base not in seen and want[base](m):
            picks.append(m)
            seen.add(base)
    Ts = set(attention_T)
    for m in result.instances:
        if m["step"] == first_step and m["spec"].startswith("Attention"):
            T = int(m["spec"].split("T=")[1].split(",")[0].split("}")[0]) if "T=" in m["spec"] else None
            if T in Ts:
                picks.append(m)
                Ts.discard(T)
    return picks


def compare_subcircuits(result: FoldResult, store: SnapshotStore, weights: WeightSource | None = None,
                        picks: list[dict] | None = None) -> dict:
    _attach_specs(result)
    if result.program is None:
        return {"status": "skipped", "reason": "fold ran without Program()"}
    vals = Values(result, store, weights)
    circuit = result.program.circuit
    root = result.program.root
    n_in = len(root.body.nodes) - len(result.body.nodes)
    report: dict[str, Any] = {"status": "pass", "instances": []}
    for m in picks if picks is not None else select_instances(result):
        entry = {"node": m["node"], "name": m["name"], "spec": m["spec"], "step": m["step"]}
        node = result.body.nodes[m["node"]]
        try:
            t0 = time.time()
            args: list[int] = []
            for a in node.args:
                args += _encodings(vals.of_refs(a.refs))
            expect: list[int] = []
            for k, view in enumerate(m["output_views"]):
                if view is None:
                    raise LookupError(f"output {k} view unknown")
                s = store.find(view, at=m["event"])
                if s is None:
                    raise LookupError(f"no snapshot of event {m['event']} covers output {k} {view!r}")
                expect += _encodings(store.extract(s, view))
            call = FunctionCall(circuit, circuit.root_scope, n_in + m["node"])
            got = list(evaluate_call(circuit, call, args))
            if len(got) != len(expect):
                raise LookupError(f"{len(got)} return encodings vs {len(expect)} snapshot words")
            diff = [i for i in range(len(got)) if got[i] != expect[i]]
            entry.update(status="pass" if not diff else "fail", n_args=len(args), n_returns=len(got), n_diff=len(diff),
                         first_diff=diff[:4], gates=call.hi - call.lo, seconds=round(time.time() - t0, 3))
        except LookupError as e:
            entry.update(status="skipped", reason=str(e))
        except Exception as e:  # evaluator / MUFU table problems are reported, not fatal
            entry.update(status="error", reason=f"{type(e).__name__}: {e}")
        report["instances"].append(entry)
    st = Counter(e["status"] for e in report["instances"])
    report["summary"] = dict(st)
    report["status"] = "fail" if st.get("fail") or st.get("error") else ("pass" if st.get("pass") else "skipped")
    return report


# --- 4. program structure ------------------------------------------------------------------------------------

def _consumed_nodes(result: FoldResult) -> set[int]:
    used: set[int] = set()
    for n in result.body.nodes:
        for a in n.args:
            for space, idx, _, _ in runs(a.refs):
                if space == "n":
                    used.add(idx)
    for space, idx, _, _ in runs(result.body.ret):
        if space == "n":
            used.add(idx)
    return used


def classify_extra(result: FoldResult, extra: list[dict]) -> dict[str, dict]:
    """Group `match_program`'s extra nodes into named classes (spec base, phase, hint tail, live/dead)."""
    by_node = {m["node"]: m for m in result.instances}
    used = _consumed_nodes(result)
    lm = lm_head_nodes(result)
    classes: dict[str, dict] = {}
    for e in extra:
        k = e["index"]
        m = by_node.get(k, {})
        base = e["id"].split("{")[0]
        step = m.get("step")
        phase = "prefill" if step == 0 else "decode" if step is not None else "?"
        hint = (m.get("hint") or "")
        tail = hint.rsplit(".", 1)[-1] if hint else ""
        live = "live" if k in used else "dead"
        if base.startswith("RMSNormFusedCuda") and phase == "prefill" and "layers." not in (m.get("name") or "") and live == "dead":
            name = "final norm applied to every prefill row in vLLM; Serve_v4 normalises only the last position (dead extra instances)"
        elif base.startswith("Gemm") and k in lm:
            name = f"lm_head Gemm over extra gathered rows ({phase}, {live})"
        elif base.startswith("Embedding"):
            name = f"extra Embedding rows ({phase}, {live})"
        else:
            name = f"unclassified: {base} ({phase}, hint tail {tail!r}, {live})"
        c = classes.setdefault(name, {"count": 0, "specs": Counter(), "steps": Counter(), "nodes": []})
        c["count"] += 1
        c["specs"][e["id"]] += 1
        c["steps"][str(step)] += 1
        if len(c["nodes"]) < 8:
            c["nodes"].append({"index": k, "name": e.get("name")})
    return classes


def compare_structure(result: FoldResult, *, C=b1.B0_CONFIG, LP: int = 32, STEPS: int = 15, BN: int | None = None,
                      family=None) -> dict:
    """`match_program` against the hand-authored family `Serve_v4{C, LP, STEPS, BN}`; `BN` defaults to the FA2 rule for
    `C["D"]` (128 at head dim 64, 64 at head dim 128).  `C` carries no `BN` key in `b1.*_CONFIG`; a profile config does."""
    if BN is None:
        BN = C.get("BN") or b1.fa2_kblock_n(C["D"])
    C = {k: v for k, v in C.items() if k != "BN"}
    fam = family if family is not None else bind(b1.ServeV4, C=C, LP=LP, STEPS=STEPS, BN=BN)
    W = len(result.params) - 1
    t0 = time.time()
    try:
        rep = match_program(result.body, fam, flat_weights_param=W, allow_extra=True, with_map=False)
    except Exception as e:
        rep = {"matched": False, "reason": f"{type(e).__name__}: {e}"}
    out: dict[str, Any] = {"status": "pass" if rep.get("matched") or rep.get("superset") else "fail",
                           "family": fam.id, "matched": bool(rep.get("matched")), "superset": bool(rep.get("superset")),
                           "reason": rep.get("reason"), "seconds": round(time.time() - t0, 3)}
    extra = rep.get("extra_nodes")  # listed only for a superset: a match, a mismatch and a matcher that raised list none
    if not extra:
        extra = []
    out["n_extra"] = len(extra)
    out["extra_by_spec"] = dict(Counter(e["id"] for e in extra))
    out["classes"] = {k: {"count": v["count"], "specs": dict(v["specs"]), "steps": dict(v["steps"]), "examples": v["nodes"]}
                      for k, v in classify_extra(result, extra).items()}
    # gate I8 (generalization plan): a superset passes only when every extra instance falls in a named, explained class;
    # an `unclassified:` class is unexplained extra computation and fails (the M4 rule let any superset pass).
    unclassified = {k: v["count"] for k, v in out["classes"].items() if k.startswith("unclassified:")}
    out["unclassified"] = unclassified
    if out["status"] == "pass" and unclassified:
        out["status"] = "fail"
        out["reason"] = f"{sum(unclassified.values())} extra instances in unclassified classes: " + "; ".join(sorted(unclassified))
    derived = Counter(node_leaf_id(n) for n in result.body.nodes)
    leaf_bases = {n.fn.base_id for n in result.body.nodes if hasattr(n.fn, "base_id")}
    try:
        leaves, _, _, _ = expand_leaves(fam, leaf_bases)
        family_hist = Counter(leaf.id for leaf in leaves)
    except Exception as e:
        family_hist = Counter()
        out["family_expand_error"] = f"{type(e).__name__}: {e}"
    ids = sorted(set(derived) | set(family_hist))
    out["histogram"] = {i: {"derived": derived.get(i, 0), "family": family_hist.get(i, 0), "delta": derived.get(i, 0) - family_hist.get(i, 0)}
                        for i in ids}
    out["n_derived_nodes"] = len(result.body.nodes)
    out["n_family_leaves"] = sum(family_hist.values())
    if rep.get("flat_node_ids") or rep.get("family_leaf_ids"):
        out["match_program_ids"] = {"flat": rep.get("flat_node_ids"), "family": rep.get("family_leaf_ids")}
    return out


# --- CLI ------------------------------------------------------------------------------------------------------

def _max_rss_gb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Counter):
        return {str(k): v for k, v in x.items()}
    if isinstance(x, ViewDesc):
        return repr(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, Path):
        return str(x)
    return x


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("--log")
    snapshots: str | None = option("--snapshots", default=None, help="directory with snapshots.json, values/, tokens.json")
    profile: str = option("--profile")
    record: str | None = option("--record", default=None, help="accepted record dir with adapter/*/ru/*/boundary.json (default: none, no record comparison)")
    out: str = option("--out")
    strict: bool = option("--strict", default=False, action="store_true")
    no_program: bool = option("--no-program", default=False, action="store_true", help="skip Program() (comparison 3 is then skipped)")
    lp: int = option("--lp", type=int, default=32)
    steps: int = option("--steps", type=int, default=15)
    bn: int | None = option("--bn", type=int, default=None, help="Serve_v4 BN (default: the profile config's / the FA2 rule at D)")
    safetensors: str | None = option("--safetensors", default=None, help="glob of the checkpoint's model*.safetensors (default: the HF cache of the profile's model)")
    cos_sin: str | None = option("--cos-sin", default=None, help="uint16 .npy of the rotary cos_sin table (default: <snapshots>/cos_sin_cache.npy, else the B0 fixture)")


def main(a: Options) -> int:
    t0 = time.time()
    profile = load_profile(a.profile)
    case = (profile.notes or {}).get("case", "B0")
    repo = (profile.config or {}).get("model")
    record = Path(a.record) if a.record else None
    result = fold_log(a.log, profile, strict=True if a.strict else None, with_program=not a.no_program)
    report: dict[str, Any] = {"log": a.log, "profile": a.profile, "case": case, "model": repo, "fold": summarize(result),
                              "digest": result.digest, "fold_seconds": round(time.time() - t0, 3)}
    snaps = Path(a.snapshots) if a.snapshots else None
    report["tokens"] = compare_tokens(result, snaps / "tokens.json" if snaps else None, record)
    store = None
    if snaps is not None and (snaps / "snapshots.json").exists():
        store = SnapshotStore(snaps)
        cos_sin = Path(a.cos_sin) if a.cos_sin else ((snaps / "cos_sin_cache.npy") if (snaps / "cos_sin_cache.npy").exists() else None)
        weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(a.safetensors))] if a.safetensors else None,
                               cos_sin, repo=repo, config=dict(profile.config or {}))   # family / dtype facts
        report["weights_available"] = weights.available()
        report["weights_source"] = {"safetensors": [str(p) for p in weights.paths], "cos_sin": str(weights.cos_sin_path)}
        report["boundary"] = compare_boundary(result, store, weights)
        report["subcircuits"] = compare_subcircuits(result, store, weights) if not a.no_program else {"status": "skipped", "reason": "--no-program"}
    else:
        report["boundary"] = {"status": "skipped", "reason": "no snapshots.json"}
        report["subcircuits"] = {"status": "skipped", "reason": "no snapshots.json"}
    report["structure"] = compare_structure(result, C=dict(profile.config), LP=a.lp, STEPS=a.steps, BN=a.bn)
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)   # DESIGN §11: every full-size run documents its footprint
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=1, sort_keys=False))
    for k in ("tokens", "boundary", "subcircuits", "structure"):
        print(f"{k:12s} {report[k]['status']}", file=sys.stderr)
    print(str(out))
    return 0 if all(report[k]["status"] in ("pass", "skipped") for k in ("tokens", "boundary", "subcircuits", "structure")) else 1
