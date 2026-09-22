"""Structural / unfolding comparator (CB-B): a mechanically constructed Program vs an accepted oracle record.

    python -m verity_capture.experimental.cb_b.compare --mine <build dir> --oracle <dir with program.json[, instances.jsonl]> --out <json>

Compares, without normalising anything away (every normalisation applied is *declared* in the output):
  * signature: parameter names/types and return type of the request body;
  * node sequence: for k in 0..N-1, (spec id, operand reference runs, output type) — the flat request body in
    execution order; first divergence with context; counts of mismatching nodes by class;
  * per-step segmentation (oracle: `instances.jsonl` step; mine: node-name prefix `t=k/`): node counts per step;
  * cross-step (KV / state) dependencies: for every node, the set of steps its operands come from — the wiring of
    KV state across steps is compared as (step, spec) → {referenced steps};
  * spec histogram diff;
  * digests: descriptor `program_digest` of both; body digest (sha256 over the node sequence, no statics); and the
    descriptor digest of *mine* after substituting the oracle's WORKLOAD static value (declared: the eager Program's
    root is bound to the sha256 of the workload JSON incl. prompt token *values*, which are not construction inputs
    here — if the bodies are identical this substitution is the only difference).

Node sequences are taken from `verity.ir.codec.decode_program` for both sides (the oracle's `program.json` is the
`verity-gen/program/v1` wrapper with the descriptor under "program"; mine is the bare descriptor).  If the oracle
root inlines the request body behind `n_input_nodes` input nodes, the comparison skips those and re-bases `n`
references by that count — declared in the output as `oracle_input_nodes`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from typing import Any


def load_descriptor(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        d = json.load(f)
    if "program" in d and isinstance(d["program"], dict) and "definitions" in d["program"]:
        return d["program"], {k: v for k, v in d.items() if k != "program"}
    return d, {}


def decode(desc: dict):
    from verity.ir.codec import decode_program
    from verity.ir.defs import REGISTRY
    from verity_vllm.registry import b1  # noqa: F401  (populates REGISTRY with the pinned bindings)
    from verity_vllm.registry import serve3  # noqa: F401  (padding primitives)
    try:
        from verity_vllm.registry.quarantine import dense  # noqa: F401  (Gemma-2 family primitives: GeGLU, softcap, unfused norm chain)
    except Exception:  # noqa: BLE001
        pass

    return decode_program(desc, REGISTRY)


def request_body(decoded, wrapper: dict) -> tuple[list, int, dict]:
    """(nodes, n_input_nodes, info): the root body as encoded (input nodes included — both sides are `Root[...]`
    wrappers whose input nodes stand for the parameters), plus the count of leading `Input*` nodes for segmentation."""
    root = decoded.root
    body = root.body
    info = {"root_id": getattr(root, "id", None) or str(root), "root_nodes": len(body.nodes),
            "params": [[n, t.to_json()] for n, t in body.params], "ret_type": body.ret_type.to_json()}
    n_in = 0
    while n_in < len(body.nodes) and str(getattr(body.nodes[n_in].fn, "id", "")).startswith("Input"):
        n_in += 1
    info["input_nodes"] = [{"form": n.form, "fn": n.fn.id, "out": n.out.to_json()} for n in body.nodes[:n_in]]
    if wrapper.get("n_input_nodes") not in (None, n_in):
        info["n_input_nodes_wrapper_disagrees"] = wrapper.get("n_input_nodes")
    return list(body.nodes), n_in, info


def node_tuple(n, rebase: int) -> tuple[str, list, Any]:
    from verity.ir.refs import runs

    rs = []
    for c in n.args:
        r = runs(c.refs)
        if rebase:
            r = [(sp, (idx - rebase if sp == "n" else idx), lo, hi) for sp, idx, lo, hi in r]
        rs.append([list(x) for x in r])
    return n.fn.id, rs, n.out.to_json()


def seq_digest(seq: list) -> str:
    h = hashlib.sha256()
    for t in seq:
        h.update(json.dumps(t, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def step_of_names(names: list[str | None]) -> list[int]:
    out, cur = [], 0
    for nm in names:
        m = re.match(r"t=(\d+)/", nm or "")
        if m:
            cur = int(m.group(1))
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True, help="build dir (program.json, body.jsonl, summary.json)")
    ap.add_argument("--oracle", required=True, help="dir with program.json (+ instances.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=3)
    a = ap.parse_args()
    from verity.ir.codec import canonical_json, program_digest

    t0 = time.time()
    R: dict[str, Any] = {"mine": a.mine, "oracle": a.oracle, "declared_normalisations": []}

    o_desc, o_wrap = load_descriptor(os.path.join(a.oracle, "program.json"))
    m_desc, m_wrap = load_descriptor(os.path.join(a.mine, "program.json"))
    R["oracle_wrapper"] = {k: v for k, v in o_wrap.items() if k != "program"}
    R["digest"] = {"oracle_descriptor": program_digest(o_desc), "mine_descriptor": program_digest(m_desc),
                   "oracle_recorded": o_wrap.get("digest")}
    R["digest"]["descriptor_equal"] = R["digest"]["oracle_descriptor"] == R["digest"]["mine_descriptor"]
    R["roots"] = {"oracle": o_desc.get("root"), "mine": m_desc.get("root"), "oracle_program_fn": o_desc.get("program_fn"),
                  "mine_program_fn": m_desc.get("program_fn")}

    O = decode(o_desc)
    M = decode(m_desc)
    R["decode_s"] = round(time.time() - t0, 1)
    o_nodes, o_in, o_info = request_body(O, o_wrap)
    m_nodes, m_in, m_info = request_body(M, m_wrap)
    R["input_nodes"] = {"oracle": o_in, "mine": m_in, "equal": o_in == m_in}
    R["body"] = {"oracle": o_info, "mine": m_info, "oracle_nodes": len(o_nodes), "mine_nodes": len(m_nodes)}
    def request_sig(desc: dict) -> dict:          # the request function's own signature (Captured_v1{...}: prompt, weights -> samples)
        pf = desc.get("program_fn") or ""
        d = desc["definitions"].get(pf) or desc["definitions"].get(pf.replace('"', "'")) or {}
        return {"params": d.get("params"), "ret": d.get("ret"), "form": d.get("form"), "found": bool(d)}

    o_sig, m_sig = request_sig(o_desc), request_sig(m_desc)
    R["signature"] = {"root_params_equal": o_info["params"] == m_info["params"], "root_ret_equal": o_info["ret_type"] == m_info["ret_type"],
                      "request_fn": {"oracle": o_sig, "mine": m_sig}, "params_equal": o_sig["params"] == m_sig["params"],
                      "ret_equal": o_sig["ret"] == m_sig["ret"] and o_info["ret_type"] == m_info["ret_type"]}

    o_seq = [node_tuple(n, 0) for n in o_nodes]
    m_seq = [node_tuple(n, 0) for n in m_nodes]
    o_names = [n.name for n in o_nodes]
    m_names = [n.name for n in m_nodes]
    body_jsonl = os.path.join(a.mine, "body.jsonl")     # node names are annotations, not in the descriptor: take mine from the build
    if os.path.exists(body_jsonl):
        with open(body_jsonl) as f:
            mine_names = [json.loads(l)["name"] for l in f]
        if len(mine_names) + m_in == len(m_nodes):
            m_names = [None] * m_in + mine_names
        else:
            R["body"]["body_jsonl_len_mismatch"] = [len(mine_names), len(m_nodes) - m_in]
    R["digest"]["oracle_body"] = seq_digest(o_seq)
    R["digest"]["mine_body"] = seq_digest(m_seq)
    R["digest"]["body_equal"] = R["digest"]["oracle_body"] == R["digest"]["mine_body"]

    # spec histograms
    oc, mc = Counter(t[0] for t in o_seq), Counter(t[0] for t in m_seq)
    R["spec_histogram"] = {"equal": oc == mc, "oracle_only": {k: v for k, v in oc.items() if k not in mc},
                           "mine_only": {k: v for k, v in mc.items() if k not in oc},
                           "count_diff": {k: [oc[k], mc[k]] for k in set(oc) | set(mc) if oc[k] != mc[k] and k in oc and k in mc},
                           "n_specs": [len(oc), len(mc)]}

    # node-by-node
    n = min(len(o_seq), len(m_seq))
    first = None
    classes = Counter()
    mism: list[dict] = []
    for k in range(n):
        o, m = o_seq[k], m_seq[k]
        if o == m:
            continue
        if first is None:
            first = k
        if o[0] != m[0]:
            cls = "spec"
        elif o[2] != m[2]:
            cls = "out_type"
        else:
            cls = "wiring"
        classes[cls] += 1
        if len(mism) < 400:
            # per-argument diff: which operand differs, and how (operand runs are [(space, node, lo, hi)])
            arg_diff = [{"arg": i, "oracle": oa, "mine": ma} for i, (oa, ma) in enumerate(zip(o[1], m[1])) if oa != ma]
            mism.append({"k": k, "class": cls, "spec": o[0], "mine_spec": m[0], "name": m_names[k], "arg_diff": arg_diff})
    R["nodes"] = {"compared": n, "mismatching": sum(classes.values()), "by_class": dict(classes), "first_divergence": first,
                  "length_equal": len(o_seq) == len(m_seq), "mismatches": mism}
    if first is not None:
        lo, hi = max(0, first - a.context), min(n, first + a.context + 1)
        R["nodes"]["context"] = [{"k": k, "oracle": {"spec": o_seq[k][0], "args": o_seq[k][1], "out": o_seq[k][2], "name": o_names[k]},
                                  "mine": {"spec": m_seq[k][0], "args": m_seq[k][1], "out": m_seq[k][2], "name": m_names[k]},
                                  "equal": o_seq[k] == m_seq[k]} for k in range(lo, hi)]

    # per-step segmentation
    o_steps = None
    inst = os.path.join(a.oracle, "instances.jsonl")
    if os.path.exists(inst):
        o_steps = []
        with open(inst) as f:
            next(f)                                  # header line
            for line in f:
                o_steps.append(json.loads(line)["step"])
        if len(o_steps) + o_in == len(o_seq):
            o_steps = [0] * o_in + o_steps          # input nodes belong to no step; counted under step 0 on both sides
        else:
            R["body"]["instances_len_mismatch"] = [len(o_steps), len(o_seq) - o_in]
            o_steps = None
    if o_steps is None:
        o_steps = step_of_names(o_names)
        R["declared_normalisations"].append("oracle step segmentation from node-name prefixes `t=k/` (instances.jsonl absent)")
    m_steps = None
    summ = os.path.join(a.mine, "summary.json")        # my per-step node counts come from the fold's own step report
    if os.path.exists(summ):
        with open(summ) as f:
            fs = json.load(f).get("fold", {}).get("steps", [])
        counts = [int(st["nodes"]) for st in fs]
        if sum(counts) + m_in == len(m_seq):
            m_steps = [0] * m_in
            for si, c in enumerate(counts):
                m_steps += [si] * c
        else:
            R["body"]["summary_steps_len_mismatch"] = [sum(counts), len(m_seq) - m_in]
    if m_steps is None:
        m_steps = step_of_names(m_names)
        R["declared_normalisations"].append("mine step segmentation from node-name prefixes `t=k/` (fold step report absent)")
    R["steps"] = {"oracle": dict(sorted(Counter(o_steps).items())), "mine": dict(sorted(Counter(m_steps).items()))}
    R["steps"]["equal"] = R["steps"]["oracle"] == R["steps"]["mine"]

    # cross-step dependencies: (step, spec) -> Counter of referenced steps (KV / state edges)
    def deps(seq, steps):
        out: dict[str, Counter] = {}
        for k, (spec, rs, _) in enumerate(seq):
            key = f"{steps[k]}:{spec}"
            c = out.setdefault(key, Counter())
            for r in rs:
                for sp, idx, lo, hi in r:
                    if sp == "n":
                        c[str(steps[idx]) if 0 <= idx < len(steps) else "?"] += 1
                    else:
                        c[f"p{idx}"] += 1
        return out

    od, md = deps(o_seq, o_steps), deps(m_seq, m_steps)
    cross_o = {k: {s: c for s, c in v.items() if not s.startswith("p") and s != k.split(":")[0]} for k, v in od.items()}
    cross_m = {k: {s: c for s, c in v.items() if not s.startswith("p") and s != k.split(":")[0]} for k, v in md.items()}
    cross_o = {k: v for k, v in cross_o.items() if v}
    cross_m = {k: v for k, v in cross_m.items() if v}
    R["cross_step_deps"] = {"equal": cross_o == cross_m, "oracle_keys": len(cross_o), "mine_keys": len(cross_m),
                            "oracle_sample": dict(list(cross_o.items())[:6]), "mine_sample": dict(list(cross_m.items())[:6]),
                            "differing": [k for k in set(cross_o) | set(cross_m) if cross_o.get(k) != cross_m.get(k)][:20]}

    # WORKLOAD-substituted descriptor digest (declared normalisation, reported separately from the raw digests)
    def static_of(pfn: str | None, key: str) -> str | None:
        if not pfn:
            return None
        m = re.search(key + r'=["\']([^"\']+)["\']', pfn)
        return m.group(1) if m else None

    ow, mw = static_of(o_desc.get("program_fn"), "WORKLOAD"), static_of(m_desc.get("program_fn"), "WORKLOAD")
    op, mp = static_of(o_desc.get("program_fn"), "PROFILE"), static_of(m_desc.get("program_fn"), "PROFILE")
    R["statics"] = {"oracle": {"PROFILE": op, "WORKLOAD": ow}, "mine": {"PROFILE": mp, "WORKLOAD": mw}}
    if ow and mw and ow != mw:
        txt = canonical_json(m_desc).decode()
        subst = txt.replace(mw, ow)
        if op and mp and op != mp:
            subst = subst.replace(mp, op)
            R["declared_normalisations"].append("PROFILE static (eager: `profile_id_of(build_manifest)` — a runtime-manifest digest, "
                                                "pod/wheel-path dependent; mine: the profile module id) substituted for the "
                                                "substituted digest only")
        d2 = json.loads(subst)
        R["digest"]["mine_descriptor_with_oracle_statics"] = program_digest(d2)
        R["digest"]["descriptor_equal_modulo_statics"] = R["digest"]["mine_descriptor_with_oracle_statics"] == R["digest"]["oracle_descriptor"]
        R["declared_normalisations"].append("WORKLOAD static (sha256 of the workload JSON incl. prompt token values) substituted "
                                            "for `descriptor_equal_modulo_statics` only; raw descriptor digests reported unchanged")

    R["verdict"] = {
        "exact_expanded_circuit_equality": bool(R["signature"]["params_equal"] and R["signature"]["ret_equal"] and R["nodes"]["length_equal"]
                                                and R["nodes"]["mismatching"] == 0),
        "descriptor_digest_equal": R["digest"]["descriptor_equal"],
        "descriptor_digest_equal_modulo_declared_statics": R["digest"].get("descriptor_equal_modulo_statics"),
    }
    R["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1, default=str)
    print(json.dumps({k: R[k] for k in ("verdict", "digest", "signature", "nodes", "steps", "spec_histogram", "cross_step_deps",
                                        "declared_normalisations", "statics", "seconds")}, indent=1, default=str)[:12000])
    return 0 if R["verdict"]["exact_expanded_circuit_equality"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
