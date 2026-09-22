"""Row-9 census preparation (CB-B): turn a build dir into `vu_census.py` run dirs.

    python -m verity_capture.experimental.cb_b.row9_prepare --build <dir> --out <runs dir> [--oracle-instances <instances.jsonl>]

Writes `<runs>/flat/{program.json,instances.jsonl}` from `program.json` + `body.jsonl` and, if present,
`<runs>/padded/{program.json,instances.jsonl}` from `padded_program.json` + `padded_body.jsonl`.  `program.json` is the
`verity-gen/program/v1` wrapper the census expects (`n_input_nodes` = leading `Input*` root nodes); `instances.jsonl`
carries node → (spec, pattern, step, inserted) — `pattern` is the eager pattern id of the spec family (taken from the
oracle's instances when given, else from the fixed spec-prefix table), used by the census only to classify the eager
physical location; padding nodes (guards, muxes, constants) get pattern `root`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

PATTERN_OF_PREFIX = {
    "Gemm_v1": "gemm.matmul_kernel_persistent", "Attention_v3": "attention.fa2", "RMSNormFusedCuda_v2": "rmsnorm.fused_add_rms_norm",
    "RMSNormTriton_v1": "rmsnorm.triton_rms_norm_kernel", "RoPE_v1": "rope.rotary_embedding", "SiluMul_v1": "act.silu_and_mul",
    "Embedding_v1": "embedding.aten_embedding", "TokenSelect_v1": "sample.gumbel_two_stage",
}


def _open(p: str):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def _load_desc(build: str, name: str) -> dict:
    for cand in (os.path.join(build, name), os.path.join(build, name + ".gz")):
        if os.path.exists(cand):
            with _open(cand) as f:
                return json.load(f)
    raise FileNotFoundError(name)


def _n_input_nodes(desc: dict) -> int:
    root = desc["definitions"][desc["root"]]
    n = 0
    for node in root["body"]["nodes"] if isinstance(root.get("body"), dict) else root.get("nodes", []):
        fn = node.get("fn") or node.get("callee") or ""
        if isinstance(fn, int):
            fn = desc["callees"][fn] if "callees" in desc else str(fn)
        if str(fn).startswith("Input"):
            n += 1
        else:
            break
    return n


def write_run(desc: dict, body_lines: list[dict], out: str, patterns: dict[str, str], steps: list[int] | None, profile: str) -> dict:
    import verity_vllm.registry.b1  # noqa: F401
    import verity_vllm.registry.serve3  # noqa: F401  (padding primitives: PadGuard / SelectBf16 / BitNot …)
    from verity.ir.codec import decode_program, program_digest
    from verity.ir.defs import REGISTRY

    os.makedirs(out, exist_ok=True)
    n_in = _n_input_nodes(desc)
    digest = program_digest(desc)
    root_nodes = decode_program(desc, REGISTRY).root.body.nodes      # out leaves per node (the census's materialization check)
    with open(os.path.join(out, "program.json"), "w") as f:
        json.dump({"schema": "verity-gen/program/v1", "digest": digest, "n_input_nodes": n_in, "n_instances": len(body_lines),
                   "profile": profile, "program": desc}, f)
    with open(os.path.join(out, "instances.jsonl"), "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "profile": profile, "digest": digest, "n_instances": len(body_lines),
                            "note": "cb-b synthesized: node k of the request body = root node n_input_nodes + k"}) + "\n")
        for o in body_lines:
            k = o["k"]
            spec = o["spec"]
            inserted = o.get("inserted")
            pat = "root" if inserted else patterns.get(spec.split("{")[0], "root")
            leaves = root_nodes[n_in + k].fn.out_leaves
            f.write(json.dumps({"node": k, "spec": spec, "pattern": pat, "name": o.get("name"), "step": steps[k] if steps else None,
                                "inserted": inserted, "form": o.get("form"), "n": o.get("n"),
                                "output_views": [{"shape": [leaves], "note": "meta: leaf count, not a runtime tensor"}], "operand_views": []}) + "\n")
    return {"out": out, "digest": digest, "n_input_nodes": n_in, "n_body_nodes": len(body_lines)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-instances", default=None)
    a = ap.parse_args()
    patterns = dict(PATTERN_OF_PREFIX)
    if a.oracle_instances:
        with _open(a.oracle_instances) as f:
            next(f)
            for line in f:
                o = json.loads(line)
                patterns.setdefault(o["spec"].split("{")[0], o["pattern"])
    with open(os.path.join(a.build, "summary.json")) as f:
        summ = json.load(f)
    profile = summ["inputs"]["profile_id"]
    counts = [int(st["nodes"]) for st in summ["fold"]["steps"]]
    flat_steps: list[int] = []
    for si, c in enumerate(counts):
        flat_steps += [si] * c
    report = {}
    with _open(os.path.join(a.build, "body.jsonl")) as f:
        body = [json.loads(l) for l in f]
    report["flat"] = write_run(_load_desc(a.build, "program.json"), body, os.path.join(a.out, "flat"), patterns, flat_steps, profile)
    pb = os.path.join(a.build, "padded_body.jsonl")
    if os.path.exists(pb) or os.path.exists(pb + ".gz"):
        with _open(pb if os.path.exists(pb) else pb + ".gz") as f:
            pbody = [json.loads(l) for l in f]
        # steps of the padded body: kept nodes follow the flat order; inserted nodes take the step of the preceding kept node
        psteps, j = [], 0
        for o in pbody:
            if o.get("inserted"):
                psteps.append(psteps[-1] if psteps else 0)
            else:
                psteps.append(flat_steps[j])
                j += 1
        assert j == len(flat_steps), (j, len(flat_steps))
        report["padded"] = write_run(_load_desc(a.build, "padded_program.json"), pbody, os.path.join(a.out, "padded"), patterns, psteps, profile)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
