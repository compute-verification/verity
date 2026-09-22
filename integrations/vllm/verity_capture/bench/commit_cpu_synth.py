"""CMT-1 fallback measurement without a GPU: hashing / tree-build cost of the reference committer on synthetic step streams sized
exactly like the accepted workload's naturally-materialized stream (B0-cov_b0_b1_c256: prefill 155 tokens, 14 decode steps;
624,384 B per token + 98,308 B per step for the lm_head row and the token id = 106,995,516 B per request, the manifest's
naturally-materialized total).  Random bytes -- hashing cost does not depend on the values.  Host backend (hashlib) and the torch
backend on the CPU, per step, with raw samples.  Labels: measured on the CPU named in the output; nothing here is a GPU number.

    PYTHONPATH=. python -m verity_capture.bench.commit_cpu_synth --out DIR [--reps 3] [--torch-cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from verity_capture.experimental import commit as C

PER_TOKEN = 624_384
PER_STEP = 98_304 + 4
PROGRAM_DIGEST = bytes.fromhex("5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9")


def layouts_for(prompt_len: int, decode_steps: int) -> list[C.StepLayout]:
    out = []
    for s in range(decode_steps + 1):
        n = prompt_len if s == 0 else 1
        L = C.StepLayout(s, n)
        L.add("embedding1/out/gatherbf16x49152_v1/gatherbf16x49152", "model.embed_tokens", "bf16", (n, 576))
        L.add("rmsnormtriton1/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16", "model.layers.0.input_layernorm", "bf16", (n, 576))
        for i in range(30):
            p = f"model.layers.{i}"
            if i > 0:
                L.add("rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16", f"{p}.input_layernorm", "bf16", (n, 576))
                L.add("rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp", f"{p}.input_layernorm.residual", "bf16", (n, 576))
            L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"{p}.self_attn.qkv_proj", "bf16", (n, 960))
            L.add("rope1/out/rotated(ropeout+ropeoutadd)", f"{p}.self_attn.rotary_emb.q", "bf16", (n, 576))
            L.add("rope1/out/rotated(ropeout+ropeoutadd)", f"{p}.self_attn.rotary_emb.k", "bf16", (n, 192))
            L.add("attention3/out/f2fpbf16_v1/f2fpbf16", f"{p}.self_attn.attn", "bf16", (n, 576))
            L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"{p}.self_attn.o_proj", "bf16", (n, 576))
            L.add("rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16", f"{p}.post_attention_layernorm", "bf16", (n, 576))
            L.add("rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp", f"{p}.post_attention_layernorm.residual", "bf16", (n, 576))
            L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"{p}.mlp.gate_up_proj", "bf16", (n, 3072))
            L.add("silumul1/out/silumulbf16_v1/silumulbf16", f"{p}.mlp.act_fn", "bf16", (n, 1536))
            L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", f"{p}.mlp.down_proj", "bf16", (n, 576))
        L.add("rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16", "model.norm", "bf16", (n, 576))
        L.add("rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp", "model.norm.residual", "bf16", (n, 576))
        L.add("gemm1/out/gemmcoordinate_v1/f2fpbf16", "lm_head(logits_processor)", "bf16", (1, 49152))
        L.add("tokenselect1/out/tokenselect_v1/selecti32", "sampler", "i32", (1, 1))
        L.freeze()
        assert L.stream_bytes == n * PER_TOKEN + PER_STEP, (L.stream_bytes, n)
        out.append(L)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=155)
    ap.add_argument("--decode-steps", type=int, default=14)
    ap.add_argument("--torch-cpu", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts = layouts_for(args.prompt_len, args.decode_steps)
    total = sum(L.stream_bytes for L in layouts)
    cpu = platform.processor() or subprocess.run(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"], capture_output=True, text=True).stdout.strip()
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    print(f"[synth] {len(layouts)} steps, {total} B per request (manifest naturally-materialized total 106995516), cpu={cpu} threads={os.cpu_count()}", flush=True)
    rng = os.urandom
    streams = [rng(L.stream_bytes) for L in layouts]
    ctx = C.Context("vllm_d9105ea80_sm89_eager", "synthetic", args.prompt_len, "HuggingFaceTB/SmolLM2-135M", {"synthetic": True})
    raw = []
    roots = {}
    for rep in range(args.reps):
        c = C.Committer(PROGRAM_DIGEST, ctx, backend="host")
        per_step = []
        t0 = time.perf_counter()
        for L, s in zip(layouts, streams, strict=True):
            sc = c.commit_step(C.StepLayout.from_json(L.to_json()), s)
            per_step.append({"step": L.step, "bytes": L.stream_bytes, "leaves": L.leaf_count, "leaf_s": sc.timing["leaf_s"], "tree_s": sc.timing["tree_s"]})
        tf = time.perf_counter()
        root, pos = c.finish()
        t1 = time.perf_counter()
        roots.setdefault("host", set()).add(root.hex())
        raw.append({"backend": "host", "rep": rep, "wall_s": t1 - t0, "finish_s": t1 - tf, "steps": per_step, "hash_calls": c.hash_calls, "hash_blocks": c.hash_blocks,
                    "retained": c.persistent_bytes()})
        print(f"[synth] host rep{rep}: {t1 - t0:.3f}s total; prefill leaf {per_step[0]['leaf_s']:.3f}s tree {per_step[0]['tree_s']:.3f}s; "
              f"decode mean leaf {statistics.fmean(p['leaf_s'] for p in per_step[1:]) * 1e3:.2f}ms tree {statistics.fmean(p['tree_s'] for p in per_step[1:]) * 1e3:.2f}ms", flush=True)
        if args.torch_cpu:
            import torch
            if args.torch_threads:
                torch.set_num_threads(args.torch_threads)
            for hlb in (0, 256):
                ct = C.Committer(PROGRAM_DIGEST, ctx, backend="torch", host_levels_below=hlb)
                per_step_t = []
                t0 = time.perf_counter()
                for L, s in zip(layouts, streams, strict=True):
                    st = torch.frombuffer(bytearray(s), dtype=torch.uint8)
                    ta = time.perf_counter()
                    sc = ct.commit_step(C.StepLayout.from_json(L.to_json()), st)
                    per_step_t.append({"step": L.step, "bytes": L.stream_bytes, "leaves": L.leaf_count, "wall_s": time.perf_counter() - ta, **sc.timing})
                tf = time.perf_counter()
                root_t, _ = ct.finish()
                t1 = time.perf_counter()
                roots.setdefault(f"torch-cpu-hlb{hlb}", set()).add(root_t.hex())
                raw.append({"backend": f"torch-cpu-hlb{hlb}", "rep": rep, "wall_s": t1 - t0, "finish_s": t1 - tf, "steps": per_step_t, "threads": torch.get_num_threads(),
                            "hash_calls": ct.hash_calls, "hash_blocks": ct.hash_blocks, "root_equals_host": root_t == root})
                print(f"[synth] torch-cpu hlb={hlb} rep{rep}: {t1 - t0:.3f}s total ({torch.get_num_threads()} threads); prefill {per_step_t[0]['wall_s']:.3f}s; "
                      f"decode mean {statistics.fmean(p['wall_s'] for p in per_step_t[1:]) * 1e3:.2f}ms; root==host {root_t == root}", flush=True)
    # a few openings + negatives on the synthetic commitment (host)
    ops = []
    t_open = []
    for step in (0, 1, len(layouts) - 1):
        for inst, el in (("model.layers.7.mlp.gate_up_proj", 12345 if step == 0 else 2345), ("lm_head(logits_processor)", 49151)):
            ta = time.perf_counter()
            op = C.open(c, step, "gemm1/out/gemmcoordinate_v1/f2fpbf16", inst, el, stream=streams[step])
            t_open.append(time.perf_counter() - ta)
            ops.append((op, C.verify(root, pos, op)))
    negs = [C.verify(root, pos, ops[0][0].replace(value=bytes([ops[0][0].value[0] ^ 1]) + ops[0][0].value[1:]))[0],
            C.verify(root, pos, ops[0][0].replace(element=ops[0][0].element + 1))[0],
            C.verify(root, pos, ops[0][0], program_digest=hashlib.sha256(b"x").digest())[0]]
    doc = {"schema": "verity-gen/cmt-ref/cpu-synth/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpu": cpu, "cpu_count": os.cpu_count(),
           "host": platform.node(), "code_rev": rev, "label": "measured on CPU, synthetic random streams sized from the manifest (not GPU numbers)",
           "bytes_per_request": total, "leaves_per_request": sum(L.leaf_count for L in layouts), "steps": len(layouts),
           "roots_stable": {k: len(v) == 1 for k, v in roots.items()}, "raw": raw,
           "openings": {"n": len(ops), "all_verify": all(v[0] for _, v in ops), "open_s": t_open, "negatives_rejected": [not x for x in negs]}}
    (out / "cpu_synth.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "raw"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
