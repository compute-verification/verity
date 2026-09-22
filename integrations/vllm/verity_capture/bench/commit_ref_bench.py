"""CMT-1 reference path: acquire the naturally-materialized interface tensors of the accepted workload production-style (forward
hooks -> preallocated device buffer) and commit them per engine step; measure every component separately; verify openings against
the ORIGINAL root with the production verifier core; negatives.  Writes raw samples (jsonl) + positions + openings + a summary.

Arms (interleaved: rep 0 of every arm, then rep 1, ...; the same engine, the same request every time):
    control          event-only hooks (step start / step end CUDA events), nothing acquired
    acquire          + copies into the device buffer (no commit)
    host             + D2H of the step stream + hashlib leaves + host tree (the slow reference; one sync per step)
    torch            + torch SHA-256 on the device (eager torch ops), tree on the device, root deferred to finish()
    torch-hybrid     + torch SHA-256 leaves + lower tree levels on the device, upper levels (<= N nodes) on the host (one small sync)
    torch-compiled   + torch.compile'd compression (fused), otherwise as torch-hybrid

    cd /workspace/cp/veritor && PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.bench.commit_ref_bench \
        --out /workspace/cp/cmt1/bench --reps 3 --arms control,acquire,host,torch,torch-hybrid,torch-compiled
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import struct
import sys
import time
from pathlib import Path

from verity_vllm_adapter import profile as prof  # noqa: F401

prof.apply_env()

import torch  # noqa: E402

from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.experimental import commit as C  # noqa: E402
from verity_capture.experimental.commit import acquire as A  # noqa: E402

PROGRAM_DIGEST = "5bb5ac55c1ee110189c8d43e138ff1b41c1e7d48224777f1ef688d27c535edb9"
ARMS = {
    "control": {},
    "acquire": {"acquire": True},
    "host": {"acquire": True, "backend": "host"},
    "torch": {"acquire": True, "backend": "torch", "hlb": 0},
    "torch-hybrid": {"acquire": True, "backend": "torch", "hlb": 256},
    "torch-compiled": {"acquire": True, "backend": "torch", "hlb": 256, "compiled": True},
    "triton": {"acquire": True, "backend": "torch", "hlb": 0, "compiled": "triton"},
    "triton-hybrid": {"acquire": True, "backend": "torch", "hlb": 256, "compiled": "triton"},
}


def _ms(a: torch.cuda.Event, b: torch.cuda.Event) -> float:
    return float(a.elapsed_time(b))


class Run:
    """One (arm, rep): installs the hooks, drives the request, collects per-step events/timers, tears down."""

    def __init__(self, eng: va.Engine, arm: dict, *, ctx: C.Context, program_digest: bytes, buffer_bytes: int, keep_streams: bool) -> None:
        self.eng, self.arm, self.ctx, self.pd = eng, arm, ctx, program_digest
        self.ev_start: list[torch.cuda.Event] = []
        self.ev_precommit: list[torch.cuda.Event] = []
        self.ev_end: list[torch.cuda.Event] = []
        self.host_t: list[dict] = []
        self.handles = []
        self.acq: A.Acquirer | None = None
        self.committer: C.Committer | None = None
        self.streams: list[torch.Tensor] = []
        self.keep_streams = keep_streams
        core = getattr(eng.model, "model", eng.model)
        if arm.get("acquire"):
            if arm.get("backend"):
                self.committer = C.Committer(program_digest, ctx, backend=arm["backend"], compiled=arm.get("compiled", False),
                                             host_levels_below=int(arm.get("hlb", 0)))
            self.acq = A.Acquirer(eng.model, eng.runner, on_step=self._on_step, buffer_bytes=buffer_bytes, device="cuda", mode="copy")

        def pre(_m, _inp):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev_start.append(e)
        self.handles.append(core.embed_tokens.register_forward_pre_hook(pre))
        if self.acq is not None:
            self.acq.install()

        def post(_m, _inp, _out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev_end.append(e)
            if self.acq is None:
                self.ev_precommit.append(e)
        self.handles.append(A.hook_sampler(eng.runner, post))

    def _on_step(self, rec: A.StepRecord, stream: torch.Tensor) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev_precommit.append(e)
        if self.keep_streams:
            self.streams.append(stream.clone())
        if self.committer is None:
            return
        t = {"step": rec.step, "n_tokens": rec.n_tokens, "copies": rec.copies, "bytes": rec.bytes, "leaves": rec.layout.leaf_count}
        if self.committer.backend == "host":
            t0 = time.perf_counter()
            torch.cuda.synchronize()               # wait for the forward + copies (attributed separately: not the D2H)
            t1 = time.perf_counter()
            host = stream.cpu().numpy().tobytes()  # the D2H (padded stream)
            t2 = time.perf_counter()
            sc = self.committer.commit_step(rec.layout, host[: rec.layout.stream_bytes])
            t3 = time.perf_counter()
            t.update({"sync_wait_s": t1 - t0, "d2h_s": t2 - t1, "leaf_hash_s": sc.timing["leaf_s"], "tree_s": sc.timing["tree_s"], "commit_wall_s": t3 - t2})
        else:
            t0 = time.perf_counter()
            sc = self.committer.commit_step(rec.layout, stream)
            t1 = time.perf_counter()
            t.update({"commit_enqueue_s": sc.timing["device_enqueue_s"], "host_tail_s": sc.timing["host_tail_s"], "commit_wall_s": t1 - t0})
        self.host_t.append(t)

    def close(self) -> dict:
        for h in self.handles:
            h.remove()
        if self.acq is not None:
            self.acq.remove()
        torch.cuda.synchronize()
        n = min(len(self.ev_start), len(self.ev_end), len(self.ev_precommit))
        steps = []
        for i in range(n):
            steps.append({"step": i, "fwd_ms": _ms(self.ev_start[i], self.ev_precommit[i]), "commit_ms": _ms(self.ev_precommit[i], self.ev_end[i]),
                          "step_ms": _ms(self.ev_start[i], self.ev_end[i])})
        for i, t in enumerate(self.host_t):
            if i < n:
                steps[i].update(t)
        if self.acq is not None:
            for i, r in enumerate(self.acq.records):
                if i < n:
                    steps[i].update({"n_tokens": r.n_tokens, "copies": r.copies, "bytes": r.bytes, "leaves": r.layout.leaf_count,
                                     "hook_span_s": r.t_last_hook - r.t_first_hook})
        return {"steps": steps, "errors": (self.acq.errors if self.acq else [])}


# instance in our stream <- record (b_hooks) module / output index.  LlamaDecoderLayer returns (mlp.down_proj out, post_attention_layernorm residual);
# RMSNorm(model.norm) with residual returns (normalized, residual); logits_processor returns the logits.
_HOOK_MAP = {("model.norm", 0): "model.norm", ("model.norm", 1): "model.norm.residual", ("logits_processor", 0): "lm_head(logits_processor)",
             ("lm_head", 0): "lm_head(logits_processor)"}


def value_check(ref_c, streams, hooks_path: str, gpu: dict) -> dict:
    """Bit-level value correctness: sha256 of our acquired segment bytes == the accepted record's forward-hook digests (b_hooks, RTX 4090)."""
    import hashlib
    rec = json.load(open(hooks_path))
    per_step, compared, equal, unmatched, mism = [], 0, 0, 0, []
    for st in rec["steps"]:
        i = int(st["engine_step"])
        if i >= len(ref_c.steps):
            break
        L = ref_c.steps[i].layout
        by_inst = {sg.instance: sg for sg in L.segments}
        host = streams[i].cpu().numpy().tobytes()
        n_eq = n_cmp = 0
        for r in st["records"]:
            m = re.match(r"model\.layers\.(\d+)$", r["module"])
            if m:
                inst = f"model.layers.{m.group(1)}.mlp.down_proj" if r["index"] == 0 else f"model.layers.{m.group(1)}.post_attention_layernorm.residual"
            else:
                inst = _HOOK_MAP.get((r["module"], r["index"]))
            sg = by_inst.get(inst) if inst else None
            if sg is None or sg.nbytes != r["nbytes"]:
                unmatched += 1
                continue
            d = hashlib.sha256(host[sg.offset: sg.offset + sg.nbytes]).hexdigest()
            n_cmp += 1
            if d == r["sha256"]:
                n_eq += 1
            else:
                mism.append({"step": i, "module": r["module"], "index": r["index"], "instance": inst, "ours": d, "record": r["sha256"]})
        compared += n_cmp
        equal += n_eq
        per_step.append({"step": i, "compared": n_cmp, "equal": n_eq})
    return {"record_hooks": hooks_path, "record_schema": rec.get("schema"), "record_gpu": "see record versions.json (RTX 4090 profile)", "our_gpu": gpu["name"],
            "compared": compared, "equal": equal, "unmatched": unmatched, "per_step": per_step, "mismatches": mism[:50],
            "classes_checked": ["gemm1/out/gemmcoordinate_v1/f2fpbf16 (down_proj, lm_head)", "rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp (post_attention residual, model.norm.residual)",
                                "rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16 (model.norm)"]}


def snapshot_check(ref_c, streams, snapshots_path: str, gpu: dict) -> dict:
    """Content-addressed bit-level check for EVERY acquired segment: sha256(our segment bytes) is looked up in the accepted record's
    observer snapshot digests of the same step (a_observer/snapshots.json: sha256 of the storage bytes each observed view spans, all
    dispatch/Triton events).  A hit = the identical bytes were materialized in the record at this step.  Misses are expected for
    segments that are strided views in the record (e.g. rotary q/k slices of the qkv buffer: the snapshot spans the whole row range)."""
    import hashlib
    rec = json.load(open(snapshots_path))
    by_step: dict[int, set] = {}
    by_step_any: dict[int, set] = {}
    for e in rec["entries"]:
        st = int(e.get("step", -1))
        by_step_any.setdefault(st, set()).add((e["sha256"], int(e["nbytes"])))
        if str(e.get("path", "")).startswith("outs"):
            by_step.setdefault(st, set()).add((e["sha256"], int(e["nbytes"])))
    per_class: dict[str, dict] = {}
    per_step = []
    misses = []
    n_seg = n_hit = 0
    for i, stc in enumerate(ref_c.steps):
        L = stc.layout
        host = streams[i].cpu().numpy().tobytes()
        outs, anyset = by_step.get(i, set()), by_step_any.get(i, set())
        h_step = 0
        for sg in L.segments:
            d = hashlib.sha256(host[sg.offset: sg.offset + sg.nbytes]).hexdigest()
            key = (d, sg.nbytes)
            hit_out, hit_any = key in outs, key in anyset
            pc = per_class.setdefault(sg.class_id, {"segments": 0, "matched": 0, "matched_as_input_only": 0, "bytes": 0, "bytes_matched": 0})
            pc["segments"] += 1
            pc["bytes"] += sg.nbytes
            n_seg += 1
            if hit_out or hit_any:
                pc["matched"] += 1
                pc["bytes_matched"] += sg.nbytes
                n_hit += 1
                h_step += 1
                if not hit_out:
                    pc["matched_as_input_only"] += 1
            elif len(misses) < 400:
                misses.append({"step": i, "class_id": sg.class_id, "instance": sg.instance, "nbytes": sg.nbytes, "shape": list(sg.shape)})
        per_step.append({"step": i, "segments": len(L.segments), "matched": h_step, "record_out_digests": len(outs), "record_all_digests": len(anyset)})
    return {"record_snapshots": snapshots_path, "record_schema": rec.get("schema"), "record_layout": rec.get("layout"), "our_gpu": gpu["name"],
            "segments": n_seg, "segments_matched": n_hit, "per_class": per_class, "per_step": per_step, "misses_sample": misses,
            "note": "match = (sha256, nbytes) of our acquired segment present among the record's snapshot digests of the same engine step (outs or any path); "
                    "a miss does not mean a value mismatch when the record observed the tensor only as a strided view (different spanned bytes)"}


def sample_positions(positions: dict, k: int, rng: random.Random) -> list[tuple[int, str, str, int]]:
    out = []
    steps = positions["steps"]
    for s in steps:                                      # the sampled token of every step is always in the sample
        for seg in s["segments"]:
            if seg["class_id"] == A.C_TOK:
                out.append((s["step"], seg["class_id"], seg["instance"], 0))
    while len(out) < k:
        s = rng.choice(steps)
        seg = rng.choice(s["segments"])
        numel = 1
        for d in seg["shape"]:
            numel *= d
        out.append((s["step"], seg["class_id"], seg["instance"], rng.randrange(numel)))
    return out


def negatives(root: bytes, positions: dict, ops: list[C.Opening], streams: list[torch.Tensor], committer: C.Committer, pd: bytes) -> list[dict]:
    res = []

    def rec(name, op, expect_ok=False, **kw):
        ok, why = C.verify_opening(root, positions, op, **kw)
        res.append({"test": name, "accepted": ok, "reason": why, "expected_accept": expect_ok, "pass": ok == expect_ok})

    op = next((x for x in ops if x.class_id != A.C_TOK), ops[0])
    L = C.StepLayout.from_json(positions["steps"][op.step])
    k, o, ib = L.rank(op.class_id, op.instance, op.element)
    # altered value (leaf consistent) -> leaf hash changes -> step root mismatch
    leaf = bytearray(op.leaf)
    leaf[o] ^= 0x01
    rec("altered value bit (leaf + value consistent)", op.replace(leaf=bytes(leaf), value=bytes(leaf[o: o + ib])))
    # altered value only
    v = bytearray(op.value)
    v[0] ^= 0x80
    rec("altered value bytes (leaf unchanged)", op.replace(value=bytes(v)))
    # position binding: claim this value at another element of the same segment whose true bytes differ
    seg = L.segment(op.class_id, op.instance)
    for e2 in range(seg.numel):
        k2, o2, _ = L.rank(op.class_id, op.instance, e2)
        if k2 == k and op.leaf[o2: o2 + ib] != op.value:
            rec("this value claimed at another element (same leaf, true bytes differ)", op.replace(element=e2))
            break
    for e2 in range(seg.numel):
        k2, o2, _ = L.rank(op.class_id, op.instance, e2)
        if k2 != k:
            rec("this leaf/path claimed for an element of another leaf", op.replace(element=e2))
            break
    # another instance's leaf (same class, same step) presented for this instance
    other = next((x for x in ops if x.step == op.step and x.class_id == op.class_id and x.instance != op.instance), None)
    if other is not None:
        rec("leaf/path of another instance substituted", op.replace(leaf=other.leaf, path=other.path, leaf_index=other.leaf_index))
        rec("another instance's leaf bytes with this position's path", op.replace(leaf=other.leaf, value=other.leaf[o: o + ib]))
    # substituted from another step (same class/instance/element)
    other_s = next((x for x in ops if x.step != op.step and x.class_id == op.class_id and x.instance == op.instance and x.element == op.element), None)
    if other_s is not None:
        rec("leaf/paths of another step substituted (step index kept)", op.replace(leaf=other_s.leaf, value=other_s.value, path=other_s.path,
                                                                                   step_root=other_s.step_root, leaf_index=other_s.leaf_index))
    # wrong program digest
    rec("wrong Program digest", op, program_digest=hashlib.sha256(b"other program").digest())
    # tampered positions (a segment moved)
    import copy
    p2 = copy.deepcopy(positions)
    segs = p2["steps"][op.step]["segments"]
    if len(segs) > 1:
        segs[0], segs[1] = segs[1], segs[0]
        ok, why = C.verify_opening(root, p2, op)
        res.append({"test": "tampered positions document (two segments swapped)", "accepted": ok, "reason": why, "expected_accept": False, "pass": not ok})
    # swapped step roots in the top tree
    if len(ops) > 1 and ops[-1].step != op.step:
        rec("top path of another step", op.replace(top_path=ops[-1].top_path))
    # binding: recommit with one byte of one stream flipped -> a different root (and the original opening still verifies against the original)
    if streams:
        s0 = streams[0].clone()
        s0[1000 % s0.numel()] ^= 0x01
        c2 = C.Committer(pd, committer.ctx, backend="host")
        for i, st in enumerate(committer.steps):
            src = s0 if i == 0 else streams[i]
            c2.commit_step(C.StepLayout.from_json(st.layout.to_json()), src.cpu().numpy().tobytes()[: st.layout.stream_bytes])
        r2, _ = c2.finish()
        res.append({"test": "binding: one byte flipped in step 0 stream -> different root", "accepted": r2 == root, "reason": f"root2={r2.hex()[:16]}",
                    "expected_accept": False, "pass": r2 != root})
        # reordered leaves: swap two 64-byte leaves of step 0 -> different root
        s1 = streams[0].clone()
        a, b = s1[64:128].clone(), s1[128:192].clone()
        s1[64:128], s1[128:192] = b, a
        c3 = C.Committer(pd, committer.ctx, backend="host")
        for i, st in enumerate(committer.steps):
            src = s1 if i == 0 else streams[i]
            c3.commit_step(C.StepLayout.from_json(st.layout.to_json()), src.cpu().numpy().tobytes()[: st.layout.stream_bytes])
        r3, _ = c3.finish()
        res.append({"test": "position binding: two leaves of step 0 swapped -> different root", "accepted": r3 == root, "reason": f"root3={r3.hex()[:16]}",
                    "expected_accept": False, "pass": r3 != root})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--program-digest", default=PROGRAM_DIGEST)
    ap.add_argument("--record", default="B0-cov_b0_b1_c256")
    ap.add_argument("--record-snapshots", default=None, help="record's nonint/a_observer/snapshots.json: look up sha256 of every acquired segment among the record's per-step snapshot digests")
    ap.add_argument("--record-hooks", default=None, help="record's nonint/b_hooks/hooks.json: compare sha256 of our acquired segments with the record's digests")
    ap.add_argument("--profile-id", default="vllm_d9105ea80_sm89_eager")
    ap.add_argument("--buffer-mb", type=int, default=128)
    ap.add_argument("--n-open", type=int, default=64)
    ap.add_argument("--manifest", default="out/gen/r9/cmt-ref/interface_manifest.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd = bytes.fromhex(args.program_digest)
    wl = va.load_workload(args.workload)
    eng = va.build_engine(args.checkpoints, max_num_seqs=args.max_num_seqs, case=va.case_of(wl))
    cap = torch.cuda.get_device_capability(0)
    gpu = {"name": torch.cuda.get_device_name(0), "capability": list(cap), "sm": f"sm{cap[0]}{cap[1]}", "torch": torch.__version__,
           "cuda": torch.version.cuda, "total_mem_bytes": torch.cuda.get_device_properties(0).total_memory}
    print(f"[cmt] engine built in {eng.build_s:.1f}s on {gpu['name']} (case {eng.case}, {eng.cp['repo']})", flush=True)
    (out / "versions.json").write_text(json.dumps({**va.versions_doc(eng), "gpu": gpu}, indent=1, default=str) + "\n")
    req = wl["requests"][0]
    ctx = C.Context(args.profile_id, req["request_id"], len(req["prompt_token_ids"]), eng.cp["repo"],
                    {"record": args.record, "workload": Path(args.workload).name})

    for i in range(2):
        _, tw = va.run_requests(eng, va.throwaway_requests(eng, wl["requests"], seed=100 + i), wl["sampling"])
        print(f"[cmt] warm-up {i}: {tw['wall_s']:.3f}s, {tw['engine_steps']} steps", flush=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    buffer_bytes = args.buffer_mb << 20
    raw = open(out / "raw_samples.jsonl", "w")
    tokens_ref = None
    roots: dict[str, list[str]] = {a: [] for a in arms}
    reference: dict | None = None
    mem_base = torch.cuda.memory_allocated()
    # torch.compile warm-up outside the timed reps (compile time is reported separately)
    compile_s = None
    if "torch-compiled" in arms:
        t0 = time.perf_counter()
        try:
            from verity_capture.experimental.commit import torch_sha256 as TS
            for n in (1, 2, 5, 1000, 12000):
                x = torch.randint(0, 256, (n, 64), dtype=torch.uint8, device="cuda")
                lv = TS.fold_levels(TS.pos_leaf_digests(x, compiled=True), compiled=True)
                torch.cuda.synchronize()
                ref = TS.fold_levels(TS.pos_leaf_digests(x, compiled=False), compiled=False)
                assert torch.equal(lv[-1], ref[-1]), "compiled root != eager root"
            compile_s = time.perf_counter() - t0
            print(f"[cmt] torch.compile warm-up {compile_s:.1f}s (compiled roots == eager roots)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[cmt] torch.compile FAILED after {time.perf_counter() - t0:.1f}s: {e!r} -- arm torch-compiled dropped", flush=True)
            arms = [a for a in arms if a != "torch-compiled"]
            compile_s = {"failed": repr(e)[:500]}
    for rep in range(args.reps):
        for arm in arms:
            keep = rep == 0 and arm == next((a for a in ("triton-hybrid", "triton", "torch-hybrid", "torch", "host") if a in arms), None)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            run = Run(eng, ARMS[arm], ctx=ctx, program_digest=pd, buffer_bytes=buffer_bytes, keep_streams=keep)
            m1 = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            toks, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
            wall = time.perf_counter() - t0
            root_hex = None
            fin_s = None
            positions = None
            if run.committer is not None:
                tf = time.perf_counter()
                root, positions = run.committer.finish()
                fin_s = time.perf_counter() - tf
                root_hex = root.hex()
                roots[arm].append(root_hex)
            stats = run.close()
            peak = torch.cuda.max_memory_allocated()
            gen = next(iter(toks.values()))
            if tokens_ref is None:
                tokens_ref = gen
            steps = stats["steps"]
            sample = {"arm": arm, "rep": rep, "gpu": gpu["name"], "wall_s": wall, "engine_wall_s": timing.get("wall_s"), "engine_steps": timing.get("engine_steps"),
                      "observed_steps": len(steps), "tokens_equal_control": gen == tokens_ref, "root": root_hex, "finish_s": fin_s,
                      "mem_persistent_bytes": m1 - m0, "mem_peak_extra_bytes": peak - m1, "mem_base_bytes": mem_base,
                      "hash_calls": run.committer.hash_calls if run.committer else None, "hash_blocks": run.committer.hash_blocks if run.committer else None,
                      "retained": run.committer.persistent_bytes() if run.committer else None,
                      "step_ms_mean": statistics.fmean(s["step_ms"] for s in steps) if steps else None,
                      "fwd_ms_mean": statistics.fmean(s["fwd_ms"] for s in steps) if steps else None,
                      "commit_ms_mean": statistics.fmean(s["commit_ms"] for s in steps) if steps else None,
                      "prefill_step_ms": steps[0]["step_ms"] if steps else None,
                      "decode_step_ms_mean": statistics.fmean(s["step_ms"] for s in steps[1:]) if len(steps) > 1 else None,
                      "bytes_total": sum(s.get("bytes", 0) for s in steps), "copies_total": sum(s.get("copies", 0) for s in steps),
                      "steps": steps, "errors": stats["errors"][:5]}
            raw.write(json.dumps(sample) + "\n")
            raw.flush()
            print(f"[cmt] rep{rep} {arm:14s} wall {wall:.3f}s steps {len(steps)} step-mean {sample['step_ms_mean']:.2f}ms "
                  f"(prefill {sample['prefill_step_ms']:.2f} decode {sample['decode_step_ms_mean']:.2f}) commit-mean {sample['commit_ms_mean']:.2f}ms "
                  f"bytes {sample['bytes_total']} root {root_hex[:16] if root_hex else '-'} tok_eq {gen == tokens_ref} errs {len(stats['errors'])}", flush=True)
            if keep and run.committer is not None:
                reference = {"arm": arm, "rep": rep, "root": root, "positions": positions, "streams": run.streams, "committer": run.committer}
            elif run.committer is not None:
                run.committer = None
            del run
            torch.cuda.empty_cache()
    raw.close()

    # ---- validity on the reference run --------------------------------------------------------------------------------------
    validity: dict = {"roots_by_arm": roots}
    if reference is not None:
        ref_c, root, positions, streams = reference["committer"], reference["root"], reference["positions"], reference["streams"]
        (out / "positions.json").write_text(json.dumps(positions, indent=None, separators=(",", ":")) + "\n")
        (out / "root.json").write_text(json.dumps({"root": root.hex(), "program_digest": pd.hex(), "context": positions["context"], "arm": reference["arm"],
                                                   "gpu": gpu["name"]}, indent=1) + "\n")
        # cross-backend: host recommit of the same streams must reproduce the root
        ch = C.Committer(pd, ctx, backend="host")
        t0 = time.perf_counter()
        for i, st in enumerate(ref_c.steps):
            ch.commit_step(C.StepLayout.from_json(st.layout.to_json()), streams[i].cpu().numpy().tobytes()[: st.layout.stream_bytes])
        rh, ph = ch.finish()
        validity["host_recommit_root_equal"] = rh == root
        validity["host_recommit_s"] = time.perf_counter() - t0
        validity["positions_equal"] = ph["steps"] == positions["steps"]
        # openings
        rng = random.Random(7)
        sample_pos = sample_positions(positions, args.n_open, rng)
        ops, otimes, ver = [], [], []
        for (s, cid, inst, el) in sample_pos:
            t0 = time.perf_counter()
            op = C.open_value(ref_c, s, cid, inst, el, stream=streams[s])
            otimes.append(time.perf_counter() - t0)
            ops.append(op)
            t1 = time.perf_counter()
            ok, why = C.verify_opening(root, positions, op)
            ver.append({"ok": ok, "why": why, "verify_s": time.perf_counter() - t1, "production_leaf_check": C.engine.production_leaf_check(op)})
        validity["openings"] = {"n": len(ops), "all_verify": all(v["ok"] for v in ver), "all_production_leaf_check": all(v["production_leaf_check"] for v in ver),
                                "open_s": otimes, "verify_s": [v["verify_s"] for v in ver],
                                "open_ms_mean": statistics.fmean(otimes) * 1e3, "verify_ms_mean": statistics.fmean(v["verify_s"] for v in ver) * 1e3,
                                "path_len_step_mean": statistics.fmean(len(o.path) for o in ops), "path_len_top": len(ops[0].top_path),
                                "opening_bytes_mean": statistics.fmean(len(json.dumps(o.to_json())) for o in ops)}
        (out / "openings_sample.json").write_text(json.dumps({"root": root.hex(), "openings": [o.to_json() for o in ops], "verify": ver}, indent=None) + "\n")
        # the sampled tokens opened must equal the generated tokens
        tok_ops = [o for o in ops if o.class_id == A.C_TOK]
        opened_tokens = [struct.unpack("<i", o.value)[0] for o in sorted(tok_ops, key=lambda o: o.step)]
        validity["opened_tokens_equal_generated"] = opened_tokens == list(tokens_ref)[: len(opened_tokens)]
        validity["opened_tokens"] = opened_tokens[:20]
        validity["generated_tokens"] = list(tokens_ref)[:20]
        # negatives
        validity["negatives"] = negatives(root, positions, ops, streams, ref_c, pd)
        validity["negatives_all_pass"] = all(n["pass"] for n in validity["negatives"])
        # bytes bound per class vs manifest
        per_class: dict[str, int] = {}
        for st in ref_c.steps:
            for k, v in st.layout.class_bytes().items():
                per_class[k] = per_class.get(k, 0) + v
        binding = {"per_class_bytes_bound": per_class, "total_bytes_bound": sum(per_class.values()),
                   "leaves_total": sum(st.leaf_count for st in ref_c.steps), "steps": len(ref_c.steps),
                   "stream_bytes_per_step": [st.layout.stream_bytes for st in ref_c.steps]}
        mpath = Path(args.manifest)
        if mpath.exists():
            man = json.load(open(mpath))
            rows = []
            for c in man["classes"]:
                if c["acquisition"] != "naturally-materialized":
                    continue
                cid = c["id"]
                bound = per_class.get(cid)
                if cid in A.ROPE_MANIFEST_IDS:
                    bound = per_class.get(A.C_ROPE, 0) / 2            # one rotated tensor = both output primitives, by position
                rows.append({"id": cid, "manifest_logical_bytes": c["logical_bytes"], "bound_bytes": bound,
                             "ratio": None if not bound else bound / c["logical_bytes"]})
            binding["vs_manifest"] = rows
        validity["binding"] = binding
        if args.record_snapshots:
            validity["snapshot_check"] = snapshot_check(ref_c, streams, args.record_snapshots, gpu)
            (out / "snapshot_check.json").write_text(json.dumps(validity["snapshot_check"], indent=1) + "\n")
            sc = validity["snapshot_check"]
            print(f"[cmt] SNAPSHOT CHECK vs record observer digests: {sc['segments_matched']}/{sc['segments']} segments found bit-identical in the record's per-step output snapshots; per class: "
                  + ", ".join(f"{k.split('/')[0]}={v['matched']}/{v['segments']}" for k, v in sc['per_class'].items()), flush=True)
        if args.record_hooks:
            validity["value_check"] = value_check(ref_c, streams, args.record_hooks, gpu)
            (out / "value_check.json").write_text(json.dumps(validity["value_check"], indent=1) + "\n")
            vc = validity["value_check"]
            print(f"[cmt] VALUE CHECK vs record hooks digests: {vc['equal']}/{vc['compared']} equal ({vc['unmatched']} records without a segment)", flush=True)
        (out / "binding.json").write_text(json.dumps(binding, indent=1) + "\n")
    validity["compile_warmup_s"] = compile_s
    (out / "validity.json").write_text(json.dumps(validity, indent=1, default=str) + "\n")

    # ---- summary table ------------------------------------------------------------------------------------------------------------
    samples = [json.loads(l) for l in open(out / "raw_samples.jsonl")]
    by_arm: dict[str, list[dict]] = {}
    for s in samples:
        by_arm.setdefault(s["arm"], []).append(s)
    base = by_arm.get("control")
    lines = ["| arm | reps | run wall (s) mean [min..max] | step GPU mean (ms) | prefill step (ms) | decode step mean (ms) | commit GPU mean (ms) | persistent mem (MB) | peak extra (MB) | root stable | tokens = control |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, ss in by_arm.items():
        w = [x["wall_s"] for x in ss]
        line = (f"| {arm} | {len(ss)} | {statistics.fmean(w):.3f} [{min(w):.3f}..{max(w):.3f}] | {statistics.fmean(x['step_ms_mean'] for x in ss):.2f} | "
                f"{statistics.fmean(x['prefill_step_ms'] for x in ss):.2f} | {statistics.fmean(x['decode_step_ms_mean'] for x in ss):.2f} | "
                f"{statistics.fmean(x['commit_ms_mean'] for x in ss):.2f} | {statistics.fmean(x['mem_persistent_bytes'] for x in ss) / 2**20:.1f} | "
                f"{statistics.fmean(x['mem_peak_extra_bytes'] for x in ss) / 2**20:.1f} | {len(set(x['root'] for x in ss)) == 1 if ss[0]['root'] else '-'} | "
                f"{all(x['tokens_equal_control'] for x in ss)} |")
        lines.append(line)
    table = "\n".join(lines)
    print(table, flush=True)
    doc = {"schema": "verity-gen/cmt-ref/bench/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "argv": sys.argv,
           "arms": {a: ARMS[a] for a in arms}, "reps": args.reps, "table_md": table, "validity": {k: v for k, v in validity.items() if k != "openings"},
           "openings_summary": {k: v for k, v in validity.get("openings", {}).items() if k not in ("open_s", "verify_s")},
           "control_wall_mean_s": statistics.fmean(x["wall_s"] for x in base) if base else None}
    (out / "summary.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(f"# cmt-ref bench — {gpu['name']} — {doc['utc']}\n\n{table}\n\nvalidity: {json.dumps(doc['validity'], default=str)[:3000]}\n")


if __name__ == "__main__":
    main()
