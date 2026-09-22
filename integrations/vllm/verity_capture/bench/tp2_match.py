"""R13 lane tp, P4 step 3 — TP2 Match at the collective level, through the per-rank `worker_extension_cls` seam.

    ... tp2_match --arm capture --workload <manifest> --case LLAMA32_1B --tp 2 --out <run>/capture [--engine-arg k=v ...]
    ... tp2_match --arm control --workload <manifest> --case LLAMA32_1B --tp 2 --out <run>/control
    ... tp2_match --arm check --out <run> --build-rank0 <row>/build_step_r0 --build-rank1 <row>/build_step_r1

`capture`: the pinned BI engine at `--tp 2` with `tp2_worker.EXTENSION` in every rank; after the uninstrumented throwaway
warm-up, `tp2_coll_install` records EVERY collective of every rank in call order (step, k, kind, module, partial + received
output, sha256s; the tensors of the first `--store-steps` steps are saved) and the driver writes `tokens.json`.
`control`: the identical engine without the extension (`tokens.json` only).
`check` (CPU, offline): the three parts of the TP-aware Match this lane can execute on the collective skeleton of the run —
  1. STRUCTURE: per step and per rank, the sequence (kind, N) of the collectives equals the peer-input signature of THAT rank's
     Build (`tp_links.json` of `derive_step --tp 2 --rank r`), and the rows of every collective equal the step's token count;
  2. CROSS-RANK PROVENANCE: for every stored collective k of every step, the value each rank RECEIVED equals the registered
     Definition of the two ranks' COMMITTED partials — `AllReduce2_v1{N}` (`bf16(f32(part_rank0) + f32(part_rank1))` per element) /
     `AllGather2_v1{N}` (`[part_rank0 | part_rank1]`) — bit-exactly, and both ranks received the same value;
  3. TOKENS: the instrumented run's tokens equal the control's (non-interference of the seam), per request.
It is NOT the instance-sequence Match of `row_pod.sh` (the fold of a per-rank log against the rank Program: the fold needs a
peer-input operand class and a sharded weights Record — see tp-STATUS); a PASS here is intermediate evidence, not a green row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _engine_main(args) -> None:
    from verity_vllm_adapter import profile as prof

    prof.apply_env()
    from vllm import LLM

    from verity_capture import vllm_adapter as va
    from verity_capture.bench.tp2_capture import parse_engine_args
    from verity_capture.bench.tp2_worker import EXTENSION

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload, None)
    case = va.case_of(wl, args.case)
    env_flags = prof.apply_env()
    cp = prof.checkpoint_for(case, args.checkpoints)
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True, tensor_parallel_size=args.tp)
    kw["enable_prefix_caching"] = False
    kw["max_num_seqs"] = int(args.max_num_seqs)
    kw.update(parse_engine_args(args.engine_arg))
    instrumented = args.arm == "capture"
    if instrumented:
        kw["worker_extension_cls"] = EXTENSION
    print(f"[tp2match:{args.arm}] tp={args.tp} case={case} kwargs={ {k: v for k, v in kw.items() if k not in ('model', 'tokenizer')} }", flush=True)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case)
    topo = llm.collective_rpc("tp2_probe", timeout=args.rpc_timeout) if instrumented else None
    (out / "topology.json").write_text(json.dumps({"arm": args.arm, "tp": args.tp, "build_s": round(build_s, 1),
                                                    "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
                                                    "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "VLLM_", "CUBLAS"))},
                                                    "ranks": topo}, indent=1, default=str) + "\n")
    if not args.no_warmup:
        warm = va.throwaway_requests(eng, wl["requests"])
        _, tw = va.run_requests(eng, warm, wl["sampling"])
        print(f"[tp2match:{args.arm}] warm-up {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)
    if instrumented:
        inst = llm.collective_rpc("tp2_coll_install", timeout=args.rpc_timeout, kwargs={"out_dir": str(out), "store_steps": int(args.store_steps)})
        print(f"[tp2match:capture] collective recorder installed: {inst}", flush=True)
    t0 = time.perf_counter()
    tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
    run_s = time.perf_counter() - t0
    closed = llm.collective_rpc("tp2_coll_close", timeout=args.rpc_timeout) if instrumented else None
    doc = {"arm": args.arm, "tp": args.tp, "case": case, "model": cp["repo"], "run_s": round(run_s, 3), "timing": timing, "argv": sys.argv,
           "recorder": closed,
           "requests": [{"request_id": r["request_id"], "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"],
                         "generated_token_ids": tokens.get(r["request_id"], [])} for r in wl["requests"]]}
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp2match:{args.arm}] run {run_s:.2f}s {timing.get('engine_steps')} steps; recorder={closed}", flush=True)


def _check_main(args) -> int:
    try:
        import torch
    except ImportError:            # laptop re-check of the structure criterion only; provenance needs the stored .pt steps + torch
        torch = None

    run = Path(args.out)
    cap, ctl = run / "capture", run / "control"
    res: dict = {"schema": "tp2-match/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": str(run),
                 "campaign": os.environ.get("VERITY_CAMPAIGN", "r15"),       # [R15] was the hard-coded 'r13' label (R14 row note)
                 "what": "collective-level TP-aware Match (structure + cross-rank provenance + tokens); NOT the instance-sequence Match"}
    recs = {}
    for r in (0, 1):
        p = cap / f"coll.rank{r}.jsonl"
        if not p.exists():
            res["fail"] = f"missing {p}"
            _write(run, res)
            return 1
        recs[r] = [json.loads(l) for l in p.open() if l.strip()]
    # [R17 tp D90, rev F-r17-03] the recorder files this check read, by digest: the provenance record below is bound to THESE rows (and
    # rank_match refuses a provenance record whose recorder digests are not those of the rows it loads itself)
    recorder_sha = {str(r): hashlib.sha256((cap / f"coll.rank{r}.jsonl").read_bytes()).hexdigest() for r in (0, 1)}
    steps = sorted({x["step"] for x in recs[0]} | {x["step"] for x in recs[1]})
    tok = json.load(open(cap / "tokens.json"))
    n_tok_step = {}   # tokens per step: prefill = sum of prompt lens (one request at a time is the only admitted shape here), decode = #live requests
    prompts = [len(q["prompt_token_ids"]) for q in tok["requests"]]
    gens = [len(q["generated_token_ids"]) for q in tok["requests"]]
    # ---- 1. structure vs each rank's Build peer-input signature
    struct = {"pass": True, "per_rank": {}}
    for r, bdir in ((0, args.build_rank0), (1, args.build_rank1)):
        links = json.load(open(Path(bdir) / "tp_links.json"))
        sig = [(l["kind"], int(l["shape"][-1])) for l in links["links"]]
        # [R15 rev-pm M-0398] site identity per k: the link's module_path ('inner.model.<path> (<Class>)') must equal the record's
        # hooked module ('<path>'), so 96 width-H all-reduces are checked as the Program's SITES in the Program's ORDER, not as a
        # (kind, width) sequence that any interleaving / doubled hook / order swap would also produce.
        sig_sites = [_link_site(l) for l in links["links"]]
        assert links["rank"] == r and links["world"] == 2, links
        per_step = {}
        bad = []
        bad_sites = []
        for s in steps:
            xs = [x for x in recs[r] if x["step"] == s]
            got = [(x["kind"], int(x["shape"][-1])) for x in xs]
            # rows: the hidden-stream all-reduces carry the step's TOKENS (prefill = prompt tokens), the logits all-gather the step's SAMPLED rows
            rows_ar = sorted({int(x["shape"][0]) for x in xs if x["kind"] == "all_reduce" and len(x["shape"]) == 2})
            rows_ag = sorted({int(x["shape"][0]) for x in xs if x["kind"] == "all_gather" and len(x["shape"]) == 2})
            ok = got == sig
            got_sites = [x.get("module") for x in xs]
            # [rev-pm M-0470] fail CLOSED by absence: a link without module_path or a record without a module is a site mismatch at that
            # k (None == None never counts as equal), and the step passes only when every k is a NAMED equal pair.
            named_equal = sum(1 for g, e in zip(got_sites, sig_sites) if g is not None and e is not None and g == e)
            ok_sites = (len(got_sites) == len(sig_sites)) and named_equal == len(sig_sites)
            per_step[s] = {"n": len(xs), "rows_all_reduce": rows_ar, "rows_all_gather": rows_ag, "equal_signature": ok, "equal_sites": ok_sites,
                           "sites_named_equal": named_equal}
            if not ok:
                bad.append(s)
            if not ok_sites:
                bad_sites.append({"step": s, "sites_named_equal": named_equal, "first_mismatch": next(
                    ({"k": k, "record": g, "link": e} for k, (g, e) in enumerate(zip(got_sites, sig_sites)) if g is None or e is None or g != e),
                    {"k": min(len(got_sites), len(sig_sites)), "record": None, "link": None, "note": "length mismatch"})})
        struct["per_rank"][r] = {"build_signature_len": len(sig), "build_digest": links["program_digest"], "n_steps": len(steps),
                                 "n_collectives": len(recs[r]), "steps_mismatching": bad,
                                 "sites_checked_per_step": len(sig_sites), "steps_site_mismatching": bad_sites[:5], "n_steps_site_mismatching": len(bad_sites),
                                 "site_rule": "record.module == link.module_path minus 'inner.model.' prefix and ' (Class)' suffix, per k; "
                                              "a None on either side is a mismatch (fail closed by absence, rev-pm M-0470)",
                                 "links_without_module_path": sum(1 for e in sig_sites if e is None),
                                 "prefill": per_step[steps[0]] if steps else None,
                                 "decode_rows_all_reduce": sorted({tuple(per_step[s]["rows_all_reduce"]) for s in steps[1:]}) if len(steps) > 1 else None,
                                 "decode_rows_all_gather": sorted({tuple(per_step[s]["rows_all_gather"]) for s in steps[1:]}) if len(steps) > 1 else None,
                                 "by_kind": _count(x["kind"] for x in recs[r]), "modules": _count(f'{x["module_class"]}' for x in recs[r])}
        struct["pass"] &= not bad and not bad_sites and len(recs[r]) == len(sig) * len(steps)
    # token count per step (B=1, arrivals none: one prefill of all prompt tokens, then one row per decode step)
    if steps:
        pre = struct["per_rank"][0]["prefill"]
        struct["prefill_all_reduce_rows_equal_prompt_tokens"] = (pre["rows_all_reduce"] == [sum(prompts)])
        struct["prefill_all_gather_rows_equal_requests"] = (pre["rows_all_gather"] == [len(prompts)])
        struct["pass"] &= bool(struct["prefill_all_reduce_rows_equal_prompt_tokens"] and struct["prefill_all_gather_rows_equal_requests"])
    res["structure"] = struct
    # ---- 2. cross-rank provenance on the stored steps
    prov = {"pass": True, "steps_checked": 0, "collectives_checked": 0, "elements_checked": 0, "all_reduce_checked": 0, "all_gather_checked": 0,
            "mismatches": [], "received_equal_both_ranks": 0,
            # [rev F-r17-03] the stored tensors are BOUND to the recorder rows: sha256 of every .pt part / out (the recorder's own bytes rule:
            # contiguous, bf16 viewed as uint8) must equal the row's part_sha256 / out_sha256 at the same (rank, step, k); every mismatch or
            # missing row is a named FAIL; the digests of the recorder files themselves travel with the record
            "recorder_sha256": recorder_sha, "tensors_digested": 0, "digest_mismatches": [], "n_digest_mismatches": 0,
            "digest_rule": "sha256(tensor.contiguous() viewed uint8 for bf16, else raw) == coll.rank<r>.jsonl row part_sha256 / out_sha256 at (rank, step, k)"}

    def _sha(t) -> str:
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.contiguous().numpy().tobytes()).hexdigest()

    def _digest_fail(rank: int, s: int, k: int, what: str, why: str) -> None:
        prov["pass"] = False
        prov["n_digest_mismatches"] += 1
        if len(prov["digest_mismatches"]) < 20:
            prov["digest_mismatches"].append({"rank": rank, "step": s, "k": k, "tensor": what, "why": why})

    for s in steps:
        p0, p1 = cap / f"coll.rank0.step{s}.pt", cap / f"coll.rank1.step{s}.pt"
        if not (p0.exists() and p1.exists()):
            continue
        if torch is None:
            raise RuntimeError("stored provenance steps present but torch is not importable")
        v0, v1 = torch.load(p0), torch.load(p1)
        x0 = [x for x in recs[0] if x["step"] == s]
        x1 = [x for x in recs[1] if x["step"] == s]
        n = min(len(v0["part"]), len(v1["part"]))
        if len(v0["part"]) != len(x0) or len(v1["part"]) != len(x1) or len(v0["out"]) != len(x0) or len(v1["out"]) != len(x1):
            _digest_fail(-1, s, -1, "count", f"stored tensors rank0 {len(v0['part'])}/{len(v0['out'])} rank1 {len(v1['part'])}/{len(v1['out'])} vs recorder rows {len(x0)} / {len(x1)}")
        for k in range(n):
            a, b = v0["part"][k], v1["part"][k]
            o0, o1 = v0["out"][k], v1["out"][k]
            if k >= len(x0) or k >= len(x1):
                _digest_fail(-1, s, k, "row", "no recorder row for the stored tensor")
                continue
            for rank, (xr, part, out) in ((0, (x0[k], a, o0)), (1, (x1[k], b, o1))):
                for what, t, key in (("part", part, "part_sha256"), ("out", out, "out_sha256")):
                    want = xr.get(key)
                    got = _sha(t)
                    prov["tensors_digested"] += 1
                    if not want or got != want:
                        _digest_fail(rank, s, k, what, f"stored .pt {what} sha256 {got[:12]} != recorder row {str(want)[:12]} (the arithmetic below runs on bytes the record does not describe)")
                if int(xr.get("k", k)) != k or int(xr.get("step", s)) != s:
                    _digest_fail(rank, s, k, "key", f"recorder row keyed (step {xr.get('step')}, k {xr.get('k')}) at position {k} of step {s}")
            kind = x0[k]["kind"]
            if kind == "all_reduce":
                exp = (a.float() + b.float()).to(torch.bfloat16)
                prov["all_reduce_checked"] += 1
            elif kind == "all_gather":
                exp = torch.cat([a, b], dim=-1)
                prov["all_gather_checked"] += 1
            else:
                prov["mismatches"].append({"step": s, "k": k, "kind": kind, "why": "unbound collective kind"})
                prov["pass"] = False
                continue
            same01 = bool(torch.equal(o0, o1))
            eq0 = bool(torch.equal(o0, exp))
            eq1 = bool(torch.equal(o1, exp))
            prov["received_equal_both_ranks"] += int(same01)
            prov["collectives_checked"] += 1
            prov["elements_checked"] += int(exp.numel())
            if not (same01 and eq0 and eq1):
                d0 = int((o0.view(torch.int16) != exp.view(torch.int16)).sum()) if o0.shape == exp.shape else -1
                prov["mismatches"].append({"step": s, "k": k, "kind": kind, "module": x0[k]["module"], "received_equal_both_ranks": same01,
                                           "rank0_received_equals_definition": eq0, "rank1_received_equals_definition": eq1, "n_diff_rank0": d0})
                prov["pass"] = False
        prov["steps_checked"] += 1
    prov["n_mismatches"] = len(prov["mismatches"])
    prov["mismatches"] = prov["mismatches"][:20]
    res["provenance"] = prov
    # ---- 3. tokens: instrumented == control, per request
    tk = {"pass": None}
    if (ctl / "tokens.json").exists():
        c = json.load(open(ctl / "tokens.json"))
        A = [(q["request_id"], q["prompt_token_ids"], q["generated_token_ids"]) for q in tok["requests"]]
        B = [(q["request_id"], q["prompt_token_ids"], q["generated_token_ids"]) for q in c["requests"]]
        tk = {"pass": A == B and len(A) > 0, "n_requests": len(A), "generated_lengths": gens,
              "capture_run_s": tok.get("run_s"), "control_run_s": c.get("run_s")}
    else:
        tk["note"] = "no control arm"
    res["tokens"] = tk
    res["recorder"] = tok.get("recorder")
    res["pass"] = bool(struct["pass"] and prov["pass"] and prov["collectives_checked"] > 0 and tk.get("pass") is not False)
    _write(run, res)
    print(json.dumps({k: res[k] for k in ("pass", "structure", "provenance", "tokens")}, indent=1, default=str)[:6000])
    return 0 if res["pass"] else 1


def _link_site(link: dict) -> str | None:
    """Module path a tp_links link names, in the recorder's spelling: 'inner.model.model.layers.0.mlp (X)' -> 'model.layers.0.mlp'."""
    mp = link.get("module_path")
    if not isinstance(mp, str):
        return None
    mp = mp.split(" (", 1)[0]
    return mp[len("inner.model."):] if mp.startswith("inner.model.") else mp


def _count(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _write(run: Path, res: dict) -> None:
    (run / "tp2_match.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("capture", "control", "check"), required=True)
    ap.add_argument("--workload", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--store-steps", type=int, default=1 << 30, help="save the partial/received tensors of steps < N (all by default)")
    ap.add_argument("--rpc-timeout", type=float, default=1800.0)
    ap.add_argument("--build-rank0", default=None)
    ap.add_argument("--build-rank1", default=None)
    args = ap.parse_args()
    if args.arm == "check":
        return _check_main(args)
    _engine_main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
