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
     `AllGather2_v1{N}` (`[part_rank0 | part_rank1]`) — bit-exactly, and both ranks received the same value; a
     `fused_allreduce_rms_norm` call on its fused path, the new residual (`collectives.fused_norm.expected_residual`);
  3. TOKENS: the instrumented run's tokens equal the control's (non-interference of the seam), per request.
It is NOT the instance-sequence Match of `row_pod.sh` (the fold of a per-rank log against the rank Program: the fold needs a
peer-input operand class and a sharded weights Record — see tp-STATUS); a PASS here is intermediate evidence, not a green row.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.collectives import fused_norm as FN
from verity_vllm.config import option
from verity_vllm.pipeline.build import read_tp_links, same_module_path


def _engine_main(args) -> None:
    from verity_vllm.engine import engine_profile as prof

    prof.apply_env()
    from vllm import LLM

    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.tp.capture import parse_engine_args
    from verity_vllm.engine.rank_worker import EXTENSION

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
    # [lane vllm-tp-n] the world is the number of rank Builds given (--build-rank in rank order; --build-rank0/1 are the two-rank spelling)
    builds = _build_dirs(args)
    world = len(builds)
    ranks = list(range(world))
    res["world"] = world
    recs = {}
    for r in ranks:
        p = cap / f"coll.rank{r}.jsonl"
        if not p.exists():
            res["fail"] = f"missing {p}"
            _write(run, res)
            return 1
        recs[r] = [json.loads(l) for l in p.open() if l.strip()]
    # [R17 tp D90, rev F-r17-03] the recorder files this check read, by digest: the provenance record below is bound to THESE rows (and
    # rank_match refuses a provenance record whose recorder digests are not those of the rows it loads itself)
    recorder_sha = {str(r): hashlib.sha256((cap / f"coll.rank{r}.jsonl").read_bytes()).hexdigest() for r in ranks}
    steps = sorted({x["step"] for r in ranks for x in recs[r]})
    tok = json.load(open(cap / "tokens.json"))
    n_tok_step = {}   # tokens per step: prefill = sum of prompt lens (one request at a time is the only admitted shape here), decode = #live requests
    prompts = [len(q["prompt_token_ids"]) for q in tok["requests"]]
    gens = [len(q["generated_token_ids"]) for q in tok["requests"]]
    # ---- 1. structure vs each rank's Build peer-input signature
    struct = {"pass": True, "per_rank": {}}
    for r, bdir in enumerate(builds):
        links = read_tp_links(str(Path(bdir) / "tp_links.json"))
        sig = [(l["kind"], int(l["shape"][-1])) for l in links["links"]]
        # [R15 rev-pm M-0398] site identity per k: the link's module_path ('inner.model.<path> (<Class>)') must equal the record's
        # hooked module ('<path>'), so 96 width-H all-reduces are checked as the Program's SITES in the Program's ORDER, not as a
        # (kind, width) sequence that any interleaving / doubled hook / order swap would also produce.
        sig_sites = [l.get("module") for l in links["links"]]     # [lane vllm-tp-n] recorder vocabulary, from the recorded model_root (read_tp_links)
        sites_exact = bool(links.get("model_root"))              # a v1 file has no recorded root: dotted-suffix identity (same_module_path)
        if int(links["rank"]) != r or int(links["world"]) != world:
            raise SystemExit(f"tp_links of build {bdir}: rank {links['rank']} world {links['world']} != rank {r} of world {world}")
        if any(len(l["peers"]) != world - 1 for l in links["links"]):
            raise SystemExit(f"tp_links of rank {r}: a link does not name exactly world-1 = {world - 1} peers")
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
            named_equal = sum(1 for g, e in zip(got_sites, sig_sites) if g is not None and e is not None and (g == e if sites_exact else same_module_path(g, e)))
            ok_sites = (len(got_sites) == len(sig_sites)) and named_equal == len(sig_sites)
            per_step[s] = {"n": len(xs), "rows_all_reduce": rows_ar, "rows_all_gather": rows_ag, "equal_signature": ok, "equal_sites": ok_sites,
                           "sites_named_equal": named_equal}
            if not ok:
                bad.append(s)
            if not ok_sites:
                bad_sites.append({"step": s, "sites_named_equal": named_equal, "first_mismatch": next(
                    ({"k": k, "record": g, "link": e} for k, (g, e) in enumerate(zip(got_sites, sig_sites))
                     if g is None or e is None or not (g == e if sites_exact else same_module_path(g, e))),
                    {"k": min(len(got_sites), len(sig_sites)), "record": None, "link": None, "note": "length mismatch"})})
        struct["per_rank"][r] = {"build_signature_len": len(sig), "build_digest": links["program_digest"], "n_steps": len(steps),
                                 "n_collectives": len(recs[r]), "steps_mismatching": bad,
                                 "sites_checked_per_step": len(sig_sites), "steps_site_mismatching": bad_sites[:5], "n_steps_site_mismatching": len(bad_sites),
                                 "site_rule": ("record.module == link.module (link.module_path minus the derive's RECORDED model_root, ' (Class)' dropped), per k; "
                                               if sites_exact else
                                               "record.module is a dotted suffix of link.module_path (v1 tp_links: no recorded model_root), per k; ")
                                              + "a None on either side is a mismatch (fail closed by absence, rev-pm M-0470)",
                                 "model_root": links.get("model_root"),
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
        # [R19 tp2 F-r19-tp2-1, after moe F-r19-moe-1] the prefill's all-gathers are of TWO row classes: the logits all-gather carries the
        # step's SAMPLED rows (#requests) and must be present; a MID-MODULE all-gather (OlmoeAttention's q/k gathers, recorded since the
        # recorder patches the model module's own import) carries the step's TOKENS, exactly like the hidden-stream all-reduces.  The R13 rule
        # `rows_all_gather == [#requests]` read the 3755-row q/k gathers of row #70 as a structure FAIL (rows [8, 3755]) although every one
        # of the 66 sites matched its link at every step.  Any other row count still fails.
        rows_ag = list(pre["rows_all_gather"])
        struct["prefill_all_gather_rows_equal_requests"] = (len(prompts) in rows_ag) and set(rows_ag) <= {len(prompts), sum(prompts)}
        struct["prefill_all_gather_rows"] = {"observed": rows_ag, "sampled_rows": len(prompts), "prompt_tokens": sum(prompts),
                                             "rule": "logits all-gather rows == #requests (required); mid-module all-gather rows == prompt tokens; nothing else"}
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

    # [lane vllm-tp-n] world > 2: the Definition's chain of one-rounding bf16 adds in `allreduce_order(world)` (AllReduce_v2{WORLD,N});
    # a mismatch is a named finding.  Diagnostics travel with it: how many received values equal the f32-accumulate-then-round-once
    # alternative (`alt_f32_once_equal`) and the ascending rank-order chain (`alt_ascending_equal`) -- never a relabel, facts for the finding.
    from verity_vllm.program.registry.b1_tp2 import allreduce_order
    order = allreduce_order(world)
    prov["definition_order"] = ("world 2: bf16(f32(part0) + f32(part1)); world > 2: acc = part[order[0]]; acc = bf16(f32(acc) + f32(part[r])) "
                                f"for r in order[1:], order = {list(order)} (b1_tp2.allreduce_order)")
    prov["alt_f32_once_equal"] = 0
    prov["alt_ascending_equal"] = 0
    for s in steps:
        pts = [cap / f"coll.rank{r}.step{s}.pt" for r in ranks]
        if not all(p.exists() for p in pts):
            continue
        if torch is None:
            raise RuntimeError("stored provenance steps present but torch is not importable")
        vs = [torch.load(p) for p in pts]
        xs_r = [[x for x in recs[r] if x["step"] == s] for r in ranks]
        n = min(len(v["part"]) for v in vs)
        if any(len(vs[r]["part"]) != len(xs_r[r]) or len(vs[r]["out"]) != len(xs_r[r]) for r in ranks):
            _digest_fail(-1, s, -1, "count", "stored tensors " + " ".join(f"rank{r} {len(vs[r]['part'])}/{len(vs[r]['out'])}" for r in ranks)
                         + " vs recorder rows " + " / ".join(str(len(xs_r[r])) for r in ranks))
        for k in range(n):
            parts = [vs[r]["part"][k] for r in ranks]
            outs = [vs[r]["out"][k] for r in ranks]
            if any(k >= len(xs_r[r]) for r in ranks):
                _digest_fail(-1, s, k, "row", "no recorder row for the stored tensor")
                continue
            x0 = xs_r[0]
            for rank in ranks:
                xr, part, out = xs_r[rank][k], parts[rank], outs[rank]
                for what, t, key in (("part", part, "part_sha256"), ("out", out, "out_sha256")):
                    want = xr.get(key)
                    got = _sha(t)
                    prov["tensors_digested"] += 1
                    if not want or got != want:
                        _digest_fail(rank, s, k, what, f"stored .pt {what} sha256 {got[:12]} != recorder row {str(want)[:12]} (the arithmetic below runs on bytes the record does not describe)")
                if int(xr.get("k", k)) != k or int(xr.get("step", s)) != s:
                    _digest_fail(rank, s, k, "key", f"recorder row keyed (step {xr.get('step')}, k {xr.get('k')}) at position {k} of step {s}")
            kind = x0[k]["kind"]
            exp = _expected(kind, x0[k].get("path"), parts, outs, order, prov)
            if exp is None:
                if kind != FN.KIND:
                    prov["mismatches"].append({"step": s, "k": k, "kind": kind, "why": "unbound collective kind"})
                    prov["pass"] = False
                continue
            same_all = all(torch.equal(outs[0], o) for o in outs[1:])
            eqs = [bool(torch.equal(o, exp)) for o in outs]
            prov["received_equal_both_ranks"] += int(same_all)
            prov["collectives_checked"] += 1
            prov["elements_checked"] += int(exp.numel())
            if not (same_all and all(eqs)):
                d0 = int((outs[0].view(torch.int16) != exp.view(torch.int16)).sum()) if outs[0].shape == exp.shape else -1
                mm = {"step": s, "k": k, "kind": kind, "module": x0[k]["module"], "received_equal_both_ranks": same_all,
                      "rank0_received_equals_definition": eqs[0], "rank1_received_equals_definition": eqs[1], "n_diff_rank0": d0}
                if world > 2:
                    mm["received_equals_definition_per_rank"] = eqs
                prov["mismatches"].append(mm)
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
    res["recorder"], res["mid_module_sites_ruled_out"] = tok.get("recorder"), _ruled_out_sites(cap, ranks)
    res["pass"] = bool(struct["pass"] and prov["pass"] and prov["collectives_checked"] > 0 and tk.get("pass") is not False)
    _write(run, res)
    print(json.dumps({k: res[k] for k in ("pass", "structure", "provenance", "tokens")}, indent=1, default=str)[:6000])
    return 0 if res["pass"] else 1


def _expected(kind: str, path: str | None, parts: list, outs: list, order: Any, prov: dict) -> Any:
    """What every rank must have received for one collective of ``kind`` from the ranks' ``parts``, counted in ``prov``; None for a
    kind with no relation here.  An all-reduce is the Definition's chain in ``order`` (world > 2 adds two diagnostics); a
    `fused_norm.KIND` call on the fused path is its new residual (`fused_norm.expected_residual`), and on the unfused path (past
    FlashInfer's token budget, the communicator's route) it is counted and left to the Commit's replay."""
    import torch

    if kind == "all_reduce":
        exp = parts[order[0]]
        for r_ in order[1:]:
            exp = (exp.float() + parts[r_].float()).to(torch.bfloat16)      # world 2: exactly the R13 rule bf16(f32(a) + f32(b))
        prov["all_reduce_checked"] += 1
        if len(parts) > 2:
            alt = sum(pr.float() for pr in parts).to(torch.bfloat16)
            prov["alt_f32_once_equal"] += int(all(torch.equal(o, alt) for o in outs))
            asc = parts[0]
            for pr in parts[1:]:
                asc = (asc.float() + pr.float()).to(torch.bfloat16)
            prov["alt_ascending_equal"] += int(all(torch.equal(o, asc) for o in outs))
        return exp
    if kind == "all_gather":
        prov["all_gather_checked"] += 1
        return torch.cat(parts, dim=-1)
    if kind == FN.KIND:
        if path != FN.FUSED:
            prov["fused_norm_unfused_left_to_commit"] = prov.get("fused_norm_unfused_left_to_commit", 0) + 1
            return None
        prov["fused_norm_checked"] = prov.get("fused_norm_checked", 0) + 1
        return FN.expected_residual(parts)
    return None


def _ruled_out_sites(cap: Path, ranks: list[int]) -> dict | None:
    """The mid-module collective calls the recorder did not expect because the instance's flags rule them out (`coll_meta.rank<r>.json`,
    `collective_sites.mid_module_site_census`); None when no rank's meta lists them."""
    ruled = {r: json.loads(p.read_text()).get("mid_module_sites_ruled_out", []) for r in ranks if (p := cap / f"coll_meta.rank{r}.json").exists()}
    if not ruled:
        return None
    first = ruled[min(ruled)]
    same = len(ruled) == len(ranks) and all(v == first for v in ruled.values())
    return {"n": len(first), "same_on_all_ranks": same, "sites": first, **({} if same else {"per_rank": {str(r): v for r, v in ruled.items()}})}


def _build_dirs(args) -> list[str]:
    """The rank Build dirs in rank order: `--build-rank DIR` repeated (world = count), or the two-rank `--build-rank0/--build-rank1`."""
    if getattr(args, "build_rank", None):
        if args.build_rank0 or args.build_rank1:
            raise SystemExit("give either --build-rank DIR (repeated, rank order) or --build-rank0/--build-rank1, not both")
        if len(args.build_rank) < 2:
            raise SystemExit(f"--build-rank: {len(args.build_rank)} rank Build(s); a tensor-parallel world needs at least 2")
        return list(args.build_rank)
    if not (args.build_rank0 and args.build_rank1):
        raise SystemExit("check arm: --build-rank DIR (repeated, rank order) or --build-rank0 and --build-rank1 required")
    return [args.build_rank0, args.build_rank1]


def _count(xs) -> dict:
    h: dict = {}
    for x in xs:
        h[x] = h.get(x, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _write(run: Path, res: dict) -> None:
    (run / "tp2_match.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


@dataclass(frozen=True, kw_only=True)
class Options:
    arm: str = option("--arm", choices=("capture", "control", "check"))
    workload: str | None = option("--workload", default=None)
    out: str = option("--out")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    case: str | None = option("--case", default=None)
    tp: int = option("--tp", type=int, default=2)
    engine_arg: list[str] = option("--engine-arg", action="append", default=[])
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    no_warmup: bool = option("--no-warmup", default=False, action="store_true")
    store_steps: int = option("--store-steps", type=int, default=1 << 30,
                              help="save the partial/received tensors of steps < N (all by default)")
    rpc_timeout: float = option("--rpc-timeout", type=float, default=1800.0)
    build_rank0: str | None = option("--build-rank0", default=None)
    build_rank1: str | None = option("--build-rank1", default=None)
    build_rank: list[str] = option("--build-rank", action="append", default=[],
                                   help="[lane vllm-tp-n] rank r's Build dir, repeated in rank order (world = count)")


def main(args: Options) -> int:
    if args.arm == "check":
        return _check_main(args)
    _engine_main(args)
    return 0
