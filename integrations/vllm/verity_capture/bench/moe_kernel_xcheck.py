#!/usr/bin/env python
"""[R13 moe / MOE-01] Bit-exact correspondence of the REGISTERED fused-MoE Definitions (`verity_vllm.registry.moe`) with the vLLM kernels
Bet A's `rules/vllm_moe.py` binds them to, on the pod that runs the rows (cp-moe, L40S), under several routing patterns and tilings.

Two phases (the kernels need the vLLM venv, Python 3.11; the Definitions are evaluated by the torch-free `verity.ir` under 3.12):

  dump   (venv-cu129, GPU)  for every pattern: bf16 router logits [M, E] (seeded: uniform / peaked-on-few-experts / near-ties) ->
         `_moe_C.topk_softmax(renormalize=False)`; bf16 hidden [M, H] and expert stacks w13 [E, 2I, H], w2 [E, H, I] -> vLLM's
         `fused_experts` (the same `invoke_fused_moe_kernel` / `silu_and_mul` / `moe_sum` the engine's TritonExperts path launches), with the
         two GEMM launches hooked to record their inputs, outputs, schedule (naive or sorted) and tile config.  Writes <out>/<pattern>.npz.
  check  (venv-py312, CPU)  evaluates, from the dump's own inputs, `MoeRouterTopK_v1{E,TOPK,VPT}` on EVERY token row (all TOPK weights and
         ids bit-equal), `MoeExpertCoordinate_v1{E,K}` / `MoeExpertCoordinateW_v1{E,K}` on sampled (pair, n) coordinates of the up and down
         launches (pairs chosen to cover every distinct expert and both ends of each expert's sorted block), `SiluMul_v1{I}` on sampled
         pair rows and `MoeSum_v1{TOPK,H}` on sampled tokens.  One inexact word is a refusal of the registration, never a tolerance.
         Negative control: an injected fault on one weight word must be reported.

  PY=/workspace/venv-cu129/bin/python; PY312=/workspace/venv-py312/bin/python; cd /workspace/cp/veritor
  VLLM_BATCH_INVARIANT=1 $PY verity_capture/bench/moe_kernel_xcheck.py dump  --out /workspace/cp/moe13/xcheck [--E 64 --topk 8 --H 2048 --I 1024]
  $PY312 verity_capture/bench/moe_kernel_xcheck.py check --dumps /workspace/cp/moe13/xcheck --out /workspace/cp/moe13/xcheck/xcheck.json [--coords 24]

Patterns (M = tokens): M=1 (the b1 decode step: the NAIVE block assignment, M*TOPK*4 <= E), M=3, M=17 and M=64 sorted schedules with
uniform logits; M=33 'peaked' (most tokens route to the same 3 experts: long expert blocks, padding); M=9 'ties' (logits quantised to
few distinct bf16 values: tie-breaking by lowest index); plus E=128 rows for the router only (the second tested E).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def patterns(E: int, TOPK: int) -> list[dict]:
    return [dict(name="m1_naive", M=1, logits="uniform", seed=11),
            dict(name="m3_uniform", M=3, logits="uniform", seed=12),
            dict(name="m17_uniform", M=17, logits="uniform", seed=13),
            dict(name="m64_uniform", M=64, logits="uniform", seed=14),
            dict(name="m33_peaked", M=33, logits="peaked", seed=15),
            dict(name="m9_ties", M=9, logits="ties", seed=16)]


# ----------------------------------------------------------------------------------------------------------------------- dump (GPU)


def _words16(t) -> np.ndarray:
    import torch
    return t.detach().contiguous().view(torch.int16).cpu().numpy().astype(np.uint16)


def _words32(t) -> np.ndarray:
    import torch
    return t.detach().contiguous().view(torch.int32).cpu().numpy().astype(np.uint32)


def make_logits(kind: str, M: int, E: int, gen, torch):
    if kind == "uniform":
        x = torch.randn(M, E, generator=gen, dtype=torch.float32) * 2.0
    elif kind == "peaked":
        x = torch.randn(M, E, generator=gen, dtype=torch.float32) * 0.5
        fav = torch.randperm(E, generator=gen)[:3]
        x[:, fav] += 6.0
        x[::7] = torch.randn((M + 6) // 7, E, generator=gen, dtype=torch.float32) * 2.0  # a few tokens route elsewhere
    elif kind == "ties":
        x = torch.randint(-2, 3, (M, E), generator=gen).to(torch.float32) * 0.5
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def dump(args) -> int:
    import torch
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    dev = torch.device("cuda")
    E, TOPK, H, I = args.E, args.topk, args.H, args.I
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(2026)
    # expert stacks (bf16, the layer's shapes: w13 [E, 2I, H], w2 [E, H, I]) -- one set for every pattern
    w13 = (torch.randn(E, 2 * I, H, generator=gen, dtype=torch.float32) * (H ** -0.5)).to(torch.bfloat16).to(dev)
    w2 = (torch.randn(E, H, I, generator=gen, dtype=torch.float32) * (I ** -0.5)).to(torch.bfloat16).to(dev)
    launches: list[dict] = []
    real_invoke = fm.dispatch_fused_moe_kernel

    def hooked(A, B, C, A_scale, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight,
               top_k, config, *a, **k):
        rec = dict(A=_words16(A), B_is_w13=bool(B.data_ptr() == w13.data_ptr()), mul_routed_weight=bool(mul_routed_weight), top_k=int(top_k),
                   config={kk: (int(v) if isinstance(v, (int, np.integer)) else v) for kk, v in dict(config).items()},
                   naive=sorted_token_ids is None,
                   sorted_token_ids=None if sorted_token_ids is None else _words32(sorted_token_ids),
                   expert_ids=_words32(expert_ids), num_tokens_post_padded=_words32(num_tokens_post_padded),
                   topk_weights=None if topk_weights is None else _words32(topk_weights))
        r = real_invoke(A, B, C, A_scale, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight,
                        top_k, config, *a, **k)
        torch.cuda.synchronize()
        rec["C"] = _words16(C)
        rec["C_shape"] = list(C.shape)
        launches.append(rec)
        return r

    fm.dispatch_fused_moe_kernel = hooked
    silu_recs: list[dict] = []  # silu_and_mul's input is the up launch's C and its output the down launch's A (both recorded)
    RN = bool(args.renormalize)
    manifest = {"E": E, "TOPK": TOPK, "H": H, "I": I, "renormalize": RN, "router_only": bool(args.router_only),
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "vllm": __import__("vllm").__version__, "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"), "patterns": []}
    for pat in patterns(E, TOPK):
        M = pat["M"]
        g = torch.Generator().manual_seed(pat["seed"])
        logits = make_logits(pat["logits"], M, E, g, torch).to(dev)
        hidden = (torch.randn(M, H, generator=g, dtype=torch.float32)).to(torch.bfloat16).to(dev)
        tw = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
        ti = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        tei = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        torch.ops._moe_C.topk_softmax(tw, ti, tei, logits, RN, None, None)
        torch.cuda.synchronize()
        if args.router_only:
            np.savez_compressed(os.path.join(args.out, f"{pat['name']}.npz"), logits=_words16(logits), topk_weights=_words32(tw), topk_ids=_words32(ti))
            manifest["patterns"].append(dict(pat, router_only=True, distinct_experts=int(len(np.unique(ti.cpu().numpy())))))
            print(f"[dump] {pat['name']}: M={M} router-only renormalize={RN}", flush=True)
            continue
        launches.clear()
        out = fm.fused_experts(hidden, w13, w2, tw, ti)
        torch.cuda.synchronize()
        assert len(launches) == 2, launches
        up, down = launches
        assert up["mul_routed_weight"] is False and down["mul_routed_weight"] is True and up["top_k"] == TOPK and down["top_k"] == 1
        # the up launch reads hidden rows; the down launch reads the activation's rows: tie the pieces to the whole
        assert (up["A"] == _words16(hidden)).all()
        assert down["A"].shape[0] == M * TOPK and up["C_shape"][-1] == 2 * I, (down["A"].shape, up["C_shape"])
        ids = ti.cpu().numpy()
        np.savez_compressed(os.path.join(args.out, f"{pat['name']}.npz"),
                            logits=_words16(logits), topk_weights=_words32(tw), topk_ids=_words32(ti), hidden=_words16(hidden),
                            w13=_words16(w13), w2=_words16(w2), up_C=up["C"].reshape(M * TOPK, 2 * I), down_A=down["A"].reshape(M * TOPK, I),
                            down_C=down["C"].reshape(M * TOPK, H), out=_words16(out))
        info = dict(pat, distinct_experts=int(len(np.unique(ids))), schedule="naive" if up["naive"] else "sorted",
                    up_config=up["config"], down_config=down["config"], expert_histogram=np.bincount(ids.reshape(-1), minlength=E).tolist(),
                    num_tokens_post_padded=int(up["num_tokens_post_padded"][0]))
        manifest["patterns"].append(info)
        print(f"[dump] {pat['name']}: M={M} schedule={info['schedule']} distinct_experts={info['distinct_experts']} "
              f"up={up['config']} down={down['config']}", flush=True)
    # router-only rows for the second tested E (128 experts, top-8): every routed weight / id of 40 rows
    for E2 in (128,):
        g = torch.Generator().manual_seed(77)
        M = 40
        logits = make_logits("uniform", M, E2, g, torch).to(dev)
        tw = torch.empty(M, TOPK, dtype=torch.float32, device=dev)
        ti = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        tei = torch.empty(M, TOPK, dtype=torch.int32, device=dev)
        torch.ops._moe_C.topk_softmax(tw, ti, tei, logits, RN, None, None)
        torch.cuda.synchronize()
        np.savez_compressed(os.path.join(args.out, f"router_E{E2}.npz"), logits=_words16(logits), topk_weights=_words32(tw), topk_ids=_words32(ti))
        manifest["patterns"].append(dict(name=f"router_E{E2}", M=M, E=E2, router_only=True))
        print(f"[dump] router_E{E2}: M={M}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("[dump] ok", args.out)
    return 0


# ----------------------------------------------------------------------------------------------------------------------- check (CPU)


def _eval(fn, inputs: list[list[int]]) -> list[int]:
    from verity.ir.program import Program
    from verity_vllm.registry.serve3_reference import materialize
    P = Program(fn)
    tab = materialize(P, inputs)
    return [tab[g] for g in P.circuit.root_outputs()]


def check(args) -> int:
    from verity.ir.defs import bind
    from verity_vllm.registry import b1
    from verity_vllm.registry import moe as R
    man = json.load(open(os.path.join(args.dumps, "manifest.json")))
    E, TOPK, H, I = man["E"], man["TOPK"], man["H"], man["I"]
    RN = bool(man.get("renormalize", False))
    Router = R.MoeRouterTopKNorm if RN else R.MoeRouterTopK   # [MOE-04] renormalize=True -> the 1/sum epilogue Definition
    VPT = 8  # == rules.vllm_moe.TOPK_SOFTMAX_VPT[torch.bfloat16] (16-byte loads of bf16); the rules module needs torch, this phase does not
    rng = np.random.default_rng(7)
    report = {"schema": "moe-kernel-xcheck/v1", "renormalize": RN, "router_definition": Router.name if hasattr(Router, "name") else str(Router), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dump": man, "patterns": [], "ok": True}
    t_all = time.time()
    for pat in man["patterns"]:
        if args.patterns and pat["name"] not in args.patterns.split(","):
            continue
        d = np.load(os.path.join(args.dumps, f"{pat['name']}.npz"))
        M = pat["M"]
        Ep = pat.get("E", E)
        res = {"name": pat["name"], "M": M, "E": Ep}
        # -- router: every row, every weight and id
        fn = bind(Router, E=Ep, TOPK=TOPK, VPT=VPT)
        t0 = time.time()
        bad = 0
        for t in range(M):
            got = _eval(fn, [[int(x) for x in d["logits"][t]]])
            want = [int(x) for x in d["topk_weights"][t]] + [int(x) for x in d["topk_ids"][t]]
            bad += sum(1 for a, b in zip(got, want) if a != b)
        res["router"] = {"rows": M, "words": M * 2 * TOPK, "inexact": bad, "seconds": round(time.time() - t0, 1)}
        if pat.get("router_only"):
            report["patterns"].append(res)
            report["ok"] &= bad == 0
            print(f"[check] {pat['name']}: router {res['router']}", flush=True)
            continue
        ids = d["topk_ids"].astype(np.int64).reshape(-1)
        ws = d["topk_weights"].reshape(-1)
        NP = M * TOPK
        # pairs: for every distinct expert its first and last pair (sorted-block ends) plus random pairs, up to --coords
        pairs = []
        for e in np.unique(ids):
            where = np.nonzero(ids == e)[0]
            pairs += [int(where[0]), int(where[-1])]
        pairs = sorted(set(pairs))
        extra = [int(p) for p in rng.choice(NP, size=min(NP, args.coords), replace=False)]
        pairs = sorted(set(pairs + extra))[: max(args.coords, len(set(pairs)))]
        # -- up launch: MoeExpertCoordinate on (pair, n) coordinates, n random per pair
        fn_up = bind(R.MoeExpertCoordinate, E=E, K=H)
        t0 = time.time()
        bad_up = 0
        n_up = 0
        for p in pairs:
            t = p // TOPK
            for n in rng.choice(2 * I, size=args.per_pair, replace=False):
                slab = d["w13"][:, n, :]  # [E, H]
                got = _eval(fn_up, [[int(x) for x in d["hidden"][t]], [int(ids[p])], [int(x) for x in slab.reshape(-1)]])
                n_up += 1
                bad_up += int(got[0] != int(d["up_C"][p, n]))
        res["up"] = {"coordinates": n_up, "inexact": bad_up, "pairs": len(pairs), "seconds": round(time.time() - t0, 1),
                     "config": pat["up_config"], "schedule": pat["schedule"]}
        # -- silu_and_mul: SiluMul rows
        fn_s = bind(b1.SiluMul, I=I)
        t0 = time.time()
        bad_s = 0
        srows = pairs[: args.silu_rows]
        for p in srows:
            got = _eval(fn_s, [[int(x) for x in d["up_C"][p]]])
            bad_s += sum(1 for a, b in zip(got, d["down_A"][p]) if a != int(b))
        res["silu"] = {"rows": len(srows), "words": len(srows) * I, "inexact": bad_s, "seconds": round(time.time() - t0, 1)}
        # -- down launch: MoeExpertCoordinateW with the routed weight
        fn_dn = bind(R.MoeExpertCoordinateW, E=E, K=I)
        t0 = time.time()
        bad_dn = 0
        n_dn = 0
        for p in pairs:
            for n in rng.choice(H, size=args.per_pair, replace=False):
                slab = d["w2"][:, n, :]  # [E, I]
                got = _eval(fn_dn, [[int(x) for x in d["down_A"][p]], [int(ids[p])], [int(ws[p])], [int(x) for x in slab.reshape(-1)]])
                n_dn += 1
                bad_dn += int(got[0] != int(d["down_C"][p, n]))
        res["down"] = {"coordinates": n_dn, "inexact": bad_dn, "pairs": len(pairs), "seconds": round(time.time() - t0, 1), "config": pat["down_config"]}
        # -- moe_sum: MoeSum rows (all TOPK slots of a token)
        fn_sum = bind(R.MoeSum, TOPK=TOPK, H=H)
        t0 = time.time()
        bad_sum = 0
        toks = sorted(set(p // TOPK for p in pairs))[: args.sum_rows]
        for t in toks:
            y = d["down_C"][t * TOPK:(t + 1) * TOPK].reshape(-1)
            got = _eval(fn_sum, [[int(x) for x in y]])
            bad_sum += sum(1 for a, b in zip(got, d["out"][t]) if a != int(b))
        res["sum"] = {"rows": len(toks), "words": len(toks) * H, "inexact": bad_sum, "seconds": round(time.time() - t0, 1)}
        res["ok"] = (bad + bad_up + bad_s + bad_dn + bad_sum) == 0
        report["ok"] &= res["ok"]
        report["patterns"].append(res)
        print(f"[check] {pat['name']}: {json.dumps({k: v for k, v in res.items() if k not in ('name',)})}", flush=True)
    # -- negative control: one flipped weight word in a down coordinate must be caught (router-only dumps: a flipped logit word instead)
    if man.get("router_only"):
        d = np.load(os.path.join(args.dumps, f"{man['patterns'][0]['name']}.npz"))
        lg = [int(x) for x in d["logits"][0]]
        lg[3] ^= 0x0080
        got = _eval(bind(Router, E=man["patterns"][0].get("E", E), TOPK=TOPK, VPT=VPT), [lg])
        want = [int(x) for x in d["topk_weights"][0]] + [int(x) for x in d["topk_ids"][0]]
        report["negative_control"] = {"fault": "logits[0][3] bit 7 flipped", "detected": got != want}
        report["ok"] &= report["negative_control"]["detected"]
        report["seconds"] = round(time.time() - t_all, 1)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print("[check]", "OK" if report["ok"] else "FAIL", args.out, f"{report['seconds']}s")
        return 0 if report["ok"] else 1
    d = np.load(os.path.join(args.dumps, f"{man['patterns'][1]['name']}.npz"))
    ids = d["topk_ids"].astype(np.int64).reshape(-1)
    ws = d["topk_weights"].reshape(-1)
    # the fault must be one the bf16 OUTPUT word cannot absorb: negating the selected expert's whole weight row negates the f32 dot
    # (a single-word flip among K terms can round to the same bf16 word -- that is precision, not a missed check)
    p, n = 0, 5
    slab = d["w2"][:, n, :].copy()
    slab[ids[p], :] ^= 0x8000
    got = _eval(bind(R.MoeExpertCoordinateW, E=E, K=I), [[int(x) for x in d["down_A"][p]], [int(ids[p])], [int(ws[p])], [int(x) for x in slab.reshape(-1)]])
    # and a fault in ANOTHER expert's row must NOT be detected (the pair reads only its own expert): the gather is by the value operand
    slab2 = d["w2"][:, n, :].copy()
    other = int((ids[p] + 1) % E)
    slab2[other, :] ^= 0x8000
    got2 = _eval(bind(R.MoeExpertCoordinateW, E=E, K=I), [[int(x) for x in d["down_A"][p]], [int(ids[p])], [int(ws[p])], [int(x) for x in slab2.reshape(-1)]])
    report["negative_control"] = {"fault": "w2[e, n, :] sign bits flipped (selected expert e)", "detected": bool(got[0] != int(d["down_C"][p, n])),
                                  "other_expert_fault_ignored": bool(got2[0] == int(d["down_C"][p, n]))}
    report["ok"] &= report["negative_control"]["other_expert_fault_ignored"]
    report["ok"] &= report["negative_control"]["detected"]
    report["seconds"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("[check]", "OK" if report["ok"] else "FAIL", args.out, f"{report['seconds']}s")
    return 0 if report["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--out", required=True)
    a.add_argument("--E", type=int, default=64)
    a.add_argument("--topk", type=int, default=8)
    a.add_argument("--H", type=int, default=2048)
    a.add_argument("--I", type=int, default=1024)
    a.add_argument("--renormalize", action="store_true", help="topk_softmax(renormalize=True): Qwen3-MoE / Mixtral norm_topk_prob -> MoeRouterTopKNorm_v1")
    a.add_argument("--router-only", action="store_true", help="dump only the router (no expert GEMMs): fast E/TOPK/renormalize coverage rows")
    b = sub.add_parser("check")
    b.add_argument("--dumps", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--coords", type=int, default=24, help="pairs sampled per launch (plus every expert's block ends)")
    b.add_argument("--per-pair", type=int, default=2, help="output coordinates per sampled pair")
    b.add_argument("--silu-rows", type=int, default=6)
    b.add_argument("--sum-rows", type=int, default=4)
    b.add_argument("--patterns", default="", help="comma-separated pattern names to evaluate (default all)")
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else check(args)


if __name__ == "__main__":
    sys.exit(main())
