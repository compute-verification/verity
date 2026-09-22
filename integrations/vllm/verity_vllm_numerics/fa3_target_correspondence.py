"""Per-target correspondence of vLLM's FlashAttention-3 forward (Hopper, sm_90) with the registered ``Attention_v2`` Definition.

[R13 lane hopper, deliverable 2(a)]  On cc 9.x the pinned vLLM selects FA3 (``_vllm_fa3_C``, ``vllm_flash_attn_version=3``).
The registered numerical Definition for a Hopper target is ``Attention_v2{T,NH,KVH,D,BN,DOT=HopperBF16WgmmaDot16,INV=Fa3InvSum}``
(``verity_vllm.registry.targets.attention_spec``): FA2's per-row online-softmax recurrence with the tensor-core step
``wgmma m64nNk16`` (one group of 16 products, 26-bit adder, floor -133), the epilogue ``inv = (sum == 0 || NaN) ? 0 : rcp(sum)``,
and the key-block size ``BN`` of the launched tile -- ``fa3_kblock_n(D, max_seqlen_q, NH/KVH)``: hdim 64 -> 192 under one MMA
warpgroup (``seqlen_q * qheads_per_kvhead <= 64``: decode, short prefill) else 128; hdim 128 -> 128.  Whether the kernel vLLM
actually launches computes exactly that function -- on the model's shapes, tails included (lengths around every block boundary),
prefill, chunked prefill and decode, one sequence and varlen batches (whose ``max_seqlen_q`` decides the tile for every row),
paged KV with a scrambled block table -- is what this driver measures, bit for bit.  The CPU spelling is
``verity_vllm_numerics.fa3_model`` (C++ recurrence), itself equal word for word to the typed Definition (tests/test_fa3_model.py).

Two phases (the pinned vLLM venv is Python 3.11; the CPU model wants the MUFU tables fixture and g++, any Python >= 3.10):

``capture`` (GPU host, vLLM venv): builds the cases, calls ``vllm.vllm_flash_attn.flash_attn_varlen_func`` EXACTLY as
``FlashAttentionImpl.forward`` does under batch invariance (paged K/V cache ``(num_blocks, block_size, KVH, D)``, ``cu_seqlens_q``,
``seqused_k``, ``block_table``, ``causal=True``, ``fa_version=3``, ``num_splits=1``, no window / softcap / alibi / descale),
records the CUDA kernels the call launched (``torch.profiler``: must be ``FlashAttnFwdSm90`` only -- no combine kernel), the
device, the FA3 extension's sha256, and stores the bf16 words of q, the caches, the tables and the output per case.

``check`` (any host): evaluates ``fa3_model.attention_call`` per sequence at the launch's kBlockN (``fa3_model.kblock_n_for`` ==
``verity_vllm.registry.targets.fa3_kblock_n`` when verity.ir is importable -- cross-checked) on the gathered canonical KV rows and
compares every output word.  Rows the model flags non-finite are reported ``unresolved`` (not counted exact, not a mismatch).
A single inexact coordinate is a refusal of the (target, head dim, tile) registration, never a tolerance.

Usage (pod):
    PYTHONPATH=. VLLM_BATCH_INVARIANT=1 /workspace/venv-cu129/bin/python -m verity_vllm_numerics.fa3_target_correspondence \\
        capture --geoms llama32-1b,qwen25-15b --dir /workspace/cp/sweep/diag/fa3_target_h100
    VERITY_MUFU_TABLES=<repo>/fixtures/W11-40f0cebeb670-20260907T1800Z/mufu OMP_NUM_THREADS=32 PYTHONPATH=. \\
        /workspace/venv-py312/bin/python -m verity_vllm_numerics.fa3_target_correspondence check --dir ... --out .../fa3_target_h100.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

# (NH, KVH, D) as the models' attention layers launch them; block_size 16 = vLLM's default KV block for these rows.
GEOMS: dict[str, dict] = {
    "llama32-1b": {"NH": 32, "KVH": 8, "D": 64, "model": "unsloth/Llama-3.2-1B"},
    "qwen3-4b": {"NH": 32, "KVH": 8, "D": 128, "model": "Qwen/Qwen3-4B"},
    "qwen25-15b": {"NH": 12, "KVH": 2, "D": 128, "model": "Qwen/Qwen2.5-1.5B"},
}
BLOCK_SIZE = 16

# Case families: lists of sequences (seqlen_q, seqlen_k) in ONE launch (varlen batch).  Lengths straddle every tile boundary
# reachable at hdim 64 (128, 192, 384 = lcm) and the one-MMA-warpgroup switch (seqlen_q * qh_per_kvh <= 64 -> for Llama qh=4:
# seqlen_q <= 16).
PREFILL_LENS = (1, 5, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 383, 384, 385, 1000, 1024)
DECODE_KLENS = (1, 2, 17, 127, 128, 129, 191, 192, 193, 383, 384, 385, 576, 577, 1024, 1025, 1151, 1152)
CHUNKED = ((64, 1000), (17, 529), (16, 529), (128, 1152), (7, 200))
BATCHES = (
    ((1, 100), (1, 257), (1, 1025)),            # decode batch (max_seqlen_q 1)
    ((37, 37), (200, 200)),                     # prefill batch
    ((1, 300), (64, 64)),                       # mixed: the decode row runs in the prefill tile (max_seqlen_q 64 -> hd64 BN 128)
    ((1, 300), (16, 16)),                       # mixed: max_seqlen_q 16 -> Llama one MMA warpgroup (hd64 BN 192)
    ((5, 900), (1, 1), (1, 1152)),
)
FAMILIES = ("normal", "wide")   # operand distributions: N(0,1) bf16; q,k scaled x3 (larger |S|, more exp2 range)


def _cases(geom: dict) -> list[dict]:
    cs: list[dict] = []
    for L in PREFILL_LENS:
        cs.append({"kind": "prefill", "seqs": [(L, L)]})
    for Lk in DECODE_KLENS:
        cs.append({"kind": "decode", "seqs": [(1, Lk)]})
    for q, k in CHUNKED:
        cs.append({"kind": "chunked", "seqs": [(q, k)]})
    for b in BATCHES:
        cs.append({"kind": "batch", "seqs": list(b)})
    out = []
    for fam in FAMILIES:
        for c in cs:
            if fam == "wide" and c["kind"] not in ("prefill", "decode"):
                continue
            out.append({**c, "family": fam})
    return out


def _bf16_words(rng: np.random.Generator, shape, scale: float) -> np.ndarray:
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)  # truncation to bf16 (any bf16 word is a valid operand)


def _build_case(rng: np.random.Generator, geom: dict, case: dict) -> dict:
    NH, KVH, D = geom["NH"], geom["KVH"], geom["D"]
    qs = 3.0 if case["family"] == "wide" else 1.0
    seqs = case["seqs"]
    lq = [s[0] for s in seqs]
    lk = [s[1] for s in seqs]
    cu_q = np.concatenate([[0], np.cumsum(lq)]).astype(np.int32)
    nblk = [-(-k // BLOCK_SIZE) for k in lk]
    total_blocks = sum(nblk) + 3                       # a few blocks nobody owns (must never be read into a visible column)
    perm = rng.permutation(total_blocks).astype(np.int32)
    maxb = max(nblk)
    block_table = np.zeros((len(seqs), maxb), np.int32)
    off = 0
    for i, nb in enumerate(nblk):
        block_table[i, :nb] = perm[off:off + nb]
        off += nb
        block_table[i, nb:] = 0                         # vLLM pads with 0 (a real block id); never dereferenced for visible keys
    q = _bf16_words(rng, (int(cu_q[-1]), NH, D), qs)
    k_cache = _bf16_words(rng, (total_blocks, BLOCK_SIZE, KVH, D), qs)   # every slot filled, incl. slots past seqused_k
    v_cache = _bf16_words(rng, (total_blocks, BLOCK_SIZE, KVH, D), 1.0)
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "block_table": block_table, "cu_seqlens_q": cu_q,
            "seqused_k": np.asarray(lk, np.int32), "max_seqlen_q": int(max(lq)), "max_seqlen_k": int(max(lk)),
            "softmax_scale": float(D) ** -0.5}


# ------------------------------------------------------------------------------------------------------ capture (GPU, vLLM venv)

def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def capture(args: argparse.Namespace) -> None:
    import torch
    import vllm
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn import flash_attn_interface as fai

    os.makedirs(args.dir, exist_ok=True)
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    ok3, why3 = fai.is_fa_version_supported(3), None
    if not ok3:
        raise SystemExit(f"FA3 not supported here: {fai.fa_version_unsupported_reason(3) if hasattr(fai, 'fa_version_unsupported_reason') else ''}")
    so = [m.__file__ for n, m in sys.modules.items() if n.endswith("_vllm_fa3_C") and getattr(m, "__file__", None)]
    if not so:
        import importlib
        try:
            so = [importlib.import_module("vllm.vllm_flash_attn._vllm_fa3_C").__file__]
        except Exception:  # noqa: BLE001
            so = []
    header = {
        "schema": "verity/fa3-target-correspondence/capture/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "gpu": {"name": p.name, "compute_capability": [p.major, p.minor], "num_sms": p.multi_processor_count},
        "vllm": {"version": vllm.__version__, "fa3_extension": so[0] if so else None, "fa3_extension_sha256": _sha256_file(so[0]) if so else None},
        "torch": torch.__version__, "cuda": torch.version.cuda, "batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "call": "flash_attn_varlen_func(q, k_cache, v_cache, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, "
                "causal=True, block_table, fa_version=3, num_splits=1)  [as FlashAttentionImpl.forward, VLLM_BATCH_INVARIANT]",
        "block_size": BLOCK_SIZE, "geoms": {}, "cases": [],
    }
    n = 0
    for gname in args.geoms.split(","):
        geom = GEOMS[gname]
        header["geoms"][gname] = geom
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"fa3-target/{gname}".encode()).digest()[:8], "little"))
        for case in _cases(geom):
            c = _build_case(rng, geom, case)
            t = {k: torch.from_numpy(v.view(np.int16)).to(dev).view(torch.bfloat16) for k, v in c.items()
                 if k in ("q", "k_cache", "v_cache")}
            bt = torch.from_numpy(c["block_table"]).to(dev)
            cu = torch.from_numpy(c["cu_seqlens_q"]).to(dev)
            su = torch.from_numpy(c["seqused_k"]).to(dev)
            out = torch.empty_like(t["q"])
            kernels: list[str] = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                flash_attn_varlen_func(q=t["q"], k=t["k_cache"], v=t["v_cache"], out=out, cu_seqlens_q=cu, max_seqlen_q=c["max_seqlen_q"],
                                       seqused_k=su, max_seqlen_k=c["max_seqlen_k"], softmax_scale=c["softmax_scale"], causal=True,
                                       block_table=bt, fa_version=3, num_splits=1)
                torch.cuda.synchronize()
            for e in prof.events():
                if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type) and e.name not in kernels:
                    kernels.append(e.name)
            # a second call must reproduce the words (determinism of the launched kernel on identical inputs)
            out2 = torch.empty_like(t["q"])
            flash_attn_varlen_func(q=t["q"], k=t["k_cache"], v=t["v_cache"], out=out2, cu_seqlens_q=cu, max_seqlen_q=c["max_seqlen_q"],
                                   seqused_k=su, max_seqlen_k=c["max_seqlen_k"], softmax_scale=c["softmax_scale"], causal=True,
                                   block_table=bt, fa_version=3, num_splits=1)
            torch.cuda.synchronize()
            o = out.view(torch.int16).cpu().numpy().view(np.uint16)
            o2 = out2.view(torch.int16).cpu().numpy().view(np.uint16)
            fn = os.path.join(args.dir, f"case_{n:03d}.npz")
            np.savez_compressed(fn, out=o, **{k: v for k, v in c.items() if isinstance(v, np.ndarray)})
            attn_k = [k for k in kernels if "flash" in k.lower() or "Flash" in k]
            header["cases"].append({"n": n, "file": os.path.basename(fn), "geom": gname, "kind": case["kind"], "family": case["family"],
                                    "seqs": case["seqs"], "max_seqlen_q": c["max_seqlen_q"], "max_seqlen_k": c["max_seqlen_k"],
                                    "softmax_scale": c["softmax_scale"], "kernels": attn_k, "all_cuda_kernels": kernels,
                                    "repeat_equal": bool(np.array_equal(o, o2)), "rows": int(o.shape[0]) * geom["NH"]})
            print(f"[capture] {n:3d} {gname} {case['kind']:8s} {case['family']:6s} seqs={case['seqs']} kernels={attn_k} "
                  f"repeat_equal={np.array_equal(o, o2)}", flush=True)
            n += 1
    json.dump(header, open(os.path.join(args.dir, "cases.json"), "w"), indent=1)
    print(f"[capture] {n} cases -> {args.dir}")


# ---------------------------------------------------------------------------------------------------------- check (CPU model)

def _gather_kv(cache: np.ndarray, block_table_row: np.ndarray, seqlen_k: int) -> np.ndarray:
    """Canonical KV rows (seqlen_k, KVH, D) of one sequence from the paged cache via its block table."""
    nb = -(-seqlen_k // BLOCK_SIZE)
    rows = cache[block_table_row[:nb]].reshape(nb * BLOCK_SIZE, cache.shape[2], cache.shape[3])
    return rows[:seqlen_k]


def check(args: argparse.Namespace) -> None:
    from verity_vllm_numerics import fa3_model as FA3
    from verity_vllm_numerics.fa2_model import MufuTables

    tdir = os.environ.get("VERITY_MUFU_TABLES") or args.tables
    if not tdir:
        raise SystemExit("--tables / VERITY_MUFU_TABLES (mufu_ex2/rcp_delta_int8.xz) required")
    tables = MufuTables.load(tdir)
    hdr = json.load(open(os.path.join(args.dir, "cases.json")))
    rule_xcheck = None
    try:
        from verity_vllm.registry.targets import fa3_kblock_n as _reg_rule
        rule_xcheck = _reg_rule
    except Exception:  # noqa: BLE001
        pass
    report = {"schema": "verity/fa3-target-correspondence/check/v1", "capture": {k: hdr[k] for k in ("utc", "gpu", "vllm", "torch", "cuda", "call")},
              "model": FA3.KERNEL, "tables": tables.source, "cases": [], "totals": {}}
    tot = {"cases": 0, "rows": 0, "words": 0, "mismatch_words": 0, "mismatch_rows": 0, "unresolved_rows": 0, "kernel_unexpected": 0,
           "repeat_unequal": 0, "rule_disagreements": 0}
    by_tile: dict[str, dict] = {}
    t0 = time.time()
    for c in hdr["cases"]:
        if args.only and c["n"] not in args.only:
            continue
        geom = hdr["geoms"][c["geom"]]
        NH, KVH, D = geom["NH"], geom["KVH"], geom["D"]
        z = np.load(os.path.join(args.dir, c["file"]))
        q, out, kc, vc, bt, cu, su = (z[k] for k in ("q", "out", "k_cache", "v_cache", "block_table", "cu_seqlens_q", "seqused_k"))
        BN = FA3.kblock_n_for(D, c["max_seqlen_q"], NH // KVH)
        if args.force_bn:          # negative control: the OTHER registered tile must NOT match (a tile is a different function)
            BN = int(args.force_bn)
        if rule_xcheck is not None and rule_xcheck(D, c["max_seqlen_q"], NH // KVH) != BN:
            tot["rule_disagreements"] += 1
        # the launch = the varlen block-count prep kernel (`flash::prepare_varlen_num_blocks_kernel`, scheduling only) + exactly ONE
        # attention kernel, the sm90 forward (`FlashAttnFwdSm90<...>`); a split-KV combine (`FlashAttnFwdCombine`) would be another function
        fwd = [k for k in c["all_cuda_kernels"] if "FlashAttnFwd" in k or "flash_fwd" in k]
        exp_kernel = len(fwd) == 1 and "FlashAttnFwdSm90" in fwd[0] and not any("Combine" in k for k in c["all_cuda_kernels"])
        if not exp_kernel:
            tot["kernel_unexpected"] += 1
        if not c["repeat_equal"]:
            tot["repeat_unequal"] += 1
        mm_words = mm_rows = unres = 0
        rows = 0
        first_bad = None
        for b, (lq, lk) in enumerate(c["seqs"]):
            qb = q[int(cu[b]):int(cu[b + 1])]
            ob = out[int(cu[b]):int(cu[b + 1])]
            K = _gather_kv(kc, bt[b], int(lk))
            V = _gather_kv(vc, bt[b], int(lk))
            rr = FA3.attention_call(qb, K, V, np.arange(lq), int(lq), int(lk), float(c["softmax_scale"]), tables, kblock_n=BN)
            eq = (rr.out == ob)
            nf = rr.nonfinite                      # (lq, NH)
            bad_rows = (~eq.all(axis=2)) & ~nf
            mm_rows += int(bad_rows.sum())
            mm_words += int((~eq)[~nf].sum())
            unres += int(nf.sum())
            rows += lq * NH
            if first_bad is None and bad_rows.any():
                r, h = np.argwhere(bad_rows)[0]
                d = np.flatnonzero(~eq[r, h])[:4]
                first_bad = {"seq": b, "row": int(r), "head": int(h), "dims": d.tolist(),
                             "kernel": [f"0x{int(x):04x}" for x in ob[r, h, d]], "model": [f"0x{int(x):04x}" for x in rr.out[r, h, d]]}
        exact = mm_rows == 0 and exp_kernel and c["repeat_equal"]
        rec = {**{k: c[k] for k in ("n", "geom", "kind", "family", "seqs", "max_seqlen_q", "max_seqlen_k", "kernels")}, "D": D, "BN": BN,
               "rows": rows, "words": rows * D, "mismatch_rows": mm_rows, "mismatch_words": mm_words, "unresolved_rows": unres,
               "kernel_expected": exp_kernel, "repeat_equal": c["repeat_equal"], "exact": exact, "first_mismatch": first_bad}
        report["cases"].append(rec)
        key = f"D={D},BN={BN}"
        bt_ = by_tile.setdefault(key, {"cases": 0, "exact_cases": 0, "rows": 0, "mismatch_rows": 0, "unresolved_rows": 0})
        bt_["cases"] += 1; bt_["exact_cases"] += int(exact); bt_["rows"] += rows; bt_["mismatch_rows"] += mm_rows; bt_["unresolved_rows"] += unres
        tot["cases"] += 1; tot["rows"] += rows; tot["words"] += rows * D; tot["mismatch_words"] += mm_words
        tot["mismatch_rows"] += mm_rows; tot["unresolved_rows"] += unres
        print(f"[check] {c['n']:3d} {c['geom']} D={D} BN={BN} {c['kind']:8s} {c['family']:6s} seqs={c['seqs']} rows={rows} "
              f"mismatch_rows={mm_rows} unresolved={unres} kernel_ok={exp_kernel} -> {'EXACT' if exact else 'MISMATCH'}"
              + (f" first={first_bad}" if first_bad else ""), flush=True)
    report["totals"] = tot
    report["by_tile"] = by_tile
    report["verdict"] = "EXACT" if (tot["mismatch_rows"] == 0 and tot["kernel_unexpected"] == 0 and tot["repeat_unequal"] == 0
                                    and tot["rule_disagreements"] == 0 and tot["cases"] > 0) else "MISMATCH"
    report["elapsed_s"] = round(time.time() - t0, 1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
    print(f"[check] {report['verdict']}: {tot} by_tile={by_tile} ({report['elapsed_s']} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("capture")
    a.add_argument("--geoms", default="llama32-1b")
    a.add_argument("--dir", required=True)
    b = sub.add_parser("check")
    b.add_argument("--dir", required=True)
    b.add_argument("--out")
    b.add_argument("--tables")
    b.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")] if s else None, default=None)
    b.add_argument("--force-bn", type=int, default=None, help="negative control: evaluate every case at this kBlockN instead of the tile rule")
    args = ap.parse_args(argv)
    {"capture": capture, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
