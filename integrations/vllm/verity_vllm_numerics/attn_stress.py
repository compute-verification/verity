"""Stress (CONTRACT §11) and mutation inputs for the FlashAttention-2 model, run against the
production kernel (``vllm.vllm_flash_attn.flash_attn_varlen_func`` on a paged KV cache, exactly the
call the B1 profile / the same_runtime checker make) on the GPU host.

Writes a fixture in the ``verity-vllm/attn-fixture/v1`` layout (``attn_capture.py``) so that
``attn_check.py`` and the CPU tests read it unchanged:

  <out>/index.json            calls[] with family/case names, consts, kernel identity
  <out>/calls/NNNNN_{q,out}.npy   (n_rows, H, D) u16 BF16 words
  <out>/kv/cNNN_{k,v}.npy         (Lk, KVH, D) u16 BF16 words (structured cases); plain random-normal K/V are
  <out>/kv/pool_normal_{k,v}.npy  the first Lk rows of two shared 4096-row pools (readers slice [:Lk])
  <out>/mutations.json        one-bit flips of one K / V / Q word: production output on the mutated
                              input, whether it differs from the production output on the original
  <out>/SHA256SUMS

Families (§11): zeros (Q=0 -> uniform rows; K=0; V=0), cancellation (large +/- partial products in
Q.K), rounding boundaries (output words that land exactly on BF16 ties), subnormals (BF16 subnormal
Q/K/V), extremes (|S| ~ 1e4 -> P flushed to 0; V near BF16 max -> FP32 overflow to inf; mixed signs ->
NaN), max shifts (the running max increases at every 64-key block, so every block rescales), near-
uniform rows, tile boundaries (Lk and Lq on both sides of 16/64/128/... and causal offsets), plus
Lk = 4095/4096 beyond the natural range (reported as their own family).

GPU host only: PYTHONPATH=. python -m verity_vllm_numerics.attn_stress OUT_DIR [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

H, KVH, D = 12, 2, 128
SCALE = 1.0 / float(np.sqrt(128.0))  # Qwen2.5-1.5B head_dim 128 (python float, as vLLM passes it)
BLOCK = 16


def words(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def from_words(w: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w).view(np.int16)).view(torch.bfloat16).cuda()


def bf16(x: np.ndarray | float, shape=None) -> np.ndarray:
    a = np.asarray(x, np.float32)
    if shape is not None:
        a = np.broadcast_to(a, shape).astype(np.float32)
    return words(torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16))


def production(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> np.ndarray:
    """q (Lq, H, D), k/v (Lk, KVH, D) BF16 words -> out (Lq, H, D) BF16 words from the production kernel."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    Lq, Lk = q.shape[0], k.shape[0]
    nblocks = (Lk + BLOCK - 1) // BLOCK
    kc = torch.zeros((nblocks * BLOCK, KVH, D), dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    kc[:Lk] = from_words(k)
    vc[:Lk] = from_words(v)
    kc = kc.view(nblocks, BLOCK, KVH, D)
    vc = vc.view(nblocks, BLOCK, KVH, D)
    qt = from_words(q)
    out = torch.empty_like(qt)
    flash_attn_varlen_func(
        q=qt, k=kc, v=vc, out=out,
        cu_seqlens_q=torch.tensor([0, Lq], dtype=torch.int32, device="cuda"), max_seqlen_q=Lq,
        seqused_k=torch.tensor([Lk], dtype=torch.int32, device="cuda"), max_seqlen_k=Lk,
        softmax_scale=SCALE, causal=causal, alibi_slopes=None, window_size=None,
        block_table=torch.arange(nblocks, dtype=torch.int32, device="cuda").reshape(1, -1),
        softcap=0.0, scheduler_metadata=None, fa_version=2, q_descale=None, k_descale=None, v_descale=None, num_splits=1,
    )
    torch.cuda.synchronize()
    return words(out)


# ---- families ------------------------------------------------------------------------------------------


def normal(rng, shape, std=1.0):
    return bf16(rng.standard_normal(shape).astype(np.float32) * np.float32(std))


POOL_N = 4096


def cases(rng: np.random.Generator) -> list[dict]:
    """Every case is {family, case, q, k, v[, kv_pool]}.  Plain random-normal K/V of length n are the first n rows
    of two shared 4096-row pools (``kv_pool = "normal"``) so the fixture stays small; structured K/V are per case."""
    C: list[dict] = []
    pool_k = normal(rng, (POOL_N, KVH, D))
    pool_v = normal(rng, (POOL_N, KVH, D))

    def add(family, name, q, k, v, **extra):
        C.append({"family": family, "case": name, "q": q, "k": k, "v": v, **extra})

    def addp(family, name, q, n, k=None, v=None):
        """K and/or V from the shared pools (prefix of length n)."""
        C.append({"family": family, "case": name, "q": q, "k": pool_k[:n] if k is None else k, "v": pool_v[:n] if v is None else v,
                  "kv_pool": {"k": k is None, "v": v is None}})

    # --- zeros
    for n in (1, 64, 65, 2048):
        addp("zeros", f"q0_n{n}", bf16(0.0, (1, H, D)), n)
    addp("zeros", "k0_n300", normal(rng, (1, H, D)), 300, k=bf16(0.0, (300, KVH, D)))
    addp("zeros", "v0_n300", normal(rng, (1, H, D)), 300, v=bf16(0.0, (300, KVH, D)))
    add("zeros", "all0_n129", bf16(0.0, (1, H, D)), bf16(0.0, (129, KVH, D)), bf16(0.0, (129, KVH, D)))
    addp("zeros", "prefill_q0_L70", bf16(0.0, (70, H, D)), 70)

    # --- cancellation: Q = [+a, -a, +a, ...] * large, K rows = same sign pattern with a small perturbation, so
    # every S is a small residue of large cancelling partial products inside the MMA
    a = np.float32(64.0)
    sign = np.where(np.arange(D) % 2 == 0, 1.0, -1.0).astype(np.float32)
    for n in (64, 200):
        q = np.broadcast_to(a * sign, (1, H, D)).copy()
        kk = np.broadcast_to(sign, (n, KVH, D)) + rng.standard_normal((n, KVH, D)).astype(np.float32) * np.float32(1e-2)
        addp("cancellation", f"pm64_n{n}", bf16(q), n, k=bf16(kk))
    # rows of K that are the negation of Q's head-0 direction: logits -a*a*D vs +a*a*D in the same block
    q = normal(rng, (1, H, D), std=4.0)
    kk = np.zeros((128, KVH, D), np.float32)
    qf = (q.astype(np.uint32) << 16).view(np.float32)
    kk[0::2] = qf[0, 0]
    kk[1::2] = -qf[0, 0]
    addp("cancellation", "alternating_sign_logits_n128", q, 128, k=bf16(kk))

    # --- rounding boundaries: Q=0 -> P=1 for every key, so sum = Lk exactly and O = sum of V rows (exact for
    # small integers); with Lk=2 (inv = 0.5 exact) and V0+V1 odd in [257, 511] the normalised value is x.5 with
    # 8 significant bits -> exactly a BF16 RNE tie for every output word
    v = np.zeros((2, KVH, D), np.float32)
    v[0] = 256.0 + 2.0 * (np.arange(D)[None, :] % 127)
    v[1] = 1.0
    addp("rounding_boundaries", "cast_ties_n2", bf16(0.0, (1, H, D)), 2, v=bf16(v))
    # Lk=4 (inv = 0.25 exact): V0 + V1 = 1028 + 8j -> O * inv = 257 + 2j, odd integers in [256, 512) = BF16 ties
    v = np.zeros((4, KVH, D), np.float32)
    v[0] = 1024.0 + 8.0 * (np.arange(D)[None, :] % 32)
    v[1] = 4.0
    addp("rounding_boundaries", "cast_ties_n4", bf16(0.0, (1, H, D)), 4, v=bf16(v))
    # P on BF16 ties: S = q0*k_c exactly (one non-zero coordinate), scan k over a fine grid so many P values
    # fall near BF16 rounding ties before the PV MMA
    q = np.zeros((1, H, D), np.float32)
    q[..., 0] = 1.0
    kk = np.zeros((2048, KVH, D), np.float32)
    kk[:, :, 0] = np.linspace(-40.0, 0.0, 2048, dtype=np.float32)[:, None]
    addp("rounding_boundaries", "p_grid_n2048", bf16(q), 2048, k=bf16(kk))

    # --- subnormals (BF16 subnormal magnitude < 2^-126)
    sub = lambda shape: bf16(rng.standard_normal(shape).astype(np.float32) * np.float32(2.0**-130))
    addp("subnormals", "v_subnormal_n100", normal(rng, (1, H, D)), 100, v=sub((100, KVH, D)))
    addp("subnormals", "k_subnormal_n100", normal(rng, (1, H, D)), 100, k=sub((100, KVH, D)))
    addp("subnormals", "q_subnormal_n100", sub((1, H, D)), 100)
    addp("subnormals", "qk_subnormal_n100", sub((1, H, D)), 100, k=sub((100, KVH, D)))
    vv = pool_v[:100].copy()
    vv[::3] = sub((34, KVH, D))
    addp("subnormals", "v_mixed_subnormal_n100", normal(rng, (1, H, D)), 100, v=vv)

    # --- extremes
    addp("extremes", "logits_1e4_n200", normal(rng, (1, H, D), std=30.0), 200, k=normal(rng, (200, KVH, D), std=30.0))
    addp("extremes", "v_near_max_n64", normal(rng, (1, H, D)), 64, v=bf16(rng.uniform(1e38, 3.3e38, (64, KVH, D)).astype(np.float32)))
    addp("extremes", "v_pm_max_n64", normal(rng, (1, H, D)), 64, v=bf16(rng.choice([-3.3e38, 3.3e38], (64, KVH, D)).astype(np.float32)))
    add("extremes", "v_huge_p_small_n200", normal(rng, (1, H, D), std=30.0), normal(rng, (200, KVH, D), std=30.0), bf16(rng.uniform(-3e38, 3e38, (200, KVH, D)).astype(np.float32)))
    qq = normal(rng, (1, H, D)).copy()
    qq[0, 3, 5] = int(bf16(np.inf))
    addp("extremes", "q_inf_n64", qq, 64)
    kk = pool_k[:64].copy()
    kk[7, 1, 9] = int(bf16(np.nan))
    addp("extremes", "k_nan_n64", normal(rng, (1, H, D)), 64, k=kk)

    # --- max shifts: blocks are visited from the last one down to block 0; make the logits grow towards
    # column 0 so that every block raises the running max (rescale at every block, large factors)
    for n, step in ((640, 2.0), (2048, 0.5), (1000, 20.0)):
        q = np.zeros((1, H, D), np.float32)
        q[..., 0] = 1.0
        kk = rng.standard_normal((n, KVH, D)).astype(np.float32) * np.float32(0.1)
        kk[:, :, 0] = (np.float32(step) * (n - 1 - np.arange(n)).astype(np.float32) / 64.0)[:, None]
        addp("max_shifts", f"ramp_n{n}_step{step}", bf16(q), n, k=bf16(kk))
    # prefill flavour with causal offsets
    q = np.zeros((70, H, D), np.float32)
    q[..., 0] = 1.0
    kk = rng.standard_normal((300, KVH, D)).astype(np.float32) * np.float32(0.1)
    kk[:, :, 0] = (np.float32(3.0) * (300 - 1 - np.arange(300)).astype(np.float32) / 64.0)[:, None]
    addp("max_shifts", "ramp_prefill_Lq70_Lk300", bf16(q), 300, k=bf16(kk))

    # --- near-uniform rows: K rows equal up to one BF16 ulp -> S almost equal, P ~ 1 everywhere
    for n in (64, 1000, 2048):
        base = rng.standard_normal((1, KVH, D)).astype(np.float32)
        kw = bf16(np.broadcast_to(base, (n, KVH, D)).copy())
        kw = (kw.astype(np.int32) + rng.integers(-1, 2, kw.shape)).astype(np.uint16)  # -1, 0, +1 ulp
        addp("near_uniform", f"kulp_n{n}", normal(rng, (1, H, D)), n, k=kw)

    # --- tile boundaries (decode)
    for n in (1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 193, 1023, 1024, 1025, 2047, 2048):
        addp("tile_boundaries", f"decode_n{n}", normal(rng, (1, H, D)), n)
    # --- tile boundaries (prefill: Lq rows, causal offset off = Lk - Lq)
    for Lq in (1, 2, 63, 64, 65, 128, 129, 200):
        for off in (0, 1, 40, 63, 64, 65):
            Lk = Lq + off
            addp("tile_boundaries", f"prefill_Lq{Lq}_off{off}", normal(rng, (Lq, H, D)), Lk)
    # --- beyond the natural range
    for n in (4095, 4096):
        addp("beyond_natural_range", f"decode_n{n}", normal(rng, (1, H, D)), n)
    addp("beyond_natural_range", "prefill_Lq64_Lk4096", normal(rng, (64, H, D)), 4096)
    addp("beyond_natural_range", "prefill_Lq2048_Lk2048", normal(rng, (2048, H, D)), 2048)
    addp("beyond_natural_range", "prefill_Lq4096_Lk4096", normal(rng, (4096, H, D)), 4096)
    return C


def sample_rows(Lq: int, limit: int = 96) -> list[int]:
    """Rows to save for a prefill case: all when Lq <= limit; else the rows around every 64-row M-block
    boundary and 16-row page boundary (where the kernel geometry changes), the first/last rows, and an
    even spread of the rest, capped at ``limit``."""
    if Lq <= limit:
        return list(range(Lq))
    must = {0, 1, 2, Lq - 2, Lq - 1}
    for b in range(64, Lq, 64):
        must |= {b - 1, b, b + 1}
    must = {r for r in must if 0 <= r < Lq}
    if len(must) >= limit:
        keep = sorted(must)
        step = max(1, len(keep) // limit)
        return sorted(set(keep[::step][:limit]) | {0, Lq - 1})
    rest = [r for r in range(Lq) if r not in must]
    k = limit - len(must)
    spread = [rest[int(i * len(rest) / k)] for i in range(k)] if k else []
    return sorted(must | set(spread))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--natural", default=None, help="natural fixture dir; a few of its decode calls are also mutated")
    ap.add_argument("--mutations-per-case", type=int, default=6)
    ap.add_argument("--prefill-row-limit", type=int, default=96, help="max query rows saved per prefill case (fixture size)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "kv").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    import vllm

    t0 = time.time()
    C = cases(rng)
    calls = []
    pool_saved = False
    for i, c in enumerate(C):
        q, k, v = c["q"], c["k"], c["v"]
        Lq, Lk = q.shape[0], k.shape[0]
        o = production(q, k, v)
        rows = sample_rows(Lq, args.prefill_row_limit)
        pool = c.get("kv_pool") or {}
        if pool and not pool_saved:
            # the shared random-normal pools (a case's K/V of length n = the first n rows; the readers slice [:Lk])
            full = next(cc for cc in C if cc.get("kv_pool", {}).get("k") and cc["k"].shape[0] == POOL_N)
            np.save(out / "kv/pool_normal_k.npy", full["k"])
            fullv = next(cc for cc in C if cc.get("kv_pool", {}).get("v") and cc["v"].shape[0] == POOL_N)
            np.save(out / "kv/pool_normal_v.npy", fullv["v"])
            pool_saved = True
        if pool.get("k"):
            k_file = "kv/pool_normal_k.npy"
        else:
            k_file = f"kv/c{i:03d}_k.npy"
            np.save(out / k_file, k)
        if pool.get("v"):
            v_file = "kv/pool_normal_v.npy"
        else:
            v_file = f"kv/c{i:03d}_v.npy"
            np.save(out / v_file, v)
        np.save(out / f"calls/{i:05d}_q.npy", q[rows])
        np.save(out / f"calls/{i:05d}_out.npy", o[rows])
        calls.append({
            "seq": i, "req": i, "layer": 0, "layer_name": None, "family": c["family"], "case": c["case"],
            "phase": "decode" if Lq == 1 else "prefill", "Lq": Lq, "Lk": Lk, "rows": rows,
            "q_file": f"calls/{i:05d}_q.npy", "out_file": f"calls/{i:05d}_out.npy",
            "k_file": k_file, "v_file": v_file,
        })
        print(f"[{i:3d}] {c['family']:22s} {c['case']:28s} Lq={Lq:5d} Lk={Lk:5d} rows={len(rows)}", flush=True)

    # ---- mutations: one bit of one K / V / Q word -----------------------------------------------------------
    muts = []
    mut_targets = [c for c in calls if c["family"] in ("tile_boundaries", "max_shifts", "near_uniform") and c["phase"] == "decode"][:10]
    mut_targets += [c for c in calls if c["family"] == "tile_boundaries" and c["phase"] == "prefill"][:3]
    if args.natural:
        nat = json.load(open(Path(args.natural) / "index.json"))
        nat_calls = [c for c in nat["calls"] if c["phase"] == "decode"]
        for c in nat_calls[:: max(1, len(nat_calls) // 4)][:4]:
            c = dict(c)
            c["_base"] = args.natural
            c["family"] = "natural"
            c["case"] = f"seq{c['seq']}_L{c['layer']}_Lk{c['Lk']}"
            mut_targets.append(c)
    for c in mut_targets:
        base = Path(c.get("_base", out))
        q = np.load(base / c["q_file"])
        k = np.load(base / c["k_file"])[: c["Lk"]]
        v = np.load(base / c["v_file"])[: c["Lk"]]
        rows = c["rows"]
        if c.get("_base"):
            # natural q_file holds the saved rows only; rebuild a (Lq, H, D) query where the saved rows are real
            qq = np.zeros((c["Lq"], H, D), np.uint16)
            qq[rows] = q
            q = qq
        o_ref = production(q, k, v)[rows]
        for m in range(args.mutations_per_case):
            which = ("k", "v", "q")[m % 3]
            bit = (0, 7, 14, 0, 6, 15)[m % 6]
            arr = {"k": k, "v": v, "q": q}[which].copy()
            if which == "q":
                r = int(rng.choice(rows))
                idx = (r, int(rng.integers(0, H)), int(rng.integers(0, D)))
            else:
                idx = (int(rng.integers(0, c["Lk"])), int(rng.integers(0, KVH)), int(rng.integers(0, D)))
            old = int(arr[idx])
            new = old ^ (1 << bit)
            arr[idx] = new
            o_mut = production(arr if which == "q" else q, arr if which == "k" else k, arr if which == "v" else v)[rows]
            changed = int((o_mut != o_ref).sum())
            muts.append({
                "seq": c["seq"], "family": c["family"], "case": c["case"], "base_fixture": str(c.get("_base", "")) or None,
                "Lq": c["Lq"], "Lk": c["Lk"], "rows": rows, "operand": which, "index": list(idx), "bit": bit,
                "old_word": old, "new_word": new, "production_output_words_changed": changed,
                "production_out_mutated_file": f"calls/mut{len(muts):04d}_out.npy",
            })
            np.save(out / muts[-1]["production_out_mutated_file"], o_mut)
            print(f"  mut {len(muts)-1:3d} {c['case']:28s} {which}{list(idx)} bit {bit:2d}: {changed} output words changed", flush=True)

    index = {
        "schema": "verity-vllm/attn-fixture/v1", "kind": "stress+mutation (CONTRACT §11) against the production kernel",
        "generator": "verity_vllm_numerics/attn_stress.py", "seed": args.seed,
        "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "consts": {"num_heads": H, "num_kv_heads": KVH, "head_size": D, "block_size": BLOCK, "softmax_scale": SCALE, "causal": True,
                   "window_size": [-1, -1], "softcap": 0.0, "fa_version": 2, "num_splits": 1, "alibi": False},
        "production_call": "vllm.vllm_flash_attn.flash_attn_varlen_func(q, k_cache, v_cache, out, cu_seqlens_q=[0,Lq], max_seqlen_q=Lq, seqused_k=[Lk], max_seqlen_k=Lk, softmax_scale, causal=True, block_table, softcap=0, fa_version=2, num_splits=1) on a [nblocks,16,KVH,D] cache",
        "prefill_row_limit": args.prefill_row_limit,
        "n_calls": len(calls), "calls": calls, "n_mutations": len(muts), "seconds": time.time() - t0,
    }
    (out / "index.json").write_text(json.dumps(index, indent=1))
    (out / "mutations.json").write_text(json.dumps(muts, indent=1))
    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"n_calls": len(calls), "n_mutations": len(muts), "seconds": index["seconds"]}))


if __name__ == "__main__":
    main()
