"""GPU-truth fixture for `TopPMask_v1{V,S}` (`verity.ml.definitions.registry.topp_split`) across EVERY split geometry the vLLM
d9105ea80 split top-p pipeline can launch (S = 1, 2, 4, 8, 16, 32) with constructed pivot ties -- and the CPU checker the committed
test uses.

  produce (pod, torch venv):   verity-vllm topp-split-probe produce --out DIR [--record fstoch01_rows.npz]
  check   (any venv, no torch): verity-vllm topp-split-probe check --dir DIR [--record fstoch01_rows.npz]

Cases are REGENERATED on the CPU from the manifest (seed / V / kind / params), so the fixture ships only the kernel's outputs:
`keep` bits (the masked row is `where(keep, x_in, -inf)` -- the mask kernel stores the loaded fp32 x for kept lanes), the stats
buffer, the per-slice scalar chain of the instrumented mask kernel (M, Z, pivot_logit, dup, numkeep, numdup, num_preceding, fin) and
one bit per lane saying WHICH of the two per-thread Z values that lane's thread held (`z_group`).

Kinds:
  random     bf16-valued randn * 3 rows (the first validation's shape) at V in {1024, 3000, 4096, 128256};
  ties       a head of distinct large logits + a plateau of k IDENTICAL logits at lanes spread over the splits, both Z groups and
             (for S >= 2) straddling split boundaries (s*slice_len - 1, s*slice_len) + a small-mass tail; p chosen so the top-p
             boundary falls INSIDE the plateau (numkeep < numdup: the kernel's duplicate budget is exercised) at several offsets;
  record     the 3 preserved rows of ``--record`` (bf16 logits -> the real `apply_temperature` kernel) at every S (needs --record).
  drain      the REAL `_apply_topp_split` entry on a stacked batch of `ties` rows at B = 64, 18, 17, 9, 8, 7, 6, 5, 4, 3, 2, 1 and
             both sides of every live-count boundary of splits_for(B, num_sm) at the device's SM count (`drain_batches`; 142 SMs adds
             35/36, 188 adds 11/12, 23/24, 47/48) -- every row must equal the constexpr-S model at that S.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dataclasses import dataclass, field

from verity_vllm.commit import challenge as CH
from verity_vllm.config import option

f32 = np.float32
NEG_INF_BITS = 0xFF800000
SPLITS = (1, 2, 4, 8, 16, 32)
RANDOM_VOCABS = (1024, 3000, 4096, 128256)
TIE_VOCABS = (4096, 128256)
DRAIN_BATCHES = (64, 18, 17, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#: The manifest's base seed: every committed fixture's cases are sampled from it.
MANIFEST_SEED = 20260920


def drain_batches(num_sm: int) -> tuple[int, ...]:
    """The drain's batch sizes on a device of `num_sm` SMs, descending: DRAIN_BATCHES plus B and B + 1 for every B at which
    splits_for(B, num_sm) != splits_for(B + 1, num_sm), up to the split pipeline's largest batch."""
    from verity.ml.definitions.registry.topp_geometry import SPLIT_MAX_BATCH, splits_for
    edges = {b for n in range(1, SPLIT_MAX_BATCH) if splits_for(n, num_sm) != splits_for(n + 1, num_sm) for b in (n, n + 1)}
    return tuple(sorted(set(DRAIN_BATCHES) | edges, reverse=True))


def bf16_round(x: np.ndarray) -> np.ndarray:
    """RN-even rounding of fp32 values to bf16-representable fp32 (finite inputs)."""
    b = np.asarray(x, dtype=f32).view(np.uint32).astype(np.uint64)
    lsb = (b >> 16) & 1
    b = ((b + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32)
    return b.view(f32)


def thread_lane(v: int, V: int, S: int, spt: int, block: int = 8192) -> int:
    """Warp lane (0..31) of the thread holding vocabulary lane v in the mask kernel's 1-D [BLOCK] load of its slice."""
    slice_len = (V + S - 1) // S
    j = (v - (v // slice_len) * slice_len) % block
    return (j // spt) % 32


def gen_random(seed: int, V: int) -> np.ndarray:
    g = CH.generator(seed)
    return bf16_round((g.standard_normal(V) * 3.0).astype(f32))


def gen_ties(seed: int, V: int, S: int, k: int, keep_frac: float, head: int = 8, group_bias: int | None = None, plateau: float = 3.5) -> tuple[np.ndarray, float]:
    """Row with `k` identical plateau logits at lanes spread over the S slices (+ boundary pairs), and the p that puts the top-p
    boundary `keep_frac` of the way into the plateau.  `group_bias` (0/1) puts every plateau lane on a thread of that Z group."""
    from verity.ml.definitions.registry.topp_geometry import spt_for
    g = CH.generator(seed)
    x = (g.standard_normal(V) * 1.0 - 6.0).astype(f32)             # tail: small mass
    slice_len = (V + S - 1) // S
    spt = spt_for(V, S)
    lanes: list[int] = []
    for s in range(1, S):                                             # ties straddling split boundaries
        lanes += [s * slice_len - 1, s * slice_len]
    cand = g.permutation(V)
    for v in cand:                                                    # then spread lanes (both Z groups unless biased)
        if len(lanes) >= k:
            break
        if v in lanes or v < head:
            continue
        if group_bias is not None and ((thread_lane(int(v), V, S, spt) % max(S, 2)) >= max(S, 2) // 2) != bool(group_bias):
            continue
        lanes.append(int(v))
    lanes = sorted(set(lanes))[:k]
    while len(lanes) < k:                                             # (tiny V with many boundaries) top up
        v = int(g.integers(head, V))
        if v not in lanes:
            lanes.append(v)
    lanes = sorted(lanes)
    hv = np.linspace(9.0, 5.0, head).astype(f32)                      # distinct head
    head_lanes = [v for v in range(V) if v not in lanes][:head]
    x[head_lanes] = hv
    x[lanes] = f32(plateau)
    x = bf16_round(x)
    xd = x.astype(np.float64); m = xd.max(); e = np.exp(xd - m); pr = e / e.sum()
    head_mass = float(pr[head_lanes].sum()); pm = float(pr[lanes[0]])
    p = head_mass + pm * (keep_frac * k)
    p = float(f32(min(max(p, 0.05), 0.999)))
    return x, p


def build_manifest(seed: int, with_record: bool = False) -> list[dict]:
    cases: list[dict] = []
    cid = 0
    for S in SPLITS:
        for V in RANDOM_VOCABS:
            for r in range(2 if V == 128256 else 3):
                cases.append({"id": cid, "kind": "random", "S": S, "V": V, "seed": seed + 1000 * S + 10 * V + r, "p": 0.95 if r != 1 else 0.9}); cid += 1
        for V in TIE_VOCABS:
            for (k, kf, gb, pl) in ((3, 0.5, None, 3.5), (12, 0.25, None, 3.5), (24, 0.5, None, 3.5), (24, 0.95, None, 3.5), (40, 0.6, None, 3.5),
                                    (16, 0.5, 0, 3.5), (16, 0.5, 1, 3.5), (24, 0.5, None, 6.5), (12, 0.75, None, 7.5)):
                if V == 128256 and k in (3, 40):
                    continue
                cases.append({"id": cid, "kind": "ties", "S": S, "V": V, "seed": seed + 7 * cid, "k": k, "keep_frac": kf, "group_bias": gb, "plateau": pl}); cid += 1
        if with_record:
            for r in range(3):
                cases.append({"id": cid, "kind": "record", "S": S, "V": 128256, "row": r}); cid += 1
    cases.append({"id": cid, "kind": "drain", "V": 128256, "seed": seed + 99991, "k": 24, "keep_frac": 0.5, "p": None}); cid += 1
    return cases


def case_row(c: dict, record=None) -> tuple[np.ndarray, float]:
    if c["kind"] == "random":
        return gen_random(c["seed"], c["V"]), float(c["p"])
    if c["kind"] == "ties":
        x, p = gen_ties(c["seed"], c["V"], c["S"], c["k"], c["keep_frac"], group_bias=c.get("group_bias"), plateau=c.get("plateau", 3.5))
        return x, p
    if c["kind"] == "record":
        z, meta = record
        k = list(meta)[c["row"]]
        bf = (z[k + "_bf16_logits_bits"].astype(np.uint32) << 16).view(f32)
        return bf, float(meta[k]["device"]["top_p"])            # (tempering is done by the producer's real kernel / the checker's model)
    raise ValueError(c["kind"])


def record_temperature(c: dict, record) -> float:
    z, meta = record
    return float(meta[list(meta)[c["row"]]]["device"]["temperature"])


# ------------------------------------------------------------------------------------------------ produce (GPU)


def _mask_probe_kernel():
    """The instrumented copy of vLLM's split top-p mask kernel that `produce` launches beside the real one.  Built by the call:
    triton and vLLM exist only on the pod."""
    import triton
    import triton.language as tl
    from vllm.v1.sample.ops import topk_topp_triton as TT

    @triton.jit
    def _mask_probe(LOGITS, LOGITS_STRIDE_0, STATS, PARTS, P, Z_OUT, KEEP_OUT, SC_OUT,
                    S: tl.constexpr, F: tl.constexpr, NUM_ROUNDS: tl.constexpr, VOCAB_SIZE: tl.constexpr, BLOCK: tl.constexpr):
        # verbatim `_topp_sb_mask_kernel` (HAS_K=False) + stores of the thread's Z per lane and the per-slice scalar chain
        pid = tl.program_id(0)
        row = pid // S
        slice_id = pid % S
        p = tl.load(P + row)
        if p >= 1.0:
            return
        M, Z, mn, num_finite = TT._topp_sb_row_stats(STATS, row, S)
        if (Z <= 0.0) or (num_finite < 1.0):
            return
        max_prob = 1.0 / Z
        min_prob = tl.maximum(tl.exp(mn - M) / Z, 1e-30)
        if max_prob <= min_prob * (1.0 + 1e-6):
            return
        lo = min_prob
        hi = max_prob
        best = -1.0
        best_mass = 0.0
        best_minl = 1.0
        best_nmin = 0.0
        done = 0.0
        nomask = 0.0
        pivot = 0.0
        dup = 1.0
        numdup = 1.0
        numkeep = 1.0
        fin = -1.0
        best_fin = -1.0
        for j in range(NUM_ROUNDS):
            if done == 0.0:
                (done, nomask, pivot, dup, numdup, numkeep, lo, hi, best, best_mass, best_minl, best_nmin, sel_fidx, best_sel) = TT._topp_sb_combine(
                    PARTS + (row * NUM_ROUNDS + j) * S * F * 3, Z, p, lo, hi, best, best_mass, best_minl, best_nmin, S=S, F=F, IS_LAST=j == NUM_ROUNDS - 1)
                if sel_fidx >= 0.0:
                    fin = j * F + sel_fidx
                if best_sel >= 0.0:
                    best_fin = j * F + best_sel
        if nomask != 0.0:
            return
        if fin < 0.0:
            fin = best_fin
        logZ = tl.log(Z)
        pivot_logit = tl.log(pivot) + logZ + M
        ties = numkeep < numdup
        num_preceding = 0.0
        if ties:
            fin_round = tl.cast(fin / F, tl.int32)
            fin_slot = tl.cast(fin - tl.cast(fin_round, tl.float32) * F, tl.int32)
            pbase = PARTS + ((row * NUM_ROUNDS + fin_round) * S * F + fin_slot) * 3
            sidx = tl.arange(0, S)
            minl_s = tl.load(pbase + sidx * (F * 3) + 1)
            nmin_s = tl.load(pbase + sidx * (F * 3) + 2)
            dupc_s = tl.where(tl.abs(minl_s - dup) < 1e-9, nmin_s, 0.0)
            num_preceding = tl.sum(tl.where(sidx < slice_id, dupc_s, 0.0))
        sb = SC_OUT + pid * 8
        tl.store(sb + 0, M)
        tl.store(sb + 1, Z)
        tl.store(sb + 2, pivot_logit)
        tl.store(sb + 3, dup)
        tl.store(sb + 4, numkeep)
        tl.store(sb + 5, numdup)
        tl.store(sb + 6, num_preceding)
        tl.store(sb + 7, fin)
        slice_len = (VOCAB_SIZE + S - 1) // S
        start = slice_id * slice_len
        end = tl.minimum(start + slice_len, VOCAB_SIZE)
        ROW = LOGITS + row.to(tl.int64) * LOGITS_STRIDE_0
        offs = tl.arange(0, BLOCK)
        for i in range(start, end, BLOCK):
            mask_n = i + offs < end
            x = tl.load(ROW + i + offs, mask=mask_n, other=-float("inf"))
            keep = x > pivot_logit
            zb = tl.zeros([BLOCK], dtype=tl.float32) + Z
            tl.store(Z_OUT + i + offs, zb, mask=mask_n)
            if ties:
                dmask = tl.exp(x - M) / Z == dup
                cum = tl.cumsum(dmask.to(tl.int32)).to(tl.float32) + num_preceding
                keep_dup = dmask & (cum <= numkeep)
                keep = keep & (~dmask | keep_dup)
            tl.store(KEEP_OUT + i + offs, keep.to(tl.int32), mask=mask_n)

    return _mask_probe


def produce(out: str, record_npz: str | None, seed: int) -> None:
    import torch
    import triton
    from vllm.utils.platform_utils import num_compute_units
    from vllm.v1.sample.ops import topk_topp_triton as TT
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature

    _mask_probe = _mask_probe_kernel()
    dev = torch.device("cuda:0")
    num_sm = num_compute_units(dev.index)
    F, R = TT._SPLIT_FANOUT, TT._SPLIT_ROUNDS
    record = None
    if record_npz:
        record = (np.load(record_npz), json.load(open(record_npz + ".json")))
    cases = build_manifest(seed, with_record=record is not None)
    o = Path(out); o.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    summary = []

    def run_pipeline(x_dev: "torch.Tensor", p_dev: "torch.Tensor", S: int, V: int):
        """stats -> 5 steps -> (instrumented mask on a copy | real mask kernel in place); returns (keep_real, keep_probe, zlane, stats, sc)."""
        stats = x_dev.new_empty((S, 4)); parts = x_dev.new_empty((1, R, S, F, 3))
        TT._topp_sb_stats_kernel[(S,)](x_dev, x_dev.stride(0), stats, x_dev, p_dev, HAS_K=False, VOCAB_SIZE=V, S=S, BLOCK=8192, num_warps=8)
        for r in range(R):
            TT._topp_sb_step_kernel[(S,)](x_dev, x_dev.stride(0), stats, parts, x_dev, p_dev, r, HAS_K=False, S=S, F=F, NUM_ROUNDS=R, VOCAB_SIZE=V, BLOCK=2048, num_warps=8)
        Z_OUT = torch.zeros(V, dtype=torch.float32, device=dev)
        KEEP_OUT = torch.full((V,), -1, dtype=torch.int32, device=dev)
        SC_OUT = torch.full((S * 8,), float("nan"), dtype=torch.float32, device=dev)
        probe_in = x_dev.clone()
        _mask_probe[(S,)](probe_in, probe_in.stride(0), stats, parts, p_dev, Z_OUT, KEEP_OUT, SC_OUT, S=S, F=F, NUM_ROUNDS=R, VOCAB_SIZE=V, BLOCK=8192, num_warps=8)
        TT._topp_sb_mask_kernel[(S,)](x_dev, x_dev.stride(0), stats, parts, x_dev, p_dev, HAS_K=False, MASK_VALUE=float("-inf"), S=S, F=F, NUM_ROUNDS=R, VOCAB_SIZE=V, BLOCK=8192, num_warps=8)
        torch.cuda.synchronize()
        keep_real = (x_dev.reshape(-1).view(torch.int32).cpu().numpy().view(np.uint32) != NEG_INF_BITS)
        kp = KEEP_OUT.cpu().numpy()
        keep_probe = None if (kp < 0).any() else (kp == 1)             # early return -> untouched
        return keep_real, keep_probe, Z_OUT.cpu().numpy().view(np.uint32), stats.cpu().numpy(), SC_OUT.cpu().numpy().reshape(S, 8)

    t0 = time.time()
    for c in cases:
        if c["kind"] == "drain":
            continue
        S, V = c["S"], c["V"]
        x, p = case_row(c, record)
        if c["kind"] == "record":
            bf = torch.from_numpy((x.view(np.uint32) >> 16).astype(np.int16)).view(torch.bfloat16).to(dev).reshape(1, V)
            xd = torch.empty((1, V), dtype=torch.float32, device=dev).copy_(bf)
            apply_temperature(xd, torch.zeros(1, dtype=torch.int32, device=dev), torch.tensor([record_temperature(c, record)], dtype=torch.float32, device=dev))
            x_in = xd.reshape(-1).view(torch.int32).cpu().numpy().view(np.uint32).copy()
        else:
            xd = torch.from_numpy(x.view(np.int32).copy()).view(torch.float32).to(dev).reshape(1, V)
            x_in = x.view(np.uint32).copy()
        p_dev = torch.tensor([p], dtype=torch.float32, device=dev)
        keep_real, keep_probe, zl, stats, sc = run_pipeline(xd, p_dev, S, V)
        zu = np.unique(zl)
        pre = f"c{c['id']}_"
        arrays[pre + "keep"] = np.packbits(keep_real, bitorder="little")
        arrays[pre + "stats"] = stats
        arrays[pre + "sc"] = sc
        arrays[pre + "zvals"] = zu
        if len(zu) == 2:
            arrays[pre + "zbit"] = np.packbits(zl == zu[1], bitorder="little")
        if c["kind"] == "record":
            c["temperature"] = record_temperature(c, record)
            c["x_in_sha"] = __import__("hashlib").sha256(x_in.tobytes()).hexdigest()[:16]    # the checker's TemperatureScale model must reproduce it
        c["p"] = float(p)
        c["kept"] = int(keep_real.sum()); c["z_nunique"] = int(len(zu))
        c["probe_keep_eq_real"] = None if keep_probe is None else bool(np.array_equal(keep_probe, keep_real))
        c["ties"] = bool(np.nanmax(sc[:, 4]) < np.nanmax(sc[:, 5])) if not np.isnan(sc).all() else None
        c["numkeep"] = None if np.isnan(sc[0, 4]) else float(sc[0, 4]); c["numdup"] = None if np.isnan(sc[0, 5]) else float(sc[0, 5])
        summary.append(c)
        print(json.dumps({k: c[k] for k in ("id", "kind", "S", "V", "p", "kept", "z_nunique", "ties", "numkeep", "numdup", "probe_keep_eq_real")}), flush=True)
    # drain: the real entry point on a stacked batch
    dc = [c for c in cases if c["kind"] == "drain"][0]
    V = dc["V"]
    rows = []
    for b in range(max(DRAIN_BATCHES)):
        xb, pb = gen_ties(dc["seed"] + b, V, 16 if b % 2 == 0 else 32, dc["k"], dc["keep_frac"])
        rows.append((xb, pb))
    X = torch.from_numpy(np.stack([r[0] for r in rows]).view(np.int32).copy()).view(torch.float32).to(dev)
    P = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=dev)
    dc["rows"] = [{"seed": dc["seed"] + b, "S_gen": 16 if b % 2 == 0 else 32, "p": float(rows[b][1])} for b in range(len(rows))]
    dc["num_sm"] = int(num_sm); dc["batches"] = []
    for B in drain_batches(num_sm):
        L = X[:B].clone()
        TT._apply_topp_split(L, None, P[:B], float("-inf"), num_sm)
        torch.cuda.synchronize()
        keep = (L.view(torch.int32).cpu().numpy().view(np.uint32) != NEG_INF_BITS)
        s = 1
        while s < TT._SPLIT_MAX_SPLITS and s * 2 * B <= num_sm:
            s *= 2
        arrays[f"drain_B{B}_keep"] = np.packbits(keep, axis=1, bitorder="little")
        dc["batches"].append({"B": B, "S": s, "kept": [int(k.sum()) for k in keep]})
        print(json.dumps({"drain": B, "S": s, "kept": [int(k.sum()) for k in keep][:8]}), flush=True)
    summary.append(dc)
    np.savez_compressed(o / "fixture.npz", **arrays)
    (o / "manifest.json").write_text(json.dumps({
        "cases": summary, "seed": seed, "num_sm": int(num_sm), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "triton": triton.__version__, "vllm": __import__("vllm").__version__, "record": os.path.basename(record_npz) if record_npz else None,
        "secs": round(time.time() - t0, 1)}, indent=1))
    print("cases", len(summary), "->", o / "fixture.npz", f"{(o / 'fixture.npz').stat().st_size / 1024:.0f} KB", f"{time.time() - t0:.1f}s")


# ------------------------------------------------------------------------------------------------ check (CPU)


def check(d: str, record_npz: str | None = None, verbose: bool = True) -> dict:
    """Model vs fixture: masked row (keep), stats, per-lane Z group, drain batches.  Returns the report dict (ok = every check passed)."""
    from verity.ml.definitions.registry import topp_split as TS
    from verity.ml.definitions.registry import topp_geometry
    dd = Path(d)
    man = json.loads((dd / "manifest.json").read_text())
    z = np.load(dd / "fixture.npz")
    record = None
    if record_npz and os.path.exists(record_npz):
        record = (np.load(record_npz), json.load(open(record_npz + ".json")))
    rep = {"n": 0, "mask_ok": 0, "stats_ok": 0, "zgroup_ok": 0, "zgroup_checked": 0, "ties_cases": 0, "boundary_tie_cases": 0, "fail": [], "drain": []}
    for c in man["cases"]:
        if c["kind"] == "drain":
            continue
        pre = f"c{c['id']}_"
        S, V, p = c["S"], c["V"], c["p"]
        if c["kind"] == "record":
            if record is None:
                continue
            bf, _ = case_row(c, record)
            x_in = TS.tl_div(bf, f32(c["temperature"]), TS.Cfg())          # `_temperature_kernel` = div.full.f32 (TemperatureScale_v1)
            if c.get("x_in_sha") and __import__("hashlib").sha256(x_in.view(np.uint32).tobytes()).hexdigest()[:16] != c["x_in_sha"]:
                rep["fail"].append({"id": c["id"], "kind": "record", "S": S, "x_in": "TemperatureScale model != pod apply_temperature"})
        else:
            x_in, _ = case_row(c)
        cfg = TS.Cfg.for_vocab(V, S)
        out, stats, parts, tr = TS.topp_split_row(x_in, p, S, cfg=cfg)
        keep_k = np.unpackbits(z[pre + "keep"], bitorder="little")[:V].astype(bool)
        keep_m = out.view(np.uint32) != NEG_INF_BITS
        mask_ok = bool(np.array_equal(keep_m, keep_k))
        stats_ok = bool(np.array_equal(stats.view(np.uint32), z[pre + "stats"].view(np.uint32)))
        rep["n"] += 1; rep["mask_ok"] += mask_ok; rep["stats_ok"] += stats_ok
        if c.get("ties"):
            rep["ties_cases"] += 1
            if c["kind"] == "ties" and S >= 2:
                rep["boundary_tie_cases"] += 1
        zg_ok = None
        if pre + "zbit" in z.files and V % 16 == 0:
            # (V % 16 != 0: the real kernels' x load is sizePerThread 1 but the probe's fresh, 16-B aligned Z_OUT store may vectorize
            #  to 4, so its per-lane pattern reads through a different thread layout -- ambiguous, not checked; the masked row is.)
            zbit = np.unpackbits(z[pre + "zbit"], bitorder="little")[:V].astype(np.int64)
            zvals = z[pre + "zvals"]
            M, Zh, mn, nf = TS.row_stats(stats, S, cfg)
            # the fixture's bit 1 = the LARGER Z word; map the model's group -> its Z word and compare per lane
            lane = np.array([thread_lane(v, V, S, cfg.spt_x) for v in range(V)])
            grp = topp_geometry.z_group(lane, S)
            zw_model = np.asarray(Zh, f32).view(np.uint32)[grp]
            zw_fix = zvals[zbit]
            zg_ok = bool(np.array_equal(zw_model, zw_fix))
            rep["zgroup_checked"] += 1; rep["zgroup_ok"] += zg_ok
        if not (mask_ok and stats_ok and (zg_ok in (None, True))):
            diff = np.nonzero(keep_m != keep_k)[0]
            rep["fail"].append({"id": c["id"], "kind": c["kind"], "S": S, "V": V, "p": p, "mask_ok": mask_ok, "stats_ok": stats_ok, "zgroup_ok": zg_ok,
                                "lanes_differ": int(len(diff)), "first": [int(i) for i in diff[:8]], "numkeep_model": tr.get("numkeep"), "numdup_model": tr.get("numdup"),
                                "numkeep_kernel": c.get("numkeep"), "numdup_kernel": c.get("numdup")})
        if verbose:
            print(json.dumps({"id": c["id"], "kind": c["kind"], "S": S, "V": V, "mask_ok": mask_ok, "stats_ok": stats_ok, "zgroup_ok": zg_ok, "ties": c.get("ties"),
                              "kept": c.get("kept")}), flush=True)
    dc = [c for c in man["cases"] if c["kind"] == "drain"]
    if dc:
        dc = dc[0]; V = dc["V"]
        for b in dc["batches"]:
            B, S = b["B"], b["S"]
            assert S == topp_geometry.splits_for(B, dc["num_sm"]), (B, S)
            keep_k = z[f"drain_B{B}_keep"]
            ok_rows = 0
            for i in range(B):
                r = dc["rows"][i]
                x_i, p_i = gen_ties(r["seed"], V, r["S_gen"], dc["k"], dc["keep_frac"])
                out = TS.topp_split_row(x_i, p_i, S)[0]
                ok_rows += int(np.array_equal(out.view(np.uint32) != NEG_INF_BITS, np.unpackbits(keep_k[i], bitorder="little")[:V].astype(bool)))
            rep["drain"].append({"B": B, "S": S, "rows_ok": ok_rows})
            if ok_rows != B:
                rep["fail"].append({"drain_B": B, "S": S, "rows_ok": ok_rows})
            if verbose:
                print(json.dumps({"drain_B": B, "S": S, "rows_ok": f"{ok_rows}/{B}"}), flush=True)
    rep["ok"] = not rep["fail"] and rep["mask_ok"] == rep["n"] and rep["stats_ok"] == rep["n"] and rep["zgroup_ok"] == rep["zgroup_checked"]
    if verbose:
        print("TOTALS", json.dumps({k: v for k, v in rep.items() if k not in ("fail", "drain")}), "fail:", rep["fail"][:5])
    return rep


@dataclass(frozen=True, kw_only=True)
class ProduceOptions:
    out: str = option("--out")
    record: str | None = option("--record", default=None)
    seed: int = option("--seed", type=int, default=MANIFEST_SEED)
    cmd: str = field(default="produce", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    dir: str = option("--dir")
    record: str | None = option("--record", default=None)
    out: str | None = option("--out", default=None)
    cmd: str = field(default="check", init=False)


@dataclass(frozen=True, kw_only=True)
class ManifestOptions:
    record: bool = option("--record", default=False, action="store_true")
    cmd: str = field(default="manifest", init=False)


COMMANDS = {"produce": ProduceOptions, "check": CheckOptions, "manifest": ManifestOptions}


def main(args: ProduceOptions | CheckOptions | ManifestOptions) -> None:
    if args.cmd == "produce":
        produce(args.out, args.record, args.seed)
    elif args.cmd == "check":
        rep = check(args.dir, args.record)
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=1))
        sys.exit(0 if rep["ok"] else 1)
    else:
        for c in build_manifest(MANIFEST_SEED, with_record=args.record):
            print(json.dumps(c))
