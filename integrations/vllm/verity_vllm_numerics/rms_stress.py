"""CONTRACT §11 stress inputs and one-bit mutations for the Triton rms_norm kernel (W11; runs on the GPU pod).

Every case is run through the production entry point ``rms_norm_batch_invariant(x, weight, eps)`` (which
launches ``_rms_norm_kernel`` exactly as ``RMSNorm.forward_cuda`` does) and stored in the same fixture layout
as ``rms_capture.py`` (``calls/NNNNN_{x,out}.npy``, ``weights/``, ``index.json`` with ``family`` / ``case``).
The number of compiled Triton specialisations is recorded before and after every family so a case that
silently used a *different* compilation of the kernel is flagged (``new_specialisations``).

Families: ``zeros``, ``absorption`` (large + many small squares, order-sensitive sums), ``rounding_boundaries``
(rows whose pre-cast fp32 outputs land within 1 ulp of a BF16 tie according to the model; plus exact ties),
``subnormals`` (BF16 subnormal inputs, fp32-subnormal squares, ``eps = 0`` so the flushed sqrt input matters),
``extremes`` (overflowing squares, BF16 max, inf / NaN inputs and weights), ``near_uniform`` (constant rows
across the exponent range, one-ulp perturbations), ``random_scale`` (rows scaled by 2^-60 .. 2^60),
``n_cols`` (16 .. 4096, partial and multiple 1024-column chunks), ``beyond_natural_range`` (4096 rows in one
launch).  ``mutations.json``: one-bit flips of one x word / one w word / an eps change, each re-run through
the kernel, with the number of output words that moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verity_vllm_numerics import rms_triton_model as rm  # noqa: E402

BF16_MAX = 0x7F7F
BF16_INF = 0x7F80
BF16_NAN = 0x7FC0


def words(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def to_t(wds: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(wds).view(np.int16)).view(torch.bfloat16).cuda()


def bf16(v) -> np.ndarray:
    """float(s) -> BF16 words (RNE via torch)."""
    return words(torch.as_tensor(np.asarray(v, dtype=np.float32)).to(torch.bfloat16))


def f32(wds: np.ndarray) -> np.ndarray:
    return (wds.astype(np.uint32) << 16).view(np.float32)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def n_specialisations(kernel) -> int:
    n = 0
    dc = getattr(kernel, "device_caches", None)
    if dc:
        for v in dc.values():
            try:
                n += len(v[0])
            except Exception:  # noqa: BLE001
                pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tables", required=True, help="dir with mufu_sqrt_delta_int8.xz / mufu_rcp_delta_int8.xz (for the boundary search)")
    ap.add_argument("--natural", help="natural fixture dir (its weight is reused; some rows are mutated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boundary-search-rows", type=int, default=40000)
    ap.add_argument("--big-rows", type=int, default=1024, help="rows of the single large launch (beyond_natural_range)")
    ap.add_argument("--kernel", default="triton", choices=["triton", "cuda"],
                    help="triton: rms_norm_batch_invariant (the profile's kernel); cuda: torch.ops._C.rms_norm (vllm::rms_norm_kernel, not launched by the engine)")
    ap.add_argument("--replay-natural", help="instead of the stress families: run the natural fixture's x rows through the kernel (CUDA op reference outputs)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tables = rm.RmsTables.load(args.tables)

    from vllm.model_executor.determinism import batch_invariant as bi  # noqa: E402

    kern = bi._rms_norm_kernel
    if args.kernel == "cuda":
        import vllm._custom_ops  # noqa: F401,E402  (registers torch.ops._C)

        def launch(xt: torch.Tensor, wt: torch.Tensor, eps: float) -> torch.Tensor:
            o = torch.empty_like(xt)
            torch.ops._C.rms_norm(o, xt, wt, eps)
            return o
    else:
        def launch(xt: torch.Tensor, wt: torch.Tensor, eps: float) -> torch.Tensor:
            return bi.rms_norm_batch_invariant(xt, wt, eps)
    N = 1536
    EPS = 1e-6

    # weights: the natural one (layers.0.input_layernorm) when available, plus synthetic ones
    weights: dict[str, np.ndarray] = {}
    if args.natural:
        nidx = json.loads((Path(args.natural) / "index.json").read_text())
        for k, v in nidx["weights"].items():
            weights["natural"] = np.load(Path(args.natural) / v["file"]).view(np.uint16)
            break
    weights["ones"] = bf16(np.ones(N))
    weights["random"] = bf16(rng.normal(1.0, 0.3, N))
    weights["wide"] = bf16(np.exp2(rng.uniform(-20, 20, N)) * rng.choice([-1, 1], N))
    weights["max"] = np.full(N, BF16_MAX, np.uint16)
    weights["zeros"] = np.zeros(N, np.uint16)
    weights["nan_one"] = bf16(np.ones(N)); weights["nan_one"][777] = BF16_NAN
    wmeta = {}
    for k, v in weights.items():
        np.save(out / f"weights/{k}.npy", v)
        wmeta[k] = {"file": f"weights/{k}.npy", "n": int(v.shape[0])}

    calls: list[dict] = []
    spec_log: list[dict] = []

    def run(family: str, case: str, x: np.ndarray, wkey: str, eps: float = EPS, note: str | None = None) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.uint16)
        if x.ndim == 1:
            x = x[None]
        n = x.shape[1]
        w = weights[wkey]
        if w.shape[0] != n:
            wk2 = f"{wkey}_n{n}"
            if wk2 not in weights:
                if wkey == "ones":
                    weights[wk2] = bf16(np.ones(n))
                else:
                    weights[wk2] = bf16(rng.normal(1.0, 0.3, n))
                np.save(out / f"weights/{wk2}.npy", weights[wk2])
                wmeta[wk2] = {"file": f"weights/{wk2}.npy", "n": n}
            wkey, w = wk2, weights[wk2]
        n0 = n_specialisations(kern)
        xt = to_t(x)
        o = launch(xt, to_t(w), eps)
        torch.cuda.synchronize()
        n1 = n_specialisations(kern)
        seq = len(calls)
        rec = {"seq": seq, "family": family, "case": case, "rows": int(x.shape[0]), "n_cols": n, "eps": float(eps),
               "weight": wkey, "x_file": f"calls/{seq:05d}_x.npy", "out_file": f"calls/{seq:05d}_out.npy",
               "new_specialisations": n1 - n0, "specialisations_after": n1}
        if note:
            rec["note"] = note
        ow = words(o)
        np.save(out / rec["x_file"], x)
        np.save(out / rec["out_file"], ow)
        calls.append(rec)
        return ow

    def randn_rows(rows: int, n: int = N, scale: float = 1.0) -> np.ndarray:
        return bf16(rng.normal(0, scale, (rows, n)))

    # warm-up / compile with the natural shape (same specialisation as the engine: n_cols=1536 % 16 == 0, contiguous)
    _ = launch(to_t(randn_rows(4)), to_t(weights["ones"]), EPS)
    torch.cuda.synchronize()
    spec_log.append({"after": "warmup", "n_specialisations": n_specialisations(kern)})

    if args.replay_natural:
        # the natural fixture's inputs (layers.0.input_layernorm rows of the serving run) through this kernel
        nat = Path(args.replay_natural)
        ridx = json.loads((nat / "index.json").read_text())
        if "natural" not in weights:
            weights["natural"] = np.load(nat / next(iter(ridx["weights"].values()))["file"]).view(np.uint16)
            np.save(out / "weights/natural.npy", weights["natural"])
            wmeta["natural"] = {"file": "weights/natural.npy", "n": int(weights["natural"].shape[0])}
        for c in ridx["calls"]:
            x = np.load(nat / c["x_file"]).view(np.uint16).reshape(c["rows"], c["n_cols"])
            rec_out = run(c["phase"], f"natural_seq_{c['seq']:05d}", x, "natural", eps=c["eps"])
            calls[-1]["natural_seq"] = c["seq"]
            calls[-1]["phase"] = c["phase"]
        import triton  # noqa: E402
        import vllm  # noqa: E402

        meta = {"schema": "verity-vllm/rmsnorm-fixture/v1", "kind": "natural-inputs-replayed", "kernel": args.kernel, "seed": args.seed,
                "vllm": vllm.__version__, "torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "entry_point": "torch.ops._C.rms_norm(out, x, weight, eps)" if args.kernel == "cuda" else "rms_norm_batch_invariant",
                "source_fixture": str(nat), "note": "inputs are the natural fixture's rows (the engine never launches this kernel in the pinned profile); outputs written by this kernel",
                "weights": wmeta, "n_calls": len(calls), "calls": calls}
        (out / "index.json").write_text(json.dumps(meta, indent=1))
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
        print(json.dumps({"n_calls": len(calls), "rows": sum(c["rows"] for c in calls), "total_bytes": sum(p.stat().st_size for p in files)}, indent=1))
        return

    # ---- zeros ----------------------------------------------------------------------------------
    run("zeros", "all_zero", np.zeros((4, N), np.uint16), "natural" if "natural" in weights else "random")
    run("zeros", "all_zero_w_random", np.zeros((2, N), np.uint16), "random")
    run("zeros", "neg_zero", np.full((2, N), 0x8000, np.uint16), "random")
    single = np.zeros((16, N), np.uint16)
    pos = [0, 1, 7, 8, 9, 31, 32, 255, 256, 511, 512, 1023, 1024, 1025, 1535, 1000]
    for i, p in enumerate(pos):
        single[i, p] = bf16(1.0)[()]
    run("zeros", "single_one_at_positions", single, "random", note=f"positions {pos}")
    run("zeros", "zero_weight", randn_rows(4), "zeros")
    run("zeros", "zero_eps_zero_row", np.zeros((2, N), np.uint16), "random", eps=0.0)

    # ---- absorption (order-sensitive fp32 sums) -----------------------------------------------------
    absorb = np.tile(bf16(1.0), (16, N)).astype(np.uint16)
    big_pos = [0, 1, 8, 9, 32, 33, 64, 511, 512, 1023, 1024, 1025, 1100, 1535, 700, 300]
    for i, p in enumerate(big_pos):
        absorb[i, p] = bf16(4096.0)[()]  # 4096^2 = 2^24: every added 1.0 is a half-ulp
    run("absorption", "2^24_plus_ones_positions", absorb, "ones", note=f"big at {big_pos}")
    absorb2 = np.tile(bf16(3.0), (8, N)).astype(np.uint16)  # 9.0 each: sum accumulates with odd roundings
    for i in range(8):
        absorb2[i, rng.integers(0, N, 4)] = bf16(2048.0)[()]
    run("absorption", "2^22_plus_nines", absorb2, "random")
    wide = np.exp2(rng.uniform(-8, 8, (64, N))) * rng.choice([-1, 1], (64, N))
    run("absorption", "wide_dynamic_range_2^-8..2^8", bf16(wide), "random")
    wide2 = np.exp2(rng.uniform(-12, 12, (64, N))) * rng.choice([-1, 1], (64, N))
    run("absorption", "wide_dynamic_range_2^-12..2^12", bf16(wide2), "natural" if "natural" in weights else "random")
    # one dominant element, rest tiny (small squares below half an ulp of the big square)
    dom = bf16(rng.normal(0, 1e-3, (16, N)))
    dom[:, 5] = bf16(100.0)[()]
    run("absorption", "dominant_element", dom, "random")

    # ---- rounding boundaries (model-guided search, checked against the hardware) ----------------------
    t0 = time.perf_counter()
    cand = randn_rows(args.boundary_search_rows)
    wkey = "natural" if "natural" in weights else "random"
    md, pre = rm.rms_norm_rows(cand, weights[wkey], EPS, tables, want_pre=True)
    u = pre.view(np.uint32)
    low = u & 0xFFFF
    # distance (in fp32 ulps) to the BF16 tie 0x8000 in the low half; exact ties = 0
    dist = np.minimum(np.abs(low.astype(np.int64) - 0x8000), 0x10000)
    finite = np.isfinite(pre)
    exact_tie_rows = np.nonzero(((dist == 0) & finite).any(axis=1))[0]
    near_rows = np.nonzero(((dist <= 1) & finite).any(axis=1))[0]
    near2_rows = np.nonzero(((dist <= 4) & finite).any(axis=1))[0]
    sel_exact = exact_tie_rows[:128]
    sel_near = np.setdiff1d(near_rows, sel_exact)[:192]
    sel_near2 = np.setdiff1d(near2_rows, np.concatenate([sel_exact, sel_near]))[:128]
    if len(sel_exact):
        run("rounding_boundaries", "exact_bf16_ties_pre_cast", cand[sel_exact], wkey,
            note=f"{len(exact_tie_rows)} of {args.boundary_search_rows} random rows have >=1 output exactly on a BF16 tie (model)")
    if len(sel_near):
        run("rounding_boundaries", "within_1ulp_of_tie", cand[sel_near], wkey, note=f"{len(near_rows)} rows within 1 fp32 ulp")
    if len(sel_near2):
        run("rounding_boundaries", "within_4ulp_of_tie", cand[sel_near2], wkey, note=f"{len(near2_rows)} rows within 4 fp32 ulp")
    # rows whose mean_eps sits at an exponent boundary of the sqrt (parity switch): all-equal rows with x^2 * 1 = 2^k
    par = np.stack([np.full(N, np.exp2(k / 2.0)) for k in range(-20, 21)])  # x = 2^(k/2): x^2 = 2^k, mean = 2^k
    run("rounding_boundaries", "sqrt_exponent_parity_rows", bf16(par), "ones", note="x = 2^(k/2), k = -20..20; mean_sq = 2^k (+eps)")
    bnd_seconds = time.perf_counter() - t0

    # ---- subnormals ---------------------------------------------------------------------------------
    sub = rng.integers(1, 0x80, (8, N)).astype(np.uint16) | (rng.integers(0, 2, (8, N)).astype(np.uint16) << 15)
    run("subnormals", "bf16_subnormal_rows_eps1e-6", sub, "ones", note="x^2 underflows to 0; inv = 1/sqrt(eps); out = 1000*x*w is subnormal")
    run("subnormals", "bf16_subnormal_rows_eps0", sub[:4], "ones", eps=0.0, note="sum 0, sqrt(0) = 0, rcp(0) = inf")
    tiny = bf16(rng.normal(0, 2.0 ** -70, (8, N)))  # x^2 ~ 2^-140: fp32 subnormal squares accumulate (no ftz)
    run("subnormals", "fp32_subnormal_squares_eps0", tiny, "ones", eps=0.0, note="mean_eps subnormal -> sqrt.approx.ftz flushes -> inf")
    run("subnormals", "fp32_subnormal_squares_eps1e-6", tiny, "random")
    tiny2 = bf16(rng.normal(0, 2.0 ** -60, (8, N)))  # x^2 ~ 2^-120: normal squares, sum ~ 2^-109
    run("subnormals", "tiny_normal_squares_eps0", tiny2, "ones", eps=0.0)
    mixed = randn_rows(8)
    mixed[:, ::3] = sub[:8, ::3]
    run("subnormals", "mixed_subnormal_normal", mixed, "random")
    # products inv*x*w that are fp32-subnormal: normal x, tiny w
    wsub = bf16(np.full(N, 2.0 ** -120))
    weights["w_2^-120"] = wsub; np.save(out / "weights/w_2^-120.npy", wsub); wmeta["w_2^-120"] = {"file": "weights/w_2^-120.npy", "n": N}
    run("subnormals", "subnormal_products_via_tiny_w", bf16(rng.normal(0, 2.0 ** -10, (8, N))), "w_2^-120", note="inv*x ~ 2^10, times 2^-120 -> 2^-110 normal; with x ~ 2^-10 scale -> out subnormal range")

    # ---- extremes -----------------------------------------------------------------------------------
    run("extremes", "bf16_max_row", np.full((2, N), BF16_MAX, np.uint16), "ones", note="x^2 = inf -> inv = 0 -> out = 0")
    run("extremes", "one_bf16_max_rest_normal", np.where(np.arange(N) == 100, BF16_MAX, randn_rows(2)).astype(np.uint16), "random")
    for sc, name in ((4e17, "sum_finite_near_max"), (1e18, "sum_overflows"), (1e19, "squares_near_max")):
        run("extremes", f"scale_{name}", bf16(rng.normal(0, sc, (4, N))), "ones")
    inf_row = randn_rows(4); inf_row[0, 0] = BF16_INF; inf_row[1, 1535] = BF16_INF | 0x8000; inf_row[2, 700] = BF16_INF; inf_row[3, 8] = BF16_INF
    run("extremes", "inf_in_x", inf_row, "random", note="x^2 = inf -> sum inf -> inv 0 -> 0*inf = NaN at that column, 0 elsewhere")
    nan_row = randn_rows(4); nan_row[0, 0] = BF16_NAN; nan_row[1, 33] = 0xFFC0; nan_row[2, 1000] = 0x7F81; nan_row[3, 1535] = 0x7FFF
    run("extremes", "nan_in_x", nan_row, "random", note="NaN payloads 7fc0, ffc0, 7f81, 7fff")
    run("extremes", "nan_in_w", randn_rows(4), "nan_one")
    run("extremes", "max_w_overflowing_out", bf16(rng.normal(0, 1, (4, N))), "max", note="out = inv*x*3.39e38 overflows for |inv*x| > 1")
    run("extremes", "large_eps", randn_rows(4), "random", eps=1e6)
    run("extremes", "negative_eps", randn_rows(4), "random", eps=-0.5, note="mean_eps may be negative -> sqrt NaN")
    run("extremes", "eps_inf", randn_rows(2), "random", eps=float("inf"))

    # ---- near-uniform -------------------------------------------------------------------------------
    consts = np.array([2.0 ** k for k in range(-30, 31, 3)] + [1.0, -1.0, 3.0, 0.1, 7.0, 1e-3, 1e3, 12345.0])
    run("near_uniform", "constant_rows", bf16(np.tile(consts[:, None], (1, N))), "ones")
    run("near_uniform", "constant_rows_w_random", bf16(np.tile(consts[:, None], (1, N))), "random")
    base = bf16(np.full((16, N), 1.0))
    for i in range(16):
        base[i, rng.integers(0, N)] += 1  # one element one BF16 ulp up
    run("near_uniform", "one_ulp_perturbation", base, "random")
    alt = bf16(np.tile(np.where(np.arange(N) % 2 == 0, 1.0, -1.0), (4, 1)))
    run("near_uniform", "alternating_sign", alt, "random")

    # ---- random scale (exponent coverage of the sqrt / rcp tables) -------------------------------------
    for k in (-60, -40, -20, -10, 10, 20, 40, 60):
        run("random_scale", f"scale_2^{k}", bf16(rng.normal(0, 2.0 ** k, (8, N))), "random")

    # ---- n_cols family ------------------------------------------------------------------------------
    for n in (16, 64, 576, 1024, 2048, 3072, 4096):
        run("n_cols", f"n_cols_{n}", bf16(rng.normal(0, 1, (8, n))), "random", note="new_specialisations flags a different compilation")
    run("n_cols", "n_cols_576_wide_range", bf16(np.exp2(rng.uniform(-8, 8, (8, 576))) * rng.choice([-1, 1], (8, 576))), "random")

    # ---- beyond natural range: many rows in one launch ------------------------------------------------
    run("beyond_natural_range", f"{args.big_rows}_rows", randn_rows(args.big_rows), "natural" if "natural" in weights else "random")

    spec_log.append({"after": "families", "n_specialisations": n_specialisations(kern)})

    # ---- mutations ----------------------------------------------------------------------------------
    muts: list[dict] = []
    mut_base = randn_rows(8)
    base_out = run("mutation_base", "base_rows", mut_base, wkey)
    bits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    mseq = 0
    for b in bits:
        r = int(rng.integers(0, 8)); c = int(rng.integers(0, N))
        xm = mut_base.copy(); xm[r, c] ^= np.uint16(1 << b)
        om = words(launch(to_t(xm), to_t(weights[wkey]), EPS)); torch.cuda.synchronize()
        f = f"calls/mut{mseq:04d}_out.npy"; np.save(out / f, om); mseq += 1
        muts.append({"kind": "x_bit_flip", "bit": b, "row": r, "col": c, "base_seq": calls[-1]["seq"],
                     "x_before": f"{int(mut_base[r, c]):04x}", "x_after": f"{int(xm[r, c]):04x}", "out_file": f,
                     "production_output_words_changed": int((om != base_out).sum()),
                     "changed_in_mutated_row": int((om[r] != base_out[r]).sum()), "changed_in_other_rows": int((np.delete(om, r, 0) != np.delete(base_out, r, 0)).sum())})
    for b in bits:
        c = int(rng.integers(0, N))
        wm = weights[wkey].copy(); wm[c] ^= np.uint16(1 << b)
        om = words(launch(to_t(mut_base), to_t(wm), EPS)); torch.cuda.synchronize()
        f = f"calls/mut{mseq:04d}_out.npy"; np.save(out / f, om); mseq += 1
        muts.append({"kind": "w_bit_flip", "bit": b, "col": c, "base_seq": calls[-1]["seq"], "w_before": f"{int(weights[wkey][c]):04x}",
                     "w_after": f"{int(wm[c]):04x}", "out_file": f, "production_output_words_changed": int((om != base_out).sum())})
    for e in (1e-5, 1e-7, 0.0, 1e-3):
        om = words(launch(to_t(mut_base), to_t(weights[wkey]), e)); torch.cuda.synchronize()
        f = f"calls/mut{mseq:04d}_out.npy"; np.save(out / f, om); mseq += 1
        muts.append({"kind": "eps_change", "eps": e, "base_seq": calls[-1]["seq"], "out_file": f,
                     "production_output_words_changed": int((om != base_out).sum())})
    (out / "mutations.json").write_text(json.dumps({"weight": wkey, "eps": EPS, "mutations": muts}, indent=1))

    import triton  # noqa: E402
    import vllm  # noqa: E402

    meta = {
        "schema": "verity-vllm/rmsnorm-fixture/v1", "kind": "stress", "kernel": args.kernel, "seed": args.seed,
        "vllm": vllm.__version__, "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "entry_point": ("torch.ops._C.rms_norm(out, x, weight, eps) -> vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>" if args.kernel == "cuda"
                        else "vllm.model_executor.determinism.batch_invariant.rms_norm_batch_invariant(x, weight, eps)"),
        "kernel_source_sha256": sha256_file(Path(bi.__file__)),
        "specialisations": spec_log, "boundary_search": {"rows": args.boundary_search_rows, "exact_tie_rows": int(len(exact_tie_rows)),
                                                         "within_1ulp_rows": int(len(near_rows)), "within_4ulp_rows": int(len(near2_rows)), "seconds": bnd_seconds},
        "weights": wmeta, "n_calls": len(calls), "calls": calls,
    }
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"n_calls": len(calls), "rows": sum(c["rows"] for c in calls), "families": sorted({c["family"] for c in calls}),
                      "new_specialisations": [(c["family"], c["case"], c["new_specialisations"]) for c in calls if c["new_specialisations"]],
                      "mutations": len(muts), "undetected_mutations": [m for m in muts if m["production_output_words_changed"] == 0],
                      "boundary_search": meta["boundary_search"], "total_bytes": sum(p.stat().st_size for p in files)}, indent=1))


if __name__ == "__main__":
    main()
