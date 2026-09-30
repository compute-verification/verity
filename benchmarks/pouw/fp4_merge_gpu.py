#!/usr/bin/env python3
"""epsilon_R and the zero fraction (``fp4-merge-rate/sm120``) with the forming on the GPU, as restartable fill chunks.

``fp4_census.merge_rate`` forms an operand with the stand-in (Pearl's rank-32 row noise at delta * rms(row), then NVIDIA's
or OCP's quantizer) and counts the Strassen-pattern pre-adds of two blocks that are one FP4 value under one scale of the
format.  Here both steps run in torch on the device, and the adaptive side's partner search covers ``--partner-blocks``
k-blocks (1,024: every k-block of Qwen2.5-7B's linears but NVFP4 down_proj's 1,184; the CPU census stopped at 64).
``selftest`` gates the port: on the same noisy operand the codes and scale bytes equal ``fp4_census.quantize``'s, and every
merge statistic equals ``fp4_census.merge_rate_codes``'s, bit for bit.

    python benchmarks/pouw/fp4_merge_gpu.py selftest --device cuda
    python benchmarks/pouw/fp4_merge_gpu.py run --device cuda --acts ACTS_DIR --out-dir OUT --budget-s 300

``run`` does one unit per (format, family) and per (format, real layer, linear), each with both operands, writes each
unit's JSON to ``--out-dir`` as its checkpoint, and exits 99 once ``--budget-s`` is spent with units left (0 when all are
done, with ``summary.json``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp4_census as C  # noqa: E402


def _exp2(e: torch.Tensor) -> torch.Tensor:
    """2^e exactly, as float64, for integer e in the normal range."""
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def _rne(a: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``fp4_census``'s round-to-nearest-even onto a sorted grid of magnitudes (index of the pick)."""
    hi = torch.searchsorted(grid, a.contiguous()).clamp(0, len(grid) - 1)
    lo = (hi - 1).clamp(0, len(grid) - 1)
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick = torch.where(dlo < dhi, lo, torch.where(dhi < dlo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return torch.where(a == grid[hi], hi, pick)


def e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    pick = _rne(y.abs().clamp(max=C.E2M1_MAX), torch.as_tensor(C.E2M1_VALUES, device=y.device))
    return pick | torch.where((y < 0) & (pick > 0), 8, 0)


def scale_pairs(fmt: str, sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if fmt == "ue8m0":
        return torch.full_like(sc, 8), sc - 130
    e, m = (sc >> 3) & 0xF, sc & 7
    return torch.where(e == 0, m, 8 + m), torch.where(e == 0, -9, e - 10)


def quantize(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``fp4_census.quantize`` on the device (float64 throughout)."""
    rows, k = x.shape
    if fmt == "nvf4":
        g = float(x.abs().max()) / (C.E2M1_MAX * C.E4M3_MAX) or 1.0
        xb = x.reshape(rows, k // 16, 16) / g
        sc = _rne((xb.abs().amax(-1) / C.E2M1_MAX).clamp(max=C.E4M3_MAX), torch.as_tensor(C.E4M3_FIN, device=x.device))
        m, e = scale_pairs("ue4m3", sc)
        s = m.double() * _exp2(e)
        codes = e2m1_codes(torch.where(s[..., None] > 0, xb / torch.where(s > 0, s, 1.0)[..., None], 0.0))
        return codes.reshape(rows, k), sc, g
    xb = x.reshape(rows, k // 32, 32)
    amax = xb.abs().amax(-1)
    _, e = torch.frexp(torch.where(amax > 0, amax, 1.0))
    ex = torch.where(amax > 0, (e.to(torch.int64) - 3).clamp(-127, 127), -127)
    codes = e2m1_codes(xb / _exp2(ex)[..., None])
    return codes.reshape(rows, k), ex + 127, 1.0


def noisy_row(x: torch.Tensor, delta: float, gen: torch.Generator) -> torch.Tensor:
    """``fp4_census.noisy(mode="row")``, drawn on the device (the same law, a different stream)."""
    rows, k = x.shape
    d = x.device

    def lines(n):
        return (torch.randint(1, 129, (n, 32), generator=gen, device=d)
                * (2 * torch.randint(0, 2, (n, 32), generator=gen, device=d) - 1)).double()

    Z = lines(rows) @ lines(k).T                                   # integers below 2^20: exact in any order
    Z = Z / torch.sqrt((Z ** 2).mean(1, keepdim=True))
    return x + delta * torch.sqrt((x ** 2).mean(1, keepdim=True)) * Z


def block_ints(codes: torch.Tensor, sc: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = codes.shape
    blk = 16 if fmt == "nvf4" else 32
    m, e = scale_pairs("ue4m3" if fmt == "nvf4" else "ue8m0", sc)
    num = torch.as_tensor(C.E2M1_NUM, device=codes.device)[codes & 7]
    num = torch.where(codes & 8 != 0, -num, num)
    return num.reshape(rows, k // blk, blk) * m[..., None], e - 1


def representable(N: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp4_census.fp4_representable`` on the device."""
    lut = torch.zeros(13, dtype=torch.bool, device=N.device)
    lut[torch.as_tensor(C.E2M1_NUM, device=N.device)] = True
    M = N.abs().amax(-1)
    zero = M == 0
    Ms = torch.where(zero, 1, M)[:, None]
    strict, anyscale = zero.clone(), zero.clone()
    odd_max = 15 if fmt == "nvf4" else 1
    for qmax in (1, 2, 3, 4, 6, 8, 12):
        num = N * qmax
        q = (num // Ms).abs()
        ok = ~zero & (num % Ms == 0).all(-1) & ((q <= 12) & lut[q.clamp(max=12)]).all(-1)
        anyscale |= ok
        three = 3 if qmax % 3 == 0 else 1
        t = torch.where(Ms[:, 0] % three == 0, 2 * Ms[:, 0] // three, 0)
        odd = t // torch.where(t > 0, t & -t, 1)
        strict |= ok & (t > 0) & (odd <= odd_max)
    return strict, anyscale


def _aligned(N1, E1, N2, E2):
    lo = torch.minimum(E1, E2)
    far = (E1 - E2).abs() > 40
    one = torch.ones_like(E1)
    return N1 * (one << (E1 - lo).clamp(max=40))[:, None], N2 * (one << (E2 - lo).clamp(max=40))[:, None], far


def mergeable(N1, E1, N2, E2, fmt) -> torch.Tensor:
    A1, A2, far = _aligned(N1, E1, N2, E2)
    ok = torch.zeros(len(N1), dtype=torch.bool, device=N1.device)
    for S in (A1 + A2, A1 - A2):
        ok |= representable(S, fmt)[0] & (S.abs().amax(-1) > 0)
    return ok & ~far


def partners(N, E, live, fmt, max_blocks: int | None = None, chunk: int = 1 << 22) -> dict:
    """``fp4_census.merge_partners``: every other row at the same k-block, every other k-block in the same row."""
    rows, nb, blk = N.shape
    kb = nb if max_blocks is None else min(nb, max_blocks)
    d = N.device
    i, j = torch.triu_indices(rows, rows, 1, device=d)
    col = torch.zeros(rows, kb, dtype=torch.int32, device=d)
    step = max(1, chunk // max(1, len(i) * blk))
    for b0 in range(0, kb, step):
        b1 = min(kb, b0 + step)
        nbt = b1 - b0
        f = lambda T, r: T[r, b0:b1].reshape(len(r) * nbt, *T.shape[2:])  # noqa: E731
        ok = (mergeable(f(N, i), f(E, i), f(N, j), f(E, j), fmt) & f(live, i) & f(live, j)).reshape(len(i), nbt).int()
        col[:, b0:b1].index_add_(0, i, ok).index_add_(0, j, ok)
    row = torch.zeros(rows, kb, dtype=torch.int32, device=d)
    p, q = torch.triu_indices(kb, kb, 1, device=d)
    pstep = max(1, chunk // blk)
    for r in range(rows):
        for c0 in range(0, len(p), pstep):
            pp, qq = p[c0:c0 + pstep], q[c0:c0 + pstep]
            ok = (mergeable(N[r, pp], E[r, pp], N[r, qq], E[r, qq], fmt) & live[r, pp] & live[r, qq]).int()
            row[r].index_add_(0, pp, ok).index_add_(0, qq, ok)
    lv = live[:, :kb]
    n_live = max(1, int(lv.sum()))
    return {"same_kblock_any_row": int(((col > 0) & lv).sum()) / n_live,
            "same_row_any_kblock": int(((row > 0) & lv).sum()) / n_live, "rows": int(rows), "kblocks_searched": int(kb)}


def merge_rate_codes(codes: torch.Tensor, sc: torch.Tensor, fmt: str, partner_blocks: int | None = None) -> dict:
    """``fp4_census.merge_rate_codes`` on the device; ``partner_blocks=None`` searches every k-block."""
    N, E = block_ints(codes, sc, fmt)
    rows, nb, blk = N.shape
    live = N.abs().amax(-1) > 0
    out = {"zero_codes": int(((codes & 7) == 0).sum()) / codes.numel(), "zero_blocks": int((~live).sum()) / live.numel(),
           "blocks": int(rows * nb), "patterns": {}}
    h, hk = rows // 2, nb // 2
    idx = {"adjacent-k": ((slice(None), slice(0, 2 * hk, 2)), (slice(None), slice(1, 2 * hk, 2))),
           "adjacent-row": ((slice(0, 2 * h, 2), slice(None)), (slice(1, 2 * h, 2), slice(None))),
           "strassen-k": ((slice(None), slice(0, hk)), (slice(None), slice(hk, 2 * hk))),
           "strassen-row": ((slice(0, h), slice(None)), (slice(h, 2 * h), slice(None))),
           "strassen-diag": ((slice(0, h), slice(0, hk)), (slice(h, 2 * h), slice(hk, 2 * hk)))}
    worst = 0.0
    for pat in C.MERGE_PATTERNS:
        a, b = idx[pat]
        N1, E1, N2, E2 = N[a].reshape(-1, blk), E[a].reshape(-1), N[b].reshape(-1, blk), E[b].reshape(-1)
        both = live[a].reshape(-1) & live[b].reshape(-1)
        A1, A2, far = _aligned(N1, E1, N2, E2)
        dup = both & ~far & ((A1 == A2).all(-1) | (A1 == -A2).all(-1))
        n = max(1, int(both.sum()))
        row = {"pairs": int(both.sum())}
        for sign, S in (("+", A1 + A2), ("-", A1 - A2)):
            strict, anyscale = representable(S, fmt)
            nz = S.abs().amax(-1) > 0
            r = {"eps_strict": int((both & ~far & nz & strict).sum()) / n,
                 "eps_any_scale": int((both & ~far & nz & anyscale).sum()) / n, "cancel": int((both & ~nz).sum()) / n}
            worst = max(worst, r["eps_strict"])
            row[sign] = r
        row["dup"] = int(dup.sum()) / n
        out["patterns"][pat] = row
    out["eps_R_strict_max"] = worst
    out["eps_R_any_scale_max"] = max(out["patterns"][p][s]["eps_any_scale"] for p in C.MERGE_PATTERNS for s in "+-")
    out["partners"] = partners(N, E, live, fmt, partner_blocks)
    return out


def selftest(device: str, rows: int = 32, k: int = 512, seed: int = 11) -> dict:
    """The port against the census, on the same noisy operands: codes, scale bytes, g and every merge statistic equal."""
    rng = np.random.default_rng(seed)
    checked = 0
    for fmt in ("nvf4", "mxf4"):
        for fam in ("gaussian", "outlier-in-block", "rank-1", "zero-blocks", "scale-spread-30", "spiky-row",
                    "stride-row-grid", "grid-aligned"):
            for X in C.family(fam, rows, k, rng):
                for Xn in (X, C.noisy(X, fmt, "row", 0.25, rng)):
                    ref_codes, ref_sc, ref_g = C.quantize(Xn, fmt)
                    codes, sc, g = quantize(torch.as_tensor(Xn, device=device), fmt)
                    if not (np.array_equal(codes.cpu().numpy(), ref_codes) and np.array_equal(sc.cpu().numpy(), ref_sc)
                            and g == ref_g):
                        raise AssertionError(f"quantize differs: {fmt} {fam}")
                    ref = C.merge_rate_codes(ref_codes, ref_sc, fmt, partner_blocks=10 ** 9)
                    got = merge_rate_codes(codes, sc, fmt)
                    if got != ref:
                        raise AssertionError(f"merge_rate_codes differs: {fmt} {fam}\n{got}\n{ref}")
                    checked += 1
    return {"selftest": "pass", "operands": checked, "rows": rows, "k": k, "device": device}


def _brief(r: dict) -> dict:
    return {"eps_R_strict_max": r["eps_R_strict_max"], "eps_R_any_scale_max": r["eps_R_any_scale_max"],
            "zero_codes": r["zero_codes"], "zero_blocks": r["zero_blocks"],
            "partner_col": r["partners"]["same_kblock_any_row"], "partner_row": r["partners"]["same_row_any_kblock"]}


def units(args) -> list[tuple[str, str, str]]:
    """(kind, format, name): each family, then each real layer's linear (``layerLL:linear``), both formats."""
    out = [("fam", fmt, fam) for fam in args.families.split(",") for fmt in ("nvf4", "mxf4")]
    for f in sorted(Path(args.acts).glob("layer*.npz")) if args.acts else []:
        lins = sorted({key.rsplit("/", 1)[0] for key in np.load(f).files})
        out += [("real", fmt, f"{f.stem}:{lin}") for lin in lins for fmt in ("nvf4", "mxf4")]
    return out


def run_unit(args, kind: str, fmt: str, name: str, dev: str) -> dict:
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + zlib.crc32(f"{kind}-{fmt}-{name}".encode()))
    if kind == "fam":
        rng = np.random.default_rng(args.seed)
        sides = dict(zip("AB", C.family(name, args.rows, args.k, rng)))
    else:
        layer, lin = name.split(":")
        d = np.load(Path(args.acts) / f"{layer}.npz")
        sides = {s: d[f"{lin}/{'x' if s == 'A' else 'w'}"].astype(np.float64) for s in "AB"}
    res = {"kind": kind, "format": fmt, "name": name, "delta": args.delta, "noise": "row (stand-in), drawn on the device",
           "partner_blocks": args.partner_blocks, "sides": {}}
    for label, X in sides.items():
        t0 = time.monotonic()
        x = torch.as_tensor(X, device=dev)
        codes, sc, _ = quantize(noisy_row(x, args.delta, gen), fmt)
        r = merge_rate_codes(codes, sc, fmt, args.partner_blocks)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        r["shape"], r["seconds"] = list(X.shape), round(time.monotonic() - t0, 3)
        res["sides"][label] = r
        print(json.dumps({"unit": f"{kind}-{fmt}-{name}", "side": label, **{k: round(v, 5) for k, v in _brief(r).items()},
                          "s": r["seconds"]}), flush=True)
    return res


def cmd_run(args) -> int:
    t_start = time.monotonic()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = selftest(args.device)
    print(json.dumps(gate), flush=True)
    todo = units(args)
    stem = lambda kind, fmt, name: f"{kind}-{fmt}-{name.replace(':', '__')}"  # noqa: E731
    ran = 0
    for kind, fmt, name in todo:
        f = out / f"{stem(kind, fmt, name)}.json"
        if f.exists():
            continue
        if ran and time.monotonic() - t_start > args.budget_s:
            return 99
        ran += 1
        starts = out / f"{stem(kind, fmt, name)}.starts"
        n = int(starts.read_text()) + 1 if starts.exists() else 1
        starts.write_text(str(n))
        if n > 3:
            raise SystemExit(f"unit {f.stem} started 3 times without finishing")
        res = run_unit(args, kind, fmt, name, args.device)
        res["selftest"] = gate
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.rename(f)
    summary = {"units": [], "rows": args.rows, "k": args.k, "delta": args.delta, "seed": args.seed}
    for kind, fmt, name in todo:
        r = json.loads((out / f"{stem(kind, fmt, name)}.json").read_text())
        for label, s in r["sides"].items():
            summary["units"].append({"kind": kind, "format": fmt, "name": name, "side": label, **_brief(s)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "run"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--families", default=",".join(C.FAMILIES))
    ap.add_argument("--acts", default=None, help="fp4_quality.py --dump-acts directory (layerLL.npz)")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--k", type=int, default=16384)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partner-blocks", type=int, default=1024, help="k-blocks the partner search covers (the census: 256)")
    ap.add_argument("--out-dir", default="merge_gpu")
    ap.add_argument("--budget-s", type=float, default=300.0)
    args = ap.parse_args()
    if args.cmd == "selftest":
        print(json.dumps(selftest(args.device)))
        return
    sys.exit(cmd_run(args))


if __name__ == "__main__":
    main()
