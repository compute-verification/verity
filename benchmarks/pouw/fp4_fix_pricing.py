"""Pricing candidate fixes for the assessor's base-split break of Pearl-C4 (`internal/pouw/red-team/fp4-base-split-break.md`):
fix (a), probabilistic salt-deadness charged in the debit.

A code is ε-dead when two independent draws of v1's real noise law give it different values with probability below ε
(P_change, estimated over S draws with the unbiased collision estimator). Term 1 of the debit (2:4 groups) then counts
ε-dead codes; term 4 (forming) stays on the reference's witness, since a code that may change can't have its forming
skipped. The noise law is v1's: D = α·x + a_E·2^-4·E·Fᵀ, E and F uniform E2M1 lines of 32 (the atom's sum is exact on these
magnitudes), then the noisy block scale and the E2M1 cast.

    python benchmarks/pouw/fp4_fix_pricing.py attack                       # the attack family across spike heights S
    python benchmarks/pouw/fp4_fix_pricing.py real Qwen/Qwen2.5-0.5B --out q.json   # honest debit at each ε on a model

`attack` prints, per S: the spikes' P_change, ΔA's 2:4 violation rate per 4-group and the share of 16 x 64 fragments that fit
one 2:4 pass, the attack's cost in creditOf (the assessor's model: 0.05 + 0.95·(ΔA's part + the A0 corrections)), and the
smallest ε that charges its spikes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_emulation import E2M1_HALVES, e2m1_rn, ue4m3_dyadic, ue4m3_rn  # noqa: E402
from fp4_v4_coverage import KB1, T, bf16, side, stats, uval, witness  # noqa: E402

EPS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def draw_codes(X, alpha, ae, rng):
    """One draw of v1's forming on rows X: the E2M1 code values, as a small integer id per value (±0 merged)."""
    R, k = X.shape
    E = VALUES[rng.integers(0, 16, (R, 32))]
    F = VALUES[rng.integers(0, 16, (k, 32))]
    D = alpha[:, None] * X + (ae[:, None] / 16) * (E @ F.T)
    blk = D.reshape(R, k // 16, 16)
    s = np.maximum(uval(ue4m3_rn(np.abs(blk).max(2) / 6)), 2.0 ** -9)
    code = e2m1_rn((blk / s[:, :, None]).reshape(R, k))
    return np.where(code & 7, code, 0).astype(np.int8)


def p_change(X, alpha, ae, salts, rng):
    """P(two independent draws give the element different codes), estimated from `salts` draws (unbiased)."""
    R, k = X.shape
    counts = np.zeros((R, k, 16), np.int16)
    idx = np.arange(R * k)
    for _ in range(salts):
        c = draw_codes(X, alpha, ae, rng).reshape(-1)
        counts.reshape(-1, 16)[idx, c] += 1
    same = (counts.astype(np.float64) * (counts - 1)).sum(2) / (salts * (salts - 1))
    return 1 - same


def not_block_max(X):
    """False on each block's clean |max| element (its first occurrence), True elsewhere."""
    R, k = X.shape
    at = np.abs(X).reshape(R, k // 16, 16).argmax(2)
    keep = np.ones((R, k // 16, 16), bool)
    np.put_along_axis(keep, at[:, :, None], False, axis=2)
    return keep.reshape(R, k)


def units_of(dead):
    d4 = dead.reshape(len(dead), -1, 4).sum(2)
    return np.where(d4 == 4, 4, np.where(d4 >= 2, 2, 0)).sum(1).astype(np.float64)


# ---- the attack family ---------------------------------------------------------------------------------------------

def attack_rows(rows, k, S, rng, park=None):
    """The assessor's rows; with `park`, one element per block sits at S·ρ (the block max) and the other 13 spikes at
    park/6 of it, the centre of an E2M1 cell (4.25: code 4's, the widest below saturation)."""
    x = rng.standard_normal((rows, k))
    r = rng.choice((-1.0, 1.0), rows)
    c = rng.choice((-1.0, 1.0), k // 16)
    offs = [o for o in range(16) if o not in (0, 8)]
    for b in range(k // 16):
        for i, o in enumerate(offs):
            x[:, 16 * b + o] = S * r * c[b] * (1.0 if park is None or i == 0 else park / 6)
    return x, np.array([o in offs for o in range(16)] * (k // 16))


def stats_floor(X, rowmax):
    """v1's row statistics with the noise σ = max(ρ/4, rowmax·s), s the row's max (one factor per row: the atom's a_E)."""
    alpha, ae, dnf, rho2 = stats(X, KB1)
    if rowmax:
        s = np.abs(X).max(1)
        sig = np.maximum(np.sqrt(rho2) / 4, rowmax * s)
        ae = uval(ue4m3_rn(sig * alpha * KB1))
    return alpha, ae, dnf, rho2


def attack(args):
    rng = np.random.default_rng(20260930)
    out = []
    for S in args.S:
        x, spike = attack_rows(args.rows, args.k, S, rng, args.park)
        alpha, ae, dnf, rho2 = stats_floor(x, args.rowmax)
        pc = p_change(x, alpha, ae, args.salts, rng)
        sp = pc[:, spike]
        # ΔA against the salt-free guess A0 (the modal code), over fresh draws
        codes = [draw_codes(x, alpha, ae, rng) for _ in range(args.draws)]
        mode = np.stack(codes).astype(np.int16)
        vals = np.apply_along_axis(lambda v: np.bincount(v + 8, minlength=24).argmax() - 8, 0, mode)
        viol, clean = [], []
        for c in codes:
            dA = (c != vals)
            g = dA.reshape(args.rows, -1, 4).sum(2)
            viol.append(float((g > 2).mean()))
            fr = dA[: args.rows // 16 * 16].reshape(args.rows // 16, 16, args.k // 64, 64)
            nz4 = fr.reshape(args.rows // 16, 16, args.k // 64, 16, 4).sum(4)
            clean.append(float((nz4 <= 2).all(axis=(1, 3)).mean()))
        v, fclean = float(np.mean(viol)), float(np.mean(clean))
        delta_cost = 0.5 + 0.5 * (1 - fclean)
        costs = [0.05 + 0.95 * (delta_cost + a0) for a0 in (0.08, 0.25)]
        dead_eps = {e: float((sp < e).mean()) for e in EPS}
        term1 = {e: float(units_of(pc < e).mean() / args.k) for e in EPS}
        rec = {"S": S, "admitted": float(dnf.mean()), "spike_p_change_median": float(np.median(sp)),
               "spike_p_change_p90": float(np.quantile(sp, 0.9)), "violation_rate": v, "fragments_one_pass": fclean,
               "attack_cost": costs, "saving": [1 - c for c in costs], "spikes_eps_dead": dead_eps,
               "term1_share_of_chain": term1,
               "smallest_eps_charging_90pct": next((e for e in EPS if dead_eps[e] >= 0.9), None)}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


# ---- the count-limit candidate: the attacker's best response -----------------------------------------------------

def flip_of(eps):
    """The per-draw flip rate f of a two-valued code whose P_change is eps: P_change = 2f(1 − f)."""
    return (1 - np.sqrt(max(1 - 2 * eps, 0.0))) / 2


def countlimit(args):
    """Best response to "at most L near-deterministic codes (P_change < ε) per block": per block, the two ρ positions
    (offsets 0 and 8) are background, nonzero in ΔA with probability `bg` (the assessor's rows: 0.71); L positions are
    deterministic and fill the background's 4-groups first; the other 14 − L sit just past the threshold, flipping with
    f(ε) per draw.  All structured (rank-1 A0).  ΔA's cost is 0.5 + 0.5·(share of 16 x 64 fragments needing a second pass),
    the attack's 0.05 + 0.95·(ΔA + A0 corrections 0.08 … 0.25) as the assessor prices it."""
    from math import comb
    out = []
    for eps in args.eps:
        f = flip_of(eps)
        for L in args.L:
            # positions per 4-group: g0 = {0 (bg), 1, 2, 3}, g1 = {4..7}, g2 = {8 (bg), 9, 10, 11}, g3 = {12..15}
            det = [0, 0, 0, 0]
            left = L
            for g in (0, 2, 1, 3):
                take = min(left, 3 if g in (0, 2) else 4)
                det[g] = take
                left -= take
            viol = []
            for g in range(4):
                live = (3 if g in (0, 2) else 4) - det[g]
                bgp = args.bg if g in (0, 2) else 0.0
                p_more = 0.0
                for bgnz in (0, 1):
                    pb = bgp if bgnz else 1 - bgp
                    if pb == 0:
                        continue
                    for x in range(live + 1):
                        if bgnz + x > 2:
                            p_more += pb * comb(live, x) * f ** x * (1 - f) ** (live - x)
                viol.append(p_more)
            clean = np.prod([(1 - v) ** 64 for v in viol])          # 16 rows x 4 blocks of each group type per fragment
            delta = 0.5 + 0.5 * (1 - clean)
            saving = [1 - (0.05 + 0.95 * (delta + a0)) for a0 in (0.08, 0.25)]
            rec = {"eps": eps, "L": L, "flip": f, "violation_per_group": viol, "fragments_one_pass": float(clean),
                   "saving": saving, "closed": bool(saving[0] <= 0.0025)}
            out.append(rec)
            print(f"eps {eps:.2f} L {L:2d}: flip {f:.3f}  fragments one-pass {100*clean:5.1f}%  saving {100*saving[0]:6.1f} .. {100*saving[1]:6.1f}%  {'closed' if rec['closed'] else 'OPEN'}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


# ---- real models ---------------------------------------------------------------------------------------------------

FS, CAPS = 107.34, (0.0025, 0.001)


def modal_share(X, alpha):
    """Per row: the share of blocks whose clean UE4M3 scale byte is the row's most common one."""
    R, k = X.shape
    b = ue4m3_rn(np.abs(alpha[:, None] * X).reshape(R, k // 16, 16).max(2) / 6)
    return np.array([np.bincount(r).max() / len(r) for r in b])


def linear(X, W, rng, salts, max_cols=24, a_only=False, exclude_max=False, count_limits=()):
    from fp4_v4_coverage import DESIGNS
    p = DESIGNS["v1"]
    X, W = bf16(X), bf16(W)
    m, k = X.shape
    n = W.shape[0]
    m -= m % T
    cols = sorted(rng.choice(n // T, min(n // T, max_cols), replace=False))
    Wj = np.concatenate([W[c * T:(c + 1) * T] for c in cols])
    a, b = side(X[:m], p), side(Wj, p)
    pa = p_change(X[:m], *stats(X[:m], KB1)[:2], salts, rng)
    pb = None if a_only else p_change(Wj, *stats(Wj, KB1)[:2], salts, rng)
    if exclude_max:
        pa = np.where(not_block_max(X[:m]), pa, 1.0)
        if pb is not None:
            pb = np.where(not_block_max(Wj), pb, 1.0)
    cw = k + (32 + FS) * k / n + 256
    rows_ok = a["dnf"]
    res = {"k": k, "n": n, "macs": float(m * n * k), "eps": {}}
    for eps in EPS:
        ua = units_of(pa < eps)
        ub = b["units"] if a_only else units_of(pb < eps)   # a_only: B's term 1 stays on the witness
        deb, cred = [], []
        for r in range(m // T):
            I = slice(r * T, (r + 1) * T)
            ok = rows_ok[I]
            if ok.sum() == 0:
                continue
            nI = ok.sum()
            for c in range(len(cols)):
                J = slice(c * T, (c + 1) * T)
                cr = nI * T * cw
                t1 = (T * ua[I][ok].sum() + nI * ub[J].sum()) / cr
                t3 = 16 * (a["scr"][I][ok] @ b["scr"][J].T).sum() / cr
                t4 = T * FS / n * a["sdw"][I][ok].sum() / cr
                deb.append(t1 + t3 + t4)
                cred.append(cr)
        deb, cred = np.array(deb), np.array(cred)
        res["eps"][str(eps)] = {"max": float(deb.max()), "over": {str(c): float(cred[deb > c].sum() / cred.sum()) for c in CAPS},
                                "no_margin": {str(c): bool(deb.max() > c / 2) for c in CAPS},
                                "dead_share_a": float((pa < eps).mean())}
        # the count limit: a block of A with more than L ε-dead codes is charged in term 1 at 1 per MAC (16 per word)
        cnt = (pa < eps).reshape(m, k // 16, 16).sum(2)
        res["eps"][str(eps)]["count_hist"] = np.bincount(cnt.ravel(), minlength=17).tolist()
        for L in count_limits:
            over_units = 16.0 * (cnt > L).sum(1)
            deb, cred = [], []
            for r in range(m // T):
                I = slice(r * T, (r + 1) * T)
                ok = rows_ok[I]
                if ok.sum() == 0:
                    continue
                nI = ok.sum()
                for c in range(len(cols)):
                    J = slice(c * T, (c + 1) * T)
                    cr = nI * T * cw
                    t1 = (T * (a["units"][I][ok] + over_units[I][ok]).sum() + nI * b["units"][J].sum()) / cr
                    t3 = 16 * (a["scr"][I][ok] @ b["scr"][J].T).sum() / cr
                    t4 = T * FS / n * a["sdw"][I][ok].sum() / cr
                    deb.append(t1 + t3 + t4)
                    cred.append(cr)
            deb, cred = np.array(deb), np.array(cred)
            res["eps"][str(eps)][f"L{L}"] = {"max": float(deb.max()), "over": float(cred[deb > CAPS[0]].sum() / cred.sum()),
                                            "no_margin": bool(deb.max() > CAPS[0] / 2), "blocks_over": float((cnt > L).mean())}
    alpha_a = stats(X[:m], KB1)[0]
    ms = modal_share(X[:m], alpha_a)
    res["modal_share"] = {"median": float(np.median(ms)), "p90": float(np.quantile(ms, 0.9)),
                          "rows_over_0.97": float((ms > 0.97).mean())}
    return res


def real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[:, :args.tokens]
    lins = {nm: mod for nm, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and "lm_head" not in nm}
    got = {}
    hs = [mod.register_forward_pre_hook((lambda nm: lambda m_, a: got.__setitem__(nm, a[0][0].detach().to(torch.bfloat16).clone()))(nm))
          for nm, mod in lins.items()]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    rng = np.random.default_rng(20260930)
    res = {}
    for nm, mod in lins.items():
        x = got.pop(nm)
        if int(nm.split(".")[2]) % args.layer_step:
            continue
        res[nm] = linear(x.float().numpy(), mod.weight.detach().float().numpy(), rng, args.salts, a_only=args.a_only,
                         exclude_max=args.exclude_max, count_limits=args.count_limits)
    macs = sum(r["macs"] for r in res.values())
    summ = {"model": args.model, "tokens": int(ids.shape[1]), "salts": args.salts, "a_only": args.a_only,
            "layer_step": args.layer_step, "exclude_max": args.exclude_max, "linears": len(res), "eps": {}}
    for eps in EPS:
        e = str(eps)
        summ["eps"][e] = {"worst_tile": max(r["eps"][e]["max"] for r in res.values()),
                          "dead_share_a": sum(r["eps"][e]["dead_share_a"] * r["macs"] for r in res.values()) / macs}
        for c in CAPS:
            summ["eps"][e][f"credit_over_{c}"] = sum(r["eps"][e]["over"][str(c)] * r["macs"] for r in res.values()) / macs
            summ["eps"][e][f"macs_no_2x_margin_{c}"] = sum(r["macs"] for r in res.values() if r["eps"][e]["no_margin"][str(c)]) / macs
    for eps in EPS:
        e = str(eps)
        for L in args.count_limits:
            k_ = f"L{L}"
            summ["eps"][e][k_] = {
                "credit_over_0.0025": sum(r["eps"][e][k_]["over"] * r["macs"] for r in res.values()) / macs,
                "macs_no_2x_margin": sum(r["macs"] for r in res.values() if r["eps"][e][k_]["no_margin"]) / macs,
                "worst_tile": max(r["eps"][e][k_]["max"] for r in res.values()),
                "blocks_over": sum(r["eps"][e][k_]["blocks_over"] * r["macs"] for r in res.values()) / macs}
    summ["modal_share"] = {"rows_over_0.97": sum(r["modal_share"]["rows_over_0.97"] * r["macs"] for r in res.values()) / macs,
                           "median_of_medians": float(np.median([r["modal_share"]["median"] for r in res.values()]))}
    print(json.dumps(summ, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summ, "linears": res}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attack")
    a.add_argument("--S", type=float, nargs="+", default=[8.9, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5])
    a.add_argument("--rows", type=int, default=64)
    a.add_argument("--k", type=int, default=2048)
    a.add_argument("--salts", type=int, default=128)
    a.add_argument("--draws", type=int, default=8)
    a.add_argument("--rowmax", type=float, default=0.0, help="noise floor relative to the row max: σ = max(ρ/4, rowmax·s)")
    a.add_argument("--park", type=float, default=None, help="park 13 of the 14 spikes at this E2M1 cell centre (e.g. 4.25)")
    a.add_argument("--out", default=None)
    r = sub.add_parser("real")
    r.add_argument("model")
    r.add_argument("--tokens", type=int, default=256)
    r.add_argument("--salts", type=int, default=48)
    r.add_argument("--text", default=str(Path(__file__).resolve().parents[2] / "README.md"))
    r.add_argument("--a-only", action="store_true", help="probabilistic salt-deadness on A only; B's term 1 on the witness")
    r.add_argument("--layer-step", type=int, default=1, help="only decoder layers whose index is a multiple of this")
    r.add_argument("--exclude-max", action="store_true", help="never count a block's clean maximum as ε-dead")
    r.add_argument("--count-limits", type=lambda s: [int(v) for v in s.split(",")], default=[],
                   help="per-block limits L on ε-dead A codes; a block over L is charged in term 1 at 1 per MAC")
    r.add_argument("--out", default=None)
    c = sub.add_parser("countlimit")
    c.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    c.add_argument("--L", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12])
    c.add_argument("--bg", type=float, default=0.71)
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    {"attack": attack, "real": real, "countlimit": countlimit}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
