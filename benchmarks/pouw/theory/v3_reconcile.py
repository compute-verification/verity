"""ACCEL ALGO-COST-3 (CMP-118, A27/A28): where ALGO-COST-2's A25 (`v3_caps.py` here) and PROTO-DESIGN-3's v3
(`accel_score/v3_caps.py` at bbe690dfe5) disagree, which inputs cause it, and v3's caps under the right reading.

Both price the routes at A = B = 0 (the tile route, PD2-1's O-route, the zero-high-digit peel) in D5's job LP
(`d5_routes.job`), at the same (t, L′). They differ in how the honest job's hashing enters D5's LP:

- **comps** (v3): the protocol's SHA-512 compressions a MAC from nofold's word model, priced by D5 at θ (tensor at
  θ's s8, hashing at the schedules of D5's `sha_menu`);
- **lane** (A25): the hash scale bisected so that D5's honest ALU load equals nofold's A_nw/T. That ratio carries
  nofold's η = 0.85 on the tensor side and nofold's own price a compression (CALIB's 38.29 / 38.11 SM-clocks), so it
  understates in-model hashing by about 1.2× (H200) and 1.5× (RTX PRO 6000);

and, second order, in nofold's version (9a21fbecd6 expands b on the node; bbe690dfe5 streams b, P2), the peels'
digits (D5's fixed 2 + 2 against nofold's dz, dy), the L′ grid (A25: powers of two) and the zero-leak tolerance.

A third reading, **pub**, is comps with D5's own `pub_sha` in the LP's schedule menu. On sm_120 `sha_menu` drops
"R-hi, IMAD.HI at 32 (unmeasured)", which is D5's named `pub_sha` there, so v3's LP hashes about 1.13× slower than
the public best it credits; on the H200 the menu already holds `pub_sha` (A-wide + R-hi), so pub = comps.

Parts:
- `reconcile`: per point, the job-LP leak under each variant, D5's honest lane load, the lane load at which the leak
  starts (bisection on the hash scale) and the served-period (any-prices) tensor saving over honest tensor time;
- `caps`: v3's cap rule (the largest L′ of v3's grid within the card's tolerance: 0 on the H200, 1% at L′ ≤ 256 on
  the RTX PRO 6000) per static shape under each reading, with γ, and MAC-weighted per model;
- `picks`: on the H200, v3's pick against the least-γ zero-leak L′, each with its margin and served-period saving;
- `pv`: P19 (RT-PROTO-2): v3's PV caps per attended length n, whether L′(n) ≤ n/32 (no O-route saving at any
  price), the served-period saving at v3's picks, and coverage with PV credited only where L′ ≤ n/32.

With --d5 DIR (a public_best.py whose `tile_job` takes comps and model_sha, a6cb33fefe on), the same questions
against BET-ALGO-LAB's own hull, and D5 v2's check:
- `d5`: v3's picks (comps) under D5's `tile_job`, model_sha off and on, against PD3-4's `d5_routes.job`;
- `u13`: the H200 with in-model SHA-512 at RT-UNITS' 35.4 SM-clocks (U13) and CALIB's 38.29: v3's pick, and the
  largest and least-γ zero-leak L′;
- `perlp`: per shape and L′: γ, job-only leak, and what the makespan hull frees against the public best's loads as a
  share of best.c (RT-PROTO-2's unit), with 07:04Z's rule under its two readings (machine time; the honest program
  at the same prices) at a tensor-only price;
- `pvtable`: PV per n, v3's pick against L′ = n/32;
- `issue`: whether the hull's lower issue load is a route or degeneracy: each menu's least issue at makespan best.c.

    python3 benchmarks/pouw/theory/v3_reconcile.py --v3 DIR --a25 DIR [--d5 DIR] [--part reconcile,...] [--json OUT]

DIR --v3 holds nofold.py, d5_routes.py (bbe690dfe5) and public_best.py (D5); DIR --a25 holds nofold.py
(9a21fbecd6); DIR --d5 holds D5's public_best.py at the head to check. Staging, from a checkout with the branches
fetched:

    P=benchmarks/pouw/accel_score
    for f in nofold d5_routes; do git show bbe690dfe5:$P/$f.py > /tmp/in-v3/$f.py; done
    git show 9a21fbecd6:$P/nofold.py > /tmp/in-a25/nofold.py
    git show 126ba960ca:benchmarks/pouw/theory/public_best.py > /tmp/in-v3/public_best.py
    git show a6cb33fefe:benchmarks/pouw/theory/public_best.py > /tmp/in-d5/public_best.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from dataclasses import replace

V3_LPS = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048)
V3_CARDS = {"H200": (2, None, 1e-4), "RTX PRO 6000": (3, 256, 0.01)}
READINGS = ("comps", "pub", "lane")


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Codes:
    def __init__(self, v3: str, a25: str, d5: str | None = None):
        sys.path.insert(0, v3)
        import public_best as pb
        self.pb = pb
        self.dr = load("d5_routes_v3", f"{v3}/d5_routes.py")
        self.nf = {"v3": load("nofold_v3", f"{v3}/nofold.py"), "a25": load("nofold_a25", f"{a25}/nofold.py")}
        self._menu = pb.sha_menu
        self.d5 = load("public_best_d5", f"{d5}/public_best.py") if d5 else None

    def d5_job(self, name: str, K: int, N: int, t: int, Lp: int, model_sha: bool = True, scale: float = 1.0,
               weight_coins: bool = True, card=None) -> dict:
        """D5's own `tile_job` at nofold's digits and compressions (comps times scale), on card (default: D5's)."""
        d5, nf = self.d5, self.nf["v3"]
        c = card or (d5.SM120 if name == "RTX PRO 6000" else d5.H200)
        _, bit, depth = d5.BIT_COST[c.family][0]
        r = nf.v2_round(name, K, N, t * Lp, 16, tiles=t, weight_coins=weight_coins)
        return d5.tile_job(c, t, Lp, K, N, bit, depth, nf.V_WIDTH[name], digits=r["digits"], comps=r["comps"] * scale,
                           model_sha=model_sha)

    def card(self, name: str):
        return self.pb.SM120 if name == "RTX PRO 6000" else self.pb.H200

    @contextlib.contextmanager
    def sha(self, reading: str):
        """pub: D5's `pub_sha` joins the LP's schedule menu (every schedule of the card's family)."""
        if reading == "pub":
            self.pb.sha_menu = lambda card: list(self.pb.SHA_SCHEDULES[card.family])
        try:
            yield
        finally:
            self.pb.sha_menu = self._menu

    def point(self, name: str, K: int, N: int, t: int, Lp: int, nofold: str = "v3", how: str = "comps",
              digits: str = "nofold", weight_coins: bool = True) -> dict:
        pb, dr, nf = self.pb, self.dr, self.nf[nofold]
        c = self.card(name)
        _, bit, depth = pb.BIT_COST[c.family][0]
        r = nf.v2_round(name, K, N, t * Lp, 16, tiles=t, weight_coins=weight_coins)
        if "comps" not in r:                      # 9a21fbecd6 returns neither; same formulas as bbe690dfe5
            mac, sha = nf.V2_CARDS[name][:2]
            n_terms = K / Lp
            r["comps"] = (r["hash"] + r["A"] + r["coin"]) / (nf.ETA * mac / sha)
            r["digits"] = (2 if (128 + nf.V_WIDTH[name]) * n_terms < 2 ** 14 else 3,
                           2 if 128 * n_terms < 2 ** 14 else 3)
        dg = r["digits"] if digits == "nofold" else (2, 2)
        v = nf.V_WIDTH[name]
        with self.sha("pub" if how == "pub" else "menu"):
            if how in ("comps", "pub"):
                j = dr.job(c, t, Lp, K, N, bit, depth, v, 1.0, digits=dg, comps=r["comps"])
            else:
                j = dr.job_at(c, t, Lp, K, N, bit, depth, v, r["A_nw"] / r["T"], digits=dg)
        base = dict(t=t, Lp=Lp, nofold_load=r["A_nw"] / r["T"], any=r["saved"] / r["T"], saved=r["saved"],
                    T=r["T"], gamma_max=r["gamma_max"], gamma=r["gamma"], digits=dg)
        if j is None:
            return dict(base, leak=0.0, load=None, gamma_lp=None, mix={}, binding=[])
        return dict(base, leak=1 - j["c_opt"], load=pb._alu_load(j["honest"]), gamma_lp=j["honest"].c,
                    mix={k.split(":")[-1]: round(float(x), 3) for k, x in j["mix"].items() if x > 1e-3},
                    binding=j["best"].binding)

    def threshold(self, name: str, K: int, N: int, t: int, Lp: int, tol: float = 1e-4, digits=(2, 2)) -> float:
        """The least honest ALU load (D5's, at θ) with job-LP leak below tol: bisection on the hash scale."""
        pb, dr = self.pb, self.dr
        c = self.card(name)
        _, bit, depth = pb.BIT_COST[c.family][0]
        v = self.nf["v3"].V_WIDTH[name]

        def at(s):
            j = dr.job(c, t, Lp, K, N, bit, depth, v, s, digits=digits)
            return 1 - j["c_opt"], pb._alu_load(j["honest"])

        lo, hi = 0.05, 8.0
        if at(hi)[0] >= tol:
            return float("inf")
        if at(lo)[0] < tol:
            return at(lo)[1]
        for _ in range(28):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if at(mid)[0] >= tol else (lo, mid)
        return at(hi)[1]

    def cap(self, name: str, K: int, N: int, how: str, weight_coins: bool = True) -> dict:
        """v3's rule (`accel_score/v3_caps.cap`): the largest L′ of v3's grid (t L′ ≤ K, L′ under the card's limit)
        whose leak is within the card's tolerance, else the least-leak L′; one tile below t·64 inner indices."""
        t, lmax, tol = V3_CARDS[name]
        t = t if t * V3_LPS[0] <= K else 1
        tried = []
        for Lp in sorted((x for x in V3_LPS if t * x <= K and (lmax is None or x <= lmax)), reverse=True) or [K]:
            p = self.point(name, K, N, t, Lp, how=how, weight_coins=weight_coins)
            tried.append(p)
            if p["leak"] < tol:
                return p
        return min(tried, key=lambda x: x["leak"])


def reconcile(co: Codes, out: dict) -> None:
    nf = co.nf["v3"]
    print("== H200, t = 2: job-LP leak per L′ under each variant; D5 lane load (comps / lane); threshold lane load")
    print("   v3 = nofold bbe6 + comps + nofold digits; A25 = nofold 9a21 + lane-matched + 2+2 digits;")
    print("   x-lane = nofold bbe6 + lane-matched; x-9a21 = nofold 9a21 + comps; any = served-period tensor saving")
    for model, shapes in nf.SHAPES.items():
        for K, N, _ in shapes:
            for Lp in (256, 512, 640, 768, 1024):
                if 2 * Lp > K:
                    continue
                v3 = co.point("H200", K, N, 2, Lp)
                a25 = co.point("H200", K, N, 2, Lp, nofold="a25", how="lane", digits="d5")
                xl = co.point("H200", K, N, 2, Lp, how="lane")
                x9 = co.point("H200", K, N, 2, Lp, nofold="a25")
                th = co.threshold("H200", K, N, 2, Lp, digits=v3["digits"])
                out["reconcile"].append(dict(card="H200", model=model, K=K, N=N, t=2, Lp=Lp, v3=v3, a25=a25,
                                             x_lane=xl, x_9a21=x9, threshold=th))
                print(f"  {model:16} {K:5}x{N:<6} L′{Lp:5} | v3 {v3['leak']:6.2%} A25 {a25['leak']:6.2%} "
                      f"x-lane {xl['leak']:6.2%} x-9a21 {x9['leak']:6.2%} | load {v3['load']:.2f} / "
                      f"{a25['load']:.2f} (nofold {v3['nofold_load']:.2f}) thr {th:.2f} margin "
                      f"{v3['load'] / th - 1:+6.1%} | γ_lp {v3['gamma_lp']:.2f} γ_max {v3['gamma_max']:.2f} "
                      f"any {v3['any']:5.1%} [{v3['mix']}]")
    print("\n== RTX PRO 6000, t 2 / 3, L′ 64 and 256: the same columns, plus pub (pub_sha in the LP's menu)")
    for model, shapes in nf.SHAPES.items():
        for K, N, _ in shapes:
            for t in (2, 3):
                for Lp in (64, 128, 192, 256):
                    v3 = co.point("RTX PRO 6000", K, N, t, Lp)
                    pub = co.point("RTX PRO 6000", K, N, t, Lp, how="pub")
                    a25 = co.point("RTX PRO 6000", K, N, t, Lp, nofold="a25", how="lane", digits="d5")
                    xl = co.point("RTX PRO 6000", K, N, t, Lp, how="lane")
                    out["reconcile"].append(dict(card="RTX PRO 6000", model=model, K=K, N=N, t=t, Lp=Lp, v3=v3,
                                                 pub=pub, a25=a25, x_lane=xl))
                    if Lp in (64, 256) or (K, N) == (1536, 65536):
                        print(f"  {model:16} {K:5}x{N:<6} t{t} L′{Lp:4} | v3 {v3['leak']:6.2%} pub "
                              f"{pub['leak']:6.2%} A25 {a25['leak']:6.2%} x-lane {xl['leak']:6.2%} | load "
                              f"{v3['load']:.2f} / pub {pub['load']:.2f} / lane {a25['load'] or 0:.2f} "
                              f"(nofold {v3['nofold_load']:.2f}) | γ_lp {v3['gamma_lp']:.2f} / pub "
                              f"{pub['gamma_lp']:.2f} γ_max {v3['gamma_max']:.2f} any {v3['any']:5.1%} "
                              f"[{v3['mix']}]")


def caps(co: Codes, out: dict) -> None:
    nf = co.nf["v3"]
    print("\n== v3's cap rule per static shape under each reading: L′ (leak) ...; MAC-weighted per model: γ_lp "
          "(D5's honest at θ), γ_max (nofold), job-LP leak (cost-weighted), served-period saving (of honest tensor)")
    for name in V3_CARDS:
        for model, shapes in nf.SHAPES.items():
            for how in READINGS:
                W = G = Gm = L = Lw = S = Tt = 0.0
                cells = []
                for K, N, reps in shapes:
                    p = co.cap(name, K, N, how)
                    out["caps"].append(dict(card=name, model=model, K=K, N=N, how=how, **p))
                    w = K * N * reps
                    W, G, Gm = W + w, G + w * p["gamma_lp"], Gm + w * p["gamma_max"]
                    L, Lw = L + w * p["gamma_lp"] * p["leak"], Lw + w * p["gamma_lp"]
                    S, Tt = S + w * p["saved"], Tt + w * p["T"]
                    cells.append(f"{K}x{N} {p['Lp']}" + (f" ({p['leak']:.2%})" if p["leak"] >= 1e-4 else ""))
                print(f"  {name:13} {model:16} {how:5} γ_lp {G / W:5.2f} γ_max {Gm / W:5.2f} leak {L / Lw:6.3%} "
                      f"served {S / Tt:6.2%} | " + "; ".join(cells))


def picks(co: Codes, out: dict) -> None:
    """The H200 at job-only prices (comps; pub is the same there): v3's rule (the largest zero-leak L′) against the
    least-γ zero-leak L′ (γ_lp, D5's honest cost at θ), each with its margin (honest lane load over the load at which
    the leak starts) and its served-period saving. A larger L′ that commits as many words a cell (S = ⌈K/L⌉) only adds
    peels, so it is dominated."""
    nf = co.nf["v3"]
    print("\n== H200, t = 2, job-only prices: v3's pick (largest zero-leak L′) | least-γ zero-leak L′; each: L′, S, "
          "γ_lp, γ_max, margin, served-period saving")
    for model, shapes in nf.SHAPES.items():
        agg = {"v3": [0.0] * 4, "least": [0.0] * 4}
        for K, N, reps in shapes:
            cand = [co.point("H200", K, N, 2, Lp) for Lp in V3_LPS if 2 * Lp <= K]
            clean = [p for p in cand if p["leak"] < 1e-4]
            big, least = max(clean, key=lambda p: p["Lp"]), min(clean, key=lambda p: p["gamma_lp"])
            cells = []
            for tag, p in (("v3", big), ("least", least)):
                th = co.threshold("H200", K, N, 2, p["Lp"], digits=p["digits"])
                p["margin"] = p["load"] / th - 1
                p["S"] = -(-K // (2 * p["Lp"]))
                w = K * N * reps
                a = agg[tag]
                a[0], a[1], a[2], a[3] = a[0] + w, a[1] + w * p["gamma_lp"], a[2] + w * p["saved"], a[3] + w * p["T"]
                cells.append(f"L′ {p['Lp']:4} S {p['S']:3} γ_lp {p['gamma_lp']:.2f} γ_max {p['gamma_max']:.2f} "
                             f"margin {p['margin']:+6.1%} served {p['any']:5.1%}")
            out["picks"].append(dict(model=model, K=K, N=N, v3=big, least=least))
            print(f"  {model:16} {K:5}x{N:<6} " + " | ".join(cells))
        print(f"  {model:16} MAC-weighted: v3 γ_lp {agg['v3'][1] / agg['v3'][0]:.2f} served "
              f"{agg['v3'][2] / agg['v3'][3]:5.2%} | least-γ γ_lp {agg['least'][1] / agg['least'][0]:.2f} served "
              f"{agg['least'][2] / agg['least'][3]:5.2%}")


def pv(co: Codes, out: dict) -> None:
    nf = co.nf["v3"]
    print("\n== P19: v3's PV caps per attended length n (K = n, N = d_v, weight side streamed), reading comps and pub;"
          "\n   bins with L′ > n/32 (the O-route saves at any price); MAC-weighted served-period saving and γ_lp;"
          "\n   coverage with PV credited at every n, and only where L′ ≤ n/32 is possible (n ≥ 2048)")
    for name in V3_CARDS:
        for m in nf.WORLD:
            hist, st, qk, pvm, ix = nf.world_macs(m)
            tot = st + qk + pvm + ix
            bins = sorted({nf.pv_bucket(n) for n in hist})
            mac = {}
            for n, c in hist.items():
                mac[nf.pv_bucket(n)] = mac.get(nf.pv_bucket(n), 0.0) + c * m["hq"] * m["dv"] * n
            for how in ("comps", "pub"):
                picks = {nb: co.cap(name, nb, m["dv"], how, weight_coins=False) for nb in bins}
                over = [nb for nb in bins if picks[nb]["Lp"] > nb / 32]
                w = sum(mac.values())
                served = sum(mac[nb] * picks[nb]["saved"] for nb in bins) / sum(mac[nb] * picks[nb]["T"] for nb in bins)
                leak = sum(mac[nb] * picks[nb]["gamma_lp"] * picks[nb]["leak"] for nb in bins) / sum(
                    mac[nb] * picks[nb]["gamma_lp"] for nb in bins)
                g = sum(mac[nb] * picks[nb]["gamma_lp"] for nb in bins) / w
                t = V3_CARDS[name][0]
                fixed = {}
                for nb in bins:
                    if nb < 2048:
                        continue
                    lp = max(x for x in V3_LPS if x <= min(picks[nb]["Lp"], nb / 32))
                    fixed[nb] = picks[nb] if lp == picks[nb]["Lp"] else co.point(name, nb, m["dv"], t, lp, how=how,
                                                                                 weight_coins=False)
                wf = sum(mac[nb] for nb in fixed)
                g_f = sum(mac[nb] * fixed[nb]["gamma_lp"] for nb in fixed) / wf if wf else float("nan")
                served_f = sum(mac[nb] * fixed[nb]["saved"] for nb in fixed) / sum(
                    mac[nb] * fixed[nb]["T"] for nb in fixed) if wf else float("nan")
                leak_f = max((fixed[nb]["leak"] for nb in fixed), default=float("nan"))
                ok10 = {nb: p for nb, p in fixed.items() if p["gamma_lp"] <= 10}
                w10 = sum(mac[nb] for nb in ok10)
                g10 = sum(mac[nb] * ok10[nb]["gamma_lp"] for nb in ok10) / w10 if w10 else float("nan")
                cov_all = 1 - (qk + ix) / tot
                cov_f = 1 - (qk + ix + (w - wf)) / tot
                cov10 = 1 - (qk + ix + (w - w10)) / tot
                out["pv"].append(dict(card=name, model=m["name"], how=how, bins=len(bins), over=over,
                                      n_max=max(bins), pv_share=w / tot,
                                      lp=[(nb, picks[nb]["t"], picks[nb]["Lp"], picks[nb]["leak"],
                                           picks[nb]["gamma_lp"], picks[nb]["any"]) for nb in bins],
                                      served=served, leak=leak, gamma_lp=g, gamma_fixed=g_f, served_fixed=served_f,
                                      leak_fixed=leak_f, coverage_all=cov_all, coverage_fixed=cov_f,
                                      gamma_fixed_10=g10, coverage_fixed_10=cov10,
                                      n_min_10=min(ok10, default=None)))
                print(f"  {name:13} {m['name']:16} {how:5} n ≤ {max(bins)}, PV {w / tot:4.1%} of MACs; L′ "
                      f"{min(p['Lp'] for p in picks.values())}–{max(p['Lp'] for p in picks.values())}; "
                      f"{len(over)}/{len(bins)} bins L′ > n/32 ({sum(mac[nb] for nb in over) / w:5.1%} of PV MACs);"
                      f" job leak {leak:6.3%}, served {served:6.2%}, γ_lp {g:5.2f}, coverage {cov_all:5.1%}\n"
                      f"      L′ ≤ n/32 from n 2048: γ_lp {g_f:5.2f}, served {served_f:6.2%}, worst job leak "
                      f"{leak_f:6.3%}, coverage {cov_f:5.1%}; and only bins with γ_lp ≤ 10 (from n "
                      f"{min(ok10, default=None)}): γ_lp {g10:5.2f}, coverage {cov10:5.1%}")


def d5(co: Codes, out: dict) -> None:
    nf = co.nf["v3"]
    print("\n== v3's picks (comps) under D5's tile_job, model_sha off / on, against PD3-4's d5_routes.job: job-only "
          "leak and γ_LP")
    for name in V3_CARDS:
        for model, shapes in nf.SHAPES.items():
            W = Lo = Lm = Lp_ = Gm = Gp = 0.0
            for K, N, reps in shapes:
                p = co.cap(name, K, N, "comps")
                j0, j1 = (co.d5_job(name, K, N, p["t"], p["Lp"], model_sha=ms) for ms in (False, True))
                l0, l1 = 1 - j0["c_opt"], 1 - j1["c_opt"]
                w = K * N * reps
                W, Lo, Lm, Lp_ = W + w, Lo + w * p["leak"], Lm + w * l0, Lp_ + w * l1
                Gm, Gp = Gm + w * j0["honest"].c, Gp + w * j1["honest"].c
                out["d5"].append(dict(card=name, model=model, K=K, N=N, t=p["t"], Lp=p["Lp"], pd34=p["leak"], d5=l0,
                                      d5_model=l1, g_pd34=p["gamma_lp"], g_d5=j0["honest"].c, g_d5_model=j1["honest"].c))
                print(f"  {name:12} {model:16} {K:5}x{N:<6} t{p['t']} L′{p['Lp']:4} | PD3-4 {p['leak']:6.3%} D5 "
                      f"{l0:6.3%} model_sha {l1:6.3%} | γ {p['gamma_lp']:.3f} / {j0['honest'].c:.3f} / "
                      f"{j1['honest'].c:.3f}")
            print(f"  {name:12} {model:16} MAC-weighted leak PD3-4 {Lo / W:6.3%} D5 {Lm / W:6.3%} model_sha "
                  f"{Lp_ / W:6.3%}; γ D5 {Gm / W:.2f} model_sha {Gp / W:.2f}")


def u13(co: Codes, out: dict) -> None:
    nf, d5_ = co.nf["v3"], co.d5
    s = d5_.sha_clocks(d5_.H200, d5_.PUB_SHA["sm_90"][0])[0]
    for target in (35.4, 38.29):
        f = target / s
        print(f"\n== H200, in-model SHA-512 at {target} SM-clocks (×{f:.4f} of D5's {s:.2f}): v3's pick | largest | "
              "least-γ zero-leak L′")
        for model, shapes in nf.SHAPES.items():
            W = G1 = G2 = 0.0
            for K, N, reps in shapes:
                v3 = co.cap("H200", K, N, "comps")
                res = {}
                for Lp in (x for x in V3_LPS if 2 * x <= K):
                    j = co.d5_job("H200", K, N, 2, Lp, scale=f)
                    res[Lp] = (1 - j["c_opt"], j["honest"].c)
                clean = [lp for lp, (lk, _) in res.items() if lk < 1e-4]
                big, least = max(clean), min(clean, key=lambda lp: res[lp][1])
                w = K * N * reps
                W, G1, G2 = W + w, G1 + w * res[big][1], G2 + w * res[least][1]
                out["u13"].append(dict(sha=target, model=model, K=K, N=N, v3=v3["Lp"], v3_leak=res[v3["Lp"]][0],
                                       largest=big, least=least, by_lp=res))
                print(f"  {model:16} {K:5}x{N:<6} v3 L′ {v3['Lp']:4}: leak {res[v3['Lp']][0]:6.3%} γ "
                      f"{res[v3['Lp']][1]:.2f} | L′ {big:4} γ {res[big][1]:.2f} | L′ {least:4} γ {res[least][1]:.2f}")
            print(f"  {model:16} MAC-weighted γ: largest zero-leak {G1 / W:.2f}, least-γ zero-leak {G2 / W:.2f}")


def perlp(co: Codes, out: dict) -> None:
    """Freed loads are best's minus the makespan hull's, a share of best.c: exact for tensor where the hull sits on
    the O-route, an arbitrary vertex on rows that don't bind (`issue`)."""
    nf = co.nf["v3"]
    for name, t, lmax in (("H200", 2, 2048), ("RTX PRO 6000", 3, 256)):
        print(f"\n== {name}, t = {t}, model_sha: L′: γ / job-only leak / tensor freed / issue freed (of best.c)")
        for model, shapes in nf.SHAPES.items():
            for K, N, reps in shapes:
                rows = []
                for Lp in (x for x in V3_LPS if t * x <= K and x <= lmax):
                    j = co.d5_job(name, K, N, t, Lp)
                    b, h = j["best"], j["route"]
                    fr = {k: (b.loads.get(k, 0) - h.loads.get(k, 0)) / b.c for k in ("tensor", "issue", "smem")}
                    bt = b.loads["tensor"] / b.c
                    rows.append(dict(card=name, model=model, K=K, N=N, reps=reps, t=t, Lp=Lp, gamma=j["honest"].c,
                                     leak=1 - j["c_opt"], freed=fr, best_tensor=bt,
                                     machine=bt - fr["tensor"], priced=1 - fr["tensor"] / bt))
                out["perlp"].extend(rows)
                print(f"  {model:16} {K:5}x{N:<6} " + "; ".join(
                    f"{r['Lp']}: {r['gamma']:.2f}/{r['leak']:.2%}/{r['freed']['tensor']:.1%}/{r['freed']['issue']:.2%}"
                    for r in rows))
    print("\n== 07:04Z's rule, max pub(L′)/T_honest(L′) with γ ≤ 10, at a tensor-only price (H200): T_honest as "
          "machine time (hull tensor / γ), as the honest program at the same price (1 - saving), and machine time "
          "with zero job-only leak")
    shapes = {}
    for r in out["perlp"]:
        if r["card"] == "H200":
            shapes.setdefault((r["model"], r["K"], r["N"], r["reps"]), []).append(r)
    agg = {}
    for (model, K, N, reps), rs in shapes.items():
        ok = [r for r in rs if r["gamma"] <= 10]
        pick = {"machine": max(ok, key=lambda r: r["machine"]), "priced": max(ok, key=lambda r: r["priced"]),
                "machine, zero leak": max((r for r in ok if r["leak"] < 1e-4), key=lambda r: r["machine"])}
        w = K * N * reps
        a = agg.setdefault(model, {k: [0.0, 0.0, 0.0] for k in pick})
        for k, p in pick.items():
            a[k][0], a[k][1], a[k][2] = a[k][0] + w, a[k][1] + w * p["gamma"], a[k][2] + w * p["leak"]
        print(f"  {model:16} {K:5}x{N:<6} " + " | ".join(
            f"{k}: L′ {p['Lp']} γ {p['gamma']:.2f} leak {p['leak']:.2%} saved {1 - p['priced']:.1%}"
            for k, p in pick.items()))
    for model, a in agg.items():
        print(f"  {model:16} MAC-weighted γ: " + "; ".join(f"{k} {v[1] / v[0]:.2f} (job-only leak {v[2] / v[0]:.2%})"
                                                         for k, v in a.items()))


def pvtable(co: Codes, out: dict) -> None:
    print("\n== PV per n (N = d_v 128, no weight coins): v3's pick t, L′: γ / job-only leak / served-period saving | "
          "L′ = n/32")
    for name, how in (("H200", "comps"), ("RTX PRO 6000", "pub")):
        t, lmax, _ = V3_CARDS[name]
        print(f"  {name} ({how})")
        for n in (128, 256, 512, 1024, 1152, 2048, 3072, 4096, 8192, 16384):
            p = co.cap(name, n, 128, how, weight_coins=False)
            lp = min(n // 32, lmax or n)
            q = co.point(name, n, 128, t, lp, how=how, weight_coins=False) if lp in V3_LPS else None
            out["pvtable"].append(dict(card=name, n=n, v3=p, n32=q))
            qs = f"L′ {lp}: {q['gamma_lp']:.2f} / {q['leak']:.2%} / {q['any']:.1%}" if q else "—"
            print(f"    n {n:5} | t{p['t']} L′ {p['Lp']:4}: {p['gamma_lp']:.2f} / {p['leak']:.2%} / {p['any']:.1%} | {qs}")


def issue(co: Codes, out: dict) -> None:
    nf, d5_ = co.nf["v3"], co.d5
    print("\n== H200 issue rows: the least issue capacity each menu needs at makespan best.c (bisection on the card's "
          "issue rate), best's (honest menu) against the hull's (every route)")
    for K, N, Lp in ((16384, 16384, 512), (16384, 65536, 512), (4096, 6144, 640), (8192, 8192, 512),
                     (2048, 128, 64), (4096, 128, 128)):
        wc = N > 128
        j0 = co.d5_job("H200", K, N, 2, Lp, weight_coins=wc)
        c0, res = j0["best"].c, {}
        for key in ("best", "route"):
            lo, hi = 0.0, 0.6
            for _ in range(30):
                mid = (lo + hi) / 2
                card = replace(d5_.H200, issue=d5_.H200.issue * (1 - mid))
                j = co.d5_job("H200", K, N, 2, Lp, weight_coins=wc, card=card)
                lo, hi = (mid, hi) if j[key].c <= c0 * (1 + 1e-7) else (lo, mid)
            res[key] = 1 - lo
        vertex = (j0["best"].loads["issue"] - j0["route"].loads["issue"]) / c0
        out["issue"].append(dict(K=K, N=N, Lp=Lp, best=res["best"], hull=res["route"], vertex=vertex))
        print(f"  {K}x{N} L′{Lp}: makespan vertex 'frees' {vertex:+.2%} of best.c; least issue at best.c: best "
              f"{res['best']:.4f}, hull {res['route']:.4f} of capacity (saving {1 - res['route'] / res['best']:+.2%})")


PARTS = dict(reconcile=reconcile, caps=caps, picks=picks, pv=pv, d5=d5, u13=u13, perlp=perlp, pvtable=pvtable,
             issue=issue)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="/tmp/in-v3")
    ap.add_argument("--a25", default="/tmp/in-a25")
    ap.add_argument("--d5")
    ap.add_argument("--part", default="reconcile,caps,picks,pv")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    co = Codes(a.v3, a.a25, a.d5)
    out = {k: [] for k in PARTS}
    for part in a.part.split(","):
        PARTS[part](co, out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
