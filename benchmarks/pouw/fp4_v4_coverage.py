"""Pearl-C4's honest per-tile replayed debit on a real model, for v1 and the candidate v4 (rank-2 FFMA noise), on the basis
the cap tests: terms 1, 3 and 4 of `theory-pearl-c4-domain.md` §2, term 4 on the reference's witness (W), the 90·ρ² screen
in term 3. Adapted from bc-a8466279's `internal/pouw/cheap-binding/pearlc4-coverage.py` (v1's numbers reproduce it).

The witness forms each block under the extreme draws of the design's noise support, C + N and C − N with N its largest
noise magnitude, and v1's clean draw C as well (v4 has no zero-noise draw: its codes are nonzero and cannot cancel); an
element whose code is the same under every draw is counted salt-dead (an over-count, so the debit is sound). The block's
other elements are taken at their clean values, as in the reference (for v4 they carry at most 0.94·a in the witness
draws, which this stand-in leaves out). v1: N = 72·a_E (the atom, E ≡ ±6 and F ≡ ±6 at 2^-4). v4: N = a·36·(J1max + 5/4·J2max), from nonzero E2M1 row and
column codes and the per-row dyadic jitters J in [1, 5/4) (`docs/pouw/pearl-c4-v4.md`).

    python benchmarks/pouw/fp4_v4_coverage.py Qwen/Qwen2.5-0.5B --design v4 --cap 0.001 --out v4-qwen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_emulation import E2M1_HALVES, e2m1_rn, ue4m3_dyadic, ue4m3_rn  # noqa: E402

T = 64
KB1 = 16 / np.sqrt(2346.125)
#: v4's noise, per unit a: n = J1·e1·f1 + (5/4)·J2·e2·f2 over nonzero E2M1 codes (E[(ef)²] = (137/14)²) and J uniform
#: in [1, 5/4) (E[J²] = 61/48); a = ue4m3(σα·KB4) makes the noise's standard deviation σα.
V4_VAR = (137 / 14) ** 2 * (61 / 48) * (1 + 25 / 16)
KB4 = 1 / np.sqrt(V4_VAR)
J_MAX = (2 ** 14 + 4095) / 2 ** 14
N4 = 36 * (J_MAX + 1.25 * J_MAX)
DESIGNS = {
    #        noise magnitude bound / a,  a's constant, credited f_s (FP4 units), fused column, peel per word
    "v1": dict(nmax=72.0, kb=KB1, fs=107.34, fused=32, peel=256, clean_draw=True),
    "v4": dict(nmax=N4, kb=KB4, fs=4 * 8.38 + 12.675 + 2 * 8.38 + 2 * 8.0, fused=8, peel=64, clean_draw=False),
}


def bf16(x):
    a = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    r = ((a + 0x7FFF + ((a >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32).astype(np.float64)


def uval(code):
    m, e = ue4m3_dyadic(code)
    return m * np.exp2(e)


def e2v(code):
    mag = E2M1_HALVES[code & 7] / 2
    return np.where(code & 8, -mag, mag)


def stats(X, kb):
    s = np.abs(X).max(1)
    rho2 = (X[:, ::8] ** 2).mean(1)
    sig = np.sqrt(rho2) / 4
    alpha = 2688 / np.maximum(s + 4 * sig, 2.0 ** -32)
    ae = uval(ue4m3_rn(sig * alpha * kb))
    return alpha, ae, ae >= 2.0 ** -6, rho2


def witness(X, alpha, ae, nmax, clean_draw=True):
    R, k = X.shape
    C = (alpha[:, None] * X).reshape(R, k // 16, 16)
    A = np.abs(C)
    top = A.max(2, keepdims=True)
    second = np.sort(A, axis=2)[:, :, -2:-1]
    rest = np.where(A == top, second, top)
    N = (nmax * ae)[:, None, None]
    vals = []
    for d in ((C + N, C, C - N) if clean_draw else (C + N, C - N)):
        r = 1 / np.maximum(uval(ue4m3_rn(np.maximum(np.abs(d), rest) / 6)), 2.0 ** -9)
        vals.append(e2v(e2m1_rn(d * r)))
    same = vals[0] == vals[-1]
    for v in vals[1:-1]:
        same &= vals[0] == v
    return same.reshape(R, k)


def side(X, p):
    alpha, ae, dnf, rho2 = stats(X, p["kb"])
    dead = np.zeros(X.shape, bool)
    for a in range(0, len(X), 512):
        dead[a:a + 512] = witness(X[a:a + 512], alpha[a:a + 512], ae[a:a + 512], p["nmax"], p["clean_draw"])
    scr = (np.abs(X).reshape(len(X), -1, 16).max(2) ** 2) > 90 * rho2[:, None]
    d4 = dead.reshape(len(X), -1, 4).sum(2)
    units = np.where(d4 == 4, 4, np.where(d4 >= 2, 2, 0)).sum(1).astype(np.float64)
    return dict(dnf=dnf, units=units, scr=scr.astype(np.float64), sdw=dead.sum(1).astype(np.float64),
                dead_share=float(dead.mean()))


def linear(X, W, rng, p, max_cols=24):
    X, W = bf16(X), bf16(W)
    m, k = X.shape
    n = W.shape[0]
    m -= m % T
    cols = sorted(rng.choice(n // T, min(n // T, max_cols), replace=False))
    Wj = np.concatenate([W[c * T:(c + 1) * T] for c in cols])
    a, b = side(X[:m], p), side(Wj, p)
    cw = k + (p["fused"] + p["fs"]) * k / n + p["peel"]
    out = []
    for r in range(m // T):
        I = slice(r * T, (r + 1) * T)
        ok = a["dnf"][I]
        if ok.sum() == 0:
            continue
        nI = ok.sum()
        for c in range(len(cols)):
            J = slice(c * T, (c + 1) * T)
            cred = nI * T * cw
            t1a = T * a["units"][I][ok].sum() / cred
            t1b = nI * b["units"][J].sum() / cred
            t3 = 16 * (a["scr"][I][ok] @ b["scr"][J].T).sum() / cred
            t4 = T * p["fs"] / n * a["sdw"][I][ok].sum() / cred
            out.append((cred, t1a, t1b, t3, t4))
    o = np.array(out)
    deb = o[:, 1:].sum(1)
    iw = int(np.argmax(deb))
    return dict(k=k, n=n, tiles=len(out), macs=float(m * n * k), dnf_fail=float(1 - a["dnf"].mean()),
                dead_share_a=a["dead_share"], cred=o[:, 0].tolist(), debit=deb.tolist(),
                max_debit=float(deb.max()), mean_debit=float((o[:, 0] * deb).sum() / o[:, 0].sum()),
                worst_terms=[float(v) for v in o[iw, 1:]])


def two_level(res: dict, rho_u: float, rho_t: float) -> dict:
    """A unit (one linear) is admitted at ρ_u when its credit-weighted mean tile debit is at most ρ_u; inside an admitted
    unit a tile whose debit exceeds the per-tile allowance ρ_t is left uncredited.  The honest completeness is the
    admitted credit's share, MAC-weighted over the model."""
    macs = sum(r["macs"] for r in res.values())
    kept = 0.0
    out_units = []
    for nm, r in res.items():
        if r["mean_debit"] > rho_u:
            out_units.append(nm)
            continue
        kept += r["macs"] * sum(c for c, d in zip(r["cred"], r["debit"]) if d <= rho_t) / sum(r["cred"])
    return {"rho_u": rho_u, "rho_t": rho_t, "credit_admitted": kept / macs, "units_over_rho_u": len(out_units),
            "their_mac_share": sum(res[nm]["macs"] for nm in out_units) / macs,
            "units_without_2x_margin": sum(r["mean_debit"] > rho_u / 2 for r in res.values()),
            "worst_unit_mean": max(r["mean_debit"] for r in res.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--design", choices=sorted(DESIGNS), default="v4")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--text", default=str(Path(__file__).resolve().parents[2] / "README.md"))
    ap.add_argument("--caps", default="0.0025,0.001")
    ap.add_argument("--fuse", action="store_true",
                    help="replay q/k/v and gate/up as the fused GEMMs a server runs (one A′ formed per shared input)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = DESIGNS[args.design]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[:, :args.tokens]
    lins = {nm: mod for nm, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and "lm_head" not in nm}
    got = {}

    def hook(nm):
        def f(mod, a):
            got[nm] = a[0][0].detach().to(torch.bfloat16).clone()      # the model's BF16 input, exact at half the memory
        return f

    hs = [mod.register_forward_pre_hook(hook(nm)) for nm, mod in lins.items()]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    rng = np.random.default_rng(20260930)
    groups: dict[str, list[str]] = {}
    for nm in lins:
        key = nm
        if args.fuse:
            head, _, leaf = nm.rpartition(".")
            if leaf in ("q_proj", "k_proj", "v_proj"):
                key = head + ".qkv_proj"
            elif leaf in ("gate_proj", "up_proj"):
                key = head + ".gate_up_proj"
        groups.setdefault(key, []).append(nm)
    res = {}
    for key, names in groups.items():
        xs = [got.pop(nm) for nm in names]
        if any(not torch.equal(xs[0], x) for x in xs[1:]):
            raise ValueError(f"{key}: fused linears read different inputs")
        w = np.concatenate([lins[nm].weight.detach().float().numpy() for nm in names])
        res[key] = linear(xs[0].float().numpy(), w, rng, p)
    macs = sum(r["macs"] for r in res.values())
    summ = {"model": args.model, "design": args.design, "tokens": int(ids.shape[1]), "linears": len(res),
            "dnf_fail_share": sum(r["dnf_fail"] * r["macs"] for r in res.values()) / macs,
            "worst_tile_debit": max(r["max_debit"] for r in res.values()), "caps": {}}
    for cap in (float(c) for c in args.caps.split(",")):
        rej = {nm: sum(c for c, d in zip(r["cred"], r["debit"]) if d > cap) / sum(r["cred"]) for nm, r in res.items()}
        summ["caps"][str(cap)] = {
            "rejected_credit_share": sum(rej[nm] * r["macs"] for nm, r in res.items()) / macs,
            "linears_with_rejections": sum(v > 0 for v in rej.values()),
            "linears_without_2x_margin": sum(r["max_debit"] > cap / 2 for r in res.values()),
            "their_mac_share": sum(r["macs"] for nm, r in res.items() if r["max_debit"] > cap / 2) / macs}
    summ["two_level"] = [two_level(res, 0.001, rt) for rt in (0.001, 0.0025, 0.005, 0.01)]
    downs = sorted((nm for nm in res if nm.endswith("down_proj")), key=lambda s: int(s.split(".")[2]))
    summ["first_last_down_proj"] = [res[downs[0]]["max_debit"], res[downs[-1]]["max_debit"]]
    summ["every_down_proj"] = [min(res[d]["max_debit"] for d in downs), max(res[d]["max_debit"] for d in downs)]
    print(json.dumps(summ, indent=1), flush=True)
    if args.out:
        for r in res.values():
            r["cred"] = [round(c) for c in r["cred"]]
            r["debit"] = [round(d, 7) for d in r["debit"]]
        Path(args.out).write_text(json.dumps({"summary": summ, "linears": res}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
