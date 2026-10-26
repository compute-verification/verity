"""machine-model theta [--check]           a device's θ (theta.toml): each figure against its document, then d by each
                                          delivery route, Strassen's ALU-for-tensor-core exchange rate, B_RF and
                                          c ≈ ρ₈/B_RF, and each honest profile's r_H against r_min and γ, each on
                                          the data sheet and measured, with its arithmetic; --check prints the
                                          figures alone; exits 1 when a figure doesn't hold"""
from __future__ import annotations

from verity_machine_model import theta as T


def add(sub) -> None:
    c = sub.add_parser("theta")
    c.add_argument("--check", action="store_true", help="the file's rules and its figures only")


def _filled(xs) -> str:
    return f" (sheet value for want of a run: {', '.join(xs)})" if xs else ""


def _mix(m: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.3f}" for k, v in m.items() if v >= 5e-4)


def run(args) -> int:
    t = T.theta(args.data)
    fs = T.figures(t)
    on_sheet = sum(r.sheet is not None for r in t.rates.values())
    measured = sum(r.measured is not None for r in t.rates.values())
    print(f"{t.gpu}: {len(t.rates)} rates ({on_sheet} on a vendor document, {measured} measured, "
          f"{len(t.rates) - measured} gaps), {len(t.capacities)} capacities, {len(t.paths)} exact paths")
    print(f"figures, 2 · sheet · {t.sms} SMs · clock against the document's (within {100 * T.TOL:g}%):")
    for f in fs:
        print(f"  {f.key:20} 2 · {t.rates[f.key].sheet:g} · {t.sms} · {f.mhz:g} MHz ({f.clock}) = {f.value:.2f} "
              f"against {f.figure:g}: {'holds' if f.holds else 'FAILS'}")
    ok = all(f.holds for f in fs)
    if args.check:
        return 0 if ok else 1
    print(f"\nd, one delivery in INT8 MAC units: ρ₈ ({T.INT8}) over the route's deliveries a SM-clock")
    for reading in T.READINGS:
        for r in T.routes(t, reading):
            print(f"  {reading:8} {r.name}: {r.rho8:g} / {r.per_clk:.4g} [{r.how}] = {r.d:.2f}{_filled(r.filled)}")
    hs = T.folds(t, args.data)
    L = hs[0].L
    s0 = T.strassen(t, "sheet", L)
    print(f"\nStrassen one level deep on R·Uᵀ inside each L = {L} segment: 7 products of k = {L // 2} for 8, "
          f"5/(4L) = {s0.extra:.5f} extra deliveries a naive MAC")
    for reading in T.READINGS:
        s = T.strassen(t, reading, L)
        head = (f"  {reading:8} tensor time 7/8 · ρ_k{L} / ρ_k{L // 2} = 7/8 · {s.rho_l:g} / {s.rho_half:g} = "
                f"{s.tensor:.4f} of naive{_filled(s.filled)}")
        if not s.rates:
            print(f"{head}: the level saves no tensor time, so there is no exchange")
            continue
        print(f"{head}; exchange 5d/(4L) / (1 − {s.tensor:.4f}), MAC-times a MAC saved:")
        for name, x in s.rates.items():
            print(f"             {name}: {x:.3f}")
    print("\nB_RF, the most register bits all units write a SM-clock (warp instructions a SM-clock at the optimum); "
          "c ≈ ρ₈/B_RF")
    for reading in (*T.READINGS, "measured only"):
        b = T.bound(t, reading)
        dropped = f"; left out, unmeasured: {', '.join(b.dropped)}" if b.dropped else ""
        print(f"  {reading:13} {b.words:.1f} words = {b.bits:.0f} bits; c = {b.rho8:g} / {b.bits:.0f} = {b.c:.4f}; "
              f"at {_mix(b.mix)}{_filled(b.filled)}{dropped}")
    best = min(hs, key=lambda f: f.r_min)
    print(f"\nr_H against r_min, lane instructions a committed word of each fold's classes; the least r_min here is "
          f"{best.fold}'s {best.r_min:g}, the fold's own count (no lower bound proves it)")
    for f in hs:
        print(f"  honest {f.profile} ({f.status}, {f.fold}, L = {f.L}): "
              + ", ".join(f"{c} {v:.4f} (the fold's {f.minimal[c]:g})" for c, v in f.per_word.items())
              + f": r_H = {f.r_h:.4f}, r_min = {f.r_min:g}")
        g = (f"{f.measured:g} / {f.lone:.2f} = {f.gamma:.3f}" if f.gamma is not None
             else "not measured (a provisional profile)")
        print(f"    γ, SMSP-cycles a MMA over the lone INT8 MMA's (words · depth · {T.SMSPS} / ρ₈ measured): {g}")
        for reading in T.READINGS:
            x = f.exchange[reading]
            tail = (f"; over the cheapest exchange {x:.3f}: {100 * f.eps(reading):.2f}%" if x is not None
                    else "; Strassen frees no tensor time to trade it for: 0")
            print(f"    {reading:8} slack (r_H − r_min) at each class's lane rate: {f.slack[reading]:.4f} tensor "
                  f"MAC-times a MAC{tail}")
    return 0 if ok else 1
