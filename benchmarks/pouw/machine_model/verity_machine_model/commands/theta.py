"""machine-model theta [--check]           a device's θ (theta.toml): each figure against its document, then d by each
                                          delivery route and the mma.sync seed, Strassen's ALU-for-tensor-core
                                          exchange rate (at the honest L and the no-fold L), B_RF and c ≈ ρ₈/B_RF,
                                          each honest profile's r_H against r_min and γ, those ratios per second,
                                          every unit's verdict, and SHA-512 against the public figures and
                                          the card's ceiling, each on the data sheet and measured, with its
                                          arithmetic; --check prints the figures alone; exits 1 when a figure
                                          doesn't hold"""
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
    print(f"  seed: mma.sync D ≠ C writes C, with the products it computes anyway, into a new accumulator: marginal d 0 "
          f"while those products are needed; charged its issue slot, ρ₈ / (issue · 128 words) = "
          f"{T.seed(t, 'sheet'):.2f} / {T.seed(t, 'measured'):.2f} (sheet / measured)")
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
    print(f"  at the no-fold receipt's L, where the k = L/2 products are k32 MMAs at full rate (post-additions only; the "
          f"pre-additions and operand fix-ups uncounted), the cheapest route's exchange:")
    for big in T.NOFOLD_L:
        cells = []
        for reading in T.READINGS:
            s = T.strassen(t, reading, big)
            name, x = min(s.rates.items(), key=lambda kv: kv[1]) if s.rates else ("no exchange", None)
            cells.append(f"{reading} {x:.4g} ({name})" if x is not None else f"{reading}: {name}")
        print(f"    L = {big}: " + "; ".join(cells))
    print("\nB_RF, the most register bits all units write a SM-clock (warp instructions a SM-clock at the optimum); "
          "c ≈ ρ₈/B_RF")
    for reading in (*T.READINGS, "measured only"):
        b = T.bound(t, reading)
        dropped = f"; left out, unmeasured: {', '.join(b.dropped)}" if b.dropped else ""
        print(f"  {reading:13} {b.words:.1f} words = {b.bits:.0f} bits; c = {b.rho8:g} / {b.bits:.0f} = {b.c:.4f}; "
              f"at {_mix(b.mix)}{_filled(b.filled)}{dropped}")
    print("  each writer alone, register words a SM-clock (rate / per · words; the LP adds issue and the joint rows), "
          "sheet / measured (* the sheet's, for want of a run):")
    for w in t.writers:
        xs = [t.at(w.rate, rd) for rd in T.READINGS]
        print(f"    {w.name}: " + " / ".join("none" if v is None else f"{v / w.per * w.words:.1f}{'*' if f else ''}"
                                             for v, f in xs))
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
    _per_second(t, L)
    _units(t)
    _sha(t)
    return 0 if ok else 1


def _per_second(t: T.Theta, L: int) -> None:
    print(f"\nper second, the tensor figures at their clock and the lanes at {t.clocks['fmax'].mhz:g} MHz: each "
          f"tensor-over-lane ratio above times the clocks' ratio (1 where the card's figures share one clock); "
          f"c with every writer at the lane clock")
    for reading in T.READINGS:
        k = T.tick(t, reading)
        d = min(r.d for r in T.routes(t, reading) if r.name in T.ADDING)
        s = T.strassen(t, reading, L)
        x = f"{min(s.rates.values()):.3f} → {k * min(s.rates.values()):.3f}" if s.rates else "none"
        c = T.bound(t, reading).c
        print(f"  {reading:8} × {k:.4f}: least adding d {d:.2f} → {k * d:.2f}; exchange at L = {L} {x}; "
              f"c {c:.4f} → {k * c:.4f}")


def _units(t: T.Theta) -> None:
    print(f"\nunits ({len(T.UNITS)}), each with its verdict; a bounded or open unit's most work a SM-clock "
          f"over the rate it stands in for, on the sheet / measured:")
    for verdict, meaning in T.VERDICTS.items():
        us = [u for u in t.units.values() if u.verdict == verdict]
        if not us:
            continue
        print(f"  {verdict}: {meaning}")
        for u in us:
            line = f"    {u.key}: {u.what}"
            if (s := T.share(t, u, "sheet")) is not None:
                m = T.share(t, u, "measured")
                line += f"; ≤ {u.bound:g} against {u.against}: {100 * s:.2f}% / {100 * m:.2f}%"
            print(line + (f"; settles it: {u.gap}" if u.gap else ""))
    for reading in T.READINGS:
        tot = T.free(t, reading)
        print(f"  {reading:8} bounded units together add at most "
              + (", ".join(f"{100 * v:.2f}% of {k}" for k, v in tot.items()) or "nothing"))


def _sha(t: T.Theta) -> None:
    if "sha512" not in t.rates:
        return
    r = t.rates["sha512"]
    print(f"\nSHA-512, compressions a SM-clock: sheet {r.sheet:g} ({r.grade}); ours {r.measured:g} ({r.runs})")
    for s in T.sha(t):
        p = s.public
        print(f"  {p.form:10} {p.impl} on the {p.card}: {p.gps:g} G/s over {p.sms} SMs at ≤ {p.mhz:g} MHz = "
              f"≥ {s.per_clk:.6f}; ours over it {r.measured / s.per_clk:.3f}, c_sha ≤ {s.c:.3f}")
    if t.sha_mix is None:
        return
    m = t.sha_mix
    print(f"  the card's ceiling for the built kernel ({m['alu']} ALU, {m.get('imad', 0)} IMAD, {m.get('other', 0)} other "
          f"instructions; {m['rotates']} rotations and {m['adds']} adds ({m['add_form']}) IMAD can take; "
          f"{m['source']}): an LP over ALU, IMAD and issue")
    for hi in (True, False):
        for reading in T.READINGS:
            c = T.sha_ceiling(t, reading, hi)
            print(f"    {'IMAD.HI' if hi else 'no IMAD.HI':10} {reading:8} {c.clocks:.2f} SM-clocks (built "
                  f"{c.built:.2f}; rotations {c.hi_rotates:.0f} by IMAD.HI, {c.shf_rotates:.0f} by SHF, {c.adds:.0f} adds "
                  f"to IMAD) = {c.rate:.6f}; ours over it {r.measured / c.rate:.3f}")
    print(f"    ALU and IMAD on this mix: {m['overlap']}")
