"""Phase-2 arithmetic for E1's sections of Daniel's draft. Plain Python, no inputs.

Units: k in bits (1 kB = 8e3 bits, 2.5 MB = 2e7 bits). phi = 6.7e9 FLOPs/raw bit, kappa = 6 (T0 = 6 M).
K2 shape: d = 7168, aOut (fan-out e of the MLP input) = 37248 = 2*2048*9 + 384.
"""
kB = 8e3
phi, kap = 6.7e9, 6.0
d, e = 7168, 37248
htok = 17

print("== 5.1 column cap P_col = e * pbar, and where the joint form's leading term is tight (h <= 2 P_col)")
for pbar in (2.8, 3.0, 5.5):
    Pcol = e * pbar
    print(f"pbar {pbar}: P_col = {Pcol:.3g} bits = {Pcol/kB:.1f} kB; tight up to h = {2*Pcol/kB:.1f} kB")

print("== 5.1 replay-consistent beta_x = (htok + F)/d, F from B1 (Qwen d=1536), FP8")
for name, F in (("LoRA-16/full FT", 580), ("LoRA-16/full FT", 1320), ("small adapters", 2900), ("small adapters", 4900)):
    print(f"{name}: F={F}: beta_x <= {(htok+F)/1536:.2f} bits/coord")
bx_lo, bx_hi = (htok + 580) / 1536, (htok + 1320) / 1536

print("== 5.1 separate-budget estimate phi pbar beta_x/(kappa k), pbar FP8 2.8-5.5")
for k, lab in ((8e6, "1 MB"), (2e7, "2.5 MB")):
    old = (phi * 2.8 * 2.6 / (kap * k), phi * 5.5 * 4.7 / (kap * k))
    new = (phi * 2.8 * bx_lo / (kap * k), phi * 5.5 * bx_hi / (kap * k))
    print(f"{lab}: draft-rate {old[0]:.0f}-{old[1]:.0f}; replay-consistent {new[0]:.0f}-{new[1]:.0f}; joint x4 {4*new[0]:.0f}-{4*new[1]:.0f}")

print("== 5.1 fan-out face rho >= c beta_x/(kappa e (h - e pbar)), matched c = k phi")
for k, lab in ((8e6, "1 MB"), (2e7, "2.5 MB")):
    for bx in (bx_lo, bx_hi, 1.0, 2.6, 4.9):
        r = k * phi * bx / (kap * e * (k - e * 3.0))
        print(f"{lab} beta_x {bx:.2f}: {r:.3g}")
print("Lean FanOutCoverBound check alpha0 s f/(aOut k):", 0.147 * 4.52e16 / (37248 * 2e7))
print("gain of fan-out over joint form at h: h/(4 P_col):", {lab: round(k / (4 * e * 3.0), 1) for k, lab in ((8e6, '1MB'), (2e7, '2.5MB'))})
print("gain over separate form: h_w/P_col:", {lab: round(k / (e * 3.0), 1) for k, lab in ((8e6, '1MB'), (2e7, '2.5MB'))})

print("== cheap-token tile vs column tile at h (products per block), p=3, beta_marg=3")
for k in (8e6, 2e7):
    for F in (0, 2700, 6200):
        col = e * (k - e * 3) / 3
        tok = k * k / (4 * 3 * (htok + F)) if k / 6 <= d * e else None
        print(f"h={k/kB/1e3:.1f} MB F={F}: column {col:.3g}, full-token {tok:.3g}, ratio tok/col {tok/col:.3g}")

print("== 5.2 cheap-input accumulator corollary rho >= c h_eff b_ps/(kappa k^2), matched c")
for k, lab in ((8e6, "1 MB"), (2e7, "2.5 MB")):
    c = k * phi
    for bps in (1, 16):
        print(f"{lab} b_ps={bps}: {c*htok*bps/(kap*k*k):.3g}")
print("Prop 4 rescale 1.34e17/4.52e16:", 1.34e17 / 4.52e16, 1.6e4 * 1.34e17 / 4.52e16, 6.1e4 * 1.34e17 / 4.52e16)

print("== 5.3 rank threshold h_v < beta_v d (kB)")
rates = {"bf16 | A": (0.47, 3.0), "FP8 | A": (0.18, 0.63), "NVFP4 | A": (0.06, 0.37),
         "FP8 raw": (5.3, 11), "FP8 sweep | A (max)": (13, 13)}
for name, (lo, hi) in rates.items():
    print(f"{name}: d=7168 {lo*d/kB:.2f}-{hi*d/kB:.2f} kB; d=1536 {lo*1536/kB:.3f}-{hi*1536/kB:.3f} kB")
print("tokens an inference session holds at 0.56 kB:", 0.56 * kB / htok)
print("FP8|A estimate phi beta_v/(kappa d):", phi * 0.18 / (kap * d), phi * 0.63 / (kap * d))
print("conditioning share 1 - |A/raw:", 1 - 0.18 / 5.3, 1 - 0.63 / 11, " FP8|A over bf16|A:", 0.18 / 0.47, 0.63 / 3.0)
print("replay-consistent output-side rate at the next boundary = fan-out face: phi F/(kappa d e), F scaled d/1536:")
for F in (580, 1320):
    FV = F * d / 1536
    print(f"F={F} -> {FV:.0f} at d=7168: {phi*FV/(kap*d*e):.3g}")

print("== 5.4 packing g* = min(floor((k-s)/u), floor(c/F))")
F = 2e14
for k, s, u, c, lab in ((2e7, 49 * kB, 2.6 * kB, 1.34e17, "2.5MB matched, 9.8kB arm"),
                        (8e6, 49 * kB, 2.6 * kB, 1.4e17, "1MB fixed c, 9.8kB arm"),
                        (8e4, 6 * kB, 2.6 * kB, 8e4 * phi, "10kB matched, 1.2kB arm"),
                        (8e4, 6 * kB, 2.6 * kB, 1.4e17, "10kB fixed c, 1.2kB arm")):
    gin, gw = int((k - s) // u), int(c // F)
    g = min(gin, gw)
    print(f"{lab}: input-bound g {gin}, work-bound g {gw}, g* {g}, rho ~ c/(g F) = {c/(g*F):.2f}; data lower bound c u/(k F) = {c*u/(k*F):.2f}")

print("== P0: steps after registration that one block can replay, n0 <= c/F (seeded data)")
for lab, Fstep in (("GRPO Qwen-1.5B (2e14)", 2e14), ("frontier small-batch RL, 19200 tok x 6 x 49e9", 19200 * 6 * 49e9),
                   ("frontier pretraining step 1.2e18", 1.2e18), ("Pythia-160m step", 2e15)):
    print(f"{lab}: F={Fstep:.3g}, n0 max = {1.34e17/Fstep:.1f}")
print("Pythia-160m step data bits (1024 x 2048 x 17):", 1024 * 2048 * 17, "=", 1024 * 2048 * 17 / 8e6, "MB")
