"""BET-BANDWIDTH (CMP-122): what the output-bandwidth bound of the ACCEL brief's 4(g) certifies on the H200, exactly.

The bound: an accepted run writes every committed transcript bit into a register (premise ABSORB), so its time is at
least F n^3 / B_RF, where F is the committed bits a credited MAC (32 / L at prefix stride L) and B_RF is the most
register bits a second all of the card's units can write together under (G2). This script computes, in exact rationals
at the H200's rates (CALIB's theta, `benchmarks/pouw/machine_model/data/devices/sm_90/theta.toml` on
`cursor/compute-research-accel-calib-ea24`; the sheet reading and the measured one):

- `B_RF`: the LP over CALIB's register writers (`[[writer]]`), with issue and every pipe as (G2) constraints;
- per protocol setting (shape, L, fold, whether the fold's states are committed too): the honest kernel's time a MAC
  under perfect overlap, its gamma against plain INT8 wgmma, and three readings of what the floor certifies:
  `c_4g` = (F / B_RF) / t_hon, the brief's time form; `c_px`, the best single price vector (FreedOfFloor's mu), an LP;
  `c_R` = 1 - eps under R-ACCEL-1's sum LP (`Period.FreedWithin`), eps the most freed compute a credited MAC, where the
  other work Y is exact INT8 MACs (on the tensor cores alone, or on any unit), valued one a MAC; and `c_DT`, the
  constant of `FreedOfFloor`'s floor premise at the honest problem's dual price;
- the hashing cap: forcing every committed word through SHA-512 (4.79 G compressions a second, CMP-107) caps c_4g at
  1024 rho_SHA / B_RF at every L, and costs gamma = (F / 1024) rho_8 / rho_SHA;
- the F that c_R >= 0.99 needs, and the gamma an honest kernel must pay to commit it.

Every honest time is a lower bound (perfect overlap of pipes, each at its peak), and every adversary is held only to
the bound's floor, which is what a proof from the floor alone must cover.

    uv run python benchmarks/pouw/theory/bet_bandwidth.py
"""
from __future__ import annotations

from fractions import Fraction as Q

SMS = 132
FLEET = 1000
PERIODS = {"week": 7, "month": 30, "year": 365}
CLK = {"tensor": Q(1830 * 10**6), "fmax": Q(1980 * 10**6)}
RHO_SHA = Q(479, 100) * 10**9  # SHA-512 compressions a second on the H200 (CMP-107, note:compute-accounting/20261025T2330Z-report-h200-pouw)

# Per SM-clock, by theta key: (sheet, measured or None, clock).
THETA = {
    "tensor.s8.k32": (Q(4096), Q("4095.6"), "tensor"),
    "tensor.s8.k16": (Q(4096), Q("1995.7"), "tensor"),
    "tensor.f16.k16.sync": (Q(2048), Q("1343.8"), "tensor"),
    "tensor.tf32.sync": (Q(1024), Q("671.9"), "tensor"),
    "issue": (Q(4), Q("3.998"), "fmax"),
    "alu": (Q(64), Q("63.01"), "fmax"),
    "imad": (Q(64), Q("63.99"), "fmax"),
    "imad.wide": (Q(32), None, "fmax"),
    "dp4a": (Q(64), None, "fmax"),
    "ffma": (Q(128), Q("127.9"), "fmax"),
    "dfma": (Q(64), None, "fmax"),
    "smem": (Q(128), Q("127.96"), "fmax"),
}
WGMMA_SLOT = {"sheet": Q(1), "measured": Q("62.741")}  # JI90_0: a wgmma's SMSP-cycles beside saturated wgmma
OTHER_SLOT = {"sheet": Q(1), "measured": Q("0.981")}


def rate(key: str, reading: str) -> Q:
    """Units a second over the card."""
    sheet, measured, clock = THETA[key]
    per_clk = measured if reading == "measured" and measured is not None else sheet
    return per_clk * SMS * CLK[clock]


def resources(reading: str) -> dict[str, Q]:
    """(G2) capacities a second: `tc` in busy seconds, `issue` in SMSP-cycles, pipes in warp instructions, smem in B."""
    return {
        "tc": Q(1),
        "issue": 4 * SMS * CLK["fmax"],
        "alu": rate("alu", reading) / 32,
        "imad": rate("imad", reading) / 32,
        "fma": rate("ffma", reading) / 32,
        "fp64": rate("dfma", reading) / 32,
        "smem": rate("smem", reading),
    }


def writers(reading: str) -> dict[str, tuple[Q, dict[str, Q], Q]]:
    """CALIB's `[[writer]]` classes: name -> (bits written a warp instruction, resource use a warp instruction, MACs)."""
    s, o = WGMMA_SLOT[reading], OTHER_SLOT[reading]
    imad_wide_per_imad = rate("imad", reading) / rate("imad.wide", reading)
    return {
        "wgmma m64n256k32 s8 (SS)": (Q(4096 * 32), {"tc": Q(131072) / rate("tensor.s8.k32", reading), "issue": s,
                                                     "smem": Q(2560)}, Q(131072)),
        "mma.sync m16n8k16 s8": (Q(128 * 32), {"tc": Q(2048) / rate("tensor.s8.k16", reading), "issue": o}, Q(2048)),
        "mma.sync m16n8k8 f16.f32": (Q(128 * 32), {"tc": Q(1024) / rate("tensor.f16.k16.sync", reading), "issue": o},
                                     Q(1024)),
        "mma.sync m16n8k4 tf32": (Q(128 * 32), {"tc": Q(512) / rate("tensor.tf32.sync", reading), "issue": o}, Q(512)),
        "LDS.128": (Q(128 * 32), {"issue": o, "smem": Q(512)}, Q(0)),
        "DFMA": (Q(64 * 32), {"issue": o, "fp64": Q(1)}, Q(0)),
        "IMAD.WIDE": (Q(64 * 32), {"issue": o, "imad": imad_wide_per_imad}, Q(0)),
        "IMAD": (Q(32 * 32), {"issue": o, "imad": Q(1)}, Q(0)),
        "IADD3/LOP3/SHF/LEA.HI": (Q(32 * 32), {"issue": o, "alu": Q(1)}, Q(0)),
        "FFMA": (Q(32 * 32), {"issue": o, "fma": Q(1)}, Q(0)),
    }


def simplex(c: list[Q], A: list[list[Q]], b: list[Q]) -> tuple[Q, list[Q], list[Q]]:
    """max c.x st A x <= b, x >= 0, b >= 0, exactly (Bland's rule). Returns the value, x and the duals of the rows."""
    m, n = len(A), len(c)
    assert all(bi >= 0 for bi in b)
    T = [list(A[i]) + [Q(int(i == j)) for j in range(m)] + [b[i]] for i in range(m)]
    z = [-ci for ci in c] + [Q(0)] * m + [Q(0)]
    basis = [n + i for i in range(m)]
    while True:
        col = next((j for j in range(n + m) if z[j] < 0), None)
        if col is None:
            break
        rows = [(T[i][-1] / T[i][col], basis[i], i) for i in range(m) if T[i][col] > 0]
        if not rows:
            raise ValueError("unbounded")
        _, _, r = min(rows)
        p = T[r][col]
        T[r] = [v / p for v in T[r]]
        for i in range(m):
            if i != r and T[i][col] != 0:
                f = T[i][col]
                T[i] = [vi - f * vr for vi, vr in zip(T[i], T[r])]
        if z[col] != 0:
            f = z[col]
            z = [vz - f * vr for vz, vr in zip(z, T[r])]
        basis[r] = col
    x = [Q(0)] * (n + m)
    for i, j in enumerate(basis):
        x[j] = T[i][-1]
    return z[-1], x[:n], z[n:n + m]


def lp(objective: dict[str, Q], rows: list[tuple[dict[str, Q], Q]]) -> tuple[Q, dict[str, Q], list[Q]]:
    """`simplex` on named variables, with its optimality certificate checked: the duals are feasible and close the gap."""
    names = sorted({k for k in objective} | {k for row, _ in rows for k in row})
    c = [objective.get(k, Q(0)) for k in names]
    A = [[row.get(k, Q(0)) for k in names] for row, _ in rows]
    b = [cap for _, cap in rows]
    value, x, duals = simplex(c, A, b)
    assert all(y >= 0 for y in duals) and all(
        sum(duals[i] * A[i][j] for i in range(len(A))) >= c[j] for j in range(len(c)))
    assert sum(y * bi for y, bi in zip(duals, b)) == value == sum(cj * xj for cj, xj in zip(c, x))
    assert all(sum(A[i][j] * x[j] for j in range(len(c))) <= b[i] for i in range(len(A)))
    return value, dict(zip(names, x)), duals


def capacity_rows(reading: str, use: dict[str, dict[str, Q]]) -> list[tuple[dict[str, Q], Q]]:
    """One (G2) row a resource over variables whose resource use a unit is `use[var]`."""
    caps = resources(reading)
    return [({v: u[r] for v, u in use.items() if r in u}, cap) for r, cap in caps.items()]


def b_rf(reading: str) -> tuple[Q, dict[str, Q]]:
    """The most register bits a second, over every mix of writers (G2)."""
    w = writers(reading)
    value, x, _ = lp({k: bits for k, (bits, _, _) in w.items()}, capacity_rows(reading, {k: u for k, (_, u, _) in w.items()}))
    return value, {k: v for k, v in x.items() if v}


# ---- honest settings ------------------------------------------------------------------------------------------------
# A fold's lane instructions a word absorbed (ALU pipe, IMAD pipe): duo is LOP3 + LEA.HI + IMAD (segment._duo_absorb),
# li one LOP3 and one IMAD a pair, tree3 the fan-in floor 1 / (f - 1) at f = 3 (ISA-FANIN), split over both pipes.
FOLDS = {"duo": (Q(2), Q(1)), "li": (Q(1, 2), Q(1, 2)), "tree3": (Q(1, 4), Q(1, 4))}

# shape -> (theta rate key, MACs a warp instruction, words a warp instruction, operand MACs a smem byte, RF-loaded?)
SHAPES = {
    "wgmma s8 k32": ("tensor.s8.k32", Q(131072), Q(4096), Q(512, 10), False),
    "mma.sync s8 k16": ("tensor.s8.k16", Q(2048), Q(128), Q(32), True),
    "mma.sync f16 k8": ("tensor.f16.k16.sync", Q(1024), Q(128), Q(16), True),
    "mma.sync tf32 k4": ("tensor.tf32.sync", Q(512), Q(128), Q(8), True),
}


def honest_use(reading: str, shape: str, L: int, fold: str) -> tuple[dict[str, Q], Q]:
    """The honest kernel's resource use a credited MAC, and its register bits written a MAC."""
    key, macs, words, intensity, loads = SHAPES[shape]
    s = WGMMA_SLOT[reading] if shape.startswith("wgmma") else OTHER_SLOT[reading]
    o = OTHER_SLOT[reading]
    alu, imad = FOLDS[fold]
    wpm = Q(1, L)
    use = {"tc": 1 / rate(key, reading), "issue": s / macs, "smem": 1 / intensity,
           "alu": alu * wpm / 32, "imad": imad * wpm / 32}
    use["issue"] += o * (alu + imad) * wpm / 32
    written = words * 32 / macs + (alu + imad) * wpm * 32
    if loads:
        use["issue"] += o / (intensity * 512)
        written += 8 / intensity
    return use, written


def t_of(reading: str, use: dict[str, Q]) -> Q:
    caps = resources(reading)
    return max(u / caps[r] for r, u in use.items())


def c_price(reading: str, use: dict[str, Q], F: Q) -> Q:
    """max over price vectors lam of min over writers of F (lam . writer time a bit) / (lam . honest time a MAC)."""
    caps = resources(reading)
    w = writers(reading)
    hon = {r: u / caps[r] for r, u in use.items()}
    rows = [({"t": Q(1), **{r: -F * (u / caps[r]) / bits for r, u in wu.items()}}, Q(0)) for bits, wu, _ in w.values()]
    rows.append(({r: v for r, v in hon.items()}, Q(1)))
    value, _, _ = lp({"t": Q(1)}, rows)
    return value


# ---- R-ACCEL-1's sum LP ---------------------------------------------------------------------------------------------
def jobs(reading: str, menu: str) -> dict[str, dict[str, Q]]:
    """Other work Y, exact INT8 MACs, value 1 a MAC: resource use a MAC. `tc` is the INT8 wgmma GEMM alone."""
    o = OTHER_SLOT[reading]
    out = {"Y wgmma s8": {"tc": 1 / rate("tensor.s8.k32", reading), "issue": WGMMA_SLOT[reading] / 131072,
                          "smem": Q(10, 512)}}
    if menu == "all":
        dp4a_per_imad = rate("imad", reading) / rate("dp4a", reading)
        out |= {
            "Y IDP.4A": {"issue": o / 128, "imad": dp4a_per_imad / 128},
            "Y FFMA": {"issue": o / 32, "fma": Q(1, 32)},
            "Y DFMA": {"issue": o / 32, "fp64": Q(1, 32)},
            "Y IMAD": {"issue": o / 32, "imad": Q(1, 32)},
        }
    return out


def c_dither(reading: str, menu: str, F: Q, hon: dict[str, Q]) -> Q:
    """DITHER-THM's form (`Models.Pouw.Guarantees.Dither.FreedOfFloor`): the largest c with mu . X >= c mu . (W hon)
    for every X the floor admits, at mu the dual of the honest problem (its kernel or none, beside Y), which
    complementary slackness pins. That is F (mu's cheapest written bit) / (mu . hon)."""
    Y = jobs(reading, menu)
    _, _, duals = lp({**{k: Q(1) for k in Y}, "honest": Q(1)}, capacity_rows(reading, {**Y, "honest": hon}))
    mu = dict(zip(resources(reading), duals))
    bit = min(sum(mu[r] * u for r, u in wu.items()) / bits for bits, wu, _ in writers(reading).values())
    return F * bit / sum(mu[r] * u for r, u in hon.items())


def freed(reading: str, menu: str, F: Q, w0: Q | None, hon: dict[str, Q]) -> tuple[Q, Q, Q]:
    """(eps, W, Y on the tensor cores): the adversary held only to the floor (register bits >= F a credited MAC)
    against the best honest play (its kernel `hon`, used a MAC, or none, beside Y). `w0` caps the adversary's credited
    MACs a second (None: its best). eps(W) does not grow with W (V_adv is concave), so the cap's small W is its sup."""
    Y = jobs(reading, menu)
    w = writers(reading)
    honest_value, _, _ = lp({**{k: Q(1) for k in Y}, "honest": Q(1)}, capacity_rows(reading, {**Y, "honest": hon}))
    use = {**Y, **{"x " + k: u for k, (_, u, _) in w.items()}}
    floor = ({"W": F, **{"x " + k: -bits for k, (bits, _, _) in w.items()}}, Q(0))
    extra = [({"W": Q(1)}, w0)] if w0 is not None else []
    value, x, _ = lp({**{k: Q(1) for k in Y}, "W": Q(1)}, capacity_rows(reading, use) + [floor] + extra)
    W = x.get("W", Q(0))
    return ((value - honest_value) / W if W else Q(0)), W, x.get("Y wgmma s8", Q(0))


def fmt(q: Q, digits: int = 3) -> str:
    v = float(q)
    return f"{v:.{digits}g}" if abs(v) < 1e4 else f"{v:.3e}"


def main() -> None:
    rho8 = rate("tensor.s8.k32", "sheet")
    t_plain = {r: 1 / rate("tensor.s8.k32", r) for r in ("sheet", "measured")}
    print("H200 (sm_90), CALIB theta; per-second rates at the tensor (1830 MHz) and fmax (1980 MHz) clocks\n")
    brf = {}
    for reading in ("sheet", "measured"):
        brf[reading], mix = b_rf(reading)
        print(f"B_RF ({reading}): {fmt(brf[reading])} bits/s = {fmt(brf[reading] / (SMS * CLK['fmax']))} bits a SM-clock at fmax")
        w = writers(reading)
        for k, v in mix.items():
            print(f"    {k:28s} {fmt(v * w[k][0])} bits/s")
    tc_max = Q(128 * 32, 2048) * rate("tensor.s8.k16", "sheet")
    print(f"tensor cores' most register bits (smallest k, every format): {fmt(tc_max)} bits/s = {fmt(tc_max / brf['sheet'])} of B_RF")
    print(f"plain INT8 at L = 32 commits {fmt(32 * rho8 / 32)} bits/s: c_4g <= {fmt(rho8 / brf['sheet'])} with no fold")
    sha_cap = 1024 * RHO_SHA / brf["sheet"]
    print(f"\nhashing cap (every committed bit through SHA-512): c_4g <= 1024 rho_SHA / B_RF = {fmt(sha_cap)} at every L;"
          f" gamma_hash = 32 rho_8 / (1024 L rho_SHA) = {fmt(32 * rho8 / (1024 * RHO_SHA))} / L"
          f" ({fmt(32 * rho8 / (1024 * RHO_SHA * 32))} at L = 32; gamma_hash <= 10 needs L >= {fmt(32 * rho8 / (1024 * RHO_SHA * 10))})")

    settings = [
        ("wgmma s8 k32", 32, "duo", False), ("wgmma s8 k32", 32, "li", False), ("wgmma s8 k32", 32, "tree3", False),
        ("wgmma s8 k32", 64, "li", False),
        ("mma.sync s8 k16", 16, "duo", False), ("mma.sync s8 k16", 16, "li", False),
        ("mma.sync s8 k16", 16, "tree3", False), ("mma.sync s8 k16", 16, "tree3", True),
        ("mma.sync f16 k8", 8, "tree3", False), ("mma.sync f16 k8", 8, "tree3", True),
        ("mma.sync tf32 k4", 4, "tree3", True),
    ]
    rows = []
    print("\nc_R = 1 - sup (freed / credited MAC) and c_DT = DITHER-THM's floor constant at the dual price, each with Y on"
          " the tensor cores alone (TC) and on every unit (all)")
    print("setting                              reading   F b/MAC  gamma   c_4g   c_px   c_R: TC all       c_DT: TC all"
          "       gamma_hash(all F)")
    for shape, L, fold, commit_fold in settings:
        for reading in ("sheet", "measured"):
            use, written = honest_use(reading, shape, L, fold)
            alu, imad = FOLDS[fold]
            F = Q(32, L) * (1 + (alu + imad if commit_fold else 0))
            t = t_of(reading, use)
            gamma = t / t_plain[reading]
            c4g = F / brf[reading] / t
            cpx = c_price(reading, use, F)
            w0 = rate("tensor.s8.k32", reading) / 10**6
            eps_tc, _, _ = freed(reading, "tc", F, w0, use)
            eps_all, _, _ = freed(reading, "all", F, w0, use)
            cdt_tc = c_dither(reading, "tc", F, use)
            cdt_all = c_dither(reading, "all", F, use)
            g_hash = F * rho8 / (1024 * RHO_SHA)
            name = f"{shape}, L={L}, {fold}{' +states' if commit_fold else ''}"
            rows.append((name, reading, gamma, c4g, cpx, 1 - eps_all))
            print(f"{name:36s} {reading:9s} {fmt(F):7s} {fmt(gamma):7s} {fmt(c4g):6s} {fmt(cpx):6s} "
                  f"{fmt(1 - eps_tc):4s} {fmt(1 - eps_all):8s} {fmt(cdt_tc):4s} {fmt(cdt_all):8s} {fmt(g_hash)}")

    print("\nR-ACCEL-1 sum LP, adversary at its best credited rate (eps there; the sup is the marginal eps above):")
    for reading in ("sheet", "measured"):
        today, _ = honest_use(reading, "wgmma s8 k32", 32, "duo")
        for F in (Q(1), Q(2), Q(8), Q(32)):
            eps, W, y_tc = freed(reading, "all", F, None, today)
            peak = rate("tensor.s8.k32", reading)
            print(f"    {reading:9s} F = {fmt(F):3s} bits/MAC: eps = {fmt(eps)}, credited {fmt(W / peak)} of the INT8 peak,"
                  f" Y wgmma {fmt(y_tc / peak)} of it")

    print("\nthe F that c_R >= 0.99 needs (all-unit Y, marginal), and the least gamma an honest kernel pays to commit it:")
    for reading in ("sheet", "measured"):
        w0 = rate("tensor.s8.k32", reading) / 10**6
        today, _ = honest_use(reading, "wgmma s8 k32", 32, "duo")
        eps1, _, _ = freed(reading, "all", Q(1), w0, today)
        d_bit = 1 - eps1
        F_need = Q(99, 100) / d_bit
        eps_chk, _, _ = freed(reading, "all", F_need, w0, today)
        g_writes = Q(3, 2) * F_need * rate("tensor.s8.k32", reading) / brf[reading]
        g_tc = F_need / 2
        print(f"    {reading:9s} value a written bit {fmt(d_bit, 4)} MACs; F >= {fmt(F_need, 4)} bits/MAC (c_R there "
              f"{fmt(1 - eps_chk, 4)}); gamma >= 1.5 F rho_8 / B_RF = {fmt(g_writes)} (fold at the fan-in floor, every "
              f"writer at its peak), >= F / 2 = {fmt(g_tc)} if the words come off the tensor cores")
        cap = d_bit * brf[reading] / rate("tensor.s8.k32", reading)
        print(f"    {reading:9s} so at every setting the share of pooled compute R-ACCEL-1 certifies is at most "
              f"d_bit B_RF / rho_8 = {fmt(cap)} (c_R credited = d_bit F a credited MAC, and F a second <= B_RF)")
        Y = jobs(reading, "all")
        _, _, duals = lp({k: Q(1) for k in Y}, capacity_rows(reading, Y))
        mu = dict(zip(resources(reading), duals))
        per_price = {k: bits / sum(mu[r] * u for r, u in wu.items()) for k, (bits, wu, _) in writers(reading).items()}
        computing = max(v for k, v in per_price.items() if not k.startswith("LDS"))
        print(f"    {reading:9s} c_DT ceiling for computed (not loaded) words, at every gamma: bits a unit of mu of the best "
              f"computing writer / the cheapest writer = {fmt(computing)} / {fmt(max(per_price.values()))} = "
              f"{fmt(computing / max(per_price.values()))}")

    print(f"\nrule L, per {FLEET} H200s at the INT8 sheet peak, whole fleet on the setting (sheet reading); the model term"
          " is the same at delta = 1% and 1e-6, and the sampling terms only lower what is certified:")
    sheet_rows = [r for r in rows if r[1] == "sheet"]
    picks = {"today (wgmma, L=32, duo)": sheet_rows[0],
             "best c_R at gamma <= 10": max((r for r in sheet_rows if r[2] <= 10), key=lambda r: r[5]),
             "best c_4g at gamma <= 10": max((r for r in sheet_rows if r[2] <= 10), key=lambda r: r[3])}
    for label, (name, _, gamma, c4g, _, cR) in picks.items():
        print(f"    {label}: {name}, gamma {fmt(gamma)}, c_R {fmt(cR)}, c_4g {fmt(c4g)} (c_4g under ABSORB only)")
        for period, days in PERIODS.items():
            pooled = FLEET * rho8 * days * 86400
            credited = pooled / gamma
            print(f"        {period:5s} pooled {fmt(pooled)} MACs; credited {fmt(credited)} ({fmt(1 / gamma)} of pooled);"
                  f" certified <= {fmt(cR * credited)} under R-ACCEL-1 ({fmt(cR / gamma)} of pooled),"
                  f" <= {fmt(c4g * credited)} by the time form ({fmt(c4g / gamma)})")


if __name__ == "__main__":
    main()
