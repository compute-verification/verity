"""The recursion's lambda (Claims.Recursion.lam) at the deployed parameters, mirroring the Lean definitions.

Lean sources (branch cursor/claims-recursion-ks-proof-f39a @ 76f4508e7b):
  Security/Definitions/Claims/Recursion.lean        NI, kappa, gamma, gammaE, cBe, lam, Nums.{k,kappa,gamma}
  Security/Proofs/Flock/Soundness/Accounting/Schedule.lean   fast100, queriesAt
  verity/core/service/flock/Flock/Zk.lean            queries0 (m1Queries0), PadLayout
  Security/Proofs/Flock/Claims/NumbersGamma.lean     weightOf
  Security/Proofs/Flock/Claims/NumbersPads.lean      padDomain (padMsg_eq)
  Security/Proofs/Flock/Claims/Deploy.lean           repRounds, rounds, V.k
  Security/Proofs/Flock/Claims/Numbers.lean          waits = 210, cap = 2^256, Flock kappa
  Security/Proofs/Flock/Claims/Recursion/BoundWeight.lean    coverW
"""
from math import ceil, e, log2, sqrt, exp

QUERIES_AT = {1: 218, 2: 106, 3: 71, 4: 53, 5: 43, 6: 36, 7: 32}
FAST100 = {22: (6, [9, 6, 3]), 23: (6, [10, 7, 4]), 24: (6, [11, 8, 5]), 25: (6, [12, 9, 6, 3]),
           26: (6, [13, 10, 7, 4]), 27: (6, [14, 11, 8, 5]), 28: (4, [17, 14, 11, 8, 5]),
           29: (5, [17, 14, 11, 8, 5]), 30: (6, [17, 14, 11, 8, 5]), 31: (6, [18, 15, 12, 9, 6, 3]),
           32: (6, [19, 16, 13, 10, 7, 4]), 33: (6, [20, 17, 14, 11, 8, 5]),
           34: (6, [21, 18, 15, 12, 9, 6, 3]), 35: (6, [22, 19, 16, 13, 10, 7, 4])}
WAITS, CAP = 210, 2 ** 256
EPS_TABLE = 2.0 ** -205  # NumbersTable: tableErr <= 2^-205 per table, plain and --zk (an upper bound)


def sched(m, zk):
    """[(logLen, queries)] per level, and k0. --zk: level 0's queries are queries0 (m1Schedule)."""
    k0, cols = FAST100[m]
    lv = []
    for l, c in enumerate(cols):
        r = 4 if (m == 22 and l == 2) else l + 1
        lv.append((c + r, QUERIES_AT[r]))
    if zk:
        q0 = {25: 277, 26: 242, 27: 229}.get(m, 219 if m <= 31 else 218)
        lv[0] = (lv[0][0], q0)
    return k0, lv


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def table_weight(m, klog, zk):
    k0, lv = sched(m, zk)
    N = next_pow2(396 + 2 * (m - 6) + 2 * (klog - 6) + 4 * k0) * 8 if zk else 0
    return 2 * sum(2 ** d * q for d, q in lv) + 2 * N * N + 2 + 4 * 2 ** lv[0][0] / e


def rep_rounds(m, klog, claims, zk):
    k0, lv = sched(m, zk)
    L = len(lv)
    if zk:
        return (m + (klog - 6)) + (claims + 1) + (k0 + 7 + 11 * (L - 1)) + 4
    return (3 + (m - 6)) + (3 + (klog - 6)) + (claims + 1) + (k0 + 5 + 11 * (L - 1))


def kE(N0):
    return ceil(4 * N0 / e)


def flock_kappa(eps, N0):
    k = kE(N0)
    return 4 * eps + 2.0 ** -WAITS + 2 * k / CAP + exp(-k / 8)


def flock_gamma(rounds, C):
    return 16 * (sqrt(2 * rounds * C) + rounds) ** 2


class Session:
    """One Flock session: verifier rounds V.k, weight C, N0, eps_table."""

    def __init__(self, Vk, C, N0, eps):
        self.Vk, self.C, self.N0, self.eps = Vk, C, N0, eps

    @classmethod
    def of_tables(cls, tables, zk):  # tables: [(m, klog, claims)]
        Vk = 2 + sum(2 * rep_rounds(m, kl, c, zk) for m, kl, c in tables)
        C = sum(table_weight(m, kl, zk) for m, kl, c in tables)
        N0 = sum(2 ** sched(m, zk)[1][0][0] for m, kl, c in tables)
        return cls(Vk, C, N0, len(tables) * EPS_TABLE)


def lam(inner, outers, w):
    """Claims.Recursion.lam with the inner record's weight at IVStar set to w (gamma_I = Flock gamma at w)."""
    NI = 1 + WAITS * kE(inner.N0)
    kI = inner.Vk
    kx = kI + 1 + sum(o.Vk for o in outers)  # inner rounds, kS = 1 staging round, each V* session's VO.k
    kappa = flock_kappa(inner.eps, inner.N0) + NI * sum(flock_kappa(o.eps, o.N0) for o in outers)
    gamma = (NI * sum(sqrt(flock_gamma(o.Vk, o.C) * kx) for o in outers)) ** 2
    gI = flock_gamma(kI, w)
    gammaE = (sqrt(gI * kx) + NI * sqrt(kI * kx)) ** 2
    cBe = 4 + 2 * sum(WAITS * kE(o.N0) for o in outers)
    t1 = log2(1 / kappa)
    t2 = 255.5 - log2(gamma) / 2
    t3 = 128 - log2(gammaE * cBe) / 2
    return dict(t1=t1, t2=t2, t3=t3, lam=min(t1, t2, t3) - 2, NI=NI, kI=kI, kx=kx, gammaE=gammaE, cBe=cBe,
                gI=gI)


def coverW(inner):
    NI = 1 + WAITS * kE(inner.N0)
    return (sqrt(inner.C) + NI * sqrt(inner.Vk / 32)) ** 2


# Inner: the deployed cap (outer-shape finding): m = 35, k_log = 24, claims = 8, plain (inner sessions are not --zk).
INNER_TABLE = (35, 24, 8)
# V* at an m = 35 inner table: 7 RecOpen levels (rec-reprice's staged sizes: m and block log = k_log) and the
# algebra's 3 parts at the cap; the parts' sizes are not recorded per part, so each is taken at the one-statement
# algebra's (dense m 29, k_log 26). Outer claims are not recorded: taken at 8, as the inner's.
VSTAR = [(32, 23, 8), (31, 23, 8), (31, 23, 8), (30, 23, 8), (30, 23, 8), (29, 22, 8), (28, 22, 8)] + [(29, 26, 8)] * 3


def inner_of(n, bounds):
    if bounds:  # REC1's: k_I = 4402 n, C_DE = n 2^33, N0 = n 2^23
        return Session(4402 * n, n * 2.0 ** 33, n * 2.0 ** 23, n * EPS_TABLE)
    return Session.of_tables([INNER_TABLE] * n, zk=False)


def outers_of(bounds):
    if bounds:  # Lean's per-table bounds at one --zk table each: V.k <= 4402, C <= 2^33, N0 <= 2^23
        return [Session(4402, 2.0 ** 33, 2.0 ** 23, EPS_TABLE) for _ in VSTAR]
    return [Session.of_tables([t], zk=True) for t in VSTAR]


if __name__ == "__main__":
    for label, bounds in (("deployed cap shape", False), ("Lean bounds (REC1's inner, one-table outer)", True)):
        print(f"== {label}")
        outs = outers_of(bounds)
        print("   V* sessions: Vk", [o.Vk for o in outs], " log2 C", [round(log2(o.C), 2) for o in outs],
              " log2 N0", [int(log2(o.N0)) for o in outs])
        for n in (1, 64, 200):
            inn = inner_of(n, bounds)
            cw = coverW(inn)
            a, b = lam(inn, outs, inn.C), lam(inn, outs, cw)
            print(f"   n={n:3d}  k_I={inn.Vk}  log2 C_DE={log2(inn.C):.2f}  log2 coverW={log2(cw):.2f}  "
                  f"log2 N_I={log2(a['NI']):.2f}  k(x)={a['kx']}  log2 cBe={log2(a['cBe']):.2f}")
            for tag, r in (("w=C_DE ", a), ("w=coverW", b)):
                print(f"     {tag}: log2(1/kappa)={r['t1']:.2f}  255.5-lg(gamma)/2={r['t2']:.2f}  "
                      f"128-lg(gE*cBe)/2={r['t3']:.2f}  lambda={r['lam']:.2f}  log2 gamma_I={log2(r['gI']):.2f}")
            print(f"     bits lost in the third term: {a['t3'] - b['t3']:.2f}")
