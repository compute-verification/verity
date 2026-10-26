"""pw8 re-red-team (EndToEndWindow on cursor/pouw-work-bound-stmt2-c7a9 at 8814ab8bb, WindowEps's three links on
cursor/pouw-window-eps-stmt2-c7a9 at 01b0e2a4c): probes against main's window, draw, service and tile unit.

1. the draw: `core_check.stream_subset` (Lean's `Draw.subset`) is a uniform K-subset, and `window.sampling`'s law and
   `instance_ts`'s sessions are what both drafts model (`Law.subset n K`, `serviceLaw`).
2. ε at δ: EndToEndWindow's first term is (1 − W/(1000·N·w_max))^K′ with ε = 1/1000 fixed; at the served ε = 1%, δ = 2^-128
   windows K′ buys only about 2^-13 at that threshold, so the statement doesn't certify what those windows claim.
3. the deadline: `AcceptsBy` needs received <= t_d, but main's `Check.registration_deadline` defaults to None and served_zk
   never sets it, so the service accepts runs the statement doesn't cover.
4. ServedTie on main: the window's tile unit reads no -h3 leaf (no `tile_read` in any Program), ignores its index fields,
   and `window_problem` is Python only, so a window whose committed outputs aren't the served leaves is accepted.
5. ksJoint_le's κ = 2^logLen/(e·Kr) at main's fast100 level-0 lengths, and the Kr it needs.
6. the recheck of WindowEps at 373694a79: FlockKnowledgeSound at analysisZC (κ = 0, g over 1 − rateZ), sized at those
   lengths.

Runs on a CPU in seconds against a checkout of main: ``PYTHONPATH=MAIN uv run python benchmarks/rate_redteam/pw8_rered_probes.py``.
"""
from __future__ import annotations

import inspect
import math
import random
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as W
from verity.ml.definitions.pouw.circuit import pc8
from verity.core.protocols.sampled_proofs import service as S

ROOT = Path(W.__file__).resolve().parents[4]


def probe_draw() -> None:
    print("== 1. the draw ==")
    n, k, runs = 7, 3, 70_000
    rng = random.Random(8)
    counts = Counter()
    for _ in range(runs):
        src = rng.randbytes(2 * k * CC.width(n) + 64)
        counts[tuple(CC.stream_subset(k, n, src))] += 1
    subsets = list(combinations(range(n), k))
    exp = runs / len(subsets)
    chi2 = sum((counts[s] - exp) ** 2 / exp for s in subsets)
    print(f"  stream_subset({k}, {n}): {len(counts)} of {len(subsets)} subsets seen, chi2 = {chi2:.1f} on "
          f"{len(subsets) - 1} dof (5% critical ≈ 48.6)")
    work = W.bucket(1000)
    samp = W.sampling(Fraction(1, 1000), Fraction(1, 2 ** 40), work, 4096 * 64 * 64)
    k_ = samp["law"]["k"]
    reg = {"population": work.n, "law": samp["law"]}
    drawn = sorted(rng.sample(range(work.n), min(k_, work.n)))
    print(f"  bucket(1000): credited {work.credited}, N = {work.n} = pad(1001) = {W.pad(1001)}; law {samp['law']}")
    cap = 4
    if len(drawn) == k_:
        ts = [W.instance_ts(reg, drawn, s, cap) for s in range(W.sessions(k_, cap))]
        print(f"  sessions(K′={k_}, cap={cap}) = {len(ts)}; last session {ts[-1]} (filler N − 1 = {work.n - 1})")


def log2_miss(q: Fraction, k: int) -> float:
    return k * math.log2(float(q))


def log2_hyper(n: int, b: int, k: int) -> float:
    """log2 of C(n − b, k)/C(n, k): a uniform k-subset of n misses a fixed b-set."""
    if k > n - b:
        return -math.inf
    lc = lambda a, c: math.lgamma(a + 1) - math.lgamma(c + 1) - math.lgamma(a - c + 1)
    return (lc(n - b, k) - lc(n, k)) / math.log(2)


def probe_eps() -> None:
    print("== 2. ε at δ: EndToEndWindow's first term (1 − W/(1000·N·w_max))^K′ against the code's draw ==")
    for eps, delta, lo, hi in ((Fraction(1, 1000), Fraction(1, 2 ** 40), 87, 143),
                               (Fraction(1, 100), Fraction(1, 2 ** 128), 87, 143)):
        full, worst_bin, worst_hyp, worst_stmt, worst_stmt_rest, worst_prod = [], -1e9, -1e9, -1e9, -1e9, -1e9
        for ell in range(lo, hi + 1):
            w = W.Work.at(ell)
            k = W.k_prime(eps, delta, w.credited, w.n, 1)
            share = Fraction(w.credited, w.n)
            b = math.ceil(eps * w.credited)
            worst_hyp = max(worst_hyp, log2_hyper(w.n, b, k))
            stmt = log2_miss(1 - share / 1000, k)
            worst_stmt = max(worst_stmt, stmt)
            if k == w.n:
                full.append(ell)
                worst_prod = max(worst_prod, log2_miss(1 - Fraction(b, w.n), k))
            else:
                worst_bin = max(worst_bin, log2_miss(1 - eps * share, k))
                worst_stmt_rest = max(worst_stmt_rest, stmt)
        print(f"  ε = {eps}, δ = 2^{math.log2(float(delta)):.0f}, levels {lo}..{hi}: K′ = N (every tile drawn) at "
              f"{len(full)} levels{f' ({full[0]}..{full[-1]})' if full else ''}; elsewhere k_prime's (1 − p)^K′ <= "
              f"2^{worst_bin:.1f}; the true miss of an ε-set <= 2^{worst_hyp:.1f} over all levels")
        print(f"    the statement's first term at ε = 1/1000: <= 2^{worst_stmt_rest:.1f} where K′ < N, up to "
              f"2^{worst_stmt:.1f} where K′ = N (its docstring says it is at most k_prime's δ)")
        if full:
            print(f"    WindowEps's first term where K′ = N, the sessions' product law at served_zk's one unit a session "
                  f"((1 − b/N)^K′, b = ⌈ε·credited⌉): up to 2^{worst_prod:.1f}, where the service's draw misses nothing")
    print("  so at ε = 1% the statement bounds Pr[accept ∧ T < (1 − γ_u)(0.999)W] by about 2^-13, not the claimed "
          "(1 − γ_u)(0.99)W at 2^-128; and where K′ = N its binomial term exceeds δ though the draw misses nothing")


def probe_deadline() -> None:
    print("== 3. the registration deadline ==")
    f = S.Check.__dataclass_fields__["registration_deadline"]
    print(f"  Check.registration_deadline default: {f.default!r}")
    prog = "0" * 128
    c = S.Check(prog, {"query": "x"}, proof_deadline=60.0)
    print(f"  Check(program, query, proof_deadline=60) builds, registration_deadline = {c.registration_deadline!r}")
    sc = (ROOT / "benchmarks/pouw/served_zk/served_commit.py").read_text()
    job = (ROOT / "benchmarks/pouw/served_zk/job.sh").read_text()
    print(f"  served_commit.spec_of reads params.get('registration_deadline'): "
          f"{'registration_deadline=params.get(\"registration_deadline\")' in sc}; job.sh sets it: "
          f"{'registration_deadline' in job}")


def probe_served_tie() -> None:
    print("== 4. ServedTie on main ==")
    d = pc8.side_tile(4096)
    print(f"  {d.defn.name} reads {[n for n, _t in d.signature()[0]]}")
    body = inspect.getsource(pc8._tile_side_body)
    print(f"  its body passes ia/jb on: {'ia' in body.split('return', 1)[1] or 'jb' in body.split('return', 1)[1]}; "
          f"SIDE_PORTS regions {pc8.SIDE_PORTS['regions']}")
    readers = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "verity").rglob("*.py")
                     if "/tests/" not in str(p) and "tile_read(" in p.read_text() and p.name != "window.py")
    print(f"  modules outside window.py that call tile_read: {readers}")
    print(f"  window_problem is Python only: {'It is not proved yet: no Program states it' in W.__doc__}")


def probe_kappa() -> None:
    print("== 5. ksJoint_le's κ = 2^logLen/(e·Kr) at fast100's level-0 lengths ==")
    level0 = {22: 10, 25: 13, 28: 18, 31: 19, 35: 23}
    for m, ll in level0.items():
        kr = math.ceil(2 ** (ll + 40) / math.e)
        adv = 2 ** -385
        best = 2 * math.sqrt(2 ** ll * adv / math.e)
        print(f"  m = {m}: logLen = {ll}; κ <= 2^-40 needs Kr >= 2^{math.log2(kr):.1f}, and then Kr·adv₀ <= 2^-40 needs "
              f"adv₀ <= 2^{-40 - math.log2(kr):.1f}; at adv₀ = q²/2^513, q = 2^64: min over Kr ≈ 2^{math.log2(best):.0f}")
    print("  a session state holding a collision of H has adv₀ = 1, so κ + g >= 1 there: the term needs counted session "
          "queries to be small")


def probe_analysis_zc() -> None:
    print("== 6. FlockKnowledgeSound at analysisZC: κ = 0, g = (ε⁻ + √(k·adv₀))/(1 − rateZ), rateZ = 2^logLen/(e·k) ==")
    level0 = {22: 10, 25: 13, 28: 18, 31: 19, 35: 23}
    adv = 2.0 ** -385
    for m, ll in level0.items():
        kmin = 2 ** ll / math.e
        k2 = math.ceil(2 * kmin)
        best = min((math.sqrt(k * adv) / (1 - kmin / k), k) for k in (math.ceil(kmin * 2 ** (i / 8)) for i in range(1, 400)))
        print(f"  m = {m}: logLen = {ll}; rateZ < 1 needs k > 2^{math.log2(kmin):.1f}; at k = 2^{math.log2(k2):.1f} "
              f"(rateZ = 1/2) g = 2·ε⁻ + 2^{math.log2(2 * math.sqrt(k2 * adv)):.1f}; min over k of the √ term "
              f"2^{math.log2(best[0]):.1f} at k = 2^{math.log2(best[1]):.1f}")
    print("  at adv₀ = q²/2^513, q = 2^64, the √ term is negligible and g is about ε⁻/(1 − rateZ); at a state holding a "
          "collision of H adv₀ = 1 and g >= 1, so g is small only on average over H at counted queries (CollCounted)")


if __name__ == "__main__":
    probe_draw()
    probe_eps()
    probe_deadline()
    probe_served_tie()
    probe_kappa()
    probe_analysis_zc()
