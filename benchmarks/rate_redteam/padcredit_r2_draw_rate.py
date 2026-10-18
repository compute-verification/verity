"""PADCREDIT R2: a selection unit is caught only at its own draw rate, so its gated cone multiplies the sampling term.

A two-call static window (STATIC-WIN's layout): call c has m_c batch slots (side A rows, ok_A,i gating the n_c tiles
of slot i) and n_c weight rows (side B rows, ok_B,j gating the m_c tiles of row j); a tile is (c, i, j) of weight w,
N = sum m_c n_c + 1 tiles. The draw names K' units of its population with replacement, each at its Q-mass.

Attacks, each gaining a fraction phi of the credited tiles at escape probability delta (no drawn unit is wrong):
* tiles: skip phi.N tiles (the baseline the window's K' is sized for);
* ok_B: forge ok_B = 1 on refused (dead) weight rows; each gains its m_c tiles, computed honestly on the dead row, so
  only the row unit's own draw catches it;
* ok_A: the same on an A slot, gaining its n_c tiles.
Draw laws: tiles only (today: row units not drawn, so selections are never caught); uniform over tiles and row units;
cone-proportional (row unit g at mass selConeWt(g) = (1 + gated tiles) w); and selections always proved.

Runs anywhere: ``uv run python benchmarks/rate_redteam/padcredit_r2_draw_rate.py``.
"""
import math

CALLS = [(16, 4096), (16, 14336)]
DELTA = 2.0 ** -40
EPS = 0.01
N = sum(m * n for m, n in CALLS) + 1
L = math.log(1 / DELTA)


def k_for(q_per_gain: float) -> float:
    """Draws K' so that a cheater gaining EPS of the tiles escapes with probability at most DELTA, when each unit of
    gain carries detection mass q_per_gain per draw: (1 - EPS.N.q)^K' <= DELTA."""
    return L / -math.log1p(-EPS * N * q_per_gain)


def phi_at(k: float, q_per_gain: float) -> float:
    """The fraction of tiles a cheater gains at escape DELTA with K' = k draws: (1 - phi.N.q)^k = DELTA."""
    return -math.expm1(-L / k) / (N * q_per_gain)


rows_a = sum(m for m, _ in CALLS)
rows_b = sum(n for _, n in CALLS)
k0 = k_for(1 / N)
print(f"window: calls (m, n) = {CALLS}, N = {N} tiles, rows A = {rows_a}, rows B = {rows_b}; eps = {EPS:.0%} at delta = 2^-40")
print(f"tiles-only K' for eps: {k0:.0f}")
print()
print("law                         attack   gain/unit  mass/draw (per gained tile)   phi at K'_0     K' for eps")
pop = N + rows_a + rows_b
cone_total = N + sum(n * (1 + m) for m, n in CALLS) + sum(m * (1 + n) for m, n in CALLS)
for law in ("uniform over tiles + rows", "cone-proportional"):
    for attack in ("tiles", "ok_B", "ok_A"):
        if attack == "tiles":
            gain, mass = 1, (1 / pop if law.startswith("uniform") else 1 / cone_total)
        else:
            m, n = CALLS[0]
            gain = m if attack == "ok_B" else n
            mass = (1 / pop) if law.startswith("uniform") else (1 + gain) / cone_total
        q = mass / gain
        phi = phi_at(k0, q)
        print(f"{law:27s} {attack:6s} {gain:9d}  {q:.3e}                      {min(phi, 1):8.4f}    {k_for(q):12.0f}")
print()
print("selection units never drawn (today: Pc8RowA/B are the committer's CPU forming): the ok attacks are never caught,")
print("phi = 1 at every K'.")
print(f"always proving every row unit: {rows_a + rows_b} forming proofs a window, against K'_0 = {k0:.0f} tile proofs;")
print(f"weights' ok_B proved once at weight registration, A rows only: {rows_a} a window.")
