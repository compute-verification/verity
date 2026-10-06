"""Torch-free core of the B-Ligero ``q`` leak (red-team-zk; `reports/redteam_zk.md` finding 1).

The chain message ``q_d = Q_d + Z_H s_chain_d`` of ``archive/direct/ligero/protocol.py`` equals ``Q_d`` on ``H``
(the mask vanishes there), and ``Q_d(h_j)`` is the per-column share of the cross-column constraints.  Given the
verifier's coefficients ``rc`` and the public statement, walking each VU from its public ``c_0 = 0`` recovers
every intermediate accumulator: ``D`` equations in the 3 unknown components per step.  This module is the
reference used by the fixture loader test (`fixtures/redteam-zk`); the torch harness
(`archive/direct/ligero/redteam/q_leak.py`) calls the same recovery.
"""
from __future__ import annotations

P = 15 * 2**27 + 1
GEN = 31


def _inv(a: int) -> int:
    return pow(a % P, P - 2, P)


def fold_to_H(q: list[int], l: int) -> list[int]:
    """Evaluations on ``H = <w_l>`` (natural order) of the polynomial with coefficients ``q``: reduce mod ``x^l - 1``,
    then a plain DFT (fine for the fixture sizes)."""
    acc = [0] * l
    for i, c in enumerate(q):
        acc[i % l] = (acc[i % l] + c) % P
    w = pow(GEN, (P - 1) // l, P)
    out = []
    for k in range(l):
        wk = pow(w, k, P)
        s, x = 0, 1
        for c in acc:
            s = (s + c * x) % P
            x = x * wk % P
        out.append(s)
    return out


def solve_overdetermined(A: list[list[int]], b: list[int]) -> tuple[list[int] | None, bool]:
    """Unique solution of a full-rank u x u subsystem of the e x u system and whether all e equations hold."""
    e, u = len(A), len(A[0])
    rows = [[x % P for x in row] + [bb % P] for row, bb in zip(A, b)]
    for col in range(u):
        pr = next((i for i in range(col, e) if rows[i][col]), None)
        if pr is None:
            return None, False
        rows[col], rows[pr] = rows[pr], rows[col]
        iv = _inv(rows[col][col])
        rows[col] = [(x * iv) % P for x in rows[col]]
        for i in range(e):
            if i != col and rows[i][col]:
                f = rows[i][col]
                rows[i] = [(x - f * y) % P for x, y in zip(rows[i], rows[col])]
    return [rows[i][u] for i in range(u)], all(rows[i][u] == 0 for i in range(u, e))


def layout_masks(l: int, n_vus: int, steps: int) -> tuple[list[int], list[int], list[int]]:
    """``chain.Layout.masks``: start / link / end indicator per column."""
    start, link, end = [], [], []
    for j in range(l):
        in_vu = j < steps * n_vus
        i = j % steps
        start.append(int((not in_vu) or i == 0))
        end.append(int(in_vu and i == steps - 1))
        link.append(int(in_vu and i != steps - 1))
    return start, link, end


def recover(QH: list[list[int]], rc: list[list[list[int]]], y16: list[int], Yc: list[int], Y16c: int,
            l: int, n_vus: int, steps: int) -> dict:
    """``QH``: (D, l) values of ``q`` on H; ``rc``: (D, 7, l) verifier coefficients (link x3, start x3, end);
    ``Yc``: constant terms of the three output-component expressions; ``Y16c``: of the epilogue word.
    Returns the recovered accumulator components entering every column, and the consistency flags."""
    D = len(QH)
    start, link, end = layout_masks(l, n_vus, steps)
    rec = [[0, 0, 0] for _ in range(l)]
    solved = [False] * l
    consistent = [False] * l
    end_ok = []
    for v in range(n_vus):
        j0 = steps * v
        c_prev = [0, 0, 0]
        solved[j0] = consistent[j0] = True
        for i in range(steps):
            j = j0 + i
            known = []
            for d in range(D):
                k = 0
                for x in range(3):
                    rl_prev = rc[d][x][j - 1] * link[j - 1] if j >= 1 else 0
                    k += rl_prev * c_prev[x] + rc[d][3 + x][j] * start[j] * c_prev[x]
                    k += rc[d][x][j] * link[j] * Yc[x]
                k += rc[d][6][j] * end[j] * ((y16[j] - Y16c) % P)
                known.append(k % P)
            if i < steps - 1:
                A = [[rc[d][x][j] * link[j] % P for x in range(3)] for d in range(D)]
                bvec = [(known[d] - QH[d][j]) % P for d in range(D)]
                sol, ok = solve_overdetermined(A, bvec)
                if sol is None:
                    break
                rec[j + 1] = sol
                solved[j + 1] = True
                consistent[j + 1] = ok
                c_prev = sol
            else:
                end_ok.append(all((known[d] - QH[d][j]) % P == 0 for d in range(D)))
    return {"components": rec, "solved": solved, "consistent": consistent, "end_positions_consistent": end_ok}


def decode_fp32(word: int) -> tuple[int, int, int]:
    return (word >> 31) & 1, (word >> 23) & 0xFF, word & 0x7FFFFF


def matches_recorded(rec: dict, accs_per_vu: list[list[int]], steps: int) -> tuple[int, int]:
    """(exact matches, total) of the recovered accumulator entering column ``j0 + i + 1`` against ``accs[i]``."""
    total = match = 0
    for v, accs in enumerate(accs_per_vu):
        j0 = steps * v
        for i in range(steps - 1):
            total += 1
            if tuple(rec["components"][j0 + i + 1]) == decode_fp32(int(accs[i])):
                match += 1
    return match, total
