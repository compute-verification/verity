#!/usr/bin/env python3
"""Referee check of QSTAR's ISA-CAP (note:20261017T0344Z-draft-eps1-qstar, the packed case): incidences one instruction
creates.

ISA-CAP: every sm_120 ALU instruction creates ≤ 32/7 ≈ 4.57 incidences, with "IMAD ≤ 3 by Kronecker packing". Outer
products pack more, inside class P: an R-only word times a U-only word is a mixed-radix tuple of single-entry fields
φ = R[i, 32ℓ+p]·U[j, p] (129 values each), every one a field no input holds.
- IMAD, 8-bit slots (s = 16, 2 rows × 2 columns): 4 in one 32-bit word.
- IMAD.WIDE, 8-bit slots (s = 24, 3 rows × 2 columns): 6. At radix 129 (`isa_cap.c`): 8, four in each destination word.
- IDP.4A.U8.S8 at radix 129 (`isa_cap.c`): a 1-entry and a 4-entry field in one 32-bit word, 5. So no cap below 5 holds
  even counted per destination word.
Each is checked over every R bit and U entry (U ∈ ±1..±64). The joint statement's load bound is then restated as a
function of the cap κ: Ld ≥ (28 − 1.03κ)N/(c_L − κ).

  uv run python benchmarks/pouw/theory/isa_cap_check.py [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent
U_VALUES = np.r_[-64:0, 1:65]


def outer_imad(rows, cols, s, width, signed=True):
    """Every (R bits, U values): the destination of a·b + c mod 2^width, and whether each slot (i, j) at bit 8i + s·j holds
    R[i]·U[j] + 64 exactly. IMAD.WIDE is signed s32 × s32 → s64, so b keeps its sign there (signed=False is .U32); IMAD's
    low word is the same either way."""
    ok = True
    offs = [8 * i + s * j for i in range(rows) for j in range(cols)]
    assert len(set(offs)) == len(offs) and max(offs) + 8 <= width and all(
        abs(x - y) >= 8 for x, y in itertools.combinations(offs, 2))
    c = sum(64 << o for o in offs)
    us = np.array(list(itertools.product(U_VALUES, repeat=cols)), dtype=object)
    for r in itertools.product((0, 1), repeat=rows):
        a = sum(int(x) << (8 * i) for i, x in enumerate(r))
        assert a < 1 << 32
        for u in us:
            b = sum(int(x) << (s * j) for j, x in enumerate(u))
            assert -(1 << 31) <= b < 1 << 31
            if width == 32 or not signed:
                b %= 1 << 32
            d = (a * b + c) % (1 << width)
            for i in range(rows):
                for j in range(cols):
                    ok &= ((d >> (8 * i + s * j)) & 0xFF) - 64 == r[i] * int(u[j])
            if not ok:
                return False
    return ok


def radix_packings(radix, tmp):
    """`isa_cap.c` at this radix: {"IDP4A": (cases, fails), "IMADW": (cases, fails)}."""
    exe = Path(tmp) / f"isa_cap{radix}"
    subprocess.run(["gcc", "-O2", "-fopenmp", f"-DRADIX={radix}", "-o", str(exe), str(HERE / "isa_cap.c")], check=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return {name: (int(cases), int(fails)) for name, cases, fails in (line.split() for line in out.splitlines())}


def load_bound(cap, c_load, bytes_per_load=32):
    """Bytes of table loads a word, from 28N ≤ κA + c_L·Ld with A + Ld < 1.03N (G < 2.03N)."""
    return bytes_per_load * (28 - 1.03 * cap) / (c_load - cap)


def table_eps(a, cap, c_load, honest=1.877):
    """SPEC's table route read as one route (D2): words at 1 + A + Ld slots and Ld = (28 − κA)/c_L loads, as many as the
    level's bytes allow, each saving (honest − slots)/32. Up to the bandwidth and bytes a load, which scale it, this is
    the saving per load; honest is VF3o-16's Dither-like threshold."""
    ld = (28 - cap * a) / c_load
    return (honest - 1 - a - ld) / ld


def main() -> int:
    checks = Checks()
    checks.check("IMAD (r1 + 2^8 r2)·(u1 + 2^16 u2) + c writes 4 single-entry fields R[i]·U[j] in 8-bit slots of one "
                 "32-bit word, exhaustively over R ∈ {0,1}², U ∈ (±1..±64)²: 4 incidences, not QSTAR's ≤ 3",
                 outer_imad(2, 2, 16, 32), "slots at bits 0, 8, 16, 24")
    checks.check("IMAD.WIDE (r1 + 2^8 r2 + 2^16 r3)·(u1 + 2^24 u2) + c writes 6 fields (low word 4, high word 2), "
                 "exhaustively: 6 incidences in one instruction, above ISA-CAP's 32/7 ≈ 4.57",
                 outer_imad(3, 2, 24, 64), "slots at bits 0, 8, 16, 24, 32, 40")
    with tempfile.TemporaryDirectory() as tmp:
        got = radix_packings(129, tmp)
        bad = radix_packings(128, tmp) if checks.mutations else None
    cases, fails = got["IDP4A"]
    checks.check("IDP.4A.U8.S8 with a = (r1 + 129 r2_0, 129 r2_1, 129 r2_2, 129 r2_3) and b = U[j, p_0..3] writes "
                 "r1·u_0 and Σ_k r2_k·u_k into one 32-bit word, at every R ∈ {0,1}^5 and U ∈ (±1..±64)^4: 5 incidences "
                 "in one word, so no cap below 5 holds even per destination word", cases == 1 << 33 and fails == 0,
                 f"{cases} cases, {fails} wrong")
    cases, fails = got["IMADW"]
    checks.check("signed IMAD.WIDE with a = r1 + 2^29 r2 and b = 8 Σ_j 129^j u_j writes four r1·u_j into the low word and "
                 "four r2·u_j into the high word, at every R ∈ {0,1}² and U ∈ (±1..±64)^4: 8 incidences in one "
                 "instruction, 4 a destination word", cases == 1 << 30 and fails == 0, f"{cases} cases, {fails} wrong")
    b_wide = 8 * 64 * sum(129 ** j for j in range(4))
    p_ok = 8 * 129 ** 4 <= 1 << 32 and 129 * (128 * 4 + 1) <= 1 << 32 and 1 + 129 <= 255 and b_wide < 1 << 31
    checks.check("every destination word is a class-P tuple from an R-only and a U-only free input: IMAD.WIDE's words "
                 "4 fields of 129 values (8·129^4 ≤ 2^32 for the low word's ×8), IDP.4A's a 129-value field under a "
                 "513-value one (M_2 = 129), IDP.4A's operand bytes ≤ 255, IMAD.WIDE's b in s32", p_ok,
                 f"8·129^4 = {8 * 129 ** 4}, 129·513 = {129 * 513}, |b| ≤ {b_wide}")
    caps = {"32/7": 32 / 7, "5 a word": 5, "8": 8, "27": 27}
    table = {k: (load_bound(v, 128), load_bound(v, 216)) for k, v in caps.items()}
    checks.check("the joint statement's load bound as a function of the cap κ, single-field L2 tables (c_L = 128) / any "
                 "format (216): 6.0 / 3.5 B at 32/7, 5.9 / 3.5 B at 5 a destination word (IMAD.WIDE as 2 slots), "
                 "5.3 / 3.0 B at 8 an instruction, 0.06 B at Theorem C's own 27; the 1.22 issue floor needs only κ ≤ 128",
                 abs(table["32/7"][0] - 6.04) < 0.01 and abs(table["32/7"][1] - 3.53) < 0.01
                 and 5.25 < table["8"][0] < 5.3 and 3.0 < table["8"][1] < 3.1 and table["27"][0] < 0.1,
                 "; ".join(f"κ = {k}: {a:.2f} / {b:.2f} B" for k, (a, b) in table.items()))
    formats = (128, 216, 32, 108)

    def best_a(cap, c_load):
        a_max = min(28 / cap, (1.877 - 1 - 28 / c_load) / (1 - cap / c_load))
        grid = np.linspace(0, a_max * (1 - 1e-9), 2001)
        return grid[int(np.argmax([table_eps(a, cap, c_load) for a in grid]))]

    flip = 28 / (1.877 - 1)
    checks.check("SPEC's per-level table ε doesn't move with κ: read as one route, d(saving per load)/dA has the sign of "
                 "κ(honest − 1) − 28 whatever c_L, so no ALU creation pays below κ = 28/0.877 ≈ 31.9; the best point "
                 "stays at A = 0 at κ = 32/7, 5 and 8 on every format, and leaves it at κ = 33",
                 all(best_a(cap, c) == 0 for cap in (32 / 7, 5, 8) for c in formats)
                 and all(best_a(33, c) > 0 for c in (128, 216, 108)) and 31.8 < flip < 32,
                 f"flip at κ = {flip:.2f}; best A at κ = 33: {[round(float(best_a(33, c)), 3) for c in (128, 216, 108)]}")
    if checks.mutations:
        checks.mutation("IMAD.WIDE.U32 (b read unsigned) breaks the high word's fields: the 6-field route needs the "
                        "signed form", not outer_imad(3, 2, 24, 64, signed=False))
        checks.mutation("radix 128 is one value short for a 1-entry field: both packings fail", bad["IDP4A"][1] > 0
                        and bad["IMADW"][1] > 0, f"IDP.4A {bad['IDP4A'][1]}, IMAD.WIDE {bad['IMADW'][1]} wrong")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
