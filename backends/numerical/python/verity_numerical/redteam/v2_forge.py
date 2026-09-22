"""red-team-v2: concrete forgeries against the checker-v2 provers.

Two families:

* ``gkr_chain`` builds adversarial VU directories for Candidate A's ``verity-gkr run-vu`` / ``negatives-vu`` in
  ``--variant v2`` (Goldilocks): honest 96/steps-per-VU chains from the constant zero state whose *rows* each
  satisfy the unit circuit, with exactly one chain-level edit (a non-initial unit-0 state; an epilogue that reads a
  non-final unit's state; a flipped ``ovf`` into the epilogue; two VUs cross-linked; a B=1 batch).  The unit circuit
  and the ``chain.txt`` linkage are v2's own (``gkr_export.v2``); we only rewrite the committed rows, so any
  rejection is the linkage / boundary, not row validity.  Each is expected to be REJECTED; the loader test and the
  pod run (``verity-gkr negatives-vu``) record ACCEPTED/REJECTED.

* ``zero_swap`` answers the b-air-v2 VACUOUS claim adversarially (battery ``zero_swap`` under v2): is there any
  unit input where a ``-0`` incoming state (sign 1, M = 0) and a ``+0`` incoming state (sign 0, M = 0) produce a
  *different accepted output word*?  We run the real v2 reference (``checker.v2.step``) on both states over a large
  operand sweep and report the first divergence, or confirm the claim.

The gkr dirs are small (2-3 VUs x a few steps): safe on the laptop.
"""
from __future__ import annotations

import json
import random
import shutil
import struct
from pathlib import Path

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.v2.gadgets import Pack
from verity_numerical.checker.v2.step import gen_unit, vu_output, zero_state

from verity_numerical.gkr_export.v2 import (
    epilogue_witness_v2,
    unit_witness_v2,
    verify_row,
    vu_rows_v2,
)
from verity_numerical.gkr_export.vu import field_api

STEP = 16
GOLDILOCKS = 2**64 - 2**32 + 1


def _read_rows(path: Path) -> tuple[int, list[list[int]]]:
    b = path.read_bytes()
    n, m = struct.unpack("<QQ", b[:16])
    rows = []
    off = 16
    for _ in range(n):
        rows.append(list(struct.unpack(f"<{m}Q", b[off:off + 8 * m])))
        off += 8 * m
    return m, rows


def _write_rows(rows: list[list[int]], path: Path, m: int | None = None) -> None:
    ncols = len(rows[0]) if rows else (m or 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(rows), ncols))
        for r in rows:
            f.write(struct.pack(f"<{ncols}Q", *(v % GOLDILOCKS for v in r)))


def _bf16(rng: random.Random) -> int:
    return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(2**7)


def build_gkr_chain(out: Path, p: Params = REAL, seed: int = 424242) -> dict:
    """Honest base dir + one edit per adversarial dir under ``out/neg/<name>/`` (negatives-vu layout)."""
    api = field_api("goldilocks", "v2")
    unit, epi = api.build_unit(), api.build_epilogue()
    uctx, upaths = unit
    ectx, epaths = epi
    m = api.modulus
    out.mkdir(parents=True, exist_ok=True)
    api.export_circuit(uctx, out / "circuit.txt")
    api.export_circuit(ectx, out / "epilogue.txt")
    api.export_extra(uctx, ectx, out)  # chain.txt
    rng = random.Random(seed)
    steps = 3
    vus = 2

    # honest base
    all_rows: list[list[int]] = []
    erows: list[list[int]] = []
    y16s: list[int] = []
    per_vu_rows: list[list[list[int]]] = []
    for _ in range(vus):
        a = [_bf16(rng) for _ in range(steps * STEP)]
        b = [_bf16(rng) for _ in range(steps * STEP)]
        y16 = vu_output(p, a, b)
        rows, erow = vu_rows_v2(unit, epi, a, b, y16, p=p, modulus=m)
        per_vu_rows.append(rows)
        all_rows += rows
        erows.append(erow)
        y16s.append(y16)
    _write_rows(all_rows, out / "units.bin", len(uctx.cols))
    _write_rows(erows, out / "epilogue.bin", len(ectx.cols))
    _write_rows([[y] for y in y16s], out / "public.bin", 1)

    neg = out / "neg"
    neg.mkdir(exist_ok=True)
    G = len(p.groups)
    # column indices
    st_out_off = 9 + 2 * sum(p.groups)  # unused; state_out at 5+2k..9+2k
    cases: list[tuple[str, list[list[int]], list[list[int]], list[int], str]] = []

    # (1) honest control: must be ACCEPTED
    cases.append(("honest", list(all_rows), list(erows), list(y16s),
                  "the honest 2-VU batch; positive control"))

    # (2) unit-0 non-initial state: set VU0 unit0 incoming state (x.e) to floor+1, its row regenerated honestly
    a0 = [all_rows[i] for i in range(steps)]
    # regenerate VU0 unit0 from a forged incoming state (non-zero e) with its own honest row
    ops0_a = [all_rows[0][5 + i] for i in range(STEP)]
    ops0_b = [all_rows[0][5 + STEP + i] for i in range(STEP)]
    alt_state = (0, p.floor + 1, 0, 1)  # not the constant zero state (e != floor)
    h, s_out, ov = gen_unit(p, alt_state, ops0_a, ops0_b)
    row0 = unit_witness_v2(uctx, upaths, alt_state, ops0_a, ops0_b, s_out, h)
    assert verify_row(uctx, row0, m) is None, "forged unit-0 row must be row-valid"
    broken = list(all_rows)
    broken[0] = row0
    cases.append(("unit0_noninitial_state", broken, list(erows), list(y16s),
                  "VU0 unit0 incoming state e=floor+1 (row self-consistent); chain init link must reject"))

    # (3) epilogue reads a non-final unit's state (VU0 epilogue = VU0 unit0's outgoing state)
    s_wrong = tuple(per_vu_rows[0][0][5 + 2 * STEP + i] for i in range(4))  # unit0 state_out
    y32_wrong = Pack.check(IntCtx(), p, s_wrong, [(0, 0)] * G)
    he = Epilogue.gen(p, y32_wrong)
    y16_wrong = Epilogue.check(IntCtx(), p, y32_wrong, he)
    erow_wrong = epilogue_witness_v2(ectx, epaths, s_wrong, [(0, 0)] * G, y16_wrong, he)
    assert verify_row(ectx, erow_wrong, m) is None, "epilogue-from-wrong-unit must be row-valid"
    e_break = list(erows)
    e_break[0] = erow_wrong
    cases.append(("epilogue_wrong_unit", list(all_rows), e_break, [y16_wrong, y16s[1]],
                  "VU0 epilogue reads unit0's state, self-consistent, public word=cast(that state); epi link must reject"))

    # (4) flip ovf into the epilogue on the final unit only (epi.ovf[0]=1) with a consistent (infinity) word
    e_ovf = list(erows)
    ebad = list(erows[0])
    ebad[5] = 1  # ovf[0]
    st = tuple(ebad[1:5])
    flags = [(ebad[5 + 2 * g], ebad[6 + 2 * g]) for g in range(G)]
    y32b = Pack.check(IntCtx(), p, st, flags)
    heb = Epilogue.gen(p, y32b)
    y16b = Epilogue.check(IntCtx(), p, y32b, heb)
    ebad = epilogue_witness_v2(ectx, epaths, st, flags, y16b, heb)
    e_ovf[0] = ebad
    cases.append(("epilogue_ovf_flip", list(all_rows), e_ovf, [y16b, y16s[1]],
                  "VU0 epilogue ovf[0]=1 (infinity word) while the unit's ovf link value is 0; ovf link must reject"))

    # (5) two VUs swapped epilogues (cross-link): VU0 epilogue is VU1's, VU1 epilogue is VU0's; public words swapped too
    cases.append(("swap_vu_epilogues", list(all_rows), [erows[1], erows[0]], [y16s[1], y16s[0]],
                  "epilogues (and public words) of VU0/VU1 swapped; each epi state != its own unit-95 state -> link rejects"))

    # (6) THE ONE-COLUMN ATTACK: epilogue row with one=0 kills every assertion in that row (GKR realises each
    #     constant as one_wire*coef); lookups read columns directly and still pass with all-zero range hints.  The
    #     cast that binds y16 to the state vanishes, so we free the public word to an arbitrary value while the
    #     chain-linked state (s,e,M,z,ovf,sgn) stays honest.  If ACCEPTED this forges the public output word.
    erow_one0 = list(erows[0])
    erow_one0[0] = 0                       # one := 0
    honest_word = y16s[0]
    erow_one0[9] = honest_word ^ 0xBEEF    # arbitrary forged public word
    for c in range(10, len(erow_one0)):    # all cast hints -> 0 (0 is a valid R16/R15 row)
        erow_one0[c] = 0
    e_one0 = list(erows)
    e_one0[0] = erow_one0
    cases.append(("epilogue_one_zero", list(all_rows), e_one0, [honest_word ^ 0xBEEF, y16s[1]],
                  "VU0 epilogue one:=0 kills all row assertions; y16 forged, cast hints 0; state stays chain-honest"))

    for name, urows, erws, ys, desc in cases:
        d = neg / name
        d.mkdir(parents=True, exist_ok=True)
        _write_rows(urows, d / "units.bin", len(uctx.cols))
        _write_rows(erws, d / "epilogue.bin", len(ectx.cols))
        _write_rows([[y] for y in ys], d / "public.bin", 1)
        shutil.copy(out / "chain.txt", d / "chain.txt")

    # B=1 honest single-VU dir (edge case; must ACCEPT)
    d1 = neg / "b1_honest"
    d1.mkdir(parents=True, exist_ok=True)
    _write_rows(per_vu_rows[0], d1 / "units.bin", len(uctx.cols))
    _write_rows([erows[0]], d1 / "epilogue.bin", len(ectx.cols))
    _write_rows([[y16s[0]]], d1 / "public.bin", 1)
    shutil.copy(out / "chain.txt", d1 / "chain.txt")

    # B=1 forgery: single VU, public word flipped
    d1b = neg / "b1_wrong_word"
    d1b.mkdir(parents=True, exist_ok=True)
    _write_rows(per_vu_rows[0], d1b / "units.bin", len(uctx.cols))
    _write_rows([erows[0]], d1b / "epilogue.bin", len(ectx.cols))
    _write_rows([[y16s[0] ^ 1]], d1b / "public.bin", 1)
    shutil.copy(out / "chain.txt", d1b / "chain.txt")

    manifest = {
        "vus": vus, "steps": steps, "unit_columns": len(uctx.cols), "epilogue_columns": len(ectx.cols),
        "cases": [c[0] for c in cases] + ["b1_honest", "b1_wrong_word"],
        "expected_accept": ["honest", "b1_honest"],
        "expected_reject": [c[0] for c in cases if c[0] != "honest"] + ["b1_wrong_word"],
        "note_one_zero": "epilogue_one_zero is expected_reject IFF the verifier pins the one column; if ACCEPTED it is a critical forgery",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def zero_swap_scan(p: Params = REAL, n: int = 4000, seed: int = 7) -> dict:
    """The b-air-v2 VACUOUS claim, adversarially: for random operand units, run the v2 unit on a +0 incoming state
    (0, floor, 0, 1) and a -0 incoming state (1, floor, 0, 1) and compare the packed output words.  A single input
    with different accepted words refutes the vacuity claim.
    """
    from verity_numerical.checker.v2.step import check_unit_expr

    rng = random.Random(seed)
    k = sum(p.groups)
    pos = zero_state(p)                 # (0, floor, 0, 1) = +0
    negz = (1, p.floor, 0, 1)           # -0: sign 1, magnitude 0

    def word(state, a, b):
        h, _, _ = gen_unit(p, state, a, b)
        ctx = IntCtx()
        st_out, ovfs = check_unit_expr(ctx, p, state, a, b, h, static=False)
        return Pack.check(IntCtx(), p, st_out, ovfs)

    divergences = []
    for _ in range(n):
        a = [_bf16(rng) for _ in range(k)]
        b = [_bf16(rng) for _ in range(k)]
        wp = word(pos, a, b)
        wn = word(negz, a, b)
        if wp != wn:
            divergences.append({"a": a, "b": b, "word_pos0": wp, "word_neg0": wn})
            if len(divergences) >= 5:
                break
    # also the all-zero-operand corner (S = 0 exactly: sign is free but z masks it)
    az = [0x0000] * k
    corner = {"word_pos0": word(pos, az, az), "word_neg0": word(negz, az, az)}
    return {"trials": n, "divergences": divergences, "all_zero_corner": corner,
            "claim_vacuous": not divergences}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m verity_numerical.redteam.v2_forge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gkr-chain")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=424242)
    z = sub.add_parser("zero-swap")
    z.add_argument("--n", type=int, default=4000)
    args = ap.parse_args(argv)
    if args.cmd == "gkr-chain":
        print(json.dumps(build_gkr_chain(Path(args.out), seed=args.seed)))
    else:
        print(json.dumps(zero_swap_scan(n=args.n), indent=1))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
