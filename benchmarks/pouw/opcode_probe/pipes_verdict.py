"""pipes.cu's verdict from a run's rows.jsonl: whether sm_120 runs each lane opcode the coverage table places on alu or
sfu (machine_model/data/cover/rows.toml, CMP-166) within that pipe's ρ, alone and beside the pipe's own partner.

    python3 pipes_verdict.py OUT/rows.jsonl

Rates are warp instructions a SM-clock. alu's ρ is 64 lanes (2), sfu's 16 (0.5), f64's 2 (0.0625); a BAR.RED is charged
16 alu lanes, half a LOP3. The 32-bit conversions are held to sfu, alone and beside 8 MUFU.EX2, and the 64-bit ones to
f64, alone and beside DFMA (sharing f64's pipe they hold 0.0625); one 64-bit pair beside 24 MUFU.EX2 is only reported
(on the XU the two would hold 0.5). The add rows are on fma, issue-bound (4), so their probes only report where the card
runs them. The role-split barrier (mix_bar16_lop3) runs beside a LOP3 stream that isn't waiting on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SLACK = 1.001
ALU, XU, F64, ISSUE = 2.0, 0.5, 0.0625, 4.0
PARTNERS = {"LOP3.LUT": "lop3", "MUFU.EX2": "mufu", "DFMA": "dfma"}
OFF_PIPE = {"FADD"}
ON_ALU = ("plop3", "sgxt", "bmsk")
ON_XU = ("i2f_f2i", "i2f16_f2i16", "frnd", "cvt_s32", "cvt_u16", "cvt_s8", "cvt_frnd", "cvt_i2i", "cvt_i2ip")
ON_F64 = ("f2f64", "i2f64", "s64", "cvt_u64", "cvt_frnd64", "cvt_f64_u32", "cvt_f64_s64")
ON_FMA = ("iadd64", "iabs")


def _split(row: dict) -> tuple[dict[str, float], float]:
    """The probe's own opcodes' rates, and its partner's (0 without one)."""
    own, partner = {}, 0.0
    for key, rate in row["rates"].items():
        pattern = key.split(":", 1)[1]
        if pattern in PARTNERS:
            partner += rate
        elif pattern not in OFF_PIPE:
            own[pattern] = rate
    return own, partner


def verdict(rows: dict[str, dict]) -> tuple[bool, list[str]]:
    failed, said = [], []

    def need(name: str) -> dict | None:
        if name not in rows:
            failed.append(f"no row for {name}")
        return rows.get(name)

    for name, cap, pipe, mix in [(n, ALU, "alu", "lop3") for n in ON_ALU] + [(n, XU, "sfu", "mufu") for n in ON_XU]:
        alone, beside = need(name), need(f"mix_{name}_{mix}")
        if alone is None or beside is None:
            continue
        own, _ = _split(alone)
        x = sum(own.values())
        bx, bp = _split(beside)
        pair = sum(bx.values()) + bp
        what = "+".join(own)
        said.append(f"{what} {x:.4f} alone, {sum(bx.values()):.4f} + {bp:.4f} = {pair:.4f} beside {mix.upper()}")
        if x > cap * SLACK:
            failed.append(f"{what} alone {x:.4f} > {pipe}'s {cap:g}")
        if pair > cap * SLACK:
            failed.append(f"{what} beside {mix.upper()} {pair:.4f} > {pipe}'s {cap:g}")

    alone, beside = need("bar_or"), need("mix_bar_or_lop3")
    if alone is not None and beside is not None:
        x = sum(_split(alone)[0].values())
        bx, bp = _split(beside)
        load = bp + sum(bx.values()) / 2
        said.append(f"BAR.RED.OR {x:.4f} alone; LOP3 + BAR/2 = {bp:.4f} + {sum(bx.values()):.4f}/2 = {load:.4f}")
        if load > ALU * SLACK:
            failed.append(f"LOP3 + BAR.RED.OR/2 {load:.4f} > alu's {ALU:g}")

    for name in ON_F64:
        alone, xu, f64 = need(name), need(f"mix_{name}_mufu24"), need(f"mix_{name}_dfma")
        if alone is None or xu is None or f64 is None:
            continue
        own, _ = _split(alone)
        x, what = sum(own.values()), "+".join(own)
        xx, xm = _split(xu)
        fx, fd = _split(f64)
        pair, beside = sum(xx.values()) + xm, sum(fx.values()) + fd
        where = "on the XU" if pair <= XU * SLACK else "beside the XU"
        said.append(f"{what} {x:.4f} alone, {sum(fx.values()):.4f} + {fd:.4f} = {beside:.4f} beside DFMA, "
                    f"{sum(xx.values()):.4f} + {xm:.4f} = {pair:.4f} beside 24 MUFU ({where})")
        if x > F64 * SLACK:
            failed.append(f"{what} alone {x:.4f} > f64's {F64:g}")
        if beside > F64 * SLACK:
            failed.append(f"{what} beside DFMA {beside:.4f} > f64's {F64:g}")

    alone, beside = need("bar16"), need("mix_bar16_lop3")
    if alone is not None and beside is not None:
        x = sum(_split(alone)[0].values())
        bx, bp = _split(beside)
        load = bp + sum(bx.values()) / 2
        said.append(f"BAR.RED.OR (16 warps) {x:.4f} alone; beside a LOP3 stream LOP3 + BAR/2 = {bp:.4f} + "
                    f"{sum(bx.values()):.4f}/2 = {load:.4f}")
        if load > ALU * SLACK:
            failed.append(f"LOP3 + BAR.RED.OR/2 {load:.4f} > alu's {ALU:g} beside a LOP3 stream")

    for name in ON_FMA:
        alone, beside = need(name), need(f"mix_{name}_lop3")
        if alone is None or beside is None:
            continue
        own, _ = _split(alone)
        bx, bp = _split(beside)
        x, pair = sum(own.values()), sum(bx.values()) + bp
        where = "beside alu" if pair > ALU * SLACK else "on alu"
        said.append(f"{'+'.join(own)} {x:.4f} alone, {pair:.4f} with LOP3 ({where})")
        if x > ISSUE * SLACK or pair > ISSUE * SLACK:
            failed.append(f"{'+'.join(own)} past the issue slots ({x:.4f}, {pair:.4f})")

    for pattern, name in PARTNERS.items():
        if name in rows:
            said.append(f"{pattern} alone {rows[name]['rates'].get('0:' + pattern, 0.0):.4f}")
    throttled = sorted(n for n, r in rows.items() if r.get("throttled") or r.get("tags"))
    if throttled:
        said.append("tagged: " + ", ".join(f"{n} {rows[n].get('tags') or ['throttled']}" for n in throttled))
    return not failed, failed + said


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else Path("rows.jsonl")
    rows = {r["probe"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("binary") == "pipes"}
    held, lines = verdict(rows)
    print(("verdict: hypothesis held: " if held else "verdict: hypothesis failed: ") + "; ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
