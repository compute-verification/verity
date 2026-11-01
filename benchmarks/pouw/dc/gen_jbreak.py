"""Writes probes/sm90_jbreak.inc, the probe list of probes/sm90_jbreak.cu (the stress of the H200 rows).

  python3 gen_jbreak.py            # rewrites probes/sm90_jbreak.inc
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
WG = {"s8_n256": "IGMMA.64x256x32.S8.S8*", "s8_n128": "IGMMA.64x128x32.S8.S8*", "s8_n64": "IGMMA.64x64x32.S8.S8*",
      "s8_n256_rs": "IGMMA.64x256x32.S8.S8*", "f16_n256": "HGMMA.64x256x16.F32*", "tf32_n256": "HGMMA.64x256x8.F32.TF32*"}
X = {"fmuli": "J_FMULI", "faddi": "J_FADDI", "axi": "J_AXI", "imli": "J_IMLI", "ffma": "J_FFMA", "li": "J_LI",
     "mix": "J_MIX", "lif": "J_LIF"}


def lane_ops(lane: str, r: int, c: int) -> list[str]:
    """LaneC::op's opcode for ops 0 .. r - 1 on c chains (sm90_jbreak.cu)."""
    out = []
    for q in range(r):
        k = q // c
        out.append({"fmuli": "FMUL", "faddi": "FADD", "ffma": "FFMA",
                    "axi": "LOP3.LUT" if k & 1 else "ADD",
                    "imli": "LOP3.LUT" if k & 1 else "IMAD",
                    "li": "IMAD" if q & 1 else "LOP3.LUT",
                    "mix": "FFMA" if q & 1 else "I*ADD*",
                    "lif": "FFMA" if q % 8 == 7 else ("LOP3.LUT" if (q % 8 + q // 8) % 2 == 0 else "IMAD")}[lane])
    return out


def lane_pat(lane: str, r: int, c: int = 8) -> str:
    """The lane's SASS for r ops. axi's add of an immediate lowers to IADD3, VIADD or IMAD.IADD, so only its xors are
    declared (jbreak.py counts every instruction of the loop from the SASS)."""
    ops = lane_ops(lane, r, c)
    return ",".join(f"{op}={ops.count(op)}" for op in dict.fromkeys(ops) if op != "ADD")


def probes() -> list[str]:
    out = []

    def jb(form, ns, u, p, lane, c, r, w):
        name = f"{form.replace('_', '')}_s{ns}u{u}p{p}_{lane}c{c}_r{r}_w{w}"
        pat = f"{WG[form]}={4 * ns * u},{lane_pat(lane, r * ns * u, c)}"
        out.append(f'JB({name}, {form}, {ns}, {u}, {p}, {X[lane]}, {c}, {r}, {w}, "{pat}")')

    def jbu(lane, c, r, mw, lw):
        pat = f"{WG['s8_n256']}=4;{lane_pat(lane, r, c)}"
        out.append(f'JBU({lane}c{c}_r{r}_m{mw}l{lw}, {X[lane]}, {c}, {r}, {mw}, {lw}, "{pat}", "{4 * mw};{4 * lw}")')

    def jba(lane, c, w):
        out.append(f'JBA({lane}c{c}_w{w}, {X[lane]}, {c}, {w}, "{lane_pat(lane, 256, c)}")')

    def jbk(form, lane, r):
        out.append(f'JBK({form.replace("_", "")}_{lane}_r{r}, {form}, {X[lane]}, {r}, "{WG[form]}=4,{lane_pat(lane, r)}")')

    def jbku(form, lane, r):
        out.append(f'JBKU({form.replace("_", "")}_{lane}_r{r}, {form}, {X[lane]}, {r}, "{WG[form]}=4;{lane_pat(lane, r)}")')

    # facet_break's break beside wgmma: one-read FP lanes on 2-3 chains, 2 warps a SMSP, across the facet
    for lane in ("fmuli", "faddi"):
        for c in (2, 3):
            for r in (128, 192, 256, 320, 384, 512):
                jb("s8_n256", 1, 1, 0, lane, c, r, 2)
    for lane in ("axi", "imli", "ffma", "li", "mix"):
        for r in (192, 256, 384):
            jb("s8_n256", 1, 1, 0, lane, 3, r, 2)
    for lane in ("fmuli", "imli"):
        for c in (4, 8):
            for r in (256, 384):
                jb("s8_n256", 1, 1, 0, lane, c, r, 2)
    for w in (1, 3):
        for r in (256, 384):
            jb("s8_n256", 1, 1, 0, "fmuli", 3, r, w)
        jb("s8_n256", 1, 1, 0, "imli", 3, 256, w)
    for lane in ("fmuli", "imli"):
        for r in (256, 384):
            jb("s8_n256", 1, 2, 0, lane, 3, r, 2)
            jb("s8_n256", 1, 1, 1, lane, 3, r, 2)
    # accumulator sets: smaller N, several groups in flight (the tensor counted in slots)
    for ns in (1, 2, 3):
        for r in (128, 192):
            jb("s8_n128", ns, 1, 0, "fmuli", 3, r, 2)
    jb("s8_n128", 2, 1, 0, "imli", 3, 128, 2)
    jb("s8_n128", 2, 1, 0, "fmuli", 3, 128, 3)
    for ns in (2, 4, 6):
        for r in (64, 96):
            jb("s8_n64", ns, 1, 0, "fmuli", 3, r, 2)
    jb("s8_n64", 4, 1, 0, "imli", 3, 64, 2)
    jb("s8_n64", 2, 1, 0, "fmuli", 3, 64, 4)
    # exact points: lane warps until the wgmma warpgroups finish, 256-512 ops a trip
    for lane, cs in (("fmuli", (2, 3, 8)), ("faddi", (3,)), ("axi", (3, 8)), ("imli", (2, 3, 8)), ("ffma", (3,)),
                     ("li", (8,))):
        for c in cs:
            jbu(lane, c, 256, 2, 1)
    for lane, c in (("fmuli", 3), ("imli", 3), ("li", 8)):
        jbu(lane, c, 256, 1, 2)
    for lane in ("fmuli", "imli"):
        jbu(lane, 3, 512, 2, 1)
    # the lanes alone
    for lane in ("fmuli", "faddi", "axi", "imli"):
        for w in (1, 2):
            jba(lane, 3, w)
    jba("fmuli", 2, 2)
    jba("imli", 2, 2)
    jba("li", 8, 1)
    jba("li", 8, 2)
    # the menu clause: other wgmma kinds in honest's shape, at L = 32's and L = 64's depth ratio, and exact
    for form, rs in (("s8_n256", (512, 256)), ("s8_n256_rs", (512, 256)), ("f16_n256", (256, 128)),
                     ("tf32_n256", (128, 64))):
        for r in rs:
            jbk(form, "mix", r)
    for form, r in (("s8_n256", 512), ("s8_n256_rs", 512), ("f16_n256", 256), ("tf32_n256", 128)):
        jbku(form, "lif", r)
    return out


def main() -> None:
    lines = probes()
    (HERE / "probes" / "sm90_jbreak.inc").write_text(
        "// Generated by gen_jbreak.py; edit that, not this. One line a probe (sm90_jbreak.cu's macros).\n"
        + "\n".join(lines) + "\n")
    print(len(lines), "probes")


if __name__ == "__main__":
    main()
