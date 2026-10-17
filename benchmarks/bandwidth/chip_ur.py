"""Uniform registers past the ones ptxas allocates: rewrite `urt.cubin` (bwchip's `urt` kernel) to load its two
parameters into URa and URb and read them back from there, so `bwchip --ur` can see whether URa holds a value apart
from URb.

`urt(out, v1, pad, v2)` assembles to `LDCU URx, c[0x0][0x388]` (v1), `LDCU URy, c[0x0][0x398]` (v2), one ALU
instruction reading URx and one reading URy; each thread writes `tid·v1` and `tid ^ v2`.  An LDCU's destination is
byte 2 of its low instruction word and an ALU instruction's uniform source byte 4.  The rewrite finds the four words
in the cubin's SASS, sets those bytes, and checks the rewritten cubin's SASS names URa and URb where URx and URy were;
it refuses (exit 3) any cubin whose SASS isn't that shape.

    python3 chip_ur.py CUOBJDUMP urt.cubin A B OUT.cubin
"""

import re
import subprocess
import sys

LINE = re.compile(r"/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\* 0x([0-9a-f]{16}) \*/")


def sass(cuobjdump: str, path: str) -> list[tuple[str, int]]:
    out = subprocess.run([cuobjdump, "-sass", path], check=True, capture_output=True, text=True).stdout
    return [(m.group(1), int(m.group(2), 16)) for m in LINE.finditer(out)]


def plan(code: list[tuple[str, int]]) -> list[tuple[int, int, int]]:
    """(word, byte index, register) for v1's and v2's loads and reads, in the order (v1 load, v1 read, v2 ...)."""
    out = []
    for offset in ("0x388", "0x398"):
        loads = [(text, word) for text, word in code if re.fullmatch(rf"LDCU UR(\d+), c\[0x0\]\[{offset}\]", text)]
        if len(loads) != 1:
            raise SystemExit(f"REFUSED: {len(loads)} loads of c[0x0][{offset}]")
        text, word = loads[0]
        reg = int(re.search(r"UR(\d+)", text).group(1))
        if (word >> 16) & 0xFF != reg:
            raise SystemExit(f"REFUSED: {text}: byte 2 of {word:#018x} is not UR{reg}")
        reads = [(t, w) for t, w in code if not t.startswith("LDCU") and re.search(rf"\bUR{reg}\b", t)]
        if len(reads) != 1 or (reads[0][1] >> 32) & 0xFF != reg:
            raise SystemExit(f"REFUSED: UR{reg}'s reads are {reads}")
        out += [(word, 2, reg), (reads[0][1], 4, reg)]
    return out


def main() -> None:
    cuobjdump, src, a, b, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    if not (0 <= a < 255 and 0 <= b < 255 and a != b):
        raise SystemExit("REFUSED: a and b are distinct uniform registers, 0..254")
    code = sass(cuobjdump, src)
    steps = plan(code)
    used = {int(r) for text, _ in code for r in re.findall(r"\bUR(\d+)\b", text)} - {steps[0][2], steps[2][2]}
    others = used | {r + 1 for r in used}  # a named register may be the low half of a 64-bit pair
    if {a, b} & others:
        raise SystemExit(f"REFUSED: UR{a} or UR{b} is one the kernel uses for something else ({sorted(others)})")
    blob = bytearray(open(src, "rb").read())
    for (word, at, _), new in zip(steps, (a, a, b, b)):
        raw = word.to_bytes(8, "little")
        if blob.count(raw) != 1:
            raise SystemExit(f"REFUSED: {word:#018x} is not unique in the cubin")
        i = blob.index(raw)
        blob[i + at] = new
    open(dst, "wb").write(bytes(blob))
    want = plan(sass(cuobjdump, dst))
    if [r for _, _, r in want] != [a, a, b, b]:
        raise SystemExit(f"REFUSED: the rewritten SASS names {[r for _, _, r in want]}, not UR{a}, UR{b}")
    print(f"urt: v1 in UR{a}, v2 in UR{b}")


if __name__ == "__main__":
    main()
