"""Compare lane_capture's H200 outputs with the semantics derive_minimal_ssn.py reads (its `prmt`, `shf` and `sgxt`) and
the documented function of every other form, word for word.

  python3 check.py CAPTURE.bin [--sass FILE] [--json OUT]   # exit 0 iff every word of every form matches

`--sass` takes cuobjdump -sass of the binary and names, for each form, the opcodes its kernel holds beyond those every
kernel holds (the loads, the store and the index arithmetic), so the verdict says which SASS form was captured.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path

M32, M64 = (1 << 32) - 1, (1 << 64) - 1
_spec = importlib.util.spec_from_file_location(
    "derive_minimal_ssn", Path(__file__).resolve().parents[2] / "theory" / "derive_minimal_ssn.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def s32(x: int) -> int:
    return x - (1 << 32) if x >> 31 else x


def lop3(lut: int):
    def f(a, b, c):
        out = 0
        for i in range(8):
            if lut >> i & 1:
                out |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
        return out & M32
    return f


def byte(x: int, k: int, signed: bool) -> int:
    v = x >> 8 * k & 0xFF
    return v - 256 if signed and v >> 7 else v


def half(x: int, k: int, signed: bool) -> int:
    v = x >> 16 * k & 0xFFFF
    return v - 65536 if signed and v >> 15 else v


def dp4a(sa: bool, sb: bool):
    return lambda a, b, c: (sum(byte(a, k, sa) * byte(b, k, sb) for k in range(4)) + (s32(c) if sa or sb else c)) & M32


def dp2a(hi: bool, signed: bool):
    return lambda a, b, c: (sum(half(a, k, signed) * byte(b, k + 2 * hi, signed) for k in range(2))
                            + (s32(c) if signed else c)) & M32


def bmsk(pos: int, w: int) -> int:
    return ((1 << w) - 1) << pos & M32


ONE = {
    "lop3_96": lop3(0x96), "lop3_e8": lop3(0xE8), "lop3_ca": lop3(0xCA), "lop3_1e": lop3(0x1E),
    "mad_lo": lambda a, b, c: (a * b + c) & M32,
    "mad_hi": lambda a, b, c: ((a * b >> 32) + c) & M32,
    "mad_hi_s": lambda a, b, c: ((s32(a) * s32(b) >> 32) + c) & M32,
    "iadd3": lambda a, b, c: (a + b + c) & M32,
    "lea3": lambda a, b, c: ((a << 3) + b) & M32,
    "lea_hi5": lambda a, b, c: ((((b << 32 | a) << 5 & M64) >> 32) + c) & M32,
    "prmt": lambda a, b, c: D.prmt(a, b, c),
    "shf_l_wrap": lambda a, b, c: D.shf(a, b, c & 31, True, False, True),
    "shf_l_clamp": lambda a, b, c: D.shf(a, b, min(c, 32), True, False, True),
    "shf_r_wrap": lambda a, b, c: D.shf(a, b, c & 31, False, False, False),
    "shf_r_clamp": lambda a, b, c: D.shf(a, b, min(c, 32), False, False, False),
    "szext_cs": lambda a, b, c: D.sgxt(a, min(c, 32), True),
    "szext_cu": lambda a, b, c: D.sgxt(a, min(c, 32), False),
    "szext_ws": lambda a, b, c: D.sgxt(a, c & 31, True),
    "szext_wu": lambda a, b, c: D.sgxt(a, c & 31, False),
    "bmsk_c": lambda a, b, c: bmsk(min(b, 32), min(c, 32)),
    "bmsk_w": lambda a, b, c: bmsk(b & 31, c & 31),
    "dp4a_ss": dp4a(True, True), "dp4a_uu": dp4a(False, False), "dp4a_su": dp4a(True, False),
    "dp2a_lo": dp2a(False, True), "dp2a_hi": dp2a(True, False),
    "vabsdiff4": lambda a, b, c: sum(abs(byte(a, k, False) - byte(b, k, False)) << 8 * k for k in range(4)),
    "min_s": lambda a, b, c: min(s32(a), s32(b)) & M32,
    "max_u": lambda a, b, c: max(a, b),
    "max3_s": lambda a, b, c: max(s32(a), s32(b), s32(c)) & M32,
    "addmax_s": lambda a, b, c: max(s32((a + b) & M32), s32(c)) & M32,
    "abs_s": lambda a, b, c: abs(s32(a)) & M32,
    "popc": lambda a, b, c: bin(a).count("1"),
    "bfind_u": lambda a, b, c: a.bit_length() - 1 & M32,
    "brev": lambda a, b, c: int(f"{a:032b}"[::-1], 2),
    "sel": lambda a, b, c: a if c & 1 else b,
}
TWO = {
    "mad_wide": lambda a, b, c: divmod((a * b + (c << 32 | (a ^ b))) & M64, 1 << 32)[::-1],
    "shl64": lambda a, b, c: (D.shf(a, b, c & 63, True, False, False), D.shf(a, b, c & 63, True, False, True)),
    "shr64": lambda a, b, c: (D.shf(a, b, c & 63, False, False, False), D.shf(a, b, c & 63, False, False, True)),
    "sar64": lambda a, b, c: (D.shf(a, b, c & 63, False, True, False), D.shf(a, b, c & 63, False, True, True)),
}


def read(path: Path):
    raw = path.read_bytes()
    magic, n, forms = struct.unpack_from("<3I", raw)
    if magic != 0x31434C56:
        raise SystemExit(f"check: {path} is not a lane_capture file")
    names, off = [], 12
    for _ in range(forms):
        end = raw.index(b"\0", off)
        names.append(raw[off:end].decode())
        off = end + 1
    abc = [struct.unpack_from(f"<{n}I", raw, off + 4 * n * k) for k in range(3)]
    off += 12 * n
    outs = {}
    for k, name in enumerate(names):
        w = struct.unpack_from(f"<{2 * n}I", raw, off + 8 * n * k)
        outs[name] = (w[0::2], w[1::2])
    return n, abc, outs


def sass_forms(path: Path) -> dict[str, list[str]]:
    ops: dict[str, set[str]] = {}
    cur = None
    for line in path.read_text().splitlines():
        m = re.search(r"Function : lc_(\w+)", line)
        if m:
            cur = m.group(1)
            ops[cur] = set()
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m and cur:
            ops[cur].add(m.group(1))
    common = set.intersection(*ops.values()) if ops else set()
    return {k: sorted(v - common) for k, v in ops.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--sass", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    n, (A, B, C), outs = read(a.capture)
    sass = sass_forms(a.sass) if a.sass else {}
    rows, bad = [], 0
    for name, (x, y) in outs.items():
        if name in ONE:
            want = [(ONE[name](A[i], B[i], C[i]), 0) for i in range(n)]
        elif name in TWO:
            want = [TWO[name](A[i], B[i], C[i]) for i in range(n)]
        else:
            raise SystemExit(f"check: no semantics for form {name}")
        miss = [i for i in range(n) if (x[i], y[i]) != want[i]]
        bad += bool(miss)
        rows.append({"form": name, "sass": sass.get(name), "inputs": n, "mismatches": len(miss),
                     "first": [{"a": A[i], "b": B[i], "c": C[i], "got": [x[i], y[i]], "want": list(want[i])}
                               for i in miss[:4]]})
        print(f"{'ok ' if not miss else 'BAD'} {name:12} {len(miss):6} of {n}  {' '.join(sass.get(name) or [])}")
    missing = (set(ONE) | set(TWO)) - set(outs)
    if missing:
        print(f"check: forms with semantics but no capture: {sorted(missing)}", file=sys.stderr)
    verdict = {"inputs": n, "forms": len(rows), "forms_matching": len(rows) - bad, "rows": rows}
    if a.json:
        a.json.write_text(json.dumps(verdict, indent=1))
    print(f"check: {len(rows) - bad} of {len(rows)} forms match on every one of {n} inputs")
    return 0 if bad == 0 and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
