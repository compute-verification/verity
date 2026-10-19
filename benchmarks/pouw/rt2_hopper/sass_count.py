"""Count each kernel's SASS loop in `cuobjdump -sass` output, and gate sm90_clock's kernels on it.

A kernel's loop is the instructions from the target of its last backward BRA to that BRA. Counts are by opcode with its
modifiers (`LOP3.LUT`, `IMAD.WIDE.U32`, `REDUX.XOR`), and by class as DS-90's census names them where `loads.class_of`
has one (ALU, IMAD, SHFL and so on), else the opcode's first word.

The gate (`--gate`): no local-memory access (LDL, STL) in any kernel; each `clk<...>` kernel's loop holds the IGMMA its
load issues (4 a trip for the three wgmma loads, none for li_alone); each `li_exact<...>` loop holds li's 64 LOP3 and 64
IMAD a unit (256 and 256 a trip of four units), the checkpoint's REDUX (4 a trip a segment, 16 a set, 8 for CP_SEG4U's
two a segment, with its 3 LOP3; for CP_RED,
CP_RED2 and CP_RED8, 1, 2 and 8 RED a checkpoint and no REDUX) and its per-lane XOR (4 LOP3 a checkpoint; 3 for CP_RED2,
none for CP_RED8); each `roles<X,R>` kernel's MMA loop holds its 4 IGMMA and the
poll's LDS, and its lane loop exactly R of the lane's op (IMAD for X = 1, LOP3 for X = 2); each `lirate<K,W>` loop holds
at least 256 IMAD a trip and 256 LOP3 plus its checkpoint's lane XOR, and exactly its checkpoint's REDUX, RED and STG (4
checkpoints a trip; CK_SEG4's, CK_SEG4U's and CK_SEG4M's 4 in one STG.E.128); each `lirate_mix<NR,W>` loop (8 units a trip) holds 8 − NR REDUX and STG, 2·NR RED, at least 512
IMAD and 512 + 4(8 − NR) + 3·NR LOP3. Exit 1 when a check fails.

Run: `python sass_count.py SASS [--json OUT] [--gate]`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

FUNC = re.compile(r"^\s*Function : (\S+)")
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)(.*?);")
BRA = re.compile(r"BRA\s+(?:`\(\S+\)\s*)?(0x[0-9a-f]+)")


def kernels(text: str) -> dict[str, list[tuple[int, str, str]]]:
    out: dict[str, list[tuple[int, str, str]]] = {}
    cur = None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = INS.search(line)
        if m and cur is not None:
            out[cur].append((int(m.group(1), 16), m.group(3), m.group(4)))
    return out


def bodies(ins: list[tuple[int, str, str]]) -> list[list[tuple[int, str, str]]]:
    """Every backward branch's body, from its target to the branch."""
    out = []
    for addr, op, rest in ins:
        if op.split(".")[0] != "BRA":
            continue
        m = BRA.search(op + rest)
        if m and int(m.group(1), 16) < addr:
            out.append([x for x in ins if int(m.group(1), 16) <= x[0] <= addr])
    return out


def loop(ins: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """The kernel's timed loop: of its backward branches' bodies, the one with the most IGMMA, then the longest (a wgmma
    loop is shorter than the tile fill's)."""
    rank = lambda b: (sum(1 for _, o, _ in b if o.startswith("IGMMA")), len(b))
    return max(bodies(ins), key=rank, default=[])


def demangle_tag(name: str) -> str:
    m = re.match(r"_Z\d+([A-Za-z_]\w*?)I((?:Li\d+E)+)E", name)
    return f"{m.group(1)}<{','.join(re.findall(r'Li(\d+)E', m.group(2)))}>" if m else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--json")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    ks = kernels(open(args.sass).read())
    rows = {}
    fails = []
    for name, ins in ks.items():
        tag = demangle_tag(name)
        body = loop(ins)
        ops = Counter(op for _, op, _ in body)
        base = Counter(op.split(".")[0] for _, op, _ in body)
        local = sum(1 for _, op, _ in ins if op.split(".")[0] in ("LDL", "STL"))
        rows[tag] = {"mangled": name, "instructions": len(ins), "loop": len(body), "local": local,
                     "loop_ops": dict(sorted(ops.items())), "loop_base": dict(sorted(base.items()))}
        if not tag.startswith(("clk<", "li_exact<", "roles<", "lirate<", "lirate_mix<")):
            continue
        if local:
            fails.append(f"{tag}: {local} local-memory accesses (a spill)")
        if tag.startswith("roles<"):
            x, r = (int(v) for v in tag[len("roles<"):-1].split(","))
            lane_op = {1: "IMAD", 2: "LOP3.LUT"}[x]
            got = sum(n for op, n in ops.items() if op.startswith("IGMMA"))
            if got != 4 or base["LDS"] < 1:
                fails.append(f"{tag}: {got} IGMMA and {base['LDS']} LDS in the MMA loop, want 4 and the poll's LDS")
            lane = max((sum(1 for _, o, _ in b if o == lane_op) for b in bodies(ins)
                        if not any(o.startswith("IGMMA") for _, o, _ in b)), default=0)
            rows[tag]["lane_loop_ops"] = lane
            if lane != r:
                fails.append(f"{tag}: {lane} {lane_op} in the lane loop, want {r}")
        if tag.startswith("clk<"):
            want = 0 if tag == "clk<3>" else 4
            got = sum(n for op, n in ops.items() if op.startswith("IGMMA"))
            if got != want:
                fails.append(f"{tag}: {got} IGMMA in the loop, want {want}")
        if tag.startswith("li_exact<"):
            cp = int(tag[len("li_exact<"):-1])
            ncp = 16 if cp == 2 else 4
            # a checkpoint's REDUX, RED and lane-XOR LOP3: CP_SEG, CP_SEG4, CP_SET 1, 0, 4; CP_RED 0, 1, 4;
            # CP_RED2 0, 2, 3; CP_RED8 0, 8, 0; CP_SEG4U 2, 0, 3
            redux, red, xl = {3: (0, 1, 4), 4: (0, 2, 3), 5: (0, 8, 0), 6: (2, 0, 3)}.get(cp, (1, 0, 4))
            redux, red = redux * ncp, red * ncp
            got_red = base["RED"] + base["REDG"]
            if base["REDUX"] != redux or got_red != red:
                fails.append(f"{tag}: {base['REDUX']} REDUX and {got_red} RED a trip, want {redux} and {red}")
            if base["LOP3"] < 256 + xl * ncp:
                fails.append(f"{tag}: {base['LOP3']} LOP3 a trip, want at least {256 + xl * ncp}")
            imad = sum(n for op, n in ops.items() if op == "IMAD")
            if imad < 256:
                fails.append(f"{tag}: {imad} IMAD a trip, want at least 256")
        if tag.startswith("lirate<"):
            k = int(tag[len("lirate<"):-1].split(",")[0])
            # CK_NONE, CK_SEG, CK_RED, CK_RED2, CK_RED8, CK_SEG4, CK_SEG4U, CK_SEG4M: a trip's lane-XOR LOP3, REDUX,
            # RED and STG
            xl, redux, red, stg = [(0, 0, 0, 0), (16, 4, 0, 4), (16, 0, 4, 0), (12, 0, 8, 0), (0, 0, 32, 0),
                                   (16, 4, 0, 1), (12, 8, 0, 1), (15, 5, 0, 1)][k]
            got = (base["REDUX"], base["RED"] + base["REDG"], base["STG"])
            if got != (redux, red, stg):
                fails.append(f"{tag}: {got} REDUX, RED, STG a trip, want {(redux, red, stg)}")
            wide = sum(n for op, n in ops.items() if op.startswith("STG.E.128"))
            if k >= 5 and wide != 1:
                fails.append(f"{tag}: {wide} STG.E.128 a trip, want 1")
            if base["LOP3"] < 256 + xl:
                fails.append(f"{tag}: {base['LOP3']} LOP3 a trip, want at least {256 + xl}")
            imad = sum(n for op, n in ops.items() if op == "IMAD")
            if imad < 256:
                fails.append(f"{tag}: {imad} IMAD a trip, want at least 256")
        if tag.startswith("lirate_mix<"):
            nr = int(tag[len("lirate_mix<"):-1].split(",")[0])
            want = (8 - nr, 2 * nr, 8 - nr)
            got = (base["REDUX"], base["RED"] + base["REDG"], base["STG"])
            if got != want:
                fails.append(f"{tag}: {got} REDUX, RED, STG a trip, want {want}")
            if base["LOP3"] < 512 + 4 * (8 - nr) + 3 * nr:
                fails.append(f"{tag}: {base['LOP3']} LOP3 a trip, want at least {512 + 4 * (8 - nr) + 3 * nr}")
            imad = sum(n for op, n in ops.items() if op == "IMAD")
            if imad < 512:
                fails.append(f"{tag}: {imad} IMAD a trip, want at least 512")
    for tag, r in rows.items():
        if tag.startswith(("clk<", "li_exact<", "roles<", "lirate<", "lirate_mix<")):
            extra = f", lane loop {r['lane_loop_ops']}" if "lane_loop_ops" in r else ""
            print(f"{tag}: {r['instructions']} instructions, loop {r['loop']}: {r['loop_ops']}{extra}")
    for f in fails:
        print(f"REFUSED {f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"kernels": rows, "refused": fails}, f, indent=1)
    if args.gate:
        print("GATE", "REFUSED" if fails else "PASS")
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
