"""SHA-512's priced work a compression on the H200 (sm_90a), from the committer's own SASS: stdlib only.

    python3 sha512_sass.py PROBE_SASS [ROWS_SASS] OUT_JSON

PROBE_SASS is `cuobjdump -sass` of sha512_probe.cu, whose loop trip is one block's loads and one `sha::compress`. The
count a compression is the loop's instructions a trip, its loads (LDG) left out; with SHA_ROLL=1 the rounds' inner loop
runs 4 trips a compression, so its body counts 4 times. ROWS_SASS, if given, is native_rows.cu's, summarized by kernel.

Each instruction is priced in W1 units (one dense int8 MAC = 1) at two readings of CORE-H32's H200 table
(r20261029-021534-a5a3): the floor, 32 a register written (`h200Measured`, what the rank bound charges); and each
instruction alone (LOP3, SHF, IADD3, LEA, PRMT, SEL, MOV 64.13; IMAD 64.0; IMAD.WIDE 80.3 for its two registers; the
uniform datapath 2,048 a word over a warp's 32 lanes). Against HASH-38's bar (0.01803 compressions an SM-clock, 4,096
MACs an SM-clock) a compression takes 227,180 MAC-times.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
LABEL = re.compile(r"^\s*(\.L_x_\d+):")
FUNC = re.compile(r"Function : (\S+)")
TARGET = re.compile(r"BRA\s+(?:\S+\s+)?(?:`\((\.L_x_\d+)\)|0x([0-9a-f]+))")

FLOOR = 32.0
ALONE = {"IMAD": 64.0, "IMAD.WIDE": 80.3, "UNIFORM": 2048.0 / 32}
ALONE_DEFAULT = 64.13
BAR_MAC_TIMES = 4096 / 0.01803
NO_WRITE = ("BRA", "EXIT", "BSYNC", "BSSY", "NOP", "STG", "STS", "ST", "RET", "CALL", "BAR", "WARPSYNC", "DEPBAR")


def parse(path: Path) -> dict[str, list[dict]]:
    """Every function's instructions in order: address, opcode with modifiers, predicate, operands, and the label (if any)
    each branch targets."""
    funcs: dict[str, list[dict]] = {}
    cur, pending = None, []
    labels: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        if m := FUNC.search(line):
            cur = m.group(1)
            funcs[cur], labels[cur], pending = [], {}, []
            continue
        if cur is None:
            continue
        if m := LABEL.match(line):
            pending.append(m.group(1))
            continue
        if m := INSN.search(line):
            addr, text = int(m.group(1), 16), m.group(2).strip()
            for lab in pending:
                labels[cur][lab] = addr
            pending = []
            pred = ""
            if text.startswith("@"):
                pred, text = text.split(None, 1)
            parts = text.split(None, 1)
            op, ops = parts[0], (parts[1] if len(parts) > 1 else "")
            tgt = None
            if t := TARGET.search(text):
                tgt = t.group(1) or int(t.group(2), 16)
            funcs[cur].append({"addr": addr, "op": op, "pred": pred, "ops": ops, "target": tgt})
    for f, ins in funcs.items():
        for i in ins:
            if isinstance(i["target"], str):
                i["target"] = labels[f].get(i["target"])
    return funcs


def writes(i: dict) -> tuple[int, str]:
    """Registers an instruction writes and its price class."""
    op = i["op"]
    base = op.split(".")[0]
    if base in NO_WRITE or op.startswith("ISETP") or op.startswith("PLOP3") or op.startswith("UISETP"):
        return 0, "none"
    dst = i["ops"].split(",")[0].strip()
    if base.startswith("U") and base not in ("UNUSED",):
        return (1 if dst.startswith("UR") and dst != "URZ" else 0), "UNIFORM"
    if not re.match(r"R\d+", dst):
        return 0, "none"
    n = 4 if ".128" in op else 2 if (".64" in op or ".WIDE" in op) else 1
    cls = "IMAD.WIDE" if op.startswith("IMAD.WIDE") else "IMAD" if base == "IMAD" else base
    return n, cls


def price(ins: list[dict], mult: Counter | None = None) -> dict:
    """Counts and W1 work of a run of instructions, each weighted by mult[addr] (default 1)."""
    ops, regs, w_floor, w_alone, n = Counter(), 0, 0.0, 0.0, 0.0
    for i in ins:
        k = (mult or {}).get(i["addr"], 1)
        if i["op"].startswith("LDG"):
            continue
        r, cls = writes(i)
        ops[i["op"].split(".")[0]] += k
        n += k
        regs += k * r
        if r:
            w_floor += k * FLOOR * r
            w_alone += k * (ALONE["IMAD.WIDE"] if cls == "IMAD.WIDE" else ALONE.get(cls, ALONE_DEFAULT) * r)
    return {"instructions": n, "registers_written": regs, "w1_floor": w_floor, "w1_alone": w_alone,
            "by_opcode": dict(ops.most_common())}


def loops(ins: list[dict]) -> list[tuple[int, int]]:
    """(first, last) index pairs of backward branches' bodies, outermost first."""
    idx = {i["addr"]: n for n, i in enumerate(ins)}
    out = [(idx[i["target"]], n) for n, i in enumerate(ins)
           if i["op"].startswith("BRA") and isinstance(i["target"], int) and i["target"] <= i["addr"] and i["target"] in idx]
    return sorted(out, key=lambda x: x[0] - x[1])


def per_compression(ins: list[dict], inner_trips: int) -> dict:
    """The outer loop's trip (a block's loads and one compression); a loop nested inside it runs inner_trips times."""
    ls = loops(ins)
    if not ls:
        raise SystemExit("no loop in the probe: expected one trip a block")
    a, b = ls[0]
    body = ins[a:b + 1]
    mult: Counter = Counter()
    inner = [(c, d) for c, d in ls[1:] if a <= c and d <= b]
    for c, d in inner:
        for i in ins[c:d + 1]:
            mult[i["addr"]] = inner_trips
    r = price(body, mult)
    r["loads_left_out"] = sum(1 for i in body if i["op"].startswith("LDG"))
    r["inner_loops"] = len(inner)
    return r


def main() -> None:
    probe, out_path = Path(sys.argv[1]), Path(sys.argv[-1])
    funcs = parse(probe)
    (name, ins), = [(k, v) for k, v in funcs.items() if "probe" in k]
    r = per_compression(ins, 4)
    r.update({"function": name, "rolled": r["inner_loops"] > 0,
              "bar_mac_times": BAR_MAC_TIMES,
              "issue_bound_mac_times": r["instructions"] * 32,
              "w1_floor_over_bar": r["w1_floor"] / BAR_MAC_TIMES, "w1_alone_over_bar": r["w1_alone"] / BAR_MAC_TIMES})
    out = {"compression": r}
    if len(sys.argv) > 3:
        out["native_rows"] = {k: price(v) for k, v in parse(Path(sys.argv[2])).items()}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"{name}: {r['instructions']:.0f} instructions a compression ({r['registers_written']:.0f} registers written; "
          f"loads left out {r['loads_left_out']}, rolled {r['rolled']}); W1 at the floor {r['w1_floor']:.0f} "
          f"({r['w1_floor_over_bar']:.3f} of the bar's {BAR_MAC_TIMES:.0f} MAC-times), alone {r['w1_alone']:.0f} "
          f"({r['w1_alone_over_bar']:.3f}); top opcodes {list(r['by_opcode'].items())[:8]}")


if __name__ == "__main__":
    main()
