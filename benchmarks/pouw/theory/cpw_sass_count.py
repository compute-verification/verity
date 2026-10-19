"""Count `cpw_sass.cu`'s loop body (one set: 8 units) in `cuobjdump -sass`, per mode, by census class (`IMAD`: IMAD,
IMUL and their forms; `ALU`: IADD3, IADD, LOP3, LOP, SHF, PRMT, LEA; any other class by its opcode's base), and each
checkpoint route's warp instructions a set beyond mode 0's (the fold alone). The loop body is the largest span from a
label or address to a branch back to it. With `--device FILE` (the binary's output on a device) each mode's measured
SMSP-cycles a warp-set is set beside its count: their ratio is the warp instructions it issued a SMSP-cycle. Prints one
JSON object."""
from __future__ import annotations

import argparse
import collections
import json
import re

IMAD = re.compile(r"^(IMAD|IMUL)$")
ALU = re.compile(r"^(IADD3|IADD|LOP3|LOP|SHF|PRMT|LEA)$")
MODES = {0: "cpw_none", 1: "cpw_redux", 2: "cpw_red64", 3: "cpw_atoms64", 4: "cpw_shfl", 5: "cpw_egress",
         6: "cpw_pair2"}
INS = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*)$")
LABEL = re.compile(r"^\s*\.(L_x_\d+):")
CC_ARCH = {"12.0": "sm_120a"}
UNITS = 8


def cls(op: str) -> str:
    base = op.split(".")[0]
    if IMAD.match(base):
        return "IMAD"
    if ALU.match(base):
        return "ALU"
    return base


def parse(path: str) -> dict[tuple[str, str], tuple[list[str], dict[str, int], list[tuple[int, str]]]]:
    arch, kernel, out = "?", None, {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*arch = (sm_\w+)", line)
        if m:
            arch, kernel = m.group(1), None
            continue
        m = re.search(r"Function : (\S+)", line)
        if m:
            kernel = m.group(1)
            out[(arch, kernel)] = ([], {}, [])
            continue
        if kernel is None:
            continue
        ops, where, branches = out[(arch, kernel)]
        m = LABEL.match(line)
        if m:
            where[m.group(1)] = len(ops)
            continue
        m = INS.match(line)
        if m:
            addr, op, rest = int(m.group(1), 16), m.group(3), m.group(4)
            where[hex(addr)] = len(ops)
            if op.split(".")[0] == "BRA":
                t = re.search(r"\.(L_x_\d+)|\b(0x[0-9a-f]+)\b", rest)
                if t:
                    branches.append((len(ops), t.group(1) or hex(int(t.group(2), 16))))
            ops.append(op)
    return out


def count(path: str) -> dict[str, dict[str, dict]]:
    res: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for (arch, k), (ops, where, branches) in parse(path).items():
        m = re.search(r"cpwILi(\d)E", k)
        if not m:
            continue
        name = MODES[int(m.group(1))]
        spans = [(where[t], i) for i, t in branches if t in where and where[t] < i]
        lo, hi = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(ops) - 1)
        body = ops[lo:hi + 1]
        c = collections.Counter(cls(op) for op in body)
        res[arch][name] = {"body_per_set": len(body), "per_unit": round(len(body) / UNITS, 4),
                           "classes_per_set": dict(c.most_common()), "opcodes": dict(collections.Counter(body).most_common())}
    for arch, ks in res.items():
        base = ks.get("cpw_none")
        if base:
            for v in ks.values():
                v["beyond_fold_per_set"] = v["body_per_set"] - base["body_per_set"]
                v["beyond_fold_per_unit"] = round(v["beyond_fold_per_set"] / UNITS, 4)
    return dict(res)


def device(path: str) -> tuple[str | None, dict[str, list[dict[str, float]]]]:
    arch, rows = None, collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        m = re.match(r"device cc=(\S+)", line)
        if m:
            arch = CC_ARCH.get(m.group(1))
        m = re.match(r"(cpw_\w+) (.*)", line)
        if m:
            row = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", m.group(2))}
            rows[m.group(1)].append(row)
    return arch, dict(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--device", help="the binary's output on one device")
    a = ap.parse_args()
    doc = {"sass": count(a.sass)}
    if a.device:
        arch, rows = device(a.device)
        doc["device"] = {"arch": arch, "rows": rows}
        for k, rs in rows.items():
            c = doc["sass"].get(arch, {}).get(k)
            for r in rs:
                if c and r.get("smsp_cyc_per_warp_set"):
                    r["warp_ipc_smsp"] = round(c["body_per_set"] / r["smsp_cyc_per_warp_set"], 4)
    print(json.dumps(doc, indent=1))
