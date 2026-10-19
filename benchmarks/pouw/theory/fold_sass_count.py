"""Count each fold kernel's loop body in `cuobjdump -sass` of `fold_sass.cu`, per arch, by census class, with
facet_break/loads.py's vector rules (`IMAD`: IMAD, IMUL and their forms, IMAD.HI and IMAD.MOV among them; `ALU`: IADD3,
IADD, LOP3, LOP, SHF, PRMT, LEA), a word absorbed. The loop body is the largest span from a label to a branch back to
it; a pass absorbs 32 words (fx1) or 64 (li: 32 pairs). With `--device FILE` (the binary's output on a device), each
kernel's measured SMSP-cycles a warp-word is set beside its count: their ratio is the warp instructions it issued a
SMSP-cycle. Prints one JSON object."""
from __future__ import annotations

import argparse
import collections
import json
import re

IMAD = re.compile(r"^(IMAD|IMUL)$")
ALU = re.compile(r"^(IADD3|IADD|LOP3|LOP|SHF|PRMT|LEA)$")
WORDS = {"fx1_chain": 32, "li_chain": 64, "li_heavy_chain": 64, "li_heavy_imad_chain": 64,
         "li_heavy_imad25_chain": 64}
INS = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*)$")
LABEL = re.compile(r"^\s*\.(L_x_\d+):")
CC_ARCH = {"9.0": "sm_90a", "10.0": "sm_100a", "12.0": "sm_120a"}


def cls(op: str) -> str:
    base = op.split(".")[0]
    if IMAD.match(base):
        return "IMAD"
    if ALU.match(base):
        return "ALU"
    return base


def parse(path: str) -> dict[tuple[str, str], tuple[list[str], dict[str, int], list[tuple[int, str]]]]:
    """(arch, kernel) -> (opcodes in order, label or address -> index of its instruction, (index, target) of each
    branch). cuobjdump names a branch's target by its address (`BRA 0x2ec0`), nvdisasm by a label (`.L_x_3`)."""
    arch, kernel, out = "?", None, {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*arch = (sm_\w+)", line)
        if m:
            arch = m.group(1)
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
        w = WORDS.get(k)
        if w is None:
            continue
        spans = [(where[t], i) for i, t in branches if t in where and where[t] < i]
        lo, hi = max(spans, key=lambda s: s[1] - s[0]) if spans else (0, len(ops) - 1)
        body = ops[lo:hi + 1]
        c = collections.Counter(cls(op) for op in body)
        res[arch][k] = {"words": w, "body": len(body), "per_word": {x: round(n / w, 4) for x, n in c.most_common()},
                        "per_word_total": round(len(body) / w, 4), "lop3": sum(op.startswith("LOP3") for op in body),
                        "opcodes": dict(collections.Counter(body).most_common())}
    return dict(res)


def device(path: str) -> tuple[str | None, dict[str, dict[str, float]]]:
    arch, rows = None, {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"device cc=(\S+)", line)
        if m:
            arch = CC_ARCH.get(m.group(1))
        m = re.match(r"(\w+_chain) (.*)", line)
        if m:
            rows[m.group(1)] = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", m.group(2))}
    return arch, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--device", help="the binary's output on one device")
    a = ap.parse_args()
    doc = {"sass": count(a.sass)}
    if a.device:
        arch, rows = device(a.device)
        doc["device"] = {"arch": arch, "rows": rows}
        for k, r in rows.items():
            c = doc["sass"].get(arch, {}).get(k)
            if c and r.get("tp_smsp_cyc_per_warp_word"):
                r["warp_ipc_smsp"] = round(c["per_word_total"] / r["tp_smsp_cyc_per_warp_word"], 4)
    print(json.dumps(doc, indent=1))
