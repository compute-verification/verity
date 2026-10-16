#!/usr/bin/env python3
"""Combine the issue-ceiling run's JSONL with the SASS cuobjdump wrote: confirm the opcode that issued in each loop,
derive the per-call SASS behind every 'warp instr/SM/clk', and render the issue table and CE7 summary.

$OUT/issue_ceiling.jsonl  op_alone / mma_alone / mix rows (warp instructions a SM a clock, MMA MAC rate)
$OUT/sass/issue_ceiling.sass   every kernel's SASS; this reads each loop and counts opcodes
$OUT/ce7.jsonl            CE7 vs cuBLASLt int8, if the ce7 section ran
-> $OUT/issue.md, $OUT/issue.json
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ADDR = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z@][A-Z0-9_.]*)")
BRA = re.compile(r"\bBRA\b.*?(0x[0-9a-fA-F]+)")
FUNC = re.compile(r"Function\s*:\s*(\S+)")


def demangle(name):
    try:
        out = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or name
    except Exception:
        return name


def parse_sass(path):
    """Return {demangled_function: {opcode: count_in_hottest_loop}} plus the full-body histogram as a fallback."""
    if not path.exists():
        return {}
    funcs = {}
    cur = None
    lines_by_func = {}
    for line in path.read_text(errors="replace").splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            lines_by_func[cur] = []
            continue
        if cur is not None:
            lines_by_func[cur].append(line)
    for raw_name, lines in lines_by_func.items():
        instrs = []  # (addr_int, opcode, raw)
        for line in lines:
            m = ADDR.search(line)
            if not m:
                continue
            addr = int(m.group(1), 16)
            op = m.group(2).lstrip("@")
            instrs.append((addr, op, line))
        if not instrs:
            continue
        # hottest loop: the backward BRA whose body (target..bra) has the most instructions
        best = None
        for i, (addr, op, raw) in enumerate(instrs):
            if op != "BRA":
                continue
            mt = BRA.search(raw)
            if not mt:
                continue
            tgt = int(mt.group(1), 16)
            if tgt >= addr:
                continue
            body = [x for x in instrs if tgt <= x[0] <= addr]
            if best is None or len(body) > len(best):
                best = body
        hist = {}
        for _, op, _ in (best or instrs):
            base = op.split(".")[0]
            hist[base] = hist.get(base, 0) + 1
        funcs[demangle(raw_name)] = {"loop": hist, "loop_len": len(best or []), "whole": len(instrs)}
    return funcs


def find_func(funcs, *needles):
    for name, info in funcs.items():
        if all(n in name for n in needles):
            return name, info
    return None, None


MMA_OP = {"imma": "IMMA", "omma": "OMMA"}
XOP_SASS_OP = {
    "iadd3": "IADD3", "lop3": "LOP3", "prmt": "PRMT", "shfl": "SHFL", "imad": "IMAD",
    "idp4a": "IDP4A", "hfma2": "HFMA2", "lds128": "LDS", "bra": "BRA",
}


def main():
    out = Path(sys.argv[1])
    rows = []
    jl = out / "issue_ceiling.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    funcs = parse_sass(out / "sass" / "issue_ceiling.sass")

    # per-call SASS: the whole MMA / op expansion, from the alone microkernels (loop body has 8 calls)
    mma_sass, op_sass = {}, {}
    for key in ("imma", "omma"):
        _, info = find_func(funcs, "k_mma", key) if funcs else (None, None)
        if info:
            n = info["loop"].get(MMA_OP[key], 0)
            mma_sass[key] = n / 8.0  # 8 MMAs an iteration
    for key in XOP_SASS_OP:
        _, info = find_func(funcs, "k_op", f"X_{key.upper()}") if funcs else (None, None)
        if info is None and funcs:
            # demangled template arg is the numeric enum value, not the name; fall back to total loop length / 8
            _, info = find_func(funcs, "k_op", key)
        if info:
            op_sass[key] = info["loop_len"] / 8.0  # best-effort: whole loop body / 8 calls

    result = {"mma_sass_per_call": mma_sass, "op_sass_per_call": op_sass, "mix": []}

    md = ["# Mixed-stream issue ceiling (sm_120, RTX PRO 6000)\n"]
    # alone rates
    md.append("## Alone\n")
    md.append("| stream | warp instr/SM/clk | per SMSP | MMA MAC/SM/clk |")
    md.append("|---|---|---|---|")
    mma_alone_macs = {}
    for r in rows:
        if r.get("mode") == "op_alone":
            v = r["op_calls_per_sm_per_clk"]
            md.append(f"| {r['op']} | {v:.3f} | {v/4:.3f} | — |")
        if r.get("mode") == "mma_alone":
            v = r["mma_calls_per_sm_per_clk"]
            mma_alone_macs[r["mma"]] = r["mma_macs_per_sm_per_clk"]
            md.append(f"| {r['mma']} (MMA) | {v:.4f} | {v/4:.4f} | {r['mma_macs_per_sm_per_clk']:.1f} |")

    md.append("\n## Mixed (MMA interleaved with a single-pipe op)\n")
    md.append("| MMA | op | ops/MMA | warp instr/SM/clk (min) | per SMSP | MMA MAC/SM/clk | MMA frac of alone |")
    md.append("|---|---|---|---|---|---|---|")
    peak_smsp = 0.0
    for r in rows:
        if r.get("mode") != "mix":
            continue
        wi = r["warp_instr_min_per_sm_per_clk"]
        smsp = r["warp_instr_min_per_smsp_per_clk"]
        peak_smsp = max(peak_smsp, smsp)
        md.append(f"| {r['mma']} | {r['op']} | {r['ops_per_mma']:.0f} | {wi:.3f} | {smsp:.4f} | "
                  f"{r['mma_macs_per_sm_per_clk']:.1f} | {r['mma_macs_frac_of_alone']:.3f} |")
        result["mix"].append({k: r[k] for k in r if k not in ("probe",)})
    md.append(f"\n**Peak warp instr/SMSP/cycle over every mix: {peak_smsp:.4f}** (ceiling 1.0 = single-issue).\n")
    result["peak_warp_instr_per_smsp_min"] = peak_smsp

    # SASS confirmation
    md.append("## SASS (cuobjdump) per-call instruction counts\n")
    md.append(f"MMA per call: {mma_sass}")
    md.append(f"\nop per loop-body/8: {op_sass}\n")

    # CE7
    ce = out / "ce7.jsonl"
    if ce.exists():
        md.append("## CE7 vs cuBLASLt int8\n")
        md.append("| shape | kernel | ms | GMAC/s | bit-exact vs CPU | CE7/int8 |")
        md.append("|---|---|---|---|---|---|")
        for line in ce.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("mode") == "ce7_result":
                md.append(f"| {r.get('shape','')} | {r.get('kernel','')} | {r.get('ms',0):.4f} | "
                          f"{r.get('gmac_s',0):.1f} | {r.get('exact','')} | {r.get('ratio_vs_int8','')} |")
            result.setdefault("ce7", []).append(r)

    (out / "issue.md").write_text("\n".join(md) + "\n")
    (out / "issue.json").write_text(json.dumps(result, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
