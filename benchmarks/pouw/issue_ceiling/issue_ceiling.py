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


TMPL_INT = re.compile(r"Li(\d+)E")


def tmpl_ints(mangled):
    """The integer template arguments in a mangled name, in order (k_mix<M,X,R> -> [M,X,R])."""
    return [int(n) for n in TMPL_INT.findall(mangled)]


def parse_sass(path):
    """Return {mangled_function: info} where info has the hottest loop's opcode histogram, its length and the
    whole-function length. Keyed by the mangled name so the template arguments (M,X,R) join to the JSONL exactly;
    demangling turns the template ints into enum values and loses that join (combiner bug fixed 16 Oct)."""
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
            if op.split(".")[0] != "BRA":  # the back-edge is BRA.U; match the base opcode
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
        funcs[raw_name] = {
            "mangled": raw_name,
            "demangled": demangle(raw_name),
            "tmpl": tmpl_ints(raw_name),
            "loop": hist,
            "loop_len": len(best or []),
            "whole": len(instrs),
        }
    return funcs


def by_tmpl(funcs, stem, tmpl):
    """The function whose mangled name is `stem` with template ints `tmpl` (e.g. ('k_mix',[0,8,64]))."""
    for info in funcs.values():
        if info["mangled"].startswith(f"_Z{len(stem)}{stem}I") and info["tmpl"] == list(tmpl):
            return info
    return None


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

    # Enum orders must match issue_ceiling.cu: M_IMMA=0, M_OMMA=1; X_NAME below.
    M_NAME = ["imma_s8_16832", "omma_nvf4_16864"]
    M_SASS = ["IMMA", "OMMA"]
    X_NAME = ["iadd3", "lop3", "prmt", "shfl", "imad", "idp4a", "hfma2", "lds128", "bra"]
    X_SASS = ["IADD3", "LOP3", "PRMT", "SHFL", "IMAD", "IDP", "HFMA2", "LDS", "BRA"]
    m_idx = {n: i for i, n in enumerate(M_NAME)}
    x_idx = {n: i for i, n in enumerate(X_NAME)}

    # SASS of each alone kernel: confirm the opcode that issued and the per-call (loop_len/8) instruction count.
    mma_sass, op_sass, op_sass_confirm = {}, {}, {}
    for m, key in enumerate(M_NAME):
        info = by_tmpl(funcs, "k_mma", [m])
        if info:
            mma_sass[key] = {"per_call": info["loop"].get(M_SASS[m], 0) / 8.0,
                             "loop_len": info["loop_len"], "loop": info["loop"]}
    for x, key in enumerate(X_NAME):
        info = by_tmpl(funcs, "k_op", [x])
        if info:
            op_sass[key] = {"per_call_true": info["loop_len"] / 8.0,
                            "intended_per_call": info["loop"].get(X_SASS[x], 0) / 8.0,
                            "loop_len": info["loop_len"], "loop": info["loop"]}
            op_sass_confirm[key] = info["loop"].get(X_SASS[x], 0)

    result = {"mma_sass": mma_sass, "op_sass": op_sass, "mix": []}

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
    md.append("The *min* columns count one SASS instruction per MMA call and per op call (a lower bound on issue). "
              "The *SASS-exact* columns count every instruction in the loop body cuobjdump emits (address generation, "
              "operand moves and the back-edge BRA included): the true warp-instructions the SM issued.")
    md.append("")
    md.append("| MMA | op | ops/MMA | min /SM/clk | min /SMSP | SASS-exact /SM/clk | SASS-exact /SMSP | "
              "MMA MAC/SM/clk | MMA frac of alone |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    peak_smsp = 0.0
    peak_true_smsp = 0.0
    peak_true_row = None
    for r in rows:
        if r.get("mode") != "mix":
            continue
        wi = r["warp_instr_min_per_sm_per_clk"]
        smsp = r["warp_instr_min_per_smsp_per_clk"]
        peak_smsp = max(peak_smsp, smsp)
        warps = r.get("warps", 16)
        M = m_idx.get(r["mma"])
        X = x_idx.get(r["op"])
        R = int(round(r["ops_per_mma"] * 8))
        info = by_tmpl(funcs, "k_mix", [M, X, R]) if funcs else None
        if info:
            true_sm = info["loop_len"] * r["iters"] * warps / r["cyc"]
            true_smsp = true_sm / 4.0
            if true_smsp > peak_true_smsp:
                peak_true_smsp = true_smsp
                peak_true_row = (r["mma"], r["op"], r["ops_per_mma"], info["loop_len"], true_smsp)
            true_sm_s = f"{true_sm:.3f}"
            true_smsp_s = f"{true_smsp:.4f}"
        else:
            true_sm = true_smsp = None
            true_sm_s = true_smsp_s = "—"
        md.append(f"| {r['mma']} | {r['op']} | {r['ops_per_mma']:.0f} | {wi:.3f} | {smsp:.4f} | "
                  f"{true_sm_s} | {true_smsp_s} | "
                  f"{r['mma_macs_per_sm_per_clk']:.1f} | {r['mma_macs_frac_of_alone']:.3f} |")
        row = {k: r[k] for k in r if k not in ("probe",)}
        if true_smsp is not None:
            row["warp_instr_sass_per_sm_per_clk"] = true_sm
            row["warp_instr_sass_per_smsp_per_clk"] = true_smsp
            row["loop_len_sass"] = info["loop_len"]
        result["mix"].append(row)
    md.append(f"\n**Peak warp instr/SMSP/cycle over every mix — min bound: {peak_smsp:.4f}; "
              f"SASS-exact: {peak_true_smsp:.4f}** (ceiling 1.0 = one warp instruction per SMSP per cycle).")
    if peak_true_row:
        md.append(f"SASS-exact peak at {peak_true_row[0]} + {peak_true_row[1]} @ {peak_true_row[2]:.0f} ops/MMA "
                  f"(loop body {peak_true_row[3]} instructions).\n")
    result["peak_warp_instr_per_smsp_min"] = peak_smsp
    result["peak_warp_instr_per_smsp_sass"] = peak_true_smsp

    # SASS confirmation: did the opcode I intended issue?
    md.append("## SASS (cuobjdump): did the intended opcode issue?\n")
    md.append("| op | intended SASS | count in loop (alone, ÷8 = per call) | loop body len | verdict |")
    md.append("|---|---|---|---|---|")
    for x, key in enumerate(X_NAME):
        s = op_sass.get(key)
        if not s:
            md.append(f"| {key} | {X_SASS[x]} | — | — | no SASS |")
            continue
        n = op_sass_confirm.get(key, 0)
        if n >= 8:
            verdict = "issued as intended"
        elif key == "bra":
            verdict = "**if-converted to predicated op** (only the loop back-edge BRA remains)"
        elif n == 0:
            verdict = "**if-converted / not emitted**"
        else:
            verdict = "partial"
        md.append(f"| {key} | {X_SASS[x]} | {n} | {s['loop_len']} | {verdict} |")
    md.append("")
    for m, key in enumerate(M_NAME):
        s = mma_sass.get(key)
        if s:
            md.append(f"- `{key}`: {s['loop'].get(M_SASS[m], 0)}× {M_SASS[m]} in the loop body "
                      f"(÷8 = {s['per_call']:.2f} per MMA call), loop body {s['loop_len']} instructions.")
    md.append("")

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
