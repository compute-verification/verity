"""ε_skip's per-instruction counts from SASS, on any architecture the kernels build for (INT-HB row 20).

    python eps_counts.py OUT --src TAG=APV_CU [--src ...] [--json FILE]

eps_budget.py prices a peel design's skippable work with BC-KERNEL r2's and D2S-KERNEL r3's exact ncu counts on sm_120:
APV_TAIL (apv_l7qs's tail stage, warp instructions a 128-cell tile by class: IMMA 4, IMAD 4.0, the rest 11.5, which ncu
read as apv_l7qs − apv_cc), c_apv_serve (the served f over a plain epilogue, 4.4 a tile) and c_vfold (vfold − vfold_x,
0.6884 / 0.7265 / 0.7221 warp instructions an entry at 2K / 8K / 32K). Off sm_120 it reads the same numbers. This counts
the same differences from each architecture's SASS (eps_counts.sh's `nvdisasm -c -gi` dumps): every instruction is
placed by the source lines its inline chain passes through, and each region is weighted by how often a warp runs it at
the shape ncu measured (Llama-3.1-8B, 32 query and 8 KV heads):

- the tail: apv_l7qs's instructions inlined from `atail`, once a warp-tile (8 tiles of 128 cells), without its `y2s`
  block (s <= 4096 skips it); `issue`'s (the TMA refill, thread 0's) at 1/16, its warp of the CTA's 16;
- the served epilogue: apv_body's `OUT != 2` block, apv_l7qs's less apv_plain's, a warp-tile;
- the sums: vfold<1, D3> less vfold<0, 0> over a warp's s / 2 entries (a CTA's 8 warps take 4 channels x s keys): the
  key-chunk loop's body s / 1024 times, its `threadIdx.x < 64` block on 2 of the 8 warps, the digest loop on the warps
  whose threads hash (min(nC, 256) of them, nC = s / 128) and everything else once. D3 is on past 4096 tokens.

A static count runs a spin-wait once and leaves out whatever only scheduling differs in, so sm_120a's numbers here are
the check on the rule against ncu's (`RECORDED`); the other architectures are read against sm_120a's, by the same rule.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: eps_budget.py's sm_120 values (BC-KERNEL r2 runs r20261017-090559-fd46 / -092503-35ff; D2S-KERNEL r3 -113844-b913)
RECORDED = {"tail": {"IMMA": 4.0, "IMAD": 4.0, "ALU": 11.5}, "serve": 4.4,
            "vfold": {2048: 0.6884, 8192: 0.7265, 32768: 0.7221}}
SHAPES = (2048, 8192, 32768)
TILES_A_WARP = 8      # a warp's 32 x 32 cells: MI 2 x NI 4 m16n8 tiles
WARPS_A_CTA = 16      # apv's 512 threads; thread 0 issues the TMA refills
VF_WARPS = 8          # vfold's 256 threads
KERNELS = {"apv_l7qs": r"_Z8apv_l7qs6ApArgs$", "apv_cc": r"_Z6apv_cc6ApArgs$", "apv_plain": r"_Z9apv_plain6ApArgs$",
           "vfold_d3": r"_Z5vfoldILb1ELb1EE", "vfold": r"_Z5vfoldILb1ELb0EE", "vfold_x": r"_Z5vfoldILb0ELb0EE"}

_FUNC = re.compile(r"^\s*\.text\.(\S+?):\s*$")
_LOC = re.compile(r'"([^"]+)", line (\d+)')
_INSTR = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")


@dataclass(frozen=True)
class Instr:
    addr: int
    op: str
    frames: tuple[tuple[str, int], ...]   # (file's basename, line), innermost first


def parse(text: str) -> dict[str, list[Instr]]:
    """nvdisasm -c -gi's SASS a function: each instruction with the inline chain of the `//## File` line above it."""
    funcs: dict[str, list[Instr]] = {}
    cur, frames = None, ()
    for line in text.splitlines():
        if m := _FUNC.match(line):
            cur, frames = m.group(1), ()
            funcs.setdefault(cur, [])
        elif "//##" in line:
            frames = tuple((Path(f).name, int(n)) for f, n in _LOC.findall(line))
        elif cur is not None and (m := _INSTR.search(line)):
            body = m.group(2).strip()
            if body.startswith("@"):
                body = body.split(None, 1)[1] if " " in body else ""
            op = body.split(None, 1)[0] if body else ""
            if op:
                funcs[cur].append(Instr(int(m.group(1), 16), op, frames))
    return funcs


def block(lines: list[str], pattern: str, start: int = 0) -> tuple[int, int]:
    """1-based (first, last) lines of the brace block that opens on the first line from `start` matching `pattern`."""
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            depth, opened = 0, False
            for j in range(i, len(lines)):
                text = re.sub(r"//.*", "", lines[j])
                if j == i:
                    text = text[rx.search(text).end():]
                for ch in text:
                    if ch == "{":
                        depth, opened = depth + 1, True
                    elif ch == "}":
                        depth -= 1
                        if opened and depth == 0:
                            return i + 1, j + 1
            break
    raise ValueError(f"no block for {pattern!r}")


def regions(src: str) -> dict[str, tuple[int, int]]:
    """apv.cu's line ranges the counts read; `y2s` and the vfold ones only where the tree has them (BC-KERNEL r3 and
    D2S-KERNEL r3 added them). An empty range matches nothing."""
    ln = src.splitlines()
    none = (0, -1)
    atail = block(ln, r"DEV void atail\(")
    out = {"atail": atail, "issue": block(ln, r"DEV void issue\(\)"),
           "epilogue": block(ln, r"else if constexpr \(OUT != 2\)")}
    y2s = block(ln, r"if \(y2s\) \{", atail[0] - 1) if any("if (y2s) {" in x for x in ln) else none
    out["y2s"] = y2s if atail[0] <= y2s[0] <= atail[1] else none
    if any("c0 < nC; c0 += 8" in x for x in ln):
        loop = block(ln, r"for \(int c0 = 0; c0 < nC; c0 \+= 8\)")
        out.update(vf_loop=loop, vf_tid64=block(ln, r"if \(threadIdx\.x < 64\)", loop[0] - 1),
                   vf_hash=block(ln, r"for \(int i = threadIdx\.x; i < nC; i \+= VF_NT\)"))
    else:
        out.update(vf_loop=none, vf_tid64=none, vf_hash=none)
    return out


def within(ins: Instr, r: tuple[int, int], file: str = "apv.cu") -> bool:
    return any(f == file and r[0] <= n <= r[1] for f, n in ins.frames)


def base(op: str) -> str:
    return op.split(".", 1)[0]


def collapse(c: Counter) -> dict[str, float]:
    """eps_budget's APV_TAIL classes: IMMA, IMAD and the rest (priced as ALU's)."""
    out = {"IMMA": 0.0, "IMAD": 0.0, "ALU": 0.0}
    for op, n in c.items():
        out[base(op) if base(op) in ("IMMA", "IMAD") else "ALU"] += n
    return out


def tail(fn: list[Instr], R: dict) -> Counter:
    c: Counter = Counter()
    for i in fn:
        if within(i, R["atail"]) and not within(i, R["y2s"]):
            c[base(i.op)] += (1 / WARPS_A_CTA if within(i, R["issue"]) else 1.0) / TILES_A_WARP
    return c


def epilogue(fn: list[Instr], R: dict) -> Counter:
    c: Counter = Counter()
    for i in fn:
        if within(i, R["epilogue"]):
            c[base(i.op)] += 1.0 / TILES_A_WARP
    return c


def vf_weight(i: Instr, R: dict, s: int) -> float:
    """How many times an average warp of vfold's CTA runs instruction `i` at s tokens."""
    nC = s // 128
    if within(i, R["vf_loop"]):
        return (s // 1024) * (2 / VF_WARPS if within(i, R["vf_tid64"]) else 1.0)
    if within(i, R["vf_hash"]):
        return min(-(-min(nC, 256) // 32), VF_WARPS) / VF_WARPS
    return 1.0


def vf_count(fn: list[Instr], R: dict, s: int) -> Counter:
    c: Counter = Counter()
    for i in fn:
        c[base(i.op)] += vf_weight(i, R, s)
    return c


def diff(a: Counter, b: Counter, scale: float = 1.0) -> Counter:
    out: Counter = Counter()
    for k in set(a) | set(b):
        v = (a.get(k, 0.0) - b.get(k, 0.0)) * scale
        if abs(v) > 1e-12:
            out[k] = v
    return out


def kernels(funcs: dict[str, list[Instr]]) -> dict[str, list[Instr]]:
    out = {}
    for name, rx in KERNELS.items():
        hits = [f for f in funcs if re.search(rx, f)]
        if len(hits) > 1:
            raise ValueError(f"{name}: {len(hits)} functions match {rx}: {hits}")
        if hits:
            out[name] = funcs[hits[0]]
    return out


def ptxas(log: str) -> dict[str, dict]:
    """ptxas -v's registers and spill bytes a kernel."""
    out, cur = {}, None
    for line in log.splitlines():
        if m := re.search(r"Compiling entry function '(\S+)'", line):
            cur = m.group(1)
        elif cur and (m := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)):
            out.setdefault(cur, {}).update(spill_stores=int(m.group(1)), spill_loads=int(m.group(2)))
        elif cur and (m := re.search(r"Used (\d+) registers", line)):
            out.setdefault(cur, {})["registers"] = int(m.group(1))
    return out


def counts(sass: str, src: str, log: str = "") -> dict:
    """One build's counts: the tail and served epilogue a tile, the sums an entry at each shape, with their parts."""
    R, ks = regions(src), kernels(parse(sass))
    out: dict = {"regions": R, "kernels": {k: len(v) for k, v in ks.items()}}
    if "apv_l7qs" in ks:
        t = tail(ks["apv_l7qs"], R)
        out["tail"] = {"by_opcode": dict(t), "classes": collapse(t), "total": sum(t.values())}
        if "apv_plain" in ks:
            e, p = epilogue(ks["apv_l7qs"], R), epilogue(ks["apv_plain"], R)
            sv = diff(e, p)
            out["serve"] = {"by_opcode": dict(sv), "total": sum(sv.values()), "epilogue": sum(e.values()),
                            "plain_epilogue": sum(p.values())}
            if "apv_cc" in ks:
                out["serve"]["cc_epilogue"] = sum(epilogue(ks["apv_cc"], R).values())
    if "vfold_x" in ks:
        out["vfold"] = {}
        for s in SHAPES:
            k = "vfold_d3" if s > 4096 else "vfold"
            if k in ks:
                d = diff(vf_count(ks[k], R, s), vf_count(ks["vfold_x"], R, s), 2 / s)
                out["vfold"][s] = {"by_opcode": dict(d), "total": sum(d.values())}
    regs = ptxas(log)
    out["ptxas"] = {k: v for f, v in regs.items() for k, rx in KERNELS.items() if re.search(rx, f)}
    return out


def table(res: dict) -> str:
    rows = ["| source | arch | tail IMMA | tail IMAD | tail other | tail total | serve | vfold 2K | vfold 8K | vfold 32K |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    r = RECORDED
    rows.append(f"| ncu (eps_budget) | sm_120 | {r['tail']['IMMA']:.2f} | {r['tail']['IMAD']:.2f} | {r['tail']['ALU']:.2f} "
                f"| {sum(r['tail'].values()):.2f} | {r['serve']:.2f} | "
                + " | ".join(f"{r['vfold'][s]:.4f}" for s in SHAPES) + " |")
    for tag, by in res.items():
        for arch, c in by.items():
            t = c.get("tail", {}).get("classes", {})
            vf = c.get("vfold", {})
            cell = lambda v, f=".2f": format(v, f) if v is not None else "–"  # noqa: E731
            rows.append(f"| {tag} | {arch} | {cell(t.get('IMMA'))} | {cell(t.get('IMAD'))} | {cell(t.get('ALU'))} | "
                        f"{cell(c.get('tail', {}).get('total'))} | {cell(c.get('serve', {}).get('total'))} | "
                        + " | ".join(cell(vf.get(s, {}).get("total"), ".4f") for s in SHAPES) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path, help="eps_counts.sh's OUT (TAG/ARCH/apv.gi.sass.gz)")
    p.add_argument("--src", action="append", required=True, help="TAG=path to that tree's apv.cu")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    res: dict = {}
    for spec in a.src:
        tag, path = spec.split("=", 1)
        src = Path(path).read_text()
        for d in sorted((a.out / tag).iterdir()):
            f = d / "apv.gi.sass.gz"
            if f.exists():
                log = (d / "build.log").read_text() if (d / "build.log").exists() else ""
                res.setdefault(tag, {})[d.name] = counts(gzip.decompress(f.read_bytes()).decode(), src, log)
    print(table(res))
    if a.json:
        a.json.write_text(json.dumps({"recorded_sm120": RECORDED, "counts": res}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
