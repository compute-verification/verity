"""Each pipe's duty over Dither's overlapped call, from run.sh's MODE=fill FILL_PROF=1 outputs: Nsight Compute's pipe
utilization of every kernel of `v31 fill-once` (ncu-VTAG-LL-SHAPE.raw.csv, OVL_CALLS calls) and fill-bench's own time of an
overlapped call (fill-bench-VTAG-LL.jsonl's ovl arm, beside cuBLASLt's lt).

A kernel's busy time on a pipe is its share of the pipe's peak over its elapsed cycles (ncu's pct_of_peak_sustained_active,
scaled by sm__cycles_active / sm__cycles_elapsed) times its duration; a pipe's duty is the call's busy time over the
overlapped call's wall time, so kernels that overlap add. ncu runs kernels one at a time, so this takes each kernel's pipe
use as unchanged by overlap. The weight side (fill_b_kernel, perm_kernel: once a period) and the operands' generation are
left out, as fill-bench's ovl arm forms W' once, and so is everything before the first call's bits_kernel (cuBLASLt's
algorithm search under V31_LTW).

    python3 h200_pipes.py OUT_DIR    # a markdown table, one row a shape and L
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PCT = re.compile(r"^sm__(inst_executed_pipe_|pipe_)(\w+?)(_cycles_active)?\.avg\.pct_of_peak_sustained_active$")
SKIP = ("gen_kernel", "fill_b_kernel", "perm_kernel", "ne_kernel")
HASH = ("salt_", "leaf_", "node_kernel", "level_kernel", "quad_", "seg_kernel", "top_small")
SCALE = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
#: the pipes the table names (a column each), by ncu's pipe name; the rest go in `other pipes`
SHOWN = (("tensor", ("tensor_op_gmma", "tensor_op_imma", "tensor", "tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma")),
         ("alu", ("alu",)), ("fma", ("fma", "fmaheavy")), ("lsu", ("lsu",)))


def _num(s: str) -> float:
    s = s.replace(",", "").strip()
    return float(s) if s and s not in ("n/a", "N/A") else float("nan")


def kind(name: str) -> str | None:
    """ncu's demangled name ("void gemm_kernel<true, 2>(GArgs)"): our kernels by their base name; cuBLASLt's (V31_LTW)
    by the gemm, xmma or cutlass in its name."""
    base = re.split(r"[<(]", re.sub(r"^void\s+", "", name), maxsplit=1)[0].strip()
    if base.startswith(SKIP):
        return None
    if base.startswith(HASH):
        return "hash"
    if base.startswith("gemm_kernel") or any(x in name.lower() for x in ("gemm", "xmma", "cutlass")):
        return "gemm"
    return "other"


def read_ncu(path: Path) -> list[dict]:
    """One dict a kernel: name, kind, seconds, and {(issue|cycles, pipe): % of peak over its elapsed cycles}."""
    rows = list(csv.reader(path.read_text().splitlines()))
    head = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    cols, units = rows[head], rows[head + 1]
    at = {c: i for i, c in enumerate(cols)}
    unit = units[at["gpu__time_duration.sum"]]
    if unit not in SCALE:
        raise ValueError(f"{path.name}: gpu__time_duration's unit {unit!r} is none of {sorted(SCALE)}")
    data = sorted((r for r in rows[head + 2:] if len(r) == len(cols)), key=lambda r: int(_num(r[at["ID"]])))
    first = next((x for x, r in enumerate(data) if r[at["Kernel Name"]].startswith("bits_kernel")), None)
    if first is None:
        raise ValueError(f"{path.name}: no bits_kernel, so no call to profile")
    out = []
    for r in data[first:]:
        name = r[at["Kernel Name"]]
        k = kind(name)
        if k is None:
            continue
        t = _num(r[at["gpu__time_duration.sum"]]) * SCALE[unit]
        act, el = _num(r[at["sm__cycles_active.avg"]]), _num(r[at["sm__cycles_elapsed.avg"]])
        frac = act / el if el > 0 else 1.0
        pipes = {}
        for c, i in at.items():
            m = PCT.match(c)
            if m:
                pipes[("cycles" if m.group(3) or m.group(1) == "pipe_" else "issue", m.group(2))] = _num(r[i]) * frac
        out.append({"name": name, "kind": k, "s": t, "pipes": pipes})
    return out


def busy(kernels: list[dict], calls: int) -> tuple[dict, dict]:
    """Per call: each pipe's busy seconds (the larger of its issue and cycles-active shares), and each kind's seconds."""
    pipe, kinds = defaultdict(float), defaultdict(float)
    for k in kernels:
        kinds[k["kind"]] += k["s"] / calls
        best: dict[str, float] = {}
        for (_, p), v in k["pipes"].items():
            if v == v:
                best[p] = max(best.get(p, 0.0), v)
        for p, v in best.items():
            pipe[p] += v / 100 * k["s"] / calls
    return pipe, kinds


def shown(pipe: dict, names: tuple[str, ...]) -> float:
    return max((pipe.get(n, 0.0) for n in names), default=0.0)


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    bench = {}
    for f in sorted(out.glob("fill-bench-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.startswith("{"):
                j = json.loads(line)
                bench[(f.stem[len("fill-bench-"):], j["shape"])] = j
    print("| shape | L | variant | ovl ms | cuBLASLt ms | gamma ovl | tensor duty | alu | fma | lsu | idle tensor | gemm ms | "
          "hashing ms | other ms | kernel ms / ovl ms | other pipes (duty > 5%) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    found = 0
    for f in sorted(out.glob("ncu-*.raw.csv")):
        m = re.match(r"ncu-(.+)-L(\d+)-(.+)\.raw$", f.stem)
        if not m:
            continue
        vtag, L, shape = m.group(1), int(m.group(2)), m.group(3)
        log = f.with_name(f.name.replace(".raw.csv", ".log"))
        mc = re.search(r": (\d+) calls\b", log.read_text()) if log.exists() else None
        calls = int(mc.group(1)) if mc else 4
        b = bench.get((f"{vtag}-L{L}", shape))
        if not b:
            print(f"{f.name}: no fill-bench line for {vtag}-L{L} {shape}", file=sys.stderr)
            continue
        ovl, lt = b["arms"]["ovl"]["ms"] * 1e-3, b["arms"]["lt"]["ms"] * 1e-3
        pipe, kinds = busy(read_ncu(f), calls)
        duty = {p: v / ovl for p, v in pipe.items()}
        cols = [100 * shown(duty, names) for _, names in SHOWN]
        named = {n for _, names in SHOWN for n in names}
        rest = ", ".join(f"{p} {100 * v:.0f}%" for p, v in sorted(duty.items(), key=lambda x: -x[1]) if p not in named
                         and v > 0.05)
        kt = sum(kinds.values())
        print(f"| {shape} | {L} | {vtag} | {1e3 * ovl:.3f} | {1e3 * lt:.3f} | {ovl / lt:.3f} | {cols[0]:.1f}% | {cols[1]:.1f}% | "
              f"{cols[2]:.1f}% | {cols[3]:.1f}% | {100 - cols[0]:.1f}% | {1e3 * kinds['gemm']:.3f} | {1e3 * kinds['hash']:.3f} | "
              f"{1e3 * kinds['other']:.3f} | {kt / ovl:.2f} | {rest or '-'} |")
        found += 1
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
