"""Each pipe's duty over Dither's overlapped call, from run.sh's MODE=fill FILL_PROF=1 outputs: Nsight Compute's pipe
utilization of every kernel of `v31 fill-once` (ncu-VTAG-LL-SHAPE.raw.csv, OVL_CALLS calls) and fill-bench's own time of an
overlapped call (fill-bench-VTAG-LL.jsonl's ovl arm, beside cuBLASLt's lt; on V31_GSM's split, its ovl_w_gN arm, whose
calls form W' each as fill-once's do, so the weight side's kernels count there).

A kernel's busy time on a pipe is its share of the pipe's peak over its elapsed cycles (ncu's pct_of_peak_sustained_active,
scaled by sm__cycles_active / sm__cycles_elapsed) times its duration; a pipe's duty is the call's busy time over the
overlapped call's wall time, so kernels that overlap add. ncu runs kernels one at a time, so this takes each kernel's pipe
use as unchanged by overlap. The weight side (fill_b_kernel, perm_kernel: once a period) and the operands' generation are
left out, as fill-bench's ovl arm forms W' once, and so is everything before the first call's bits_kernel (cuBLASLt's
algorithm search under V31_LTW).

Where OUT_DIR has `v31 fill-step`'s lines too (fill-step-VTAG-LL.jsonl), a second table gives each pipe's duty over each
step and N: its calls' busy cycles (each shape's ncu file, its calls the weight side's kernels included, as a step's
calls form W' each) over the step's cycles at its own SM clock. A pipe's work is the same on any SMs, so one ncu run on
every SM gives each split's duty.

    python3 h200_pipes.py OUT_DIR    # markdown tables: one row a shape and L; one a step and N
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
SKIP_W = ("gen_kernel", "ne_kernel")   # over an ovl_w arm, whose calls form W' each
HASH = ("salt_", "leaf_", "node_kernel", "level_kernel", "quad_", "seg_kernel", "top_small")
SCALE = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
#: the pipes the table names (a column each), by ncu's pipe name; the rest go in `other pipes`
SHOWN = (("tensor", ("tensor_op_gmma", "tensor_op_imma", "tensor", "tensor_type_hmma_hgmma_qgmma_imma_igmma_bmma_bgmma")),
         ("alu", ("alu",)), ("fma", ("fma", "fmaheavy")), ("lsu", ("lsu",)))


def _num(s: str) -> float:
    s = s.replace(",", "").strip()
    return float(s) if s and s not in ("n/a", "N/A") else float("nan")


def kind(name: str, skip: tuple[str, ...] = SKIP) -> str | None:
    """ncu's demangled name ("void gemm_kernel<true, 2>(GArgs)"): our kernels by their base name; cuBLASLt's (V31_LTW)
    by the gemm, xmma or cutlass in its name."""
    base = re.split(r"[<(]", re.sub(r"^void\s+", "", name), maxsplit=1)[0].strip()
    if base.startswith(skip):
        return None
    if base.startswith(HASH):
        return "hash"
    if base.startswith("gemm_kernel") or any(x in name.lower() for x in ("gemm", "xmma", "cutlass")):
        return "gemm"
    return "other"


def read_ncu(path: Path, skip: tuple[str, ...] = SKIP) -> list[dict]:
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
        k = kind(name, skip)
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
        out.append({"name": name, "kind": k, "s": t, "cycles": el, "pipes": pipes})
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


def busy_cycles(kernels: list[dict], calls: int) -> dict:
    """Per call: each pipe's busy cycles of an SM, averaged over the device's (the larger of its issue and cycles-active
    shares of the kernel's elapsed cycles): the pipe's work, the same on any SMs and at any clock."""
    pipe = defaultdict(float)
    for k in kernels:
        best: dict[str, float] = {}
        for (_, p), v in k["pipes"].items():
            if v == v:
                best[p] = max(best.get(p, 0.0), v)
        for p, v in best.items():
            pipe[p] += v / 100 * k["cycles"] / calls
    return pipe


def shown(pipe: dict, names: tuple[str, ...]) -> float:
    return max((pipe.get(n, 0.0) for n in names), default=0.0)


def step_duty(out: Path) -> int:
    """fill-step's lines in out (fill-step-VTAG-LL.jsonl): for each step and N, each pipe's duty over the step, its
    calls' busy cycles (each shape's ncu-VTAG-LL-SHAPE.raw.csv, else the one ncu file of that L and shape) over the step's
    cycles (its seconds at its SM clock). A step whose shapes some ncu file lacks is left out. The rows printed."""
    ncu = {}
    for f in sorted(out.glob("ncu-*.raw.csv")):
        m = re.match(r"ncu-(.+)-L(\d+)-(.+)\.raw$", f.stem)
        if m:
            ncu.setdefault((int(m.group(2)), m.group(3)), {})[m.group(1)] = f
    rows = 0
    for f in sorted(out.glob("fill-step-*.jsonl")):
        tag = f.stem[len("fill-step-"):]
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            j = json.loads(line)
            vtag = tag.removesuffix(f"-L{j['L']}")
            per: dict[str, float] = defaultdict(float)
            for s in j["shapes"]:
                by = ncu.get((j["L"], s["shape"]), {})
                p = by.get(vtag) or (next(iter(by.values())) if len(by) == 1 else None)
                if p is None:
                    print(f"{f.name}: no ncu file for L {j['L']} {s['shape']}", file=sys.stderr)
                    break
                log = p.with_name(p.name.replace(".raw.csv", ".log"))
                mc = re.search(r": (\d+) calls\b", log.read_text()) if log.exists() else None
                for q, v in busy_cycles(read_ncu(p, SKIP_W), int(mc.group(1)) if mc else 6).items():
                    per[q] += s["calls"] * v
            else:
                for g in j["gsms"]:
                    a = j["arms"].get(f"step_g{g['n']}")
                    if not a or not a.get("mhz"):
                        continue
                    duty = {q: v / (a["ms"] * 1e-3 * a["mhz"] * 1e6) for q, v in per.items()}
                    cols = [100 * shown(duty, names) for _, names in SHOWN]
                    named = {n for _, names in SHOWN for n in names}
                    rest = ", ".join(f"{q} {100 * v:.0f}%" for q, v in sorted(duty.items(), key=lambda x: -x[1])
                                     if q not in named and v > 0.05)
                    if not rows:
                        print("\n| step | L | variant | N | step s | SM MHz | tensor duty | alu | fma | lsu | idle tensor | "
                              "idle alu | idle fma | idle lsu | other pipes (duty > 5%) |")
                        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                    print(f"| {j['step']} | {j['L']} | {vtag} | {g['n']} | {a['ms'] * 1e-3:.3f} | {a['mhz']:.0f} | "
                          + " | ".join(f"{c:.1f}%" for c in cols) + " | "
                          + " | ".join(f"{100 - c:.1f}%" for c in cols) + f" | {rest or '-'} |")
                    rows += 1
    return rows


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
        g = b.get("gsm") or 0
        arm = f"ovl_w_g{g}" if g and f"ovl_w_g{g}" in b["arms"] else "ovl"
        ovl, lt = b["arms"][arm]["ms"] * 1e-3, b["arms"]["lt"]["ms"] * 1e-3
        pipe, kinds = busy(read_ncu(f, SKIP if arm == "ovl" else SKIP_W), calls)
        vtag += "" if arm == "ovl" else f" ({arm})"
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
    found += step_duty(out)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
