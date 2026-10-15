"""Assemble a census run's outputs into the instance table: $OUT/census.json and $OUT/census.md.

Reads each section's JSON lines (`<section>.jsonl`), the SASS of each binary (`sass/<bin>.sass`, from cuobjdump), the
lease's nvidia-smi records, and writes one row per constant: constant, value, unit, run id, GPU UUID, method. Rates of
the CUDA-pipe probe are recomputed per SASS instruction: the opcodes of each kernel's main loop (the longest
backward-branch range) are counted, so a rate is for what ptxas emitted, not for the PTX. Stdlib only.

    python3 census.py OUT_DIR [RUN_ID]
"""

import json
import pathlib
import re
import statistics
import sys
from collections import Counter

INSTR = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
FUNC = re.compile(r"Function : (\S+)")


def sass_loops(path):
    """{mangled kernel name: (Counter of full opcodes in its longest loop, loop length)}."""
    out = {}
    if not path.exists():
        return out
    name, instrs = None, []

    def flush():
        if name is None:
            return
        best = (Counter(), 0)
        for i, (addr, text) in enumerate(instrs):
            m = re.match(r"(?:@!?U?P\w+\s+)?BRA(?:\.\w+)*\s+(?:!?U?P\w+,\s*)?(0x[0-9a-f]+)", text)
            if not m:
                continue
            tgt = int(m.group(1), 16)
            if tgt >= addr:
                continue
            body = [t for a, t in instrs if tgt <= a <= addr]
            if len(body) > best[1]:
                ops = Counter()
                for t in body:
                    t = re.sub(r"^@!?U?P\w+\s+", "", t)
                    ops[t.split()[0]] += 1
                best = (ops, len(body))
        out[name] = best

    for line in path.read_text(errors="replace").splitlines():
        f = FUNC.search(line)
        if f:
            flush()
            name, instrs = f.group(1), []
            continue
        m = INSTR.search(line)
        if m and name:
            instrs.append((int(m.group(1), 16), m.group(2)))
    flush()
    return out


def base(op):
    return op.split(".")[0]


def jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def smi_summary(out):
    s = {}
    samples = out / "smi-samples.csv"
    if samples.exists():
        clk, pw, reasons = [], [], Counter()
        for line in samples.read_text().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 7:
                continue
            try:
                clk.append(float(p[1].split()[0]))
                pw.append(float(p[3].split()[0]))
                reasons[p[5]] += 1
            except ValueError:
                pass
        if clk:
            s.update(sm_clock_mhz_median=statistics.median(clk), sm_clock_mhz_min=min(clk), sm_clock_mhz_max=max(clk),
                     power_w_max=max(pw), power_w_median=statistics.median(pw), samples=len(clk),
                     clock_event_reasons=dict(reasons))
    before = out / "smi-before.txt"
    if before.exists():
        txt = before.read_text()
        sec = None
        for line in txt.splitlines():
            if line.strip() and not line.startswith(" " * 8) and line.startswith("    ") and ":" not in line:
                sec = line.strip()
            m = re.match(r"\s+(.+?)\s+:\s+(.+)", line)
            if m and sec:
                s.setdefault("smi_q", {})[f"{sec} / {m.group(1).strip()}"] = m.group(2).strip()
    return s


def main():
    out = pathlib.Path(sys.argv[1])
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    run_id = sys.argv[2] if len(sys.argv) > 2 else meta.get("run_id", "?")
    lease = json.loads((out / "lease.json").read_text()) if (out / "lease.json").exists() else {}
    uuid = lease.get("gpu_uuid", "?")
    rows = []

    def row(constant, value, unit, method, section, **extra):
        rows.append(dict(constant=constant, value=value, unit=unit, run_id=run_id, gpu_uuid=uuid, method=method,
                         section=section, **extra))

    smi = smi_summary(out)
    sass = {}
    for f in (out / "sass").glob("*.sass") if (out / "sass").exists() else []:
        sass.update(sass_loops(f))

    # ---- tensor ----
    tens = jsonl(out / "tensor.jsonl")
    kinds = []
    for r in tens:
        if r.get("kind") and r["kind"] not in kinds:
            kinds.append(r["kind"])
    for k in kinds:
        thr = [r for r in tens if r.get("kind") == k and r.get("mode") == "throughput"]
        lat = [r for r in tens if r.get("kind") == k and r.get("mode") == "latency"]
        if not thr:
            continue
        best = max(thr, key=lambda r: r["macs_per_sm_per_clk"])
        key = next((n for n in sass if f"I{len(k)}{k}Li{best['nacc']}E" in n), None)
        ops, n = sass.get(key, (Counter(), 0))
        mma_ops = {o: c for o, c in ops.items() if base(o) in ("QMMA", "OMMA", "HMMA", "IMMA", "DMMA", "BMMA")}
        per_instr = best["nacc"]
        if mma_ops:
            sass_desc = ", ".join(f"{o} x{c}" for o, c in mma_ops.items()) + f" per {per_instr} MMAs"
        else:
            sass_desc = f"no tensor opcode: emulated, {n} SASS instructions per {per_instr} MMAs ({dict(ops.most_common(4))})"
        warps_desc = "; ".join(f"{r['warps']}w {r['macs_per_sm_per_clk']:.1f}" for r in thr)
        meth = (f"tensor.cu k_mma: {best['ptx']}; one block a SM, {best['warps']} warps x {best['nacc']} independent "
                f"accumulators, clock64 per block; SASS: {sass_desc}; MACs/SM/clk by warps: {warps_desc}")
        row(f"tensor.{k}.macs_per_sm_per_clk", round(best["macs_per_sm_per_clk"], 2), "MAC/SM/clk", meth, "tensor",
            macs_per_instr=best["macs_per_instr"], stored_macs_per_instr=best["stored_macs_per_instr"],
            sass=list(mma_ops) or "emulated")
        row(f"tensor.{k}.slot_cycles", round(best["smsp_cycles_per_instr"], 3), "SMSP cycles per warp MMA",
            "4 x SM cycles / warp MMAs a SM at saturation (tensor.cu)", "tensor")
        row(f"tensor.{k}.device_macs_per_s", round(best["device_tmacs_per_s"] * 1e12, -9), "MAC/s",
            f"all {best['distinct_sms']} SMs, event-timed, SM clock {best['mhz']:.0f} MHz (cyc_max/event)", "tensor")
        if lat:
            row(f"tensor.{k}.latency_cycles", round(lat[0]["cycles_per_dependent_instr"], 2), "SM cycles",
                "one warp, one accumulator chain (tensor.cu)", "tensor")

    # ---- CUDA pipes ----
    for r in jsonl(out / "cuda_pipes.jsonl"):
        if "op1" not in r:
            continue
        o1, o2 = r["op1"], r["op2"]
        prefix = f"_Z6k_pipeI{len(o1) + 3}op_{o1}" + ("S0_E" if o1 == o2 else f"{len(o2) + 3}op_{o2}E")
        key = next((n for n in sass if n.startswith(prefix)), None)
        ops, n = sass.get(key, (Counter(), 0))
        if not n:
            continue
        per_iter = r["warps"] * r["iters"] / r["cyc_med"]
        overhead = {"IADD3", "ISETP", "BRA", "UIADD3", "UISETP", "NOP"}
        keep_iadd3 = o1.startswith("iadd3") or o2.startswith("iadd3")
        work = Counter({o: c for o, c in ops.items() if base(o) not in overhead or (keep_iadd3 and base(o) == "IADD3")})
        top = work.most_common(4) or ops.most_common(4)
        label = o1 if o1 == o2 else f"{o1}+{o2}"
        if r["warps"] != 32:
            label += f"@{r['warps']}w"
        desc = ", ".join(f"{o} {c}" for o, c in top)
        rates = {o: c * per_iter for o, c in top}
        main_op, main_c = top[0]
        meth = (f"cuda_pipes.cu k_pipe<{o1},{o2}>: {r['warps']} warps, one block a SM, 8-register rotation; loop body "
                f"{n} SASS ({desc}); rates per SASS opcode: " + ", ".join(f"{o} {v:.3f}" for o, v in rates.items())
                + f"; all SASS {n * per_iter:.3f} warp instr/SM/clk")
        row(f"cuda.{label}.warp_instr_per_sm_per_clk", round(main_c * per_iter, 4),
            f"warp {main_op}/SM/clk", meth, "cuda", lanes_per_sm_per_clk=round(main_c * per_iter * 32, 2),
            total_issue_per_sm_per_clk=round(n * per_iter, 4), loop_hist=dict(ops))

    # ---- co-issue ----
    for r in jsonl(out / "coissue.jsonl"):
        if r.get("mode") == "same_kernel":
            row(f"coissue.{r['mma']}+{r['xop']}.k{r['xops_per_mma']:g}.w{r['warps']}.mma_slowdown",
                round(r["mma_slowdown"], 4), "x (time both / MMA alone)",
                f"coissue.cu same kernel: 8 MMAs + {r['xops_per_mma']*8:g} {r['xop']} an iteration; MMA alone "
                f"{r['mma_macs_per_sm_per_clk_alone']:.1f} -> {r['mma_macs_per_sm_per_clk_both']:.1f} MAC/SM/clk; "
                f"{r['xop']} alone {r['x_warp_ops_per_sm_per_clk_alone']:.3f} -> both {r['x_warp_ops_per_sm_per_clk_both']:.3f}"
                f" PTX warp ops/SM/clk; both/(sum of alone) {r['both_over_sum']:.3f}", "coissue",
                x_slowdown=round(r["x_slowdown"], 4), both_over_sum=round(r["both_over_sum"], 4))
        elif r.get("mode") == "split_warps":
            row(f"coissue.split.{r['mma']}+{r['xop']}.m{r['mma_warps']}x{r['x_warps']}.both_over_sum",
                round(r["both_over_sum"], 4), "time both / (time MMA alone + time op alone)",
                f"coissue.cu k_split: one block a SM, {r['mma_warps']} warps only MMAs + {r['x_warps']} warps only "
                f"{r['xop']}; alone {r['mma_macs_per_sm_per_clk_alone']:.1f} MAC/SM/clk and "
                f"{r['x_warp_ops_per_sm_per_clk_alone']:.3f} PTX warp ops/SM/clk; while both ran the MMAs kept "
                f"{r['mma_fraction_during_overlap']:.3f} and the {r['xop']} warps {r['x_fraction_during_overlap']:.3f} "
                f"of their alone rates (role spans are the slowest warp's: with fewer than 4 op warps, the MMA warps "
                f"on a sub-partition that has an op warp)", "coissue",
                mma_fraction_during_overlap=round(r["mma_fraction_during_overlap"], 4),
                x_fraction_during_overlap=round(r["x_fraction_during_overlap"], 4))
        elif r.get("mode") == "two_kernels":
            row(f"coissue.two_kernels.{r['mma']}+{r['xop']}.mma_rate_together", round(r["mma_macs_per_sm_per_clk_together"], 2),
                "MAC/SM/clk", f"coissue.cu two streams, one 4-warp block of each a SM: MMA alone "
                f"{r['mma_macs_per_sm_per_clk_alone']:.1f}, {r['xop']} alone {r['x_warp_ops_per_sm_per_clk_alone']:.3f} -> "
                f"together {r['x_warp_ops_per_sm_per_clk_together']:.3f} PTX warp ops/SM/clk; overlap "
                f"{r['overlap_fraction']:.2f}, {r['x_blocks_on_mma_sms']} of the op kernel's blocks on SMs running MMA blocks",
                "coissue")

    # ---- memory side ----
    ms = jsonl(out / "memside.jsonl")
    for r in ms:
        p = r.get("probe")
        if p == "red":
            row(f"l2red.{r['op']}.{r['pattern']}.{r['target']}.elem_adds_per_s", float(f"{r['elem_adds_per_s']:.4g}"),
                "element adds/s", f"memside.cu k_red: {r['op']} ({r['bytes_per_lane']} B, {r['elems_per_lane']} elements a "
                f"lane), {r['pattern']} addresses into a {'32 MiB (L2-resident)' if r['target'] == 'l2' else '4 GiB'} buffer; "
                f"lane ops/s {r['lane_ops_per_s']:.4g}, payload {r['payload_bytes_per_s']:.4g} B/s", "memside")
        elif p == "bulk_load":
            row(f"tma.bulk_load.{r['target']}.chunk{r['chunk']}.bpsm{r['blocks_per_sm']}.bytes_per_s",
                float(f"{r['bytes_per_s']:.4g}"), "B/s", "memside.cu k_bulk_load: cp.async.bulk global->shared::cta, "
                "mbarrier complete_tx, 4-stage ring, one thread a block", "memside")
        elif p == "bulk_red":
            row(f"tma.bulk_reduce.{r['op']}.{r['target']}.chunk{r['chunk']}.elem_adds_per_s",
                float(f"{r['elem_adds_per_s']:.4g}"), "element adds/s",
                f"memside.cu k_bulk_red: cp.reduce.async.bulk.global.shared::cta.bulk_group.{r['op']}, 4 blocks a SM; "
                f"payload {r['payload_bytes_per_s']:.4g} B/s", "memside")
    mix = [r for r in ms if r.get("probe") == "mix"]
    pairs = sorted({(r["role_a"], r["role_b"]) for r in mix})
    unit = {"ffma": "FFMA lane ops", "qmma_e4m3": "MACs", "red_v4_f32": "FP32 L2 adds", "ldg_l2": "L2 bytes"}
    for a, b in pairs:
        sel = {r["case"]: r for r in mix if (r["role_a"], r["role_b"]) == (a, b)}
        if not all(c in sel for c in ("a_alone", "b_alone", "both")):
            continue
        ra = sel["both"]["a_units_per_sm_per_clk"] / sel["a_alone"]["a_units_per_sm_per_clk"]
        rb = sel["both"]["b_units_per_sm_per_clk"] / sel["b_alone"]["b_units_per_sm_per_clk"]
        row(f"mix.{a}+{b}.rate_fraction_{a}", round(ra, 4), "rate beside / rate alone",
            f"memside.cu k_mix: 8 warps {a} + 8 warps {b} in one block a SM; {a} {sel['a_alone']['a_units_per_sm_per_clk']:.3f}"
            f" -> {sel['both']['a_units_per_sm_per_clk']:.3f} {unit[a]}/SM/clk; {b} {sel['b_alone']['b_units_per_sm_per_clk']:.3f}"
            f" -> {sel['both']['b_units_per_sm_per_clk']:.3f} {unit[b]}/SM/clk", "memside", partner_fraction=round(rb, 4))

    # ---- memory ----
    mem = jsonl(out / "memory.jsonl")
    for r in mem:
        p = r.get("probe")
        if p == "props":
            for k in ("sms", "l2_bytes", "persisting_l2_max_bytes", "smem_per_sm_bytes", "smem_per_block_optin_bytes",
                      "regs_per_sm", "max_threads_per_sm", "max_blocks_per_sm", "mem_bus_bits", "mem_total_bytes",
                      "sm_clock_khz_attr", "mem_clock_khz_attr", "generic_compression", "async_engines", "max_cluster"):
                if k in r:
                    row(f"props.{k}", r[k], "", f"cudaGetDeviceProperties / cuDeviceGetAttribute ({r.get('name')}, cc {r.get('cc')})",
                        "memory")
        elif p == "compress_alloc":
            row("compression.generic_granted", r["granted"], "CUmemAllocationCompType",
                "cuMemCreate(CU_MEM_ALLOCATION_COMP_GENERIC) then cuMemGetAllocationPropertiesFromHandle", "memory")
        elif p == "dram":
            for d in ("read", "write"):
                row(f"dram.{r['alloc']}.{r['pattern']}.{d}_bytes_per_s", float(f"{r[d + '_bytes_per_s']:.4g}"), "B/s",
                    f"memory.cu: 4 GiB {'cudaMalloc' if r['alloc'] == 'plain' else 'cuMemCreate COMP_GENERIC'}, "
                    f"pattern {r['pattern']}, {'ld.global.cg 16 B, 4 in flight' if d == 'read' else 'st.global 16 B'}, best of reps",
                    "memory")
        elif p == "dram_copy":
            row("dram.copy_d2d.read_plus_write_bytes_per_s", float(f"{r['read_plus_write_bytes_per_s']:.4g}"), "B/s",
                f"cudaMemcpyAsync D2D of {r['bytes']} B random data", "memory")
        elif p == "l2_sweep":
            row(f"l2.{r['alloc']}.{r['pattern']}.footprint{r['footprint_mib']}MiB.read_bytes_per_s",
                float(f"{r['read_bytes_per_s']:.4g}"), "B/s", f"memory.cu k_read over a {r['footprint_mib']} MiB footprint "
                f"{r['reps']} times in one launch (ld.global.cg)", "memory")
        elif p == "smem":
            row(f"smem.{r['op']}.bytes_per_sm_per_clk", round(r["bytes_per_sm_per_clk"], 3), "B/SM/clk",
                f"memory.cu k_smem: conflict-free {r['op']}, 32 warps, one block a SM, clock64; device "
                f"{r['device_bytes_per_s']:.4g} B/s at {r['mhz']:.0f} MHz", "memory")
        elif p == "read_path":
            where = "l2" if r["footprint_bytes"] <= 64 << 20 else "dram"
            row(f"read_path.{r['data']}.{where}.p{r['path_id']}.b{r['blocks_per_sm']}.bytes_per_s",
                float(f"{r['read_bytes_per_s']:.4g}"), "B/s",
                f"memory.cu k_read_path: {r['path']}, {r['blocks_per_sm']} x 512-thread blocks a SM (grid-stride), "
                f"{r['data']} over {r['footprint_bytes'] >> 20} MiB", "memory")
        elif p == "fill_method":
            reads = ", ".join(f"{x:.3g}" for x in r["m5_reads_bytes_per_s"])
            row(f"fill_method.zeros.{r['method']}.t{r['trial']}.m5_read.bytes_per_s",
                float(f"{r['m5_best_after_first']:.4g}"), "B/s",
                f"memory.cu fill_methods: fresh 4 GiB COMP_GENERIC allocation, zeros written by {r['method']}; best of "
                f"reads 2-6 with M5's k_read (one __ldg a lane, 8 x 512 a SM); all six: {reads}", "memory")
            row(f"fill_method.zeros.{r['method']}.t{r['trial']}.cg_read.bytes_per_s",
                float(f"{r['cg_b4_read_bytes_per_s']:.4g}"), "B/s",
                f"memory.cu fill_methods: same allocation read with ld.global.cg x4, 4 x 512 a SM, best of 4", "memory")
        elif p == "gather":
            row(f"gather.{r['target']}.row{r['row_bytes']}.bytes_per_s", float(f"{r['bytes_per_s']:.4g}"), "B/s",
                f"memory.cu k_gather: random rows of {r['row_bytes']} B from a {'4 GiB' if r['target'] == 'dram' else '32 MiB'}"
                f" buffer (ld.global.cg 16 B a lane); {r['rows_per_s']:.4g} rows/s", "memory")

    # ---- links, host ----
    for r in jsonl(out / "links.jsonl"):
        if "constant" in r:
            rows.append(dict(constant=r["constant"], value=r["value"], unit=r.get("unit", ""), run_id=run_id,
                             gpu_uuid=r.get("gpu_uuid", uuid), method=r.get("method", ""), section="links"))
    for r in jsonl(out / "host.jsonl"):
        if "constant" in r:
            rows.append(dict(constant=r["constant"], value=r["value"], unit=r.get("unit", ""), run_id=run_id,
                             gpu_uuid="host", method=r.get("method", ""), section="host"))

    # ---- clocks and power ----
    if smi:
        for k in ("sm_clock_mhz_median", "sm_clock_mhz_min", "sm_clock_mhz_max", "power_w_max", "power_w_median"):
            if k in smi:
                row(f"clocks.lease.{k}", smi[k], "", f"nvidia-smi --query-gpu every 200 ms over the lease ({smi['samples']} "
                    f"samples; clock-event reasons {smi['clock_event_reasons']})", "clocks")
        q = smi.get("smi_q", {})
        for k, v in q.items():
            if re.search(r"Clocks|Power|Performance State|Limit", k) and re.search(r"MHz|W$|P\d", v):
                row(f"clocks.smi_q.{k}", v, "", "nvidia-smi -q -d CLOCK,POWER,PERFORMANCE before the probes (read-only)",
                    "clocks")

    doc = dict(meta=dict(meta, run_id=run_id, gpu_uuid=uuid, smi=smi), rows=rows,
               raw={f.stem: jsonl(f) for f in sorted(out.glob("*.jsonl"))})
    (out / "census.json").write_text(json.dumps(doc, indent=1))
    lines = ["| constant | value | unit | run id | GPU UUID | method |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['constant']}` | {r['value']} | {r['unit']} | `{r['run_id']}` | `{r['gpu_uuid'][:12]}` | "
                     f"{r['method'].replace('|', '/')} |")
    (out / "census.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {out / 'census.json'}")


if __name__ == "__main__":
    main()
