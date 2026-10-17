"""run.sh EG_SET=eps1's (or eps1lag's, eps1sleep's, eps1wave's) profile against the machine model: per arm and shape
(each shape's m, n, k from its gate's meta), ncu's executed SASS instructions a MMA by census class, the bound
max_r sum_k n_k u_kr (SMSP-cycles a MMA, the model's time for that mix) with its binding row, the measured cycles a
MMA (the hot bench's, and ncu's one call), eps_sched = 1 - bound / measured, issue, tensor and every pipe's utilization
and the stall reasons; the fold-cost sweep's table; and the profiles as TOML for `machine-model eps`.

    python3 eps_profile.py OUT_DIR --mm MACHINE_MODEL_DIR [--epoch VF3l-16] [--json PATH] [--toml PATH] [--toml-arms A,B]

MACHINE_MODEL_DIR is `benchmarks/pouw/machine_model` (verity_machine_model and its data/). An opcode with no class is
priced as NOROW (issue only), so the bound is never above what the model says, and is listed under `unpriced`.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HONEST = "e_l2q"
SWEEP = [(0.0, "e_plain"), (1.0, "f10_l2q"), (1.5, "f15_l2q"), (2.0, "f20_l2q"), (2.5, "f25_l2q"), (3.0, "e_l2q"),
         (4.0, "f40_l2q")]
SHAPES = {"o": (8192, 8192, 8192), "o2k": (2048, 4096, 4096), "gu2k": (2048, 28672, 4096)}
NOROW = {"BAR", "BRA", "BSSY", "BSYNC", "CCTL", "CS2R", "DEPBAR", "ELECT", "ERRBAR", "EXIT", "FENCE", "MEMBAR", "NANOSLEEP",
         "NOP", "RET", "S2R", "SYNCS", "VOTE", "WARPSYNC", "YIELD", "CALL", "BMOV", "ACQBULK", "STG", "ST", "STL", "LDL",
         "PLOP3", "P2R", "R2P", "MOV"}
ALU = {"LOP3", "IADD3", "SHF", "PRMT", "LEA", "ISETP", "I2FP", "IABS", "SGXT", "BMSK", "LOP", "IADD", "IMNMX", "FLO_ALU"}


def kind_of(op: str, tma: str) -> str | None:
    """The census class of a SASS mnemonic (with its modifiers), or None: priced as NOROW and listed as unpriced."""
    b = op.split(".")[0]
    if b in ("IMMA", "HMMA", "QMMA", "OMMA"):
        return "MMA"
    if b == "IMAD":
        return "IMADW" if ".WIDE" in op else "IMAD"
    if b == "IDP":
        return "IMAD"
    if b == "F2FP":
        return "F2FS" if "SATFINITE" in op else "ALU"
    if b in ALU:
        return "ALU"
    if b in ("SEL", "FSEL"):
        return "SEL"
    if b == "VIMNMX":
        return "VIMNMX"
    if b in ("FFMA", "FADD", "FMUL"):
        return "FFMA"
    if b in ("HFMA2", "HADD2", "HMUL2"):
        return "HFMA2"
    if b in ("POPC", "BREV", "FLO", "MUFU"):
        return "XU"
    if b == "SHFL":
        return "SHFL"
    if b == "LDSM":
        w = {"4": "X4", "2": "X2"}.get(op.split(".")[-1], "X1")
        return f"LDSM_{w}{'T' if '.MT88' in op else ''}"
    if b == "LDS":
        return "LDS128" if ".128" in op else "LDS32"
    if b == "STS":
        return "STS128" if ".128" in op else None
    if b == "LDG":
        return "LDG128" if ".128" in op else "LDG32"
    if b == "UTMALDG":
        return tma
    if b in ("RED", "REDG"):
        return "RED64" if ".64" in op else "RED32"
    if b.startswith("U") or b in ("R2UR", "S2UR"):
        return "UNI"
    if b in NOROW:
        return "NOROW"
    return None


def opcode(src: str) -> str:
    s = src.strip()
    if s.startswith("@"):
        s = s.split(None, 1)[1] if " " in s else ""
    return s.split(None, 1)[0].rstrip(";") if s else ""


def table(path: Path) -> list[dict]:
    """The rows of an ncu --csv export, from its header line on."""
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    i = next((i for i, x in enumerate(lines) if x.startswith('"') and ("Source" in x or "ID" in x)), None)
    return [] if i is None else list(csv.DictReader(io.StringIO("\n".join(lines[i:]))))


def num(v: str) -> float:
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return float("nan")


def sass_counts(path: Path) -> tuple[dict[str, float], dict[str, float], list[tuple[str, float, float]]]:
    """Executed warp instructions by mnemonic, over the kernel and over its hot loop (instructions executed at least half
    as often as the most executed MMA), and per instruction (mnemonic, executed, stall samples). An mbarrier's retry loop
    (YIELD; SYNCS.PHASECHK...TRYWAIT; @!P BRA; BRA, ptxas's out-of-line spin, or with a suspend-time hint TRYWAIT;
    @!P NANOSLEEP.SYNCS; @!P SYNCS.PHASECHK; @!P BRA; BRA) is counted as WAIT.<mnemonic>: how often it runs is how long a
    warp waited, not work."""
    rows = table(path)
    if not rows:
        return {}, {}, []
    ex = next(k for k in rows[0] if k.strip() == "Instructions Executed")
    st = next((k for k in rows[0] if k.strip().startswith("Warp Stall Sampling (All")), None)
    per = [(opcode(r["Source"]), num(r[ex]), num(r[st]) if st else 0.0) for r in rows if r.get("Source")]
    per = [p for p in per if p[0] and p[1] == p[1]]
    for i in range(len(per) - 3):
        if per[i][0] == "YIELD" and per[i + 1][0].startswith("SYNCS.PHASECHK") and per[i + 2][0] == "BRA":
            for j in range(i, i + 3 + (per[i + 3][0] == "BRA")):
                per[j] = ("WAIT." + per[j][0],) + per[j][1:]
    for i in range(len(per) - 4):
        if (per[i][0].startswith("SYNCS.PHASECHK") and per[i + 1][0].startswith("NANOSLEEP")
                and per[i + 2][0].startswith("SYNCS.PHASECHK") and per[i + 3][0] == "BRA"):
            for j in range(i, i + 4 + (per[i + 4][0] == "BRA")):
                per[j] = ("WAIT." + per[j][0],) + per[j][1:]
    top = max((n for op, n, _ in per if kind_of(op, "") == "MMA"), default=0.0)
    allc, hot = defaultdict(float), defaultdict(float)
    for op, n, _ in per:
        allc[op] += n
        if n >= 0.5 * top:
            hot[op] += n
    return dict(allc), dict(hot), per


SCALE = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "s": 1e3, "second": 1e3,
         "Ghz": 1e3, "GHz": 1e3, "Mhz": 1.0, "MHz": 1.0, "Khz": 1e-3, "hz": 1e-6}


def raw_metrics(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """The values of an ncu --page raw --csv export (its last row) and their units (the row under the header)."""
    rows = table(path)
    if len(rows) < 2:
        return {}, {}
    return {k: num(v) for k, v in rows[-1].items() if k}, {k: v for k, v in rows[0].items() if k}


def first(raw: dict[str, float], *names: str) -> tuple[float, str]:
    return next(((raw[n], n) for n in names if raw.get(n, float("nan")) == raw.get(n, float("nan"))), (float("nan"), ""))


def by_class(counts: dict[str, float], tma: str) -> tuple[dict[str, float], dict[str, float]]:
    cls, unpriced = defaultdict(float), defaultdict(float)
    for op, n in counts.items():
        if op.startswith("WAIT."):
            cls["WAIT"] += n
            continue
        k = kind_of(op, tma)
        if k is None:
            unpriced[op] += n
            k = "NOROW"
        cls[k] += n
    return dict(cls), dict(unpriced)


def bound(per_mma: dict[str, float], census: dict[str, dict[str, float]], cycle_u: float) -> list[tuple[float, str]]:
    """Every row's time for the mix, SMSP-cycles a MMA, largest first."""
    tot = defaultdict(float)
    for k, n in per_mma.items():
        for row, u in (census.get(k) or census["NOROW"]).items():
            tot[row] += n * u / cycle_u
    return sorted(((v, r) for r, v in tot.items()), reverse=True)


def pick(m: dict[str, float], *pats: str) -> dict[str, float]:
    return {k: v for k, v in m.items() if any(re.search(p, k) for p in pats) and v == v}


def analyse(rep: Path, census: dict, cycle_u: float, tma: str) -> dict | None:
    """One report's profile. `bound` prices the executed mix with WAIT (the spin) as NOROW; `bound_work` leaves it out."""
    b = str(rep)[: -len(".ncu-rep")] if str(rep).endswith(".ncu-rep") else str(rep)
    allc, hot, per = sass_counts(Path(b + ".sass.csv"))
    raw, units = raw_metrics(Path(b + ".raw.csv"))
    if not allc:
        return None
    out = {"report": Path(b).name}
    for scope, counts in (("kernel", allc), ("loop", hot)):
        cls, unp = by_class(counts, tma)
        mma = cls.get("MMA", 0.0)
        if not mma:
            continue
        pm = {k: v / mma for k, v in sorted(cls.items(), key=lambda x: -x[1])}
        rows = bound(pm, census, cycle_u)
        work = bound({k: v for k, v in pm.items() if k != "WAIT"}, census, cycle_u)
        out[scope] = {"mma": mma, "inst_per_mma": sum(counts.values()) / mma, "wait_per_mma": pm.get("WAIT", 0.0),
                      "per_mma": pm, "opcodes_per_mma": {k: v / mma for k, v in sorted(counts.items(), key=lambda x: -x[1])},
                      "unpriced": {k: v / mma for k, v in unp.items()}, "bound": rows[0][0], "facet": rows[0][1],
                      "facets": rows[:8], "bound_work": work[0][0], "facet_work": work[0][1], "facets_work": work[:8]}
    nsm = raw.get("device__attribute_multiprocessor_count", float("nan"))
    el, _ = first(raw, "sm__cycles_elapsed.avg", "smsp__cycles_elapsed.avg", "l1tex__cycles_elapsed.avg")
    act, _ = first(raw, "sm__cycles_active.avg")
    k = out.get("kernel")
    if k:
        mps = k["mma"] / (4 * nsm)   # MMAs a SMSP
        out["mma_per_smsp"] = mps
        out["measured"] = el / mps
        out["measured_active"] = act / mps
        out["eps_sched"] = 1 - k["bound"] / out["measured"]
        out["eps_sched_active"] = 1 - k["bound"] / out["measured_active"]
        out["eps_sched_work"] = 1 - k["bound_work"] / out["measured"]
        if "loop" in out:
            out["eps_sched_loop_bound"] = 1 - out["loop"]["bound"] / out["measured"]
    t, tn = first(raw, "gpu__time_duration.sum")
    out["ms"] = t * SCALE.get(units.get(tn, "ns"), float("nan"))
    hz, hn = first(raw, "sm__cycles_elapsed.avg.per_second", "smsp__cycles_elapsed.avg.per_second",
                   "gpc__cycles_elapsed.avg.per_second")
    out["mhz"] = hz * SCALE.get(units.get(hn, "hz"), float("nan"))
    out["sm_count"] = nsm
    out["issue_pct"] = raw.get("smsp__issue_active.avg.pct_of_peak_sustained_active", float("nan"))
    out["tensor_pct"], _ = first(raw, "sm__inst_executed_pipe_tensor_subpipe_imma.avg.pct_of_peak_sustained_active",
                                 "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_active")
    out["pipes_pct"] = pick(raw, r"^sm__(inst_executed_)?pipe_.*\.avg\.pct_of_peak_sustained_active$",
                            r"^sm__pipe_.*cycles_active\.avg\.pct_of_peak_sustained_active$")
    out["pipe_inst"] = pick(raw, r"^sm__inst_executed_pipe_[a-z0-9_]+\.sum$")
    stalls = pick(raw, r"^smsp__average_warps?_(latency_)?issue_stalled_.*_per_issue_active\.ratio$")
    if not stalls:
        stalls = pick(raw, r"^smsp__pcsamp_warps_issue_stalled_[a-z_]+$")
    out["stalls"] = dict(sorted(stalls.items(), key=lambda x: -x[1]))
    samp = defaultdict(float)
    for op, _, s in per:
        samp[op.split(".")[0]] += s
    tot = sum(samp.values()) or 1.0
    out["stall_samples_by_opcode"] = {k: v / tot for k, v in sorted(samp.items(), key=lambda x: -x[1])[:12]}
    out["warps_eligible"] = raw.get("smsp__warps_eligible.avg.per_cycle_active", float("nan"))
    return out


def bench(o: Path) -> dict[tuple[int, int, int], dict]:
    out = {}
    p = o / "bench.jsonl"
    for line in (p.read_text().splitlines() if p.exists() else []):
        if line.startswith("{"):
            b = json.loads(line)
            out[(b["m"], b["n"], b["k"])] = b["arms"]
    return out


def run_shapes(o: Path, names: set[str]) -> dict[str, tuple[int, int, int]]:
    """Each profiled shape's (m, n, k) by its name in run.sh, from the gate's meta (gate_NAME_meta.json), else SHAPES."""
    out = {}
    for name in ["o"] + sorted(names - {"o"}):
        meta = o / f"gate_{name}_meta.json"
        if meta.exists():
            g = json.loads(meta.read_text())
            out[name] = (g["m"], g["n"], g["k"])
        elif name in SHAPES:
            out[name] = SHAPES[name]
    return out


def f(x, d=3):
    return "" if x is None or x != x else f"{x:.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--mm", type=Path, required=True)
    ap.add_argument("--epoch", default="")
    ap.add_argument("--tma", default="UTMA16K", help="the class of the mainloop's TMA box (MLW: 128 x 128 bytes)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--toml", type=Path)
    ap.add_argument("--toml-arms", default=f"{HONEST},e_l4q,e_plain,plain", help="the arms --toml writes, comma-separated")
    a = ap.parse_args()
    sys.path.insert(0, str(a.mm))
    from verity_machine_model import model as mmod
    data = a.mm / "data"
    m = mmod.model(data)
    epoch = a.epoch or mmod.vector(m.current, data).epoch
    census = mmod.census(m, epoch, data)
    res: dict[str, dict] = {}
    for rep in sorted((a.out / "ncu").glob("*.sass.csv")):
        name = rep.name[: -len(".sass.csv")]
        r = analyse(Path(str(rep)[: -len(".sass.csv")] + ".ncu-rep"), census, m.cycle_u, a.tma)
        if r:
            res[name] = r
    hot = bench(a.out)
    shapes = run_shapes(a.out, {n.split("-", 1)[0] for n in res})
    print(f"# EPS1 PROFILE: the edited GEMM against the machine model ({a.out.name}, epoch {epoch})\n")
    for sh, shape in shapes.items():
        arms = hot.get(shape, {})
        pl, ep = arms.get("plain", {}).get("ms"), arms.get("e_plain", {}).get("ms")
        print(f"## {sh} = {shape[0]} x {shape[1]} x {shape[2]}\n")
        print("Cycles are SMSP-cycles a MMA; `ncu` is the one profiled call's elapsed cycles, `hot` the bench's hot median "
              "at its clock. `bound` prices the executed mix, `work` leaves out WAIT (the mbarrier spin).\n")
        print("| arm | r | hot ms | / e_plain | / plain | MHz | inst/MMA | WAIT/MMA | fold ops/word | issue % | tensor % | "
              "bound | facet | work bound | facet | ncu | hot | eps_sched | eps_sched (work) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        base = res.get(f"{sh}-e_plain", {}).get("kernel", {})
        base = base["inst_per_mma"] - base["wait_per_mma"] if base else None
        rows = [(None, "plain")] + SWEEP + [(3.0, "e_l4q"), (1.5, "x_l2o")]
        seen = {arm for _, arm in rows}
        rows += [(3.0 if arm.startswith("e_l") else None, arm) for arm in
                 sorted({n.split("-", 1)[1] for n in res if n.startswith(sh + "-")} - seen)]
        for rr, arm in rows:
            if f"{sh}-{arm}" not in res and arm not in arms:
                continue
            r = res.get(f"{sh}-{arm}", {})
            k = r.get("kernel", {})
            hb = arms.get(arm, {})
            ms, mhz = hb.get("ms"), hb.get("mhz")
            if k and ms and mhz:
                r["measured_hot"] = ms * mhz * 1e3 / r["mma_per_smsp"]
                r["eps_sched_hot"] = 1 - k["bound"] / r["measured_hot"]
                r["eps_sched_work_hot"] = 1 - k["bound_work"] / r["measured_hot"]
            fo = (k["inst_per_mma"] - k["wait_per_mma"] - base) / 4 if k and base else None
            print(f"| {arm} | {'' if rr is None else rr} | {f(ms, 4)} | {f(ms / ep if ms and ep else None)} | "
                  f"{f(ms / pl if ms and pl else None)} | {f(mhz, 0)} | {f(k.get('inst_per_mma'), 2)} | "
                  f"{f(k.get('wait_per_mma'), 2)} | {f(fo, 2)} | {f(r.get('issue_pct'), 1)} | {f(r.get('tensor_pct'), 1)} | "
                  f"{f(k.get('bound'), 2)} | {k.get('facet', '')} | {f(k.get('bound_work'), 2)} | {k.get('facet_work', '')} | "
                  f"{f(r.get('measured'), 2)} | {f(r.get('measured_hot'), 2)} | {f(r.get('eps_sched_hot'))} | "
                  f"{f(r.get('eps_sched_work_hot'))} |")
        print()
        for arm in ("plain", "e_plain", HONEST):
            r = res.get(f"{sh}-{arm}")
            if not r:
                continue
            k = r["kernel"]
            print(f"### {arm} at {sh}: {f(k['inst_per_mma'], 2)} instructions a MMA, {r['ms']:.4f} ms under ncu at "
                  f"{r['mhz']:.0f} MHz, issue {f(r['issue_pct'], 1)}%, warps eligible {f(r['warps_eligible'], 2)} a cycle\n")
            print("classes a MMA (kernel / hot loop): " + ", ".join(
                f"{c} {v:.3f}" + (f" / {r['loop']['per_mma'].get(c, 0):.3f}" if "loop" in r else "") for c, v in k["per_mma"].items()))
            print("\nopcodes a MMA: " + ", ".join(f"{o} {v:.3f}" for o, v in list(k["opcodes_per_mma"].items())[:24]))
            if k["unpriced"]:
                print("\nunpriced (as NOROW): " + ", ".join(f"{o} {v:.3f}" for o, v in k["unpriced"].items()))
            print("\nfacets (SMSP-cycles a MMA): " + ", ".join(f"{row} {v:.2f}" for v, row in k["facets"]))
            print("\nfacets without WAIT: " + ", ".join(f"{row} {v:.2f}" for v, row in k["facets_work"]))
            print("\npipes %: " + ", ".join(f"{p.replace('sm__', '').replace('.avg.pct_of_peak_sustained_active', '')} {v:.1f}"
                                            for p, v in sorted(r["pipes_pct"].items(), key=lambda x: -x[1]) if v >= 0.5))
            print("\nstalls (per issue): " + ", ".join(
                f"{re.sub(r'^smsp__(average_warps?_(latency_)?issue_stalled_|pcsamp_warps_issue_stalled_)|_per_issue_active.ratio$', '', s)} {v:.2f}"
                for s, v in list(r["stalls"].items())[:10]))
            print("\nstall samples by opcode: " + ", ".join(f"{o} {v:.2f}" for o, v in r["stall_samples_by_opcode"].items()))
            print()
    if a.json:
        a.json.write_text(json.dumps({"epoch": epoch, "reports": res, "hot": {f"{k[0]}x{k[1]}x{k[2]}": v for k, v in hot.items()}},
                                     indent=1, default=str))
    if a.toml:
        lines = [f"# the edited GEMM's executed SASS a MMA (m16n8k32 s8, the peel's included) by census class, from ncu; "
                 f"cycles are SMSP-cycles a MMA, bounds over census {epoch}",
                 "# WAIT: the mbarrier retry loop (YIELD or NANOSLEEP, TRYWAIT, BRA), NOROW-priced in bound, left out of "
                 "work_bound; "
                 "measured: the hot median at its clock", f'run_dir = "{a.out.parent.name}"', ""]
        for sh, shape in shapes.items():
            for arm in a.toml_arms.split(","):
                r = res.get(f"{sh}-{arm}")
                if not r:
                    continue
                k = r["kernel"]
                lines += [f'[profile.{arm}."{shape[0]}x{shape[1]}x{shape[2]}"]',
                          f"inst_per_mma = {k['inst_per_mma']:.4f}", f"wait_per_mma = {k['wait_per_mma']:.4f}",
                          f"measured_cycles_per_mma = {r.get('measured_hot', r['measured']):.4f}",
                          f"ncu_cycles_per_mma = {r['measured']:.4f}",
                          f"bound_cycles_per_mma = {k['bound']:.4f}", f'facet = "{k["facet"]}"',
                          f"work_bound_cycles_per_mma = {k['bound_work']:.4f}", f'work_facet = "{k["facet_work"]}"',
                          f"eps_sched = {r.get('eps_sched_hot', r['eps_sched']):.4f}",
                          f"eps_sched_work = {r.get('eps_sched_work_hot', r['eps_sched_work']):.4f}",
                          f"issue_pct = {r['issue_pct']:.2f}", f"tensor_pct = {r['tensor_pct']:.2f}",
                          "per_mma = { " + ", ".join(f'{c} = {v:.4f}' for c, v in k["per_mma"].items()) + " }"]
                if "loop" in r:
                    lines.append("loop_per_mma = { " + ", ".join(f'{c} = {v:.4f}' for c, v in r["loop"]["per_mma"].items()) + " }")
                lines.append("")
        a.toml.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
