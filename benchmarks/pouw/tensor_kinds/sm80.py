"""sm_80's tensor table, joint issue coefficients, scalar pipes and duo, from a run_sm80.sh run.

    python sm80.py gate OUT BIN...   # gpu_kit's timing gate on every timed kernel -> OUT/gate.json, OUT/loops.json
    python sm80.py analyze OUT       # OUT/g*.jsonl with the gate's records -> OUT/sm80.json, OUT/sm80.md
    python sm80.py census OUT... --extension OUT... [--to DIR]   # analyzed runs -> the machine model's rows.csv and
                                                                 # census.csv (machine_model/data/devices/sm_80)

Every case of sm80.cu prints its kernel's template (`kern`, `struct`, `targs`), so its mangled name, and so its SASS
loops, follow from the line. `gate` runs gpu_kit's timing gate (`../gpu_kit/sass_gate.py`: clock reads, the timed
loops' exact opcode counts, no LDL/STL, a live sink) on each kernel with the counts its template implies, before the
lease, and records each timed loop's opcode counts. `analyze` reports no number for a kernel the gate refused or whose
0xA5 sink gate failed (`sink_gate` in its line), and names it.

Per SMSP, over the timed window of T cycles: a case's MMAs are its MMA warps a SMSP x iterations x PTX MMAs an
iteration, its other instructions every non-MMA SASS instruction its loops issue. On the issue facet a.#MMA + #other <=
T, each case bounds a <= (T - #other)/#MMA; a form's coefficient `a` is the least bound over every program that holds
its MMA, whatever runs beside it, since a facet holds on every program. `a_mix` and `a_iadd3` are the least over the
IADD3+FFMA and the IADD3 programs. The IADD3 lane binds on its pipe (half the issue rate on sm_80), so its programs also
bound the pipe facet a' + #IADD3/x_IADD3 <= T, x_IADD3 its rate alone.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOISE = 0.02
LANE_MIN = 0.9
REF = "ImmaS8k32"
MMA = re.compile(r"^(IMMA|HMMA|BMMA|DMMA)")
#: lane op codes (sm80.cu's OP_*) -> (name, the SASS opcode pattern the gate counts)
OPS = {0: ("iadd3", "IADD3"), 1: ("ffma", "FFMA"), 2: ("lop3", "LOP3*"), 3: ("imad", "IMAD"), 4: ("dp4a", "IDP.4A*"),
       5: ("shfl", "SHFL.BFLY*"), 6: ("popc", "POPC"), 7: ("lds", "LDS*"), 8: ("iadd3r", "IADD3"), 9: ("lop3i", "LOP3*"),
       10: ("ldsm4", "LDSM.16.M88.4"), 11: ("ldsm2", "LDSM.16.M88.2"), 12: ("ldsm1", "LDSM.16.M88*"),
       13: ("lds128", "LDS.128")}
OP_NONE = 15
#: SASS MMAs a PTX mma.sync, where not 1
SASS_PER_PTX: dict[str, int] = {}
#: each form: SASS family, its machine-model kind, its exact-word reading, and the s8 x s8 limb products one 32-step
#: committed word needs on it (s4 holds 4-bit limbs: 2 x 2; b1 holds bit planes: 7 x 7 magnitude bits, sign apart)
FORMS = {
    "ImmaS8k32": ("IMMA", "IMMA16832", "s32, wraps mod 2^32", 1),
    "ImmaU8S8k32": ("IMMA", "IMMA16832", "s32, wraps mod 2^32", 1),
    "ImmaU8U8k32": ("IMMA", "IMMA16832", "s32, wraps mod 2^32", 1),
    "ImmaS8k32Sat": ("IMMA", "IMMA16832", "s32, saturates (.satfinite)", 1),
    "ImmaS8k16": ("IMMA", "IMMA16816", "s32, wraps mod 2^32", 1),
    "ImmaS8m8k16": ("IMMA", "IMMA8816", "s32, wraps mod 2^32", 1),
    "ImmaS4k64": ("IMMA", "IMMA16864S4", "s32, wraps mod 2^32", 4),
    "ImmaU4k64": ("IMMA", "IMMA16864S4", "s32, wraps mod 2^32", 4),
    "ImmaS4k32": ("IMMA", "IMMA16832S4", "s32, wraps mod 2^32", 4),
    "ImmaS4m8k32": ("IMMA", "IMMA8832S4", "s32, wraps mod 2^32", 4),
    "BmmaAndK256": ("BMMA", "BMMA168256", "s32 popcount, wraps", 49),
    "BmmaXorK256": ("BMMA", "BMMA168256", "s32 popcount, wraps", 49),
    "BmmaAndK128": ("BMMA", "BMMA168128", "s32 popcount, wraps", 49),
    "BmmaAndM8K128": ("BMMA", "BMMA88128", "s32 popcount, wraps", 49),
    "ImmaSpS8k64": ("IMMA", "IMMASP16864", "s32, wraps mod 2^32", 1),
    "ImmaSpS8k32": ("IMMA", "IMMASP16832", "s32, wraps mod 2^32", 1),
    "ImmaSpS4k128": ("IMMA", "IMMASP168128S4", "s32, wraps mod 2^32", 4),
    "ImmaSpS4k64": ("IMMA", "IMMASP16864S4", "s32, wraps mod 2^32", 4),
    "HmmaF16k16F32": ("HMMA", "HMMA16816", "FP32 word; >= 1 op to extract", 1),
    "HmmaBf16k16F32": ("HMMA", "HMMA16816", "FP32 word; >= 1 op to extract", 1),
    "HmmaF16k16F16": ("HMMA", "HMMA16816F16", "f16 pairs: no 32-bit word", None),
    "HmmaF16k8F32": ("HMMA", "HMMA1688", "FP32 word; >= 1 op to extract", 1),
    "HmmaBf16k8F32": ("HMMA", "HMMA1688", "FP32 word; >= 1 op to extract", 1),
    "HmmaF16k8F16": ("HMMA", "HMMA1688F16", "f16 pairs: no 32-bit word", None),
    "HmmaTf32k8": ("HMMA", "HMMA1688T", "FP32 word; >= 1 op to extract", 1),
    "HmmaTf32k4": ("HMMA", "HMMA1684T", "FP32 word; >= 1 op to extract", 1),
    "HmmaSpF16k32": ("HMMA", "HMMASP16832", "FP32 word; >= 1 op to extract", 1),
    "HmmaSpF16k16": ("HMMA", "HMMASP16816", "FP32 word; >= 1 op to extract", 1),
    "HmmaSpTf32k16": ("HMMA", "HMMASP16816T", "FP32 word; >= 1 op to extract", 1),
    "DmmaF64": ("DMMA", "DMMA884", "FP64 word in 2 slots (exact integers < 2^53); >= 1 op to extract", 1),
}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mangle(kern: str, struct: str, targs: list[int]) -> str:
    return (f"_Z{len(kern)}{kern}I" + (f"{len(struct)}{struct}" if struct else "")
            + "".join(f"Li{v}E" for v in targs) + "EviPjP8BlockClk")


def demangle(name: str) -> tuple[str, str, list[int]] | None:
    """(kern, struct, targs) of a timed kernel's mangled name; None for any other function."""
    m = re.match(r"_Z(\d+)", name)
    if not m:
        return None
    n = int(m.group(1))
    kern, rest = name[m.end():m.end() + n], name[m.end() + n:]
    if not kern.startswith("k_") or kern == "k_exact" or not rest.startswith("I"):
        return None
    rest = rest[1:]
    struct = ""
    s = re.match(r"(\d+)", rest)
    if s:
        k = int(s.group(1))
        struct, rest = rest[s.end():s.end() + k], rest[s.end() + k:]
    targs = []
    while (t := re.match(r"Li(\d+)E", rest)):
        targs.append(int(t.group(1)))
        rest = rest[t.end():]
    return (kern, struct, targs) if rest == "EviPjP8BlockClk" else None


def lane_ops(x: int) -> list[int]:
    p, q = x & 15, x >> 4
    return [p] if q == OP_NONE else [p, q]


def lane_counts(x: int, n: int) -> dict[str, int]:
    """The opcode counts n lane ops of program x put in a loop, the loop counter included.

    The counter is an IADD3, or a UIADD3 when ptxas moves it to the uniform datapath (at 8 warps a SMSP, 64 registers),
    so a lane with IADD3 counts both (`*IADD3`)."""
    ops = lane_ops(x)
    out: dict[str, int] = {}
    for p in ops:
        pat = OPS[p][1]
        out[pat] = out.get(pat, 0) + n // len(ops)
    if "IADD3" in out:
        out["*IADD3"] = out.pop("IADD3") + 1
    return out


def mixu_u(r: int) -> int:
    """sm80.cu's mixu_u: k_mixu's groups of 8 MMAs an iteration at R lane ops a group, where the sweep doesn't set U."""
    return max(1, min(8, 192 // (8 + r)))


def expect(kern: str, struct: str, targs: list[int]) -> list[dict[str, int]]:
    """Each timed loop's opcode counts, as the kernel's template sets them."""
    fam = f"{FORMS[struct][0]}*" if struct else None
    s = SASS_PER_PTX.get(struct, 1)
    if kern == "k_opalone":
        return [lane_counts(targs[0], 64)]
    if kern == "k_alone":
        return [{fam: 8 * (targs[1] if len(targs) > 1 else 1) * s}]
    if kern == "k_duo":
        return [{fam: 8 * s}]
    if kern == "k_duofold":
        return [{"IMAD": 32}]
    if kern == "k_spec":
        R, x = targs
        return [{fam: 8 * s}, lane_counts(x, R)]
    if kern == "k_mixu":
        R, x = targs[:2]
        u = targs[3] if len(targs) > 3 else mixu_u(R)
        return [{fam: 8 * u * s, **lane_counts(x, u * R)}]
    raise ValueError(kern)


def gate(out: Path, bins: list[Path]) -> int:
    G = _load("gpu_kit_sass_gate", HERE.parent / "gpu_kit" / "sass_gate.py")
    verdicts, loops_rec, refused = {}, {}, 0
    for b in bins:
        g = re.search(r"_g(\d+)$", b.name).group(1)
        for fn in G.functions(b):
            k = demangle(fn.name)
            if not k:
                continue
            key = f"g{g}:{fn.name}"
            rec = G.judge_kernel(fn, expect(*k))
            verdicts[key] = {"ok": rec["ok"], "reasons": rec["reasons"], "sass_sha256": rec["sass_sha256"],
                             "loops": rec["loops"], "sink": rec["sink"]}
            clocks = G.clock_reads(fn)
            lo, hi = (clocks[0], clocks[-1]) if len(clocks) >= 2 else (0, len(fn.insns))
            loops_rec[key] = [dict(Counter(ins.name for ins in fn.insns[j:t + 1] if ins.op[0] != "NOP"))
                              for j, t in G.loops(fn) if lo < j and t < hi and not G.elect_loop(fn, j, t)]
            refused += not rec["ok"]
            if not rec["ok"]:
                print(f"REFUSED {key}: " + "; ".join(rec["reasons"]))
    (out / "gate.json").write_text(json.dumps({"version": G.VERSION, "kernels": verdicts}, indent=1) + "\n")
    (out / "loops.json").write_text(json.dumps(loops_rec, indent=1) + "\n")
    print(f"gate: {len(verdicts) - refused} of {len(verdicts)} kernels pass")
    return 0 if verdicts else 8


def _split(loop: dict[str, int]) -> tuple[int, int]:
    """(SASS MMAs, other instructions) of a loop's opcode counts."""
    mma = sum(c for o, c in loop.items() if MMA.match(o))
    return mma, sum(loop.values()) - mma


def _count(loop: dict[str, int], pat: str) -> int:
    from fnmatch import fnmatchcase
    return sum(c for o, c in loop.items() if fnmatchcase(o, pat))


def cases(out: Path) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """(timed cases with their counts, exactness lines, device lines, exclusions) of a run."""
    gate_k = json.loads((out / "gate.json").read_text())["kernels"]
    loops = json.loads((out / "loops.json").read_text())
    timed, exact, heads, excluded = [], [], [], []
    for f in sorted(out.glob("g*.jsonl")):
        g = re.match(r"g(\d+)\.jsonl", f.name).group(1)
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
        if not any(r.get("done") for r in rows):
            excluded.append(f"group {g}: no done line (the binary did not finish)")
        for r in rows:
            if r.get("mode") == "exact":
                exact.append(r)
                continue
            if "kern" not in r:
                if "device" in r:
                    heads.append(r)
                continue
            name = mangle(r["kern"], r["struct"], r["targs"])
            key = f"g{g}:{name}"
            v = gate_k.get(key)
            why = ("no SASS for the kernel" if v is None else "; ".join(v["reasons"]) if not v["ok"]
                   else "" if r["sink_gate"] else "the 0xA5 sink gate failed")
            if why:
                excluded.append(f"{key} ({r['mode']} {r['struct']} {r['targs']}): {why}")
                continue
            L = loops[key]
            T = r["cyc_med"]
            it = r["iters"]
            if r["kern"] == "k_opalone":
                (lp,) = L
                r.update(group=g, kernel=name, T=T, loop=lp, instr_per_smsp_cycle=8 * it * sum(lp.values()) / T,
                         ops_per_smsp_cycle=8 * it * 64 / T)
                r["per_op"] = {OPS[p][0]: 8 * it * _count(lp, OPS[p][1]) / T for p in lane_ops(r["targs"][0])}
                timed.append(r)
                continue
            if r["kern"] == "k_duofold":
                (lp,) = L
                nw = r["targs"][0]
                r.update(group=g, kernel=name, T=T, loop=lp, words_per_smsp_cycle=nw * it * 1024 / T,
                         instr_per_word=(sum(lp.values()) - 2) / 32, smsp_cycles_per_32_words=T / (nw * it * 32))
                timed.append(r)
                continue
            mma_l = [x for x in L if _split(x)[0]]
            lane_l = [x for x in L if not _split(x)[0]]
            want_lane = 1 if r["kern"] == "k_spec" else 0
            if len(mma_l) != 1 or len(lane_l) != want_lane:
                excluded.append(f"{key}: loops {L}")
                continue
            sass_mma, other_m = _split(mma_l[0])
            other_l = sum(lane_l[0].values()) if lane_l else 0
            n_mma = r["mma_warps_smsp"] * it * r["mma_ptx_iter"]
            n_other = r["mma_warps_smsp"] * it * other_m + r["lane_warps_smsp"] * it * other_l
            n_iadd3 = (r["mma_warps_smsp"] * it * _count(mma_l[0], "IADD3")
                       + r["lane_warps_smsp"] * it * (_count(lane_l[0], "IADD3") if lane_l else 0))
            r.update(group=g, kernel=name, T=T, n_mma=n_mma, n_other=n_other, n_iadd3=n_iadd3,
                     sass_per_ptx=sass_mma / r["mma_ptx_iter"], mma_ops=sorted(o for o in mma_l[0] if MMA.match(o)),
                     cycles_per_mma=T / n_mma, other_per_mma=n_other / n_mma, a_bound=(T - n_other) / n_mma,
                     loops=L)
            timed.append(r)
    return timed, exact, heads, excluded


def lane_name(x: int) -> str:
    return "+".join(OPS[p][0] for p in lane_ops(x))


def analyze(out: Path) -> tuple[dict, list[str]]:
    timed, exact, heads, excluded = cases(out)
    fails = list(excluded)
    lanes = {}
    for r in (r for r in timed if r["kern"] == "k_opalone"):
        lanes[lane_name(r["targs"][0])] = {"instr_per_smsp_cycle": r["instr_per_smsp_cycle"],
                                           "ops_per_smsp_cycle": r["ops_per_smsp_cycle"], "per_op": r["per_op"],
                                           "warp_instr_per_sm_clk": 4 * r["instr_per_smsp_cycle"],
                                           "mhz": r["mhz"], "nvml_mhz": [r["nvml_mhz_before"], r["nvml_mhz_after"]],
                                           "loop": r["loop"]}
    x_iadd3 = lanes.get("iadd3", {}).get("per_op", {}).get("iadd3")
    mix = lanes.get("iadd3+ffma", {}).get("instr_per_smsp_cycle")
    if mix is None:
        fails.append("opalone: no iadd3+ffma row")
    elif mix < LANE_MIN:
        fails.append(f"opalone: the IADD3+FFMA lane issues {mix:.3f} a SMSP-cycle alone, below {LANE_MIN}")
    forms: dict[str, dict] = {}
    by_form: dict[str, list[dict]] = defaultdict(list)
    for r in timed:
        if r["struct"]:
            by_form[r["struct"]].append(r)
    for s, pts in by_form.items():
        fam, kind, word, limbs = FORMS[s]
        alone = [p for p in pts if p["kern"] == "k_alone"]
        mixed = [p for p in pts if p["kern"] in ("k_spec", "k_mixu")]
        lane_pts = {lane: [p for p in mixed if p["lane"] == lane] for lane in (16, 240)}
        a = min((p["a_bound"] for p in pts if "a_bound" in p), default=None)
        a_mix = min((p["a_bound"] for p in lane_pts[16]), default=None)
        a_iadd3 = min((p["a_bound"] for p in lane_pts[240]), default=None)
        a_pipe = (min(((p["T"] - p["n_iadd3"] / x_iadd3) / p["n_mma"] for p in lane_pts[240]), default=None)
                  if x_iadd3 else None)
        c = min((p["cycles_per_mma"] for p in alone), default=None)
        p0 = pts[0]
        mhz = [p["mhz"] for p in pts]
        nv = [m for p in pts for m in (p["nvml_mhz_before"], p["nvml_mhz_after"]) if m]
        forms[s] = {
            "name": p0["form"], "kind": kind, "sass": sorted({o for p in pts if "mma_ops" in p for o in p["mma_ops"]}),
            "sass_per_ptx": sorted({p["sass_per_ptx"] for p in pts if "sass_per_ptx" in p}),
            "macs": p0["macs"], "slots": p0["slots"], "words": p0["words"], "word": word, "limb_products": limbs,
            "alone_smsp_cycles_per_mma": c,
            "alone_by_warps": {p["mode"]: p["cycles_per_mma"] for p in alone},
            "slots_per_sm_clk": 4 * p0["slots"] / c if c else None,
            "words_per_sm_clk": 4 * p0["words"] / c if c else None,
            "macs_per_sm_clk": 4 * p0["macs"] / c if c else None,
            "a": a, "a_mix": a_mix, "a_iadd3": a_iadd3, "a_iadd3_pipe": a_pipe,
            "programs": len(mixed), "mhz_clock64": [min(mhz), max(mhz)], "mhz_nvml": [min(nv), max(nv)] if nv else None,
            "points": [{k: p[k] for k in ("mode", "kern", "targs", "lane", "cycles_per_mma", "other_per_mma", "a_bound",
                                          "mhz", "nvml_mhz_before", "nvml_mhz_after", "group")} for p in mixed + alone],
        }
    ref = forms.get(REF, {})
    for s, f in forms.items():
        r0, v = ref.get("a"), f["a"]
        f["a_per_slot_vs_ref"] = (v / f["slots"]) / (r0 / ref["slots"]) if v and r0 else None
        depth = {"ImmaS8k32": 32, "ImmaU8S8k32": 32, "ImmaU8U8k32": 32, "ImmaS8k32Sat": 32, "ImmaS8k16": 16,
                 "ImmaS8m8k16": 16}.get(s)
        f["depth"] = depth or None
    rho = max((f["slots_per_sm_clk"] for f in forms.values() if f["slots_per_sm_clk"]), default=None)
    words_max = max((f["words_per_sm_clk"] for f in forms.values() if f["words_per_sm_clk"]), default=None)
    # duo
    duofold = {f"nw{r['targs'][0]}": {"words_per_smsp_cycle": r["words_per_smsp_cycle"],
                                      "smsp_cycles_per_32_words": r["smsp_cycles_per_32_words"],
                                      "instr_per_word": r["instr_per_word"], "loop": r["loop"], "mhz": r["mhz"]}
               for r in timed if r["kern"] == "k_duofold"}
    duo = {}
    for r in (r for r in timed if r["kern"] == "k_duo"):
        f = forms.get(r["struct"], {})
        c0 = f.get("alone_smsp_cycles_per_mma")
        duo[f"{r['struct']}/nw{r['targs'][0]}"] = {
            "cycles_per_mma": r["cycles_per_mma"], "alone_cycles_per_mma": c0,
            "words_per_sm_clk": 4 * r["words"] / r["cycles_per_mma"],
            "fold_smsp_cycles_per_32_words_beside": (r["cycles_per_mma"] - c0) / (r["words"] / 32) if c0 else None,
            "other_per_mma": r["other_per_mma"], "loops": r["loops"], "mhz": r["mhz"],
            "nvml_mhz": [r["nvml_mhz_before"], r["nvml_mhz_after"]]}
    # a scalar class beside a saturated MMA (spec, R = 64): the summed load of its two programs alone
    beside = {}
    for r in (r for r in timed if r["kern"] == "k_spec" and r["targs"][0] == 64 and r["lane"] not in (16, 240)):
        f = forms.get(r["struct"], {})
        ln = lanes.get(lane_name(r["lane"]))
        c0 = f.get("alone_smsp_cycles_per_mma")
        if not ln or not c0:
            continue
        n_lane = 2 * r["iters"] * 64
        load = r["n_mma"] * c0 / r["T"] + n_lane / (ln["ops_per_smsp_cycle"] * r["T"])
        beside[f"{r['struct']}+{lane_name(r['lane'])}"] = {
            "cycles_per_mma": r["cycles_per_mma"], "alone_cycles_per_mma": c0, "lane_alone_ops_per_smsp_cycle":
            ln["ops_per_smsp_cycle"], "summed_load": load, "a_bound": r["a_bound"], "mhz": r["mhz"]}
    exact_by: dict[str, dict] = defaultdict(lambda: {"cases": 0, "wrong": 0, "wrapped": 0, "saturated": 0})
    for e in exact:
        x = exact_by[e["struct"]]
        x["cases"] += 1
        x["form"] = e["form"]
        x["semantics"] = e["semantics"]
        for k in ("wrong", "wrapped", "saturated"):
            x[k] += e[k]
    for s, x in exact_by.items():
        if x["wrong"]:
            fails.append(f"exact {s}: {x['wrong']} wrong elements")
    q = {}
    for s in ("ImmaS8k32", "ImmaS8k16", "ImmaS8m8k16"):
        f = forms.get(s)
        if f and f["a"] is not None:
            q[s] = {"a": f["a"], "words_per_lane": f["words"] / 32, "q0": f["a"] / (f["words"] / 32)}
    meta = out / "meta.json"
    summary = {"run_id": json.loads(meta.read_text()).get("run_id") if meta.exists() else None,
               "device": heads[:1], "lanes": lanes, "forms": forms, "rho_T_slots_per_sm_clk": rho,
               "words_per_sm_clk_max": words_max, "duofold": duofold, "duo": duo, "beside": beside,
               "exact": dict(exact_by), "qstar0": q, "excluded": excluded}
    return summary, fails


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def render(s: dict) -> str:
    L = [f"Run {s.get('run_id', '?')}; clocks read, never set (default boost).", "",
         "## Forms (alone: least SMSP-cycles a PTX MMA over 2, 4 and 8 warps a SMSP, 8, 32 and 64 MMAs a loop iteration)", "",
         "| form | SASS | slots | alone SMSP-cyc a MMA | slots/SM/clk | MACs/SM/clk | a, every program | a, IADD3+FFMA | "
         "a, IADD3 (issue) | a, IADD3 (pipe) | a a slot vs IMMA.16832 | word | MHz clock64 / NVML |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, f in s["forms"].items():
        nv = f["mhz_nvml"]
        nvs = "–" if not nv else f"{nv[0]}–{nv[1]}"
        L.append(f"| {f['name']} | {', '.join(f['sass'])} | {f['slots']:.0f} | {_f(f['alone_smsp_cycles_per_mma'], 2)} | "
                 f"{_f(f['slots_per_sm_clk'], 2)} | {_f(f['macs_per_sm_clk'], 0)} | {_f(f['a'])} | {_f(f['a_mix'])} | "
                 f"{_f(f['a_iadd3'])} | {_f(f['a_iadd3_pipe'])} | {_f(f['a_per_slot_vs_ref'])} | "
                 f"{f['word']} | {f['mhz_clock64'][0]:.0f}–{f['mhz_clock64'][1]:.0f} / {nvs} |")
    L += ["", f"ρ_T = {_f(s['rho_T_slots_per_sm_clk'], 2)} slots/SM/clk; most output words {_f(s['words_per_sm_clk_max'], 2)}/SM/clk.",
          "", "## Lanes alone (8 warps a SMSP, ILP 8)", "",
          "| lane | warp instr / SMSP-cycle (all) | per op | warp instr/SM/clk |", "|---|---|---|---|"]
    for k, v in s["lanes"].items():
        L.append(f"| {k} | {_f(v['instr_per_smsp_cycle'])} | "
                 + ", ".join(f"{o} {_f(x)}" for o, x in v["per_op"].items()) + f" | {_f(v['warp_instr_per_sm_clk'])} |")
    L += ["", "## Beside a saturated MMA (2 MMA + 2 lane warps a SMSP, 8 ops a MMA)", "",
          "| program | SMSP-cyc a MMA | MMA alone | lane alone ops/SMSP-cyc | summed load (1 = serial, 2 = overlap) |",
          "|---|---|---|---|---|"]
    for k, v in s["beside"].items():
        L.append(f"| {k} | {_f(v['cycles_per_mma'])} | {_f(v['alone_cycles_per_mma'])} | "
                 f"{_f(v['lane_alone_ops_per_smsp_cycle'])} | {_f(v['summed_load'])} |")
    L += ["", "## duo", "", "| program | SMSP-cyc a MMA | MMA alone | words/SM/clk | fold SMSP-cyc a 32 words beside |",
          "|---|---|---|---|---|"]
    for k, v in s["duo"].items():
        L.append(f"| {k} | {_f(v['cycles_per_mma'])} | {_f(v['alone_cycles_per_mma'])} | {_f(v['words_per_sm_clk'], 2)} | "
                 f"{_f(v['fold_smsp_cycles_per_32_words_beside'])} |")
    for k, v in s["duofold"].items():
        L.append(f"| duo alone {k} | – | – | {_f(4 * v['words_per_smsp_cycle'], 2)} | {_f(v['smsp_cycles_per_32_words'])} "
                 f"({_f(v['instr_per_word'], 2)} instr a word) |")
    L += ["", "## Exact s32 word (constant operands, random C near ±2^31)", "",
          "| form | semantics | cases | wrong | wrapped | saturated |", "|---|---|---|---|---|---|"]
    for k, v in s["exact"].items():
        L.append(f"| {v['form']} | {v['semantics']} | {v['cases']} | {v['wrong']} | {v['wrapped']} | {v['saturated']} |")
    L += ["", "## q*₀ (MMA issue a word, against one scalar instruction's issue slot)", ""]
    for k, v in s["qstar0"].items():
        L.append(f"- {k}: a = {_f(v['a'])} issue-cycles a MMA / {v['words_per_lane']:.0f} words a lane = "
                 f"**q*₀ {_f(v['q0'])}**")
    if s["excluded"]:
        L += ["", "## Excluded (no number reported)", ""] + [f"- {x}" for x in s["excluded"]]
    return "\n".join(L) + "\n"


#: the scalar census kinds (sm_120's names where one exists) -> (row, the lanes whose per-op rate prices it)
SCALAR = {"ALU": ("A", ["iadd3", "lop3"]), "FFMA": ("I", ["ffma"]), "IMAD": ("I", ["imad", "dp4a"]),
          "IDP4A": ("I", ["dp4a", "imad"]), "XU": ("XU", ["popc"]), "SHFL": ("SHFL", ["shfl"]),
          "LDS32": ("LDSi", ["lds"]), "LDS128": ("R6r", ["lds128"]), "LDSM_X1": ("LDSM_X1", ["ldsm1"]),
          "LDSM_X2": ("LDSM_X2", ["ldsm2"]), "LDSM_X4": ("LDSM_X4", ["ldsm4"])}
#: kinds with no pipe row, priced on the issue rows only: every instruction a profile counts that no row above holds
#: (branches, barriers, cp.async's LDGSTS, the uniform datapath, stores, special registers)
ISSUE_ONLY = ["NOROW"]
ROWS = {
    "issue": "1 warp instruction a SMSP-cycle, every kind (the best scalar issue: IADD3+FFMA alternating, {iadd3+ffma} a "
             "SMSP-cycle)",
    "M": "tensor pipe: each MMA kind alone, 4 / its least SMSP-cycles a MMA warp instructions/SM/clk + 2% (best of 2, 4 "
         "and 8 warps a SMSP and of 8, 32 and 64 MMAs a loop iteration, 8 accumulators a warp; the loop's 3 instructions "
         "included)",
    "A": "ALU pipe: IADD3 and LOP3 + 2% (IADD3+LOP3 issue {iadd3+lop3} a SMSP-cycle: one pipe)",
    "I": "IMAD pipe: IMAD, FFMA and IDP.4A + 2% (IMAD+FFMA {imad+ffma}, IDP.4A+FFMA {dp4a+ffma} a SMSP-cycle: one pipe "
         "on sm_80)",
    "XU": "XU: POPC + 2%",
    "SHFL": "SHFL.BFLY + 2%",
    "LDSi": "LDS.32 + 2% (one warp a SM-cycle: 128 B/SM/clk)",
    "R6r": "LDS.128 {lds128} warp instructions a SMSP-cycle alone + 2%, each lane's 16 bytes its own address",
    "LDSM_X1": "LDSM.16.M88 (ldmatrix x1) {ldsm1} warp instructions a SMSP-cycle alone + 2%",
    "LDSM_X2": "LDSM.16.M88.2 (ldmatrix x2) {ldsm2} warp instructions a SMSP-cycle alone + 2%",
    "LDSM_X4": "LDSM.16.M88.4 (ldmatrix x4) {ldsm4} warp instructions a SMSP-cycle alone + 2%",
    "JIM": "M1's issue facet: a(kind) #MMA + every other warp instruction 1 <= T SMSP-cycles, a the least (T - #other)/#MMA "
           "over every gated program that holds the kind's MMA (beside IADD3+FFMA, which issues {iadd3+ffma} a SMSP-cycle "
           "alone, beside IADD3 and the other lanes, duo, alone); every coefficient x 0.98 (2% noise) / {margin} "
           "(decision 76: the most the search's latest extension, {ext}, moved a kind's least)",
}


def census(outs: list[Path], ext: list[Path] = ()) -> tuple[str, str]:
    """sm_80's census (machine_model/data/devices/sm_80/'s rows.csv and census.csv: one epoch, no since or until) from
    analyzed runs: every kind at one warp instruction a SMSP-cycle on `issue`, each MMA kind's least alone cycles as a
    rate, each scalar class's fastest rate, both at the 2% noise margin, and each kind's least joint coefficient; a run
    whose lane gate failed gives no joint coefficient and doesn't stand for the issue row. The joint row takes a
    convergence margin: `ext` are the runs of the search's latest extension, and the row is divided by 1 + m, m the most
    they moved a kind's least below the other runs'. Past the noise the search hasn't converged, and the census
    refuses."""
    alone: dict[str, tuple[float, set]] = {}
    joint: dict[str, tuple[float, set]] = {}
    before: dict[str, tuple[float, set]] = {}
    rate: dict[str, tuple[float, set]] = {}
    pairs: dict[str, float] = {}
    ext_runs: set[str] = set()
    issue_runs: set[str] = set()

    def keep(d, k, v, run, better):
        if v is None:
            return
        old = d.get(k)
        if old is None or better(v, old[0]):
            d[k] = (v, {run})
        elif abs(v - old[0]) <= 1e-9 * abs(v):
            old[1].add(run)

    lt = lambda a, b: a < b
    for out, latest in [(p, False) for p in outs] + [(p, True) for p in ext]:
        rec = json.loads((out / "sm80.json").read_text())
        s, fails = rec["summary"], rec["gate_failures"]
        run = s.get("run_id") or out.name
        if latest:
            ext_runs.add(run)
        lane_ok = not any(f.startswith("opalone") for f in fails)
        if lane_ok and "iadd3+ffma" in s["lanes"]:
            issue_runs.add(run)
        for st, f in s["forms"].items():
            kind = FORMS[st][1]
            keep(alone, kind, f["alone_smsp_cycles_per_mma"], run, lt)
            if lane_ok:
                keep(joint, kind, f["a"], run, lt)
                if not latest:
                    keep(before, kind, f["a"], run, lt)
        for kind, (_, lanes) in SCALAR.items():
            for ln in lanes:
                v = s["lanes"].get(ln, {}).get("per_op", {}).get(ln)
                keep(rate, kind, None if v is None else 4 * v, run, lambda a, b: a > b)
        for ln, v in s["lanes"].items():
            pairs[ln] = max(pairs.get(ln, 0.0), v["instr_per_smsp_cycle"])
    lines, cites = [], defaultdict(set)
    fmt = {k: f"{v:.3f}" for k, v in pairs.items()}
    fmt.update({k: f"{rate[c][0] / 4:.3f}" for c, (_, lns) in SCALAR.items() for k in lns[:1] if c in rate})
    if issue_runs:
        for kind in sorted(alone) + list(SCALAR) + ISSUE_ONLY:
            lines.append(f"{kind},issue,,,1,,,,{' '.join(sorted(issue_runs))}")
        cites["issue"] |= issue_runs
    for kind, (v, runs) in sorted(alone.items()):
        lines.append(f"{kind},M,{4 / v:.6g},{NOISE},,,,,{' '.join(sorted(runs))}")
        cites["M"] |= runs
    for kind, (v, runs) in sorted(rate.items()):
        row = SCALAR[kind][0]
        lines.append(f"{kind},{row},{v:.6g},{NOISE},,,,,{' '.join(sorted(runs))}")
        cites[row] |= runs
    if joint:
        if not ext_runs:
            raise SystemExit("census: name the runs of the schedule search's latest extension (--extension OUT...); "
                             "decision 76 takes the joint row's margin from them")
        moved = {k: before[k][0] / v - 1 if k in before else math.inf for k, (v, _) in joint.items()}
        worst, m = max(moved.items(), key=lambda x: x[1])
        if m > NOISE:
            raise SystemExit(f"census: the search's latest extension moved {worst}'s a {m:.1%} below the earlier runs', "
                             f"past the {NOISE:.0%} noise: extend the search again (decision 76)")
        scale = (1 - NOISE) / (1 + m)
        for kind, (v, runs) in sorted(joint.items()):
            lines.append(f"{kind},JIM,,,{v * scale:.6g},,,,{' '.join(sorted(runs))}")
            cites["JIM"] |= runs
        cites["JIM"] |= ext_runs
        for other in list(SCALAR) + ISSUE_ONLY:
            lines.append(f"{other},JIM,,,{scale:.6g},,,,{' '.join(sorted(cites['JIM']))}")
        fmt.update(margin=f"{1 + m:.4f}", ext=" ".join(sorted(ext_runs)))
    rows = ["row,status,runs,description"]
    for row, desc in ROWS.items():
        if cites[row]:
            status = "joint" if row.startswith("JI") else "measured"
            rows.append(f'{row},{status},{" ".join(sorted(cites[row]))},"{desc.format_map(defaultdict(lambda: "?", fmt))}"')
    return "\n".join(rows) + "\n", "kind,row,rate,noise,cycles,u,since,until,runs\n" + "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if argv[1] == "gate":
        return gate(Path(argv[2]), [Path(a) for a in argv[3:]])
    if argv[1] == "analyze":
        out = Path(argv[2])
        summary, fails = analyze(out)
        (out / "sm80.json").write_text(json.dumps({"summary": summary, "gate_failures": fails}, indent=1) + "\n")
        md = render(summary)
        (out / "sm80.md").write_text(md)
        print(md)
        for x in fails:
            print("GATE", x)
        return 0
    if argv[1] == "census":
        to, outs, ext = None, [], []
        args = iter(argv[2:])
        dest = outs
        for a in args:
            if a == "--to":
                to = Path(next(args))
            elif a == "--extension":
                dest = ext
            else:
                dest.append(Path(a))
        rows_csv, census_csv = census(outs, ext)
        if to:
            to.mkdir(parents=True, exist_ok=True)
            (to / "rows.csv").write_text(rows_csv)
            (to / "census.csv").write_text(census_csv)
        print(rows_csv + "\n" + census_csv, end="")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
