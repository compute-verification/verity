"""M1 of EPS1 TENSOR: fit each MMA form's joint issue coefficient from a tensor_kinds run (sm_120, or sm_89 with
TK_ARCH=89: there each form also has an alone4 point, and JI3e, sm_120's rows, isn't loaded).

    python tensor_kinds.py OUT_DIR            # reads OUT_DIR/tensor_kinds.jsonl and OUT_DIR/sass/tensor_kinds.sass
    python tensor_kinds.py refit OUT_DIR...   # VF3n-16's JI3n rows from every program of these runs, as JSON

For every timed case it reads the kernel's loops in the SASS (a loop is the instructions from a label to the backward
branch that targets it), counts MMA and other instructions an iteration, and turns the run's per-SM counts into
per-SMSP counts over the timed window. On the facet a·#MMA + #other <= cycles (one issue a SMSP-cycle for every other
instruction, as JI3e_0), each case bounds a <= (cycles - #other) / #MMA; a form's coefficient is the least bound over
its cases. The form's issue price a slot is a / slots. The tensor theorem's issue form is tight when no form's price a
slot is below IMMA s8 m16n8k32's (2% noise margin). The fit counts only if the lane op (IADD3 and FFMA alternating)
alone issues at >= 0.9 a SMSP-cycle; a slower lane op binds the mixed points on its own rate and overstates a. Each
form's programs also load VF3e's joint rows JI3e_0/1/2, fitted beside IMMA s8 m16n8k32 and OMMA only; a load above
0.98 is a program the row says can't run that fast (marked **over**).

Writes OUT_DIR/tensor_kinds.json and OUT_DIR/tensor_kinds.md; exits 3 when a gate fails (the case is named).

`refit` writes the machine model's JI3n_0..2 (VF3n-16): each JI3e row's MMA coefficient refit over every program of the
runs (a program constrains a row whatever the lane gate says) and VF3e's, at the noise margin with the row's other
coefficient kept, then the whole row divided by SCHED, the schedule spread between builds of one program. MMA takes every
form's least; each form's kind (KIND) gets its own.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MMA = re.compile(r"^(IMMA|HMMA|QMMA|OMMA|DMMA|BMMA)")
REF = "imma_s8_m16n8k32"
NOISE = 0.02
#: The mixed points' lane op (IADD3 and FFMA alternating) alone must issue at least this many warp instructions a
#: SMSP-cycle, or the mixed points bind on a pipe's own rate instead of the issue facet and overstate a.
LANE_MIN = 0.9
LANES = ("iadd3", "ffma", "iadd3_ffma")
MIXED_LANE = "iadd3_ffma"
#: VF3e's joint issue rows (note:20261016T0751Z-draft-rs-vf3e) as the machine model's census holds them, fitted beside
#: IMMA s8 m16n8k32 and OMMA: SMSP-cycles a MMA and a other instruction. Every program of every form must load each at
#: most 1 - NOISE.
JI3E = {"JI3e_0": (8.115163161, 1.0), "JI3e_1": (4510.17215313632 / 300.08, 0.317063107),
        "JI3e_2": (4662.85212194032 / 300.08, 0.171692378)}
#: The schedule spread `refit` covers: one program's time varied up to 6.91% between builds (M1, programs within 5% of a
#: JI3e facet: r20261017-040022-5e4d against r20261017-041103-d35d and r20261017-041754-7630).
SCHED = 1.07
#: Each form's kind in the machine model (VF3n-16): one a SASS form.
KIND = {"imma_s8_m16n8k32": "IMMA16832", "imma_u8s8_m16n8k32": "IMMA16832", "imma_s8_m16n8k32_sat": "IMMA16832",
        "imma_s8_m16n8k16": "IMMA16816", "imma_s8_m8n8k16": "IMMA8816", "imma_sp_s8_m16n8k64": "IMMASP16864",
        "imma_sp_s8_m16n8k32": "IMMASP16832", "hmma_f16_m16n8k16_f32": "HMMA16816", "hmma_bf16_m16n8k16_f32": "HMMA16816",
        "hmma_f16_m16n8k16_f16": "HMMA16816F16", "hmma_sp_f16_m16n8k32_f32": "HMMASP16832",
        "hmma_tf32_m16n8k8": "HMMA1688T", "hmma_tf32_m16n8k4": "HMMA1684T", "qmma_e4m3_m16n8k32_f32": "QMMA16832",
        "qmma_e5m2_m16n8k32_f32": "QMMA16832", "qmma_e2m1_m16n8k32_f32": "QMMA16832",
        "qmma_sf_e4m3_m16n8k32_f32": "QMMASF16832", "qmma_sp_e4m3_m16n8k64_f32": "QMMASP16864",
        "omma_nvf4_m16n8k64": "OMMASF16864",
        # sm_89's other forms
        "imma_s4_m16n8k64": "IMMA16864S4", "imma_s4_m16n8k32": "IMMA16832S4", "imma_s4_m8n8k32": "IMMA8832S4",
        "bmma_and_m16n8k256": "BMMA168256", "bmma_xor_m16n8k256": "BMMA168256", "bmma_and_m16n8k128": "BMMA168128",
        "bmma_and_m8n8k128": "BMMA88128", "hmma_f16_m16n8k8_f32": "HMMA1688", "hmma_bf16_m16n8k8_f32": "HMMA1688",
        "hmma_f16_m16n8k8_f16": "HMMA1688F16", "qmma_e4m3_m16n8k16_f32": "QMMA16816",
        "qmma_e4m3_m16n8k16_f16": "QMMA16816F16", "qmma_e4m3_m16n8k32_f16": "QMMA16832F16",
        "imma_sp_s4_m16n8k64": "IMMASP16864S4", "imma_sp_s4_m16n8k128": "IMMASP168128S4",
        "hmma_sp_f16_m16n8k16_f32": "HMMASP16816", "hmma_sp_tf32_m16n8k16": "HMMASP16816T"}
#: The kinds VF3e's programs counted as MMA, so VF3e's coefficient bounds theirs too.
VF3E_KINDS = {"IMMA16832", "OMMASF16864"}
STRUCTS = {
    "ImmaS8k32": "imma_s8_m16n8k32",
    "ImmaU8S8k32": "imma_u8s8_m16n8k32",
    "ImmaS8k16": "imma_s8_m16n8k16",
    "ImmaS8m8k16": "imma_s8_m8n8k16",
    "ImmaSpS8k64": "imma_sp_s8_m16n8k64",
    "HmmaF16k16F32": "hmma_f16_m16n8k16_f32",
    "HmmaF16k16F16": "hmma_f16_m16n8k16_f16",
    "HmmaTf32k8": "hmma_tf32_m16n8k8",
    "QmmaE4m3k32": "qmma_e4m3_m16n8k32_f32",
    "OmmaNvf4k64": "omma_nvf4_m16n8k64",
    "ImmaS8k32Sat": "imma_s8_m16n8k32_sat",
    "ImmaSpS8k32": "imma_sp_s8_m16n8k32",
    "HmmaBf16k16F32": "hmma_bf16_m16n8k16_f32",
    "HmmaSpF16k32": "hmma_sp_f16_m16n8k32_f32",
    "HmmaTf32k4": "hmma_tf32_m16n8k4",
    "QmmaE5m2k32": "qmma_e5m2_m16n8k32_f32",
    "QmmaE2m1k32": "qmma_e2m1_m16n8k32_f32",
    "QmmaSfE4m3k32": "qmma_sf_e4m3_m16n8k32_f32",
    "QmmaSpE4m3k64": "qmma_sp_e4m3_m16n8k64_f32",
    # sm_89
    "ImmaS4k64": "imma_s4_m16n8k64",
    "ImmaS4k32": "imma_s4_m16n8k32",
    "ImmaS4m8k32": "imma_s4_m8n8k32",
    "BmmaAndK256": "bmma_and_m16n8k256",
    "BmmaXorK256": "bmma_xor_m16n8k256",
    "BmmaAndK128": "bmma_and_m16n8k128",
    "BmmaAndM8k128": "bmma_and_m8n8k128",
    "HmmaF16k8F32": "hmma_f16_m16n8k8_f32",
    "HmmaF16k8F16": "hmma_f16_m16n8k8_f16",
    "HmmaBf16k8F32": "hmma_bf16_m16n8k8_f32",
    "QmmaE4m3k16F32": "qmma_e4m3_m16n8k16_f32",
    "QmmaE4m3k16F16": "qmma_e4m3_m16n8k16_f16",
    "QmmaE4m3k32F16": "qmma_e4m3_m16n8k32_f16",
    "ImmaSpS4k64": "imma_sp_s4_m16n8k64",
    "ImmaSpS4k128": "imma_sp_s4_m16n8k128",
    "HmmaSpF16k16": "hmma_sp_f16_m16n8k16_f32",
    "HmmaSpTf32k16": "hmma_sp_tf32_m16n8k16",
    "QmmaSpE4m3k64Ada": "qmma_sp_e4m3_m16n8k64_f32",
}
#: PTX MMAs an iteration of the loop that issues them (k_alone, k_alone4 and k_spec: 8; k_mixu: 64).
PTX_MMAS = {"alone": 8, "alone4": 8, "spec": 8, "mixu": 64}
#: MMA warps a SMSP in each kernel's MMA loop (k_mixu: every warp; k_spec: half of them).
MMA_WARPS = {"alone": 2, "alone4": 4, "spec": 2, "mixu": 4}
#: The lane op of a k_spec/k_mixu template's third argument: the IADD3+FFMA stream or IADD3 alone.
LANE_MODE = {"2": "", "0": "_iadd3"}


def parse_sass(text: str) -> dict[str, list[tuple[int, str, str]]]:
    """Function name -> [(address, label or '', instruction)] in address order (cuobjdump -sass)."""
    funcs: dict[str, list[tuple[int, str, str]]] = {}
    cur = None
    pending = ""
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        lab = re.match(r"\s*(\.L_x_\d+):", line)
        if lab:
            pending = lab.group(1)
            continue
        ins = re.match(r"\s*/\*([0-9a-fA-F]{4,})\*/\s+(.*?)\s*;", line)
        if ins:
            funcs[cur].append((int(ins.group(1), 16), pending, ins.group(2)))
            pending = ""
    return funcs


def opcode(ins: str) -> str:
    toks = ins.split()
    if toks and toks[0].startswith("@"):
        toks = toks[1:]
    return toks[0] if toks else ""


def loops(body: list[tuple[int, str, str]]) -> list[dict]:
    """Each backward branch's loop (target a label or an address): its MMA opcodes and counts an iteration."""
    by_label = {lab: i for i, (_, lab, _) in enumerate(body) if lab}
    by_addr = {a: i for i, (a, _, _) in enumerate(body)}
    out = []
    for j, (_, _, ins) in enumerate(body):
        if opcode(ins).split(".")[0] != "BRA":
            continue
        m = re.search(r"`\((\.L_x_\d+)\)", ins)
        h = re.search(r"\b0x([0-9a-fA-F]+)\b", ins)
        i = by_label.get(m.group(1)) if m else (by_addr.get(int(h.group(1), 16)) if h else None)
        if i is None or i >= j:  # i == j: the idle `BRA self` after EXIT ending every function
            continue
        seg = [o for o in (opcode(x) for _, _, x in body[i : j + 1]) if o != "NOP"]
        mma = [o for o in seg if MMA.match(o)]
        # sm_89's ptxas moves part of an add.u32 stream to the FMA pipe as IMAD.IADD: an add all the same
        out.append({"mma": len(mma), "mma_ops": sorted(set(mma)), "other": len(seg) - len(mma),
                    "iadd3": sum(o.startswith(("IADD3", "IMAD.IADD")) for o in seg),
                    "ffma": sum(o.startswith("FFMA") for o in seg)})
    return out


def kernel_key(name: str) -> tuple[str, str, int] | None:
    """(mode, form, R) of a mangled kernel name, e.g. _Z6k_specI11QmmaE4m3k32Li32EEviPjP8BlockClk."""
    m = re.fullmatch(r"_Z9k_opaloneILi(\d)EEviPjP8BlockClk", name)
    if m:
        return "opalone", LANES[int(m.group(1))], 64
    m = re.match(r"_Z\d+(k_alone4|k_alone|k_spec|k_mixu)I(\d+)", name)
    if not m:
        return None
    n, rest = int(m.group(2)), name[m.end():]
    struct, tail = rest[:n], rest[n:]
    r = re.match(r"Li(\d+)E(?:Li(\d)E)?", tail)
    mode = m.group(1)[2:] + (LANE_MODE.get(r.group(2), f"_lane{r.group(2)}") if r and r.group(2) else "")
    return mode, STRUCTS.get(struct, struct), int(r.group(1)) if r else 0


def analyze(out: Path) -> tuple[dict, list[str]]:
    rows = [json.loads(x) for x in (out / "tensor_kinds.jsonl").read_text().splitlines() if x.startswith("{")]
    cases = [r for r in rows if "form" in r]
    done = [r for r in rows if r.get("done")]
    num_sms = done[0]["num_sms"] if done else None
    arch = done[0].get("sm", 120) if done else 120
    fails: list[str] = [] if num_sms else ["the run did not finish (no done line)"]
    groups = {r.get("group", -1) for r in done}
    if arch == 89 and done and groups != {0, 1, 2, 3}:
        fails.append(f"sm_89: groups {sorted(groups)} finished, want 0-3")
    sass = parse_sass((out / "sass" / "tensor_kinds.sass").read_text())
    kl = {}
    for fn, body in sass.items():
        k = kernel_key(fn)
        if k:
            kl[k] = loops(body)
    by_form: dict[str, list[dict]] = defaultdict(list)
    lanes = {}
    for r in [r for r in cases if r["mode"] == "opalone"]:
        L = kl.get(("opalone", r["form"], 64))
        if not L or len(L) != 1 or L[0]["mma"] or L[0]["iadd3"] + L[0]["ffma"] < 64:
            fails.append(f"opalone {r['form']}: loops {L} (want one loop of >= 64 IADD3 and FFMA)")
            continue
        # 8 warps a SMSP, every instruction of the loop body an iteration
        lanes[r["form"]] = 8 * r["iters"] * L[0]["other"] / r["cyc_med"]
    if MIXED_LANE not in lanes:
        if not any(f"opalone {MIXED_LANE}" in f for f in fails):
            fails.append(f"opalone: no {MIXED_LANE} row")
    elif lanes[MIXED_LANE] < LANE_MIN:
        fails.append(f"opalone: the lane op issues {lanes[MIXED_LANE]:.3f} a SMSP-cycle alone, below {LANE_MIN}")
    for r in cases:
        mode, form = r["mode"], r["form"]
        if mode == "opalone":
            continue
        base = mode.split("_")[0]
        R = int(round(r["ops_per_mma"] * 8)) if not base.startswith("alone") else 0
        key = (mode, form, R)
        L = kl.get(key)
        if L is None:
            fails.append(f"{key}: no SASS for the kernel")
            continue
        mma_loops = [x for x in L if x["mma"]]
        op_loops = [x for x in L if not x["mma"]]
        if len(mma_loops) != 1 or (base == "spec") != (len(op_loops) == 1):
            fails.append(f"{key}: loops {L} (want one MMA loop{' and one lane loop' if base == 'spec' else ''})")
            continue
        ml = mma_loops[0]
        sass_per_ptx = ml["mma"] / PTX_MMAS[base]
        if ml["mma"] % PTX_MMAS[base]:
            fails.append(f"{key}: {ml['mma']} SASS MMAs a loop, not a multiple of {PTX_MMAS[base]}")
        want_iadd3 = 8 * R if base == "mixu" else 0
        if base == "mixu" and ml["iadd3"] + ml["ffma"] < want_iadd3:
            fails.append(f"{key}: {ml['iadd3'] + ml['ffma']} IADD3 and FFMA in the loop, want >= {want_iadd3}")
        if base == "spec" and op_loops[0]["iadd3"] + op_loops[0]["ffma"] < R:
            fails.append(f"{key}: {op_loops[0]['iadd3'] + op_loops[0]['ffma']} IADD3 and FFMA in the lane loop, "
                         f"want >= {R}")
        it, w = r["iters"], MMA_WARPS[base]
        n_mma, n_other = w * it * PTX_MMAS[base], w * it * ml["other"]
        if base == "spec":
            n_other += 2 * it * op_loops[0]["other"]
        T = r["cyc_med"]
        if num_sms and r["distinct_sms"] != num_sms:
            fails.append(f"{key}: blocks on {r['distinct_sms']} of {num_sms} SMs")
        by_form[form].append({
            "mode": mode, "R": R, "ops_per_mma": n_other / n_mma, "cycles_per_mma": T / n_mma,
            "a_bound": (T - n_other) / n_mma, "sass_mma_per_ptx": sass_per_ptx, "mma_ops": ml["mma_ops"],
            "slots": r["slots"], "macs": r["macs"], "mhz": r["mhz"],
        })
    summary = {}
    for form, pts in by_form.items():
        mixed = [p for p in pts if not p["mode"].startswith("alone")]
        alone = [p["cycles_per_mma"] for p in pts if p["mode"].startswith("alone")]
        a = min(p["a_bound"] for p in mixed) if mixed else None
        by_lane = {lane: [p["a_bound"] for p in mixed if p["mode"].endswith("_iadd3") == (lane == "iadd3")]
                   for lane in (MIXED_LANE, "iadd3")}
        slots = pts[0]["slots"]
        rows = {}
        # JI3e is sm_120's; on another target its loads say nothing
        for row, (am, ao) in (JI3E.items() if arch == 120 else ()):
            def load(p, am=am, ao=ao):
                return (am + ao * p["ops_per_mma"]) / p["cycles_per_mma"]
            w = max(pts, key=load)
            rows[row] = {"load": load(w), "at": f"{w['mode']} R={w['R']}", "over": load(w) > 1 - NOISE}
        summary[form] = {
            "slots": slots, "macs": pts[0]["macs"], "mma_ops": pts[0]["mma_ops"],
            "alone_smsp_cycles_per_mma": min(alone) if alone else None,
            # 4 SMSPs an SM: output slots a SM a clock at the form's own pipe rate, and useful MACs
            "words_per_sm_clk": 4 * slots / min(alone) if alone else None,
            "macs_per_sm_clk": 4 * pts[0]["macs"] / min(alone) if alone else None,
            "a": a, "a_per_slot": a / slots if a is not None else None,
            "a_by_lane": {k: min(v) if v else None for k, v in by_lane.items()},
            "ji3e": rows, "points": pts,
        }
    ref = summary[REF]["a_per_slot"] if REF in summary else None
    for form, s in summary.items():
        s["vs_ref"] = s["a_per_slot"] / ref if ref and s["a_per_slot"] else None
        s["below_ref"] = bool(s["vs_ref"] is not None and s["vs_ref"] < 1 - NOISE)
    if lanes:
        summary["lane_op_alone"] = {k: {"issue_per_smsp_cycle": v} for k, v in lanes.items()}
    return summary, fails


def render(summary: dict) -> str:
    lines = ["| form | SASS | slots | alone SMSP-cycles a MMA | words/SM/clk | a (issue-cycles a MMA) | a: IADD3+FFMA / "
             "IADD3 | a a slot | vs IMMA s8 k32 | worst JI3e_0 / _1 / _2 load |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for form, s in summary.items():
        if form == "lane_op_alone":
            continue
        f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"
        lines.append(f"| {form} | {', '.join(s['mma_ops'])} | {s['slots']:.0f} | {f(s['alone_smsp_cycles_per_mma'], 2)} | "
                     f"{f(s.get('words_per_sm_clk'), 2)} | {f(s['a'])} | "
                     + " / ".join(f(v) for v in s.get("a_by_lane", {}).values()) + " | "
                     f"{f(s['a_per_slot'], 4)} | {f(s['vs_ref'])}{' **below**' if s['below_ref'] else ''} | "
                     + (" / ".join(f"{v['load']:.3f}{' **over**' if v['over'] else ''} ({v['at']})"
                                   for v in s["ji3e"].values()) or "–") + " |")
    if "lane_op_alone" in summary:
        rates = ", ".join(f"{k} {v['issue_per_smsp_cycle']:.3f}" for k, v in summary["lane_op_alone"].items())
        lines.append(f"\nLane ops alone, 8 warps a SMSP, warp instructions a SMSP-cycle: {rates} "
                     f"(mixed points use {MIXED_LANE}).")
    return "\n".join(lines) + "\n"


def refit(outs: list[Path], sched: float = SCHED) -> dict:
    """JI3n_0..2 from every program of the runs in `outs` and VF3e's (the module docstring)."""
    pts = []
    for out in outs:
        summary, _ = analyze(out)
        pts += [(form, p) for form, s in summary.items() if form != "lane_op_alone" for p in s["points"]]
    rows = {}
    for r, (am, ao) in JI3E.items():
        kinds: dict[str, float] = {}
        for form, p in pts:
            k = KIND.get(form, form)
            bound = (1 - NOISE) * p["cycles_per_mma"] - ao * p["ops_per_mma"]
            kinds[k] = min(kinds.get(k, am if k in VF3E_KINDS else float("inf")), bound)
        rows[r.replace("JI3e", "JI3n")] = {"MMA": min(am, *kinds.values()) / sched, "other": ao / sched,
                                           **{k: v / sched for k, v in sorted(kinds.items())}}
    return {"programs": len(pts), "noise": NOISE, "sched": sched, "rows": rows}


def main(argv: list[str]) -> int:
    if argv[1] == "refit":
        print(json.dumps(refit([Path(a) for a in argv[2:]]), indent=1))
        return 0
    out = Path(argv[1])
    summary, fails = analyze(out)
    (out / "tensor_kinds.json").write_text(json.dumps({"summary": summary, "gate_failures": fails}, indent=1))
    md = render(summary)
    (out / "tensor_kinds.md").write_text(md + ("\nGate failures:\n" + "\n".join(f"- {x}" for x in fails) if fails else ""))
    print(md)
    for x in fails:
        print("GATE", x)
    return 3 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
