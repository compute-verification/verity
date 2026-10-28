"""pgemm.cu's run (pipe.sh) into the honest sm_100 kernel's numbers: its rate at each commit spacing against ρ_T5, the
TMEM stage alone, ε_tensor at L = 32 with the measured points, the fold's issue cost beside the MMA, and q*.

    python3 pipe.py gate-build OUT                  refuse a pg100 that spills, can't launch, or lacks UTCIMMA / LDTM.x32
    python3 pipe.py analyze OUT [--census RUN_OUT ...] [--json F]
    python3 pipe.py agree OUT_A OUT_B               a repeat's rates against the first's, point by point

OUT is pipe.sh's output (rate.jsonl, gate/*/result.json and ref.json, sass/pgemm.sass). --census takes the B200 probe runs
(run.sh sm100, fetched with `research fetch RUN --all`): their TMEM region and mma.sync points, with the kernel's measured
points added, give ε_tensor at L = 32 by TENSOR's two-stage LP (analyze.tmem_model). Without it the census lines in
`../machine_model/data/devices/sm_100/` give ρ_T5, the issue facet and the scalar rates, and ε is not computed.

Arms (pgemm.cu's MODE): 0 the pipeline alone (loaders hand the barrier back), 1 the TMEM stage (every word loaded, not
folded), 2 honest (loaded and folded by duo), 3 the MMA warp waiting on its own commit (no loaders). MI 1: one MMA warp a
tile.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parent / "machine_model" / "data" / "devices" / "sm_100" / "census.csv"
ARMS = {0: "pipeline", 1: "TMEM stage", 2: "honest", 3: "MMA self-wait"}
SLOTS_MMA, LANES, SMSPS = 128 * 256, 32, 4
#: duo's lane instructions a word by pipe (dc.cuh): u = y ^ p ^ k (LOP3), y = u a + z (IMAD), z = rotl(u, r) + y (LEA.HI)
DUO = {"ALU": 2, "IMAD": 1}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def census_lines(path: Path = CENSUS) -> dict[tuple[str, str], dict]:
    with path.open(newline="") as f:
        return {(r["kind"], r["row"]): r for r in csv.DictReader(f)}


def machine(path: Path = CENSUS) -> dict:
    """ρ_T5 (slots/SM/clk), the issue facet JI100_0 (SMSP-cycles a UTCIMMA 128x256, a warp instruction of any other
    kind), the ALU and IMAD pipes (warp instructions/SM/clk) from the device census."""
    c = census_lines(path)
    return {"rho_T5": float(c[("UTCIMMA_M128_N256", "M")]["rate"]) * SLOTS_MMA,
            "c_M": float(c[("UTCIMMA_M128_N256", "JI100_0")]["cycles"]),
            "c_X": float(c[("ALU", "JI100_0")]["cycles"]),
            "ALU": float(c[("ALU", "A")]["rate"]), "IMAD": float(c[("IMAD", "I")]["rate"]),
            "ld_x32": float(c[("LDTM_32X32B_X32", "TL")]["rate"]) * 1024}


# ---- SASS ----------------------------------------------------------------------------------------------------------------
FUNC = re.compile(r"Function : _Z5pg100ILi(\d)ELi(\d)ELi(\d)ELb(\d)E")
INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]*)")


def kernels(sass: str) -> dict[tuple[int, int, int, int], list[str]]:
    """(NB, LW, MODE, MI) -> the kernel's opcodes in order."""
    out, cur = {}, None
    for line in sass.splitlines():
        m = FUNC.search(line)
        if m:
            cur = out.setdefault(tuple(int(x) for x in m.groups()), [])
        elif cur is not None:
            i = INSTR.search(line)
            if i:
                cur.append(i.group(2))
    return out


def pipe_of(op: str) -> str:
    base = op.split(".")[0]
    if base in ("LOP3", "LEA", "SHF", "IADD3", "ISETP", "SEL", "PRMT", "LOP"):
        return "ALU"
    if base == "IMAD":
        return "IMAD"
    if base == "LDTM":
        return "LDTM"
    return "other"


def fold_profile(ops: list[str], nb: int, lw: int) -> dict:
    """The honest loader's issue a word, from its main loop's straight-line tile passes: every instruction from the first
    tcgen05.ld of a group to the last of it (TPW tiles of BN / 32 loads, unrolled), over the words those loads bring
    (the last load's 32 words are folded after the range, so the range holds the fold of all but one load)."""
    bn, tpw = 512 // nb, nb // lw
    loads = [i for i, op in enumerate(ops) if op.startswith("LDTM")]
    per_group = tpw * bn // 32
    if len(loads) < per_group:
        return {}
    seg = ops[loads[0]:loads[per_group - 1]]
    words = (per_group - 1) * 32
    by = {}
    for op in seg:
        by[pipe_of(op)] = by.get(pipe_of(op), 0) + 1
    return {"instructions": len(seg) / words, **{k: v / words for k, v in sorted(by.items())}, "words": words}


KERNELS = 38   # (NB, LW) in (2, 1) (2, 2) (4, 1) (4, 2), four modes, and (4, 4) without honest; one MMA warp or one a tile
REGS_SMSP = 16384


def gate_build(out: Path) -> int:
    log = (out / "build.log").read_text()
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    regs = {tuple(int(x) for x in m.groups()[:4]): int(m.group(5)) for m in re.finditer(
        r"Compiling entry function '_Z5pg100ILi(\d)ELi(\d)ELi(\d)ELb(\d)E[^']*'.*?Used (\d+) registers", log, re.S)}
    ks = kernels((out / "sass" / "pgemm.sass").read_text())
    bad = []
    if len(ks) != KERNELS or len(spills) < KERNELS or len(regs) != KERNELS:
        bad.append(f"{len(ks)} pg100 kernels in the SASS, {len(spills)} ptxas spill and {len(regs)} register reports; "
                   f"{KERNELS} expected")
    if any(s != ("0", "0") for s in spills):
        bad.append(f"ptxas reports spills: {spills}")
    for (nb, lw, mode, mi), ops in sorted(ks.items()):
        n_mma = sum(op.startswith("UTCIMMA") for op in ops)
        n_ld = sum(op == "LDTM.x32" for op in ops)
        local = sum(op.split(".")[0] in ("LDL", "STL") for op in ops)
        r = regs.get((nb, lw, mode, mi), 0)
        cap = min(255, REGS_SMSP // (-(-(4 * lw + (nb if mi else 1)) // 4) * 32) // 8 * 8)
        ok = n_mma > 0 and local == 0 and (mode in (0, 3) or n_ld > 0) and 0 < r <= cap
        name = f"pg100<{nb},{lw},{mode},{mi}>"
        print(f"{'PASS' if ok else 'REFUSED'} {name}: UTCIMMA {n_mma}, LDTM.x32 {n_ld}, LDL/STL {local}, registers {r} "
              f"(launchable {cap})")
        if not ok:
            bad.append(name)
    for b in bad:
        print(f"REFUSED {b}", file=sys.stderr)
    return 1 if bad else 0


# ---- the run -------------------------------------------------------------------------------------------------------------
def load(out: Path) -> tuple[list[dict], list[dict]]:
    rates = [json.loads(x) for x in (out / "rate.jsonl").read_text().splitlines() if x.strip()]
    gates = []
    for d in sorted((out / "gate").glob("*")):
        res = d / "result.json"
        ref = d / "ref.json"
        if res.exists():
            g = json.loads(res.read_text().strip().splitlines()[-1])
            g["ref_ok"] = bool(ref.exists() and json.loads(ref.read_text().strip().splitlines()[-1]).get("ok"))
            gates.append(g)
    return rates, gates


def best(rates: list[dict], arm: int, L: int) -> dict | None:
    pts = [r for r in rates if r["arm"] == arm and r["L"] == L]
    return max(pts, key=lambda r: r["slots_per_clk"], default=None)


def issue_beside(w: float, prof: dict, mach: dict, L: int) -> dict:
    """Honest's issue at w committed words/SM/clk: the fold's (and the loads') warp instructions a SM-clock by pipe, and
    the MMA's on the issue facet, against 4 issue slots and the ALU and IMAD pipes."""
    per = {k: w * prof.get(k, 0.0) / LANES for k in ("instructions", "ALU", "IMAD", "LDTM")}
    mma = w * (L // 32) / SLOTS_MMA
    return {"fold_and_loads_issue": per["instructions"], "issue_fraction": (per["instructions"] * mach["c_X"]
            + mma * mach["c_M"]) / SMSPS, "ALU_fraction": per["ALU"] / mach["ALU"],
            "IMAD_fraction": per["IMAD"] / mach["IMAD"]}


def qstar(mach: dict, L: int, ld_issue_word: float) -> dict:
    """q*: the issue (SMSP-cycles) honest's tensor route spends a committed word a lane: L / 32 UTCIMMA 128x256 at c_M a
    1024 words a lane, plus the tcgen05.ld that bring each word to a register at c_X (the kernel's SASS count: ptxas
    makes tcgen05.wait::ld a scoreboard wait, no instruction)."""
    mma = (L // 32) * mach["c_M"] / (SLOTS_MMA / LANES)
    return {"mma_only": mma, "with_loads": mma + ld_issue_word * mach["c_X"]}


def honest_profile(ks: dict) -> tuple[tuple[int, int], dict]:
    """The fold's issue a word from the honest kernel whose straight-line group holds the most words (duo is the same code
    in every (NB, LW); a short group's range misses fold ptxas scheduled after its last load)."""
    profs = {(nb, lw): fold_profile(ops, nb, lw) for (nb, lw, mode, mi), ops in ks.items() if mode == 2 and not mi}
    key = max(profs, key=lambda t: profs[t].get("words", 0))
    return key, profs[key]


def eps_tensor_l32(census_dirs: list[Path], rates: list[dict], rho5: float) -> dict:
    """ε_tensor at L = 32 with honest's measured TMEM stage (arm 1: every word loaded, the fold off) against TENSOR's
    two-stage adversary, who keeps t ρ_T5 tcgen05 words and fills the shared pipe's slack at t with mma.sync. Three t:
    T1's (the adversary has the whole pipe: ε = 1 − honest / ρ_T5), the probes' TMEM region's balanced point (load
    bandwidth only, the kernel's points added), and that capped by the round trip: a tcgen05 word at L = 32 must leave
    TMEM before the next k-step lands on it, so no program keeps more than TMEM's 128 x 512 words in flight, and arm 3
    (the MMA waiting on its own commit, no read) bounds any program's tcgen05 words from above."""
    census = _module("dc_census", HERE / "census.py")
    analyze = census.analyze
    dev = census.FamilyRuns("sm100", census_dirs)
    base = dev.model()
    region = list(base["tmem_points"]) + [(r["slots_per_clk"] / rho5, r["committed_words_per_clk"] / rho5)
                                          for r in rates if r["arm"] in (1, 2) and r["L"] == 32]
    h = analyze.balanced(region)
    rho_sync = max(v["rate"] * v["slots_per"] for k, v in dev.tensor().items() if k.startswith("IMMA"))
    honest = best(rates, 1, 32)
    if not honest:
        return {}
    w = honest["committed_words_per_clk"]
    out = {"honest_TMEM_stage": w, "honest_fraction": w / rho5, "rho_sync": rho_sync,
           "region_balanced_probes": base["honest_fraction_L32"], "eps_probes_lp": base.get("eps_tensor_L32"),
           "T1": 1 - w / rho5}
    rt = max((b for b in (best(rates, 0, 32), best(rates, 3, 32)) if b), key=lambda r: r["slots_per_clk"], default=None)
    out["honest_config"] = {k: honest.get(k) for k in ("nb", "lw", "mi")}
    caps = {"bandwidth": h} | ({"round_trip": min(h, rt["slots_per_clk"] / rho5)} if rt else {})
    if rt:
        out["round_trip_config"] = {k: rt.get(k) for k in ("nb", "lw", "mi", "arm")}
    for name, t in caps.items():
        extra = analyze.upper(base["sync_points"], t) * rho_sync
        out[name] = {"t": t, "tcgen05_words": t * rho5, "mma_sync_words": extra, "adversary_words": t * rho5 + extra,
                     "eps": 1 - w / (t * rho5 + extra)}
    return out


def analyze_run(out: Path, census_dirs: list[Path]) -> dict:
    rates, gates = load(out)
    mach = machine()
    rho5 = mach["rho_T5"]
    ks = kernels((out / "sass" / "pgemm.sass").read_text()) if (out / "sass" / "pgemm.sass").exists() else {}
    for r in rates:
        r["tensor_fraction"] = r["slots_per_clk"] / rho5
    res: dict = {"rho_T5": rho5, "gates": {"n": len(gates), "exact": sum(g["exact"] and g["ref_ok"] for g in gates)},
                 "rates": rates, "machine": mach, "by_L": {}}
    if ks:
        res["fold_kernel"], res["fold_profile"] = honest_profile(ks)
    prof = res.get("fold_profile", {})
    for L in sorted({r["L"] for r in rates}):
        row = {}
        for arm in ARMS:
            b = best(rates, arm, L)
            if b:
                row[ARMS[arm]] = {k: b.get(k) for k in ("nb", "lw", "mi", "slots_per_clk", "committed_words_per_clk",
                                                         "tensor_fraction", "mhz", "group_clk")}
        h = best(rates, 2, L)
        if h and prof:
            row["issue_beside"] = issue_beside(h["committed_words_per_clk"], prof, mach, L)
        row["qstar"] = qstar(mach, L, prof.get("LDTM", 1 / 32))
        res["by_L"][L] = row
    if census_dirs:
        res["eps_L32"] = eps_tensor_l32(census_dirs, rates, rho5)
    return res


def text(res: dict) -> str:
    lines = [f"rho_T5 {res['rho_T5']:.2f} slots/SM/clk (census); gates exact {res['gates']['exact']}/{res['gates']['n']}",
             "", "| L | arm | (NB, LW, MI) | slots/SM/clk | committed words/SM/clk | of rho_T5 | clk a group |",
             "|---|---|---|---|---|---|---|"]
    for L, row in res["by_L"].items():
        for arm in ARMS.values():
            b = row.get(arm)
            if b:
                g = f"{b['group_clk']:.0f}" if b.get("group_clk") else ""
                lines.append(f"| {L} | {arm} | ({b['nb']}, {b['lw']}, {b.get('mi') or 0}) | {b['slots_per_clk']:.2f} | "
                             f"{b['committed_words_per_clk']:.2f} | {b['tensor_fraction']:.4f} | {g} |")
    lines.append("")
    p = res.get("fold_profile")
    if p:
        lines.append(f"honest's loader (pg100<{res['fold_kernel'][0]},{res['fold_kernel'][1]},2>, {p['words']} words): "
                     f"{p['instructions']:.3f} instructions a word a lane (ALU {p.get('ALU', 0):.3f}, IMAD "
                     f"{p.get('IMAD', 0):.3f}, LDTM {p.get('LDTM', 0):.4f}, other {p.get('other', 0):.3f})")
    for L, row in res["by_L"].items():
        q = row["qstar"]
        line = f"L = {L}: q* {q['mma_only']:.4f} (MMA), {q['with_loads']:.4f} (with the loads)"
        i = row.get("issue_beside")
        if i:
            line += (f"; at honest's rate the loader issues {i['fold_and_loads_issue']:.3f} warp instructions/SM/clk: "
                     f"{i['issue_fraction']:.3f} of the SM's issue (4 SMSPs), ALU pipe {i['ALU_fraction']:.3f}, IMAD pipe "
                     f"{i['IMAD_fraction']:.3f}")
        lines.append(line)
    e = res.get("eps_L32")
    if e:
        lines += ["", f"eps_tensor at L = 32, honest's measured TMEM stage {e['honest_TMEM_stage']:.2f} words/SM/clk "
                  f"({e['honest_fraction']:.4f} of rho_T5; the probes' LP assumed {e['region_balanced_probes']:.4f} and "
                  f"gave {e['eps_probes_lp']:.4f}): T1 {e['T1']:.4f}"]
        for name in ("bandwidth", "round_trip"):
            c = e.get(name)
            if c:
                lines.append(f"- adversary capped by {name.replace('_', ' ')}: t {c['t']:.4f}, tcgen05 "
                             f"{c['tcgen05_words']:.2f} + mma.sync {c['mma_sync_words']:.2f} = "
                             f"{c['adversary_words']:.2f} words/SM/clk, eps {c['eps']:.4f}")
    return "\n".join(lines)


def agree(a: Path, b: Path) -> dict:
    """Two runs' rates point by point: the ratio of slots/SM/clk for every (NB, LW, MI, arm, L) both have."""
    key = ("nb", "lw", "mi", "arm", "L")
    ra = {tuple(r.get(k) for k in key): r for r in load(a)[0]}
    rb = {tuple(r.get(k) for k in key): r for r in load(b)[0]}
    ratios = {k: rb[k]["slots_per_clk"] / ra[k]["slots_per_clk"] for k in ra.keys() & rb.keys()}
    worst = max(ratios, key=lambda k: abs(ratios[k] - 1), default=None)
    return {"points": len(ratios), "worst": worst and {"point": dict(zip(key, worst)), "ratio": ratios[worst]},
            "within_2pct": sum(abs(r - 1) <= 0.02 for r in ratios.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["gate-build", "analyze", "agree"])
    ap.add_argument("out", type=Path)
    ap.add_argument("other", type=Path, nargs="?")
    ap.add_argument("--census", type=Path, nargs="*", default=[])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "gate-build":
        return gate_build(a.out)
    if a.cmd == "agree":
        print(json.dumps(agree(a.out, a.other)))
        return 0
    res = analyze_run(a.out, a.census)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
