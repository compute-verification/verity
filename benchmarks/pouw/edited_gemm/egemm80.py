"""egemm80.cu's SASS gate, its run's summary, and its honest profiles in the machine model's classes (sm_80's census).

    python3 egemm80.py gate SASS BUILD_LOG    every arm's mainloop (its k-stage loop): the MMAs, the fold and the loads it
                                              must hold, no local memory, no spill; JSON to stdout, exit 1 on a refusal
    python3 egemm80.py summary OUT            the gate's verdicts and each arm's SMSP-cycles a MMA, from the bench (median
                                              item at its NVML clock) and from the hold (the steady clock's half)

A k-stage is 128 of k: 4 atoms of 8 MMAs a warp (e16: 16, two k16 halves an atom). Its fold is duo (LOP3, IMAD, LEA.HI) on
4 words a lane a committed MMA, with a LOP3 tie every 4th MMA, and its loads 4 LDSM.16.M88.4 an atom and 4 LDGSTS a
thread. A profile counts every instruction of the kernel a mainloop MMA: the k-stage loop's at its KT - 1 trips a tile
and every other instruction of the kernel once a tile (the first stage, the flush, the peel, the epilogue and the
per-CTA setup, so an over-count), each in sm_80's census classes (`CLASS`).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ARMS = {0: "p32", 1: "e16", 2: "e32", 3: "e64"}
#: per arm: (mainloop MMA SASS, its census kind, MMAs a k-stage, committed MMAs a k-stage, L)
SHAPE = {"p32": ("IMMA.16832", "IMMA16832", 32, 0, 32), "e16": ("IMMA.16816", "IMMA16816", 64, 64, 16),
         "e32": ("IMMA.16832", "IMMA16832", 32, 32, 32), "e64": ("IMMA.16832", "IMMA16832", 32, 16, 64)}
TIES = {"p32": 0, "e16": 16, "e32": 8, "e64": 8}
SMSPS = 4
INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")


def cls(op: str) -> str:
    """The census class of a SASS opcode on sm_80 (sm_80's census: data/devices/sm_80). ALU holds every integer and
    logic op that isn't IMAD (on A, as the ALU pipe's measured IADD3 and LOP3); IMAD every IMAD form (IMAD.WIDE counted
    twice by `weight`); NOROW everything with no pipe row (issue only: the loads by cp.async, barriers, branches, uniform
    datapath, stores, special registers)."""
    if op.startswith("IMMA.16832"):
        return "IMMA16832"
    if op.startswith("IMMA.16816"):
        return "IMMA16816"
    if op.startswith("LDSM.16.M88.4"):
        return "LDSM_X4"
    if op.startswith("LDSM.16.M88.2"):
        return "LDSM_X2"
    if op.startswith("LDSM.16.M88"):
        return "LDSM_X1"
    if op.startswith("IMAD"):
        return "IMAD"
    if op.split(".")[0] in ("LOP3", "IADD3", "LEA", "SHF", "ISETP", "PRMT", "IMNMX", "SEL", "IABS", "BMSK", "SGXT", "PLOP3",
                            "MOV", "LOP", "SHL", "SHR", "P2R", "R2P", "FLO", "BREV"):
        return "ALU"
    return "NOROW"


def weight(op: str) -> float:
    return 2.0 if op.startswith("IMAD.WIDE") else 1.0


def functions(sass: str) -> dict[str, list[tuple[int, str, str]]]:
    out: dict[str, list[tuple[int, str, str]]] = {}
    cur = None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = INSN.search(line)
        if cur and m:
            out[cur].append((int(m.group(1), 16), m.group(2), m.group(3)))
    return out


def arm_of(name: str) -> str | None:
    m = re.match(r"_Z7egemm80ILi(\d)EEv4Args", name)
    return ARMS.get(int(m.group(1))) if m else None


def mainloop(ins: list[tuple[int, str, str]], mma: str) -> list[tuple[int, str, str]]:
    """The backward branch's body holding the most MMAs of the arm's form (the k-stage loop)."""
    best: list = []
    for addr, op, rest in ins:
        if op == "BRA":
            m = re.search(r"0x([0-9a-f]+)", rest)
            if m and int(m.group(1), 16) < addr:
                body = [x for x in ins if int(m.group(1), 16) <= x[0] <= addr]
                if sum(x[1].startswith(mma) for x in body) > sum(x[1].startswith(mma) for x in best):
                    best = body
    return best


def gate(sass_path: Path, build_log: Path) -> dict:
    fns = functions(sass_path.read_text())
    log = build_log.read_text()
    out: dict = {"arms": {}}
    ok = True
    for name, ins in fns.items():
        arm = arm_of(name)
        if arm is None:
            continue
        mma, _, n_mma, n_com, _ = SHAPE[arm]
        body = mainloop(ins, mma)
        c = Counter(op for _, op, _ in body)
        fam = lambda p: sum(v for k, v in c.items() if k.startswith(p))
        words = 4 * n_com
        v = {"function": name, "loop_insns": len(body), "loop_mma": fam(mma), "loop_lea_hi": fam("LEA.HI"),
             "loop_lop3": fam("LOP3"), "loop_imad": fam("IMAD"), "loop_ldsm4": fam("LDSM.16.M88.4"),
             "loop_ldgsts": fam("LDGSTS"), "loop_bar": fam("BAR"), "local": sum(op.startswith(("LDL", "STL")) for _, op, _ in ins),
             "loop_counts": dict(sorted(c.items())), "function_counts": dict(sorted(Counter(op for _, op, _ in ins).items()))}
        fold = v["loop_lea_hi"] + v["loop_lop3"] + v["loop_imad"] + fam("SHF") + fam("IADD3")
        why = []
        if v["loop_mma"] != n_mma:
            why.append(f"{v['loop_mma']} {mma} in the loop, not {n_mma}")
        if v["loop_ldsm4"] != 16:
            why.append(f"{v['loop_ldsm4']} LDSM.16.M88.4, not 16")
        if v["loop_ldgsts"] != 4:
            why.append(f"{v['loop_ldgsts']} LDGSTS, not 4")
        if v["local"]:
            why.append(f"{v['local']} local-memory instructions")
        if words and (v["loop_lea_hi"] < words or v["loop_lop3"] < words + TIES[arm] or fold < 3 * words + TIES[arm]):
            why.append(f"the fold: {v['loop_lop3']} LOP3, {v['loop_imad']} IMAD, {v['loop_lea_hi']} LEA.HI for {words} words")
        m = re.search(rf"Compiling entry function '{re.escape(name)}'.*?(\d+) bytes spill stores, (\d+) bytes spill loads", log,
                      re.S)
        if m and (int(m.group(1)) or int(m.group(2))):
            why.append(f"spills: {m.group(1)} B stores, {m.group(2)} B loads")
        v["refused"] = why
        ok &= not why
        out["arms"][arm] = v
    if set(out["arms"]) != set(SHAPE):
        ok = False
        out["missing"] = sorted(set(SHAPE) - set(out["arms"]))
    out["ok"] = ok
    return out


def profile(g: dict, arm: str, K: int) -> dict[str, float]:
    """Warp instructions a mainloop MMA by census class (the module docstring's count)."""
    v = g["arms"][arm]
    mma, kind, n_mma, _, _ = SHAPE[arm]
    KT = K // 128
    per_tile_main = n_mma * KT
    n: Counter = Counter()
    rest = Counter(v["function_counts"])
    for op, k in v["loop_counts"].items():
        n[cls(op)] += weight(op) * k * (KT - 1)
        rest[op] -= k
    for op, k in rest.items():
        n[cls(op)] += weight(op) * k
    return {k: round(x / per_tile_main, 4) for k, x in sorted(n.items(), key=lambda t: -t[1]) if x}


def _lines(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def summary(out: Path) -> dict:
    g = json.loads((out / "gate-sass.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    ref = _lines(out / "ref.jsonl")
    res: dict = {"run": meta.get("run_id"), "M": meta["M"], "N": meta["N"], "K": meta["K"], "sass_gate": g["ok"],
                 "poison_gate": _lines(out / "gate.jsonl"), "ref": ref[0] if ref else None, "arms": {}}
    bench = _lines(out / "bench.jsonl")
    for arm in SHAPE:
        r: dict = {"refused": g["arms"].get(arm, {}).get("refused")}
        info = next((x for x in bench if x.get("arm") == arm and "mmas" in x), None)
        items = [x for x in bench if x.get("gk") == "item" and x.get("arm") == arm]
        if info and items:
            ms = statistics.median(x["ms"] / x["calls"] for x in items)
            mhz = statistics.median(x["sm_mhz"]["during"] or x["sm_mhz"]["after"] for x in items)
            r["bench"] = {"ms": round(ms, 5), "nvml_mhz": mhz, "cycles_per_mma": round(ms * 1e-3 * mhz * 1e6 * SMSPS * info["sms"] / info["mmas"], 4),
                          "throttle": sorted({x["throttle"] for x in items})}
            mma_main = info["mmas"] - (meta["M"] // 128) * (meta["N"] // 128) * 16 * 8 * 2
            r["bench"]["cycles_per_main_mma"] = round(ms * 1e-3 * mhz * 1e6 * SMSPS * info["sms"] / mma_main, 4)
        hold = _lines(out / f"hold-{arm}.jsonl")
        if hold:
            tail = hold[len(hold) // 2:]
            ms = statistics.median(x["ms_per_call"] for x in tail)
            mhz = statistics.median(x["sm_mhz"] for x in tail)
            mma_main = hold[0]["mmas"] - (meta["M"] // 128) * (meta["N"] // 128) * 16 * 8 * 2
            r["hold"] = {"seconds": len(hold), "ms": round(ms, 5), "nvml_mhz_steady": mhz,
                         "nvml_mhz_min": min(x["sm_mhz"] for x in hold), "nvml_mhz_max": max(x["sm_mhz"] for x in hold),
                         "cycles_per_main_mma": round(ms * 1e-3 * mhz * 1e6 * SMSPS * hold[0]["sms"] / mma_main, 4),
                         "throttle": sorted({x["throttle"] for x in hold})}
        if arm in g["arms"]:
            r["per_mma"] = profile(g, arm, meta["K"])
        res["arms"][arm] = r
    return res


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "gate":
        g = gate(Path(sys.argv[2]), Path(sys.argv[3]))
        print(json.dumps(g, indent=1))
        for arm, v in g["arms"].items():
            print(f"{arm}: loop {v['loop_insns']} insns, {v['loop_mma']} MMA, {v['loop_lop3']} LOP3, {v['loop_imad']} IMAD, "
                  f"{v['loop_lea_hi']} LEA.HI, {v['loop_ldsm4']} LDSM.x4, {v['loop_ldgsts']} LDGSTS"
                  + (f"  REFUSED: {'; '.join(v['refused'])}" if v["refused"] else "  PASS"), file=sys.stderr)
        return 0 if g["ok"] else 1
    if len(sys.argv) == 3 and sys.argv[1] == "summary":
        print(json.dumps(summary(Path(sys.argv[2])), indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
