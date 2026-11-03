"""sbench's side of PoUB beside its arms (`sbench.py --bw-passes GIB`): `Side` runs `dbw_side.py`
(the audit's verifier, and its prover on sbench's GPU) and starts each session as sbench needs it; `report` reads both
timelines, sbench's decode rows and the audit's passes, on the host's monotonic clock.

Each `--bw-prover-args` entry is a side of its own, one after another (the first registered, used and closed before the
next prover starts), each with its prover's own options (the default side none: `device.py`'s), the lever under test:

    registration        the prover's calibration, and so the allowance A and the pace's pass, beside the dither arms'
                        decode ops, run back to back from the side's start to its `ready` (`registered`), as
                        `node.py` calibrates beside its engines' served load
    session 0           idle: the arms built and gated, nothing generating (the passes' latency without Dither)
    rep r, off block    every arm's ops as without --bw-passes, no session running
    rep r, on block     session 1 + r started first (paced to outlast the block, `paced`), the same ops once its first
                        coin is out, then its verdict awaited
    session 1 + reps    a tampered prover (skip:0.0625, 1/16 of its blocks dropped) beside one decode op of each dither
                        arm: refused

A pass's latency is its coin's release to its root's arrival at the verifier (`windows_seen`: K = 1, so a group is a
pass); a pass is an arm's when one decode op of that arm's spans it whole, `between` when no op overlaps it and `mixed`
otherwise.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import threading
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
TAMPER = "skip:0.0625"
T_MS, DRAWS = 1000, 2773


class SideError(RuntimeError):
    """The side reported an error or exited: the rest of the run times nothing beside the passes."""


class Side:
    def __init__(self, out: Path, *, plan: list[str], gib: float, py: str, lib: str, kv: str, lean_exe: str | None,
                 prover_cpus: str, verifier_cpus: str | None, side_args: list[str], prover_args: list[str]) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        cmd = [py, str(HERE / "dbw_side.py"), "--out", str(out), "--plan", ",".join(plan), "--gib", str(gib),
               "--lib", lib, "--kv", kv, "--prover-cpus", prover_cpus, *(["--lean-exe", lean_exe] if lean_exe else []),
               *side_args, "--", *prover_args]
        if verifier_cpus:
            cmd = ["taskset", "-c", verifier_cpus, *cmd]
        self.cmd = cmd
        with open(out / "side.log", "w") as log:
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        self.passes_at: dict[int, int] = {}

    def next(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            raise SideError(f"the side exited ({self.p.wait()}): {self.out / 'side.log'}")
        ev = json.loads(line)
        with open(self.out / "events.jsonl", "a") as f:
            f.write(line)
        if ev["event"] == "error":
            raise SideError(ev["why"])
        if ev["event"] == "passes":
            self.passes_at[ev["k"]] = ev["at_ns"]
        return ev

    def expect(self, kind: str) -> dict:
        while (ev := self.next())["event"] != kind:
            pass
        return ev

    def send(self, msg: dict) -> None:
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def start(self, k: int, paced: str) -> dict:
        """Session k, its first coin sent once the prover is armed (after its untimed warm-up passes)."""
        self.send({"op": "session", "k": k, "paced": paced})
        return self.expect("started")

    def end(self, k: int) -> dict:
        ev = self.expect("ended")
        if ev["k"] != k:
            raise SideError(f"session {ev['k']} ended, not {k}")
        return ev | {"passes_at_ns": self.passes_at.get(k)}

    def stop(self) -> dict:
        self.send({"op": "stop"})
        ev = self.expect("stopped")
        self.p.stdin.close()
        self.p.wait()
        return ev


def registered(side: Side, serve: Callable[[], dict]) -> tuple[dict, list[dict]]:
    """The side's `ready`, `serve()` (a block of the dither arms' decode ops) run back to back from the side's start
    until it comes, and the blocks it returned: the prover calibrates inside its registration, so its passes' high
    percentile, the allowance taken from it, are those beside Dither serving and not those of an idle GPU."""
    got: dict = {}

    def wait() -> None:
        try:
            got["ready"] = side.expect("ready")
        except SideError as e:
            got["error"] = e

    t = threading.Thread(target=wait, daemon=True)
    t.start()
    blocks = []
    while t.is_alive():
        blocks.append(serve())
    t.join()
    if "error" in got:
        raise got["error"]
    return got["ready"], blocks


def plan(reps: int) -> list[str]:
    return ["honest"] * (1 + reps) + [TAMPER]


def paced(block_s: float, cover: float, stretch_s: float = T_MS / 1000) -> str:
    """A K = 1 session whose passes outlast `cover` times a block of `block_s` seconds, a stretch taking `stretch_s`
    (`window_s` of an earlier session; T_MS before any): W stretches of T_MS's groups (and the closing one), DRAWS
    openings a stretch, as the audit's 1 s sessions (`run.sh`'s, gate run 7's)."""
    return f"1:{T_MS}:{DRAWS}:{max(1, math.ceil(cover * block_s / stretch_s) + 1)}"


def window_s(session: dict) -> float:
    """The seconds a stretch of `session` took on the verifier's clock: its passes' span over its W stretches and the
    closing one. A stretch's groups are counted at the calibration's bare pass, so a session's passes, slower than
    that, take longer than T_MS a stretch, and more so beside the arms."""
    p = session["passes_ns"]
    if not p:
        return T_MS / 1000
    return (p[-1][1] - p[0][0]) / 1e9 / (session["pace"]["windows"] + 1)


def beside_s(session: dict, lo: int, hi: int) -> float | None:
    """The seconds a stretch of `session` took beside a block from `lo` to `hi`: its passes wholly inside the block, at
    their rate (None for fewer than two)."""
    p = [x for x in session["passes_ns"] if lo <= x[0] and x[1] <= hi]
    if len(p) < 2:
        return None
    return (p[-1][1] - p[0][0]) / 1e9 / len(p) * session["pace"]["groups"]


def stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    s = sorted(xs)
    def at(q):
        return s[min(len(s) - 1, int(q * len(s)))]
    return {"n": len(s), "mean": round(statistics.fmean(s), 4), "p50": round(at(0.5), 4), "p90": round(at(0.9), 4),
            "p99": round(at(0.99), 4), "max": round(s[-1], 4)}


def attribute(passes: list[list[int]], spans: list[tuple[str, int, int]]) -> dict[str, list[float]]:
    """Each pass's latency in ms, under the label of the op span that holds it whole, `between` where none overlaps it,
    else `mixed`."""
    out: dict[str, list[float]] = {}
    for lo, hi in passes:
        whole = [label for label, a, b in spans if a <= lo and hi <= b]
        label = whole[0] if whole else "mixed" if any(a < hi and lo < b for _, a, b in spans) else "between"
        out.setdefault(label, []).append((hi - lo) / 1e6)
    return out


def median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None


def tpot_gamma(dec: list[dict], arms: list[str]) -> dict:
    """Each arm's TPOT off and on over decode rows `dec` (medians over the reps), and each dither arm's γ over each
    other arm off and on."""
    tpot = {arm: {ph: median([x["tpot_b32_ms"] for x in dec if x["arm"] == arm and x["bw"] == ph])
                  for ph in ("off", "on")} for arm in arms}
    for t in tpot.values():
        if t["off"] and t["on"]:
            t["on_over_off"] = round(t["on"] / t["off"], 4)
    gamma = {}
    for d in (x for x in arms if x.startswith("dither")):
        for base in (x for x in arms if not x.startswith("dither")):
            g = {ph: round(tpot[d][ph] / tpot[base][ph], 4) for ph in ("off", "on") if tpot[d][ph] and tpot[base][ph]}
            if tpot[d]["on"] and tpot[base]["off"]:
                g["dither_on_over_base_off"] = round(tpot[d]["on"] / tpot[base]["off"], 4)
            gamma[f"{d}/{base}"] = g
    return {"tpot_b32_ms": tpot, "gamma": gamma}


def report(rows: list[dict], sides: list[dict], arms: list[str]) -> dict:
    """The run's numbers beside the passes: each arm's TPOT off and on and each dither arm's γ (`tpot_gamma`), over
    every side's blocks and each side's own, each side's pass latency idle and by what ran beside it, its sessions'
    verdicts, whether every on block lay inside its session's passes, what served beside its calibration
    (`calibrated_beside`: the decode ops from the side's start to its `ready`, the share of that span they decoded, and
    their TPOT), and the memory lines. An on row counts only when its whole decode op lies inside its session's passes
    (`uncovered`, those that don't): one whose passes end in it times its N1 and N2 runs under different loads, and its
    TPOT is noise (negative even)."""
    dec = [x for x in rows if x["op"] == "decode" and "bw" in x]
    k_of = {(side["i"], blk["rep"], blk["phase"]): blk["k"] for side in sides for blk in side.get("blocks", ())}
    passes = {(side["i"], k): s["passes_ns"] for side in sides for k, s in side.get("sessions", {}).items()}

    def inside(x: dict) -> bool:
        p = passes.get((x.get("side"), k_of.get((x.get("side"), x["rep"], x["bw"]))))
        return bool(p) and "spans_ns" in x and p[0][0] <= x["spans_ns"][0] and x["spans_ns"][-1] <= p[-1][1]

    timed = [x for x in dec if x["bw"] == "off" or inside(x)]
    uncovered: dict[int, dict[str, int]] = {}
    for x in dec:
        if x["bw"] not in ("off", "cal") and not inside(x):
            by = uncovered.setdefault(x.get("side"), {})
            by[x["arm"]] = by.get(x["arm"], 0) + 1
    out_sides = []
    for side in sides:
        if "ready" not in side:
            out_sides.append({"i": side["i"], "prover_args": side["prover_args"], "error": side.get("error"),
                              "honest_accepted": False, "tampered_refused": False})
            continue
        sessions = side["sessions"]
        by: dict[str, list[float]] = {}
        covered = []
        for blk in side["blocks"]:
            if blk["k"] is None:
                continue
            s = sessions.get(blk["k"])
            spans = [(x["arm"], x["spans_ns"][0], x["spans_ns"][-1]) for x in dec
                     if x.get("side") == side["i"] and x["rep"] == blk["rep"] and x["bw"] == blk["phase"]]
            if s is None or not s["passes_ns"]:
                covered.append({"k": blk["k"], "phase": blk["phase"], "covered": False})
                continue
            first, last = s["passes_ns"][0][0], s["passes_ns"][-1][1]
            covered.append({"k": blk["k"], "phase": blk["phase"], "covered": first <= blk["lo"] and blk["hi"] <= last,
                            "block_s": round((blk["hi"] - blk["lo"]) / 1e9, 3),
                            "passes_s": round((last - first) / 1e9, 3)})
            for label, xs in attribute(s["passes_ns"], spans).items():
                by.setdefault(label if blk["phase"] == "on" else f"{blk['phase']}:{label}", []).extend(xs)
        idle = sessions.get(0)
        cal = side["ready"]["calibration"]
        beside = side.get("calibrated_beside")
        if beside and beside["blocks"]:
            lo, hi = beside["blocks"][0]["lo"], beside["blocks"][-1]["hi"]
            ops = [x for x in dec if x.get("side") == side["i"] and x["bw"] == "cal"]
            beside = {"arms": beside["arms"], "ops": len(ops), "serving_s": round((hi - lo) / 1e9, 3),
                      "ready_at_s": round((side["ready"]["t_ns"] - lo) / 1e9, 3),
                      "decoding": round(sum(x["spans_ns"][-1] - x["spans_ns"][0] for x in ops if "spans_ns" in x)
                                        / (hi - lo), 4),
                      "tpot_b32_ms": {arm: median([x["tpot_b32_ms"] for x in ops if x["arm"] == arm])
                                      for arm in beside["arms"]}}
        out_sides.append({
            "i": side["i"], "prover_args": side["prover_args"], "gib": side["ready"]["gib"],
            **tpot_gamma([x for x in timed if x.get("side") == side["i"]], arms),
            "uncovered": uncovered.get(side["i"], {}),
            "calibrated_beside": beside,
            "blocks": side["ready"]["blocks"], "allowance_ms": round(side["ready"]["allowance_ns"] / 1e6, 3),
            "calibration_pass_ms": {q: round(cal["pass_ns"][q] / 1e6, 4) for q in ("p50", "p99", "p999", "max")
                                    if q in cal.get("pass_ns", {})},
            "read_ms": cal.get("read_ms"),
            "pass_ms": {"idle": stats([(b - a) / 1e6 for a, b in idle["passes_ns"]]) if idle else None}
            | {label: stats(xs) for label, xs in sorted(by.items())},
            "coverage": covered,
            "verdicts": {k: {"accepted": s["accepted"], "failure": s["failure"], "plan": side["plan"][k],
                             "passes": len(s["passes_ns"]), "pace": s["pace"],
                             "pass_ms": stats([(b - a) / 1e6 for a, b in s["passes_ns"]])}
                         for k, s in sorted(sessions.items())},
            "honest_accepted": len(sessions) == len(side["plan"]) and all(
                s["accepted"] for k, s in sessions.items() if side["plan"][k] == "honest"),
            "tampered_refused": len(sessions) == len(side["plan"]) and all(
                not s["accepted"] for k, s in sessions.items() if side["plan"][k] != "honest"),
            "memory": {k: {"memory": s["memory"], "paced": s["paced"]} for k, s in sorted(sessions.items())
                       if s["accepted"]},
            "stopped": side.get("stopped")})
    return {**tpot_gamma(timed, arms), "sides": out_sides}
