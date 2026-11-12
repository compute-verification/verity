"""The scan's cost from one scan's kept files (README.md): two numbers, the stages they are made of, and the guard.

    result.py write RUN_DIR --scan-exit RC    after run.sh's scan: RUN_DIR/out/scan_cost.json and the run's result.json
    result.py table DOC                       a document's round, a session a line and a stage a column
    result.py compare CANDIDATE [--to DOC]    a candidate beside the baseline (baseline.json's run), or beside DOC
    result.py baseline [--lock DOC]           the baseline, or DOC's run written as the baseline

DOC and CANDIDATE are a `scan_cost.json`, a directory that holds one (or `out/scan_cost.json`), or a run's id, whose
document is fetched from the evidence store (`research data show`, `research data fetch`).

**What is read.** Everything is the scan's own (`benchmarks/network_traces/scan_zk.sh`), under RUN_DIR/out:
`sczk/summary.json` (the class, the execution phases in the registration, T and the tick, and for each session of the
honest round its prover's seconds, its staging and the Lean verifier's wall), each of those sessions' `prove.log`
and `lean-verify.log` (the stages), `sczk/answers.json` (the guard), `host.txt` (the cores the job had) and
`lean.jsonl` (each Lean verifier process's core and CPU seconds, `SC_LEAN_ONE`).

**The numbers.** A round is the honest round: every block of the group's G execution phases proved, a session a
member. With `phase_s` = G · T · tick:

- proving seconds for each second of execution phase: (the sessions' seconds + staging) / `phase_s`. A session's
  seconds are its prover process's, from its start to its exit, the ZK server's inside them and the live verifier's
  replies inside its waits. Staging is each member's statement staged for its session. Lowering the class's three
  Definitions is paid once a class, whatever the traffic, and is beside the number, not in it.
- the Lean verifier's CPU seconds for each second of execution phase, on one core: the CPU seconds (user and
  system) of the processes that verify the round's sessions / `phase_s`, each process pinned to one core. Its wall
  seconds are beside the number and are not it: the core is one of those the prover's work runs on, so the wall is
  contended. A scan whose verifier was not pinned has no such number.

**The stages** of a session's seconds, from its `prove.log` (the process's line and the prover's `LIVE` record), in
order and summing to the process's seconds: to Register (the process's seconds less `e2e_s`: its start, with its
statement built, up to its Register), Register to the repetitions (`e2e_s` − `tp_s`), level 0 (`prove_total_s` less
the repetitions), the repetitions (the sum of `buckets[*]."t.prove"`), and the tail (`tp_s` − `prove_total_s`). The
Lean verifier's, from its `lean-verify.log`: reading its inputs (`inputs:`), the sampled statement's setup (`setup`),
and the session's checks (the session's line less the setup, which it includes).

**The identity** is what two results must share to compare (`IDENTITY`): the class, the sample (how many of the
round's units its answers say were proved), T and the tick, the traffic, the cores the job was pinned to and the Lean
verifier's. The group is a setting, beside the identity: the numbers are for a second of execution phase at any group.

**The guard**: the scan exited 0, and each execution phase's answer accepts with the public part `public.json` pins
for the class (the bound 0, every block sampled and proved, every verdict run; over a group, G times the blocks).
Where it does not hold, no measurement is written: the document says why and the run's validation fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from verity_bench import kit, settings

HERE = Path(__file__).resolve().parent
DOC, SCHEMA, BASELINE_SCHEMA = "scan_cost.json", "scan-cost/v1", "scan-cost/baseline/v1"
PUBLIC, BASELINE = HERE / "public.json", HERE / "baseline.json"
#: what two results share to compare
IDENTITY = ("class", "sample", "ticks", "tick_ms", "source", "cores", "verifier_cores")
#: the metrics' ids (benchmarks/metrics.toml), the names their measurements are written under
PROVING, VERIFIER = "network-accounting/scan-proving-cost", "network-accounting/scan-verifier-cost"
#: a session's stages, in order, and the Lean verifier's
STAGES = ("to_register_s", "register_to_reps_s", "level0_s", "reps_s", "tail_s")
LEAN_STAGES = ("inputs_s", "setup_s", "check_s")
ROUND = "honest"
#: the round's seconds, each a measurement: its name and its key in a document's `round`
PARTS = (("sessions", "sessions_s"), ("staging", "staging_s"), ("to Register", "to_register_s"),
         ("Register to the repetitions", "register_to_reps_s"), ("level 0", "level0_s"), ("the repetitions", "reps_s"),
         ("the tail", "tail_s"), ("Lean verifier CPU", "lean_cpu_s"), ("Lean verifier wall", "lean_s"),
         ("Lean verifier wall: inputs", "lean_inputs_s"), ("Lean verifier wall: setup", "lean_setup_s"),
         ("Lean verifier wall: checks", "lean_check_s"))
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
NOT_PINNED = "the Lean verifier was not pinned to one core (scan_zk.sh's SC_LEAN_ONE): its seconds are no core's"
NO_LINE = "lean.jsonl has no line for a session's verify process: its CPU seconds are not known"


class Refused(Exception):
    """A document that can't be read or written, or two that can't be compared; its text says why and what to do."""


# ---- reading a scan's kept files ---------------------------------------------------------------------------------------------


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{path} can't be read ({e}): the scan keeps it when it reaches its end") from None


def _seconds(text: str) -> float | None:
    """A kept log's process seconds (`# rc N in S s`)."""
    m = re.search(r"^# rc -?\d+ in ([0-9.]+) s", text, re.M)
    return float(m.group(1)) if m else None


def _ms(text: str, line: str) -> float | None:
    found = re.findall(line, text, re.M)
    return int(found[-1]) / 1000 if found else None


def prover(log: str) -> dict:
    """A session's `prove.log` by stage (module docstring): each figure a field of the prover's LIVE line or of the
    log's process line, or a difference of two. A log without them gives no stage."""
    live = next((json.loads(x[5:]) for x in log.splitlines() if x.startswith("LIVE\t")), None) or {}
    process = _seconds(log)
    reps = [b["t.prove"] for b in live.get("buckets") or [] if isinstance(b, dict) and "t.prove" in b]
    need = [process, *(live.get(k) for k in ("e2e_s", "tp_s", "prove_total_s"))]
    if None in need or not reps:
        return {"process_s": process}
    built = re.search(r"^statement \S+ .*? built in ([0-9.]+)s", log, re.M)
    out = {"process_s": process, "to_register_s": process - live["e2e_s"], "register_to_reps_s": live["e2e_s"] - live["tp_s"],
           "level0_s": live["prove_total_s"] - sum(reps), "reps_s": sum(reps), "tail_s": live["tp_s"] - live["prove_total_s"],
           "statement_built_s": float(built.group(1)) if built else None, "witness_s": live.get("witness_s"),
           "wait_s": live.get("wait_e2e_s"),
           **{name: sum(b.get(key) or 0 for b in live["buckets"])
              for name, key in (("upstream_s", "t.upstream"), ("zk_inner_s", "t.zk_inner"), ("pads_s", "t.zk_pads_commit"))}}
    return {k: None if v is None else round(v, 3) for k, v in out.items()}


def lean(log: str) -> dict:
    """A session's `lean-verify.log` by stage: its process's seconds, reading its inputs, the sampled statement's
    setup, and the session's checks (the session's line less the setup, which it holds)."""
    process, inputs = _seconds(log), _ms(log, r"^inputs: (\d+) ms")
    setup, session = _ms(log, r"^setup(?: \([^)]*\))?: (\d+) ms"), _ms(log, r"^\S*sessions/\S+: (\d+) ms")
    check = None if setup is None or session is None else round(session - setup, 3)
    return {"process_s": process, "inputs_s": inputs, "setup_s": setup, "check_s": check}


def _lines(path: Path) -> list[dict]:
    """A kept `.jsonl` file's lines; none where the scan kept no such file."""
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _cores(out: Path) -> int | None:
    """The cores the scan had, as its own `host.txt` says (`threads N`: what its provers size their threads by)."""
    host = out / "host.txt"
    m = re.search(r"^threads (\d+)$", host.read_text(encoding="utf-8"), re.M) if host.is_file() else None
    return int(m.group(1)) if m else None


def _verifier(out: Path) -> tuple[dict, dict]:
    """The Lean verifier's `verify` processes as `lean.jsonl` holds them: the cores each was pinned to (1, or None
    where the scan pinned none) with their wall and CPU seconds over the whole scan, and each one's line by the
    session archive it verified."""
    rows = [r for r in _lines(out / "lean.jsonl") if r.get("command") == ["verify"]]
    pinned = bool(rows) and all(isinstance(r.get("core"), int) for r in rows)
    return ({"cores": 1 if pinned else None, "processes": len(rows),
             "wall_s": round(sum(r.get("wall_s") or 0 for r in rows), 2),
             "cpu_s": round(sum(r.get("cpu_s") or 0 for r in rows), 2)},
            {r["archive"]: r for r in rows if r.get("archive")})


def _peak_gb(out: Path) -> float | None:
    """The largest flock-circuit process's peak RSS (`rss.jsonl`, SC_RSS), in GiB."""
    return max((r["peak_gb"] for r in _lines(out / "rss.jsonl") if r.get("peak_gb") is not None), default=None)


def sessions(out: Path, summary: dict, verified: dict) -> list[dict]:
    """The honest round's sessions in its members' order, each with its summary's figures, its logs' stages and its
    Lean verifier process's CPU seconds (`verified`, `_verifier`'s lines by archive)."""
    rnd, members = summary["rounds"][ROUND], summary["members"][ROUND]
    blocks = {m["name"]: m["n"] for m in members}
    rows = []
    for name, t in rnd["audit"]["timing"].items():
        logs = out / "sczk" / ROUND / f"session-{name}"
        text = {f: (logs / f).read_text(encoding="utf-8", errors="replace") if (logs / f).is_file() else ""
                for f in ("prove.log", "lean-verify.log")}
        live = t.get("timing") or {}
        rows.append({"member": name, "blocks": blocks.get(name), "k_log": live.get("k_log"), "dense_m": live.get("dense_m"),
                     "session_s": t["session_s"], "stage_s": (rnd.get(name) or {}).get("stage_s"),
                     "lean_verify_s": t.get("lean_verify_s"),
                     "lean_cpu_s": (verified.get(f"sczk/{ROUND}/session-{name}/archive") or {}).get("cpu_s"),
                     "prover": prover(text["prove.log"]), "lean": lean(text["lean-verify.log"])})
    order = {m["name"]: i for i, m in enumerate(members)}
    return sorted(rows, key=lambda r: order.get(r["member"], len(order)))


# ---- the guard ---------------------------------------------------------------------------------------------------------------


def pinned(cls: str, group: int, pins: dict | None = None) -> dict:
    """The public part every execution phase's answer must have for a scan of `cls` at a group of `group`:
    `public.json`'s copy for the class, each of its lists of an execution phase's blocks (0 to n − 1) over `group`
    times the blocks (a group's answer is one answer over all its blocks)."""
    pins = _json(PUBLIC) if pins is None else pins
    if cls not in pins:
        raise Refused(f"{PUBLIC.name} pins no public part for the class {cls} (it has {', '.join(sorted(pins))}): add "
                      "the class's, from an answer checked by hand")
    return {k: list(range(group * len(v))) if isinstance(v, list) and v == list(range(len(v))) else v
            for k, v in pins[cls].items()}


def _phases(out: Path) -> list[dict]:
    """The answers the scan kept, one an execution phase of the group (`answers.json`)."""
    return _json(out / "sczk" / "answers.json").get("windows") or []


def sample(summary: dict, phases: list[dict]) -> str | None:
    """How much of the round its answers say was proved: every unit, or how many of the round's; None with no
    answer to read it from."""
    units = sum(m["n"] for m in summary["members"][ROUND])
    proved = {len(((w.get("answer") or {}).get("public") or {}).get("proved") or []) for w in phases}
    if not proved:
        return None
    return "every unit" if proved == {units} else f"{min(proved)} of {units} units"


def guard(out: Path, summary: dict, scan_exit: int, pins: dict | None = None) -> dict:
    """Whether the measurement stands (module docstring), and each reason it does not."""
    why = [] if scan_exit == 0 else [f"the scan exited {scan_exit}"]
    group = summary["group"]
    try:
        want = pinned(summary["setup"], group, pins)
        phases = _phases(out)
    except Refused as e:
        return {"holds": False, "scan_exit": scan_exit, "why": [*why, str(e)]}
    if len(phases) != group:
        why.append(f"answers.json holds {len(phases)} execution phases' answers, and the group is {group}")
    for i, w in enumerate(phases):
        answer = w.get("answer") or {}
        if answer.get("verdict") != "accept":
            why.append(f"execution phase {i}'s answer is {answer.get('verdict')!r}: {answer.get('reason')}")
        elif answer.get("public") != want:
            got = answer.get("public") or {}
            keys = sorted(k for k in {*want, *got} if want.get(k) != got.get(k))
            why.append(f"execution phase {i}'s public part is not {PUBLIC.name}'s for {summary['setup']} in {', '.join(keys)}")
    return {"holds": not why, "scan_exit": scan_exit, "why": why}


# ---- the document ------------------------------------------------------------------------------------------------------------


def _sum(rows: list[dict], *keys: str) -> float | None:
    """The sum over `rows` of the figure at `keys`, None where a row lacks it."""
    vals = []
    for r in rows:
        for k in keys:
            r = (r or {}).get(k)
        vals.append(r)
    return None if None in vals or not vals else round(sum(vals), 3)


def document(out: Path, scan_exit: int, run: str | None = None, commit: str | None = None, pins: dict | None = None) -> dict:
    """A scan's kept files under `out` as the benchmark's document: its identity and settings, the guard, each
    session with its stages, the round's sums, and the numbers (None where the round does not give one)."""
    summary = _json(out / "sczk" / "summary.json")
    if not isinstance((summary.get("rounds") or {}).get(ROUND), dict) or "audit" not in summary["rounds"][ROUND]:
        raise Refused(f"{out / 'sczk' / 'summary.json'} holds no {ROUND} round with its audit: the scan did not reach it")
    try:
        return _document(out, summary, scan_exit, run, commit, pins)
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"{out}'s kept files lack what the {ROUND} round is read by ({type(e).__name__}: {e}): the scan "
                      "did not finish the round") from None


def _document(out: Path, summary: dict, scan_exit: int, run: str | None, commit: str | None, pins: dict | None) -> dict:
    group, ticks, tick_ms = summary["group"], summary["T"], summary["params"]["bucket_ms"]
    phase_s = group * ticks * tick_ms / 1000
    verifier, verified = _verifier(out)
    rows = sessions(out, summary, verified)
    answers = _phases(out) if (out / "sczk" / "answers.json").is_file() else []
    rnd = {"sessions_s": _sum(rows, "session_s"), "staging_s": _sum(rows, "stage_s"),
           **{k: _sum(rows, "prover", k) for k in STAGES}, "lean_cpu_s": _sum(rows, "lean_cpu_s"),
           "lean_s": _sum(rows, "lean_verify_s"), **{f"lean_{k}": _sum(rows, "lean", k) for k in LEAN_STAGES}}
    one = verifier["cores"] == 1 and rnd["lean_cpu_s"] is not None
    proving = None if None in (rnd["sessions_s"], rnd["staging_s"]) else rnd["sessions_s"] + rnd["staging_s"]
    lowering = {k.split("/", 1)[1]: v.get("lower_s") for k, v in (summary.get("stagers") or {}).items() if k.startswith("joint/")}
    return {
        "schema": SCHEMA, "run": run, "commit": commit,
        "identity": {"class": summary["setup"], "sample": sample(summary, answers), "ticks": ticks, "tick_ms": tick_ms,
                     "source": summary["source"], "cores": _cores(out), "verifier_cores": verifier["cores"]},
        "group": group, "phase_s": phase_s, "guard": guard(out, summary, scan_exit, pins),
        "sessions": rows, "round": rnd, "lowering_s": lowering, "verifier": verifier, "peak_gb": _peak_gb(out),
        "numbers": {"proving": None if proving is None else round(proving / phase_s, 3),
                    "proving_s_a_tick": None if proving is None else round(proving / (group * ticks), 3),
                    "verifier": round(rnd["lean_cpu_s"] / phase_s, 3) if one else None,
                    "verifier_wall": None if rnd["lean_s"] is None else round(rnd["lean_s"] / phase_s, 3)}}


def no_verifier(doc: dict) -> str | None:
    """Why a document has no verifier number, None where it has one."""
    if doc["numbers"]["verifier"] is not None:
        return None
    return NOT_PINNED if doc["identity"]["verifier_cores"] != 1 else NO_LINE


def measurements(doc: dict) -> list[dict]:
    """A document's numbers as the kit's writer takes them: the two metrics under their ids, each naming the
    measurements it is computed from, then those: the round's seconds by stage and the group's execution phases'."""
    n, r = doc["numbers"], doc["round"]
    stages = [name for name, key in PARTS[2:7] if r[key] is not None]
    out = [{"name": PROVING, "value": n["proving"], "unit": "proving seconds a second of execution phase",
            "inputs": ["sessions", "staging", "execution phases", *stages]},
           {"name": "proving seconds a tick", "value": n["proving_s_a_tick"], "unit": "s",
            "inputs": ["sessions", "staging"]},
           {"name": VERIFIER, "value": n["verifier"],
            "unit": "Lean verifier CPU seconds a second of execution phase, on one core",
            "inputs": [] if n["verifier"] is None else ["Lean verifier CPU", "execution phases"],
            **({"why": why} if (why := no_verifier(doc)) else {})},
           {"name": "Lean verifier wall seconds a second of execution phase", "value": n["verifier_wall"], "unit": "s",
            "inputs": ["Lean verifier wall", "execution phases"]},
           {"name": "execution phases", "value": doc["phase_s"], "unit": "s", "inputs": []}]
    out += [{"name": name, "value": r[key], "unit": "s a round", "inputs": []} for name, key in PARTS if r[key] is not None]
    return [m for m in out if m["value"] is not None or "why" in m]


# ---- write -------------------------------------------------------------------------------------------------------------------


def _put(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write(run_dir: Path, scan_exit: int) -> tuple[dict | None, bool, str]:
    """The run's records of a scan whose kept files are under `run_dir/out`: `out/scan_cost.json` and the run's
    `result.json`, its validation the guard and its fingerprint the identity with the group, then, where the result
    stands (the guard holds and the round has its proving seconds), each measurement through the kit's writer.
    Returns the document (None where the scan left no round), whether the result stands, and the validation's line."""
    out, run = run_dir / "out", settings.RESEARCH_RUN_ID
    stamp = kit.host_stamp()
    try:
        doc = document(out, scan_exit, run, stamp["commit"])
    except Refused as e:
        doc, holds, detail = None, False, f"no round to measure: {e}"
    else:
        _put(out / DOC, doc)
        holds = doc["guard"]["holds"] and doc["numbers"]["proving"] is not None
        detail = "the guard holds" if holds else "the guard does not hold: " + "; ".join(doc["guard"]["why"]) \
            if not doc["guard"]["holds"] else "the guard holds, and the summary lacks a session's seconds or staging: no number"
    kept = [f"out/{DOC}", "out/sczk/answers.json"] if doc is not None else []
    _put(run_dir / kit.RESULT, {
        "schema": kit.RESULT_SCHEMA, "run_id": run,
        "workload_fingerprint": {"benchmark": "scan_cost", **({**doc["identity"], "group": doc["group"]} if doc else {})},
        "validation": {"status": "passed" if holds else "failed", "checker": "benchmarks/scan_cost/result.py",
                       "detail": detail, "evidence": kept},
        "measurements": [], "measurement_files": kept[:1], "artifacts": []})
    if holds:
        writer = kit.Writer(run_dir, run, stamp)
        for m in measurements(doc):
            writer.measurement(**m)
    return doc, holds, detail


# ---- table, compare, baseline ------------------------------------------------------------------------------------------------


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def table(doc: dict) -> str:
    """A document as text: its identity, the guard, the numbers, then the round, a session a line."""
    i, n, r = doc["identity"], doc["numbers"], doc["round"]
    head = ["member", "blocks", "session", "staging", "to Register", "Register to reps", "level 0", "reps", "tail",
            "Lean CPU", "Lean wall", "inputs", "setup", "checks"]
    rows = [[s["member"], str(s["blocks"]), _f(s["session_s"]), _f(s["stage_s"]), *(_f(s["prover"].get(k)) for k in STAGES),
             _f(s.get("lean_cpu_s")), _f(s["lean_verify_s"]), *(_f(s["lean"].get(k)) for k in LEAN_STAGES)]
            for s in doc["sessions"]]
    rows.append(["the round", str(sum(s["blocks"] or 0 for s in doc["sessions"])), _f(r["sessions_s"]), _f(r["staging_s"]),
                 *(_f(r[k]) for k in STAGES), _f(r["lean_cpu_s"]), _f(r["lean_s"]),
                 *(_f(r[f"lean_{k}"]) for k in LEAN_STAGES)])
    width = [max(len(x[c]) for x in [head, *rows]) for c in range(len(head))]
    lines = [
        f"scan_cost: run {doc['run'] or 'local'}, commit {doc['commit'] or 'unknown'}",
        f"  identity: class {i['class']}, {i['sample']}, T = {i['ticks']} ticks of {i['tick_ms']} ms, {i['cores']} cores, "
        f"the Lean verifier on {i['verifier_cores'] or 'cores not pinned'}; {i['source']}",
        f"  group {doc['group']}: {doc['phase_s']:g} s of execution phase a round",
        "  guard: " + ("holds" if doc["guard"]["holds"] else "does not hold: " + "; ".join(doc["guard"]["why"])),
        f"  proving seconds a second of execution phase: {_f(n['proving'])} ({_f(n['proving_s_a_tick'])} s a tick)",
        f"  Lean verifier CPU seconds a second of execution phase, on one core: {_f(n['verifier'])}"
        + (f" ({why})" if (why := no_verifier(doc)) else "") + f"; its wall beside it: {_f(n['verifier_wall'])}",
        "  the round, in seconds (the Lean verifier's inputs, setup and checks are its wall's):",
        *("  " + "  ".join(c.rjust(w) if k else c.ljust(w) for k, (c, w) in enumerate(zip(x, width, strict=True)))
          for x in [head, *rows])]
    return "\n".join(lines)


def same_identity(base: dict, cand: dict) -> None:
    """Refused unless the identities `base` and `cand` are one, naming each field that differs."""
    differs = [f"{f}: {base.get(f)!r} and {cand.get(f)!r}" for f in IDENTITY if base.get(f) != cand.get(f)]
    if differs:
        raise Refused("the two results' identities differ, so their numbers don't compare (" + "; ".join(differs)
                      + "): run the candidate at the baseline's identity (run.sh's header)")


def compare(base: dict, cand: dict) -> str:
    """A candidate beside a baseline, as text; Refused where their identities differ or either's guard does not hold,
    since a number then says nothing about the other."""
    same_identity(base["identity"], cand["identity"])
    for what, d in (("baseline", base), ("candidate", cand)):
        if not d["guard"]["holds"]:
            raise Refused(f"the {what}'s guard does not hold ({'; '.join(d['guard']['why'])}): it has no number to compare")
    lines = [f"scan_cost: {cand['run'] or 'local'} (group {cand['group']}) beside {base['run'] or 'local'} (group "
             f"{base['group']}), at one identity"]
    for name, key in (("proving seconds a second of execution phase", "proving"),
                      ("Lean verifier CPU seconds a second of execution phase, on one core", "verifier")):
        b, c = base["numbers"][key], cand["numbers"][key]
        ratio = "" if None in (b, c) or not b else f", {c / b:.3f} of the baseline"
        lines.append(f"  {name}: {_f(c)} beside {_f(b)}{ratio}")
    lines.append("  a second of execution phase by stage, in seconds, the candidate beside the baseline:")
    for name, key in PARTS:
        b, c = (None if d["round"][key] is None else d["round"][key] / d["phase_s"] for d in (base, cand))
        lines.append(f"    {name}: {_f(c)} beside {_f(b)}")
    return "\n".join(lines)


def fetched(run: str, call: Callable = subprocess.run) -> dict:
    """Run `run`'s document from the evidence store: its run record's `out/scan_cost.json`."""
    shown = call(["research", "data", "show", run, "--json"], capture_output=True, text=True)
    try:
        doc = json.loads(shown.stdout)
        art = (doc.get("attempt") or doc)["outputs"]["run_record"]
    except (ValueError, KeyError, TypeError, AttributeError):
        raise Refused(f"`research data show {run} --json` names no run record ({shown.stderr.strip()[-200:] or 'no output'}): "
                      "run it from a checkout whose environment has `research` (`uv run`)") from None
    with tempfile.TemporaryDirectory() as d:
        got = call(["research", "data", "fetch", art, "--to", d, "--path", f"out/{DOC}"], capture_output=True, text=True)
        path = Path(d) / "out" / DOC
        if not path.is_file():
            raise Refused(f"{run}'s run record holds no out/{DOC} ({got.stderr.strip()[-200:] or 'nothing fetched'}): it is "
                          "no run of benchmarks/scan_cost/run.sh that reached its result")
        return _json(path)


def opened(ref: str, call: Callable = subprocess.run) -> dict:
    """The document `ref` names: a file, a directory holding one, or a run's id."""
    path = Path(ref)
    for p in (path, path / DOC, path / "out" / DOC):
        if p.is_file():
            doc = _json(p)
            break
    else:
        if not RUN_ID.fullmatch(ref):
            raise Refused(f"{ref} is no {DOC}, no directory holding one and no run's id")
        doc = fetched(ref, call)
    if doc.get("schema") != SCHEMA:
        raise Refused(f"{ref} is no {SCHEMA} document: it is written by `result.py write`")
    return doc


def baseline(path: Path | None = None) -> dict:
    """The baseline: the run every candidate is compared with, its commit, identity and group. No number."""
    path = BASELINE if path is None else path
    if not path.is_file():
        raise Refused(f"{path} doesn't exist: no baseline is pinned yet (`result.py baseline --lock DOC` pins one)")
    doc = _json(path)
    if doc.get("schema") != BASELINE_SCHEMA or not RUN_ID.fullmatch(str(doc.get("run"))):
        raise Refused(f"{path} is no {BASELINE_SCHEMA} file naming a run: `result.py baseline --lock DOC` writes one")
    return doc


def lock(doc: dict, path: Path | None = None) -> dict:
    """`doc`'s run written as the baseline; Refused for a document with no run, whose guard does not hold, or that
    lacks either number: a candidate is compared with the baseline on both."""
    path = BASELINE if path is None else path
    if not RUN_ID.fullmatch(str(doc.get("run"))) or not doc["guard"]["holds"]:
        raise Refused("a baseline is a run whose guard holds: this document "
                      + ("names no run" if doc["guard"]["holds"] else "says " + "; ".join(doc["guard"]["why"])))
    absent = [name for name, key in ((PROVING, "proving"), (VERIFIER, "verifier")) if doc["numbers"][key] is None]
    if absent:
        why = no_verifier(doc) if VERIFIER in absent else None
        raise Refused(f"a baseline has both numbers: this document has no {' and no '.join(absent)}"
                      + (f" ({why})" if why else ""))
    out = {"schema": BASELINE_SCHEMA, "run": doc["run"], "commit": doc["commit"], "identity": doc["identity"],
           "group": doc["group"]}
    _put(path, out)
    return out


def main(argv: list[str] | None = None, *, call: Callable = subprocess.run) -> int:
    """0 when the command did what it says; for `write`, 1 where the guard does not hold or the scan left no round;
    2 where a document can't be read or two can't be compared. `call` runs the `research` commands."""
    ap = argparse.ArgumentParser(prog="result.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="a scan's document and the run's result.json")
    w.add_argument("run_dir", type=Path)
    w.add_argument("--scan-exit", type=int, required=True, help="the exit of the scan that left out/sczk")
    sub.add_parser("table", help="a document's round, a session a line").add_argument("doc")
    c = sub.add_parser("compare", help="a candidate beside the baseline; refuses differing identities")
    c.add_argument("candidate")
    c.add_argument("--to", metavar="DOC", help="the document to compare with (default: baseline.json's run's)")
    b = sub.add_parser("baseline", help="the pinned baseline, or pin one")
    b.add_argument("--lock", metavar="DOC", help="write this document's run as the baseline")
    a = ap.parse_args(argv)
    try:
        if a.command == "write":
            doc, holds, detail = write(a.run_dir, a.scan_exit)
            print(f"scan_cost: {detail}" if doc is None else f"{table(doc)}\nscan_cost: {detail}")
            return 0 if holds else 1
        if a.command == "table":
            print(table(opened(a.doc, call)))
        elif a.command == "compare":
            cand = opened(a.candidate, call)
            if a.to is None:
                same_identity(baseline()["identity"], cand["identity"])   # the baseline file's, before its run is fetched
            print(compare(opened(a.to if a.to is not None else baseline()["run"], call), cand))
        else:
            print(json.dumps(lock(opened(a.lock, call)) if a.lock else baseline(), indent=1, sort_keys=True))
    except Refused as e:
        print(f"scan_cost: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
