#!/usr/bin/env python3
"""The wide level-0 commitment leaf against today's leaf on one card: `run.sh`'s Python, with the steady harness's
modules (`verity_bench.steady`, `steady_node`, `steady_cards`) on its path. Each command writes under the run's `out/`.

  wleaf.py lease --max-min M [--on CARD] -- CMD...
      CMD under one gpu-lease of the card CARD, else of one card (`kit.lease`) on the NUMA node of the run's CPUs
      (`steady_cards.cards_near`), refusing a CARD on another node,
      pinned to them, preemptible, as the steady harness's point takes its lease.
  wleaf.py host
      inside the lease: the kit's host stamp (`kit.host_stamp`, the lease's one card) and the card's NUMA node, to
      out/host.json, which `record` writes the result under.
  wleaf.py step TAG [--wide] --staged Z --type T --pin PIN --sessions N --warm W --ahead A
      one step on the leased card, as the steady harness's timed step runs it (`steady_node.timed`): the card's
      cotenants before and after (`steady_cards.snapshot`), `server_rate.py` at one prover (with `--wide-leaf` for a
      wide step), and the rows, the card's NUMA node and the sessions kept by `steady_node.keep` under out/steps/TAG/.
  wleaf.py record
      after the lease: each step's ten checks (`steady.verdict`) on its own files, out/summary.json, and the result
      (`record`): its validation and, where every step passed, the two measurements.

The steps interleave today's leaf and the wide one (a-own, b-wide, c-wide, d-own, e-own, f-wide) on the same statement,
so a drift of the card or the node over the lease reaches both alike.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

#: GPU-seconds a statement with the wide leaf over the same with today's leaf
TIME = "kernels/wide-leaf-gpu-seconds-over-today"
#: the proof's bytes with the wide leaf over the same with today's leaf
SIZE = "kernels/wide-leaf-proof-bytes-over-today"
HERE = Path(__file__).resolve().parent
#: the CPUs the steps are timed on, a GPU slot's: with fewer or more a ratio is another point's
CORES = 10
NOT_SAID = ("the prover's CPU witness sets the pace, so the ratio is of this card with this slot's cores; it doesn't say "
            "what share of a monitoring period's proving these statements carry; each step's acceptance is under the "
            "provisional unit cover (the statement staged unbound)")


def statement() -> dict[str, Any]:
    import tomllib
    return tomllib.loads((HERE / "statement.toml").read_text())


def lease(a: argparse.Namespace) -> int:
    from verity_bench import kit, steady_cards, steady_node
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != CORES:
        print(f"wleaf: the run has {len(cpus)} CPUs ({steady_node._spans(cpus)}), and the steps are timed on {CORES}, "
              "a GPU slot's: launch it through a slot of the node whose line gives that many", file=sys.stderr)
        return 2
    near = steady_cards.cards_near(cpus, steady_node._ask)
    if a.on is not None and near.get("cards") and a.on not in near["cards"]:
        print(f"wleaf: card {a.on} isn't on the NUMA node of the run's CPUs ({steady_node._spans(cpus)}), whose cards are "
              f"{near['cards']}: name a card and cores of one node", file=sys.stderr)
        return 2
    on = str(a.on) if a.on is not None else ",".join(map(str, near["cards"])) if near.get("cards") else None
    *argv, dashdash = kit.lease(1, a.max_min, on=on, cores=steady_node._spans(cpus))
    print(f"wleaf: one card {'of ' + on if on else '(the node says no card near the run CPUs: ' + str(near.get('why')) + ')'}"
          f", pinned to {len(cpus)} CPUs", flush=True)
    return subprocess.run([*argv, "--preemptible", dashdash, *a.command], check=False).returncode


def host(a: argparse.Namespace) -> int:
    from verity_bench import kit, settings, steady, steady_cards, steady_lease, steady_node
    out = Path(settings.RESEARCH_RUN_DIR) / "out"
    device = settings.CUDA_VISIBLE_DEVICES or ""
    stamp = kit.host_stamp()
    record = {"host": stamp, "device": device, "cpus": sorted(os.sched_getaffinity(0)),
              "versions": {"driver": stamp.get("driver"), "cuda_toolkit": steady_lease.toolkit_on_node(steady_node._ask)},
              steady.CARD_NUMA: steady_cards.card_numa_node(device, steady_node._ask)}
    (out / "host.json").write_text(json.dumps(record, indent=1))
    print(f"wleaf: {record['host'].get('card')} (device {device}, NUMA node {record[steady.CARD_NUMA]}), "
          f"driver {record['host'].get('driver')}", flush=True)
    return 0 if device and "," not in device else 2


def step(a: argparse.Namespace) -> int:
    from verity_bench import settings, steady, steady_cards, steady_node
    run_dir = Path(settings.RESEARCH_RUN_DIR)
    device = settings.CUDA_VISIBLE_DEVICES or ""
    if not device or "," in device:
        print(f"wleaf: a step runs under a gpu-lease of one card (CUDA_VISIBLE_DEVICES is {device!r})", file=sys.stderr)
        return 2
    sd, scratch = run_dir / "out" / "steps" / a.tag, Path(a.scratch) / f"rate-{a.tag}"
    sd.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    row, run_id = {"provers": 1}, settings.RESEARCH_RUN_ID
    steady_cards.snapshot(sd / Path(steady.card_file(row, "start")).name, run_id, device, steady_node._ask, _now)
    argv = [sys.executable, "benchmarks/one_stage/server_rate.py", "--bin", steady_node.prover()["path"],
            "--staged", a.staged, "--types", a.type, "--provers", "1", "--out", str(scratch), "--device", device,
            "--sessions", str(a.sessions), "--warm", str(a.warm), "--ahead", str(a.ahead),
            *(["--wide-leaf"] if a.wide else [])]
    with open(sd / "server-rate.log", "w") as log:
        rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    steady_cards.snapshot(sd / Path(steady.card_file(row, "end")).name, run_id, device, steady_node._ask, _now)
    item = SimpleNamespace(row=SimpleNamespace(name=a.type, period=None), provers=1)
    node = steady_cards.card_numa_node(device, steady_node._ask)
    read = json.loads((run_dir / "out" / "host.json").read_text())["versions"]
    n = steady_node.keep(scratch, sd / "rate", item, read, node)
    kept = steady_node.keep_session(scratch / f"{a.type}-k1", sd / "rate" / f"{a.type}-k1", a.pin)
    (sd / "step.json").write_text(json.dumps({"tag": a.tag, "wide": a.wide, "exit": rc, "rows": n, "kept": kept,
                                              "argv": argv[1:]}, indent=1))
    print(f"wleaf: step {a.tag}: exit {rc}, {n} row(s)", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return rc


def _now() -> float:
    import time
    return time.time()


def summarize(run_dir: Path, name: str) -> list[dict[str, Any]]:
    """Each step's row as `record` reads it: its GPU-seconds a statement, its proofs' bytes and its ten checks' verdict,
    each check on the step's own files (its rate under out/steps/TAG/rate, its card files beside it, the run's lease
    usage)."""

    from verity_bench import steady
    base, steps = steady.files_in(run_dir), []
    for sd in sorted((run_dir / "out" / "steps").glob("*")):
        tag = sd.name

        def files(path: str, tag: str = tag) -> str | None:
            if path.startswith("out/rate/"):
                path = f"out/steps/{tag}/rate/" + path[len("out/rate/"):]
            elif path.startswith("out/cotenants-"):
                path = f"out/steps/{tag}/" + path[len("out/"):]
            return base(path)

        st = json.loads((sd / "step.json").read_text()) if (sd / "step.json").is_file() else {}
        rp = sd / "rate" / "rate.json"
        rows = json.loads(rp.read_text()) if rp.is_file() else []
        sessions = sd / "rate" / f"{name}-k1" / "sessions.jsonl"
        lives = [json.loads(line).get("live") or {} for line in sessions.read_text().splitlines() if line.strip()] \
            if sessions.is_file() else []
        sizes = sorted({sum(live["proof_bytes"]) for live in lives
                        if not live.get("warm") and isinstance(live.get("proof_bytes"), list)})
        if len(rows) != 1:
            steps.append({"step": tag, "wide": bool(st.get("wide")), "status": steady.FAIL,
                          "why": f"the step kept {len(rows)} rows, not one (exit {st.get('exit')})"})
            continue
        v = steady.verdict(rows[0], files)
        rate = rows[0].get("statements_per_s")
        steps.append({"step": tag, "wide": bool(st.get("wide")), "status": v.status, "why": v.why,
                      "gpu_s": 1 / rate if rate else None, "proof_bytes": sizes,
                      "median_prove_total_s": (rows[0].get("median") or {}).get("prove_total_s"),
                      "accepted": rows[0].get("accepted"), "verdict": v.record()})
    return steps


def record(out: Path, steps: Sequence[Mapping[str, Any]], host: Mapping[str, Any], stage: Mapping[str, Any],
           model: str) -> None:
    """The run's result through the kit's writer under `host`, the stamp taken inside the lease: its validation (passed
    only where every step's ten checks pass, today's leaf and the wide one each timed and each step's proofs of one
    size) and then the two measurements, GPU-seconds a statement and proof bytes, each the wide leaf's median over
    today's, its member the model and the card's short name."""

    from verity_bench import kit
    w = dataclasses.replace(kit.writer(out), stamp=host)
    own = [s for s in steps if not s["wide"]]
    wide = [s for s in steps if s["wide"]]
    bad = [f"{s['step']}: {s['status']}" + (f" ({s['why']})" if s.get("why") else "") for s in steps
           if s["status"] != "pass" or not s.get("gpu_s") or len(s.get("proof_bytes") or []) != 1]
    if not own or not wide:
        bad.append(f"{len(own)} steps on today's leaf and {len(wide)} on the wide one, where each needs one or more")
    m = stage["k_log"] + math.ceil(math.log2(math.ceil(stage["n"] / stage.get("vus_per_block", 1))))
    w.validation("passed" if not bad else "failed", "steady.verdict",
                 f"{len(steps)} steps on {stage['definition']} x {stage['n']} (m = {m}): "
                 + ("each step's ten checks pass" if not bad else "; ".join(bad)), ["out/summary.json"])
    if bad:
        return
    card = "-".join(host["card"].lower().split()).removeprefix("nvidia-")
    own_s, wide_s = [s["gpu_s"] for s in own], [s["gpu_s"] for s in wide]
    own_b, wide_b = own[0]["proof_bytes"][0], wide[0]["proof_bytes"][0]
    common = {"m": m, "k_log": stage["k_log"], "n": stage["n"], "definition": stage["definition"], "provers": 1,
              "cores": len(host.get("cpus") or []) or None, "not_said": NOT_SAID}
    w.measurement(TIME, statistics.median(wide_s) / statistics.median(own_s), "times today's leaf", mark="measured",
                  member=f"{model}-{card}",
                  extra={**common, "own_gpu_s": own_s, "wide_gpu_s": wide_s,
                         "pairs": [round(b / a, 4) for a, b in zip(own_s, wide_s, strict=False)]})
    w.measurement(SIZE, wide_b / own_b, "times today's proof", mark="measured", member=f"{model}-{card}",
                  extra={**common, "own_proof_bytes": own_b, "wide_proof_bytes": wide_b})


def finish(a: argparse.Namespace) -> int:
    from verity_bench import settings
    run_dir = Path(settings.RESEARCH_RUN_DIR)
    out, st = run_dir / "out", statement()
    steps = summarize(run_dir, st["name"])
    host = json.loads((out / "host.json").read_text())
    stage = json.loads((Path(a.stage) / "stage.json").read_text())
    (out / "summary.json").write_text(json.dumps({"statement": st, "host": host, "steps": steps}, indent=1, default=str))
    for s in steps:
        print(f"wleaf: {s['step']} {'wide' if s['wide'] else 'own'}: {s.get('gpu_s')} GPU-s a statement, proof bytes "
              f"{s.get('proof_bytes')}, {s['status']}" + (f": {s['why'][:300]}" if s.get("why") else ""), flush=True)
    record(out, steps, {**host["host"], "cpus": host.get("cpus")}, stage, st["model"])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lease")
    s.add_argument("--max-min", type=int, required=True)
    s.add_argument("--on", type=int, help="the card to lease, by index (else any card near the run's CPUs)")
    s.add_argument("command", nargs="+")
    sub.add_parser("host")
    s = sub.add_parser("step")
    s.add_argument("tag")
    s.add_argument("--wide", action="store_true")
    for name in ("staged", "type", "pin", "scratch"):
        s.add_argument(f"--{name}", required=True)
    for name in ("sessions", "warm", "ahead"):
        s.add_argument(f"--{name}", type=int, required=True)
    s = sub.add_parser("record")
    s.add_argument("--stage", required=True, help="the statement's directory, holding its stage.json")
    a = p.parse_args()
    return {"lease": lease, "host": host, "step": step, "record": finish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
