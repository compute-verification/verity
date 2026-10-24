"""Prove path: a committed TP run and its advice through the proof service, for network accounting under arbitrary vLLM usage.

Each rank's step units are Programs a registered compiler made from the advice (`pipeline.advice_windows`), not a declared
workload's. The prover commits each rank's window over them from the rank's sealed bundle, its registration binding the compiler's
and the advice's digests. The verifier compiles the same advice with its own copy of the compiler, derives its own window and
audits the rank through the one-stage audit's per-rank `service.Stream` (`benchmarks/one_stage/a5.py`'s `audit`): register,
select on the verifier's own live coins (read only after the receipt), C-Flock `--zk` behind the Lean firewall on the drawn
units, Lean's verdicts, and the outcome, which the Lean verifier of record's `Run.run` must agree with. README.md states what
changes when Programs come from advice.

    prove_path.py advice --workload F --out DIR
    prove_path.py commit --tp-commit DIR --build-root DIR --advice DIR --partition F --out WINDOWS [--ranks 0,1] [--jobs N] \\
        [--forge MEMBER:UNIT]
    prove_path.py audit --tp-commit DIR --build-root DIR --advice DIR --windows WINDOWS --partition F --template-sets DIR \\
        --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --firewall flock-firewall --out DIR \\
        [--ranks 0,1] [--jobs N] [--forged]

* **advice**: a declared run's advice from its workload file (`declared_advice`), ``DIR/advice.json``. A served run's advice
  comes from its step trace instead (`verity-vllm advice`, I1); this driver reads either through `read_advice`.
* **commit** (the prover): each rank's stand-in window from its bundle over the compiler's Programs for the advice
  (`advice_windows.commit_window`), ``WINDOWS/rank<r>/``. ``--forge MEMBER:UNIT`` commits that unit with bit 0 of its first
  output word flipped (`forging`): its rows, commit strings and roots all agree with the flipped word, so only the proof can
  refuse it.
* **audit** (the verifier): each rank's window from the record, the advice and its own compiler (`advice_windows.
  window_of_advice`), audited by a5 on the verifier's coins, with the advice's negatives (`advice_negatives`) beside a5's;
  ``OUT/summary.json`` over the ranks. ``--forged``: the prover runs with its self-check skipped (``FC_ZK_SELF_CHECK=skip``) so it
  sends proofs of the false witness, and a rank passes when its registration passed every check before the draw and Lean
  refused the outcome (`forged_row`).

The compiler is `advice_windows.Traced`, the Build's Programs of record, refused unless of record, until the ahead-of-time compiler
(`verity_vllm.predict.steps.compile_steps`) makes a TP rank's Programs. The advice's encoding is its canonical JSON
(`encode`) until I1's encoding lands; its digest is SHA-512 over the encoding and its length the encoding's bits.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_vllm.pipeline import advice_windows as AW  # noqa: E402
from verity_vllm.pipeline.tp import serving_rows as SRT  # noqa: E402

ADVICE = "advice.json"


def encode(advice: dict) -> bytes:
    return canonical_json(advice)


def read_advice(d: str | os.PathLike) -> tuple[dict, bytes]:
    """The advice in ``d`` (``advice.json``) and its encoding, refused unless the file is that encoding."""
    data = (Path(d) / ADVICE).read_bytes()
    doc = json.loads(data)
    if encode(doc) != data:
        raise ValueError(f"{Path(d) / ADVICE} is not the advice's canonical encoding")
    return doc, data


def declared_advice(workload_file: str | os.PathLike) -> dict:
    """A declared run's advice in I1's fields: request i is ``r<i>`` (tp-commit's name for it), its prompt whole at its arrival
    step from position 0, then one decode row a step until its cap, finishing by length."""
    doc = json.loads(Path(workload_file).read_text())
    reqs, rows = [], {}
    for i, q in enumerate(doc["requests"]):
        lp, cap, s0 = len(q["prompt_token_ids"]), int(q["max_tokens"]), int(q.get("arrive_step") or 0)
        reqs.append({"id": f"r{i}", "prompt_len": lp, "output_len": cap, "finish": "length"})
        rows.setdefault(s0, []).append({"request": i, "kind": "prefill", "start": 0, "n": lp, "cache_hit": 0, "preempted": False})
        for k in range(1, cap):
            rows.setdefault(s0 + k, []).append({"request": i, "kind": "decode", "start": lp + k - 1, "n": 1, "cache_hit": 0,
                                                "preempted": False})
    return {"requests": reqs, "steps": [{"rows": rows.get(s, [])} for s in range(max(rows) + 1 if rows else 0)], "release": []}


def requests_of(advice: dict) -> list[dict]:
    """The compiler's requests (`advice_windows`: ``request_id``, ``arrive_step``, ``prompt_len``, ``max_tokens``) from the advice:
    each request that ran a row, arriving at its first row's step, at the shape its rows ran (``max_tokens`` one more than its
    decode rows, which include any trailing rows past its finish). Refused by name where the rows are not a whole prompt from
    position 0 and then one decode row a step, as I2 v0 refuses them (`predict.steps.shape_of`): a prefix-cache hit, a chunked
    prefill, a preemption, a gap or a recompute."""
    reqs = advice["requests"]
    ran: dict[int, list[tuple[int, dict]]] = {i: [] for i in range(len(reqs))}
    for s, step in enumerate(advice["steps"]):
        for row in step["rows"]:
            i = int(row["request"])
            if not 0 <= i < len(reqs):
                raise ValueError(f"step {s}: a row of request {i}, outside the advice's {len(reqs)} requests")
            if row.get("preempted"):
                raise ValueError(f"step {s}: request {reqs[i]['id']} preempted (each request runs from its prefill without preemption)")
            ran[i].append((s, row))
    out = []
    for i, q in enumerate(reqs):
        if not ran[i]:
            continue
        lp, (s0, first) = int(q["prompt_len"]), ran[i][0]
        if first.get("cache_hit") or first["start"]:
            raise ValueError(f"request {q['id']}: a prefix-cache hit at step {s0} (prefix caching must be off)")
        if first["n"] != lp:
            raise ValueError(f"request {q['id']}: a chunked prefill, {first['n']} of its {lp} prompt tokens at step {s0}")
        for k, (s, row) in enumerate(ran[i][1:], start=1):
            if (s, row["start"], row["n"]) != (s0 + k, lp + k - 1, 1):
                raise ValueError(f"request {q['id']}: its row {k} is (step, start, n) = {(s, row['start'], row['n'])}, not the decode "
                                 f"{(s0 + k, lp + k - 1, 1)}")
        out.append({"request_id": str(q["id"]), "arrive_step": s0, "prompt_len": lp, "max_tokens": len(ran[i])})
    if not out:
        raise ValueError("no request of the advice ran a row")
    return out


def compiled(tp_commit: str, build_root: str, advice_dir: str) -> tuple[Any, list[dict], dict]:
    """The registered compiler, the requests it compiles from the advice, and the window's binding (`advice_windows.binding`)."""
    compiler = AW.Traced.for_commit(tp_commit, build_root)
    advice, data = read_advice(advice_dir)
    return compiler, requests_of(advice), AW.binding(compiler, hashlib.sha512(data).hexdigest(), 8 * len(data))


@contextlib.contextmanager
def forging(member: int, unit: int) -> Iterator[None]:
    """Inside, serving rows commit member ``member``'s unit ``unit`` with bit 0 of its first output word flipped, before its commit
    strings and roots are made over it."""
    from verity_vllm.pipeline import serving_rows as SRW

    real = SRW.read_all

    def read_all(ms, rows, store, weights, arrivals=None):
        got = real(ms, rows, store, weights, arrivals)
        port = ms[member].outs[0][0]
        got[member][0]["outs"][port][unit, 0] ^= 1
        return got

    SRW.read_all = read_all
    try:
        yield
    finally:
        SRW.read_all = real


def ranks_of(tp_commit: str, ranks: str | None) -> tuple[int, list[int]]:
    world = int(SRT.record(tp_commit)["world"])
    return world, [int(x) for x in ranks.split(",")] if ranks else list(range(world))


def in_processes(fn, jobs: int, items: list) -> Iterator[tuple[Any, Any]]:
    """``fn(item)`` of each item, in up to ``jobs`` spawned processes (a rank's Programs don't pickle, so a process does a rank);
    -> (item, result) as each finishes."""
    if jobs <= 1 or len(items) <= 1:
        for x in items:
            yield x, fn(x)
        return
    with ProcessPoolExecutor(min(jobs, len(items)), mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(fn, x): x for x in items}
        for f in as_completed(futures):
            yield futures[f], f.result()


def commit_rank(job: tuple[argparse.Namespace, int]) -> dict:
    a, r = job
    compiler, requests, bound = compiled(a.tp_commit, a.build_root, a.advice)
    forge = [int(x) for x in a.forge.split(":")] if a.forge else None
    with forging(*forge) if forge else contextlib.nullcontext():
        got = AW.commit_window(a.tp_commit, r, a.partition, Path(a.out) / f"rank{r}", compiler, requests, bound, workers=a.workers)
    return {**got, "forged": {"member": forge[0], "unit": forge[1]} if forge else None}


def commit(a: argparse.Namespace) -> int:
    _world, ranks = ranks_of(a.tp_commit, a.ranks)
    rows = {}
    for (_a, r), got in in_processes(commit_rank, a.jobs, [(a, r) for r in ranks]):
        rows[str(r)] = got
        print(f"[prove-path commit] rank {r}: {json.dumps(got, sort_keys=True, default=str)}", flush=True)
    Path(a.out, "commit.json").write_text(json.dumps(rows, indent=1, sort_keys=True, default=str))
    return 0


def advice_negatives(r: int, world: int, mine: Any, held: dict, served: Path) -> dict[str, dict]:
    """What the verifier refuses before any draw about the advice: the rank's registration naming another compiler, other advice,
    no binding at all (a window over Programs of record passed off as the advice's), and another rank's registration."""
    w = json.loads((served / "registration.json").read_text())["window"]
    other = hashlib.sha512(b"other advice").hexdigest()
    cases = {"registration-names-another-compiler (source)": {**w, "compiler": {**w["compiler"], "digest": other}},
             "registration-names-other-advice (source)": {**w, "advice": {**w["advice"], "digest": other}},
             "registration-binds-no-compiler-or-advice (source)": {k: v for k, v in w.items() if k not in SRT.BINDING}}
    theirs = served.parent / f"rank{(r + 1) % world}" / "registration.json"
    if world > 1 and theirs.exists():
        cases["registration-of-another-rank (source)"] = json.loads(theirs.read_text())["window"]
    out = {}
    for name, x in cases.items():
        v = SRT.source(x, mine, held)
        out[name] = {"refused": not v["ok"], "codes": v["codes"]}
    return out


def forged_row(s: dict, forged: dict | None) -> dict:
    """A forged rank's verdict: its registration passed every check before the draw, the forged unit was drawn, and the outcome is
    a reject with Lean's verifier refusing a drawn member."""
    verdicts = (s.get("audit") or {}).get("verdicts") or {}
    lean = {k: v for k, v in verdicts.items() if k.startswith("lean-verify/")}
    unit = None
    if forged is not None:
        unit = next(m["base"] for m in s.get("members") or [] if m["k"] == forged["member"]) + forged["unit"]
    checks = {"checks_before_draw": bool(s.get("checks_before_draw")) and all(s["checks_before_draw"].values()),
              "forged_unit_drawn": unit is not None and unit in (s.get("draw") or []),
              "rejected": (s.get("audit") or {}).get("accepted") is False,
              "lean_refused": any(v is False for v in lean.values())}
    return {"ok": all(checks.values()), "checks": checks, "why": (s.get("audit") or {}).get("why")}


def audit_rank(job: tuple[argparse.Namespace, int, int]) -> dict:
    import a5 as A5

    a, world, r = job
    t0 = time.time()
    compiler, requests, bound = compiled(a.tp_commit, a.build_root, a.advice)
    mine, held = AW.window_of_advice(a.tp_commit, r, a.partition, compiler, requests, bound)
    window_s = round(time.time() - t0, 2)
    served = Path(a.windows) / f"rank{r}"
    more = advice_negatives(r, world, mine, held, served)
    s, ok = A5.audit(a, A5.template_sets(Path(a.template_sets)), Path(a.out) / f"rank{r}", mine, served,
                     lambda w: SRT.source(w, mine, held), a.port + A5.RANK_PORTS * r,
                     {"rank": r, "tp": held["tp"], "compiler": bound["compiler"], "advice": bound["advice"],
                      "requests": requests, "verifier_window_s": window_s}, more, rank=r)
    row = {"ok": ok, "seconds": round(time.time() - t0, 1),
           **{x: s.get(x) for x in ("run_root", "population", "members", "registration", "source", "refused", "checks_before_draw",
                                    "audit", "draw", "draw_counts", "negatives", "coins")}}
    if a.forged:
        forged = (json.loads((Path(a.windows) / "commit.json").read_text()).get(str(r)) or {}).get("forged")
        row["forged"] = {"unit": forged, **forged_row(s, forged)}
        row["ok"] = row["forged"]["ok"]
    return row


def audit(a: argparse.Namespace) -> int:
    rec = SRT.record(a.tp_commit)
    world, ranks = ranks_of(a.tp_commit, a.ranks)
    a.coins, a.verifier_coins, a.window_root = None, True, rec["tp_run_root"]
    a.rule = json.loads(Path(a.partition).read_text()).get("law")
    if a.forged:
        os.environ["FC_ZK_SELF_CHECK"] = "skip"
    _compiler, requests, bound = compiled(a.tp_commit, a.build_root, a.advice)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tp_run_root": rec["tp_run_root"], "world": world, "compiler": bound["compiler"], "advice": bound["advice"],
                     "requests": requests, "rule": a.rule, "delta": a.delta, "forged": a.forged,
                     "coins": {"source": "verifier", "what": "each rank's service.Stream on the verifier's own live coins (the "
                                                             "Lean's IO.getRandomBytes), read once the stream received the rank's "
                                                             "registration"},
                     "ranks": {}}
    for (_a, _w, r), row in in_processes(audit_rank, a.jobs, [(a, world, r) for r in ranks]):
        summary["ranks"][str(r)] = row
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    summary["ok"] = sorted(summary["ranks"]) == sorted(map(str, ranks)) and all(x["ok"] for x in summary["ranks"].values())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    shown = {r: {x: s.get(x) for x in ("ok", "population", "draw", "draw_counts", "seconds")}
             | ({"forged": s["forged"]} if a.forged else {"accepted": (s.get("audit") or {}).get("accepted"),
                                                         "negatives_refused": all((s.get("negatives") or {"none": False}).values())})
             for r, s in summary["ranks"].items()}
    print("PROVE-PATH " + json.dumps({"tp_run_root": rec["tp_run_root"], "forged": a.forged, "ok": summary["ok"], "ranks": shown},
                                     sort_keys=True, default=str))
    return 0 if summary["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("advice", help="a declared run's advice from its workload file")
    p.add_argument("--workload", required=True)
    p.add_argument("--out", required=True)
    for name in ("commit", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--tp-commit", required=True, help="the TP Commit's record (its --out: runs.jsonl, versions.json)")
        p.add_argument("--build-root", required=True, help="each rank's Programs of record in rank<r>/ (the stand-in compiler's)")
        p.add_argument("--advice", required=True, help="the advice's dir (advice.json)")
        p.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
        p.add_argument("--ranks", default=None, help="comma-separated (default every rank)")
        p.add_argument("--jobs", type=int, default=1, help="ranks at once, each in its own process")
        p.add_argument("--out", required=True)
    p = sub.choices["commit"]
    p.add_argument("--forge", help="MEMBER:UNIT: commit that unit with bit 0 of its first output word flipped")
    p.add_argument("--workers", type=int, default=None, help="commit workers per rank (default min(16, CPUs))")
    p = sub.choices["audit"]
    p.add_argument("--windows", required=True, help="the prover's windows, rank<r>/ each (prove_path.py commit --out)")
    p.add_argument("--template-sets", required=True, help="one input set per member template, for the lowering")
    p.add_argument("--m0-pythonpath", required=True)
    p.add_argument("--bin", required=True)
    p.add_argument("--lean", required=True)
    p.add_argument("--firewall", required=True)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--port", type=int, default=7651)
    p.add_argument("--session-timeout", type=float, default=14400)
    p.add_argument("--forged", action="store_true", help="the windows hold a forged unit (commit --forge): every rank must be refused")
    a = ap.parse_args(argv)
    if a.cmd == "advice":
        Path(a.out).mkdir(parents=True, exist_ok=True)
        doc = declared_advice(a.workload)
        requests_of(doc)
        (Path(a.out) / ADVICE).write_bytes(encode(doc))
        print(json.dumps({"advice": str(Path(a.out) / ADVICE), "requests": len(doc["requests"]), "steps": len(doc["steps"]),
                          "digest": hashlib.sha512(encode(doc)).hexdigest(), "bits": 8 * len(encode(doc))}))
        return 0
    return commit(a) if a.cmd == "commit" else audit(a)


if __name__ == "__main__":
    raise SystemExit(main())
