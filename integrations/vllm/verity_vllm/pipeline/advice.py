"""`verity-vllm advice RUN`: the compiler's advice (I1) for a served run, from its scheduler step trace and its Commit record.

The precommitted compiler turns advice into each step's circuit; the advice says everything serving decided that nothing
declared before the run predicts.  This command writes it from what the run recorded:

    step_trace.jsonl   `engine.step_trace`: each executed step's rows in the model runner's order, with each row's first position
                       and length, the requests first scheduled or resumed, prefix-cache hits, preemptions and finishes
    the Commit record  each request's id as the client sent it (the body's `user`), its prompt and output token ids:
                       ``requests.jsonl`` (one request a line), or a server's ``windows.jsonl`` (its windows' requests)

and refuses a run whose two records disagree: a prompt or output length, a request the trace never saw, a row whose first
position isn't where the request's last row ended (`continuity`: no position is computed twice and none is skipped, except
from a prefix-cache hit at first scheduling or a reset at a resume from preemption).

    advice.bin    the advice, encoded
    advice.json   the same advice, as its dataclasses' fields
    report.json   the windows (steps between idle points, where every request that arrived has finished) and the bits each is
                  charged, the round trip (`decode(encode(a)) == a`), the run's counts, and the digests of what it read

    verity-vllm advice RUN [--trace FILE] [--io FILE] [--out DIR]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, option
from verity_vllm.engine import step_trace as ST

from . import advice_draft as I1

SCHEMA = "verity-vllm/advice-report/v1"


def finish_of(rec: dict[str, Any] | None) -> str:
    """A trace finish as I1's: a stop with no stop reason is the model's EOS, an int one a stop token id, a str one a stop string."""
    if rec is None:
        return "open"
    reason, why = rec["reason"], rec.get("stop_reason")
    if reason == "stop":
        return "eos" if why is None else ("stop_token" if isinstance(why, int) else "stop_string")
    return reason


def io_record(path: str | Path) -> dict[str, dict[str, Any]]:
    """The Commit record's requests by engine request id: ``{"user", "prompt_ids", "output_ids"}``.  Reads ``requests.jsonl``
    (``request``, ``user``, ``prompt_ids``, ``output_ids``) or ``windows.jsonl`` (each line a window with ``requests``, each with
    ``engine_id``, ``user`` or ``http_id``, ``prompt_ids`` and ``output_ids`` or ``streamed_ids``)."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        for r in doc.get("requests") if isinstance(doc.get("requests"), list) else [doc]:
            rid = str(r.get("request") or r.get("engine_id"))
            if rid in out:
                raise UsageError(f"{path}: request {rid} is recorded twice")
            out[rid] = {"user": str(r.get("user") or r.get("http_id") or rid), "prompt_ids": list(r["prompt_ids"]),
                        "output_ids": list(r.get("output_ids") if r.get("output_ids") is not None else r["streamed_ids"])}
    return out


@dataclass
class _Req:
    rid: str
    prompt_len: int
    arrive: int
    computed: int = 0
    emitted: int = 0
    scheduled: bool = False
    finish: dict[str, Any] | None = None


def from_trace(recs: list[dict[str, Any]], io: dict[str, dict[str, Any]] | None = None) -> tuple[I1.Advice, list[tuple[int, int]]]:
    """The run's advice and its windows ([first, end) step ranges between idle points), from a step trace and, when given, the
    Commit record's requests (their `user` ids, and the lengths the trace is held to)."""
    reqs: dict[str, _Req] = {}
    for r in recs:
        if r["kind"] == "arrive":
            if r["request"] in reqs:
                raise ValueError(f"request {r['request']} arrives twice")
            reqs[r["request"]] = _Req(r["request"], int(r["prompt_len"]), int(r["before_step"]))
    index = {rid: i for i, rid in enumerate(reqs)}
    steps = ST.steps(recs)
    outside = {r["request"]: r for r in recs if r["kind"] == "finish"}
    out_steps: list[I1.StepAdvice] = []
    windows: list[tuple[int, int]] = []
    lo = 0
    for st in steps:
        s = st["step"]
        rows: list[I1.Row] = []
        for rid, start, n in st["rows"]:
            q = reqs.get(rid)
            if q is None:
                raise ValueError(f"step {s}: request {rid} has a row but never arrived")
            if q.finish is not None and q.finish.get("at", s) < s - 1:
                raise ValueError(f"step {s}: request {rid} has a row {s - q.finish['at']} steps after it finished")
            hit = 0
            if not q.scheduled:
                hit = int(st["cache_hit"].get(rid, 0))
                if start != hit:
                    raise ValueError(f"step {s}: request {rid} is first scheduled at position {start}, not at its cache hit {hit}")
            elif rid in st["resumed"]:
                if start > q.computed:
                    raise ValueError(f"step {s}: request {rid} resumes at position {start}, past the {q.computed} it had computed")
            elif start != q.computed:
                raise ValueError(f"step {s}: request {rid}'s row starts at {start}, not where its last row ended ({q.computed})")
            kind = "decode" if n == 1 and start >= q.prompt_len and q.scheduled and rid not in st["resumed"] else "prefill"
            rows.append(I1.Row(index[rid], kind, int(start), int(n), hit))
            q.scheduled, q.computed = True, int(start) + int(n)
        for rid in st["preempted"]:
            q = reqs[rid]
            rows.append(I1.Row(index[rid], "decode" if q.computed > q.prompt_len else "prefill", q.computed, 0, 0, True))
        for rid, k in st["emitted"].items():
            reqs[rid].emitted += int(k)
        for rid, f in st["finished"].items():
            reqs[rid].finish = {**f, "at": s}
        for rid, f in outside.items():
            if f["after_step"] == s and reqs[rid].finish is None:
                reqs[rid].finish = {**f, "at": s}
        out_steps.append(I1.StepAdvice(tuple(rows)))
        # idle after step s: every request that could be scheduled by step s has finished, and none has a row at the step after
        # (under async scheduling a finished request's last row can follow its finish by one step)
        arrived = {rid for rid, q in reqs.items() if q.arrive <= s}
        later = {rid for rid, *_ in steps[s + 1]["rows"]} if s + 1 < len(steps) else set()
        if arrived and all(reqs[rid].finish is not None for rid in arrived) and not arrived & later:
            windows.append((lo, s + 1))
            lo = s + 1
    if lo < len(steps):
        windows.append((lo, len(steps)))
    advice_reqs = []
    for rid, q in reqs.items():
        user = rid
        if io is not None:
            rec = io.get(rid)
            if rec is None:
                raise ValueError(f"request {rid} is in the trace but not in the Commit record")
            if len(rec["prompt_ids"]) != q.prompt_len:
                raise ValueError(f"request {rid}: the trace's prompt is {q.prompt_len} tokens, the Commit record's {len(rec['prompt_ids'])}")
            if len(rec["output_ids"]) != q.emitted:
                raise ValueError(f"request {rid}: the trace emitted {q.emitted} tokens, the Commit record holds {len(rec['output_ids'])}")
            user = rec["user"]
        advice_reqs.append(I1.RequestAdvice(user, q.prompt_len, q.emitted, finish_of(q.finish)))
    if io is not None and (extra := sorted(set(io) - set(reqs))):
        raise ValueError(f"the Commit record holds {len(extra)} requests the trace never saw: {extra[:3]}")
    return I1.Advice(tuple(advice_reqs), tuple(out_steps), ()), windows


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(run: str | Path, out: str | Path | None = None, *, trace: str | Path | None = None, io: str | Path | None = None) -> dict[str, Any]:
    run = Path(run)
    trace = Path(trace) if trace else run / "step_trace.jsonl"
    if io is None:
        io = next((p for p in (run / "requests.jsonl", run / "windows.jsonl") if p.exists()), None)
    recs = ST.load(trace)
    a, windows = from_trace(recs, io_record(io) if io else None)
    data = I1.encode(a)
    back = I1.decode(data)
    out = Path(out) if out else run / "advice"
    out.mkdir(parents=True, exist_ok=True)
    (out / "advice.bin").write_bytes(data)
    (out / "advice.json").write_text(json.dumps(dataclasses.asdict(a), separators=(",", ":")) + "\n")
    steps = ST.steps(recs)
    per = []
    for lo, hi in windows:
        ws = a.steps[lo:hi]
        per.append({"window": [lo, hi], "steps": hi - lo, "rows": sum(len(s.rows) for s in ws), "tokens": sum(r.n for s in ws for r in s.rows),
                    "requests": len({r.request for s in ws for r in s.rows}), "bits": I1.bits(a, (lo, hi))})
    rows = [r for s in a.steps for r in s.rows]
    report = {"schema": SCHEMA, "run": str(run), "i1": I1.__name__, "trace": {"path": str(trace), "sha256": _sha(trace)},
              "commit_record": {"path": str(io), "sha256": _sha(Path(io))} if io else None, "config": recs[0]["config"],
              "advice": {"bytes": len(data), "bits": 8 * len(data), "sha256": hashlib.sha256(data).hexdigest()},
              "roundtrip": back == a, "windows": per,
              "counts": {"requests": len(a.requests), "steps": len(a.steps), "rows": len(rows),
                         "chunked_prefill_rows": sum(1 for r in rows if r.kind == "prefill" and r.n and r.start + r.n < a.requests[r.request].prompt_len),
                         "prefill_rows": sum(1 for r in rows if r.kind == "prefill"), "decode_rows": sum(1 for r in rows if r.kind == "decode"),
                         "preempted_rows": sum(1 for r in rows if r.preempted), "cache_hit_rows": sum(1 for r in rows if r.cache_hit),
                         "mixed_steps": sum(1 for s in a.steps if len({r.kind for r in s.rows}) > 1),
                         "arrivals_during_decode": sum(1 for r in recs if r["kind"] == "arrive" and 0 < r["before_step"] < len(steps)),
                         "finishes": {f: sum(1 for q in a.requests if q.finish == f) for f in sorted({q.finish for q in a.requests})},
                         "row_order": sorted({s["order"] for s in steps})}}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


@dataclass(frozen=True, kw_only=True)
class Options:
    """Write a served run's advice (I1) from its step trace and Commit record; exit 0 iff it round-trips."""
    run: str = option("run", help="the run directory (step_trace.jsonl, and requests.jsonl or windows.jsonl)")
    trace: str | None = option("--trace", default=None, help="the step trace (default RUN/step_trace.jsonl)")
    io: str | None = option("--io", default=None, help="the Commit record's requests (default RUN/requests.jsonl or RUN/windows.jsonl)")
    out: str | None = option("--out", default=None, help="where advice.bin, advice.json and report.json go (default RUN/advice)")


def main(a: Options) -> int:
    try:
        rep = write(a.run, a.out, trace=a.trace, io=a.io)
    except ValueError as e:
        raise UsageError(f"{a.run}: {e}") from e
    for w in rep["windows"]:
        print(f"[advice] window {w['window']}: {w['steps']} steps, {w['rows']} rows, {w['requests']} requests, {w['bits']} bits", flush=True)
    print(f"[advice] {rep['counts']['requests']} requests, {rep['counts']['steps']} steps, {rep['advice']['bits']} bits; "
          f"round trip {'equal' if rep['roundtrip'] else 'NOT EQUAL'} ({rep['i1']})", flush=True)
    return 0 if rep["roundtrip"] else 1
