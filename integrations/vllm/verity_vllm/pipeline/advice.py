"""`verity-vllm advice RUN`: the compiler's advice (I1, `verity.core.protocols.certifier.advice`) for a served run, from its
scheduler step trace and its Commit record.

The precommitted compiler turns advice into each step's circuit; the advice says everything serving decided that nothing
declared before the run predicts.  This command writes it from what the run recorded:

    step_trace.jsonl   `engine.step_trace`: each executed step's rows in the model runner's order, with each row's first position
                       and length, the requests first scheduled or resumed, prefix-cache hits, preemptions and finishes
    the Commit record  each request's id as the client sent it (the body's `user`), its prompt and output token ids:
                       ``requests.jsonl`` (one request a line), or a server's ``windows.jsonl`` (its windows' requests)

and refuses a run whose two records disagree (a prompt or output length, a request the trace never saw), a row whose first
position isn't where the request's last row ended (`continuity`: no position is computed twice and none is skipped, except
from a prefix-cache hit at first scheduling or at a resume from preemption), and a finish advice v0 can't say (a stop string,
an abort).  A cache hit's KV `sources` are the runs of earlier-computed positions whose requests share the hit's tokens, read
from the Commit record's token ids.  Each kept output's `release` bucket is the bucket of width `--tau` seconds, counted from
the first step, in which the engine emitted it (the step's clock reading): with no certifier on the path, the earliest bucket
its frame could be released in.  A trace without a clock takes the step index as the bucket.

The advice is not published (Daniel, 24 Oct): its canonical encoding is committed as hiding rows (`advice_windows.commit_advice`:
hm96-sha512 rows under `verity.ml.commitments.advice`, the tree's domain bound to the registered compiler), and only the root and
the advice's length go public, as the binding a window's registration carries (`advice_windows.binding`, the window provenance's
``binding``, which `serving_rows.after_commit(binding=…)` takes).  Under ``--out``:

    registration.json     public: the binding (the compiler's name and digest, the advice's root and ``total_bits``), and the bits
                          I1 charges each egress window of `--T` buckets (``bits_by_window``)
    private/advice.bin    the prover's: the advice, I1's canonical encoding
    private/advice.json   the same advice, I1's `to_json`
    private/opening.json  the advice's opening (`advice_windows.opening`: its encoding and every row's salt)
    private/report.json   the round trip (`decode(encode(a)) == a`), the run's counts (they describe the advice), and the digests of
                          what it read

    verity-vllm advice RUN [--trace FILE] [--io FILE] [--out DIR] [--T 600] [--tau 0.1] [--compiler-name N --compiler-digest D]
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.certifier import advice as I1
from verity.ml.circuits.codec import canonical_json
from verity_vllm.config import UsageError, option
from verity_vllm.engine import step_trace as ST

from . import advice_windows as AW

SCHEMA = "verity-vllm/advice-report/v3"
REGISTRATION = "verity-vllm/advice-registration/v1"
COMPILER = "verity_vllm.predict.steps.compile_steps"


@dataclass(frozen=True)
class Compiler:
    """The registered compiler as the binding names it: ``name`` and ``digest`` (128 hex digits)."""
    name: str
    digest: str


def registered_compiler(name: str = COMPILER, digest: str | None = None) -> Compiler:
    """The compiler the advice is bound to; without ``digest``, the SHA-512 of `name` and the tree's code identity
    (`source_identity.code_identity`)."""
    if digest is None:
        # TODO: COMPILER's own registered digest once compile_steps lands with one
        from verity_vllm.pipeline import source_identity as SI
        digest = hashlib.sha512(canonical_json({"name": name, "code": SI.code_identity(SI.TREE)["sha256"]})).hexdigest()
    return Compiler(name, digest)


def finish_of(rid: str, rec: dict[str, Any] | None) -> str:
    """A trace finish as I1's: a stop with no stop reason (the model's EOS) or an int one (a stop token id) is `eos`."""
    if rec is None:
        raise ValueError(f"request {rid} never finished")
    reason, why = rec["reason"], rec.get("stop_reason")
    if reason == "stop" and (why is None or isinstance(why, int)):
        return "eos"
    if reason == "length":
        return "length"
    raise ValueError(f"request {rid} finished by {reason}{f' ({why!r})' if why is not None else ''}, which advice v0 can't say "
                     f"(its finishes are {', '.join(I1.FINISHES)})")


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
    computed: int = 0
    emitted: int = 0
    index: int | None = None                    # its I1 index: the order of first rows
    preempted: bool = False                     # preempted, and not yet resumed
    finish: dict[str, Any] | None = None
    made: list[list[int]] = field(default_factory=list)  # the positions whose KV it computed, as disjoint [lo, hi)


def _cover(made: list[list[int]], lo: int, hi: int) -> None:
    made.append([lo, hi])
    made.sort()
    merged = [made[0]]
    for a, b in made[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    made[:] = merged


def _reach(made: list[list[int]], at: int) -> int:
    """How far from `at` the computed positions run without a gap (`at` itself when it isn't computed)."""
    return next((b for a, b in made if a <= at < b), at)


def _sources(q: _Req, hit: int, tokens: dict[str, list[int]], reqs: dict[str, _Req]) -> tuple[I1.Source, ...]:
    """The runs of positions 0 … hit − 1 and the requests whose computed KV each reuses: at each position, the request (first by
    index) that computed it with the same tokens up to there and runs furthest.  Its own KV only on a resume."""
    mine = tokens[q.rid]
    out: list[I1.Source] = []
    at = 0
    while at < hit:
        best, reach = None, at
        for p in sorted((p for p in reqs.values() if p.index is not None), key=lambda p: p.index):
            if p is q and not q.preempted:
                continue
            theirs = tokens[p.rid]
            end = min(_reach(p.made, at), hit, len(theirs))
            agree = at
            while agree < end and theirs[agree] == mine[agree]:
                agree += 1
            if agree > reach and theirs[:at] == mine[:at]:
                best, reach = p, agree
        if best is None:
            raise ValueError(f"request {q.rid}'s cache hit of {hit} tokens: no request computed position {at} with its tokens")
        if out and out[-1].request == best.index:
            out[-1] = I1.Source(best.index, out[-1].n + reach - at)
        else:
            out.append(I1.Source(best.index, reach - at))
        at = reach
    return tuple(out)


@dataclass(frozen=True)
class Built:
    advice: I1.Advice
    release_rule: dict[str, Any]


def from_trace(recs: list[dict[str, Any]], io: dict[str, dict[str, Any]] | None = None, *, tau: float = 0.1) -> Built:
    """The run's advice from a step trace and, when given, the Commit record's requests (their `user` ids, the lengths the trace
    is held to, and the tokens a cache hit's sources are read from)."""
    reqs: dict[str, _Req] = {}
    for r in recs:
        if r["kind"] == "arrive":
            if r["request"] in reqs:
                raise ValueError(f"request {r['request']} arrives twice")
            reqs[r["request"]] = _Req(r["request"], int(r["prompt_len"]))
    tokens = {rid: rec["prompt_ids"] + rec["output_ids"] for rid, rec in io.items()} if io is not None else None
    steps = ST.steps(recs)
    outside = {r["request"]: r for r in recs if r["kind"] == "finish"}
    order: list[_Req] = []
    built: list[list[tuple[_Req, int, int, int, bool]]] = []
    for st in steps:
        s = st["step"]
        rows = []
        for rid, start, n in st["rows"]:
            q = reqs.get(rid)
            if q is None:
                raise ValueError(f"step {s}: request {rid} has a row but never arrived")
            if q.finish is not None and q.finish["at"] < s - 1:
                raise ValueError(f"step {s}: request {rid} has a row {s - q.finish['at']} steps after it finished")
            start, n, hit, resume = int(start), int(n), 0, rid in st["resumed"]
            if q.index is None:
                hit = int(st["cache_hit"].get(rid, 0))
                if start != hit:
                    raise ValueError(f"step {s}: request {rid} is first scheduled at position {start}, not at its cache hit {hit}")
                q.index = len(order)
                order.append(q)
            elif resume:
                if not q.preempted:
                    raise ValueError(f"step {s}: request {rid} resumes without a preemption")
                if start > q.computed:
                    raise ValueError(f"step {s}: request {rid} resumes at position {start}, past the {q.computed} it had computed")
                hit = start
            elif start != q.computed:
                raise ValueError(f"step {s}: request {rid}'s row starts at {start}, not where its last row ended ({q.computed})")
            rows.append((q, start, n, hit, resume))
            q.computed = start + n
        for q, start, n, _, _ in rows:
            _cover(q.made, start, start + n)
        out_rows = []
        for q, start, n, hit, resume in rows:
            sources: tuple[I1.Source, ...] = ()
            if hit:
                if tokens is None:
                    raise ValueError(f"step {s}: request {q.rid} has a cache hit, whose sources need the Commit record's tokens")
                sources = _sources(q, hit, tokens, reqs)
            q.preempted = q.preempted and not resume
            out_rows.append((q, start, n, hit, resume, sources))
        built.append(out_rows)
        for rid in st["preempted"]:
            reqs[rid].preempted = True
        for rid, k in st["emitted"].items():
            reqs[rid].emitted += int(k)
        for rid, f in st["finished"].items():
            reqs[rid].finish = {**f, "at": s}
        for rid, f in outside.items():
            if f["after_step"] == s and reqs[rid].finish is None:
                reqs[rid].finish = {**f, "at": s}
    if unscheduled := [rid for rid, q in reqs.items() if q.index is None]:
        raise ValueError(f"{len(unscheduled)} requests arrived and never ran: {unscheduled[:3]}")
    advice_reqs = []
    for q in order:
        user = q.rid
        if io is not None:
            rec = io.get(q.rid)
            if rec is None:
                raise ValueError(f"request {q.rid} is in the trace but not in the Commit record")
            if len(rec["prompt_ids"]) != q.prompt_len:
                raise ValueError(f"request {q.rid}: the trace's prompt is {q.prompt_len} tokens, the Commit record's {len(rec['prompt_ids'])}")
            if len(rec["output_ids"]) != q.emitted:
                raise ValueError(f"request {q.rid}: the trace emitted {q.emitted} tokens, the Commit record holds {len(rec['output_ids'])}")
            user = rec["user"]
        advice_reqs.append(I1.RequestAdvice(user, q.prompt_len, q.emitted, finish_of(q.rid, q.finish)))
    if io is not None and (extra := sorted(set(io) - set(reqs))):
        raise ValueError(f"the Commit record holds {len(extra)} requests the trace never saw: {extra[:3]}")
    i1_steps = tuple(I1.StepAdvice(tuple(
        I1.Row(q.index, "decode" if n == 1 and start >= q.prompt_len else "prefill", start, n, hit, resume, sources)
        for q, start, n, hit, resume, sources in rows)) for rows in built)
    clocked = all("t" in st for st in steps)
    t0 = steps[0]["t"] if clocked and steps else 0.0
    bucket = [math.floor((st["t"] - t0) / tau) if clocked else st["step"] for st in steps]
    # the kept outputs in I1's order (step, then row): a row keeps its sample when it reaches the request's known tokens
    known = [q.prompt_len for q in advice_reqs]
    kept = [0] * len(advice_reqs)
    release = []
    for s, st in enumerate(i1_steps):
        for row in st.rows:
            r = row.request
            if row.start + row.n == known[r] and kept[r] < advice_reqs[r].output_len:
                release.append(bucket[s])
                kept[r] += 1
                known[r] += 1
    rule = {"rule": "engine clock", "tau_s": tau} if clocked else {"rule": "step index"}
    try:
        return Built(I1.Advice(tuple(advice_reqs), i1_steps, tuple(release)), rule)
    except I1.MalformedAdvice as e:
        raise ValueError(f"I1 refuses the advice: {e}") from e


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def registration(compiler: Any, a: I1.Advice, root: bytes, rows: int, T: int) -> dict[str, Any]:
    """What goes public of the advice: the binding (the compiler, the root of the advice's hiding commitment and its length in
    bits) and the bits I1 charges each egress window of ``T`` buckets.  Nothing of the advice itself."""
    return {"schema": REGISTRATION, "scheme": AW.SCHEME, "binding": AW.binding(compiler, root.hex(), int(I1.total_bits(a))),
            "rows": rows, "row_bits": AW.ROW_BITS, "T": int(T),
            "bits_by_window": {str(w): int(n) for w, n in sorted(I1.bits_by_window(a, int(T)).items())}}


def commit(compiler: Any, a: I1.Advice, T: int = 600, salts: list[bytes] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The advice committed: its public registration (`registration`) and its private opening (`advice_windows.opening`)."""
    data = I1.encode(a)
    reg = AW.commit_advice(compiler, data, salts)
    return registration(compiler, a, reg.root, len(reg.rows), T), AW.opening(data, reg)


def write(run: str | Path, out: str | Path | None = None, *, trace: str | Path | None = None, io: str | Path | None = None,
          T: int = 600, tau: float = 0.1, compiler: Any = None) -> dict[str, Any]:
    run = Path(run)
    trace = Path(trace) if trace else run / "step_trace.jsonl"
    if io is None:
        io = next((p for p in (run / "requests.jsonl", run / "windows.jsonl") if p.exists()), None)
    recs = ST.load(trace)
    b = from_trace(recs, io_record(io) if io else None, tau=tau)
    a = b.advice
    data = I1.encode(a)
    back = I1.decode(data)
    compiler = compiler if compiler is not None else registered_compiler()
    public, opened = commit(compiler, a, T)
    out = Path(out) if out else run / "advice"
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(json.dumps(public, indent=1) + "\n")
    (private / "advice.bin").write_bytes(data)
    (private / "advice.json").write_text(json.dumps(I1.to_json(a), separators=(",", ":")) + "\n")
    (private / "opening.json").write_text(json.dumps(opened) + "\n")
    steps = ST.steps(recs)
    charged = I1.bits_by_window(a, T)
    outputs = I1.outputs(a)
    per = [{"window": w, "buckets": [w * T, (w + 1) * T], "bits": charged[w],
            "outputs": sum(1 for x in a.release if x // T == w),
            "steps": len({o.step for o, x in zip(outputs, a.release) if x // T == w})} for w in sorted(charged)]
    rows = [r for s in a.steps for r in s.rows]
    report = {"schema": SCHEMA, "run": str(run), "i1": {"module": I1.__name__, "schema": I1.SCHEMA, "version": I1.VERSION},
              "trace": {"path": str(trace), "sha256": _sha(trace)},
              "commit_record": {"path": str(io), "sha256": _sha(Path(io))} if io else None, "config": recs[0]["config"],
              "advice": {"bytes": len(data), "bits": I1.total_bits(a), "sha256": hashlib.sha256(data).hexdigest(),
                         "sha512": hashlib.sha512(data).hexdigest()},
              "roundtrip": back == a, "T": T, "release": b.release_rule, "windows": per, "registration": public,
              "counts": {"requests": len(a.requests), "steps": len(a.steps), "rows": len(rows), "outputs": len(a.release),
                         "chunked_prefill_rows": sum(1 for r in rows if r.kind == "prefill" and r.start + r.n < a.requests[r.request].prompt_len),
                         "prefill_rows": sum(1 for r in rows if r.kind == "prefill"), "decode_rows": sum(1 for r in rows if r.kind == "decode"),
                         "preempted_rows": sum(1 for r in rows if r.preempted), "cache_hit_rows": sum(1 for r in rows if r.cache_hit),
                         "mixed_steps": sum(1 for s in a.steps if len({r.kind for r in s.rows}) > 1),
                         "arrivals_during_decode": sum(1 for r in recs if r["kind"] == "arrive" and 0 < r["before_step"] < len(steps)),
                         "finishes": {f: sum(1 for q in a.requests if q.finish == f) for f in sorted({q.finish for q in a.requests})},
                         "row_order": sorted({s["order"] for s in steps})}}
    (private / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


@dataclass(frozen=True, kw_only=True)
class Options:
    """Write a served run's advice (I1) from its step trace and Commit record, committed: registration.json public, the advice and
    its opening under private/; exit 0 iff it round-trips."""
    run: str = option("run", help="the run directory (step_trace.jsonl, and requests.jsonl or windows.jsonl)")
    trace: str | None = option("--trace", default=None, help="the step trace (default RUN/step_trace.jsonl)")
    io: str | None = option("--io", default=None, help="the Commit record's requests (default RUN/requests.jsonl or RUN/windows.jsonl)")
    out: str | None = option("--out", default=None, help="where registration.json and private/ go (default RUN/advice)")
    compiler_name: str = option("--compiler-name", default=COMPILER, help="the registered compiler the advice is bound to")
    compiler_digest: str | None = option("--compiler-digest", default=None,
                                         help="its digest, 128 hex digits (default: SHA-512 of the name and the tree's code identity)")
    T: int = option("--T", type=int, default=600, help="egress buckets per window, for the bits I1 charges each window")
    tau: float = option("--tau", type=float, default=0.1, help="the egress bucket's width in seconds, for each output's release bucket")


def main(a: Options) -> int:
    try:
        rep = write(a.run, a.out, trace=a.trace, io=a.io, T=a.T, tau=a.tau,
                    compiler=registered_compiler(a.compiler_name, a.compiler_digest))
    except ValueError as e:
        raise UsageError(f"{a.run}: {e}") from e
    reg = rep["registration"]
    for w, bits in reg["bits_by_window"].items():
        print(f"[advice] window {w} (buckets {int(w) * reg['T']}-{(int(w) + 1) * reg['T'] - 1}): {bits} bits", flush=True)
    bound = reg["binding"]["advice"]
    print(f"[advice] {bound['bits']} bits at T={reg['T']}, committed as {reg['rows']} hiding rows, root {bound['root'][:16]}…; "
          f"round trip {'equal' if rep['roundtrip'] else 'NOT EQUAL'} ({rep['i1']['module']})", flush=True)
    return 0 if rep["roundtrip"] else 1
