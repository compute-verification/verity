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
from a prefix-cache hit at first scheduling or at a resume from preemption), a finish the advice can't say (a stop string; an
abort outside a Commit window; an abort mid-prefill, before the request's first output, since an aborted request's Program
ends at its last kept output; an abort while it waits after a preemption, since it has no row in the step it is freed at), a
request that arrived and never ran, and a row after an abort's step.  A request its client aborted before it was ever
scheduled has no row and no output, so the advice drops it, exactly; the Commit record keeps it, aborted after 0 ids, and must
hold no output for it.  A cache hit's KV `sources` are the runs of earlier-computed positions whose requests share the hit's
tokens, read
from the Commit record's token ids.  Each kept output's `release` bucket is the bucket of width `--tau` seconds, counted from
the first step, in which the engine emitted it (the step's clock reading): with no certifier on the path, the earliest bucket
its frame could be released in.  A trace without a clock takes the step index as the bucket.

The advice is not published: its canonical encoding is committed as hiding rows (`advice_windows.commit_advice`:
hm96-sha512 rows under `verity.ml.commitments.advice`, the tree's domain bound to the registered compiler), and only the root and
the advice's length go public, as the binding a window's registration carries (`advice_windows.binding`, the window provenance's
``binding``, which `serving_rows.after_commit(binding=…)` takes).  Under ``--out``:

    registration.json     public: the binding (the compiler's name and digest, the advice's root and ``total_bits``), the
                          registered compiler, `advice_windows.Capped` with ℓ (the cap on the advice's length its digest covers)
                          and the compiler it wraps (``compiler``), the bits
                          I1 charges each egress window of `--T` buckets (``bits_by_window``), and with `--weights-of-record`
                          the committer's hiding root of each weight parameter the window circuit registers, as its weights
                          registration records it (``registered_roots``: `registered_weights`)
    private/advice.bin    the prover's: the advice, I1's canonical encoding
    private/advice.json   the same advice, I1's `to_json`
    private/opening.json  the advice's opening (`advice_windows.opening`: its encoding and every row's salt)
    private/report.json   the round trip (`decode(encode(a)) == a`), the run's counts (they describe the advice), and the digests of
                          what it read

A `verity-vllm serve-openai` run (its ``advice_trace.jsonl``, the server's scheduler step trace with each Commit window's open and
close, and its ``requests.jsonl``, whose ``user`` names each request) gets one advice a closed Commit window (`windows_of`), each
under ``w<k>/`` with its registration naming the window's Commit (run id and run root), listed in ``windows.json``; a window still
open is left for a later call, and a window the advice refuses is named there.  While the server runs, a window holding a request
with no Commit record yet is pending, not refused: the server writes a request's record at the close of the window it finishes in,
after the closes before it, so one running on past its window's close has none until then.  A pending window and the ones after
it are left for a later call, with no ``windows.json`` written and exit code 3; once the server has stopped with every record
written (its ``summary.json``'s ``final``), a request with none is refused.  A window's advice is `window_advice`'s: I1 v0
with the requests it carries across its open and close, since a window closes after ``window_steps`` steps whatever is running,
and the requests its clients abort (the trace's outside finish, ``abort``, after the step in flight).
The windows are read in order, each opening with the state the last one left (`Left`: where each unfinished request's next
row continues, its kept outputs and the KV it computed), so a request that runs on into the next window is held there to the
position it reached, and its record (lengths and finish) is the Commit record's.

    verity-vllm advice RUN [--trace FILE] [--io FILE] [--out DIR] [--window K] [--T 600] [--tau 0.1] [--compiler-name N --compiler-digest D] [--cap BITS]
        [--weights-of-record FILE]
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from verity.core.protocols.certifier import advice as I1
from verity.ml.windows import interface as IF
from verity_vllm.config import UsageError, option
from verity_vllm.engine import step_trace as ST

from . import advice_windows as AW
from . import window_advice as WA

SCHEMA = "verity-vllm/advice-report/v3"
REGISTRATION = "verity-vllm/advice-registration/v1"
SERVED = "verity-vllm/advice-windows-index/v1"
COMPILER = "verity_vllm.program.predict.steps.compile_steps"
#: a served run's exit code while a window it reads waits on a Commit record the running server hasn't written yet
PENDING = 3


CAP_BITS = AW.CAP_BITS


@dataclass(frozen=True)
class Compiler:
    """The compiler the cap wraps (`advice_windows.Capped`): ``name`` and ``digest`` (128 hex digits)."""
    name: str
    digest: str


def registered_compiler(name: str = COMPILER, digest: str | None = None, cap_bits: int = CAP_BITS) -> AW.Capped:
    """The compiler the advice is bound to: `advice_windows.Capped` (whose digest names the inner compiler's and ℓ =
    ``cap_bits``) over ``name``, whose digest is ``digest``, else the verifier's `interface.compiler_digest` of ``name`` over
    the integration's import closure (`research_tools.CLOSURE`) in this git work tree."""
    if digest is None:
        from verity_vllm.pipeline import research_tools as RT
        from verity_vllm.pipeline import source_identity as SI
        digest = IF.compiler_digest(name, SI.TREE, [g.removesuffix("/**") for g in RT.CLOSURE])
    return AW.Capped(Compiler(name, digest), cap_bits)


def finish_of(rid: str, rec: dict[str, Any] | None) -> str:
    """A trace finish as the advice's: a stop with no stop reason (the model's EOS) or an int one (a stop token id) is `eos`, and
    a client's abort `abort` (`window_advice`'s only)."""
    if rec is None:
        raise ValueError(f"request {rid} never finished")
    reason, why = rec["reason"], rec.get("stop_reason")
    if reason == "stop" and (why is None or isinstance(why, int)):
        return "eos"
    if reason in ("length", "abort"):
        return reason
    raise ValueError(f"request {rid} finished by {reason}{f' ({why!r})' if why is not None else ''}, which the advice can't say "
                     f"(its finishes are {', '.join(WA.FINISHES)})")


def _aborted_before_output(rid: str) -> str:
    return (f"request {rid} was aborted before its first output, which the advice refuses: an aborted request's Program ends at "
            "its last kept output, and it kept none")


def _aborted_preempted(rid: str, step: int) -> str:
    return (f"request {rid} was aborted after step {step} while it waited after a preemption, which the advice refuses: it has no "
            "row in the step it is freed at")


def io_record(path: str | Path) -> dict[str, dict[str, Any]]:
    """The Commit record's requests by engine request id: ``{"user", "prompt_ids", "output_ids", "finish"}``.  Reads
    ``requests.jsonl`` (``request``, ``user``, ``prompt_ids``, ``output_ids``, ``finish``) or ``windows.jsonl`` (each line a window
    with ``requests``, each with ``engine_id``, ``user`` or ``http_id``, ``prompt_ids`` and ``output_ids`` or ``streamed_ids``);
    ``finish`` is None where the record has none."""
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
                        "output_ids": list(r.get("output_ids") if r.get("output_ids") is not None else r["streamed_ids"]),
                        "finish": r.get("finish")}
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
            if (p is q and not q.preempted) or (theirs := tokens.get(p.rid)) is None:
                continue
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
class Left:
    """A request a Commit window leaves to the next (`window_advice.close`): its trace state at the close (its finish's step
    counted from the next window's first), its record and its advice state, and whether the next window must carry it."""
    q: _Req
    record: WA.RequestAdvice
    state: WA.Carried
    must: bool


@dataclass(frozen=True)
class Built:
    advice: WA.Advice
    release_rule: dict[str, Any]
    left: tuple[Left, ...] = ()
    dropped: tuple[str, ...] = ()               # the requests aborted before they were ever scheduled (`_dropped`)


def _dropped(reqs: dict[str, _Req], outside: dict[str, dict[str, Any]], io: dict[str, dict[str, Any]] | None) -> tuple[str, ...]:
    """The requests that arrived and were never scheduled, each one its client aborted, which the advice drops: with no row, the
    advice of the others is the same without them.  The Commit record keeps each, aborted after 0 ids; one it says kept an
    output, or the trace says emitted one, is refused, and so is a request that never ran and wasn't aborted."""
    unscheduled = [rid for rid, q in reqs.items() if q.index is None]
    gone = tuple(rid for rid in unscheduled if (outside.get(rid) or {}).get("reason") == "abort")
    if never := [rid for rid in unscheduled if rid not in gone]:
        raise ValueError(f"{len(never)} requests arrived and never ran: {never[:3]}")
    for rid in gone:
        rec = (io or {}).get(rid) or {"output_ids": [], "finish": None}
        fin = rec.get("finish") or {"reason": "abort"}
        if reqs[rid].emitted or rec["output_ids"] or fin.get("reason") != "abort" or fin.get("k"):
            raise ValueError(f"request {rid} was aborted before it was ever scheduled, which the advice drops, and the trace emitted "
                             f"{reqs[rid].emitted} tokens for it, the Commit record {len(rec['output_ids'])}, finished by "
                             f"{fin.get('reason')}" + (f" after {fin['k']} ids" if fin.get("k") is not None else ""))
    return gone


def from_trace(recs: list[dict[str, Any]], io: dict[str, dict[str, Any]] | None = None, *, tau: float = 0.1,
               carry: tuple[Left, ...] | None = None) -> Built:
    """The run's advice from a step trace and, when given, the Commit record's requests (their `user` ids, the lengths the trace
    is held to, and the tokens a cache hit's sources are read from).  ``carry`` makes it a served Commit window's
    (`window_advice`): what the last window left (`Built.left`; ``()`` for a window that opens on an idle engine), each
    request it must carry and each it may that has a row here coming first, held to its state; and a request may end the window
    unfinished, its record then the Commit record's.  Without it the trace is a whole session's, every request finished.  Either
    way a request its client aborted before it was ever scheduled is dropped (`Built.dropped`)."""
    window = carry is not None
    ids = {row[0] for r in recs if r["kind"] == "step" for row in r["rows"]}
    carried = [x for x in carry or () if x.must or x.q.rid in ids]
    reqs: dict[str, _Req] = {x.q.rid: replace(x.q, index=k, made=[list(m) for m in x.q.made]) for k, x in enumerate(carried)}
    order: list[_Req] = [reqs[x.q.rid] for x in carried]
    for r in recs:
        if r["kind"] == "arrive":
            q = reqs.get(r["request"])
            if q is not None and q.index is not None and q.index < len(carried) and q.prompt_len == int(r["prompt_len"]):
                continue
            if q is not None:
                raise ValueError(f"request {r['request']} arrives twice")
            reqs[r["request"]] = _Req(r["request"], int(r["prompt_len"]))
    tokens = {rid: rec["prompt_ids"] + rec["output_ids"] for rid, rec in io.items()} if io is not None else None
    steps = ST.steps(recs)
    outside = {r["request"]: r for r in recs if r["kind"] == "finish"}
    built: list[list[tuple[_Req, int, int, int, bool]]] = []
    for st in steps:
        s = st["step"]
        rows = []
        for rid, start, n in st["rows"]:
            q = reqs.get(rid)
            if q is None:
                raise ValueError(f"step {s}: request {rid} has a row but never arrived")
            if q.finish is not None and q.finish["at"] < s - (0 if q.finish.get("outside") else 1):
                raise ValueError(f"step {s}: request {rid} has a row {s - q.finish['at']} steps after it finished"
                                 + (f" ({q.finish['reason']} after step {q.finish['at']}, outside the scheduler)" if q.finish.get("outside") else ""))
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
                if f["reason"] == "abort" and reqs[rid].preempted:
                    raise ValueError(_aborted_preempted(rid, s))
                reqs[rid].finish = {**f, "at": s}
    dropped = _dropped(reqs, outside, io)
    advice_reqs = []
    for q in order:
        was = carried[q.index].record if q.index < len(carried) else None
        user, total = (was.id, was.output_len) if was is not None else (q.rid, q.emitted)
        rec = io.get(q.rid) if io is not None else None
        if rec is not None:
            if len(rec["prompt_ids"]) != q.prompt_len:
                raise ValueError(f"request {q.rid}: the trace's prompt is {q.prompt_len} tokens, the Commit record's {len(rec['prompt_ids'])}")
            user, total = rec["user"], len(rec["output_ids"])
        elif io is not None and was is None:
            raise ValueError(f"request {q.rid} is in the trace but not in the Commit record")
        if q.emitted > total or (q.emitted < total and not window) or (q.emitted < total and q.finish is not None):
            raise ValueError(f"request {q.rid}: the trace emitted {q.emitted} tokens, the Commit record holds {total}")
        if q.finish is not None or not window:
            finish = finish_of(q.rid, q.finish)
            if finish == "abort" and not window:
                raise ValueError(f"request {q.rid} finished by abort, which a whole session's advice (I1 v0) can't say; a served "
                                 f"Commit window's (window_advice) can")
        elif rec is not None or was is None:
            # it finishes in a later window: the Commit record's finish, or the abort the trace records after this window
            finish = finish_of(q.rid, (rec and rec.get("finish")) or outside.get(q.rid))
        else:
            finish = was.finish
        if finish == "abort" and total == 0:
            raise ValueError(_aborted_before_output(q.rid))
        if was is not None and (user, total, finish) != (was.id, was.output_len, was.finish):
            raise ValueError(f"request {q.rid} is carried in as {was}, and its Commit record says {user!r}, {total} outputs, {finish}")
        advice_reqs.append(WA.RequestAdvice(user, q.prompt_len, total, finish))
    if io is not None and (extra := sorted(set(io) - set(reqs))):
        raise ValueError(f"the Commit record holds {len(extra)} requests the trace never saw: {extra[:3]}")
    i1_steps = tuple(I1.StepAdvice(tuple(
        I1.Row(q.index, "decode" if n == 1 and start >= q.prompt_len else "prefill", start, n, hit, resume, sources)
        for q, start, n, hit, resume, sources in rows)) for rows in built)
    clocked = all("t" in st for st in steps)
    t0 = steps[0]["t"] if clocked and steps else 0.0
    bucket = [math.floor((st["t"] - t0) / tau) if clocked else st["step"] for st in steps]
    # the kept outputs in I1's order (step, then row): a row keeps its sample when it reaches the request's known tokens
    kept = [x.state.kept for x in carried] + [0] * (len(advice_reqs) - len(carried))
    known = [q.prompt_len + k for q, k in zip(advice_reqs, kept)]
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
        if not window:
            I1.Advice(tuple(q.i1() for q in advice_reqs), i1_steps, tuple(release))
        a = WA.Advice(tuple(advice_reqs), i1_steps, tuple(release), tuple(x.state for x in carried))
    except I1.MalformedAdvice as e:
        raise ValueError(f"I1 refuses the advice: {e}") from e
    left = []
    for c in WA.close(a):
        q = order[c.request]
        if c.must and q.finish is not None:
            raise ValueError(f"request {q.rid} finished at step {q.finish['at']}, and its rows keep {c.state.kept} of its "
                             f"{a.requests[c.request].output_len} outputs")
        f = None if q.finish is None else {**q.finish, "at": q.finish["at"] - len(steps)}
        left.append(Left(replace(q, index=None, finish=f, made=[list(m) for m in q.made]), a.requests[c.request], c.state, c.must))
    return Built(a, rule, tuple(left), dropped)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def registered_weights(registration: dict[str, Any], serve: dict[str, Any] | None = None) -> dict[str, str]:
    """What a window's registration records of the weights it served (``registered_roots``): the committer's weights
    registration's hiding root for each weight parameter its window circuit registers (`records.weights_registration`,
    ``weights/<label>@<words>`` -> its entry, `window_weights.weights_registration`), written once per model and key
    before the first window's coins, the roots the verifier pins, which `interface.input_count` holds each weight's reads
    to. Refused (ValueError) unless it is such a registration, every value a weight parameter's, and, given the run's
    ``serve.json``, of its model at its revision."""
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.records.weights_registration import WeightsRegistration

    r = WeightsRegistration.of(registration, "the weights registration")
    if other := sorted(p for p in r.values if not p.startswith(f"{WR.PARAM}/")):
        raise ValueError(f"the weights registration registers {other}, not weight parameters {WR.PARAM}/<label>@<words>")
    if serve is not None and (r.repo, r.revision) != (serve.get("model"), serve.get("revision")):
        raise ValueError(f"the weights registration is of {r.repo}@{r.revision}, and the run served "
                         f"{serve.get('model')}@{serve.get('revision')}")
    return r.roots


def registration(compiler: Any, a: I1.Advice, root: bytes, rows: int, T: int,
                 registered_roots: dict[str, str] | None = None) -> dict[str, Any]:
    """What goes public of the advice: the binding (the compiler, the root of the advice's hiding commitment and its length in
    bits), the bits I1 charges each egress window of ``T`` buckets and, given, the root of each registered parameter
    (``registered_roots``, `registered_weights`).  Nothing of the advice itself."""
    c = codec(a)
    return {"schema": REGISTRATION, "scheme": AW.SCHEME, "binding": AW.binding(compiler, root.hex(), int(c.total_bits(a))),
            "compiler": {"name": str(compiler.name), "digest": compiler.digest, "cap_bits": int(compiler.cap_bits),
                         "inner": {"name": str(compiler.inner.name), "digest": compiler.inner.digest}},
            "rows": rows, "row_bits": AW.ROW_BITS, "T": int(T),
            "bits_by_window": {str(w): int(n) for w, n in sorted(c.bits_by_window(a, int(T)).items())},
            **({"registered_roots": dict(registered_roots)} if registered_roots else {})}


def codec(a: Any) -> Any:
    """The codec of ``a``: `window_advice` for a Commit window's advice, else I1."""
    return WA if isinstance(a, WA.Advice) else I1


def commit(compiler: Any, a: Any, T: int = 600, salts: list[bytes] | None = None,
           registered_roots: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The advice (I1's, or a Commit window's) committed: its public registration (`registration`, with ``registered_roots``) and
    its private opening (`advice_windows.opening`).  Advice over the compiler's ℓ (``compiler``: an `advice_windows.Capped`)
    is refused: the compiler is total only up to it."""
    data = codec(a).encode(a)
    cap = int(compiler.cap_bits)
    if 8 * len(data) > cap:
        raise ValueError(f"the advice is {8 * len(data)} bits, over the registered cap ℓ = {cap} bits")
    reg = AW.commit_advice(compiler, data, salts)
    return registration(compiler, a, reg.root, len(reg.rows), T, registered_roots), AW.opening(data, reg)


def windows_of(recs: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    """A served trace's records by closed Commit window (`StepTrace.window`), each a trace of its own: the header, the arrivals of
    the requests with rows in the window, its steps renumbered from 0, and their outside finishes; and the windows still open.  A
    window's steps are those scheduled from its open to its close, wherever async scheduling recorded them; one with none is left
    out."""
    opened: dict[int, int] = {}
    closed: dict[int, int] = {}
    for r in recs:
        if r["kind"] == "window":
            (opened if r["event"] == "open" else closed)[int(r["window"])] = int(r["before_step"])
    out: dict[int, list[dict[str, Any]]] = {}
    for w in sorted(closed):
        lo, hi = opened.get(w, 0), closed[w]
        steps = sorted((r for r in recs if r["kind"] == "step" and lo <= r["step"] < hi), key=lambda r: r["step"])
        if not steps:
            continue
        ids = {row[0] for st in steps for row in st["rows"]}
        out[w] = [recs[0], *({**r, "before_step": max(0, r["before_step"] - lo)} for r in recs if r["kind"] == "arrive" and r["request"] in ids),
                  *({**st, "step": st["step"] - lo} for st in steps),
                  *({**r, "after_step": r["after_step"] - lo} for r in recs if r["kind"] == "finish" and r["request"] in ids)]
    return out, sorted(set(opened) - set(closed))


def served(recs: list[dict[str, Any]]) -> bool:
    """Whether a trace is a server's, in Commit windows."""
    return any(r["kind"] == "window" for r in recs)


def write(run: str | Path, out: str | Path | None = None, *, trace: str | Path | None = None, io: str | Path | None = None,
          T: int = 600, tau: float = 0.1, compiler: Any = None, window: int | None = None,
          registered_roots: dict[str, str] | None = None) -> dict[str, Any]:
    """The run's advice, committed (`write_one`); a `verity-vllm serve-openai` run's, one advice a closed Commit window
    (`write_served`); each registration with ``registered_roots``, the root of each parameter its window circuit registers."""
    run = Path(run)
    if trace is None:
        trace = next((p for p in (run / ST.SERVED, run / "step_trace.jsonl") if p.exists()), run / "step_trace.jsonl")
    trace = Path(trace)
    if io is None:
        io = next((p for p in (run / "requests.jsonl", run / "windows.jsonl") if p.exists()), None)
    recs = ST.load(trace)
    compiler = compiler if compiler is not None else registered_compiler()
    out = Path(out) if out else run / "advice"
    if served(recs):
        return write_served(run, out, recs, trace, io, T=T, tau=tau, compiler=compiler, window=window,
                            registered_roots=registered_roots)
    return write_one(run, out, recs, trace, io, io_record(io) if io else None, T=T, tau=tau, compiler=compiler,
                     registered_roots=registered_roots)


def write_served(run: Path, out: Path, recs: list[dict[str, Any]], trace: Path, io: str | Path | None, *, T: int, tau: float,
                 compiler: Any, window: int | None = None, registered_roots: dict[str, str] | None = None) -> dict[str, Any]:
    """A server's advice, one a closed Commit window (`windows_of`; only ``window`` when given): each under ``out/w<k>/`` as
    `write_one` writes it, its registration naming the window's Commit (its run id and run root, from windows.jsonl).  The
    windows are read in order, each opening with what the last one left (`from_trace`'s ``carry``), so ``window`` reads the
    ones before it too.  A window whose advice is refused (a finish the advice can't say, a trace the Commit record contradicts)
    is named in ``refused``, and the others are written; the window after a refused one opens on nothing carried.  Until the
    server has `stopped`, a window holding a request with no Commit record yet is named in ``pending``, and neither it nor any
    window after it is read.  ``out/windows.json`` (public) lists each window's registration, written only when none is pending."""
    if io is None:
        raise ValueError("a served run's advice needs its Commit record, requests.jsonl")
    io_all = io_record(io)
    final = stopped(run)
    lines = (run / "windows.jsonl").read_text().splitlines() if (run / "windows.jsonl").exists() else []
    commits = {int(d["window"]): {"run_id": d.get("run_id"), "run_root": d.get("run_root")}
               for d in (json.loads(line) for line in lines if line.strip())}
    per, still_open = windows_of(recs)
    if window is not None and window not in per:
        raise ValueError(f"window {window} is not a closed window with steps (closed: {sorted(per)}; open: {still_open})")
    reports: dict[str, Any] = {}
    refused: dict[str, str] = {}
    pending: dict[str, str] = {}
    carry: tuple[Left, ...] = ()
    gone: int | None = None
    for w, sub in per.items():
        if window is not None and w > window:
            break
        ids = {r["request"] for r in sub if r["kind"] == "arrive"}
        if not final and (unrecorded := sorted(ids - set(io_all))):
            pending[str(w if window is None else window)] = (
                f"request {unrecorded[0]} of window {w} has no Commit record yet, and the server is still running"
                + (f" (window {window} opens on what window {w} leaves)" if window is not None and w != window else ""))
            break
        try:
            b = from_trace(sub, {i: q for i, q in io_all.items() if i in ids}, tau=tau, carry=carry)
        except ValueError as e:
            if window is None or w == window:
                refused[str(w)] = str(e) + (f" (window {gone} before it is refused, so it opens on nothing carried)" if gone is not None else "")
            carry, gone = (), w
            continue
        carry, gone = b.left, None
        if window is None or w == window:
            reports[str(w)] = _written(run, out / f"w{w}", b, sub, trace, io, T=T, compiler=compiler,
                                       extra={"window": w, "commit": commits.get(w)}, registered_roots=registered_roots)
    index = {"schema": SERVED, "run": str(run), "trace": str(trace), "T": int(T),
             "windows": {w: {"registration": f"w{w}/registration.json", "root": r["registration"]["binding"]["advice"]["root"],
                             "bits": r["registration"]["binding"]["advice"]["bits"], "commit": r["registration"]["commit"]}
                         for w, r in reports.items()},
             "refused": refused, "open": still_open}
    if not pending:
        out.mkdir(parents=True, exist_ok=True)
        (out / "windows.json").write_text(json.dumps(index, indent=1) + "\n")
    return {**index, "pending": pending, "reports": reports}


def stopped(run: Path) -> bool:
    """Whether ``run``'s server has stopped with every Commit record written: its ``summary.json`` says ``final`` (one without
    the key is from a server that wrote each window's records on its engine loop, before any summary)."""
    try:
        return json.loads((run / "summary.json").read_text()).get("final", True) is True
    except (OSError, ValueError):
        return False


def write_one(run: Path, out: Path, recs: list[dict[str, Any]], trace: Path, io: str | Path | None, io_map: dict[str, Any] | None, *,
              T: int, tau: float, compiler: Any, extra: dict[str, Any] | None = None,
              registered_roots: dict[str, str] | None = None) -> dict[str, Any]:
    """One advice from ``recs`` and the Commit record's requests ``io_map``, committed: ``out/registration.json`` (public, with
    ``extra`` and ``registered_roots``) and ``out/private/``."""
    return _written(run, out, from_trace(recs, io_map, tau=tau), recs, trace, io, T=T, compiler=compiler, extra=extra,
                    registered_roots=registered_roots)


def _written(run: Path, out: Path, b: Built, recs: list[dict[str, Any]], trace: Path, io: str | Path | None, *, T: int, compiler: Any,
             extra: dict[str, Any] | None = None, registered_roots: dict[str, str] | None = None) -> dict[str, Any]:
    a = b.advice
    data = WA.encode(a)
    back = WA.decode(data)
    public, opened = commit(compiler, a, T, registered_roots=registered_roots)
    public.update(extra or {})
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(json.dumps(public, indent=1) + "\n")
    (private / "advice.bin").write_bytes(data)
    (private / "advice.json").write_text(json.dumps(WA.to_json(a), separators=(",", ":")) + "\n")
    (private / "opening.json").write_text(json.dumps(opened) + "\n")
    steps = ST.steps(recs)
    charged = WA.bits_by_window(a, T)
    outputs = WA.outputs(a)
    per = [{"window": w, "buckets": [w * T, (w + 1) * T], "bits": charged[w],
            "outputs": sum(1 for x in a.release if x // T == w),
            "steps": len({o.step for o, x in zip(outputs, a.release) if x // T == w})} for w in sorted(charged)]
    rows = [r for s in a.steps for r in s.rows]
    report = {"schema": SCHEMA, "run": str(run),
              "i1": {"module": (WA if a.version else I1).__name__, "schema": WA.SCHEMA if a.version else I1.SCHEMA, "version": a.version},
              "trace": {"path": str(trace), "sha256": _sha(trace)},
              "commit_record": {"path": str(io), "sha256": _sha(Path(io))} if io else None, "config": recs[0]["config"],
              "advice": {"bytes": len(data), "bits": WA.total_bits(a), "sha256": hashlib.sha256(data).hexdigest(),
                         "sha512": hashlib.sha512(data).hexdigest()},
              "roundtrip": back == a, "T": T, "release": b.release_rule, "windows": per, "registration": public,
              "carry": {"in": len(a.carried), "out": sum(x.must for x in b.left), "may": sum(not x.must for x in b.left)},
              "counts": {"requests": len(a.requests), "steps": len(a.steps), "rows": len(rows), "outputs": len(a.release),
                         "chunked_prefill_rows": sum(1 for r in rows if r.kind == "prefill" and r.start + r.n < a.requests[r.request].prompt_len),
                         "prefill_rows": sum(1 for r in rows if r.kind == "prefill"), "decode_rows": sum(1 for r in rows if r.kind == "decode"),
                         "preempted_rows": sum(1 for r in rows if r.preempted), "cache_hit_rows": sum(1 for r in rows if r.cache_hit),
                         "mixed_steps": sum(1 for s in a.steps if len({r.kind for r in s.rows}) > 1),
                         "arrivals_during_decode": sum(1 for r in recs if r["kind"] == "arrive" and 0 < r["before_step"] < len(steps)
                                                       and r["request"] not in b.dropped),
                         "aborted_unscheduled": len(b.dropped),
                         "finishes": {f: sum(1 for q in a.requests if q.finish == f) for f in sorted({q.finish for q in a.requests})},
                         "row_order": sorted({s["order"] for s in steps})}}
    (private / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


@dataclass(frozen=True, kw_only=True)
class Options:
    """Write a served run's advice (I1) from its step trace and Commit record, committed: registration.json public, the advice and
    its opening under private/; a serve-openai run's, one a closed Commit window under w<k>/; exit 0 iff each round-trips, 3
    (`PENDING`) while a window waits on a Commit record its running server hasn't written yet."""
    run: str = option("run", help="the run directory (step_trace.jsonl or a server's advice_trace.jsonl, and requests.jsonl or windows.jsonl)")
    trace: str | None = option("--trace", default=None, help="the step trace (default RUN/advice_trace.jsonl, else RUN/step_trace.jsonl)")
    io: str | None = option("--io", default=None, help="the Commit record's requests (default RUN/requests.jsonl or RUN/windows.jsonl)")
    out: str | None = option("--out", default=None, help="where registration.json and private/ go (default RUN/advice)")
    window: int | None = option("--window", type=int, default=None, help="a server's run: only this Commit window (default every closed one)")
    compiler_name: str = option("--compiler-name", default=COMPILER, help="the registered compiler the advice is bound to")
    compiler_digest: str | None = option("--compiler-digest", default=None,
                                         help="its digest, 128 hex digits (default: interface.compiler_digest of the name over "
                                              "the integration's import closure in this git work tree)")
    cap: int = option("--cap", type=int, default=CAP_BITS,
                      help="ℓ, the cap on the advice's length in bits the compiler is total up to: the registered compiler is "
                           "advice_windows.Capped over the one named, whose digest covers ℓ (default 2^23, 1 MiB)")
    T: int = option("--T", type=int, default=600, help="egress buckets per window, for the bits I1 charges each window")
    tau: float = option("--tau", type=float, default=0.1, help="the egress bucket's width in seconds, for each output's release bucket")
    weights_of_record: str | None = option("--weights-of-record", default=None, metavar="JSON",
                                           help="the committer's weights registration (window_weights.weights_registration), "
                                                "whose hiding root for each weight parameter each window's registration "
                                                "records (default: none recorded)")


def _say(rep: dict[str, Any], where: str = "") -> bool:
    reg = rep["registration"]
    for w, bits in reg["bits_by_window"].items():
        print(f"[advice]{where} egress window {w} (buckets {int(w) * reg['T']}-{(int(w) + 1) * reg['T'] - 1}): {bits} bits", flush=True)
    bound = reg["binding"]["advice"]
    print(f"[advice]{where} {bound['bits']} bits at T={reg['T']}, committed as {reg['rows']} hiding rows, root {bound['root'][:16]}…; "
          f"round trip {'equal' if rep['roundtrip'] else 'NOT EQUAL'} ({rep['i1']['module']})", flush=True)
    return bool(rep["roundtrip"])


def main(a: Options) -> int:
    try:
        roots = None
        if a.weights_of_record:
            serve = Path(a.run) / "serve.json"
            roots = registered_weights(json.loads(Path(a.weights_of_record).read_text()),
                                       json.loads(serve.read_text()) if serve.exists() else None)
        rep = write(a.run, a.out, trace=a.trace, io=a.io, T=a.T, tau=a.tau, window=a.window,
                    compiler=registered_compiler(a.compiler_name, a.compiler_digest, a.cap), registered_roots=roots)
    except ValueError as e:
        raise UsageError(f"{a.run}: {e}") from e
    if rep.get("schema") != SERVED:
        return 0 if _say(rep) else 1
    ok = [_say(r, f" commit window {w}:") for w, r in rep["reports"].items()]
    for w, why in rep["refused"].items():
        print(f"[advice] commit window {w}: refused: {why}", flush=True)
    for w, why in rep["pending"].items():
        print(f"[advice] commit window {w}: pending: {why}; a later call takes it", flush=True)
    index = f"{Path(a.out) if a.out else Path(a.run) / 'advice'}/windows.json"
    print(f"[advice] {len(ok)} commit windows committed, {len(rep['refused'])} refused, {len(rep['pending'])} pending, open "
          f"{rep['open']}; " + ("no index until none is pending" if rep["pending"] else f"index {index}"), flush=True)
    if rep["pending"]:
        return PENDING if all(ok) else 1
    return 0 if ok and all(ok) else 1
