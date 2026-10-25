"""negatives: what network accounting of arbitrary vLLM usage must refuse, beside honest traffic (G-B of
note:network-accounting/20261024T0429Z-draft-arbitrary-vllm-plan), each a pure function on a run's artifacts that
writes a tampered copy, and each naming the check that refuses it by its refusal code.

    python benchmarks/network_traces/negatives.py table [--markdown]
    python benchmarks/network_traces/negatives.py make NAME --run DIR --out DIR [--advice FILE] [--commit FILE]
    python benchmarks/network_traces/negatives.py n2 --run DIR
    python benchmarks/network_traces/negatives.py judge NAME --run DIR

**A run** is a certifier-serve run's directory (`certifier_serve.py rehearse`'s, one per strategy): its `plan.json`,
`calibration.json`, `deployment.json` and `certified.jsonl`, and, as their interfaces land, the advice (I1, ADV-SPEC:
`verity.core.protocols.certifier.advice`, its `encode` bytes) and the Commit record (I3, SERVE-COMMIT).

**What is tampered, by whom** (`Negative.tampers`):
- ``plan``: the shim's plan (`active_replay.adversary_plan`), served by `certifier_serve.py serve --tamper NAME`. The
  certifier certifies what the shim wrote, so a tampering of the I/O that the certifier signs is the shim's, made while
  serving; `rehearse --strategies honest,NAME,...` runs it. `make` writes the shim's plan (`shim.json`).
- ``certified``: the certified records the prover commits (`certified.jsonl`) after the run, under the certificates the
  certifier issued.
- ``frames``: the certified frames the prover opens under each record's frames root (the I/O tie's reads).
- ``advice``: the advice string the prover hands the compiler after serving (I1).
- ``commit``: the Commit record of the run's committed values (I3).

**The checks and their codes** (`Negative.code`). A code is ``CHECK.PART``: the check that refuses it, and the part of
it that fails. On the proof service's stream every refusal shows only ``reject`` (the violation's kind stays in the
verifier); the code is what the verifier of record establishes, which `n2` reads here in the clear for N2's checks:
- ``NetworkHeadCheck_v2.{opening,fields,counts,signed}``: N2's head check of each end (`head_check.reference_parts`):
  the head opens under the registered root, its fields (link, end, window, grid, status complete), its counts the
  profile's row, and its record digest under the certifier's salt the certificate's signed digest;
- ``NetworkServedCounts_v1``: N2's served counts on a served link (`served.reference`);
- ``audit.VIOLATION``: N2's audit of both records (`audit.Audit`, the Python reference of the Lean check of record);
- pending checks, by the names their drafts give them: ``NetworkAdvisedCounts_v1`` (N2-ADVICE's advised counts on an
  advised link, `LinkWindows.advised_check`: the egress head's counts the release of the advice's frames, the ingress
  head's the request frames the advice needs; ``.view`` when the advice is malformed or doesn't cover the window, which
  refuses every window), ``NetworkIoTie_v1`` (IO-TIE's tie of a drawn frame's token id to the Commit's at its (request,
  position)) and ``NetworkFrameRead`` (its read of a frame under the record's frames root); and, until their owners
  name them, ``IoUnit.render`` (under the attested-edge design, the egress unit rendering committed outputs to bytes),
  ``StepUnit`` (PROVE-PATH: the compiled step Program's digest against the committed trace's, and the committed values
  passing it) and ``Commit`` (the Commit's opening under its registered root).

`Negative.waits` names the interfaces a negative waits on; one with none runs today. `table` renders the registry.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import active_replay as ar  # noqa: E402
import certifier_serve as cs  # noqa: E402
from verity.core.protocols.certifier import audit as host_audit, head_check as HC  # noqa: E402
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402

FORMAT = "verity-network-negatives/v1"
ADVICE_MODULE = "verity.core.protocols.certifier.advice"
ADVISED_MODULE = "verity.core.protocols.certifier.advised"
SAMPLED = "runner.sampler/sampled_token_ids"
HEAD = "NetworkHeadCheck_v2"
SERVED = "NetworkServedCounts_v1"
ADVISED = "NetworkAdvisedCounts_v1"
TIE = "NetworkIoTie_v1"
CHECKS_TODAY = (HEAD, SERVED, "audit")  # the checks `n2` runs; every other code's check is pending
ENDS = ("egress", "ingress")


class NotApplicable(ValueError):
    """The run has nothing this tampering can change: say, advice with no chunked prefill to split differently."""


# --- the certified records ---------------------------------------------------------------------------------------------

def copy_run(src: Path, dst: Path) -> Path:
    """A copy of the run directory `src` at `dst`, which must not exist."""
    shutil.copytree(src, dst)
    return dst


def write_certified(path: Path, records: dict[tuple[Direction, int], CertifiedRecord], refused: list) -> None:
    """`certified.jsonl` as `certifier_serve.CertifyingPublisher` writes it: one line per record, by window and end,
    then the link-windows the signer refused."""
    lines = [cs.certified_json(records[k]) for k in sorted(records, key=lambda k: (k[1], k[0].code))]
    lines += [{"refused": r} for r in refused]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))


def one_count(record: CertifiedRecord, by: int = 1) -> CertifiedRecord:
    """The record with its head `by` frames off in one bucket (one more in its first bucket that has frames and room,
    else its first with room; one fewer in its first with frames), the certifier's salt and certificate unchanged."""
    counts, r = list(record.head.counts), record.head.allocation
    if by > 0:
        t = next((t for t, n in enumerate(counts) if 0 < n < r), None)
        t = t if t is not None else next(t for t, n in enumerate(counts) if n < r)
    else:
        t = next(t for t, n in enumerate(counts) if n > 0)
    counts[t] += by
    return CertifiedRecord(replace(record.head, counts=tuple(counts)), record.salt, record.certificate)


def tampered_head(src: Path, dst: Path, window: int = 0, by: int = 1) -> Path:
    """The run with window `window`'s egress head `by` frames off (`one_count`) under its original certificate: what a
    prover that commits a head other than the certified one hands over (N2-ZK's tampered head)."""
    records, refused = cs.read_certified(src / "certified.jsonl")
    key = (Direction.EGRESS, window)
    if key not in records:
        raise NotApplicable(f"{src}: no certified egress record of window {window}")
    copy_run(src, dst)
    records[key] = one_count(records[key], by)
    write_certified(dst / "certified.jsonl", records, refused)
    return dst


def frame_byte(frames: list[bytes], k: int = 0, at: int = -1) -> list[bytes]:
    """The window's frames with byte `at` of frame `k` flipped (its low bit), every other frame and the record's frames
    root kept: what a prover that opens a frame other than the certified one hands over. Its path no longer reaches
    the root (`certificate.verify_frame`), which is what the frame read of the I/O tie (`NetworkFrameRead`), or the
    warden's root under the attested-edge design, refuses."""
    out = list(frames)
    f = bytearray(out[k])
    f[at] ^= 1
    out[k] = bytes(f)
    return out


def frames_edit(src: Path, dst: Path) -> Path:
    raise NotApplicable("the run keeps its frames' root only (`certified.jsonl`'s heads); the frames negative waits "
                        "for a record that keeps the frames (WARDEN's attested cells, IO-TIE's frame reads)")


def shim(strategy: str) -> Callable[[Path, Path], Path]:
    """The tampering of `strategy` as the shim serves it: the run's plan as `active_replay.adversary_plan` makes it, in
    `shim.json` beside a copy of the run's plan and calibration (`certifier_serve.py serve --tamper STRATEGY` serves it)."""
    def make(src: Path, dst: Path) -> Path:
        plan, report = (json.loads((src / f).read_text()) for f in ("plan.json", "calibration.json"))
        dst.mkdir(parents=True)
        for f in ("plan.json", "calibration.json"):
            shutil.copy(src / f, dst / f)
        (dst / "shim.json").write_text(json.dumps(ar.adversary_plan(plan, ar.Setup(report), strategy)))
        return dst
    make.__name__ = f"shim_{strategy}"
    return make


# --- the advice (I1) ---------------------------------------------------------------------------------------------------
# On I1 v0 (ADV-SPEC, `verity.core.protocols.certifier.advice`): Advice(requests, steps, release),
# RequestAdvice(id, prompt_len, output_len, finish), StepAdvice(rows), Row(request, kind, start, n, cache_hit,
# preempted, sources), Source(request, n), and `release` one egress bucket a kept output. Each tampering returns new advice and
# leaves its argument, and keeps the advice valid under v0's schedule rules (a row continues at its request's computed
# position, or at its cache hit when it admits the request; `kind` is decode exactly when n = 1 past the prompt; a row
# keeps a sample when it reaches its request's known tokens), so that what refuses it is a check of the advice against
# the certified I/O or the committed values, not its form. The advice is built once, at the end (`_build`): I1's
# classes refuse an invalid advice on construction.

def _rows(advice) -> list[tuple[int, int, Any]]:
    return [(s, j, row) for s, step in enumerate(advice.steps) for j, row in enumerate(step.rows)]


def _plain(advice) -> list[list]:
    return [list(step.rows) for step in advice.steps]


def _at(row, request, start: int, n: int, **changes):
    """`row` over `n` positions of `request` from `start`, of that span's kind (decode exactly when n = 1 past the
    prompt). The kinds are literals: `tests/test_seed_ports.py` reads a computed `kind=` in a `replace` as a port's."""
    if n == 1 and start >= request.prompt_len:
        return replace(row, kind="decode", start=start, n=n, **changes)
    return replace(row, kind="prefill", start=start, n=n, **changes)


def _kept(requests, steps: list[list]) -> list[tuple[int, int, int, int]]:
    """(step, row, request, index) of each kept output of `steps` (lists of rows), in I1 v0's order (`advice.outputs`):
    a row keeps a sample when it reaches its request's known tokens (its prompt and the outputs kept before) while the
    request has kept fewer than its `output_len`."""
    known, kept, out = [q.prompt_len for q in requests], [0] * len(requests), []
    for s, rows in enumerate(steps):
        for j, row in enumerate(rows):
            r = row.request
            if row.start + row.n == known[r] and kept[r] < requests[r].output_len:
                out.append((s, j, r, kept[r]))
                kept[r], known[r] = kept[r] + 1, known[r] + 1
    return out


def sampling_rows(advice, request: int) -> list[tuple[int, int]]:
    """(step, row) of each row that keeps a sample of request `request`, in order."""
    return [(s, j) for s, j, r, _ in _kept(advice.requests, _plain(advice)) if r == request]


def _buckets(advice) -> dict[tuple[str, int], int]:
    """Each kept output's release bucket, by its request's id and its index."""
    kept = _kept(advice.requests, _plain(advice))
    return {(advice.requests[r].id, k): b for (_, _, r, k), b in zip(kept, advice.release)}


def _build(advice, requests: list, steps: list[list], buckets: dict[tuple[str, int], int]):
    """The advice of `requests` and `steps` (lists of rows; an empty one is no step), each kept output released in its
    bucket in `buckets`, an output new to it in the bucket of the output before it."""
    steps = [rows for rows in steps if rows]
    release: list[int] = []
    for _, _, r, k in _kept(requests, steps):
        release.append(buckets.get((requests[r].id, k), release[-1] if release else min(buckets.values(), default=0)))
    step = type(advice.steps[0])
    return replace(advice, requests=tuple(requests), steps=tuple(step(tuple(rows)) for rows in steps),
                   release=tuple(release))


def output_length(advice, request: int = 0, by: int = 1):
    """Request `request`'s output length one more (`by` = 1) or one less (−1), the rest of the advice as served. One
    more: its row past its last kept sample keeps one (the row async scheduling runs past an EOS), or, if it has none,
    a decode row in the step after its last row does. One less: its last kept sample discarded, as a row past an EOS
    is, and any row of it after that left out. The advice's frames then disagree with what the egress certified."""
    if by not in (1, -1):
        raise ValueError("an output length one more (by = 1) or one less (by = -1)")
    q, steps = advice.requests[request], _plain(advice)
    if q.output_len + by < 1:
        raise NotApplicable(f"request {request} keeps one output: it can't keep fewer")
    requests = list(advice.requests)
    requests[request] = replace(q, output_len=q.output_len + by)
    if by == -1:
        s0 = sampling_rows(advice, request)[-1][0]
        steps = [[row for row in rows if row.request != request or s <= s0] for s, rows in enumerate(steps)]
    elif sum(o[2] == request for o in _kept(requests, steps)) < requests[request].output_len:
        s, last = max((s, row) for s, rows in enumerate(steps) for row in rows if row.request == request)
        end = last.start + last.n
        row = _at(last, q, end, 1, cache_hit=0, preempted=False, sources=())
        if s + 1 == len(steps):
            steps.append([])
        steps[s + 1].append(row)
    if sum(o[2] == request for o in _kept(requests, steps)) != requests[request].output_len:
        raise NotApplicable(f"request {request}'s rows can't keep {requests[request].output_len} outputs")
    return _build(advice, requests, steps, _buckets(advice))


def missing_request(advice, request: int = -1):
    """Request `request` left out with its rows, the later requests' indices moved down: the advice serves one request
    fewer than the ingress certified."""
    i = range(len(advice.requests))[request]

    def moved(row):
        if any(src.request == i for src in row.sources):
            raise NotApplicable(f"a row reads request {i}'s KV: it can't be left out alone")
        return replace(row, request=row.request - (row.request > i),
                       sources=tuple(replace(src, request=src.request - (src.request > i)) for src in row.sources))

    steps = [[moved(row) for row in rows if row.request != i] for rows in _plain(advice)]
    return _build(advice, [q for k, q in enumerate(advice.requests) if k != i], steps, _buckets(advice))


def extra_request(advice, request: int = 0, suffix: str = "~extra"):
    """A request more: a copy of request `request` under another id, next to it in the requests' order, each of its
    rows right after the original's and each output released with the original's, so the advice is whole but serves
    one request more than the ingress certified."""
    i = range(len(advice.requests))[request]
    q = advice.requests[i]

    def moved(row, copy: bool = False):
        own = copy and row.preempted
        return replace(row, request=i + 1 if copy else row.request + (row.request > i),
                       sources=tuple(replace(src, request=i + 1 if own and src.request == i else
                                             src.request + (src.request > i)) for src in row.sources))

    steps = [[x for row in rows for x in ([moved(row), moved(row, True)] if row.request == i else [moved(row)])]
             for rows in _plain(advice)]
    requests = [*advice.requests[:i + 1], replace(q, id=f"{q.id}{suffix}"), *advice.requests[i + 1:]]
    buckets = _buckets(advice)
    buckets |= {(f"{q.id}{suffix}", k): b for (rid, k), b in buckets.items() if rid == q.id}
    return _build(advice, requests, steps, buckets)


def release_moved(advice, output: int = 0, by: int = 1):
    """Kept output `output`'s frame (the first by default) released `by` buckets later, and each output after it that
    was released before that bucket moved up to it, so the release stays in order. The advice's release disagrees
    with the egress buckets certified."""
    if not advice.release:
        raise NotApplicable("the advice keeps no output")
    rel = list(advice.release)
    k = range(len(rel))[output]
    rel[k] += by
    rel[k + 1:] = [max(b, rel[k]) for b in rel[k + 1:]]
    return replace(advice, release=tuple(rel))


def _covers(intervals: list[tuple[int, int]], lo: int, hi: int) -> bool:
    return set(range(lo, hi)) <= {p for a, b in intervals for p in range(a, b)}


def _hit_sources(advice, steps: list[list], s: int, row, hit: int):
    """`row`'s sources for a cache hit of `hit` positions at step `s`: its own runs cut to `hit`, or extended by the KV
    of a request other than its own, admitted by the end of step `s`, that computed the positions it adds (the run
    before, if that request computed them too); None if no request did."""
    runs = list(row.sources)
    if hit <= row.cache_hit:
        out, at = [], 0
        for src in runs:
            if (n := min(src.n, hit - at)) > 0:
                out.append(replace(src, n=n))
            at += src.n
        return tuple(out)
    made: dict[int, list[tuple[int, int]]] = {}
    for rows in steps[:s + 1]:
        for x in rows:
            made.setdefault(x.request, []).append((x.start, x.start + x.n))
    Source = type(runs[0]) if runs else getattr(sys.modules.get(type(row).__module__), "Source", None)
    if runs and runs[-1].request != row.request and _covers(made[runs[-1].request], row.cache_hit - runs[-1].n, hit):
        return tuple(runs[:-1] + [replace(runs[-1], n=runs[-1].n + hit - row.cache_hit)])
    donor = next((d for d in sorted(made) if d != row.request and _covers(made[d], row.cache_hit, hit)), None)
    return None if donor is None else tuple(runs + [Source(donor, hit - row.cache_hit)])


def start_shifted(advice, request: int | None = None, by: int = 1):
    """A request's first row (`request`'s, by default the first's that can) starting `by` positions later with its
    last position kept: a prefix-cache hit of `by` positions more, their KV named as another request's that computed
    them by that step (`by` < 0: fewer). Under v0 every other row's start is its request's continuation, so a row's
    start moves only with its cache hit. Its frames are the same, so N2 accepts it, but its compiled Program reads KV
    where the committed trace computed it."""
    steps, first = _plain(advice), {}
    for s, j, row in _rows(advice):
        first.setdefault(row.request, (s, j, row))
    for r in range(len(advice.requests)) if request is None else [range(len(advice.requests))[request]]:
        s, j, row = first[r]
        hit = row.cache_hit + by
        if hit < 0 or row.n - by < 1 or (sources := _hit_sources(advice, steps, s, row, hit)) is None:
            continue
        steps[s][j] = _at(row, advice.requests[r], hit, row.n - by, cache_hit=hit, sources=sources)
        return _build(advice, list(advice.requests), steps, _buckets(advice))
    raise NotApplicable(f"no request's first row can start {by} later: no other request computed the KV it would read")


def chunk_split(advice):
    """One prompt token moved across a chunk boundary of a request's prefill: of two consecutive chunks, the first one
    token shorter and the next one longer, starting one earlier; or, when no request's prefill is chunked, the first
    prefill row of two tokens or more split in two, its last token in a step of its own right after it. Every request
    is served the same positions with the same outputs, released in the same buckets, so the I/O agrees; the steps
    compile to other Programs than the committed trace realized."""
    steps, requests, last = _plain(advice), advice.requests, {}
    for s, j, b in _rows(advice):
        if b.request in last and b.start < requests[b.request].prompt_len and not b.preempted:
            t, k = last[b.request]
            a = steps[t][k]
            if a.n >= 2:
                steps[t][k] = replace(a, n=a.n - 1)
                steps[s][j] = replace(b, kind="prefill", start=b.start - 1, n=b.n + 1)
                return _build(advice, list(requests), steps, _buckets(advice))
        last[b.request] = (s, j)
    for s, j, a in _rows(advice):
        q = requests[a.request]
        if a.start < q.prompt_len and a.n >= 2:
            end = a.start + a.n
            steps[s][j] = replace(a, n=a.n - 1)
            steps.insert(s + 1, [_at(a, q, end - 1, 1, cache_hit=0, preempted=False, sources=())])
            return _build(advice, list(requests), steps, _buckets(advice))
    raise NotApplicable("no prefill row of two tokens or more to split")


def advice_module():
    """I1's module (ADV-SPEC), or None until it lands."""
    try:
        return importlib.import_module(ADVICE_MODULE)
    except ModuleNotFoundError:
        return None


def advice_edit(edit: Callable) -> Callable[[Path, Path, Path], Path]:
    """`edit` on the advice at `advice` (I1's `encode` bytes), written to `dst/advice.bin` beside a copy of the run."""
    def make(src: Path, dst: Path, advice: Path) -> Path:
        mod = advice_module()
        if mod is None:
            raise NotApplicable(f"no advice codec yet: {ADVICE_MODULE} (I1, ADV-SPEC) hasn't landed")
        copy_run(src, dst)
        (dst / "advice.bin").write_bytes(mod.encode(edit(mod.decode(advice.read_bytes()))))
        return dst
    make.__name__, make.edit = f"advice_{edit.__name__}", edit
    return make


# --- the Commit record (I3) --------------------------------------------------------------------------------------------
# On a stand-in of I3: per step, each batch row's request, the committed `c00.input/ids` and
# `runner.sampler/sampled_token_ids`, under the Commit's registered root. SERVE-COMMIT's record (`serve_openai.py`'s
# `windows.jsonl` and `requests.jsonl`) holds each window's run root and step roots and each request's (window, step,
# row range), not the committed values, which stay in the committer's host copy; the stand-in is those values.

def wrong_token(commit: dict, step: int = 0, row: int = 0, vocab: int | None = None) -> dict:
    """The committed sampled id of `row` at `step` changed after the run (+1, mod `vocab` when given), the Commit's
    registered root kept: its opening no longer reproduces the root, so the tie and the step unit each refuse it."""
    out = json.loads(json.dumps(commit))
    ids = out["steps"][step][SAMPLED]
    ids[row] = (ids[row] + 1) % vocab if vocab else ids[row] + 1
    return out


def commit_edit(edit: Callable) -> Callable[[Path, Path, Path], Path]:
    def make(src: Path, dst: Path, commit: Path) -> Path:
        copy_run(src, dst)
        (dst / "commit.json").write_text(json.dumps(edit(json.loads(commit.read_text()))))
        return dst
    make.__name__ = f"commit_{edit.__name__}"
    return make


# --- the registry ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Negative:
    """One tampering: its `family` (the goal's four: `io`, `advice`, `token`, `circuit`), what it `tampers` (`plan`,
    `certified`, `frames`, `advice` or `commit`), the `check` that must refuse it and its refusal `code` (`served`: the code on a
    served link where it differs), the interfaces it `waits` on (owner names; none: it runs today), and `make`, the pure
    function that writes the tampered copy."""

    name: str
    family: str
    tampers: str
    what: str
    check: str
    code: str
    make: Callable
    served: str | None = None
    waits: tuple[str, ...] = ()
    also: tuple[str, ...] = field(default=())

    def code_on(self, profile: str | None) -> str:
        return self.served if profile == "served" and self.served else self.code

    @property
    def pending(self) -> bool:
        return bool(self.waits)


NEGATIVES = {n.name: n for n in [
    Negative("late", "io", "plan", "the shim holds a window-0 session past every sync; the certifier faults the window",
             "N2's head check (the record's status)", f"{HEAD}.fields", shim("late"),
             also=("audit.status:coverage-never-established",)),
    Negative("extra", "io", "plan", "the shim writes one event more than the endpoint sent; certified",
             "N2's counts against the profile", f"{HEAD}.counts", shim("extra"), served=SERVED),
    Negative("drop", "io", "plan", "the shim drops the first event of a response; certified",
             "N2's counts against the profile", f"{HEAD}.counts", shim("drop"), served=SERVED,
             also=(ADVISED, TIE)),
    Negative("head", "io", "certified", "window 0's egress head one count off, its certificate the original's",
             "N2's head check (the signed digest)", f"{HEAD}.signed", tampered_head, also=("audit.signature",)),
    Negative("token", "io", "plan", "the shim writes every token of window 0's responses with other bytes than the "
             "endpoint sent: certified, its counts honest's", "the I/O tie", TIE, shim("token"),
             waits=("IO-TIE", "CERT-VARLEN"), also=("IoUnit.render",)),
    Negative("frame", "io", "frames", "one byte of a certified frame changed, the record's frames root kept",
             "the I/O tie's frame read (under the attested-edge design, the warden's root)", "NetworkFrameRead",
             frames_edit, waits=("IO-TIE", "WARDEN")),
    Negative("advice-length", "advice", "advice", "a request's output length one more than its certified frames, "
             "its next row keeping the sample", "N2's advised counts", ADVISED, advice_edit(output_length),
             waits=("ADV-SPEC", "N2-ADVICE")),
    Negative("advice-missing", "advice", "advice", "a certified request missing from the advice, with its rows",
             "N2's advised counts", ADVISED, advice_edit(missing_request), waits=("ADV-SPEC", "N2-ADVICE"),
             also=(TIE,)),
    Negative("advice-extra", "advice", "advice", "a request in the advice that the ingress didn't certify",
             "N2's advised counts", ADVISED, advice_edit(extra_request), waits=("ADV-SPEC", "N2-ADVICE"),
             also=(TIE,)),
    Negative("advice-release", "advice", "advice", "a kept output released one bucket later than certified",
             "N2's advised counts", ADVISED, advice_edit(release_moved), waits=("ADV-SPEC", "N2-ADVICE")),
    Negative("advice-start", "advice", "advice", "a request's first row starting one position later, claiming a "
             "prefix-cache hit on another request's KV, its end kept: N2 accepts it",
             "the step unit (committed values against the compiled Program)", "StepUnit.values",
             advice_edit(start_shifted), waits=("ADV-SPEC", "COMPILER", "PROVE-PATH")),
    Negative("wrong-token", "token", "commit", "a committed sampled id changed after the run; the registered root kept",
             "the I/O tie and the step unit (the Commit's opening)", "Commit.opening", commit_edit(wrong_token),
             waits=("SERVE-COMMIT", "IO-TIE", "PROVE-PATH"), also=(TIE, "StepUnit.opening")),
    Negative("advice-chunk", "circuit", "advice", "a prefill chunk split one token differently (an unchunked one in "
             "two): N2 accepts it",
             "the step unit (the compiled Program's digest against the committed trace's)", "StepUnit.program",
             advice_edit(chunk_split), waits=("ADV-SPEC", "COMPILER", "PROVE-PATH")),
]}


def make(name: str, src: Path, dst: Path, *, advice: Path | None = None, commit: Path | None = None) -> Path:
    """The tampered copy of the run `src` for negative `name`, written at `dst`."""
    n = NEGATIVES[name]
    if n.tampers == "advice":
        if advice is None:
            raise NotApplicable(f"{name} tampers the advice: give it (--advice)")
        return n.make(src, dst, advice)
    if n.tampers == "commit":
        if commit is None:
            raise NotApplicable(f"{name} tampers the Commit record: give it (--commit)")
        return n.make(src, dst, commit)
    return n.make(src, dst)


def table(markdown: bool = True) -> str:
    """The registry, one row per negative: what it tampers, the check that refuses it, its code, and its status."""
    q = "`" if markdown else ""
    rows = [("negative", "family", "tampers", "check", "code", "status")]
    for n in NEGATIVES.values():
        code = f"{q}{n.code}{q}" + (f" (served: {q}{n.served}{q})" if n.served else "")
        rows.append((f"{q}{n.name}{q}", n.family, n.tampers, n.check, code,
                     f"pending: {', '.join(n.waits)}" if n.waits else "runs today"))
    if not markdown:
        return "\n".join("\t".join(r) for r in rows)
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


# --- N2's refusal codes ------------------------------------------------------------------------------------------------

def link(run: Path) -> tuple[LinkWindows, dict]:
    """The run's link and its plan, as `certifier_serve.check` builds them."""
    plan, report, dep = (json.loads((run / f).read_text()) for f in ("plan.json", "calibration.json", "deployment.json"))
    setup = ar.Setup(report)
    profile = plan.get("occupancy") or {}
    lw = LinkWindows(dep["links"]["n2"], setup.egress_params, setup.sync_set, cs.read_public(dep["certifier_key"]),
                     dep["run_id"], bytes.fromhex(dep["session"]), ingress=setup.ingress_params,
                     proof_deadline=cs.PROOF_DEADLINE_S, occupancy=(profile.get("egress"), profile.get("ingress")),
                     served=cs.served_profile(plan))
    return lw, plan


def codes(lw: LinkWindows, opening, root: bytes, syncs: list[ClockSync]) -> list[str]:
    """What fails of N2's check on the opening under `root`, as codes: each end's head check parts, the served counts
    on a served link, and each end's audit violation (`audit.Audit`, the Lean check's reference)."""
    out = []
    for end, inst in zip(ENDS, lw.head_checks(opening, root)):
        if inst is None:
            out.append(f"{HEAD}.shape")
        else:
            out += [f"{HEAD}.{part}" for part, ok in HC.reference_parts(inst).items() if not ok]
    if lw.served is not None and not lw.served_check(opening, syncs):
        out.append(SERVED)
    egress, ingress = lw._audit(host_audit.Audit(), opening, syncs)
    w = opening.window
    out += [f"audit.{v[w].violation}" for v in (egress, ingress) if len(v) > w and v[w].violation is not None]
    return sorted(set(out))


def link_received(advice, request_frames, received: list[int]) -> list[int]:
    """`received` as a prover declares it beside `advice`: the request frames the advice's requests take under the
    link's framing (`request_frames[prompt_len − 1]` each) that `received` doesn't sum to go to its first window that
    receives any, as a prover that serves other requests than were received would declare them, so the advice stays
    well formed (`advised.ends`) and only the counts can refuse it."""
    got = list(received)
    w = next((w for w, g in enumerate(got) if g > 0), 0)
    got[w] += sum(request_frames[q.prompt_len - 1] for q in advice.requests
                  if q.prompt_len <= len(request_frames)) - sum(got)
    return got


def n2_advice(advice, request_frames, received: list[int]):
    """I1 v0 advice as N2-ADVICE's `advised` reads it (`advised.LinkAdvice`: the advice and each window's `received`,
    `link_received`'s). Raises ModuleNotFoundError until N2-ADVICE lands."""
    AD = importlib.import_module(ADVISED_MODULE)
    return AD.LinkAdvice(advice, tuple(link_received(advice, request_frames, received)))


def advised_codes(link, window: int, syncs, sync_set, request_frames, T: int, R: int, RIN: int, OWN: int, ingress,
                  egress) -> list[str]:
    """N2-ADVICE's code on one advised link-window with heads of counts `ingress` and `egress`, as
    `LinkWindows.advised_check` decides it (`advised.view` under the declared `syncs`, Σ_sync and the link's framing,
    then `advised.reference`), on the advice as N2 reads it (`n2_advice`): ``.view`` when the advice is malformed or
    doesn't cover the window, the advised counts' when they refuse, else none. Raises ModuleNotFoundError until
    N2-ADVICE lands."""
    AD = importlib.import_module(ADVISED_MODULE)
    try:
        v = AD.view(link, window, T, R, syncs, sync_set, request_frames)
    except ValueError:
        return [f"{ADVISED}.view"]
    ok = AD.reference(T, RIN, OWN, ingress, egress, v.released, v.floor, v.received)
    return [] if ok else [ADVISED]


def honest_heads(link, T: int, R: int, syncs, sync_set, request_frames,
                 own: int = 0) -> list[tuple[list[int], list[int]]]:
    """Each window's (ingress, egress) head counts that an honest advised run of `link` (`n2_advice`) certifies under
    N2-ADVICE's rule: the egress the advice's releases, the ingress the request frames its floor needs by each bucket,
    the rest of the window's in its last bucket."""
    AD = importlib.import_module(ADVISED_MODULE)
    out = []
    for w in range(len(link.received)):
        v = AD.view(link, w, T, R, syncs, sync_set, request_frames)
        ingress, prev = [], 0
        for t, f in enumerate(v.floor):
            ingress.append(f - prev + (own if t == 0 else 0))
            prev = f
        ingress[-1] += v.received - prev
        out.append((ingress, list(v.released)))
    return out


def advice_suite(advice, *, T: int, R: int, syncs, request_frames, received: list[int], sync_set=None,
                 RIN: int | None = None) -> dict[str, dict]:
    """Every advice negative made from a run's `advice` (I1's): whether it applies to it, whether it's valid advice that
    round-trips through I1's codec, its bits, and N2-ADVICE's codes (`advised_codes`) on the heads an honest run of the
    advice certifies (`honest_heads`, under the link's framing `request_frames` with each window's `received` and
    declared sync; Σ_sync by default the declared syncs' deltas and rhos, `RIN` their largest ingress count). `ok`: the
    codes are exactly the one it names, or none for a step-unit negative, which N2 must accept (its check is
    pending). A negative that releases more than R frames in a bucket is refused by the view (``.view``) instead. An
    honest advice the view refuses has no honest heads, so its row says why and no negative runs."""
    mod = advice_module()
    if mod is None:
        raise NotApplicable(f"no advice codec yet: {ADVICE_MODULE} (I1, ADV-SPEC) hasn't landed")
    sync_set = sync_set or SyncSet(tuple(sorted({s.delta for s in syncs})), tuple(sorted({s.rho for s in syncs})))

    def codes(link) -> list[str]:
        return sorted({c for w, (i, e) in enumerate(heads)
                       for c in advised_codes(link, w, syncs, sync_set, request_frames, T, R, RIN, 0, i, e)})

    try:
        heads = honest_heads(n2_advice(advice, request_frames, received), T, R, syncs, sync_set, request_frames)
    except ValueError as err:
        why = f"N2-ADVICE's view refuses the honest advice: {err}"
        return {"honest": {"bits": mod.total_bits(advice), "codes": [f"{ADVISED}.view"], "ok": False, "why": why}} | {
            name: {"code": n.code, "check": n.check, "waits": list(n.waits), "ran": False, "why": why}
            for name, n in NEGATIVES.items() if n.tampers == "advice"}
    RIN = RIN or max(1, *(c for i, _ in heads for c in i))
    out = {"honest": {"bits": mod.total_bits(advice), "codes": codes(n2_advice(advice, request_frames, received))}}
    out["honest"]["ok"] = out["honest"]["codes"] == []
    for name, n in NEGATIVES.items():
        if n.tampers != "advice":
            continue
        row: dict[str, Any] = {"code": n.code, "check": n.check, "waits": list(n.waits)}
        try:
            b = n.make.edit(advice)
        except NotApplicable as err:
            out[name] = row | {"ran": False, "why": str(err)}
            continue
        got = codes(n2_advice(b, request_frames, received))
        want = [] if n.code.startswith("StepUnit") else [n.code]
        out[name] = row | {"ran": True, "valid": mod.decode(mod.encode(b)) == b, "bits": mod.total_bits(b),
                           "steps": len(b.steps), "outputs": len(b.release), "codes": got, "ok": got == want}
    return out


def advice_run(path: Path, *, T: int = 50, R: int = 32, frames: int = 1, delta: int = 0, rho: int = 1000,
               shift: int = 0) -> dict:
    """`advice_suite` on the advice at `path` (I1's encode bytes), each request `frames` ingress frames whatever its
    prompt's length, all received in window 0, every window's sync (δ, ρ): its shape and the rows (`negatives`). With
    `shift`, every release `shift` buckets later, for an advice whose clock put an output in its request's own bucket,
    which N2-ADVICE's floor refuses."""
    mod = advice_module()
    if mod is None:
        raise NotApplicable(f"no advice codec yet: {ADVICE_MODULE} (I1, ADV-SPEC) hasn't landed")
    adv = mod.decode(path.read_bytes())
    if shift:
        adv = replace(adv, release=tuple(b + shift for b in adv.release))
    windows = max(adv.release, default=0) // T + 1
    framing = (frames,) * max((q.prompt_len for q in adv.requests), default=1)
    rows = advice_suite(adv, T=T, R=R, syncs=[ClockSync(delta, Fraction(rho))] * windows, request_frames=framing,
                        received=[frames * len(adv.requests)] + [0] * (windows - 1))
    return {"format": FORMAT, "advice": str(path), "requests": len(adv.requests), "steps": len(adv.steps),
            "outputs": len(adv.release), "windows": windows, "T": T, "R": R, "shift": shift, "negatives": rows}


def n2(run: Path, clock=time.monotonic) -> dict:
    """N2's verdict on every link-window of the run's `certified.jsonl`, through the proof service's stream as
    `certifier_serve.check` takes it (Selection All, the Lean audit, `unproved_outcome`: nothing is proved, so the
    verdict reads the opening) without asking the certifier's salts, and the link's network verdict; each refused
    link-window's codes (`codes`), and their union."""
    lw, plan = link(run)
    syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
    st = lw.stream(budget=plan["windows"], delta=cs.DELTA, challenger="negatives", coins=cs._no_coins, clock=clock)
    certified, refused = cs.read_certified(run / "certified.jsonl")
    rows, previous = [], None
    for w in range(plan["windows"]):
        pair = certified.get((Direction.EGRESS, w)), certified.get((Direction.INGRESS, w))
        if None in pair:
            rows.append({"window": w, "accepted": False, "codes": ["no-certified-record"]})
            break
        opening = lw.commit(w, *pair)
        reg = lw.register(st, lw.registration(opening, previous))
        previous = reg.digest
        draw = st.select(reg)
        outcome = st.deliver(draw, [lw.session(opening, draw)]) or lw.unproved_outcome(st, draw, opening, syncs)
        found = codes(lw, opening, bytes.fromhex(reg.record["roots"][0]["root"]), syncs)
        rows.append({"window": w, "accepted": outcome.accepted, "symbols": list(outcome.verdict.codes),
                     "codes": found if not outcome.accepted else []})
        if outcome.accepted and found:
            rows[-1]["diagnosed"] = found
    network = lw.network_verdict(st, plan["windows"])
    return {"accepted": network["accepted"], "network": network, "windows": rows, "issuer_refused": refused,
            "profile": (plan.get("occupancy") or {}).get("kind"),
            "codes": sorted({c for r in rows for c in r["codes"]})}


def judge(name: str, verdict: dict) -> dict:
    """Whether `verdict` (`n2`'s on the negative's run) refuses negative `name` by its code: the network verdict
    rejects and a refused link-window's codes include the code the negative names on the run's profile."""
    n = NEGATIVES[name]
    code = n.code_on(verdict.get("profile"))
    return {"negative": name, "code": code, "refused": not verdict["accepted"],
            "by_code": not verdict["accepted"] and code in verdict["codes"], "codes": verdict["codes"],
            "pending": list(n.waits)}


def suite(runs: dict[str, Path], work: Path, *, advice: Path | None = None, commit: Path | None = None,
          clock=time.monotonic) -> dict[str, dict]:
    """Every negative's row for the one command's report, from a rehearsal's runs by strategy (`runs`, which must hold
    honest): a shim negative is its own run (`rehearse --strategies`), any other is made from honest's under `work`.
    A row says what the negative tampers, the check and code it names and what it waits on; whether it ran, and if so
    N2's verdict, its refused link-windows and their codes, and `ok`: refused by its code; for a shim negative whose
    check is pending (token), N2's accepting it, since only that check may refuse it; else None. An advice negative's
    verdict is `advice_run`'s, N2-ADVICE's codes on the heads an honest run of `advice` certifies (`honest_codes`
    theirs on the honest advice), since the certified records don't change with it; `ok` needs the honest advice
    accepted."""
    out, judged = {}, None
    for name, n in NEGATIVES.items():
        row = {"family": n.family, "tampers": n.tampers, "check": n.check, "code": n.code, "waits": list(n.waits)}
        run = runs.get(name)
        if run is None and n.tampers != "plan":
            try:
                run = make(name, runs["honest"], work / name, advice=advice, commit=commit)
            except NotApplicable as err:
                row |= {"ran": False, "why": str(err)}
        elif run is None:
            row |= {"ran": False, "why": f"no run of it: rehearse --strategies honest,{name},..."}
        if run is not None and n.tampers == "advice":
            if judged is None:
                try:
                    judged = advice_run(advice)["negatives"]
                except ModuleNotFoundError as err:
                    judged = {"why": f"N2-ADVICE's advised check hasn't landed ({err.name})"}
            got = judged.get(name)
            if got is None:
                row |= {"ran": False, "copy": str(run), "why": judged["why"]}
            else:
                honest = judged["honest"]
                row |= {"copy": str(run), "honest_codes": honest["codes"],
                        "heads": "an honest run of the advice's, under N2-ADVICE's rule"}
                row |= {k: v for k, v in got.items() if k not in ("code", "check", "waits")}
                if got["ran"]:
                    row["ok"] = got["ok"] and honest["ok"]
            run = None
        if run is not None:
            v = n2(run, clock=clock)
            got = judge(name, v)
            row |= {"ran": True, "run": str(run), "code": got["code"], "n2_accepted": v["accepted"],
                    "refused_windows": [r["window"] for r in v["windows"] if r["codes"]], "codes": v["codes"],
                    "by_code": got["by_code"]}
            if not n.pending:
                row["ok"] = got["by_code"]
            elif n.tampers == "plan":
                row["ok"] = v["accepted"]
            else:
                row["ok"] = None
        out[name] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="negatives.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table", help="the negatives, their checks and codes")
    t.add_argument("--markdown", action="store_true")
    m = sub.add_parser("make", help="write a negative's tampered copy of a run")
    m.add_argument("name", choices=list(NEGATIVES))
    m.add_argument("--advice", type=Path, help="the run's advice (I1's encode bytes)")
    m.add_argument("--commit", type=Path, help="the run's Commit record (I3)")
    m.add_argument("--out", required=True, type=Path)
    x = sub.add_parser("n2", help="N2's verdict of record on a run, with the refused link-windows' codes")
    j = sub.add_parser("judge", help="whether N2 refuses a negative's run by the code it names")
    j.add_argument("name", choices=list(NEGATIVES))
    for q in (m, x, j):
        q.add_argument("--run", required=True, type=Path)
    a = sub.add_parser("advice", help="every advice negative of a run's advice, and N2-ADVICE's codes on it")
    a.add_argument("--advice", required=True, type=Path, help="the run's advice (I1's encode bytes)")
    a.add_argument("--T", type=int, default=50, help="buckets a window (the advice's release buckets')")
    a.add_argument("--R", type=int, default=32, help="egress frames a bucket")
    a.add_argument("--frames", type=int, default=1, help="ingress frames a request")
    a.add_argument("--delta", type=int, default=0, help="each window's declared sync: δ buckets ...")
    a.add_argument("--rho", type=int, default=1000, help="... and ρ frames a bucket")
    a.add_argument("--shift", type=int, default=0, help="every release this many buckets later")
    args = ap.parse_args(argv)
    if args.command == "advice":
        try:
            doc = advice_run(args.advice, T=args.T, R=args.R, frames=args.frames, delta=args.delta, rho=args.rho,
                             shift=args.shift)
        except NotApplicable as err:
            raise SystemExit(str(err))
        print(json.dumps(doc, indent=1))
        return 0 if all(r.get("ok") is not False for r in doc["negatives"].values()) else 1
    if args.command == "table":
        print(table(args.markdown))
        return 0
    if args.command == "make":
        print(make(args.name, args.run, args.out, advice=args.advice, commit=args.commit))
        return 0
    verdict = n2(args.run)
    if args.command == "n2":
        print(json.dumps(verdict, indent=1))
        return 0
    out = judge(args.name, verdict)
    print(json.dumps(out))
    return 0 if out["by_code"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
