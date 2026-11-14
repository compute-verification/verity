"""negatives: what network certification of arbitrary vLLM usage must refuse, beside honest traffic, each naming the
check that refuses it by its refusal code, and the advice and Commit tamperings each a pure edit.

    python benchmarks/network_traces/negatives.py table [--markdown]
    python benchmarks/network_traces/negatives.py advice --advice FILE [--ids REQUESTS.jsonl] [--frame-bytes B]
        [--model M] [--T 50] [--R 32]
    python benchmarks/network_traces/negatives.py warden --run DIR

The advice is I1's (`verity.protocols.accounting.certifier.verifier.advice`, its `encode` bytes) and the Commit record
I3's (serve-openai's).

**What is tampered, by whom** (`Negative.tampers`):
- ``plan``: what the endpoint sent, tampered by a shim between the endpoint and the network certifier while serving:
  the network certifier certifies what the shim wrote.
- ``certified``: the certified records the prover commits (`certified.jsonl`) after the run, under the certificates the
  certifier issued.
- ``frames``: the certified frames the prover opens under each record's frames root (the I/O tie's reads).
- ``advice``: the advice string the prover hands the compiler after serving (I1).
- ``commit``: the Commit record of the run's committed values (I3).

**The checks and their codes** (`Negative.code`). A code is ``CHECK.PART``: the check that refuses it, and the part of
it that fails. On the proof system's stream every refusal shows only ``reject`` (the violation's kind stays in the
verifier); the code is what the verifier establishes:
- ``NetworkHeadCheck_v2.{opening,fields,counts,signed}``: N2's head check of each end (`head_check.reference_parts`):
  the head opens under the registered root, its fields (link, end, window, grid, status complete), its counts the
  profile's row, and its record digest under the certifier's salt the certificate's signed digest;
- ``NetworkServedCounts_v1``: N2's served counts on a served link (`served.reference`);
- ``audit.VIOLATION``: N2's audit of both records (`audit.Audit`, the Python reference of the Lean check of record);
- ``Issuer.refused``: no check's but the certifier's: it certifies no record whose status isn't complete
  (`certificate.Issuer`, which lists the link-window in `refused`), so the window has no certified record and the
  link stops there;
- pending checks, by the names their drafts give them: ``IoTieRequest`` (`io_request`'s ingress tie, a unit a bucket
  row of the warden's ingress link-window at selection All, `io_request.reference`: each lane's cells, read from its
  carry-in, are its connection's pads and canonical id-request cells, leaving its carry-out, on the verifier's lanes
  from the advice's prompt lengths and the warden's release; its verdict on every window, under the advice's digest
  and with the release giving `received`, is the tie, `tie_service.IoTieVerdict`),
  ``IoTieWindow.{cells_root,port,content}`` (the I/O tie, a unit at selection All over one egress link-window,
  `io_tie.reference_parts`: the network certifier's cells open under their root, the Commit's port rows of sampled ids
  open at each bucket's steps under the Commit's root, and every cell is the verifier's map's, computed from advice v0
  and the network certifier's admissions: a `PAD` or the `END` on its request's stream, B zero bytes in a slot no
  connection holds, or a data cell on the request's stream whose id is the port entry the map names; ``.map`` when the
  verifier can't compute the map, an advice whose release isn't the network certifier's schedule or whose requests
  aren't its connections, which refuses every window); and, until their owners name them,
  ``StepUnit`` (the prove path's: the
  compiled step Program's digest against the committed trace's, and the committed values passing it) and ``Commit``
  (the Commit's opening under its registered root).

**The tie on a fake upstream** (`tie_suite`): the network certifier's cells of a run of the advice served through a
network certifier of r slots a connection (`tie_schedule`, `tie_served`: each request admitted at its first release, its
outputs released on the network certifier's schedule), each kept output a data cell carrying its sampled id (`--ids`,
`verity-vllm advice`'s `requests.jsonl`), and the Commit's port, a row of sampled ids a step, both committed; each
negative it reads is the tampered cells, port or advice (whose map the verifier computes), and `io_tie.reference_parts`
names the part that refuses it (`TIE_PARTS`).

**The ingress tie on a fake upstream** (`advice_suite`): the warden's own ingress of the advice's requests, id
requests at the link's cell size (each from its record, `--ids` as serve-openai writes it: prompt ids, user and what it
brought), each window's arriving whole so their request cells are the honest `received`'s (`certified_ingress`), under
the warden's framing (`link_framing`). Each advice negative is IoTieRequest's reference on every bucket row on the
verifier's lanes from its advice and the warden's release (`request_tie`), whose verdict the verifier takes
from its own tie source (`tie_service.Ties`).

`Negative.waits` names the interfaces a negative waits on; one with none runs. `table` renders the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # the root, for benchmarks.network_traces

from verity.protocols.accounting.certifier.verifier import record_files as RF  # noqa: E402
from verity.protocols.accounting.certifier.verifier import relay_check as RC  # noqa: E402
from verity.protocols.accounting.certifier.verifier.link_frame import (DATA, HEADER,  # noqa: E402
                                                                       )
from verity.protocols.accounting.certifier.verifier.id_form import ID_NAME_BYTES, id_request  # noqa: E402
from verity.protocols.accounting.certifier.verifier import tie_service as TS, warden as W  # noqa: E402

FORMAT = "verity-network-negatives/v1"
ADVICE_MODULE = "verity.protocols.accounting.certifier.verifier.advice"
IO_TIE_MODULE = "verity.protocols.accounting.certifier.verifier.io_tie"
IO_REQUEST_MODULE = "verity.protocols.accounting.certifier.verifier.io_request"
WARDEN_ROWS_MODULE = "verity.protocols.accounting.certifier.verifier.warden_rows"
SAMPLED = "runner.sampler/sampled_token_ids"
HEAD = "NetworkHeadCheck_v2"
SERVED = "NetworkServedCounts_v1"
TIE = "IoTieWindow"
REQUEST_TIE = "IoTieRequest"
#: the verifier's registered read of a network certifier cell at its position under the network certifier's signed root
#: (`warden_rows.opens`, the Lean verifier's check of a proof block's registered port)
WARDEN_READ = "Flock.Registered.checkPort"
#: the part of the I/O tie (`TIE`) that refuses each negative it reads; None: one it must accept while its outputs are
#: honest's (each one's step, batch row, request, index and bucket), which content refuses once the edit moves them
TIE_PARTS = {"drop": "content", "token": "content", "frame": "cells_root", "advice-length": "content",
             "advice-missing": "content", "advice-extra": "map", "advice-release": "map", "advice-start": None,
             "wrong-token": "port", "advice-chunk": None, "prompt-split": None}
ISSUER_REFUSED = "Issuer.refused"
FRAME_BYTES = 4096  # the link's frame size


class NotApplicable(ValueError):
    """The run has nothing this tampering can change: say, advice with no chunked prefill to split differently."""


# --- the network certifier's cells -------------------------------------------------------------------------------------


def frame_byte(frames: list[bytes], k: int = 0, at: int = -1) -> list[bytes]:
    """The window's frames with byte `at` of frame `k` flipped (its low bit), every other frame and the record's frames
    root kept: what a prover that opens a frame other than the certified one hands over. Its path no longer reaches
    the root (`certificate.verify_frame`); under the network certifier, the I/O tie recomputes the cells' root and
    refuses it (`TIE`.cells_root, `tie_suite`)."""
    out = list(frames)
    f = bytearray(out[k])
    f[at] ^= 1
    out[k] = bytes(f)
    return out


def cell_reads(WR, cells, k: int, row: bytes) -> bool:
    """Whether `row`, opened at position `k` of the network certifier's committed `cells` (`warden_rows.Cells`, a row a
    bucket) under that position's salt and path, climbs to their root: the verifier's registered read
    (`warden_rows.opens`)."""
    return WR.opens(cells.domain, cells.root, k, WR.bc(row, cells.salts[k]), cells.path(k))


def warden_frame(run: Path, at: int | None = None) -> dict:
    """`frame` on a warden-serve run's own cells (`warden_rows`: `cells.jsonl`'s registrations and the
    prover's `cells.salts`, the rows' salts and the wiring's): the first egress execution phase with a data cell, its
    registration (v2) checked under the network certifier's key and the schedule, and its record's bucket rows and
    wiring committed under their salts to the registered head; then byte `at` of that cell (its first payload byte by
    default) changed in its bucket's row, and the row opened at its position. The honest row's read climbs to the
    signed root and the changed one's doesn't (`WARDEN_READ`)."""
    try:
        WR = importlib.import_module(WARDEN_ROWS_MODULE)
        WS = importlib.import_module("benchmarks.network_traces.warden_serve")
    except ModuleNotFoundError as err:
        raise NotApplicable(f"no warden cells: {err.name} isn't in this tree") from None
    link, direction = RC.CERTIFIED[RC.EGRESS]
    if missing := [f for f in ("cells.salts", f"{RC.EGRESS}.records") if not (run / f).is_file()]:
        raise NotApplicable(f"{run}: no {' or '.join(missing)}: the run kept its registrations only (public), so "
                            "no unit can open its cells")
    dep = json.loads((run / "deployment.json").read_text())
    params = WS.WardenParams(**dep["params"])
    key, session = RF.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = {r.window: r for r in RC.read_records(run, params)[RC.EGRESS]}
    regs = {r["window"]: (r, salts, wiring_salt) for d, salts, wiring_salt in RC.read_cells(run)
            if (r := d.get("registration")) and (r["link"], r["direction"]) == (link, direction.value)}
    B = params.cell_bytes
    for w in sorted(set(regs) & set(records)):
        rows = WR.window_rows(records[w], params)
        t, j = next(((t, j) for t, r in enumerate(rows) for j in range(len(r) // B) if r[j * B] == DATA), (None, None))
        if t is None:
            continue
        reg, salts, wiring_salt = regs[w]
        head, _ = WR.check_registration(reg, key, session, params, link, direction, w)
        cells = WR.Cells.of(link, direction, records[w], params, session, salts, wiring_salt)
        if cells.head != head:
            raise ValueError(f"{run}: egress window {w}'s rows and wiring under their salts aren't its registered head")
        bad = bytearray(rows[t])
        bad[j * B + (HEADER.size if at is None else at % B)] ^= 1
        honest, tampered = cell_reads(WR, cells, t, rows[t]), cell_reads(WR, cells, t, bytes(bad))
        return {"negative": "frame", "window": w, "row": t, "slot": j, "rows": len(rows), "cell_bytes": B,
                "row_bytes": len(rows[t]), "code": WARDEN_READ, "honest_reads": honest, "tampered_reads": tampered,
                "ok": honest and not tampered}
    raise NotApplicable(f"{run}: no registered egress link-window with a data cell")


# --- the advice (I1) ---------------------------------------------------------------------------------------------------
# On I1 v0 (`verity.protocols.accounting.certifier.verifier.advice`): Advice(requests, steps, release),
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


def _relaid(steps: list[list], r: int, request, prompt_len: int) -> list[list]:
    """`steps` with request `r`'s rows for a prompt of `prompt_len` tokens: a shorter one cut from the front of its
    prefill, each chunk keeping a token, a longer one's last chunk longer, and every row after the prompt moved with
    it, so every row stays in its step and the chunk that ends the prompt ends the new one: each kept output stays
    where it was. Raises NotApplicable for a request whose rows read cached KV (a cache hit, a resume, another
    request's KV), or whose prompt has more chunks than `prompt_len` tokens."""
    by, end = prompt_len - request.prompt_len, request.prompt_len
    chunks = sum(row.request == r and row.start < end for rows in steps for row in rows)
    if prompt_len < chunks:
        raise NotApplicable(f"request {r}'s prompt is prefilled in {chunks} chunks: a {prompt_len}-token one can't be")
    cut, at, out = max(0, -by), 0, []
    for rows in steps:
        out.append([])
        for row in rows:
            if row.request != r:
                out[-1].append(row)
                continue
            if row.cache_hit or row.preempted or row.sources:
                raise NotApplicable(f"request {r}'s rows read cached KV: its prompt can't move alone")
            if row.start >= end:
                out[-1].append(replace(row, start=row.start + by))
                continue
            take = min(cut, row.n - 1)
            n, cut = row.n - take + max(0, by) * (row.start + row.n == end), cut - take
            out[-1].append(replace(row, start=at, n=n))
            at += n
    return out


def prompt_split(advice, request: int | None = None):
    """Request `request`'s and the next one's prompt lengths swapped, their total kept (as (338,
    16) served, (16, 338) in the advice): by default the first request whose prompt is longer than the next one's,
    else the first whose prompt is shorter, each prefilled to the other's length and decoded on from there
    (`_relaid`), the rest as served, so every output is honest's, released in its bucket. Where the two take different
    frames under the link's framing, the earlier one's end moves (down, for a longer first: phase 6's acausal release).
    Either way the requests' frames are honest's, so only the tie of each request's prompt length to the certified
    ingress (IoTieRequest) refuses it."""
    qs, steps, why = advice.requests, _plain(advice), "no request's prompt length differs from the next one's"
    pairs = [range(len(qs) - 1)[request]] if request is not None else \
        sorted((j for j in range(len(qs) - 1) if qs[j].prompt_len != qs[j + 1].prompt_len),
               key=lambda j: qs[j].prompt_len < qs[j + 1].prompt_len)
    for j in pairs:
        lens = {j: qs[j + 1].prompt_len, j + 1: qs[j].prompt_len}
        if lens[j] == lens[j + 1]:
            why = f"requests {j} and {j + 1} have one prompt length"
            continue
        if any(src.request in lens for rows in steps for row in rows for src in row.sources):
            why = f"a row reads request {j}'s or {j + 1}'s KV"
            continue
        try:
            moved = _relaid(_relaid(steps, j, qs[j], lens[j]), j + 1, qs[j + 1], lens[j + 1])
        except NotApplicable as err:
            why = str(err)
            continue
        requests = [replace(q, prompt_len=lens[k]) if k in lens else q for k, q in enumerate(qs)]
        return _build(advice, requests, moved, _buckets(advice))
    raise NotApplicable(why)


def advice_module():
    """I1's module (`advice`), or None where it isn't installed."""
    try:
        return importlib.import_module(ADVICE_MODULE)
    except ModuleNotFoundError:
        return None


# --- the Commit record (I3) --------------------------------------------------------------------------------------------
# On a stand-in of I3: per step, each batch row's request, the committed `c00.input/ids` and
# `runner.sampler/sampled_token_ids`, under the Commit's registered root. serve-openai's record (`serve_openai.py`'s
# `windows.jsonl` and `requests.jsonl`) holds each window's run root and step roots and each request's (window, step,
# row range), not the committed values, which stay in the committer's host copy; the stand-in is those values.

def wrong_token(commit: dict, step: int = 0, row: int = 0, vocab: int | None = None) -> dict:
    """The committed sampled id of `row` at `step` changed after the run (+1, mod `vocab` when given), the Commit's
    registered root kept: its opening no longer reproduces the root, so the tie and the step proof block each refuse
    it."""
    out = json.loads(json.dumps(commit))
    ids = out["steps"][step][SAMPLED]
    ids[row] = (ids[row] + 1) % vocab if vocab else ids[row] + 1
    return out


# --- the registry ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Negative:
    """One tampering: its `family` (the goal's four: `io`, `advice`, `token`, `circuit`), what it `tampers` (`plan`,
    `certified`, `frames`, `advice` or `commit`), the `check` that must refuse it and its refusal `code` (`served`: the code on a
    served link where it differs), the interfaces it `waits` on (each by what it is; none: it runs), and `edit`,
    the pure edit of the advice or the Commit record it tampers (None: it tampers the link's records)."""

    name: str
    family: str
    tampers: str
    what: str
    check: str
    code: str
    edit: Callable | None = None
    served: str | None = None
    waits: tuple[str, ...] = ()
    also: tuple[str, ...] = field(default=())

    def code_on(self, profile: str | None) -> str:
        return self.served if profile == "served" and self.served else self.code

    @property
    def pending(self) -> bool:
        return bool(self.waits)


NEGATIVES = {n.name: n for n in [
    Negative("late", "io", "plan", "the shim holds a window-0 session past every sync; the certifier faults the window "
             "and certifies none of it", "the certifier (it certifies no incomplete status)", ISSUER_REFUSED),
    Negative("extra", "io", "plan", "the shim writes one event more than the endpoint sent; certified",
             "N2's counts against the profile", f"{HEAD}.counts", served=SERVED),
    Negative("drop", "io", "plan", "the shim drops the first event of a response; certified",
             "N2's counts against the profile", f"{HEAD}.counts", served=SERVED,
             also=(f"{TIE}.content",)),
    Negative("head", "io", "certified", "window 0's egress head one count off, its certificate the original's",
             "N2's head check (the signed digest)", f"{HEAD}.signed", also=("audit.signature",)),
    Negative("token", "io", "plan", "the shim writes every token of window 0's responses with other bytes than the "
             "endpoint sent: certified, its counts honest's", "the I/O tie (a data cell's id against its port entry's)",
             f"{TIE}.content", waits=("the I/O tie", "the warden's cells")),
    Negative("frame", "io", "frames", "one byte of a certified cell changed, the cells' root kept",
             "the I/O tie (the warden's cells under their root)", f"{TIE}.cells_root",
             waits=("the I/O tie", "the warden's cells")),
    Negative("advice-length", "advice", "advice", "a request's output length one more than its certified frames, "
             "its next row keeping the sample", "the I/O tie (a kept output's data cell against the certified cells)",
             f"{TIE}.content", output_length,
             waits=("the advice module", "the I/O tie", "the warden's cells")),
    Negative("advice-missing", "advice", "advice", "a certified request missing from the advice, with its rows",
             "IoTieRequest (each certified request against the advice's)", REQUEST_TIE, missing_request,
             waits=("the advice module", "the I/O tie"), also=(f"{TIE}.content",)),
    Negative("advice-extra", "advice", "advice", "a request in the advice that the ingress didn't certify",
             "IoTieRequest (each certified request against the advice's)", REQUEST_TIE, extra_request,
             waits=("the advice module", "the I/O tie"), also=(f"{TIE}.content",)),
    Negative("prompt-split", "advice", "advice", "two requests' prompt lengths swapped, their total kept "
             "((338, 16) served, (16, 338) in the advice), every output and frame honest's",
             "IoTieRequest (each request's prompt length against the certified ingress cells)", REQUEST_TIE,
             prompt_split, waits=("the advice module", "the I/O tie")),
    Negative("advice-release", "advice", "advice", "a kept output released one bucket later than certified",
             "the I/O tie (the verifier's map of the advice's outputs against the warden's schedule)", f"{TIE}.map",
             release_moved, waits=("the advice module", "the I/O tie", "the warden's cells")),
    Negative("advice-start", "advice", "advice", "a request's first row starting one position later, claiming a "
             "prefix-cache hit on another request's KV, its end kept: N2 accepts it",
             "the step unit (committed values against the compiled Program)", "StepUnit.values",
             start_shifted, waits=("the advice module", "the compiler", "the prove path")),
    Negative("wrong-token", "token", "commit", "a committed sampled id changed after the run; the registered root kept",
             "the I/O tie and the step unit (the Commit's opening)", "Commit.opening", wrong_token,
             waits=("the Commit record", "the I/O tie", "the prove path"), also=(f"{TIE}.port", "StepUnit.opening")),
    Negative("advice-chunk", "circuit", "advice", "a prefill chunk split one token differently (an unchunked one in "
             "two): N2 accepts it",
             "the step unit (the compiled Program's digest against the committed trace's)", "StepUnit.program",
             chunk_split, waits=("the advice module", "the compiler", "the prove path")),
]}


def table(markdown: bool = True) -> str:
    """The registry, one row per negative: what it tampers, the check that refuses it, its code, and its status."""
    q = "`" if markdown else ""
    rows = [("negative", "family", "tampers", "check", "code", "status")]
    for n in NEGATIVES.values():
        code = f"{q}{n.code}{q}" + (f" (served: {q}{n.served}{q})" if n.served else "")
        rows.append((f"{q}{n.name}{q}", n.family, n.tampers, n.check, code,
                     f"pending: {', '.join(n.waits)}" if n.waits else "runs"))
    if not markdown:
        return "\n".join("\t".join(r) for r in rows)
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


# --- the ingress tie on a fake upstream (`io_request`) -----------------------------------------------------------------


def link_received(advice, request_frames, received: list[int]) -> list[int]:
    """`received` as a prover declares it beside `advice`: the request frames the advice's requests take under the
    link's framing (`request_frames[prompt_len − 1]` each) that `received` doesn't sum to go to its first window that
    receives any, as a prover that serves other requests than were received would declare them, so the advice's frames
    are the windows' and only its tie to the certified ingress (`request_tie`) can refuse it."""
    got = list(received)
    w = next((w for w, g in enumerate(got) if g > 0), 0)
    got[w] += sum(request_frames[q.prompt_len - 1] for q in advice.requests
                  if q.prompt_len <= len(request_frames)) - sum(got)
    return got


def io_request_module():
    """IoTieRequest's module (`io_request`), or None where it isn't installed."""
    try:
        return importlib.import_module(IO_REQUEST_MODULE)
    except ModuleNotFoundError:
        return None


def link_framing(frame_bytes: int, longest: int) -> tuple[int, ...]:
    """The link's public framing at `frame_bytes`-byte cells (`tie_service.AdvisedProfile.request_frames`): entry
    n − 1 the request cells the warden cuts an id request of an n-token prompt into (`tie_service.framing`, the framing
    the ingress tie holds the profile to), for prompts of 1 to `longest` tokens."""
    return TS.framing(frame_bytes, max(1, longest))


def read_requests(path: Path) -> dict[str, dict]:
    """Each request's record by the advice's request id: serve-openai's `requests.jsonl` rows that carry the prompt
    (`prompt_ids`), by the request's `user` (its http id without one), with what it brought (`ingress.brought`)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r.get("user") or r["http_id"]: r for r in rows if "prompt_ids" in r}


def id_requests(advice, records: dict[str, dict] | None, frame_bytes: int, model: str = "") -> list[bytes]:
    """The id requests (`id_form.id_request` at `frame_bytes`-byte cells and offset 0, as the network certifier's cells
    carry them) of the advice's requests in its order, the ingress the network certifier certified: each from its record
    (`read_requests`: its prompt ids, user, and the
    max_tokens, temperature, top_p and seed it brought, a seed it brought none of 0, id requests' default), or, for a
    request without one, a prompt of its length of id 0 under its own id. Raises NotApplicable for a record whose
    prompt isn't the advice's length or that no id request carries."""
    out = []
    for q in advice.requests:
        r = (records or {}).get(q.id)
        if r is None:
            named = len(q.id) <= ID_NAME_BYTES - 2 and all(" " <= c <= "~" and c not in '"\\' for c in q.id)
            fields = {"user": q.id if named else "", "prompt": [0] * q.prompt_len, "max_tokens": q.output_len}
        else:
            got = (r.get("ingress") or {}).get("brought") or {}
            if len(r["prompt_ids"]) != q.prompt_len:
                raise NotApplicable(f"request {q.id}'s record has a {len(r['prompt_ids'])}-token prompt, the advice "
                                    f"{q.prompt_len}")
            fields = {"user": r.get("user") or "", "prompt": r["prompt_ids"], "max_tokens": got.get("max_tokens", 0),
                      "temperature": got.get("temperature", 0.0), "top_p": got.get("top_p", 1.0),
                      "seed": got.get("seed", 0)}
        try:
            out.append(id_request(model=model, frame_bytes=frame_bytes, offset=0, **fields))
        except ValueError as err:
            raise NotApplicable(f"request {q.id} isn't an id request: {err}") from None
    return out


class Ingress(NamedTuple):
    """A link's ingress through the network certifier's own ingress on a fake upstream, as IoTieRequest reads it: its
    shape (`io_request.Shape`), each session bucket's row (witness, the network certifier's registered row), each
    connection's release (`io_request.Release`) and T, the buckets a window."""

    shape: Any
    rows: tuple[tuple[bytes, ...], ...]
    releases: tuple[Any, ...]
    T: int


def certified_ingress(Q, requests: list[bytes], received: list[int], *, T: int, frame_bytes: int) -> Ingress:
    """The network certifier's own ingress (`io_request.serve`) of `requests` (id requests at offset 0, in arrival
    order) at `frame_bytes`-byte cells, T ≥ 2 buckets a window, `len(received)` windows: each request in the window its
    first cell falls in when the windows take `received`'s request cells in order (the network certifier's cut), the
    window's k-th arriving whole at its bucket k mod (T − 1), on a lane of its own and r the most cells a request takes,
    so all its cells are in the bucket after. A request whose cells straddle two windows' `received` puts them all in
    the first, which the tie refuses."""
    if T < 2:
        raise NotApplicable(f"the warden's ingress releases a request the bucket after it arrives: T = {T} < 2")
    cells = [len(W.cells(r, frame_bytes, 1)) // frame_bytes for r in requests]
    ends, at, k, arrivals = list(itertools.accumulate(received)), 0, defaultdict(int), []
    for c in cells:
        w = next((w for w, e in enumerate(ends) if at < e), len(ends) - 1)
        arrivals.append(w * T + k[w] % (T - 1))
        k[w], at = k[w] + 1, at + c
    lanes = max([1, *(sum(a <= t <= a + 1 for a in arrivals) for t in range(len(received) * T))])
    params = W.WardenParams(cell_bytes=frame_bytes, slots=max([1, *cells]), lanes=lanes, window=T, delay=1)
    link = Q.serve(params, requests, arrivals, [len(r) for r in requests], buckets=len(received) * T)
    return Ingress(Q.Shape(frame_bytes, lanes, params.slots), tuple(tuple(r) for r in link.rows),
                   tuple(link.releases), T)


def request_tie(Q, cells: Ingress, link) -> tuple[bool, ...]:
    """IoTieRequest's verdict on each ingress window of `link` (`tie_service.LinkAdvice`): its reference on every bucket
    row of the window against the verifier's lanes of the bucket (`io_request.lanes`: the advice's prompt lengths in
    its order and the network certifier's release); every window refused where the release isn't the network certifier's
    schedule for the advice's requests or doesn't put `received`'s request cells in each window
    (`tie_service.IngressTie.tie`). The verifier takes the verdict from IoTieRequest proofs it verified; this
    is their reference in the clear."""
    refused, T = (False,) * len(link.received), cells.T
    lens = [q.prompt_len for q in link.advice.requests]
    if len(link.received) * T != len(cells.rows):
        return refused
    try:
        lanes = [Q.lanes(cells.shape, lens, cells.releases, t) for t in range(len(cells.rows))]
    except ValueError:
        return refused
    if TS.window_cells(cells.releases, T) != {w: c for w, c in enumerate(link.received) if c}:
        return refused
    return tuple(all(Q.reference(cells.shape, Q.Instance(cells.rows[t], lanes[t])) for t in range(w * T, (w + 1) * T))
                 for w in range(len(link.received)))


def advice_suite(advice, *, T: int, request_frames, received: list[int], frame_bytes: int,
                 records: dict[str, dict] | None = None, model: str = "") -> dict[str, dict]:
    """Every advice negative made from a run's `advice` (I1's): whether it applies to it, whether it's valid advice that
    round-trips through I1's codec, its bits, and the ingress tie's verdict on it by window (``request_tie``):
    IoTieRequest's reference (`request_tie`) on the warden's ingress at `frame_bytes`-byte cells, whose framing
    `request_frames` must be (`link_framing`), of the honest advice's id requests (`id_requests`, from `records` and
    `model`), each window's the honest `received`'s (`certified_ingress`), since a tampering of the advice leaves the
    certified ingress; ``IoTieRequest`` is among its codes when a window is refused. `ok`: the codes are exactly those
    it names, its code and `also`, that the ingress tie reads (none for a negative that only the I/O tie or the step
    unit refuses, whose requests are honest's)."""
    mod, Q = advice_module(), io_request_module()
    if mod is None:
        raise NotApplicable(f"no advice codec: {ADVICE_MODULE} (I1) isn't in this tree")
    if Q is None:
        raise NotApplicable(f"no ingress tie: {IO_REQUEST_MODULE} isn't in this tree")
    cells = certified_ingress(Q, id_requests(advice, records, frame_bytes, model),
                              link_received(advice, request_frames, received), T=T, frame_bytes=frame_bytes)

    def judged(b) -> dict:
        tie = list(request_tie(Q, cells, TS.LinkAdvice(b, tuple(link_received(b, request_frames, received)))))
        return {"codes": [] if all(tie) else [REQUEST_TIE], "request_tie": tie}

    honest = judged(advice)
    out = {"honest": {"bits": mod.total_bits(advice)} | honest | {"ok": honest["codes"] == []}}
    for name, n in NEGATIVES.items():
        if n.tampers != "advice":
            continue
        row: dict[str, Any] = {"code": n.code, "check": n.check, "waits": list(n.waits)}
        try:
            b = n.edit(advice)
        except NotApplicable as err:
            out[name] = row | {"ran": False, "why": str(err)}
            continue
        got = judged(b)
        moved = {j: [p.prompt_len, q.prompt_len] for j, (p, q) in enumerate(zip(advice.requests, b.requests))
                 if len(b.requests) == len(advice.requests) and p.prompt_len != q.prompt_len}
        out[name] = row | {"ran": True, "valid": mod.decode(mod.encode(b)) == b, "bits": mod.total_bits(b),
                           "steps": len(b.steps), "outputs": len(b.release)} | got | \
            ({"prompt_lens": moved} if moved else {}) | {"ok": got["codes"] == [c for c in (n.code, *n.also)
                                                                                if c == REQUEST_TIE]}
    return out


def advice_run(path: Path, *, T: int = 50, R: int = 32, frame_bytes: int = FRAME_BYTES, ids: Path | None = None,
               model: str = "") -> dict:
    """`advice_suite` on the advice at `path` (I1's encode bytes) under the link's framing at `frame_bytes`-byte
    frames (`link_framing`), all its requests' frames received in window 0: its shape and the rows (`negatives`, or why
    not), each tied by IoTieRequest's reference on the id requests of the run's records (`ids`, `read_requests`;
    `model` the name they were sent under). With the run's sampled `ids` (`read_ids`), `tie_suite` on it at T buckets
    of R cells (`tie`)."""
    mod = advice_module()
    if mod is None:
        raise NotApplicable(f"no advice codec: {ADVICE_MODULE} (I1) isn't in this tree")
    adv = mod.decode(path.read_bytes())
    windows = max(adv.release, default=0) // T + 1
    framing = link_framing(frame_bytes, max((q.prompt_len for q in adv.requests), default=1))
    received = [sum(framing[q.prompt_len - 1] for q in adv.requests)] + [0] * (windows - 1)
    try:
        rows = advice_suite(adv, T=T, request_frames=framing, received=received, frame_bytes=frame_bytes,
                            records=read_requests(ids) if ids else None, model=model)
    except NotApplicable as err:
        rows = {"why": str(err)}
    doc = {"format": FORMAT, "advice": str(path), "requests": len(adv.requests), "steps": len(adv.steps),
           "outputs": len(adv.release), "windows": windows, "T": T, "R": R, "frame_bytes": frame_bytes,
           "received": received, "negatives": rows}
    if ids is not None:
        try:
            doc["tie"] = tie_suite(adv, read_ids(ids), T=T, R=R)
        except NotApplicable as err:
            doc["tie"] = {"why": str(err)}
    return doc


# --- the I/O tie on a fake upstream (`io_tie`) -------------------------------------------------------------------------
# No network certifier writes cells into a run's directory yet, and no Commit's port is laid out as the tie reads it, so
# the tie's negatives run on what a fake upstream makes of a run's advice and sampled ids, as `io_tie`'s own tests make
# them: a network certifier of r slots a connection (`tie_schedule`), whose cells are the ones the map admits (each kept
# output a data cell on its request's stream, a PAD or the END on its stream elsewhere in its lane, B zero bytes in a
# slot no connection holds), and the Commit's port, a row of sampled ids a step, both committed under fake salts. The
# map is the verifier's, from the advice and the network certifier's admissions.

def io_tie_module():
    """The I/O tie's module (`io_tie`), or None where it isn't installed."""
    try:
        return importlib.import_module(IO_TIE_MODULE)
    except ModuleNotFoundError:
        return None


def _commit_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_ids(path: Path) -> dict[str, list[int]]:
    """Each request's sampled ids by the advice's request id: `verity-vllm advice`'s `requests.jsonl` (`user`,
    `output_ids`), or serve-openai's (`user`, `streamed_ids`; a request without a `user` by its `http_id`, as its advice
    names it)."""
    return {r.get("user") or r["http_id"]: list(r["output_ids"] if "output_ids" in r else r["streamed_ids"])
            for r in _commit_rows(path)}


def commit_aborted(row: dict) -> bool:
    """Whether a Commit record's request (a `read_ids` row) finished by its client's abort: an ``output_ids`` row's
    `finish_reason` ``abort``, or serve-openai's `finish` null (the endpoint writes an aborted request with none) or of
    reason ``abort``. A row with no finish field isn't an abort."""
    if "output_ids" in row:
        return row.get("finish_reason") == "abort"
    return "finish" in row and (row["finish"] is None or row["finish"].get("reason") == "abort")


def read_aborted(path: Path) -> set[str]:
    """The requests of `read_ids`' record that their clients aborted (`commit_aborted`), by the same ids."""
    return {r.get("user") or r["http_id"] for r in _commit_rows(path) if commit_aborted(r)}


def tie_outputs(IT, advice, ids: dict[str, list[int]]):
    """The tie's view of `advice`'s kept outputs, in I1 v0's order (`advice.outputs`): each one's `io_tie.Output`
    (step, batch row, request, index), its release bucket, and its id, the request's index-th sampled id (`ids`, by
    request id; 0 past them, as for a request the run never served)."""
    outputs, out_ids = [], []
    for s, j, r, k in _kept(advice.requests, _plain(advice)):
        xs = ids.get(advice.requests[r].id, ())
        outputs.append(IT.Output(s, j, r, k))
        out_ids.append(xs[k] if k < len(xs) else 0)
    return outputs, list(advice.release), out_ids


def tie_schedule(advice, slots: int | None = None) -> tuple[list[int], int]:
    """The fake upstream's network certifier for `advice`: each request's admission bucket and r, the slots a connection
    holds a bucket (`slots`, or by default the most outputs a request releases in its first release bucket), with no
    delay: request i is admitted at its first kept output's release bucket, a request that keeps none at the one before
    it (bucket 0 for the first). The admissions are the network certifier's wiring, which the verifier takes from the
    certified record, so every tampering of the advice is read against the honest advice's."""
    first: dict[int, int] = {}
    there: dict[int, int] = {}
    for (_, _, q, _), b in zip(_kept(advice.requests, _plain(advice)), advice.release):
        first.setdefault(q, b)
        there[q] = there.get(q, 0) + (b == first[q])
    admitted, last = [], 0
    for i in range(len(advice.requests)):
        last = first.get(i, last)
        admitted.append(last)
    return admitted, slots or max([1, *there.values()])


def tie_served(advice, schedule):
    """`advice` as the fake upstream's network certifier serves it (`tie_schedule`'s): each kept output k of request i
    released at admitted[i] + ⌊k / r⌋, the bucket the network certifier sends its cell in. An advice's release is the
    network certifier's schedule, which the verifier's map holds it to, so an advice no network certifier served is read
    as this one's."""
    admitted, r = schedule
    return replace(advice, release=tuple(admitted[q] + k // r for _, _, q, k in _kept(advice.requests, _plain(advice))))


def tie_negative(edited, served, schedule):
    """An advice negative of the `served` advice as the fake upstream's network certifier reads it: an output the served
    advice doesn't keep (by its request's id and index) released where the network certifier would send it, admitted[i]
    + ⌊k / r⌋ for its request i (a request past the network certifier's connections where the negative releases it);
      every other output where
    the negative releases it, so a moved release stays moved."""
    admitted, r = schedule
    honest = _buckets(served)
    kept = _kept(edited.requests, _plain(edited))
    return replace(edited, release=tuple(
        b if (edited.requests[q].id, k) in honest or q >= len(admitted) else admitted[q] + k // r
        for (_, _, q, k), b in zip(kept, edited.release)))


def _ends(advice, schedule) -> list[int]:
    """Each request's `END` bucket under the network certifier's schedule: its admission plus ⌊n / r⌋, n its kept
    outputs."""
    admitted, r = schedule
    n = [0] * len(advice.requests)
    for _, _, q, _ in _kept(advice.requests, _plain(advice)):
        n[q] += 1
    return [a + k // r for a, k in zip(admitted, n)]


def tie_shape(IT, advices, schedule, *, T: int, R: int | None = None, B: int = 256):
    """One shape for the tie of every advice in `advices` (the honest one and its tamperings), so that each reaches the
    proof block's parts: R (unless given) r slots for each of the most connections live in a bucket under the network
    certifier's `schedule` (`tie_schedule`; a connection live from its admission to its `END`'s bucket), P its widest
    step's batch, S the most steps a bucket's outputs were sampled in, and DS the depth of a port over the longest
    session."""
    admitted, r = schedule
    live, most_s, P, steps = 1, 1, 1, 2
    for a in advices:
        buckets: dict[int, set[int]] = {}
        for (s, *_), b in zip(_kept(a.requests, _plain(a)), a.release):
            buckets.setdefault(b, set()).add(s)
        most_s = max([most_s, *map(len, buckets.values())])
        if len(a.requests) == len(admitted):
            ends = _ends(a, schedule)
            live = max([live, *(sum(x <= b <= y for x, y in zip(admitted, ends))
                                for b in range(max(ends, default=0) + 1))])
        P, steps = max([P, *(len(st.rows) for st in a.steps)]), max(steps, len(a.steps))
    return IT.Shape(B, T, R or r * live, P=P, S=most_s, DS=(steps - 1).bit_length())


def tie_port(IT, shape, advice, outputs, ids, seed: bytes = b"negatives"):
    """The Commit's port of the run of `advice`: a row a step of its session, its batch rows' sampled ids (a kept
    output's id, 0 for a row that keeps none), empty rows to 2^DS steps, under fake salts (a prover samples fresh ones):
    its rows, salts, tree levels and domain id."""
    got = {(o.step, o.row): x for o, x in zip(outputs, ids)}
    rows = [IT.port_row([got.get((s, j), 0) for j in range(len(advice.steps[s].rows))] if s < len(advice.steps) else [],
                        shape.P) for s in range(1 << shape.DS)]
    pd = IT.domain(hashlib.sha512(b"port" + seed).digest(), len(rows))
    salts = [IT.fake_salt(seed + b"p", s) for s in range(len(rows))]
    return rows, salts, IT.rows_levels(pd.domain_id, rows, salts), pd.domain_id


def tie_map(IT, shape, window: int, outputs, release, schedule):
    """The map and each bucket's steps the verifier computes for egress window `window` from the advice's `outputs` and
    `release` and the network certifier's admissions and r (`schedule`, no delay; `io_tie.window_map`, a ValueError when
    it can't). Advice request i is on stream i + 1, v0's order."""
    admitted, r = schedule
    return IT.window_map(shape, outputs, release, window, admitted, 0, r)


def tie_cells(IT, shape, cmap, steps, port) -> list[bytes]:
    """The network certifier's cells of an egress window on the fake upstream: each slot the one cell its record admits
    (`io_tie.expected_cell`), a data cell the Commit's id at its entry (`port`, `tie_port`'s, opened at its bucket's
    steps), a `PAD` or the `END` on its stream, B zero bytes in a slot no connection holds."""
    rows, salts, levels, _ = port
    cells = []
    for t, st in enumerate(steps):
        opened = IT.open_port(levels, rows, salts, st)[0]
        cells += [IT.expected_cell(m, shape.B, IT.port_id(shape, opened, m.e) if 0 <= m.e < shape.E else None)
                  for m in cmap[t * shape.R:(t + 1) * shape.R]]
    return cells


def tie_instance(IT, shape, window: int, cells, outputs, release, schedule, port, seed: bytes = b"negatives"):
    """Window `window`'s tie: `cells` committed as the network certifier commits them (a row a bucket, under fake
    salts), the map and each bucket's steps from the advice (`tie_map`), and each bucket's port rows opened at its steps
    from the Commit's (`port`, `tie_port`'s)."""
    return _instance(IT, shape, window, cells, *tie_map(IT, shape, window, outputs, release, schedule), port, seed)


def _instance(IT, shape, window: int, cells, cmap, steps, port, seed: bytes):
    rows, salts, levels, port_domain = port
    tag = seed + window.to_bytes(4, "big")
    cd = IT.domain(hashlib.sha512(b"cells" + tag).digest(), shape.T)
    c_salts = [IT.fake_salt(tag + b"c", t) for t in range(shape.T)]
    opened = [IT.open_port(levels, rows, salts, st) for st in steps]
    return IT.Instance(shape, tuple(cells), tuple(c_salts),
                       IT.rows_root(cd.domain_id, IT.bucket_rows(shape, cells), c_salts), cd.domain_id, tuple(cmap),
                       tuple(steps), tuple(o[0] for o in opened), tuple(o[1] for o in opened),
                       tuple(o[2] for o in opened), levels[-1][0], port_domain)


def _first_data(IT, cells) -> int:
    for k, c in enumerate(cells):
        p = IT.parse_cell(c)
        if p is not None and p.data:
            return k
    raise NotApplicable("the window has no data cell")


def cell_byte(IT, inst):
    """`frame` on the tie: the last digit of the first data cell's id changed (`frame_byte`), the cells' root kept."""
    k = _first_data(IT, inst.cells)
    return replace(inst, cells=tuple(frame_byte(list(inst.cells), k, IT.ID_AT + IT.ID_DIGITS - 1)))


def cell_token(IT, inst) -> list[bytes]:
    """`token` on the tie: the cells with the first data cell carrying another id than the sampled one, which the
    network certifier certifies as it saw it (the cells to recommit)."""
    k = _first_data(IT, inst.cells)
    p, cells = IT.parse_cell(inst.cells[k]), list(inst.cells)
    cells[k] = IT.data_cell((p.id + 1) % (1 << 32), p.stream, inst.shape.B)
    return cells


def cell_dropped(IT, inst) -> list[bytes]:
    """`drop` on the tie: the cells with a `PAD` on its stream in place of the first data cell, as the network certifier
    releases that slot when the event never came (to recommit)."""
    k, cells = _first_data(IT, inst.cells), list(inst.cells)
    cells[k] = IT.pad_cell(inst.shape.B, IT.parse_cell(cells[k]).stream)
    return cells


def port_token(IT, inst):
    """`wrong-token` on the tie: the Commit's sampled id at the first data cell's port entry changed after the run, the
    Commit's root kept."""
    k = _first_data(IT, inst.cells)
    t, (s, r) = inst.shape.bucket(k), divmod(inst.map[k].e, inst.shape.P)
    row = bytearray(inst.port[t][s])
    row[4 * r:4 * r + 4] = ((int.from_bytes(row[4 * r:4 * r + 4], "big") + 1) % (1 << 32)).to_bytes(4, "big")
    bucket = inst.port[t][:s] + (bytes(row),) + inst.port[t][s + 1:]
    return replace(inst, port=inst.port[:t] + (bucket,) + inst.port[t + 1:])


def tie_suite(advice, ids: dict[str, list[int]], *, T: int, R: int | None = None, B: int = 256,
              slots: int | None = None) -> dict[str, dict]:
    """Every negative the I/O tie reads (`TIE_PARTS`), on a fake upstream's run of `advice` whose requests sampled `ids`
    (by request id), served through a network certifier of r = `slots` slots a connection (`tie_schedule`,
    `tie_served`): each egress window's cells and the Commit's port, honest and as each negative makes them (a cell or
    port negative in the first window with a data cell; an advice negative the map the verifier computes from its advice
    and the network certifier's admissions, over the honest cells and port), with the parts of `io_tie.reference_parts`
    that fail, by window (``map``, at window −1, when the verifier can't compute the map, with `why`: an advice whose
    release isn't the network certifier's schedule, or whose requests aren't its connections). One shape serves them all
    (`tie_shape`). `ok`: the part it names fails; for one it must accept, none does while its outputs are honest's, and
    content does once the edit moved them (`moved`). NotApplicable when the network certifier can't serve the advice in
    R slots."""
    IT = io_tie_module()
    if IT is None:
        raise NotApplicable(f"no I/O tie: {IO_TIE_MODULE} isn't in this tree")
    schedule = tie_schedule(advice, slots)
    advice = tie_served(advice, schedule)
    edited: dict[str, Any] = {}
    for name in TIE_PARTS:
        if NEGATIVES[name].tampers == "advice":
            try:
                edited[name] = tie_negative(NEGATIVES[name].edit(advice), advice, schedule)
            except NotApplicable as err:
                edited[name] = err
    shape = tie_shape(IT, [advice, *(b for b in edited.values() if not isinstance(b, NotApplicable))], schedule,
                      T=T, R=R, B=B)
    outputs, release, out_ids = tie_outputs(IT, advice, ids)
    port = tie_port(IT, shape, advice, outputs, out_ids)
    windows = max([*release, *_ends(advice, schedule)], default=0) // T + 1
    try:
        maps = {w: tie_map(IT, shape, w, outputs, release, schedule) for w in range(windows)}
    except ValueError as err:
        raise NotApplicable(f"the advice isn't served through a warden of r = {schedule[1]} slots a connection in "
                            f"R = {shape.R}: {err}") from err
    cells = {w: tie_cells(IT, shape, *maps[w], port) for w in range(windows)}
    honest = {w: _instance(IT, shape, w, cells[w], *maps[w], port, b"negatives") for w in range(windows)}

    def failing(insts: dict) -> dict[int, list[str]]:
        out = {}
        for w, inst in insts.items():
            bad = [p for p, ok in IT.reference_parts(inst).items() if not ok]
            if bad:
                out[w] = bad
        return out

    first = next((w for w in range(windows) if any(IT.parse_cell(c).data for c in cells[w])), 0)
    h = honest[first]
    cell_made = {"frame": lambda: cell_byte(IT, h),
                 "token": lambda: tie_instance(IT, shape, first, cell_token(IT, h), outputs, release, schedule, port),
                 "drop": lambda: tie_instance(IT, shape, first, cell_dropped(IT, h), outputs, release, schedule, port),
                 "wrong-token": lambda: port_token(IT, h)}
    got = failing(honest)
    out: dict[str, dict] = {"honest": {"shape": {k: getattr(shape, k) for k in ("B", "T", "R", "P", "S", "DS", "N", "M")},
                                       "windows": windows, "failing": got, "ok": not got}}
    for name, part in TIE_PARTS.items():
        row: dict[str, Any] = {}
        try:
            if name in cell_made:
                bad = failing({first: cell_made[name]()})
            else:
                if isinstance(b := edited[name], NotApplicable):
                    raise b
                o, r, _ = tie_outputs(IT, b, ids)
                if part is None and (o, r) != (outputs, release):
                    part, row["moved"] = "content", True
                span = max(windows, max(r, default=0) // T + 1)
                empty = [IT.pad_cell(shape.B)] * shape.N
                try:
                    bad = failing({w: tie_instance(IT, shape, w, cells.get(w, empty), o, r, schedule, port)
                                   for w in range(span)})
                except NotApplicable:
                    raise
                except ValueError as err:
                    bad, row["why"] = {-1: ["map"]}, str(err)
        except NotApplicable as err:
            out[name] = {"code": f"{TIE}.{part}" if part else None, "ran": False, "why": str(err)}
            continue
        codes = sorted({f"{TIE}.{p}" for parts in bad.values() for p in parts})
        code = f"{TIE}.{part}" if part else None
        out[name] = {"code": code, **row, "ran": True, "failing": bad, "codes": codes,
                     "ok": (code in codes) if part else not codes}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="negatives.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table", help="the negatives, their checks and codes")
    t.add_argument("--markdown", action="store_true")
    w = sub.add_parser("warden", help="frame on a warden-serve run's own cells, by the verifier's registered read")
    w.add_argument("--run", required=True, type=Path)
    a = sub.add_parser("advice", help="every advice negative of a run's advice, and the ingress tie's codes on it")
    a.add_argument("--advice", required=True, type=Path, help="the run's advice (I1's encode bytes)")
    a.add_argument("--T", type=int, default=50, help="buckets a window (the advice's release buckets')")
    a.add_argument("--R", type=int, default=32, help="egress frames a bucket")
    a.add_argument("--frame-bytes", type=int, default=FRAME_BYTES,
                   help="the link's frame size: each request's frames are its id request's at this size")
    a.add_argument("--model", default="", help="the model the run's requests named, in their id requests")
    a.add_argument("--ids", type=Path, help="the run's records (serve-openai's requests.jsonl): each request's prompt "
                   "and what it brought, for IoTieRequest's ingress, and its sampled ids, for the I/O tie")
    args = ap.parse_args(argv)
    if args.command == "advice":
        try:
            doc = advice_run(args.advice, T=args.T, R=args.R, frame_bytes=args.frame_bytes, ids=args.ids,
                             model=args.model)
        except NotApplicable as err:
            raise SystemExit(str(err))
        print(json.dumps(doc, indent=1))
        rows = [r for part in ("negatives", "tie") for r in (doc.get(part) or {}).values() if isinstance(r, dict)]
        return 0 if all(r.get("ok") is not False for r in rows) else 1
    if args.command == "table":
        print(table(args.markdown))
        return 0
    try:
        out = warden_frame(args.run)
    except NotApplicable as err:
        raise SystemExit(str(err))
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
