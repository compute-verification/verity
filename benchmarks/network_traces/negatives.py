"""negatives: what network accounting of arbitrary vLLM usage must refuse, beside honest traffic, each a pure function on a run's artifacts that
writes a tampered copy, and each naming the check that refuses it by its refusal code.

    python benchmarks/network_traces/negatives.py table [--markdown]
    python benchmarks/network_traces/negatives.py make NAME --run DIR --out DIR [--advice FILE] [--commit FILE]
    python benchmarks/network_traces/negatives.py n2 --run DIR
    python benchmarks/network_traces/negatives.py judge NAME --run DIR
    python benchmarks/network_traces/negatives.py advice --advice FILE [--ids REQUESTS.jsonl] [--frame-bytes B]
        [--model M] [--T 50] [--R 32] [--delta 0] [--rho 1000] [--shift 0]
    python benchmarks/network_traces/negatives.py warden --run DIR
    python benchmarks/network_traces/negatives.py tie-run --run TARBALL|DIR [--ids REQUESTS.jsonl]

**A run** is a certifier-serve run's directory (`certifier_serve.py rehearse`'s, one per strategy): its `plan.json`,
`calibration.json`, `deployment.json` and `certified.jsonl`, and, as their interfaces land, the advice (I1:
`verity.core.protocols.certifier.advice`, its `encode` bytes) and the Commit record (I3, serve-openai's).

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
it that fails. On the proof system's stream every refusal shows only ``reject`` (the violation's kind stays in the
verifier); the code is what the verifier of record establishes, which `n2` reads here in the clear for N2's checks:
- ``NetworkHeadCheck_v2.{opening,fields,counts,signed}``: N2's head check of each end (`head_check.reference_parts`):
  the head opens under the registered root, its fields (link, end, window, grid, status complete), its counts the
  profile's row, and its record digest under the certifier's salt the certificate's signed digest;
- ``NetworkServedCounts_v1``: N2's served counts on a served link (`served.reference`);
- ``audit.VIOLATION``: N2's audit of both records (`audit.Audit`, the Python reference of the Lean check of record);
- pending checks, by the names their drafts give them: ``NetworkAdvisedCounts_v1`` (N2-ADVICE's advised counts on an
  advised link, `LinkWindows.advised_check`: the egress head's counts the release of the advice's frames, the ingress
  head's the request frames the advice needs; ``.view`` when the advice is malformed, doesn't cover the window or its
  requests aren't the certified ingress's (the tie, IoTieRequest's verdict, `advised.IoTieVerdict`, from the
  verifier's own tie source), which refuses every window), ``IoTieRequest`` (`io_request`'s ingress tie, a unit a bucket
  row of the warden's ingress link-window at selection All, `io_request.reference`: each lane's cells, read from its
  carry-in, are its connection's pads and canonical id-request cells, leaving its carry-out, on the verifier's lanes
  from the advice's prompt lengths and the warden's release; its verdict on every window, under the advice's digest
  and with the release giving `received`, is the tie N2's advised view needs),
  ``IoTieWindow.{cells_root,port,content}`` (the I/O tie, a unit at selection All over one egress link-window,
  `io_tie.reference_parts`: the warden's cells open under their root, the Commit's port rows of sampled ids open at
  each bucket's steps under the Commit's root, and every cell is the verifier's map's, computed from advice v0 and
  the warden's admissions: a `PAD` or the `END` on its request's stream, B zero bytes in a slot no connection holds,
  or a data cell on the request's stream whose id is the port entry the map names; ``.map`` when the verifier can't
  compute the map, an advice whose release isn't the warden's schedule or whose requests aren't its connections,
  which refuses every window); and, until their owners name them,
  ``StepUnit`` (the prove path's: the
  compiled step Program's digest against the committed trace's, and the committed values passing it) and ``Commit``
  (the Commit's opening under its registered root).

**The tie on a fake upstream** (`tie_suite`): the warden's cells of a run of the advice served through a warden of r
slots a connection (`tie_schedule`, `tie_served`: each request admitted at its first release, its outputs released on
the warden's schedule), each kept output a data cell carrying its sampled id (`--ids`, `verity-vllm advice`'s
`requests.jsonl`), and the Commit's port, a row of sampled ids a step, both committed; each negative it reads is the
tampered cells, port or advice (whose map the verifier computes), and `io_tie.reference_parts` names the part that
refuses it (`TIE_PARTS`).

**The tie on a certifier run** (`tie-run`, `tie_runs`): the I/O tie on each run of a gate run's tarball (or of a
directory of runs), each egress window's certified frames held to its certified head, as one JSON document
(`TIE_RUN`): `runs`, by strategy, each with its `verdict` (``accepted``, ``refused``, or ``not-applicable`` when the run
lacks what the tie opens, which `missing` names by `TIE_MISSING`'s keys), its `codes` (``IoTieWindow.PART``), the
failing parts by window, its `finish` (on a compact plan its records held to the sampled ids,
`active.compact_finish_violations`, which refuses it when they aren't), and `ok` (`tie_ok`: honest accepted, a
negative the tie reads refused by its part, `TIE_PARTS` or, for a shim tamper only a run has, `TIE_RUN_PARTS`); and
`stand_ins`, what the tie takes for what the run doesn't keep (`TIE_STAND_INS`). It exits 0 when every run has a
verdict and none is `ok` false, else 1.

**The ingress tie on a fake upstream** (`advice_suite` with a frame size): the warden's own ingress of the advice's
requests, id requests at that cell size (each from its record, `--ids` as serve-openai writes it: prompt ids, user and
what it brought), each window's arriving whole so their request cells are the honest `received`'s
(`certified_ingress`), under the warden's framing (`link_framing`). Each advice negative is IoTieRequest's reference
on every bucket row on the verifier's lanes from its advice and the warden's release (`request_tie`), whose verdict is
the tie source's for `N2_LINK` (`n2_advice`). Without a frame size, the tie is the stand-in on the prompt lengths.

`Negative.waits` names the interfaces a negative waits on; one with none runs. `table` renders the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import shutil
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import active_replay as ar  # noqa: E402
import certifier_serve as cs  # noqa: E402
from verity.core.protocols.certifier import audit as host_audit, head_check as HC  # noqa: E402
from verity.core.protocols.certifier.active import (CANONICAL_STREAM, DATA, HEADER, ID_NAME_BYTES,  # noqa: E402
                                                     compact_finish_violations, compact_record_data, decode_frame,
                                                     encode_frame, id_event_data, id_request)
from verity.core.protocols.certifier.certificate import (CertifiedRecord, Direction, frames_root,  # noqa: E402
                                                          real_frames)
from verity.core.protocols.certifier import tie_service as TS, warden as W  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402

FORMAT = "verity-network-negatives/v1"
ADVICE_MODULE = "verity.core.protocols.certifier.advice"
ADVISED_MODULE = "verity.core.protocols.certifier.advised"
IO_TIE_MODULE = "verity.core.protocols.certifier.io_tie"
IO_REQUEST_MODULE = "verity.core.protocols.certifier.io_request"
WARDEN_ROWS_MODULE = "verity.core.protocols.certifier.warden_rows"
SAMPLED = "runner.sampler/sampled_token_ids"
HEAD = "NetworkHeadCheck_v2"
SERVED = "NetworkServedCounts_v1"
ADVISED = "NetworkAdvisedCounts_v1"
TIE = "IoTieWindow"
REQUEST_TIE = "IoTieRequest"
#: the verifier's registered read of a warden cell at its position under the warden's signed root (`warden_rows.opens`,
#: the Lean verifier's check of a unit's registered port)
WARDEN_READ = "Flock.Registered.checkPort"
#: the part of the I/O tie (`TIE`) that refuses each negative it reads; None: one it must accept while its outputs are
#: honest's (each one's step, batch row, request, index and bucket), which content refuses once the edit moves them
TIE_PARTS = {"drop": "content", "token": "content", "frame": "cells_root", "advice-length": "content",
             "advice-missing": "content", "advice-extra": "map", "advice-release": "map", "advice-start": None,
             "wrong-token": "port", "advice-chunk": None, "prompt-split": None}
#: the part of the I/O tie that refuses each shim tamper it reads only on a certifier run (`tie-run`), not on a fake
#: upstream (`tie_suite`): extra's one event past a response's end is a data cell no committed output is
TIE_RUN_PARTS = {"extra": "content"}
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
    prover that commits a head other than the certified one hands over (n2_zk's tampered head)."""
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
    the root (`certificate.verify_frame`); under the warden, the I/O tie recomputes the cells' root and refuses it
    (`TIE`.cells_root, `tie_suite`)."""
    out = list(frames)
    f = bytearray(out[k])
    f[at] ^= 1
    out[k] = bytes(f)
    return out


def frames_edit(src: Path, dst: Path) -> Path:
    raise NotApplicable("the run keeps its frames' root only (`certified.jsonl`'s heads); the frames negative waits "
                        "for a record that keeps the cells (the warden's: `warden_frame` on a warden-serve run), and "
                        "runs on the tie's fake upstream (--ids)")


def cell_reads(WR, cells, k: int, row: bytes) -> bool:
    """Whether `row`, opened at position `k` of the warden's committed `cells` (`warden_rows.Cells`, a row a bucket)
    under that position's salt and path, climbs to their root: the verifier's registered read (`warden_rows.opens`)."""
    return WR.opens(cells.domain, cells.root, k, WR.bc(row, cells.salts[k]), cells.path(k))


def warden_frame(run: Path, at: int | None = None) -> dict:
    """`frame` on a warden-serve run's own cells (`warden_rows`: `cells.jsonl`'s registrations and the
    prover's `cells.salts`, the rows' salts and the wiring's): the first egress link-window with a data cell, its
    registration (v2) checked under the warden's key and the schedule, and its record's bucket rows and wiring
    committed under their salts to the registered head; then byte `at` of that cell (its first payload byte by
    default) changed in its bucket's row, and the row opened at its position. The honest row's read climbs to the
    signed root and the changed one's doesn't (`WARDEN_READ`)."""
    try:
        WR, WS = importlib.import_module(WARDEN_ROWS_MODULE), importlib.import_module("warden_serve")
    except ModuleNotFoundError as err:
        raise NotApplicable(f"no warden cells: {err.name} isn't in this tree") from None
    link, direction = WS.CERTIFIED[WS.EGRESS]
    if missing := [f for f in ("cells.salts", f"{WS.EGRESS}.records") if not (run / f).is_file()]:
        raise NotApplicable(f"{run}: no {' or '.join(missing)}: the run kept its registrations only (public), so "
                            "no unit can open its cells")
    dep = json.loads((run / "deployment.json").read_text())
    params = WS.WardenParams(**dep["params"])
    key, session = cs.read_public(dep["key"]), bytes.fromhex(dep["session"])
    records = {r.window: r for r in WS.read_records(run, params)[WS.EGRESS]}
    regs = {r["window"]: (r, salts, wiring_salt) for d, salts, wiring_salt in WS.read_cells(run)
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
# On I1 v0 (`verity.core.protocols.certifier.advice`): Advice(requests, steps, release),
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
    16) served, (16, 338) advised): by default the first request whose prompt is longer than the next one's, else the
    first whose prompt is shorter, each prefilled to the other's length and decoded on from there (`_relaid`), the
    rest as served, so every output is honest's, released in its bucket. Where the two take different frames under
    the link's framing, the earlier one's end moves (down, for a longer first: phase 6's acausal release); where they
    take the same, `advised.ends` doesn't move. Either way the requests' frames are honest's, so only the tie of each
    request's prompt length to the certified ingress (IoTieRequest) refuses it."""
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


def advice_edit(edit: Callable) -> Callable[[Path, Path, Path], Path]:
    """`edit` on the advice at `advice` (I1's `encode` bytes), written to `dst/advice.bin` beside a copy of the run."""
    def make(src: Path, dst: Path, advice: Path) -> Path:
        mod = advice_module()
        if mod is None:
            raise NotApplicable(f"no advice codec: {ADVICE_MODULE} (I1) isn't in this tree")
        copy_run(src, dst)
        (dst / "advice.bin").write_bytes(mod.encode(edit(mod.decode(advice.read_bytes()))))
        return dst
    make.__name__, make.edit = f"advice_{edit.__name__}", edit
    return make


# --- the Commit record (I3) --------------------------------------------------------------------------------------------
# On a stand-in of I3: per step, each batch row's request, the committed `c00.input/ids` and
# `runner.sampler/sampled_token_ids`, under the Commit's registered root. serve-openai's record (`serve_openai.py`'s
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
    served link where it differs), the interfaces it `waits` on (each by what it is; none: it runs), and `make`,
    the pure function that writes the tampered copy."""

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
             also=(ADVISED, f"{TIE}.content")),
    Negative("head", "io", "certified", "window 0's egress head one count off, its certificate the original's",
             "N2's head check (the signed digest)", f"{HEAD}.signed", tampered_head, also=("audit.signature",)),
    Negative("token", "io", "plan", "the shim writes every token of window 0's responses with other bytes than the "
             "endpoint sent: certified, its counts honest's", "the I/O tie (a data cell's id against its port entry's)",
             f"{TIE}.content", shim("token"), waits=("the I/O tie", "the warden's cells")),
    Negative("frame", "io", "frames", "one byte of a certified cell changed, the cells' root kept",
             "the I/O tie (the warden's cells under their root)", f"{TIE}.cells_root", frames_edit,
             waits=("the I/O tie", "the warden's cells")),
    Negative("advice-length", "advice", "advice", "a request's output length one more than its certified frames, "
             "its next row keeping the sample", "N2's advised counts", ADVISED, advice_edit(output_length),
             waits=("the advice module", "N2's advised counts"), also=(f"{TIE}.content",)),
    Negative("advice-missing", "advice", "advice", "a certified request missing from the advice, with its rows",
             "N2's advised view (the tie, IoTieRequest's verdict)", f"{ADVISED}.view", advice_edit(missing_request),
             waits=("the advice module", "N2's advised counts"), also=(REQUEST_TIE, f"{TIE}.content")),
    Negative("advice-extra", "advice", "advice", "a request in the advice that the ingress didn't certify",
             "N2's advised view (the tie, IoTieRequest's verdict)", f"{ADVISED}.view", advice_edit(extra_request),
             waits=("the advice module", "N2's advised counts"), also=(REQUEST_TIE, f"{TIE}.content")),
    Negative("prompt-split", "advice", "advice", "two requests' prompt lengths swapped, their total kept "
             "((338, 16) served, (16, 338) advised), every output and frame honest's",
             "IoTieRequest (each request's prompt length against the certified ingress cells), whose verdict N2's "
             "advised view needs", REQUEST_TIE, advice_edit(prompt_split),
             waits=("the advice module", "N2's advised counts", "the I/O tie"), also=(f"{ADVISED}.view",)),
    Negative("advice-release", "advice", "advice", "a kept output released one bucket later than certified",
             "N2's advised counts", ADVISED, advice_edit(release_moved),
             waits=("the advice module", "N2's advised counts"), also=(f"{TIE}.content",)),
    Negative("advice-start", "advice", "advice", "a request's first row starting one position later, claiming a "
             "prefix-cache hit on another request's KV, its end kept: N2 accepts it",
             "the step unit (committed values against the compiled Program)", "StepUnit.values",
             advice_edit(start_shifted), waits=("the advice module", "the compiler", "the prove path")),
    Negative("wrong-token", "token", "commit", "a committed sampled id changed after the run; the registered root kept",
             "the I/O tie and the step unit (the Commit's opening)", "Commit.opening", commit_edit(wrong_token),
             waits=("the Commit record", "the I/O tie", "the prove path"), also=(f"{TIE}.port", "StepUnit.opening")),
    Negative("advice-chunk", "circuit", "advice", "a prefill chunk split one token differently (an unchunked one in "
             "two): N2 accepts it",
             "the step unit (the compiled Program's digest against the committed trace's)", "StepUnit.program",
             advice_edit(chunk_split), waits=("the advice module", "the compiler", "the prove path")),
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
                     f"pending: {', '.join(n.waits)}" if n.waits else "runs"))
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
    receives any, as a prover that serves other requests than were received would declare them, so the advice's frames
    are the windows' (`advised.ends`) and only its tie to the certified ingress (`n2_advice`'s) or the counts can
    refuse it."""
    got = list(received)
    w = next((w for w, g in enumerate(got) if g > 0), 0)
    got[w] += sum(request_frames[q.prompt_len - 1] for q in advice.requests
                  if q.prompt_len <= len(request_frames)) - sum(got)
    return got


#: the advised link `n2_advice`'s stand-in tie names
N2_LINK = "responses"


def n2_registrations(windows: int) -> tuple[str, ...]:
    """The stand-in tie source's record of `N2_LINK`'s ingress registrations: a stand-in digest a window."""
    return tuple(hashlib.sha512(f"negatives ingress registration {N2_LINK}/{w}".encode()).hexdigest()
                 for w in range(windows))


def io_request_module():
    """IoTieRequest's module (`io_request`), or None where it isn't installed."""
    try:
        return importlib.import_module(IO_REQUEST_MODULE)
    except ModuleNotFoundError:
        return None


def link_framing(frame_bytes: int, longest: int) -> tuple[int, ...]:
    """The link's public framing at `frame_bytes`-byte cells (`advised.AdvisedProfile.request_frames`): entry n − 1
    the request cells the warden cuts an id request of an n-token prompt into (`tie_service.framing`, the framing the
    ingress tie holds the profile to), for prompts of 1 to `longest` tokens."""
    return TS.framing(frame_bytes, max(1, longest))


def read_requests(path: Path) -> dict[str, dict]:
    """Each request's record by the advice's request id: serve-openai's `requests.jsonl` rows that carry the prompt
    (`prompt_ids`), by the request's `user` (its http id without one), with what it brought (`ingress.brought`)."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r.get("user") or r["http_id"]: r for r in rows if "prompt_ids" in r}


def id_requests(advice, records: dict[str, dict] | None, frame_bytes: int, model: str = "") -> list[bytes]:
    """The id requests (`active.id_request` at `frame_bytes`-byte cells and offset 0, as the warden's cells carry
    them) of the advice's requests in its order, the ingress the warden certified: each from its record
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
    """A link's ingress through the warden's own ingress on a fake upstream, as IoTieRequest reads it: its shape
    (`io_request.Shape`), each session bucket's row (witness, the warden's registered row), each connection's release
    (`io_request.Release`) and T, the buckets a window."""

    shape: Any
    rows: tuple[tuple[bytes, ...], ...]
    releases: tuple[Any, ...]
    T: int


def certified_ingress(Q, requests: list[bytes], received: list[int], *, T: int, frame_bytes: int) -> Ingress:
    """The warden's own ingress (`io_request.serve`) of `requests` (id requests at offset 0, in arrival order) at
    `frame_bytes`-byte cells, T ≥ 2 buckets a window, `len(received)` windows: each request in the window its first
    cell falls in when the windows take `received`'s request cells in order (the warden's cut), the window's k-th
    arriving whole at its bucket k mod (T − 1), on a lane of its own and r the most cells a request takes, so all its
    cells are in the bucket after. A request whose cells straddle two windows' `received` puts them all in the first,
    which the tie refuses."""
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
    """IoTieRequest's verdict on each ingress window of `link` (`advised.LinkAdvice`): its reference on every bucket
    row of the window against the verifier's lanes of the bucket (`io_request.lanes`: the advice's prompt lengths in
    its order and the warden's release); every window refused where the release isn't the warden's schedule for the
    advice's requests or doesn't put `received`'s request cells in each window (`tie_service.IngressTie.tie`). The
    verifier of record takes the verdict from IoTieRequest proofs it verified; this is their reference in the
    clear."""
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


def n2_advice(advice, request_frames, received: list[int], certified=None, cells: Ingress | None = None):
    """I1 v0 advice as `advised` reads it, (link, tie): `advised.LinkAdvice` (the advice and each window's
    `received`, `link_received`'s) and the tie, IoTieRequest's verdict on `N2_LINK` (`advised.IoTieVerdict`), which
    the verifier of record takes only from its own tie source, never beside the advice. Until IoTieRequest runs here
    this is that source's stand-in: with the certified ingress `cells`, IoTieRequest's reference on them
    (`request_tie`); without, its reference in the clear on the prompt lengths, every ingress window accepted when the
    advice's prompt lengths are `certified`'s requests', in arrival order, else every one refused. A tampering of the
    advice leaves the certified ingress, so `certified` is the honest advice; by default `advice` itself. Raises
    ModuleNotFoundError without `advised`, and with `cells` without `io_request`."""
    AD = importlib.import_module(ADVISED_MODULE)
    link = AD.LinkAdvice(advice, tuple(link_received(advice, request_frames, received)))
    regs = n2_registrations(len(link.received))
    if cells is not None:
        accepted = request_tie(importlib.import_module(IO_REQUEST_MODULE), cells, link)
    else:
        ok = certified is None or [q.prompt_len for q in advice.requests] == [q.prompt_len for q in certified.requests]
        accepted = (ok,) * len(regs)
    return link, AD.IoTieVerdict(N2_LINK, regs, AD.digest(link), accepted)


def stated(advised):
    """`n2_advice`'s (link, tie) under a tie the prover states, every ingress window accepted on its own advice: what
    N2's advised counts decide without IoTieRequest."""
    link, tie = advised
    return link, replace(tie, accepted=(True,) * len(tie.accepted))


def _n2_view(AD, advised, window: int, T: int, R: int, syncs, sync_set, request_frames):
    """`advised.view` of the (link, tie) `n2_advice` gives, held to `N2_LINK` and its stand-in registrations, on
    windows the certifier ran under the declared `syncs`."""
    link, tie = advised
    return AD.view(link, window, T, R, syncs, sync_set, request_frames, tie=tie, link_name=N2_LINK,
                   registrations=n2_registrations(len(link.received)), clocks=syncs)


def advised_codes(advised, window: int, syncs, sync_set, request_frames, T: int, R: int, RIN: int, OWN: int, ingress,
                  egress) -> list[str]:
    """The advised counts' code on one advised link-window with heads of counts `ingress` and `egress`, as
    `LinkWindows.advised_check` decides it (`advised.view` under the declared `syncs`, Σ_sync and the link's framing,
    then `advised.reference`), on the advice as N2 reads it and its tie (`n2_advice`'s (link, tie)): ``.view`` when
    the advice is malformed, isn't tied or doesn't cover the window, the advised counts' when they refuse, else none.
    Raises ModuleNotFoundError without `advised`."""
    AD = importlib.import_module(ADVISED_MODULE)
    try:
        v = _n2_view(AD, advised, window, T, R, syncs, sync_set, request_frames)
    except ValueError:
        return [f"{ADVISED}.view"]
    ok = AD.reference(T, RIN, OWN, ingress, egress, v.released, v.floor, v.received)
    return [] if ok else [ADVISED]


def honest_heads(advised, T: int, R: int, syncs, sync_set, request_frames,
                 own: int = 0) -> list[tuple[list[int], list[int]]]:
    """Each window's (ingress, egress) head counts that an honest advised run of `advised` (`n2_advice`'s (link,
    tie)) certifies under `advised`'s rule: the egress the advice's releases, the ingress the request frames its floor
    needs by each bucket, the rest of the window's in its last bucket."""
    AD = importlib.import_module(ADVISED_MODULE)
    out = []
    for w in range(len(advised[0].received)):
        v = _n2_view(AD, advised, w, T, R, syncs, sync_set, request_frames)
        ingress, prev = [], 0
        for t, f in enumerate(v.floor):
            ingress.append(f - prev + (own if t == 0 else 0))
            prev = f
        ingress[-1] += v.received - prev
        out.append((ingress, list(v.released)))
    return out


def advice_suite(advice, *, T: int, R: int, syncs, request_frames, received: list[int], sync_set=None,
                 RIN: int | None = None, frame_bytes: int | None = None, records: dict[str, dict] | None = None,
                 model: str = "") -> dict[str, dict]:
    """Every advice negative made from a run's `advice` (I1's): whether it applies to it, whether it's valid advice that
    round-trips through I1's codec, its bits, and `advised`'s codes (`advised_codes`, tied to the certified ingress,
    `n2_advice`) on the heads an honest run of the advice certifies (`honest_heads`, under the link's framing
    `request_frames` with each window's `received` and declared sync; Σ_sync by default the declared syncs' deltas and
    rhos, `RIN` their largest ingress count). The tie: with `frame_bytes`, whose warden's framing `request_frames` must
    be (`link_framing`), IoTieRequest's reference (`request_tie`) on the warden's ingress of the advice's id requests
    (`id_requests`, from `records` and `model`), each window's the honest `received`'s (`certified_ingress`), each
    row with its verdict by window (`request_tie`) and ``IoTieRequest`` among its codes when it refuses; else the
    stand-in on the prompt lengths. `untied`: `advised`'s codes under a tie the prover states (`stated`), what N2
    refuses without IoTieRequest. `ok`: the codes are exactly the checks the suite reads among the ones it names, its
    code and `also` (none for a step-unit negative, which N2 must accept: its check is pending). A negative that
    releases more than R frames in a bucket is refused by the view (``.view``) instead. An honest advice the view
    refuses has no honest heads, so its row says why and no negative runs."""
    mod = advice_module()
    if mod is None:
        raise NotApplicable(f"no advice codec: {ADVICE_MODULE} (I1) isn't in this tree")
    Q = io_request_module() if frame_bytes is not None else None
    sync_set = sync_set or SyncSet(tuple(sorted({s.delta for s in syncs})), tuple(sorted({s.rho for s in syncs})))

    def codes(link) -> list[str]:
        return sorted({c for w, (i, e) in enumerate(heads)
                       for c in advised_codes(link, w, syncs, sync_set, request_frames, T, R, RIN, 0, i, e)})

    try:
        heads = honest_heads(n2_advice(advice, request_frames, received), T, R, syncs, sync_set, request_frames)
    except ValueError as err:
        why = f"N2's advised view refuses the honest advice: {err}"
        return {"honest": {"bits": mod.total_bits(advice), "codes": [f"{ADVISED}.view"], "ok": False, "why": why}} | {
            name: {"code": n.code, "check": n.check, "waits": list(n.waits), "ran": False, "why": why}
            for name, n in NEGATIVES.items() if n.tampers == "advice"}
    RIN = RIN or max(1, *(c for i, _ in heads for c in i))
    cells = None if Q is None else certified_ingress(Q, id_requests(advice, records, frame_bytes, model),
                                                     link_received(advice, request_frames, received), T=T,
                                                     frame_bytes=frame_bytes)
    read = {ADVISED, f"{ADVISED}.view"} | ({REQUEST_TIE} if cells is not None else set())

    def judged(b, certified=None) -> tuple[dict, Any]:
        advised = n2_advice(b, request_frames, received, certified, cells)
        if cells is None:
            return {"codes": codes(advised)}, advised
        tie = list(advised[1].accepted)
        return {"codes": sorted(codes(advised) + ([] if all(tie) else [REQUEST_TIE])), "request_tie": tie}, advised

    honest, _ = judged(advice)
    out = {"honest": {"bits": mod.total_bits(advice)} | honest | {"ok": honest["codes"] == []}}
    for name, n in NEGATIVES.items():
        if n.tampers != "advice":
            continue
        row: dict[str, Any] = {"code": n.code, "check": n.check, "waits": list(n.waits)}
        try:
            b = n.make.edit(advice)
        except NotApplicable as err:
            out[name] = row | {"ran": False, "why": str(err)}
            continue
        got, advised = judged(b, advice)
        want = sorted(c for c in {n.code, *n.also} if c in read)
        moved = {j: [p.prompt_len, q.prompt_len] for j, (p, q) in enumerate(zip(advice.requests, b.requests))
                 if len(b.requests) == len(advice.requests) and p.prompt_len != q.prompt_len}
        out[name] = row | {"ran": True, "valid": mod.decode(mod.encode(b)) == b, "bits": mod.total_bits(b),
                           "steps": len(b.steps), "outputs": len(b.release)} | got | \
            ({"prompt_lens": moved} if moved else {}) | {"untied": codes(stated(advised)), "ok": got["codes"] == want}
    return out


def advice_run(path: Path, *, T: int = 50, R: int = 32, frame_bytes: int = cs.FRAME_BYTES, delta: int = 0,
               rho: int = 1000, shift: int = 0, ids: Path | None = None, model: str = "") -> dict:
    """`advice_suite` on the advice at `path` (I1's encode bytes) under the link's framing at `frame_bytes`-byte
    frames (`link_framing`), all its requests' frames received in window 0, every window's sync (δ, ρ): its shape and
    the rows (`negatives`, or why not), each tied by IoTieRequest's reference once `io_request` is importable, on the
    id requests of the run's records (`ids`, `read_requests`; `model` the name they were sent under). With `shift`,
    every release `shift` buckets later, for an advice whose clock put an output earlier than a declared δ allows,
    which `advised`'s floor refuses. Under δ = 0 an output in its request's own bucket (a served first token, the
    engine's clock with no certifier on the path) is accepted. With the run's sampled `ids` (`read_ids`), `tie_suite`
    on it at T buckets of R cells (`tie`)."""
    mod = advice_module()
    if mod is None:
        raise NotApplicable(f"no advice codec: {ADVICE_MODULE} (I1) isn't in this tree")
    adv = mod.decode(path.read_bytes())
    if shift:
        adv = replace(adv, release=tuple(b + shift for b in adv.release))
    windows = max(adv.release, default=0) // T + 1
    framing = link_framing(frame_bytes, max((q.prompt_len for q in adv.requests), default=1))
    received = [sum(framing[q.prompt_len - 1] for q in adv.requests)] + [0] * (windows - 1)
    try:
        rows = advice_suite(adv, T=T, R=R, syncs=[ClockSync(delta, Fraction(rho))] * windows, request_frames=framing,
                            received=received, frame_bytes=frame_bytes, records=read_requests(ids) if ids else None,
                            model=model)
    except ModuleNotFoundError as err:
        rows = {"why": f"N2's advised check isn't in this tree ({err.name})"}
    except NotApplicable as err:
        rows = {"why": str(err)}
    doc = {"format": FORMAT, "advice": str(path), "requests": len(adv.requests), "steps": len(adv.steps),
           "outputs": len(adv.release), "windows": windows, "T": T, "R": R, "frame_bytes": frame_bytes,
           "received": received, "shift": shift, "negatives": rows}
    if ids is not None:
        try:
            doc["tie"] = tie_suite(adv, read_ids(ids), T=T, R=R)
        except NotApplicable as err:
            doc["tie"] = {"why": str(err)}
    return doc


# --- the I/O tie on a fake upstream (`io_tie`) -------------------------------------------------------------------------
# No warden writes cells into a run's directory yet, and no Commit's port is laid out as the tie reads it, so the tie's
# negatives run on what a fake upstream makes of a run's advice and sampled ids, as `io_tie`'s own tests make them: a
# warden of r slots a connection (`tie_schedule`), whose cells are the ones the map admits (each kept output a data cell
# on its request's stream, a PAD or the END on its stream elsewhere in its lane, B zero bytes in a slot no connection
# holds), and the Commit's port, a row of sampled ids a step, both committed under fake salts. The map is the
# verifier's, from the advice and the warden's admissions.

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
    """The fake upstream's warden for `advice`: each request's admission bucket and r, the slots a connection holds a
    bucket (`slots`, or by default the most outputs a request releases in its first release bucket), with no delay:
    request i is admitted at its first kept output's release bucket, a request that keeps none at the one before it
    (bucket 0 for the first). The admissions are the warden's wiring, which the verifier takes from the certified
    record, so every tampering of the advice is read against the honest advice's."""
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
    """`advice` as the fake upstream's warden serves it (`tie_schedule`'s): each kept output k of request i released at
    admitted[i] + ⌊k / r⌋, the bucket the warden sends its cell in. An advice's release is the warden's schedule, which
    the verifier's map holds it to, so an advice no warden served is read as this one's."""
    admitted, r = schedule
    return replace(advice, release=tuple(admitted[q] + k // r for _, _, q, k in _kept(advice.requests, _plain(advice))))


def tie_negative(edited, served, schedule):
    """An advice negative of the `served` advice as the fake upstream's warden reads it: an output the served advice
    doesn't keep (by its request's id and index) released where the warden would send it, admitted[i] + ⌊k / r⌋ for
    its request i (a request past the warden's connections where the negative releases it); every other output where
    the negative releases it, so a moved release stays moved."""
    admitted, r = schedule
    honest = _buckets(served)
    kept = _kept(edited.requests, _plain(edited))
    return replace(edited, release=tuple(
        b if (edited.requests[q].id, k) in honest or q >= len(admitted) else admitted[q] + k // r
        for (_, _, q, k), b in zip(kept, edited.release)))


def _ends(advice, schedule) -> list[int]:
    """Each request's `END` bucket under the warden's schedule: its admission plus ⌊n / r⌋, n its kept outputs."""
    admitted, r = schedule
    n = [0] * len(advice.requests)
    for _, _, q, _ in _kept(advice.requests, _plain(advice)):
        n[q] += 1
    return [a + k // r for a, k in zip(admitted, n)]


def tie_shape(IT, advices, schedule, *, T: int, R: int | None = None, B: int = 256):
    """One shape for the tie of every advice in `advices` (the honest one and its tamperings), so that each reaches the
    unit's parts: R (unless given) r slots for each of the most connections live in a bucket under the warden's
    `schedule` (`tie_schedule`; a connection live from its admission to its `END`'s bucket), P its widest step's
    batch, S the most steps a bucket's outputs were sampled in, and DS the depth of a port over the longest session."""
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
    output's id, 0 for a row that keeps none), empty rows to 2^DS steps, under fake salts (a prover draws fresh ones):
    its rows, salts, tree levels and domain id."""
    got = {(o.step, o.row): x for o, x in zip(outputs, ids)}
    rows = [IT.port_row([got.get((s, j), 0) for j in range(len(advice.steps[s].rows))] if s < len(advice.steps) else [],
                        shape.P) for s in range(1 << shape.DS)]
    pd = IT.domain(hashlib.sha512(b"port" + seed).digest(), len(rows))
    salts = [IT.fake_salt(seed + b"p", s) for s in range(len(rows))]
    return rows, salts, IT.rows_levels(pd.domain_id, rows, salts), pd.domain_id


def tie_map(IT, shape, window: int, outputs, release, schedule):
    """The map and each bucket's steps the verifier computes for egress window `window` from the advice's `outputs` and
    `release` and the warden's admissions and r (`schedule`, no delay; `io_tie.window_map`, a ValueError when it
    can't). Advice request i is on stream i + 1, v0's order."""
    admitted, r = schedule
    return IT.window_map(shape, outputs, release, window, admitted, 0, r)


def tie_cells(IT, shape, cmap, steps, port) -> list[bytes]:
    """The warden's cells of an egress window on the fake upstream: each slot the one cell its record admits
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
    """Window `window`'s tie: `cells` committed as the warden commits them (a row a bucket, under fake salts), the map
    and each bucket's steps from the advice (`tie_map`), and each bucket's port rows opened at its steps from the
    Commit's (`port`, `tie_port`'s)."""
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
    """`token` on the tie: the cells with the first data cell carrying another id than the sampled one, which the warden
    certifies as it saw it (the cells to recommit)."""
    k = _first_data(IT, inst.cells)
    p, cells = IT.parse_cell(inst.cells[k]), list(inst.cells)
    cells[k] = IT.data_cell((p.id + 1) % (1 << 32), p.stream, inst.shape.B)
    return cells


def cell_dropped(IT, inst) -> list[bytes]:
    """`drop` on the tie: the cells with a `PAD` on its stream in place of the first data cell, as the warden releases
    that slot when the event never came (to recommit)."""
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
    (by request id), served through a warden of r = `slots` slots a connection (`tie_schedule`, `tie_served`): each
    egress window's cells and the Commit's port, honest and as each negative makes them (a cell or port negative in
    the first window with a data cell; an advice negative the map the verifier computes from its advice and the
    warden's admissions, over the honest cells and port), with the parts of `io_tie.reference_parts` that fail, by
    window (``map``, at window −1, when the verifier can't compute the map, with `why`: an advice whose release isn't
    the warden's schedule, or whose requests aren't its connections). One shape serves them all (`tie_shape`). `ok`:
    the part it names fails; for one it must accept, none does while its outputs are honest's, and content does once
    the edit moved them (`moved`). NotApplicable when the warden can't serve the advice in R slots."""
    IT = io_tie_module()
    if IT is None:
        raise NotApplicable(f"no I/O tie: {IO_TIE_MODULE} isn't in this tree")
    schedule = tie_schedule(advice, slots)
    advice = tie_served(advice, schedule)
    edited: dict[str, Any] = {}
    for name in TIE_PARTS:
        if NEGATIVES[name].tampers == "advice":
            try:
                edited[name] = tie_negative(NEGATIVES[name].make.edit(advice), advice, schedule)
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


# --- the I/O tie on a certifier run ------------------------------------------------------------------------------------
# A certifier-serve run keeps the prover's copy of each window's egress frames (`responses.records`) beside the
# certified heads (`certified.jsonl`), so its frames can be held to their certified head; it keeps no warden cells, no
# advice and no Commit. `tie_run` runs `io_tie.reference_parts` on each egress window of it, standing in for what it
# lacks (`TIE_STAND_INS`), and names what keeps the tie from opening a run (`TIE_MISSING`).

TIE_RUN = "verity-network-tie-run/v1"
#: what a run must keep for the tie to open it, by the key `tie_run`'s `missing` names it by
TIE_MISSING = {
    "io-tie": f"the I/O tie: {IO_TIE_MODULE} isn't in this tree",
    "id-events": "token-id events: no egress frame is an id event (`active.id_event`), or on a compact plan a compact "
                 "record (`active.compact_record`), so none is a data cell the tie reads an id from; a fixed plan's "
                 "frames are text events, and only a varied plan (`certifier_serve.py plan --traffic varied`, with "
                 "`--compact B` its records) writes them",
    "commit": "the Commit's rows: the endpoint's sampled ids of the run's requests, by each id request's user (--ids), "
              "which the run doesn't keep (the shim drops `user` before it forwards a request)",
}
#: what the tie on a run takes in place of what the unit reads and the run doesn't keep
TIE_STAND_INS = {
    "cells_root": "each window's frames held to its certified egress head (record v2's frames root, under its "
                  "certificate), then committed under fake salts in place of the warden's bucket rows",
    "cells": "each certified id event as v4's data cell on the stream its ordinal names, and on a compact plan each "
             "compact record (`active.compact_record_data`) as the data cell of its id, in an id event's frame "
             "(`certifier_serve.FRAME_BYTES`); any other frame as it is",
    "port": "the Commit's port from --ids: one row a certified request's output, in (stream, index) order, under fake "
            "salts, each bucket opening the rows of its own data cells",
    "map": "the certified placement, each data cell its stream's next output, since no advice says where an output "
           "goes, and any other slot held to B zero bytes (`io_tie.UNHELD`, the certifier's padding); a committed "
           "output no cell carries (`unplaced`), which the advice's map would place in its release window, fails "
           "content at window -1",
}
#: the tie's parts in the order a run's `failing` lists them; ``map`` on every window when the certified ingress, which
#: names each stream's request, doesn't hold
TIE_ORDER = ("map", "cells_root", "port", "content")
NOT_APPLICABLE = "not-applicable"


def tie_cell(IT, slot: bytes | None, B: int, compact: bool = False) -> bytes:
    """A certified egress slot as the tie's B-byte cell: padding a pad, an id event's canonical frame (or under
    `compact` a compact record) the v4 data cell of its id on the stream its ordinal names, any other frame as it is."""
    if slot is None:
        return IT.pad_cell(B)
    if compact:
        got = compact_record_data(slot)
        return slot if got is None else IT.data_cell(got[1], got[0], B)
    kind, stream, payload = decode_frame(slot)
    got = id_event_data(payload) if (kind, stream) == (DATA, CANONICAL_STREAM) else None
    if got is None or slot != encode_frame(DATA, CANONICAL_STREAM, payload, B):
        return slot
    return IT.data_cell(got[1], got[0], B)


def _is_id_event(slot: bytes) -> bool:
    kind, stream, payload = decode_frame(slot)
    return (kind, stream) == (DATA, CANONICAL_STREAM) and id_event_data(payload) is not None


def frames_held(certified: CertifiedRecord | None, record, params, key, session: bytes, link: str) -> bool:
    """Whether a window's frames (`record`, the prover's copy) are the certified ones: the certificate verifies under
    the certifier's key and session, the head is the link's on the grid's parameters, and its counts and frames root
    are the frames' (`certificate.frames_root` of their `real_frames`)."""
    if certified is None or record is None or not certified.verify(key, session):
        return False
    h, p = certified.head, params
    if (h.link, h.buckets, h.allocation, h.frame_bytes) != (link, p.buckets, p.allocation, p.frame_bytes):
        return False
    try:
        frames = real_frames(record.grid, params)
    except ValueError:
        return False
    return (h.counts == tuple(sum(f is not None for f in row) for row in record.grid)
            and h.root == frames_root(frames, params))


def tie_run(run: Path, ids: dict[str, list[int]] | None, seed: bytes = b"tie-run",
            aborted: Collection[str] = ()) -> dict:
    """The I/O tie on each egress window of the certifier-serve run `run`, `ids` the endpoint's sampled ids by request
    user (`read_ids`; None when not given), with `TIE_STAND_INS` for what the run doesn't keep. `verdict`: refused
    when a window's parts fail (`failing`, by window, in `TIE_ORDER`; window -1 the committed outputs no cell carries;
    `codes` their ``IoTieWindow.PART``) or the finish doesn't hold (`finish`), accepted when neither,
    `NOT_APPLICABLE` when the run lacks what the tie opens (`missing`, by `TIE_MISSING`'s keys). `finish`: whether
    each response's length and finish are the Commit's; on a compact plan the link's records held to the sampled ids,
    the users the endpoint says their clients `aborted` (`read_aborted`) and each request's `max_tokens`
    (`active.compact_finish_violations`, each (ordinal, why) in `finish_violations`), on id events true, their finish
    being their ids'; None without the ids or the events. The windows whose frames are `held` to their certified
    heads, at each end; the run's `id_events` (a compact plan's records), its certified requests' committed
    `outputs`, the data cells `placed` on one, and the committed outputs no cell carries (`unplaced`)."""
    IT = io_tie_module()
    plan, report, dep = (json.loads((run / f).read_text())
                         for f in ("plan.json", "calibration.json", "deployment.json"))
    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    key, session, link = cs.read_public(dep["certifier_key"]), bytes.fromhex(dep["session"]), dep["links"]["n2"]
    certified, _ = cs.read_certified(run / "certified.jsonl")
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    windows, egress = plan["windows"], {r.window: r for r in outs}
    ends = {Direction.EGRESS: (egress, e), Direction.INGRESS: ({r.window: r for r in ins}, i)}
    held = {d.value: [w for w in range(windows)
                      if frames_held(certified.get((d, w)), recs.get(w), p, key, session, link)]
            for d, (recs, p) in ends.items()}
    compact = bool(plan.get("compact"))
    real = [s for r in sorted(outs, key=lambda r: r.window) for row in r.grid for s in row if s is not None]
    events = sum((compact_record_data(s) is not None) if compact else _is_id_event(s) for s in real)
    missing = {k: TIE_MISSING[k] for k, lacks in (("io-tie", IT is None), ("id-events", not events),
                                                  ("commit", ids is None)) if lacks}
    records = cs.certified_io(plan, setup, ins, outs)["records"] if events and ids is not None else None
    users = None if records is None else {r["stream"]: r["user"] for r in records}
    finish = compact_finish_violations({q: ids.get(u, ()) for q, u in users.items()}, plan["varied"]["eos_token_ids"],
                                       real, max_tokens={r["stream"]: r["max_tokens"] for r in records},
                                       aborted={q for q, u in users.items() if u in aborted}
                                       ) if compact and users is not None else []
    out = {"verdict": NOT_APPLICABLE, "codes": [], "failing": {}, "windows": windows,
           "held": {end: len(ws) for end, ws in held.items()}, "id_events": events, "missing": missing,
           "finish": None if users is None else not finish, "finish_violations": [list(v) for v in finish]}
    if missing:
        return out
    step, rows = {}, []
    for q in sorted(users):
        for k, x in enumerate(ids.get(users[q], ())):
            step[q, k] = len(rows)
            rows.append(IT.port_row([x], 1))
    DS = max(1, (len(rows) - 1).bit_length())
    rows += [IT.port_row([], 1)] * ((1 << DS) - len(rows))
    pd = IT.domain(hashlib.sha512(b"port" + seed).digest(), len(rows))
    salts = [IT.fake_salt(seed + b"p", s) for s in range(len(rows))]
    port = rows, salts, IT.rows_levels(pd.domain_id, rows, salts), pd.domain_id
    nxt: dict[int, int] = defaultdict(int)
    placed: dict[int, tuple[list[bytes], list]] = {}
    B = cs.FRAME_BYTES if compact else e.frame_bytes
    for w in range(windows):
        grid = egress[w].grid if w in egress else [(None,) * e.allocation] * e.buckets
        cells, at = [tie_cell(IT, s, B, compact) for row in grid for s in row], []
        for c in cells:
            p = IT.parse_cell(c)
            if p is not None and p.data:
                at.append((p.stream, step.get((p.stream, nxt[p.stream]))))
                nxt[p.stream] += 1
            else:
                at.append(None)
        placed[w] = cells, at
    R = e.allocation

    def bucket_steps(at: list, t: int) -> list[int]:
        return sorted({x[1] for x in at[t * R:(t + 1) * R] if x and x[1] is not None})

    S = max([1, *(len(bucket_steps(at, t)) for _, at in placed.values() for t in range(e.buckets))])
    shape = IT.Shape(B, e.buckets, R, P=1, S=S, DS=DS)
    ingress_held = len(held[Direction.INGRESS.value]) == windows
    for w, (cells, at) in placed.items():
        cmap, steps = [], []
        for t in range(e.buckets):
            mine = bucket_steps(at, t)
            sigma = {s: n for n, s in enumerate(mine)}
            cmap += [IT.MapEntry(x[0], sigma[x[1]]) if x and x[1] is not None else IT.UNHELD
                     for x in at[t * R:(t + 1) * R]]
            steps.append(tuple(mine + [mine[-1] if mine else 0] * (S - len(mine))))
        inst = _instance(IT, shape, w, cells, cmap, steps, port, seed)
        bad = {p for p, ok in IT.reference_parts(inst).items() if not ok}
        bad |= {"cells_root"} if w not in held[Direction.EGRESS.value] else set()
        bad |= {"map"} if not ingress_held else set()
        if bad:
            out["failing"][w] = [p for p in TIE_ORDER if p in bad]
    committed = {q: len(ids.get(u, ())) for q, u in users.items()}
    unplaced = sum(max(0, n - nxt[q]) for q, n in committed.items())
    if unplaced:
        out["failing"][-1] = ["content"]
    out["codes"] = sorted({f"{TIE}.{p}" for parts in out["failing"].values() for p in parts})
    out["verdict"] = "refused" if out["failing"] or out["finish"] is False else "accepted"
    return out | {"outputs": sum(committed.values()),
                  "placed": sum(1 for _, at in placed.values() for x in at if x and x[1] is not None),
                  "unplaced": unplaced}


def run_dirs(src: Path, scratch: Path) -> dict[str, Path]:
    """The certifier-serve runs in `src` by name (each directory with a `certified.jsonl`, named for its strategy): a
    gate run's tarball (`c…/out/STRATEGY/`) extracted into `scratch`, a directory of runs, or one run's directory."""
    if src.is_file():
        with tarfile.open(src) as tf:
            tf.extractall(scratch, filter="data")
        src = scratch
    found: dict[str, Path] = {}
    for p in sorted(src.rglob("certified.jsonl")):
        if found.setdefault(p.parent.name, p.parent) != p.parent:
            raise NotApplicable(f"{src}: two runs named {p.parent.name}")
    return found


def tie_ok(name: str, row: dict) -> bool | None:
    """Whether the tie's verdict on run `name` is the one it owes: honest's accepted, a negative's it reads
    (`TIE_PARTS`, `TIE_RUN_PARTS`) refused by its part; None for any other run, or one with no verdict."""
    if row["verdict"] == NOT_APPLICABLE:
        return None
    if name == "honest":
        return row["verdict"] == "accepted"
    part = TIE_PARTS.get(name) or TIE_RUN_PARTS.get(name)
    return f"{TIE}.{part}" in row["codes"] if part else None


def tie_runs(src: Path, ids: Path | None = None) -> dict:
    """`tie-run`'s document (`TIE_RUN`): the tie's row on each run in `src` (`run_dirs`, `tie_run`, the Commit's ids and
    aborts `ids`' own, `read_ids` and `read_aborted`), by name, with
    `ok` (`tie_ok`), and what it stands in for (`TIE_STAND_INS`)."""
    sampled = read_ids(ids) if ids is not None else None
    gone = read_aborted(ids) if ids is not None else set()
    with tempfile.TemporaryDirectory() as scratch:
        runs = run_dirs(src, Path(scratch))
        if not runs:
            raise NotApplicable(f"{src}: no certifier-serve run (no certified.jsonl in it)")
        rows = {name: tie_run(run, sampled, aborted=gone) for name, run in runs.items()}
    return {"format": TIE_RUN, "check": TIE, "source": str(src), "ids": None if ids is None else str(ids),
            "stand_ins": TIE_STAND_INS,
            "runs": {name: {"verdict": r["verdict"], "ok": tie_ok(name, r)} | r for name, r in rows.items()}}


def n2(run: Path, clock=time.monotonic) -> dict:
    """N2's verdict on every link-window of the run's `certified.jsonl`, through the proof system's stream as
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
          ids: Path | None = None, clock=time.monotonic) -> dict[str, dict]:
    """Every negative's row for the one command's report, from a rehearsal's runs by strategy (`runs`, which must hold
    honest): a shim negative is its own run (`rehearse --strategies`), any other is made from honest's under `work`.
    A row says what the negative tampers, the check and code it names and what it waits on; whether it ran, and if so
    N2's verdict, its refused link-windows and their codes, and `ok`: refused by its code; for a shim negative whose
    check is pending (token), N2's accepting it, since only that check may refuse it; else None. An advice negative's
    verdict is `advice_run`'s, `advised`'s codes on the heads an honest run of `advice` certifies (`honest_codes`
    theirs on the honest advice), since the certified records don't change with it; `ok` needs the honest advice
    accepted. With the run's sampled `ids` too, each negative the I/O tie reads carries the tie's verdict on a fake
    upstream (`tie`, `tie_suite`'s row), and its `ok` needs the tie to refuse it by its part (or accept it) and accept
    the honest windows."""
    out, judged, tie = {}, None, {}
    if advice is not None and ids is not None:
        try:
            doc = advice_run(advice, ids=ids)
            judged, tie = doc["negatives"], doc["tie"]
        except NotApplicable as err:
            tie = {"why": str(err)}
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
                judged = advice_run(advice)["negatives"]
            got = judged.get(name)
            if got is None:
                row |= {"ran": False, "copy": str(run), "why": judged["why"]}
            else:
                honest = judged["honest"]
                row |= {"copy": str(run), "honest_codes": honest["codes"],
                        "heads": "an honest run of the advice's, under N2's advised rule"}
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
        t = tie.get(name) if name in TIE_PARTS else None
        if name in TIE_PARTS and t is None and ids is not None:
            t = {"ran": False, "why": tie.get("why", "no advice (--advice)")}
        if t is not None:
            row["tie"] = t
            if t["ran"]:
                ok = t["ok"] and tie["honest"]["ok"]
                row["ok"] = ok if row.get("ok") is None else row["ok"] and ok
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
    w = sub.add_parser("warden", help="frame on a warden-serve run's own cells, by the verifier's registered read")
    for q in (m, x, j, w):
        q.add_argument("--run", required=True, type=Path)
    a = sub.add_parser("advice", help="every advice negative of a run's advice, and N2's advised codes on it")
    a.add_argument("--advice", required=True, type=Path, help="the run's advice (I1's encode bytes)")
    a.add_argument("--T", type=int, default=50, help="buckets a window (the advice's release buckets')")
    a.add_argument("--R", type=int, default=32, help="egress frames a bucket")
    a.add_argument("--frame-bytes", type=int, default=cs.FRAME_BYTES,
                   help="the link's frame size: each request's frames are its id request's at this size")
    a.add_argument("--model", default="", help="the model the run's requests named, in their id requests")
    a.add_argument("--delta", type=int, default=0, help="each window's declared sync: δ buckets ...")
    a.add_argument("--rho", type=int, default=1000, help="... and ρ frames a bucket")
    a.add_argument("--shift", type=int, default=0, help="every release this many buckets later")
    a.add_argument("--ids", type=Path, help="the run's records (serve-openai's requests.jsonl): each request's prompt "
                   "and what it brought, for IoTieRequest's ingress, and its sampled ids, for the I/O tie")
    r = sub.add_parser("tie-run", help="the I/O tie's verdict on each run of a gate run's tarball or a run directory")
    r.add_argument("--run", required=True, type=Path,
                   help="a gate run's tarball (c…/out/STRATEGY/), a directory of runs, or one run's directory")
    r.add_argument("--ids", type=Path, help="the endpoint's sampled ids by request user (requests.jsonl: user, "
                   "output_ids), the Commit's rows")
    args = ap.parse_args(argv)
    if args.command == "tie-run":
        try:
            doc = tie_runs(args.run, args.ids)
        except NotApplicable as err:
            raise SystemExit(str(err))
        print(json.dumps(doc, indent=1))
        return 0 if all(r["verdict"] != NOT_APPLICABLE and r["ok"] is not False for r in doc["runs"].values()) else 1
    if args.command == "advice":
        try:
            doc = advice_run(args.advice, T=args.T, R=args.R, frame_bytes=args.frame_bytes, delta=args.delta,
                             rho=args.rho, shift=args.shift, ids=args.ids, model=args.model)
        except NotApplicable as err:
            raise SystemExit(str(err))
        print(json.dumps(doc, indent=1))
        rows = [r for part in ("negatives", "tie") for r in (doc.get(part) or {}).values() if isinstance(r, dict)]
        return 0 if all(r.get("ok") is not False for r in rows) else 1
    if args.command == "table":
        print(table(args.markdown))
        return 0
    if args.command == "make":
        print(make(args.name, args.run, args.out, advice=args.advice, commit=args.commit))
        return 0
    if args.command == "warden":
        try:
            out = warden_frame(args.run)
        except NotApplicable as err:
            raise SystemExit(str(err))
        print(json.dumps(out))
        return 0 if out["ok"] else 1
    verdict = n2(args.run)
    if args.command == "n2":
        print(json.dumps(verdict, indent=1))
        return 0
    out = judge(args.name, verdict)
    print(json.dumps(out))
    return 0 if out["by_code"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
