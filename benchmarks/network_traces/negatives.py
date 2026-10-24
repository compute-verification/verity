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
  position)); and, until their owners name them, ``StepUnit`` (PROVE-PATH: the compiled step Program's digest against
  the committed trace's, and the committed values passing it) and ``Commit`` (the Commit's opening under its registered
  root).

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
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402

FORMAT = "verity-network-negatives/v1"
ADVICE_MODULE = "verity.core.protocols.certifier.advice"
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
# On I1's dataclasses as drafted: Advice(requests, steps, release), RequestAdvice(id, prompt_len, output_len, finish),
# StepAdvice(rows), Row(request, kind, start, n, cache_hit, preempted), and N2-ADVICE's additions where the advice has
# them: each request's `ingress_frames` and each window's `received`. Each returns new advice and leaves its argument,
# and each keeps the advice well formed under N2-ADVICE's frame rule (`advised.frames`: a row samples its request's next
# token when `start + n` is the prompt's length plus the tokens sampled before it), so that what refuses it is a check
# of the advice against the certified I/O or the committed values, not its form.

def _rows(advice) -> list[tuple[int, int, Any]]:
    return [(s, j, row) for s, step in enumerate(advice.steps) for j, row in enumerate(step.rows)]


def _set_row(advice, s: int, j: int, row):
    steps = list(advice.steps)
    rows = list(steps[s].rows)
    rows[j] = row
    steps[s] = replace(steps[s], rows=rows)
    return replace(advice, steps=steps)


def sampling_rows(advice, request: int) -> list[tuple[int, int]]:
    """(step, row) of each row that samples request `request`'s next token, in order (`advised.frames`' rule)."""
    prompt, sampled, out = advice.requests[request].prompt_len, 0, []
    for s, j, row in _rows(advice):
        if row.request == request and row.n >= 1 and row.start + row.n == prompt + sampled:
            out.append((s, j))
            sampled += 1
    return out


def _received(advice, request, by: int):
    """The advice with `by` times request `request`'s ingress frames added to the first window's `received` that can
    take them, when the advice carries N2-ADVICE's ingress fields."""
    if not hasattr(advice, "received"):
        return advice
    got, n = list(advice.received), advice.requests[request].ingress_frames * by
    w = next(w for w, g in enumerate(got) if g > 0 and g + n >= 0)
    got[w] += n
    return replace(advice, received=got)


def output_length(advice, request: int = 0, by: int = 1):
    """Request `request`'s output length one more (`by` = 1: a decode row samples one token past its last, in the step
    after its last sampling row's, a step appended at the last release bucket if there is none) or one less (−1: its
    last sampling row, a decode row, left out). The advice's frames then disagree with what the egress certified."""
    q, rows = advice.requests[request], sampling_rows(advice, request)
    requests = list(advice.requests)
    requests[request] = replace(q, output_len=q.output_len + by)
    out = replace(advice, requests=requests)
    if by == -1:
        s, j = rows[-1]
        row = advice.steps[s].rows[j]
        if q.output_len < 2 or row.n != 1:
            raise NotApplicable(f"request {request}'s last token isn't sampled by a decode row")
        steps = list(out.steps)
        steps[s] = replace(steps[s], rows=[r for k, r in enumerate(steps[s].rows) if k != j])
        return replace(out, steps=steps)
    if by != 1:
        raise ValueError("an output length one more (by = 1) or one less (by = -1)")
    if not rows:
        raise NotApplicable(f"request {request} samples no token")
    s, j = rows[-1]
    last = advice.steps[s].rows[j]
    row = replace(last, kind="decode", start=q.prompt_len + q.output_len - 1, n=1,
                  **{f: v for f, v in (("cache_hit", 0), ("preempted", False)) if hasattr(last, f)})
    s, steps, release = s + 1, list(out.steps), list(out.release)
    if s == len(steps):
        steps.append(replace(steps[-1], rows=()))
        release += release[-1:]
    steps[s] = replace(steps[s], rows=list(steps[s].rows) + [row])
    return replace(out, steps=steps, release=release)


def missing_request(advice, request: int = -1):
    """Request `request` left out, with its rows, the later requests' indices moved down (and its ingress frames out of
    `received`): the advice serves one request fewer than the ingress certified."""
    i = range(len(advice.requests))[request]
    advice = _received(advice, i, -1)
    steps = [replace(st, rows=[replace(r, request=r.request - (r.request > i)) for r in st.rows if r.request != i])
             for st in advice.steps]
    return replace(advice, requests=[q for k, q in enumerate(advice.requests) if k != i], steps=steps)


def extra_request(advice, request: int = 0, suffix: str = "~extra"):
    """A request more: a copy of request `request` under another id, its rows beside the original's in the same steps
    (and its ingress frames in `received`), so the advice is whole but serves one request more than the ingress
    certified."""
    new = len(advice.requests)
    q = advice.requests[request]
    steps = [replace(st, rows=list(st.rows) + [replace(r, request=new) for r in st.rows if r.request == request])
             for st in advice.steps]
    out = replace(advice, requests=list(advice.requests) + [replace(q, id=f"{q.id}{suffix}")], steps=steps)
    return _received(out, new, 1)


def release_moved(advice, step: int | None = None, by: int = 1):
    """Step `step`'s output released `by` buckets later (by default the first step that samples a token), and each later
    step released before that bucket moved up to it, so the release stays in order. The advice's release disagrees with
    the egress buckets certified."""
    if step is None:
        step = min((s for i in range(len(advice.requests)) for s, _ in sampling_rows(advice, i)), default=None)
        if step is None:
            raise NotApplicable("no step samples a token")
    rel = list(advice.release)
    rel[step] += by
    rel[step + 1:] = [max(b, rel[step]) for b in rel[step + 1:]]
    return replace(advice, release=rel)


def start_shifted(advice, step: int = -1, by: int = -1):
    """The first decode row of step `step` (the last by default) starting `by` positions later, its last position kept
    (`n` − `by`): by default one earlier, a decode row of two positions. Its frames are the same, so N2 accepts it,
    but its compiled Program computes other positions than the committed values were computed at."""
    s = range(len(advice.steps))[step]
    j = next((j for j, r in enumerate(advice.steps[s].rows)
              if r.kind == "decode" and r.n - by >= 1 and r.start + by >= 0), None)
    if j is None:
        raise NotApplicable(f"step {s} has no decode row that can start {by} later with its end kept")
    row = advice.steps[s].rows[j]
    return _set_row(advice, s, j, replace(row, start=row.start + by, n=row.n - by))


def chunk_split(advice):
    """One prompt token moved across the boundary of a request's two consecutive prefill chunks: the first chunk one
    token shorter, the next one longer and starting one earlier. The positions each request is served are the same,
    but two steps' Programs aren't the ones the committed trace realized."""
    by_request: dict[int, list] = {}
    for s, j, row in _rows(advice):
        if row.kind == "prefill":
            by_request.setdefault(row.request, []).append((s, j, row))
    for chunks in by_request.values():
        for (s, j, a), (t, k, b) in zip(chunks, chunks[1:]):
            if a.n >= 2 and b.start == a.start + a.n and t > s:
                out = _set_row(advice, s, j, replace(a, n=a.n - 1))
                return _set_row(out, t, k, replace(b, start=b.start - 1, n=b.n + 1))
    raise NotApplicable("no request's prefill is chunked: no chunk to split differently")


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
# On I3 as drafted: per step, each batch row's `user` and token positions, `c00.input/ids` and
# `runner.sampler/sampled_token_ids`, under the Commit's registered root.

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
    `certified`, `advice` or `commit`), the `check` that must refuse it and its refusal `code` (`served`: the code on a
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
             waits=("IO-TIE", "CERT-VARLEN")),
    Negative("advice-length", "advice", "advice", "a request's output length one more than its certified frames, "
             "with the decode row that samples it", "N2's advised counts", ADVISED, advice_edit(output_length),
             waits=("ADV-SPEC", "N2-ADVICE")),
    Negative("advice-missing", "advice", "advice", "a certified request missing from the advice, with its rows",
             "N2's advised counts", ADVISED, advice_edit(missing_request), waits=("ADV-SPEC", "N2-ADVICE"),
             also=(TIE,)),
    Negative("advice-extra", "advice", "advice", "a request in the advice that the ingress didn't certify",
             "N2's advised counts", ADVISED, advice_edit(extra_request), waits=("ADV-SPEC", "N2-ADVICE"),
             also=(TIE,)),
    Negative("advice-release", "advice", "advice", "a step's output released one bucket later than certified",
             "N2's advised counts", ADVISED, advice_edit(release_moved), waits=("ADV-SPEC", "N2-ADVICE")),
    Negative("advice-start", "advice", "advice", "a decode row starting one position earlier, its end kept: N2 "
             "accepts it", "the step unit (committed values against the compiled Program)", "StepUnit.values",
             advice_edit(start_shifted), waits=("ADV-SPEC", "COMPILER", "PROVE-PATH")),
    Negative("wrong-token", "token", "commit", "a committed sampled id changed after the run; the registered root kept",
             "the I/O tie and the step unit (the Commit's opening)", "Commit.opening", commit_edit(wrong_token),
             waits=("SERVE-COMMIT", "IO-TIE", "PROVE-PATH"), also=(TIE, "StepUnit.opening")),
    Negative("advice-chunk", "circuit", "advice", "a prefill chunk split one token differently: N2 accepts it",
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


def advised_codes(advice, window: int, syncs, T: int, R: int, RIN: int, OWN: int, ingress, egress) -> list[str]:
    """N2-ADVICE's code on one advised link-window with heads of counts `ingress` and `egress`, as
    `LinkWindows.advised_check` decides it (`advised.view`, then `advised.reference`): ``.view`` when the advice is
    malformed or doesn't cover the window, the advised counts' when they refuse, else none. Raises ModuleNotFoundError
    until N2-ADVICE lands."""
    AD = importlib.import_module("verity.core.protocols.certifier.advised")
    try:
        v = AD.view(advice, window, T, R, syncs)
    except ValueError:
        return [f"{ADVISED}.view"]
    ok = AD.reference(T, R, RIN, OWN, ingress, egress, v.handed, v.floor, v.queued, v.received)
    return [] if ok else [ADVISED]


def n2(run: Path, clock=time.monotonic) -> dict:
    """N2's verdict of record on every link-window of the run's `certified.jsonl`, through the proof service's stream
    as `certifier_serve.check` takes it (Selection All, the Lean audit) without asking the certifier's salts, and the
    link's network verdict; each refused link-window's codes (`codes`), and their union."""
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
        outcome = st.deliver(draw, [lw.session(opening, draw)]) or lw.outcome(st, draw, opening, syncs)
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
    args = ap.parse_args(argv)
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
