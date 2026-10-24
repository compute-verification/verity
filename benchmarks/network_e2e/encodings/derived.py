"""ENC-DERIVED: the fields of advice v0 (I1) that cost zero bits, because the certified I/O, the committed values, the
registered config or the advice's own history already fix them, and the advice that remains.

The advice is public and its length is the leak (`A_c`). A field leaves the string only when every public consumer of it
(the compiler, `compile_steps`; N2-ADVICE's derived counts; the verifier's `bits`) can recompute it from public data, or
when no public consumer needs it and the unit that does (the tie, IO-TIE) reads it from witness. A field a public consumer
needs but only witness fixes stays in the string: it is *forced* (one accepted value given the I/O, once a check holds it
there), so a bound that charged the advice's freedom rather than its length would charge it 0, but it is sent.

**What each checker holds.** In the clear (public): the residual advice below and the registered config (`Config`: block
size, the KV pool's size, the token budget, the stop ids, the scheduler mode). As witness, inside proofs only: the
certified ingress (each request's `user` id, prompt ids and `max_tokens`), the certified egress (the frames' token ids and
the head's counts) and the committed values (each row's sampled id).

**The derivations** are one walk over the steps (`_walk`), the same for the encoder (`residual`) and the verifier
(`reconstruct`), so they can't drift apart. Each request's known tokens K are its prompt length plus its sampling rows so
far (in async mode the in-flight one too). A row starts at the request's previous row's end (its first row at its
prefix-cache hit, a resumed row at its re-hit), schedules n = min(K − start, the step's budget left, the long-prefill
threshold) tokens (vLLM d9105ea8's running and waiting loops), and samples unless it ends short of K (the GPU runner's
discard rule). `FIELDS` is the table of every I1 field: its verdict, derivation, checker and cost.

**The residual** (`Residual`): each step's requests in scheduling order (the scheduler's choice, ENC-REPLAY's), and
- `prompt_len`, one a request: the compiled Programs' shapes read it (a request's final prefill chunk is P − start), so
  it is sent, forced by the certified id count;
- `hits`: the prefix-cache hit of each request that hits, in blocks, forced by the prompts' history but not checked yet;
- `resumes`: each resumed row's re-hit, unless the pool bound rules preemption out, which the verifier checks;
- `finish_eos`: async mode only, a bit a request (`output_len` subtracts the decode the scheduler ran after the stop);
- `ns`: n where the walk can't derive it: only when the rows aren't in scheduling order (the budget left is unknown);
- `release`: each step's egress bucket on an unshaped egress, where its choices are the timing channel; none on a
  shaped one, where `served.derive` fixes them from the ingress anchors, the output lengths and the declared sync.

`bits_table` measures v0 and the residual per certifier window, at the widths `V0` assumes for I1 (ADV-SPEC's `bits`
replaces them when it lands), after checking `rebuild(residual(a)) == a`.

    python derived.py table --trace TRACE.json[.gz] ... --out TABLE.json [--buckets 600 --bucket-ms 100]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:                                                    # I1, once ADV-SPEC lands it
    from verity.core.protocols.certifier.advice import Advice, RequestAdvice, Row, StepAdvice  # type: ignore
except ImportError:
    @dataclass(frozen=True)
    class RequestAdvice:
        id: str
        prompt_len: int
        output_len: int
        finish: str                                     # "eos" or "length"

    @dataclass(frozen=True)
    class Row:
        request: int
        kind: str                                       # "prefill" or "decode"
        start: int
        n: int
        cache_hit: int
        preempted: bool

    @dataclass(frozen=True)
    class StepAdvice:
        rows: tuple[Row, ...]

    @dataclass(frozen=True)
    class Advice:
        requests: tuple[RequestAdvice, ...]
        steps: tuple[StepAdvice, ...]
        release: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    """The registered serving config, public."""
    block_size: int
    num_gpu_blocks: int
    max_num_batched_tokens: int
    max_model_len: int
    eos_token_id: int
    stop_token_ids: tuple[int, ...]
    scheduler: str = "sync"                             # "async": the scheduler runs one step ahead of the outputs
    long_prefill_token_threshold: int = 0
    max_num_seqs: int = 256
    enable_prefix_caching: bool = True
    rows_in_scheduling_order: bool = True

    @property
    def stops(self) -> frozenset[int]:
        return frozenset((self.eos_token_id, *self.stop_token_ids))


@dataclass(frozen=True)
class CertifiedRequest:
    """One certified ingress request, witness: its `user` id, prompt ids and `max_tokens`."""
    id: str
    prompt_ids: tuple[int, ...]
    max_tokens: int


@dataclass(frozen=True)
class Holdings:
    """Witness, inside the tie only: the certified requests in ingress order, and each request's sampled ids as the
    committed values hold them (equal, by the tie, to its certified egress frames' ids)."""
    requests: tuple[CertifiedRequest, ...]
    outputs: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Residual:
    """The advice that remains (the module's docstring)."""
    rows: tuple[tuple[int, ...], ...]
    prompt_len: tuple[int, ...]
    hits: dict[int, int] = field(default_factory=dict)
    resumes: dict[tuple[int, int], int] | None = None
    finish_eos: tuple[bool, ...] | None = None
    ns: tuple[int, ...] = ()
    release: tuple[int, ...] | None = None


# --- traces ----------------------------------------------------------------------------------------------------------

def load_trace(path: str | Path) -> dict:
    """A step trace (`sched_trace`'s format, `enc-derived/sched-trace/v1`, optionally gzipped): `config` (`Config`'s
    fields), `requests` in certified-ingress order (`id`, `prompt_ids`, `max_tokens`, `output_ids`, `finish`), and `steps`
    (`t_end_s`, `rows`: `request`, `start`, `n`, `samples`, `new`, `resumed`)."""
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return json.loads(raw)


def config_of(trace: dict) -> Config:
    c = trace["config"]
    return Config(block_size=c["block_size"], num_gpu_blocks=c["num_gpu_blocks"], max_num_batched_tokens=c["max_num_batched_tokens"],
                  max_model_len=c["max_model_len"], eos_token_id=c["eos_token_id"], stop_token_ids=tuple(c["stop_token_ids"]),
                  scheduler=c.get("scheduler", "sync"), long_prefill_token_threshold=c.get("long_prefill_token_threshold", 0),
                  max_num_seqs=c.get("max_num_seqs", 256), enable_prefix_caching=c.get("enable_prefix_caching", True),
                  rows_in_scheduling_order=c.get("rows_in_scheduling_order", True))


def holdings_of(trace: dict) -> Holdings:
    return Holdings(requests=tuple(CertifiedRequest(r["id"], tuple(r["prompt_ids"]), r["max_tokens"]) for r in trace["requests"]),
                    outputs=tuple(tuple(r["output_ids"]) for r in trace["requests"]))


def release_buckets(trace: dict, bucket_ms: float) -> tuple[int, ...]:
    """Unshaped: each step's output tokens leave in the bucket its step ends in."""
    return tuple(int(s["t_end_s"] * 1000.0 // bucket_ms) for s in trace["steps"])


def advice_of(trace: dict, bucket_ms: float = 100.0) -> Advice:
    """I1 as drafted, from a trace: requests in ingress order; each row's kind (prefill when it starts before the
    prompt's end), cache_hit (the first row's start, else 0) and preempted (a row resumed after preemption)."""
    requests = tuple(RequestAdvice(r["id"], len(r["prompt_ids"]), len(r["output_ids"]), r["finish"]) for r in trace["requests"])
    seen: set[int] = set()
    steps = []
    for s in trace["steps"]:
        rows = []
        for r in s["rows"]:
            i = r["request"]
            first = i not in seen
            seen.add(i)
            rows.append(Row(i, "prefill" if r["start"] < requests[i].prompt_len else "decode", r["start"], r["n"],
                            r["start"] if first else 0, bool(r["resumed"])))
        steps.append(StepAdvice(tuple(rows)))
    return Advice(requests, tuple(steps), release_buckets(trace, bucket_ms))


# --- the derivations -------------------------------------------------------------------------------------------------

def pool_bound(rows: Sequence[Sequence[tuple[int, int, int]]], cfg: Config) -> bool:
    """True when no allocation can fail, so nothing is preempted: at every step the blocks of every request between its
    first and last row, each counted at its furthest end so far as if no block were shared, fit the pool less vLLM's
    null block. `rows`: each step's (request, start, n)."""
    last: dict[int, int] = {}
    for s, step in enumerate(rows):
        for i, _, _ in step:
            last[i] = s
    held: dict[int, int] = {}
    total = 0
    for s, step in enumerate(rows):
        for i, start, n in step:
            blocks = -(-(start + n) // cfg.block_size)
            if blocks > held.get(i, 0):
                total += blocks - held.get(i, 0)
                held[i] = blocks
        if total > cfg.num_gpu_blocks - 1:
            return False
        for i, _, _ in step:
            if last[i] == s:
                total -= held.pop(i)
    return True


@dataclass(frozen=True)
class Walk:
    """What the walk gives back in the clear: every row's I1 fields, each request's output length, and the step of each
    explicit n it read."""
    rows: tuple[tuple[Row, ...], ...]
    output_len: tuple[int, ...]
    explicit_n: tuple[int, ...]


def _walk(rows: Sequence[Sequence[int]], prompt_len: Sequence[int], hits: dict[int, int],
          resumes: dict[tuple[int, int], int] | None, finish_eos: Sequence[bool] | None, cfg: Config,
          n_of: Callable[[int, int], int]) -> Walk:
    """The one derivation (the module's docstring). `n_of(s, k)` gives n where the walk can't derive it."""
    resumed_at = resumes or {}
    end: dict[int, int] = {}
    samples: dict[int, int] = {}
    built, explicit = [], []
    for s, step in enumerate(rows):
        used, out = 0, []
        for k, i in enumerate(step):
            P = prompt_len[i]
            first = i not in end
            resumed = (s, k) in resumed_at
            start = hits.get(i, 0) * cfg.block_size if first else resumed_at[(s, k)] if resumed else end[i]
            K = P + samples.get(i, 0)
            if K - start <= 0:
                raise ValueError(f"step {s} row {k}: request {i} has nothing left to compute")
            if K - start == 1:
                n = 1
            elif cfg.rows_in_scheduling_order:
                n = min(K - start, cfg.max_num_batched_tokens - used)
                if cfg.long_prefill_token_threshold > 0:
                    n = min(n, cfg.long_prefill_token_threshold)
            else:
                n = n_of(s, k)
                explicit.append(s)
            if not 0 < n <= cfg.max_num_batched_tokens - used:
                raise ValueError(f"step {s} row {k}: n = {n} with {cfg.max_num_batched_tokens - used} of the budget left")
            used += n
            if start + n >= K:
                samples[i] = samples.get(i, 0) + 1
            end[i] = start + n
            out.append(Row(i, "prefill" if start < P else "decode", start, n, start if first else 0, resumed))
        built.append(tuple(out))
    output_len = []
    for i in range(len(prompt_len)):
        if i not in samples:
            raise ValueError(f"request {i} never sampled")
        output_len.append(samples[i] - (1 if finish_eos is not None and finish_eos[i] else 0))
    if resumes is None and not pool_bound([[(r.request, r.start, r.n) for r in step] for step in built], cfg):
        raise ValueError("no resumes sent, but the rows don't keep within the KV pool")
    return Walk(tuple(built), tuple(output_len), tuple(explicit))


def residual(advice: Advice, cfg: Config, *, shaped: bool = False) -> Residual:
    """The advice that remains of `advice`: what `reconstruct` needs beside the config to give every public field back."""
    rows = tuple(tuple(r.request for r in st.rows) for st in advice.steps)
    if any(r.cache_hit % cfg.block_size for st in advice.steps for r in st.rows):
        raise ValueError("a cache hit that isn't whole blocks")
    prompt_len = tuple(r.prompt_len for r in advice.requests)
    hits = {r.request: r.cache_hit // cfg.block_size for st in advice.steps for r in st.rows if r.cache_hit}
    resumes = None if pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in advice.steps], cfg) else {
        (s, k): r.start for s, st in enumerate(advice.steps) for k, r in enumerate(st.rows) if r.preempted}
    finish_eos = tuple(a.finish == "eos" for a in advice.requests) if cfg.scheduler == "async" else None
    ns: list[int] = []

    def n_of(s: int, k: int) -> int:
        ns.append(advice.steps[s].rows[k].n)
        return ns[-1]

    _walk(rows, prompt_len, hits, resumes, finish_eos, cfg, n_of)
    return Residual(rows, prompt_len, hits, resumes, finish_eos, tuple(ns), None if shaped else tuple(advice.release))


def reconstruct(res: Residual, cfg: Config) -> Walk:
    """The public fields from the residual and the config alone: what the compiler and N2-ADVICE read."""
    ns = iter(res.ns)

    def n_of(s: int, k: int) -> int:
        n = next(ns, None)
        if n is None:
            raise ValueError("residual: fewer explicit n than rows that need one")
        return n

    walk = _walk(res.rows, res.prompt_len, res.hits, res.resumes, res.finish_eos, cfg, n_of)
    if next(ns, None) is not None:
        raise ValueError("residual: more explicit n than rows that need one")
    return walk


def stop_rule(output: Sequence[int], prompt_len: int, max_tokens: int, cfg: Config) -> str | None:
    """The tie's check that forces output_len and gives finish, in vLLM d9105ea8's `check_stop` order: a stop id first,
    then the length cap (max_tokens, or the model's length). None when the committed ids can't be an honest output."""
    stops = cfg.stops
    if not output or len(output) > max_tokens or any(t in stops for t in output[:-1]):
        return None
    if output[-1] in stops:
        return "eos"
    return "length" if len(output) == max_tokens or prompt_len + len(output) >= cfg.max_model_len else None


def tie_fields(walk: Walk, prompt_len: Sequence[int], held: Holdings, cfg: Config) -> list[RequestAdvice]:
    """The request fields the tie derives from witness, and its checks of the public lengths: id (the i-th certified
    request's), prompt_len against the certified id count, output_len against the committed outputs, finish by
    `stop_rule`."""
    out = []
    for i, (P, L) in enumerate(zip(prompt_len, walk.output_len, strict=True)):
        req, ids = held.requests[i], held.outputs[i]
        if len(req.prompt_ids) != P:
            raise ValueError(f"request {i}: prompt_len {P}, its certified prompt has {len(req.prompt_ids)} ids")
        if len(ids) != L:
            raise ValueError(f"request {i}: {L} outputs by the rows, {len(ids)} committed")
        finish = stop_rule(ids, P, req.max_tokens, cfg)
        if finish is None:
            raise ValueError(f"request {i}: its committed outputs break the stop rule")
        out.append(RequestAdvice(req.id, P, L, finish))
    return out


def rebuild(res: Residual, cfg: Config, held: Holdings, release: Sequence[int]) -> Advice:
    """The whole of v0 from the residual, the config and the tie's witness: `rebuild(residual(a), …) == a`. `release` is
    read only on a shaped egress, where N2-ADVICE derives it."""
    walk = reconstruct(res, cfg)
    requests = tie_fields(walk, res.prompt_len, held, cfg)
    if res.finish_eos is not None and [r.finish == "eos" for r in requests] != list(res.finish_eos):
        raise ValueError("residual: a finish bit disagrees with the stop rule")
    return Advice(tuple(requests), tuple(StepAdvice(rows) for rows in walk.rows),
                  tuple(res.release if res.release is not None else release))


# --- bits ------------------------------------------------------------------------------------------------------------

V0 = {"id": "utf8", "prompt_len": 16, "output_len": 16, "finish": 1, "rows_in_step": 16, "request": 16, "kind": 1,
      "start": 16, "n": 16, "cache_hit": 16, "preempted": 1, "release": 16}
"""The widths assumed for I1's fixed-width codec until ADV-SPEC's `bits` lands; `id` is its UTF-8 bytes and a length
byte."""

FORCED = ("prompt_len", "hits", "finish", "resumes")
"""The residual's fields that only witness fixes but a public consumer needs: sent, and forced once a check holds them."""


def _windows(advice: Advice, buckets: int) -> tuple[list[int], dict[int, int]]:
    """Each step's window (its release bucket's), and each request's (its first row's)."""
    wins = [b // buckets for b in advice.release]
    first: dict[int, int] = {}
    for s, st in enumerate(advice.steps):
        for r in st.rows:
            first.setdefault(r.request, wins[s])
    return wins, first


def _add(out: dict[int, dict[str, int]], w: int, f: str, b: int) -> None:
    d = out.setdefault(w, {})
    d[f] = d.get(f, 0) + b


def v0_bits(advice: Advice, buckets: int, widths: dict = V0) -> dict[int, dict[str, int]]:
    """v0's bits per certifier window, by field."""
    wins, first = _windows(advice, buckets)
    out: dict[int, dict[str, int]] = {}
    for i, r in enumerate(advice.requests):
        _add(out, first[i], "id", 8 * len(r.id.encode()) + 8)
        for f in ("prompt_len", "output_len", "finish"):
            _add(out, first[i], f, widths[f])
    for s, st in enumerate(advice.steps):
        _add(out, wins[s], "rows_in_step", widths["rows_in_step"])
        _add(out, wins[s], "release", widths["release"])
        for f in ("request", "kind", "start", "n", "cache_hit", "preempted"):
            _add(out, wins[s], f, widths[f] * len(st.rows))
    return out


def derived_bits(res: Residual, advice: Advice, cfg: Config, buckets: int, widths: dict = V0) -> dict[int, dict[str, int]]:
    """The residual's bits per window, at v0's widths for what it keeps: rows (rows_in_step and request), prompt_len,
    hits (a flag a request, a start-width block count each hit), finish (async), resumes (a count a window, then a row
    index and a start each), ns and release."""
    wins, first = _windows(advice, buckets)
    out: dict[int, dict[str, int]] = {}
    for i in range(len(advice.requests)):
        _add(out, first[i], "prompt_len", widths["prompt_len"])
        _add(out, first[i], "hits", 1 + (widths["start"] if i in res.hits else 0))
        if res.finish_eos is not None:
            _add(out, first[i], "finish", widths["finish"])
    for s, step in enumerate(res.rows):
        _add(out, wins[s], "rows_in_step", widths["rows_in_step"])
        _add(out, wins[s], "request", widths["request"] * len(step))
        if res.release is not None:
            _add(out, wins[s], "release", widths["release"])
    for s in reconstruct(res, cfg).explicit_n:
        _add(out, wins[s], "n", widths["n"])
    if res.resumes is not None:
        for w in set(wins):
            _add(out, w, "resumes", 16)
        for s, _ in res.resumes:
            _add(out, wins[s], "resumes", 32 + widths["start"])
    return out


def bits_table(trace: dict, *, buckets: int = 600, bucket_ms: float = 100.0, widths: dict = V0) -> dict:
    """Per window: v0's bits and the residual's, unshaped (release kept) and shaped (release derived), with each field's
    share, after checking that the residual rebuilds v0 exactly."""
    cfg = config_of(trace)
    adv = advice_of(trace, bucket_ms)
    held = holdings_of(trace)
    res = residual(adv, cfg)
    if rebuild(res, cfg, held, adv.release) != adv:
        raise AssertionError("the residual doesn't rebuild v0 on this trace")
    shaped = residual(adv, cfg, shaped=True)
    v0 = v0_bits(adv, buckets, widths)
    dv = derived_bits(res, adv, cfg, buckets, widths)
    ds = derived_bits(shaped, adv, cfg, buckets, widths)
    wins, first = _windows(adv, buckets)
    rows = []
    for w in sorted(v0):
        steps = [s for s, x in enumerate(wins) if x == w]
        rows.append({"window": w, "steps": len(steps), "rows": sum(len(adv.steps[s].rows) for s in steps),
                     "requests": sum(1 for x in first.values() if x == w),
                     "v0_bits": sum(v0[w].values()), "derived_bits": sum(dv.get(w, {}).values()),
                     "derived_shaped_bits": sum(ds.get(w, {}).values()),
                     "forced_bits": sum(dv.get(w, {}).get(f, 0) for f in FORCED),
                     "v0_by_field": v0[w], "derived_by_field": dv.get(w, {})})
    totals = {k: sum(r[k] for r in rows) for k in ("steps", "rows", "requests", "v0_bits", "derived_bits",
                                                   "derived_shaped_bits", "forced_bits")}
    return {"kind": "enc-derived/bits/v1", "config": asdict(cfg), "buckets": buckets, "bucket_ms": bucket_ms, "widths": widths,
            "pool_bound": res.resumes is None, "hits": len(res.hits), "explicit_n": len(res.ns),
            "preempted_rows": sum(r.preempted for st in adv.steps for r in st.rows),
            "finish": {f: sum(r.finish == f for r in adv.requests) for f in ("eos", "length")},
            "windows": rows, "totals": totals}


# --- the field table -------------------------------------------------------------------------------------------------

FIELDS = [
    {"field": "requests[i].id", "verdict": "derived", "bits": "0", "from": "certified ingress order",
     "derivation": "request i is the i-th certified request; its id is that request's `user` field", "checker": "tie",
     "cost": "0: the tie addresses request i by its ingress index, with no id to compare"},
    {"field": "prompt_len", "verdict": "stays, forced", "bits": "16 a request",
     "from": "certified request's id count (witness)", "derivation": "len(prompt ids); the Programs read it (a final prefill "
     "chunk is P - start), so it is public", "checker": "tie",
     "cost": "a 16-bit equality of the certified count with a public value: 15 ANDs a request"},
    {"field": "output_len", "verdict": "derived", "bits": "0", "from": "rows (public); forced by the committed sampled ids",
     "derivation": "the request's sampling rows (async: less 1 when finish is eos)", "checker": "compiler, N2-ADVICE; tie",
     "cost": "0 in the clear; the tie's stop rule: is_stop on each committed id (3 stop ids of 17 bits: 3*16 + 2 = 50 ANDs), "
             "L <= max_tokens and L == max_tokens (16 + 15): 50 L + 31 ANDs a request"},
    {"field": "finish", "verdict": "derived (sync); forced (async)", "bits": "0 (sync) / 1 a request (async)",
     "from": "committed sampled ids, certified max_tokens", "derivation": "eos iff the last committed id is a stop id",
     "checker": "tie", "cost": "0 beyond the stop rule"},
    {"field": "start", "verdict": "derived", "bits": "0", "from": "rows' history",
     "derivation": "the request's previous row's end; its first row's is its cache hit, a resumed row's its re-hit",
     "checker": "compiler", "cost": "0"},
    {"field": "n", "verdict": "derived", "bits": "0 (16 a non-decode row when rows aren't in scheduling order)",
     "from": "rows' history, prompt_len, max_num_batched_tokens",
     "derivation": "min(K - start, the step's budget left, the long-prefill threshold), K = prompt_len + sampling rows so far",
     "checker": "compiler", "cost": "0"},
    {"field": "kind", "verdict": "derived", "bits": "0", "from": "start, prompt_len", "derivation": "prefill iff start < prompt_len",
     "checker": "compiler", "cost": "0"},
    {"field": "cache_hit", "verdict": "stays, forced (not checked yet), sparse", "bits": "1 a request + 16 a hit",
     "from": "every earlier block's token ids (witness), the schedule, the pool's size, LRU eviction",
     "derivation": "the longest resident block prefix, at most num_tokens - 1 rounded down to a block",
     "checker": "none yet: forcing it needs the block pool replayed over witness token ids",
     "cost": "unbuilt; until then a prover may report any shorter hit"},
    {"field": "preempted", "verdict": "derived under the pool bound; else stays, with the re-hit", "bits": "0 / 48 a resume",
     "from": "rows, num_gpu_blocks", "derivation": "pool_bound: no allocation can fail, so nothing is preempted",
     "checker": "verifier", "cost": "0 (in the clear)"},
    {"field": "release", "verdict": "stays (unshaped); derived (shaped)", "bits": "16 a step / 0",
     "from": "unshaped: the egress head (witness), whose choices are the timing channel; shaped: the ingress anchors, the "
             "output lengths and the declared sync", "derivation": "shaped: served.derive with per-request output lengths",
     "checker": "N2-ADVICE", "cost": "shaped: N2-ADVICE's derive circuit (served.size: 678,795 ANDs at T = 600, FIXLEN)"},
]


def render(table: dict) -> str:
    lines = ["| window | steps | rows | requests | v0 bits | derived | derived, shaped | of which forced | derived / v0 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table["windows"] + [dict(table["totals"], window="all")]:
        lines.append(f"| {r['window']} | {r['steps']} | {r['rows']} | {r['requests']} | {r['v0_bits']:,} | {r['derived_bits']:,} | "
                     f"{r['derived_shaped_bits']:,} | {r['forced_bits']:,} | {r['derived_bits'] / max(r['v0_bits'], 1):.3f} |")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="bits per window, v0 against derived")
    t.add_argument("--trace", required=True, type=Path, nargs="+")
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--buckets", type=int, default=600)
    t.add_argument("--bucket-ms", type=float, default=100.0)
    a = ap.parse_args(list(argv) if argv is not None else None)
    tables = {}
    for p in a.trace:
        tables[p.name] = bits_table(load_trace(p), buckets=a.buckets, bucket_ms=a.bucket_ms)
        print(f"## {p.name}\n\n{render(tables[p.name])}\n")
    a.out.write_text(json.dumps({"kind": "enc-derived/tables/v1", "fields": FIELDS, "traces": tables}, indent=1))


if __name__ == "__main__":
    main()
