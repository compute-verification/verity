"""ENC-DERIVED: the fields of advice v0 (I1, `verity.core.protocols.certifier.advice`) that cost zero bits, because the
certified I/O, the committed values, the registered config or the advice's own history already fix them, and the advice
that remains.

The advice is public and its length is the leak (`A_c`). A field leaves the string only when every public consumer of it
(the compiler; N2-ADVICE; the verifier's `bits`) can recompute it from public data, or when no public consumer needs it
and the unit that does (the tie, IO-TIE) reads it from witness. A field a public consumer needs but only witness fixes
stays in the string: it is *forced* (one accepted value given the I/O, once a check holds it there), so a bound that
charged the advice's freedom rather than its length would charge it 0, but it is sent.

**What each checker holds.** In the clear: the residual below and the registered config (`Config`: block size, the KV
pool's size, the token budget, the stop ids, the scheduler mode). As witness, inside proofs only: the certified ingress
(each request's `user` id, prompt ids and `max_tokens`, in ingress order), the certified egress and the committed values
(each kept row's sampled id).

**The derivations** are one walk over the steps (`_walk`), the same for the encoder (`residual`) and the verifier
(`reconstruct`), so they can't drift apart. A request's known tokens K are its prompt length plus its sampling rows so
far. A row that admits its request (its first, or one that resumes it after a preemption) starts at its prefix-cache hit;
any other starts where the request's previous row ended. It schedules n = min(K − start, the step's budget left, the
long-prefill threshold) tokens (vLLM d9105ea8's running and waiting loops), and samples when it reaches K (the GPU runner's
discard rule). `kind` is I1's: decode exactly when n = 1 at or past the prompt. `FIELDS` is the table of every I1 field:
its verdict, derivation, checker and cost.

**The residual** (`Residual`): each step's requests in scheduling order (the scheduler's choice, ENC-REPLAY's), and
- `prompt_len`, one a request: the compiled Programs' shapes read it (a request's final prefill chunk is P − start), so
  it is sent, forced by the certified id count;
- `hits` and `sources`: each admitting row's prefix-cache hit, in blocks, and the requests that computed its KV, forced
  by the prompts' history and vLLM's block pool but not checked yet;
- `resumes`: which rows after their request's first resume it, by position among those rows, unless the pool bound rules
  preemption out, which the verifier checks;
- `lag`: async mode only, a bit for each request whose last two sampling rows are in consecutive steps: whether the last
  one ran after the stop (the async scheduler schedules a request's next decode before it reads the sample, and then
  drops that decode's), which `output_len` subtracts; forced by the committed ids (a lag follows a stop id);
- `order`: each request's certified ingress index, only when it isn't the request's own (I1 numbers requests by first
  row, and vLLM's FCFS admits them in arrival order, so the two differ only if the engine got them out of ingress order);
- `ns`: n where the walk can't derive it: only when the rows aren't in scheduling order (the budget left is unknown);
- `release`: each kept output's egress bucket on an unshaped egress, where its choices are the timing channel; none on a
  shaped one, where `served.derive` fixes them from the ingress anchors, the output lengths and the declared sync.

`bits_table` measures v0 (I1's own `bits_by_window`) and the residual per certifier window, the residual at its own least
widths and charged by I1's rule (`charge`), after checking that `rebuild(residual(a)) == a`.

    python derived.py table --trace TRACE.json[.gz] ... --out TABLE.json [--T 600 --bucket-ms 100]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from verity.core.protocols.certifier import advice as I1
from verity.core.protocols.certifier.advice import Advice, RequestAdvice, Row, Source, StepAdvice


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
    """Witness, inside the tie only: the certified requests in ingress order, and each one's kept sampled ids as the
    committed values hold them (equal, by the tie, to its certified egress frames' ids)."""
    requests: tuple[CertifiedRequest, ...]
    outputs: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Residual:
    """The advice that remains (the module's docstring). Requests are I1's, numbered by first row."""
    rows: tuple[tuple[int, ...], ...]
    prompt_len: tuple[int, ...]
    hits: tuple[int, ...]                               # each admitting row's, in blocks
    sources: tuple[tuple[Source, ...], ...] = ()        # each nonzero hit's
    resumes: tuple[int, ...] | None = None              # positions among the rows after their request's first; None: the pool bound
    lag: tuple[bool, ...] | None = None
    order: tuple[int, ...] | None = None                # None: request i is the i-th certified one
    ns: tuple[int, ...] = ()
    release: tuple[int, ...] | None = None


# --- traces ----------------------------------------------------------------------------------------------------------

def load_trace(path: str | Path) -> dict:
    """A step trace (`sched_trace`'s format, `enc-derived/sched-trace/v2`, optionally gzipped): `config` (`Config`'s
    fields), `requests` in certified-ingress order (`id`, `prompt_ids`, `max_tokens`, `output_ids`, `finish`), and `steps`
    (`t_end_s`, `rows` in scheduling order: `request` (its ingress index), `start`, `n`, `samples`, `new`, `resumed`, and
    on a row that admits its request with a hit, `sources`: [ingress index, tokens] runs)."""
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


def advice_of(trace: dict, bucket_ms: float = 100.0) -> tuple[Advice, tuple[int, ...]]:
    """I1 from a trace, and each I1 request's ingress index. A row admits its request when it is its first or resumes it;
    such a row's `cache_hit` is its start. Each kept output is released in the bucket its step ends in (unshaped)."""
    order: list[int] = []
    num: dict[int, int] = {}
    for s in trace["steps"]:
        for r in s["rows"]:
            if r["request"] not in num:
                num[r["request"]] = len(order)
                order.append(r["request"])
    reqs = trace["requests"]
    requests = tuple(RequestAdvice(reqs[t]["id"], len(reqs[t]["prompt_ids"]), len(reqs[t]["output_ids"]), reqs[t]["finish"])
                     for t in order)
    seen: set[int] = set()
    kept = [0] * len(order)
    steps, release = [], []
    for s in trace["steps"]:
        rows = []
        bucket = int(s["t_end_s"] * 1000.0 // bucket_ms)
        for r in s["rows"]:
            i = num[r["request"]]
            admits = i not in seen or r["resumed"]
            seen.add(i)
            rows.append(Row(i, "decode" if r["n"] == 1 and r["start"] >= requests[i].prompt_len else "prefill", r["start"], r["n"],
                            r["start"] if admits else 0, bool(r["resumed"]),
                            tuple(Source(num[a], n) for a, n in r.get("sources", ()))))
            if r["samples"] and kept[i] < requests[i].output_len:
                kept[i] += 1
                release.append(bucket)
        steps.append(StepAdvice(tuple(rows)))
    return Advice(requests, tuple(steps), tuple(release)), tuple(order)


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
    """What the walk gives back in the clear: I1's steps, each request's output length, the step of each explicit n it
    read, and the requests it read a lag bit for."""
    steps: tuple[StepAdvice, ...]
    output_len: tuple[int, ...]
    explicit_n: tuple[int, ...]
    lag_requests: tuple[int, ...]


def _walk(rows: Sequence[Sequence[int]], prompt_len: Sequence[int], cfg: Config, *, pool: bool,
          hit_of: Callable[[int, int], int], sources_of: Callable[[int, int], tuple[Source, ...]],
          resumed_of: Callable[[int, int], bool], n_of: Callable[[int, int], int], lag_of: Callable[[int], bool]) -> Walk:
    """The one derivation (the module's docstring). The callbacks give, for step s's row k, what the residual holds:
    `hit_of` an admitting row's hit in blocks, `sources_of` a nonzero hit's sources, `resumed_of` whether a row after its
    request's first resumes it (read unless `pool`, which instead checks the pool bound), `n_of` n where the walk can't
    derive it; `lag_of(i)` whether request i's last sampling row ran after its stop, asked in async mode only of a request
    whose last two sampling rows are in consecutive steps (only there can the scheduler have run a decode past a stop)."""
    end: dict[int, int] = {}
    samples: dict[int, int] = {}
    sample_steps: dict[int, list[int]] = {}
    steps, explicit = [], []
    for s, step in enumerate(rows):
        used, out = 0, []
        for k, i in enumerate(step):
            if not 0 <= i < len(prompt_len):
                raise ValueError(f"step {s} row {k}: request {i} of {len(prompt_len)}")
            first = i not in end
            if first and i != len(end):
                raise ValueError(f"step {s} row {k}: request {i} before request {len(end)}: requests are numbered by first row")
            resumed = not first and not pool and resumed_of(s, k)
            P = prompt_len[i]
            if first or resumed:
                hit = hit_of(s, k) * cfg.block_size
                sources, start = (sources_of(s, k) if hit else ()), hit
            else:
                hit, sources, start = 0, (), end[i]
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
                sample_steps.setdefault(i, []).append(s)
            end[i] = start + n
            out.append(Row(i, "decode" if n == 1 and start >= P else "prefill", start, n, hit, resumed, sources))
        steps.append(StepAdvice(tuple(out)))
    output_len, lagged = [], []
    for i in range(len(prompt_len)):
        if i not in samples:
            raise ValueError(f"request {i} never sampled")
        lag = False
        if cfg.scheduler == "async" and len(sample_steps[i]) >= 2 and sample_steps[i][-1] == sample_steps[i][-2] + 1:
            lag = lag_of(i)
            lagged.append(i)
        output_len.append(samples[i] - lag)
    if pool and not pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in steps], cfg):
        raise ValueError("no preemption sent, but the rows don't keep within the KV pool")
    return Walk(tuple(steps), tuple(output_len), tuple(explicit), tuple(lagged))


def _sampling_rows(advice: Advice, i: int) -> int:
    P, k = advice.requests[i].prompt_len, 0
    for st in advice.steps:
        for r in st.rows:
            if r.request == i and r.start + r.n >= P + k:
                k += 1
    return k


def residual(advice: Advice, cfg: Config, *, order: Sequence[int] | None = None, shaped: bool = False) -> Residual:
    """The advice that remains of `advice`, whose requests are certified requests `order` (None: in ingress order): what
    `reconstruct` needs beside the config to give every public field back."""
    pool = pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in advice.steps], cfg)
    if pool and any(r.preempted for st in advice.steps for r in st.rows):
        raise ValueError("a resumed row where the pool bound rules preemption out")
    hits: list[int] = []
    sources: list[tuple[Source, ...]] = []
    resumes: list[int] = []
    later = [0]
    ns: list[int] = []
    lags: list[bool] = []

    def hit_of(s: int, k: int) -> int:
        h = advice.steps[s].rows[k].cache_hit
        if h % cfg.block_size:
            raise ValueError(f"step {s} row {k}: a cache hit of {h} isn't whole blocks")
        hits.append(h // cfg.block_size)
        return hits[-1]

    def sources_of(s: int, k: int) -> tuple[Source, ...]:
        sources.append(advice.steps[s].rows[k].sources)
        return sources[-1]

    def resumed_of(s: int, k: int) -> bool:
        later[0] += 1
        if advice.steps[s].rows[k].preempted:
            resumes.append(later[0] - 1)
        return advice.steps[s].rows[k].preempted

    def n_of(s: int, k: int) -> int:
        ns.append(advice.steps[s].rows[k].n)
        return ns[-1]

    def lag_of(i: int) -> bool:
        lags.append(_sampling_rows(advice, i) - advice.requests[i].output_len == 1)
        return lags[-1]

    rows = tuple(tuple(r.request for r in st.rows) for st in advice.steps)
    prompt_len = tuple(q.prompt_len for q in advice.requests)
    _walk(rows, prompt_len, cfg, pool=pool, hit_of=hit_of, sources_of=sources_of, resumed_of=resumed_of, n_of=n_of,
          lag_of=lag_of)
    own = order is None or tuple(order) == tuple(range(len(advice.requests)))
    return Residual(rows, prompt_len, tuple(hits), tuple(sources), None if pool else tuple(resumes),
                    tuple(lags) if cfg.scheduler == "async" else None, None if own else tuple(order), tuple(ns),
                    None if shaped else tuple(advice.release))


def reconstruct(res: Residual, cfg: Config) -> Walk:
    """The public fields from the residual and the config alone: what the compiler and N2-ADVICE read."""
    held = {"hit": iter(res.hits), "sources": iter(res.sources), "n": iter(res.ns), "lag": iter(res.lag or ())}
    resumes = iter(res.resumes or ())
    later, resume = [0], [next(resumes, None)]

    def take(name: str):
        value = next(held[name], None)
        if value is None:
            raise ValueError(f"residual: fewer {name} values than the rows read")
        return value

    def resumed_of(s: int, k: int) -> bool:
        later[0] += 1
        if resume[0] != later[0] - 1:
            return False
        resume[0] = next(resumes, None)
        return True

    walk = _walk(res.rows, res.prompt_len, cfg, pool=res.resumes is None, hit_of=lambda s, k: take("hit"),
                 sources_of=lambda s, k: take("sources"), resumed_of=resumed_of, n_of=lambda s, k: take("n"),
                 lag_of=lambda i: take("lag"))
    if resume[0] is not None or any(next(it, None) is not None for it in held.values()):
        raise ValueError("residual: more values than the rows read, or resumes out of order")
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


def tie_fields(walk: Walk, res: Residual, held: Holdings, cfg: Config) -> list[RequestAdvice]:
    """The request fields the tie derives from witness, and its checks of the public lengths: request i is certified
    request `order[i]` (i itself unless sent), whose `user` is its id; prompt_len against the certified id count;
    output_len against the committed outputs; finish by `stop_rule`."""
    R = len(res.prompt_len)
    order = res.order if res.order is not None else tuple(range(R))
    if len(held.requests) != R or sorted(order) != list(range(R)):
        raise ValueError(f"the advice's {R} requests aren't the {len(held.requests)} certified ones, each once")
    out = []
    for i, (P, L) in enumerate(zip(res.prompt_len, walk.output_len, strict=True)):
        req, ids = held.requests[order[i]], held.outputs[order[i]]
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
    requests = tie_fields(walk, res, held, cfg)
    for i, lag in zip(walk.lag_requests, res.lag or (), strict=True):
        if lag and requests[i].finish != "eos":
            raise ValueError(f"request {i}: a decode after its last output, which no stop id ended")
    return Advice(tuple(requests), walk.steps, tuple(res.release if res.release is not None else release))


# --- bits ------------------------------------------------------------------------------------------------------------

class Records(NamedTuple):
    """An encoding's bits by record, as I1 charges them, each by field: the header, each request's record, each step's,
    and each kept output's release."""
    header: dict[str, int]
    requests: list[dict[str, int]]
    steps: list[dict[str, int]]
    release: int


def _uvar(value: int) -> int:
    return I1.WIDTH_BITS + value.bit_length()


def v0_records(adv: Advice) -> Records:
    """I1's layout (its module's docstring), by field; `charge` of it is I1's `bits_by_window`."""
    lens = [x for q in adv.requests for x in (q.prompt_len, q.output_len)]
    lens += [x for st in adv.steps for r in st.rows for x in (r.start, r.n, r.cache_hit, *(s.n for s in r.sources))]
    w_req, w_len = max(len(adv.requests) - 1, 0).bit_length(), max(lens, default=0).bit_length()
    w_src = max((len(r.sources) - 1 for st in adv.steps for r in st.rows if r.cache_hit), default=0).bit_length()
    header = {"header": 8 + _uvar(len(adv.requests)) + _uvar(len(adv.steps)) + 3 * I1.WIDTH_BITS}
    requests = [{"id": _uvar(len(q.id.encode())) + 8 * len(q.id.encode()), "prompt_len": w_len, "output_len": w_len, "finish": 1}
                for q in adv.requests]
    steps = [{"rows_in_step": w_req, "request": w_req * len(st.rows), "kind": len(st.rows), "start": w_len * len(st.rows),
              "n": w_len * len(st.rows), "cache_hit": w_len * len(st.rows), "preempted": len(st.rows),
              "sources": sum(w_src + len(r.sources) * (w_req + w_len) for r in st.rows if r.cache_hit)} for st in adv.steps]
    return Records(header, requests, steps, max(adv.release, default=0).bit_length())


def derived_records(res: Residual, walk: Walk, cfg: Config) -> Records:
    """The residual's layout, by field, each at its least width as I1 does: the header (I1's, a width for hits in blocks,
    a flag each for order, resumes and release being sent, and when resumes are, their width and count); a request's
    prompt_len, its order (when sent) and its lag bit (when asked); a step's row count and requests, each resumed row's
    position (when sent), a hit flag for each admitting row with the hit's blocks when nonzero and then its sources (I1's:
    a count, and each a request and its blocks), and an explicit n where read; a kept output's release (when sent)."""
    bs = cfg.block_size
    w_req = max(len(res.prompt_len) - 1, 0).bit_length()
    w_len = max((*res.prompt_len, *res.ns), default=0).bit_length()
    w_hit = max((*res.hits, *(s.n // bs for src in res.sources for s in src)), default=0).bit_length()
    w_src = max((len(src) - 1 for src in res.sources), default=0).bit_length()
    w_res = max(res.resumes or (0,)).bit_length()
    header = {"header": 8 + _uvar(len(res.prompt_len)) + _uvar(len(res.rows)) + 4 * I1.WIDTH_BITS + 3}
    if res.resumes is not None:
        header["resumes"] = I1.WIDTH_BITS + _uvar(len(res.resumes))
    lagged = set(walk.lag_requests)
    requests = [{"prompt_len": w_len, "order": w_req if res.order is not None else 0, "lag": int(i in lagged)}
                for i in range(len(res.prompt_len))]
    explicit: dict[int, int] = {}
    for s in walk.explicit_n:
        explicit[s] = explicit.get(s, 0) + 1
    seen: set[int] = set()
    steps = []
    for s, st in enumerate(walk.steps):
        d = {"rows_in_step": w_req, "request": w_req * len(st.rows), "resumes": 0, "hits": 0, "sources": 0,
             "n": w_len * explicit.get(s, 0)}
        for r in st.rows:
            first = r.request not in seen
            seen.add(r.request)
            if r.preempted:
                d["resumes"] += w_res
            if first or r.preempted:
                d["hits"] += 1 + (w_hit if r.cache_hit else 0)
                if r.cache_hit:
                    if any(x.n % bs for x in r.sources):
                        raise ValueError(f"step {s}: a source that isn't whole blocks")
                    d["sources"] += w_src + len(r.sources) * (w_req + w_hit)
        steps.append(d)
    return Records(header, requests, steps,
                   max(res.release, default=0).bit_length() if res.release is not None else 0)


def _earliest(adv: Advice, T: int) -> tuple[list[int], list[int]]:
    """I1's charge windows: each step's (the earliest window of an output at that step or later; the last window when
    none follows), and each request's first step."""
    windows = [b // T for b in adv.release]
    own: list[int | None] = [None] * len(adv.steps)
    for o, w in zip(I1.outputs(adv), windows):
        own[o.step] = w if own[o.step] is None else min(own[o.step], w)
    earliest, at = [0] * len(adv.steps), None
    for s in range(len(adv.steps) - 1, -1, -1):
        if own[s] is not None:
            at = own[s] if at is None else min(at, own[s])
        earliest[s] = max(windows, default=0) if at is None else at
    first: dict[int, int] = {}
    for s, st in enumerate(adv.steps):
        for r in st.rows:
            first.setdefault(r.request, s)
    return earliest, [first[i] for i in range(len(adv.requests))]


def charge(adv: Advice, T: int, rec: Records) -> dict[int, dict[str, int]]:
    """`rec`'s bits per egress window of T buckets, by field, by I1's rule (`bits_by_window`): an output's release to its
    own window, a step's record to the earliest window of an output at that step or later, a request's to its first
    step's, the header to step 0's."""
    out: dict[int, dict[str, int]] = {}

    def add(w: int, d: dict[str, int]) -> None:
        for f, b in d.items():
            if b:
                out.setdefault(w, {})[f] = out.get(w, {}).get(f, 0) + b

    if not adv.release:
        for d in (rec.header, *rec.requests, *rec.steps):
            add(0, d)
        return out
    earliest, first = _earliest(adv, T)
    add(earliest[0], rec.header)
    for i, d in enumerate(rec.requests):
        add(earliest[first[i]], d)
    for s, d in enumerate(rec.steps):
        add(earliest[s], d)
    for b in adv.release:
        add(b // T, {"release": rec.release})
    return out


FORCED = ("prompt_len", "hits", "sources", "lag", "resumes", "order")
"""The residual's fields that only witness fixes but a public consumer needs: sent, and forced once a check holds them."""


def bits_table(trace: dict, *, T: int = 600, bucket_ms: float = 100.0) -> dict:
    """Per window: v0's bits (I1's `bits_by_window`) and the residual's, unshaped (release kept) and shaped (release
    derived), by field, after checking that the residual rebuilds v0 exactly."""
    cfg = config_of(trace)
    adv, order = advice_of(trace, bucket_ms)
    held = holdings_of(trace)
    res = residual(adv, cfg, order=order)
    shaped = residual(adv, cfg, order=order, shaped=True)
    if rebuild(res, cfg, held, ()) != adv or rebuild(shaped, cfg, held, adv.release) != adv:
        raise AssertionError("the residual doesn't rebuild v0 on this trace")
    v0 = charge(adv, T, v0_records(adv))
    if {w: sum(d.values()) for w, d in v0.items()} != I1.bits_by_window(adv, T):
        raise AssertionError("v0_records doesn't charge as I1's bits_by_window")
    walk = reconstruct(res, cfg)
    dv, ds = charge(adv, T, derived_records(res, walk, cfg)), charge(adv, T, derived_records(shaped, walk, cfg))
    earliest, first = _earliest(adv, T)
    windows = []
    for w in sorted(v0):
        steps = [s for s, x in enumerate(earliest) if x == w]
        windows.append({"window": w, "steps": len(steps), "rows": sum(len(adv.steps[s].rows) for s in steps),
                        "requests": sum(1 for f in first if earliest[f] == w),
                        "outputs": sum(1 for b in adv.release if b // T == w),
                        "v0_bits": sum(v0[w].values()), "derived_bits": sum(dv.get(w, {}).values()),
                        "derived_shaped_bits": sum(ds.get(w, {}).values()),
                        "forced_bits": sum(dv.get(w, {}).get(f, 0) for f in FORCED),
                        "v0_by_field": v0[w], "derived_by_field": dv.get(w, {})})
    totals = {k: sum(r[k] for r in windows) for k in ("steps", "rows", "requests", "outputs", "v0_bits", "derived_bits",
                                                      "derived_shaped_bits", "forced_bits")}
    by_field: dict[str, dict[str, int]] = {"v0": {}, "derived": {}}
    for r in windows:
        for side, key in (("v0", "v0_by_field"), ("derived", "derived_by_field")):
            for f, b in r[key].items():
                by_field[side][f] = by_field[side].get(f, 0) + b
    return {"kind": "enc-derived/bits/v2", "config": asdict(cfg), "T": T, "bucket_ms": bucket_ms,
            "pool_bound": res.resumes is None, "order_sent": res.order is not None, "explicit_n": len(res.ns),
            "admitting_rows": len(res.hits), "hits": sum(h > 0 for h in res.hits), "sources": sum(len(s) for s in res.sources),
            "resumed_rows": sum(r.preempted for st in adv.steps for r in st.rows),
            "lag_asked": len(walk.lag_requests), "lag_set": sum(res.lag or ()),
            "finish": {f: sum(q.finish == f for q in adv.requests) for f in ("eos", "length")},
            "windows": windows, "totals": totals, "by_field": by_field}


# --- the field table -------------------------------------------------------------------------------------------------

FIELDS = [
    {"field": "requests[i].id", "verdict": "derived", "bits": "0 (w_req a request when order is sent)",
     "from": "certified ingress order", "derivation": "request i is certified request order[i], i itself unless order is "
     "sent; its id is that request's `user`", "checker": "tie",
     "cost": "0: the tie addresses the certified request by index, with no id to compare"},
    {"field": "order (new)", "verdict": "sent only when not the identity", "bits": "0, else w_req a request",
     "from": "the engine's admission order against the certified ingress order",
     "derivation": "FCFS admits in arrival order, so request i (by first row) is the i-th certified request unless the "
     "engine received them out of ingress order", "checker": "tie (each certified request used once)", "cost": "0"},
    {"field": "prompt_len", "verdict": "stays, forced", "bits": "w_len a request",
     "from": "certified request's id count (witness)", "derivation": "len(prompt ids); the Programs read it (a final prefill "
     "chunk is P - start), so it is public", "checker": "tie",
     "cost": "a 16-bit equality of the certified count with a public value: 15 ANDs a request"},
    {"field": "output_len", "verdict": "derived", "bits": "0", "from": "rows (public); forced by the committed sampled ids",
     "derivation": "the request's sampling rows (async: less the lag bit)", "checker": "compiler, N2-ADVICE; tie",
     "cost": "0 in the clear; the tie's stop rule: is_stop on each committed id (3 stop ids of 17 bits: 3*16 + 2 = 50 ANDs), "
             "L <= max_tokens and L == max_tokens (16 + 15): 50 L + 31 ANDs a request"},
    {"field": "finish", "verdict": "derived", "bits": "0", "from": "committed sampled ids, certified max_tokens",
     "derivation": "eos iff the last committed id is a stop id, else length (max_tokens or the model's length)",
     "checker": "tie", "cost": "0 beyond the stop rule"},
    {"field": "lag (new, async only)", "verdict": "forced", "bits": "1 a request whose last two sampling rows are in consecutive steps",
     "from": "committed sampled ids", "derivation": "the last sampling row ran after the stop: its sample is dropped",
     "checker": "tie (a lag follows a stop id)", "cost": "0 beyond the stop rule"},
    {"field": "rows[k].request", "verdict": "stays", "bits": "w_req a row, and w_req a step for its row count",
     "from": "the scheduler's choices", "derivation": "none here: which requests each step runs, in scheduling order, "
     "depends on the arrival times, the timing channel (ENC-REPLAY's piece)", "checker": "compiler (the Programs' shapes)",
     "cost": "0"},
    {"field": "start", "verdict": "derived", "bits": "0", "from": "rows' history, cache_hit",
     "derivation": "the request's previous row's end; an admitting row's (its first, or a resumed one) is its hit",
     "checker": "compiler", "cost": "0"},
    {"field": "n", "verdict": "derived", "bits": "0 (w_len a non-decode row when rows aren't in scheduling order)",
     "from": "rows' history, prompt_len, max_num_batched_tokens",
     "derivation": "min(K - start, the step's budget left, the long-prefill threshold), K = prompt_len + sampling rows so far",
     "checker": "compiler", "cost": "0"},
    {"field": "kind", "verdict": "derived", "bits": "0", "from": "start, n, prompt_len",
     "derivation": "decode iff n = 1 and start >= prompt_len (I1's rule)", "checker": "compiler", "cost": "0"},
    {"field": "cache_hit", "verdict": "stays on admitting rows only, forced (not checked yet)",
     "bits": "1 an admitting row + w_hit (blocks) a hit; 0 on every other row (I1: w_len every row)",
     "from": "every earlier block's token ids (witness), the schedule, the pool's size, LRU eviction",
     "derivation": "the longest resident block prefix, at most num_tokens - 1 rounded down to a block",
     "checker": "none yet: forcing it needs vLLM's block pool replayed over witness token ids",
     "cost": "unbuilt; until then a prover may report any shorter hit"},
    {"field": "sources", "verdict": "stays, forced (not checked yet)", "bits": "I1's, with n in blocks (w_hit, not w_len)",
     "from": "which request computed each hit block: the block pool's history over witness token ids",
     "derivation": "vLLM's block for a hash is the one the earliest still-cached computation filled",
     "checker": "none yet: the KV the circuit reads must be the named request's; with equal tokens computed twice, a "
     "prover may name either", "cost": "unbuilt"},
    {"field": "preempted", "verdict": "derived under the pool bound; else sent sparse (resumes)",
     "bits": "0 / a count and w_res a resumed row (I1: 1 every row)",
     "from": "rows, num_gpu_blocks", "derivation": "pool_bound: no allocation can fail, so nothing is preempted; "
     "else which rows resume is FCFS's pop of the last running request when the pool runs out, which needs the shared "
     "blocks (witness), so it is sent", "checker": "verifier (the bound, in the clear); none yet for the sent resumes",
     "cost": "0 (in the clear)"},
    {"field": "release", "verdict": "stays (unshaped); derived (shaped)", "bits": "w_rel a kept output / 0",
     "from": "unshaped: the egress head (witness), whose choices are the timing channel; shaped: the ingress anchors, the "
             "output lengths and the declared sync", "derivation": "shaped: served.derive with per-request output lengths",
     "checker": "N2-ADVICE", "cost": "shaped: N2-ADVICE's derive circuit (served.size: 678,795 ANDs at T = 600, FIXLEN)"},
]


def render(table: dict) -> str:
    lines = ["| window | steps | rows | requests | outputs | v0 bits | derived | derived, shaped | of which forced | derived / v0 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table["windows"] + [dict(table["totals"], window="all")]:
        lines.append(f"| {r['window']} | {r['steps']} | {r['rows']} | {r['requests']} | {r['outputs']} | {r['v0_bits']:,} | "
                     f"{r['derived_bits']:,} | {r['derived_shaped_bits']:,} | {r['forced_bits']:,} | "
                     f"{r['derived_bits'] / max(r['v0_bits'], 1):.3f} |")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="bits per window, v0 against derived")
    t.add_argument("--trace", required=True, type=Path, nargs="+")
    t.add_argument("--out", required=True, type=Path)
    t.add_argument("--T", type=int, default=600, help="buckets a certifier window")
    t.add_argument("--bucket-ms", type=float, default=100.0)
    a = ap.parse_args(list(argv) if argv is not None else None)
    tables = {}
    for p in a.trace:
        tables[p.name] = bits_table(load_trace(p), T=a.T, bucket_ms=a.bucket_ms)
        print(f"## {p.name}\n\n{render(tables[p.name])}\n")
    a.out.write_text(json.dumps({"kind": "enc-derived/tables/v2", "fields": FIELDS, "traces": tables}, indent=1))


if __name__ == "__main__":
    main()
