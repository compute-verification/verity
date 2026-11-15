"""The derived rules: the fields of advice v0 (I1, `verity.protocols.resource_accounting.network.verifier.advice`) that cost zero bits,
because the certified I/O, the committed values, the registered config or the advice's own history already fix them, and
the advice that remains.

The advice is public and its length is the leak (`A_c`): the sampled leak bound (`hAs`) needs at most 2^c accepted
advice strings charged ≤ c bits, so the encoding must be injective on everything the audit accepts. A field leaves the
string only when it is *canonical and checked*: a function of the config, the rest of the string and the certified I/O
that the unit holding that I/O recomputes and holds the advice to, so for a given I/O no choice hides in it. Any other
field is sent and charged, a *forced* one (fixed by witness, but by a rule no unit checks yet) included.

**What each checker holds.** In the clear: the residual below and the registered config (`Config`: block size, the KV
pool's size, the token budget, the stop ids, the scheduler mode). As witness, inside proofs only: the certified ingress
(each request's `user` id, prompt ids and `max_tokens`, in ingress order), the certified egress and the committed values
(each kept row's sampled id). The compile slice's proof block holds every request's tokens (its prompt ids, then its kept sampled ids).

**The derivations** are one walk over the steps (`walk_steps`), the same for the encoder and the decoder, so they can't drift
apart. It takes each step's rows in vLLM d9105ea8's scheduling order: the rows' own (`row_order` "scheduling"), or, for
rows in the runner's batch order ("batch"), the order vLLM's FCFS lists give them (the running requests by admission,
then the resumed ones, then the new ones in arrival order, the certified ingress order). A request's known tokens K are
its prompt length plus its sampling rows so far. A row that admits its request (its first, or one that resumes it after
a preemption) starts at its prefix-cache hit; any other starts where the request's previous row ended. It schedules
n = min(K − start, the step's budget left, the long-prefill threshold) tokens (vLLM's running and waiting loops), and
samples when it reaches K (the GPU runner's discard rule). `kind` is I1's: decode exactly when n = 1 at or past the
prompt. Two fields come from the tokens (`Prefixes`):
- a hit's `sources` are canonical: for each hit unit, the earliest-admitted (lowest-numbered) request that has computed
  the whole unit by this step (I1's rule) from the same tokens through it. Prefixes are looked up by a chain of SHA-512
  keys over whole units (`unit_key`), and a key names a request only where its tokens match, so no collision and no
  interpreter changes what is derived;
- while nothing can have been evicted (`walk_steps`'s no-eviction bound), a `cache_hit` is the longest prefix, in hit units
  and at most K − 1 tokens, whose every unit some row scheduled before it computed (vLLM's lookup: a block is cached as it
  is scheduled, and stays cached until evicted). A shorter or longer hit is refused. Past the bound, hits are sent.
The request records come from the tie (`tie_fields`: the prompt and output lengths against the certified ids, `finish`
by the stop rule, `stop_rule`), and `pool_bound` holds rows with no preemption sent to the KV pool.

**The residual** (`Residual`): each step's requests in I1's row order (the scheduler's choice, the replay's), and
- `prompt_len`, one a request, sent, forced by the certified id count (the tie checks it);
- `hits`: each admitting row's prefix-cache hit, in hit units (`Config.unit`: the block, finer only in a hybrid model,
  whose hit may end inside a block), only from the step where the no-eviction bound fails (or in async or explicit
  mode);
- `resumes`: which rows after their request's first resume it, by position among those rows, unless the pool bound rules
  preemption out, which the verifier checks;
- `lag`: async mode only, a bit for each request whose last two sampling rows are in consecutive steps: whether the last
  one ran after the stop (the async scheduler schedules a request's next decode before it reads the sample, and then
  drops that decode's), which `output_len` subtracts; forced by the committed ids (a lag follows a stop id);
- `order`: a step's new requests' certified ingress indices, in I1's order, only where the walk's prediction misses;
- `ns`: n where the walk can't derive it: only when the scheduling order is unknown (`row_order` "explicit");
- `release`: each kept output's egress bucket on an unshaped egress, where its choices are the timing channel; none on a
  shaped one.

`residual` builds it from an advice and `reconstruct` and `rebuild` (the compile slice's proof block) give the advice back,
by the same walk. `charge` puts an encoding's bits by field (`i1_records`, I1's layout, or `derived_records`, the
residual's) in the egress windows I1's rule charges them to.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from verity.protocols.resource_accounting.network.verifier.advice import (
    Advice, outputs, RequestAdvice, Row, Source, StepAdvice, WIDTH_BITS,
)

__all__ = ["CertifiedRequest", "charge", "Config", "config_of", "derived_records", "earliest_windows", "FORCED", "Holdings",
    "i1_records", "pool_bound", "PREFIX_ROOT", "Prefixes", "rebuild", "reconstruct", "Records", "Residual", "residual",
    "stop_rule", "tie_fields", "TOKEN_BYTES", "unit_key", "Walk", "walk_steps"]


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
    row_order: str = "scheduling"                    # I1's rows: "scheduling", "batch" (the runner's) or "explicit" (n sent)
    hit_unit: int = 0                                # vLLM's hash_block_size (prefix_match_unit); 0: block_size

    @property
    def stops(self) -> frozenset[int]:
        return frozenset((self.eos_token_id, *self.stop_token_ids))

    @property
    def unit(self) -> int:
        """The token boundary a cache hit lands on: the block, except in a hybrid model whose groups hash finer, where a
        hit may end inside a block (its tail copied on write)."""
        return self.hit_unit or self.block_size


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
    hits: tuple[int, ...]                               # each admitting row's the walk can't derive, in hit units
    resumes: tuple[int, ...] | None = None              # positions among the rows after their request's first; None: the pool bound
    lag: tuple[bool, ...] | None = None
    order: tuple[tuple[int, tuple[int, ...]], ...] = ()  # (step, its new requests' ingress indices) where the walk's prediction misses
    ns: tuple[int, ...] = ()
    release: tuple[int, ...] | None = None


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


PREFIX_ROOT = bytes(64)  # the key of the empty prefix
TOKEN_BYTES = 4         # a token id's canonical bytes: unsigned, big-endian


def unit_key(parent: bytes, tokens: Sequence[int]) -> bytes:
    """The key of a prefix of whole hit units: SHA-512 over the key of the prefix before its last unit, then that unit's
    token ids, each in `TOKEN_BYTES` big-endian bytes (Verity's one hash; vLLM's prefix cache chains its block hashes the
    same way, with sha256)."""
    if not all(type(t) is int and 0 <= t < 1 << 8 * TOKEN_BYTES for t in tokens):
        raise ValueError(f"token ids are integers in [0, 2^{8 * TOKEN_BYTES})")
    return hashlib.sha512(parent + b"".join(t.to_bytes(TOKEN_BYTES, "big") for t in tokens)).digest()


def _chain(keys: list[bytes], tokens: Sequence[int], unit: int, u: int) -> bytes:
    """The key of `tokens`' first u + 1 hit units of `unit` tokens, memoized in `keys`."""
    if (u + 1) * unit > len(tokens):
        raise ValueError(f"{len(tokens)} tokens, not {u + 1} whole units of {unit}")
    while len(keys) <= u:
        keys.append(unit_key(keys[-1] if keys else PREFIX_ROOT, tokens[len(keys) * unit:(len(keys) + 1) * unit]))
    return keys[u]


def _units(lo: int, start: int, end: int, unit: int) -> range:
    """The units a run from `lo` covers whole once a row has computed `start` … `end` − 1 of it, and not before."""
    return range(max(-(-lo // unit), start // unit), end // unit)


class Prefixes:
    """Witness, in the compile slice's proof block: every request's tokens (its prompt ids, then its kept sampled ids), I1-numbered as
    the walk numbers them, and for each prefix of whole hit units whose last unit some request has computed KV at so far
    (all of it, in one run from a unit boundary), the requests that did. A prefix is keyed by `unit_key`'s chain; a key
    names a request only where that request's tokens through the prefix are the asking one's, so no collision can change
    what is derived, and the prover and the compile slice's proof block derive the same on any interpreter."""

    def __init__(self, tokens: Sequence[Sequence[int]], unit: int) -> None:
        if type(unit) is not int or unit < 1:
            raise ValueError(f"a hit unit of {unit!r} tokens")
        self.unit = unit
        self.tokens: list[tuple[int, ...]] = []
        self.keys: list[list[bytes]] = []
        self.by: dict[bytes, set[int]] = {}
        self.run: list[tuple[int, int]] = []    # each request's latest run: (from, to)
        for t in tokens:
            self.push(t)

    def push(self, tokens: Sequence[int]) -> int:
        """The next request's tokens; its number."""
        self.tokens.append(tuple(tokens))
        self.keys.append([])
        self.run.append((0, 0))
        return len(self.tokens) - 1

    def _key(self, i: int, u: int) -> bytes:
        """The key of request i's first u + 1 hit units."""
        return _chain(self.keys[i], self.tokens[i], self.unit, u)

    def _owner(self, key: bytes, tokens: Sequence[int]) -> int | None:
        return min((a for a in self.by.get(key, ()) if self.tokens[a][:len(tokens)] == tokens), default=None)

    def owner(self, i: int, u: int) -> int | None:
        """The lowest-numbered request that has computed request i's unit u from the same tokens through it, or None."""
        return self._owner(self._key(i, u), self.tokens[i][:(u + 1) * self.unit])

    def add(self, i: int, start: int, end: int) -> None:
        """Request i's row computed KV at positions start … end − 1. A row continues its request's run when it starts
        where the run ended, and starts a new one otherwise (an admission or a resume, at a unit boundary); each unit the
        run now covers whole is computed by i."""
        lo, hi = self.run[i]
        if start != hi or start == 0:
            lo = start
        self.run[i] = (lo, end)
        for u in _units(lo, start, end, self.unit):
            self.by.setdefault(self._key(i, u), set()).add(i)

    def cached(self, i: int, K: int) -> int:
        """vLLM's lookup while nothing has been evicted: the longest prefix of request i, in whole units and at most
        K − 1 tokens, each of whose units some request has computed (vLLM caches a unit's block as it schedules it)."""
        h, U = 0, self.unit
        while h + U <= K - 1 and self.owner(i, h // U) is not None:
            h += U
        return h

    def lookup(self, tokens: Sequence[int], K: int, extra: Sequence[tuple[Sequence[int], int, int]] = ()) -> int:
        """`cached` for tokens not yet numbered, with the rows in `extra` computed too: each the tokens of a request not
        yet numbered, and the positions start … end − 1 its admitting row computes."""
        tokens, keys, h, U = tuple(tokens), [], 0, self.unit
        while h + U <= K - 1:
            u, through = h // U, tokens[:h + U]
            if (self._owner(_chain(keys, tokens, U, u), through) is None
                    and not any(u in _units(a, a, b, U) and tuple(ts[:h + U]) == through for ts, a, b in extra)):
                break
            h += U
        return h

    def sources(self, i: int, hit: int) -> tuple[Source, ...]:
        """The canonical sources of request i's hit: unit by unit, the lowest-numbered request that has computed the
        whole unit from the same tokens through it, adjacent units of one request merged into one run."""
        if hit % self.unit:
            raise ValueError(f"request {i}'s hit of {hit} isn't whole units of {self.unit}")
        runs: list[list[int]] = []
        for u in range(hit // self.unit):
            a = self.owner(i, u)
            if a is None:
                raise ValueError(f"request {i}'s hit of {hit}: no request computed its unit {u} from its tokens")
            if runs and runs[-1][0] == a:
                runs[-1][1] += self.unit
            else:
                runs.append([a, self.unit])
        return tuple(Source(a, n) for a, n in runs)


@dataclass(frozen=True)
class Walk:
    """What the walk gives back: I1's steps, each request's output length, the step of each explicit n it read, the
    requests it read a lag bit for, the step of each hit it read, each request's certified ingress index, and the steps
    whose new requests' ingress indices it read (with them), where its prediction missed."""
    steps: tuple[StepAdvice, ...]
    output_len: tuple[int, ...]
    explicit_n: tuple[int, ...]
    lag_requests: tuple[int, ...]
    sent_hits: tuple[int, ...] = ()
    order: tuple[int, ...] = ()
    order_sent: tuple[tuple[int, tuple[int, ...]], ...] = ()


def walk_steps(rows: Sequence[Sequence[int]], prompt_len: Sequence[int], cfg: Config, *, held: Holdings, pool: bool,
               hit_of: Callable[[int, int, int | None], int], resumed_of: Callable[[int, int], bool],
               n_of: Callable[[int, int], int], lag_of: Callable[[int], bool],
               order_of: Callable[[int, Sequence[int]], Sequence[int] | None]) -> Walk:
    """The one derivation (the module's docstring), over the certified requests' tokens (`held`, witness). The callbacks
    give what the residual holds: `hit_of(s, t, cached)` the hit in tokens of step s's row admitting certified request t,
    given `cached`, the hit the walk derives, or None where it derives none; `resumed_of(s, k)` whether step s's row k,
    after its request's first, resumes it (read unless `pool`, which instead checks the pool bound); `n_of(s, k)` n where
    the walk can't derive it; `lag_of(i)` whether request i's last sampling row ran after its stop, asked in async mode
    only of a request whose last two sampling rows are in consecutive steps (only there can the scheduler have run a
    decode past a stop); `order_of(s, numbers)` the certified ingress indices of step s's new requests `numbers`, or None
    where the walk's prediction stands.

    The request order: vLLM's FCFS admits a step's new requests from the waiting queue's head, so they are the lowest
    ingress indices not yet admitted, in arrival order. I1 numbers them by first row: in scheduling order that is arrival
    order, and in batch order it is Model Runner V2's, which sorts the step's requests by scheduled tokens, stably, so
    new requests go by n and, at equal n, by arrival.

    The scheduling order, for rows in batch order: the running requests in the running list's order (each appended when
    admitted), then the waiting ones, resumed requests first (each prepended when preempted, the latest preemption
    foremost) and then new ones in arrival order, the last of which alone can be cut short.

    The no-eviction bound: vLLM's free queue hands out every block no request has used, and every one freed uncached,
    before any cached block (a freed cached block joins its tail, LRU), so nothing is evicted while the blocks handed out
    so far fit the pool less its null block. The requests admitted by step s, each counting the blocks its prompt and
    kept outputs fill as if none were shared, bound those, so while that count fits, a hit is derived; once it doesn't,
    it never does again. No allocation failing (`pool_bound`) is weaker: cached free blocks are evicted before the pool
    is full. Derived hits also need the scheduling order and the sync scheduler, whose rows all compute known tokens (the
    async one caches a block only once its sampled ids come back)."""
    R = len(prompt_len)
    if len(held.requests) != R or len(held.outputs) != R:
        raise ValueError(f"the advice's {R} requests aren't the {len(held.requests)} certified ones")
    certified = [tuple(q.prompt_ids) + tuple(o) for q, o in zip(held.requests, held.outputs)]
    plen = [len(q.prompt_ids) for q in held.requests]
    life = [-(-len(t) // cfg.block_size) for t in certified]
    prefixes = Prefixes((), cfg.unit)
    arrival: list[int] = []                         # I1 request i is certified request arrival[i]
    taken: set[int] = set()
    lowest = 0                                      # the lowest ingress index not yet admitted
    derive = cfg.scheduler == "sync" and cfg.row_order != "explicit"
    filled = 0
    end: dict[int, int] = {}
    samples: dict[int, int] = {}
    sample_steps: dict[int, list[int]] = {}
    opened: set[int] = set()
    admitted: dict[int, int] = {}                   # each request's place in the running list: its latest admission
    last: dict[int, int] = {}                       # each request's latest step
    admissions = 0
    steps, explicit, sent, order_sent = [], [], [], []
    for s, step in enumerate(rows):
        if len(set(step)) != len(step):
            raise ValueError(f"step {s}: a request in two rows")
        adm = []
        for k, i in enumerate(step):
            if not 0 <= i < R:
                raise ValueError(f"step {s} row {k}: request {i} of {R}")
            first = i not in opened
            if first and i != len(opened):
                raise ValueError(f"step {s} row {k}: request {i} before request {len(opened)}: requests are numbered by first row")
            opened.add(i)
            adm.append((first, not first and not pool and resumed_of(s, k)))
        new = [k for k in range(len(step)) if adm[k][0]]
        numbers = [step[k] for k in new]
        told = tuple(order_of(s, numbers) or ()) if new else ()
        fcfs, arriving = _arriving(s, numbers, told, lowest, taken, plen, prompt_len)
        filled += sum(life[t] for t in arriving)
        derive = derive and filled <= cfg.num_gpu_blocks - 1

        def admit(t: int, K: int, extra: Sequence[tuple[Sequence[int], int, int]] = ()) -> int:
            cached = (0 if not cfg.enable_prefix_caching else
                      prefixes.lookup(certified[t], K, extra) if derive else None)
            if cached is None:
                sent.append(s)
            return hit_of(s, t, cached)

        def number(ordered: Sequence[int], predicted: Sequence[int] | None) -> None:
            for i, t in zip(numbers, ordered, strict=True):
                if plen[t] != prompt_len[i]:
                    raise ValueError(f"request {i}: prompt_len {prompt_len[i]}, its certified prompt has {plen[t]} ids")
                arrival.append(t)
                taken.add(t)
                prefixes.push(certified[t])
            if tuple(ordered) != tuple(predicted or ()):
                order_sent.append((s, tuple(ordered)))

        used, made, early = 0, {}, {}

        def run(ks: Sequence[int]) -> None:
            nonlocal used
            for k in ks:
                i, (first, resumed) = step[k], adm[k]
                K, hit = prompt_len[i] + samples.get(i, 0), 0
                if first or resumed:
                    hit = early[arrival[i]] if first and arrival[i] in early else admit(arrival[i], K)
                start = hit if first or resumed else end[i]
                if K - start <= 0:
                    raise ValueError(f"step {s} row {k}: request {i} has nothing left to compute")
                left = cfg.max_num_batched_tokens - used
                if K - start == 1:
                    n = 1
                elif cfg.row_order == "explicit":
                    n = n_of(s, k)
                    explicit.append(s)
                else:
                    n = min(_want(K, start, first or resumed, cfg), left)
                if not 0 < n <= left:
                    raise ValueError(f"step {s} row {k}: n = {n} with {left} of the budget left")
                used += n
                if start + n >= K:
                    samples[i] = samples.get(i, 0) + 1
                    sample_steps.setdefault(i, []).append(s)
                end[i] = start + n
                prefixes.add(i, start, start + n)
                made[k] = (start, n, hit, resumed)

        if cfg.row_order == "batch":
            seq = sorted((k for k in range(len(step)) if not adm[k][0]),
                         key=lambda k: (1, -last[step[k]], admitted[step[k]]) if adm[k][1] else (0, admitted[step[k]]))
            run(seq)
            if new:
                predicted = (_predict(fcfs, cfg.max_num_batched_tokens - used, admit, plen, certified, early, cfg)
                             if arriving == fcfs else None)
                number(told or predicted, predicted)
                later = sorted(new, key=lambda k: arrival[step[k]])
                run(later)
                seq += later
        else:
            if new:
                number(told or fcfs, fcfs)
            seq = list(range(len(step)))
            run(seq)
        while lowest in taken:
            lowest += 1
        out = []
        for k, i in enumerate(step):
            start, n, hit, resumed = made[k]
            out.append(Row(i, "decode" if n == 1 and start >= prompt_len[i] else "prefill", start, n, hit, resumed,
                           prefixes.sources(i, hit)))
        for k in seq:
            if adm[k][0] or adm[k][1]:
                admitted[step[k]] = admissions
                admissions += 1
            last[step[k]] = s
        steps.append(StepAdvice(tuple(out)))
    output_len, lagged = _lengths(R, samples, sample_steps, lag_of, cfg)
    if pool and not pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in steps], cfg):
        raise ValueError("no preemption sent, but the rows don't keep within the KV pool")
    return Walk(tuple(steps), tuple(output_len), tuple(explicit), tuple(lagged), tuple(sent), tuple(arrival),
                tuple(order_sent))


def _arriving(s: int, numbers: Sequence[int], told: Sequence[int], lowest: int, taken: set[int], plen: Sequence[int],
              prompt_len: Sequence[int]) -> tuple[list[int], list[int]]:
    """Step s's new requests (I1's `numbers`) as certified requests: the ones FCFS admits next, the lowest ingress indices
    not yet admitted (`taken`), and the ones arriving, `told` where it names others. An order sent that isn't as many
    certified requests not yet admitted, or whose prompts aren't the new requests' lengths, is refused."""
    fcfs, t = [], lowest
    while len(fcfs) < len(numbers):
        if t not in taken:
            fcfs.append(t)
        t += 1
    if told and (len(told) != len(numbers) or len(set(told)) != len(told)
                 or any(not 0 <= t < len(plen) or t in taken for t in told)):
        raise ValueError(f"step {s}: the order sent for its {len(numbers)} new requests isn't as many certified requests "
                         f"not yet admitted")
    arriving = sorted(told) if told and sorted(told) != fcfs else fcfs
    if sorted(plen[t] for t in arriving) != sorted(prompt_len[i] for i in numbers):
        raise ValueError(f"step {s}: the new requests' prompt_len {[prompt_len[i] for i in numbers]} aren't their "
                         f"certified prompts' id counts {[plen[t] for t in arriving]}")
    return fcfs, arriving


def _predict(fcfs: Sequence[int], left: int, admit: Callable[..., int], plen: Sequence[int],
             certified: Sequence[Sequence[int]], early: dict[int, int], cfg: Config) -> list[int]:
    """The batch order's numbering of a step's new requests where they arrive as FCFS has them: each admitted in turn
    (its hit kept in `early`) against the budget `left`, then Model Runner V2's stable sort by the tokens scheduled."""
    extra, ns = [], {}
    for t in fcfs:
        early[t] = hit = admit(t, plen[t], extra)
        ns[t] = 1 if plen[t] - hit == 1 else min(_want(plen[t], hit, True, cfg), left)
        left -= max(ns[t], 0)
        extra.append((certified[t], hit, hit + max(ns[t], 0)))
    return sorted(fcfs, key=ns.__getitem__)


def _lengths(R: int, samples: dict[int, int], sample_steps: dict[int, list[int]], lag_of: Callable[[int], bool],
             cfg: Config) -> tuple[list[int], list[int]]:
    """Each request's output length, its sampling rows less a lag bit, and the requests a lag bit was read for: in async
    mode, those whose last two sampling rows are in consecutive steps."""
    output_len, lagged = [], []
    for i in range(R):
        if i not in samples:
            raise ValueError(f"request {i} never sampled")
        lag = False
        if cfg.scheduler == "async" and len(sample_steps[i]) >= 2 and sample_steps[i][-1] == sample_steps[i][-2] + 1:
            lag = lag_of(i)
            lagged.append(i)
        output_len.append(samples[i] - lag)
    return output_len, lagged


def _want(K: int, start: int, admits: bool, cfg: Config) -> int:
    """What a row asks of the budget: the rest of its known tokens, at most the long-prefill threshold, and a running
    row no further than the model's length less one."""
    want = K - start
    if cfg.long_prefill_token_threshold > 0:
        want = min(want, cfg.long_prefill_token_threshold)
    if not admits:
        want = min(want, cfg.max_model_len - 1 - start)
    return want


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
    request `walk.order[i]`, whose `user` is its id; prompt_len against the certified id count; output_len against the
    committed outputs; finish by `stop_rule`."""
    R = len(res.prompt_len)
    order = walk.order
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


def config_of(trace: Mapping[str, Any]) -> Config:
    """The registered config of a step trace, or of any document holding `Config`'s fields under ``config``."""
    c = trace["config"]
    return Config(block_size=c["block_size"], num_gpu_blocks=c["num_gpu_blocks"], max_num_batched_tokens=c["max_num_batched_tokens"],
                  max_model_len=c["max_model_len"], eos_token_id=c["eos_token_id"], stop_token_ids=tuple(c["stop_token_ids"]),
                  scheduler=c.get("scheduler", "sync"), long_prefill_token_threshold=c.get("long_prefill_token_threshold", 0),
                  max_num_seqs=c.get("max_num_seqs", 256), enable_prefix_caching=c.get("enable_prefix_caching", True),
                  row_order=c.get("row_order", "scheduling" if c.get("rows_in_scheduling_order", True) else "explicit"),
                  hit_unit=c.get("hit_unit", 0))


# --- the residual -----------------------------------------------------------------------------------------------------

def _sampling_rows(advice: Advice, i: int) -> int:
    P, k = advice.requests[i].prompt_len, 0
    for st in advice.steps:
        for r in st.rows:
            if r.request == i and r.start + r.n >= P + k:
                k += 1
    return k


def residual(advice: Advice, cfg: Config, held: Holdings, *, order: Sequence[int] | None = None,
             shaped: bool = False) -> Residual:
    """The advice that remains of `advice`, whose requests are certified requests `order` (None: in ingress order): what
    `reconstruct` needs beside the config and the tokens to give every field back. Refused unless the walk gives
    `advice` back: a hit other than the longest cached one while nothing can have been evicted, a source other than the
    earliest-admitted request, or a start, n or kind off vLLM's rules."""
    order = tuple(range(len(advice.requests))) if order is None else tuple(order)
    if sorted(order) != list(range(len(advice.requests))):
        raise ValueError(f"the advice's {len(advice.requests)} requests aren't certified requests {order}, each once")
    number = {t: i for i, t in enumerate(order)}
    prompt_len = tuple(q.prompt_len for q in advice.requests)
    pool = pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in advice.steps], cfg)
    if pool and any(r.preempted for st in advice.steps for r in st.rows):
        raise ValueError("a resumed row where the pool bound rules preemption out")
    hits: list[int] = []
    resumes: list[int] = []
    later = [0]
    ns: list[int] = []
    lags: list[bool] = []
    row_of = [{r.request: r for r in st.rows} for st in advice.steps]

    def hit_of(s: int, t: int, cached: int | None) -> int:
        i = number[t]
        if i not in row_of[s]:
            raise ValueError(f"step {s}: the walk admits request {i} (certified request {t}), which has no row there")
        h = row_of[s][i].cache_hit
        if cached is not None:
            if h != cached:
                raise ValueError(f"step {s} request {i}: a hit of {h} where nothing can have been evicted, and the "
                                 f"longest cached prefix is {cached}")
            return h
        if h % cfg.unit:
            raise ValueError(f"step {s} request {i}: a cache hit of {h} isn't whole hit units ({cfg.unit})")
        hits.append(h // cfg.unit)
        return h

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
    walk = walk_steps(rows, prompt_len, cfg, held=held, pool=pool, hit_of=hit_of, resumed_of=resumed_of, n_of=n_of,
                      lag_of=lag_of, order_of=lambda s, numbers: tuple(order[i] for i in numbers))
    for s, (mine, theirs) in enumerate(zip(walk.steps, advice.steps, strict=True)):
        for j, (a, b) in enumerate(zip(mine.rows, theirs.rows, strict=True)):
            if a != b:
                off = [f"{f} {getattr(b, f)!r}, where the walk gives {getattr(a, f)!r}"
                       for f in ("start", "n", "kind", "cache_hit", "preempted", "sources") if getattr(a, f) != getattr(b, f)]
                raise ValueError(f"step {s} row {j} (request {b.request}): {'; '.join(off)}")
    return Residual(rows, prompt_len, tuple(hits), None if pool else tuple(resumes),
                    tuple(lags) if cfg.scheduler == "async" else None, walk.order_sent, tuple(ns),
                    None if shaped else tuple(advice.release))


def reconstruct(res: Residual, cfg: Config, held: Holdings) -> Walk:
    """I1's steps from the residual, the config and the tokens: the compile slice's proof block's derivation, which the
    compiler then reads."""
    values: dict[str, Any] = {"hit": iter(res.hits), "n": iter(res.ns), "lag": iter(res.lag or ())}
    resumes = iter(res.resumes or ())
    later, resume = [0], [next(resumes, None)]
    told = iter(res.order)
    entry = [next(told, None)]

    def order_of(s: int, numbers: Sequence[int]) -> Sequence[int] | None:
        if entry[0] is None or entry[0][0] != s:
            return None
        ordered, entry[0] = entry[0][1], next(told, None)
        return ordered

    def take(name: str) -> Any:
        value = next(values[name], None)
        if value is None:
            raise ValueError(f"residual: fewer {name} values than the rows read")
        return value

    def resumed_of(s: int, k: int) -> bool:
        later[0] += 1
        if resume[0] != later[0] - 1:
            return False
        resume[0] = next(resumes, None)
        return True

    walk = walk_steps(res.rows, res.prompt_len, cfg, held=held, pool=res.resumes is None,
                      hit_of=lambda s, t, cached: cached if cached is not None else take("hit") * cfg.unit,
                      resumed_of=resumed_of, n_of=lambda s, k: take("n"), lag_of=lambda i: take("lag"), order_of=order_of)
    if resume[0] is not None or any(next(it, None) is not None for it in values.values()):
        raise ValueError("residual: more values than the rows read, or resumes out of order")
    if entry[0] is not None or walk.order_sent != tuple((s, tuple(o)) for s, o in res.order):
        raise ValueError("residual: an order entry the walk doesn't read, or one equal to its prediction")
    return walk


def rebuild(res: Residual, cfg: Config, held: Holdings, release: Sequence[int]) -> Advice:
    """The whole of v0 from the residual, the config and the tie's witness: `rebuild(residual(a), …) == a`. `release` is
    read only on a shaped egress, where it is derived (`served.derive`)."""
    walk = reconstruct(res, cfg, held)
    requests = tie_fields(walk, res, held, cfg)
    for i, lag in zip(walk.lag_requests, res.lag or (), strict=True):
        if lag and requests[i].finish != "eos":
            raise ValueError(f"request {i}: a decode after its last output, which no stop id ended")
    return Advice(tuple(requests), walk.steps, tuple(res.release if res.release is not None else release))


# --- bits -------------------------------------------------------------------------------------------------------------

class Records(NamedTuple):
    """An encoding's bits by record, as I1 charges them, each by field: the header, each request's record, each step's,
    and each kept output's release."""
    header: dict[str, int]
    requests: list[dict[str, int]]
    steps: list[dict[str, int]]
    release: int


def _uvar(value: int) -> int:
    return WIDTH_BITS + value.bit_length()


def i1_records(adv: Advice) -> Records:
    """I1's layout (its module's docstring), by field; `charge` of it is I1's `bits_by_window`."""
    lens = [x for q in adv.requests for x in (q.prompt_len, q.output_len)]
    lens += [x for st in adv.steps for r in st.rows for x in (r.start, r.n, r.cache_hit, *(s.n for s in r.sources))]
    w_req, w_len = max(len(adv.requests) - 1, 0).bit_length(), max(lens, default=0).bit_length()
    w_src = max((len(r.sources) - 1 for st in adv.steps for r in st.rows if r.cache_hit), default=0).bit_length()
    header = {"header": 8 + _uvar(len(adv.requests)) + _uvar(len(adv.steps)) + 3 * WIDTH_BITS}
    requests = [{"id": _uvar(len(q.id.encode())) + 8 * len(q.id.encode()), "prompt_len": w_len, "output_len": w_len, "finish": 1}
                for q in adv.requests]
    steps = [{"rows_in_step": w_req, "request": w_req * len(st.rows), "kind": len(st.rows), "start": w_len * len(st.rows),
              "n": w_len * len(st.rows), "cache_hit": w_len * len(st.rows), "preempted": len(st.rows),
              "sources": sum(w_src + len(r.sources) * (w_req + w_len) for r in st.rows if r.cache_hit)} for st in adv.steps]
    return Records(header, requests, steps, max(adv.release, default=0).bit_length())


def derived_records(res: Residual, walk: Walk, cfg: Config) -> Records:
    """The residual's layout, by field, each at its least width as I1 does: the header (I1's, a width for hits in hit
    units, a flag each for order, resumes and release being sent, when order is, its count of steps, and when resumes
    are, their width and count); a request's prompt_len and its lag bit (when asked); a step's row count and requests,
    the step's position and its new requests' ingress indices where the order is sent, each resumed row's position (when
    sent), a hit flag for each admitting row whose hit the walk reads, with the hit's units when nonzero, and an explicit
    n where read; a kept output's release (when sent). The hit unit is the registered config's, and sources are
    canonical, so neither costs anything here."""
    w_req = max(len(res.prompt_len) - 1, 0).bit_length()
    w_len = max((*res.prompt_len, *res.ns), default=0).bit_length()
    w_hit = max(res.hits, default=0).bit_length()
    w_res = max(res.resumes or (0,)).bit_length()
    w_step = max(len(res.rows) - 1, 0).bit_length()
    header = {"header": 8 + _uvar(len(res.prompt_len)) + _uvar(len(res.rows)) + 3 * WIDTH_BITS + 3}
    if res.order:
        header["order"] = _uvar(len(res.order))
    if res.resumes is not None:
        header["resumes"] = WIDTH_BITS + _uvar(len(res.resumes))
    lagged = set(walk.lag_requests)
    requests = [{"prompt_len": w_len, "lag": int(i in lagged)} for i in range(len(res.prompt_len))]
    told = {s: len(o) for s, o in res.order}
    explicit: dict[int, int] = {}
    for s in walk.explicit_n:
        explicit[s] = explicit.get(s, 0) + 1
    sent = set(walk.sent_hits)
    seen: set[int] = set()
    steps = []
    for s, st in enumerate(walk.steps):
        d = {"rows_in_step": w_req, "request": w_req * len(st.rows), "resumes": 0, "hits": 0, "n": w_len * explicit.get(s, 0),
             "order": w_step + w_req * told[s] if s in told else 0}
        for r in st.rows:
            first = r.request not in seen
            seen.add(r.request)
            if r.preempted:
                d["resumes"] += w_res
            if (first or r.preempted) and s in sent:
                d["hits"] += 1 + (w_hit if r.cache_hit else 0)
        steps.append(d)
    return Records(header, requests, steps,
                   max(res.release, default=0).bit_length() if res.release is not None else 0)


def earliest_windows(adv: Advice, T: int) -> tuple[list[int], list[int]]:
    """I1's charge windows: each step's (the earliest window of an output at that step or later; the last window when
    none follows), and each request's first step."""
    windows = [b // T for b in adv.release]
    own: list[int | None] = [None] * len(adv.steps)
    for o, w in zip(outputs(adv), windows):
        prev = own[o.step]
        own[o.step] = w if prev is None else min(prev, w)
    earliest, at = [0] * len(adv.steps), None
    for s in range(len(adv.steps) - 1, -1, -1):
        mine = own[s]
        if mine is not None:
            at = mine if at is None else min(at, mine)
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
    earliest, first = earliest_windows(adv, T)
    add(earliest[0], rec.header)
    for i, d in enumerate(rec.requests):
        add(earliest[first[i]], d)
    for s, d in enumerate(rec.steps):
        add(earliest[s], d)
    for b in adv.release:
        add(b // T, {"release": rec.release})
    return out


FORCED = ("prompt_len", "hits", "lag", "resumes", "order")
"""The residual's fields that witness fixes but that are sent: checked (prompt_len, lag, order) or fixed by vLLM's rules
over witness that nothing replays yet (hits past the no-eviction bound, resumes)."""
