"""The derived rules: the fields of advice v0 (I1, `verity.core.protocols.certifier.advice`) that cost zero bits,
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
(each kept row's sampled id). The compile unit holds every request's tokens (its prompt ids, then its kept sampled ids).

**The derivations** are one walk over the steps (`_walk`), the same for the encoder (`residual`) and the compile unit
(`reconstruct`), so they can't drift apart. It takes each step's rows in vLLM d9105ea8's scheduling order: the rows' own
(`row_order` "scheduling"), or, for rows in the runner's batch order ("batch"), the order vLLM's FCFS lists give them
(the running requests by admission, then the resumed ones, then the new ones in arrival order, the certified ingress
order). A request's known tokens K are its prompt length plus its sampling rows so far. A row that admits its request
(its first, or one that resumes it after a preemption) starts at its prefix-cache hit; any other starts where the
request's previous row ended. It schedules n = min(K − start, the step's budget left, the long-prefill threshold) tokens
(vLLM's running and waiting loops), and samples when it reaches K (the GPU runner's discard rule). `kind` is I1's: decode
exactly when n = 1 at or past the prompt. Two fields come from the tokens (`Prefixes`):
- a hit's `sources` are canonical: for each hit unit, the earliest-admitted (lowest-numbered) request that has computed
  the whole unit by this step (I1's rule) from the same tokens through it. Prefixes are looked up by a chain of SHA-512
  keys over whole units (`unit_key`), and a key names a request only where its tokens match, so no collision and no
  interpreter changes what is derived;
- while nothing can have been evicted (`_walk`'s no-eviction bound), a `cache_hit` is the longest prefix, in hit units
  and at most K − 1 tokens, whose every unit some row scheduled before it computed (vLLM's lookup: a block is cached as it
  is scheduled, and stays cached until evicted). A shorter or longer hit is refused. Past the bound, hits are sent.
`FIELDS` is the table of every I1 field: its verdict, derivation, checker and cost.

**The residual** (`Residual`): each step's requests in I1's row order (the scheduler's choice, the replay's), and
- `prompt_len`, one a request, sent, forced by the certified id count (the tie checks it);
- `hits`: each admitting row's prefix-cache hit, in hit units (`Config.unit`: the block, finer only in a hybrid model,
  whose hit may end inside a block), only from the step where the no-eviction bound fails (or in async or explicit
  mode): which cached blocks LRU eviction left is vLLM's block pool over witness, which nothing replays yet;
- `resumes`: which rows after their request's first resume it, by position among those rows, unless the pool bound rules
  preemption out, which the verifier checks;
- `lag`: async mode only, a bit for each request whose last two sampling rows are in consecutive steps: whether the last
  one ran after the stop (the async scheduler schedules a request's next decode before it reads the sample, and then
  drops that decode's), which `output_len` subtracts; forced by the committed ids (a lag follows a stop id);
- `order`: a step's new requests' certified ingress indices, in I1's order, only where the walk's prediction misses.
  vLLM's FCFS admits a step's new requests as the lowest ingress indices not yet admitted, in arrival order, and I1
  numbers them by first row: in batch order, Model Runner V2's stable sort by scheduled tokens. So a step is sent only
  if the engine got its requests out of the certified admission order, or a runner orders its batch another way;
- `ns`: n where the walk can't derive it: only when the scheduling order is unknown (`row_order` "explicit");
- `release`: each kept output's egress bucket on an unshaped egress, where its choices are the timing channel; none on a
  shaped one, where `served.derive` fixes them from the ingress anchors, the output lengths and the declared sync.

`bits_table` measures v0 (I1's own `bits_by_window`) and the residual per certifier window, the residual at its own least
widths and charged by I1's rule (`charge`), after checking that `rebuild(residual(a)) == a`.

    python derived.py table --trace TRACE.json[.gz] ... --out TABLE.json [--T 600 --bucket-ms 100]

A `--trace` that is a directory is one of the trace corpus's (`load_corpus_trace`; `--block-size`, `--eos`,
`--stop-ids`, `--row-order`).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
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
                  row_order=c.get("row_order", "scheduling" if c.get("rows_in_scheduling_order", True) else "explicit"),
                  hit_unit=c.get("hit_unit", 0))


def holdings_of(trace: dict) -> Holdings:
    return Holdings(requests=tuple(CertifiedRequest(r["id"], tuple(r["prompt_ids"]), r["max_tokens"]) for r in trace["requests"]),
                    outputs=tuple(tuple(r["output_ids"]) for r in trace["requests"]))


def _jsonl(path: Path) -> list[dict]:
    """A JSON-lines file's records (or its `.gz`'s), less an unterminated last line (a writer killed mid-line)."""
    gz = path.with_name(path.name + ".gz")
    raw = path.read_bytes() if path.exists() or not gz.exists() else gzip.decompress(gz.read_bytes())
    lines = raw.decode("utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def _flag(argv: Sequence[str], name: str, default: int | None = None) -> int | None:
    return int(argv[argv.index(name) + 1]) if name in argv else default


def load_corpus_trace(d: str | Path, *, block_size: int = 16, eos_token_id: int = 128001,
                      stop_token_ids: Sequence[int] = (), num_gpu_blocks: int | None = None,
                      row_order: str = "batch") -> dict:
    """One trace of the trace corpus (`benchmarks/network_e2e/traces`, `vllm-trace-corpus/v1`: a directory with the
    joined `steps.jsonl` and `requests.jsonl`, `meta.json` and, when kept, `server.log`) in `load_trace`'s format.

    The registered config is the server's argv; the block size and the stop ids are the model's, which the corpus doesn't
    record, so they are parameters, and the walk's tie refuses a stop id they miss. The pool's size, unless given, is
    `server.log`'s "GPU KV cache size" in blocks: vLLM logs the pool's blocks over the blocks a max_model_len request
    takes, times max_model_len, which is never more than the pool, so the bound stays sound. Without either it is 0, so
    the pool bound fails and the resumes are sent.

    A step's rows are in the model runner's batch order (`row_order` "batch", I1's) or in vLLM's scheduling order
    ("scheduling": running, then resumed, then new; the hook gives each list in its loop's order). The requests are in
    the engine's arrival order, which stands for the certified ingress order. The corpus doesn't record whose KV a hit
    reads; `advice_of` names the canonical sources."""
    d = Path(d)
    argv = json.loads((d / "meta.json").read_text()).get("server_argv") or []
    budget = _flag(argv, "--max-num-batched-tokens")
    if budget is None:
        raise ValueError(f"{d}: the server's argv has no --max-num-batched-tokens, the token budget the walk needs")
    if num_gpu_blocks is None:
        log = d / "server.log"
        found = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.read_text(errors="replace")) if log.exists() else []
        num_gpu_blocks = int(found[-1].replace(",", "")) // block_size if found else 0
    config = {"block_size": _flag(argv, "--block-size", block_size), "num_gpu_blocks": num_gpu_blocks,
              "max_num_batched_tokens": budget, "max_model_len": _flag(argv, "--max-model-len"),
              "max_num_seqs": _flag(argv, "--max-num-seqs", 256),
              "long_prefill_token_threshold": _flag(argv, "--long-prefill-token-threshold", 0),
              "scheduler": "async" if "--async-scheduling" in argv else "sync",
              "enable_prefix_caching": "--no-enable-prefix-caching" not in argv,
              "eos_token_id": eos_token_id, "stop_token_ids": list(stop_token_ids), "row_order": row_order}
    if "--async-scheduling" not in argv and "--no-async-scheduling" not in argv:
        raise ValueError(f"{d}: the server's argv doesn't say whether it schedules asynchronously")
    if config["max_model_len"] is None:
        raise ValueError(f"{d}: the server's argv has no --max-model-len")
    bs = config["block_size"]
    requests = []
    for q in sorted(_jsonl(d / "requests.jsonl"), key=lambda q: q.get("arrival", 0.0)):
        if q["finish_reason"] not in ("stop", "length"):
            raise ValueError(f"{d}: request {q['id']} finished {q['finish_reason']!r}, which the walk can't derive")
        requests.append({"id": q["id"], "prompt_ids": q["prompt_ids"], "max_tokens": q["max_tokens"],
                         "output_ids": q["output_ids"], "finish": "eos" if q["finish_reason"] == "stop" else "length"})
    index = {q["id"]: t for t, q in enumerate(requests)}
    seen: set[int] = set()
    steps, t0 = [], None
    for s in _jsonl(d / "steps.jsonl"):
        if not s["total_num_scheduled_tokens"]:
            continue
        t0 = s["t"] if t0 is None else t0
        running = [r for r in s["cached"] if not r["resumed"]]
        resumed = [r for r in s["cached"] + s["new"] if r["resumed"]]
        new = [r for r in s["new"] if not r["resumed"]]
        rows = {}
        for r in running + resumed + new:
            if not r["num_scheduled_tokens"]:
                continue
            if r["id"] not in index:
                raise ValueError(f"{d}: step {s['step']} schedules {r['id']}, which the engine never freed")
            t, start, n = index[r["id"]], r["num_computed_tokens"], r["num_scheduled_tokens"]
            row = {"request": t, "start": start, "n": n, "new": t not in seen, "resumed": bool(r["resumed"]),
                   "samples": start + n >= len(requests[t]["prompt_ids"]) + r.get("num_output_tokens", 0)}
            if (row["new"] or row["resumed"]) and start % bs:
                raise ValueError(f"{d}: step {s['step']}: {r['id']}'s hit of {start} isn't whole {bs}-token blocks")
            seen.add(t)
            rows[r["id"]] = row
        if row_order == "batch":
            if sorted(s.get("batch") or ()) != sorted(rows):
                raise ValueError(f"{d}: step {s['step']}'s batch order doesn't hold its scheduled requests")
            ordered = [rows[i] for i in s["batch"]]
        else:
            ordered = list(rows.values())
        steps.append({"t_end_s": round(s.get("t_output", s["t"]) - t0, 6), "rows": ordered})
    return {"kind": "enc-derived/sched-trace/v2", "source": str(d), "config": config, "requests": requests, "steps": steps}


def advice_of(trace: dict, bucket_ms: float = 100.0) -> tuple[Advice, tuple[int, ...]]:
    """I1 from a trace, and each I1 request's ingress index. A row admits its request when it is its first or resumes it;
    such a row's `cache_hit` is its start, and its sources the canonical ones (`Prefixes`), whichever block vLLM read.
    Each kept output is released in the bucket its step ends in (unshaped)."""
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
    prefixes = Prefixes([tuple(reqs[t]["prompt_ids"]) + tuple(reqs[t]["output_ids"]) for t in order], config_of(trace).unit)
    seen: set[int] = set()
    kept = [0] * len(order)
    steps, release = [], []
    for s in trace["steps"]:
        rows = []
        bucket = int(s["t_end_s"] * 1000.0 // bucket_ms)
        for r in s["rows"]:
            prefixes.add(num[r["request"]], r["start"], r["start"] + r["n"])
        for r in s["rows"]:
            i = num[r["request"]]
            admits = i not in seen or r["resumed"]
            seen.add(i)
            hit = r["start"] if admits else 0
            rows.append(Row(i, "decode" if r["n"] == 1 and r["start"] >= requests[i].prompt_len else "prefill", r["start"], r["n"],
                            hit, bool(r["resumed"]), prefixes.sources(i, hit)))
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
    """Witness, in the compile unit: every request's tokens (its prompt ids, then its kept sampled ids), I1-numbered as
    the walk numbers them, and for each prefix of whole hit units whose last unit some request has computed KV at so far
    (all of it, in one run from a unit boundary), the requests that did. A prefix is keyed by `unit_key`'s chain; a key
    names a request only where that request's tokens through the prefix are the asking one's, so no collision can change
    what is derived, and the prover and the compile unit derive the same on any interpreter."""

    def __init__(self, tokens: Sequence[Sequence[int]], unit: int):
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


def _walk(rows: Sequence[Sequence[int]], prompt_len: Sequence[int], cfg: Config, *, held: Holdings, pool: bool,
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
        fcfs, t = [], lowest
        while len(fcfs) < len(new):
            if t not in taken:
                fcfs.append(t)
            t += 1
        told = tuple(order_of(s, numbers) or ()) if new else ()
        if told and (len(told) != len(new) or len(set(told)) != len(told) or any(not 0 <= t < R or t in taken for t in told)):
            raise ValueError(f"step {s}: the order sent for its {len(new)} new requests isn't as many certified requests "
                             f"not yet admitted")
        arriving = sorted(told) if told and sorted(told) != fcfs else fcfs
        if sorted(plen[t] for t in arriving) != sorted(prompt_len[i] for i in numbers):
            raise ValueError(f"step {s}: the new requests' prompt_len {[prompt_len[i] for i in numbers]} aren't their "
                             f"certified prompts' id counts {[plen[t] for t in arriving]}")
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
                predicted = None
                if arriving == fcfs:
                    left, extra, ns = cfg.max_num_batched_tokens - used, [], {}
                    for t in fcfs:
                        early[t] = hit = admit(t, plen[t], extra)
                        ns[t] = 1 if plen[t] - hit == 1 else min(_want(plen[t], hit, True, cfg), left)
                        left -= max(ns[t], 0)
                        extra.append((certified[t], hit, hit + max(ns[t], 0)))
                    predicted = sorted(fcfs, key=ns.__getitem__)
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
    output_len, lagged = [], []
    for i in range(R):
        if i not in samples:
            raise ValueError(f"request {i} never sampled")
        lag = False
        if cfg.scheduler == "async" and len(sample_steps[i]) >= 2 and sample_steps[i][-1] == sample_steps[i][-2] + 1:
            lag = lag_of(i)
            lagged.append(i)
        output_len.append(samples[i] - lag)
    if pool and not pool_bound([[(r.request, r.start, r.n) for r in st.rows] for st in steps], cfg):
        raise ValueError("no preemption sent, but the rows don't keep within the KV pool")
    return Walk(tuple(steps), tuple(output_len), tuple(explicit), tuple(lagged), tuple(sent), tuple(arrival),
                tuple(order_sent))


def _want(K: int, start: int, admits: bool, cfg: Config) -> int:
    """What a row asks of the budget: the rest of its known tokens, at most the long-prefill threshold, and a running
    row no further than the model's length less one."""
    want = K - start
    if cfg.long_prefill_token_threshold > 0:
        want = min(want, cfg.long_prefill_token_threshold)
    if not admits:
        want = min(want, cfg.max_model_len - 1 - start)
    return want


def _sampling_rows(advice: Advice, i: int) -> int:
    P, k = advice.requests[i].prompt_len, 0
    for st in advice.steps:
        for r in st.rows:
            if r.request == i and r.start + r.n >= P + k:
                k += 1
    return k


def tokens_of(held: Holdings, order: Sequence[int], prompt_len: Sequence[int]) -> list[tuple[int, ...]]:
    """Each I1 request's tokens, from certified request `order[i]`: its prompt ids, then its committed sampled ids."""
    R = len(prompt_len)
    if len(held.requests) != R or sorted(order) != list(range(R)):
        raise ValueError(f"the advice's {R} requests aren't the {len(held.requests)} certified ones, each once")
    for i, (t, P) in enumerate(zip(order, prompt_len)):
        if len(held.requests[t].prompt_ids) != P:
            raise ValueError(f"request {i}: prompt_len {P}, its certified prompt has {len(held.requests[t].prompt_ids)} ids")
    return [tuple(held.requests[t].prompt_ids) + tuple(held.outputs[t]) for t in order]


def canonical_sources(advice: Advice, tokens: Sequence[Sequence[int]], unit: int) -> Advice:
    """`advice` with each hit's sources the canonical ones (`Prefixes`, in hit units of `unit` tokens) over each I1
    request's `tokens`."""
    prefixes, steps = Prefixes(tokens, unit), []
    for st in advice.steps:
        for r in st.rows:
            prefixes.add(r.request, r.start, r.start + r.n)
        steps.append(StepAdvice(tuple(Row(r.request, r.kind, r.start, r.n, r.cache_hit, r.preempted,
                                          prefixes.sources(r.request, r.cache_hit)) for r in st.rows)))
    return Advice(advice.requests, tuple(steps), advice.release)


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
    walk = _walk(rows, prompt_len, cfg, held=held, pool=pool, hit_of=hit_of, resumed_of=resumed_of, n_of=n_of,
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
    """I1's steps from the residual, the config and the tokens: the compile unit's derivation, which the compiler and
    N2's advised counts then read."""
    values = {"hit": iter(res.hits), "n": iter(res.ns), "lag": iter(res.lag or ())}
    resumes = iter(res.resumes or ())
    later, resume = [0], [next(resumes, None)]
    told = iter(res.order)
    entry = [next(told, None)]

    def order_of(s: int, numbers: Sequence[int]) -> Sequence[int] | None:
        if entry[0] is None or entry[0][0] != s:
            return None
        ordered, entry[0] = entry[0][1], next(told, None)
        return ordered

    def take(name: str):
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

    walk = _walk(res.rows, res.prompt_len, cfg, held=held, pool=res.resumes is None,
                 hit_of=lambda s, t, cached: cached if cached is not None else take("hit") * cfg.unit,
                 resumed_of=resumed_of, n_of=lambda s, k: take("n"), lag_of=lambda i: take("lag"), order_of=order_of)
    if resume[0] is not None or any(next(it, None) is not None for it in values.values()):
        raise ValueError("residual: more values than the rows read, or resumes out of order")
    if entry[0] is not None or walk.order_sent != tuple((s, tuple(o)) for s, o in res.order):
        raise ValueError("residual: an order entry the walk doesn't read, or one equal to its prediction")
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


def rebuild(res: Residual, cfg: Config, held: Holdings, release: Sequence[int]) -> Advice:
    """The whole of v0 from the residual, the config and the tie's witness: `rebuild(residual(a), …) == a`. `release` is
    read only on a shaped egress, where N2's advised counts derive it."""
    walk = reconstruct(res, cfg, held)
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
    header = {"header": 8 + _uvar(len(res.prompt_len)) + _uvar(len(res.rows)) + 3 * I1.WIDTH_BITS + 3}
    if res.order:
        header["order"] = _uvar(len(res.order))
    if res.resumes is not None:
        header["resumes"] = I1.WIDTH_BITS + _uvar(len(res.resumes))
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


FORCED = ("prompt_len", "hits", "lag", "resumes", "order")
"""The residual's fields that witness fixes but that are sent: checked (prompt_len, lag, order) or fixed by vLLM's rules
over witness that nothing replays yet (hits past the no-eviction bound, resumes)."""


def bits_table(trace: dict, *, T: int = 600, bucket_ms: float = 100.0) -> dict:
    """Per window: v0's bits (I1's `bits_by_window`) and the residual's, unshaped (release kept) and shaped (release
    derived), by field, after checking that the residual rebuilds v0 exactly."""
    cfg = config_of(trace)
    adv, order = advice_of(trace, bucket_ms)
    held = holdings_of(trace)
    res = residual(adv, cfg, held, order=order)
    shaped = residual(adv, cfg, held, order=order, shaped=True)
    if rebuild(res, cfg, held, ()) != adv or rebuild(shaped, cfg, held, adv.release) != adv:
        raise AssertionError("the residual doesn't rebuild v0 on this trace")
    v0 = charge(adv, T, v0_records(adv))
    if {w: sum(d.values()) for w, d in v0.items()} != I1.bits_by_window(adv, T):
        raise AssertionError("v0_records doesn't charge as I1's bits_by_window")
    walk = reconstruct(res, cfg, held)
    seen: set[int] = set()
    admitting = new_steps = 0
    for st in adv.steps:
        new_steps += any(r.request not in seen for r in st.rows)
        for r in st.rows:
            admitting += r.request not in seen or r.preempted
            seen.add(r.request)
    num = {t: i for i, t in enumerate(order)}
    vllm_sources = [(tuple(Source(num[a], n) for a, n in r["sources"]), adv.steps[s].rows[j].sources)
                    for s, st in enumerate(trace["steps"]) for j, r in enumerate(st["rows"]) if "sources" in r]
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
            "pool_bound": res.resumes is None, "new_steps": new_steps, "order_sent": len(res.order),
            "explicit_n": len(res.ns),
            "admitting_rows": admitting, "hits": sum(r.cache_hit > 0 for st in adv.steps for r in st.rows),
            "hits_sent": len(res.hits), "hits_derived": admitting - len(res.hits),
            "sources": sum(len(r.sources) for st in adv.steps for r in st.rows),
            "vllm_sources": len(vllm_sources), "vllm_sources_not_earliest": sum(a != b for a, b in vllm_sources),
            "resumed_rows": sum(r.preempted for st in adv.steps for r in st.rows),
            "lag_asked": len(walk.lag_requests), "lag_set": sum(res.lag or ()),
            "finish": {f: sum(q.finish == f for q in adv.requests) for f in ("eos", "length")},
            "windows": windows, "totals": totals, "by_field": by_field}


# --- the field table -------------------------------------------------------------------------------------------------

FIELDS = [
    {"field": "requests[i].id", "verdict": "derived", "bits": "0",
     "from": "certified ingress order, the walk's request order", "derivation": "request i is certified request "
     "order[i] (the walk's); its id is that request's `user`", "checker": "tie",
     "cost": "0: the tie addresses the certified request by index, with no id to compare"},
    {"field": "order (new)", "verdict": "derived; a step's sent sparse where the prediction misses",
     "bits": "0 / a count, and w_step + w_req a new request of each step sent (v0: none, the id says it)",
     "from": "the certified admission order (the warden's public schedule), rows, n",
     "derivation": "FCFS admits a step's new requests from the waiting queue's head: the lowest ingress indices not yet "
     "admitted, in arrival order. I1 numbers them by first row: in scheduling order, arrival order; in batch order, Model "
     "Runner V2's stable sort of the step by scheduled tokens, so by n and, at equal n, by arrival",
     "checker": "compile unit (it derives n in arrival order anyway); a sent step equal to the prediction is refused, "
                "and the tie uses each certified request once", "cost": "0"},
    {"field": "prompt_len", "verdict": "stays, forced", "bits": "w_len a request",
     "from": "certified request's id count (witness)", "derivation": "len(prompt ids); the Programs read it (a final prefill "
     "chunk is P - start), so it is public", "checker": "tie",
     "cost": "a 16-bit equality of the certified count with a public value: 15 ANDs a request"},
    {"field": "output_len", "verdict": "derived", "bits": "0", "from": "rows (public); forced by the committed sampled ids",
     "derivation": "the request's sampling rows (async: less the lag bit)",
     "checker": "compiler, N2's advised counts; tie",
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
     "depends on the arrival times, the timing channel (the replay's piece)",
     "checker": "compiler (the Programs' shapes)", "cost": "0"},
    {"field": "start", "verdict": "derived", "bits": "0", "from": "rows' history, cache_hit",
     "derivation": "the request's previous row's end; an admitting row's (its first, or a resumed one) is its hit",
     "checker": "compile unit (a derived hit makes it witness-dependent)", "cost": "0"},
    {"field": "n", "verdict": "derived", "bits": "0 (w_len a non-decode row when the scheduling order is unknown)",
     "from": "rows' history, prompt_len, max_num_batched_tokens, max_model_len, the certified ingress order",
     "derivation": "min(K - start, the step's budget left, the long-prefill threshold; a running row's max_model_len - 1 - "
     "start), K = prompt_len + sampling rows so far, the budget taken in vLLM's scheduling order (in batch order: running "
     "rows by admission, then resumed, then new ones in arrival order, so only the last arrival is cut short)",
     "checker": "compile unit", "cost": "0"},
    {"field": "kind", "verdict": "derived", "bits": "0", "from": "start, n, prompt_len",
     "derivation": "decode iff n = 1 and start >= prompt_len (I1's rule)", "checker": "compile unit", "cost": "0"},
    {"field": "cache_hit", "verdict": "canonical while nothing can have been evicted; else sent sparse (hits)",
     "bits": "0 / 1 an admitting row + w_hit (hit units) a hit (I1: w_len every row)",
     "from": "every request's tokens (witness), the rows scheduled before it, num_gpu_blocks",
     "derivation": "while the blocks of every request admitted so far, counted unshared from its prompt and kept outputs, "
     "fit num_gpu_blocks - 1 (vLLM hands out unused and uncached free blocks before evicting a cached one), in sync mode "
     "with the scheduling order known: the longest prefix, at most K - 1 tokens in whole hit units (the block, or a hybrid "
     "model's finer hash_block_size), each of whose units a row scheduled before it computed from the same tokens; "
     "past the bound, which cached blocks LRU eviction left is vLLM's block pool over witness, so the hit is sent",
     "checker": "compile unit: the derived hit, and a hit of any other length, shorter included, is refused; none yet "
                "for a sent one", "cost": "a token comparison per hit unit of each admitting row"},
    {"field": "sources", "verdict": "canonical", "bits": "0 (I1: a count, and w_req + w_len a run)",
     "from": "every request's tokens (witness), rows (which request computed KV where by each step)",
     "derivation": "hit unit by hit unit, the earliest-admitted (lowest-numbered) request that has computed the whole "
     "unit by this step (I1's rule) from the same tokens through it, in runs; looked up by a SHA-512 chain over whole "
     "units, every match checked on the tokens",
     "checker": "compile unit (it holds every request's tokens and checks their equality anyway); any other source is "
                "refused", "cost": "the token equality it checks for the KV read, and the lowest number among them"},
    {"field": "preempted", "verdict": "derived under the pool bound; else sent sparse (resumes)",
     "bits": "0 / a count and w_res a resumed row (I1: 1 every row)",
     "from": "rows, num_gpu_blocks", "derivation": "pool_bound: no allocation can fail, so nothing is preempted; "
     "else which rows resume is FCFS's pop of the last running request when the pool runs out, which needs the shared "
     "blocks (witness), so it is sent", "checker": "verifier (the bound, in the clear); none yet for the sent resumes",
     "cost": "0 (in the clear)"},
    {"field": "release", "verdict": "stays (unshaped); derived (shaped)", "bits": "w_rel a kept output / 0",
     "from": "unshaped: the egress head (witness), whose choices are the timing channel; shaped: the ingress anchors, the "
             "output lengths and the declared sync", "derivation": "shaped: served.derive with per-request output lengths",
     "checker": "N2's advised counts",
     "cost": "shaped: N2's advised counts' derive circuit (served.size: 678,795 ANDs at T = 600, FIXLEN)"},
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
    c = t.add_argument_group("a trace-corpus trace directory (`load_corpus_trace`)")
    c.add_argument("--block-size", type=int, default=16)
    c.add_argument("--eos", type=int, default=128001)
    c.add_argument("--stop-ids", type=int, nargs="*", default=())
    c.add_argument("--row-order", choices=("batch", "scheduling"), default="batch")
    a = ap.parse_args(list(argv) if argv is not None else None)
    tables = {}
    for p in a.trace:
        trace = load_corpus_trace(p, block_size=a.block_size, eos_token_id=a.eos, stop_token_ids=a.stop_ids,
                                  row_order=a.row_order) if p.is_dir() else load_trace(p)
        tables[p.name] = bits_table(trace, T=a.T, bucket_ms=a.bucket_ms)
        print(f"## {p.name}\n\n{render(tables[p.name])}\n")
    a.out.write_text(json.dumps({"kind": "enc-derived/tables/v2", "fields": FIELDS, "traces": tables}, indent=1))


if __name__ == "__main__":
    main()
