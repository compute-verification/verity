"""A vLLM general plugin (entry point group ``vllm.general_plugins``, beside `vllm_trace_plugin`) that makes the V1 scheduler
hostile: from a seed, every ``schedule()`` call makes adversarial but valid choices, so that the honest and hostile traces of
one plan (`corpus.py`) test whether a request's output ids depend on what it is batched with.

`register` reads ``VERITY_ADV_SEED`` (an integer; unset, the plugin does nothing) and ``VERITY_ADV_CACHE`` and wraps three
methods of `Scheduler` (the `AsyncScheduler` inherits them):

* `schedule`. Before the real call, with the seed's generator, it
  - preempts random running requests, one call in `1 / P_PREEMPT`, up to `PREEMPT_SHARE` of them
    (`Scheduler._preempt_request`: their blocks are freed and they go back to the waiting queue, to be recomputed when the
    scheduler resumes them), each request at most `MAX_FORCED` times, so every request still finishes;
  - shuffles the running list and the waiting queue: the order the scheduler serves and admits requests in, which Model
    Runner V2 keeps among rows of one size (`sort_batch_req_ids` is a stable sort by row size);
  - holds back a random share of the waiting requests (a share drawn per call) and the ones it just preempted, and puts them
    back at the front of the queue after the call; when nothing runs it leaves at least one request admissible.

  The real `schedule` then admits what it can of the rest, under its own checks (the token budget, the KV blocks, the
  256-row cap), so the step stays one the scheduler could have made.
* `_reserve_prefill_lookahead`, the last cut of every row's token count: half the time, a row of n > 1 tokens (a prefill,
  or a resumed request's recompute) is cut to a uniform count in [1, n - 1], so prefills split at random positions within
  the step's budget.
* `_get_local_prefix_cache_hit`, the prefix-cache lookup of a request's first scheduling (and of each resume after a
  preemption). ``VERITY_ADV_CACHE``: ``refuse`` refuses a hit half the time (the request recomputes those tokens),
  ``shortest`` cuts every hit longer than a block to its first block, ``natural`` (the default) leaves hits alone.

Each call's choices are a line of ``adversary.PID.jsonl`` in ``VERITY_TRACE_DIR`` (when it is set), keyed by the scheduler's
``current_step``, which the tracer records too (``sched_step``). An error in a choice is counted and written to
``adversary_errors.PID.txt``; the call goes on with the honest choice.
"""
from __future__ import annotations

import json
import os
import random
import time
import traceback
from collections import Counter
from pathlib import Path

from vllm_trace_plugin import client_id

SEED = "VERITY_ADV_SEED"
CACHE = "VERITY_ADV_CACHE"
TRACE_DIR = "VERITY_TRACE_DIR"
CACHE_MODES = ("natural", "refuse", "shortest")
P_PREEMPT = 0.1
PREEMPT_SHARE = 1 / 8
MAX_FORCED = 2
P_SPLIT = 0.5
P_REFUSE = 0.5
MAX_ERRORS = 20


class Adversary:
    """One scheduler's hostile choices and their record."""

    def __init__(self, seed: int, cache: str, out: Path | None):
        if cache not in CACHE_MODES:
            raise ValueError(f"{CACHE}={cache!r}; known: {CACHE_MODES}")
        self.rng = random.Random(seed)
        self.cache = cache
        self.forced: Counter = Counter()
        self.call: dict | None = None
        self.errors = 0
        self.file = self.errors_path = None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            self.file = open(out / f"adversary.{pid}.jsonl", "a", encoding="utf-8")
            self.errors_path = out / f"adversary_errors.{pid}.txt"

    def error(self, where: str) -> None:
        self.errors += 1
        if self.errors_path is not None and self.errors <= MAX_ERRORS:
            with open(self.errors_path, "a", encoding="utf-8") as f:
                f.write(f"--- {where} (error {self.errors})\n{traceback.format_exc()}\n")

    def preempt(self, sched) -> list:
        if not sched.running or self.rng.random() >= P_PREEMPT:
            return []
        eligible = [r for r in sched.running
                    if self.forced[r.request_id] < MAX_FORCED and getattr(r.status, "name", None) == "RUNNING"]
        k = self.rng.randint(1, max(1, int(len(sched.running) * PREEMPT_SHARE)))
        victims = self.rng.sample(eligible, min(k, len(eligible)))
        now = time.monotonic()
        drop = bool(getattr(sched, "requires_kv_delivery", False))
        for r in victims:
            sched.running.remove(r)
            sched._preempt_request(r, now, drop_stale_output=drop)
            self.forced[r.request_id] += 1
        return victims

    def before(self, sched) -> list:
        """Preempt, shuffle and hold back; returns the held-back requests, in queue order."""
        call = self.call = {"preempted": [], "held": 0, "admissible": 0, "splits": [], "hits": []}
        victims = []
        try:
            victims = self.preempt(sched)
        except Exception:  # noqa: BLE001 -- the engine must serve on
            self.error("preempt")
        call["preempted"] = [client_id(r.request_id) for r in victims]
        self.rng.shuffle(sched.running)
        waiting = list(sched.waiting)
        if not waiting:
            return []
        out = set(victims)
        share = self.rng.random()
        held = [r for r in waiting if r in out or self.rng.random() < share]
        out.update(held)
        keep = [r for r in waiting if r not in out]
        if not keep and not sched.running and (free := [r for r in held if r not in out]):
            back = self.rng.choice(free)
            held.remove(back)
            keep.append(back)
        self.rng.shuffle(keep)
        sched.waiting.remove_requests(waiting)
        for r in keep:
            sched.waiting.add_request(r)
        call["held"], call["admissible"] = len(held), len(keep)
        return held

    def after(self, sched, held: list) -> None:
        for r in reversed(held):
            sched.waiting.prepend_request(r)
        call, self.call = self.call, None
        if self.file is not None and call is not None and (call["preempted"] or call["admissible"] or call["held"]
                                                            or call["splits"] or call["hits"] or sched.running):
            call["step"] = getattr(sched, "current_step", None)
            self.file.write(json.dumps(call, separators=(",", ":")) + "\n")
            self.file.flush()

    def split(self, request, n: int) -> int:
        if n <= 1 or self.rng.random() >= P_SPLIT:
            return n
        m = self.rng.randint(1, n - 1)
        if self.call is not None:
            self.call["splits"].append([client_id(request.request_id), n, m])
        return m

    def hit(self, sched, request, found: tuple) -> tuple:
        blocks, n, boundary, diverged = found
        used = found
        if n > 0 and self.cache == "refuse" and self.rng.random() < P_REFUSE:
            used = (sched.kv_cache_manager.empty_kv_cache_blocks, 0, boundary, False)
        elif self.cache == "shortest" and n > sched.block_size:
            used = (sched.kv_cache_manager.truncate_computed_blocks(blocks, sched.block_size), sched.block_size, boundary,
                    False)
        if n > 0 and self.call is not None:
            self.call["hits"].append([client_id(request.request_id), n, used[1]])
        return used


def _adversary(sched) -> Adversary:
    a = getattr(sched, "_verity_adversary", None)
    if a is None:
        out = os.environ.get(TRACE_DIR)
        a = sched._verity_adversary = Adversary(int(os.environ[SEED]), os.environ.get(CACHE, "natural"),
                                                Path(out) if out else None)
    return a


def install(cls) -> None:
    """Wrap `cls`'s schedule, _reserve_prefill_lookahead and _get_local_prefix_cache_hit (once)."""
    if getattr(cls, "_verity_adversarial", False):
        return
    schedule, lookahead, lookup = cls.schedule, cls._reserve_prefill_lookahead, cls._get_local_prefix_cache_hit

    def hostile_schedule(self, *args, **kwargs):
        a = _adversary(self)
        held = []
        try:
            held = a.before(self)
        except Exception:  # noqa: BLE001
            a.error("schedule")
        try:
            return schedule(self, *args, **kwargs)
        finally:
            try:
                a.after(self, held)
            except Exception:  # noqa: BLE001
                a.error("after schedule")

    def hostile_lookahead(self, request, num_computed_tokens, num_new_tokens):
        a = _adversary(self)
        try:
            num_new_tokens = a.split(request, num_new_tokens)
        except Exception:  # noqa: BLE001
            a.error("_reserve_prefill_lookahead")
        return lookahead(self, request, num_computed_tokens, num_new_tokens)

    def hostile_lookup(self, request):
        found = lookup(self, request)
        a = _adversary(self)
        try:
            return a.hit(self, request, found)
        except Exception:  # noqa: BLE001
            a.error("_get_local_prefix_cache_hit")
            return found

    cls.schedule = hostile_schedule
    cls._reserve_prefill_lookahead = hostile_lookahead
    cls._get_local_prefix_cache_hit = hostile_lookup
    cls._verity_adversarial = True


def register() -> None:
    if os.environ.get(SEED):
        from vllm.v1.core.sched.scheduler import Scheduler
        install(Scheduler)
