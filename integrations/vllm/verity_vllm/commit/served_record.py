"""The served Commit's record (I3): what each request's committed values are, by position, whatever vLLM's scheduler did.

A committing engine under arbitrary traffic runs a request in rows the scheduler picks while serving: a whole prompt or a
chunk of it, one decode token a step, a recompute from position 0 after a preemption, a suffix after a prefix-cache hit, and,
under async scheduling, a row past the token that finished it (an EOS, a stop token or string the scheduler hadn't seen
yet). None of that is a rule here; each is a fact the record states. A `NativeHostCommitter` with a `SamplerInputsSource`
attached keeps each step's rows (`Row`: the runner's request id, its token rows [lo, hi) of the step, its first position and
its prefill length) and binds, beside the sampled ids, the seed and the position each row's draw read (`SAMPLER_INPUTS`). `Record.absorb`
reads a finished window back from the committed bytes into one `RequestRecord` per request:

- ``inputs``: position p -> the committed input id at p (the step-input port's ``step.input/ids``, else ``c00.input/ids`` on
  the compiled path or ``model.embed_tokens/in`` on the eager one), every row that computed p; a recompute must commit the
  same id again;
- ``sampled``: position p -> the committed id sampled from p's logits, the token at p + 1, for every row whose last position
  reaches its prefill length (Model Runner V2 samples every row and drops a partial prefill's; those are ``discarded``);
- ``seeds``, ``rows``, the prefix-cache hit (a first row starting past 0) and the preemptions (a row starting before the
  request's computed end).

`RequestRecord.check` holds that to the streamed I/O: with t = prompt ids + streamed ids, P prompt ids and K streamed,
the committed inputs at every position in [hit, P + K - 1) are t's, the committed samples at every position in
[P - 1, P + K - 1) are the next of t's, every seed is the one the request carried (the client's, or the served default,
`engine.served_seed`), and a cached prefix [0, hit) is other requests' committed inputs, each committed no later than the
request's first row by a request whose committed ids agree with t up to there (`RequestRecord.prefix_cover`: a turn's
prefix can be its first turn's prompt and its previous turn's suffix). Rows past the finish
(positions from P + K - 1 on) are ``overrun``: computed and committed, never streamed, each input there the previous
committed sample. Given the request's finish, the stream must also end where it says (`finish_problems`), so a stream
cut short can't pass with its missing ids taken for an overrun; a client's abort ends it anywhere short of ``max_tokens``,
and then at most one committed sample runs past it (the step in flight when the abort was applied, under async scheduling).
A request aborted before it was ever scheduled streamed nothing and has no committed row, and `Record.check` passes it.

Stdlib only, but `committed_ints`, which reads a committer's retained bytes with numpy (and torch for a GPU block).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

SCHEMA = "verity-vllm/served-record/v1"
#: The step-input port's two tensors, the first two of every step's stream (`StepInputPort`).
PORT_NAMES = ("step.input/ids", "step.input/positions")
#: The step's input ids, in the order they are read: the step-input port's, then the compiled path's name, then the eager
#: path's (`NativeHostCommitter`'s ``<module>/in`` pre-hook on the ``VocabParallelEmbedding``).
IDS_NAMES = (PORT_NAMES[0], "c00.input/ids", "model.embed_tokens/in")
SAMPLED = "runner.sampler/sampled_token_ids"
#: Each sampled row's seed and the position its draw read, one i64 a row (`SamplerInputsSource`).
SAMPLER_INPUTS = ("runner.sampler/seed", "runner.sampler/pos")
_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def base_id(runner_id: str, known: Mapping[str, Any] | None = None) -> str:
    """The request a runner id names: itself when ``known`` holds it, else without vLLM's ``-<8 hex>`` suffix (the V1
    input processor's internal id)."""
    if known is not None and runner_id in known:
        return runner_id
    return _SUFFIX.sub("", runner_id)


@dataclass(frozen=True)
class Row:
    """One request's rows in one step, as the model runner laid them out: token rows [lo, hi) of the step's flattened
    batch, from position ``start`` (its computed tokens before the step), of a request whose prefill is ``prefill_len``
    tokens (its prompt, or prompt and outputs on a resume)."""

    request: str
    lo: int
    hi: int
    start: int
    prefill_len: int

    @property
    def n(self) -> int:
        return self.hi - self.lo

    @property
    def last(self) -> int:
        return self.start + self.n - 1

    @property
    def samples(self) -> bool:
        """Its sampled id is an output token: the row's end reaches the prefill (vLLM: ``seq_len >= prefill_len``)."""
        return self.start + self.n >= self.prefill_len

    @property
    def kind(self) -> str:
        if self.start + self.n < self.prefill_len:
            return "prefill-chunk"
        return "decode" if self.n == 1 and self.start >= self.prefill_len else "prefill"


class SamplerInputsSource:
    """The committer source (`NativeHostCommitter.attach_source`) that, on each sampled step, binds the seed and the position
    each row's draw read (Model Runner V2's `gumbel_sample`: ``seeds[idx_mapping]`` and ``positions[logits_indices]``), one
    i64 a row under `SAMPLER_INPUTS`, and keeps the step's rows in ``rows`` (`Row`'s fields, keyed by the committer's step;
    a window's step 0 starts them afresh)."""

    short_name = "sampler_inputs"
    own_names = frozenset(SAMPLER_INPUTS)

    def __init__(self, committer: Any) -> None:
        self.com = committer
        self.rows: dict[int, list[tuple]] = {}

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def on_step_begin(self) -> None:
        pass

    def extra_step_bytes(self, envelope: Any, model: Any) -> dict[str, int]:
        return {"per_seq_bytes": 8 * len(SAMPLER_INPUTS), "tensors_per_step": len(SAMPLER_INPUTS)}

    def on_sample(self, ib: Any, n_sampled: int, job: Any) -> None:
        com = self.com
        if job.step == 0:
            self.rows = {}
        ss = getattr(getattr(com.runner, "sampler", None), "sampling_states", None)
        if ib is None or ss is None or not hasattr(ss, "seeds"):
            com.errors.append(f"sampler inputs: step {job.step} has no input batch or sampling states to read")
            return
        n = int(ib.num_reqs)
        if n != n_sampled:
            com.errors.append(f"sampler inputs: step {job.step} samples {n_sampled} rows for {n} requests")
            return
        com._acquire(SAMPLER_INPUTS[0], ss.seeds.gpu[ib.idx_mapping.long()], job)
        com._acquire(SAMPLER_INPUTS[1], ib.positions[ib.logits_indices].long(), job)
        qsl = [int(x) for x in ib.query_start_loc_np[: n + 1]]
        self.rows[job.step] = [(str(ib.req_ids[i]), qsl[i], qsl[i + 1], int(ib.num_computed_tokens_np[i]), int(ib.prefill_len_np[i]))
                               for i in range(n)]


class StepInputPort:
    """The committer source that starts every step's stream with the step-input port the certifier's served-rows tie and
    IO-TIE's ``IoTieRequest`` read (`verity.ml.protocols.certifier.served_tie`): the step's input ids, then its positions,
    each ``T`` little-endian i64s (IB = PB = 8), zero past the step's N scheduled tokens, ``T`` the engine's token budget
    (``max_num_batched_tokens``). So the ids are the stream's bytes [0, 8T) and the positions [8T, 16T), at fixed leaves of
    the step's tree when 8T is a whole number of chunks (refused otherwise). A pre-hook on the top model takes them from
    its call (``input_ids``, ``positions``) before any module of it commits."""

    short_name = "step_input_port"
    own_names = frozenset(PORT_NAMES)
    IB = PB = 8

    def __init__(self, committer: Any, T: int) -> None:
        chunk = int(getattr(committer, "chunk", 4096) or 4096)
        if T < 1 or (T * self.IB) % chunk:
            raise ValueError(f"step-input port: T = {T} i64s is not a whole number of {chunk}-byte chunks; "
                             f"pass --max-num-batched-tokens a multiple of {chunk // self.IB}")
        self.com, self.T, self._h = committer, int(T), None

    def describe(self) -> dict[str, Any]:
        """The port as a sealed window's context states it: names, T, the element bytes and each tensor's byte range."""
        T, IB, PB = self.T, self.IB, self.PB
        return {"names": list(PORT_NAMES), "T": T, "IB": IB, "PB": PB, "ids_bytes": [0, T * IB], "positions_bytes": [T * IB, T * (IB + PB)]}

    def install(self) -> None:
        self._h = self.com.model.register_forward_pre_hook(self._pre, with_kwargs=True)

    def uninstall(self) -> None:
        if self._h is not None:
            self._h.remove()
            self._h = None

    def on_step_begin(self) -> None:
        pass

    def extra_step_bytes(self, envelope: Any, model: Any) -> dict[str, int]:
        return {"per_step_bytes": self.T * (self.IB + self.PB), "tensors_per_step": len(PORT_NAMES)}

    def _pre(self, _m: Any, args: tuple, kwargs: dict) -> None:
        import torch

        com, job = self.com, getattr(self.com, "_cur", None)
        if job is None:
            return None
        ids = kwargs.get("input_ids", args[0] if args else None)
        pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
        n = int((getattr(com, "_plan_shape", None) or (0, 0))[0])
        if ids is None or pos is None or ids.dim() != 1 or pos.dim() != 1 or not 0 < n <= min(self.T, int(ids.shape[0]), int(pos.shape[0])):
            com.errors.append(f"step-input port: step {job.step} has N = {n} tokens for T = {self.T}, "
                              f"ids {None if ids is None else tuple(ids.shape)}, positions {None if pos is None else tuple(pos.shape)}")
            return None
        for name, t in zip(PORT_NAMES, (ids, pos)):
            out = torch.zeros(self.T, dtype=torch.int64, device=t.device)
            out[:n] = t[:n].long()
            com._acquire(name, out, job)
        return None


def served_rows(committer: Any) -> dict[int, list[tuple]]:
    """The rows its `SamplerInputsSource` kept, by step; empty without one."""
    src = next((s for s in getattr(committer, "sources", ()) if isinstance(s, SamplerInputsSource)), None)
    return src.rows if src is not None else {}


def committed_ints(committer: Any, step: int, name: str) -> list[int] | None:
    """The integer tensor ``name`` as committed at ``step``, read back from the bytes the committer retains (the retained
    GPU-tree block, its host copy, or the packed step stream); None when the step committed no such tensor."""
    got = committed_raw(committer, step, name)
    return None if got is None else ints_of(*got)


def ints_of(raw: bytes, dtype: str) -> list[int]:
    """A committed integer tensor's bytes (`committed_raw`) as ints."""
    import numpy as np

    return np.frombuffer(raw, dtype=np.dtype(str(dtype).rsplit(".", 1)[-1])).astype(np.int64).tolist()


def committed_raw(committer: Any, step: int, name: str) -> tuple[bytes, str] | None:
    """`committed_ints`' bytes of ``name`` at ``step`` and its dtype. A plan's metas (`plan_cache.PlanMetas`) are read from its
    columns, so no `TensorMeta` is built for the read."""
    from verity_vllm.commit.committer.plan_cache import PlanMetas

    metas = committer._layouts.get(step, ())
    if isinstance(metas, PlanMetas):
        if name not in metas.names:
            return None
        i = metas.names.index(name)
        meta: Any = SimpleNamespace(dtype=metas.dtypes[i], nbytes=metas.sizes[i], dev_off=metas.offsets[i], stream_off=metas.offsets[i])
    else:
        meta = next((m for m in metas if m.name == name), None)
    if meta is None:
        return None
    raw: Any = None
    blk = (getattr(committer, "_gpu_blocks", {}) or {}).get(step, (None, 0))[0]
    if blk is not None:
        off = int(meta.dev_off) if int(getattr(meta, "dev_off", -1)) >= 0 else int(meta.stream_off)
        part = blk[off: off + meta.nbytes]
        raw = (part.cpu() if hasattr(part, "cpu") else part).numpy().tobytes()
    elif getattr(meta, "host", None) is not None:
        raw = bytes(meta.host)
    elif step in (getattr(committer, "_streams", {}) or {}) and int(meta.stream_off) >= 0:
        raw = bytes(committer._streams[step][meta.stream_off: meta.stream_off + meta.nbytes])
    return None if raw is None else (raw, str(meta.dtype))


@dataclass
class RequestRecord:
    """One request's committed values by position (the module docstring)."""

    request: str
    inputs: dict[int, int] = field(default_factory=dict)
    sampled: dict[int, int] = field(default_factory=dict)
    discarded: dict[int, int] = field(default_factory=dict)
    seeds: set[int] = field(default_factory=set)
    rows: list[list[Any]] = field(default_factory=list)     # [window, step, lo, hi, start, prefill_len, kind]
    runner_ids: list[str] = field(default_factory=list)
    hit: int | None = None
    preemptions: int = 0
    recomputed: int = 0
    end: int = 0                                            # positions computed so far
    conflicts: list[str] = field(default_factory=list)
    first_at: dict[int, tuple[int, int]] = field(default_factory=dict)   # position -> (window, step) it was first committed

    def add(self, window: int, step: int, row: Row, ids: Sequence[int], sampled: int, seed: int | None) -> None:
        if self.hit is None:
            self.hit = row.start
        elif row.start < self.end:
            self.preemptions += 1
        for k, tok in enumerate(ids):
            p = row.start + k
            if p in self.inputs:
                self.recomputed += 1
                if self.inputs[p] != tok:
                    self.conflicts.append(f"position {p}: committed input {tok} at step {step}, {self.inputs[p]} before")
            else:
                self.first_at[p] = (window, step)
            self.inputs[p] = int(tok)
        if row.samples:
            if row.last in self.sampled and self.sampled[row.last] != sampled:
                self.conflicts.append(f"position {row.last}: committed sample {sampled} at step {step}, {self.sampled[row.last]} before")
            self.sampled[row.last] = int(sampled)
        else:
            self.discarded[row.last] = int(sampled)
        if seed is not None:
            self.seeds.add(int(seed))
        self.end = row.start + row.n
        self.rows.append([window, step, row.lo, row.hi, row.start, row.prefill_len, row.kind])

    def check(self, prompt_ids: Sequence[int], streamed_ids: Sequence[int], *, seed: int | None,
              prefix_sources: Mapping[str, RequestRecord] | None = None, finish: Mapping[str, Any] | None = None,
              decode: Callable[[Sequence[int]], str] | None = None) -> dict[str, Any]:
        """The record against what was served (the module docstring); ``seed`` the seed the request carried into the
        engine, ``prefix_sources`` the records a cached prefix may have come from, ``finish`` how the stream ended
        (`finish_problems`)."""
        t = [int(x) for x in prompt_ids] + [int(x) for x in streamed_ids]
        P, K = len(prompt_ids), len(streamed_ids)
        hit = int(self.hit or 0)
        problems: list[str] = list(self.conflicts)
        if K == 0:
            problems.append("nothing was streamed")
        if finish is not None and K:
            problems.extend(finish_problems(t[P:], finish, decode))
        if not self.rows:
            problems.append("no committed row")
        last = P + K - 1                                    # the last streamed token's position: never fed back but in an overrun

        def first_bad(want: range, got: Mapping[int, int], of: Any, what: str) -> None:
            missing = [p for p in want if p not in got]
            wrong = [p for p in want if p in got and got[p] != of(p)]
            if missing:
                problems.append(f"{what}: {len(missing)} position(s) not committed, from {missing[0]}")
            if wrong:
                p = wrong[0]
                problems.append(f"{what}: {len(wrong)} position(s) differ, first {p}: committed {got[p]}, served {of(p)}")

        first_bad(range(hit, last), self.inputs, lambda p: t[p], "inputs")
        first_bad(range(P - 1, last), self.sampled, lambda p: t[p + 1], "samples")
        early = sorted(p for p in self.sampled if p < P - 1)
        if early:
            problems.append(f"samples: an output sampled inside the prompt, at position {early[0]}")
        over_in = sorted(p for p in self.inputs if p >= last)
        over_s = sorted(p for p in self.sampled if p >= last)
        if finish is not None and finish.get("reason") == "abort" and len(over_s) > 1:
            problems.append(f"finish: aborted after {K} ids, and {len(over_s)} committed samples run past them, more than the one "
                            "of the step in flight")
        for p in over_in:
            want = t[p] if p == last else self.sampled.get(p - 1)
            if self.inputs[p] != want:
                problems.append(f"overrun: the input committed at position {p} is {self.inputs[p]}, not the token before it ({want})")
                break
        if seed is not None and self.seeds - {int(seed)}:
            problems.append(f"seeds: committed {sorted(self.seeds)}, the request carried {seed}")
        spans: list[list[Any]] = []
        if hit:
            spans, gap = self.prefix_cover(t, hit, prefix_sources or {})
            if gap is not None:
                problems.append(f"prefix cache: position {gap} of [0, {hit}) was read from cache, and no other request committed "
                                "it by then with the served ids up to it")
        return {"passed": not problems, "prompt_ids": P, "streamed_ids": K, "cache_hit": hit,
                "prefix_source": spans[0][0] if spans else None, "prefix_spans": spans,
                "rows": len(self.rows), "preemptions": self.preemptions, "recomputed_positions": self.recomputed,
                "discarded_samples": len(self.discarded), "overrun_rows": len(over_s), "seeds": sorted(self.seeds),
                "problems": problems[:8]}

    def prefix_cover(self, t: Sequence[int], hit: int, sources: Mapping[str, RequestRecord]) -> tuple[list[list[Any]], int | None]:
        """Who computed the cached prefix [0, ``hit``) of the served ids ``t``: spans ``[request, lo, hi)`` of other requests that
        committed ``t``'s ids there no later than this request's first row, each with no committed id before ``hi`` that
        differs from ``t``'s (a cached block's KV is its whole prefix's); the first position no span covers, else None. A
        turn's prefix can be several requests' (its first turn's prompt, its previous turn's suffix)."""
        first = tuple(self.rows[0][:2]) if self.rows else (0, 0)
        reach: dict[str, int] = {}
        for rid, r in sources.items():
            if rid == self.request:
                continue
            reach[rid] = min((q for q, v in r.inputs.items() if q < hit and v != t[q]), default=hit)

        def covers(rid: str, p: int) -> bool:
            r = sources[rid]
            return p < reach[rid] and p in r.inputs and r.first_at.get(p, first) <= first

        spans: list[list[Any]] = []
        p = 0
        while p < hit:
            best, best_hi = None, p
            for rid in reach:
                hi = p
                while hi < hit and covers(rid, hi):
                    hi += 1
                if hi > best_hi:
                    best, best_hi = rid, hi
            if best is None:
                return spans, p
            spans.append([best, p, best_hi])
            p = best_hi
        return spans, None

    def to_json(self) -> dict[str, Any]:
        """The record as I3 publishes it: rows, and the committed ids by position."""
        return {"request": self.request, "runner_ids": self.runner_ids, "cache_hit": self.hit, "preemptions": self.preemptions,
                "rows": self.rows, "inputs": [[p, v] for p, v in sorted(self.inputs.items())],
                "sampled": [[p, v] for p, v in sorted(self.sampled.items())],
                "discarded": [[p, v] for p, v in sorted(self.discarded.items())], "seeds": sorted(self.seeds)}


class Record:
    """Every request's `RequestRecord` over the windows of one committing engine. ``known`` maps a request id to anything
    (the admitted requests); a runner id is matched to it by `base_id`."""

    def __init__(self, known: Mapping[str, Any] | None = None) -> None:
        self.known = known
        self.requests: dict[str, RequestRecord] = {}
        self.windows: list[dict[str, Any]] = []

    def absorb(self, committer: Any, window: int) -> dict[str, Any]:
        """The window just finalized by ``committer`` into the requests' records; returns the window's readback record."""
        from verity_vllm.commit.committer.plan_cache import PlanMetas

        layouts = getattr(committer, "_layouts", {}) or {}
        names = {n for metas in layouts.values() for n in (metas.names if isinstance(metas, PlanMetas) else (m.name for m in metas))}
        ids_name = next((n for n in IDS_NAMES if n in names), None)
        errors: list[str] = []
        rows_of = served_rows(committer)
        if layouts and not rows_of:
            errors.append("the committer kept no rows (no SamplerInputsSource attached)")
        if ids_name is None and rows_of:
            errors.append(f"no committed input ids under any of {list(IDS_NAMES)}")
        held: list[str] = []
        steps = sorted(set(layouts) | set(rows_of))
        for s in steps:
            rows = [Row(*r) for r in rows_of.get(s, ())]
            if not rows:
                errors.append(f"step {s}: committed with no rows")
                continue
            ids = committed_ints(committer, s, ids_name) if ids_name else None
            sampled = committed_ints(committer, s, SAMPLED)
            seeds = committed_ints(committer, s, SAMPLER_INPUTS[0])
            pos = committed_ints(committer, s, SAMPLER_INPUTS[1])
            gone = [n for n, v in ((ids_name, ids), (SAMPLED, sampled), *zip(SAMPLER_INPUTS, (seeds, pos))) if v is None]
            if gone:
                errors.append(f"step {s}: nothing committed under {gone}")
                continue
            if not (len(sampled) == len(seeds) == len(pos) == len(rows)) or rows[-1].hi > len(ids):
                errors.append(f"step {s}: {len(rows)} rows over {rows[-1].hi} tokens, committed {len(ids)} ids, {len(sampled)} samples, "
                              f"{len(seeds)} seeds, {len(pos)} positions")
                continue
            for i, row in enumerate(rows):
                if pos[i] != row.last:
                    errors.append(f"step {s} row {i}: the draw read position {pos[i]}, the row ends at {row.last}")
                rid = base_id(row.request, self.known)
                if self.known is not None and rid not in self.known:
                    errors.append(f"step {s}: runner request {row.request} is no admitted request")
                    continue
                rec = self.requests.setdefault(rid, RequestRecord(rid))
                if row.request not in rec.runner_ids:
                    rec.runner_ids.append(row.request)
                rec.add(window, s, row, ids[row.lo:row.hi], sampled[i], seeds[i])
                if rid not in held:
                    held.append(rid)
        rec = {"window": window, "ids_name": ids_name, "steps": len(steps), "requests": held, "readback_errors": errors[:10],
               "readback_errors_n": len(errors), "passed": not errors}
        self.windows.append(rec)
        return rec

    def check(self, request: str, prompt_ids: Sequence[int], streamed_ids: Sequence[int], *, seed: int | None,
              finish: Mapping[str, Any] | None = None, decode: Callable[[Sequence[int]], str] | None = None) -> dict[str, Any]:
        rec = self.requests.get(request)
        if rec is None:
            gone = finish is not None and finish.get("reason") == "abort" and not streamed_ids and not finish.get("k")
            return {"passed": gone, "problems": [] if gone else ["no committed row"], "prompt_ids": len(prompt_ids),
                    "streamed_ids": len(streamed_ids)}
        return rec.check(prompt_ids, streamed_ids, seed=seed, prefix_sources=self.requests, finish=finish, decode=decode)


def finish_problems(out: Sequence[int], finish: Mapping[str, Any], decode: Callable[[Sequence[int]], str] | None = None) -> list[str]:
    """Whether the stream ``out`` ends where its finish says, so that a stream cut short can't pass as a finished one whose
    committed rows past the cut are an async overrun. ``finish``: ``reason`` (vLLM's ``finish_reason``), ``stop_reason``,
    ``max_tokens`` and ``eos_ids``. A ``length`` finish streamed ``max_tokens`` ids; a client's ``abort`` streamed the ``k``
    it records, fewer than ``max_tokens`` (at its cap vLLM has finished it on length); a ``stop`` on the model's EOS
    (``stop_reason`` None) ends on one of ``eos_ids``, on a stop token (an int) ends on it, and on a stop string ends on
    the id whose text first holds it (with ``decode``; vLLM keeps every id of a stop-string finish and cuts the text)."""
    reason, why, cap = finish.get("reason"), finish.get("stop_reason"), finish.get("max_tokens")
    if reason == "length":
        return [] if cap is None or len(out) == int(cap) else [f"finish: length after {len(out)} ids, max_tokens {cap}"]
    if reason == "abort":
        k = finish.get("k")
        if k is not None and len(out) != int(k):
            return [f"finish: aborted after {k} ids, {len(out)} streamed"]
        return [] if cap is None or len(out) < int(cap) else [f"finish: aborted after {len(out)} ids, at its max_tokens {cap}"]
    if reason != "stop":
        return [f"finish: {reason!r}, neither stop, length nor abort"]
    if why is None:
        eos = {int(x) for x in finish.get("eos_ids") or ()}
        return [] if not eos or out[-1] in eos else [f"finish: stopped on EOS, the last streamed id {out[-1]} is none of {sorted(eos)}"]
    if isinstance(why, int):
        return [] if out[-1] == why else [f"finish: stopped on token {why}, the last streamed id is {out[-1]}"]
    if decode is not None and stop_string_end(out, str(why), int(finish.get("min_tokens") or 0), decode) != len(out):
        return [f"finish: stop string {why!r} is not first matched at the last streamed id"]
    return []


def stop_string_end(out: Sequence[int], stop: str, min_tokens: int, decode: Callable[[Sequence[int]], str]) -> int | None:
    """How many ids vLLM's detokenizer streams before ``stop`` stops it: once more than ``min_tokens`` ids are out, each id's
    new text, with the ``len(stop) - 1`` characters before it, is searched for the string."""
    prev = decode(out[:min_tokens])
    for k in range(min_tokens + 1, len(out) + 1):
        text = decode(out[:k])
        new = len(text) - len(prev)
        if new > 0 and stop in text[-(new + len(stop) - 1):]:
            return k
        prev = text
    return None


def window_commit(rc: Any) -> dict[str, Any]:
    """The Commit's own verdict on a finalized window (`RunCommitment`): a run root, no committer errors, and the identity
    check not failed."""
    extra = getattr(rc, "extra", None) or {}
    ident = extra.get("identity_check") or {}
    errors = list(extra.get("errors") or [])
    root = bytes(getattr(rc, "run_root", b"") or b"")
    return {"passed": bool(root) and not errors and int(extra.get("errors_n") or 0) == 0 and ident.get("ok") is not False,
            "run_root": root.hex(), "steps": len(getattr(rc, "steps", ()) or ()), "errors": errors[:10],
            "errors_n": int(extra.get("errors_n") or 0),
            "identity_check": {k: ident.get(k) for k in ("ok", "steps_checked", "violations_n", "violations") if k in ident}}
