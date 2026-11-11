"""Opened values: the one byte source of the value checks.

A value check reads committed bytes through an `OpenedReader`: the leaves that hold the bytes are opened from the committer
(`open_range`) and verified against the run commitment of record (`verify_range`) before a byte is returned.  A read that does not verify
raises `OpeningNotVerified`, which every check turns into a named FAIL, never 'not evaluated'.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Iterable

ORDINAL_SEP = "#ord"      # `<member name>#ord<tensor_ordinal>`: the reader's exact meta of a shared module's k-th invocation


def meta_by_name(metas: list, name: str) -> Any:
    """The committed meta a reader is asked for by member name, or None.
    A SHARED module (vLLM's one RotaryEmbedding, called once per layer) publishes one meta per INVOCATION under the SAME
    member name; the binding map binds each as its own identity (invocation k, tensor_ordinal = that meta's ordinal).  A reader asked by name
    alone can only return the FIRST meta's bytes (layer 0's words for every layer: on OLMoE every RoPE_v1 / Attention_v3 row past
    layer 0 is unequal).  `<name>#ord<ordinal>` selects the meta EXACTLY.  A BARE name with
    more than one meta at the step is a shared member asked without its ordinal: there is no first-meta answer to give (that read IS the
    alias class) -- None by name, so a third caller cannot regress silently."""
    base, _sep, ordinal = name.partition(ORDINAL_SEP)
    if _sep:
        return next((x for x in metas if x.name == base and int(getattr(x, "ordinal", -1)) == int(ordinal)), None)
    same = [x for x in metas if x.name == base]
    return same[0] if len(same) == 1 else None


class OpeningNotVerified(RuntimeError):
    """A value a check was about to consume did not verify against the run root.  The check's result is False (named), never 'not
    evaluated': the committer's memory no longer holds what the root commits to."""


OPENED_METHOD = ("every byte a value check consumes is opened from the committer (`open_range`: the retained copy's leaves + the boundary "
                 "siblings of its tree) and verified against the ORIGINAL run commitment (`verify_range`: leaf digests recomputed, folded to "
                 "the step root, step root in the run root) before it is compared; a read that does not verify fails the check by name")


class OpenedReader:
    """The value checks' one byte source.  `reader(step, name, lo, hi)` -> bytes | None, `reader.read_meta(step, meta, lo, hi)` for readers
    that hold the meta: the bytes [lo, hi) of the member (clamped to the member), taken from an opening of every leaf that holds them, verified
    against `run` (the run commitment of record) before they are returned.  None = the member is not committed at the step or its bytes are
    not retained here (retain=none / sparse retention: they open by replay after release) -- not compared.  A read whose opening does not
    verify raises `OpeningNotVerified`; unverified bytes are never returned.  The first verified read of a step authenticates its bare root
    (context binding + layout digest + run-root membership, once per step); every read re-hashes its own leaves."""

    def __init__(self, com: Any, run: Any) -> None:
        self.com, self.run = com, run
        self._roots: dict[int, bytes] = {}
        self._own: dict[int, set[int]] = {}
        self.stats: Counter = Counter()
        self.seconds = 0.0
        self.failures: list[dict[str, Any]] = []

    def __call__(self, step: int, name: str, lo: int, hi: int) -> bytes | None:
        metas = (getattr(self.com, "_layouts", None) or {}).get(step)
        m = meta_by_name(metas, name) if metas else None
        return None if m is None else self.read_meta(step, m, lo, hi)

    def read_meta(self, step: int, m: Any, lo: int = 0, hi: int | None = None) -> bytes | None:
        own = self._own.get(step)
        if own is None:
            own = self._own[step] = {id(x) for x in (getattr(self.com, "_layouts", None) or {}).get(step, [])}
        if id(m) not in own:
            raise ValueError(f"step {step} {getattr(m, 'name', m)!r}: not a meta of this committer's layout at the step (its offsets would "
                             "open another tensor's leaves)")
        t0 = time.perf_counter()
        try:
            lo, hi = max(0, int(lo)), int(m.nbytes) if hi is None else min(int(hi), int(m.nbytes))
            if hi <= lo:
                return b""
            a, b, off = self.com.leaf_span(step, m, lo, hi)
            ro = self.com.open_range(step, a, b)
            if ro is None:
                self.stats["not_retained"] += 1
                return None
            if not self.com.verify_range(self.run, ro, self._roots):
                self.stats["failed"] += 1
                if len(self.failures) < 10:
                    self.failures.append({"step": step, "name": m.name, "ordinal": int(getattr(m, "ordinal", -1)), "byte_lo": lo, "byte_hi": hi,
                                          "leaves": [a, b]})
                raise OpeningNotVerified(f"step {step} {m.name} (ordinal {getattr(m, 'ordinal', '?')}) bytes [{lo}, {hi}) = leaves [{a}, {b}): "
                                         "the opening does not verify against the run root")
            self.stats["reads"] += 1
            self.stats["leaves"] += b - a
            self.stats["bytes"] += hi - lo
            return ro.value[off:off + hi - lo]
        finally:
            self.seconds += time.perf_counter() - t0

    def or_none(self, step: int, name: str, lo: int, hi: int) -> bytes | None:
        """For a consumer that fails an unread entry by name: a read that does not verify is None (and counted in `failed`)."""
        try:
            return self(step, name, lo, hi)
        except OpeningNotVerified:
            return None

    def record(self) -> dict[str, Any]:
        return {"source": "opened", "method": OPENED_METHOD, "run_root": bytes(self.run.run_root).hex(), "reads": int(self.stats["reads"]),
                "leaves": int(self.stats["leaves"]), "bytes": int(self.stats["bytes"]), "not_retained": int(self.stats["not_retained"]),
                "failed": int(self.stats["failed"]), "steps_bound": len(self._roots), "seconds": round(self.seconds, 3),
                "failures": list(self.failures)}


def opened_check(reader: Any, check: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """`check(rec)` on a fresh record.  A read whose opening does not verify makes the record a mismatch (result False), named, keeping what
    `check` had filled in.  With an `OpenedReader`, `_opened` carries the reads made during the call: a forked evaluator's only way to
    report them (`opened_in_children`)."""
    rec: dict[str, Any] = {}
    stats = getattr(reader, "stats", None)
    n0, s0 = (dict(stats), reader.seconds) if stats is not None else ({}, 0.0)
    try:
        rec = check(rec)
    except OpeningNotVerified as e:
        rec = rec | {"result": False, "why": f"opened value not verified: {e}"[:300]}
    if stats is not None:
        rec["_opened"] = {k: int(v) - int(n0.get(k, 0)) for k, v in stats.items()} | {"seconds": reader.seconds - s0}
    return rec


def opened_in_children(results: Iterable[tuple[Any, dict[str, Any], Any]], *, forked: bool) -> dict[str, Any]:
    """Pops every `(item, record, _)` record's `_opened` and, when the records came from forked evaluators, sums them ({} otherwise: the
    parent's reader counted its own reads)."""
    tot: Counter = Counter()
    for _item, rec, _dt in results:
        d = rec.pop("_opened", None)
        if d and forked:
            tot.update(d)
    return {k: (round(v, 3) if k == "seconds" else int(v)) for k, v in tot.items()}


class UnverifiedUnlinked:
    """A store whose `words` answers (None, why) for a read whose opening does not verify: the linkage names it unlinked (result False)."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def words(self, *args: Any) -> tuple[Any, str]:
        try:
            return self._store.words(*args)
        except OpeningNotVerified as e:
            return None, f"opened value not verified: {e}"


def filed(commit_row: dict[str, Any], check: str, record: dict[str, Any]) -> str:
    """Files `record` (an `OpenedReader.record()`) as `commit_row["opened_values"][check]`; returns its one-line account for the log."""
    commit_row.setdefault("opened_values", {"method": OPENED_METHOD})[check] = record
    kids = f"; forked evaluators {record['forked_children']}" if record.get("forked_children") else ""
    return (f"{record.get('reads')} reads, {record.get('leaves')} leaves, {record.get('bytes')} B verified against the run root in "
            f"{record.get('seconds')} s ({record.get('steps_bound')} steps bound); not retained {record.get('not_retained')}, failed "
            f"{record.get('failed')}{kids}")
