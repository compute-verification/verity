"""A client's abort on a compact run: the certified I/O (`certifier_serve.certified_io`) and the I/O tie's rule
(`active.compact_finish_violations`) on test_compact_eos_phase12's honest rehearsal, its target response (finish
`length` at `max_tokens`) cut short as the link carries one its client aborted: its kept records, none marked.

No record says an abort and the certifier can't see one on the link, so both take it from the Commit (`aborted`) and
accept the unmarked end only where the Commit's finish is an abort after 1 ≤ k < `max_tokens` outputs with no EOS id
among them; the advice charges that `output_len` in full. Refused: an unmarked end the Commit doesn't call an abort,
one at `max_tokens`, one on an EOS id, a marked end the Commit calls an abort, and (in the tie, which accepted it
before) a `COMPACT_LAST` short of `max_tokens`.
"""
from __future__ import annotations

from test_certifier_serve import cs
from test_compact_eos_phase12 import _caps, _cells, _cut, _sampled, run  # noqa: F401  (run: the rehearsal fixture)
from test_compact_phase10 import _as_id_events
from verity.core.protocols.certifier import active


def _io(run, outs, aborted=(), **plan) -> dict:
    return cs.certified_io({**run.plan, **plan}, run.setup, run.ins, outs, aborted)


def _finish(io: dict, stream: int) -> tuple[str, int]:
    mine = next(x for x in io["records"] if x["stream"] == stream)
    return mine["finish"], len(mine["output"])


def _tie(run, commit: dict[int, list[int]], outs, aborted=()) -> list[tuple[int, str]]:
    return active.compact_finish_violations(commit, run.eos, _cells(outs), max_tokens=_caps(run.honest),
                                            aborted=aborted)


def test_an_honest_abort_is_accepted_where_the_commit_says_so_and_only_there(run):
    """Cut after k + 1 of its outputs, k + 1 < `max_tokens`, the response is an abort the certified I/O accepts for
    the user the Commit says aborted and refuses for any other, as id events too; the tie, the Commit's ids cut there
    as well, accepts it for the ordinal the Commit says aborted and refuses it for any other."""
    t = run.target
    stream, user, full = t["stream"], t["user"], len(t["output"])
    for k in range(full - 1):
        cut = _cut(run.outs, stream, k, active.compact_record(stream, t["output"][k]))
        got = _io(run, cut, {user})
        assert got["accepted"] and _finish(got, stream) == ("abort", k + 1) and got["finish"]["abort"] == 1, k
        assert got["tokens"] == run.honest["tokens"] - (full - 1 - k)
        assert _io(run, cut)["violations"] == [[stream, "no finish"]], k
        ids = _io(run, _as_id_events(cut), {user}, compact=None)
        assert ids["accepted"] and _finish(ids, stream) == ("abort", k + 1), k
        assert _io(run, _as_id_events(cut), compact=None)["violations"] == [[stream, "short of max_tokens without EOS"]]
        commit = _sampled(run.honest) | {stream: t["output"][:k + 1]}
        assert _tie(run, commit, cut, {stream}) == [], k
        assert _tie(run, commit, cut) == [(stream, f"finish none on output {k} where the Commit's is length")], k


def test_an_abort_at_max_tokens_is_refused(run):
    t = run.target
    stream, user, full = t["stream"], t["user"], len(t["output"])
    cut = _cut(run.outs, stream, full - 1, active.compact_record(stream, t["output"][-1]))
    assert _io(run, cut, {user})["violations"] == [[stream, "an abort at max_tokens"]]
    assert _io(run, _as_id_events(cut), {user}, compact=None)["violations"] == \
        [[stream, "a finish length where the Commit's is abort"]]
    assert _tie(run, _sampled(run.honest), cut, {stream}) == \
        [(stream, f"an abort at {full} outputs where max_tokens is {full}")]


def test_an_abort_ending_on_an_eos_id_is_refused(run):
    """The response cut after k + 1 outputs, the last an EOS id with no mark: the model stopped there, so it is no
    abort, and the Commit that says it is one is refused too."""
    t, eos_id = run.target, min(run.eos)
    stream, user = t["stream"], t["user"]
    for k in (0, 2):
        cut = _cut(run.outs, stream, k, active.compact_record(stream, eos_id))
        assert _io(run, cut, {user})["violations"] == [[stream, "an EOS id outside a stop record"]], k
        assert _io(run, _as_id_events(cut), {user}, compact=None)["violations"] == \
            [[stream, "a finish eos where the Commit's is abort"]], k
        commit = _sampled(run.honest) | {stream: [*t["output"][:k], eos_id]}
        assert _tie(run, commit, cut, {stream}) == [(stream, "an abort on an EOS id")], k


def test_a_marked_end_the_commit_calls_an_abort_is_refused(run):
    """The honest records, the target ending on `COMPACT_LAST` at `max_tokens`, against a Commit that says its client
    aborted it."""
    t = run.target
    stream, full = t["stream"], len(t["output"])
    assert _io(run, run.outs, {t["user"]})["violations"] == [[stream, "a finish length where the Commit's is abort"]]
    assert _tie(run, _sampled(run.honest), run.outs, {stream}) == \
        [(stream, f"finish length on output {full - 1} where the Commit's is none")]


def test_a_length_finish_short_of_max_tokens_is_refused_by_the_tie_too(run):
    """`COMPACT_LAST` after k + 1 < `max_tokens` outputs, the Commit's ids cut there with no EOS and no abort (a
    response that ran out of context): the certified I/O refused it already, and the tie, which held only the
    records to the ids, now refuses it too."""
    t = run.target
    stream, full = t["stream"], len(t["output"])
    for k in range(full - 1):
        cut = _cut(run.outs, stream, k, active.compact_record(stream, t["output"][k], "length"))
        assert _io(run, cut)["violations"] == [[stream, "a length finish short of max_tokens"]], k
        commit = _sampled(run.honest) | {stream: t["output"][:k + 1]}
        assert _tie(run, commit, cut) == [(stream, f"a length finish at {k + 1} outputs where max_tokens is {full}")], k
