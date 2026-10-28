"""A client's abort in the I/O tie's compact rule (`active.compact_finish_violations`).

No compact record says an abort: a response its client aborted ends on an unmarked record, and the certifier can't see
the abort on the link (one connection, no per-request cancel), so any abort on a certified link comes from the prover's
side. The tie takes it from the Commit (`aborted`) and accepts the unmarked end only where the Commit's finish is an
abort after 1 ≤ k < `max_tokens` outputs, none of them EOS. What makes that sound is the charge, not an observation:
the advice charges each request's `output_len`, the abort's included, in full. Beside it, a `COMPACT_LAST` short of
`max_tokens` is now refused here too, as `certifier_serve.certified_io` refuses it.
"""

from __future__ import annotations

from verity.core.protocols.certifier import active as A

B = 8
EOS = (0,)


def _cells(ordinal: int, ids: list[int], last: str | None) -> list[bytes]:
    """`ids` as cells, every one unmarked but the last, which carries `last`."""
    *body, end = ids
    return [A.compact_cell(A.compact_record(ordinal, t), B) for t in body] + \
        [A.compact_cell(A.compact_record(ordinal, end, last), B)]


def _violations(ids: list[int], cap: int, last: str | None, aborted: bool) -> list[tuple[int, str]]:
    cells = _cells(1, ids, last) if ids else []
    return A.compact_finish_violations({1: ids}, EOS, cells, max_tokens={1: cap}, aborted={1} if aborted else ())


def test_an_honest_abort_is_its_unmarked_records_and_nothing_else():
    """The Commit's client aborted request 1 after 2 of its 4: its records, the Commit's ids with no mark, are its
    response. `compact_records_of` an aborted response marks nothing."""
    assert _violations([7, 8], 4, None, aborted=True) == []
    assert A.compact_records_of(1, [7, 8], EOS, aborted=True) == [A.compact_record(1, 7), A.compact_record(1, 8)]
    assert _violations([7, 8, 9], 4, None, aborted=True) == []


def test_an_unmarked_end_the_commit_doesnt_call_an_abort_is_refused():
    assert _violations([7, 8], 4, None, aborted=False) == [(1, "finish none on output 1 where the Commit's is length")]
    assert _violations([7, 8, 9, 6], 4, None, aborted=False) == \
        [(1, "finish none on output 3 where the Commit's is length")]


def test_an_abort_at_max_tokens_is_refused():
    """A response that reached `max_tokens` finished on its length before any abort could take it."""
    assert _violations([7, 8, 9, 6], 4, None, aborted=True) == [(1, "an abort at 4 outputs where max_tokens is 4")]
    assert _violations([7, 8, 9, 6, 5], 4, None, aborted=True) == [(1, "an abort at 5 outputs where max_tokens is 4")]


def test_an_abort_on_an_eos_id_is_refused():
    """A response whose model sampled EOS stopped there: an abort that ends on it, or carries it before its end, is
    refused, and so is the stop record the Commit's abort doesn't have."""
    assert _violations([7, 0], 4, None, aborted=True) == [(1, "an abort on an EOS id")]
    assert _violations([0, 7], 4, None, aborted=True) == [(1, "an abort on an EOS id")]
    assert _violations([7, 0], 4, "stop", aborted=True) == [(1, "finish stop on output 1 where the Commit's is none")]


def test_a_marked_end_where_the_commit_says_abort_and_an_abort_with_no_output_are_refused():
    assert _violations([7, 8], 4, "length", aborted=True) == \
        [(1, "finish length on output 1 where the Commit's is none")]
    assert _violations([], 4, None, aborted=True) == [(1, "an abort before any output")]


def test_a_length_finish_short_of_max_tokens_is_refused():
    """`COMPACT_LAST` alone where the Commit sampled no EOS and stopped short of `max_tokens`: the records are the
    Commit's ids one for one, and this was accepted before."""
    assert _violations([7, 8, 9, 6], 4, "length", aborted=False) == []
    assert _violations([7, 8], 4, "length", aborted=False) == \
        [(1, "a length finish at 2 outputs where max_tokens is 4")]
    assert _violations([7, 8, 0], 4, "stop", aborted=False) == []
