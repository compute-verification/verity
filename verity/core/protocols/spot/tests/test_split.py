"""One sample proved in several sessions (`split`, `check_split`): the parts the rule cuts, the one part of a sample
that fits one session, and each refusal of sessions that are not the parts in order, by its code."""

from __future__ import annotations

import pytest

from verity.core.protocols.spot.one_stage import draw as D

#: a subset sample of 8 of 20 proof blocks
SUBSET = {**D.parse_design("subset:8"), "population": 20, "units": [0, 2, 3, 7, 11, 12, 16, 19]}
#: a stratified sampling design over 10 proof blocks, templates of 4 and 6 of them: k_s = 2 and 3
DESIGN = {**D.parse_design("stratified:5"), "strata": [{"template": "SplitTestA_v1", "units": [[0, 4]], "k": 2},
                                                        {"template": "SplitTestB_v1", "units": [[4, 10]], "k": 3}]}
SAMPLE = {**DESIGN, "population": 10, "units": [1, 3, 4, 7, 9]}
#: a work sampling design over 6 proof blocks, templates of 2 (no work) and 4 of them: k_s = 1 and 4
WORK = {**D.parse_design("work:4"), "strata": [{"template": "SplitTestA_v1", "units": [[0, 2]], "work": 0, "floor": 1, "k": 1},
                                                {"template": "SplitTestB_v1", "units": [[2, 6]], "work": 10, "floor": 1, "k": 4}]}
#: two of its samples: one proves a closure of one more proof block, the other none
CLOSED = {**WORK, "population": 6, "units": [0, 2, 3, 4, 5], "closure": [1]}
OPEN = {**WORK, "population": 6, "units": [1, 2, 3, 4, 5]}
#: a Bernoulli sample of no proof block
EMPTY = {**D.parse_design("bernoulli:1/2"), "population": 4, "units": []}


def test_the_samples_are_samples_of_their_designs():
    assert D.check_draw(SUBSET, {"population": 20}, D.parse_design("subset:8")).ok
    assert D.check_draw(SAMPLE, {"population": 10}, DESIGN).ok
    assert D.check_draw(CLOSED, {"population": 6}, WORK).ok and D.check_draw(OPEN, {"population": 6}, WORK).ok
    assert D.check_draw(EMPTY, {"population": 4}, D.parse_design("bernoulli:1/2")).ok


def test_a_sample_that_fits_one_session_is_one_part():
    """With no `session_units`, or one the sample doesn't pass, the one part is every proof block the sample proves, and
    `check_split` accepts a session's proof blocks exactly where `check_proved` finds them to be the sample's."""
    for sample in (SUBSET, SAMPLE, CLOSED, OPEN, EMPTY):
        whole = D.proved(sample)
        for bound in (None, max(1, len(whole)), len(whole) + 1, 1 << 40):
            assert D.split(sample, bound) == [whole]
            assert D.check_split(sample, [whole], bound).ok and D.check_split(sample, (tuple(whole),), bound).ok
        assert D.split(sample) == [whole] and D.check_split(sample, [whole]).ok
    for proved in ([1, 3, 4, 7, 9], [1, 3, 4, 7], [1, 3, 4, 7, 8], [1, 3, 4, 7, 9, 9], [3, 1, 4, 7, 9], [0, 1, 3, 4, 7, 9], []):
        want = proved == [1, 3, 4, 7, 9]
        assert D.check_split(SAMPLE, [proved]).ok == D.check_proved(SAMPLE, proved, len(proved)).ok == want, proved
    for proved in ([0, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5]):
        assert D.check_split(CLOSED, [proved]).ok == D.check_proved(CLOSED, proved, len(proved)).ok == (len(proved) == 6)


def test_a_sample_past_the_bound_is_cut_into_consecutive_parts():
    """The parts are consecutive runs of `session_units` of the sample's proof blocks, ascending, the last run the rest: they
    are as many as ceil(K / session_units), every one but the last is full, none is empty, and together they are the
    sample in order."""
    assert D.split(SUBSET, 4) == [[0, 2, 3, 7], [11, 12, 16, 19]]
    assert D.split(SAMPLE, 3) == [[1, 3, 4], [7, 9]]
    assert D.split(SAMPLE, 2) == [[1, 3], [4, 7], [9]]
    assert D.split(SAMPLE, 1) == [[1], [3], [4], [7], [9]]
    for sample, bound in ((SUBSET, 4), (SUBSET, 3), (SAMPLE, 3), (SAMPLE, 2), (SAMPLE, 1)):
        parts = D.split(sample, bound)
        assert D.check_split(sample, parts, bound).ok and D.check_split(sample, [tuple(p) for p in parts], bound).ok
    for k in range(1, 13):
        sample = {**D.parse_design(f"subset:{k}"), "population": 24, "units": list(range(1, 2 * k, 2))}
        for bound in range(1, 14):
            parts = D.split(sample, bound)
            assert len(parts) == max(1, -(-k // bound)) and [u for p in parts for u in p] == sample["units"]
            assert all(len(p) == bound for p in parts[:-1]) and 1 <= len(parts[-1]) <= bound
            assert D.check_split(sample, parts, bound).ok


def test_a_bound_of_no_proof_block_or_that_is_no_integer_is_refused():
    for bad in (0, -1, True, "2", 2.0, [2]):
        why = D.session_units_problem(bad)
        assert why == ("session_units is the most proof blocks one session proves, an integer of at least one, "
                       f"not {bad!r}")
        with pytest.raises(ValueError, match="session_units is the most proof blocks one session proves"):
            D.split(SAMPLE, bad)
        v = D.check_split(SAMPLE, [[1, 3, 4, 7, 9]], bad)
        assert v.codes == ("session-units",) and v.detail == {"session-units": why}
    assert D.session_units_problem(None) is None and D.session_units_problem(1) is None


def _refusal(sessions, sample=SAMPLE, bound=2) -> dict:
    v = D.check_split(sample, sessions, bound)
    assert not v.ok and tuple(v.detail) == v.codes
    return v.detail


def test_sessions_that_are_not_the_parts_in_order_are_refused_by_name():
    """Against the parts [1, 3], [4, 7], [9]: a proof block no session proved, one proved twice, a session that proved
    blocks of another part, the parts in another order, and sessions that are not as many as the parts."""
    assert D.split(SAMPLE, 2) == [[1, 3], [4, 7], [9]]
    assert _refusal([[1, 3], [4], [9]]) == {"dropped": [7], "parts": {"session": 1, "proved": [4], "part": [4, 7]}}
    assert _refusal([[1, 3], [3, 4, 7], [9]]) == {"doubled": [3],
                                                  "parts": {"session": 1, "proved": [3, 4, 7], "part": [4, 7]}}
    assert _refusal([[1, 3], [4, 7], [9, 9]]) == {"doubled": [9], "parts": {"session": 2, "proved": [9, 9], "part": [9]}}
    assert _refusal([[1, 4], [3, 7], [9]]) == {"parts": {"session": 0, "proved": [1, 4], "part": [1, 3]}}
    assert _refusal([[1], [3, 4], [7, 9]]) == {"parts": {"session": 0, "proved": [1], "part": [1, 3]}}
    assert _refusal([[4, 7], [1, 3], [9]]) == {"order": [1, 0, 2]}
    assert _refusal([[9], [4, 7], [1, 3]]) == {"order": [2, 1, 0]}
    assert _refusal([[1, 3], [4, 7]]) == {"sessions": {"sessions": 2, "parts": 3}, "dropped": [9]}
    assert _refusal([[1, 3], [4, 7], [9], []]) == {"sessions": {"sessions": 4, "parts": 3}}
    assert _refusal([[1, 3], [4, 7], [9], [9]]) == {"sessions": {"sessions": 4, "parts": 3}, "doubled": [9]}
    assert _refusal([[1, 3, 4, 7, 9]]) == {"sessions": {"sessions": 1, "parts": 3},
                                           "parts": {"session": 0, "proved": [1, 3, 4, 7, 9], "part": [1, 3]}}
    assert _refusal([]) == {"sessions": {"sessions": 0, "parts": 3}, "dropped": [1, 3, 4, 7, 9]}


def test_sessions_of_other_proof_blocks_or_of_none_are_refused():
    """A proof block outside the sample, a session's proof blocks out of ascending order, and sessions that are no lists
    of integers."""
    assert _refusal([[1, 3], [4, 7], [8]]) == {"dropped": [9], "parts": {"session": 2, "proved": [8], "part": [9]}}
    assert _refusal([[1, 3], [4, 7], [8, 9]]) == {"parts": {"session": 2, "proved": [8, 9], "part": [9]}}
    assert _refusal([[3, 1], [4, 7], [9]]) == {"parts": {"session": 0, "proved": [3, 1], "part": [1, 3]}}
    for bad in ([[1, 3], [4, 7], "9"], [[1, 3], [4, True], [9]], [[1, 3], [4, 7], [9.0]], [[1, 3], [4, 7], [[9]]], "139", None,
                {0: [1, 3]}):
        assert list(_refusal(bad)) == ["not-sessions"], bad
    assert list(_refusal([[0, 2, 3, 7], [11, 12, 16]], SUBSET, 4)) == ["dropped", "parts"]
    assert list(_refusal([[0, 2, 3, 7], [11, 12, 16], [19]], SUBSET, 4)) == ["sessions", "parts"]
    assert _refusal([[11, 12, 16, 19], [0, 2, 3, 7]], SUBSET, 4) == {"order": [1, 0]}


def test_a_stratum_short_of_its_count_is_refused_whatever_the_split():
    """`check_draw` holds the sample to each stratum's k_s and `check_split` the sessions to the sample: a sample short
    in a stratum is refused by the first (``strata-count``) though sessions prove its parts, and sessions that prove a
    full sample less one proof block of a stratum are refused by the second (``dropped``)."""
    short = {**SAMPLE, "units": [1, 4, 5, 7, 9]}
    v = D.check_draw(short, {"population": 10}, DESIGN)
    assert v.codes == ("strata-count",)
    assert v.detail["strata-count"] == {"SplitTestA_v1": {"drawn": 1, "k": 2}, "SplitTestB_v1": {"drawn": 4, "k": 3}}
    assert D.check_split(short, D.split(short, 2), 2).ok
    fewer = {**SAMPLE, "units": [1, 4, 7, 9]}
    assert D.check_draw(fewer, {"population": 10}, DESIGN).detail == {"strata-count": {"SplitTestA_v1": {"drawn": 1, "k": 2}}}
    assert _refusal([[1], [4, 7], [9]]) == {"dropped": [3], "parts": {"session": 0, "proved": [1], "part": [1, 3]}}
    assert _refusal([[1, 4], [7, 9]]) == {"sessions": {"sessions": 2, "parts": 3}, "dropped": [3],
                                          "parts": {"session": 0, "proved": [1, 4], "part": [1, 3]}}


def test_a_closure_is_cut_with_the_sampled_proof_blocks_and_moves_the_parts():
    """A work sample's closure is proved with its sampled proof blocks, so the parts cut both together; two samples of one
    work sampling design whose closures differ in size prove different numbers of proof blocks, so how many parts there
    are and how large the last one is follow the sample, where under a design of fixed count they follow the design."""
    assert D.fixed_count(WORK) is None and D.fixed_count(DESIGN) == 5
    assert (len(D.proved(CLOSED)), len(D.proved(OPEN))) == (6, 5)
    assert D.split(CLOSED, 3) == [[0, 1, 2], [3, 4, 5]] and D.split(OPEN, 3) == [[1, 2, 3], [4, 5]]
    assert D.split(CLOSED, 5) == [[0, 1, 2, 3, 4], [5]] and D.split(OPEN, 5) == [[1, 2, 3, 4, 5]]
    assert D.check_split(CLOSED, [[0, 1, 2], [3, 4, 5]], 3).ok
    assert _refusal([[0, 2, 3], [4, 5]], CLOSED, 3) == {"dropped": [1],
                                                        "parts": {"session": 0, "proved": [0, 2, 3], "part": [0, 1, 2]}}
