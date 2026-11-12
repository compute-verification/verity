"""`StopCheck_v1` (`verity.definitions.registry.stop_check`) against its reference on every short stream, on a served stream cut
short or run past its stop, with no stop ids, and the statics it refuses."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.registry import stop_check as X
from verity.definitions.registry import stop_rule as SR

GLM_EOS = (154820, 154827, 154829)


def _check(rows: list[list[int]], **statics) -> list[int]:
    fn = bind(X.StopCheck, T=len(rows[0]), **statics)
    return [int(b) for b in np.asarray(evaluate_batch(fn, np.array(rows, dtype=np.uint32), kernel="reference")).reshape(-1)]


@pytest.mark.parametrize("T", [1, 2, 3])
@pytest.mark.parametrize("STOP", [(3, 7), ()])
@pytest.mark.parametrize("MAX_TOKENS, PROMPT_LEN", [(2, 8), (4, 8), (9, 61), (0, 8)])
def test_the_check_is_its_reference(T, STOP, MAX_TOKENS, PROMPT_LEN):
    """Every stream of T outputs over the stop ids, an other id and `NOT_AN_ID`, on each finish, with max_tokens and the model's
    length (64) each reached within T or not."""
    words = [3, 7, 5, X.NOT_AN_ID]
    rows = [[words[k // 4 ** i % 4] for i in range(T)] for k in range(4 ** T)]
    for FINISH in [f for f in SR.FINISH if f != 1 or STOP]:
        st = dict(STOP=STOP, MAX_TOKENS=MAX_TOKENS, PROMPT_LEN=PROMPT_LEN, MAX_MODEL_LEN=64, FINISH=FINISH)
        assert _check(rows, **st) == [X.reference(r, **st) for r in rows]


def test_the_check_refuses_a_stream_cut_short_or_run_past_its_stop_and_passes_an_honest_finish():
    """Under three stop ids, on one stream whose first stop id is output 4 and a max_tokens of 6: the honest stop (5 outputs), the
    honest max_tokens finish of a stream with no stop id (6), and a stream that runs on (3, finish 0) pass; the stream cut short
    and called a stop (3) or a max_tokens finish (3), run past its stop to a later one (8) or to its max_tokens (6), or run past
    max_tokens (7) is refused."""
    s, plain = [11, 12, 13, 14, 154827, 15, 16, 154820], [11, 12, 13, 14, 15, 16, 17]
    cases = {"honest stop": (s[:5], 1, 1), "honest length": (plain[:6], 2, 1), "runs on": (s[:3], 0, 1),
             "cut short, a stop": (s[:3], 1, 0), "cut short, length": (s[:3], 2, 0), "run past to a later stop": (s, 1, 0),
             "run past to max_tokens": (s[:6], 2, 0), "run past max_tokens": (plain, 2, 0)}
    got = {k: _check([ids], STOP=GLM_EOS, MAX_TOKENS=6, PROMPT_LEN=1024, MAX_MODEL_LEN=131072, FINISH=f)[0]
           for k, (ids, f, _) in cases.items()}
    assert got == {k: ok for k, (_, _, ok) in cases.items()}


def test_with_no_stop_ids_only_the_length_caps_end_a_stream():
    """No stop ids (ignore_eos): a stream of any token ids ends exactly at max_tokens, or at the model's length before it."""
    st = dict(STOP=(), PROMPT_LEN=8, MAX_MODEL_LEN=64)
    assert _check([[0, 154820, 2], [5, 6, 7]], MAX_TOKENS=3, FINISH=2, **st) == [1, 1]
    assert _check([[0, 154820, 2]], MAX_TOKENS=4, FINISH=2, **st) == [0]
    assert _check([[0, 154820, 2]], MAX_TOKENS=4, FINISH=0, **st) == [1]
    assert _check([[0, 1, 2]], MAX_TOKENS=9, FINISH=3, **{**st, "PROMPT_LEN": 61}) == [1]


@pytest.mark.parametrize("bad", [{"T": 0}, {"T": (1 << 31) - 1}, {"STOP": (7, 3)}, {"STOP": (3, 3)}, {"STOP": (1 << 32,)},
                                 {"STOP": [3]}, {"MAX_TOKENS": -1}, {"PROMPT_LEN": 1 << 31}, {"FINISH": 4}, {"FINISH": "length"},
                                 {"STOP": (), "FINISH": 1}])
def test_the_check_refuses_statics_with_no_check(bad):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(X.StopCheck, **{"T": 2, "STOP": (3,), "MAX_TOKENS": 4, "PROMPT_LEN": 8, "MAX_MODEL_LEN": 64, "FINISH": 1, **bad})
