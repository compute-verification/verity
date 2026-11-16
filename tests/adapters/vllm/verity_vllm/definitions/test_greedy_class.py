"""A class's selects and length check (`greedy_stages.GreedyCombineFirst_v1`, `GreedyCombineLive_v1`,
`greedy_stop.GreedyStopCheckClass_v1`) on made-up requests at a small class: each step's row puts its argmax on a chosen
token, the selects run as a chain (each reads the step before's bit and count), and the check reads the served ids, the
last bit and count.  Each property the class's design asks of them has a test here that fails without it."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.types import Array
from verity_vllm.definitions.registry import greedy_stages as G
from verity_vllm.definitions.registry import greedy_stop as GS
from verity_vllm.definitions.registry import stop_rule as X
from verity_vllm.definitions.registry.prims import I32

V, N, O, P, MML = 320, 16, 4, 8, 64
STOP = (37, 301)
NINF, ONE = 0xFF80, 0x3F80
K = G.count_bits(O)
FIRST = bind(G.GreedyCombineFirst, V=V, N=N, O=O, STOP=STOP, MAX_MODEL_LEN=MML)
LIVE = bind(G.GreedyCombineLive, V=V, N=N, O=O, STOP=STOP, MAX_MODEL_LEN=MML)


def _check(mml: int = MML, p: int = P):
    return bind(GS.GreedyStopCheckClass, V=V, N=N, P=p, T=O, STOP=STOP, MAX_MODEL_LEN=mml)


def _row(token: int) -> np.ndarray:
    row = np.full(V, NINF, np.uint16)
    row[token] = ONE
    return row


def _args(row: np.ndarray) -> list:
    _token, words = G.twin(V, row)
    return [[int(v) for v in row[V - N:]], *([int(w) for w in words[u.key][0]] for u in G.plan(V)[:-1])]


def _bits(c: int) -> list[int]:
    """A count or a bit as the u16 word a select reads it from, its bits least significant first."""
    return [(c >> j) & 1 for j in range(16)]


def _out(leaves) -> tuple[int, int, int, int]:
    """A select's outputs as (o0 the id, o1, o2 the count, o3 the id)."""
    v = [int(x) for x in leaves]
    return v[0] | v[1] << 16, v[2], sum(b << j for j, b in enumerate(v[3:3 + K])), v[3 + K] | v[4 + K] << 16


def _select(token: int, request, count: int | None = None, live: int | None = None, fn=None):
    if count is None:
        return _out(evaluate(fn or FIRST, *_args(_row(token)), list(request)))
    return _out(evaluate(fn or LIVE, *_args(_row(token)), list(request), _bits(count), _bits(live)))


def _chain(tokens: list[int], request, mml: int = MML, p: int = P) -> dict:
    """The O selects over steps whose argmaxes are ``tokens``, and the check: each step's outputs, the ids, and the bit."""
    fn_first = bind(G.GreedyCombineFirst, V=V, N=N, O=O, STOP=STOP, MAX_MODEL_LEN=mml)
    fn_live = bind(G.GreedyCombineLive, V=V, N=N, O=O, STOP=STOP, MAX_MODEL_LEN=mml)
    steps = [_select(tokens[0], request, fn=fn_first)]
    for t in tokens[1:]:
        _id, live, count, _key = steps[-1]
        steps.append(_select(t, request, count, live, fn=fn_live))
    ids = [s[0] for s in steps]
    _id, live, count, _key = steps[-1]
    check = int(evaluate(_check(mml, p), *_args(_row(tokens[-1])), ids, list(request), _bits(count), _bits(live))[0])
    want = GS.class_reference(_row(tokens[-1]), ids, request, count, live, V, N, p, STOP, mml)
    assert check == want, (tokens, request, steps)
    return {"steps": steps, "ids": ids, "keys": [s[3] for s in steps], "bits": [s[1] for s in steps],
            "counts": [s[2] for s in steps], "check": check}


def _req(lp=P, max_tokens=O, min_tokens=0, ignore_eos=0):
    return [lp, max_tokens, min_tokens, ignore_eos]


# ---- whole requests ------------------------------------------------------------------------------------------------------------

def test_a_request_that_stops_at_step_1_on_a_stop_id():
    """The first id is a stop id: it is served, its key is 0, every later step is off and serves 0, the count holds at 1, and
    the check passes at n = 1."""
    r = _chain([37, 5, 6, 7], _req(max_tokens=100))
    assert r["ids"] == [37, 0, 0, 0] and r["keys"] == [0, 0, 0, 0]
    assert r["bits"] == [0, 0, 0, 0] and r["counts"] == [1, 1, 1, 1] and r["check"] == 1


def test_a_request_that_stops_at_the_last_step_on_a_stop_id():
    """The O-th id is a stop id: all O are served, each key is its token but the last's, and the check passes at n = O."""
    r = _chain([5, 6, 7, 301], _req(max_tokens=100))
    assert r["ids"] == [5, 6, 7, 301] and r["keys"] == [5, 6, 7, 0]
    assert r["bits"] == [1, 1, 1, 0] and r["counts"] == [1, 2, 3, 4] and r["check"] == 1


def test_a_request_that_never_stops_on_a_stop_id_serves_o_ids_at_max_tokens():
    """No stop id, max_tokens = O: n = O by the length, every id served, and the check passes."""
    r = _chain([5, 6, 7, 8], _req(max_tokens=O))
    assert r["ids"] == [5, 6, 7, 8] and r["keys"] == [5, 6, 7, 0] and r["counts"] == [1, 2, 3, 4] and r["check"] == 1


def test_a_request_the_rule_does_not_end_within_o_is_refused():
    """No stop id and max_tokens past O: the chain stops at O (its bit 0, its count O), and the check refuses (finish 0)."""
    r = _chain([5, 6, 7, 8], _req(max_tokens=100))
    assert r["ids"] == [5, 6, 7, 8] and r["bits"][-1] == 0 and r["counts"][-1] == O and r["check"] == 0


def test_a_request_that_stops_at_max_tokens():
    """max_tokens = 2: two ids served, the second's key 0, the rest 0, and the check passes at n = 2."""
    r = _chain([5, 6, 7, 8], _req(max_tokens=2))
    assert r["ids"] == [5, 6, 0, 0] and r["keys"] == [5, 0, 0, 0] and r["counts"] == [1, 2, 2, 2] and r["check"] == 1


def test_a_request_that_stops_at_the_model_s_length():
    """The model's length less the prompt's is 3, under max_tokens: three ids, finish 3, and the check passes."""
    r = _chain([5, 6, 7, 8], _req(max_tokens=100), mml=P + 3)
    assert r["ids"] == [5, 6, 7, 0] and r["counts"][-1] == 3 and r["check"] == 1
    assert X.reference([5, 6, 7, 0], 100, STOP, P, P + 3) == (3, 3)


def test_max_tokens_below_1_never_stops_and_is_refused():
    """max_tokens 0: the rule never stops, so the chain runs to O and the check refuses."""
    r = _chain([37, 6, 7, 8], _req(max_tokens=0))
    assert r["ids"] == [37, 6, 7, 8] and r["check"] == 0


def test_min_tokens_other_than_0_is_refused_by_the_check_alone():
    """min_tokens 1 changes no select (the chain is the request's without it) and the check refuses it."""
    a, b = _chain([5, 37, 7, 8], _req(max_tokens=100)), _chain([5, 37, 7, 8], _req(max_tokens=100, min_tokens=1))
    assert a["steps"] == b["steps"] and a["check"] == 1 and b["check"] == 0


def test_ignore_eos_switches_the_stop_ids_off_in_the_selects_and_the_check():
    """The second id is a stop id: without ignore_eos it ends there; with it the request runs to max_tokens 3."""
    a = _chain([5, 37, 7, 8], _req(max_tokens=3))
    b = _chain([5, 37, 7, 8], _req(max_tokens=3, ignore_eos=1))
    assert a["ids"] == [5, 37, 0, 0] and a["check"] == 1
    assert b["ids"] == [5, 37, 7, 0] and b["keys"] == [5, 37, 0, 0] and b["check"] == 1


def test_a_prompt_of_other_than_p_ids_is_refused():
    r = _chain([5, 37, 7, 8], _req(lp=P - 1, max_tokens=100))
    assert r["ids"] == [5, 37, 0, 0] and r["check"] == 0


@pytest.mark.parametrize("p", [1, 3, 43])
def test_a_class_is_keyed_by_its_prompt_s_exact_length_a_power_of_two_or_not(p):
    """P is the prompt's exact length, any integer: at P = 43 a prompt of 43 ids passes, one of 42 or 44 is refused."""
    assert _chain([5, 37, 7, 8], _req(lp=p, max_tokens=100), p=p)["check"] == 1
    for lp in (p - 1, p + 1):
        assert _chain([5, 37, 7, 8], _req(lp=lp, max_tokens=100), p=p)["check"] == 0


def test_the_check_ties_n_to_the_last_count_and_bit():
    """A check handed a count or bit other than the chain's refuses: the rule's n and the selects' count are one number,
    and a chain still live is not done."""
    ids, req, row = [5, 37, 0, 0], _req(max_tokens=100), _row(8)

    def ok(count, live, ids=ids):
        return int(evaluate(_check(), *_args(row), ids, req, _bits(count), _bits(live))[0])

    assert ok(2, 0) == 1 and ok(1, 0) == 0 and ok(3, 0) == 0 and ok(2, 1) == 0
    assert ok(2, 0, [5, 37, 9, 0]) == 0 and ok(2, 0, [5, 37, 0, 9]) == 0


def test_the_check_compares_the_last_id_with_the_token_only_where_the_last_step_is_live():
    """At n = O the O-th id must be the last row's token; at n < O the last row's token is not 0 and the check passes."""
    req = _req(max_tokens=O)
    good = int(evaluate(_check(), *_args(_row(8)), [5, 6, 7, 8], req, _bits(O), _bits(0))[0])
    bad = int(evaluate(_check(), *_args(_row(9)), [5, 6, 7, 8], req, _bits(O), _bits(0))[0])
    early = int(evaluate(_check(), *_args(_row(9)), [5, 37, 0, 0], _req(max_tokens=100), _bits(2), _bits(0))[0])
    assert (good, bad, early) == (1, 0, 1)


@pytest.mark.parametrize("seed", range(6))
def test_the_chain_ends_where_the_stop_rule_does(seed):
    """Random tokens (stop ids among them) and limits: the served ids are the rule's first n then zeros, the last count is
    n, and the check is 1 exactly where the rule ends within O."""
    rng = np.random.default_rng(seed)
    for _ in range(3):
        tokens = [int(rng.choice([5, 6, 37, 301, 200])) for _ in range(O)]
        req = _req(max_tokens=int(rng.integers(0, 6)), ignore_eos=int(rng.integers(0, 2)))
        mml = int(rng.choice([MML, P + 2]))
        r = _chain(tokens, req, mml)
        n, finish = X.reference(tokens, req[1], STOP if not req[3] else (0xFFFFFFFF,), P, mml)
        assert r["ids"] == tokens[:n] + [0] * (O - n) and r["counts"][-1] == n, (tokens, req, mml)
        assert r["check"] == int(finish != 0), (tokens, req, mml)


# ---- one select ---------------------------------------------------------------------------------------------------------------

def test_the_first_bit_and_count_are_constants_of_the_first_select():
    """The first select has no live or count parameter, so neither is a boundary row, and writes the count 1 whatever it
    reads; the decode steps' select reads both."""
    first = [p for p, _ in FIRST.signature()[0]]
    assert "live" not in first and "count" not in first and first[-1] == "request"
    assert [p for p, _ in LIVE.signature()[0]][-3:] == ["request", "count", "live"]
    for token, req in ((5, _req()), (37, _req()), (5, _req(max_tokens=0)), (5, _req(lp=0, max_tokens=1))):
        assert _select(token, req)[2] == 1


@pytest.mark.parametrize("count", range(O + 1))
def test_once_off_off_and_a_step_off_writes_the_count_it_read(count):
    """At live 0 a select writes the bit 0, both ids 0 and the count it read, whatever its token and limits."""
    for token, req in ((5, _req(max_tokens=100)), (37, _req()), (8, _req(max_tokens=0, ignore_eos=1))):
        assert _select(token, req, count, 0) == (0, 0, count, 0)


def test_the_count_never_passes_o():
    """A live step reading O - 1 writes O and the bit 0; one reading O (no chain reaches it) holds O and writes the bit 0."""
    req = _req(max_tokens=100)
    assert _select(5, req, O - 1, 1) == (5, 0, O, 0)
    assert _select(5, req, O, 1) == (5, 0, O, 0)
    assert G.class_reference(5, req, O, STOP, MML, O, 1) == (5, 0, O, 0)


def test_token_is_anded_with_its_own_bit_and_key_with_the_next():
    """A live step that stops serves its token (o0) and hands the next step 0 (o3); one that doesn't hands its token on; a
    step off serves 0."""
    req = _req(max_tokens=100)
    assert _select(37, req, 1, 1) == (37, 0, 2, 0)
    assert _select(6, req, 1, 1) == (6, 1, 2, 6)
    assert _select(6, req, 1, 0) == (0, 0, 1, 0)
    assert _select(301, req) == (301, 0, 1, 0) and _select(6, req) == (6, 1, 1, 6)


def test_the_limits_are_values_of_one_row_not_statics():
    """The selects and the check share their statics across requests (no max_tokens or prompt length among them), and read
    the limits from their ``request`` parameter, `LIMIT_FIELDS` in order: max_tokens 1 ends the first step, 5 doesn't."""
    for fn in (FIRST, LIVE, _check()):
        assert not {"MAX_TOKENS", "PROMPT_LEN", "FINISH"} & set(fn.bindings)
        assert dict(fn.signature()[0])["request"] == Array(4, I32)
    assert G.LIMIT_FIELDS == ("lp", "max_tokens", "min_tokens", "ignore_eos")
    assert _select(6, _req(max_tokens=1))[1] == 0 and _select(6, _req(max_tokens=5))[1] == 1
    assert _select(6, _req(lp=MML - 1, max_tokens=5))[1] == 0


@pytest.mark.parametrize("seed", range(4))
def test_a_select_is_its_reference(seed):
    """Random tokens, limits, counts and bits: the select's four outputs are `class_reference`'s."""
    rng = np.random.default_rng(seed)
    for _ in range(6):
        token = int(rng.choice([0, 5, 37, 301, 319]))
        req = [int(rng.integers(0, 70)), int(rng.integers(-1, 7)), 0, int(rng.integers(0, 2))]
        count, live = int(rng.integers(0, O + 1)), int(rng.integers(0, 2))
        assert _select(token, req, count, live) == G.class_reference(token, req, O, STOP, MML, count, live)
        assert _select(token, req) == G.class_reference(token, req, O, STOP, MML)


def test_a_token_past_2_16_matches_a_stop_id_by_both_words():
    """At GLM-5.3's vocabulary a stop id's high word is 2: a token equal to it in one word only is no stop."""
    fn = bind(G.GreedyCombineLive, V=V, N=N, O=O, STOP=(0x25 | 2 << 16,), MAX_MODEL_LEN=MML)
    assert _select(37, _req(max_tokens=100), 1, 1, fn=fn)[1] == 1


def test_a_reader_reads_only_the_count_s_bits_and_the_bit_s_bit_0():
    """``count`` and ``live`` are a u16 word each, wired from outputs of fewer bits zero-padded: the selects and the check
    read the count's low log2(O) + 1 bits and the bit's bit 0, so the other bits of either word change nothing."""
    req, row = _req(max_tokens=100), _row(6)

    def high(w, extra):
        return [b | e for b, e in zip(_bits(w), _bits(extra))]

    want = _out(evaluate(LIVE, *_args(row), req, _bits(1), _bits(1)))
    assert _out(evaluate(LIVE, *_args(row), req, high(1, 0xFFF0 & ~((1 << K) - 1)), high(1, 0xFFFE))) == want == (6, 1, 2, 6)
    ids = [5, 37, 0, 0]
    assert int(evaluate(_check(), *_args(row), ids, req, high(2, 1 << 15), high(0, 0xFFFE))[0]) == 1


def test_every_port_is_whole_words_of_a_row():
    """`interface.layout` takes the three as units: each parameter whole u16 words, each output field one or two."""
    from verity.commitments.periods import interface as IF

    for fn in (FIRST, LIVE, _check()):
        L = IF.layout(fn)
        assert [w for _p, w in L.rows][-2:] == ([3, 8] if fn is FIRST else [1, 1])
    assert [w for _p, w in IF.layout(LIVE).outs] == [2, 1, 1, 2]


# ---- statics ------------------------------------------------------------------------------------------------------------------

def test_glm53_statics_bind():
    """At GLM-5.3's statics (154,880 lanes in 20 chunks of 7,744, its three EOS ids, a model of 131,072 ids) and O = 128,
    the count is 8 bits; the check at P = 1,024 reads 128 ids."""
    s = dict(V=154880, N=7744, O=128, STOP=(154820, 154827, 154829), MAX_MODEL_LEN=131072)
    first, live = bind(G.GreedyCombineFirst, **s), bind(G.GreedyCombineLive, **s)
    check = bind(GS.GreedyStopCheckClass, V=154880, N=7744, P=1024, T=128, STOP=s["STOP"], MAX_MODEL_LEN=131072)
    assert len(first.signature()[0]) == 21 and len(live.signature()[0]) == 23 and len(check.signature()[0]) == 24
    assert str(live.signature()[1]) == "(Array<2,Value<16>>, Value<1>, Array<8,Value<1>>, Array<2,Value<16>>)"
    assert dict(check.signature()[0])["count"] == dict(live.signature()[0])["count"] == G.WORD_BITS


@pytest.mark.parametrize("bad", [dict(O=3), dict(O=1), dict(STOP=(5, 3)), dict(MAX_MODEL_LEN=1 << 31)])
def test_the_selects_refuse_a_binding_with_no_class(bad):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(G.GreedyCombineLive, **{**dict(V=V, N=N, O=O, STOP=STOP, MAX_MODEL_LEN=MML), **bad})


@pytest.mark.parametrize("bad", [dict(P=0), dict(P=1 << 31), dict(T=6), dict(STOP=(3, 3))])
def test_the_check_refuses_a_binding_with_no_class(bad):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(GS.GreedyStopCheckClass, **{**dict(V=V, N=N, P=P, T=O, STOP=STOP, MAX_MODEL_LEN=MML), **bad})
