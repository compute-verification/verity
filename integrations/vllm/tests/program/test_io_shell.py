"""`registry.io_shell` (ingress v0): the encoder's bytes in the layout, padded to whole SHA-512 blocks, `IngressPrompt_v1` as
`reference` on bodies in the schema and on each way out of it (other padding among them), the statics it refuses, the
Boolean Definition against its word, `TokenFeed_v1` down a class Program's positions, and the network
certifier's scan state read (`ScanSlots_v2`), output collection (`OutCollect_v1`) and entry (`SlotEntry_v1`), and those
two inside the token select's combine (`SelectCollect_v1`, `SelectEntry_v1`)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_io_shell as BX
from verity.ml.definitions.registry import io_shell as X
from verity.ml.definitions.registry.dead_free import TokenSelectCombineV2

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_words

MODEL, V = "served-model", 128256
SMALL = {"MODEL": "m", "LP": 3, "V": 1000}


def _bodies(st: dict, n: int, seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    out = [X.encode(st["MODEL"], [0] * st["LP"], 1, st["V"]), X.encode(st["MODEL"], [st["V"] - 1] * st["LP"], 999999, st["V"])]
    while len(out) < n:
        ids = [int(rng.integers(0, st["V"] // 10 ** int(rng.integers(0, X.id_width(st["V"]))))) for _ in range(st["LP"])]
        out.append(X.encode(st["MODEL"], ids, int(rng.integers(1, 3000)), st["V"]))
    return out


def _pad(body: bytes) -> bytes:
    return body + b" " * (-len(body) % X.BLOCK)


def _corrupt(st: dict, body: bytes) -> list[bytes]:
    """Each byte of the first field, the separator after it and max_tokens' field set to a space, '0', '9', 'a', '/', ':' and
    0xB0; the model's last byte and the closing brace changed; and the padding's first and last spaces made other JSON
    whitespace, the closing brace moved to the end, and a NUL."""
    _, fields, _ = X.layout(**st)
    (f0, w), (mt, mw) = fields[0], fields[-1]
    out = []
    for i in [*range(f0, f0 + w + 1), *range(mt, mt + mw)]:
        out += [body[:i] + bytes([c]) + body[i + 1:] for c in b" 09a/:\xb0" if c != body[i]]
    m_end, end = body.index(b'","prompt"') - 1, body.rindex(b"}")
    assert end < len(body) - 1 and set(body[end + 1:]) == {0x20}
    return out + [body[:m_end] + b"M" + body[m_end + 1:], body[:end] + b" " + body[end + 1:], body[:end] + body[end + 1:] + b"}",
                  body[:end + 1] + b"\n" + body[end + 2:], body[:-1] + b"\t", body[:-1] + b"\r", body[:-1] + b"\0"]


def _word(st: dict, bodies: list[bytes]) -> np.ndarray:
    arr = np.frombuffer(b"".join(bodies), dtype=np.uint8).reshape(len(bodies), -1).astype(np.uint32)
    return np.asarray(evaluate_batch(bind(X.IngressPrompt, **st), arr, kernel="reference"), dtype=np.uint64).reshape(len(bodies), -1)


def _ref(st: dict, bodies: list[bytes]) -> np.ndarray:
    return np.array([[ok, *ids, m] for ok, ids, m in (X.reference(b, **st) for b in bodies)], dtype=np.uint64)


def test_a_body_is_valid_json_in_the_layout_and_its_length_depends_on_the_statics_alone():
    a, b = X.encode(MODEL, [128000, 9, 0], 64, V), X.encode(MODEL, [1, 22, 333], 256, V)
    assert len(a) == len(b) == X.layout(MODEL, 3, V)[2] == X.length(MODEL, 3, V)
    assert json.loads(a) == {"model": MODEL, "prompt": [128000, 9, 0], "max_tokens": 64, "temperature": 0, "stream": True}
    assert X.reference(a, MODEL, 3, V) == (1, [128000, 9, 0], 64)


@pytest.mark.parametrize("st", [SMALL, {"MODEL": "a b", "LP": 1, "V": 10}, {"MODEL": MODEL, "LP": 16, "V": V},
                                {"MODEL": MODEL, "LP": 200, "V": V}], ids=["small", "w1", "served", "served-200"])
def test_a_body_is_whole_sha512_blocks_of_fewest_trailing_spaces(st):
    """The body is a whole number of 128-byte SHA-512 blocks: the request's JSON then the fewest spaces that get there, so a
    body one block shorter couldn't hold the JSON.  A body without its padding, or with a block more, has no layout."""
    body = X.encode(st["MODEL"], [st["V"] - 1] * st["LP"], 7, st["V"])
    unpadded = body.rstrip(b" ")
    assert len(body) % X.BLOCK == 0 and len(body) - X.BLOCK < len(unpadded) <= len(body) == X.length(**st)
    lits = dict(X.layout(**st)[0])
    assert all(lits.get(i) == 0x20 for i in range(len(unpadded), len(body)))
    assert [X.reference(b, **st)[0] for b in _corrupt(st, body)[-6:]] == [0] * 6
    for other in (unpadded, body + b" " * X.BLOCK):
        with pytest.raises(ValueError, match="the body is"):
            X.reference(other, **st)


@pytest.mark.parametrize("st", [SMALL, {"MODEL": "a b", "LP": 1, "V": 10}, {"MODEL": MODEL, "LP": 2, "V": V}], ids=["small", "w1", "served"])
def test_the_definition_is_the_reference_in_and_out_of_the_schema(st):
    bodies = _bodies(st, 24, 0)
    bad = [c for b in bodies[:3] for c in _corrupt(st, b)]
    got, want = _word(st, bodies + bad), _ref(st, bodies + bad)
    assert np.array_equal(got, want)
    assert got[: len(bodies), 0].all() and not got[len(bodies):, 0].all()


@pytest.mark.parametrize("body, why", [
    (_pad(b'{"model":"m","prompt":[ 01,  2,  3],"max_tokens":     5,"temperature":0,"stream":true}'), "a leading zero"),
    (_pad(b'{"model":"m","prompt":[ 1 ,  2,  3],"max_tokens":     5,"temperature":0,"stream":true}'), "a space after a digit"),
    (_pad(b'{"model":"m","prompt":[   ,  2,  3],"max_tokens":     5,"temperature":0,"stream":true}'), "no digit"),
    (_pad(b'{"model":"m","prompt":[  1,  2,  3],"max_tokens":     0,"temperature":0,"stream":true}'), "max_tokens 0"),
    (_pad(b'{"model":"m","prompt":[  1,  2,  3],"max_tokens":     5,"temperature":1,"stream":true}'), "temperature 1"),
    (_pad(b'{"model":"n","prompt":[  1,  2,  3],"max_tokens":     5,"temperature":0,"stream":true}'), "another model"),
])
def test_a_body_out_of_the_schema_is_not_ok(body, why):
    assert X.reference(body, **SMALL)[0] == 0, why
    assert _word(SMALL, [body])[0, 0] == 0, why


def test_an_id_past_the_vocabulary_is_not_ok():
    st = {"MODEL": "m", "LP": 1, "V": 500}
    body = X.encode("m", [499], 7, 500).replace(b"499", b"500")
    assert X.reference(body, **st) == (0, [500], 7)
    assert _word(st, [body]).tolist() == [[0, 500, 7]]


@pytest.mark.parametrize("kw", [{"MODEL": 'a"b', "LP": 1, "V": 10}, {"MODEL": "m", "LP": 0, "V": 10}, {"MODEL": "m", "LP": 1, "V": 1},
                                {"MODEL": "", "LP": 1, "V": 10}])
def test_statics_without_a_layout_are_refused(kw):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(X.IngressPrompt, **kw)


def test_the_token_feed_is_its_reference_down_a_prompt_and_past_it():
    """`TokenFeed_v1` chained down a class Program of NP = 5 prompt ids from each prompt length lp = 1..5: positions below lp
    take the prompt's ids, every later one the select it reads, and each carries (lp − p − 1, ids[p + 1:], 0s) on; the first
    position takes ids[0] whatever its slot's count, and the statics it has no layout for are refused: FIRST=1 too, after
    FIRST=True at the same NP is bound (as `window_weights.WEIGHTS_CLASS`'s feed is)."""
    NP, ids = 5, [7, 0xFFFFFFFF, 0, 123456, 1 << 31]
    sels = [1000 + p for p in range(NP)]
    first, rest = bind(X.TokenFeed, NP=NP, FIRST=True), bind(X.TokenFeed, NP=NP, FIRST=False)

    def run(fn, *rows):
        args = [np.array([r], dtype=np.uint64).astype(np.uint32) for r in rows]
        return [int(v) for v in np.asarray(evaluate_batch(fn, *args, kernel="reference"), dtype=np.uint64).reshape(-1)]

    for lp in range(1, NP + 1):
        slot, toks = [lp, *ids], []
        for p in range(NP):
            sel = None if p == 0 else sels[p - 1]
            got = run(first, slot) if sel is None else run(rest, slot, [sel])
            tok, carried = X.feed_reference(slot, sel, NP)
            assert got == [tok, *carried]
            toks.append(got[0])
            slot = got[1:]
            assert slot == [(lp - p - 1) & 0xFFFFFFFF, *ids[p + 1:], *[0] * (p + 1)]
        assert toks == ids[:lp] + sels[lp - 1:NP - 1]
    assert run(first, [0, *ids])[0] == ids[0]
    bind(X.TokenFeed, NP=2, FIRST=True)
    for kw in ({"NP": 0, "FIRST": True}, {"NP": 2, "FIRST": 1}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(X.TokenFeed, **kw)


def _run(fn, *rows):
    args = [np.array([r], dtype=np.uint64).astype(np.uint32) for r in rows]
    return [int(v) for v in np.asarray(evaluate_batch(fn, *args, kernel="reference"), dtype=np.uint64).reshape(-1)]


def _lane(tok: int, slot: list[int], occ: int, head: list[int]) -> list[int]:
    return [tok, *slot, occ, *head]


def test_the_scan_slots_read_is_its_reference_and_reads_each_field_where_it_is_packed():
    """`ScanSlots_v2` on a 40-word row with two lanes: random rows give `scan_reference`'s lanes, and a row packed field by
    field (occupancy, lp, max_tokens, and ids at `position`'s index, K0 = 2 in the first cell and K = 3 in each later one)
    gives each lane its first position's `TokenFeed_v1{FIRST}` on its row (ids cut to IDB bits), its occupancy word and its
    head, at the prompt lengths where the decrement's borrow runs furthest (0, 1, 2^31 and 2^32 − 1). A field past the
    row is refused, and its gates are the decrements' 30 ANDs a lane."""
    NS, NP, IDB, K0, K = 40, 4, 17, 2, 3
    OCC, LP, MAX, IDS = (3, 300), (40, 340), (80, 380), (120, 420)
    st = {"NS": NS, "NP": NP, "IDB": IDB, "K0": K0, "K": K, "OCC": OCC, "LP": LP, "MAX": MAX, "IDS": IDS}
    fn = bind(BX.ScanSlotsV2, **st)
    assert BX.ScanSlotsV2.traced(SimpleNamespace(**st)).counts["and"] == 30 * 2
    state = np.random.default_rng(1).integers(0, 1 << 16, size=(3, NS), dtype=np.uint64)
    got = np.asarray(evaluate_words(fn, state.astype(np.uint32)), dtype=np.uint64)
    for i in range(3):
        ref = X.scan_reference([int(w) for w in state[i]], NP, IDB, K0, K, OCC, LP, MAX, IDS)
        assert [int(v) for v in got[i].reshape(-1)] == [v for lane in ref for v in _lane(*lane)]
    for lp0, lp1 in ((3, 0), (1, 1 << 31), (X.M32, 2)):
        lanes = [(1, lp0, 64, [5, (1 << IDB) - 1, 0, 77]), (0, lp1, 9, [0, 0, 0, 0])]
        x = 0
        for (occ, lp, m, ids), o, l, mx, at in zip(lanes, OCC, LP, MAX, IDS):
            x |= occ << o | lp << l | m << mx
            x |= sum(v << at + IDB * X.position(K0, K, p) for p, v in enumerate(ids))
        words = [[x >> 16 * k & 0xFFFF for k in range(NS)]]
        out = [int(v) for v in np.asarray(evaluate_words(fn, np.array(words, dtype=np.uint32)), dtype=np.uint64).reshape(-1)]
        assert out == [v for occ, lp, m, ids in lanes
                       for v in _lane(*X.feed_reference([lp, *ids], None, NP), occ, [lp, m])]
        assert out[1] == (lp0 - 1) & X.M32 and out[2:NP + 1] == lanes[0][3][1:] and out[NP + 1] == 0
    assert [X.position(K0, K, p) for p in range(NP)] == [0, 1, 3, 4]
    for bad in ({"IDS": (120, 16 * NS - IDB)}, {"IDB": 33}, {"OCC": (3,)}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(BX.ScanSlotsV2, **{**st, **bad})


@pytest.mark.parametrize("KO", [1, 2, 5])
def test_the_output_collector_keeps_the_ko_samples_from_the_prompts_last_position_on(KO):
    """`OutCollect_v1` chained down NP + KO positions from each prompt length lp = 1..4: each step is `collect_reference`,
    and after the last, ids[i] is the sample at position lp − 1 + i, ``max_tokens`` carried through."""
    first, rest = bind(X.OutCollect, KO=KO, FIRST=True), bind(X.OutCollect, KO=KO, FIRST=False)
    for lp in range(1, 5):
        state = _run(first, [lp, 9], [100])
        assert state == X.collect_reference([lp, 9], 100, KO, True)
        for p in range(1, lp + KO):
            nxt = _run(rest, state, [100 + p])
            assert nxt == X.collect_reference(state, 100 + p, KO, False)
            state = nxt
        assert state[1:] == [9, *[100 + lp - 1 + i for i in range(KO)]]
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(X.OutCollect, KO=0, FIRST=True)


@pytest.mark.parametrize("KO, STOP", [(2, ()), (5, (103,)), (5, (102, 104))], ids=["no-stop", "one-stop", "two-stops"])
def test_a_slots_entry_counts_its_outputs_cuts_its_ids_and_is_zero_when_empty(KO, STOP):
    """`SlotEntry_v1` against `entry_reference` at each occupancy word, ``max_tokens`` (2^31 and 2^32 − 1 unsigned) and
    prompt length lp (the last state's w = lp − NP − KO): an empty slot's every word is zero; an occupied slot's n is
    min(max_tokens, the first stop's index + 1) and its ids output i's low 17 bits for i < n, zeros past it (ids carry a bit
    at 2^20 the cut drops), except that a request the class can't hold, its prompt not 1 to NP ids or that n past KO, has
    n = KO + 1 and no ids."""
    NP = 3
    e, ids = bind(X.SlotEntry, KO=KO, IDB=17, NP=NP, STOP=STOP), [100 + i for i in range(KO)]
    for occ in (0, 1, 2, 3):
        for m in (0, 1, 2, 4, 9, 1 << 31, X.M32):
            for lp in (0, 1, NP, NP + 1, 1 << 31):
                state = [(lp - NP - KO) & X.M32, m, *[x | (1 << 20 if x not in STOP else 0) for x in ids]]
                bits = _run(e, [occ], state)
                words = [sum(bits[16 * k + b] << b for b in range(16)) for k in range(len(bits) // 16)]
                assert words == X.entry_reference(occ, state, KO, 17, NP, STOP)
                n = min([m, *[i + 1 for i, x in enumerate(ids) if x in STOP]]) if 1 <= lp <= NP else KO + 1
                n = 0 if not occ & 1 else n if n <= KO else KO + 1
                packed = sum(w << 16 * k for k, w in enumerate(words))
                assert packed == n | sum(x << 32 + 17 * i for i, x in enumerate(ids[:n] if n <= KO else ()))
    assert X.entry_words(64, 17) == 70
    for bad in ({"STOP": (1, 1)}, {"NP": 0}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(X.SlotEntry, **{"KO": 2, "IDB": 17, "NP": 4, "STOP": (), **bad})


@pytest.mark.parametrize("first", [True, False], ids=["first", "later"])
def test_a_select_collected_and_the_last_ones_entry_are_the_combine_then_the_collector_and_the_entry(first):
    """`SelectCollect_v1` is `TokenSelectCombine_v2`'s select and `OutCollect_v1` over it, and `SelectEntry_v1` those and
    `SlotEntry_v1` over the state, on groups' bests drawn from a few bf16 values (ties and -inf among them), their indices,
    heads or states whose w is in and out of the collector's KO, and both occupancy words."""
    V, C, KO, NP, STOP = 200, 64, 3, 3, (107,)
    combine = bind(TokenSelectCombineV2, V=V, C=C)
    sc = bind(X.SelectCollect, V=V, C=C, KO=KO, FIRST=first)
    se = bind(X.SelectEntry, V=V, C=C, KO=KO, FIRST=first, IDB=17, NP=NP, STOP=STOP)
    words = (np.array([-1.0, 0.0, 1.0, 2.0, -np.inf], dtype=np.float32).view(np.uint32) >> 16).tolist()
    rng = np.random.default_rng(5)
    for _ in range(24):
        x0, best = int(rng.choice(words)), [int(w) for w in rng.choice(words, 4)]
        idx = [int(i) for i in rng.integers(0, C, 4)]
        w = int(rng.integers(-NP - KO, 3)) & X.M32
        a = [int(rng.integers(1, 5)), 9] if first else [w, 9, *[100 + i for i in range(KO)]]
        tok = _run(combine, [x0], best, idx)[0]
        state = X.collect_reference(a, tok, KO, first)
        assert _run(sc, [x0], best, idx, a) == [tok, *state]
        for occ in (0, 1):
            bits = _run(se, [x0], best, idx, a, [occ])
            got = [sum(bits[16 * k + b] << b for b in range(16)) for k in range(len(bits) // 16)]
            assert got == X.entry_reference(occ, state, KO, 17, NP, STOP)
    for fn, bad in ((X.SelectCollect, {"V": 1, "C": 1, "KO": 2, "FIRST": True}),
                    (X.SelectEntry, {"V": 200, "C": 64, "KO": 2, "FIRST": True, "IDB": 17, "NP": 4, "STOP": (1, 1)})):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(fn, **bad)


@pytest.mark.parametrize("st", [SMALL, {"MODEL": MODEL, "LP": 2, "V": V}], ids=["small", "served"])
def test_the_boolean_definition_is_its_word(st):
    target = bind(BX.IngressPromptV2, **st)
    assert SW.boolean_version(word_of(target)).id == target.id
    bodies = _bodies(st, 12, 1)
    bodies += [c for b in bodies[:2] for c in _corrupt(st, b)]
    arr = np.frombuffer(b"".join(bodies), dtype=np.uint8).reshape(len(bodies), -1).astype(np.uint32)
    got = np.asarray(evaluate_words(target, arr), dtype=np.uint64).reshape(len(bodies), -1)
    assert np.array_equal(got, _ref(st, bodies))
