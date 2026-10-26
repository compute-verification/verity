"""`registry.io_shell` (ingress v0): the encoder's bytes in the layout, padded to whole SHA-512 blocks, `IngressPrompt_v1` as
`reference` on bodies in the schema and on each way out of it (other padding among them), the statics it refuses, the
Boolean Definition against its word, `TokenFeed_v1` down a class Program's positions, a slot's `SlotRow_v1` and
`SlotMask_v1`, and the network certifier's scan state read (`ScanSlots_v1`), output collection (`OutCollect_v1`) and
entry (`SlotEntry_v1`)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_io_shell as BX
from verity.ml.definitions.registry import io_shell as X

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
    position takes ids[0] whatever its slot's count, and the statics it has no layout for are refused."""
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
    for kw in ({"NP": 0, "FIRST": True}, {"NP": 2, "FIRST": 1}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(X.TokenFeed, **kw)


def test_a_slots_row_opens_into_its_occupancy_and_feed_and_an_empty_slot_outputs_no_token():
    """`SlotRow_v1` copies a slot's ingress row (occupancy, lp, ids) out as its occupancy word and the first `TokenFeed_v1`'s
    slot; `SlotMask_v1` passes the select through while the occupancy word's low bit is 1 and otherwise outputs `EMPTY`,
    which no token id is, at every occupancy word and token tried."""
    def run(fn, *rows):
        args = [np.array([r], dtype=np.uint64).astype(np.uint32) for r in rows]
        return [int(v) for v in np.asarray(evaluate_batch(fn, *args, kernel="reference"), dtype=np.uint64).reshape(-1)]

    row, ids = bind(X.SlotRow, NP=3), [7, 0, 0xFFFFFFFF]
    for occ, lp in ((1, 3), (0, 0), (0xFFFFFFFE, 1)):
        assert run(row, [occ, lp, *ids]) == [occ, lp, *ids]
    mask = bind(X.SlotMask)
    for occ in (0, 1, 2, 3, 0x80000000, 0xFFFFFFFF):
        for tok in (0, 5, 128255, 0xFFFFFFFF):
            assert run(mask, [occ], [tok]) == [X.mask_reference(occ, tok)] == [tok if occ & 1 else X.EMPTY]
    assert X.EMPTY >= 1 << 31
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(X.SlotRow, NP=0)


def _run(fn, *rows):
    args = [np.array([r], dtype=np.uint64).astype(np.uint32) for r in rows]
    return [int(v) for v in np.asarray(evaluate_batch(fn, *args, kernel="reference"), dtype=np.uint64).reshape(-1)]


def test_the_scan_slots_read_is_its_reference_and_reads_each_field_where_it_is_packed():
    """`ScanSlots_v1` on a 40-word row with two lanes: random rows give `slots_reference`'s rows and heads, and a row packed
    field by field (occupancy, lp, max_tokens, and ids at `position`'s index, K0 = 2 in the first cell and K = 3 in each
    later one) reads each lane's fields back, ids cut to IDB bits.  A field past the row is refused."""
    NS, NP, IDB, K0, K = 40, 4, 17, 2, 3
    OCC, LP, MAX, IDS = (3, 300), (40, 340), (80, 380), (120, 420)
    st = {"NS": NS, "NP": NP, "IDB": IDB, "K0": K0, "K": K, "OCC": OCC, "LP": LP, "MAX": MAX, "IDS": IDS}
    fn = bind(X.ScanSlots, **st)
    state = np.random.default_rng(1).integers(0, 1 << 16, size=(3, NS), dtype=np.uint64)
    got = np.asarray(evaluate_words(fn, state.astype(np.uint32)), dtype=np.uint64)
    for i in range(3):
        ref = X.slots_reference([int(w) for w in state[i]], NP, IDB, K0, K, OCC, LP, MAX, IDS)
        assert [int(v) for v in got[i].reshape(-1)] == [v for row, head in ref for v in (*row, *head)]
    lanes = [(1, 3, 64, [5, (1 << IDB) - 1, 0, 77]), (0, 0, 9, [0, 0, 0, 0])]
    x = 0
    for (occ, lp, m, ids), o, l, mx, at in zip(lanes, OCC, LP, MAX, IDS):
        x |= occ << o | lp << l | m << mx
        x |= sum(v << at + IDB * X.position(K0, K, p) for p, v in enumerate(ids))
    words = [[x >> 16 * k & 0xFFFF for k in range(NS)]]
    out = [int(v) for v in np.asarray(evaluate_words(fn, np.array(words, dtype=np.uint32)), dtype=np.uint64).reshape(-1)]
    assert out == [v for occ, lp, m, ids in lanes for v in (occ, lp, *ids, lp, m)]
    assert [X.position(K0, K, p) for p in range(NP)] == [0, 1, 3, 4]
    for bad in ({"IDS": (120, 16 * NS - IDB)}, {"IDB": 33}, {"OCC": (3,)}):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(X.ScanSlots, **{**st, **bad})


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


@pytest.mark.parametrize("st", [SMALL, {"MODEL": MODEL, "LP": 2, "V": V}], ids=["small", "served"])
def test_the_boolean_definition_is_its_word(st):
    target = bind(BX.IngressPromptV2, **st)
    assert SW.boolean_version(word_of(target)).id == target.id
    bodies = _bodies(st, 12, 1)
    bodies += [c for b in bodies[:2] for c in _corrupt(st, b)]
    arr = np.frombuffer(b"".join(bodies), dtype=np.uint8).reshape(len(bodies), -1).astype(np.uint32)
    got = np.asarray(evaluate_words(target, arr), dtype=np.uint64).reshape(len(bodies), -1)
    assert np.array_equal(got, _ref(st, bodies))
