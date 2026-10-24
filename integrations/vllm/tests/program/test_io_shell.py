"""`registry.io_shell` (ingress v0): the encoder's bytes in the layout, padded to whole SHA-512 blocks, `IngressPrompt_v1` as
`reference` on bodies in the schema and on each way out of it (other padding among them), the statics it refuses, and the
Boolean Definition against its word."""
from __future__ import annotations

import json

import numpy as np
import pytest
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_io_shell as BX
from verity_vllm.program.registry import io_shell as X

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


@pytest.mark.parametrize("st", [SMALL, {"MODEL": MODEL, "LP": 2, "V": V}], ids=["small", "served"])
def test_the_boolean_definition_is_its_word(st):
    target = bind(BX.IngressPromptV2, **st)
    assert SW.boolean_version(word_of(target)).id == target.id
    bodies = _bodies(st, 12, 1)
    bodies += [c for b in bodies[:2] for c in _corrupt(st, b)]
    arr = np.frombuffer(b"".join(bodies), dtype=np.uint8).reshape(len(bodies), -1).astype(np.uint32)
    got = np.asarray(evaluate_words(target, arr), dtype=np.uint64).reshape(len(bodies), -1)
    assert np.array_equal(got, _ref(st, bodies))
