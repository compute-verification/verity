"""The seeded weights and their registration root (`e2e_h200.served_weights`)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments import hm96
from verity.ml.commitments.gates import embedding
from e2e_h200 import served_weights as W

FIXTURE = Path(__file__).parent / "fixtures" / "weights-v1.json"
SEED, KEY, PROG = W.TEST_SEED, W.TEST_SALT_KEY, W.TEST_PROGRAM
XOR8 = PrimitiveDefinition("SeededWeightsTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)


def reading_program(shape: W.Shape, names=None, bits=None) -> Program:
    """A program whose root parameters are the tensors (`names`, default all), each at its elements' width (`bits`
    overrides), named as the tensor."""
    ts = [t for t in W.tensors(shape) if names is None or t.name in names]
    width = {t.name: (bits or {}).get(t.name, {"s8": 8, "f32": 32, "bf16": 16}[t.dtype]) for t in ts}

    def sig(S):
        return tuple((t.name, Array(t.rows, Array(t.cols, Value(width[t.name]))) if t.rows > 1
                      else Array(t.cols, Value(width[t.name]))) for t in ts), Value(8)

    def body(B, S, *params):
        a = next(p for t, p in zip(ts, params) if t.kind == "s8")
        return B.call(XOR8, a[0][0], a[0][1])

    return Program(bind(CompositeDefinition("SeededWeightsTestReader", 0, (), sig, body, register=False)))


def test_fixture_is_the_generators():
    """The committed fixture is what the generator and the registration give now: a change to either moves it."""
    assert json.loads(W.dumps(W.fixture())) == json.loads(FIXTURE.read_text())


def test_row_stream_is_sha512_counter():
    key = hashlib.sha512(b"verity/seeded-weights/v1\0" + SEED + b"\0l0.wq").digest()
    want = b"".join(hashlib.sha512(key + (5).to_bytes(8, "little") + i.to_bytes(8, "little")).digest() for i in range(3))
    assert W.row_bytes(SEED, "l0.wq", 5, 150) == want[:150]
    codes = W.generate(W.H200, SEED, "l0.wq", [5])[0]
    assert codes.dtype == np.int8 and codes[:150].tobytes() == want[:150]


def test_element_ranges():
    for t in W.tensors(W.PHASE1):
        x = W.generate(W.PHASE1, SEED, t.name)
        if t.kind == "scale":
            assert x.dtype == np.float32 and (x >= 2.0 ** t.exponent).all() and (x < 2.0 ** (t.exponent + 1)).all()
        elif t.kind == "gain":
            f = (x.astype(np.uint32) << 16).view(np.float32)
            assert ((f >= 1) & (f < 2)).all()
        elif t.kind == "embed":
            f = np.abs((x.astype(np.uint32) << 16).view(np.float32))
            assert ((f >= 0.5) & (f < 1)).all()
    assert W.tensor(W.H200, "l0.wq.scale").exponent == W.scale_exponent(16384) - 3 == -17
    assert W.tensor(W.H200, "l0.wdown.scale").exponent == -15


def test_h200_sizes():
    ts = W.tensors(W.H200)
    layer_s8 = sum(t.nbytes for t in ts if t.kind == "s8" and t.name.startswith("l"))
    assert layer_s8 == 139_695_489_024 and f"{layer_s8 / 2**30:.2f}" == "130.10"
    assert all(t.leaves & (t.leaves - 1) == 0 for t in ts), "a tree with pads"
    assert [t.offset for t in ts] == [sum(u.nbytes for u in ts[:i]) for i in range(len(ts))]
    assert W.tensor(W.H200, "l0.wup").words == 4096 and W.tensor(W.H200, "l0.wup").leaves == 65536 * 2


def test_stream_and_write(tmp_path):
    whole = W.stream(W.TINY, SEED, 0, W.stream_bytes(W.TINY))
    assert whole == b"".join(W.tensor_bytes(W.TINY, SEED, t.name) for t in W.tensors(W.TINY))
    assert W.stream(W.TINY, SEED, 1000, 9000) == whole[1000:9000]
    out = tmp_path / "w.bin"
    W.write(W.TINY, SEED, out, workers=2, chunk_bytes=4096)
    assert out.read_bytes() == whole
    m = W.manifest(W.TINY, SEED, digests=True, workers=2)
    assert m == W.manifest(W.TINY, SEED, digests=True)
    assert [t["sha256"] for t in m["tensors"]] == [hashlib.sha256(W.tensor_bytes(W.TINY, SEED, t.name)).hexdigest()
                                                   for t in W.tensors(W.TINY)]


def test_root_is_core_commit():
    """Each tensor's root and entry are core's ``Registered.commit`` of its rows under the same salts, in one chunk or many,
    in one process or two."""
    roots = W.root(W.TINY, SEED, KEY, PROG)
    for t in W.tensors(W.TINY):
        c = W.commit(W.TINY, SEED, KEY, PROG, t.name)
        assert c.root == roots[t.name] and c.entry() == W.entry(W.TINY, PROG, t.name, roots[t.name])
    assert W.root(W.TINY, SEED, KEY, PROG, chunk_bytes=64, workers=2) == roots
    assert W.root(W.TINY, SEED, KEY, PROG, ["l0.wq", "head.scale"]) == {k: roots[k] for k in ("l0.wq", "head.scale")}


def test_root_opens_under_the_verifiers_entries():
    """For a program reading the tensors, each entry is what the verifier derives from that program
    (``registered_entries``, the values held across windows: leaves, words, word_bits, domain), and a row of each kind
    opens against the parallel root under that domain."""
    prog = reading_program(W.TINY)
    r = W.reader(W.TINY, prog)
    ts = W.tensors(W.TINY)
    words = {t.name: W.registered_words(t, r.word_bits(t.name)) for t in ts}
    want = REG.registered_entries(prog, words, {t.name: t.schema for t in ts}, window="ab" * 64, fixed=list(words))
    roots = W.root(W.TINY, SEED, KEY, prog, chunk_bytes=256, workers=2)
    for e in W.entries(W.TINY, prog, roots):
        v = want[e["value"]]
        assert (e["leaves"], e["words"], e.get("word_bits", 16), e["domain"]) == (
            v["leaves"], v["words"], v["word_bits"], v["domain"])
    assert (r.word_bits("l0.wq.scale"), words["l0.wq.scale"], words["l0.wq"]) == (32, 1, W.TINY.row_codes // 2)
    for name in ("tok", "pos", "l0.attn_norm", "l0.wq", "l0.wq.scale", "head"):
        t, c = W.tensor(W.TINY, name), W.commit(W.TINY, SEED, KEY, prog, name)
        assert c.root == roots[name] and c.domain.domain_id.hex() == want[name]["domain"]
        k = t.leaves - 1
        assert REG.verify_read(c.domain, roots[name], k, c.bcs[k], c.path(k), t.schema)
    plain = W.root(W.TINY, SEED, KEY, PROG)
    assert W.registration_digest(W.TINY, prog, roots) != W.registration_digest(W.TINY, PROG, plain)


def test_tok_is_one_coordinate_a_row():
    """``tok`` is (H, V) in ``hm96-sha512/row/v2``: row j, coordinate j over the tokens, opens whole by
    an ordinary registered read at position j and at no other, and the unit that reads it selects token t's coordinate
    j, the generator's element t of tok's generation row j."""
    s, prog = W.TINY, reading_program(W.TINY, ["tok", "l0.wq"])
    t = W.tensor(s, "tok")
    assert (t.rows, t.cols, t.words, t.leaves, t.schema) == (s.width, s.vocab, s.vocab, s.width, REG.SCHEMA_V2)
    c = W.commit(s, SEED, KEY, prog, "tok")
    assert W.root(s, SEED, KEY, prog, ["tok"])["tok"] == c.root
    inst, table = hm96.SHA512, W.generate(s, SEED, "tok")
    for j in (0, 5, s.width - 1):
        row = list(c.rows[j])
        bc = inst.commit_string(inst.default_key, REG.row_digest(row, 16, REG.SCHEMA_V2), c.salts[j])
        assert bc == c.bcs[j]
        assert REG.verify_read(c.domain, c.root, j, bc, c.path(j), REG.SCHEMA_V2)
        assert not REG.verify_read(c.domain, c.root, j ^ 1, bc, c.path(j), REG.SCHEMA_V2)
        raw = W.row_bytes(SEED, "tok", j, 2 * s.vocab)
        for tok in (0, 97, s.vocab - 1):
            u = int.from_bytes(raw[2 * tok:2 * tok + 2], "little")
            assert row[tok] == u & 0x8000 | 0x3F00 | u & 0x7F == table[j, tok]
    assert [c.rows[j][97] for j in range(s.width)] == table[:, 97].tolist()


def test_tok_one_token_a_row():
    """With `Shape.tok_rows` ``"token"``, ``tok`` is (V, H) in row/v2: token r's embedding opens at position r and at
    no other."""
    s = dataclasses.replace(W.TINY, name="tiny-tok-token", tok_rows="token")
    prog = reading_program(s, ["tok", "l0.wq"])
    t = W.tensor(s, "tok")
    assert (t.rows, t.cols, t.words, t.leaves, t.schema) == (s.vocab, s.width, s.width, s.vocab, REG.SCHEMA_V2)
    c = W.commit(s, SEED, KEY, prog, "tok")
    assert W.root(s, SEED, KEY, prog, ["tok"])["tok"] == c.root
    for tok in (0, 97, s.vocab - 1):
        assert list(c.rows[tok]) == W.generate(s, SEED, "tok", [tok])[0].tolist()
        assert REG.verify_read(c.domain, c.root, tok, c.bcs[tok], c.path(tok), REG.SCHEMA_V2)
        assert not REG.verify_read(c.domain, c.root, tok ^ 1, c.bcs[tok], c.path(tok), REG.SCHEMA_V2)


def test_tok_in_the_embed_units_format():
    """With `Shape.tok_rows` ``"token"`` and `Shape.tok_schema` ``hm96-sha512/row-seg/v1``, a token's row opens under
    ``EmbeddingRead_v1``'s reference at the hidden token, and at no other."""
    s = dataclasses.replace(W.TINY, name="tiny-row-seg", tok_rows="token", tok_schema=REG.SCHEMA_SEG)
    prog = reading_program(s, ["tok", "l0.wq"])
    assert W.tensor(s, "tok").schema == embedding.SCHEMA
    c = W.commit(s, SEED, KEY, prog, "tok")
    assert W.root(s, SEED, KEY, prog, ["tok"])["tok"] == c.root
    assert c.root != W.commit(dataclasses.replace(s, tok_schema=REG.SCHEMA_V2), SEED, KEY, prog, "tok").root
    for tok in (0, 97, s.vocab - 1):
        row = W.generate(s, SEED, "tok", [tok])[0].tolist()
        path = c.path(tok)
        h, ok = embedding.reference(s.vocab, tok, c.root, c.domain.domain_id, row, c.salts[tok],
                                    [path[i:i + 64] for i in range(0, len(path), 64)])
        assert ok == 1 and h == row
        _, bad = embedding.reference(s.vocab, tok ^ 1, c.root, c.domain.domain_id, row, c.salts[tok],
                                     [path[i:i + 64] for i in range(0, len(path), 64)])
        assert bad == 0


def test_registration_binds_seed_key_and_program():
    roots = W.root(W.TINY, SEED, KEY, PROG)
    d = W.registration_digest(W.TINY, PROG, roots)
    assert W.registration_digest(W.TINY, PROG, W.root(W.TINY, SEED, bytes(32), PROG)) != d
    assert W.registration_digest(W.TINY, PROG, W.root(W.TINY, b"another seed", KEY, PROG)) != d
    other = "0" * 128
    assert W.registration_digest(W.TINY, other, W.root(W.TINY, SEED, KEY, other)) != d
    assert W.root(W.TINY, SEED, KEY, other)["l0.wq"] != roots["l0.wq"]


def test_refusals():
    with pytest.raises(ValueError, match="salt key is 32 bytes"):
        W.root(W.TINY, SEED, b"short", PROG)
    with pytest.raises(ValueError, match="has no tensor"):
        W.generate(W.TINY, SEED, "l9.wq")
    with pytest.raises(ValueError, match="has rows"):
        W.generate(W.TINY, SEED, "l0.wq", [W.tensor(W.TINY, "l0.wq").rows])
    with pytest.raises(ValueError, match="multiple of row_codes"):
        W.tensors(W.Shape("bad", 48, 1, 1, 16, 1, 64, 256, 8, 32))
    with pytest.raises(ValueError, match="'coordinate' or 'token'"):
        W.tensors(dataclasses.replace(W.TINY, tok_rows="column"))
    with pytest.raises(ValueError, match="program digest"):
        W.root(W.TINY, SEED, KEY, "the-old-string-program")
    with pytest.raises(ValueError, match="has no parameter 'l0.wk'"):
        W.root(W.TINY, SEED, KEY, reading_program(W.TINY, ["tok", "l0.wq"]), ["l0.wk"])
    with pytest.raises(ValueError, match="reads 8 rows of it"):
        W.reader(W.TINY, reading_program(W.TINY, ["l0.wq.scale", "l0.wq"], bits={"l0.wq.scale": 8}))
