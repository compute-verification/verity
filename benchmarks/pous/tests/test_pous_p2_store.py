"""The encoded weights store (`p2_v1/store.py`, CMP-95 item 2): the real store's layout JSON and the toy fixture
(`p2_v1/store-toy/`) regenerate as pinned, the layout, `W` and the staging buffers by their digests; the toy's `C*` decodes,
layer by layer, to its `W`'s tensors by `reference.decode`, by the registered codec and by `layout.decode_tensor`,
passes the public-encoder setup with the pinned root, and refuses each negative block with its flag; the spot vectors
decode at the real store's indices."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity")

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from p2_v1 import reference  # noqa: E402
from p2_v1 import store as S  # noqa: E402

from verity.core.protocols.pous import protocol, setup  # noqa: E402
from verity.core.protocols.pous.layout import Layout, decode_tensor  # noqa: E402
from verity.core.protocols.pous.schemes.p2 import P2, P2Params, decode_segment, encode_segment, expand_prefix  # noqa: E402
from verity.core.protocols.pous.verifier import VerifierKey  # noqa: E402

TOY = json.loads((S.TOY_DIR / "toy.json").read_text())
SPOT = json.loads((S.TOY_DIR / "spot.json").read_text())
C_STAR = (S.TOY_DIR / "c-star.bin").read_bytes()
STORE = S.toy_store(TOY)
PLAIN = S.toy_plain(STORE)
W, STAGING = PLAIN["w.bin"], PLAIN["staging.bin"]
SALT = bytes.fromhex(TOY["salt"])
NB = STORE.params.block_bytes


def _segments(c_star: bytes = C_STAR) -> list[list[bytes]]:
    n = STORE.params.blocks_per_segment
    return [[c_star[NB * (s * n + v):NB * (s * n + v + 1)] for v in range(n)] for s in range(STORE.params.segments)]


def test_layout_json_regenerates():
    doc = S._dump(S.build(S.MQA16K).to_json())
    assert hashlib.sha512(doc.encode()).hexdigest() == S.LAYOUT_SHA512, "a layout change: pin the new digest"


def test_real_layout():
    """Every tensor from a block boundary, back to back by blocks, so no block holds two; W to whole segments."""
    store, pb = S.build(S.MQA16K), 2054
    at = 0
    for t in store.tensors:
        assert (t.first_block, t.offset, t.blocks) == (at, at * pb, -(-t.size // pb))
        assert store.layout.placement(t.name).offset == t.offset and t.staging_offset % S.STAGING_ALIGN == 0
        at += t.blocks
    p = store.params
    assert (at, p.blocks_per_segment, p.segments, p.blocks) == (68_013_623, 65_536, 1_038, 68_026_368)
    assert (p.w_bytes, p.blocks * p.block_bytes) == (139_726_159_872, 139_862_212_608)
    assert sum(t.size for t in store.tensors) == 139_699_683_328
    for l in range(52):
        qkv, o, up, down = store.layer(l)
        assert (qkv.n, qkv.k, o.n, o.k, up.n, up.k, down.n, down.k) == (16512, 16384, 16384, 16384, 65536, 16384, 16384, 65536)
        assert qkv.first_block == 1_307_915 * l and store.staging_bytes(l) == 2_686_451_712
    (head,) = store.layer(52)
    assert (head.name, head.first_block, head.blocks) == ("lm_head.weight", 68_011_580, 2_043)
    aligned = Layout.of(P2Params(S.W_BITS, S.C0, S.PER_SEGMENT), [(t.name, t.size) for t in store.tensors], align=True)
    assert aligned.params.segments > p.segments + 50


def test_toy_regenerates():
    """The toy's W, commitment, salt and key prefix from the generator's code, the default toy's; its files by their
    pinned hashes."""
    assert TOY["layout"] == json.loads(S._dump(S.build(S.toy_shape(), 11).to_json()))
    assert {t.layer for t in STORE.tensors} == {0, 1, 2} and STORE.layer(2)[0].name == "lm_head.weight"
    straddles = [t.name for t in STORE.tensors if len(STORE.layout.segments(t.name)) > 1]
    assert straddles == ["model.layers.0.mlp.down_proj.weight"] and STORE.layer(1)[0].first_block % 11
    commitment = setup.commit(STORE.params, W)
    assert commitment.hex() == TOY["commitment"]
    assert setup.draw_salt(bytes.fromhex(TOY["source"]), commitment) == SALT
    assert expand_prefix(STORE.params, SALT).hex() == TOY["prefix"]
    for data, key in ((W, "w_sha512"), (C_STAR, "c_star_sha512"), (STAGING, "staging_sha512")):
        assert hashlib.sha512(data).hexdigest() == TOY[key]


def test_toy_decode():
    """Each layer's staging buffer is its tensors' bytes of W, by `reference.decode`, and every tensor and segment decodes
    the same by the registered codec, `layout.decode_tensor` and `protocol.decode_tensor`."""
    def block(i: int) -> bytes:
        return C_STAR[NB * i:NB * (i + 1)]

    for l, start, size, sha in TOY["staging"]:
        buf = S.decode_layer(STORE, C_STAR, SALT, l)
        assert buf == STAGING[start:start + size] and hashlib.sha512(buf).hexdigest() == sha
        for t in STORE.layer(l):
            assert buf[t.staging_offset:t.staging_offset + t.size] == W[t.offset:t.offset + t.size]
            assert decode_tensor(STORE.layout, None, SALT, t.name, block) == W[t.offset:t.offset + t.size]
            assert protocol.decode_tensor(P2(STORE.params), STORE.layout, SALT, t.name, block) == W[t.offset:t.offset + t.size]
    seg = STORE.params.segment_bytes
    for s, blocks in enumerate(_segments()):
        assert decode_segment(STORE.params, None, SALT, s, blocks) == W[s * seg:(s + 1) * seg]


def test_toy_public_setup():
    """`Dec(C*)` matches the commitment, so `C* = Enc(W)` (each `D_j` is a bijection on `[0, p)`), with the pinned root."""
    vk = setup.verifier_key_from_encoding(STORE.params, None, bytes.fromhex(TOY["commitment"]), SALT, _segments())
    assert vk.root.hex() == TOY["root"] == VerifierKey.from_segments(STORE.params, SALT, _segments()).root.hex()


def test_toy_refusals():
    assert S.refusal(STORE, C_STAR, SALT, 0) is None
    for neg in TOY["negatives"]:
        bad = S.negative(neg, C_STAR, STORE.params)
        assert S.refusal(STORE, bad, SALT, neg["layer"]) == neg["flag"], neg["id"]
        assert S.refusal(STORE, bad, SALT, 1) is None
    other = bytes(SALT[:-1]) + bytes([SALT[-1] ^ 1])
    assert S.refusal(STORE, C_STAR, other, 0) == 2


def test_spot_vectors():
    """Blocks at the real store's indices (segments up to 1,037, v up to 65,535): keys, then decode."""
    real = S.build(S.MQA16K).params
    salt = bytes.fromhex(SPOT["salt"])
    assert SPOT["params"] == real.manifest() and salt == SALT
    prefix = expand_prefix(real, salt)
    assert prefix.hex() == SPOT["prefix"]
    ref = reference.Params(real.w, real.c0, real.payload_bytes, real.blocks_per_segment, real.segments)
    sha = lambda b: hashlib.sha512(b).hexdigest()  # noqa: E731
    spot_c = (S.TOY_DIR / "spot-c.bin").read_bytes()
    assert sha(spot_c) == SPOT["c_sha512"] and len(spot_c) == NB * len(SPOT["vectors"])
    assert [v["id"] for v in SPOT["vectors"]] == ["w-first", "qkv0-last", "down51-first", "head-last", "w-last"]
    for j, v in enumerate(SPOT["vectors"]):
        assert divmod(v["i"], real.blocks_per_segment) == (v["s"], v["v"])
        k, t, attempt = reference.keys_after(prefix, ref, v["s"], v["v"])
        assert (sha(k.to_bytes(NB, "big")), sha(t.to_bytes(NB, "big")), attempt) == (v["K_sha512"], v["t_sha512"], v["t_attempt"])
        m = S.decode_block(real, prefix, v["i"], spot_c[NB * j:NB * (j + 1)])
        assert m == S.spot_payload(v["i"], v["used"], real.payload_bytes) and sha(m) == v["m_sha512"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="about two minutes of roots: VERITY_POUS_SLOW=1")
def test_toy_reencodes():
    seg = STORE.params.segment_bytes
    for s, blocks in enumerate(_segments()):
        assert encode_segment(STORE.params, None, SALT, s, W[s * seg:(s + 1) * seg]) == blocks
