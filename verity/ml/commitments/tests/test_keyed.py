"""``verity.ml.commitments.keyed`` reproduces its vectors.  d over an A instance of four 1 KiB segments, a words instance of
four chunks over two calls, the next call's equal A, another salt_w, another ks_w and a one-leaf subtree.  d resumed from S_w
at word level is the same digest, at no compression more than SHA-512(M_top) from the IV.  Two mutations are negative tests:
a length word without the 128 key bytes, and the top node at level D.  M_top's first block holds the domain id, and a message
that isn't the root's is refused."""
from __future__ import annotations

import hashlib

import pytest

from verity.errors import ProtocolError
from verity.ml.commitments import keyed
from verity.ml.commitments.gates.sha512 import compress_ref, midstate, pad_tail, words_be
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, _message, _uint, merkle_depth, tree_levels


def _sha(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


def _stream(label: bytes, n: int) -> bytes:
    return b"".join(_sha(label + i.to_bytes(8, "big")) for i in range(-(-n // 64)))[:n]


KS, SALT = _sha(b"lead/keyed/vector/ks_w"), _sha(b"lead/keyed/vector/salt_w")
KS2, SALT2 = _sha(b"lead/keyed/vector/ks_w/2"), _sha(b"lead/keyed/vector/salt_w/2")
A = _stream(b"lead/keyed/vector/a", 4096)
SEGMENTS = [A[i:i + 1024] for i in range(0, len(A), 1024)]
WORDS = [c for w in (_stream(b"lead/keyed/vector/words/0", 655360), _stream(b"lead/keyed/vector/words/1", 4096))
         for c in (w[i:i + 262144] for i in range(0, len(w), 262144))]
A_SCHEMA, W_SCHEMA = "pouw/lead/a/v1", "pouw/lead/words/v1"

S_W = ("a643d39e0757b1f9c08d7feaf6451f256fc0e863c515b44262dcfebfe5113f4d7974c10007d58f184127c15a186d481cda8618febc0585e6acee9ce79"
       "18760b6")
# (ks_w, salt_w, position, schema, values, d, the unkeyed root)
VECTORS = {
    "A": (KS, SALT, 197, A_SCHEMA, SEGMENTS,
          "3bf65c529446a801fb137bc9eae0bae2c005c7d79fc48cd7b1a33b48e4dbf64a059f7100b300cb6f52d1d8ad39278e800c94f2d0c5b010af31d49817ec11f4c1",
          "0a1f61d80a5c7fc5e25ab2af0af0e5baba7d4e9533bf6059012211a8eb20913334898cb7e1e092231851c511ead95e5aabcf8fb2a92d9fc3057a6648b7389337"),
    "words": (KS, SALT, (1 << 63) + 3, W_SCHEMA, WORDS,
              "d6776c42b1346f3ace17242764e08fcb23621d8adc544d190e197266a069f51d933b859b783b990cba2a1d3ee4226a92f8298acb94ec3caa003e4dbaae93b9d3",
              "165f68e06ac1433e2471c34a85181e71991fc984d4e28f5232aad3f162396712ab6d25a8212bd017e8fee6ef888dc4e9b22bba68fa10a472a2872aad90d76211"),
    "A_next_call": (KS, SALT, 198, A_SCHEMA, SEGMENTS,
                    "a0fdd2b23876a727a9c5c47ccb85262157fa129750d6898d7560db2f955e7d78fee89498f08a55ef66529adf02c77697ee4745488439abc4eeefd92045def6f8",
                    "8b85460cd2d523ed3419dd8f9a27c3ed44a6379f9ac737f58fd8f90f17f222b5d8cad36d67d4c284a2674735a8c31a3f7566fb31dd5b7e259acac50a4583e860"),
    "A_other_salt": (KS, SALT2, 197, A_SCHEMA, SEGMENTS,
                     "c970a0667c863f673b0ebeef9634b58a93eb333ce75c0600064d6f72a272ae4ada6998f58923a9e0a58de16595fa40ecbca35bf6998d7f701c29c934f737bf50",
                     "5344ee02f6c418cde199559998fad3fead6c62bdcc31680fd917039b548e8e6c526c351530d682562ca0f8c6ac87e953e81ce6c61c51dd760b65939c96fec5a5"),
    "A_other_key": (KS2, SALT, 197, A_SCHEMA, SEGMENTS,
                    "621f580dfaed7f5c003d3d18951e71417b71afa923751ecca16d004e8d19f8b4a34794974f8f63663ff49ffa165a22e5dd5284976eb0100fd7878ece98a7b1f1",
                    "0a1f61d80a5c7fc5e25ab2af0af0e5baba7d4e9533bf6059012211a8eb20913334898cb7e1e092231851c511ead95e5aabcf8fb2a92d9fc3057a6648b7389337"),
    "one_leaf": (KS, SALT, 199, A_SCHEMA, SEGMENTS[:1],
                 "21ea90132419867d07995334c4021454dfcfe7ca594854e590049bca06089ed7618b6b94bd53c5c0cabdf425308480f464f1a22b8919e6e6a062514c74ecc12e",
                 "0579b187e835ebd3241903387a0b69043e46675f1c22fba6526a4f871faa38af21d0ab5997bde572820ab48c0df682576ba7cb33e714067b335c30615e609928"),
}
# d over the top node at level D, from the reference with that mutation
LEVEL_D = {"A": "6833d028c564c7acc17a50fde2de5da89b4fb2a73d7b6a08c9f90ba7ea6c1e901629dd20b703d27d2caff84e4a0260f0a7ee4326264ee8beb6c77f5db8459e99",
           "words": "7e263d1cd7b63219ebdfd97a5f4d7c6feb8d1d2bf9720b217893642b31afe577793c58cbee43d3168c99bea06d8169809478364f78fc12eef174a8368eaff7b1"}


def _instance(salt_w: bytes, position: int, schema: str, values: list[bytes]):
    dom = CommitmentDomain(_sha(b"pouw/lead/instance/v1" + salt_w + position.to_bytes(8, "big")), -1,
                           RangeIndexedDomain(0, len(values)), hash="sha512")
    return dom, tree_levels(dom, [dom.leaf(r, r, schema, v) for r, v in enumerate(values)])


def _resumed(state: list[int], message: bytes, length: int) -> tuple[bytes, int]:
    """SHA-512 from the chaining value ``state`` over ``message``, padded for a total of ``length`` bytes."""
    tail = message + pad_tail(length)
    cv = list(state)
    for i in range(0, len(tail), 128):
        cv = compress_ref(cv, words_be(tail[i:i + 128]))
    return b"".join(w.to_bytes(8, "big") for w in cv), len(tail) // 128


def _top(name: str) -> tuple[bytes, bytes, bytes, str]:
    ks_w, salt_w, p, schema, values, _, _ = VECTORS[name]
    dom, levels = _instance(salt_w, p, schema, values)
    return ks_w, salt_w, keyed.top_message(dom, schema, values, levels), levels[-1][0].hex()


@pytest.mark.parametrize("name", VECTORS)
def test_vectors(name):
    ks_w, salt_w, m, root = _top(name)
    assert root == VECTORS[name][6]
    assert keyed.digest(ks_w, salt_w, m).hex() == VECTORS[name][5] != root


@pytest.mark.parametrize("name", VECTORS)
def test_d_is_sha512_resumed_from_s_w_at_no_extra_compression(name):
    ks_w, salt_w, m, _ = _top(name)
    state = midstate(ks_w + salt_w)
    if (ks_w, salt_w) == (KS, SALT):
        assert "".join(f"{w:016x}" for w in state) == S_W
    d, after = _resumed(state, m, 128 + len(m))
    assert d == keyed.digest(ks_w, salt_w, m)
    assert after == (len(m) + 17 + 127) // 128 == {275: 3, 1185: 10}[len(m)]


@pytest.mark.parametrize("name", VECTORS)
def test_a_length_word_without_the_key_bytes_is_not_d(name):
    ks_w, salt_w, m, _ = _top(name)
    d, _ = _resumed(midstate(ks_w + salt_w), m, len(m))
    assert d != keyed.digest(ks_w, salt_w, m) and d.hex() != VECTORS[name][5]


@pytest.mark.parametrize("name", LEVEL_D)
def test_the_top_node_at_level_d_is_not_the_roots_message(name):
    ks_w, salt_w, p, schema, values, want, root = VECTORS[name]
    dom, levels = _instance(salt_w, p, schema, values)
    D = merkle_depth(len(values))
    wrong = _message(b"node", dom.domain_id, _uint(D), _uint(0), *levels[D - 1])
    assert _sha(wrong).hex() != root
    assert keyed.digest(ks_w, salt_w, wrong).hex() == LEVEL_D[name] != want


def test_one_leaf_top_is_its_leaf_message():
    _, salt_w, p, schema, values, _, _ = VECTORS["one_leaf"]
    dom, levels = _instance(salt_w, p, schema, values)
    assert keyed.top_message(dom, schema, values, levels) == dom.leaf_prefix(0, 0, schema, 1024) + values[0]


def test_a_windows_keyed_messages_lead_with_their_domain_ids():
    """Bytes 49..113 of M_top (after the frame, the tag and the id's length) are the domain id, so a window's instances, at
    distinct positions under one salt_w, have distinct first blocks."""
    firsts = set()
    for name in ("A", "words", "A_next_call", "one_leaf"):
        _, salt_w, p, schema, values, _, _ = VECTORS[name]
        dom, levels = _instance(salt_w, p, schema, values)
        m = keyed.top_message(dom, schema, values, levels)
        assert m[49:113] == dom.domain_id
        firsts.add(m[:128])
    assert len(firsts) == 4


def test_refusals():
    _, salt_w, p, schema, values, _, _ = VECTORS["A"]
    dom, levels = _instance(salt_w, p, schema, values)
    m = keyed.top_message(dom, schema, values, levels)
    for ks_w, s in ((KS[:63], SALT), (KS, SALT + b"\0"), (b"", b"")):
        with pytest.raises(ProtocolError):
            keyed.digest(ks_w, s, m)
    forged = (*levels[:-1], (_sha(b"another root"),))
    narrow = CommitmentDomain(dom.binding, -1, RangeIndexedDomain(0, len(values)), hash="sha256")
    one, one_levels = _instance(salt_w, p, schema, values[:1])
    for args in ((dom, schema, values, forged), (dom, schema, [], ()), (dom, schema, values, levels[:-1]),
                 (narrow, schema, values, levels), (one, W_SCHEMA, values[:1], one_levels)):
        with pytest.raises(ProtocolError):
            keyed.top_message(*args)


@pytest.mark.parametrize("name", VECTORS)
def test_the_top_from_its_bytes_is_top_message(name):
    """A hasher's top (the root's two children, or the one leaf's value) gives `top_message`'s M_top, and refuses a top of
    another size or a domain of another hash."""
    _, salt_w, p, schema, values, _, _ = VECTORS[name]
    dom, levels = _instance(salt_w, p, schema, values)
    D = merkle_depth(len(values))
    top = values[0] if D == 0 else b"".join(levels[D - 1])
    assert keyed.top_from(dom, schema, len(values), top) == keyed.top_message(dom, schema, values, levels)
    narrow = CommitmentDomain(dom.binding, -1, RangeIndexedDomain(0, len(values)), hash="sha256")
    bad = [(narrow, schema, len(values), top), (dom, schema, 0, top), (dom, schema, True, top)]
    if D:
        bad += [(dom, schema, len(values), top[:64]), (dom, schema, len(values), top + bytes(64))]
    for args in bad:
        with pytest.raises(ProtocolError):
            keyed.top_from(*args)
