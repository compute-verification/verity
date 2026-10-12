"""The code reproduces every pinned vector (`vectors.json`, written by `remote_bandwidth_vectors.py`), and the codec is
kv/v0's at p512."""

from __future__ import annotations

import json

import remote_bandwidth_vectors as rv

from verity.core.protocols.remote_bandwidth import codec

# `benchmarks/word_codec/tests/kv_vectors.json` ("p512/k2", cursor/ra-word-codec-10bf at 56818bb77): the kv/v0 codec
# the KV kernel (`kv.cu`) implements, under its vector seed SHA256("verity/word-codec/kv/vectors").
KV_SEED = "9a2a3bccf24b46d0788b084ea6ab8a259f06781facbe053a212734936d48be4a"
KV_P512_K2 = {
    "c": ["2c61dd69fb21c4b458cc8178648a8ad285673e5864a05a9e19afaaaf3ca8fcca06848e55f6076466016eacfc49ad08e5b017041fa41cfd02b02659a6db39b7d0",
          "d3d85c02ec9acf259e2ec29739d6d230b820fea028d78ee4e167e28b348fa360028b22e7a9e81e88ddb1312c21e3c95d7edff57ff7683a4e25bae677814f3e7f"],
    "tau": "a2e9e84a2c2e860177109ca66cfda353f9bfabb9ab8ec71d4d1c32b710e8488e68be390cde83efec99849319e5dec84f5f6d48e1df63230c1b8554eceb6034e1",
    "words": [
        (0, "c375c470a796e21ce35a5b006c8b22f09e85da03b42db8e595f75e1b19efca463e799703779f9f53e522f7bec10b625e384d9d9efe526a9ec50bf1f5e1081d1",
         "557c9915c722e47147bbb1d8f07f23cea3e7024b5bd17251103bcf1cc529dead643bb5afbe19e1a12097d4043b71e6b8b5f4b828518d25f3f24db991d1cd428a"),
        (1, "cf765b84904f0fbc1c43353a2e36aeba869a95d830e5e673603d80ecda9a0412e04926756442a8eb96681f9466cc19a7e1bd822d81176ea33fd97d3c375af9dc",
         "c2f5ef0e23826e655de508a5e44f1becfdf308af21107280798f405541aabd5c97c8d0d545ac4934239df50a618d17b2de6364e4d4b6b4bb08e3dfa34dfe8b5"),
    ],
}


def test_vectors_are_reproduced() -> None:
    assert rv.generate() == json.loads(rv.PATH.read_text())


def test_vectors_cover_zero_weights_a_short_last_leaf_a_partial_plan_and_a_chain() -> None:
    pinned = json.loads(rv.PATH.read_text())
    assert pinned["codec"]["words"][0]["payload"] == "00" * 64
    assert rv.BLOCKS % 16 == 5 and rv.PLANS[1].length % 16 == 10 and list(rv.PLANS[1])[:2] == [16, 17]
    chain = pinned["chain"]["passes"]
    assert chain[0]["previous"] == pinned["registration"]["digest"]
    assert all(b["previous"] == a["root"] for a, b in zip(chain, chain[1:]))
    assert chain[0]["plan_digest"] == chain[2]["plan_digest"] != chain[1]["plan_digest"]


def test_codec_is_kv_v0_at_p512() -> None:
    k = codec.key(bytes.fromhex(KV_SEED), 2)
    assert [format(c, "x") for c in k.c] == KV_P512_K2["c"] and format(k.tau, "x") == KV_P512_K2["tau"]
    for i, m, y in KV_P512_K2["words"]:
        assert codec.encode_word(int(m, 16), i, k) == int(y, 16)
        assert codec.decode_word(int(y, 16), i, k) == int(m, 16)
