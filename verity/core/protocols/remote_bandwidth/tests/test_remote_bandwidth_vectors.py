"""The code reproduces every pinned vector (`vectors.json`, written by `remote_bandwidth_vectors.py`), and the codec is
kv/v0's: masked at p1024, k = 2 and 3, and at p512."""

from __future__ import annotations

import itertools
import json

import pytest
import remote_bandwidth_vectors as rv

from verity.core.protocols.remote_bandwidth import audit, codec, coins

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

# kv.py at p1024 masked (cursor/ra-codec-mix-ea59 at 088223599, #1894), `key("p1024", SEED, k, mask=True)` under the same
# seed: c_0, c_1 and τ are its kv_vectors.json's "p1024/k2" (the mask changes neither), K is the masked key's, and each
# word is `encode_word(m, i, key)` at k = 2 and at k = 3, m from random.Random(f"p1024/k{k}/masked") at i = 0, 1, 2^33.
KV_P1024_MASKED = {
    "c": ["13b5cc00227db2db761c3937199c9cc967ccad726954daa0b37212ba59510301f885ac0074e7c469d3b47e6ad6801cba70327316194d1019f9310c9e261f006c"
          "6dd2fb9b1e3c11a1f3addfcb4ad93e35de97e3af3cb7a2fe1b746ada8d2bd4ed233f97790d2ec4eb5242a0dc4575b8460cbcf4d92054eb9293889bb2f7bab986",
          "ad4b74fa2887bc01dd79fc9deba87c164194aedbebf0d52ae8b1c5ac76c5b48d708f9d093fe4babc45658308eae2dd6b5569b2b5ce7b6cdcfa7ef6a5a5e3469e"
          "52fe5815bbfc233d08f1db524c0eb41ed7dd9331b4a8223da2016f6ba22f2f8e05ce52294282ea4719754fd676c71a58706cc6fec2e829ac337df2c809e0d377",
          "10cb40656b09f86f5d57dbd547d6fa56555b05b1fbe2e8c506c3e5b011517bcf610bec025c2a6eec11a6e022f33f453858ed4e3162a6ffbb9cfdb74e1403211b"
          "5ef2aaea3bb09efe6b24939ef0f0f3c2b1ba116beb928517da253807a744ee6b7f8261807f90875252549602b0523763ce8f0a76e3b0e7983fba98936077a27e"],
    "tau": "f9822c4a3bd5977f940b17d79fee2cffbbc3f5bf184617f3b298478c9879a5d81c55f11dc837e7d24399b8b9a1051c32e529b6e72a16a7fe237e6efad9bc9a56"
           "02aa789d3e398fb98705b8348c32057cd8aee605143f8ade37c51aa3bddbe718ed8229aa3a6a5e8ba909951de40494bbdf270629c116dd6edaf08a86c97be0da",
    "mask": "266f3755d7e950e3ce57616c22a79398fa584d593645eb51252a73d2825417bca53c10a1d92ed3112e79d2e2872dd6336cdafc512adf4bed109e05b710222b92"
            "27ada256d3d6b3e858e07d9a9ad1db96b767f232bbd13c828f444cb861afb8f9cee512ce5770b3adf9a8a90b20e935cff22720c43ac2267a38b79a4d8f2942c3",
    "words": {
        2: [
            (0, "c895add5957e674104a58100291f858dbb8a1ef4b904be0f82cdfef7eeb273620e330434dd5b693368a2814ae84d80538d66db1fd14032725c7a6e750465d693"
                "598e6a9a805cd9acdcb6c282d451c52d3580f2f144eddd4ae405b554a61c1b8cb85793254391cbb070db2af5a88cada892ffde9dd06cd1d69873b68f0f8e51",
                "2505e657c7655790e3f74d79f47095bc10554dbf2543315b47f6ce3030b9b84844442bf1e2e8d1b364583b6ca1dcf60a9c2bc5b0d59200377ffe5201cdc8eda0"
                "380a4547370c75ca97c92b0b86a46130a277a19162b3f166f0af7414f58d3aee428321e1fc100065187639f478d6bfa8489fbf5347cc055d5f0de2497d380700"),
            (1, "1bd42c04bd2c81cfe268d4137b263ebdf9c5d0d7287b856ddf1d8baf70b2c932b95ec92f002629460f661d708e77760bf79cbcc7c21a8207cbd1090cb2671398"
                "a5bc4ee23c5f008cfa3d54c96b38fb3ac92fa52ad535b3fc98d87a1cbd3eb3ec3ef78d55b4624f0d4c0a92499c3e08f193ec348743ea5a152692a36da4352f20",
                "cd28c364ed8ffdf5bfa676f7181ea313fc99858103819123295d22b51ad7140f9532d761f597f016ebbd015521dcfdbe88423ed8c00316721a42248a9856e10d"
                "9ab14b035627566b1f2740babba9cebe7ca9da4bc3196a829585a66289359da9bb74577082343417b25af7880e45e35cd2c5f3834b3b74c93f633625f7f46954"),
            (8589934592, "924b9cfc766a165f54de2a75f057beae1c7f207a7e675c3ad871623b26db9bc2ec05c12b4fec498a2cdffa7cc32a175a7694dcf7bf1c97f6742d9cd27ea4cd6d"
                         "9d467769eccdeb3ee6394ad0b69bc75b864847c48a4c19d1d9c271100c117ca66f853dbe98a2aab30aea8e67e2d84c71a172731e6d60e58fdb08a6d63abbb744",
                         "c7429006ec052a50a8ea08b3f1b5ccb32c9397c61e7ad6fb370a4acaa974fd9b4b2964d25e91dc6e1a7443be9108b136531825b4de6a371e1763666e4cf367d9"
                         "b590f0edd339c6cba418d5de4724701f1c82502700d10d3531d68944980264dd09239455fecb45cdfb738c1e49616f68ad9ae33966c97aac79957316fc731d8c"),
        ],
        3: [
            (0, "ea3f3002bf464b963f30e04e0cf358bd869f42c25764fa954b7a14a3f024080874cc32a863206914937202cc53be9dc34b8ae0d1abf3438556de2b8bbaad4799"
                "34bbe562f813d985be5fbedfbcb250dfea8487682b6abc886b14ffb8adc78fdf066b4709a73f38ef45327c645b1d07409124368f9f8318ccc5be44558e88eea4",
                "3ccc0306e1b8abcef1cdf424118aac32c528efbf75a11f1a14228a21b539e03fda94d29996393c9c1a4903204edd7a2ee5853de2bbb5e41627c1a9b9d1f00389"
                "e2395e05392a2d5136aa56971446b7040da2a3ba9832d81144c0e7d04c8f49fb6283f7ab6fdd207e8f1606f965dcb89af6bff57eab587acbb12f8028cea77be4"),
            (1, "a1a7b8d7eac91b4144f0cabec79f88cb31a66769852e49cfe74e052ce1c5b25957f17a972f18266786703bb003b95b15fad651122f11434d1e8132959cfc828f"
                "bcff096462fd33b7384457b5878f3bb6781423ffa9d02935b6f2233768ee5f0f3839cc7d8ee4f36b9fb934cbe03696f2cd0d777bf76102a00bfecf1d2e7fbbc2",
                "268815a411c4e113b8d6ce3ca2aacc5ad503c34890615c2bd0ca3cc1e3428065b565c871fde70a25e87ac6cc9ea05a34b9d0f820829018d588b40cc8345ef71a"
                "2c97e6e27f213d4c7e599d53f48c25957a59b89238d76c2b32990861e8021e0985f53146d4d1a4702ebe80620adde29e8f26e6cd023ff232b621cf73c87ce74f"),
            (8589934592, "e8bcd0e0ccf86027b88b9cccaf77ee5151f50a46fc9f39fb934fcc5411380295beedc807d8151ed9eb8fcc5c0c84af03107d720fbc40479585f591bad9bf1369"
                         "f9b56fbc8ce493f009b88d5ac1ad98c59ef1aea9f9e0dff1e616eb0db8c8ab9e3c9954eb98f37efafc7a4efd4cf21cdd6dd5886de0a627d7cdd755b181961ede",
                         "1b3908384f6c1b3b552f9a9a029a59936c6f4c09169f185a7b8248a634106e427297f6aaff5f5635fc229f11c358b890bb07030b7e5b1b4191046cf7f1e4cbeb"
                         "ca88c9ee1da0069b1b9e2be8c7fdaab92055f97929a393ec570c8c79b15d399cadf4fa946a1a9bb83c279f32a6effe26b327c1e0c2ae783d9310a7771423f052"),
        ],
    },
}


@pytest.mark.slow
def test_vectors_are_reproduced() -> None:
    """About 6 s: the store's 21 blocks encoded under the default codec (`rv.store`, once a process)."""
    assert rv.generate() == json.loads(rv.PATH.read_text())


def test_the_registration_and_the_codecs_words_reproduce_the_pinned_ones() -> None:
    """The quick tier's share of the vectors: the payload root, the registration and the pinned words, encoded one by one
    rather than with the store."""
    pinned = json.loads(rv.PATH.read_text())
    payloads, salts = [rv.payload(j) for j in range(rv.BLOCKS)], rv.payload_salts()
    root = audit.payload_tree(rv.NODE, payloads, salts).commitment.root
    reg = audit.register(rv.SOURCE, rv.NODE, rv.BLOCKS, root, nonce=rv.NONCE)
    assert pinned["payload"]["root"] == root.hex() and pinned["registration"]["digest"] == reg.digest.hex()
    assert pinned["registration"]["codec"] == pinned["codec"]["name"] == reg.codec.name == "p1024/k3/masked"
    k, f = reg.key, reg.codec.field
    assert pinned["codec"]["c"] == [format(c, "x") for c in k.c] and pinned["codec"]["tau"] == format(k.tau, "x")
    assert pinned["codec"]["mask"] == format(k.mask, "x") and pinned["codec"]["seed"] == reg.codec_seed.hex()
    for v in pinned["codec"]["words"]:
        j, w, s = v["j"], v["w"], f.word_bytes
        assert payloads[j][s * w:s * (w + 1)].hex() == v["payload"]
        m = int.from_bytes(payloads[j][s * w:s * (w + 1)], "little")
        assert codec.encode_word(m, f.block_words * j + w, k).to_bytes(s, "little").hex() == v["word"]


def test_vectors_cover_zero_weights_a_short_last_leaf_a_partial_plan_and_the_chain() -> None:
    pinned = json.loads(rv.PATH.read_text())
    assert pinned["codec"]["words"][0]["payload"] == "00" * codec.DEFAULT.field.word_bytes
    assert rv.BLOCKS % 16 == 5 and rv.PLANS[1].length % 16 == 10 and list(rv.PLANS[1])[:2] == [16, 17]
    passes = pinned["passes"]["passes"]
    assert pinned["passes"]["challenge"] == pinned["coins"]["coin_0"]["coin"] != pinned["openings"]["coin"]
    assert passes[0]["previous"] == pinned["registration"]["digest"]
    assert all(b["previous"] == a["root"] for a, b in itertools.pairwise(passes))
    assert len({p["seed"] for p in passes}) == 3
    assert passes[0]["plan_digest"] == passes[2]["plan_digest"] != passes[1]["plan_digest"]
    assert pinned["openings"]["index"] == rv.PARAMS.windows and pinned["payload"]["schema"] == "hm96-sha512/row/v2"


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


@pytest.mark.slow
def test_the_provers_passes_and_the_sessions_draws_reproduce_the_pinned_ones() -> None:
    """The generator computes receipts and draws on its own; the audit's `Prover` and `Session` must agree with it, at the
    default codec.  Slow for the same store as `test_vectors_are_reproduced`, which encodes it once for both."""
    store = rv.store()
    reg = store.registration
    challenger = coins.Challenger(reg.digest, 0, rv.PARAMS.coins, entropy=rv.Entropy(rv.COIN_LABEL))
    clock = Clock()
    s = audit.Session(rv.PARAMS, reg, audit.NoiseTiles(96), challenger, clock=clock)
    prover = audit.Prover(store, audit.NoiseTiles(96), s.commitment, entropy=rv.Entropy(rv.LEAF_SALT_LABEL))
    pinned = json.loads(rv.PATH.read_text())
    challenge, previous = s.start(), reg.digest
    for i, (plan, want) in enumerate(zip(rv.PLANS, pinned["passes"]["passes"], strict=True)):
        ps = prover.run(challenge, i, previous, plan)
        assert ps.root.hex() == want["root"] and ps.plan.runs == tuple(map(tuple, want["plan"]))
        assert s.receive(0, i, ps.root, ps.plan.runs) is None
        previous = ps.root
    assert s.boundary() is None
    clock.now = s.deadline(0) + 1
    assert list(s.close(0)) == [unit(u) for u in pinned["openings"]["draws"]]


def unit(u: int) -> tuple[int, int]:
    """Draw u of the window's units as (pass, position)."""
    for k, plan in enumerate(rv.PLANS):
        if u < plan.length:
            return k, u
        u -= plan.length
    raise IndexError(u)


def test_codec_is_kv_v0_at_p512() -> None:
    k = codec.key(bytes.fromhex(KV_SEED), codec.Codec(codec.P512, 2, False))
    assert [format(c, "x") for c in k.c] == KV_P512_K2["c"] and format(k.tau, "x") == KV_P512_K2["tau"] and k.mask is None
    for i, m, y in KV_P512_K2["words"]:
        assert codec.encode_word(int(m, 16), i, k) == int(y, 16)
        assert codec.decode_word(int(y, 16), i, k) == int(m, 16)


@pytest.mark.parametrize("c", [codec.FALLBACK, codec.DEFAULT], ids=lambda c: c.name)
def test_codec_is_kv_v0_masked_at_p1024(c) -> None:
    assert codec.P1024.p == (1 << 1024) - 105 and c.field == codec.P1024 and c.masked
    k, pin = codec.key(bytes.fromhex(KV_SEED), c), KV_P1024_MASKED
    assert [format(v, "x") for v in k.c] == pin["c"][:c.rounds] and format(k.tau, "x") == pin["tau"]
    assert format(k.mask, "x") == pin["mask"]
    for i, m, y in pin["words"][c.rounds]:
        assert codec.encode_word(int(m, 16), i, k) == int(y, 16)
        assert codec.decode_word(int(y, 16), i, k) == int(m, 16)
