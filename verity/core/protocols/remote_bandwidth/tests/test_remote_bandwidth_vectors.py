"""The code reproduces every pinned vector (`vectors.json`, written by `remote_bandwidth_vectors.py`), the codec is
kv/v0's (masked at p1024, k = 2 and 3, and at p512), and sketch-sha512 is the tensor-core codec's reference's."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

import pytest
import remote_bandwidth_vectors as rv
from remote_bandwidth_fakes import Clock

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts

# `benchmarks/codec_tensor/sketch_receipt.py` (at 5b8decfc5), numpy, run on these
# inputs: seed k is SHA512(SKETCH_PINS ‖ u8 k), "counter" n bytes of SHA512(SKETCH_PINS ‖ label ‖ u32be c) for
# c = 0, 1, …; each case (seed, block, z, j, σ_j, y) with σ_j its `sketch` and y its `SketchReceipter`'s, a receipt
# that binds z; each unbound case (seed, block, j, σ_j, y), of the block alone, with σ_j its `sketches` of the block
# and y its `receipt_of`.
SKETCH_PINS = b"verity/remote-bandwidth/sketch-pins/v0\0"
SKETCH_UNBOUND_CASES = [
    (0, "0x80", 0,
     "0028000000d9ffff00d9ffff00d3ffff000a00000000000000f5ffff00230000001700000003000000ffffff00e5ffff00f0ffff0006000000230000000e0000",
     "f596fd5b1e1e0c96eef5f72c7b3493de91630a3d42a7f425a33b9ffa93195f06b119fa811cbef77bf75131e8d91dca2fe454d25455e0f3278496605ca8f66157"),
    (0, "0x7f", 1,
     "50d8ffffb2260000b2260000a62c000014f6ffff00000000ea0a000046ddffff2ee9ffff06fdfffffe000000ca1a0000e00f00000cfaffff46ddffff1cf2ffff",
     "9e87b84c847bd01b02de09afd66c5555b2ba4669df567e4d8ece2281dc565288c35fbf2b084ce8006a3d6f1780214d1d0e3d5ab4a3cacc4f0875d3125dbc9885"),
    (0, "counter", 20,
     "3901000005090000450e0000dffcffff33ebfffff7100000290a00004bf4ffffa5fcffff5ff9ffff690b0000f10d00002d040000391400000f0b00007ff4ffff",
     "bd5414ec1f5250e9583c1ad977da6812607a90edad7f0342d7ef14f813f1cc44873380376f0ef331d220597f9ae7f2369f22483cc335737c4a083e708905c5c2"),
    (1, "counter", (1 << 64) - 1,
     "7fffffff7df8ffff0d04000029eafffff7210000fdeaffff5700000099ffffff73f3ffff1913000061fcffffd5190000bf07000061f0ffffc518000053f3ffff",
     "a18c9d926b1d13b42737a8550d5f638e432541486e8a7678e3856d2a48de24b6b78af95837775f96e23ec4cf35ad714d2077dce9b8b9a8220a0575baa70c26c1"),
]
SKETCH_SIGN_BITS_SHA512 = [
    "02c8dbb7ab436a92e8cf7006a0907d635b75cadec8d7959d15c7cebd007a5bdbdf645569e2540abd3906f2c3b566674a6a3580b79177bd9b7d892d8ba3ef3108",
    "4a6c53f126bd07a216ca31b753133d080c85ad79a6ca693ad82e3bcf1e2e36169110a36a4a8365075678a8d5eee65a19da5ed96c0e87b53478cfff3ad8c2a792",
]
SKETCH_CASES = [
    (0, "0x80", "one", 0,
     "b02700004ed9ffff4ed9ffff5ad3ffffec0900000000000016f5ffffba220000d2160000fa02000002ffffff36e5ffff20f0fffff4050000ba220000e40d0000",
     "81417a3dfc2da9e7b9ed8d5a5379c80ad5f7a5dbdd92b5b88a7f43bd3aaf5b222c9364513ed69ac7b2f846b48b1679d85d5697df7f6d8d169074a90279cb2dfa"),
    (0, "0x7f", "short", 1,
     "c4e3ffffe610000018140000281a0000c8f9ffffc00a0000500f000006f2ffff0af7ffff0608000010000000781f0000f4090000eeffffff16eeffff10ffffff",
     "42d5e0d3d4651466e7918aeb312f1d9bee48f26d06a9673008331b786b309ef42e58c140bf47e691389d62204ba831f9e99edeb6d14a08bc7614caecb8641a56"),
    (0, "counter", "folded", 20,
     "4afeffffc80b000060f0ffffac080000b2f7ffff6e07000042fbffff54100000f00500004e00000038210000083500000609000032daffff8cf5ffffa20b0000",
     "a60cb97c4444fea57afde1f4466f6fbb59286edf99a560e5a0407273d3644344b46e8c036e71068d31a73cddb8857fc5a51d444a0bcce49fe6cb7a5b8bb98e13"),
    (1, "counter", "short", (1 << 64) - 1,
     "3f100000a9fbffff47ffffff3bf1ffff6b280000db040000110d0000e10900000bfefffff71100008dfcffff1b1e00005b0500002fecffffbb040000eb040000",
     "6dd31d65ca1d94c620420c240e3834c3d09159020309c455514f6384f5bc0d566c31936d25ef206d6481b907ed3cff83ac3e4e245c2d7c9e9ca7004e99b3de6d"),
]

# The same reference at 64 rows (at f666d887b, `SketchReceipter(seed, 64)`, `signs(seed,
# 64)`), on the same inputs: the sign bits' SHA-512 per seed, and each case with SHA-512 of σ_j's 256 bytes in place of
# σ_j.
SKETCH64_SIGN_BITS_SHA512 = [
    "2ffa7ce504e16a4a8d9e0d19abb2318ce0d7e4f6245c0dd64d15e8bff5640a035da1da231aa63a56ae45d67291d831543c4fde5b020fbde8494451b5eaf023b1",
    "e53861f81d5515c5e6fd423e66ef64c6d84e3c770acdcb98fecf7345e30c2da6cde8efeff25c2235e61fc701e7d8a537906499326f3cd8dfb630a4271c6db80a",
]
SKETCH64_CASES = [
    (0, "0x80", "one", 0,
     "12543d8b982b5dd2a3e001e128f77d320a96908cd1d5c35bdc00549106147f720d65153356833a8ee0c8aa3d789449d32ae9651d18acaa83c2498bf3a43a8b2d",
     "2aff6a29daa438d20b11ead8d629a639d31de55435ad18dcbad8571100cc4f671625fb3d262b9e4c9520e6682b7067f0ab854293810d4010c86d97a65beea9da"),
    (0, "0x7f", "short", 1,
     "29c7ca9063e3c31ae69ee4af19cb105f7f35c15843eb9aa2850c3963a6e596b7429a27eb056326344688bd69b39a032097ee20805a9356bf6cb2963ffa7c1c74",
     "44c6291a83cb95aa8975dbdfa2490eae7e92ae948c6801b735512ec50590aabdb11931a2bf2745ee169cac6fe1b7778ce4eef7f6cc89565e99864d9974c5efa4"),
    (0, "counter", "folded", 20,
     "1130863d7c53c5a7290ec883daadfe074e43208e1040741391ecbc1aea9a99ecba254160d9e1f5adc4553086ba3575fdbd79cd28d67eb54015e8597a72c4758b",
     "d8cbd5b0bc3f2f0672e5bfeac4580af1bf0d9b280e37228ff594c429a831019764824d6c4feaf06ca26d359f4c2ca5b8e8ba62d471a3554c3b7a678a0016c668"),
    (1, "counter", "short", (1 << 64) - 1,
     "fb7ffb9430b69637cdab800965f3ab07e27f0e73be1243c3c57ea4cc6f9966ba939d84ca923d90c2720fa904433a4033edcb4b446eddb26ea08034b280de6d0a",
     "af5c0c67da2fd9c06b0ebf26827408faf0d9a904e6271a4a1997f0d725b6b6187ca4f6bcf75b6c79d37e949df6257c535868ea7320be7d0265733b27c0d7c903"),
]
SKETCH64_UNBOUND_CASES = [
    (0, "0x80", 0,
     "f6ef757affdfff0e2fc03f3871458ed7b8a46dc80c3b1368b9d0e6be0e4aadf16697e9109e2e0670b5785fb37b48bedb550116619005adf251d1638695176fc1",
     "0eb1e5ba0305b80b1f8029bf0027144bafb71bdfab2a0146a09c5cbdfb8ae8df0ddf93b78f78802e8ef1f6205a09936e9fda1bed694cecd9b488e960eb8abda8"),
    (0, "0x7f", 1,
     "da8204d492a263c2d87f973f06a01cb49adb25341ce91eb3abcbcce932bb029081fc30b3c8266a77b11694393ab572a11fa096c398dbbf61abf1e07aa50d0064",
     "6b6b08fbf5dea1439d82e03aa25f900ea2124a00da078adcf45dcaea62430cc02058cb689b2cc69e5f8f3e021e5bacf0cac91ca77fd57f258e07e1174f86ba98"),
    (0, "counter", 20,
     "3fa5828e6ea5cee3ce548d2e3100a99615ed19cc5fc0d411c818eaac08530647a9660d802606ddbbed24a134d2a3766935e663487d90b00764a8c357550146f4",
     "da3866b87520a10c6e183a69de5c04d3051b749d6c7a6e4173c411253e58d159f4b8ca1d08dee4bb6b5efcb586a325e39f0b66a7d2127b750e40b74261f462cf"),
    (1, "counter", (1 << 64) - 1,
     "fb24d011c28947cf9d50722c439582cdccdec8fb713d562742637266e7dd2ae740619d35c56aa02dfb4de0b94e637a364221fed9d6d1e754571abd1f382ee585",
     "a391d7324adcc28de3d7d5b780c433a2cdceb8755b795dd7d818401aa814074279c1e00b21e08307d961a433026089aadcb2f0e49f0ff966903906ad4cef9b95"),
]

# `benchmarks/word_codec/tests/kv_vectors.json` ("p512/k2", at 56818bb77): the kv/v0 codec
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

# kv.py at p1024 masked (at 088223599), `key("p1024", SEED, k, mask=True)` under the same
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

# kv.py's bound key (kv_vectors.json's "bound", its "p1024/k3/masked"), under the same seed: the token-layer of slot 3's
# layer 17, block 3 · 80 + 17, is SHA512(BOUND_LABEL ‖ u32be c) for c = 0 … 63, d its digest, b its offset and the
# words' SHA-512 that of their 32 little-endian words.
KV_BOUND_LABEL = b"verity/word-codec/kv/bound-vectors"
KV_BOUND = {
    "block": 257,
    "digest": "f5597dc95882e28736b9d6ea22e7c6f8de38e8299e1072048d913a5f740506544110c9a5a419f7058e9b8a8f6cc1df10c2c4d6e5159b7131d6f6429cbd8719a9",
    "b": "92e5925319a3cfb7146e7cb4826ac25e88c9c85dce2fa938bde6ca4ac8480b79d2337cd0d18252ab6101c7c2c98008a3e836d6f87121fe9b548cc493a7cbdf05",
    "words_sha512": "19fd802486f9d44335f474e428159469768fb12670b776684c686ca27840d43f412d18652032671e7db2621411301b1b8788b1ece4ba45c964e3c1309e6c4b99",
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
    assert pinned["registration"]["receipt"] == reg.receipt == receipts.DEFAULT == "sketch-sha512"
    assert pinned["registration"]["bind"] is reg.bind is False
    assert pinned["registration"]["layout"] == reg.layout == receipts.LAYOUT == "store-order/v0"
    assert pinned["registration"]["rows"] == reg.rows == receipts.SKETCH_ROWS == 16
    assert set(pinned["variants"]) == {rv.name(*v) for v in rv.VARIANTS} and len(rv.VARIANTS) == 4
    assert "sketch-sha512/rows64" in pinned["variants"] and pinned["sketch"]["rows_allowed"] == [16, 64]
    for kind, bind, rows in rv.VARIANTS:
        v = audit.register(rv.SOURCE, rv.NODE, rv.BLOCKS, root, reg.codec, kind, bind=bind, rows=rows, nonce=rv.NONCE)
        assert pinned["variants"][rv.name(kind, bind, rows)]["registration"] == {
            "receipt": kind, "bind": bind, "layout": receipts.LAYOUT, "rows": rows, "codec_seed": reg.codec_seed.hex(),
            "digest": v.digest.hex()}
    k, f = reg.key, reg.codec.field
    assert pinned["codec"]["c"] == [format(c, "x") for c in k.c] and pinned["codec"]["tau"] == format(k.tau, "x")
    assert pinned["codec"]["mask"] == format(k.mask, "x") and pinned["codec"]["seed"] == reg.codec_seed.hex()
    for v in pinned["codec"]["words"]:
        j, w, s = v["j"], v["w"], f.word_bytes
        assert payloads[j][s * w:s * (w + 1)].hex() == v["payload"]
        m = int.from_bytes(payloads[j][s * w:s * (w + 1)], "little")
        assert codec.encode_word(m, f.block_words * j + w, k).to_bytes(s, "little").hex() == v["word"]


def test_the_bound_blocks_reproduce_the_pinned_ones() -> None:
    """Blocks 0 and 20 under their bound keys: the digest is the payload leaf's x_j, and the block decodes with it and
    differs from the registered (seed's key) block in every word."""
    pinned = json.loads(rv.PATH.read_text())
    k = codec.key(bytes.fromhex(pinned["codec"]["seed"]), codec.DEFAULT)
    s = k.codec.field.word_bytes
    assert [v["j"] for v in pinned["codec"]["bound"]] == list(rv.BOUND_BLOCKS)
    for v in pinned["codec"]["bound"]:
        j, payload = v["j"], rv.payload(v["j"])
        d = codec.digest(payload)
        assert j or pinned["payload"]["leaf_0"]["x"] == d.hex()
        assert d.hex() == v["digest"] and format(codec.binding(d, j, k), "x") == v["b"]
        block, seeds = codec.encode(payload, j, k, bound=True), codec.encode(payload, j, k)
        assert block[:s].hex() == v["word_0"] and hashlib.sha512(block).hexdigest() == v["block_sha512"]
        assert codec.decode(block, j, k, d) == payload and codec.decode(seeds, j, k, d) is None
        assert all(block[s * w:s * (w + 1)] != seeds[s * w:s * (w + 1)] for w in range(k.codec.field.block_words))


def test_the_coin_form_blocks_reproduce_the_pinned_ones() -> None:
    """Blocks 0 and 20 under the coin form of their bound keys: the digest of the form without a coin, another offset,
    and the words of the word path at that offset, which is all of the key a device that takes b_j as its input reads
    (`kv.cu`, `paged.cu`).  Each decodes under its digest and its coin only: not without the coin, not under another
    coin, and the form without a coin's encoding not under the coin."""
    pinned = json.loads(rv.PATH.read_text())
    k = codec.key(bytes.fromhex(pinned["codec"]["seed"]), codec.DEFAULT)
    f, coin = k.codec.field, bytes.fromhex(pinned["codec"]["coin"]["c_a"])
    s = f.word_bytes
    assert coin == rv.APPEND_COIN and pinned["codec"]["coin"]["index"] == codec.COIN_INDEX == (1 << 32) - 4
    for v, pre in zip(pinned["codec"]["coin"]["bound"], pinned["codec"]["bound"], strict=True):
        j, payload = v["j"], rv.payload(v["j"])
        d = codec.digest(payload)
        b = codec.binding(d, j, k, coin)
        assert j == pre["j"] and v["digest"] == pre["digest"] == d.hex() and format(b, "x") == v["b"] != pre["b"]
        block = codec.encode(payload, j, k, bound=True, coin=coin)
        words = [codec.encode_word(int.from_bytes(payload[s * w:s * (w + 1)], "little"), f.block_words * j + w, k, b)
                 for w in range(f.block_words)]
        assert block == b"".join(x.to_bytes(s, "little") for x in words)
        assert block[:s].hex() == v["word_0"] and hashlib.sha512(block).hexdigest() == v["block_sha512"]
        assert codec.decode(block, j, k, d, coin) == payload
        assert codec.decode(block, j, k, d) is None and codec.decode(block, j, k, d, bytes(codec.COIN_BYTES)) is None
        assert codec.decode(codec.encode(payload, j, k, bound=True), j, k, d, coin) is None


def test_vectors_cover_zero_weights_a_short_last_leaf_a_partial_plan_and_the_chain() -> None:
    pinned = json.loads(rv.PATH.read_text())
    assert pinned["codec"]["words"][0]["payload"] == "00" * codec.DEFAULT.field.word_bytes
    assert rv.BLOCKS % 16 == 5 and rv.PLANS[1].length % 16 == 10 and list(rv.PLANS[1])[:2] == [16, 17]
    passes = pinned["passes"]["passes"]
    coin_0 = pinned["coins"]["coin_0"]
    assert pinned["passes"]["challenge"] == coin_0["coin"][:2 * coins.COIN_BYTES] != pinned["openings"]["coin"]
    assert coin_0["after"] == pinned["coins"]["window"] and pinned["openings"]["after"] == passes[-1]["root"]
    assert passes[0]["previous"] == pinned["registration"]["digest"]
    assert all(b["previous"] == a["root"] for a, b in itertools.pairwise(passes))
    assert len({p["seed"] for p in passes}) == 3
    assert passes[0]["plan_digest"] == passes[2]["plan_digest"] != passes[1]["plan_digest"]
    assert pinned["openings"]["position"] == rv.PARAMS.windows and pinned["payload"]["schema"] == "hm96-sha512/row/v2"
    chains = {rv.DEFAULT: pinned["passes"]} | {v: pinned["variants"][rv.name(*v)]["passes"] for v in rv.VARIANTS}
    digests = {pinned["registration"]["digest"]} | {v["registration"]["digest"] for v in pinned["variants"].values()}
    windows = {pinned["coins"]["window"]} | {v["window"] for v in pinned["variants"].values()}
    coins_roots = {v["coins_root"] for v in pinned["variants"].values()} | {pinned["coins"]["root"]}
    assert len(digests) == len(windows) == 5 and len(coins_roots) == 1
    assert windows == {coins.window_root(bytes.fromhex(d), 0).hex() for d in digests}
    pc = pinned["coins"]
    c = coins.LiveCommitment(pc["scheme"], pc["epoch"], tuple(map(tuple, pc["families"])), bytes.fromhex(pc["root"]))
    o = coins.LiveOpening(0, coin_0["label"], coin_0["index"], *(bytes.fromhex(coin_0[k]) for k in ("after", "coin", "salt")),
                          tuple(map(bytes.fromhex, coin_0["path"])))
    assert coins.layout_reason(c, 0, rv.PARAMS.windows) is None and coins.coin_reason(c, o, coins.CHALLENGE, 0) is None
    for (kind, bind, rows), c in chains.items():
        assert c["passes"][0]["previous"] in digests and all(b["previous"] == a["root"]
                                                             for a, b in itertools.pairwise(c["passes"]))
        assert [p["plan_digest"] for p in c["passes"]] == [p["plan_digest"] for p in passes]
        assert c["challenge"] == pinned["passes"]["challenge"]              # the same coins, another window R
        assert ("z_bytes" in c) == bind and all(("z_1" in p) == bind for p in c["passes"])   # z only where it is bound
        if kind == receipts.SKETCH_SHA512:
            assert len({p["sign_bits_sha512"] for p in c["passes"]}) == 3   # the signs are re-keyed every pass
            assert all(len(bytes.fromhex(s)) == 4 * rows for p in c["passes"] for s in p["sigma"].values())
    assert len({c["passes"][0]["root"] for c in chains.values()}) == 5


@pytest.mark.slow
@pytest.mark.parametrize("kind, bind, rows", [rv.DEFAULT, *rv.VARIANTS])
def test_the_provers_passes_and_the_sessions_draws_reproduce_the_pinned_ones(kind, bind, rows) -> None:
    """The generator computes receipts and draws on its own; the audit's `Prover` and `AuditRun` must agree with it, at
    the default codec, under each receipt kind, of the block alone and binding z, and at 64 rows.  Slow for the same
    store as `test_vectors_are_reproduced`, which encodes it once for all of them."""
    store = rv.store() if (kind, bind, rows) == rv.DEFAULT else rv.registered(kind, bind, rows)
    reg = store.registration
    challenger = coins.local(reg.digest, 0, rv.PARAMS.windows, rv.Entropy(rv.COIN_LABEL))
    clock = Clock()
    tiles = audit.NoiseTiles(96) if bind else None
    s = audit.AuditRun(dataclasses.replace(rv.PARAMS, receipt=kind, bind=bind, rows=rows), reg, tiles, challenger,
                       not_before_ns=0, clock=clock)
    prover = audit.Prover(store, tiles, entropy=rv.Entropy(rv.LEAF_SALT_LABEL))
    pinned = json.loads(rv.PATH.read_text())
    pinned = pinned if (kind, bind, rows) == rv.DEFAULT else pinned["variants"][rv.name(kind, bind, rows)]
    challenge, previous, passes = s.start(), reg.digest, []
    for i, (plan, want) in enumerate(zip(rv.PLANS, pinned["passes"]["passes"], strict=True)):
        passes.append(prover.run(challenge, i, previous, plan))
        ps = passes[-1]
        assert ps.root.hex() == want["root"] and ps.plan.runs == tuple(map(tuple, want["plan"]))
        assert s.receive(0, i, ps.root, ps.plan.runs, step=i) is None
        previous = ps.root
    clock.now = rv.PARAMS.window_ns
    assert s.boundary() is None
    clock.now = s.deadline(0) + 1
    draws = s.close(0)
    assert list(draws) == [unit(u) for u in pinned["openings"]["draws"]]
    assert s.verify(0, [prover.open(passes[i], p) for i, p in draws]) == audit.Verdict(0, True)
    assert s.record().public()["receipt"] == kind and s.record().public()["bind"] is bind
    assert s.record().public()["rows"] == rows


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


def _counter(label: bytes, n: int) -> bytes:
    return b"".join(hashlib.sha512(SKETCH_PINS + label + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n]


def test_sketch_sha512_is_ra_codec_tensors_reference() -> None:
    """Byte for byte with the numpy reference the GPU reader is held to: the signs, σ_j at the int8 extremes, of the
    block alone and under a one-byte, a repeated and a folded z, and y up to the last u64 index."""
    seeds = [hashlib.sha512(SKETCH_PINS + bytes([k])).digest() for k in range(2)]
    assert [hashlib.sha512(b"".join(receipts.sign_bits(s))).hexdigest() for s in seeds] == SKETCH_SIGN_BITS_SHA512
    blocks = {"0x80": b"\x80" * codec.BLOCK_BYTES, "0x7f": b"\x7f" * codec.BLOCK_BYTES,
              "counter": _counter(b"block", codec.BLOCK_BYTES)}
    zs = {"one": b"\x01", "short": bytes(range(100)), "folded": _counter(b"z", 5000)}
    receipters = [receipts.SketchReceipter(s) for s in seeds]
    for k, b, z, j, sigma, y in SKETCH_CASES:
        assert receipters[k].sketch(blocks[b], zs[z]).hex() == sigma
        assert receipters[k](j, blocks[b], zs[z]).hex() == y
        assert receipts.receipter(receipts.SKETCH_SHA512, seeds[k])(j, blocks[b], zs[z]).hex() == y
    for k, b, j, sigma, y in SKETCH_UNBOUND_CASES:
        assert receipters[k].sketch(blocks[b]).hex() == sigma
        assert receipters[k](j, blocks[b]).hex() == y == receipters[k](j, blocks[b], bytes(codec.BLOCK_BYTES)).hex()


def test_sketch_sha512_at_64_rows_is_ra_codec_tensors_reference() -> None:
    """The same inputs at k_s = 64, a registration's other row count: the signs under the key that binds k = 64, σ_j's
    256 bytes and y, three compressions after the seed's block."""
    seeds = [hashlib.sha512(SKETCH_PINS + bytes([k])).digest() for k in range(2)]
    assert [hashlib.sha512(b"".join(receipts.sign_bits(s, 64))).hexdigest() for s in seeds] == SKETCH64_SIGN_BITS_SHA512
    blocks = {"0x80": b"\x80" * codec.BLOCK_BYTES, "0x7f": b"\x7f" * codec.BLOCK_BYTES,
              "counter": _counter(b"block", codec.BLOCK_BYTES)}
    zs = {"one": b"\x01", "short": bytes(range(100)), "folded": _counter(b"z", 5000)}
    receipters = [receipts.SketchReceipter(s, 64) for s in seeds]
    for k, b, z, j, sigma, y in SKETCH64_CASES:
        assert hashlib.sha512(receipters[k].sketch(blocks[b], zs[z])).hexdigest() == sigma
        assert receipters[k](j, blocks[b], zs[z]).hex() == y
        assert receipts.receipter(receipts.SKETCH_SHA512, seeds[k], 64)(j, blocks[b], zs[z]).hex() == y
    for k, b, j, sigma, y in SKETCH64_UNBOUND_CASES:
        assert hashlib.sha512(receipters[k].sketch(blocks[b])).hexdigest() == sigma
        assert receipters[k](j, blocks[b]).hex() == y


@pytest.mark.parametrize("c", [codec.FALLBACK, codec.DEFAULT], ids=lambda c: c.name)
def test_codec_is_kv_v0_masked_at_p1024(c) -> None:
    assert codec.P1024.p == (1 << 1024) - 105 and c.field == codec.P1024 and c.masked
    k, pin = codec.key(bytes.fromhex(KV_SEED), c), KV_P1024_MASKED
    assert [format(v, "x") for v in k.c] == pin["c"][:c.rounds] and format(k.tau, "x") == pin["tau"]
    assert format(k.mask, "x") == pin["mask"]
    for i, m, y in pin["words"][c.rounds]:
        assert codec.encode_word(int(m, 16), i, k) == int(y, 16)
        assert codec.decode_word(int(y, 16), i, k) == int(m, 16)


def test_the_bound_key_is_kv_v0s() -> None:
    k, j = codec.key(bytes.fromhex(KV_SEED)), KV_BOUND["block"]
    payload = b"".join(hashlib.sha512(KV_BOUND_LABEL + c.to_bytes(4, "big")).digest() for c in range(64))
    d = codec.digest(payload)
    assert d.hex() == KV_BOUND["digest"] and format(codec.binding(d, j, k), "x") == KV_BOUND["b"]
    block = codec.encode(payload, j, k, bound=True)
    assert hashlib.sha512(block).hexdigest() == KV_BOUND["words_sha512"] and codec.decode(block, j, k, d) == payload
