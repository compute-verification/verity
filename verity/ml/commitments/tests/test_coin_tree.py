"""`verity.ml.commitments.coin_tree` is ``verity/ml/flock/live/src/coin_tree.rs`` byte for byte: docs/coin-tree-v2.md §9's
vectors (the key, the root, ``Hello``'s coins and answer, an opening and each intermediate), its openings check and its
forgeries do not, and every case of ``coin_open_vectors.json`` (which Rust's ``Checked`` and the Lean check answer) stops or
hands on the same coins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity.ml.commitments import coin_tree as CT

VECTORS = Path(__file__).resolve().parents[4] / "verity/ml/flock/live/src/coin_open_vectors.json"
NONCE = bytes(range(CT.NONCE_BYTES))
SPEC9 = CT.Spec(("session", "t/rep0", "t/rep1"), 2, 3)
#: the live protocol's `Resp::Coins` tag: a frame is the tag, u32le(W), then W words of 16 bytes
RESP_COINS = 2


def _resp_coins(words: list[int]) -> bytes:
    return bytes([RESP_COINS]) + len(words).to_bytes(4, "little") + CT.unwords(words)


def _spec_tape():
    """§9's tape: SHA-512("verity/coin-tree/v2/test-tape\\0" ‖ u64le(i)), i = 0, 1, …, read in order."""
    stream = b"".join(hashlib.sha512(b"verity/coin-tree/v2/test-tape\0" + i.to_bytes(8, "little")).digest() for i in range(4096))
    at = [0]

    def rand(n: int) -> bytes:
        out = stream[at[0]:at[0] + n]
        at[0] += n
        return out

    return rand


def _tape(seed: int):
    """coin_tree.rs's ``tape``: each call's bytes in 64-byte chunks, chunk ``i`` of the tape SHA-512(seed ‖ u64le(i))."""
    ctr = [0]

    def rand(n: int) -> bytes:
        out = b""
        for _ in range(0, n, 64):
            out += hashlib.sha512(bytes([seed]) + ctr[0].to_bytes(8, "little")).digest()
            ctr[0] += 1
        return out[:n]

    return rand


def test_the_draw_reproduces_the_specs_vectors():
    t = CT.CoinTree.draw(SPEC9, NONCE, _spec_tape())
    r = t.rules
    assert t.key.hex() == (
        "53d43b82d13a8d6cd586807a53763a5864e185bca221ce92e9d427b309dee61dba712538f39a2df44ab848082f9399912409587ef73bec0e93bb890a30618c78"
        "6071e1daec3262a1a4172b6133844b9d6095bfc7e1c10907c13f346242ba55f6da7ce6933f81e691d1e5c423a13ecee129e4c48c3ce4accf0203a6629c9b29"
        "73d922a4ed7d7ac9a477ecca699155231c4558ca66b6291f43b23e3df6f2d43a80f69b5cdc1961ae302bbf547bdac99657384b447ddc8cff866d6429b101e6"
        "dfca9ec72e302c96d55bba508c16ed1abe3d1057c421c2ffbf1266c6485e90f8e1e22081b72242fe8396aca78cc5908947d6ebde3c7def16cc0dbb65ed499807e625")
    assert t.root().hex() == ("468d899b6bde51c72568b9cd72b6d35c809454fe82b991cd46647fec631224cc"
                              "37af0b859a1133ce849b690e6583b7a12ddb90b2126611e6dcbaad8daee35740")
    assert json.dumps({**SPEC9.to_json(), "nonce": NONCE.hex()}, sort_keys=True, separators=(",", ":")) == (
        '{"block":3,"nonce":"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f","rounds_log":2,'
        '"scheme":"coin-tree/hm96-sha512/v2","streams":["session","t/rep0","t/rep1"]}')
    assert _resp_coins(t.answer_words()).hex() == "0214000000" + t.root().hex() + t.key.hex()
    s, k, n = 1, 2, 2
    c, w = t.open(s, k, n)
    assert [CT.f128_bytes(x).hex() for x in c] == ["063456f31ef84a3d219bb0b5fd492f5e", "401968e9dd010aa780e6c0f421c2b05f"]
    assert len(w) == 34 == SPEC9.opening_words(n)
    assert hashlib.sha512(CT.unwords(w)).hexdigest() == (
        "6aef2a682b4c872097f6af4f4e167f45a42c11eaf3fb889cc538f90636fbc0f40f16bec75cc10e2bcfd74650086ea766e7a9caf995a1956c89841f108aeb02a3")
    assert CT.unwords(w[n:n + 4]).hex() == (
        "9b2994f96eacf0551dc8d8c41ac00f8cce122d575342691400262daf8489fbbef9becea2e575a2e073dcfe696d676eee0bc349ff68dfdda8f278c378fec23c51")
    t0 = CT._chain(r, t.blocks[s][k], 0, SPEC9.block, r.chain_end())
    assert t0.hex() == ("b75c0977d0607fed217d48f346ff7a85f0e6874793939b0368f3286c8828dece"
                        "613fd57cd88f037710bb7b9f180b1725afb2f47c94cf2ac57b383411f5045205")
    salt = t.salts[s][k]
    x = r.h(r.round_tag, s.to_bytes(4, "big"), k.to_bytes(4, "big"), t0)
    assert x.hex() == ("c75f2678385f0b50c6dd3a849a024dad914a0741c35374601deb048d071522b5"
                       "dbb403561feda7da12d3f510ed964c4784bff2c580f9439e2f5456ed3777f1dc")
    assert bytes(p ^ q for p, q in zip(x, r.mask(salt))).hex() == (
        "d03b0fdcadbbc02b5e8b6ad48a16aa16a9e1d499553c1429b396b5de1a12e8df2457d1dfa8908e4eb3b856e3c22dcc141e9809ea8731ea4a0e9bdcc077eaf043")
    assert r.h(r.salt_prefix, salt).hex() == (
        "5a5a8122e97ed2a930a1a15ef7cff32f4b325fe1b68eb63fd3fdf1cf8b1feaa2abaeb6ac59b4befc7a6ada40408d65773bb6f57c36518d4629f287d7b0a5c033")
    assert r.leaf(s, k, t0, salt).hex() == (
        "e370b4556e90d4e8947636c4bb3264314f2bfc716ec732a2907ff9fd79d164c5886113c94171638c7243ac95ffe7262eef0653517264ad50a3dd4cfa1515165a")
    assert r.h(r.tail).hex() == (
        "3ea637a0d4617c22acbf0b6fbb4ea41519715341e8cfa7ac44176184e300db5a0ce5e709add52b48901e928adeb88d9fea4fafa82a295044e44719a586e52176")
    assert r.pad().hex() == (
        "6e03778139201207311f8606aef149711173deeb9c783b33ba9524ceb81e0c74669c7984a3f2b8c1b82304e66a4ebd84a285046ed467226e30aff27ec64f6320")
    assert CT.check(SPEC9, r, t.root(), s, k, n, w) == c
    for wrong in (CT.Rules(bytes(CT.NONCE_BYTES), t.key), CT.Rules(NONCE, bytes(CT.KEY_BYTES))):
        with pytest.raises(ValueError, match="do not open"):
            CT.check(SPEC9, wrong, t.root(), s, k, n, w)
    assert t.record() == {"key": t.key.hex(), "nonce": NONCE.hex(), "root": t.root().hex(), "spec": SPEC9.to_json()}
    assert CT.Spec.of_json({**SPEC9.to_json(), "nonce": NONCE.hex()}) == SPEC9


def test_openings_check_and_forgeries_do_not():
    spec = CT.Spec((CT.SESSION, "t/rep0", "t/rep1"), 4, 32)
    t = CT.CoinTree.draw(spec, NONCE, _tape(1))
    root, r = t.root(), t.rules
    for s, k, n in ((0, 0, 27), (1, 3, 1), (2, 15, 32), (1, 0, 0)):
        c, w = t.open(s, k, n)
        assert CT.check(spec, r, root, s, k, n, w) == c
        if n:
            with pytest.raises(ValueError):
                CT.check(spec, r, root, s, k, n, [w[0] ^ 1, *w[1:]])
        with pytest.raises(ValueError):
            CT.check(spec, r, root, s, (k + 1) % 16, n, w)
        with pytest.raises(ValueError):
            CT.check(spec, r, root, (s + 1) % 3, k, n, w)
    c5, _ = t.open(1, 2, 5)
    c9, w9 = t.open(1, 2, 9)
    assert c5 == c9[:5]
    with pytest.raises(ValueError):
        CT.check(spec, r, root, 1, 2, 5, w9)
    assert CT.CoinTree.draw(spec, NONCE, _tape(2)).root() != root
    assert CT.CoinTree.draw(spec, bytes([1]) * CT.NONCE_BYTES, _tape(1)).root() != root


def _frame_words(frame: str) -> list[int]:
    b = bytes.fromhex(frame)
    assert b[0] == RESP_COINS and int.from_bytes(b[1:5], "little") * 16 == len(b) - 5
    return CT.words(b[5:])


def _walk(case: dict) -> list[str | None]:
    """The case's steps through this module's checks, as the prover's side takes them: each step's coins (hex) or None
    for a stop, which ends the walk."""
    out: list[str | None] = []
    rules = root = None
    rounds: dict[str, int] = {}
    for st in case["steps"]:
        try:
            if "hello" in st:
                nonce = CT.hello_nonce(st["hello"])
                out.append("")
            elif "answer" in st:
                root, key = CT.answer_of(_frame_words(st["answer"]))
                rules = CT.Rules(nonce, key)
                out.append("")
            else:
                if rules is None:
                    raise ValueError("coins before Hello's answer")
                stream = st["open"]
                if stream not in SPEC9.streams:
                    raise ValueError(f"{stream} is outside the tree")
                k = rounds.get(stream, 0)
                rounds[stream] = k + 1
                coins = CT.check(SPEC9, rules, root, SPEC9.streams.index(stream), k, st["n"], _frame_words(st["frame"]))
                out.append(CT.unwords(coins).hex())
        except ValueError:
            out.append(None)
            break
    return out


def test_every_case_of_the_lean_checks_vectors_stops_or_hands_on_what_checked_does():
    cases = json.loads(VECTORS.read_text())["cases"]
    assert len(cases) > 30
    for case in cases:
        want = [None if "stop" in st else st.get("coins", "") for st in case["steps"]]
        assert _walk(case) == want, case["case"]
