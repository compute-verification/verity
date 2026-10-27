"""The toy served from its P2 store under the remote bandwidth audit (`p2_v1/serve.py`): the decode-then-matmul engine
gives the plaintext model's tokens, its receipts from the labels' partials are the reference's, an honest session is
accepted with each window's openings answered before the next step, a step that skips a label's read is caught, and
the kernel fixture keeps its pinned bytes and checks by hand."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity")

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from p2_v1 import serve as V  # noqa: E402
from p2_v1 import store as S  # noqa: E402

from verity.core.protocols.remote_bandwidth import p2, receipts  # noqa: E402

STORE, DOC, C_STAR = V.toy()
SALT = bytes.fromhex(DOC["salt"])


@pytest.fixture(scope="module")
def p2_store() -> p2.Store:
    return p2.Store(V.registration(STORE, DOC, C_STAR), V.blocks_of(STORE, C_STAR))


def engine() -> V.Engine:
    return V.Engine(STORE, C_STAR, SALT)


def test_the_decoded_model_is_the_plaintext_model() -> None:
    served, plain = engine(), V.Engine(STORE, C_STAR, SALT, w=S.toy_plain(STORE)["w.bin"])
    a, b = 3, 3
    for _ in range(5):
        a, partials = served.step(a, p2.Sketcher(hashlib.sha512(b"s").digest()))
        b, none = plain.step(b)
        assert a == b and sorted(partials) == list(range(32)) and none == {}
    assert served.decoded == 5 * 32 and served.audited == 32 and STORE.params.blocks == 33


def test_a_steps_receipts_are_the_reference_passes(p2_store) -> None:
    seed = hashlib.sha512(b"a pass seed").digest()
    sk = p2.Sketcher(seed)
    _, partials = engine().step(7, sk)
    sigmas, ys = V.unit_receipts(sk, p2_store.registration.blocks, partials)
    assert ys == [sk(g, p2_store.unit(g)) for g in range(8)]
    assert sigmas == [sk.sketch(p2_store.unit(g)) for g in range(8)]


def test_an_honest_served_session_is_accepted(p2_store) -> None:
    out = V.serve(engine(), p2_store, prompt=1, windows=3, per_window=2)
    assert out.outcome == "accepted" and len(out.tokens) == 6
    assert out.passes == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert [(v.window, v.accepted, n) for v, n in out.verdicts] == [(0, True, 3), (1, True, 5), (2, True, 6)]
    plain = V.Engine(STORE, C_STAR, SALT, w=S.toy_plain(STORE)["w.bin"])
    token, want = 1, []
    for _ in range(6):
        token, _ = plain.step(token)
        want.append(token)
    assert out.tokens == want


def test_a_step_that_skips_a_blocks_read_is_caught(p2_store) -> None:
    """Label 13's partial from zeros, as a server that never read it would have it, in window 1's second step: that
    unit's receipt is wrong, and every receipt is opened."""

    def step(e: V.Engine, token: int, sk: p2.Sketcher):
        token, partials = e.step(token, sk)
        if len(e.cache[0]) == 4:
            partials[13] = sk.block(13, bytes(p2.BLOCK_BYTES))
        return token, partials

    out = V.serve(engine(), p2_store, prompt=1, windows=3, per_window=2, step=step)
    assert out.outcome == "receipt"
    assert [(v.window, v.accepted, v.failure, v.at) for v, _ in out.verdicts] == [
        (0, True, None, None), (1, False, "receipt", (1, 3))]


@pytest.fixture(scope="module")
def kernel_fixture() -> bytes:
    return V.fixture_bytes()


def test_the_kernel_fixture_keeps_its_bytes(kernel_fixture) -> None:
    assert hashlib.sha512(kernel_fixture).hexdigest() == V.FIXTURE_SHA512, (
        "the fixture moved: kernels build against the published bytes, so pin the new digest and republish them")


def test_the_kernel_fixture_checks_by_hand(kernel_fixture) -> None:
    """Each label's partial from the fixture's own sign rows and `C*`'s bytes, each unit's σ the sum of its four,
    y the hash of σ, and the passes chained."""
    doc = json.loads(kernel_fixture)
    assert hashlib.sha512(C_STAR).hexdigest() == doc["store"]["c_star_sha512"] and doc["store"]["unaudited"] == [32]
    previous = doc["registration"]["digest"]
    for ps in doc["passes"]:
        seed = bytes.fromhex(ps["seed"])
        assert ps["previous"] == previous
        assert seed == receipts.seed(bytes.fromhex(doc["challenge"]), ps["pass"], bytes.fromhex(previous))
        signs = [bytes.fromhex(r) for r in ps["signs"]]
        assert len(signs) == 16 and {len(r) for r in signs} == {1028}
        partials = [struct.unpack("<16i", bytes.fromhex(x)) for x in ps["partials"]]
        for j in (0, 13, 31):
            block, row = C_STAR[2056 * j:2056 * (j + 1)], signs[5][257 * (j % 4):257 * (j % 4 + 1)]
            want = sum((-1 if row[x // 8] >> (x % 8) & 1 else 1) * (v - 256 if v > 127 else v)
                       for x, v in enumerate(block))
            assert partials[j][5] == want
        for g in range(8):
            sigma = struct.pack("<16i", *(sum(partials[4 * g + b][c] for b in range(4)) for c in range(16)))
            assert ps["sigmas"][g] == sigma.hex()
            y = hashlib.sha512(receipts.sketch_prefix(seed) + g.to_bytes(8, "big") + sigma).hexdigest()
            assert ps["receipts"][g] == y
        previous = ps["root"]
