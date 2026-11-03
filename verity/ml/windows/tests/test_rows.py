"""`rows`' bytes on a small case, pinned: the salts of ports (nonces past 32 bits included) and of a row table's segment, and
`commit_ports`' roots and ``b || c`` strings over every port kind (rows of each schema, given strings, words) on every
chunking and on the parallel path, so a change to the frame, the salt stream or the hiding leaves that moves a byte fails
here; and the salts are RFC 8439's ChaCha20 stream."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
from verity.ml.windows import rows as SR

KEY = bytes(range(32))
SALTS = {
    0: "ee8cc127eaa774461efe38871317cf3a416910244adcf78554d60d9761a0811734546ee86ab4fe77c573f4546ddfa76867f68a08d7a091468499750b1be9067b",
    1: "818319e7a0a50c602d29b1bffa25bf162f5d6472c5ce22d356a54090f8d7d128a906ce573bc860c654956e384f9b4c04e301e4a6564b9c7b63588d26b1684687",
    7: "f8f75db1d91f624888cdb425e1abce1499a58485a059d87fa2dce63696e716720817ca85e32be59c0e1cbb0d143b98776637afc4a074ad8fceba4833c76a1c90",
    (1 << 32) + 3: "b9bb8019afa79c4af51e8ecb85054f580f5406c52817d3e8e5fd7c5cd4ceaa1fdd827964d1fe15306f949c8011f4f678e23ca1c894d958ce9f2d6219189b8b0b",
    (1 << 40) + 5: "7df0108c76b5ae573036687dc0eb1d6c796e7743096323e49eeb3f039966f30a05e51a4cd8ffca6b39b224ef8535e1ce3569fd809cbc225104c297ec874c26f8",
}
SEGMENT_SALTS = "d896d410d9d288b07b559d42889091e1ee5b0a8c63dd565430f079c9a50a1be788e950754bf8e1f1e226c4a9ebae2d4bec0c5b672b41a31e1ea1723c2e3b6ade"
#: each port's root, and the SHA-512 of its rows' ``b || c`` strings
ROOTS = {
    "a": "69ff48fcbde3bc047508d52457c4fd51606472e5493dbae59c80e409b50a93d7165cb1e2932076e452babe73b6607ad2edcf3628bef56960975508deab7fb37c",
    "b": "7bcf50f142c041405d254287398c2165d9c02903752016fda15be55b3db5c37b39343e87aa002e191bd28d8a060779c3cfa4458bf4a87dd8ac0f4b9051845224",
    "c": "6594452e6e0b8a8193c7844946dbf5343f7cbf76309f60dd644f351d5339a8299f2a6e7f02363e4798d9c90ccb8c146efd2269753e0deea9b16f3dbc6cadd3c6",
    "d": "5b1a7b460851f788dc9d59130e46df8eb662feb2131beafd89d6a848e0c05f49dfda1572f0582499300ebfaefb96da24a1fd69906889a2d070f54364110b0c2a",
    "s": "e3b1e26d5e37f90967416bd7b4ad530b9d688e95eb28a53caa2297251402a50c1b30b87f5baab0bf80ddcf247e1ea3ecba9cff1db7bec1646b855c077aed4669",
    "w": "978fb55b38a4427145253f0687be0f18efc790cfb4615667c4cec6d750f296acb3df4dae66ac67a30e08475cb28f9e7d93a2fc93a873c498622106159c38faa0",
}
BCS = {
    "a": "04e2344548814df2490077291e40cfb7b79a3bf4d597dfe16f52fe05566b49366e2b12d2ec724c1d0f96265aba677fc9a8547eb7a94689af0a23570ff874f44c",
    "b": "e73720a2d2cae4d2d5897b8b2d47aa9d836ae94017d6f7ba1956f84f41a2d8ec97b7cf666c9f298b77f5d33984d5d496bd0766924482ead13113162c92026f9e",
    "c": "b8e18d597ba636bfb53bdd7f1d38e765fd8a9d20f69dcfcec96c02eca90382e51ace99997370b72381d6b2c9666c3d3363fa08ef72b36aa2d038915316eaef48",
    "d": "d1948f7929eaeebcc2514b6a0db64e4e1fb170678bd9ac1b157a1e7154eb773fff9bfdf18091b76ede0d6224d7f5f2c12986cf73d0d6863c8d77e649dba629b2",
    "s": "7df050dde248c210c5ae71bbf3d53f13161abcda23808d0244ea04ba0f8b0c526bf36638dd80b19d92598fd0645439d9effcfd4b8b189b7311151f486c6ed357",
}
COUNTS = {"a": 1, "b": 5, "c": 8, "d": 13, "w": 7, "s": 6}
SCHEMAS = {"c": SCHEMA_HM96_SHA512_ROW_V2, "d": SCHEMA_HM96_SHA512_ROW_SEG}


def _sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def _ports() -> tuple[dict, dict, dict, dict, dict]:
    rng = np.random.default_rng(5)
    doms = {p: CommitmentDomain(hashlib.sha512(p.encode()).digest(), -1, RangeIndexedDomain(0, n), "sha512") for p, n in COUNTS.items()}
    rows = {"a": rng.integers(0, 1 << 16, (1, 64), dtype=np.uint16), "b": rng.integers(0, 1 << 16, (5, 64), dtype=np.uint16),
            "c": rng.integers(0, 1 << 16, (8, 40), dtype=np.uint16), "d": rng.integers(0, 1 << 16, (13, 600), dtype=np.uint16)}
    ys = {p: SR.salts(KEY, k, COUNTS[p]) for k, p in enumerate(rows)}
    words = {"w": rng.integers(0, 1 << 16, 7, dtype=np.uint16)}
    strings = {"s": bytes(rng.integers(0, 256, 6 * SR.BC_BYTES, dtype=np.uint8))}
    return doms, rows, ys, words, strings


def test_salts_are_pinned() -> None:
    assert {port: _sha512(SR.salts(KEY, port, 5)) for port in SALTS} == SALTS
    assert _sha512(SR.segment_salts(KEY, 3, 11)) == SEGMENT_SALTS
    assert SR.salts(KEY, 2, 0) == b""


@pytest.mark.parametrize("chunk,workers", [(1, 1), (2, 1), (2, 2), (4, 1), (SR.CHUNK, 1)])
def test_commit_ports_is_pinned(chunk: int, workers: int) -> None:
    doms, rows, ys, words, strings = _ports()
    roots, bc = SR.commit_ports({p: doms[p] for p in (*rows, *words)}, rows, words, ys, workers=workers, chunk=chunk,
                                schemas=SCHEMAS)
    more, bc_s = SR.commit_ports({"s": doms["s"]}, {}, {}, {}, chunk=chunk, strings=strings,
                                 schemas={"s": SCHEMA_HM96_SHA512_ROW_V2})
    assert {p: r.hex() for p, r in {**roots, **more}.items()} == ROOTS
    assert {p: _sha512(b) for p, b in {**bc, **bc_s}.items()} == BCS


def test_salts_are_rfc8439s_chacha20() -> None:
    """A port's salts are RFC 8439's ChaCha20 stream under the window's key with the port's index as the nonce (12 bytes LE),
    row i at offset 192 i: section 2.3.2's block 1, and the ``cryptography`` package's stream."""
    nonce = int.from_bytes(bytes.fromhex("000000090000004a00000000"), "little")
    assert SR.salts(KEY, nonce, 1)[64:128].hex() == (
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    for bad in (-1, 1 << 96):
        with pytest.raises(ValueError, match="12 bytes"):
            SR.salts(KEY, bad, 1)
    crypto = pytest.importorskip("cryptography.hazmat.primitives.ciphers")
    for port in (3, (1 << 64) + 9):
        enc = crypto.Cipher(crypto.algorithms.ChaCha20(KEY, bytes(4) + port.to_bytes(12, "little")), mode=None).encryptor()
        assert SR.salts(KEY, port, 1042) == enc.update(bytes(1042 * SR.SALT_BYTES))
