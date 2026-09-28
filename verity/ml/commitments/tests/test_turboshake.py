"""TurboSHAKE128 against RFC 9861 and an independent implementation, and Keccak-f[1600] against hashlib."""

import hashlib

import pytest

from verity.ml.crypto.turboshake import RATE, keccak_p, turboshake128

#: (message length n, domain byte, output length, digest): M = RFC 9861's ptn(n) (byte i is i mod 251).  The first is the
#: RFC's own vector; the rest came from pycryptodome 3.23's TurboSHAKE128 and cover both sides of the 168-byte rate.
VECTORS = [
    (0, 0x1f, 32, "1e415f1c5983aff2169217277d17bb538cd945a397ddec541f1ce41af2c1b74c"),
    (0, 0x1f, 200, "1e415f1c5983aff2169217277d17bb538cd945a397ddec541f1ce41af2c1b74c3e8ccae2a4dae56c84a04c2385c03c15e8193bdf58737363321691c05462c8dfdbdf137ce385dc51640ac13897b9078b56b752345f19ee63011fb016abd57cf2a5ca9bf410aee71044042719e1c3ebea94c398909bd8ec9b443e62b0cc0fd7c6b79519f0c470ebd12a0a423e74e845baf888e5d635b534049fe87b2528159ac3b5b69ad78425efe13728a261f2a4be2d4ea83a3b8c3be12fed74555f2410f0d0aa56d0d63967f8e9"),
    (1, 0x1f, 32, "55cedd6f60af7bb29a4042ae832ef3f58db7299f893ebb9247247d856958daa9"),
    (17, 0x1, 32, "6f0f5f330a7114ed345b97d012f8a8bac5ba32f1c0aafab22ef880737bf0c103"),
    (167, 0x6, 32, "9b90d05d7d32b2ce1a06c5da8a9d47a8a84576fce34885809d7fd558b2f68bb7"),
    (168, 0x1f, 32, "ed5bf22a6a67e3cfe1d1f974a9dee10da9da2fe264f55359ec56c16541ac5456"),
    (169, 0x21, 64, "664a263829a4cf992f2e3181d1790eeac487edee372c295c6fdf207f83759eb7415cf598f46540c1c746937e0ee04f69783db27eed152273f77903b9a0b72a55"),
    (289, 0xb, 32, "1231c6e1d03445af318ce5968f39779273960c17edb6ab3435e2058914661494"),
    (4096, 0x22, 32, "a6d532034c51b5e0e2f8445b73e09c4a6b1255e1897789c14231a5e1e1f88018"),
]


def ptn(n: int) -> bytes:
    return bytes(i % 251 for i in range(n))


@pytest.mark.parametrize("n,domain,length,digest", VECTORS)
def test_vectors(n, domain, length, digest):
    assert turboshake128(ptn(n), domain, length).hex() == digest


def test_rfc_9861_empty_message():
    assert turboshake128(b"", 0x1F, 32).hex() == "1e415f1c5983aff2169217277d17bb538cd945a397ddec541f1ce41af2c1b74c"


def test_keccak_f_is_shake128_permutation():
    """With all 24 rounds and SHAKE's padding, the sponge is hashlib's SHAKE128."""
    for n in (0, 5, 167, 168, 400):
        msg = ptn(n)
        padded = bytearray(msg) + b"\x1f"
        padded += bytes(-len(padded) % RATE)
        padded[-1] ^= 0x80
        state = [0] * 25
        for off in range(0, len(padded), RATE):
            for i in range(RATE // 8):
                state[i] ^= int.from_bytes(padded[off + 8 * i:off + 8 * i + 8], "little")
            keccak_p(state, 24)
        out = b"".join(lane.to_bytes(8, "little") for lane in state[:RATE // 8])[:32]
        assert out == hashlib.shake_128(msg).digest(32)


def test_domain_byte_range():
    for bad in (0x00, 0x80):
        with pytest.raises(ValueError):
            turboshake128(b"", bad)
