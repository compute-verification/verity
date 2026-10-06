"""The protocol's own values on the served -h3 path are SHA-512: `tagged_sha512` (the first 32 bytes of SHA-512 over the tag,
a zero byte and the parts' frames), each pinned.  -h3's tree bindings take it under v2 tags; `weights_root`, the coin
commitment's binding and Pearl-C's draw-after take it for every format.  h0's SHA-256 frames keep `tagged` under v1 (the
legacy format, left as it is)."""
from __future__ import annotations

import hashlib

from verity.primitives.randomness import frame
from verity.protocols.accounting.work.pouw import audit as A
from verity.protocols.accounting.work.pouw import coins as EC
from verity.protocols.accounting.work.pouw.protocol import tagged, tagged_sha512
from verity.protocols.accounting.work.pouw.schemes import pearl_c_work as W


def spec(tag: str, *parts) -> bytes:
    return hashlib.sha512(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest()[:32]


def test_tagged_sha512_is_sha512_truncated():
    assert tagged_sha512("verity/pouw/x/v1", b"a", 3) == spec("verity/pouw/x/v1", b"a", 3)
    assert len(tagged_sha512("verity/pouw/x/v1")) == 32


def test_h3_tree_bindings_are_sha512_under_v2_and_h0s_keep_sha256():
    salt = b"\x11" * 32
    assert A.activation_binding(salt, 7, A.HM96) == spec("verity/pouw/activation-rows/v2", salt, 7)
    assert A.activation_binding(salt, 7, A.HM96).hex() == "3a492e1c546bcfeeb11fed19234dca2632c8f95841637711775058ca73ac5bae"
    assert A.weight_binding(9, A.HM96) == spec("verity/pouw/weight-rows/v2", 9)
    assert A.weight_binding(9, A.HM96).hex() == "409d779f6493d6014a59f2b6bbc2de1f7eb32931fa6c373a4596b55586d1bad9"
    assert A.activation_binding(salt, 7) == tagged("verity/pouw/activation-rows/v1", salt, 7)
    assert A.weight_binding(9) == tagged("verity/pouw/weight-rows/v1", 9)


def test_weights_root_coins_binding_and_draw_after_are_sha512():
    roots = {2: b"\x22" * 64, 1: b"\x33" * 64}
    assert A.weights_root(roots) == spec("verity/pouw/weights/v2", 1, roots[1], 2, roots[2])
    assert A.weights_root(roots).hex() == "b9a2238442f6272ec208e6db5b74c5955634df3ab11d6832b711ca6dfab2616a"
    assert EC.domain(7).binding == spec("verity/pouw/epoch-coins/v2", 7)
    assert EC.domain(7).binding.hex() == "bab653836a2d151591ef458fc221ffbd8cf6441a9f06843cca1409c070b26e50"
    assert W.draw_after(b"\x44" * 64, b"\x55" * 32) == spec("verity/pouw/pearl-c/draw-after/v2", b"\x44" * 64, b"\x55" * 32)
    assert W.draw_after(b"\x44" * 64, b"\x55" * 32).hex() == "7d38b40f8889e1c82b9544024e04eca42a521a7805a87d66f56dee81e40d3094"
