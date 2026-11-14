"""The 64-row sign key with no card: `device.sign_key` and `stream_head` give the wide key's vector (K, row 0's first
16 bytes, row 63's last 16) at 64 rows and randomness v2's head at 16; and `sign_key.py`, which rows.sh runs on a
built library's messages, finds the vector in messages framed as `rs_chain_set` frames them and names what differs in
messages that aren't. What the device's own kernels make of those messages needs a card (`signs_match`)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/accounting/poub/prover/remote_session"))

import device
import sign_key
from verity.commitments import randomness

FRAMES = randomness.frame(randomness.frame("row") + randomness.frame(0))


def row(tag: bytes, key: bytes, c: int) -> bytes:
    head = tag + key + randomness.frame(randomness.frame("row") + randomness.frame(c))
    return b"".join(hashlib.sha512(head + k.to_bytes(8, "big")).digest() for k in range(8))


def test_the_hosts_wide_key_and_its_rows_are_the_vectors():
    key = device.sign_key(bytes(64), 64)
    assert key.hex() == sign_key.VECTOR["K"]
    assert row(device.STREAM_WIDE, key, 0)[:16].hex() == sign_key.VECTOR["row 0"]
    assert row(device.STREAM_WIDE, key, 63)[-16:].hex() == sign_key.VECTOR["row 63"]


def test_a_pass_head_is_the_wide_streams_at_64_rows_and_randomness_v2s_at_16():
    seed = bytes(range(64))
    source = device.rb._digest(seed, "a pass seed")
    wide = device.stream_head(seed, 64)
    assert wide == device.STREAM_WIDE + device.sign_key(source, 64) + FRAMES and len(wide) == 128
    narrow = randomness.derive(source, device.rb.SIGN_DOMAIN, {"tag": "sketch-sha512", "k": 16, "n": 4096})
    assert device.stream_head(seed, 16) == randomness._STREAM + narrow.value + FRAMES and len(FRAMES) + 60 == 91
    assert device.sign_key(source, 16) == narrow.value


def messages(rows: int) -> tuple[bytes, list[int], bytes, list[int]]:
    """rs_chain_set's messages at `rows`, as it frames them."""
    wide = rows == 64
    domain = device.SIGN_DOMAIN_WIDE if wide else device.rb.SIGN_DOMAIN
    ctx = b"".join(randomness.frame(k) + randomness.frame({**sign_key.CONTEXT, "k": rows}[k])
                   for k in sorted(sign_key.CONTEXT))
    tag = device.DERIVE_WIDE if wide else randomness._DERIVE
    derive = tag + randomness.frame(bytes(64)) + randomness.frame(domain) + randomness.frame(ctx)
    stream = device.STREAM_WIDE if wide else randomness._STREAM
    head = stream + bytes(64 if wide else 32) + FRAMES
    return derive, [len(derive), len(tag) + 9], head, [len(head) + 8, len(stream), len(head) - 1, len(head)]


@pytest.mark.parametrize("rows", [64, 16])
def test_the_scripts_hash_of_rs_chain_sets_messages_is_the_reference(rows):
    derive, dn, head, rn = messages(rows)
    assert (len(derive), len(head)) == ((238, 128) if rows == 64 else (233, 91))
    assert sign_key.differs(*sign_key.derived(derive, dn, head, rn, rows), rows) == []


def test_the_script_names_what_differs():
    derive, dn, head, rn = messages(64)
    other = head.replace(device.STREAM_WIDE, randomness._STREAM + bytes(5))
    assert sign_key.differs(*sign_key.derived(derive, dn, other, rn, 64), 64) == ["row 0", "row 63"]
    other = derive.replace(device.DERIVE_WIDE, randomness._DERIVE + bytes(5))
    assert sign_key.differs(*sign_key.derived(other, dn, head, rn, 64), 64) == ["K", "row 0", "row 63"]
    derive, dn, head, rn = messages(16)
    other = b"w" + head[1:]
    assert sign_key.differs(*sign_key.derived(derive, dn, other, rn, 16), 16) == [f"row {c}" for c in range(16)]
