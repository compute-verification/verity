"""The 64-row sign key with no card: the verifier's `receipts.sign_key` and `sign_bits` give the wide key's vector (K,
row 0's first 16 bytes, row 63's last 16) at 64 rows, its 16-row signs stay those of `randomness.derive` under
`sketch-signs/v0`, and `device.stream_head` is a head of the verifier's key; and `sign_key.py`, which rows.sh runs on a
built library's messages, finds the verifier's key and rows in messages framed as `rs_chain_set` frames them and names
what differs in messages that aren't. What the device's own kernels make of those messages needs a card
(`signs_match`)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/accounting/poub/prover/remote_session"))

import device
import sign_key
from verity.commitments import randomness
from verity.protocols.accounting.poub.verifier import receipts

FRAMES = randomness.frame(randomness.frame("row") + randomness.frame(0))
CONTEXT = {"k": 64, "n": 4096, "tag": "sketch-sha512"}
#: the 64-row sign key and two of its rows at a seed of 64 zero bytes: `sketch-signs/v1` under the wide derive
VECTOR = {"K": "b844824fb8704537ad7e865cc186b22f116ef99a0e48d8034d76cc07cf9ab10b"
               "d3e8ca87fb456891833fe7d994a5141c4b52bf3446f163a1a954024dfa5e8583",
          "row 0": "3f2ad9ff6856261e62ea25da12b64840", "row 63": "3333608eddf01f1e530f7d92d002db91"}


def test_the_verifiers_wide_key_and_its_rows_are_the_vectors():
    key, rows = receipts.sign_key(bytes(64), 64), receipts.sign_bits(bytes(64), 64)
    assert isinstance(key, randomness.WideKey) and key.value.hex() == VECTOR["K"]
    assert rows[0][:16].hex() == VECTOR["row 0"] and rows[63][-16:].hex() == VECTOR["row 63"]


@pytest.mark.parametrize("seed", [bytes(64), bytes(range(64))])
def test_the_verifiers_16_row_signs_are_the_32_byte_keys_under_v0(seed):
    want = randomness.derive(seed, receipts.SIGN_DOMAIN, {**CONTEXT, "k": 16})
    assert receipts.sign_key(seed, 16) == want
    assert receipts.sign_bits(seed, 16) == tuple(want.stream(("row", c), 512) for c in range(16))
    if seed == bytes(64):
        assert receipts.sign_bits(seed, 16)[0][:16].hex() == "fbec5d4f3cbc4bbed972e48453b9013e"


def test_a_pass_head_is_the_verifiers_key_under_its_stream():
    seed = bytes(range(64))
    wide = device.stream_head(seed, 64)
    assert wide == randomness._STREAM_WIDE + receipts.sign_key(seed, 64).value + FRAMES and len(wide) == 128
    narrow = device.stream_head(seed, 16)
    assert narrow == randomness._STREAM + receipts.sign_key(seed, 16).value + FRAMES and len(narrow) == 91


def messages(rows: int) -> tuple[bytes, list[int], bytes, list[int]]:
    """rs_chain_set's messages at `rows`, as it frames them."""
    wide = rows == 64
    domain = receipts.SIGN_DOMAIN_WIDE if wide else receipts.SIGN_DOMAIN
    ctx = b"".join(randomness.frame(k) + randomness.frame({**CONTEXT, "k": rows}[k]) for k in sorted(CONTEXT))
    tag = randomness._DERIVE_WIDE if wide else randomness._DERIVE
    derive = tag + randomness.frame(bytes(64)) + randomness.frame(domain) + randomness.frame(ctx)
    stream = randomness._STREAM_WIDE if wide else randomness._STREAM
    head = stream + bytes(64 if wide else 32) + FRAMES
    return derive, [len(derive), len(tag) + 9], head, [len(head) + 8, len(stream), len(head) - 1, len(head)]


@pytest.mark.parametrize("rows", [64, 16])
def test_the_scripts_hash_of_rs_chain_sets_messages_is_the_verifiers(rows):
    derive, dn, head, rn = messages(rows)
    assert (len(derive), len(head)) == ((238, 128) if rows == 64 else (233, 91))
    key, got = sign_key.derived(derive, dn, head, rn, rows)
    assert sign_key.differs(key, got, rows) == []
    if rows == 64:
        assert (key.hex(), got[0][:16].hex(), got[63][-16:].hex()) == (VECTOR["K"], VECTOR["row 0"], VECTOR["row 63"])


def test_the_script_names_what_differs():
    derive, dn, head, rn = messages(64)
    other = head.replace(randomness._STREAM_WIDE, randomness._STREAM + bytes(5))
    assert sign_key.differs(*sign_key.derived(derive, dn, other, rn, 64), 64) == [f"row {c}" for c in range(64)]
    other = derive.replace(randomness._DERIVE_WIDE, randomness._DERIVE + bytes(5))
    assert sign_key.differs(*sign_key.derived(other, dn, head, rn, 64), 64) == ["the key"] + [
        f"row {c}" for c in range(64)]
    derive, dn, head, rn = messages(16)
    other = b"w" + head[1:]
    assert sign_key.differs(*sign_key.derived(derive, dn, other, rn, 16), 16) == [f"row {c}" for c in range(16)]
