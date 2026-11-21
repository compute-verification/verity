"""`verity.commitments.session_coins`, the memory attester's coin tree and signed header, which the attester writes and a
supplied function's body reads: the tags are the domains' table's and the Lean's literals are held equal to it (the commitments'
Lean package doesn't require the identity package), the attester's names are the moved ones, and the bytes are the formats'."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from verity.commitments import domains as D
from verity.commitments import session_coins as SC
from verity.commitments.crypto import lms
from verity.protocols.accounting.memory.pous.attester import memory_attester as MA

LEAN = Path(__file__).resolve().parents[3] / "verity/commitments/lean/Commitments/SessionCoins.lean"
DEF = re.compile(r'^def ([A-Z_0-9]+) : (?:String|ByteArray) := "([^"\\]*)"(\.toUTF8\.push 0)?$', re.M)


def fixed(label: str, n: int) -> bytes:
    out, i = b"", 0
    while len(out) < n:
        out += hashlib.sha512(label.encode() + i.to_bytes(4, "big")).digest()
        i += 1
    return out[:n]


def test_the_lean_literals_are_the_domains_entries():
    lean = {name: (text, bool(nul)) for name, text, nul in DEF.findall(LEAN.read_text())}
    want = {"PROTOCOL": (D.POUS_MEMORY_CHALLENGER, False), "COIN_TREE": (D.POUS_COIN_TREE_HM96_SHA512, False),
            "COIN_TAG": (D.POUS_MEMORY_CHALLENGER_COIN, True), "COIN_BINDING": (D.POUS_MEMORY_CHALLENGER_COIN_TREE, True),
            "COIN_SCHEMA": (D.POUS_COIN_HM96_SHA512, False), "MEMORY_CHALLENGER_RECORD": (D.MEMORY_CHALLENGER_RECORD, False)}
    assert {k: lean.get(k) for k in want} == want


def test_python_reads_the_tags_from_the_table_and_the_attester_has_the_moved_names():
    assert SC.COIN_TAG == D.POUS_MEMORY_CHALLENGER_COIN.encode() + b"\0" and SC.COIN_BINDING.endswith(b"\0")
    assert lms.ROLES["memory-challenger-record"] == D.MEMORY_CHALLENGER_RECORD
    for name in ("coin_digest", "coin_leaf"):
        assert getattr(MA, name) is getattr(SC, name)
    assert MA._domain is SC.domain
    assert not any(hasattr(MA, n) for n in ("PROTOCOL", "COIN_TREE", "RECORDS", "COIN_TAG", "COIN_BINDING", "COIN_SCHEMA",
                                            "COMPLETE", "LATE")), "the attester reads the commitments' names, never a second one"


def session():
    sid = fixed("session", 32)
    key = SC.HM96.key_from_label(b"session-coins-test")
    salts = [fixed(f"salt{j}", SC.HM96.salt_bytes) for j in range(3)]
    tree = MA.CoinTree(sid, 1 << 20, [5, 1023, 70000], key, salts)
    return MA.SessionCommitment(sid, tree.root, key, 1 << 20, 3, 854_000, 2056, fixed("binding", 64)), tree


def test_the_coin_digest_and_the_header_are_the_formats_bytes():
    c, tree = session()
    assert SC.coin_digest(c.session, 2, 70000) == hashlib.sha512(
        b"verity-pous/memory-challenger/coin/v1\0" + c.session + (2).to_bytes(4, "big") + (70000).to_bytes(8, "big")).digest()
    head = SC.header(c.session, c.coin_root, hashlib.sha512(c.coin_key).digest(), c.binding, c.blocks, c.k, c.limit_ns,
                    c.block_bytes, "complete", 3)
    assert head == json.dumps({**c.header(), "ended": "complete", "rounds": 3}, sort_keys=True, separators=(",", ":")).encode()
    assert len(head) == 753 and b'"ended":"complete"' in head and b'"rounds":3' in head
    rec = fixed("record", 64)
    ctx = SC.signed_context(fixed("lms-I", 16), c.session)
    assert len(ctx) == 106
    assert hashlib.sha512(ctx + (64).to_bytes(8, "big") + rec).digest() == lms.memory_attester_message(fixed("lms-I", 16), c.session, rec)


def test_an_opening_opens_under_its_root_and_nothing_else_does():
    c, tree = session()
    for j in range(3):
        o = tree.opening(j)
        args = (c.session, c.coin_root, c.coin_key, c.blocks, c.k)
        assert SC.verify(*args, o.j, o.index, o.salt, o.path) and MA.verify_coin(c, o)
        assert not SC.verify(*args, o.j, o.index + 1, o.salt, o.path)
        assert not SC.verify(*args, o.j, o.index, o.salt[:-1], o.path)
        assert not SC.verify(*args, o.j, o.index, o.salt, o.path[:-1])
        assert not SC.verify(c.session, c.coin_root, c.coin_key, c.blocks, 2, 2, o.index, o.salt, o.path)
        assert not SC.verify(c.session, c.coin_root, c.coin_key, o.index, c.k, o.j, o.index, o.salt, o.path)
        assert not SC.verify(c.session, bytes(64), c.coin_key, c.blocks, c.k, o.j, o.index, o.salt, o.path)
