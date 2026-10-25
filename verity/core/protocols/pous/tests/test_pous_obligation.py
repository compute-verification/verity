"""PoUS as an obligation at test scale (`obligation`): the KV pages' codec under coin-keyed tags, the memory challenger's
coin per page write, and the verdict on its log, for the honest run, the three attacks (`obligation_cases.CASES`) and
an edit reaching each of `obligation.REASONS`; then one window with writes during it, and the space leg's KV window
(`period.kv_window`). Lean's verdict is held to this one in `check/`'s `test_obligation_difftest.py`."""

from __future__ import annotations

import copy
import itertools
import random

import pytest

import obligation_cases as C
from verity.core.protocols.pous import obligation as O
from verity.core.protocols.pous import period


def test_pages_decode_on_use():
    """Each page's latest version decodes to what was written last, and every block is a codeword below p."""
    rng = random.Random(1)
    mc, store = C.challenger(rng, 3, 5), O.PageStore(C.SALT, 3, 5)
    last = {}
    for p in [0, 1, 2, 1, 1, 0]:
        last[p] = rng.randbytes(O.params(5).segment_bytes)
        store.write(mc, p, last[p])
    assert {p: store.plaintext(p) for p in last} == last
    assert store.versions == {0: 1, 1: 2, 2: 0}
    q = O.params(5)
    assert all(int.from_bytes(b, "big") < q.p for _, bs, _ in store.held.values() for b in bs)


def test_a_fresh_coin_rekeys_the_page():
    """The same content under two coins of one page version, or one coin at two versions, encodes differently: the tag
    carries the coin and the version."""
    q, plain = O.params(4), bytes(O.params(4).segment_bytes)
    a, b = bytes(32), bytes([1]) + bytes(31)
    encs = {tuple(O.encode_page(q, O.page_salt(C.SALT, 0, v, c), plain)) for v, c in [(0, a), (0, b), (1, a)]}
    assert len(encs) == 3


def test_keystream_is_the_mask():
    """`keystream` is each block's mask K_j cut to the payload, so it moves with the coin."""
    q = O.params(3)
    k0, k1 = (O.keystream(q, O.page_salt(C.SALT, 2, 0, bytes([c]) * 32)) for c in (0, 1))
    assert len(k0) == q.segment_bytes and k0 != k1


def test_the_challenger_draws_one_coin_per_committed_version():
    mc = C.challenger(random.Random(0), 2, 2)
    with pytest.raises(ValueError, match="uncommitted or has its coin"):
        mc.coin(0, 0)
    mc.commit(0, 0, bytes(64))
    mc.coin(0, 0)
    with pytest.raises(ValueError, match="uncommitted or has its coin"):
        mc.coin(0, 0)
    assert [e["op"] for e in mc.log()["events"]] == ["commit", "coin"]


@pytest.mark.parametrize("name", sorted(C.CASES))
@pytest.mark.parametrize("seed", range(3))
def test_cases(name, seed):
    build, reason = C.CASES[name]
    got = O.verdict(build(seed))
    if reason is None:
        assert got == {"verdict": "accept", "reads": 6, "windows": 1, "writes": 6}
    else:
        assert got["verdict"] == "reject" and got["reason"] == reason, got


def _at(log, op, k=0):
    """The index of the `k`-th event with `op`."""
    return [i for i, e in enumerate(log["events"]) if e["op"] == op][k]


def _flip(h: str) -> str:
    return f"{int(h[0], 16) ^ 1:x}{h[1:]}"


def _edit(log, f):
    log = copy.deepcopy(log)
    f(log, log["events"])
    return log


EDITS = {
    "page": lambda log, ev: ev[_at(log, "read")].update(page=log["pages"]),
    "version": lambda log, ev: ev.insert(_at(log, "snapshot"), {"op": "commit", "page": 0, "version": 9,
                                                                   "c": "00" * 64}),
    "recommit": lambda log, ev: ev.insert(_at(log, "snapshot"), dict(ev[_at(log, "commit")])),
    "coin": lambda log, ev: ev.insert(_at(log, "snapshot"), dict(ev[_at(log, "coin")])),
    "early": lambda log, ev: ev.insert(_at(log, "snapshot"), {"op": "store", "page": 0, "version": 7, "blocks": []}),
    "restore": lambda log, ev: ev.insert(_at(log, "snapshot"), dict(ev[_at(log, "store")])),
    "decode": lambda log, ev: ev[_at(log, "store")]["blocks"].__setitem__(0, _flip(ev[_at(log, "store")]["blocks"][0])),
    "snapshot": lambda log, ev: ev[_at(log, "snapshot")].update(root=_flip(ev[_at(log, "snapshot")]["root"])),
    "order": lambda log, ev: ev.insert(0, dict(ev[_at(log, "read")])),
    "late": lambda log, ev: ev[_at(log, "read")].update(ns=log["limit_ns"] + 1),
    "wrong": lambda log, ev: ev[_at(log, "read")].update(answer=_flip(ev[_at(log, "read")]["answer"])),
    "reads": lambda log, ev: ev.__delitem__(slice(_at(log, "read"), None)),
}


def test_every_reason_is_reached():
    assert set(EDITS) | {"stale"} == set(O.REASONS)


@pytest.mark.parametrize("reason", sorted(EDITS))
def test_an_edit_is_rejected_for_its_reason(reason):
    got = O.verdict(_edit(C.honest(4), EDITS[reason]))
    assert got["verdict"] == "reject" and got["reason"] == reason, got


MALFORMED = {
    "salt length": lambda log, ev: log.update(salt="00" * 23),
    "uppercase hex": lambda log, ev: ev[_at(log, "commit")].update(c=ev[_at(log, "commit")]["c"].upper()),
    "odd hex": lambda log, ev: ev[_at(log, "snapshot")].update(root="0"),
    "coin length": lambda log, ev: ev[_at(log, "coin")].update(coin="00" * 31),
    "unknown op": lambda log, ev: ev.insert(0, {"op": "erase"}),
    "missing field": lambda log, ev: ev[_at(log, "read")].pop("ns"),
    "negative": lambda log, ev: ev[_at(log, "read")].update(block=-1),
    "bool": lambda log, ev: log.update(pages=True),
    "no pages": lambda log, ev: log.update(pages=0),
    "too many blocks": lambda log, ev: log.update(blocks=O.MAX + 1),
    "not an object": lambda log, ev: ev.insert(0, ["commit"]),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_a_malformed_log_is_refused(name):
    with pytest.raises(ValueError):
        O.verdict(_edit(C.honest(5), MALFORMED[name]))


def test_a_window_writes_during_its_reads():
    """`window`: every page preallocated, the content's first half written before the snapshot and the rest between its
    reads, which the snapshot answers; each page decodes to its last write, and the log is accepted."""
    rng = random.Random(2)
    mc, size = C.challenger(rng, 3, 4), O.params(4).segment_bytes
    content = rng.randbytes(7 * size + 5)
    store, latest = O.window(mc, content, 5)
    ops = [e["op"] for e in mc.log()["events"]]
    snap = ops.index("snapshot")
    assert (ops[:snap].count("store"), ops[snap:].count("store")) == (3 + 4, 4)
    assert latest[1] == content[7 * size:].ljust(size, b"\0")
    assert [store.plaintext(p) for p in range(3)] == latest
    assert O.verdict(mc.log()) == {"verdict": "accept", "reads": 5, "windows": 1, "writes": 11}


def test_the_space_legs_kv_window():
    """`period.kv_window` without Lean: the window accepted, every page decoded on use, its writes counted."""
    rng = random.Random(3)
    got = period.kv_window(rng.randbytes(100), C.SALT, 2, 3, 4, coins=rng.randbytes,
                           clock=itertools.count(0, C.STEP_NS).__next__)
    assert got["verdict"] == {"verdict": "accept", "reads": 4, "windows": 1, "writes": 8}
    assert got["lean"] is None and got["writes"] == 8 and got["page_bytes"] == 18
