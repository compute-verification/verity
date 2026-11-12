"""The committer's history tables (`commit.history_tables`): a table's root after each window's appends is the root the
verifier recomputes from the strings the window commits, across a cut that continues the table from its edge; the window
can commit those rows from the table's strings with the same root as hashing them in full; tables append from their edge
only, are kept per request until it ends, and the switch is off by default."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.commitments import history as HT
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG
from verity.commitments.windows import rows as CS
from verity_vllm.commit import history_tables as H

PORT = "DsaLatentRow_v1/c0/o0"
PROGRAM = "ab" * 64


def _rows(seed: int, k: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, (k, H.WORDS), dtype=np.uint16)


def _committed(rows: np.ndarray, salts: bytes, strings: bytes | None = None) -> tuple[bytes, list[bytes]]:
    """The writer port's root and its rows' committed strings as a window commits them (`rows.commit_ports`): hashed in full
    under ``salts``, or from given ``strings``."""
    dom = CommitmentDomain(hashlib.sha512(PORT.encode()).digest(), -1, RangeIndexedDomain(0, len(rows)), "sha512")
    if strings is None:
        roots, bc = CS.commit_ports({PORT: dom}, {PORT: rows}, {}, {PORT: salts}, schemas={PORT: SEG})
    else:
        roots, bc = CS.commit_ports({PORT: dom}, {}, {}, {}, schemas={PORT: SEG}, strings={PORT: strings})
    s = bc[PORT]
    return roots[PORT], [s[i:i + CS.BC_BYTES] for i in range(0, len(s), CS.BC_BYTES)]


def test_a_tables_root_after_each_windows_appends_is_the_root_over_its_committed_strings():
    """Window 1 appends request r0's first five latent rows at layer 3, window 2 (the request open at its edge) the next four
    from the edge: after each, the table's registered root is `history.check`'s recomputation over the strings the windows
    committed, in position order, at the full 2^17 capacity; a string the window didn't commit fails it. The window commits
    the rows from the table's strings with the root hashing them in full under the table's salts makes."""
    hs = H.Histories(PROGRAM)
    one, two = _rows(1, 5), _rows(2, 4)
    assert list(hs.append("r0", 3, 0, one)) == [0, 1, 2, 3, 4] and hs.edge("r0", 3) == 5
    t = hs.table("r0", 3)
    root1, strings1 = _committed(one, hs.salts("r0", 3, range(5)))
    assert b"".join(strings1) == hs.strings("r0", 3, range(5)) and H.check(t, strings1)
    assert _committed(one, b"", hs.strings("r0", 3, range(5)))[0] == root1
    entry1 = hs.entries()[("r0", 3)]
    assert entry1 == {"value": H.name("r0", 3), "root": t.root.hex(), "leaves": HT.CAPACITY, "words": H.WORDS,
                      "domain": HT.domain(PROGRAM, H.name("r0", 3), H.WORDS).domain_id.hex()}
    assert list(hs.append("r0", 3, 5, two)) == [5, 6, 7, 8]
    _root2, strings2 = _committed(two, hs.salts("r0", 3, range(5, 9)))
    assert H.check(t, strings1 + strings2) and not H.check(t, strings1)
    assert hs.entries()[("r0", 3)]["root"] != entry1["root"]
    bad = bytearray(strings2[1])
    bad[0] ^= 1
    assert not H.check(t, strings1 + [strings2[0], bytes(bad), *strings2[2:]])
    assert not H.check(t, strings1 + strings2[::-1])


def test_a_table_appends_in_position_order_from_its_edge_once():
    """A table is append-only from its edge: a gap and a rewrite of a held position are refused, as are a reused salt and
    rows past its capacity (`history.HistoryTable`); each (request, layer) has its own domain."""
    hs = H.Histories(PROGRAM, capacity=8)
    hs.append("r0", 0, 0, _rows(3, 3))
    for start in (4, 2, 0):
        with pytest.raises(ValueError, match="appends at its edge 3"):
            hs.append("r0", 0, start, _rows(4, 1))
    with pytest.raises(ValueError, match="salt is never reused"):
        hs.append("r0", 0, 3, _rows(5, 1), [hs.table("r0", 0).salts[0]])
    with pytest.raises(ValueError, match="capacity"):
        hs.append("r0", 0, 3, _rows(6, 6))
    assert hs.edge("r0", 0) == 3 and hs.edge("r1", 0) == 0
    hs.append("r1", 0, 0, _rows(7, 1))
    hs.append("r0", 1, 0, _rows(8, 1))
    doms = {k: e["domain"] for k, e in hs.entries().items()}
    assert len(set(doms.values())) == 3


def test_tables_are_retained_per_request_until_it_ends():
    """Every row and salt of a request's tables is held, 1,344 bytes a row a layer, until the request ends; ending one
    request releases only its tables."""
    assert H.ROW_BYTES == 1344
    hs = H.Histories(PROGRAM, capacity=16)
    hs.append("r0", 0, 0, _rows(9, 5))
    hs.append("r0", 1, 0, _rows(10, 5))
    hs.append("r1", 0, 0, _rows(11, 2))
    assert hs.retained() == 12 * 1344 and hs.retained("r0") == 10 * 1344 and hs.requests() == ["r0", "r1"]
    assert hs.end("r0") == 10 * 1344
    assert hs.requests() == ["r1"] and list(hs.entries()) == [("r1", 0)] and hs.edge("r0", 0) == 0


def test_a_committed_window_releases_every_request_it_doesnt_name_open():
    """Once a window is committed or refused, its requests' tables are released but those it names open, which the next
    window appends to from their edge; a window of no tables releases nothing."""
    hs = H.Histories(PROGRAM, capacity=16)
    for r in ("r0", "r1", "r2"):
        hs.append(r, 0, 0, _rows(12, 2))
    assert H.release(hs, ["r1"]) == 4 * 1344 and hs.requests() == ["r1"] and hs.edge("r1", 0) == 2
    assert H.release(hs) == 2 * 1344 and hs.requests() == [] and H.release(None) == 0


def test_the_switch_is_off_by_default():
    """`SWITCH` is off unless the registration sets it."""
    assert not H.on(None) and not H.on({}) and H.on({H.SWITCH: True})


def test_a_run_keeps_one_set_of_tables_across_its_windows_and_a_window_of_none_takes_none():
    """A serving process keeps one `Histories` per run's committer program, refused without one, and none for a window
    without the switch."""
    assert H.of_request({"program": PROGRAM}, False) is None
    first = H.of_request({"program": PROGRAM * 2}, True)
    assert first is H.of_request({"program": PROGRAM * 2}, True) and first.program == PROGRAM * 2
    assert H.of_request({"program": PROGRAM * 3}, True) is not first
    with pytest.raises(ValueError, match="committer"):
        H.of_request({}, True)
