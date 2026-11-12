"""The prover's history table (`history.HistoryTable`) against the definition it mirrors (`history.root`, `check`): its root
after every append is the root over its rows' strings from scratch, which `check` accepts; each written rank opens to its
row and salt, and each rank past the length to the placeholder's zero row under the zero salt, under the table read's own
reference (`gates.embedding.reference`); and it refuses a row past its capacity, a salt used twice and a row of another
width."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.commitments import history as H
from verity.ml.commitments.gates import embedding as E


def table(capacity: int = 16, words: int = 3) -> H.HistoryTable:
    return H.HistoryTable("program", "latent/0", words, capacity)


def rows(k: int, words: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, (k, words), dtype=np.uint16)


def opened(t: H.HistoryTable, rank: int, row, salt: bytes) -> int:
    p = t.path(rank)
    return E.reference(t.leaves, rank, t.root, t.domain.domain_id, [int(w) for w in row], salt,
                       [p[k:k + 64] for k in range(0, len(p), 64)])[1]


def test_its_root_after_every_append_is_the_root_over_its_strings():
    t, ph = table(), H.placeholder(3)
    assert t.root == H.root(t.domain, [], ph)
    for k, seed in ((3, 1), (1, 2), (0, 3), (5, 4), (7, 5)):
        got = t.append(rows(k, seed=seed))
        assert len(got) == k and len(t.rows) == got.stop
        assert t.bcs == [H.string(r, 3, s) for r, s in zip(t.rows, t.salts)]
        assert t.root == H.root(t.domain, t.bcs, ph)
        H.check("program", "latent/0", 3, t.bcs, t.root, 16)
    assert t.entry() == {"value": "latent/0", "root": t.root.hex(), "leaves": 16, "words": 3,
                         "domain": H.domain("program", "latent/0", 3, 16).domain_id.hex()}


def test_a_written_rank_opens_to_its_row_and_a_later_one_to_the_placeholder():
    t = table()
    t.append(rows(6))
    assert all(opened(t, r, t.rows[r], t.salts[r]) == 1 for r in range(6))
    assert all(opened(t, r, [0, 0, 0], bytes(E.SALT_BYTES)) == 1 for r in range(6, 16))
    assert opened(t, 2, t.rows[3], t.salts[3]) == 0
    assert opened(t, 6, t.rows[0], t.salts[0]) == 0


def test_the_root_binds_its_program_table_order_and_capacity():
    salts = [bytes([i]) * E.SALT_BYTES for i in range(1, 5)]
    r = rows(4)
    base = table()
    base.append(r, salts)
    for other in (H.HistoryTable("other", "latent/0", 3, 16), H.HistoryTable("program", "latent/1", 3, 16),
                  H.HistoryTable("program", "latent/0", 3, 32)):
        other.append(r, salts)
        assert other.root != base.root
    swapped = table()
    swapped.append(r[[1, 0, 2, 3]], [salts[1], salts[0], *salts[2:]])
    assert swapped.root != base.root


def test_it_refuses_a_row_past_its_capacity_a_reused_salt_and_another_width():
    t = table(capacity=4)
    t.append(rows(3))
    with pytest.raises(ValueError, match="past the table's capacity"):
        t.append(rows(2))
    with pytest.raises(ValueError, match="never reused"):
        t.append(rows(1), [t.salts[0]])
    with pytest.raises(ValueError, match="never reused"):
        table().append(rows(2), [bytes(E.SALT_BYTES)] * 2)
    with pytest.raises(ValueError, match="u16 words"):
        t.append(rows(1, words=4))
    with pytest.raises(ValueError, match="power of two"):
        H.HistoryTable("program", "latent/0", 3, 12)
    assert len(t.rows) == 3 and t.root == H.root(t.domain, t.bcs, H.placeholder(3))


def test_at_the_pinned_capacity_its_root_is_the_definitions():
    t = H.HistoryTable("program", "latent/0", 576)
    t.append(rows(5, words=576))
    assert t.leaves == H.CAPACITY and len(t.path(0)) == 17 * H.DIGEST_BYTES
    H.check("program", "latent/0", 576, t.bcs, t.root)
