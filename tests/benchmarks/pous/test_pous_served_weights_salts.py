"""PoUS's served weights take their row salts from a salt file of fresh operating-system bytes, never from a key: a
tensor committed whole and its root computed in chunks agree under one salt file, the interface's fixture is what its
stored public test salts give, and two draws of the salts share no row."""
from __future__ import annotations

import json
from pathlib import Path

from verity.commitments import salt_file
from verity.protocols.resource_accounting.memory.pous.prover import served_weights as S

FIXTURES = Path(__file__).parent / "fixtures"
TEST_SALTS = FIXTURES / "weights-v1-tiny.salts"


def test_a_tensor_committed_whole_has_the_root_its_chunks_give_under_one_salt_file():
    want = S.root(S.TINY, S.TEST_SEED, TEST_SALTS, S.TEST_PROGRAM, ["l0.wq"], chunk_bytes=256)
    assert S.commit(S.TINY, S.TEST_SEED, TEST_SALTS, S.TEST_PROGRAM, "l0.wq").levels[-1][0] == want["l0.wq"]


def test_the_fixture_is_what_its_stored_test_salts_give():
    got = json.loads(S.dumps(S.fixture(TEST_SALTS)))
    assert got == json.loads((FIXTURES / "weights-v1.json").read_text())
    assert got["salts"]["note"].startswith("public test data")


def test_two_draws_of_the_salts_share_no_row(tmp_path):
    names = [n for n, _ in S.reader(S.TINY, S.TEST_PROGRAM).widths]
    a, b = salt_file.write(tmp_path / "a.salts", S.draw_salts(S.TINY, names)), tmp_path / "b.salts"
    salt_file.write(b, S.draw_salts(S.TINY, names))
    assert salt_file.params(a) == {n: S.tensor(S.TINY, n).leaves for n in names}
    rows = [salt_file.read(p, n)[192 * j:192 * (j + 1)] for p in (a, b) for n in names
            for j in range(S.tensor(S.TINY, n).leaves)]
    assert len(set(rows)) == len(rows)
