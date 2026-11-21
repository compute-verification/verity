"""PoUS's served weights take their row salts from a salt file of fresh operating-system bytes, never from a key: two draws
of the salts share no row."""
from __future__ import annotations

from verity.commitments import salt_file
from verity.protocols.accounting.memory.pous.prover import served_weights as S


def test_two_draws_of_the_salts_share_no_row(tmp_path):
    names = [n for n, _ in S.reader(S.TINY, S.TEST_PROGRAM).widths]
    a, b = salt_file.write(tmp_path / "a.salts", S.draw_salts(S.TINY, names)), tmp_path / "b.salts"
    salt_file.write(b, S.draw_salts(S.TINY, names))
    assert salt_file.params(a) == {n: S.tensor(S.TINY, n).leaves for n in names}
    rows = [salt_file.read(p, n)[192 * j:192 * (j + 1)] for p in (a, b) for n in names
            for j in range(S.tensor(S.TINY, n).leaves)]
    assert len(set(rows)) == len(rows)
