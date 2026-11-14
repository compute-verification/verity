"""`rows.py --ctas-at`: a row count runs at the receipt CTA counts its word names, in the word's order, and at
`--sketch-ctas` alone where no word names it; a word for a row count that doesn't run, or one whose counts aren't
numbers, ends the run saying what a word is. `rows.reference`, the whole-store check's receipts, is the reference
receipter's at both row counts, at the int8 extremes and at any first block. Every path of the tree `rows.sh` names is
one the tree has, so the build a card waits on doesn't end at a moved file."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rows
from rows import rb

ROOT = Path(__file__).resolve().parents[3]


def test_a_row_count_takes_its_words_counts_in_order_and_the_default_elsewhere():
    assert rows.ctas_at(["64=96,128,264"], [16, 64, 16, 64], 128) == {16: [128], 64: [96, 128, 264]}
    assert rows.ctas_at([], [64, 16], 0) == {64: [0], 16: [0]}
    assert rows.ctas_at(["64=132", "16=128"], [64, 16], 64) == {64: [132], 16: [128]}


@pytest.mark.parametrize("word", ["48=128", "64", "64=", "64=128,", "64=a", "x=128", "64=-1", "64=128+p0"])
def test_a_word_that_names_no_running_row_count_or_no_counts_ends_the_run(word):
    with pytest.raises(SystemExit, match="ROWS=N"):
        rows.ctas_at([word], [64], 128)


@pytest.mark.parametrize("k", rb.ROWS)
def test_the_whole_store_reference_is_the_reference_receipters(k):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(k)
    seed = rb.seed(rng.bytes(rb.CHALLENGE_BYTES), 0, bytes(rb.DIGEST_BYTES))
    blocks = [rng.bytes(rb.BLOCK_BYTES) for _ in range(5)]
    blocks += [bytes([b]) * rb.BLOCK_BYTES for b in (0x80, 0x7F, 0xFF, 0xA5)]
    # every byte the int8 extreme whose product with row 0's sign is positive: the largest σ a row can hold
    row0 = np.unpackbits(np.frombuffer(rb.sign_bits(seed, k)[0], np.uint8), bitorder="little")
    blocks.append(np.where(row0 == 1, 0x80, 0x7F).astype(np.uint8).tobytes())
    j0 = (1 << 40) + 3
    got = rows.reference(torch, seed, k)(torch.frombuffer(bytearray(b"".join(blocks)), dtype=torch.uint8), j0)
    r = rb.receipter(rb.SKETCH_SHA512, seed, k)
    assert got == b"".join(r(j0 + i, b) for i, b in enumerate(blocks))


def test_every_path_rows_sh_names_is_in_the_tree():
    script = (ROOT / "benchmarks/remote_session/rows.sh").read_text()
    names = dict(re.findall(r"\b(RS|PR)=(\S+)", script))
    assert names == {"RS": "benchmarks/remote_session", "PR": "verity/protocols/accounting/poub/prover"}
    paths = {m.group(1).replace("$RS", names["RS"]).replace("$PR", names["PR"])
             for m in re.finditer(r"((?:\$RS|\$PR|verity|benchmarks|tools)/[\w./-]+)", script)}
    paths = {p.rstrip(".") for p in paths if ".lake/" not in p}
    assert {"verity/protocols/accounting/poub/prover/remote_session/session.cu",
            "verity/protocols/accounting/poub/prover/remote_session/prover.py",
            "verity/protocols/accounting/poub/verifier/remote_session/verifier_process.py"} <= paths
    assert sorted(p for p in paths if not (ROOT / p).exists()) == []
