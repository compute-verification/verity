"""`rows.py --ctas-at`: a row count runs at the receipt CTA counts its word names, in the word's order, and at
`--sketch-ctas` alone where no word names it; a word for a row count that doesn't run, or one whose counts aren't
numbers, ends the run saying what a word is. `rows.reference`, the whole-store check's receipts, is the reference
receipter's at both row counts, at the int8 extremes and at any first block. `rows.whole_store` asks a leaf pass
for its receipts for its one pass and puts the store's mode back, and leaves `rs_pass`'s mode alone. Every path of the tree `rows.sh` names is
one the tree has, so the build a card waits on doesn't end at a moved file; the base tree it compiles links every
directory `session.cu`'s includes reach, and its build's checks run on the Python its steps run on."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[3] / "benchmarks" / "remote_session")))

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


class _Stop(Exception):
    pass


class _Store:
    """The store's pass mode as `whole_store` sees it, and the mode its pass ran under; the pass stops the check."""

    def __init__(self, leaf_ctas: int, leaf_receipts: bool = False):
        self.leaf_ctas, self.leaf_receipts, self.poison, self.ran = leaf_ctas, leaf_receipts, False, None

    def pass_mode(self, leaf_ctas: int = 0, receipts: bool = False) -> None:
        self.leaf_ctas, self.leaf_receipts = leaf_ctas, receipts

    def run_pass(self, seed):
        self.ran = (self.leaf_ctas, self.leaf_receipts, self.poison)
        raise _Stop


class _Backend:
    def __init__(self, store: _Store):
        self.store = store


@pytest.mark.parametrize("leaf, kept, during", [(188, False, (188, True, True)), (160, True, (160, True, True)),
                                                (-1, False, (-1, True, True)), (0, False, (0, False, True))])
def test_the_whole_store_pass_asks_a_leaf_pass_for_its_receipts_and_puts_the_mode_back(leaf, kept, during):
    s = _Store(leaf, kept)
    with pytest.raises(_Stop):
        rows.whole_store(_Backend(s), 64, np.random.default_rng(0))
    assert s.ran == during
    assert (s.leaf_ctas, s.leaf_receipts, s.poison) == (leaf, kept, False)


def test_every_path_rows_sh_names_is_in_the_tree():
    script = (ROOT / "benchmarks/remote_session/rows.sh").read_text()
    names = dict(re.findall(r"\b(RS|PR)=(\S+)", script))
    assert names == {"RS": "benchmarks/remote_session", "PR": "verity/protocols/resource_accounting/memory/poub/prover"}
    paths = {m.group(1).replace("$RS", names["RS"]).replace("$PR", names["PR"])
             for m in re.finditer(r"((?:\$RS|\$PR|verity|benchmarks|tools)/[\w./-]+)", script)}
    paths = {p.rstrip(".") for p in paths if ".lake/" not in p}
    assert {"verity/protocols/resource_accounting/memory/poub/prover/remote_session/session.cu",
            "verity/protocols/resource_accounting/memory/poub/prover/remote_session/prover.py",
            "verity/protocols/resource_accounting/memory/poub/verifier/remote_session/verifier_process.py"} <= paths
    assert sorted(p for p in paths if not (ROOT / p).exists()) == []


def test_the_base_tree_links_every_directory_session_cu_includes_reach():
    script = (ROOT / "benchmarks/remote_session/rows.sh").read_text()
    pr = "verity/protocols/resource_accounting/memory/poub/prover"
    links = {m.group(1).replace("$PR", pr) for m in re.finditer(r'ln -s "\$TREE/([^"]+)" "\$B/base/\1"', script)}
    assert "verity/circuits" in links
    reached, todo = set(), [ROOT / pr / "remote_session/session.cu"]
    while todo:
        f = todo.pop()
        for name in re.findall(r'^#include "([^"]+)"', f.read_text(), re.M):
            p = (f.parent / name).resolve()
            assert p.is_file(), f"{f.relative_to(ROOT)} includes {name}, which is no file of the tree"
            if p not in reached:
                reached.add(p)
                todo.append(p)
    outside = sorted(str(p.relative_to(ROOT)) for p in reached if not any(p.is_relative_to(ROOT / d) for d in links))
    assert outside == [], f"rows.sh's base tree links {sorted(links)}: link the directories of {outside} too"


def test_the_build_checks_run_on_the_python_the_steps_run_on():
    lines = (ROOT / "benchmarks/remote_session/rows.sh").read_text().splitlines()
    runs = [i for i, line in enumerate(lines) if re.search(r"\$RS/\w+\.py", line) and not line.lstrip().startswith("#")]
    py = [i for i, line in enumerate(lines) if line.startswith("export PY=")]
    assert len(runs) >= 4 and len(py) == 1 and py[0] < min(runs)
    assert [lines[i] for i in runs if re.search(r"\bpython3? \$RS/", lines[i])] == []
