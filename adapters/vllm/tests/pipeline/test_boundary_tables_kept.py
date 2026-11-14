"""`boundary_tables.Kept`: what rank 0 keeps of each table on the node, what it refuses to keep or to read back, a served
table as `load` gives it (`Served`), and the private directory's refusal."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from verity.commitments import history as HT

from verity_vllm.pipeline import boundary_tables as BT

RUN, PROGRAM, LEAVES, WORDS = "a" * 128, "cd" * 64, 16, 4
#: a kept table's window, run, class, slot and layer
AT = (0, RUN, 4, 8, 1, 3)


def kept(tmp_path: Path) -> tuple[BT.Kept, HT.HistoryTable]:
    """A table of three rows, kept at `AT` under ``tmp_path``/tables."""
    t = HT.HistoryTable(PROGRAM, BT.name(*AT[1:]), WORDS, LEAVES)
    t.append(np.arange(3 * WORDS, dtype=np.uint16).reshape(3, WORDS))
    k = BT.Kept(tmp_path / "tables")
    k.write(*AT, PROGRAM, t)
    return k, t


def test_round_2_reads_a_span_from_its_first_position_on(tmp_path):
    k, t = kept(tmp_path)
    got = k.span(*AT, range(0, 3))
    assert got["rows"].tolist() == np.stack(t.rows).tolist() and got["salts"] == t.salts
    assert got["strings"] == t.bcs and got["entry"] == t.entry() and got["positions"] == range(0, 3)
    assert k.span(*AT, range(0, 0))["strings"] == []


@pytest.mark.parametrize("suffix", ["rows", "salts"])
def test_a_kept_table_whose_rows_or_salts_lost_bytes_is_not_read(tmp_path, suffix):
    k, t = kept(tmp_path)
    (f,) = (tmp_path / "tables").rglob(f"layer-3.{suffix}")
    f.write_bytes(f.read_bytes()[:-2])
    rows, salts = 2 * 3 * WORDS, 3 * len(t.salts[0])
    held = {"rows": (rows - 2, salts), "salts": (rows, salts - 2)}[suffix]
    why = f"{t.value}: its files hold {held[0]} row bytes and {held[1]} salt bytes, not 3 rows' {rows} and {salts}"
    with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
        k.rows(*AT)


@pytest.mark.parametrize("salts", [[b"\1" * 8], [b"\1" * 8, b"\2" * 9]])
def test_a_table_is_kept_only_with_one_salt_of_one_length_a_row(tmp_path, salts):
    t = NS(value="t", words=WORDS, rows=np.zeros((2, WORDS), np.uint16), salts=salts, entry=dict)
    with pytest.raises(ValueError, match=f"^t: one salt of one length a row, not {len(salts)} for 2 rows$"):
        BT.Kept(tmp_path).write(*AT, PROGRAM, t)
    assert not list(tmp_path.rglob("layer-*"))


def test_a_served_table_is_loaded_only_when_one_window_kept_its_name(tmp_path):
    k, t = kept(tmp_path)
    assert k.load(t.value).entry() == t.entry()
    k.write(1, *AT[1:], PROGRAM, t)
    for name, n in ((t.value, 2), (BT.name(RUN, 4, 8, 0, 3), 0)):
        why = f"rank 0 kept {n} tables named {name} under {tmp_path / 'tables'}, not one"
        with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
            k.load(name)


def test_a_served_table_prints_neither_its_rows_nor_its_salts(tmp_path):
    k, t = kept(tmp_path)
    s, e = k.load(t.value), t.entry()
    assert s.rows.tolist() == np.stack(t.rows).tolist() and s.salts == b"".join(t.salts) and s.strings() == t.bcs
    assert repr(s) == (f"Served(value={t.value!r}, program={PROGRAM!r}, root={e['root']!r}, leaves={LEAVES}, words={WORDS}, "
                       f"domain={e['domain']!r})")


def test_a_private_directory_inside_what_leaves_the_node_is_refused_with_both_paths(tmp_path):
    run = tmp_path.resolve() / "run"
    why = (f"the history tables' private directory {run / 'private'} is inside {run}, which leaves the node: put it "
           f"outside every run directory")
    with pytest.raises(ValueError, match=f"^{re.escape(why)}$"):
        BT.private_root(run / "private", None, run)
    assert BT.private_root(tmp_path / "private", None, run) == tmp_path.resolve() / "private"
