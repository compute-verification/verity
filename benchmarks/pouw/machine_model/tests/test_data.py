"""data/ is well formed: every census line names a row, an epoch span and one value; vectors and attacks name what exists."""
from __future__ import annotations

import csv
import json
import re

import pytest

from verity_machine_model.model import (DATA, STATUSES, all_attacks, attacks, census, entries, holds, model, rows,
                                        u_of, vector, vectors)

M = model()
ROWS = rows()
CITE = re.compile(r"(r\d{8}-\d{6}-[0-9a-f]{4}|note:\d{8}T\d{4}Z-[a-z0-9-]+)")


def test_epochs_are_vectors():
    names = set(vectors())
    assert len(set(M.epochs)) == len(M.epochs)
    assert set(M.epochs) <= names, "an epoch is named after the vector that introduced it"
    assert {vector(n).epoch for n in names} == set(M.epochs)
    assert M.current in names


def test_rows_are_well_formed():
    assert all(r.status in STATUSES for r in ROWS.values())
    assert all(r.runs and all(CITE.fullmatch(x) for x in r.runs) for r in ROWS.values())


def test_census_lines_are_well_formed():
    with open(DATA / "census.csv", newline="") as f:
        lines = list(csv.DictReader(f))
    for d in lines:
        where = f"census.csv: {d['kind']} on {d['row']}"
        assert d["row"] in ROWS, f"{where}: add the row to rows.csv"
        assert sum(d[c] != "" for c in ("rate", "cycles", "u")) == 1, f"{where}: give one of rate, cycles and u"
        assert d["noise"] == "" or d["rate"] != "", f"{where}: noise derates a rate only"
        for e in (d["since"], d["until"]):
            assert e == "" or e in M.epochs, f"{where}: {e!r} is not in model.toml's epochs"
        if d["since"] and d["until"]:
            assert M.epochs.index(d["since"]) <= M.epochs.index(d["until"]), where
        assert d["runs"] and all(CITE.fullmatch(x) for x in d["runs"].split()), f"{where}: cite its runs or note"
    assert len(entries(M)) == len(lines)


@pytest.mark.parametrize("epoch", M.epochs)
def test_one_value_per_kind_and_row_at_each_epoch(epoch):
    c = census(M, epoch)   # raises on two lines for one kind and row
    assert all(u > 0 for v in c.values() for u in v.values())
    assert set(M.honest) <= set(c) and "MMA" in c


def test_u_of():
    assert u_of(M.cycle_u, "", "", "1", "") == M.cycle_u
    assert u_of(M.cycle_u, "2", "", "", "") == M.cycle_u * 4 / 2
    assert u_of(M.cycle_u, "2", "0.02", "", "") == M.cycle_u * 4 / (2 * 1.02)
    assert u_of(M.cycle_u, "", "", "", "123.5") == 123.5
    with pytest.raises(ValueError, match="exactly one of rate, cycles and u"):
        u_of(M.cycle_u, "2", "", "1", "")
    with pytest.raises(ValueError, match="noise derates a rate only"):
        u_of(M.cycle_u, "", "0.02", "1", "")


@pytest.mark.parametrize("name", vectors())
def test_vectors_name_what_exists(name):
    v = vector(name)
    c = census(M, v.epoch)
    assert len(set(v.rows)) == len(v.rows) and set(v.rows) <= set(ROWS)
    assert {k for k, _ in v.credits} <= set(c)
    assert not v.superseded_by or v.superseded_by in vectors()


def test_attacks_are_a_file_per_family():
    for p in sorted((DATA / "attacks").glob("*.jsonl")):
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert len({r["family"] for r in recs}) == 1, p.name
        assert p.stem == re.sub(r"[^a-z0-9]+", "-", recs[0]["family"].lower()).strip("-"), p.name
        for r in recs:
            assert set(r) <= {"design", "family", "name", "program", "on", "since", "until"}, p.name
            assert all(e in M.epochs for e in (r.get("since"), r.get("until")) if e), p.name


@pytest.mark.parametrize("epoch", M.epochs)
def test_attacks_price_known_kinds(epoch):
    c, a = census(M, epoch), attacks(M, epoch)
    assert len({(x.design, x.family, x.name) for x in a}) == len(a)
    assert all(set(x.program) <= set(c) and all(v >= 0 for v in x.program.values()) for x in a)
    assert any(x.design == M.adopted for x in a)


def test_holds():
    assert holds(M, "", "", "VF") and holds(M, "VF3e", "", "VF3j") and not holds(M, "VF3e", "", "VF3d")
    assert not holds(M, "VF3j-8", "VF3j-8", "VF3j") and holds(M, "", "VF", "VF") and not holds(M, "", "VF", "VF2")
    assert len(all_attacks()) == len(attacks(M, M.epochs[-1]))
