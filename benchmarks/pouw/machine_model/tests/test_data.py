"""data/ is well formed: every census line names a row, an epoch span and one value; vectors and attacks name what exists."""
from __future__ import annotations

import csv
import json
import re

import pytest

from verity_machine_model.model import (CLAIMS, COPIES, DATA, DEVICES, STATUSES, all_attacks, attacks, census,
                                        census_files, census_lines, entries, floors, holds, model, rows, run_lists,
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


CENSUS_COLUMNS = ("kind", "row", "rate", "noise", "cycles", "u", "since", "until", "runs")
BLOB_LIMIT = 256 * 1024   # tests/test_repository.py's cap on a tracked file


def test_census_is_a_file_per_epoch():
    """data/census/ is EPOCH.csv for each epoch that brought lines, each line's since its file's, and runs.toml; no file
    comes near the repository's cap, so the census grows by files, not by one file growing."""
    files = census_files(M)
    assert {p.name for p in (DATA / "census").iterdir()} == {p.name for p in files} | {"runs.toml"}, (
        "data/census/ holds EPOCH.csv for epochs in model.toml's epochs, and runs.toml")
    named = run_lists()
    used = set()
    for p in files:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            assert tuple(r.fieldnames or ()) == CENSUS_COLUMNS, p.name
            for d in r:
                assert d["since"] == p.stem, f"census/{p.name}: {d['kind']} on {d['row']} is from {d['since']!r}"
                used |= {c[1:] for c in d["runs"].split() if c.startswith("@")}
    for p in [*files, DATA / "census" / "runs.toml"]:
        assert p.stat().st_size <= BLOB_LIMIT // 4, (
            f"census/{p.name} is {p.stat().st_size} bytes, past a quarter of the {BLOB_LIMIT}-byte cap: name a run list "
            f"its lines share in census/runs.toml and cite it as @NAME")
    assert used == set(named), f"runs.toml: {sorted(set(named) - used)} cited by no line; {sorted(used - set(named))} not in it"
    assert all(xs and all(CITE.fullmatch(x) for x in xs) for xs in named.values())


def test_census_lines_are_well_formed():
    lines = census_lines(M)
    named = run_lists()
    for d in lines:
        where = f"census/{d['since']}.csv: {d['kind']} on {d['row']}"
        assert d["row"] in ROWS, f"{where}: add the row to rows.csv"
        assert sum(d[c] != "" for c in ("rate", "cycles", "u")) == 1, f"{where}: give one of rate, cycles and u"
        assert d["noise"] == "" or d["rate"] != "", f"{where}: noise derates a rate only"
        for e in (d["since"], d["until"]):
            assert e == "" or e in M.epochs, f"{where}: {e!r} is not in model.toml's epochs"
        if d["since"] and d["until"]:
            assert M.epochs.index(d["since"]) <= M.epochs.index(d["until"]), where
        assert d["runs"] and all(CITE.fullmatch(x) or x.startswith("@") and x[1:] in named for x in d["runs"].split()), (
            f"{where}: cite its runs or note, or a run list of census/runs.toml as @NAME")
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


def test_floors_name_census_kinds():
    fs, c = floors(), census(M, M.epochs[-1])
    assert {f.claim for f in fs} == set(CLAIMS)
    assert len({(f.kind, f.claim) for f in fs}) == len(fs)
    assert all(f.kind in c and f.words > 0 and f.floor > 0 for f in fs), [f.kind for f in fs if f.kind not in c]


# The kinds that write memory and no register. A word they write reaches registers only through a load or an atomic's
# returned word, each a copy route π_c ≥ 16 prices itself, so its claim needn't list them.
MEMORY_ONLY = {"STS128", "STSM_X1", "STSM_X2", "STSM_X4", "STSM_X1T", "STSM_X2T", "STSM_X4T", "STSM8T_X1", "STSM8T_X2",
               "STSM8T_X4", "RED32", "RED64", "REDV", "REDG32", "REDG64", "REDAS32", "REDAS64", "SURED", "TMAred512",
               "TMA512", "UTMA", "UTMA8", "UTMA512", "UTMA1K", "UTMA2K", "UTMA4K", "UTMA8K", "UTMA16K", "UTMA32K",
               "UTMA48K", "UTMA64K", "UTMABIG", "G4_16", "G4_32", "G4_256", "G4_1K", "LDGSTS16", "UBLKCP4K", "DSMST4P"}


def test_every_kind_is_a_copy_route_or_writes_memory_only():
    """π_c ≥ 16 reads every kind that writes registers, so it holds for every copy route there is."""
    kinds = set(census(M, M.epochs[-1]))
    copies = {f.kind for f in floors() if f.claim == COPIES}
    assert not copies & MEMORY_ONLY and MEMORY_ONLY <= kinds
    assert kinds - MEMORY_ONLY <= copies, (f"{sorted(kinds - MEMORY_ONLY - copies)}: give each a {COPIES} line in "
                                           f"data/floors.csv (the lane-words it writes), or add it to MEMORY_ONLY if it "
                                           f"writes no register")


def test_adopted_lines_are_faster():
    """Each line adopted from a probe at an epoch is cheaper than the line it closed at the epoch before."""
    lines = census_lines(M)
    u = lambda d: u_of(M.cycle_u, d["rate"], d["noise"], d["cycles"], d["u"])
    for epoch, before, run in [("VF3k-16", "VF3j-8", "r20261016-183617-3914"),
                               ("VF3l-16", "VF3k-16", "r20261016-210926-d856"),
                               ("VF3m-16", "VF3l-16", "r20261017-034149-3f95"),
                               ("VF3m-16", "VF3l-16", "r20261017-035245-089f")]:
        for d in (d for d in lines if d["since"] == epoch and run in d["runs"].split()):
            old = [e for e in lines if (e["kind"], e["row"]) == (d["kind"], d["row"]) and e["until"] == before]
            assert old == [] or u(d) < u(old[0]), f"{d['kind']} on {d['row']}"
    assert sum(d["since"] == "VF3k-16" for d in lines) == 18 + 5
    assert sum(d["since"] == "VF3l-16" for d in lines) == 6
    assert sum(d["since"] == "VF3m-16" for d in lines) == 30 + 15 + 1   # the joint rows, the side units, SHFL


def test_sm89_census_is_well_formed():
    """data/devices/sm_89/ (opcode_probe/sm89/census.py): one epoch, every line on a row it defines, every kind on issue
    and JI89."""
    d = DEVICES / "sm_89"
    rs = rows(d)
    assert all(r.status in STATUSES and r.runs and all(CITE.fullmatch(x) for x in r.runs) for r in rs.values())
    with open(d / "census.csv", newline="") as f:
        lines = list(csv.DictReader(f))
    for x in lines:
        assert x["row"] in rs and not x["since"] and not x["until"], x
        assert sum(x[c] != "" for c in ("rate", "cycles", "u")) == 1 and all(CITE.fullmatch(r) for r in x["runs"].split())
    c = census(M, M.epochs[-1], d)   # raises on two lines for one kind and row
    assert all(u > 0 for v in c.values() for u in v.values())
    assert all({"issue", "JI89"} <= set(v) for v in c.values())
    assert {r for v in c.values() for r in v} == set(rs)


def test_violated_rows_close():
    """VF3m-16 closes every joint row EPS1's frontiers put a point past (r20261017-034143-b1de, r20261017-035229-7c35):
    R2-R4, the KIMAD, KHFMA2 and KSHFL vertices, and the ASSUMED SHFL; KALU holds and stays."""
    c = census(M, "VF3m-16")
    closed = {"R2", "R3", "R4", *(f"KIMAD{i}" for i in range(4)), *(f"KHFMA2{i}" for i in range(4)),
              *(f"KSHFL{i}" for i in range(3))}
    assert not closed & {r for v in c.values() for r in v}
    assert closed <= {r for v in census(M, "VF3l-16").values() for r in v}
    assert {f"KALU{i}" for i in range(4)} <= set(c["ALU"]) and ROWS["SHFL"].status == "measured"
    assert c["SHFL"]["SHFL"] == pytest.approx(u_of(M.cycle_u, "0.4997201", "0.02", "", ""))


def test_joint_issue_rows_refit_over_every_mma_form():
    """VF3n-16 closes JI3e_0..2, which EPS1 TENSOR's M1 programs load past 1 (IMMA m16n8k16 and TF32 k4 beside IADD3,
    r20261017-040022-5e4d, r20261017-041103-d35d, r20261017-041754-7630), for JI3n_0..2: each the refit at the 2% noise
    margin ÷ 1.07 (the schedule spread between builds), with MMA every form's least and each form its own kind."""
    c, old = census(M, "VF3n-16"), census(M, "VF3m-16")
    ji3e, ji3n = [f"JI3e_{i}" for i in range(3)], [f"JI3n_{i}" for i in range(3)]
    assert not set(ji3e) & {r for v in c.values() for r in v}
    forms = [k for k in c if k not in old and set(ji3n) <= set(c[k])]
    assert len(forms) == 14 and {"IMMA16832", "IMMA16816", "HMMA1684T", "OMMASF16864"} <= set(forms)
    for e, n, mma in zip(ji3e, ji3n, (7.5574, 13.3816, 14.4395)):
        assert ROWS[n].status == "joint"
        assert c["MMA"][n] == pytest.approx(min(c[k][n] for k in forms)) == pytest.approx(mma * M.cycle_u, abs=0.05)
        assert c["MMA"][n] < old["MMA"][e] / 1.07
        assert all(c[k][n] == pytest.approx(v[e] / 1.07) for k, v in old.items() if e in v and k != "MMA")


def test_joint_issue_rows_take_the_schedule_search():
    """VF3o-16 closes JI3n_0..2 for JI3o_0..2: M1c's 600 schedules (r20261017-052102-d832) join M1's programs in the refit
    at the 2% noise margin, and each row's ÷ 1.07 gives way to its measured margin, the most a searched program loads
    the row fitted to M1 alone (over 0.98). Every kind but the MMA forms keeps VF3e's other coefficient over that margin."""
    c, old = census(M, "VF3o-16"), census(M, "VF3n-16")
    ji3n, ji3o = [f"JI3n_{i}" for i in range(3)], [f"JI3o_{i}" for i in range(3)]
    assert not set(ji3n) & {r for v in c.values() for r in v}
    forms = [k for k in old if k not in census(M, "VF3m-16") and set(ji3n) <= set(old[k])]
    assert len(forms) == 14
    for n, o, mma, margin in zip(ji3n, ji3o, (7.2340, 13.8781, 14.6943), (1.03783, 1.02623, 1.02466)):
        assert ROWS[o].status == "joint" and "r20261017-052102-d832" in ROWS[o].runs
        assert c["MMA"][o] == pytest.approx(min(c[k][o] for k in forms)) == pytest.approx(mma * M.cycle_u, abs=0.05)
        assert all(c[k][o] == pytest.approx(v[n] * 1.07 / margin, rel=1e-5)
                   for k, v in old.items() if n in v and k not in forms and k != "MMA")
