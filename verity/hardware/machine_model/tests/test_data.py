"""data/ is well formed: every census line names a row, an epoch span and one value; vectors and attacks name what exists."""
from __future__ import annotations

import csv
import json
import re
import tomllib

import pytest

from verity_machine_model.model import (CLAIMS, COPIES, DATA, DEVICES, STATUSES, all_attacks, attacks, census,
                                        census_files, census_lines, citations, cites, entries, floors, holds,
                                        kind_lists, model, rows, run_lists, u_of, vector, vectors)

M = model()
ROWS = rows()
CITE = re.compile(r"(r\d{8}-\d{6}-[0-9a-f]{4}|note:\d{8}T\d{4}Z-[a-z0-9-]+|art:[0-9a-f]{64})")
PROSE_CITE = re.compile(r"@([a-z0-9][a-z0-9-]*[a-z0-9])")


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def prose_cites() -> dict[str, set[str]]:
    """Each TOML file of data/ and the device directories, and the @NAMEs its strings cite."""
    return {str(p.relative_to(DATA)): {m for s in _strings(tomllib.loads(p.read_text())) for m in PROSE_CITE.findall(s)}
            for p in sorted(DATA.rglob("*.toml"))}


def test_epochs_are_vectors():
    names = set(vectors())
    assert len(set(M.epochs)) == len(M.epochs)
    assert set(M.epochs) <= names, "an epoch is named after the vector that introduced it"
    assert {vector(n).epoch for n in names} == set(M.epochs)
    assert M.current in names


def test_rows_are_well_formed():
    assert all(r.status in STATUSES for r in ROWS.values())
    assert all(r.runs and all(CITE.fullmatch(x) for x in r.runs) for r in ROWS.values())


def test_every_citation_is_a_note_or_run_record_some_line_cites():
    """data/citations.toml names notes and run records, each cited as @NAME by a census or rows.csv line's runs or by a
    TOML string in data/ or a device directory; every @NAME a TOML string cites is one of them."""
    names = citations()
    assert names and all(CITE.fullmatch(v) and v.startswith(("note:", "art:")) for v in names.values())
    used = set()
    for p in [*DATA.glob("*.csv"), *DATA.glob("census/*.csv"), *DEVICES.glob("sm_*/*.csv"),
              *DEVICES.glob("sm_*/census/*.csv")]:
        with open(p, newline="") as f:
            used |= {c[1:] for d in csv.DictReader(f) for c in d.get("runs", "").split() if c.startswith("@")}
    prose = prose_cites()
    unknown = {p: sorted(c - set(names)) for p, c in prose.items() if c - set(names)}
    assert not unknown, f"cite a run record or note as @NAME only once data/citations.toml names it: {unknown}"
    used |= set().union(*prose.values())
    assert set(names) <= used, f"citations.toml: {sorted(set(names) - used)} cited by no line"


CENSUS_COLUMNS = ("kind", "row", "rate", "noise", "cycles", "u", "since", "until", "runs")
BLOB_LIMIT = 256 * 1024   # tests/test_repository.py's cap on a tracked file


def test_census_is_a_file_per_epoch():
    """data/census/ is EPOCH.csv for each epoch that brought lines, each line's since its file's, and runs.toml; no file
    comes near the repository's cap, so the census grows by files, not by one file growing."""
    files = census_files(M)
    assert {p.name for p in (DATA / "census").iterdir()} == {p.name for p in files} | {"runs.toml", "kinds.toml"}, (
        "data/census/ holds EPOCH.csv for epochs in model.toml's epochs, runs.toml and kinds.toml")
    named, kinds = {k: v for k, v in run_lists().items() if k not in citations()}, kind_lists()
    used, used_kinds = set(), set()
    for p in files:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            assert tuple(r.fieldnames or ()) == CENSUS_COLUMNS, p.name
            for d in r:
                assert d["since"] == p.stem, f"census/{p.name}: {d['kind']} on {d['row']} is from {d['since']!r}"
                used |= {c[1:] for c in d["runs"].split() if c.startswith("@") and c[1:] not in citations()}
                used_kinds |= {d["kind"][1:]} if d["kind"].startswith("@") else set()
    for p in [*files, DATA / "census" / "runs.toml", DATA / "census" / "kinds.toml"]:
        assert p.stat().st_size <= BLOB_LIMIT // 4, (
            f"census/{p.name} is {p.stat().st_size} bytes, past a quarter of the {BLOB_LIMIT}-byte cap: name a run list "
            f"its lines share in census/runs.toml and cite it as @NAME, or a kind list a row prices at one value in "
            f"census/kinds.toml and write its line as kind @NAME")
    assert used == set(named), f"runs.toml: {sorted(set(named) - used)} cited by no line; {sorted(used - set(named))} not in it"
    assert all(xs and all(CITE.fullmatch(x) for x in xs) for xs in named.values())
    assert used_kinds == set(kinds), (f"kinds.toml: {sorted(set(kinds) - used_kinds)} written by no line; "
                                      f"{sorted(used_kinds - set(kinds))} not in it")
    assert all(xs and len(set(xs)) == len(xs) and not any(x.startswith("@") for x in xs) for xs in kinds.values())


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
    kinds = kind_lists()
    assert len(entries(M)) == sum(len(kinds[d["kind"][1:]]) if d["kind"].startswith("@") else 1 for d in lines)


@pytest.mark.parametrize("epoch", M.epochs)
def test_one_value_per_kind_and_row_at_each_epoch(epoch):
    c = census(M, epoch)   # raises on two lines for one kind and row
    assert all(u > 0 for v in c.values() for u in v.values())
    assert set(M.honest) <= set(c) and "MMA" in c


def test_sm80_census_is_well_formed():
    """data/devices/sm_80/ (tensor_kinds/sm80.py census): one epoch, every line on a row it defines with one value, every
    MMA kind on M and the joint row JIM."""
    d = DATA / "devices" / "sm_80"
    rows80 = rows(d)
    assert all(r.status in STATUSES and r.runs and all(CITE.fullmatch(x) for x in r.runs) for r in rows80.values())
    with open(d / "census.csv", newline="") as f:
        lines = list(csv.DictReader(f))
    for x in lines:
        where = f"devices/sm_80/census.csv: {x['kind']} on {x['row']}"
        assert x["row"] in rows80, f"{where}: add the row to devices/sm_80/rows.csv"
        assert sum(x[c] != "" for c in ("rate", "cycles", "u")) == 1, f"{where}: give one of rate, cycles and u"
        assert x["noise"] == "" or x["rate"] != "", f"{where}: noise derates a rate only"
        assert not x["since"] and not x["until"], f"{where}: a device directory has one epoch, no since or until"
        assert x["runs"] and all(CITE.fullmatch(r) for r in cites(x["runs"], run_lists(d))), (
            f"{where}: cite its runs or note")
    assert len(lines) == len(entries(M, d))
    c = census(M, M.epochs[-1], d)   # raises on two lines for one kind and row
    assert all(u > 0 for v in c.values() for u in v.values())
    mma = {k for k, v in c.items() if "M" in v}
    assert mma and all("JIM" in c[k] for k in mma) and {r for v in c.values() for r in v} == set(rows80)


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
        assert sum(x[c] != "" for c in ("rate", "cycles", "u")) == 1
        assert all(CITE.fullmatch(r) for r in cites(x["runs"], run_lists(d))), x
    c = census(M, M.epochs[-1], d)   # raises on two lines for one kind and row
    assert all(u > 0 for v in c.values() for u in v.values())
    assert all({"issue", "JI89"} <= set(v) for v in c.values())
    assert {r for v in c.values() for r in v} == set(rs)


@pytest.mark.parametrize("d", sorted(p for p in DEVICES.glob("sm_*") if (p / "model.toml").is_file()),
                         ids=lambda p: p.name)
def test_datasheet_vector(d):
    """A device's DS-NN reads its datasheet rows, and where it has one DS_uniform, the measured UP line exactly (no
    document states the uniform datapath's rate); every kind is on DS_issue at one SMSP-cycle, and no
    data-sheet line prices a kind above what its measured rate alone does by more than the 2% margin (a peak the chip
    beats is a wrong peak, a wrong form's MACs or a wrong pipe)."""
    rs, m = rows(d), model(d)
    v = vector(f"DS-{d.name[3:]}", d)
    assert {r for r in rs if rs[r].status == "datasheet"} == set(v.rows) - {"DS_uniform"} and "DS_issue" in v.rows
    c = census(m, v.epoch, d)
    assert all(row["DS_issue"] == pytest.approx(m.cycle_u) for row in c.values())
    if "DS_uniform" in v.rows:
        assert rs["DS_uniform"].status == "measured"
        assert {k for k, row in c.items() if "DS_uniform" in row} == {"UNI"} and c["UNI"]["DS_uniform"] == c["UNI"]["UP"]
    with open(d / "census.csv", newline="") as f:
        alone = {}
        for x in csv.DictReader(f):
            if x["rate"]:
                alone[x["kind"]] = max(alone.get(x["kind"], 0.0), u_of(m.cycle_u, x["rate"], "", "", ""))
    over = {(k, r): round(u / alone[k], 4) for k, row in c.items() for r, u in row.items()
            if r in v.rows and k in alone and u > 1.02 * alone[k]}
    assert not over, f"data-sheet lines above the kind's measured rate alone: {over}"


@pytest.mark.parametrize("name", ["sm_90", "sm_100"])
def test_datasheet_epoch(name):
    """sm_90's and sm_100's DS lines are census/DS-NN.csv, an epoch after DC-NN whose lines close where they open
    (dc/census.py --write rewrites census.csv and keeps none of them), and that epoch's census is DC-NN's with the DS
    lines added, so DS-NN prices on DC's census and moves no other vector."""
    d = DEVICES / name
    m, e = model(d), f"DS-{name[3:]}"
    v = vector(e, d)
    ds = set(v.rows)
    assert v.epoch == e and m.epochs[1] == e
    assert [p.relative_to(d).as_posix() for p in census_files(m, d)] == ["census.csv", f"census/{e}.csv"]
    with open(d / "census.csv", newline="") as f:
        stray = [x["kind"] for x in csv.DictReader(f) if x["row"] in ds]
    assert not stray, f"census.csv: {stray} on a DS row; a DS line goes in census/DS-NN.csv"
    with open(d / "census" / f"{e}.csv", newline="") as f:
        lines = list(csv.DictReader(f))
    assert lines and all(x["since"] == x["until"] == e and x["row"] in ds for x in lines)
    measured = {k: rest for k, row in census(m, e, d).items() if (rest := {r: u for r, u in row.items() if r not in ds})}
    assert measured == census(m, m.epochs[0], d)


def test_datasheet_vector_sm120():
    """DS-120 reads data/'s datasheet rows alone at its own epoch, whose lines close where they open; every kind there is
    on DS_issue at one SMSP-cycle and every MMA form on DS_tensor, and no data-sheet line prices a kind above its time
    alone on the current vector's census by more than the 2% margin."""
    v = vector("DS-120")
    assert {r for r in ROWS if ROWS[r].status == "datasheet"} == set(v.rows) and v.epoch == "DS-120"
    assert all(d["since"] == d["until"] == "DS-120" for d in census_lines(M) if d["row"] in v.rows)
    c, cur = census(M, "DS-120"), census(M, vector(M.current).epoch)
    assert all(row.get("DS_issue") == pytest.approx(M.cycle_u) for row in c.values())
    assert all("DS_tensor" in c[k] for k, row in cur.items() if "M" in row)
    alone = {k: max(row.values()) for k, row in cur.items()}
    over = {(k, r): round(u / alone[k], 4) for k, row in c.items() for r, u in row.items()
            if r in v.rows and k in alone and u > 1.02 * alone[k]}
    assert not over, f"data-sheet lines above the kind's time alone: {over}"


def test_violated_rows_close():
    """VF3m-16 closes every joint row the measured frontiers put a point past:
    R2-R4, the KIMAD, KHFMA2 and KSHFL vertices, and the ASSUMED SHFL; KALU holds and stays."""
    c = census(M, "VF3m-16")
    closed = {"R2", "R3", "R4", *(f"KIMAD{i}" for i in range(4)), *(f"KHFMA2{i}" for i in range(4)),
              *(f"KSHFL{i}" for i in range(3))}
    assert not closed & {r for v in c.values() for r in v}
    assert closed <= {r for v in census(M, "VF3l-16").values() for r in v}
    assert {f"KALU{i}" for i in range(4)} <= set(c["ALU"]) and ROWS["SHFL"].status == "measured"
    assert c["SHFL"]["SHFL"] == pytest.approx(u_of(M.cycle_u, "0.4997201", "0.02", "", ""))


def test_joint_issue_rows_refit_over_every_mma_form():
    """VF3n-16 closes JI3e_0..2, which the tensor_kinds probe's M1 programs load past 1 (IMMA m16n8k16 and TF32 k4
    beside IADD3), for JI3n_0..2: each the refit at the 2% noise margin ÷ 1.07 (the schedule spread between builds),
    with MMA every form's least and each form its own kind."""
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
    """VF3o-16 closes JI3n_0..2 for JI3o_0..2: the schedule search's 600 schedules (M1c) join M1's programs in the refit
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


def test_uniform_takes_the_issue_slot():
    """VF3p-16: a uniform instruction takes the SMSP's issue slot, 1 on
    issue and on each joint issue row 0.7416 of a vector instruction's coefficient, the share that holds every uniform.cu
    point at the 2% noise margin over JI3o's coefficients (0.8168 holds them exactly; the honest step beside 32 FFMA sets
    it at JI3o_0); UP refit to the fastest form alone, 0.766 a SMSP-cycle."""
    c, old = census(M, "VF3p-16"), census(M, "VF3o-16")
    assert "issue" not in old["UNI"] and c["UNI"]["issue"] == pytest.approx(M.cycle_u)
    for n in ("JI3o_0", "JI3o_1", "JI3o_2"):
        assert n not in old["UNI"] and c["UNI"][n] / c["FFMA"][n] == pytest.approx(0.7416, abs=1e-6)
    assert c["UNI"]["UP"] == pytest.approx(u_of(M.cycle_u, "3.0621", "", "", ""))
    assert c["UNI"]["UP"] < old["UNI"]["UP"]
    assert {k: v for k, v in c.items() if k != "UNI"} == {k: v for k, v in old.items() if k != "UNI"}
