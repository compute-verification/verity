"""onsite_periods.py over committed stand-in onsite records (`fixtures/onsite_periods/`, cut from the writer's own
record; nothing here runs or imports the writer): each cell's proved or scaled label held to the record's rules, every
way a label can break refused, and the tables."""
from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import onsite_periods as P

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "onsite_periods"
STATEMENT = "NetTiming.Guarantees.MonitoringPeriodLeakAccepted"
LOCK = {"guarantees": {"NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted": {
    "signature": [f"NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted : {STATEMENT}"]}}}


def stand_in(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["m1", "m2", "m1-week", "m1-key-open"])
def test_stand_ins_hold_their_rules(name):
    assert P.check(stand_in(name), LOCK) == []


def test_the_reader_imports_only_the_standard_library():
    tree = ast.parse(Path(P.__file__).read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= sys.stdlib_module_names | {"__future__"}, names


def test_a_signature_wrapped_after_its_colon_is_locked():
    wrapped = {"guarantees": {"NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted": {
        "signature": ["NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted.{u_1, u_2} :", f"  {STATEMENT}"]}}}
    assert P.locked(wrapped, STATEMENT) and P.locked(LOCK, STATEMENT)
    assert not P.locked(wrapped, "NetTiming.Guarantees.Period")
    assert P.check(stand_in("m1-week"), wrapped) == []


def test_a_short_run_proves_no_node_cell():
    m1 = stand_in("m1")
    labels = P.expected(m1["periods"], 1, P.Fraction(m1["bound"]["period_s"]))
    assert labels["run.per_phase"] == ("proved", None)
    assert all(labels[f"{p}.node"] == ("scaled", "run-to-period") for p in P.PERIODS)


def test_k_over_a_phase_is_proved_only_at_one_phase():
    m2, m1 = stand_in("m2"), stand_in("m1-week")
    assert m2["bound"]["phases"] == 2 and m1["bound"]["phases"] == 1
    assert P.expected(m2["periods"], 2, P.Fraction(m2["bound"]["period_s"]))["run.per_phase"] == ("scaled", "per-phase")
    assert P.expected(m1["periods"], 1, P.Fraction(m1["bound"]["period_s"]))["run.per_phase"] == ("proved", None)


def test_the_node_is_proved_only_where_the_run_covers_the_period():
    m1 = stand_in("m1-week")
    labels = P.expected(m1["periods"], 1, P.Fraction(m1["bound"]["period_s"]))
    assert labels["week.node"] == ("proved", None)
    assert labels["month.node"] == labels["year.node"] == ("scaled", "run-to-period")
    m2 = stand_in("m2")
    assert all(P.expected(m2["periods"], 2, P.Fraction(m2["bound"]["period_s"]))[f"{p}.node"] ==
               ("scaled", "run-to-period") for p in P.PERIODS)
    assert {labels[f"{p}.{c}{x}"] for p in P.PERIODS for c in ("datacenter", "world") for x in ("", ".per_phase")} == {
        ("scaled", "datacenter"), ("scaled", "world")}


def broken(name: str, edit) -> list[str]:
    rec = copy.deepcopy(stand_in(name))
    edit(rec)
    return P.check(rec, LOCK)


@pytest.mark.parametrize("name, edit, says", [
    ("m2", lambda r: r["periods"]["week"]["datacenter"].update(how="proved", rule=None), "periods.week.datacenter is proved"),
    ("m2", lambda r: r["periods"]["run"]["per_phase"].update(how="proved", rule=None), "periods.run.per_phase is proved"),
    ("m2", lambda r: r["periods"]["year"]["node"].update(how="proved", rule=None), "periods.year.node is proved"),
    ("m1-week", lambda r: r["periods"]["week"]["node"].update(rule="run-to-period"), "periods.week.node is proved by run"),
    ("m1-week", lambda r: r["periods"]["month"]["node"].update(rule="per-phase"), "its rule makes it scaled by run-to"),
    ("m2", lambda r: r["periods"]["rules"]["world"].update(sources=[]), "which the record's rules don't give"),
    ("m1-week", lambda r: r["verdicts"]["the network certifier"].update(verdict="reject"), "network certifier's verdict"),
    ("m1-week", lambda r: r["periods"]["run"].update(statement="NetTiming.Guarantees.Other"), "the lock doesn't hold"),
    ("m2", lambda r: r["bound"].update(phases=1), "over m = 1"),
    ("m2", lambda r: r["bound"]["of_record"].update(K=r["bound"]["K"]), "isn't K"),
    ("m2", lambda r: r["bound"]["of_record"].pop("tight"), "states no tight rule"),
    ("m2", lambda r: r.update(periods_tight=r["periods"]), "has periods_tight"),
    ("m2", lambda r: r["periods"].update(conditional_on=["certifier-key"]), "periods is conditional on ['certifier-key']"),
    ("m1-key-open", lambda r: (r["periods"].pop("conditional_on"), r["bound"]["of_record"].pop("conditional_on")),
     "periods.run is proved, and the network certifier's verdict is not established"),
    ("m1-key-open", lambda r: r["verdicts"]["the network certifier"].update(answer="reject"), "its answer reject"),
    ("m1-key-open", lambda r: r["verdicts"]["the network certifier"].update(open=[]), "network certifier's verdict"),
    ("m1-key-open", lambda r: r["verdicts"]["the network certifier"]["open"].append("auditor-listing"),
     "open ['certifier-key', 'auditor-listing']"),
    ("m2", lambda r: r["bound"].update(notes=[]), "overcount note"),
    ("m2", lambda r: r.update(format="verity/onsite-network-record/v0"), "not verity/onsite-network-record/v1"),
])
def test_a_label_against_its_rule_is_refused(name, edit, says):
    why = broken(name, edit)
    assert any(says in x for x in why), why


def test_tables(tmp_path):
    md = P.markdown(stand_in("m1-week"))
    week = next(x for x in md.splitlines() if x.startswith("| week"))
    node, dc, world = week.split(" | ")[1:4]
    assert node.startswith(f"**proved** (`{STATEMENT}`)") and dc.startswith("**scaled** (datacenter)")
    assert world.startswith("**scaled** (world)") and "× the served model's" in world
    assert "Note: a known overcount" in md and "ℓ_a = ['8,388,608']" in md and "How it tightens:" in md
    assert "conditional" not in md.split("## K of record")[1].split("## Rules")[0]
    md2 = P.markdown(stand_in("m2"))
    assert "over one phase **scaled** (per-phase)" in md2
    md3 = P.markdown(stand_in("m1-key-open"))
    grid = md3.split("## K of record")[1].split("## Rules")[0]
    assert "**proved**" not in grid and f"**conditional** (`{STATEMENT}`, on certifier-key)" in grid
    assert "**scaled** (world; conditional on certifier-key)" in grid
    assert "not established, its Lean answer accept, open ['certifier-key']" in md3


def test_main_refuses_a_broken_record(tmp_path, capsys):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(LOCK))
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(stand_in("m2")))
    rec = stand_in("m2")
    rec["periods"]["month"]["world"]["how"] = "proved"
    bad.write_text(json.dumps(rec))
    assert P.main([str(good), "--lock", str(lock), "--out", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "t.md").read_text().startswith("# The onsite record of")
    assert P.main([str(bad), "--lock", str(lock)]) == 1
    assert "REFUSED periods.month.world is proved" in capsys.readouterr().err
