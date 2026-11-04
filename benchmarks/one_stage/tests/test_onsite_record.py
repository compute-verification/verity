"""`onsite_record.py` on toy served runs' $OUTs. A run whose network certifier, PoUW and composites are missing, as a
rehearsal's, records each of them missing with its why, and no bound and no period figure. A run whose network
certifier accepted in the rows form records K term by term in the notation table's symbols, each naming the public
fields it sums, with its overcount as a note, the seconds it covers, the line that published its verdict, the tree, and
the connections its schedule's caps count (other caps are the connections' error); its run's own period proved, and
each monitoring period's node proved where the run's period covers it and else scaled from the run, and the
datacenter's and the world's scaled from the node's, each scaled cell naming its rule, whose sources (the census lines
by SHA-512) the record states, beside its figure over one execution phase. A public whose terms don't sum to K, or in
the clock form, is the bound's error. The served run's advice charge stays out of the record, ℓ_{a,j} is stated with
its source (the tied audits' free inputs, no advice length and no cap), and a test run's measurement is stated beside
the bound as that run's. The figure of record, every period's cells included, is that K only where the verdict held
both checks the tight ℓ_{a,j} rests on and the lock (a toy one in Security's lock's place) leaves no premise of its
guarantee open, and otherwise charges ℓ_a + 1 bits for each tied audit of each phase, the tight K's cells beside it,
conditional on what is open. A tied audit drawn in part charges its slot entries in k_{skip,j}; each cell states its
share of the schedule's egress capacity, and the served egress is missing. The run's `publish`
leaves the record as written. The run is of record only when every verdict, the onsite composite's included,
accepted."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import onsite_record as OR  # noqa: E402
import served_verdicts as SV  # noqa: E402
import leak_bounds as LB  # noqa: E402

NODE = "computeinstance-toy"
SCAN = {"delta": "1/1048576", "drawn": 126, "units": 126, "wrong_units": 0, "free": 9088, "faulty_sets": 0,
        "faulty_bits": 0, "faulty_entries": 0, "jitter": 63}
#: the schedule's 4 · 63 connections of cap 4 (⌈log2 5⌉ = 3 each, as cells and again as schedules) and one execution
#: phase's scan
PUBLIC = {"K": 756 + 756 + 9088 + 63, "audits": [{"delta": "1/100", "drawn": 2, "units": 1024, "wrong_units": 0}],
          "cells": 756, "eps": "1/1048576", "scans": [SCAN], "schedules": 756, "windows": 1}
K = PUBLIC["K"]
#: the period's one compute audit's registration, whose window names no advice (ℓ_a = 0), and the served one's, whose
#: advice is padded to the compiler's cap
NO_ADVICE = {"format": "verity/registration/v2", "window": {"circuit": "cd" * 64}}
SERVED = {"format": "verity/registration/v2", "window": {"advice": {"bits": 2 ** 23, "root": "ef" * 64},
                                                         "compiler": {"digest": "01" * 64, "name": "B"}}}
#: K of record with both checks open and the one audit's ℓ_a + 1 = 1 bit for the one phase
KR = K + 1
#: T = 63 buckets of 20 ms: an execution phase of 63/50 s
SCHEDULE = {"bucket_ms": 20, "cell_bytes": 512, "delay": 4, "egress_cells": 4, "every": 63, "ingress_buckets": 63,
            "ingress_cells": 4, "lanes": 4, "queue": 16, "slots": 8, "window": 63}
MEASURED = {"format": "verity/advice-charge/v2", "label": "measured on test run toy-test: …", "run": "toy-test",
            "commit_periods": [{"period": 0, "bits": 480}, {"period": 1, "bits": 2390}],
            "bits": {"count": 2, "min": 480, "median": 1435, "mean": 1435.0, "max": 2390, "total": 2870},
            "by_phase": [{"phase": 0, "bits": 2870, "commit_periods": 2}],
            "phase_bits": {"count": 1, "min": 2870, "median": 2870, "mean": 2870.0, "max": 2870, "total": 2870},
            "prompt_len": {"count": 5}, "output_len": {"count": 5}, "finishes": {"eos": 3, "length": 2, "abort": 0}}
TREE = "ab12" * 10
#: the certifier's published verdict line, after an earlier attempt's
LINE = 'THE NETWORK CERTIFIER {"rc": 0, "verdict": "accept"}'
#: SCHEDULE's egress capacity a phase: its 4 · 8 cells a bucket over 63 buckets exceed its caps' 252 · 4 = 1008 cells
CAPACITY = 1008 * 512 * 8
#: the premises of a lock recording no named assumption for the guarantee the tight K rests on
NO_PREMISES = {"is": OR.PREMISES, "from": "Security/lean-audit.json guarantees[g].assumptions",
               "guarantees": {g: {"assumptions": []} for g in OR.RESTS_ON}, "open": []}


@pytest.fixture(autouse=True)
def lock(tmp_path_factory, monkeypatch) -> Path:
    """A lock in Security's lock's place recording each guarantee the tight K rests on with no named assumption."""
    f = tmp_path_factory.mktemp("lock") / "lean-audit.json"
    f.write_text(json.dumps({"guarantees": {g: {"assumptions": []} for g in OR.RESTS_ON}}))
    monkeypatch.setattr(OR, "LOCK", f)
    return f


def answer(public: dict | None, verdict: str = "accept") -> dict:
    return {"protocol": OR.NETWORK, "verdict": verdict, "reason": None, "commitment": "ab" * 64, "public": public,
            "profile": {"link": "served", "windows": 1, "placement": {"trusted-host": "the certifier on the gate's host"}}}


def network_row(public: dict | None) -> dict:
    return {"protocol": OR.NETWORK, "owner": "network-accounting", "verdict": "accept", "reason": None,
            "commitment": "ab" * 64, "quantity": public, "delta": None, "epsilon": None, "runs": [],
            "answer": answer(public)}


def served_out(tmp_path: Path, *, network: dict | None = None, work: dict | None = None, case: dict | None = None,
               composites: dict[str, dict] | None = None) -> Path:
    """A toy $OUT: sampled proofs accepted, PoUW and the network certifier missing unless given, and the given
    composites and certifier case."""
    out = tmp_path / "out"
    rows = {"integrity": {"protocol": "verity/sampled-proofs/one-stage/v1", "owner": "compute-accounting",
                          "verdict": "accept", "reason": None, "answer": {}},
            "work": work or {"owner": "compute-accounting", "verdict": "missing", "why": "POUW_CMD unset: toy"},
            "communication": network or {"owner": "network-accounting", "verdict": "missing",
                                         "why": "CERTIFIER_CMD unset: the network certifier in front of the serving "
                                                "endpoint"}}
    (out / "record").mkdir(parents=True)
    (out / "record" / "verdicts.json").write_text(json.dumps(
        {"format": OR.EXTENDS, "window": 3, "node": NODE, "run": "toy-run", "protocols": rows}))
    for name, doc in (composites or {}).items():
        (out / "composite" / name).mkdir(parents=True)
        (out / "composite" / name / "composite.json").write_text(json.dumps(doc))
    if case is not None:
        (out / OR.CASE).parent.mkdir(parents=True)
        (out / OR.CASE).write_text(json.dumps(case))
    return out


def remote_composite(verdict: str = "accept") -> dict:
    cases = [{"id": i, "verdict": "accept", "params_sha512": "00", "commitment": "00"}
             for i in (OR.INFERENCE_ONLY, "verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v1")]
    return {"product": OR.REMOTE, "commitment": "00", "missing": [], "exceptions": [], "cases": cases,
            "verdict": {"id": OR.REMOTE, "verdict": verdict}}


def commitment(*audits: dict) -> str:
    return json.dumps({"audits": list(audits), "windows": [{}]}).encode().hex()


def accepted(tmp_path: Path, public: dict = PUBLIC, windows: int = 1, caps: list[int] | None = None,
             schedule: dict = SCHEDULE, audits: tuple[dict, ...] = (NO_ADVICE,)) -> Path:
    """A run whose network certifier accepted ``public``, its case the period's of ``windows`` execution phases at
    ``schedule``, whose caps are the schedule's (`Schedule.caps`) unless given, and whose commitment holds ``audits``'
    registrations."""
    pouw = {"protocol": "verity/pouw/period/v1", "owner": "compute-accounting", "verdict": "accept", "reason": None}
    if caps is None:
        caps = [schedule["egress_cells"]] * (schedule["lanes"] * windows * schedule["window"])
    out = served_out(tmp_path, network=network_row(public), work=pouw,
                     case={"commitment": commitment(*audits), "evidence": "00",
                           "params": {"windows": windows, "cells": caps, "schedule": schedule}},
                     composites={"remote": remote_composite()})
    (out / "machine.txt").write_text(f"ubuntu\n{NODE}\n{TREE}\n96\nserve dir /serve/d\n")
    (out / "verdicts.txt").write_text('THE NETWORK CERTIFIER {"rc": 1, "verdict": "error"}\nRECORD {}\n'
                                      f"{LINE}\nCOMPOSITE CASE verity/certifier/v0 accept\n")
    return out


def numbers(x) -> list:
    """Every number in a JSON document."""
    if isinstance(x, dict):
        return [n for v in x.values() for n in numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in numbers(v)]
    return [x] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def test_a_rehearsal_records_each_missing_part_with_its_why_and_no_figure(tmp_path, capsys):
    out = served_out(tmp_path)
    assert OR.main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    v = rec["verdicts"]
    assert list(v) == sorted(OR.VERDICTS) and v["sampled proofs"]["verdict"] == "accept"
    assert v["PoUW"] == {"verdict": "missing", "why": "POUW_CMD unset: toy", "from": "record/verdicts.json protocols.work"}
    assert v["the network certifier"]["why"].startswith("CERTIFIER_CMD unset")
    assert v["remote"]["why"] == "remote's composite didn't run: the run has no composite/remote/composite.json"
    assert v["inference-only"]["verdict"] == v["onsite"]["verdict"] == "missing"
    assert rec["of_record"] == {"holds": False, "missing": ["PoUW", "inference-only", "remote", "the network certifier",
                                                            "onsite"], "not_accepting": []}
    assert rec["bound"] == {"missing": "the network certifier's verdict is missing (CERTIFIER_CMD unset: the network "
                                       "certifier in front of the serving endpoint): only its accepted answer certifies K"}
    assert rec["periods"] == {"missing": "no bound"}
    assert rec["advice_charge"] == {"served": OR.ADVICE, "charged": {"missing": "no bound"},
                                    "cap": {"missing": "no bound"}, "measured": {"missing": OR.NO_MEASUREMENT}}
    assert numbers({k: rec[k] for k in ("bound", "periods", "verdicts", "advice_charge")}) == []
    assert (rec["composite_error"]["stated"], rec["composite_error"]["vacuous"]) == (None, False)
    assert rec["composite_error"]["terms"][3] == {"subprotocol": "the network certifier", "missing": "no accepted bound",
                                                  "from": f"{OR.EPS['source']}: its eps (`EPS`)"}
    assert "window" not in rec and rec["tree"] is None
    line = capsys.readouterr().out.strip()
    assert line.startswith("ONSITE ") and json.loads(line[7:])["K"] is None


def test_an_accepted_rows_form_answer_is_K_term_by_term_with_its_overcount_and_connections(tmp_path):
    rec = OR.record(accepted(tmp_path))
    b = rec["bound"]
    assert (b["K"], b["phases"], b["period_s"], b["statement"]) == (
        K, 1, "63/50", "NetTiming.Guarantees.PeriodLeakAccepted")
    assert "eps" not in b and b["delta_net"] == {"note": OR.UNVALUED, "terms": [
        {"symbol": "eps", "value": "1/1048576", **OR.EPS}, {"symbol": "ε_ks", "times": 1, **OR.ERRORS["ε_ks"]},
        {"symbol": "ε_bind", "times": 1, **OR.ERRORS["ε_bind"]}, {"symbol": "ε_tap", "times": 1, **OR.ERRORS["ε_tap"]}]}
    assert all(t["source"] for t in b["delta_net"]["terms"]) and "δ part" in OR.EPS["is"]
    assert (rec["run"], rec["tree"], b["verdict_line"]) == ("toy-run", TREE, LINE)
    assert [(t["symbol"], t.get("phase"), t["bits"], t["lean"]) for t in b["terms"]] == [
        ("k_{sched}", None, 756, ["schedules"]), ("k_{fail}", None, 756, ["cells"]),
        ("\\ell_{a,j}", 0, 9088, ["scans[0].free"]),
        ("k_{skip,j}", 0, 0, ["scans[0].faulty_sets", "scans[0].faulty_bits", "scans[0].faulty_entries"]),
        ("T_j \\log_2 d_{jit}", 0, 63, ["scans[0].jitter"])]
    assert sum(t["bits"] for t in b["terms"]) == b["K"]
    assert b["notes"] == [OR.OVERCOUNT] and "k_sched" in OR.OVERCOUNT and "k_fail" in OR.OVERCOUNT
    assert b["connections"] == {"count": 4 * 63, "from": OR.CONNECTIONS}
    assert rec["placement"] == {"trusted-host": "the certifier on the gate's host"}
    assert rec["advice_charge"]["charged"] == {"is": OR.CHARGED, "bits_per_phase": [9088], "share": 9088 / K,
                                               "lean": "scans[j].free"}
    assert "no cap and no advice length" in OR.CHARGED and "audits[].free" in OR.CHARGED
    assert b["of_record"] == {"K": KR, "held": [], "open": ["free", "circuit"], "is": OR.OPEN, "premises": NO_PREMISES,
                              "advice": {"bits_per_phase": [1], "tied": OR.TIED_ALL}}
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [0], "entries_per_phase": [0]}
    assert rec["advice_charge"]["cap"] == {"ell_a": [0], "is": OR.CAP}
    assert "A cap, in no term" in OR.CAP and "`CAP`" in OR.OPEN and "`PREMISES`" in OR.OPEN
    assert rec["periods"]["run"]["bits"] == KR and rec["periods_tight"]["run"]["bits"] == K
    assert rec["periods_tight"]["conditional_on"] == ["free", "circuit"]
    assert rec["periods"]["egress"] == {"capacity_bits_per_phase": CAPACITY, "is": OR.EGRESS,
                                        "served": {"missing": OR.SERVED_EGRESS}}


def test_the_served_audits_cap_is_charged_while_a_check_is_open_and_the_tight_K_is_beside_it(tmp_path, capsys):
    """The served shape: one V3c audit whose advice is padded to ℓ_a = 2^23, tied to the one phase's scan. With neither
    check held, the figure of record charges ℓ_a + 1 beside ℓ_{a,j}, and every cell is that figure's; the tight K's cells
    are beside them, conditional on both checks; the ONSITE line gives both."""
    out = accepted(tmp_path, audits=(SERVED,))
    assert OR.main([str(out)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    of = rec["bound"]["of_record"]
    assert (of["K"], of["open"], rec["advice_charge"]["cap"]["ell_a"], of["advice"]["bits_per_phase"]) == (
        K + 2 ** 23 + 1, ["free", "circuit"], [2 ** 23], [2 ** 23 + 1])
    assert rec["advice_charge"]["charged"]["bits_per_phase"] == [9088]
    assert rec["bound"]["K"] == K and rec["periods"]["year"]["datacenter"]["bits"] > \
        rec["periods_tight"]["year"]["datacenter"]["bits"]
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["K"], line["K_tight"], line["open"]) == (K + 2 ** 23 + 1, K, ["free", "circuit"])


@pytest.mark.parametrize("held, open_", [(["free", "circuit"], []), (["circuit", "free"], []), (["free"], ["circuit"]),
                                         (["circuit", "x"], ["free"])])
def test_the_figure_of_record_is_the_tight_K_only_where_the_verdict_held_both_checks(tmp_path, held, open_):
    rec = OR.record(accepted(tmp_path, dict(PUBLIC, held=held), audits=(SERVED,)))
    of = rec["bound"]["of_record"]
    assert (of["held"], of["open"]) == ([c for c in OR.CHECKS if c not in open_], open_)
    assert of["K"] == (K if not open_ else K + 2 ** 23 + 1)
    assert ("periods_tight" in rec) == bool(open_) and rec["periods"]["run"]["bits"] == of["K"]


@pytest.mark.parametrize("guarantees, found", [
    ({OR.RESTS_ON[0]: {"assumptions": ["NetTiming.Assumptions.B", "NetTiming.Assumptions.A"]}},
     {OR.RESTS_ON[0]: {"assumptions": ["NetTiming.Assumptions.B", "NetTiming.Assumptions.A"]}}),
    ({}, {OR.RESTS_ON[0]: {"missing": "not in the lock"}})])
def test_a_premise_the_lock_leaves_open_keeps_the_figure_of_record_charged_with_both_checks_held(
        tmp_path, lock, guarantees, found):
    """Both checks held, and the lock records a named assumption of the guarantee the tight K rests on, or doesn't list
    it: each is an open premise, so the figure of record still charges ℓ_a + 1 for the tied audit, and the tight K's
    cells beside it are conditional on those premises; with one check open too, on both."""
    lock.write_text(json.dumps({"guarantees": guarantees}))
    open_ = sorted(a for f in found.values() for a in f.get("assumptions", ())) or list(OR.RESTS_ON)
    rec = OR.record(accepted(tmp_path / "held", dict(PUBLIC, held=["free", "circuit"]), audits=(SERVED,)))
    of = rec["bound"]["of_record"]
    assert (of["held"], of["open"], of["K"]) == (["free", "circuit"], open_, K + 2 ** 23 + 1)
    assert of["premises"] == dict(NO_PREMISES, guarantees=found, open=open_)
    assert rec["periods_tight"]["conditional_on"] == open_ and rec["periods_tight"]["run"]["bits"] == K
    rec = OR.record(accepted(tmp_path / "free", dict(PUBLIC, held=["free"]), audits=(SERVED,)))
    assert rec["periods_tight"]["conditional_on"] == ["circuit", *open_]


def test_a_scans_tied_audits_are_the_ones_it_charges_and_else_every_audit_is(tmp_path):
    """Two audits, the served one and one with no advice: with the public naming none, each phase charges both; with
    scans[j].tied, each phase only its own."""
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="1/524288", windows=2, scans=[SCAN, SCAN])
    rec = OR.record(accepted(tmp_path / "all", two, windows=2, audits=(SERVED, NO_ADVICE)))
    assert rec["bound"]["of_record"]["advice"] == {"bits_per_phase": [2 ** 23 + 2] * 2, "tied": OR.TIED_ALL}
    assert rec["advice_charge"]["cap"]["ell_a"] == [2 ** 23, 0]
    tied = dict(two, scans=[dict(SCAN, tied=[0]), dict(SCAN, tied=[1])])
    of = OR.record(accepted(tmp_path / "tied", tied, windows=2, audits=(SERVED, NO_ADVICE)))["bound"]["of_record"]
    assert (of["advice"]["bits_per_phase"], of["advice"]["tied"], of["K"]) == (
        [2 ** 23 + 1, 1], "scans[j].tied", two["K"] + 2 ** 23 + 2)


def test_a_commitment_the_record_cant_read_leaves_no_figure_of_record_and_no_periods(tmp_path):
    out = accepted(tmp_path)
    case = json.loads((out / OR.CASE).read_text())
    (out / OR.CASE).write_text(json.dumps(dict(case, commitment="00")))
    rec = OR.record(out)
    assert rec["bound"]["of_record"]["error"].startswith(f"{OR.CASE} commitment: JSONDecodeError")
    assert rec["advice_charge"]["cap"] == rec["bound"]["of_record"]
    assert rec["periods"]["missing"].startswith("no figure of record") and "periods_tight" not in rec
    assert rec["bound"]["K"] == K and rec["bound"]["connections"]["count"] == 252


def test_w63k4_at_a_4_s_tick_charges_800_bits_a_phase_beside_its_advice_whatever_the_traffic(tmp_path):
    """Three execution phases of w63k4's schedule at a 4 s tick: its 4 lanes over T = 32 buckets carry at most 128
    connections a phase, each capped at 5 egress cells, so k_sched and k_fail charge 128 · ⌈log2 6⌉ = 384 bits a phase
    each, the period's check's terms however few connections were served; with the jitter T_j log2 d_jit = 32 at
    d_jit = 2, a phase's terms but ℓ_{a,j} and k_{skip,j} are 800 bits."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    scan = dict(SCAN, jitter=schedule["window"])
    public = dict(PUBLIC, K=3 * (768 + 9088 + 32), cells=3 * 384, schedules=3 * 384, windows=3, scans=[scan] * 3,
                  eps="3/1048576")
    rec = OR.record(accepted(tmp_path, public, windows=3, schedule=schedule))
    b = rec["bound"]
    assert [t["bits"] for t in b["terms"] if t.get("phase") is None] == [1152, 1152]
    jitter = [t["bits"] for t in b["terms"] if t["symbol"] == "T_j \\log_2 d_{jit}"]
    assert jitter == [32] * 3 and (1152 + 1152) // 3 + jitter[0] == 800
    assert [t.get("times") for t in b["delta_net"]["terms"]] == [None, 3, 3, 3]
    assert rec["periods"]["run"]["delta_net"] == OR.delta_net(Fraction(3, 2 ** 20), 3)
    assert b["connections"] == {"count": 3 * 128, "from": OR.CONNECTIONS}
    assert (rec["periods"]["phase_s"], rec["periods"]["run"]["seconds"], b["period_s"]) == ("128", "384", "384")
    assert rec["periods"]["week"]["node"]["parameters"]["connections"] == str(128 * math.ceil(7 * 86400 / 128))
    assert rec["periods"]["egress"]["capacity_bits_per_phase"] == 4 * 4 * 32 * 512 * 8


@pytest.mark.parametrize("drawn, entries", [(1, 896), (16, 0)])
def test_w63k4s_tied_audit_drawn_in_part_charges_its_slot_entries_and_drawn_whole_doesnt(tmp_path, drawn, entries):
    """Monday's defaults: w63k4 at a 4 s tick over three execution phases, each phase's tied V3c audit drawn subset:1
    (charged), so k_{skip,j} is its E_w = 896 slot-entry bits beside the schedule's and the jitter's 800: with
    ℓ_{a,j} = 0, both checks held and no open premise, 1,696 bits a phase of record, of the schedule's egress capacity
    of 4 lanes · 4 slots · 32 buckets of 512-byte cells a phase. Drawn whole, its 16 proof blocks proved, the entries
    and its δ leave the bound."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    scan = dict(SCAN, delta=f"1/{2 ** 64}", free=0, faulty_entries=entries, jitter=32)
    audit = {"delta": f"1/{2 ** 64}", "drawn": drawn, "units": 16, "wrong_units": 0}
    charged = 3 if drawn < 16 else 0
    public = dict(PUBLIC, K=3 * (800 + entries), cells=3 * 384, schedules=3 * 384, windows=3, scans=[scan] * 3,
                  audits=[audit] * 3, eps=str(Fraction(3 + charged, 2 ** 64)), held=["free", "circuit"])
    rec = OR.record(accepted(tmp_path, public, windows=3, schedule=schedule, audits=(SERVED,)))
    b, run = rec["bound"], rec["periods"]["run"]
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [entries] * 3, "entries_per_phase": [entries] * 3}
    assert b["delta"]["charged"] == str(Fraction(charged, 2 ** 64))
    assert (b["of_record"]["K"], b["of_record"]["open"], "periods_tight" in rec) == (3 * (800 + entries), [], False)
    capacity = 4 * 4 * 32 * 512 * 8
    assert run["per_phase"]["bits"] == 800 + entries and run["share_of_capacity"] == (800 + entries) / capacity
    assert rec["periods"]["year"]["datacenter"]["share_of_capacity"] == (800 + entries) / capacity
    assert rec["periods"]["egress"]["capacity_bits_per_phase"] == capacity == 2 ** 21


@pytest.mark.parametrize("caps", [[], [4, 4], [5] * (4 * 63)])
def test_a_case_whose_caps_arent_its_schedules_is_the_connections_error(tmp_path, caps):
    """The period's check refuses caps that aren't the schedule's (`Schedule.caps`), an empty list included, so a
    case's other caps beside an accepted answer leave the connections and the periods unfilled."""
    rec = OR.record(accepted(tmp_path, caps=caps))
    why = (f"{OR.CASE}: ValueError: the case's {len(caps)} caps aren't its schedule's (Schedule.caps: egress_cells = 4 "
           "for each of its lanes · m · T = 252 connections), which the period's check refuses")
    assert rec["bound"]["K"] == K and rec["bound"]["connections"] == {"error": why} and rec["periods"] == {"error": why}


def census_line(name: str, *fields: str) -> dict:
    d = next(d for d in json.loads((LB.CENSUS / name).read_text()) if d["role"] == "reference")
    digest = hashlib.sha512(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    return {"file": f"benchmarks/census/{name}", "id": d["id"], "read": {f: d[f] for f in fields},
            "sha512": digest.hexdigest()}


def test_the_runs_period_is_proved_and_every_monitoring_period_scaled_beside_one_phase(tmp_path):
    p = OR.record(accepted(tmp_path))["periods"]
    phase = Fraction(63 * 20, 1000)
    one = {"bits": KR, "share": KR / LB.WEIGHTS.bits, "share_of_capacity": KR / CAPACITY, "vacuous": False, "how": "proved",
           "delta_net": [{"symbol": "eps", "value": "1/1048576"}, {"symbol": "ε_ks", "times": 1},
                         {"symbol": "ε_bind", "times": 1}, {"symbol": "ε_tap", "times": 1}],
           "eps_per_phase_below": "1", "statement": LB.STATEMENTS["per_period"]}
    assert p["phase_s"] == str(phase) and p["run"] == one | {
        "seconds": str(phase), "phases": 1, "connections": 252, "per_phase": one}
    links = {"node": 1, "datacenter": LB.datacenter_links(), "world": LB.world_links()}
    rules = {"node": "run-to-period", "datacenter": "datacenter", "world": "world"}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase)
        assert (p[name]["seconds"], p[name]["phases"]) == (seconds, w)
        for where, n in links.items():
            c = p[name][where]
            assert (c["bits"], c["delta_net"], c["how"], c["rule"]) == (
                KR * w * n, OR.delta_net(Fraction(1, 2 ** 20) * w * n, w * n), "scaled", rules[where])
            assert c["delta_net"][0]["value"] == str(Fraction(w * n, 2 ** 20))
            assert c["vacuous"] == (w * n >= 2 ** 20) and c["share"] == KR * w * n / LB.WEIGHTS.bits
            assert c["share_of_capacity"] == KR / CAPACITY
            assert c["eps_per_phase_below"] == str(Fraction(1, w * n))
            assert c["parameters"] == {"phases": w, "phase_s": str(phase), "connections": str(252 * w * n), "links": n}
            assert c["per_phase"] == (one if where == "node" else {
                "bits": KR * n, "share": KR * n / LB.WEIGHTS.bits, "share_of_capacity": KR / CAPACITY,
                "delta_net": OR.delta_net(Fraction(n, 2 ** 20), n),
                "vacuous": n >= 2 ** 20, "eps_per_phase_below": str(Fraction(1, n)), "how": "scaled", "rule": where,
                "statement": OR.STATEMENTS[where]})
    assert p["week"]["datacenter"]["statement"] == LB.STATEMENTS["datacenter"]
    assert {k: r["sources"] for k, r in p["rules"].items()} == {
        "per-phase": [OR.FROM_BOUND], "run-to-period": [OR.FROM_BOUND, OR.FROM_SCHEDULE],
        "datacenter": [census_line("deployments.json", "machines")],
        "world": [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
                  census_line("deployments.json", "gpus_per_machine")]}
    assert p["rules"]["datacenter"]["sources"][0]["id"] == "dc-12500-machines-600s"
    assert {k: r["rule"] for k, r in p["rules"].items()} == OR.RULES


def test_eps_is_split_into_the_scans_and_the_charged_audits_delta_beside_each_cells_threshold(tmp_path):
    """The served defaults: the scan drawn whole at 2^-20, and a tied audit drawn subset:1 at δ = 1/2, whose δ the
    period's eps carries. The bound says which δ is which; a cell whose eps a phase is past its threshold is vacuous."""
    public = dict(PUBLIC, eps="524289/1048576", audits=[{"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}])
    rec = OR.record(accepted(tmp_path, public))
    assert rec["bound"]["delta"] == {"scan": "1/1048576", "charged": "1/2", "eps_per_phase": "524289/1048576",
                                     "is": OR.DELTA}
    run, year = rec["periods"]["run"], rec["periods"]["year"]["datacenter"]
    assert (run["vacuous"], run["eps_per_phase_below"]) == (False, "1")
    assert year["vacuous"] and Fraction(year["eps_per_phase_below"]) < Fraction(public["eps"])
    assert Fraction(year["eps_per_phase_below"]) == Fraction(1, year["parameters"]["phases"] * LB.datacenter_links())


def test_scans_at_two_deltas_are_the_bounds_error(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="3/1048576", windows=2,
               scans=[SCAN, dict(SCAN, delta="1/524288")])
    rec = OR.record(accepted(tmp_path, two, windows=2))
    assert rec["bound"]["error"].startswith("eps 3/1048576 isn't m · δ_scan plus the charged audits' δ")


def with_deltas(out: Path, **deltas: object) -> Path:
    """``out`` with its window report's rows (by key) stating each given δ."""
    f = out / "record" / "verdicts.json"
    doc = json.loads(f.read_text())
    for key, d in deltas.items():
        doc["protocols"][key]["delta"] = d
    f.write_text(json.dumps(doc))
    return out


@pytest.mark.parametrize("eps, stated, vacuous", [("1/1048576", "262145/524288", False),
                                                  ("524289/1048576", "524289/524288", True)])
def test_the_composites_error_sums_each_subprotocols_delta_with_its_source(tmp_path, eps, stated, vacuous):
    """Sampled proofs at its served δ = 1/2 and PoUW at 2^-20 from their rows, and the network certifier's eps from its
    bound: one tied audit's draw read at sampled proofs' 1/2 and again inside the network's eps is two terms. Their sum is
    stated as a lower bound on the error, inference-only's δ missing and the rest unvalued."""
    public = dict(PUBLIC, eps=eps, audits=[{"delta": str(Fraction(eps) - Fraction(1, 2 ** 20)), "drawn": 1,
                                            "units": 16, "wrong_units": 8}])
    rec = OR.record(with_deltas(accepted(tmp_path, public), integrity="1/2", work=2.0 ** -20))
    ce = rec["composite_error"]
    assert [(t["subprotocol"], t.get("delta")) for t in ce["terms"]] == [
        ("sampled proofs", "1/2"), ("PoUW", "1/1048576"), ("inference-only", None), ("the network certifier", eps)]
    assert ce["terms"][0]["from"] == "record/verdicts.json protocols.integrity.delta: its answer's integrity-profile δ"
    assert ce["terms"][2] == {"subprotocol": "inference-only", "missing": OR.NO_NCI_DELTA}
    assert ce["terms"][3]["from"].startswith(OR.EPS["source"])
    assert (ce["stated"], ce["vacuous"], ce["is"], ce["unvalued"]) == (stated, vacuous, OR.COMPOSITE, OR.COMPOSITE_UNVALUED)
    assert "counts in both" in OR.COMPOSITE and "lower bound" in OR.COMPOSITE_UNVALUED
    assert OR.summary(rec)["composite_error"] == {"stated": stated, "vacuous": vacuous}


def test_a_composite_term_the_run_doesnt_state_is_missing_and_one_that_isnt_a_number_its_error(tmp_path):
    rec = OR.record(with_deltas(accepted(tmp_path), integrity="half"))
    sampled, pouw, _nci, net = rec["composite_error"]["terms"]
    assert sampled["error"] == "its row's δ 'half' isn't a number" and pouw["missing"] == "its row states no δ"
    assert rec["composite_error"]["stated"] == net["delta"] == PUBLIC["eps"]


def test_a_node_cell_is_proved_where_the_runs_period_covers_its_monitoring_period(tmp_path):
    """An execution phase of 63 buckets of 10,000 s covers a week but not a month: the node's week is the run's K,
    proved, and its datacenter scaled from that by its machines; its month is scaled from the run."""
    p = OR.record(accepted(tmp_path, schedule=dict(SCHEDULE, bucket_ms=10_000_000)))["periods"]
    week, month = p["week"], p["month"]
    assert (week["node"]["bits"], week["node"]["how"], "rule" not in week["node"]) == (KR, "proved", True)
    assert week["node"]["parameters"] == {"phases": 1, "phase_s": "630000", "links": 1, "connections": "252"}
    n = LB.datacenter_links()
    assert (week["datacenter"]["bits"], week["datacenter"]["rule"]) == (KR * n, "datacenter")
    w = math.ceil(Fraction(LB.PERIODS["month"], 630000))
    assert (month["node"]["bits"], month["node"]["rule"], month["phases"]) == (KR * w, "run-to-period", w)


def test_a_scaled_cell_over_several_phases_rounds_up(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63 + 1, eps="1/524288", windows=2,
               scans=[SCAN, dict(SCAN, faulty_sets=1)])
    p = OR.record(accepted(tmp_path, two, windows=2))["periods"]
    kr = two["K"] + 2
    w = math.ceil(Fraction(LB.PERIODS["month"]) / Fraction(63, 50))
    assert w % 2 and p["run"]["bits"] == kr and p["run"]["per_phase"]["rule"] == "per-phase"
    assert p["month"]["node"]["bits"] == math.ceil(Fraction(kr, 2) * w) != Fraction(kr, 2) * w
    assert p["month"]["node"]["parameters"]["connections"] == str(252 * w)
    assert p["run"]["per_phase"]["bits"] == math.ceil(Fraction(kr, 2))


def test_a_case_of_another_period_or_none_leaves_the_periods_unfilled(tmp_path):
    out = accepted(tmp_path, windows=2)
    why = f"{OR.CASE}: ValueError: the case's period has 2 execution phases, and its answer 1"
    rec = OR.record(out)
    assert rec["periods"] == {"error": why} and rec["bound"]["connections"] == {"error": why}
    (out / OR.CASE).unlink()
    rec = OR.record(out)
    assert rec["bound"]["K"] == K and rec["periods"] == {
        "missing": f"no {OR.CASE}: the execution phase's length is its params' schedule"}


def test_the_record_leaves_the_node_as_written(tmp_path):
    out = accepted(tmp_path)
    f = tmp_path / "advice-charge.json"
    f.write_text(json.dumps(MEASURED))
    assert OR.main([str(out), "--advice-charge", str(f)]) == 0
    text = (out / "record" / "onsite.json").read_text()
    assert SV.public_doc(text) == text


def test_the_published_form_which_drops_windows_gives_the_same_record(tmp_path):
    full = OR.record(accepted(tmp_path / "full"))
    out = accepted(tmp_path / "published", {k: v for k, v in PUBLIC.items() if k != "windows"})
    case = json.loads((out / OR.CASE).read_text())
    case["params"].pop("windows")
    (out / OR.CASE).write_text(json.dumps(case))
    assert OR.record(out) == full


@pytest.mark.parametrize("public, why", [
    (dict(PUBLIC, K=K + 1), f"K is {K + 1}, and its terms sum to {K}"),
    ({k: v for k, v in PUBLIC.items() if k != "scans"}, "not NetworkCheck.Period's rows form"),
    (dict(PUBLIC, scans=[{k: v for k, v in SCAN.items() if k != "faulty_entries"}]),
     "the rows form's public lacks a term: KeyError: 'faulty_entries'"),
    (dict(PUBLIC, windows=2), "1 scans for the period's 2 execution phases"),
    (dict(PUBLIC, K=756 + 756, scans=[], windows=0), "the rows form's public has no scan"),
])
def test_a_public_that_isnt_the_rows_forms_sum_is_the_bounds_error(tmp_path, public, why):
    rec = OR.record(accepted(tmp_path, public))
    assert rec["bound"]["error"].startswith(why) and rec["periods"] == {"missing": "no bound"}


def test_a_test_runs_measured_advice_charge_is_beside_the_bound_as_that_runs(tmp_path, capsys):
    out = accepted(tmp_path)
    f = tmp_path / "advice-charge.json"
    f.write_text(json.dumps(MEASURED))
    assert OR.main([str(out), "--advice-charge", str(f)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["advice_charge"]["measured"] == {
        "label": MEASURED["label"], "run": "toy-test", "from": str(f), "bits_per_commit_period": MEASURED["bits"],
        "bits_per_phase": MEASURED["phase_bits"], "phases": OR.TEST_PHASES, "commit_periods_measured": 2,
        "finishes": MEASURED["finishes"],
        "prompt_len": MEASURED["prompt_len"], "output_len": MEASURED["output_len"]}
    assert (rec["advice_charge"]["served"], rec["advice_charge"]["charged"]["bits_per_phase"]) == (OR.ADVICE, [9088])
    assert json.loads(capsys.readouterr().out.strip()[7:])["advice_measured_on"] == "toy-test"
    f.write_text(json.dumps(MEASURED | {"phase_bits": None, "by_phase": {"missing": "no clock"}}))
    assert OR.main([str(out), "--advice-charge", str(f)]) == 0
    rec = json.loads((out / "record" / "onsite.json").read_text())
    assert rec["advice_charge"]["measured"]["bits_per_phase"] == {"missing": "no clock"}
    capsys.readouterr()
    f.write_text(json.dumps(MEASURED | {"format": "verity/other/v1"}))
    assert OR.main([str(out), "--advice-charge", str(f)]) == 1
    assert "not verity/advice-charge/v2" in capsys.readouterr().err


def test_the_run_is_of_record_only_with_the_onsite_composite_accepting(tmp_path):
    out = accepted(tmp_path)
    rec = OR.record(out)
    assert {k: v["verdict"] for k, v in rec["verdicts"].items()} == {
        "sampled proofs": "accept", "PoUW": "accept", "inference-only": "accept", "remote": "accept",
        "the network certifier": "accept", "onsite": "missing"}
    assert rec["verdicts"]["onsite"]["why"] == "the run has no composite of verity/remote-network/v0"
    assert rec["of_record"] == {"holds": False, "missing": ["onsite"], "not_accepting": []}
    (out / "composite" / "remote-network").mkdir()
    cases = remote_composite()["cases"] + [{"id": OR.NETWORK, "verdict": "accept", "params_sha512": "00",
                                            "commitment": "00"}]
    doc = remote_composite() | {"product": OR.ONSITE, "cases": cases, "verdict": {"id": OR.ONSITE, "verdict": "accept"}}
    (out / "composite" / "remote-network" / "composite.json").write_text(json.dumps(doc))
    assert OR.record(out)["of_record"] == {"holds": True, "missing": [], "not_accepting": []}
    (out / "composite" / "remote-network" / "composite.json").write_text(json.dumps(
        doc | {"missing": [OR.NETWORK], "verdict": {"id": OR.ONSITE, "verdict": "reject", "refusal": "components"}}))
    rec = OR.record(out)
    assert rec["verdicts"]["onsite"] == {"protocol": OR.ONSITE, "verdict": "reject", "refusal": "components",
                                         "from": "composite/remote-network/composite.json",
                                         "not_established": [OR.NETWORK]}
    assert rec["of_record"] == {"holds": False, "missing": [], "not_accepting": ["onsite"]}


def test_a_run_with_no_window_report_writes_no_record(tmp_path, capsys):
    assert OR.main([str(tmp_path)]) == 1
    assert "record/verdicts.json is missing" in capsys.readouterr().err and not (tmp_path / "record").exists()
