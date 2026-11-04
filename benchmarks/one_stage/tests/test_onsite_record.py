"""`onsite_record.py` on toy served runs' $OUTs. A run whose network certifier, PoUW and composites are missing, as a
rehearsal's, records each of them missing with its why, and no bound and no period figure. A run whose network
certifier accepted in the rows form records K term by term in the notation table's symbols, each naming the public
fields it sums, with its overcount as a note, the seconds it covers, the line that published its verdict, the tree, and
the connections its schedule's caps count (other caps are the connections' error); its run's own period proved, and
each monitoring period's node proved where the run's period covers it and else scaled from the run, and the
datacenter's and the world's scaled from the node's, each scaled cell naming its rule, whose sources (the census lines
by SHA-512) the record states, beside its figure over one execution phase. A public whose terms don't sum to K, or in
the clock form, is the bound's error. The served run's advice charge stays out of the record, the registered width
stands as its upper bound, and a test run's measurement is stated beside the bound as that run's. The run's `publish`
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
#: T = 63 buckets of 20 ms: an execution phase of 63/50 s
SCHEDULE = {"bucket_ms": 20, "cell_bytes": 512, "delay": 4, "egress_cells": 4, "every": 63, "ingress_buckets": 63,
            "ingress_cells": 4, "lanes": 4, "queue": 16, "slots": 8, "window": 63}
MEASURED = {"format": "verity/advice-charge/v1", "label": "measured on test run toy-test: …", "run": "toy-test",
            "commit_periods": [{"period": 0, "bits": 480}, {"period": 1, "bits": 2390}],
            "bits": {"count": 2, "min": 480, "median": 1435, "mean": 1435.0, "max": 2390, "total": 2870},
            "prompt_len": {"count": 5}, "output_len": {"count": 5}, "finishes": {"eos": 3, "length": 2, "abort": 0}}
TREE = "ab12" * 10
#: the certifier's published verdict line, after an earlier attempt's
LINE = 'THE NETWORK CERTIFIER {"rc": 0, "verdict": "accept"}'


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


def accepted(tmp_path: Path, public: dict = PUBLIC, windows: int = 1, caps: list[int] | None = None,
             schedule: dict = SCHEDULE) -> Path:
    """A run whose network certifier accepted ``public``, its case the period's of ``windows`` execution phases at
    ``schedule``, whose caps are the schedule's (`Schedule.caps`) unless given."""
    pouw = {"protocol": "verity/pouw/period/v1", "owner": "compute-accounting", "verdict": "accept", "reason": None}
    if caps is None:
        caps = [schedule["egress_cells"]] * (schedule["lanes"] * windows * schedule["window"])
    out = served_out(tmp_path, network=network_row(public), work=pouw,
                     case={"commitment": "00", "evidence": "00",
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
    assert rec["advice_charge"] == {"served": OR.ADVICE, "registered": {"missing": "no bound"},
                                    "measured": {"missing": OR.NO_MEASUREMENT}}
    assert numbers({k: rec[k] for k in ("bound", "periods", "verdicts", "advice_charge")}) == []
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
        ("\\ell_{a,i}", 0, 9088, ["scans[0].free"]),
        ("k_{skip,i}", 0, 0, ["scans[0].faulty_sets", "scans[0].faulty_bits", "scans[0].faulty_entries"]),
        ("k_i \\log_2 d_{jit}", 0, 63, ["scans[0].jitter"])]
    assert sum(t["bits"] for t in b["terms"]) == b["K"]
    assert b["notes"] == [OR.OVERCOUNT] and "k_sched" in OR.OVERCOUNT and "k_fail" in OR.OVERCOUNT
    assert b["connections"] == {"count": 4 * 63, "from": OR.CONNECTIONS}
    assert rec["placement"] == {"trusted-host": "the certifier on the gate's host"}
    assert rec["advice_charge"]["registered"] == {
        "label": "registered width, an upper bound on the honest charge", "bits_per_phase": [9088],
        "share": 9088 / K, "lean": "scans[i].free"}


def test_w63k4_at_a_4_s_tick_charges_768_bits_a_link_window_for_its_128_connections_whatever_the_traffic(tmp_path):
    """Three execution phases of w63k4's schedule at a 4 s tick: its 4 lanes over T = 32 buckets carry at most 128
    connections a phase, each capped at 5 egress cells, so k_sched and k_fail charge 128 · ⌈log2 6⌉ = 384 bits a phase
    each, the period's check's terms however few connections were served."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    public = dict(PUBLIC, K=3 * (768 + 9088 + 63), cells=3 * 384, schedules=3 * 384, windows=3, scans=[SCAN] * 3)
    rec = OR.record(accepted(tmp_path, public, windows=3, schedule=schedule))
    b = rec["bound"]
    assert [t["bits"] for t in b["terms"] if t.get("phase") is None] == [1152, 1152]
    assert [t.get("times") for t in b["delta_net"]["terms"]] == [None, 3, 3, 3]
    assert rec["periods"]["run"]["delta_net"] == OR.delta_net(Fraction(1, 2 ** 20), 3)
    assert b["connections"] == {"count": 3 * 128, "from": OR.CONNECTIONS}
    assert (rec["periods"]["phase_s"], rec["periods"]["run"]["seconds"], b["period_s"]) == ("128", "384", "384")
    assert rec["periods"]["week"]["node"]["parameters"]["connections"] == str(128 * math.ceil(7 * 86400 / 128))


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
    one = {"bits": K, "share": K / LB.WEIGHTS.bits, "vacuous": False, "how": "proved",
           "delta_net": [{"symbol": "eps", "value": "1/1048576"}, {"symbol": "ε_ks", "times": 1},
                         {"symbol": "ε_bind", "times": 1}, {"symbol": "ε_tap", "times": 1}],
           "statement": LB.STATEMENTS["per_period"]}
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
                K * w * n, OR.delta_net(Fraction(1, 2 ** 20) * w * n, w * n), "scaled", rules[where])
            assert c["delta_net"][0]["value"] == str(Fraction(w * n, 2 ** 20))
            assert c["vacuous"] == (w * n >= 2 ** 20) and c["share"] == K * w * n / LB.WEIGHTS.bits
            assert c["parameters"] == {"phases": w, "phase_s": str(phase), "connections": str(252 * w * n), "links": n}
            assert c["per_phase"] == (one if where == "node" else {
                "bits": K * n, "share": K * n / LB.WEIGHTS.bits, "delta_net": OR.delta_net(Fraction(n, 2 ** 20), n),
                "vacuous": n >= 2 ** 20, "how": "scaled", "rule": where, "statement": OR.STATEMENTS[where]})
    assert p["week"]["datacenter"]["statement"] == LB.STATEMENTS["datacenter"]
    assert {k: r["sources"] for k, r in p["rules"].items()} == {
        "per-phase": [OR.FROM_BOUND], "run-to-period": [OR.FROM_BOUND, OR.FROM_SCHEDULE],
        "datacenter": [census_line("deployments.json", "machines")],
        "world": [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
                  census_line("deployments.json", "gpus_per_machine")]}
    assert p["rules"]["datacenter"]["sources"][0]["id"] == "dc-12500-machines-600s"
    assert {k: r["rule"] for k, r in p["rules"].items()} == OR.RULES


def test_a_node_cell_is_proved_where_the_runs_period_covers_its_monitoring_period(tmp_path):
    """An execution phase of 63 buckets of 10,000 s covers a week but not a month: the node's week is the run's K,
    proved, and its datacenter scaled from that by its machines; its month is scaled from the run."""
    p = OR.record(accepted(tmp_path, schedule=dict(SCHEDULE, bucket_ms=10_000_000)))["periods"]
    week, month = p["week"], p["month"]
    assert (week["node"]["bits"], week["node"]["how"], "rule" not in week["node"]) == (K, "proved", True)
    assert week["node"]["parameters"] == {"phases": 1, "phase_s": "630000", "links": 1, "connections": "252"}
    n = LB.datacenter_links()
    assert (week["datacenter"]["bits"], week["datacenter"]["rule"]) == (K * n, "datacenter")
    w = math.ceil(Fraction(LB.PERIODS["month"], 630000))
    assert (month["node"]["bits"], month["node"]["rule"], month["phases"]) == (K * w, "run-to-period", w)


def test_a_scaled_cell_over_several_phases_rounds_up(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63 + 1, eps="1/524288", windows=2,
               scans=[SCAN, dict(SCAN, faulty_sets=1)])
    p = OR.record(accepted(tmp_path, two, windows=2))["periods"]
    w = math.ceil(Fraction(LB.PERIODS["month"]) / Fraction(63, 50))
    assert w % 2 and p["run"]["bits"] == two["K"] and p["run"]["per_phase"]["rule"] == "per-phase"
    assert p["month"]["node"]["bits"] == math.ceil(Fraction(two["K"], 2) * w) != Fraction(two["K"], 2) * w
    assert p["month"]["node"]["parameters"]["connections"] == str(252 * w)
    assert p["run"]["per_phase"]["bits"] == math.ceil(Fraction(two["K"], 2))


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
        "commit_periods_measured": 2, "finishes": MEASURED["finishes"], "prompt_len": MEASURED["prompt_len"],
        "output_len": MEASURED["output_len"]}
    assert (rec["advice_charge"]["served"], rec["advice_charge"]["registered"]["bits_per_phase"]) == (OR.ADVICE, [9088])
    assert json.loads(capsys.readouterr().out.strip()[7:])["advice_measured_on"] == "toy-test"
    f.write_text(json.dumps(MEASURED | {"format": "verity/other/v1"}))
    assert OR.main([str(out), "--advice-charge", str(f)]) == 1
    assert "not verity/advice-charge/v1" in capsys.readouterr().err


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
