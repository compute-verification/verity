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
both checks the tight ℓ_{a,j} rests on and the lock (a toy one in Security's lock's place) lists every premise's
discharge, taking no assumption but a standing one, and otherwise charges ℓ_a + 1 bits for each tied audit of each
phase, ℓ_a the cap of the run's published advice registration (no figure where it published none, and an error where an
audit's own registration binds an advice), the tight K's cells beside it, conditional on what is open, by name. A tied
audit drawn in part charges its slot entries in k_{skip,j}; each cell states its
share of the schedule's egress capacity, and the served egress is missing. Sampled proofs' coins are live only where
every COINS line names the live challenger at a named peer; otherwise, or with none, live-coins is open by name, the
figure of record charges ℓ_a + 1 for each tied audit whatever else holds, and it, its cells and the sampled-proofs
verdict are conditional on the coins. One program a day is never claimed, open by name beside the day each ADVICE-LINE
says, whatever the lines say. The run's `publish`
leaves the record as written. The run is of record only when every verdict, the onsite composite's included,
accepted. Its caller's serving settings, S and MAX_TOKENS, both or neither, label the record and its line, missing where
it states none; runs' records at two settings go beside each other, each run's figures labelled with its own settings
and ℓ_a, and a record of another format, or stating no settings, is refused by path."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
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
#: the period's one compute audit's registration, V3c's class window's, which binds no advice or compiler
NO_ADVICE = {"format": "verity/registration/v2", "window": {"circuit": "cd" * 64}}
#: the run's advice registration as `served_verdicts.py public-registration` publishes it, its advice committed padded to
#: its compiler's cap ℓ_a = 2^23 (`advice_windows.pad`)
PUBLISHED = SV.public_registration({"row_bits": 256, "rows": 2 ** 23 // 256,
                                    "binding": {"advice": {"root": "ab" * 64, "bits": 2 ** 23}},
                                    "compiler": {"name": "c", "digest": "d", "cap_bits": 2 ** 23}})
#: K of record with both checks open and the one audit's ℓ_a + 1 bits for the one phase
KR = K + 2 ** 23 + 1
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
#: stand-ins for the theorems that discharge the premises `PREMISES` names no discharge of yet, and those premises as
#: they are
TOY = {n: f"Toy.{n}" for n, p in OR.PREMISES.items() if not p["discharged_by"]}
UNNAMED = {n: OR.PREMISES[n] for n in TOY}
#: the guarantees of `RESTS_ON` as they are, and those no statement of which is pinned yet
RESTS_ON = {g: dict(r) for g, r in OR.RESTS_ON.items()}
UNPINNED = [g for g, r in OR.RESTS_ON.items() if r["statement"] is None]
ACCEPTED = "NetTiming.SecurityProofs.PeriodLeakAccepted"
#: the theorem that discharges the free-input count's premise
(COUNTED,) = OR.PREMISES["free-count"]["discharged_by"]
#: the Property of the Lean audit over the network certifier's signed records that discharges schedules-covered
(COVERED,) = OR.PREMISES["schedules-covered"]["discharged_by"]


def discharged() -> dict:
    """``premises`` of a lock listing every discharge and each guarantee the tight K rests on at the statement it
    names, none with an assumption."""
    found = {n: {"is": p["is"], "discharged_by": list(p["discharged_by"]), "missing": [], "unpinned": {}, "assumes": [],
                 "takes": []} for n, p in OR.PREMISES.items()}
    return {"is": OR.DISCHARGE,
            "from": "Security/lean-audit.json and Security/Properties/lean-audit.json guarantees and reads",
            "premises": found, "standing": OR.STANDING, "counts": OR.COUNTS, "open": [], "unread": [],
            "rests_on": {g: {"assumptions": [], "statement": r["statement"], "takes": r["takes"]}
                         for g, r in OR.RESTS_ON.items()}}


def lock_doc(listed: dict[str, list[str]], stated: dict[str, str | None] | None = None) -> dict:
    """A lock listing each guarantee of ``listed`` with its named assumptions, each of `RESTS_ON` and `DISCHARGES` at
    the statement ``stated`` gives it, else at the one `RESTS_ON` or `DISCHARGES` pins, among its reads."""
    pins = OR.RESTS_ON | OR.DISCHARGES
    stated = {g: r["statement"] for g, r in pins.items()} | (stated or {})
    guarantees = {g: {"assumptions": a} | ({"signature": [f"{g} : {g}.Stated"]} if g in stated else {})
                  for g, a in listed.items()}
    reads = {"Definitions.Toy": {"definitions": {f"{g}.Stated": h for g, h in stated.items() if h is not None}}}
    return {"guarantees": guarantees, "reads": reads}


def write_lock(f: Path, given: dict[str, list[str] | None] | None = None,
               stated: dict[str, str | None] | None = None) -> None:
    """A lock at ``f`` listing every discharge of `PREMISES` and each guarantee the tight K rests on with no named
    assumption, each guarantee of ``given`` instead with the assumptions given, or unlisted where None, and each of
    `RESTS_ON` at the statement it names unless ``stated`` gives another."""
    names = [t for p in OR.PREMISES.values() for t in p["discharged_by"]] + list(OR.RESTS_ON)
    listed = {g: [] for g in names} | (given or {})
    f.write_text(json.dumps(lock_doc({g: a for g, a in listed.items() if a is not None}, stated)))


@pytest.fixture(autouse=True)
def lock(tmp_path_factory, monkeypatch) -> Path:
    """A lock in Security's lock's place discharging every premise, each of `UNNAMED` by its `TOY` stand-in, each
    discharge at a stand-in statement `DISCHARGES` pins, taking nothing, and listing each guarantee of `RESTS_ON` at the
    statement it names, each of `UNPINNED` at a stand-in's."""
    for name, toy in TOY.items():
        monkeypatch.setitem(OR.PREMISES, name, dict(OR.PREMISES[name], discharged_by=(toy,)))
    for t in (t for p in OR.PREMISES.values() for t in p["discharged_by"]):
        monkeypatch.setitem(OR.DISCHARGES, t, {"statement": f"toy-{t}", "takes": {}})
    for g in UNPINNED:
        monkeypatch.setitem(OR.RESTS_ON, g, dict(OR.RESTS_ON[g], statement=f"toy-{g}"))
    f = tmp_path_factory.mktemp("lock") / "lean-audit.json"
    write_lock(f)
    monkeypatch.setattr(OR, "LOCKS", (f,))
    return f


def answer(public: dict | None, verdict: str = "accept") -> dict:
    return {"protocol": OR.NETWORK, "verdict": verdict, "reason": None, "commitment": "ab" * 64, "public": public,
            "profile": {"link": "served", "windows": 1, "placement": {"trusted-host": "the certifier on the gate's host"}}}


def network_row(public: dict | None) -> dict:
    return {"protocol": OR.NETWORK, "owner": "network-accounting", "verdict": "accept", "reason": None,
            "commitment": "ab" * 64, "quantity": public, "delta": None, "epsilon": None, "runs": [],
            "answer": answer(public)}


#: the sampled-proofs part of a COINS line naming the live challenger, as served_verdicts.sh says it under PP_CHALLENGER
GATE = "the challenger gate:research@toy-gate (peer toy-peer)"


def coins_line(said: str, lead: str = "") -> str:
    """The run's COINS line as served_verdicts.sh says it, its sampled-proofs part ``said``, led by ``lead``."""
    return (f"COINS {lead}sampled proofs: {said}; PoUW H1: not served (H1_GATE empty); the bandwidth audit's: "
            "vy-auditor-2's bw_party, inside PoUS's piece")


def served_out(tmp_path: Path, *, network: dict | None = None, work: dict | None = None, case: dict | None = None,
               composites: dict[str, dict] | None = None, coins: str | None = GATE) -> Path:
    """A toy $OUT: sampled proofs accepted, PoUW and the network certifier missing unless given, the given composites
    and certifier case, and its COINS line, its sampled-proofs part ``coins`` (none where None)."""
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
    if coins is not None:
        (out / "verdicts.txt").write_text(coins_line(coins) + "\n")
    return out


def remote_composite(verdict: str = "accept") -> dict:
    cases = [{"id": i, "verdict": "accept", "params_sha512": "00", "commitment": "00"}
             for i in (OR.INFERENCE_ONLY, "verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v1")]
    return {"product": OR.REMOTE, "commitment": "00", "missing": [], "exceptions": [], "cases": cases,
            "verdict": {"id": OR.REMOTE, "verdict": verdict}}


def commitment(*audits: dict) -> str:
    return json.dumps({"audits": list(audits), "windows": [{}]}).encode().hex()


def accepted(tmp_path: Path, public: dict = PUBLIC, windows: int = 1, caps: list[int] | None = None,
             schedule: dict = SCHEDULE, audits: tuple[dict, ...] = (NO_ADVICE,),
             registration: dict | None = PUBLISHED, coins: str | None = GATE) -> Path:
    """A run whose network certifier accepted ``public``, its case the period's of ``windows`` execution phases at
    ``schedule``, whose caps are the schedule's (`Schedule.caps`) unless given, and whose commitment holds ``audits``'
    registrations, with ``registration`` its published advice registration, none where None, and ``coins`` its COINS
    line's sampled-proofs part (`served_out`)."""
    pouw = {"protocol": "verity/pouw/period/v1", "owner": "compute-accounting", "verdict": "accept", "reason": None}
    if caps is None:
        caps = [schedule["egress_cells"]] * (schedule["lanes"] * windows * schedule["window"])
    out = served_out(tmp_path, network=network_row(public), work=pouw,
                     case={"commitment": commitment(*audits), "evidence": "00",
                           "params": {"windows": windows, "cells": caps, "schedule": schedule}},
                     composites={"remote": remote_composite()}, coins=coins)
    (out / "machine.txt").write_text(f"ubuntu\n{NODE}\n{TREE}\n96\nserve dir /serve/d\n")
    with open(out / "verdicts.txt", "a") as f:
        f.write(f'THE NETWORK CERTIFIER {{"rc": 1, "verdict": "error"}}\nRECORD {{}}\n{LINE}\n'
                "COMPOSITE CASE verity/certifier/v0 accept\n")
    if registration is not None:
        (out / OR.REGISTRATION).write_text(json.dumps(registration))
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
    assert b["of_record"] == {"K": KR, "held": [], "open": ["free", "circuit"], "is": OR.OPEN, "premises": discharged(),
                              "advice": {"bits_per_phase": [2 ** 23 + 1], "tied": OR.TIED_ALL}}
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [0], "entries_per_phase": [0]}
    assert rec["advice_charge"]["cap"] == {"ell_a": [2 ** 23], "is": OR.CAP}
    assert "A cap, in no term" in OR.CAP and "`CAP`" in OR.OPEN and "`DISCHARGE`" in OR.OPEN
    assert rec["periods"]["run"]["bits"] == KR and rec["periods_tight"]["run"]["bits"] == K
    assert rec["periods_tight"]["conditional_on"] == ["free", "circuit"]
    assert rec["periods"]["egress"] == {"capacity_bits_per_phase": CAPACITY, "is": OR.EGRESS,
                                        "served": {"missing": OR.SERVED_EGRESS}}


def test_the_served_audits_cap_is_charged_while_a_check_is_open_and_the_tight_K_is_beside_it(tmp_path, capsys):
    """The served shape: one V3c audit, its class window binding no advice, tied to the one phase's scan, and the run's
    advice registration published padded to its compiler's cap ℓ_a = 2^23. With neither check held, the figure of
    record charges ℓ_a + 1 beside ℓ_{a,j}, and every cell is that figure's; the tight K's cells are beside them,
    conditional on both checks; the ONSITE line gives both."""
    out = accepted(tmp_path)
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
    rec = OR.record(accepted(tmp_path, dict(PUBLIC, held=held)))
    of = rec["bound"]["of_record"]
    assert (of["held"], of["open"]) == ([c for c in OR.CHECKS if c not in open_], open_)
    assert of["K"] == (K if not open_ else K + 2 ** 23 + 1)
    assert ("periods_tight" in rec) == bool(open_) and rec["periods"]["run"]["bits"] == of["K"]


def held_both(out: Path) -> dict:
    """The figure of record of ``out``'s served audit, both checks held."""
    return OR.record(accepted(out, dict(PUBLIC, held=["free", "circuit"])))


def test_with_no_discharge_in_the_lock_every_premise_is_open_by_name_though_both_checks_are_held(
        tmp_path, lock, monkeypatch):
    """A lock listing the pinned guarantees the tight K rests on and none of the theorems that discharge its premises,
    with PeriodTiedCount, not yet stated, unlisted, and the premises with no discharge named: every premise is open, by
    name, and so is PeriodTiedCount, so the figure of record charges ℓ_a + 1 for the tied audit, and the tight K's cells
    beside it are conditional on each."""
    for name, premise in UNNAMED.items():
        monkeypatch.setitem(OR.PREMISES, name, premise)
    for g in UNPINNED:
        monkeypatch.setitem(OR.RESTS_ON, g, RESTS_ON[g])
    lock.write_text(json.dumps(lock_doc({g: [] for g in OR.RESTS_ON if g not in UNPINNED})))
    rec = held_both(tmp_path)
    of = rec["bound"]["of_record"]
    assert UNPINNED == ["NetTiming.SecurityProofs.PeriodTiedCount"]
    assert (of["held"], of["open"], of["K"]) == (["free", "circuit"], [*OR.PREMISES, *UNPINNED], K + 2 ** 23 + 1)
    assert [p["missing"] for p in of["premises"]["premises"].values()] == [
        list(p["discharged_by"]) for p in OR.PREMISES.values()]
    assert [of["premises"]["premises"][n]["discharged_by"] for n in UNNAMED] == [[]] * len(UNNAMED)
    assert of["premises"]["rests_on"][UNPINNED[0]] == {"missing": "not in the lock",
                                                       "takes": RESTS_ON[UNPINNED[0]]["takes"]}
    assert rec["periods_tight"]["conditional_on"] == [*OR.PREMISES, *UNPINNED]
    assert of["premises"]["counts"] == OR.COUNTS and not set(OR.COUNTS) & set(of["open"])
    assert rec["periods_tight"]["run"]["bits"] == K
    assert all(f in OR.PREMISES["sampled-leg-bridge"]["is"]
               for f in ("SampledCodeRuns", "SessionKS,", "SessionRuns", "ListWithin", "Binds"))
    assert "Python" in OR.DISCHARGE
    assert "no opening reaches the verifier" in OR.PREMISES["session-wires"]["is"]
    tags = OR.PREMISES["session-tags"]["is"]
    assert "circuitOf at Tags.circuit" in tags and "Tags.circuit.withCoins c" in tags


@pytest.mark.parametrize("given, open_", [
    ({}, []),
    ({"Properties.NetworkPeriod.rowsHeld": ["NetTiming.Assumptions.SampledCodeRuns"]}, ["held-performed"]),
    ({"Properties.NetworkPeriod.rowsHeld": ["Verity.Assumptions.TrustedHost"]}, []),
    ({COUNTED: None}, ["free-count"]),
    ({COVERED: None}, ["schedules-covered"]),
    ({COVERED: ["Verity.Assumptions.TrustedHost"]}, []),
    ({ACCEPTED: ["NetTiming.Assumptions.SampledCodeRuns"]}, ["NetTiming.Assumptions.SampledCodeRuns"]),
    ({ACCEPTED: ["Verity.Assumptions.TrustedHost"]}, []),
    ({ACCEPTED: ["Accounting.Covered"]}, []),
    ({ACCEPTED: ["NetTiming.TapOpens"]}, []),
    ({ACCEPTED: None}, [ACCEPTED])])
def test_the_tight_K_is_of_record_only_where_every_premise_is_discharged_and_none_but_a_standing_assumption_taken(
        tmp_path, lock, given, open_):
    """Both checks held: a premise whose discharge the lock doesn't list, or lists taking an assumption that doesn't
    stand (SampledCodeRuns, a premise about our own Python), is open, and so is an assumption of a guarantee the tight
    K rests on that is neither standing nor named in what it takes, by its name, and that guarantee where the lock
    doesn't list it. trusted-host and certifier-coverage stand, and a premise its takes names is that premise's: so
    schedules-covered is open by its name until the Lean audit's Property is in the lock, and discharged once it is,
    resting on trusted-host alone. With one check open too, the tight K is conditional on both."""
    write_lock(lock, given)
    rec = held_both(tmp_path / "held")
    of = rec["bound"]["of_record"]
    assert (of["open"], of["K"], "periods_tight" in rec) == (open_, K + 2 ** 23 + 1 if open_ else K, bool(open_))
    assert of["premises"]["open"] == open_
    p = of["premises"]["premises"]
    assert p["held-performed"]["assumes"] == (["NetTiming.Assumptions.SampledCodeRuns"] if "held-performed" in open_
                                              else [])
    assert p["free-count"]["missing"] == ([COUNTED] if "free-count" in open_ else [])
    assert p["schedules-covered"]["missing"] == ([COVERED] if "schedules-covered" in open_ else [])
    rec = OR.record(accepted(tmp_path / "free", dict(PUBLIC, held=["free"])))
    assert rec["periods_tight"]["conditional_on"] == ["circuit", *open_]


#: COINS lines' sampled-proofs parts that name no live challenger, as served_verdicts.sh says them
DEVELOPMENT = {
    "served": (f"a local development challenger session on {NODE}, the served node, not an auditor's (PP_CHALLENGER "
               "local)"),
    "deferred": ("a local development challenger session on the follow-on's node, where step 6 defers V3c's audit, not "
                 "an auditor's (PP_CHALLENGER local)"),
    "follow-on": ("a local development challenger session on cpu-6, the follow-on's node, where it runs V3c's audit, "
                  "not an auditor's (PP_CHALLENGER local)"),
    "no peer": "the challenger gate:research@toy-gate (peer ?)",
    "not a gate": "the challenger vy-challenger-1 (peer toy-peer)",
}


def with_coins(tmp_path: Path, said: list[str]) -> dict:
    """The record, written by `main`, of a run with both checks held whose verdicts.txt opens with the lines
    ``said``."""
    out = accepted(tmp_path, dict(PUBLIC, held=["free", "circuit"]), coins=None)
    (out / "verdicts.txt").write_text("".join(x + "\n" for x in said) + (out / "verdicts.txt").read_text())
    assert OR.main([str(out)]) == 0
    return json.loads((out / "record" / "onsite.json").read_text())


@pytest.mark.parametrize("said", [
    *([coins_line(s)] for s in DEVELOPMENT.values()),
    [f"COINS sampled proofs: {DEVELOPMENT['follow-on']}; the other protocols' as the served node's COINS line "
     "names them"],
    [coins_line(GATE), coins_line(DEVELOPMENT["follow-on"])],
    ["COINS the bandwidth audit's: vy-auditor-2's bw_party, inside PoUS's piece"],
    [],
])
def test_coins_no_line_names_live_leave_the_figure_of_record_and_each_cell_conditional_on_them(tmp_path, capsys, said):
    """A run whose COINS lines name a local development challenger session (the served node's, a deferral's or its
    follow-on's), a gate at no named peer, another challenger or no sampled proofs' coins, or that published none, has
    its coins open, live-coins by name: the sampled-proofs verdict, the run's being of record, the figure of record and
    each of its cells are conditional on them, and the figure of record charges ℓ_a + 1 for the tied audit though both
    checks are held and every premise discharged, the tight K beside it conditional on the coins alone."""
    rec, cond = with_coins(tmp_path, said), [OR.COINS_OPEN]
    parts = [x.partition("sampled proofs: ")[2].partition(";")[0] or None for x in said]
    assert rec["coins"] == {"open": True, "sampled_proofs": parts, "from": "verdicts.txt COINS", "is": OR.COINS} | (
        {} if said else {"missing": OR.NO_COINS})
    of = rec["bound"]["of_record"]
    assert rec["verdicts"]["sampled proofs"]["conditional_on"] == rec["of_record"]["conditional_on"] == cond
    assert rec["composite_error"]["conditional_on"] == cond
    assert (of["held"], of["open"], of["conditional_on"], of["K"]) == (["free", "circuit"], cond, cond, KR)
    assert rec["periods"]["conditional_on"] == rec["periods_tight"]["conditional_on"] == cond
    assert (rec["periods"]["run"]["bits"], rec["periods_tight"]["run"]["bits"]) == (KR, K)
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["conditional_on"], line["K"], line["K_tight"], line["open"]) == (cond, KR, K, cond)
    assert all(x in OR.COINS for x in ("sampled-proofs verdict", "E_w", "never as a live draw's"))


@pytest.mark.parametrize("lead", ["", "Pearl+'s epoch coins are in-process and the record claims no Pearl+ verdict; "])
def test_coins_a_line_names_live_add_no_condition_and_the_tight_K_is_of_record(tmp_path, capsys, lead):
    """A COINS line naming the live challenger (PP_CHALLENGER gate:USER@HOST at a named peer), whatever leads it, leaves
    the coins closed: nothing is conditional on them, and with both checks held and every premise discharged the tight
    K is of record. A record written before the record read the coins is conditional on them in its line."""
    rec = with_coins(tmp_path, [coins_line(GATE, lead)])
    assert rec["coins"] == {"open": False, "sampled_proofs": [GATE], "from": "verdicts.txt COINS", "is": OR.COINS}
    of = rec["bound"]["of_record"]
    assert not any("conditional_on" in x for x in (rec["verdicts"]["sampled proofs"], rec["of_record"], of,
                                                   rec["periods"], rec["composite_error"]))
    assert (of["K"], of["open"], "periods_tight" in rec) == (K, [], False)
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["conditional_on"], line["K"], line["open"]) == ([], K, [])
    assert OR.summary({k: v for k, v in rec.items() if k != "coins"})["conditional_on"] == [OR.COINS_OPEN]


#: the run's ADVICE-LINE as served_verdicts.py says it on a serve-openai --advice run, declaring B for day {}
DECLARED = ("ADVICE-LINE: B (compiler digest 0123456789abcdef…, its advice toy, ℓ_a 32 bits) registered by the server "
            "once, before its first window, the registration log's builder and the compiler prove_path.py recomputes "
            "from serve.json, and declared day {}'s one program; every closed window's advice committed: PASS")
NO_B = ("ADVICE-LINE: FAIL: no advice_registration.json: the server registered no B (serve-openai without "
        "--advice)")


@pytest.mark.parametrize("days", [["0"], ["3", "3"], [None], []])
def test_one_program_a_day_is_never_claimed_but_open_by_name_beside_the_day_each_advice_line_says(tmp_path, capsys,
                                                                                                    days):
    """Whatever the run's ADVICE-LINEs say (a day's declaration of B, two of them, a line naming no day) or where it
    published none (missing), the record doesn't claim one program a day: it is open by name in the record and its
    line, on both counts (R2's day isn't the challenger's signed time, and the audit reads the node's copy of the
    registration log, not the challenger's signed one), saying the day's registration and its refusals still run. The
    figures don't read it: with the coins live and every premise discharged the tight K is of record."""
    rec = with_coins(tmp_path, [coins_line(GATE), *(NO_B if d is None else DECLARED.format(d) for d in days)])
    assert rec["one_program_a_day"] == {"open": True, "advice_line_day": days, "from": "verdicts.txt ADVICE-LINE",
                                        "is": OR.DAY} | ({} if days else {"missing": OR.NO_ADVICE_LINE})
    assert (rec["bound"]["of_record"]["K"], rec["bound"]["of_record"]["open"]) == (K, [])
    line = json.loads(capsys.readouterr().out.strip()[7:])
    assert (line["not_claimed"], line["conditional_on"], line["K"]) == ([OR.DAY_OPEN], [], K)
    assert all(x in OR.DAY for x in ("doesn't claim", "R2's day isn't yet the challenger's signed time",
                                     "doesn't yet read the challenger's signed registration log",
                                     "(duplicate-declaration)", "(R2-declared)", "which no verdict here does"))


def test_a_discharge_the_properties_lock_lists_counts_as_one_the_guarantees_lock_does(tmp_path, lock, monkeypatch):
    """The Properties' lock holds the theorems that discharge premises at the deployed code, the guarantees' lock the
    guarantee the tight K rests on: the record reads both, and with the Properties' lock left out each premise is open."""
    discharges = [t for p in OR.PREMISES.values() for t in p["discharged_by"]]
    write_lock(lock, dict.fromkeys(discharges))
    props = tmp_path / "properties-lean-audit.json"
    props.write_text(json.dumps(lock_doc({t: [] for t in discharges})))
    monkeypatch.setattr(OR, "LOCKS", (lock, props))
    assert held_both(tmp_path / "both")["bound"]["of_record"]["open"] == []
    monkeypatch.setattr(OR, "LOCKS", (lock,))
    assert held_both(tmp_path / "one")["bound"]["of_record"]["open"] == list(OR.PREMISES)


@pytest.mark.parametrize("text, error", [(None, "FileNotFoundError"), ("{", "JSONDecodeError"), ("{}", "KeyError")])
def test_a_lock_that_cant_be_read_lists_nothing_and_is_named_so_every_premise_it_would_discharge_is_open(
        tmp_path, lock, monkeypatch, text, error):
    """The Properties' lock missing, not JSON, or with no guarantees: the record states it under ``unread``, by its path
    in the tree and with why, and reads the other lock, so only the premise whose discharge that lock alone would list
    is open and the figure of record charges the cap, never a crash."""
    write_lock(lock, {COUNTED: None})
    props = tmp_path / "tree" / "Security" / "Properties" / "lean-audit.json"
    props.parent.mkdir(parents=True)
    if text is not None:
        props.write_text(text)
    monkeypatch.setattr(OR, "LOCKS", (lock, props))
    of = held_both(tmp_path / "out")["bound"]["of_record"]
    assert (of["open"], of["K"]) == (["free-count"], KR)
    (why,) = of["premises"]["unread"]
    assert why.startswith(f"{props}: {error}: ")
    (why,) = OR.premises((lock, props), root=tmp_path / "tree")["unread"]
    assert why.startswith(f"Security/Properties/lean-audit.json: {error}: ") and str(tmp_path) not in why


def test_an_assumption_a_premise_stands_for_is_open_only_while_that_premise_is(tmp_path, lock, monkeypatch):
    """A guarantee the tight K rests on takes the wires premise as a named hypothesis, which the lock lists: named in
    what it takes, by the wires premise, nothing is open while that is discharged; with one of its discharges missing,
    the premise is open, by its own name alone; not named there, the assumption is open by its own name."""
    takes = dict(OR.RESTS_ON[ACCEPTED]["takes"], **{"NetTiming.WiresFit": "wires-fit"})
    monkeypatch.setitem(OR.RESTS_ON, ACCEPTED, dict(OR.RESTS_ON[ACCEPTED], takes=takes))
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"]})
    assert held_both(tmp_path / "a")["bound"]["of_record"]["open"] == []
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"], "Properties.NetworkCertifier.circuitWiresFit": None})
    assert held_both(tmp_path / "b")["bound"]["of_record"]["open"] == ["wires-fit"]
    monkeypatch.setitem(OR.RESTS_ON, ACCEPTED, RESTS_ON[ACCEPTED])
    write_lock(lock, {ACCEPTED: ["NetTiming.WiresFit"]})
    assert held_both(tmp_path / "c")["bound"]["of_record"]["open"] == ["NetTiming.WiresFit"]


def test_a_discharge_counts_only_at_the_statement_its_pin_names(tmp_path, lock, monkeypatch):
    """The lock records no antecedent, and a Property may take a deployment's premise inline (`ScanScopeAcc` takes
    `StmtFlattens` and `ScopeMatches`): a discharge the lock lists but `DISCHARGES` doesn't pin, or pins at another
    statement, leaves its premise open with the lock's digest to read it at, and pinned there it discharges it."""
    stated = "ab" * 32
    monkeypatch.delitem(OR.DISCHARGES, COUNTED)
    write_lock(lock, stated={COUNTED: stated})
    of = held_both(tmp_path / "unpinned")["bound"]["of_record"]
    assert (of["open"], of["K"], of["premises"]["premises"]["free-count"]["unpinned"]) == (
        ["free-count"], KR, {COUNTED: stated})
    monkeypatch.setitem(OR.DISCHARGES, COUNTED, {"statement": "ff" * 32, "takes": {}})
    assert held_both(tmp_path / "restated")["bound"]["of_record"]["open"] == ["free-count"]
    monkeypatch.setitem(OR.DISCHARGES, COUNTED, {"statement": stated, "takes": {}})
    assert held_both(tmp_path / "pinned")["bound"]["of_record"]["open"] == []


def test_a_discharge_that_takes_a_premise_discharges_only_once_that_premise_is_and_two_never_each_other(
        tmp_path, lock, monkeypatch):
    """The wires premise's discharge takes the deployment's scope check, a named assumption of its, by the sampled
    leg's bridge: nothing is open while that premise is discharged, both are while it isn't, and two premises whose
    discharges take each other stay open, each discharging nothing."""
    wired, bridge = OR.PREMISES["wires-fit"]["discharged_by"][0], TOY["sampled-leg-bridge"]
    scope = "Properties.NetworkScan.ScopeMatches"
    monkeypatch.setitem(OR.DISCHARGES, wired, dict(OR.DISCHARGES[wired], takes={scope: "sampled-leg-bridge"}))
    write_lock(lock, {wired: [scope]})
    of = held_both(tmp_path / "a")["bound"]["of_record"]
    assert (of["open"], of["premises"]["premises"]["wires-fit"]["takes"]) == ([], ["sampled-leg-bridge"])
    write_lock(lock, {wired: [scope], bridge: None})
    assert held_both(tmp_path / "b")["bound"]["of_record"]["open"] == ["wires-fit", "sampled-leg-bridge"]
    monkeypatch.setitem(OR.DISCHARGES, bridge, dict(OR.DISCHARGES[bridge], takes={"hw": "wires-fit"}))
    write_lock(lock, {wired: [scope]})
    of = held_both(tmp_path / "c")["bound"]["of_record"]
    assert (of["open"], of["K"]) == (["wires-fit", "sampled-leg-bridge"], KR)
    assert of["premises"]["premises"]["sampled-leg-bridge"]["assumes"] == []


@pytest.mark.parametrize("stated", ["ff" * 32, None])
def test_a_guarantee_the_tight_K_rests_on_is_open_by_name_where_the_lock_states_it_otherwise(tmp_path, lock, stated):
    """The lock records a guarantee's named assumptions, never its antecedents, so a statement other than the one whose
    hypotheses and antecedents `RESTS_ON` names (another digest among the lock's reads, or none) may take one the
    record doesn't name: that guarantee is open, by its name, with why, until it is read again and pinned."""
    write_lock(lock, stated={ACCEPTED: stated})
    rec = held_both(tmp_path)
    of = rec["bound"]["of_record"]
    assert (of["open"], of["K"]) == ([ACCEPTED], KR)
    assert of["premises"]["rests_on"][ACCEPTED] == {"assumptions": [], "statement": stated, "restated": OR.RESTATED,
                                                    "takes": OR.RESTS_ON[ACCEPTED]["takes"]}
    assert rec["periods_tight"]["conditional_on"] == [ACCEPTED]


def test_a_guarantee_the_tight_K_rests_on_with_no_statement_pinned_is_open_even_where_the_lock_lists_it(
        tmp_path, lock, monkeypatch):
    """PeriodTiedCount, listed before what it takes was read at its statement: open by its name, whatever the lock
    states it as."""
    (g,) = UNPINNED
    monkeypatch.setitem(OR.RESTS_ON, g, RESTS_ON[g])
    write_lock(lock, stated={g: "toy"})
    of = held_both(tmp_path)["bound"]["of_record"]
    assert of["open"] == [g] and of["premises"]["rests_on"][g]["restated"] == OR.RESTATED


def test_each_guarantee_the_tight_K_rests_on_is_named_by_what_it_takes_at_its_statement_in_security_lock(monkeypatch):
    """Security's lock records no guarantee's antecedents, so `RESTS_ON` names each hypothesis and antecedent of each
    statement by hand, read at the statement's digest among the lock's reads: each guarantee of `RESTS_ON` the tree's
    lock lists is there at that digest, with no named assumption it doesn't name or that doesn't stand, and each Prop it
    takes is a premise's or standing. A hypothesis or antecedent a statement gains changes its digest, and fails here.
    So does a discharge `DISCHARGES` pins, in either lock, and it pins only theorems `PREMISES` names."""
    monkeypatch.undo()
    locks = {f.relative_to(OR.ROOT): json.loads(f.read_text()) for f in OR.LOCKS}
    assert ACCEPTED in locks[Path("Security/lean-audit.json")]["guarantees"]
    assert set(OR.DISCHARGES) <= {t for p in OR.PREMISES.values() for t in p["discharged_by"]}
    for g, r in (OR.RESTS_ON | OR.DISCHARGES).items():
        pins = "RESTS_ON" if g in OR.RESTS_ON else "DISCHARGES"
        assert set(r["takes"].values()) <= set(OR.PREMISES) | set(OR.STANDING) | set(OR.COUNTS), g
        for name, lock in locks.items():
            if g not in lock["guarantees"]:
                continue
            got = OR.statement(lock, g)
            assert got is not None and got == r["statement"], (
                f"{g}'s statement in {name} is {got}, not {r['statement']}, the one onsite_record.{pins} names what it "
                f"takes at: read its hypotheses, antecedents and the conditions its good event reads again, name each in "
                f"its takes with the premise (PREMISES), standing assumption (STANDING) or term of what K counts (COUNTS) "
                f"that covers it, and pin {got}")
            assert set(lock["guarantees"][g]["assumptions"]) <= set(r["takes"]) | set(OR.STANDING), g


def test_a_scans_tied_audits_are_the_ones_it_charges_and_else_every_audit_is(tmp_path):
    """Two audits, each charged the run's cap: with the public naming none, each phase charges both; with
    scans[j].tied, each phase only those it names."""
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="1/524288", windows=2, scans=[SCAN, SCAN])
    rec = OR.record(accepted(tmp_path / "all", two, windows=2, audits=(NO_ADVICE, NO_ADVICE)))
    assert rec["bound"]["of_record"]["advice"] == {"bits_per_phase": [2 * (2 ** 23 + 1)] * 2, "tied": OR.TIED_ALL}
    assert rec["advice_charge"]["cap"]["ell_a"] == [2 ** 23, 2 ** 23]
    tied = dict(two, scans=[dict(SCAN, tied=[0, 1]), dict(SCAN, tied=[1])])
    of = OR.record(accepted(tmp_path / "tied", tied, windows=2, audits=(NO_ADVICE, NO_ADVICE)))["bound"]["of_record"]
    assert (of["advice"]["bits_per_phase"], of["advice"]["tied"], of["K"]) == (
        [2 * (2 ** 23 + 1), 2 ** 23 + 1], "scans[j].tied", two["K"] + 3 * (2 ** 23 + 1))


@pytest.mark.parametrize("binds", [{"advice": {"bits": 2 ** 23, "root": "ef" * 64}},
                                   {"compiler": {"digest": "01" * 64, "name": "B"}}])
def test_an_audit_whose_registration_binds_an_advice_or_a_compiler_leaves_no_figure_of_record(tmp_path, binds):
    """An audit's own advice is in no term of K and isn't the run's cap: the period takes only a circuit the verifier
    registered (`records_audit.py period` refuses the rest), so such an audit is the figure of record's error."""
    bound = {**NO_ADVICE, "window": {**NO_ADVICE["window"], **binds}}
    rec = OR.record(accepted(tmp_path, audits=(NO_ADVICE, bound)))
    assert rec["bound"]["of_record"] == rec["advice_charge"]["cap"] == {
        "error": f"{OR.CASE} commitment: ValueError: the registrations of audits [1] bind an advice or a compiler, whose "
                 "bits no term of K charges (records_audit.py period refuses them)"}
    assert rec["periods"]["missing"].startswith("no figure of record") and rec["bound"]["K"] == K


def test_a_run_that_published_no_advice_registration_or_one_unpadded_has_no_figure_of_record(tmp_path):
    """ℓ_a is the cap of the run's published advice registration (`REGISTRATION`), never 0 for want of one: with none
    the figure of record is missing, and one whose advice isn't padded to its compiler's cap is its error. The tight K
    stays the bound's."""
    rec = OR.record(accepted(tmp_path / "none", registration=None))
    assert rec["bound"]["of_record"] == rec["advice_charge"]["cap"] == {"missing": OR.NO_REGISTRATION}
    assert rec["periods"] == {"missing": f"no figure of record: {OR.NO_REGISTRATION}"} and rec["bound"]["K"] == K
    unpadded = {**PUBLISHED, "binding": {"advice": {"root": "ab" * 64, "bits": 392}}}
    rec = OR.record(accepted(tmp_path / "unpadded", registration=unpadded))
    assert rec["bound"]["of_record"] == {
        "error": f"{OR.REGISTRATION}: ValueError: its advice's length 392 isn't padded to its compiler's cap {2 ** 23}"}


@pytest.mark.parametrize("registration, cap", [
    ({k: v for k, v in PUBLISHED.items() if k != "compiler"}, None), ({**PUBLISHED, "compiler": {"name": "c"}}, None),
    ({**PUBLISHED, "compiler": {"cap_bits": 0}}, 0), ({**PUBLISHED, "compiler": {"cap_bits": -8}}, -8),
    ({**PUBLISHED, "compiler": {"cap_bits": True}}, True), ({**PUBLISHED, "compiler": {"cap_bits": "8388608"}}, "8388608"),
    ({**PUBLISHED, "compiler": {"cap_bits": 8.0}}, 8.0), ([PUBLISHED], None)])
def test_a_published_advice_registration_stating_no_cap_of_at_least_1_bit_refuses_the_figure_of_record_by_name(
        tmp_path, registration, cap):
    """No compiler, or a compiler whose cap_bits is missing, 0, negative, or not a whole number, would charge each tied
    audit's advice 0 bits while a check or a premise is open: the record refuses the figure of record by name, never
    charging 0, whatever the order its pieces land in. The tight K stays the bound's, its cells beside no figure."""
    rec = OR.record(accepted(tmp_path, registration=registration))
    why = (f"{OR.REGISTRATION}: ValueError: its compiler states no cap_bits of at least 1 bit ({cap!r}), the ℓ_a the "
           "figure of record charges each tied audit while a check or a premise is open, so it charges none rather "
           "than 0")
    assert rec["bound"]["of_record"] == rec["advice_charge"]["cap"] == {"error": why}
    assert rec["periods"] == {"missing": f"no figure of record: {why}"} and rec["bound"]["K"] == K


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
    """w63k4's schedule with a toy tied audit: w63k4 at a 4 s tick over three execution phases, each phase's tied audit
    one of 16 proof blocks (V3c's registration has millions) drawn subset:1 (charged), its draw read at δ_i = 2^-64
    with a bound of 15 beside its answer's 8 at 1/2, so k_{skip,j} is its E_w = 896 slot-entry bits beside the
    schedule's and the jitter's 800: with ℓ_{a,j} = 0, both checks held and no open premise, 1,696 bits a phase of
    record, of the schedule's egress capacity of 4 lanes · 4 slots · 32 buckets of 512-byte cells a phase. eps is each
    phase's scan's 2^-64 and its charged audit's δ_i, never the answer's 1/2. Drawn whole, every one of its proof
    blocks proved, the entries and its δ_i leave the bound. With a check open, each phase's figure of record adds the
    run's advice cap ℓ_a + 1 = 2^23 + 1, 8,390,305 bits a phase drawn in part."""
    sys.path.insert(0, str(HERE.parent.parent / "network_traces"))
    import scan_zk as SZ
    schedule = dataclasses.asdict(SZ.W63K4.params) | {"bucket_ms": 4000}
    scan = dict(SCAN, delta=f"1/{2 ** 64}", free=0, faulty_entries=entries, jitter=32)
    audit = {"delta": "1/2", "drawn": drawn, "units": 16, "wrong_units": 8 if drawn < 16 else 0}
    charged = [dict(audit, delta=f"1/{2 ** 64}", wrong_units=15)] * 3 if drawn < 16 else []
    public = dict(PUBLIC, K=3 * (800 + entries), cells=3 * 384, schedules=3 * 384, windows=3, scans=[scan] * 3,
                  audits=[audit] * 3, charged=charged, eps=str(Fraction(3 + len(charged), 2 ** 64)),
                  held=["free", "circuit"])
    rec = OR.record(accepted(tmp_path, public, windows=3, schedule=schedule))
    b, run = rec["bound"], rec["periods"]["run"]
    assert b["skip"] == {"is": OR.SKIP, "bits_per_phase": [entries] * 3, "entries_per_phase": [entries] * 3}
    assert b["delta"]["charged"] == str(Fraction(len(charged), 2 ** 64)) and b["delta"]["charged_audits"] == charged
    assert (b["of_record"]["K"], b["of_record"]["open"], "periods_tight" in rec) == (3 * (800 + entries), [], False)
    capacity = 4 * 4 * 32 * 512 * 8
    assert run["per_phase"]["bits"] == 800 + entries and run["share_of_capacity"] == (800 + entries) / capacity
    assert rec["periods"]["year"]["datacenter"]["share_of_capacity"] == (800 + entries) / capacity
    assert rec["periods"]["egress"]["capacity_bits_per_phase"] == capacity == 2 ** 21
    opened = OR.record(accepted(tmp_path / "open", dict(public, held=["free"]), windows=3, schedule=schedule))
    assert opened["bound"]["of_record"]["open"] == ["circuit"]
    assert opened["periods"]["run"]["per_phase"]["bits"] == 800 + entries + 2 ** 23 + 1 == (
        8_390_305 if drawn < 16 else 8_389_409)


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
    one = {"bits": KR, "share": KR / LB.WEIGHTS.bits, "share_served": KR / LB.SERVED.bits,
           "share_of_capacity": KR / CAPACITY, "vacuous": False, "how": "proved",
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
            assert c["share_served"] == KR * w * n / LB.SERVED.bits and c["share_of_capacity"] == KR / CAPACITY
            assert c["eps_per_phase_below"] == str(Fraction(1, w * n))
            assert c["parameters"] == {"phases": w, "phase_s": str(phase), "connections": str(252 * w * n), "links": n}
            assert c["per_phase"] == (one if where == "node" else {
                "bits": KR * n, "share": KR * n / LB.WEIGHTS.bits, "share_served": KR * n / LB.SERVED.bits,
                "share_of_capacity": KR / CAPACITY,
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
    world = census_line("adversary.json", "model", "total_params", "weights_bytes", "weights_datatype")
    assert p["weights"] == {
        "share": {"of": LB.WEIGHTS.label, "is": OR.WEIGHTS, "bits": 8 * world["read"]["weights_bytes"], "source": world},
        "share_served": {"of": "the demo's served model (Llama-3.2-1B) in BF16", "bits": LB.SERVED.bits}}
    assert world["read"]["model"] == "DeepSeek-V4-Pro" and world["id"] == "world-inference-2027"


def test_eps_is_split_into_the_scans_and_the_charged_audits_delta_beside_each_cells_threshold(tmp_path):
    """The served defaults: the scan drawn whole at 2^-20, and a tied audit drawn subset:1 at δ = 1/2, whose δ the
    period's eps carries. The bound says which δ is which; a cell whose eps a phase is past its threshold is vacuous."""
    public = dict(PUBLIC, eps="524289/1048576", audits=[{"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}])
    rec = OR.record(accepted(tmp_path, public))
    assert rec["bound"]["delta"] == {"scan": "1/1048576", "charged": "1/2", "eps_per_phase": "524289/1048576",
                                     "is": OR.DELTA, "charged_audits": {"missing": OR.NO_READINGS}}
    run, year = rec["periods"]["run"], rec["periods"]["year"]["datacenter"]
    assert (run["vacuous"], run["eps_per_phase_below"]) == (False, "1")
    assert year["vacuous"] and Fraction(year["eps_per_phase_below"]) < Fraction(public["eps"])
    assert Fraction(year["eps_per_phase_below"]) == Fraction(1, year["parameters"]["phases"] * LB.datacenter_links())


def test_scans_at_two_deltas_are_the_bounds_error(tmp_path):
    two = dict(PUBLIC, K=PUBLIC["K"] + 9088 + 63, eps="3/1048576", windows=2,
               scans=[SCAN, dict(SCAN, delta="1/524288")])
    rec = OR.record(accepted(tmp_path, two, windows=2))
    assert rec["bound"]["error"].startswith("eps 3/1048576 isn't m · δ_scan plus the charged audits' δ")


def test_an_eps_that_isnt_the_scans_delta_plus_each_charged_readings_delta_i_is_the_bounds_error(tmp_path):
    """Where the public lists each charged audit's reading at δ_i, eps is m · δ_scan plus those δ_i exactly: one that
    counts the audit's answer's δ = 1/2 in their place is the bound's error, so the record never reports it."""
    audit = {"delta": "1/2", "drawn": 1, "units": 16, "wrong_units": 8}
    public = dict(PUBLIC, eps="524289/1048576", audits=[audit],
                  charged=[dict(audit, delta=f"1/{2 ** 64}", wrong_units=15)])
    rec = OR.record(accepted(tmp_path, public))
    assert rec["bound"]["error"] == ("eps 524289/1048576 isn't m · δ_scan plus each charged reading's δ_i: m · δ_scan "
                                     f"is 1/1048576, and the readings' δ_i are ['1/{2 ** 64}']")
    assert "of_record" not in rec["bound"]


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


def test_a_composite_term_stated_as_a_float_is_the_ratio_it_was_given_as(tmp_path):
    """2^-64 as a float is 1/2^64 and 0.01 is 1/100, never a nearby decimal or ratio."""
    sampled, pouw, _nci, _net = OR.record(with_deltas(accepted(tmp_path), integrity=0.01, work=2.0 ** -64))[
        "composite_error"]["terms"]
    assert (sampled["delta"], pouw["delta"]) == ("1/100", f"1/{2 ** 64}")


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
    kr = two["K"] + 2 * (2 ** 23 + 1)
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


def test_the_record_lists_the_re_proof_inputs_and_which_of_outs_the_run_wrote(tmp_path):
    """What a later re-proof of the sampled units needs is listed by name, whatever the verdicts: the serve dir's, which
    the record can't see, by their paths there, naming no window, and OUT's with whether the run wrote each; how each must
    be kept is stated beside them."""
    out = accepted(tmp_path)
    got = OR.record(out)["reproof"]
    assert got["kept"] == OR.REPROOF_KEPT and list(got["inputs"]) == list(OR.REPROOF)
    serve = {k for k, v in got["inputs"].items() if v["in"] == "serve dir"}
    assert {"values", "openings", "draw", "coins", "advice", "weights"} <= serve
    assert not any("present" in got["inputs"][k] for k in serve) and not SV.NAMED.search(json.dumps(got))
    written = {OR.REGISTRATION, str(OR.CASE), "record/verdicts.json"}
    assert {v["path"]: v["present"] for v in got["inputs"].values() if v["in"] == "OUT"} == {
        v[1]: v[1] in written for v in OR.REPROOF.values() if v[0] == "OUT"}
    for at, path, _ in OR.REPROOF.values():
        if at == "OUT":
            (out / path).parent.mkdir(parents=True, exist_ok=True)
            (out / path).touch()
    assert all(v["present"] for v in OR.record(out)["reproof"]["inputs"].values() if v["in"] == "OUT")
    assert OR.record(served_out(tmp_path / "rehearsal"))["reproof"]["kept"] == OR.REPROOF_KEPT
    assert got["copy"] == {"missing": OR.NO_COPY}


def test_the_serve_dirs_re_proof_inputs_are_checked_against_their_copys_listing_by_path_alone(tmp_path):
    """The serve dir's inputs, witness, are copied off the node: the record names the copy's node and path and says of
    each input whether the listing holds each file its path names and at least one under each directory it names, and
    nothing of the listing's window names; one file short is that input not copied, and a copy that isn't NODE:/PATH
    or a listing it can't read is the copy's error."""
    def files(path: str) -> list[str]:
        out = []
        for part in path.split(", "):
            m = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", part)
            for p in [m[1] + x + m[3] for x in m[2].split(",")] if m else [part]:
                p = p.replace("<K>", "7").replace("<class>", "c64")
                out.append(p + "m0-private.bin" if p.endswith("/") else p)
        return out

    out = accepted(tmp_path)
    serve = [p for at, p, _ in OR.REPROOF.values() if at == "serve dir"]
    listing = tmp_path / "private" / "listing.txt"
    listing.parent.mkdir()
    listing.write_text("\n".join(f for p in serve for f in files(p)) + "\n")
    got = OR.record(out, copy="vy-nebius-cpu-3:/home/research/keep/reproof", listing=listing)["reproof"]
    assert got["copy"] == {"node": "vy-nebius-cpu-3", "path": "/home/research/keep/reproof", "not_copied": []}
    assert all(v["copied"] for v in got["inputs"].values() if v["in"] == "serve dir")
    assert not any("copied" in v for v in got["inputs"].values() if v["in"] == "OUT")
    assert not SV.NAMED.search(json.dumps(got)) and "c64" not in json.dumps(got)
    listing.write_text("\n".join(f for p in serve for f in files(p) if not f.endswith("/receipt.json")) + "\n")
    got = OR.record(out, copy="vy-nebius-cpu-3:/home/research/keep/reproof", listing=listing)["reproof"]
    assert got["copy"]["not_copied"] == ["draw"] and got["inputs"]["values"]["copied"]
    assert OR.record(out, copy="vy-nebius-cpu-3", listing=listing)["reproof"]["copy"] == {
        "error": "--reproof-copy 'vy-nebius-cpu-3' isn't NODE:/PATH"}
    assert OR.record(out, copy="n:/p", listing=tmp_path / "none.txt")["reproof"]["copy"] == {
        "error": "the copy's listing (--reproof-listing) can't be read: FileNotFoundError"}


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


def test_a_remote_run_whose_report_drops_the_certifier_takes_its_period_checks_answer_and_else_the_drop(tmp_path):
    """Remote's report drops the network certifier whatever its pieces hold; with the split certifier behind the front
    its period check's answer, beside its case under OUT/pieces/certifier, is the verdict and the bound, as onsite's
    report composes it."""
    out = accepted(tmp_path)
    report = json.loads((out / "record" / "verdicts.json").read_text())
    report["protocols"]["communication"] = SV.dropped_row("communication", "remote")
    (out / "record" / "verdicts.json").write_text(json.dumps(report))
    rec = OR.record(out)
    assert rec["verdicts"]["the network certifier"] == {
        "verdict": "missing", "why": report["protocols"]["communication"]["why"],
        "from": "record/verdicts.json protocols.communication"}
    assert "missing" in rec["bound"]
    (out / OR.CASE.parent / "answer.json").write_text(json.dumps(answer(PUBLIC)))
    rec = OR.record(out)
    assert rec["verdicts"]["the network certifier"] == {"protocol": OR.NETWORK, "verdict": "accept",
                                                        "from": "pieces/certifier/answer.json"}
    assert rec["bound"]["K"] == K and rec["placement"] == {"trusted-host": "the certifier on the gate's host"}
    assert rec == OR.record(accepted(tmp_path / "onsite")) | {"verdicts": rec["verdicts"]}


def test_a_run_with_no_window_report_writes_no_record(tmp_path, capsys):
    assert OR.main([str(tmp_path)]) == 1
    assert "record/verdicts.json is missing" in capsys.readouterr().err and not (tmp_path / "record").exists()


def published(cap: int) -> dict:
    """The run's advice registration as published, its advice padded to its compiler's cap ℓ_a = ``cap``."""
    return SV.public_registration({"row_bits": 256, "rows": -(-cap // 256),
                                   "binding": {"advice": {"root": "ab" * 64, "bits": cap}},
                                   "compiler": {"name": "c", "digest": "d", "cap_bits": cap}})


#: two runs' serving settings, (S, MAX_TOKENS), each with the ℓ_a its registration states
SETTINGS = ((1, 1, 1328), (16, 4, 7432))


def test_the_callers_serving_settings_label_the_record_and_its_line_and_are_missing_where_it_states_none(
        tmp_path, capsys):
    out = accepted(tmp_path)
    assert OR.main([str(out), "--window-steps", "16", "--max-tokens", "4"]) == 0
    text = (out / "record" / "onsite.json").read_text()
    assert json.loads(text)["serving"] == {"window_steps": 16, "max_tokens": 4, "is": OR.SERVING}
    line = capsys.readouterr().out.strip()
    assert json.loads(line[7:])["serving"] == {"window_steps": 16, "max_tokens": 4}
    assert SV.public_doc(text) == text and SV.public_line(line) == line
    assert OR.main([str(out)]) == 0
    assert json.loads((out / "record" / "onsite.json").read_text())["serving"] == {"missing": OR.NO_SERVING}
    assert json.loads(capsys.readouterr().out.strip()[7:])["serving"] is None


@pytest.mark.parametrize("args", [["--window-steps", "16"], ["--max-tokens", "4"],
                                  ["--window-steps", "0", "--max-tokens", "4"],
                                  ["--window-steps", "x", "--max-tokens", "4"]])
def test_serving_settings_are_given_both_or_neither_each_a_whole_number_of_at_least_1(tmp_path, args):
    out = accepted(tmp_path)
    with pytest.raises(SystemExit) as e:
        OR.main([str(out), *args])
    assert e.value.code == 2 and not (out / "record" / "onsite.json").exists()


def test_runs_at_two_serving_settings_go_beside_each_other_each_figure_labelled_with_its_own(tmp_path, capsys):
    paths = []
    for s, m, cap in SETTINGS:
        out = accepted(tmp_path / f"s{s}", registration=published(cap))
        assert OR.main([str(out), "--window-steps", str(s), "--max-tokens", str(m)]) == 0
        paths.append(out / "record" / "onsite.json")
    capsys.readouterr()
    f = tmp_path / "beside" / "onsite-beside.json"
    assert OR.main(["--beside", *map(str, paths), "--to", str(f)]) == 0
    doc = json.loads(f.read_text())
    assert (doc["format"], doc["is"]) == (OR.BESIDE, OR.BESIDE_IS)
    runs = doc["runs"]
    assert [(r["record"], r["run"], r["tree"], r["serving"], r["ell_a"]) for r in runs] == [
        (str(p), "toy-run", TREE, {"window_steps": s, "max_tokens": m}, [cap]) for p, (s, m, cap) in zip(paths, SETTINGS)]
    assert [(r["of_record"], r["conditional_on"], r["not_claimed"], r["K"], r["K_tight"], r["open"]) for r in runs] == [
        (False, [], [OR.DAY_OPEN], K + cap + 1, K, ["free", "circuit"]) for _, _, cap in SETTINGS]
    for p, r in zip(paths, runs):
        rec = json.loads(p.read_text())
        for name in ("periods", "periods_tight"):
            run = rec[name]["run"]
            assert r[name]["run"] == {"bits": run["bits"], "share": run["share"], "phases": 1,
                                      "per_phase": run["per_phase"]["bits"]}
            assert all(r[name][n][w] == {k: rec[name][n][w][k] for k in ("bits", "share", "how")}
                       for n in LB.PERIODS for w in ("node", "datacenter", "world"))
        assert r["periods_tight"]["conditional_on"] == ["free", "circuit"] and "conditional_on" not in r["periods"]
    assert runs[1]["periods"]["run"]["bits"] - runs[0]["periods"]["run"]["bits"] == 7432 - 1328
    line = capsys.readouterr().out.strip()
    assert line.startswith("ONSITE-BESIDE ") and [(x["serving"], x["K"]) for x in json.loads(line[14:])] == [
        (r["serving"], r["K"]) for r in runs]


def test_beside_refuses_by_path_a_record_of_another_format_or_one_stating_no_serving_settings(tmp_path, capsys):
    out = accepted(tmp_path)
    assert OR.main([str(out)]) == 0
    rec, other, f = out / "record" / "onsite.json", tmp_path / "other.json", tmp_path / "beside.json"
    other.write_text(json.dumps({"format": OR.BESIDE}))
    capsys.readouterr()
    assert OR.main(["--beside", str(rec), "--to", str(f)]) == 1
    assert capsys.readouterr().err.strip() == f"ONSITE-BESIDE: not written: {rec} {OR.UNLABELLED}"
    assert OR.main(["--beside", str(other), "--to", str(f)]) == 1
    assert f"{other} is {OR.BESIDE!r}, not {OR.FORMAT}" in capsys.readouterr().err
    assert not f.exists()


@pytest.mark.parametrize("args", [[], ["--beside", "r.json"], ["out", "--beside", "r.json", "--to", "b.json"],
                                  ["--beside", "r.json", "--to", "b.json", "--window-steps", "1", "--max-tokens", "1"]])
def test_beside_takes_the_records_and_to_alone_and_otherwise_the_record_takes_an_out(args):
    with pytest.raises(SystemExit) as e:
        OR.main(args)
    assert e.value.code == 2
