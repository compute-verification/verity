"""`onsite.py`'s window pick, the auditor's record of the run's deployment it hands the piece (its local gate's copy),
its reading of the front's and the piece's released outcomes, and its record: each zone's verdicts and K in
`PeriodLeakAccepted`'s terms, the lab's zones summed, labelled simulated. The front and the piece
themselves need the lab's namespaces and the chain's builds, so they run as recorded runs (README, "The onsite path");
here their outputs are written by hand in the shapes they release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import shutil
import tarfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
TRACES = HERE.parents[0] / "network_traces"
#: the egress cells of the gate's w63k4 schedule
W63K4_CELLS = 5


@pytest.fixture(scope="module")
def onsite():
    spec = importlib.util.spec_from_file_location("network_sim_onsite", HERE / "onsite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_window_is_the_one_commit_window_holding_the_fronts_requests(onsite, tmp_path):
    lines(tmp_path / "requests.jsonl", [{"user": f"user-{i}", "windows": [3]} for i in range(4)])
    assert onsite.window(tmp_path) == 3
    lines(tmp_path / "requests.jsonl", [{"user": "user-0", "windows": [0]}, {"user": "user-1", "windows": [0, 1]}])
    with pytest.raises(SystemExit, match="lie in 2 Commit periods, not one"):
        onsite.window(tmp_path)


def test_the_scan_class_is_the_one_the_fronts_gate_ran(onsite):
    request = "warden-rehearse --peer p --port 1 --scan --prompts --tokens 4 --scan-class w63k4 --window-per-group"
    assert onsite.scan_class({"request": request}) == "w63k4"
    assert onsite.scan_class({"request": "warden-rehearse --scan --prompts"}) == onsite.GATE_CLASS == "w63"
    assert onsite.scan_class(None) == "w63"


def test_the_gates_outcome_is_accepted_only_when_its_check_accepted_with_exit_0(onsite):
    assert onsite.gate_outcome(None) == {"accepted": None, "why": "the front wrote no session.json"}
    assert onsite.gate_outcome({"refused": "the gate refuses it: busy"}) == {"accepted": False,
                                                                            "why": "the gate refuses it: busy"}
    v = {"accepted": True, "windows": 1, "requests": 4, "problems": 0, "failed": [], "refused": 0}
    assert onsite.gate_outcome({"rc": 0, "verdict": v})["accepted"] is True
    assert onsite.gate_outcome({"rc": 2, "verdict": v})["accepted"] is False
    assert onsite.gate_outcome({"rc": 0, "verdict": v | {"accepted": False}})["accepted"] is False


def bundle(path: Path, run: str, params: dict, windows: int, requests: int) -> None:
    """A front's public bundle: the gate run's ``end.json`` and ``out/deployment.json``."""
    files = {f"{run}/end.json": {"run": run, "rc": 0, "verdict": {"accepted": True, "windows": windows,
                                                                  "requests": requests}},
             f"{run}/out/deployment.json": {"params": params}}
    with tarfile.open(path, "w:gz") as tar:
        for name, doc in files.items():
            data = json.dumps(doc).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def zone(out: Path, b: str, params: dict, *, answer: dict | None, refused: str | None = None) -> None:
    front = out / b / "front"
    front.mkdir(parents=True)
    request = "warden-rehearse --scan --prompts --tokens 4 --scan-class w63k4 --window-per-group"
    (front / "session.json").write_text(json.dumps({
        "rc": 0, "run": f"c-{b}", "request": request, "ended": "the rehearsal ended",
        "verdict": {"accepted": True, "windows": 1, "requests": 4, "problems": 0, "failed": [], "refused": 0}}))
    bundle(front / "bundle.tgz", f"c-{b}", params, 1, 4)
    piece = out / b / "piece"
    (piece / "chain" / "report").mkdir(parents=True)
    if answer is not None:
        (piece / "answer.json").write_text(json.dumps(answer))
        (piece / "chain" / "report" / "report.json").write_text(json.dumps({
            "verdict": "accepted", "path": "warden", "scope": None, "stubbed": ["the weights' load isn't certified"],
            "negatives_refused": {"scan/tampered": True, "n2/tampered": b != "b2"},
            "steps": {"scan": {"verdict": "accepted", "eps_at_delta": "2^-20"}, "tie": {"verdict": "accepted"}}}))
    if refused is not None:
        (piece / "refused.txt").write_text(refused + "\n")


UNWIRED = {"protocol": "verity/certifier/v0", "verdict": "reject", "reason": "the scan's prog is no compute "
           "audit's slot entries"}


def lab(out: Path, params: dict) -> None:
    """Three zones' released outcomes under ``out``: one whose period check rejected, one whose piece refused and one
    accepted, with the front's front.json."""
    zone(out, "b0", params, answer=UNWIRED)
    zone(out, "b1", params, answer=None, refused="the chain wrote no report (exit 7)")
    zone(out, "b2", params, answer={"protocol": "verity/certifier/v0", "verdict": "accept", "reason": None})
    out.joinpath("front.json").write_text(json.dumps({
        "host": "lab", "declaration": "leaf-spine-3",
        "zones": {b: {"zone": f"na-{b}", "certifier": f"na-{b}-cert", "address": a}
                  for b, a in (("b0", "10.77.0.1"), ("b1", "10.77.0.9"), ("b2", "10.77.0.17"))}}))


def test_the_record_is_each_zones_verdicts_and_K_in_the_bounds_terms_labelled_simulated(onsite, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TRACES))
    import leak_bounds as LB
    import scan_zk as SZ
    from verity.core.protocols.certifier import ingress_scan as IS, warden as W
    params = asdict(SZ.W63K4.params)
    assert params["egress_cells"] == W63K4_CELLS
    out = tmp_path / "out"
    unwired = UNWIRED
    lab(out, params)
    assert onsite.record(argparse.Namespace(out=str(out))) == 0
    doc = json.loads((out / "record.json").read_text())
    assert (doc["label"], doc["served"], doc["simulated"]) == ("simulated", False, list(onsite.SIMULATED))
    b0, b1, b2 = doc["zones"]["b0"], doc["zones"]["b1"], doc["zones"]["b2"]
    assert (b0["verdict"], b0["period"]["reason"]) == ("reject", unwired["reason"])
    assert b0["chain"]["verdict"] == "accepted" and b0["chain"]["steps"]["scan"]["eps_at_delta"] == "2^-20"
    assert (b1["verdict"], b1["chain"]) == ("refused", None)
    assert [z["K"]["established"] for z in (b0, b1, b2)] == [False, False, True]
    assert doc["lab"]["established"] is False

    cls, T = SZ.W63K4.cls, params["window"]
    ell_a, entries = T * IS.view_bits(cls), cls.NR * (T // cls.EVERY) * IS.entry_bits(cls)
    assert (onsite.READING, onsite.WITH_TIES) == ("view", "registered") and (ell_a, entries) == (25_600, 896)
    cell = math.ceil(math.log2(W63K4_CELLS + 1))
    caps = W.WardenParams(**params).caps(1)
    k = b0["K"]
    assert (len(caps), cell) == (128, 3)
    assert {x: k["per_phase"][x] for x in ("connections", "cell", "k_sched", "k_fail")} == {
        "connections": len(caps), "cell": cell, "k_sched": 384, "k_fail": 384}
    assert k["per_phase"]["k_sched"] == sum(math.ceil(math.log2(c + 1)) for c in caps)
    assert {x: k["per_phase"][x] for x in ("ell_a", "k_skip", "T_j", "d_jit", "jitter")} == {
        "ell_a": ell_a, "k_skip": 0, "T_j": T, "d_jit": 2, "jitter": T}
    assert k["per_phase"]["K"] == 384 + 384 + ell_a + T == 26_400
    assert "with_ties" not in k["per_phase"] and b0["with_ties"] == LB.label("registered") | {
        "ties": dict(LB.TIES), "of": onsite.PHASE_ONLY, "ell_a": entries, "K": 384 + 384 + entries + T}
    assert onsite.PHASE_ONLY == "a phase alone: no period's, lab's, datacenter's or world's figure is made from it"
    assert b0["with_ties"]["K"] == 1_696 and b0["with_ties"]["conditional_on"] == list(LB.TIES) and len(LB.TIES) == 2
    assert (k["reading"], k["bounds"]) == ("view", LB.COUNT) and "K is no leak bound" in LB.COUNT
    assert "with_ties" not in json.dumps({"K": k, "lab": doc["lab"]})
    assert all(z["K"]["conditional_on"] == list(onsite.OPEN_CHECKS) for z in (b0, b2))
    one = LB.link(cls, Fraction(4, 1) / (T * Fraction(1, 10)), T=T, tau=Fraction(1, 10), egress_cells=W63K4_CELLS)
    for period, t in k["periods"].items():
        m = Fraction(LB.PERIODS[period]) / (T * Fraction(1, 10))
        assert t["m"] == pytest.approx(float(m)) and t["bound"] == pytest.approx(one.bits(period, "view"))
        assert sum(t[x] for x in onsite.TERMS) == pytest.approx(t["bound"], rel=1e-12)
        assert Fraction(t["eps"]) == m * LB.SCAN_DELTA and t["vacuous"] == (m * LB.SCAN_DELTA >= 1)
        assert t["delta_net"] == [{"symbol": "eps", "value": t["eps"]},
                                  *({"symbol": s, "times": t["m"]} for s in ("ε_ks", "ε_bind", "ε_tap"))]
        assert t["pooled"]["1e-06"]["k_skip"] == pytest.approx(one.bits(period, "view", Fraction(1, 10**6))
                                                               - one.bits(period, "view"))
    assert (k["periods"]["week"]["vacuous"], k["periods"]["year"]["vacuous"]) == (False, True)
    assert doc["lab"]["links"] == 3 and doc["lab"]["week"]["per_period"] == pytest.approx(
        3 * k["periods"]["week"]["bound"])
    week = k["periods"]["week"]["m"]
    assert doc["lab"]["week"]["m"] == pytest.approx(3 * week)
    assert doc["lab"]["week"]["delta_net"][1:] == [{"symbol": s, "times": doc["lab"]["week"]["m"]}
                                                   for s in ("ε_ks", "ε_bind", "ε_tap")]
    dc = next(r for r in k["scopes"] if (r["period"], r["scope"]) == ("week", "datacenter"))
    assert dc["delta_net"][0] == {"symbol": "eps", "value": dc["eps"]} and dc["delta_net"][1]["times"] == pytest.approx(
        12_500 * week)
    md = (out / "record.md").read_text()
    assert md.startswith("# The onsite network path on the mock datacenter: SIMULATED, never served")
    assert "| b0 | accepted, 4 requests in 1 phase(s) | accepted (scan accepted, tie accepted) | reject: " in md
    assert "| b1 | accepted, 4 requests in 1 phase(s) | not run | refused: the chain wrote no report" in md
    assert "- the weights' load isn't certified\n" in md
    assert "Its tampered negatives refused: n2/tampered, scan/tampered; NOT refused: b2: n2/tampered." in md
    assert "K is established for b2 here" in md
    assert "**b0** (not established: the period check answered reject)" in md and "**b2** (established)" in md
    f = onsite.fmt
    assert f"| week | zone b0 | 1 | {f(week)} | no |" in md and f"| week | zone b2 | 1 | {f(week)} | yes |" in md
    assert f"| week | the lab | 3 | {f(3 * week)} | no |" in md
    assert f"| week | datacenter | 12,500 | {f(12_500 * week)} | no (scaled) |" in md
    assert "δ_net = eps + m·(ε_ks + ε_bind + ε_tap)" in md and "δ_net has no number here" in md
    assert "k_sched = k_fail = 384 (128 connections the schedule can carry, 3 bits each)" in md
    assert "whatever the requests served" in md and "a request's" not in md
    assert params["bucket_ms"] == 100 and k["per_phase"]["phase_s"] == pytest.approx(3.2)
    assert "a phase of T_j = 32 buckets of τ = 100 ms (3.2 s)" in md and "| m (3.2 s phases) |" in md
    assert "each phase T_j = 32 buckets of τ = 100 ms (3.2 s), the lab's tick" in md
    assert "(vacuous)" in md and "known overcount" in md and "Σ_j (ℓ_a,j + k_skip,j + T_j·log2 d_jit)" in md
    assert f"ℓ_a,j = {ell_a:,} (the scan's view rows, as the period check charges them)" in md
    assert "(d_jit = 2); K a phase = 26,400.\n" in md and "| b0 | 896 | 1,696 |" in md
    assert "K bounds a count expression" in md and "so K is no leak bound: every K in this section is that K" in md
    assert "rests on neither of the scan's ties" in md
    assert "No period's, lab's, datacenter's or world's figure is made from it" in md
    assert all(t in md for t in LB.TIES.values()) and "## K, the count the period check charges a zone's link" in md
    assert "can send beyond its committed program" not in md and "Every K here is conditional" not in md
    assert all(c in md for c in onsite.OPEN_CHECKS) and "ℓ_a + 1" not in md
    assert "the link figure charges the advice through the scan's inputs" in md
    for retired in ("D_j", "k_i", "link-window", "delta_net", "J is", "brief", "NET-", "advice width"):
        assert retired not in md, retired
    assert onsite.short("word " * 50).endswith(" …") and len(onsite.short("word " * 50)) <= 162


def test_no_row_of_K_leaves_the_record_without_its_readings_label(onsite, tmp_path, monkeypatch):
    """Every row of K the lab record writes, a zone's phase, its week's, month's and year's, its datacenter's and
    world's and the lab's sum's, carries its reading's label as its own fields (`leak_bounds.label`): the reading's
    name, the words that K bounds a count and is no leak bound, and the premises it is open on. Every such row is in
    the check's reading. The figure with both of the scan's ties is a phase's alone, beside K under its own heading,
    with the two ties it is conditional on and the words that no check holds them; no table and no line holds the two
    readings, and nothing is their sum. Each table has its label on the line directly above it. Take a label's field
    out of a row and `markdown` writes no table; take it out where a row is made and `record` writes no record."""
    monkeypatch.syspath_prepend(str(TRACES))
    import leak_bounds as LB
    import scan_zk as SZ
    out = tmp_path / "out"
    lab(out, asdict(SZ.W63K4.params))
    assert onsite.record(argparse.Namespace(out=str(out))) == 0
    doc = json.loads((out / "record.json").read_text())
    rows, view = onsite.k_rows(doc), LB.label("view")
    assert len(rows) == 3 * (1 + 3 + 9 + 1) + 3
    for z in doc["zones"].values():
        period_rows = [z["K"]["per_phase"], *z["K"]["periods"].values(), *z["K"]["scopes"]]
        assert len(period_rows) == 13 and all({k: r[k] for k in LB.LABEL} == view for r in period_rows)
        tied = z["with_ties"]
        assert {k: tied[k] for k in LB.LABEL} == LB.label("registered") and "which no check holds" in tied["what"]
        assert tied["conditional_on"] == ["state-tied", "view-tied"] == list(tied["ties"])
        assert not {"week", "month", "year", "periods", "scopes"} & set(tied) and "a phase alone" in tied["of"]
    assert all({k: doc["lab"][p][k] for k in LB.LABEL} == view for p in LB.PERIODS)
    assert (view["open"], view["conditional_on"]) == (["view-cut", "period-checked", "answers-verified"], [])
    assert "bounds a count expression" in view["bounds"] and "K is no leak bound" in view["bounds"]
    assert all(LB.unlabelled(r) == [] for r in rows)
    assert "27296" not in json.dumps(doc) and "28096" not in json.dumps(doc)

    md = (out / "record.md").read_text().split("\n")
    heads = {line: i for i, line in enumerate(md) if line.startswith("## ")}
    k, tied = heads[onsite.HEADINGS["view"]], heads[onsite.HEADINGS["registered"]]
    assert k < heads["## Over all compute"] < tied < heads["## Notes"]
    assert onsite.HEADINGS["registered"] == "## The count with both of the scan's ties, which no check holds"
    tables = [i for i, line in enumerate(md) if line.startswith(("| period |", "| zone | ℓ_a,j"))]
    assert len(tables) == 3 + 1 + 1
    for t in tables:
        assert md[t - 1] == "" and md[t - 2] == LB.caption("view" if t < tied else "registered"), md[t]
    assert [t > tied for t in tables] == [False, False, False, False, True]
    assert all("1,696" not in line and "896" not in line for line in md[:tied])
    assert all("26,400" not in line for line in md[tied:heads["## Notes"]]) and not any("27,296" in x for x in md)
    assert sum("| 896 | 1,696 |" in line for line in md[tied:]) == 3

    for field in LB.LABEL:
        for pick in (lambda d: d["zones"]["b2"]["K"]["periods"]["week"], lambda d: d["zones"]["b0"]["K"]["scopes"][4],
                     lambda d: d["lab"]["year"], lambda d: d["zones"]["b1"]["with_ties"],
                     lambda d: d["zones"]["b0"]["K"]["per_phase"]):
            cut = json.loads(json.dumps(doc))
            del pick(cut)[field]
            with pytest.raises(ValueError, match=f"the lab record's tables: row .*lacks its label's .*{field}"):
                onsite.markdown(cut)
    (out / "record.json").unlink()
    terms = onsite.terms
    monkeypatch.setattr(onsite, "terms", lambda k, p: {x: v for x, v in terms(k, p).items() if x != "open"})
    with pytest.raises(ValueError, match="the lab record: row 1 of K .*lacks its label's open.*leak_bounds.label"):
        onsite.record(argparse.Namespace(out=str(out)))
    assert not (out / "record.json").exists()
    monkeypatch.setattr(onsite, "terms", terms)
    whole = LB.Bound.label
    monkeypatch.setattr(LB.Bound, "label", property(lambda b: {x: v for x, v in LB.label(b.reading).items()
                                                               if x != "bounds"}))
    with pytest.raises(ValueError, match="leak: row 0 of K .*lacks its label's bounds"):
        onsite.record(argparse.Namespace(out=str(out)))
    assert not (out / "record.json").exists()
    monkeypatch.setattr(LB.Bound, "label", whole)
    assert onsite.record(argparse.Namespace(out=str(out))) == 0 and (out / "record.json").is_file()


def test_a_zone_whose_gate_didnt_accept_has_no_K(onsite, tmp_path):
    (tmp_path / "front").mkdir()
    (tmp_path / "front" / "session.json").write_text(json.dumps({"refused": "the gate refuses it: busy"}))
    got = onsite.zone_record(tmp_path)
    assert (got["verdict"], got["K"], got["gate"]["accepted"]) == ("not run", None, False)


def test_the_piece_takes_the_fronts_outcome_from_its_out(onsite, tmp_path):
    front, out, keep = tmp_path / "front", tmp_path / "out", tmp_path / "keep"
    (front / "b0" / "front").mkdir(parents=True)
    (front / "front.json").write_text("{}")
    (front / "b0" / "front" / "session.json").write_text("{}")
    args = argparse.Namespace(out=str(out), keep=str(keep), front=str(front), zone=[], lean=None, set=[])
    with pytest.raises(SystemExit, match="no zone's serve dir under"):
        onsite.piece(args)
    assert (out / "front.json").is_file() and (out / "b0" / "front" / "session.json").is_file()
    with pytest.raises(SystemExit, match="no front.json in"):
        onsite.piece(argparse.Namespace(**vars(args) | {"out": str(tmp_path / "other"), "front": None}))


def test_the_pieces_record_of_the_deployment_is_the_local_gates_own_copy(onsite, tmp_path, capsys):
    """`auditor_record`: the lab, its own auditor, hands the piece its local gate's own out/deployment.json of the kept
    run as the auditor's record (`warden_chain.RECORD`), and none, saying why, where the gate kept no copy of the run
    or the serve dir kept no run."""
    serve, run = tmp_path / "serve", "r1"
    kept = serve / onsite.CP.PRIVATE
    (kept / run).mkdir(parents=True)
    (kept / run / "fetch.tar.gz").write_bytes(b"")
    gate = kept / onsite.CP.SS.LOCAL_GATE / "var" / "certifier" / run / "out"
    gate.mkdir(parents=True)
    dep = {"key": {"public": "ab"}, "session": "cd", "params": {}}
    (gate / "deployment.json").write_text(json.dumps(dep, indent=1))
    rec = json.loads(onsite.auditor_record(serve, tmp_path / "key.json").read_text())
    assert (rec["schema"], rec["key"], rec["session"], rec["deployment"]) == ("certifier-key/v1", dep["key"], "cd", dep)
    assert rec["deployment_sha512"] == hashlib.sha512((gate / "deployment.json").read_bytes()).hexdigest()
    shutil.rmtree(kept / onsite.CP.SS.LOCAL_GATE)
    assert onsite.auditor_record(serve, tmp_path / "none.json") is None and not (tmp_path / "none.json").exists()
    assert "0 copies of run r1 at its local gate, not one" in capsys.readouterr().out
    shutil.rmtree(kept / run)
    assert onsite.auditor_record(serve, tmp_path / "none.json") is None
    assert "no auditor's record of the run's deployment (" in capsys.readouterr().out


def test_keep_is_refused_inside_out(onsite, tmp_path):
    with pytest.raises(SystemExit, match="inside --out, which the run publishes"):
        onsite.dirs(argparse.Namespace(out=str(tmp_path), keep=str(tmp_path / "keep")))
