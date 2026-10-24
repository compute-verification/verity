"""network-e2e's report on certifier-serve's runs on an injected clock against a stub endpoint (`n2_zk.virtual`): honest,
late and extra at a tiny shape, N2's verdicts only (nothing is lowered or proved here; `run.sh` proves on a node). The
report holds per run its network verdict and per link-window N2's verdict of record: the honest run is accepted on every
link-window, and late, extra and the tampered head are each refused at window 0 for their own reason (late: its egress
head's fields and counts; extra: its counts alone; tampered: its counts and signed digest). The claim is in both files
word for word and names the gaps the certifier README names. The proofs' columns are read from n2_zk's summary as
`n2_zk.prove` writes it, and a proof that wasn't held never counts as a refusal. Proofs n2_zk's own run kept are adopted
for these runs, and refused when a unit carries another run's certificate. A gate run's fetched directory and a tarball
of a rehearsal are read as the rehearsal is."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()


def _args(src, out) -> SimpleNamespace:
    return SimpleNamespace(sources=[str(src)], upstream=None, vllm=None, out=str(out), work=None, bin=None, lean=None,
                           delta=2.0 ** -20, no_prove=True, proofs=None)


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    plan = e2e.n2_zk.virtual(runs, requests=8, tokens=4, strategies=("honest", "late", "extra"))
    return SimpleNamespace(plan=plan, runs=runs)


@pytest.fixture(scope="module")
def virtual(rehearsal, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return SimpleNamespace(plan=rehearsal.plan, runs=rehearsal.runs, out=out,
                           report=e2e.run(_args(rehearsal.runs, out), [], clock=lambda: 0.0))


@pytest.mark.slow
def test_the_honest_run_is_accepted_and_each_negative_refused_for_its_own_reason(virtual):
    r = virtual.report
    windows = virtual.plan["windows"]
    assert r["format"] == e2e.FORMAT and r["ok"] is True and r["profile"] == "planned" and windows >= 2
    assert list(r["runs"]) == ["honest", "late", "extra", "tampered"]
    honest = r["runs"]["honest"]
    assert honest["network"]["accepted"] is True and honest["network"]["windows"] == windows
    rows = {(x["run"], x["window"]): x for x in r["link_windows"]}
    assert all(rows[("honest", w)]["n2"] == "accept" for w in range(windows))
    reasons = {"late": ["fields", "counts"], "extra": ["counts"], "tampered": ["counts", "signed"]}
    for name, failed in reasons.items():
        neg = r["negatives"][name]
        assert neg["refused"] is True and neg["n2_refused_windows"] == [0] and not neg["proved"], name
        assert r["runs"][name]["network"] == r["runs"][name]["network"] | {"accepted": False, "refused": [0]}, name
        assert rows[(name, 0)]["failed_parts"] == {"egress": failed, "ingress": []}, name
        assert all(rows[(name, w)]["n2"] == "accept" for w in range(1, windows)), name
    assert r["proofs"]["ran"] is False and r["proofs"]["cost_per_link_window"] is None


@pytest.mark.slow
def test_the_claim_is_in_both_files(virtual):
    doc = json.loads((virtual.out / "report.json").read_text())
    md = (virtual.out / "report.md").read_text()
    assert doc["claim"] == e2e.CLAIM and doc["claim"]["statement"].startswith("Counts only.")
    for text in [e2e.CLAIM["statement"], *e2e.CLAIM["not_covered"], *e2e.CLAIM["zero_knowledge_gaps"]]:
        assert text in md
    assert "N3 (content)" in md and "| honest | 0 | accept |" in md and "**tampered**: refused" in md
    assert doc["upstream"].startswith("http://127.0.0.1:") and doc["where"].endswith("not an auditor host")
    assert doc["where"] in md and doc["tokens"] == 4 and "Late's head checks are not proved" in md


def test_the_claims_gaps_are_the_ones_the_certifier_readme_names():
    readme = (ROOT / e2e.README).read_text()
    gaps = readme.split("## Not yet zero-knowledge", 1)[1]
    for named, ours in (("N2's unit isn't proved yet", "N2's unit isn't proved as the verifier of record"),
                        ("NET-43", "NET-43"), ("NET-42", "NET-42"), ("CertifiedRecord.verify", "CertifiedRecord.verify")):
        assert named in gaps and any(ours in g for g in e2e.CLAIM["zero_knowledge_gaps"]), named
    assert "counts only" in readme and any("N3" in x for x in e2e.CLAIM["not_covered"])


def _n2(windows: int, refused: dict[str, int]) -> dict:
    """N2's verdicts as `n2_zk.prove` records them, each run refused at the window `refused` names."""
    out = {}
    for name in ("honest", "late", "extra", "tampered"):
        bad = refused.get(name)
        out[name] = {"windows": [{"window": w, "accepted": w != bad, "codes": [] if w != bad else ["reject"],
                                  "parts": {}} for w in range(windows)],
                     "network": {"accepted": bad is None, "windows": windows, "refused": [] if bad is None else [bad],
                                 "digest": "d"}}
    return out


def _member(name: str, windows: int, bad: int | None, held: bool = True) -> dict:
    units = [{"window": w, "unit": w, "verdict": "reject" if w == bad else "accept"} for w in range(windows)]
    return {"n": windows, "ands": 1000, "stage_s": 1.0, "accepted": held, "units": units,
            "verdicts": {f"m0-verifier/{name}": held, f"lean-verify/{name}": held}, "lean_verify_s": 2.0 * windows,
            "timing": {"session_s": 9.0, "timing": {"prove_total_s": 4.0 * windows},
                       "process": {"max_child_rss_gb": 3.5}},
            "per_unit": {"prove_total_s": 4.0, "lean_verify_s": 2.0, "proof_bytes": 100.0},
            "received": {"verifier_bytes": 10, "verifier_found": [], "control": {"holds_all": True}}}


def test_the_proofs_columns_and_cost_are_read_from_n2_zk_and_an_unheld_proof_refuses_nothing(tmp_path):
    runs = {s: {"dir": tmp_path / s, "check": None, "gate": None} for s in ("honest", "late", "extra")}
    members = {"honest-egress": _member("honest-egress", 2, None), "honest-ingress": _member("honest-ingress", 2, None),
               "tampered-egress": _member("tampered-egress", 2, 0), "extra-egress": _member("extra-egress", 2, 0)}
    proofs = {"ran": True, "rc": 0, "why": None, "extra_proved": True, "summary": {"ok": True, "members": members}}
    r = e2e.report(runs, _n2(2, {"late": 0, "extra": 0, "tampered": 0}), proofs, {})
    rows = {(x["run"], x["window"]): x for x in r["link_windows"]}
    assert r["ok"] is True
    assert rows[("honest", 0)]["proof"] == {"egress": "accept", "ingress": "accept"}
    assert rows[("honest", 1)]["head_checks"] == "pass" and rows[("honest", 1)]["prove_s"] == 8.0
    assert rows[("honest", 1)]["verify_s"] == 4.0 and rows[("honest", 1)]["peak_gb"] == 3.5
    assert rows[("tampered", 0)]["head_checks"] == "refuse" and rows[("late", 0)]["proof"] is None
    assert r["negatives"]["tampered"]["proved_refused_windows"] == [0] and not r["negatives"]["late"]["proved"]
    assert r["proofs"]["cost_per_link_window"] == {"link_windows": 2, "prove_s": 8.0, "verify_s": 4.0,
                                                   "proof_bytes": 200.0, "peak_gb": 3.5}
    members["extra-egress"] = _member("extra-egress", 2, 0, held=False)
    r = e2e.report(runs, _n2(2, {"late": 0, "extra": 0, "tampered": 0}), proofs, {})
    assert r["negatives"]["extra"]["refused"] is False and r["ok"] is False
    members["honest-ingress"] = _member("honest-ingress", 2, None, held=False)
    assert e2e.report(runs, _n2(2, {"late": 0, "extra": 0, "tampered": 0}), proofs, {})["ok"] is False


def _kept(d: Path, runs: dict) -> Path:
    """n2_zk's output as `n2_zk.sh` keeps it for `runs`: each member's units, and the verifier's ``n2.json`` with the
    certificate digest each unit's public row carried."""
    members = {}
    for k in ("honest-egress", "honest-ingress", "tampered-egress", "extra-egress"):
        name, end = k.rsplit("-", 1)
        certified, _ = e2e.n2_zk.cs.read_certified(runs["honest" if name == "tampered" else name]["dir"] / "certified.jsonl")
        direction = e2e.Direction.EGRESS if end == "egress" else e2e.Direction.INGRESS
        windows = sorted(w for dd, w in certified if dd == direction)
        (d / k / "verifier").mkdir(parents=True)
        (d / k / "verifier" / "n2.json").write_text(json.dumps(
            [{"window": w, "signed": certified[(direction, w)].certificate.digest.hex()} for w in windows]))
        members[k] = _member(k, len(windows), None if name == "honest" else 0)
    (d / "summary.json").write_text(json.dumps({"ok": True, "members": members}))
    return d


@pytest.mark.slow
def test_proofs_of_these_runs_are_adopted_and_another_runs_refused(rehearsal, tmp_path):
    runs = e2e.collect([rehearsal.runs], tmp_path)
    kept = _kept(tmp_path / "n2zk", runs)
    got = e2e.adopt(kept, runs)
    assert got["rc"] == 0 and got["why"] is None and got["extra_proved"] and got["adopted"] == str(kept)
    with pytest.raises(SystemExit, match="gives no extra run"):
        e2e.adopt(kept, {k: v for k, v in runs.items() if k != "extra"})
    args = _args(rehearsal.runs, tmp_path / "out")
    args.no_prove, args.proofs = False, str(kept)
    r = e2e.run(args, [], clock=lambda: 0.0)
    assert r["ok"] is True and r["proofs"]["adopted"] == str(kept) and r["proofs"]["cost_per_link_window"]
    assert r["negatives"]["tampered"]["proved_refused_windows"] == [0] and r["negatives"]["extra"]["proofs_held"]
    assert "Proofs: adopted from" in (tmp_path / "out" / "report.md").read_text()
    n2 = kept / "honest-ingress" / "verifier" / "n2.json"
    sent = json.loads(n2.read_text())
    sent[1]["signed"] = sent[0]["signed"]
    n2.write_text(json.dumps(sent))
    with pytest.raises(SystemExit, match="proofs of another run"):
        e2e.adopt(kept, runs)


def _rehearsal(d: Path, strategies, verdicts: dict) -> Path:
    for s in strategies:
        (d / s).mkdir(parents=True)
        (d / s / "certified.jsonl").write_text("")
        (d / s / "check.json").write_text(json.dumps({"verdict": verdicts[s], "network_verdict": {"accepted": False}}))
    (d / "summary.json").write_text(json.dumps({"runs": {s: {"verdict": verdicts[s]} for s in strategies}}))
    return d


def test_a_gate_runs_directory_and_a_tarball_are_read_as_the_rehearsal(tmp_path):
    gate = tmp_path / "r1"
    _rehearsal(gate / "out", ["honest"], {"honest": "accepted"})
    (gate / "end.json").write_text(json.dumps({"run": "r1", "rc": 0, "ended": "the rehearsal ended", "pinned": "abc",
                                               "runs": {"honest": {"verdict": "accepted", "network": True}}}))
    local = _rehearsal(tmp_path / "local" / "planned-40", ["late", "extra"], {"late": "refused", "extra": "refused"})
    tgz = tmp_path / "planned-40.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        tar.add(local, arcname="planned-40")
    work = tmp_path / "work"
    work.mkdir()
    got = e2e.collect([gate, tgz], work)
    assert sorted(got) == ["extra", "honest", "late"]
    assert got["honest"]["dir"] == gate / "out" / "honest" and got["honest"]["check"]["verdict"] == "accepted"
    assert got["honest"]["gate"]["run"] == "r1" and got["honest"]["gate"]["verdict"] == "accepted"
    assert got["late"]["dir"].parent.name == "planned-40" and got["late"]["check"]["verdict"] == "refused"
    assert e2e.where(got)["where"].startswith("gate: ")
    with pytest.raises(SystemExit, match="two honest runs"):
        e2e.collect([gate, gate], work)
    with pytest.raises(SystemExit, match="no honest run"):
        e2e.collect([tgz], work)
