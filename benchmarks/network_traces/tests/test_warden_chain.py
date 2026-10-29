"""The network chain on an active warden's run (`warden_chain`, `chain_record`'s warden path), here the synthetic run
published as `warden_serve.Publisher` publishes a served one, in the gate's layout (its witness under ``witness/``, its
public files under ``out/``, ``certificates.jsonl`` as `auditor.warden_outcome` writes it): the certificates step
reruns the warden's own check and holds the public certificates to the certified lines, refusing a changed one and a
changed cell; the certified I/O read off the cells is the requests and responses served; N2's link is the warden's
(both heads at its digest, no sync, charge 0) and its references expect honest accepted and tampered refused on its
signed digest alone. A certifier rehearsal's run (its records through active.Proxy) is refused by name
(`warden_chain.PROXY`) by every reader: its files, or a certifier's deployment.json in a warden's run. The report on
the warden path requires the certificates, the endpoint, the tie, N2 and the scan, names its stubs, and refuses a step
whose input is a certifier rehearsal's or another warden run's."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_scan_zk import published, sz  # noqa: E402

import warden_chain as WC  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"network_traces_wc_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr, n2 = _load("chain_record"), _load("n2_zk")
PUBLIC = ("deployment.json", "cells.jsonl")


def gate_run(d: Path) -> Path:
    """The synthetic warden run in the gate's layout: its files under ``witness/``, its public ones copied to ``out/``
    with ``certificates.jsonl``, each certified line's certificate or refusal (`auditor.warden_outcome`)."""
    run = published(d / "flat")
    (d / "gate" / "out").mkdir(parents=True)
    shutil.copytree(run, d / "gate" / "witness")
    for name in PUBLIC:
        shutil.copyfile(run / name, d / "gate" / "out" / name)
    lines = [json.loads(x) for x in (run / "certified.jsonl").read_text().splitlines()]
    (d / "gate" / "out" / "certificates.jsonl").write_text("".join(
        json.dumps({k: x[k] for k in ("certificate", "refused") if k in x}, sort_keys=True) + "\n" for x in lines))
    return d / "gate"


def test_a_warden_runs_certificates_io_and_link_are_its_own(tmp_path):
    run = gate_run(tmp_path)
    got = WC.certificates(run)
    assert (got["verdict"], got["why"], got["certified_windows"], got["windows"]) == ("accepted", None, sz.W63.windows,
                                                                                    sz.W63.windows)
    assert got["public_is_the_records"] is True and got["clock_certificates"] == 0 and got["sync_bits"] == 0.0
    assert got["warden_run"] == "00" * 32 and got["warden_check"]["accepted"]
    io = WC.io(run)
    assert io["accepted"] and io["requests"] == len(sz.W63.prompts) and not io["violations"]
    assert [r["output"] for r in io["records"]] == [list(r) for r in sz.W63.responses]
    assert all(r["release"] is not None and r["frames"][0][1] >= r["release"] for r in io["records"])
    lw, dep, params = WC.link(run)
    assert lw.warden.digest == params.digest and lw.charge == 0.0 and lw.windows == sz.W63.windows
    assert lw.grid(0) == lw.grid(1) and lw.link == "warden" and lw.run == dep["run_id"]
    pairs, refused = WC.pairs(run, dep)
    assert len(pairs) == sz.W63.windows and refused == []
    assert all(lw.certified(w, [r.certificate for r in pair]) for w, pair in enumerate(pairs))

    public = run / "out" / "certificates.jsonl"
    lines = public.read_text().splitlines()
    public.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")
    got = WC.certificates(run)
    assert got["verdict"] == "refused" and got["why"] == "the public certificates.jsonl isn't the certified records' "\
                                                         "certificates"
    run = gate_run(tmp_path / "cell")
    path = run / "witness" / "responses.records"
    blobs = WC.WS.ar.read_blobs(path)
    rec = WC.WS.ar.unpack_record(blobs[0], sz.PARAMS.cell_bytes)
    t, j = next((t, j) for t, row in enumerate(rec.grid) for j, f in enumerate(row) if f is not None)
    grid = [list(row) for row in rec.grid]
    grid[t][j] = bytes([grid[t][j][0]]) + bytes(len(grid[t][j]) - 1)
    path.unlink()
    for b in [WC.WS.ar.pack_record(type(rec)(rec.link, rec.window, tuple(map(tuple, grid)), rec.status)), *blobs[1:]]:
        WC.WS.ar.write_blobs(path, b)
    got = WC.certificates(run)
    assert got["verdict"] == "refused" and "responses window 0 isn't certified" in got["why"]


@pytest.fixture(scope="module")
def certifier_run(tmp_path_factory):
    """A certifier rehearsal's run (`n2_zk.virtual`: certifier-serve's records through its proxy)."""
    out = tmp_path_factory.mktemp("certifier")
    n2.virtual(out, requests=4, tokens=2, strategies=("honest",))
    return out / "honest"


def test_a_certifier_rehearsals_run_is_refused_by_name_by_every_reader(tmp_path, certifier_run):
    proxy = re.escape(WC.PROXY)
    for read in (WC.deployment, WC.certificates, WC.io, WC.link):
        with pytest.raises(SystemExit, match=f"{proxy}.*plan.json is there"):
            read(certifier_run)
    with pytest.raises(SystemExit, match=proxy):
        n2.main(["prove", "--warden-run", str(certifier_run), "--dry", "--out", str(tmp_path / "n2")])
    with pytest.raises(SystemExit, match=proxy):
        cr.main(["certs", "--warden-run", str(certifier_run), "--out", str(tmp_path / "certs.json")])
    run = gate_run(tmp_path)
    shutil.copyfile(certifier_run / "deployment.json", run / "out" / "deployment.json")
    shutil.copyfile(certifier_run / "deployment.json", run / "witness" / "deployment.json")
    with pytest.raises(SystemExit, match=f"{proxy}.*its deployment.json isn't a warden's"):
        WC.certificates(run)
    run = gate_run(tmp_path / "extra")
    with pytest.raises(SystemExit, match="--extra and --late are certifier-serve's runs"):
        n2.main(["prove", "--warden-run", str(run), "--extra", str(certifier_run), "--dry", "--out",
                 str(tmp_path / "n2x")])


def test_n2_on_a_warden_run_expects_honest_accepted_and_tampered_refused_on_its_signed_digest(tmp_path):
    run = gate_run(tmp_path)
    assert n2.main(["prove", "--warden-run", str(run), "--gpu", "--dry", "--out", str(tmp_path / "n2")]) == 0
    s = json.loads((tmp_path / "n2" / "summary.json").read_text())
    assert s["warden_run"] == "00" * 32 and "no sync" in s["warden"]["syncs"] and s["gpu"] is True
    honest, tampered = (s["members"][k]["reference"] for k in ("honest", "tampered"))
    assert [x["expect"] for x in honest] == [1] * sz.W63.windows and all(not any(x["failed"].values()) for x in honest)
    assert [x["expect"] for x in tampered] == [0] + [1] * (sz.W63.windows - 1)
    assert tampered[0]["failed"] == {"egress": ["signed"], "ingress": []}
    assert s["draw"]["selection"] == "all" and n2.OTHER not in s["members"]


def test_n2_on_a_sampled_warden_period_tampers_every_window_and_proves_another_subset(tmp_path):
    """With ``--draw K``, K < n, the tampered member changes every window, so whichever K windows the draw holds refuse
    it, and other_subset (the honest records, proved on another K-subset) is a member; K outside 1 … n is refused."""
    run = gate_run(tmp_path)
    assert n2.main(["prove", "--warden-run", str(run), "--draw", "1", "--dry", "--out", str(tmp_path / "n2")]) == 0
    s = json.loads((tmp_path / "n2" / "summary.json").read_text())
    assert (s["draw"]["law"], s["draw"]["k"], s["draw"]["population"]) == ("subset:1", 1, sz.W63.windows)
    tampered, other = (s["members"][k]["reference"] for k in ("tampered", n2.OTHER))
    assert [x["expect"] for x in tampered] == [0] * sz.W63.windows
    assert all(x["failed"] == {"egress": ["signed"], "ingress": []} for x in tampered)
    assert [x["expect"] for x in other] == [1] * sz.W63.windows
    with pytest.raises(SystemExit, match="a period draw takes 1 to 2"):
        n2.main(["prove", "--warden-run", str(run), "--draw", "3", "--dry", "--out", str(tmp_path / "n2x")])


def _steps(tmp_path, run: Path) -> SimpleNamespace:
    """The warden path's step outputs on `run`: its real certificates, and the endpoint, the tie, N2 and the scan as
    their drivers write them."""
    def put(name, doc):
        (tmp_path / name).write_text(json.dumps(doc))
        return str(tmp_path / name)

    rid = "00" * 32
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    return SimpleNamespace(
        custody=put("custody.json", {"verdict": "accepted"}),
        certs=put("certs.json", WC.certificates(run)),
        endpoint=put("endpoint.json", {"warden_run": rid, "io": {"accepted": True, "requests": 5}, "chain": {
            "bits": {"bound": True, "rebuilt": True, "commit_windows": [{"bound": True, "rebuilt": True}]}}}),
        tie=put("tie.json", {"class": "warden", "warden_run": rid, "ok": True, "link": {"T": 32, "unit": "u"},
                             "tie": {"honest": [True, True], "forged": [False, True]}}),
        n2=put("n2.json", {"ok": True, "warden_run": rid, "members": {
            "honest": {"windows": [window, window], "network": {"accepted": True}, "ok": True},
            "tampered": {"windows": [window | {"accepted": False}, window], "network": {"accepted": False},
                         "ok": True}}}),
        scan=put("scan.json", {"ok": True, "T": 32, "warden_run": rid, "rounds": {
            "honest": {"audit": {"window_accepted": True}}, "forged": {"audit": {"window_accepted": False}}}}),
        job=put("job.json", {"path": cr.WARDEN, "rc": {"n2": "0", "scan": "0", "tie": "0"}}), warden_run=str(run))


def test_the_warden_path_requires_every_step_on_one_warden_run_and_refuses_a_certifier_rehearsals(tmp_path,
                                                                                                certifier_run):
    steps = _steps(tmp_path, gate_run(tmp_path / "run"))
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["stubbed"] == list(cr.WARDEN_STUBBED)
    assert got["negatives_refused"] == {"tie/forged": True, "n2/tampered": True, "scan/forged": True}
    assert got["steps"]["certificates"]["clock_certificates"] == 0 and got["d1"]["acc"] == [True] * sz.W63.windows
    assert got["steps"]["n2"]["clocks"]["source"] == "none"
    md = cr.markdown(got)
    assert "on the warden path" in md and "| tie | accepted |" in md and "certified I/O" in md
    ep = json.loads(Path(steps.endpoint).read_text())
    Path(steps.endpoint).write_text(json.dumps(ep | {"at": cr.SERVED_NODE}))
    served = cr.report(steps)
    assert served["verdict"] == "accepted" and served["stubbed"] == list(cr.WARDEN_SERVED_STUBBED)
    assert served["steps"]["endpoint"]["in_the_clear"] == cr.CLEAR["warden_endpoint_served"]
    Path(steps.endpoint).write_text(json.dumps(ep))

    def refused(name: str, edit, why: str) -> None:
        doc = json.loads(Path(getattr(steps, name)).read_text())
        Path(getattr(steps, name)).write_text(json.dumps(edit(dict(doc))))
        r = cr.report(steps)
        step = {"certs": "certificates"}.get(name, name)
        assert r["verdict"] == "refused" and why in r["steps"][step]["why"], (name, r["steps"][step])
        Path(getattr(steps, name)).write_text(json.dumps(doc))

    proxy = WC.PROXY
    refused("n2", lambda d: {k: v for k, v in d.items() if k != "warden_run"} | {"run": str(certifier_run)},
            f"N2's summary names no warden run: {proxy}")
    refused("tie", lambda d: d | {"warden_run": "11" * 32}, "the tie's summary is on warden run " + "11" * 32)
    refused("tie", lambda d: d | {"tie": {"honest": [True, True], "forged": [True, True]}}, "forged's refused")
    refused("scan", lambda d: d | {"warden_run": None}, "the scan's summary names no warden run")
    refused("endpoint", lambda d: d | {"io": {"accepted": False, "violations": [[3, "x"]]}}, "read whole off its cells")
    refused("endpoint", lambda d: {k: v for k, v in d.items() if k != "warden_run"},
            f"the endpoint's report names no warden run: {proxy}")
    certifier_certs = cr.certs(certifier_run, certifier_run)
    refused("certs", lambda d: certifier_certs, f"the certificates step's input names no warden run: {proxy}")
    steps.endpoint = steps.tie = None
    r = cr.report(steps)
    assert r["verdict"] == "refused" and r["steps"]["endpoint"]["verdict"] == "not run"
    assert r["steps"]["tie"]["verdict"] == "failed"
