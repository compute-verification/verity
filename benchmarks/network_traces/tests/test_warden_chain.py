"""The network chain on an active network certifier's run (`warden_chain`, `chain_record`'s network certifier path),
here the synthetic run published as `warden_serve.Publisher` publishes a served one, in the gate's layout (its witness
under ``witness/``, its public files under ``out/``, ``certificates.jsonl`` as `auditor.warden_outcome` writes it): the
certificates step reruns the network certifier's own check and holds the public certificates to the certified lines,
refusing a changed one and a changed cell; the run's deployment is the auditor's record's (`warden_chain.deployment_of`),
and a copy of it re-signed, rewritten or in other bytes is refused; the certified I/O read off the cells is the requests
and responses served; N2's link is the network certifier's (both heads at its digest, no sync, charge 0) and its
references expect honest accepted and tampered refused on its signed digest alone. A certifier rehearsal's run (its
records through active.Proxy) is refused by name (`warden_chain.PROXY`) by every reader: its files, or a certifier's
deployment.json in an active network certifier's run. The report on the network certifier path requires the
certificates, N2 and the scan, names its stubs, leaves d1's K open with no advice term (it opens no endpoint records or
advice, and refuses an endpoint report), and refuses a step whose input is a certifier rehearsal's or another network
certifier run's. N2's network verdicts, as its driver writes them there (no egress tie, so each member refused by name
for its egress), count a member accepted when that is its only refusal and no window is refused: an honest chain is
accepted and a negative refused or escaped the draw, never an honest member with a refused window or another refusal
by name."""

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


def gate_run(d: Path, seed: bytes = b"warden-run") -> Path:
    """The synthetic network certifier run in the gate's layout, under the network certifier key ``seed`` makes: its
    files under ``witness/``, its public ones copied to ``out/`` with ``certificates.jsonl``, each certified line's
    certificate or refusal (`auditor.warden_outcome`)."""
    run = published(d / "flat", seed=seed)
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


def test_a_runs_deployment_is_the_auditors_record_or_refused_naming_its_copy_and_field(tmp_path):
    """`deployment_of`: a run whose every deployment.json (its witness/'s and out/'s, the gate's layout) is the
    auditor's record's deployment, in the bytes the record hashes, gives the record's deployment and schedule. A run
    re-signed under another key, one copy of it rewritten, or the same deployment in other bytes is refused naming the
    copy and the field; a record that isn't a `RECORD` whose key and session are its deployment's is refused."""
    run = gate_run(tmp_path / "honest")
    record = WC.own_record(run)
    assert (record["schema"], record["key"], record["session"]) == (WC.RECORD, *map(WC.deployment(run)[0].get,
                                                                                    ("key", "session")))
    assert WC.deployment_of(run, record) == WC.deployment(run)
    other = gate_run(tmp_path / "other", seed=b"another network certifier")
    with pytest.raises(SystemExit, match="witness/deployment.json names another key than the auditor's record's$"):
        WC.deployment_of(other, record)
    out = run / "out" / "deployment.json"
    kept = out.read_bytes()
    shutil.copyfile(other / "out" / "deployment.json", out)
    with pytest.raises(SystemExit, match="out/deployment.json names another key than the auditor's record's$"):
        WC.deployment_of(run, record)
    out.write_text(json.dumps(json.loads(kept), indent=2))
    with pytest.raises(SystemExit, match="out/deployment.json isn't the bytes the auditor's record hashes$"):
        WC.deployment_of(run, record)
    out.write_bytes(kept)
    for bad in ({**record, "schema": "challenger-key/v1"}, {**record, "session": "00" * 32}, {**record, "key": {}},
                {k: v for k, v in record.items() if k != "deployment"}, None):
        with pytest.raises(SystemExit, match=f"the auditor's record isn't a {WC.RECORD} whose key and session are its "
                                             "deployment's$"):
            WC.deployment_of(run, bad)


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
    """With ``--draw K``, K < n, the tampered member changes every window, so whichever K windows the sample holds refuse
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


def warden_network(*refused: int, **why: str) -> dict:
    """A member's network verdict as `n2_zk` writes it on a network certifier link, `LinkWindows.network_verdict` given
    no egress tie: it accepts none, refuses the link by name for its egress (`service.EGRESS_TIED`) and by ``why``'s
    names, and refuses the windows ``refused``."""
    return {"accepted": False, "refused": list(refused), "egress": None,
            "why": {n2.SVC.EGRESS_TIED: "no egress tie was given"} | why}


def _steps(tmp_path, run: Path) -> SimpleNamespace:
    """The network certifier path's step outputs on `run`: its real certificates, and N2 and the scan as their drivers
    write them."""
    def put(name, doc):
        (tmp_path / name).write_text(json.dumps(doc))
        return str(tmp_path / name)

    rid = "00" * 32
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}
    return SimpleNamespace(
        custody=put("custody.json", {"verdict": "accepted"}),
        certs=put("certs.json", WC.certificates(run)),
        endpoint=None,
        n2=put("n2.json", {"ok": True, "warden_run": rid, "members": {
            "honest": {"windows": [window, window], "network": warden_network(), "ok": True},
            "tampered": {"windows": [window | {"accepted": False}, window], "network": warden_network(0),
                         "ok": True}}}),
        scan=put("scan.json", {"ok": True, "T": 32, "warden_run": rid, "rounds": {
            "honest": {"audit": {"window_accepted": True}}, "forged": {"audit": {"window_accepted": False}}}}),
        job=put("job.json", {"path": cr.WARDEN, "rc": {"n2": "0", "scan": "0"}}), warden_run=str(run))


def test_the_warden_path_requires_every_step_on_one_warden_run_and_refuses_a_certifier_rehearsals(tmp_path,
                                                                                                certifier_run):
    steps = _steps(tmp_path, gate_run(tmp_path / "run"))
    got = cr.report(steps)
    assert got["verdict"] == "accepted" and got["stubbed"] == list(cr.WARDEN_STUBBED)
    assert got["negatives_refused"] == {"n2/tampered": True, "scan/forged": True}
    assert set(got["steps"]) == {"custody", "certificates", "n2", "scan", "lean"}
    assert got["steps"]["certificates"]["clock_certificates"] == 0 and got["d1"]["acc"] == [True] * sz.W63.windows
    assert got["d1"]["K"]["open"] == cr.WARDEN_K and got["d1"]["K"]["network"]["advice"] is None
    assert got["steps"]["n2"]["clocks"]["source"] == "none"
    md = cr.markdown(got)
    assert "on the warden path" in md and "| scan | accepted |" in md and "| endpoint |" not in md
    assert all(f"- {x}" in md for x in cr.WARDEN_STUBBED)
    steps.endpoint = str(tmp_path / "endpoint.json")
    Path(steps.endpoint).write_text(json.dumps({"warden_run": "00" * 32, "io": {"accepted": True}}))
    with pytest.raises(SystemExit, match="the network certifier path opens no endpoint records or advice"):
        cr.report(steps)
    steps.endpoint = None

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
    refused("scan", lambda d: d | {"warden_run": "11" * 32}, "the scan's summary is on warden run " + "11" * 32)
    refused("scan", lambda d: d | {"warden_run": None}, "the scan's summary names no warden run")
    certifier_certs = cr.certs(certifier_run, certifier_run)
    refused("certs", lambda d: certifier_certs, f"the certificates step's input names no warden run: {proxy}")


def test_n2_counts_a_network_verdict_refused_by_name_only_for_its_egress():
    """N2's step on the network certifier path accepts the honest member's verdict that refuses only by name for the
    egress, and refuses a negative on its refused window; an honest member with a refused window or another refusal by
    name fails the step, and a sampled negative whose verdict refuses only for the egress escapes the draw when none of
    its incorrect windows was drawn and fails the step when one was."""
    window = {"accepted": True, "proof": {"proved": True}, "as_expected": True}

    def step(ok=True, **members):
        return cr.n2_step({"ok": ok, "members": {k: {"windows": [window]} | v for k, v in members.items()}}, {},
                          {"path": cr.WARDEN, "rc": {"n2": "0"}})

    got = step(honest={"network": warden_network()}, tampered={"network": warden_network(0)})
    assert (got["verdict"], got["why"], got["negatives"]) == ("accepted", None, {"tampered": True}), got
    for honest in (warden_network(0), warden_network(**{n2.SVC.PERIOD_DRAWN: "not the draw of record"})):
        got = step(honest={"network": honest}, tampered={"network": warden_network(0)})
        assert (got["verdict"], got["why"]) == ("failed", "honest: network refused"), got
    reference = [{"window": w, "expect": int(w != 0)} for w in range(4)]

    def moved(units):
        return step(False, honest={"network": warden_network()},
                    moved={"network": warden_network(), "reference": reference, "period": {"units": units}})

    got = moved([1, 2])
    assert (got["verdict"], got["negatives"]) == ("accepted", {"moved": "escaped the draw"}), got
    got = moved([0, 1])
    assert (got["verdict"], got["why"]) == ("failed", "moved: network accepted"), got
