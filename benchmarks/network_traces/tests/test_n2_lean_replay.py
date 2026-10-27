"""`n2_lean_replay.py` on a recorded run made here as `n2_zk.sh` keeps one: certifier-serve's runs on an injected clock
(`n2_zk.virtual`), each member's link-windows registered on the link's stream, delivered as a stand-in proof (the unit's
rows committed as C-Flock's public file holds them, the findings a stand-in for the verifier's own Lean run) and decided
by `LinkWindows.outcome`, with the files `n2_zk.member` writes. The replay finds the Lean and the Python agreeing with
each other and with each recorded outcome on every window, honest and tampered, and every control rejected by both; an
outcome recorded with another verdict is caught. A warden run's N2, as `chain_record.sh` keeps it on the warden path,
replays on the warden link rebuilt from the warden run's kept deployment.json, with its own controls."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity.core.protocols.certifier import service as SVC
from verity.core.protocols.sampled_proofs.one_stage import partition as P, proofs as PF
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding

HERE = Path(__file__).resolve().parents[1]
SALTS = {"pub": b"\x51" * 192, "wit": b"\x52" * 192, "o0": b"\x53" * 192}


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


n2 = _load("n2_zk")
rp = _load("n2_lean_replay")


def _bc(data: bytes, bits: int, salt: bytes) -> bytes:
    return hm96.SHA512.commit_string(hm96.SHA512.default_key, sha512_row_v2_digest(data, bits), salt)


def _member(name: str, rd, base: Path, monkeypatch) -> dict:
    """The member's windows proved by the stand-in and decided by `LinkWindows.outcome`, kept as `n2_zk.member` keeps
    them; its summary rows."""
    lw, vdir, rows, previous = rd.lw, base / name / "verifier", [], None
    vdir.mkdir(parents=True)
    stream = lw.stream(budget=rd.plan["windows"], delta=n2.cs.DELTA, challenger="n2-zk", coins=n2.cs._no_coins,
                       clock=lambda: 0.0)
    for x in rd.windows:
        pre, wdir = f"w{x.window:03d}-", base / name / f"w{x.window:03d}"
        reg = lw.register(stream, lw.registration(x.opening, previous))
        previous, draw = reg.digest, stream.select(reg)
        data = {"pub": lw.row(x.args, "pub"), "wit": lw.row(x.args, "wit"), "o0": SVC.output_row(x.word)}
        (wdir / "l0000-1").mkdir(parents=True)
        public = wdir / "pub-1.bin"
        public.write_bytes(json.dumps({"units": {"indices": [0]}, "instances": 1}).encode() + b"\n"
                           + b"".join(_bc(data[p], lw.bits(p), SALTS[p]) for p in SVC.PORTS))
        (wdir / "l0000-1" / "session.json").write_text(json.dumps({"window": x.window, "member": name}))
        delivered = SVC.Delivered(PF.Proof(wdir / "circuit.txt", public, (wdir / "l0000-1",)),
                                  {"pub": SALTS["pub"], "o0": SALTS["o0"]}, x.word)
        monkeypatch.setattr(SVC.PF, "verify", lambda proof, **kw: (
            [Finding(str(s), True, "", lw.partition["program"], {}, (0,)) for s in proof.sessions], None))
        assert stream.deliver(draw, delivered.sessions()) is None
        lw.outcome(stream, draw, x.certificates, rd.syncs, delivered, circuit_sha512="0" * 128)
        files = {"registration.json": reg.record, "draw.json": dict(draw.draw),
                 "certificates.json": [n2._certificate(c) for c in x.certificates],
                 "syncs.json": [[s.delta, str(s.rho)] for s in rd.syncs], "outcome.json": stream.records[-1],
                 "delivered.json": {"salts": {r: s.hex() for r, s in delivered.salts.items()}, "output": x.word}}
        for f, v in files.items():
            (vdir / f"{pre}{f}").write_text(json.dumps(v, indent=1, sort_keys=True, default=str))
        os.link(public, vdir / f"{pre}pub-1.bin")
        os.link(wdir / "l0000-1" / "session.json", vdir / f"{pre}session-l0000-1-session.json")
        rows.append({"window": x.window, "proof": {"refused": None, "proved": True}})
    return {"windows": rows}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    run = tmp_path_factory.mktemp("n2-run")
    runs, base = run / "out" / "runs", run / "out" / "n2zk"
    n2.virtual(runs, requests=8, tokens=4, strategies=("honest",))
    members = {"honest": n2.link_windows(runs / "honest"),
               "tampered": n2.link_windows(runs / "honest", edit=n2.one_count)}
    lw = members["honest"].lw
    with pytest.MonkeyPatch.context() as mp:
        summary = {"link": {"name": lw.link, "unit": lw.unit.id, "program": lw.partition["program"],
                            "partition": P.digest(lw.partition), "gates": [g.id for g in lw.gates()]},
                   "members": {k: _member(k, rd, base, mp) for k, rd in members.items()}}
    (base / "summary.json").write_text(json.dumps(summary))
    return run


def test_the_replay_finds_lean_and_python_agreeing_with_the_record(recorded, tmp_path):
    out = rp.replay_run(recorded, tmp_path)
    honest, tampered = out["members"]["honest"], out["members"]["tampered"]
    assert honest["ok"] and tampered["ok"], out
    assert len(honest["windows"]) >= 2 and set(honest["accepted"]) == {1}
    assert tampered["accepted"][0] == 0 and set(tampered["accepted"][1:]) == {1}
    for m in (honest, tampered):
        assert m["structure"] | {"kind": None, "by": None} == {"program": True, "partition": True, "gates": True,
                                                              "kind": None, "by": None}
        for w in m["windows"]:
            assert w["agree"] and w["as_recorded"] and w["registration"] and w["findings"] == "summary"
            assert set(w["controls"]) == set(rp.CONTROLS)
            assert all(c["passed"] and c["python"] is False and c["lean"] is False for c in w["controls"].values())
    assert rp.main(["--out", str(tmp_path / "out"), "--jobs", "1", str(recorded)]) == 0
    total = json.loads((tmp_path / "out" / "replay.json").read_text())["total"]
    n = len(honest["windows"]) + len(tampered["windows"])
    assert total | {"ok": None} == {"runs": 1, "windows": n, "agree": n, "as_recorded": n, "assumed": 0,
                                    "controls": n * len(rp.CONTROLS), "controls_passed": n * len(rp.CONTROLS),
                                    "accepted": n - 1, "rejected": 1, "ok": None} and total["ok"]


@pytest.fixture(scope="module")
def recorded_warden(tmp_path_factory):
    """A warden run's N2 as `chain_record.sh` keeps it: the synthetic warden run (`test_scan_zk.published`) proved by the
    stand-in under `out/n2/`, the summary naming the warden run at a scratch path that is gone, and the warden run's
    public deployment.json under `out/gates/g1/out/`."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_scan_zk import published

    d = tmp_path_factory.mktemp("warden")
    warden = published(d / "warden")
    run = d / "run"
    members = {"honest": n2.link_windows(warden, warden=True),
               "tampered": n2.link_windows(warden, edit=n2.one_count, warden=True)}
    lw = members["honest"].lw
    with pytest.MonkeyPatch.context() as mp:
        summary = {"run": str(d / "gone"), "warden_run": members["honest"].plan["warden_run"],
                   "link": {"name": lw.link, "unit": lw.unit.id, "program": lw.partition["program"],
                            "partition": P.digest(lw.partition), "gates": [g.id for g in lw.gates()]},
                   "members": {k: _member(k, rd, run / "out" / "n2", mp) for k, rd in members.items()}}
    (run / "out" / "n2" / "summary.json").write_text(json.dumps(summary))
    (run / "out" / "gates" / "g1" / "out").mkdir(parents=True)
    shutil.copyfile(warden / "deployment.json", run / "out" / "gates" / "g1" / "out" / "deployment.json")
    return run


def test_a_warden_runs_n2_replays_on_the_warden_link_with_its_own_controls(recorded_warden, tmp_path):
    """The link is the warden's, read off the kept deployment.json; the Lean decides (``by: lean``) and agrees with the
    Python and the record on every window; a sync off Σ_sync changes neither verdict (``unread``) and the link under
    another warden digest is rejected by both (``digest``), beside the other controls."""
    out = rp.replay_run(recorded_warden, tmp_path)
    assert out["warden"] == {"run": "00" * 32, "dir": "out/gates/g1"}
    honest, tampered = out["members"]["honest"], out["members"]["tampered"]
    assert honest["ok"] and tampered["ok"], out
    assert len(honest["windows"]) >= 2 and set(honest["accepted"]) == {1}
    assert tampered["accepted"][0] == 0 and set(tampered["accepted"][1:]) == {1}
    for m in (honest, tampered):
        assert m["structure"] == {"program": True, "partition": True, "gates": True, "kind": "warden", "by": "lean"}
        for w in m["windows"]:
            assert w["agree"] and w["as_recorded"] and w["registration"]
            assert set(w["controls"]) == set(rp.WARDEN_CONTROLS)
            for name, c in w["controls"].items():
                kept = name == "unread" and w["python"]["accepted"]
                assert c["passed"] and c["python"] == c["lean"] == kept, (name, c)
    with pytest.raises(FileNotFoundError, match="no deployment.json of warden run"):
        rp.warden_dir(recorded_warden, {"warden_run": "11" * 32, "run": str(tmp_path / "gone")})


def test_a_runs_pin_names_a_size_class_the_tree_leaves_unpinned_and_never_replaces_one():
    unpinned = rp.SV.gate(9, 7, 1, 2, 2, 1)
    assert unpinned.id.endswith("@unpinned]_v2")
    fid = unpinned.id[len("Gate["):-len("@unpinned]_v2")]
    tree = next(iter(rp.HC._pins(rp.SV.PINS).items()))
    added = rp.run_pins([f"Gate[{fid}@{'ab' * 64}]_v2", f"Gate[{tree[0]}@{'cd' * 64}]_v2"])
    try:
        assert added == [fid]
        assert rp.SV.gate(9, 7, 1, 2, 2, 1).id == f"Gate[{fid}@{'ab' * 64}]_v2"
        assert rp.HC._pins(rp.SV.PINS)[tree[0]] == tree[1]
    finally:
        rp._unpin(added)
    assert rp.SV.gate(9, 7, 1, 2, 2, 1).id == unpinned.id


def test_an_outcome_recorded_with_another_verdict_is_caught(recorded, tmp_path):
    vdir = recorded / "out" / "n2zk" / "honest" / "verifier"
    path = vdir / "w001-outcome.json"
    kept = path.read_text()
    body = json.loads(kept)
    body["audit"]["verdicts"][SVC.VERIFIER]["public"]["deadline"] = 0
    path.write_text(json.dumps(body))
    try:
        out = rp.replay_run(recorded, tmp_path)
    finally:
        path.write_text(kept)
    honest = out["members"]["honest"]
    assert not honest["ok"]
    assert [w["window"] for w in honest["windows"] if not w["as_recorded"]] == [1]
    assert all(w["agree"] for w in honest["windows"])
