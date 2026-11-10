"""scan_zk's honest round on the challenger's coins (``prove --draw K --challenger local``) at the test class, through
`scan_zk.audit` as `scan_zk.prove` runs it: the round's generator-form Program lowered and its members staged, a local
challenger session on the group's root (`a5_coins.start`, its serve.py under live_challenger's stub of the Lean
verifier's coins), and the round's `service.Stream` registering and sampling the round on its coins. Only the proofs
are stubbed (`a8.session`, `a0.lean_verify`): proving needs Flock's and the Lean's binaries, which a node's run of
``scan_zk.sh`` builds. Of the altered coins' evidence only the evidence for another window is replayed
(`a5_coins.negatives`' are a5_coins' tests')."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
STUB = HERE.parents[1] / "verity" / "composite" / "device" / "live_challenger" / "tests" / "stub_serve.py"


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_scan_zk", HERE / "scan_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sz = _load()


@pytest.mark.slow
def test_the_honest_round_on_a_local_challengers_coins_registers_its_generator_form_and_draws(tmp_path, monkeypatch):
    """The stream holds the registration's R1-R7 against its own copy of the round's Program, the generator form the
    round registers (`scan_root`, `stream_of`), so it registers and samples the round: a stratified sample of 4 with
    each template's stratum sampled, a session for each member holding a sampled unit and only for those, a
    registration of the generator form's digest and partition, and coins whose evidence holds against the group's root
    and the registration's digest. A stream that rebuilt the templates' mixed population instead refused it at R2
    (``R2-program``, ``R2-partition``) and drew nothing. Each member's registered reads, the verifier's file and its
    statement's header (the prover's and the verifier's alike, or their statement digests differ), name every port of
    its circuit, its o0 at the registration's root for it, as Lean's `Registered.checkComplete` requires.

    Runs 4 to 10 minutes on a node, about 4 alone: the synthetic run, lowering and staging the scan's three templates,
    and the Lean verifier's checks of the stream's registration, sample and run. It catches what no cheaper test
    can, since only the honest challenger round on the scan's own staged statements has both:
    - the R2 Program and partition mismatch, between the generator form the round registers and the Program the
      challenger's stream rebuilds (the stream's own window tests use toy templates, never `scan_root`);
    - a port of a staged member left out of its registered reads, which Lean's `checkComplete` refuses, so that every
      session fails Lean's verify.
    The test class is the smallest, the group is one link-window, and the draw is one more than the three strata: the
    sessions are stubbed, so a smaller draw saves nothing."""
    setup, k = sz.TEST, 4
    h = sz.host(tmp_path / "warden", draw=k, setup=setup)
    at = h.at
    M = sz.WZ._one_stage()
    import a5_coins as AC

    pg = sz.program(M, setup.cls, sz.TEMPLATES, at.G, at.T, tmp_path / sz.PROGRAMS[sz.TEMPLATES])
    members = [sz.stage(M, pg, mr, tmp_path / "honest" / f"stage-{mr.name}") for mr in h.rounds["honest"]]

    A = AC._module("a5_auditor", AC.TREE / "verity" / "composite" / "device" / "live_challenger" / "auditor.py")
    real = A.serve_argv
    monkeypatch.setattr(A, "serve_argv", lambda *a, **kw: [sys.executable, str(STUB), *real(*a, **kw)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = tmp_path / "main.txt"
    main.write_text(AC.on_main([]) + "\n")
    sessions: dict[str, list[int]] = {}

    def session(a, m, sdir, local, port, env=None):
        sessions[m["name"]] = list(local)
        return None, None, {"wall_s": 0.0}
    monkeypatch.setattr(M.a8, "session", session)
    monkeypatch.setattr(M.a0, "lean_verify", lambda *a, **kw: {"accepted": False, "why": "not proved here"})
    monkeypatch.setattr(sz.WZ, "firewall", lambda M, m: {})
    monkeypatch.setattr(AC, "negatives", lambda ev, pre: {})

    a = SimpleNamespace(challenger="local", main=str(main), gate_key=None, peer=None, challenger_key=None)
    av = SimpleNamespace(bin=None, lean=None, lease=[], gpu=False, delta=2.0 ** -20)
    with sz.challenger(a, at, tmp_path) as coins:
        res = sz.audit(M, av, at, pg, "honest", members, dict(at.trees), tmp_path / "honest", 0, k, coins)
    assert "refused" not in res["draw"], res["draw"].get("refused")
    assert (coins.published or {}).get("error") is None, coins.published.get("error")

    units, bases = res["draw"]["units"], {m["name"]: (m["base"], m["n"]) for m in members}
    first, ingress, egress = (pg.stagers[t] for t in sz.TEMPLATES)
    for sg in (first, ingress, egress):
        assert any(sg.base <= u < sg.base + sg.n for u in units), (sg.defn.id, units)
    assert sessions == {name: [u - b for u in units if b <= u < b + n]
                        for name, (b, n) in bases.items() if any(b <= u < b + n for u in units)}

    out = tmp_path / "honest"
    rec = json.loads((out / "registration.json").read_text())
    part = M.P.partition(pg.program, pg.query)
    assert rec["program"] == M.P.program_digest(pg.program) == part["program"]
    assert rec["partition"] == M.P.digest(part)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt and json.loads((out / "draw.json").read_text())["units"] == units
    assert res["coins"]["coins_reason"] is None

    roots = {r["port"]: r["root"] for r in rec["roots"]}
    for m in members:
        ports = json.loads(m["registered_file"].read_text())["ports"]
        head = json.loads((m["dir"] / f"pub-{m['n']}.bin").read_bytes().split(b"\n", 1)[0])
        inst = json.loads((m["dir"] / f"inst-{m['n']}.bin").read_bytes().split(b"\n", 1)[0])
        assert sorted(ports) == sorted(head["registered"]) == sorted(p.name for p in m["ports"]), m["name"]
        assert inst["registered"] == head["registered"], m["name"]
        assert ports["o0"]["root"] == roots[f"{m['at']}/o0"] and ports["o0"]["reads"] == list(range(m["n"]))
