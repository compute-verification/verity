import json
from pathlib import Path

from _build import T0, job

from cluster import cli, ledger, model

NEBIUS = str(Path(__file__).parents[1] / "descriptions" / "nebius.toml")


def test_validate(capsys, tmp_path):
    assert cli.main(["validate", NEBIUS]) == 0
    assert capsys.readouterr().out.strip() == "nebius: 2 nodes, 16 GPUs, 2 shares, 2 workstreams"
    bad = tmp_path / "bad.toml"
    bad.write_text('[priorities]\nwork = 1\n[nodes.a]\nowner = "p"\nmemory_gib = 0\nnuma = { 0 = "0-3" }\n')
    assert cli.main(["validate", str(bad)]) == 1
    assert "memory_gib must be >= 1" in capsys.readouterr().err


def test_plan_prints_one_action_per_line(capsys, tmp_path):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({
        "node": "vy-nebius-2", "now": "2026-10-01T00:00:00Z",
        "queue": [model.job_to_dict(job("v-1", "verity", cpus=8, memory_gib=64)),
                  model.job_to_dict(job("p-1", "pous", gpus=1, cpus=4))],
        "idle_s": {"batch": 600}}))
    assert cli.main(["plan", NEBIUS, str(snap)]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    starts = {d["alloc"]["job"]["id"]: d["alloc"] for d in lines if d["action"] == "start"}
    assert starts["v-1"]["cpus"] == "48-55" and starts["p-1"]["gpus"] == [0]


def test_ledger_commands(capsys, tmp_path):
    p = tmp_path / "l.jsonl"
    lg = ledger.Ledger(p)
    lg.append(T0, "refuse", job="x-1", reason="no share")
    assert cli.main(["ledger", "verify", str(p)]) == 0
    assert cli.main(["ledger", "failures", str(p)]) == 0
    assert cli.main(["ledger", "usage", str(p)]) == 0
    out = capsys.readouterr().out
    assert "1 records, chain intact" in out and '"what": "refused: no share"' in out
    p.write_text(p.read_text().replace("no share", "a share"))
    assert cli.main(["ledger", "verify", str(p)]) == 1


def test_agent_exit_status_tells_a_stop_from_a_crash_from_a_broken_rule(monkeypatch, tmp_path, capsys):
    """The systemd unit restarts on 1 (a crash) and never on 2 (the agent's own safety stop) or 0 (STOP file, SIGTERM)."""
    from cluster import agent
    monkeypatch.setattr(agent, "main_hook", lambda fp: None)
    for summary, rc in (({"stopped": "stop file x", "exception": None}, 0), ({"stopped": "SIGTERM", "exception": None}, 0),
                        ({"stopped": "exception", "exception": "Traceback"}, 1),
                        ({"stopped": "invariant: the plan breaks a rule: x", "exception": None}, 2)):
        monkeypatch.setattr(agent, "run", lambda *a, _s=summary, **k: _s)
        assert cli.main(["agent", NEBIUS, "--out", str(tmp_path), "--mode", "live", "--roll"]) == rc
    capsys.readouterr()
