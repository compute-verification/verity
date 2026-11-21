"""`cluster status`: node 2's ledger state as rows by owner and key, GPU-h ready and disk, from snapshots in the shape the
live node returns."""
from __future__ import annotations

import json
import re

from _build import job

from cluster import cli, ledger, model, nebius2, status
from cluster.ledger import epoch

T = epoch("2026-10-01T00:30:00Z")
DF = ("Filesystem     1024-blocks       Used  Available Capacity Mounted on\n"
      "/dev/vdc        5259290696 3638812520 1620461792      70% /workspace\n")


def node2_snapshot() -> dict:
    fill = job("gl-43527", "pous", submitter="fill:bc-7442ca43", priority="fill", gpus=1, cpus=0, max_s=480,
               node=nebius2.NODE, gpu_ids=(1,))
    sess = job("gl-16021", "pous", submitter="verity:bc-698052e1-cov-g217-proof", priority="session", kind="session",
               gpus=1, cpus=0, max_s=None, expected_s=1800, node=nebius2.NODE, run="r20260930-235745-a3d0")
    waiter = job("gl-57723", "pous", submitter="fill:bc-dbc19788", priority="fill", gpus=2, cpus=0, max_s=900,
                 node=nebius2.NODE)
    cpu = job("v-1", "verity", submitter="circuits", cpus=8, max_s=None, kind="session", node=nebius2.NODE)
    allocs = {a.id: a for a in (
        model.Allocation(id=f"gl-43527@{T - 120}", job=fill, node=nebius2.NODE, gpus=(1,), cpus=frozenset(), memory_gib=193,
                         started=T - 120),
        model.Allocation(id=f"gl-16021@{T - 600}", job=sess, node=nebius2.NODE, gpus=(2,), cpus=frozenset(), memory_gib=1,
                         started=T - 600, state="frozen", frozen_by="w", frozen_at=T - 60))}
    queue = {waiter.id: ledger.Queued(waiter, nebius2.NODE, "wait.17908.57723", T - 30),
             cpu.id: ledger.Queued(cpu, nebius2.NODE, None, T - 5)}
    st = ledger.State(allocs, queue, seq=479, head="h", t=T - 2)
    return {"node": nebius2.NODE, "t": T, "df": DF.replace("70%", "41%"), "state": ledger.state_to_dict(st),
            "state_path": "/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json"}


def test_node2_rows_are_the_ledgers_leases_and_waiters_by_submitter_and_run():
    rep = status.node2(node2_snapshot(), T)
    rows = {r["name"]: r for r in rep["jobs"]}
    lease = rows[f"gl-43527@{T - 120}"]
    assert (lease["owner"], lease["key"], lease["state"], lease["gpus"], lease["cpus"], lease["for_s"], lease["cls"]) == \
        ("fill:bc-7442ca43", "gl-43527", "running", 1, 0, 120, "fill")
    sess = rows[f"gl-16021@{T - 600}"]
    assert (sess["key"], sess["state"], sess["length_s"], sess["length_is"]) == ("r20260930-235745-a3d0", "frozen", 1800, "expected")
    wait = rows["wait.17908.57723"]
    assert (wait["state"], wait["gpus"], wait["for_s"], wait["length_s"]) == ("queued", 2, 30, 900)
    assert rep["ready"] == {"jobs": 2, "gpus": 2, "gpu_jobs": 1, "known": 1, "gpu_h": 0.5}
    assert rep["disk"]["use_pct"] == 41
    assert rep["ledger"] == {"path": "/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json", "seq": 479, "t": T - 2,
                             "age_s": 2, "window": False, "stale": False}
    assert status.node2(node2_snapshot(), T + status.STALE_S + 10)["ledger"]["stale"]
    assert [r["owner"] for r in status.node2(node2_snapshot(), T, owner="circuits")["jobs"]] == ["circuits"]


def test_node2_state_held_back_by_a_timed_window_is_stale_only_after_the_window_should_have_ended():
    timed = job("gl-314342", "pous", submitter="bc-dd22acf8", priority="timed", quiet=True, gpus=8, cpus=0, max_s=1200,
                node=nebius2.NODE, run="r20260930-221231-3dd1")
    a = model.Allocation(id=f"gl-314342@{T}", job=timed, node=nebius2.NODE, gpus=tuple(range(8)), cpus=frozenset(),
                         memory_gib=1544, started=T)
    snap = {**node2_snapshot(), "state": ledger.state_to_dict(ledger.State({a.id: a}, {}, seq=522, head="h", t=T))}
    during = status.node2(snap, T + 900)["ledger"]
    assert (during["window"], during["stale"], during["age_s"]) == (True, False, 900)
    assert status.node2(snap, T + 1200 + status.STALE_S + 1)["ledger"]["stale"]
    assert "seq 522, 15m old (a timed window: the agent writes no state until it ends)" in status.render([status.node2(snap, T + 900)])


def test_render_prints_each_nodes_header_then_its_jobs_by_owner():
    text = status.render([status.node2(node2_snapshot(), T), {"node": "vy-nebius-3", "t": T, "error": "ReadError: no route"}])
    assert "vy-nebius-2 at 2026-10-01T00:30:00Z\n  disk: /workspace 41% used (df)\n" in text
    assert "  ledger state: seq 479, 2s old (/workspace/pouw/infra/cluster/live/20260930T2320Z/state.json)\n" in text
    lines = text.splitlines()
    n2 = lines[lines.index("  ready: 2 queued, 2 GPUs asked; 0.50 GPU-h") + 1:]
    assert n2[0].split() == ["STATE", "OWNER", "KEY", "GPU", "CPU", "FOR"]
    assert n2[2].split() == ["running", "fill:bc-7442ca43", "gl-43527", "1", "-", "2m"]
    assert "vy-nebius-3: could not read: ReadError: no route" in text


def test_cli_status_reads_the_node_with_read_only_commands_and_prints_json(capsys, monkeypatch):
    sent: list[tuple[str, str]] = []
    out = {nebius2.NODE: {"state": "/live/state.json\n" + json.dumps(node2_snapshot()["state"]), "df": DF}}

    def runner(node):
        def run(cmd):
            sent.append((node, cmd))
            return out[node][next(k for k, v in status.COMMANDS.items() if v == cmd)]
        return run

    monkeypatch.setattr(status, "clock", lambda: T)
    monkeypatch.setattr(status, "main", lambda nodes, owner, as_json, runner_=runner, main=status.main:
                        main(nodes, owner, as_json, runner=runner_))
    assert cli.main(["status", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "cluster/status/v1" and doc["t"] == T
    assert [n["node"] for n in doc["nodes"]] == [nebius2.NODE]
    assert doc["nodes"][0]["ledger"]["path"] == "/live/state.json"
    assert sorted(sent) == sorted((nebius2.NODE, status.COMMANDS[k]) for k in ("state", "df"))
    writes = re.compile(r"\b(patch|delete|apply|create|edit|label|annotate|scale|cordon|drain|rm|mv|kill|tee|chmod|chown)\b|>(?!/dev/null)")
    for _, cmd in sent:
        assert not writes.search(cmd), cmd


def test_cli_status_reports_a_node_it_cannot_read_and_exits_1(capsys, monkeypatch):
    def runner(node):
        def run(cmd):
            raise status.ReadError(f"{node}: ssh: connect to host: Connection timed out")
        return run

    monkeypatch.setattr(status, "clock", lambda: T)
    monkeypatch.setattr(status, "main", lambda nodes, owner, as_json, main=status.main: main(nodes, owner, as_json, runner=runner))
    assert cli.main(["status", "--on", "nebius-2"]) == 1
    assert capsys.readouterr().out == "vy-nebius-2: could not read: ReadError: vy-nebius-2: ssh: connect to host: Connection timed out\n"


def test_a_nodes_address_is_the_environments_and_unset_its_read_fails_naming_the_variable(capsys, monkeypatch):
    """No file holds a node's address: `reach` sends each command to research@ the address the node's variable holds,
    and with it unset the read fails as an unreachable node does, naming the variable, and nothing is sent."""
    sent: list[list[str]] = []

    class Ran:
        returncode, stdout, stderr = 0, DF, ""

    monkeypatch.setattr(status.subprocess, "run", lambda argv, **kw: sent.append(argv) or Ran())
    monkeypatch.setenv("VY_NEBIUS_2_HOST", "192.0.2.12")
    assert status.reach(nebius2.NODE, None)("df -P /workspace") == DF
    assert [argv[-2:] for argv in sent] == [["research@192.0.2.12", "df -P /workspace"]]
    monkeypatch.delenv("VY_NEBIUS_2_HOST")
    monkeypatch.setattr(status, "clock", lambda: T)
    assert cli.main(["status", "--on", "nebius-2"]) == 1
    assert capsys.readouterr().out == ("vy-nebius-2: could not read: ReadError: VY_NEBIUS_2_HOST is unset: vy-nebius-2's "
                                       "address, which machines.d/vy-nebius-2.toml registers as host\n")
    assert len(sent) == 1
