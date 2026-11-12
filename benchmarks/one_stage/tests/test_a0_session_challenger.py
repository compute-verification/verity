"""a0's session on the challenger (`a0.session` under FLOCK_CHALLENGER=1): a session on the keyed coin tree (`--zk`, or a
plain one under FC_COINS=tree) starts a challenger of its own, its server takes the tree from it (`--tree-from`), and the
challenger's end and audit of the server's records come back as ``live["challenger"]``; a failed one refuses the session.
Any other session starts none, and so does one whose caller names the challenger (``tree_from``). A session's prover (`a0.prover_env`) runs `prove --zk` itself unless it is given Lean's
`flock-firewall`, which puts it behind the ZK server's send check. The server and prover are a stand-in for
`flock-circuit`, the challenger `verity_flock.challengers`' calls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity_flock import challengers as CH

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a0

ENV = ("FC_FIREWALL", "FC_FIREWALL_LEAN", "FC_FIREWALL_STATEMENT", "FC_HOST_WITNESS")

#: a stand-in for flock-circuit: ``serve`` records one session holding its argv and its environment's FC_COINS and ENV,
#: and ``prove`` prints its LIVE line with its own ENV
FAKE = '''#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
env = {{k: os.environ.get(k) for k in {env}}}
if args[0] == "serve":
    s = pathlib.Path(args[args.index("--out") + 1]) / "s0"
    s.mkdir(parents=True, exist_ok=True)
    (s / "session.json").write_text(json.dumps({{"argv": args, "coins": os.environ.get("FC_COINS"), "env": env,
                                                "verdict": {{"accepted": True}}}}))
    print("SERVING", flush=True)
elif args[0] == "prove":
    print("LIVE\\t" + json.dumps({{"accepted": True, "roots": {{}}, "env": env}}))
'''


@pytest.fixture
def run(tmp_path, monkeypatch):
    bin_ = tmp_path / "flock-circuit"
    bin_.write_text(FAKE.format(python=sys.executable, env=ENV))
    bin_.chmod(0o755)
    st = {"dir": tmp_path / "stage", "pin": "00"}
    st["dir"].mkdir()
    calls: dict[str, list] = {"start": [], "end": []}
    ends = {"ok": True}

    def start(d):
        calls["start"].append(d)
        return "proc", ["--tree-from", "127.0.0.1:1", "--tree-secret-file", str(d / "secret")]

    def end(d, p, sessions):
        calls["end"].append((d, p, sessions))
        return {"exit": 0 if ends["ok"] else 1, "sessions": len(sessions), "audit": ends["ok"], "ok": ends["ok"]}

    monkeypatch.setattr(CH, "start", start)
    monkeypatch.setattr(CH, "end", end)
    for k in ("FC_COINS", *ENV):
        monkeypatch.delenv(k, raising=False)

    def go(name, zk=False, env=None, **more):
        out = tmp_path / name
        record, live = a0.session(str(bin_), st, 4, "subset:2", 0, out, zk=zk, env=env, **more)
        return out, record, live

    return go, calls, ends


@pytest.mark.parametrize("zk, env", [(True, {}), (False, {"FC_COINS": "tree"})])
def test_a_tree_session_takes_its_tree_from_its_own_challenger(run, monkeypatch, zk, env):
    go, calls, _ = run
    monkeypatch.setenv("FLOCK_CHALLENGER", "1")
    out, record, live = go("s", zk=zk, env=env)
    assert calls["start"] == [out / "challenger"]
    argv = record["argv"]
    assert argv[argv.index("--tree-from") + 1] == "127.0.0.1:1"
    assert argv[argv.index("--tree-secret-file") + 1] == str(out / "challenger" / "secret")
    assert calls["end"] == [(out / "challenger", "proc", [str(out / "sessions" / "s0")])]
    assert live["accepted"] is True and live["challenger"]["ok"] is True


def test_a_failed_challenger_refuses_the_session(run, monkeypatch):
    go, _, ends = run
    monkeypatch.setenv("FLOCK_CHALLENGER", "1")
    ends["ok"] = False
    _, record, live = go("s", zk=True)
    assert record["verdict"]["accepted"] is True
    assert live["accepted"] is False and live["why"].startswith("challenger: ") and live["challenger"]["ok"] is False


@pytest.mark.parametrize("flag, zk, env", [(None, True, {}), ("0", True, {}), ("1", False, {})])
def test_any_other_session_starts_no_challenger(run, monkeypatch, flag, zk, env):
    go, calls, _ = run
    if flag is None:
        monkeypatch.delenv("FLOCK_CHALLENGER", raising=False)
    else:
        monkeypatch.setenv("FLOCK_CHALLENGER", flag)
    _, record, live = go("s", zk=zk, env=env)
    assert calls == {"start": [], "end": []}
    assert "--tree-from" not in record["argv"] and "challenger" not in live


def test_a_session_on_its_callers_challenger_starts_none_of_its_own(run, monkeypatch):
    """With ``tree_from`` the server takes its tree from the challenger its caller names, under FLOCK_CHALLENGER=1 too:
    no challenger is started or audited here, and the session's ``live`` carries none."""
    go, calls, _ = run
    monkeypatch.setenv("FLOCK_CHALLENGER", "1")
    _, record, live = go("s", zk=True, tree_from=["--tree-from", "127.0.0.1:2", "--tree-secret-file", "relay-secret"])
    assert calls == {"start": [], "end": []}
    argv = record["argv"]
    assert argv[argv.index("--tree-from") + 1] == "127.0.0.1:2"
    assert argv[argv.index("--tree-secret-file") + 1] == "relay-secret"
    assert live["accepted"] is True and "challenger" not in live


def test_a_session_proves_itself_unless_given_the_zk_server(run, monkeypatch):
    """Without a firewall both sides run with the ZK server off and the witness where the prover builds it, a
    FC_HOST_WITNESS the caller set removed (`a0.DIRECT`, zk_cost's `dev` arm); with one, the ZK server's send check is on
    and the caller's environment stands."""
    go, _, _ = run
    monkeypatch.delenv("FLOCK_CHALLENGER", raising=False)
    monkeypatch.setenv("FC_HOST_WITNESS", "1")
    _, record, live = go("direct", zk=True, env=a0.prover_env(None, {"partition_file": "p.json"}))
    assert record["env"] == live["env"] == {"FC_FIREWALL": "0", "FC_FIREWALL_LEAN": None, "FC_FIREWALL_STATEMENT": None,
                                            "FC_HOST_WITNESS": None}
    _, record, live = go("fw", zk=True, env=a0.prover_env("flock-firewall", {"partition_file": "p.json"}))
    assert record["env"] == live["env"] == {"FC_FIREWALL": "1", "FC_FIREWALL_LEAN": "flock-firewall",
                                            "FC_FIREWALL_STATEMENT": a0.STATEMENT_ID, "FC_HOST_WITNESS": "1"}
    assert a0.firewall_env({"partition_file": "p.json"})["FC_FIREWALL_HELLO_ARGS"] == "--partition p.json"
