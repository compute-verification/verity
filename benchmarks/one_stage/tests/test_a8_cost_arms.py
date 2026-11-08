"""a8's cost arms (`a8.cost`): ``fw`` and ``nofw`` run both sides of each session with ``--zk`` and Lean verifies with
``--zk``, ``fw`` under the ZK server's send check; ``plain`` runs both sides without ``--zk`` on the keyed coin tree
(`FC_COINS=tree`), the send check off, and Lean verifies with ``--coins tree``. The server, prover and Lean verifier are
stand-ins for `flock-circuit` and `flock-verify`."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a8

#: flock-circuit's ``serve`` records one session holding its argv and its environment's FC_COINS and FC_FIREWALL, and
#: ``prove`` prints its LIVE line
FLOCK = '''#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == "serve":
    s = pathlib.Path(args[args.index("--out") + 1]) / "s0"
    s.mkdir(parents=True, exist_ok=True)
    env = {{k: os.environ.get(k) for k in ("FC_COINS", "FC_FIREWALL")}}
    (s / "session.json").write_text(json.dumps({{"argv": args, "env": env, "verdict": {{"accepted": True}}}}))
    print("SERVING", flush=True)
elif args[0] == "prove":
    print("LIVE\\t" + json.dumps({{"accepted": True}}))
'''
#: flock-verify's ``verify`` prints an accepting VERDICT line
LEAN = '''#!{python}
import json, sys
if sys.argv[1] == "verify":
    print("VERDICT " + json.dumps({{"accepted": True}}))
'''


@pytest.mark.parametrize("arm, zk", [("fw", True), ("nofw", True), ("plain", False)])
def test_each_arm_runs_its_session_and_lean_under_its_protocol(tmp_path, monkeypatch, arm, zk):
    monkeypatch.delenv("FC_COINS", raising=False)
    tools = {}
    for name, src in (("flock-circuit", FLOCK), ("flock-verify", LEAN)):
        f = tmp_path / name
        f.write_text(src.format(python=sys.executable))
        f.chmod(0o755)
        tools[name] = str(f)
    stage = tmp_path / "stage"
    (stage / "tables").mkdir(parents=True)
    (stage / "circuit.txt").write_text("circuit")
    a = SimpleNamespace(bin=tools["flock-circuit"], lean=tools["flock-verify"], lease=[], gpu=False, points=[1],
                        arms=[arm], cold=1)
    rows = a8.cost(a, {"dir": stage, "n": 1, "pin": "00"}, tmp_path / "cost", 0)
    assert [(r["arm"], r["accepted"], r["lean"]["accepted"]) for r in rows] == [(arm, True, True)]
    sdir = tmp_path / "cost" / f"{arm}-k1-r0"
    record = json.loads(next((sdir / "sessions").glob("*/session.json")).read_text())
    prove = (sdir / "prove.log").read_text().splitlines()[0].split()
    verify = (sdir / "lean-verify.log").read_text().splitlines()[0].split()
    assert ["--zk" in x for x in (record["argv"], prove, verify)] == [zk] * 3
    assert record["env"] == {"FC_COINS": None if zk else "tree", "FC_FIREWALL": "1" if arm == "fw" else "0"}
    coins = verify[verify.index("--coins") + 1] if "--coins" in verify else None
    assert coins == (None if zk else "tree")
