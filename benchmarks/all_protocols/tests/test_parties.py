"""The toy's machine for sampled proofs' parties (`verity_all_protocols.parties`), the parties themselves core's:

* core's table holds the challenger, firewall, workers and verifier, and the example configuration places exactly those
  parties, its channels between them, each ``after`` a message an earlier channel carries;
* a configuration listing a component no party knows is refused by name;
* the product capability runs over exactly the ids it is given, an absent component's case one its own decoders refuse,
  and the commitment binds the listed components' digests under the product's own id (a stand-in ``product``, which
  records what it was given);
* the deployment gives a listening address only to a party that listens for the verifier, and states the coins
  ``simulated`` when every party is here or the challenger shares a prover's machine;
* a verifier whose listening peers run elsewhere at no address decides nothing (2);
* opt-in (``VERITY_ALL_PROTOCOLS_TOOLS=tools.json``, `build.sh`'s; minutes): the example configuration with every party on
  this machine, honest, accepted on the challenger's coins; tampered, rejected; and with the certifier listed but no
  certifier party, ``certifier: absent`` and the product rejects.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest

from verity_all_protocols import compose as C, parties as P

EXAMPLE = P.ROOT / "examples" / "toy_network" / "config.toml"
SP, CERT = "verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"


def test_the_roles_table_and_the_example_configuration_agree():
    assert set(P.ROLES) == {"challenger", "firewall", "workers", "verifier"}
    config = P.config_of(EXAMPLE)
    assert config["protocols"] == ["sampled-proofs"] and "epsilon" not in json.dumps(config)
    assert sorted(p["name"] for p in config["party"]) == sorted(P.ROLES)
    machines = {p["name"]: p["machine"] for p in config["party"]}
    assert machines["challenger"] == machines["verifier"] != machines["workers"] == machines["firewall"]
    said: set[str] = set()
    for ch in config["channel"]:
        assert ch["from"] in machines and ch["to"] in machines
        assert ch.get("after") is None or ch["after"] in said, f"{ch} is after a message no earlier channel carries"
        said |= set(ch["messages"])
    assert [c for c in config["channel"] if c.get("private")] == [
        {"from": "workers", "to": "firewall", "messages": ["witness"], "after": "draw", "private": True}]


def test_a_component_no_role_knows_is_refused(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('protocols = ["sampled-proofs", "wipe"]\n')
    with pytest.raises(SystemExit, match="protocols lists"):
        P.config_of(f)
    f.write_text("protocols = []\n")
    with pytest.raises(SystemExit, match="protocols lists"):
        P.config_of(f)


def product_exe(tmp_path: Path) -> Path:
    """A stand-in `product IDS id|check`: the subset's id, and on check a reject when any case is empty, recording its
    inputs in ``seen.json``."""
    exe = tmp_path / "product"
    exe.write_text(f"#!{sys.executable}\n" + f'''import json, sys
ids, op = sys.argv[1].split(","), sys.argv[2]
if op == "id":
    print("verity/onsite/v0" if len(ids) == 5 else "verity/product/v0")
    sys.exit(0)
p, c, e = (open(f, "rb").read() for f in sys.argv[3:6])
json.dump({{"ids": ids, "params": p.decode(), "commitment": c.hex(), "evidence": e.decode()}}, open({str(tmp_path / "seen.json")!r}, "w"))
empty = [x["id"] for x in json.loads(e)["components"] if not x["evidence"]]
print(json.dumps({{"verdict": "reject" if empty else "accept",
                  "reason": "; ".join(i + " reject: params: missing" for i in empty) or None}}))
sys.exit(1 if empty else 0)
''')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


CASES = {SP: {"commitment": "01", "evidence": "02", "params": {"k": 8}}, CERT: C.NO_CASE}
DIGESTS = {SP: "ab" * 64, CERT: ""}


def test_an_absent_components_case_is_refused_and_the_product_never_accepts_over_it(tmp_path):
    exe = product_exe(tmp_path)
    got = P.product(str(exe), [SP, CERT], CASES, DIGESTS)
    assert got == {"product": "verity/product/v0", "verdict": "reject", "reason": f"{CERT} reject: params: missing"}
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["ids"] == [SP, CERT]
    ev = {x["id"]: x for x in json.loads(seen["evidence"])["components"]}
    assert ev[CERT] == {"id": CERT, **C.NO_CASE} and ev[SP]["evidence"] == "02"
    assert seen["commitment"] == C.bind("verity/product/v0", [(CERT, ""), (SP, "ab" * 64)]).hex()


def test_the_product_over_one_listed_component_is_that_components(tmp_path):
    exe = product_exe(tmp_path)
    got = P.product(str(exe), [SP], {SP: CASES[SP]}, {SP: DIGESTS[SP]})
    assert got["verdict"] == "accept"
    assert json.loads((tmp_path / "seen.json").read_text())["ids"] == [SP]


def test_the_deployment_gives_addresses_to_listeners_and_states_its_coins():
    config = P.config_of(EXAMPLE)
    at = {"challenger": "127.0.0.1:1", "firewall": "127.0.0.1:2", "workers": "127.0.0.1:3"}
    d = P.deployment(config, False, at)
    assert d["protocol"] == SP and d["params"] == {"blocks": 1, "components": ["sampled-proofs"], "coins": "development"}
    assert {p["name"]: p.get("listen") for p in d["party"]} == {**at, "verifier": None}
    assert P.deployment(config, True, {})["params"]["coins"] == "simulated"
    shared = {**config, "party": [{**p, "machine": "prover-1"} if p["name"] == "challenger" else p
                                  for p in config["party"]]}
    assert P.coins_of(shared, False) == "simulated"


def test_a_verifier_whose_peers_listen_nowhere_decides_nothing(tmp_path, capsys):
    config = P.config_of(EXAMPLE)
    assert P.run_parties(config, tmp_path, ["verifier"], local=False, at={"workers": "127.0.0.1:9"}) == (None, 2)
    assert "challenger, firewall runs elsewhere and no --at says where it listens" in capsys.readouterr().err


TOOLS = os.environ.get("VERITY_ALL_PROTOCOLS_TOOLS")


def verdicts(state: Path) -> list[dict]:
    return [json.loads(x) for x in (state / "record" / "verdicts.jsonl").read_text().splitlines()]


@pytest.mark.slow
@pytest.mark.skipif(not TOOLS, reason="opt-in: VERITY_ALL_PROTOCOLS_TOOLS=tools.json (build.sh's); three blocks proved "
                    "under --zk on the challenger's coins, minutes")
def test_every_role_here_honest_tampered_and_the_certifier_absent(tmp_path):
    """The one test of the parties on the real machine: the challenger's host drawn on by the real audit, and the Lean
    product over the configuration's components."""
    honest, tampered, absent = (tmp_path / x for x in ("honest", "tampered", "absent"))
    assert P.main([str(EXAMPLE), "--state", str(honest), "--local", "--tools", TOOLS]) == 0
    assert P.main([str(EXAMPLE), "--state", str(tampered), "--local", "--tools", TOOLS, "--tamper", "0"]) == 1
    both = tmp_path / "both.toml"
    both.write_text(EXAMPLE.read_text().replace('protocols = ["sampled-proofs"]',
                                                'protocols = ["sampled-proofs", "certifier"]'))
    assert tomllib.loads(both.read_text())["protocols"] == ["sampled-proofs", "certifier"]
    assert P.main([str(both), "--state", str(absent), "--local", "--tools", TOOLS]) == 1
    (h,), (t,), (a,) = verdicts(honest), verdicts(tampered), verdicts(absent)
    assert h["verdict"] == "accept" and h["components"] == {"sampled-proofs": "accept"} and h["coins"] == "simulated"
    assert t["verdict"] == "reject" and t["components"] == {"sampled-proofs": "reject"}
    assert a["verdict"] == "reject" and a["components"] == {"sampled-proofs": "accept", "certifier": "absent"}
    assert h["product"] == t["product"] == a["product"] == "verity/product/v0"
    trusted = json.loads((honest / "record" / "w0" / "sampled-proofs" / "answer.json").read_text())
    assert trusted["protocol"] == SP
