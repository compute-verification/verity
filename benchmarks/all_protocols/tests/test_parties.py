"""The sampled-proofs roles of `verity run CONFIG` (`verity_all_protocols.parties`):

* the roles' table holds the challenger, firewall, workers and verifier, and the example configuration places exactly
  those roles, its channels between them, each ``after`` a message an earlier channel carries;
* a configuration listing a component no role knows is refused by name;
* the verdict line's product runs over exactly the listed components: a listed component whose role wrote nothing is
  ``absent``, its case one its own decoders refuse, so the product never accepts over it; the commitment binds the listed
  components' digests under the product's own id (a stand-in ``product``, which records what it was given);
* the verifier with no challenger listening, or no block served, decides nothing (2);
* opt-in (``VERITY_ALL_PROTOCOLS_TOOLS=tools.json``, `build.sh`'s; minutes): the example configuration with every role on
  this machine, honest, accepted on the challenger party's coins; tampered, rejected; and with the certifier listed but
  no certifier role, ``certifier: absent`` and the product rejects.
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

EXAMPLE = P.ROOT / "integrations" / "torch_loop" / "configs" / "sampled-proofs.toml"
SP, CERT = "verity/sampled-proofs/one-stage/v1", "verity/certifier/v0"


def test_the_roles_table_and_the_example_configuration_agree():
    assert set(P.PARTIES["sampled-proofs"]) == set(P.ORDER) == {"challenger", "firewall", "workers", "verifier"}
    config = P.config_of(EXAMPLE)
    assert config["protocols"] == ["sampled-proofs"] and "epsilon" not in json.dumps(config)
    assert sorted(p["name"] for p in config["party"]) == sorted(P.ORDER)
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
    with pytest.raises(SystemExit, match="each is one of"):
        P.config_of(f)
    f.write_text("protocols = []\n")
    with pytest.raises(SystemExit, match="each is one of"):
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


def record(tmp_path: Path) -> Path:
    rec = tmp_path / "record" / "w0"
    (rec / "sampled-proofs").mkdir(parents=True)
    (rec / "sampled-proofs" / "answer.json").write_text(json.dumps({"verdict": "accept", "commitment": "ab" * 64}))
    (rec / "sampled-proofs" / "case.json").write_text(json.dumps({"commitment": "01", "evidence": "02", "params": {"k": 8}}))
    return rec


def test_an_absent_component_is_named_and_the_product_never_accepts_over_it(tmp_path):
    exe = product_exe(tmp_path)
    got = P.product(str(exe), ["sampled-proofs", "certifier"], record(tmp_path))
    assert got == {"components": {"sampled-proofs": "accept", "certifier": "absent"}, "product": "verity/product/v0",
                   "verdict": "reject", "reason": f"{CERT} reject: params: missing"}
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["ids"] == [SP, CERT]
    ev = {x["id"]: x for x in json.loads(seen["evidence"])["components"]}
    assert ev[CERT] == {"id": CERT, **C.NO_CASE} and ev[SP]["evidence"] == "02"
    assert seen["commitment"] == C.bind("verity/product/v0", [(CERT, ""), (SP, "ab" * 64)]).hex()


def test_the_product_over_one_listed_component_is_that_components(tmp_path):
    exe = product_exe(tmp_path)
    got = P.product(str(exe), ["sampled-proofs"], record(tmp_path))
    assert got["components"] == {"sampled-proofs": "accept"} and got["verdict"] == "accept"
    assert json.loads((tmp_path / "seen.json").read_text())["ids"] == [SP]


def test_the_verifier_decides_nothing_without_a_challenger_or_a_block(tmp_path, capsys):
    (tmp_path / "tools.json").write_text(json.dumps({"product": "unused"}))
    config = P.config_of(EXAMPLE)
    assert P.verifier(config, tmp_path, True) == 2
    assert "no challenger is listening" in capsys.readouterr().err
    (tmp_path / "challenger").mkdir()
    (tmp_path / "challenger" / "address.json").write_text(json.dumps({"address": "127.0.0.1:9"}) + "\n")
    (tmp_path / "main.txt").write_text("f" * 40 + "\n")
    assert P.verifier(config, tmp_path, True) == 2
    assert "no block was served" in capsys.readouterr().err


TOOLS = os.environ.get("VERITY_ALL_PROTOCOLS_TOOLS")


def verdicts(state: Path) -> list[dict]:
    return [json.loads(x) for x in (state / "record" / "verdicts.jsonl").read_text().splitlines()]


@pytest.mark.slow
@pytest.mark.skipif(not TOOLS, reason="opt-in: VERITY_ALL_PROTOCOLS_TOOLS=tools.json (build.sh's); three blocks proved "
                    "under --zk on the challenger party's coins, minutes")
def test_every_role_here_honest_tampered_and_the_certifier_absent(tmp_path):
    """The one test of the roles as processes: the challenger party's session drawn on by the real audit, and the Lean
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
