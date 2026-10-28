"""The served window's onsite configuration (`demos/onsite_served/config.toml`) and its table (`served_table.py`):
the file is the table's configuration byte for byte and names no host; `verity run --dry-run` resolves every party,
channel and piece with no GPU, and refuses by name a piece that is missing, a channel to an undeclared party and a
binding the configuration has no place for; the served window runs once with every piece's hooks, and the onsite check
reads the record's composed verdict. The runner and the record are stand-ins written by each test."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from verity.core.deploy import config as K
from verity.core.deploy.manifest import Channel
from verity.core.wire import Refused
from verity_cli import run as R

ROOT = Path(R.ROOT)
DIR = ROOT / "demos" / "onsite_served"
CONFIG = DIR / "config.toml"
REG = R.registry((DIR, "served_table"))
T = sys.modules["served_table"]
KEY = "0" * 128
#: every binding the pieces read, on documentation addresses (RFC 5737)
BINDINGS = {
    "challenger": {"pouw": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}},
    "memory-challenger": {"pous": {"host": "auditor-1", "address": "192.0.2.1:7000", "command": "start-side",
                                   "key-sha512": KEY},
                          "wipe": {"host": "auditor-1", "address": "192.0.2.1:7001", "command": "start-wipe"}},
    "bandwidth-verifier": {"bandwidth": {"address": "192.0.2.2:7002"}},
    "network-certifier": {"certifier": {"gate": "user@gate-1"}},
}
UNBOUND = sorted(f"{p}.{c}.{k}" for p, cs in BINDINGS.items() for c, ks in cs.items() for k in ks)


def toml(tables: dict) -> str:
    return "".join(f"[{p}.{c}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in ks.items()) + "\n"
                   for p, cs in tables.items() for c, ks in cs.items())


def bindings(tmp: Path, tables: dict = BINDINGS) -> str:
    f = tmp / "bindings.toml"
    f.write_text(toml(tables))
    return str(f)


def said(out: str, what: str) -> list[dict]:
    return [json.loads(x.split(" ", 1)[1]) for x in out.splitlines() if x.startswith(what + " ")]


def dry(capsys, *args: str, reg: R.Registry = REG, config: Path = CONFIG) -> tuple[int, dict, str]:
    rc = R.run_main([str(config), "--dry-run", *args], reg)
    out = capsys.readouterr()
    return rc, {"plan": said(out.out, "PLAN"), "channel": said(out.out, "CHANNEL"), "dry": said(out.out, "DRY")}, out.err


# --- the configuration -------------------------------------------------------------------------------------------------

def test_the_configuration_is_the_tables_byte_for_byte():
    text = CONFIG.read_text()
    cfg = T.configuration(K.parse(text)["program"]["commit"])
    assert K.render(cfg, T.HEADER) == text and K.parse(text) == cfg
    assert R.load(CONFIG, REG)["params"]["verity/pous/p2/v4/run"] == {"challenges": 320}


def test_the_configuration_states_no_host_or_address():
    cfg = K.parse(CONFIG.read_text())
    assert not [(p["name"], f) for p in cfg["party"] for f in p if f in K.TRANSPORT]
    assert not [(c["from"], c["to"], f) for c in cfg["channel"] for f in c if f in K.TRANSPORT]
    assert {p["machine"] for p in cfg["party"]} == {T.PROVER_NODE, T.GATE_HOST, T.AUDITOR_HOST}
    assert all(p["machine"] == T.PROVER_NODE for p in cfg["party"] if p["principal"] == "developer")


def test_the_parties_are_the_manifests_names_and_onsite_is_the_product():
    cfg = R.load(CONFIG, REG)
    assert cfg["protocols"] == list(T.PROTOCOLS)
    assert {p["name"] for p in cfg["party"]} == {"workers", "firewall", "prover", "challenger", "network-certifier",
                                                 "memory-challenger", "bandwidth-verifier", "verifier"}
    assert [c for c in T.PROTOCOLS if REG.components[c].product] == list(T.ONSITE_COMPONENTS)


# --- the dry run -------------------------------------------------------------------------------------------------------

def test_a_dry_run_resolves_every_party_channel_and_piece(tmp_path, capsys):
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path))
    assert rc == 0, err
    cfg = R.load(CONFIG, REG)
    assert sorted(x["party"] for x in got["plan"]) == sorted(p["name"] for p in cfg["party"])
    assert len(got["channel"]) == len(cfg["channel"]) == 13
    assert got["dry"] == [{"configuration": K.digest(cfg), "parties": 8, "channels": 13, "entries": 19, "unbound": []}]
    pieces = {(x["party"], e["component"]) for x in got["plan"] for e in x["entries"]}
    assert {c for _, c in pieces} == {*T.PROTOCOLS, T.SERVED_WINDOW}
    hooks = {k: v for x in got["plan"] for e in x["entries"] for k, v in e["hooks"].items()}
    assert hooks["POUS_ARGS"].startswith("--rounds 320 --challenger-host auditor-1 --challenger-address 192.0.2.1:7000")
    assert hooks["SERVE_ARGS"] == "--remote-verifier 192.0.2.2:7002"
    assert (hooks["WIPE_CHALLENGER"], hooks["WIPE_ADDRESS"], hooks["WIPE_COMMAND"]) == (
        "auditor-1", "192.0.2.1:7001", "start-wipe")
    assert (hooks["H1_GATE"], hooks["H1_PEER"], hooks["H1_KEY_SHA512"]) == ("gate-command", "peer-1", KEY)
    assert hooks["CERTIFIER_FRONT"].endswith("--gate user@gate-1") and "PP_CHALLENGER" not in hooks


def test_a_dry_run_without_bindings_names_every_one_it_lacks(capsys):
    rc, got, _ = dry(capsys)
    assert rc == 0 and got["dry"][0]["unbound"] == UNBOUND
    assert "<memory-challenger.pous.address>" in next(e["hooks"]["POUS_ARGS"] for x in got["plan"] if x["party"] == "prover"
                                                      for e in x["entries"] if e["component"] == "pous")


def test_a_dry_run_on_a_machine_resolves_its_parties(capsys):
    rc, got, _ = dry(capsys, "--on", T.PROVER_NODE)
    assert rc == 0 and [x["party"] for x in got["plan"]] == ["workers", "firewall", "prover"]
    assert [e["component"] for e in got["plan"][0]["entries"]] == [T.SERVED_WINDOW, "sampled-proofs", "pouw"]


def test_the_live_challenger_is_bound_only_when_given(tmp_path, capsys):
    tables = {**BINDINGS, "challenger": {**BINDINGS["challenger"], "sampled-proofs": {"gate": "user@gate-1",
                                                                                      "peer": "peer-1"}}}
    rc, got, _ = dry(capsys, "--bindings", bindings(tmp_path, tables))
    hooks = {k: v for x in got["plan"] for e in x["entries"] for k, v in e["hooks"].items()}
    assert rc == 0 and (hooks["PP_CHALLENGER"], hooks["PP_PEER"]) == ("gate:user@gate-1", "peer-1")
    assert "PP_CHALLENGER_KEY" not in hooks


def test_a_binding_the_file_lacks_is_refused(tmp_path, capsys):
    tables = {**BINDINGS, "bandwidth-verifier": {"bandwidth": {}}}
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, tables))
    assert rc == 2 and "refused unbound" in err and "bandwidth-verifier.bandwidth.address" in err


@pytest.mark.parametrize("tables, why", [
    ({"oracle": {"pous": {"host": "auditor-1"}}}, "which the configuration has no party of"),
    ({"memory-challenger": {"bandwidth": {"address": "192.0.2.2:7002"}}}, "not a listed component that declares it"),
    ({"memory-challenger": {"pous": {"host": ["auditor-1"]}}}, "are strings or integers"),
])
def test_a_binding_the_configuration_has_no_place_for_is_refused(tmp_path, capsys, tables, why):
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, tables))
    assert rc == 2 and "refused binding" in err and why in err


def registry(**parties) -> R.Registry:
    return dataclasses.replace(REG, parties={**REG.parties, **parties})


def test_a_component_missing_a_party_s_piece_is_refused(capsys):
    pous = {p: e for p, e in REG.parties["pous"].items() if p != "memory-challenger"}
    rc, _, err = dry(capsys, reg=registry(pous=pous))
    assert rc == 2 and "refused no-party: pous has no step-0 memory-challenger" in err


def test_a_piece_whose_script_is_not_in_the_tree_is_refused(capsys):
    wipe = {**REG.parties["wipe"], "prover": dataclasses.replace(REG.parties["wipe"]["prover"],
                                                                  scripts=("benchmarks/pous/no_such_wipe.sh",))}
    rc, _, err = dry(capsys, reg=registry(wipe=wipe))
    assert rc == 2 and "refused no-piece" in err and "prover's wipe piece: benchmarks/pous/no_such_wipe.sh" in err


def test_a_channel_naming_an_undeclared_party_is_refused(tmp_path, capsys):
    cfg = K.parse(CONFIG.read_text())
    cfg["channel"].append({"from": "prover", "to": "oracle", "messages": ["Setup"]})
    path = tmp_path / "config.toml"
    path.write_text(K.render(cfg))
    rc, _, err = dry(capsys, config=path)
    assert rc == 2 and "refused undeclared-channel: prover -> oracle" in err


def test_a_component_declaring_a_channel_to_an_undeclared_party_is_refused():
    pous = REG.components["pous"]
    bad = dataclasses.replace(pous, channels=(*pous.channels, Channel("prover", "oracle", ("Setup",))))
    with pytest.raises(Refused) as e:
        K.check(K.parse(CONFIG.read_text()), {**REG.components, "pous": bad}, REG.workloads)
    assert e.value.name == "declaration" and "oracle" in e.value.detail


# --- the run -----------------------------------------------------------------------------------------------------------

def stand_in(tmp: Path, name: str, body: str) -> str:
    f = tmp / name
    f.write_text(body)
    return str(f)


RUNNER = """mkdir -p "$OUT"
env | grep -E '^(REPO|REV|SLOTS|LOAD|V3C_LAW|POUS_CMD|POUS_ARGS|SERVE_ARGS|WIPE_[A-Z]+|H1_[A-Z0-9_]+|CERTIFIER_[A-Z]+)=' \
  | sort > "$OUT/hooks.txt"
"""


def test_the_served_window_runs_once_with_every_piece_s_hooks(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", RUNNER))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state", str(state)],
                    REG)
    out = capsys.readouterr().out
    assert rc == 0
    assert said(out, "PARTY") == [{"party": p, "refused": None} for p in ("firewall", "prover", "workers")]
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks["POUS_ARGS"].startswith("--rounds 320 ") and hooks["SLOTS"] == "4" and hooks["V3C_LAW"] == "subset:1"
    assert {"WIPE_CMD", "SERVE_ARGS", "H1_GATE", "CERTIFIER_FRONT", "CERTIFIER_CMD"} <= set(hooks)
    assert json.loads((state / "bindings.json").read_text()) == BINDINGS
    assert "auditor-1" not in json.dumps(json.loads((state / "run.json").read_text()))


def test_the_served_window_with_a_binding_missing_never_starts(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", "touch \"$OUT.started\"\n"))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--state", str(state)], REG)
    assert rc == 2 and "refused unbound" in capsys.readouterr().err
    assert not (state / "out.started").exists()


RECORD = """import json, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == "window":
    print(json.dumps({"window": 2, "steps": [16, 32]}))
    sys.exit(0)
out = Path(a[a.index("--out") + 1])
out.mkdir(parents=True)
rows = {"integrity": "accept", "work": "accept", "space": "accept", "communication": "accept", "wipe": "accept"}
rows.update(json.loads(%r))
composed = "pending" if "bandwidth" not in rows else "accept"
(out / "verdicts.json").write_text(json.dumps({"protocols": {k: {"verdict": v} for k, v in rows.items()}}))
print("RECORD " + json.dumps({"composed": composed, "missing": [] if "bandwidth" in rows else ["bandwidth"],
                              "node": a[a.index("--node") + 1]}))
"""


@pytest.mark.parametrize("rows, verdict, rc", [({}, "undecided", 2), ({"bandwidth": "accept"}, "accept", 0)])
def test_the_onsite_check_is_the_records_composed_verdict(tmp_path, capsys, monkeypatch, rows, verdict, rc):
    monkeypatch.setattr(T, "RECORD", stand_in(tmp_path, "record.py", RECORD % json.dumps(rows)))
    state = tmp_path / "state"
    (state / "out").mkdir(parents=True)
    (state / "out" / "machine.txt").write_text("research\nprover-1\n")
    assert R.run_main([str(CONFIG), "--on", T.AUDITOR_HOST, "--state", str(state)], REG) == rc
    out = capsys.readouterr().out
    assert {x["party"]: x["refused"] for x in said(out, "PARTY")} == {
        "bandwidth-verifier": None, "memory-challenger": None, "verifier": None}
    (line,) = said(out, "VERDICT")
    assert line["verdict"] == verdict and line["block"] == 2 and line["product"] == T.ONSITE
    assert line["components"] == {**{c: "accept" for c in T.PROTOCOLS}, "bandwidth": rows.get("bandwidth", "absent")}
    assert line["coins"] == "development" and line["reason"] == (None if rows else "missing ['bandwidth']")
