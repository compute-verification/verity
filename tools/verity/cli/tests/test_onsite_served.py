"""The served window's onsite configuration (`demos/onsite_served/config.toml`) and its table (`served_table.py`):
the file is the table's configuration byte for byte and names no host; the network certifier's and the inference-only
policy's parties and channels are their manifests'; `verity run --dry-run` resolves every party, channel and piece with
no GPU, and refuses by name a piece that is missing, a channel to an undeclared party and a binding the configuration
has no place for; the runner's variables and the bindings' keys are pinned, and no gate piece names a script, a hook or
a binding; the served window runs once with every piece's hooks, and the onsite check
reads the record's composed verdict, undecided while the record has no row for a listed subprotocol. The runner and the
record are stand-ins written by each test."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from verity import config as K
from verity.protocol import Channel
from verity.channel import Refused
from verity.protocols.accounting.certifier.verifier import parties as NC
from verity.protocols.policy.inference_only.verifier import parties as IO
from verity.protocols.proofs.spot.verifier.service import Law
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
                          "wipe": {"host": "auditor-1", "address": "192.0.2.1:7001", "command": "start-wipe",
                                   "key-sha512": "1" * 128}},
    "bandwidth-verifier": {"bandwidth": {"address": "192.0.2.2:7002"}},
    "network-certifier": {"certifier": {"gate": "user@gate-1"}},
}
UNBOUND = sorted(f"{p}.{c}.{k}" for p, cs in BINDINGS.items() for c, ks in cs.items() for k in ks)
#: the served window's runner's variables under `BINDINGS`, and the bindings the pieces read, in the pieces' order
VARIABLES = {
    "CERTIFIER_CHAIN_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py chain "$@"',
    "CERTIFIER_PERIOD_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py period "$@"',
    "CLASSES": "4,32,4 8,32,4,4,16", "V3C_CLASS": "8,32,4,4,16",
    "CERTIFIER_FRONT": '.venv/bin/python benchmarks/network_traces/certifier_piece.py front "$1" "$2" "$3" '
                       "--gate user@gate-1",
    "DEVELOPMENT": "1", "H1_GATE": "gate-command", "H1_KEY_SHA512": KEY, "H1_PEER": "peer-1", "LOAD": "varied",
    "POUS_ARGS": "--rounds 320 --challenger-host auditor-1 --challenger-address 192.0.2.1:7000 --challenger-command "
                 f"start-side --trusted-sha512 {KEY}",
    "POUS_CMD": 'bash benchmarks/pous/served_piece.sh "$@"', "PRECISION": "verity_int8",
    "REPO": "verity-int8/Llama-3.2-1B-INT8", "REV": "15b0976d400d796ee589e7737d03a919526bf800",
    "SERVE_ARGS": "--remote-verifier 192.0.2.2:7002 --protocol ids --id-payload 503", "SLOTS": "4",
    "V3C_LAW": "rate:1/3481648", "WINDOW_STEPS": "16",
    "WIPE_ADDRESS": "192.0.2.1:7001", "WIPE_CHALLENGER": "auditor-1",
    "WIPE_CMD": 'bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"', "WIPE_COMMAND": "start-wipe",
    "WIPE_KEY_SHA512": "1" * 128,
}
KEYS = ["challenger.pouw.gate", "challenger.pouw.peer", "challenger.pouw.key-sha512", "memory-challenger.pous.host",
        "memory-challenger.pous.address", "memory-challenger.pous.command", "memory-challenger.pous.key-sha512",
        "bandwidth-verifier.bandwidth.address", "network-certifier.certifier.gate", "memory-challenger.wipe.host",
        "memory-challenger.wipe.address", "memory-challenger.wipe.command", "memory-challenger.wipe.key-sha512"]


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
    assert all(set(p) == set(K.PIN) for p in cfg["pin"])
    assert {p["machine"] for p in cfg["party"]} == {T.PROVER_NODE, T.GATE_HOST, T.AUDITOR_HOST}
    assert all(p["machine"] == T.PROVER_NODE for p in cfg["party"] if p["principal"] == "developer")


def test_the_memory_challenger_s_two_keys_are_pinned_and_hashed():
    cfg = R.load(CONFIG, REG)
    assert [(p["party"], p["protocol"], p["scheme"], p["key-sha512"]) for p in cfg["pin"]] == [
        ("memory-challenger", "pous", K.SCHEMES[0], T.PINS["pous"]),
        ("memory-challenger", "wipe", K.SCHEMES[0], T.PINS["wipe"])]
    bare = {k: v for k, v in K.parse(CONFIG.read_text()).items() if k != "pin"}
    assert K.digest(K.check(bare, REG.components, REG.workloads)) != K.digest(cfg)


def test_the_parties_are_the_manifests_names_and_onsite_is_the_product():
    cfg = R.load(CONFIG, REG)
    assert cfg["protocols"] == list(T.PROTOCOLS)
    assert {p["name"] for p in cfg["party"]} == {"workers", "firewall", "prover", "challenger", "network-certifier",
                                                 "memory-challenger", "bandwidth-verifier", "verifier"}
    assert all(REG.components[c].product for c in T.PROTOCOLS)
    assert K.composite(cfg["protocols"], REG.components) == T.ONSITE
    assert sorted(T.PROTOCOLS) == sorted(K.COMPOSITES[T.ONSITE])


def test_the_certifier_and_the_policy_are_declared_from_their_manifests_whole():
    certifier, policy = REG.components["certifier"], REG.components["inference-only"]
    assert (certifier.id, certifier.parties, certifier.channels) == (NC.ID, NC.MANIFEST.parties, NC.MANIFEST.channels)
    assert (policy.id, policy.parties, policy.channels) == (IO.ID, IO.MANIFEST.parties, IO.MANIFEST.channels)
    assert list(REG.parties["certifier"]) == [NC.WORKERS, NC.NETWORK_CERTIFIER, NC.VERIFIER]
    assert list(REG.parties["inference-only"]) == [IO.VERIFIER]
    cfg = R.load(CONFIG, REG)
    sent = {(c["from"], c["to"]): c["messages"] for c in cfg["channel"]}
    assert all(sent[c.sender, c.receiver] == list(c.messages) for c in NC.MANIFEST.channels)


# --- the dry run -------------------------------------------------------------------------------------------------------

def test_a_dry_run_resolves_every_party_channel_and_piece(tmp_path, capsys):
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path))
    assert rc == 0, err
    cfg = R.load(CONFIG, REG)
    assert sorted(x["party"] for x in got["plan"]) == sorted(p["name"] for p in cfg["party"])
    assert len(got["channel"]) == len(cfg["channel"]) == 13
    assert got["dry"] == [{"configuration": K.digest(cfg), "parties": 8, "channels": 13, "entries": 21, "unbound": []}]
    pieces = {(x["party"], e["component"]) for x in got["plan"] for e in x["entries"]}
    assert {c for _, c in pieces} == {*T.PROTOCOLS, T.SERVED_WINDOW}
    hooks = {k: v for x in got["plan"] for e in x["entries"] for k, v in e["hooks"].items()}
    assert hooks["POUS_ARGS"].startswith("--rounds 320 --challenger-host auditor-1 --challenger-address 192.0.2.1:7000")
    assert {e["component"]: e["hooks"]["SERVE_ARGS"] for x in got["plan"] for e in x["entries"]
            if "SERVE_ARGS" in e["hooks"]} == {"bandwidth": "--remote-verifier 192.0.2.2:7002", "certifier": T.ID_FORM}
    assert (hooks["WIPE_CHALLENGER"], hooks["WIPE_ADDRESS"], hooks["WIPE_COMMAND"], hooks["WIPE_KEY_SHA512"]) == (
        "auditor-1", "192.0.2.1:7001", "start-wipe", "1" * 128)
    assert (hooks["H1_GATE"], hooks["H1_PEER"], hooks["H1_KEY_SHA512"]) == ("gate-command", "peer-1", KEY)
    assert hooks["CERTIFIER_FRONT"].endswith("--gate user@gate-1") and "PP_CHALLENGER" not in hooks
    assert hooks["DEVELOPMENT"] == "1"


def test_a_dry_run_without_bindings_names_every_one_it_lacks(capsys):
    rc, got, _ = dry(capsys)
    assert rc == 0 and got["dry"][0]["unbound"] == UNBOUND
    assert "<memory-challenger.pous.address>" in next(e["hooks"]["POUS_ARGS"] for x in got["plan"] if x["party"] == "prover"
                                                      for e in x["entries"] if e["component"] == "pous")


def test_a_dry_run_on_a_machine_resolves_its_parties(capsys):
    rc, got, _ = dry(capsys, "--on", T.PROVER_NODE)
    assert rc == 0 and [x["party"] for x in got["plan"]] == ["workers", "firewall", "prover"]
    assert [e["component"] for e in got["plan"][0]["entries"]] == [T.SERVED_WINDOW, "sampled-proofs", "pouw",
                                                                   "certifier"]


def test_the_runners_variables_and_the_bindings_keys_are_these():
    cfg = R.load(CONFIG, REG)
    assert T.variables(cfg, BINDINGS) == VARIABLES
    assert T.unbound(cfg, {}) == KEYS and T.unbound(cfg, BINDINGS) == [] and sorted(KEYS) == UNBOUND
    unbound = T.variables(cfg, {})
    assert set(unbound) == set(VARIABLES)
    assert unbound["CERTIFIER_FRONT"].endswith("--gate '<network-certifier.certifier.gate>'")
    assert all(unbound[k] == VARIABLES[k] for k in ("CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD"))


def test_the_certifiers_script_is_its_developer_sides_and_no_gate_piece_names_one(tmp_path, capsys):
    cfg = R.load(CONFIG, REG)
    side, device = REG.parties["certifier"][NC.WORKERS], REG.parties["certifier"][NC.NETWORK_CERTIFIER]
    assert (side.runs, side.scripts, side.reaches, side.binds) == (T.SERVED_WINDOW, (T.CERTIFIER_PIECE,),
                                                                    NC.NETWORK_CERTIFIER, ("gate",))
    assert set(side.variables(cfg, BINDINGS)) == {"SERVE_ARGS", "CERTIFIER_FRONT", "CERTIFIER_CHAIN_CMD",
                                                    "CERTIFIER_PERIOD_CMD", "V3C_CLASS", "CLASSES"}
    assert device.runs == T.GATE
    gates = [e for ps in REG.parties.values() for e in ps.values() if e.runs == T.GATE]
    assert len(gates) == 6
    assert all((e.scripts, e.binds, e.optional, e.variables(cfg, BINDINGS)) == ((), (), (), {}) for e in gates)
    principal = {p["name"]: p["principal"] for p in cfg["party"]}
    assert all(principal[p] == "auditor" for ps in REG.parties.values() for p, e in ps.items() if e.runs == T.GATE)
    assert all(principal[p] == "developer" for ps in REG.parties.values() for p, e in ps.items()
               if e.runs == T.SERVED_WINDOW)
    rc, got, _ = dry(capsys, "--bindings", bindings(tmp_path))
    plan = {x["party"]: {e["component"]: e for e in x["entries"]} for x in got["plan"]}
    mine, its = plan[NC.WORKERS]["certifier"], plan[NC.NETWORK_CERTIFIER]["certifier"]
    assert rc == 0 and (mine["runs"], mine["reaches"], mine["scripts"]) == (T.SERVED_WINDOW, NC.NETWORK_CERTIFIER,
                                                                            [T.CERTIFIER_PIECE])
    assert (its["runs"], its["reaches"], its["scripts"], its["hooks"], its["unbound"]) == (T.GATE, NC.NETWORK_CERTIFIER,
                                                                                         [], {}, [])


def test_the_live_challenger_is_bound_only_when_given(tmp_path, capsys):
    tables = {**BINDINGS, "challenger": {**BINDINGS["challenger"], "sampled-proofs": {"gate": "user@gate-1",
                                                                                      "peer": "peer-1"}}}
    rc, got, _ = dry(capsys, "--bindings", bindings(tmp_path, tables))
    hooks = {k: v for x in got["plan"] for e in x["entries"] for k, v in e["hooks"].items()}
    assert rc == 0 and (hooks["PP_CHALLENGER"], hooks["PP_PEER"]) == ("gate:user@gate-1", "peer-1")
    assert "PP_CHALLENGER_KEY" not in hooks and "DEVELOPMENT" not in hooks


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


@pytest.mark.parametrize("fixed", ["subset:1", "stratified:1", "work:1", "work-rate:1/8", "bernoulli:1/8"])
def test_a_served_windows_rule_that_is_no_rate_is_refused_by_name_with_the_rate_to_state(fixed):
    """The served window's audit states the configuration's rule to its stream, which takes a rate: the file's own rule is
    one, and a configuration whose rule is a fixed count, or a work rate, for which the window has no work table, is
    refused at its check, by name, with the rate to state in its place."""
    cfg = K.parse(CONFIG.read_text())
    assert Law(cfg["service"]["law"]).text == cfg["service"]["law"]
    K.check(cfg, REG.components, REG.workloads)
    cfg["service"]["law"] = fixed
    with pytest.raises(Refused) as e:
        K.check(cfg, REG.components, REG.workloads)
    assert e.value.name == "law" and repr(fixed) in e.value.detail and "rate:K/N" in e.value.detail


# --- the run -----------------------------------------------------------------------------------------------------------

def stand_in(tmp: Path, name: str, body: str) -> str:
    f = tmp / name
    f.write_text(body)
    return str(f)


RUNNER = """mkdir -p "$OUT"
env | grep -E '^(REPO|REV|SLOTS|LOAD|V3C_LAW|V3C_CLASS|CLASSES|POUS_CMD|POUS_ARGS|SERVE_ARGS|WIPE_[A-Z0-9_]+|H1_[A-Z0-9_]+|CERTIFIER_[A-Z_]+)=' \
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
    assert hooks["POUS_ARGS"].startswith("--rounds 320 ") and hooks["SLOTS"] == "4"
    # the served window's rule is the configuration's, a rate: its audit states it to its stream, which takes no fixed count
    assert Law(hooks["V3C_LAW"]).text == K.parse(CONFIG.read_text())["service"]["law"]
    kept = set(VARIABLES) - {"PRECISION", "WINDOW_STEPS", "DEVELOPMENT"}
    assert {k: hooks.get(k) for k in kept} == {k: VARIABLES[k] for k in kept}
    assert {"WIPE_CMD", "SERVE_ARGS", "H1_GATE", "CERTIFIER_FRONT", "CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD"} \
        <= set(hooks) and "CERTIFIER_CMD" not in hooks
    assert hooks["WIPE_KEY_SHA512"] == "1" * 128
    assert json.loads((state / "bindings.json").read_text()) == BINDINGS
    assert "auditor-1" not in json.dumps(json.loads((state / "run.json").read_text()))
    assert hooks["SERVE_ARGS"] == "--remote-verifier 192.0.2.2:7002 --protocol ids --id-payload 503"


def test_serve_openais_flags_are_every_listed_pieces():
    cfg = R.load(CONFIG, REG)
    poub, ids = "--remote-verifier 192.0.2.2:7002", "--protocol ids --id-payload 503"
    assert T.ID_FORM == ids and T.variables(cfg, BINDINGS)["SERVE_ARGS"] == f"{poub} {ids}"
    listing = {without: {**cfg, "protocols": [c for c in cfg["protocols"] if c != without]}
               for without in ("bandwidth", "certifier")}
    assert T.variables(listing["bandwidth"], BINDINGS)["SERVE_ARGS"] == ids
    assert T.variables(listing["certifier"], BINDINGS)["SERVE_ARGS"] == poub
    neither = {**cfg, "protocols": [c for c in cfg["protocols"] if c not in listing]}
    assert "SERVE_ARGS" not in T.variables(neither, BINDINGS)
    # the runner's own preflight of the front reads the id form's flag as a word of SERVE_ARGS
    assert " --protocol ids " in f" {T.variables(cfg, BINDINGS)['SERVE_ARGS']} "


COMPOSITE = """mkdir -p "$OUT"
echo "${COMPOSITE-unset}|${COMPOSE_OFF_NODE-unset}" > "$OUT/composite.txt"
"""


@pytest.mark.parametrize("named, sets, got", [
    ({}, {"COMPOSITE": ""}, "|unset"),
    ({"COMPOSE_OFF_NODE": "1"}, {"COMPOSITE": ""}, "|1"),
    ({"COMPOSITE": "1"}, {}, "1|unset"),
    ({"COMPOSITE": ""}, {}, "|unset"),
    ({"COMPOSE_OFF_NODE": "0"}, {}, "unset|0"),
])
def test_the_served_window_asks_for_no_composite_unless_its_caller_names_one(tmp_path, capsys, monkeypatch, named,
                                                                             sets, got):
    """The runner composes off the node by default and refuses onsite's composite there, so the table hands it
    ``COMPOSITE`` empty; a caller that names ``COMPOSITE``, or ``COMPOSE_OFF_NODE=0`` for a development composite on
    the node, gets the runner what it named and nothing more."""
    for name in ("COMPOSITE", "COMPOSE_OFF_NODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in named.items():
        monkeypatch.setenv(name, value)
    assert T.composite(named) == sets
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", COMPOSITE))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state", str(state)],
                    REG)
    assert rc == 0, capsys.readouterr().err
    assert (state / "out" / "composite.txt").read_text().strip() == got


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


@pytest.mark.parametrize("rows, verdict, rc, reason", [
    ({}, "undecided", 2, "missing ['bandwidth']"),
    ({"bandwidth": "accept"}, "undecided", 2, "no row for ['inference-only']"),
    ({"bandwidth": "accept", "inference-only": "accept"}, "accept", 0, None),
])
def test_the_onsite_check_is_the_records_composed_verdict(tmp_path, capsys, monkeypatch, rows, verdict, rc, reason):
    monkeypatch.setattr(T, "VERDICTS_SCRIPT", stand_in(tmp_path, "record.py", RECORD % json.dumps(rows)))
    state = tmp_path / "state"
    (state / "out").mkdir(parents=True)
    (state / "out" / "machine.txt").write_text("research\nprover-1\n")
    assert R.run_main([str(CONFIG), "--on", T.AUDITOR_HOST, "--state", str(state)], REG) == rc
    out = capsys.readouterr().out
    assert {x["party"]: x["refused"] for x in said(out, "PARTY")} == {
        "bandwidth-verifier": None, "memory-challenger": None, "verifier": None}
    (line,) = said(out, "VERDICT")
    assert line["verdict"] == verdict and line["block"] == 2 and line["product"] == T.ONSITE
    assert line["components"] == {**{c: "accept" for c in T.PROTOCOLS},
                                  **{c: rows.get(c, "absent") for c in ("bandwidth", "inference-only")}}
    assert line["coins"] == "development" and line["reason"] == reason
