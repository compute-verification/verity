"""The served window's onsite configuration (`demos/onsite_full.toml`) and its table (`served_table.py`):
the file is the table's configuration byte for byte and names no host; the network attester's and the inference-only
policy's parties and channels are their manifests'; `verity run --dry-run` resolves every party, channel and piece with
no GPU, and refuses by name a piece that is missing, a channel to an undeclared party and a binding the configuration
has no place for; the runner's variables and the bindings' keys are pinned, and no gate piece names a script, a hook or
a binding; the served window runs once with every piece's hooks, and the onsite check
reads the published record's composed verdict, undecided while the record has no row for a listed subprotocol. The
runner is a stand-in and the published record is written by each test."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from verity import config as K
from verity.commitments.crypto import lms
from verity.protocol import Channel, Start
from verity.channel import Refused
from verity.protocols.resource_accounting.network.verifier import parties as NC
from verity.protocols.resource_accounting.memory.pous.attester import onsite
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.attester import challenger_party as CP
from verity.protocols.workload_classification.training_detection.inference_only.verifier import parties as IO
from verity.protocols.proofs.spot.verifier.service import SamplingDesign
from verity_cli import run as R

ROOT = Path(R.ROOT)
CONFIG = ROOT / "demos" / "onsite_full.toml"
REG = R.registry()
T = sys.modules["served_table"]
KEY = "0" * 128
PIN = {p["protocol"]: p["key-sha512"] for p in tomllib.loads(CONFIG.read_text())["pin"]}
WIPE_PIN = PIN["wipe"]
#: every binding the pieces read, on documentation addresses (RFC 5737)
BINDINGS = {
    "challenger": {"pouw": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}},
    "memory-challenger": {"pous": {"address": "192.0.2.1:7000"}, "wipe": {"address": "192.0.2.1:7001"}},
    "bandwidth-verifier": {"bandwidth": {"host": "192.0.2.2"}},
    "network-certifier": {"certifier": {"gate": "user@gate-1"}},
}
#: `BINDINGS` with the address PoUB's runtime parties read (`runtime`)
RUNTIME_BINDINGS = {**BINDINGS, "bandwidth-verifier": {"bandwidth": {"address": "192.0.2.2:7002"}}}
#: the memory attesters', which, unbound, are the run's on the workload's machine (`Piece.development`)
ATTESTERS = ("memory-challenger", "bandwidth-verifier")
UNBOUND = sorted(f"{p}.{c}.{k}" for p, cs in BINDINGS.items() if p not in ATTESTERS for c, ks in cs.items() for k in ks)
#: the served window's runner's variables under `BINDINGS`, and the bindings the pieces read, in the pieces' order
VARIABLES = {
    "CERTIFIER_CHAIN_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py chain "$@"',
    "CERTIFIER_PERIOD_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py period "$@"',
    "CLASSES": "4,32,4 8,32,4,4,16", "V3C_CLASS": "8,32,4,4,16",
    "CERTIFIER_FRONT": '.venv/bin/python benchmarks/network_traces/certifier_piece.py front "$1" "$2" "$3" '
                       "--gate user@gate-1",
    "DEVELOPMENT": "1", "H1_GATE": "gate-command", "H1_KEY_SHA512": KEY, "H1_PEER": "peer-1", "LOAD": "varied",
    "PRECISION": "verity_int8", "REPO": "verity-int8/Llama-3.2-1B-INT8",
    "REV": "15b0976d400d796ee589e7737d03a919526bf800", "SERVE_ARGS": "--protocol ids --id-payload 503", "SLOTS": "4",
    "V3C_LAW": "rate:1/3481648", "WINDOW_STEPS": "16", "WIPE_ADDRESS": "192.0.2.1:7001",
    "WIPE_CMD": f'WIPE_KEY_SHA512={WIPE_PIN} bash benchmarks/pous/streamed_wipe/served_wipe.sh "$1" "$2" "$3"',
    "POUS_CMD": 'bash benchmarks/pous/served_piece.sh "$@"', "ONSITE_PIECES": "benchmarks/pous/onsite_pieces.sh",
    "POUS_ARGS": f"--trusted-sha512 {PIN['pous']} --rounds 320 --challenger-address 192.0.2.1:7000 "
                 f"--bw-trusted-sha512 {PIN['bandwidth']} --bw-host 192.0.2.2",
    "BANDWIDTH_CMD": '"$PY" benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"',
}
KEYS = ["challenger.pouw.gate", "challenger.pouw.peer", "challenger.pouw.key-sha512", "network-certifier.certifier.gate"]
#: each runtime party: its protocol (None while a half isn't in the tree), its role there, its machine and its clock
RUNTIME = {
    ("memory-challenger", "pous"): ("verity/pous/memory-challenger", "memory-challenger", "auditor-host",
                                    "CLOCK_MONOTONIC ns"),
    ("prover", "pous"): ("verity/pous/memory-challenger", "prover", "prover-node", None),
    ("bandwidth-verifier", "bandwidth"): ("verity/remote-bandwidth/p2/v0", "verifier", "auditor-host",
                                          "CLOCK_MONOTONIC ns"),
    ("prover", "bandwidth"): ("verity/remote-bandwidth/p2/v0", "prover", "prover-node", None),
}
#: PoUS's and PoUB's parties as runtime pieces of their sessions (`T.POUS`, `T.POUB`), for the runtime path of `live`
RUNTIME_PARTIES = {
    (c, party): T.Piece(c, party, T.RUNTIME, f"{party}'s {c} run", reaches=reaches, binds=("address",), session=s)
    for c, s, listener in (("pous", T.POUS, T.MEMORY_ATTESTER), ("bandwidth", T.POUB, T.BANDWIDTH_VERIFIER))
    for party, reaches in ((T.PROVER, listener), (listener, ""))}


@pytest.fixture
def runtime(monkeypatch):
    """The table with PoUS's and PoUB's parties as runtime pieces, the path `live` keeps for a protocol the runtime
    starts, which no listed piece takes now."""
    for (c, party), piece in RUNTIME_PARTIES.items():
        monkeypatch.setitem(T.PARTIES, c, {**T.PARTIES[c], party: piece})


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


@pytest.mark.parametrize("demo", sorted(set(T.DEMOS) - {"onsite_full"}))
def test_each_smaller_demo_is_the_tables_byte_for_byte_and_its_own_composite(demo):
    """Each other demo file is the table's configuration of its subprotocols, at onsite_full's commit, whose composite
    (or onsite's smaller set) is its own id; only the subprotocols it lists have parameters and pins."""
    path = ROOT / "demos" / f"{demo}.toml"
    cfg = T.configuration(K.parse(CONFIG.read_text())["program"]["commit"], demo)
    assert K.render(cfg, T.HEADERS[demo]) == path.read_text()
    assert K.composite(cfg["protocols"], REG.components) in {**K.COMPOSITES, **K.UNACCEPTED}
    assert K.composite(cfg["protocols"], REG.components) != T.ONSITE
    assert {p["protocol"] for p in cfg["pin"]} == {"pous", "wipe"} & set(T.DEMOS[demo])


def test_the_configuration_states_no_host_or_address():
    cfg = K.parse(CONFIG.read_text())
    assert not [(p["name"], f) for p in cfg["party"] for f in p if f in K.TRANSPORT]
    assert not [(c["from"], c["to"], f) for c in cfg["channel"] for f in c if f in K.TRANSPORT]
    assert all(set(p) == set(K.PIN) for p in cfg["pin"])
    assert {p["machine"] for p in cfg["party"]} == {T.PROVER_NODE, T.GATE_HOST, T.AUDITOR_HOST}
    assert all(p["machine"] == T.PROVER_NODE for p in cfg["party"] if p["principal"] == "developer")


def test_the_memory_attester_s_two_keys_are_pinned_and_hashed():
    cfg = R.load(CONFIG, REG)
    assert [(p["party"], p["protocol"], p["scheme"]) for p in cfg["pin"]] == [
        (T.BANDWIDTH_VERIFIER, T.P2.COMPONENT, K.SCHEMES[0]), ("memory-challenger", "pous", K.SCHEMES[0]),
        ("memory-challenger", "wipe", K.SCHEMES[0])]
    stated = {p["protocol"]: p["key-sha512"] for p in tomllib.loads(CONFIG.read_text())["pin"]}
    assert {p["protocol"]: p["key-sha512"] for p in cfg["pin"]} == stated
    assert all(re.fullmatch("[0-9a-f]{128}", k) for k in stated.values()) and stated["pous"] != stated["wipe"]
    assert stated["bandwidth"] == stated["pous"]   # the bandwidth close is signed with the PoUS key
    assert T.configuration(cfg["program"]["commit"])["pin"] == cfg["pin"]   # the table states no key of its own
    bare = {k: v for k, v in K.parse(CONFIG.read_text()).items() if k != "pin"}
    assert K.digest(K.check(bare, REG.components, REG.workloads)) != K.digest(cfg)


def test_a_memory_attesters_placement_reads_the_configurations_pins_under_its_hash(monkeypatch):
    """The table's placement of a memory attester's session: `trusted-device` holds only for the key the onsite
    configuration pins for that protocol, under its committed hash; another protocol's pin, no key or a configuration
    nobody committed reads rehearsal. No other party gets a placement."""
    cfg = R.load(CONFIG, REG)
    piece = {c: p for c in ("pous", "wipe") for p in [T.PARTIES[c]["memory-challenger"]]}
    pin = {p["protocol"]: p["key-sha512"] for p in cfg["pin"]}
    key = {c: "ab" * 32 + c.encode().hex() for c in pin}
    monkeypatch.setitem(cfg, "pin", [p | {"key-sha512": hashlib.sha512(bytes.fromhex(key[p["protocol"]])).hexdigest()}
                                     for p in cfg["pin"]])
    monkeypatch.setattr(mc, "committed", lambda: frozenset({K.digest(cfg)}))
    for c in ("pous", "wipe"):
        where = T.placement(cfg, piece[c], {"public_key": key[c]})
        assert mc.holds(where) and K.digest(cfg) in where[mc.PREMISE] and f"{c} sessions" in where[mc.PREMISE]
        other = "wipe" if c == "pous" else "pous"
        assert not mc.holds(T.placement(cfg, piece[c], {"public_key": key[other]}))
        assert not mc.holds(T.placement(cfg, piece[c], None))
    monkeypatch.setattr(mc, "committed", lambda: frozenset())
    assert T.placement(cfg, piece["wipe"], {"public_key": key["wipe"]})[mc.PREMISE].startswith(mc.UNCOMMITTED)
    assert T.placement(cfg, T.PARTIES["pous"]["prover"], {"public_key": key["pous"]}) is None
    assert pin.keys() == {"pous", "wipe", "bandwidth"}


def test_the_registration_names_the_runs_own_configuration_and_is_not_written_yet(tmp_path):
    """The run's registration reads the configuration's hash from ``run.json``: no ``run.json``, or one under another
    hash, is refused; under this configuration's own hash it raises through the registration log's writer of the
    configuration item, which isn't written yet."""
    cfg = R.load(CONFIG, REG)
    with pytest.raises(Refused) as no_run:
        T.registration(cfg, tmp_path)
    assert no_run.value.name == "registration"
    (tmp_path / "run.json").write_text(json.dumps({"configuration": "00" * 64}))
    with pytest.raises(Refused):
        T.registration(cfg, tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"configuration": K.digest(cfg)}))
    with pytest.raises(NotImplementedError, match="proofs/architecture-on-branch"):
        T.registration(cfg, tmp_path)


def test_the_registration_writes_the_logs_configuration_item_of_the_runs_hash(tmp_path, monkeypatch):
    """The item is the registration log's, written by its writer from the configuration's hash; past it, the run's log
    that holds it is ours and isn't written yet."""
    cfg, asked = R.load(CONFIG, REG), []
    monkeypatch.setattr(T.RL, "configuration", lambda digest: asked.append(digest) or {"kind": "configuration",
                                                                                         "digest": digest})
    (tmp_path / "run.json").write_text(json.dumps({"configuration": K.digest(cfg)}))
    with pytest.raises(NotImplementedError, match=T.REGISTRATION_GOAL):
        T.registration(cfg, tmp_path)
    assert asked == [K.digest(cfg)]


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
    assert all(hooks[k] == VARIABLES[k] for k in ("WIPE_CMD", "POUS_CMD", "BANDWIDTH_CMD"))
    assert " ".join(e["hooks"]["POUS_ARGS"] for x in got["plan"] for e in x["entries"]
                    if "POUS_ARGS" in e["hooks"]) == VARIABLES["POUS_ARGS"]
    assert {e["component"]: e["hooks"]["SERVE_ARGS"] for x in got["plan"] for e in x["entries"]
            if "SERVE_ARGS" in e["hooks"]} == {"certifier": T.ID_FORM}
    assert (hooks["H1_GATE"], hooks["H1_PEER"], hooks["H1_KEY_SHA512"]) == ("gate-command", "peer-1", KEY)
    assert hooks["CERTIFIER_FRONT"].endswith("--gate user@gate-1") and "PP_CHALLENGER" not in hooks
    assert hooks["DEVELOPMENT"] == "1"


def test_a_dry_run_prints_each_runtime_party_with_its_protocol_role_machine_clock_and_channels(tmp_path, capsys,
                                                                                              runtime):
    """PoUS's rounds' and PoUB's parties are their protocols' (`Session`), each on the machine the
    configuration places it on, the clock its header names on the one that reads a clock, the auditor's side listening
    at the address the prover's side dials; PoUB's is ``absent`` while its halves aren't in the tree."""
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path, RUNTIME_BINDINGS))
    assert rc == 0, err
    runtime = {(x["party"], e["component"]): e for x in got["plan"] for e in x["entries"] if e["runs"] == T.RUNTIME}
    assert set(runtime) == set(RUNTIME)
    for (party, component), (protocol, role, machine, clock) in RUNTIME.items():
        e, s = runtime[party, component], T.PARTIES[component][party].session
        assert (e["role"], e["machine"], e["clock"], e["raises"]) == (role, machine, clock, T.NOT_YET)
        address = RUNTIME_BINDINGS[s.listener][component]["address"]
        assert e["listens" if party == s.listener else "dials"] == address
        if e["protocol"] is None:
            assert e["absent"] and set(e["absent"]) <= {h.split(":")[0] for h in s.halves}
            continue
        manifest = s.protocol().manifest
        assert e["protocol"] == protocol and e["needs"] == list(manifest.role(role).needs)
        assert ("clock" in e["needs"]) == (clock is not None)
        named = {r: c for c, r in s.roles.items()}
        assert e["channels"] == [{"from": named[c.sender], "to": named[c.receiver], "messages": list(c.messages),
                                  "carries_coins": c.carries_coins} for c in manifest.channels]


def test_the_runtime_sessions_are_one_manifest_with_every_configured_role_and_a_clock_where_one_is_read():
    for s in (T.POUS, T.POUB):
        if s.absent():
            continue
        m = s.protocol().manifest
        assert sorted(s.roles.values()) == sorted(r.name for r in m.parties)
        assert sorted(s.clocks) == sorted(r.name for r in m.parties if "clock" in r.needs)
        assert sorted(s.protocol().parties) == sorted(s.roles.values())


def test_a_dry_run_without_bindings_names_every_one_it_lacks(capsys):
    rc, got, _ = dry(capsys)
    assert rc == 0 and got["dry"][0]["unbound"] == UNBOUND
    assert next(e["hooks"]["POUS_ARGS"] for x in got["plan"] if x["party"] == "prover" for e in x["entries"]
                if e["component"] == "pous") == "--run-keys --rounds 320"


def test_a_dry_run_on_a_machine_resolves_its_parties(capsys):
    rc, got, _ = dry(capsys, "--on", T.PROVER_NODE)
    assert rc == 0 and [x["party"] for x in got["plan"]] == ["workers", "firewall", "prover"]
    assert [e["component"] for e in got["plan"][0]["entries"]] == [T.SERVED_WINDOW, "sampled-proofs", "pouw",
                                                                   "certifier"]


def test_the_runners_variables_and_the_bindings_keys_are_these():
    cfg = R.load(CONFIG, REG)
    deployed = {"VERITY_CONFIGURATION": K.digest(cfg), "VERITY_PINS": json.dumps(cfg["pin"], sort_keys=True)}
    assert T.variables(cfg, BINDINGS) == VARIABLES | deployed
    assert T.unbound(cfg, {}) == KEYS and T.unbound(cfg, BINDINGS) == [] and sorted(KEYS) == UNBOUND
    unbound = T.variables(cfg, {})
    assert set(unbound) == set(VARIABLES) - {"WIPE_ADDRESS"} | set(deployed)
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


def test_a_runtime_party_raises_before_anything_starts_while_its_machine_hands_it_nothing(tmp_path, capsys,
                                                                                         monkeypatch, runtime):
    """The prover node's runtime parties need what their protocol's start reads (the served store, the card, setup's
    request), which doesn't reach the table yet: the run raises by the row's name before the served window starts."""
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", "touch \"$OUT.started\"\n"))
    state = tmp_path / T.PROVER_NODE
    with pytest.raises(NotImplementedError) as e:
        R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path, RUNTIME_BINDINGS), "--state",
                    str(state)], REG)
    assert e.value.args == (T.NOT_YET,) and not (state / "out.started").exists()


def test_the_memory_attesters_key_is_refused_unless_this_configuration_pins_it_before_anything_else(tmp_path,
                                                                                                    capsys, runtime):
    """On the auditor's device each memory attester's start checks its key against `demos/onsite_full.toml`'s pin
    first: a state with no key is refused ``key``, and one whose key the configuration doesn't pin ``pin``, before any
    request is read or any party or the served window starts."""
    state = tmp_path / T.AUDITOR_HOST
    run = [str(CONFIG), "--on", T.AUDITOR_HOST, "--bindings", bindings(tmp_path, RUNTIME_BINDINGS), "--state", str(state)]
    assert R.run_main(run, REG) == 2 and "refused key" in capsys.readouterr().err
    other = b"a key the auditor never pinned"
    mine = state / "runs" / "pous" / T.MEMORY_ATTESTER
    mine.mkdir(parents=True)
    (mine / "seed").write_bytes(bytes(lms.DEPLOYED.N))
    lms.LeafState.create(mine / onsite.KEY_STATE, other, lms.DEPLOYED.H)
    assert R.run_main(run, REG) == 2
    err = capsys.readouterr().err
    assert "refused pin" in err and hashlib.sha512(other).hexdigest() in err
    assert not (state / "runs" / "pous" / T.MEMORY_ATTESTER / "tree.bin").exists()


def test_the_rounds_and_poubs_parties_each_have_their_protocols_start():
    for s in (T.POUS, T.POUB):
        assert sorted(s.protocol().starts) == sorted(s.roles.values())


def test_a_run_raises_at_its_request_until_the_two_sides_compare_it(tmp_path, monkeypatch, runtime):
    """With its start and its party key given, a runtime party still raises before it runs: the two sides' requests
    are compared nowhere yet (`request_bound`)."""
    monkeypatch.setitem(CP.PROTOCOL.starts, T.CS.CHALLENGER, lambda configuration, bound, state: Start({"k": 1}))
    monkeypatch.setattr(T, "held", lambda piece, role: T.Held(None, None))
    piece, cfg = T.PARTIES["pous"][T.MEMORY_ATTESTER], R.load(CONFIG, REG)
    with pytest.raises(NotImplementedError) as e:
        T.prepare(cfg, tmp_path, BINDINGS, piece)
    assert e.value.args == (T.NOT_YET,)
    monkeypatch.setattr(T, "request_bound", lambda protocol, params: None)
    assert callable(T.prepare(cfg, tmp_path, BINDINGS, piece))


def test_the_onsite_lines_trusted_device_is_each_memory_attesters_placement(tmp_path, monkeypatch, runtime):
    """The verifier's line carries each memory attester's `trusted-device` premise as `placement` reads it from the
    challenger's outcome here: under a committed configuration, ``holds`` for a session signed under the pinned key, and
    rehearsal for one that reports no key."""
    key = b"a stand-in memory attester key"
    cfg = R.load(CONFIG, REG)
    cfg = {**cfg, "pin": [{**p, "key-sha512": hashlib.sha512(key).hexdigest()} if p["protocol"] == "pous" else p
                          for p in cfg["pin"]]}
    monkeypatch.setattr(mc, "committed", lambda: frozenset({K.digest(cfg)}))
    outcomes = {"pous": {"outcome": {"public_key": key.hex()}, "refused": None}}
    monkeypatch.setattr(T, "prepare", lambda cfg, state, bindings, x: lambda: outcomes[x.component])
    monkeypatch.setattr(T, "registration", lambda cfg, state: None)
    monkeypatch.setattr(T, "onsite_check", lambda cfg, state: {"verdict": "undecided"})
    (tmp_path / T.BINDINGS).write_text(json.dumps(BINDINGS))
    line, = T.live(cfg, tmp_path, [T.MEMORY_ATTESTER, T.VERIFIER]).outcomes[T.VERIFIER]["outcome"]["verdicts"]
    assert set(line[mc.PREMISE]) == {"pous"} and line["verdict"] == "undecided"
    assert line[mc.PREMISE]["pous"].startswith("holds")


def test_the_bandwidth_verifiers_start_reads_the_memory_attesters_run_beside_it(tmp_path, monkeypatch, runtime):
    """PoUB's verifier pins the memory attester's PoUS key, which is in that party's run on the same host: its start
    gets that run's state directory (``challenger``) beside its bindings, and its own state directory."""
    got = {}

    def start(configuration, bound, state):
        got.update(bound=bound, state=state)
        raise NotImplementedError(T.NOT_YET)

    monkeypatch.setitem(T.POUB.starts, T.P2.VERIFIER, start)
    with pytest.raises(NotImplementedError):
        T.prepare(R.load(CONFIG, REG), tmp_path, RUNTIME_BINDINGS, T.PARTIES["bandwidth"][T.BANDWIDTH_VERIFIER])
    assert got["bound"] == {**RUNTIME_BINDINGS["bandwidth-verifier"]["bandwidth"],
                            "challenger": str(tmp_path / "runs" / "pous" / T.MEMORY_ATTESTER)}
    assert got["state"] == tmp_path / "runs" / "bandwidth" / T.BANDWIDTH_VERIFIER


def test_the_bandwidth_verifiers_start_is_not_yet_a_path_that_runs(tmp_path):
    start = T.POUB.protocol().starts[T.P2.VERIFIER]
    with pytest.raises(NotImplementedError, match=T.P2.GOAL):
        start({}, {"challenger": str(tmp_path)}, None)


def test_the_runtime_components_channels_are_their_protocols_manifests():
    """The configuration's channels of PoUS's rounds and PoUB are their protocols' manifests', each role
    under the configuration's name for its party, with the channel to onsite's verifier beside them."""
    for component, s in (("pous", T.POUS), ("bandwidth", T.POUB)):
        named = {r: c for c, r in s.roles.items()}
        mine = {(c.sender, c.receiver, c.messages, c.carries_coins) for c in T.MANIFESTS[component].channels
                if T.VERIFIER not in (c.sender, c.receiver)}
        assert mine == {(named[c.sender], named[c.receiver], c.messages, c.carries_coins)
                        for c in s.protocol().manifest.channels}


def test_a_binding_the_file_lacks_is_refused(tmp_path, capsys):
    tables = {**BINDINGS, "bandwidth-verifier": {"bandwidth": {}}}
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, tables))
    assert rc == 2 and "refused unbound" in err and "bandwidth-verifier.bandwidth.host" in err


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
    assert SamplingDesign(cfg["service"]["law"]).text == cfg["service"]["law"]
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


#: coreutils' env by path: agent VMs put a redacting `env` ahead of it on PATH (`tools/agent-guard/env_guard.py`)
RUNNER = """mkdir -p "$OUT"
/usr/bin/env | grep -E '^(REPO|REV|SLOTS|LOAD|V3C_LAW|V3C_CLASS|CLASSES|POUS_CMD|POUS_ARGS|ONSITE_PIECES|BANDWIDTH_CMD|SERVE_ARGS|WIPE_[A-Z0-9_]+|H1_[A-Z0-9_]+|CERTIFIER_[A-Z_]+)=' \
  | sort > "$OUT/hooks.txt"
"""


def test_the_served_window_runs_once_with_every_piece_s_hooks(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", RUNNER))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, *SERVED_PARTIES, "--bindings", bindings(tmp_path), "--state",
                     str(state)], REG)
    out = capsys.readouterr().out
    assert rc == 0
    assert said(out, "PARTY") == [{"party": p, "refused": None} for p in ("firewall", "workers")]
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks["SLOTS"] == "4" and hooks["POUS_ARGS"] == VARIABLES["POUS_ARGS"]
    # the served window's rule is the configuration's, a rate: its audit states it to its stream, which takes no fixed count
    assert SamplingDesign(hooks["V3C_LAW"]).text == K.parse(CONFIG.read_text())["service"]["law"]
    kept = set(VARIABLES) - {"PRECISION", "WINDOW_STEPS", "DEVELOPMENT"}
    assert {k: hooks.get(k) for k in kept} == {k: VARIABLES[k] for k in kept}
    assert {"SERVE_ARGS", "H1_GATE", "CERTIFIER_FRONT", "CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD"} <= set(hooks) \
        and "CERTIFIER_CMD" not in hooks
    assert json.loads((state / "bindings.json").read_text()) == BINDINGS
    assert "192.0.2.1" not in json.dumps(json.loads((state / "run.json").read_text()))
    assert hooks["SERVE_ARGS"] == "--protocol ids --id-payload 503"


#: two of the prover node's parties, whose pieces are the served window's
SERVED_PARTIES = ("--party", "workers", "--party", "firewall")


def test_serve_openais_flags_are_every_listed_pieces():
    cfg = R.load(CONFIG, REG)
    assert T.variables(cfg, BINDINGS)["SERVE_ARGS"] == T.ID_FORM == "--protocol ids --id-payload 503"
    without = {**cfg, "protocols": [c for c in cfg["protocols"] if c != "certifier"]}
    assert "SERVE_ARGS" not in T.variables(without, BINDINGS)
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
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, *SERVED_PARTIES, "--bindings", bindings(tmp_path), "--state",
                     str(state)], REG)
    assert rc == 0, capsys.readouterr().err
    assert (state / "out" / "composite.txt").read_text().strip() == got


def test_the_served_window_with_a_binding_missing_never_starts(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", "touch \"$OUT.started\"\n"))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--state", str(state)], REG)
    assert rc == 2 and "refused unbound" in capsys.readouterr().err
    assert not (state / "out.started").exists()


#: the execution phase's seal digest the served window publishes
SEAL = "5" * 128


def published(state: Path, rows: dict, composed: str = "accept", reason: str = "") -> None:
    """All the served window publishes that the onsite check reads: the execution phase by its seal's digest and the
    record, every row but ``rows``' accepting."""
    every = {"integrity": "accept", "work": "accept", "space": "accept", "communication": "accept", "wipe": "accept",
             **rows}
    for d in ("serve", "record"):
        (state / "out" / d).mkdir(parents=True, exist_ok=True)
    (state / "out" / "serve" / T.PHASE).write_text(json.dumps({"seal_sha512": SEAL}))
    (state / "out" / "record" / "verdicts.json").write_text(json.dumps(
        {"protocols": {k: {"verdict": v} for k, v in every.items()}, "composed": {"verdict": composed, "reason": reason}}))


@pytest.mark.parametrize("rows, composed, verdict, rc, reason", [
    ({"bandwidth": "missing"}, "pending", "undecided", 2, "missing ['bandwidth']"),
    ({"bandwidth": "accept"}, "accept", "undecided", 2, "no row for ['inference-only']"),
    ({"bandwidth": "accept", "inference-only": "accept"}, "accept", "accept", 0, None),
    ({"bandwidth": "accept", "inference-only": "accept", "communication": "error"}, "reject", "reject", 1,
     "the network attester: error"),
])
def test_the_onsite_check_is_the_published_records_composed_verdict(tmp_path, capsys, rows, composed, verdict, rc,
                                                                    reason):
    """The served window publishes its window's number, steps and requests to no one, so the check reads only the
    published record and the execution phase's seal digest: a check that picked the window again had nothing to pick."""
    state = tmp_path / "state"
    published(state, rows, composed, "the network attester: error")
    assert sorted(p.relative_to(state).as_posix() for p in state.rglob("*") if p.is_file()) == [
        "out/record/verdicts.json", f"out/serve/{T.PHASE}"]
    assert R.run_main([str(CONFIG), "--on", T.AUDITOR_HOST, "--party", "verifier", "--state", str(state)], REG) == rc
    out = capsys.readouterr().out
    assert {x["party"]: x["refused"] for x in said(out, "PARTY")} == {"verifier": None}
    (line,) = said(out, "VERDICT")
    assert line["verdict"] == verdict and line["block"] == SEAL and line["product"] == T.ONSITE
    assert line["components"] == {**{c: "accept" for c in T.PROTOCOLS},
                                  **{T.RECORD_NAMES[k]: rows.get(k, "absent") for k in ("bandwidth", "inference-only")},
                                  "certifier": rows.get("communication", "accept")}
    assert line["coins"] == "simulated" and line["reason"] == reason and "combined_by" not in line
    assert line["coins_on"] == {c: T.PROVER_NODE for c in ("pous", "bandwidth", "wipe")}


@pytest.mark.parametrize("left", [T.PHASE, "verdicts.json"])
def test_the_onsite_check_without_the_published_phase_or_record_is_refused_by_name(tmp_path, capsys, left):
    state = tmp_path / "state"
    published(state, {})
    next(state.rglob(left)).unlink()
    assert R.run_main([str(CONFIG), "--on", T.AUDITOR_HOST, "--party", "verifier", "--state", str(state)], REG) == 2
    refused = {x["party"]: x["refused"] for x in said(capsys.readouterr().out, "PARTY")}["verifier"]
    assert refused == ("window" if left == T.PHASE else "record")


NETWORK = ROOT / "demos" / "onsite_network.toml"
NETWORK_PIN = next(p["key-sha512"] for p in tomllib.loads(NETWORK.read_text())["pin"] if p["protocol"] == "wipe")
NETWORK_BINDINGS = {"memory-challenger": {"wipe": {"address": "192.0.2.1:7001", "host": "auditor-1",
                                                   "command": "ssh auditor-1"}},
                    "network-certifier": {"certifier": {"gate": "user@gate-1"}}}


def test_the_wipe_runs_in_the_workload_under_the_configurations_pin_and_no_party_of_it_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", RUNNER))
    state = tmp_path / "state"
    state.mkdir()
    (state / T.BINDINGS).write_text(json.dumps(NETWORK_BINDINGS))
    got = T.live(R.load(NETWORK, REG), state, [T.WORKERS, T.PROVER]).outcomes
    assert got[T.PROVER] == got[T.WORKERS] == {"outcome": {"served-window": 0}, "refused": None}
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks["WIPE_CMD"] == f'WIPE_KEY_SHA512={NETWORK_PIN} bash {T.WIPE_SCRIPT} "$1" "$2" "$3"'
    assert (hooks["WIPE_ADDRESS"], hooks["WIPE_CHALLENGER"], hooks["WIPE_COMMAND"]) == (
        "192.0.2.1:7001", "auditor-1", "ssh auditor-1")


@pytest.mark.parametrize("pinned", [True, False])
def test_served_wipe_gets_the_configurations_pin_and_no_other(tmp_path, pinned):
    cfg = R.load(NETWORK, REG)
    cmd = T.PARTIES["wipe"][T.PROVER].variables(cfg if pinned else {**cfg, "pin": []}, NETWORK_BINDINGS)["WIPE_CMD"]
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.defpath]), "HOME": str(tmp_path), "WIPE_KEY_SHA512": "f" * 128,
           "VERITY_PINS": json.dumps(cfg["pin"]), "WIPE_COMMAND": "true", "WIPE_PROTOCOL_EXE": "true",
           "GPU_LEASE_UUID": "no-such-card"}
    p = subprocess.run(["bash", "-c", cmd, "_", str(tmp_path / "serve"), "all", str(tmp_path / "wipe")], cwd=ROOT,
                       env=env, capture_output=True, text=True, check=False)
    refused = (tmp_path / "wipe" / "refused.txt").read_text()
    assert p.returncode == 3
    assert ("nvidia-smi can't read the card no-such-card" in refused) == pinned
    assert ("no challenger key is pinned" in refused) != pinned


#: the bindings both ways of starting PoUB read, so the same file starts its runtime parties and its pieces
BOTH = {**BINDINGS, "bandwidth-verifier": {"bandwidth": {**BINDINGS["bandwidth-verifier"]["bandwidth"],
                                                         **RUNTIME_BINDINGS["bandwidth-verifier"]["bandwidth"]}}}


def test_pouss_and_poubs_parties_start_from_the_table_and_none_raises(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", RUNNER))
    for machine, party in ((T.PROVER_NODE, T.PROVER), (T.AUDITOR_HOST, T.MEMORY_ATTESTER),
                           (T.AUDITOR_HOST, T.BANDWIDTH_VERIFIER)):
        state = tmp_path / party
        assert R.run_main([str(CONFIG), "--on", machine, "--party", party, *(
            ("--party", T.WORKERS) if party == T.PROVER else ()), "--bindings", bindings(tmp_path, BOTH), "--state",
            str(state)], REG) == 0, capsys.readouterr().err
        assert {"party": party, "refused": None} in said(capsys.readouterr().out, "PARTY")
    hooks = dict(x.split("=", 1) for x in (tmp_path / T.PROVER / "out" / "hooks.txt").read_text().splitlines())
    assert {k: hooks[k] for k in ("POUS_CMD", "POUS_ARGS", "BANDWIDTH_CMD", "ONSITE_PIECES")} == {
        k: VARIABLES[k] for k in ("POUS_CMD", "POUS_ARGS", "BANDWIDTH_CMD", "ONSITE_PIECES")}


#: python's stand-in for served_piece.sh: each call's arguments, one per line, then a blank line
ARGV = """#!/usr/bin/env bash
printf '%s\n' "$@" >> "$(dirname "$0")/argv.txt"; echo >> "$(dirname "$0")/argv.txt"
"""


@pytest.mark.parametrize("pins", ["configuration's", "others", "none"])
def test_pouss_and_poubs_command_line_carries_the_configurations_pins_and_no_other_key(tmp_path, pins):
    """served_piece.sh run as the workload runs it, under the pieces' variables, with a stand-in python: the PoUS
    session's key and the bandwidth close's are the configuration's pins, whatever the environment or a binding says,
    and a bound command keeps its spaces."""
    cfg = R.load(CONFIG, REG)
    other = {"pous": "a" * 128, "bandwidth": "b" * 128, "wipe": "c" * 128}
    cfg = {**cfg, "pin": {"configuration's": cfg["pin"], "none": [],
                          "others": [{**p, "key-sha512": other[p["protocol"]]} for p in cfg["pin"]]}[pins]}
    tables = {**BINDINGS, "memory-challenger": {**BINDINGS["memory-challenger"], "pous": {
        "address": "192.0.2.1:7000", "command": "ssh -i key auditor-1 challenger_party", "key-sha512": "d" * 128}}}
    hooks = {k: v for x in ("pous", "bandwidth") for k, v in T.PARTIES[x][T.PROVER].variables(cfg, tables).items()
             if k != "POUS_ARGS"}
    hooks["POUS_ARGS"] = T.variables(cfg, tables)["POUS_ARGS"]
    py = Path(stand_in(tmp_path, "python", ARGV))
    py.chmod(0o755)
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.defpath]), "HOME": str(tmp_path), "PY": str(py),
           "P2_LIB": str(tmp_path), "POUS_EXE": "true", "BW_CASE_EXE": "true", "GPU_LEASE_UUID": "no-such-card",
           "VERITY_PINS": json.dumps(cfg["pin"]), "POUS_ARGS": hooks["POUS_ARGS"]}
    p = subprocess.run(["bash", "-c", hooks["POUS_CMD"], "_", str(tmp_path / "serve"), "3", str(tmp_path / "pous")],
                       cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    calls = [c.split("\n") for c in (tmp_path / "argv.txt").read_text().split("\n\n") if "--sessions" in c]
    assert p.returncode == 0, p.stderr
    argv, = calls
    flag = {f: argv[argv.index(f) + 1] if f in argv else None
            for f in ("--trusted-sha512", "--bw-trusted-sha512", "--challenger-command", "--bw-host")}
    pinned = {p["protocol"]: p["key-sha512"] for p in cfg["pin"]}
    assert (flag["--trusted-sha512"], flag["--bw-trusted-sha512"]) == (pinned.get("pous"), pinned.get("bandwidth"))
    assert flag["--challenger-command"] == "ssh -i key auditor-1 challenger_party" and flag["--bw-host"] == "192.0.2.2"
    assert "d" * 128 not in argv


def test_an_inherited_or_bound_pous_args_adds_no_word_to_the_tables(tmp_path, capsys, monkeypatch):
    """served_piece.sh evaluates POUS_ARGS as shell words, so under `verity run` it is the table's alone: the value the
    environment carries and one a workload binding names are both replaced by the pieces' own."""
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", RUNNER))
    monkeypatch.setenv("POUS_ARGS", f"--trusted-sha512 {'f' * 128}")
    tables = {**BINDINGS, "workers": {"served-window": {"POUS_ARGS": f"--bw-trusted-sha512 {'f' * 128}"}}}
    state = tmp_path / "state"
    assert R.run_main([str(CONFIG), "--on", T.PROVER_NODE, *SERVED_PARTIES, "--bindings", bindings(tmp_path, tables),
                       "--state", str(state)], REG) == 0, capsys.readouterr().err
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks["POUS_ARGS"] == VARIABLES["POUS_ARGS"]


# --- a development run: the memory attesters unbound, on the workload's machine under keys made for the run ----------

#: `BINDINGS` without the memory attesters'
NO_ATTESTERS = {p: cs for p, cs in BINDINGS.items() if p not in ATTESTERS}
RUN_KEY = "e" * 128
#: the tree's Python's stand-in for served_piece.sh --run-keys: the key's directory and its mode, and a key's SHA-512
TPY = f"""#!/usr/bin/env bash
echo "$3 $(stat -c %a "$(dirname "$3")")" > "$(dirname "$0")/keys.txt"
echo {RUN_KEY}
"""
#: a program's stand-in that keeps its last call's arguments, NUL-separated
ARGS0 = """#!/usr/bin/env bash
printf '%s\\0' "$@" > "$(dirname "$0")/$(basename "$0").args"
"""


def executable(tmp: Path, name: str, body: str) -> Path:
    f = Path(stand_in(tmp, name, body))
    f.chmod(0o755)
    return f


def args0(f: Path) -> list[str]:
    return Path(f"{f}.args").read_text().split("\0")[:-1]


def test_unbound_the_memory_attesters_are_the_runs_and_each_piece_says_so_before_anything_starts(tmp_path, capsys):
    cfg = R.load(CONFIG, REG)
    got = T.variables(cfg, NO_ATTESTERS)
    assert got["POUS_ARGS"] == "--run-keys --rounds 320"
    assert got["WIPE_CMD"] == f'bash {T.WIPE_SCRIPT} "$1" "$2" "$3" --run-keys'
    assert not [k for k, v in got.items() if k != "VERITY_PINS" and any(p in v for p in PIN.values())]
    assert not {"WIPE_ADDRESS", "WIPE_CHALLENGER", "WIPE_COMMAND"} & set(got)
    rc, plan, err = dry(capsys, "--bindings", bindings(tmp_path, NO_ATTESTERS))
    hooks = {(x["party"], e["component"]): e["hooks"] for x in plan["plan"] for e in x["entries"]}
    assert rc == 0, err
    assert [hooks[T.PROVER, c]["DEVELOPMENT"] for c in ("pous", "bandwidth", "wipe")] == ["1"] * 3


@pytest.mark.parametrize("tables, says", [
    ({"memory-challenger": {"pous": {"host": "auditor-1"}}}, "and the deploy bindings bind memory-challenger.pous.host "
                                                             "beside it: bind memory-challenger.pous.address too"),
    ({"memory-challenger": {"wipe": {"command": "ssh auditor-1"}}}, "bind memory-challenger.wipe.address too"),
    ({"bandwidth-verifier": {"bandwidth": {"host": "192.0.2.2"}}},
     "bind bandwidth-verifier.bandwidth.host but not memory-challenger.pous.address: bind both"),
    ({"memory-challenger": {"pous": {"address": "192.0.2.1:7000"}}},
     "bind memory-challenger.pous.address but not bandwidth-verifier.bandwidth.host: bind both"),
])
def test_an_attester_bound_in_part_is_refused_by_name_with_the_fix(tmp_path, capsys, tables, says):
    """Unbound, an attester's other bindings would be dropped, and PoUB's party signs under PoUS's key: a binding given
    without its attester's address, and PoUS and PoUB not bound together, are refused before anything starts."""
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, {**NO_ATTESTERS, **tables}))
    assert rc == 2 and "refused unbound" in err and says in err


def served_piece(tmp: Path, pous_args: str, **env: str) -> tuple[subprocess.CompletedProcess, Path]:
    """served_piece.sh run as the workload runs it, with stand-ins for vLLM's Python and the tree's, and Lean built."""
    py = executable(tmp, "python", ARGV)
    run = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.defpath]), "HOME": str(tmp), "PY": str(py),
           "P2_LIB": str(tmp), "POUS_EXE": "true", "BW_CASE_EXE": "true", "BW_PARTY_EXE": "/lean/poub-verifier",
           "TPY": str(executable(tmp, "tpy", TPY)), "GPU_LEASE_UUID": "no-such-card", "VERITY_PINS": "[]",
           "POUS_ARGS": pous_args, **env}
    p = subprocess.run(["bash", "-c", f'bash {T.POUS_SCRIPT} "$@"', "_", str(tmp / "serve"), "3", str(tmp / "pous")],
                       cwd=ROOT, env=run, capture_output=True, text=True, check=False)
    return p, tmp / "argv.txt"


def test_with_run_keys_served_piece_starts_the_attesters_here_under_one_key_made_for_the_run(tmp_path):
    p, calls = served_piece(tmp_path, T.variables(R.load(CONFIG, REG), NO_ATTESTERS)["POUS_ARGS"])
    assert p.returncode == 0, p.stdout + p.stderr
    argv, = [c.split("\n") for c in calls.read_text().split("\n\n") if "--sessions" in c]
    flag = {f: argv[argv.index(f) + 1] for f in ("--trusted-sha512", "--bw-trusted-sha512", "--challenger-host",
                                                 "--challenger-address", "--challenger-command", "--bw-party",
                                                 "--bw-host", "--rounds")}
    keys = tmp_path / "serve" / "private" / "run-keys"
    attester = f"bash {ROOT / T.ATTESTER_SCRIPT} "
    assert flag == {"--trusted-sha512": RUN_KEY, "--bw-trusted-sha512": RUN_KEY,
                    "--challenger-host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
                    "--challenger-address": "127.0.0.1:47610", "--bw-host": "127.0.0.1", "--rounds": "320",
                    "--challenger-command": f"{attester}challenger_party {tmp_path / 'tpy'} {keys}",
                    "--bw-party": f"{attester}bw_party {tmp_path / 'tpy'} {keys} /lean/poub-verifier"}
    assert "--run-keys" not in argv
    assert (tmp_path / "keys.txt").read_text().split() == [str(keys / "pous"), "700"] and not keys.exists()


@pytest.mark.parametrize("flag", ["--trusted-sha512", "--bw-trusted-sha512", "--challenger-host", "--challenger-address",
                                  "--challenger-command", "--bw-host", "--bw-party"])
def test_run_keys_beside_a_pin_or_an_attesters_flag_is_refused_before_the_card(tmp_path, flag):
    p, calls = served_piece(tmp_path, f"--run-keys --rounds 320 {flag} x")
    assert p.returncode == 2 and "POUS REFUSED: --run-keys" in p.stdout and f"names {flag}: drop {flag}" in p.stdout
    assert not calls.exists() and not (tmp_path / "keys.txt").exists()


def test_the_run_attesters_take_the_gates_words_and_the_runs_key(tmp_path):
    """run_attester.sh's verbs are the gate's (host.py's party_argv) on loopback: the bandwidth party's --pous-session
    and --session read the memory challenger's device in the run's key directory, and any other word is refused."""
    py, keys = executable(tmp_path, "python", ARGS0), tmp_path / "run-keys"
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), os.defpath]), "HOME": str(tmp_path)}

    def attester(*words: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(ROOT / T.ATTESTER_SCRIPT), words[0], str(py), str(keys), *words[1:]],
                              env=env, capture_output=True, text=True, check=False)

    assert attester("challenger_party").returncode == 0
    assert args0(py) == ["-m", "verity.protocols.resource_accounting.memory.pous.attester.challenger_party",
                         "--device", str(keys / "pous"), "--listen", "127.0.0.1:47610"]
    sid, session = "1" * 64, "2" * 64
    assert attester("bw_party", "/lean/poub-verifier", "--minutes", "1", "--pous-session", sid, "--session", session,
                    "--trusted-sha512", RUN_KEY).returncode == 0
    assert args0(py)[2:] == ["--device", str(keys / "bw"), "--listen", "127.0.0.1:0", "--lean-exe", "/lean/poub-verifier",
                             "--pous-record", str(keys / "pous"), "--pous-session", sid, "--challenger",
                             str(keys / "pous"), "--session", session, "--trusted-sha512", RUN_KEY]
    refused = attester("bw_party", "/lean/poub-verifier", "--minutes", "1", "--fixture")
    assert refused.returncode == 2 and "got --fixture" in refused.stderr


@pytest.mark.parametrize("tables, on, coins", [({}, T.PROVER_NODE, "simulated"),
                                               (BINDINGS, T.AUDITOR_HOST, "development")])
def test_the_onsite_line_names_the_machine_each_memory_attesters_key_and_coins_were_made_on(tmp_path, monkeypatch,
                                                                                         tables, on, coins):
    state = tmp_path / "state"
    published(state, {})
    (state / T.BINDINGS).write_text(json.dumps(tables))
    line = T.onsite_check(R.load(CONFIG, REG), state)
    assert line["coins_on"] == {"pous": on, "bandwidth": on, "wipe": on} and line["coins"] == coins


#: nvidia-smi's stand-in: an idle card
IDLE = """#!/usr/bin/env bash
case "$*" in *memory.used*) echo 0 ;; esac
"""
#: demo.sh's stand-in: its arguments, and the directory its scratch is made in with that directory's mode
DEMO = ARGS0 + """echo "$TMPDIR $(stat -c %a "$TMPDIR")" > "$(dirname "$0")/tmpdir.txt"
"""


def served_wipe(tmp: Path, bindings_: dict, **env: str) -> tuple[subprocess.CompletedProcess, Path]:
    """served_wipe.sh run as the workload runs it on an idle card, with demo.sh's stand-in, its Lean built."""
    cfg = R.load(CONFIG, REG)
    hooks = T.PARTIES["wipe"][T.PROVER].variables(cfg, bindings_)
    executable(tmp, "nvidia-smi", IDLE)
    demo = executable(tmp, "demo.sh", DEMO)
    run = {"PATH": os.pathsep.join([str(tmp), str(Path(sys.executable).parent), os.defpath]), "HOME": str(tmp),
           "VERITY_PINS": json.dumps(cfg["pin"]), "WIPE_PROTOCOL_EXE": "true", "GPU_LEASE_UUID": "card-0",
           "WIPE_DEMO": str(demo), **{k: v for k, v in hooks.items() if k != "WIPE_CMD"}, **env}
    p = subprocess.run(["bash", "-c", hooks["WIPE_CMD"], "_", str(tmp / "serve"), "all", str(tmp / "wipe")], cwd=ROOT,
                       env=run, capture_output=True, text=True, check=False)
    return p, demo


@pytest.mark.parametrize("bound", [True, False])
def test_the_wipes_key_is_the_pin_when_bound_and_the_runs_otherwise_and_served_json_says_which(tmp_path, bound):
    tables = {"memory-challenger": {"wipe": {"address": "192.0.2.1:7001", "host": "auditor-1",
                                             "command": "ssh auditor-1"}}} if bound else {}
    p, demo = served_wipe(tmp_path, tables)
    argv, keys = args0(demo), tmp_path / "serve" / "private" / "run-keys" / "wipe"
    pinned = argv[argv.index("--trusted-sha512") + 1] if "--trusted-sha512" in argv else None
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((tmp_path / "wipe" / "served.json").read_text())["key_pin"] == ("configuration" if bound else "run")
    assert "--run-keys" not in argv and argv[argv.index("--reads") + 1] == "5500"
    if bound:
        assert pinned == WIPE_PIN and argv[argv.index("--challenger") + 1] == "auditor-1"
    else:
        assert pinned is None and not {"--challenger", "--challenger-address", "--challenger-command"} & set(argv)
        assert (tmp_path / "tmpdir.txt").read_text().split() == [str(keys), "700"] and not keys.exists()


@pytest.mark.parametrize("name", ["WIPE_KEY_SHA512", "WIPE_ADDRESS", "WIPE_CHALLENGER", "WIPE_COMMAND"])
def test_the_wipes_run_keys_beside_a_pin_or_an_attester_is_refused_before_the_card(tmp_path, name):
    p, demo = served_wipe(tmp_path, {}, **{name: ""})
    assert p.returncode == 3 and not Path(f"{demo}.args").exists()
    assert f"and {name} is set: unset {name}" in (tmp_path / "wipe" / "refused.txt").read_text()


@pytest.mark.parametrize("rows, verdict, reason", [
    ({}, "accept", None),
    ({"communication": "reject"}, "reject", "certifier reject"),
    ({"wipe": "missing"}, "undecided", "wipe missing"),
])
def test_a_set_no_lean_verifier_takes_is_its_rows_combined_in_python_and_says_so(tmp_path, monkeypatch, rows,
                                                                                verdict, reason):
    state = tmp_path / "state"
    published(state, rows)
    line = T.onsite_check(R.load(NETWORK, REG), state)
    assert (line["verdict"], line["reason"], line["product"]) == (verdict, reason, "verity/onsite-network/v0")
    assert line["combined_by"].startswith("python: no Lean verifier takes verity/onsite-network/v0")
    assert line["combined_by"].endswith("not a Lean verifier's")


def test_the_workloads_bindings_are_the_runners_settings_under_the_pieces_hooks(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", 'mkdir -p "$OUT"\necho "$VENV|$SLOTS" > "$OUT/node.txt"\n'))
    tables = {**BINDINGS, "workers": {"served-window": {"VENV": "/workspace/venv312", "SLOTS": "9"}}}
    state = tmp_path / "state"
    assert R.run_main([str(CONFIG), "--on", T.PROVER_NODE, *SERVED_PARTIES, "--bindings", bindings(tmp_path, tables),
                       "--state", str(state)], REG) == 0, capsys.readouterr().err
    assert (state / "out" / "node.txt").read_text() == "/workspace/venv312|4\n"
