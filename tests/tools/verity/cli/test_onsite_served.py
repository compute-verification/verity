"""The served window's onsite configuration (`applications/onsite_full.toml`) and its table (`served_table.py`):
the file is the table's configuration byte for byte and names no host; the network attester's and the inference-only
policy's parties and channels are their manifests'; `verity run --dry-run` resolves every party, channel and piece with
no GPU, and refuses by name a piece that is missing, a channel to an undeclared party and a binding the configuration
has no place for; the runner's variables and the bindings' keys are pinned, and no gate piece names a script, a hook or
a binding; the served window runs once with every piece's hooks, and the onsite check
reads the record's composed verdict, the onsite composite's Lean answer where the composite ran, undecided while the
record has no row for a listed subprotocol or takes no Lean answer. The runner, the record and the composite's public
output are stand-ins written by each test."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
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
from verity.protocols.resource_accounting.memory.pous.attester import wipe_challenger as WC
from verity.protocols.workload_classification.training_detection.inference_only.verifier import parties as IO
from verity.protocols.proofs.spot.verifier.service import SamplingDesign
from verity_cli import run as R

ROOT = Path(R.ROOT)
CONFIG = ROOT / "applications" / "onsite_full.toml"
REG = R.registry()
T = sys.modules["served_table"]
KEY = "0" * 128
#: every binding the pieces read, on documentation addresses (RFC 5737)
BINDINGS = {
    "challenger": {"pouw": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}},
    "memory-challenger": {"pous": {"address": "192.0.2.1:7000"}, "wipe": {"address": "192.0.2.1:7001"}},
    "bandwidth-verifier": {"bandwidth": {"address": "192.0.2.2:7002"}},
    "network-certifier": {"certifier": {"gate": "user@gate-1", "peer": "peer-1"}},
}
UNBOUND = sorted(f"{p}.{c}.{k}" for p, cs in BINDINGS.items() for c, ks in cs.items() for k in ks)
#: the served window's runner's variables under `BINDINGS`, and the bindings the pieces read, in the pieces' order
VARIABLES = {
    "CERTIFIER_CHAIN_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py chain "$@"',
    "CERTIFIER_PERIOD_CMD": '.venv/bin/python benchmarks/network_traces/certifier_piece.py period "$@"',
    "CLASSES": "4,32,4 8,32,4,4,16", "V3C_CLASS": "8,32,4,4,16",
    "CERTIFIER_FRONT": '.venv/bin/python benchmarks/network_traces/certifier_piece.py front "$1" "$2" "$3" '
                       "--gate user@gate-1 --peer peer-1",
    "DEVELOPMENT": "1", "H1_GATE": "gate-command", "H1_KEY_SHA512": KEY, "H1_PEER": "peer-1", "LOAD": "varied",
    "PRECISION": "verity_int8", "REPO": "verity-int8/Llama-3.2-1B-INT8",
    "REV": "15b0976d400d796ee589e7737d03a919526bf800", "SERVE_ARGS": "--protocol ids --id-payload 503", "SLOTS": "4",
    "V3C_LAW": "rate:1/3481648", "WINDOW_STEPS": "16",
}
KEYS = ["challenger.pouw.gate", "challenger.pouw.peer", "challenger.pouw.key-sha512", "memory-challenger.pous.address",
        "bandwidth-verifier.bandwidth.address", "network-certifier.certifier.gate", "network-certifier.certifier.peer",
        "memory-challenger.wipe.address"]
#: each runtime party: its protocol (None while a half isn't in the tree), its role there, its machine and its clock
RUNTIME = {
    ("memory-challenger", "wipe"): ("verity/pous/streamed-wipe", "memory-challenger", "auditor-host",
                                    "CLOCK_MONOTONIC_RAW ns"),
    ("prover", "wipe"): ("verity/pous/streamed-wipe", "prover", "prover-node", None),
    ("memory-challenger", "pous"): ("verity/pous/memory-challenger", "memory-challenger", "auditor-host",
                                    "CLOCK_MONOTONIC ns"),
    ("prover", "pous"): ("verity/pous/memory-challenger", "prover", "prover-node", None),
    ("bandwidth-verifier", "bandwidth"): ("verity/remote-bandwidth/p2/v0", "verifier", "auditor-host",
                                          "CLOCK_MONOTONIC ns"),
    ("prover", "bandwidth"): ("verity/remote-bandwidth/p2/v0", "prover", "prover-node", None),
}


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
    """Each other application's file is the table's configuration of its subprotocols, at onsite_full's commit, whose composite
    (or onsite's smaller set) is its own id; only the subprotocols it lists have parameters and pins."""
    path = ROOT / "applications" / f"{demo}.toml"
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
    assert not {"POUS_CMD", "POUS_ARGS", "WIPE_CMD", "BANDWIDTH_CMD"} & set(hooks)
    assert {e["component"]: e["hooks"]["SERVE_ARGS"] for x in got["plan"] for e in x["entries"]
            if "SERVE_ARGS" in e["hooks"]} == {"certifier": T.ID_FORM}
    assert (hooks["H1_GATE"], hooks["H1_PEER"], hooks["H1_KEY_SHA512"]) == ("gate-command", "peer-1", KEY)
    assert hooks["CERTIFIER_FRONT"].endswith("--gate user@gate-1 --peer peer-1") and "PP_CHALLENGER" not in hooks
    assert hooks["DEVELOPMENT"] == "1"


def test_a_dry_run_prints_each_runtime_party_with_its_protocol_role_machine_clock_and_channels(tmp_path, capsys):
    """The wipe's, PoUS's rounds' and PoUB's parties are their protocols' (`Session`), each on the machine the
    configuration places it on, the clock its header names on the one that reads a clock, the auditor's side listening
    at the address the prover's side dials; PoUB's is ``absent`` while its halves aren't in the tree."""
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path))
    assert rc == 0, err
    runtime = {(x["party"], e["component"]): e for x in got["plan"] for e in x["entries"] if e["runs"] == T.RUNTIME}
    assert set(runtime) == set(RUNTIME)
    for (party, component), (protocol, role, machine, clock) in RUNTIME.items():
        e, s = runtime[party, component], T.PARTIES[component][party].session
        assert (e["role"], e["machine"], e["clock"], e["raises"]) == (role, machine, clock, T.NOT_YET)
        address = BINDINGS[s.listener][component]["address"]
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
    for s in (T.WIPE, T.POUS, T.POUB):
        if s.absent():
            continue
        m = s.protocol().manifest
        assert sorted(s.roles.values()) == sorted(r.name for r in m.parties)
        assert sorted(s.clocks) == sorted(r.name for r in m.parties if "clock" in r.needs)
        assert sorted(s.protocol().parties) == sorted(s.roles.values())
    assert T.WIPE.clocks[T.WP.CHALLENGER][1] == T.sw.CLOCK and T.WIPE.before
    assert not T.POUS.before and not T.POUB.before


def test_a_dry_run_without_bindings_names_every_one_it_lacks(capsys):
    rc, got, _ = dry(capsys)
    assert rc == 0 and got["dry"][0]["unbound"] == UNBOUND
    assert next(e["dials"] for x in got["plan"] if x["party"] == "prover" for e in x["entries"]
                if e["component"] == "pous") == "<memory-challenger.pous.address>"


@pytest.mark.parametrize("demo", ["onsite_network", "onsite_compute"])
def test_a_dry_run_of_a_smaller_onsite_demo_resolves_every_party_and_channel(capsys, demo):
    path = ROOT / "applications" / f"{demo}.toml"
    rc, got, err = dry(capsys, config=path)
    cfg = R.load(path, REG)
    assert rc == 0, err
    assert K.composite(cfg["protocols"], REG.components) in K.UNACCEPTED
    assert sorted(x["party"] for x in got["plan"]) == sorted(p["name"] for p in cfg["party"])
    assert len(got["channel"]) == len(cfg["channel"]) and got["dry"][0]["configuration"] == K.digest(cfg)


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
    assert set(unbound) == set(VARIABLES) | set(deployed)
    assert unbound["CERTIFIER_FRONT"].endswith("--gate '<network-certifier.certifier.gate>' "
                                               "--peer '<network-certifier.certifier.peer>'")
    assert all(unbound[k] == VARIABLES[k] for k in ("CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD"))


def test_the_certifiers_script_is_its_developer_sides_and_no_gate_piece_names_one(tmp_path, capsys):
    cfg = R.load(CONFIG, REG)
    side, device = REG.parties["certifier"][NC.WORKERS], REG.parties["certifier"][NC.NETWORK_CERTIFIER]
    assert (side.runs, side.scripts, side.reaches, side.binds) == (T.SERVED_WINDOW, (T.CERTIFIER_PIECE,),
                                                                    NC.NETWORK_CERTIFIER, ("gate", "peer"))
    assert set(side.variables(cfg, BINDINGS)) == {"SERVE_ARGS", "CERTIFIER_FRONT", "CERTIFIER_CHAIN_CMD",
                                                    "CERTIFIER_PERIOD_CMD", "V3C_CLASS", "CLASSES"}
    assert device.runs == T.GATE
    gates = [e for ps in REG.parties.values() for e in ps.values() if e.runs == T.GATE]
    assert len(gates) == 3
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
                                                                                         monkeypatch):
    """The prover node's runtime parties need what their protocol's start reads (the served store, the card, setup's
    request), which doesn't reach the table yet: the run raises by the row's name before the served window starts."""
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", "touch \"$OUT.started\"\n"))
    state = tmp_path / T.PROVER_NODE
    with pytest.raises(NotImplementedError) as e:
        R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state", str(state)], REG)
    assert e.value.args == (T.NOT_YET,) and not (state / "out.started").exists()


def test_the_memory_attesters_key_is_refused_unless_this_configuration_pins_it_before_anything_else(tmp_path,
                                                                                                     capsys):
    """On the auditor's device each memory attester's start checks its key against `applications/onsite_full.toml`'s pin
    first: a state with no key is refused ``key``, and one whose key the configuration doesn't pin ``pin``, before any
    request is read or any party or the served window starts."""
    state = tmp_path / T.AUDITOR_HOST
    run = [str(CONFIG), "--on", T.AUDITOR_HOST, "--bindings", bindings(tmp_path), "--state", str(state)]
    assert R.run_main(run, REG) == 2 and "refused key" in capsys.readouterr().err
    other = b"a key the auditor never pinned"
    for component in ("pous", "wipe"):
        mine = state / "runs" / component / T.MEMORY_ATTESTER
        mine.mkdir(parents=True)
        (mine / "seed").write_bytes(bytes(lms.DEPLOYED.N))
        lms.LeafState.create(mine / onsite.KEY_STATE, other, lms.DEPLOYED.H)
    assert R.run_main(run, REG) == 2
    err = capsys.readouterr().err
    assert "refused pin" in err and hashlib.sha512(other).hexdigest() in err
    assert not (state / "runs" / "pous" / T.MEMORY_ATTESTER / "tree.bin").exists()


def test_the_wipes_the_rounds_and_poubs_parties_each_have_their_protocols_start():
    for s in (T.WIPE, T.POUS, T.POUB):
        assert sorted(s.protocol().starts) == sorted(s.roles.values())


def test_a_run_raises_at_its_request_until_the_two_sides_compare_it(tmp_path, monkeypatch):
    """With its start and its party key given, a runtime party still raises before it runs: the two sides' requests
    are compared nowhere yet (`request_bound`)."""
    monkeypatch.setitem(WC.PROTOCOL.starts, WC.CHALLENGER, lambda configuration, bound, state: Start({"k": 1}))
    monkeypatch.setattr(T, "held", lambda piece, role: T.Held(None, None))
    piece, cfg = T.PARTIES["wipe"][T.MEMORY_ATTESTER], R.load(CONFIG, REG)
    with pytest.raises(NotImplementedError) as e:
        T.prepare(cfg, tmp_path, BINDINGS, piece)
    assert e.value.args == (T.NOT_YET,)
    monkeypatch.setattr(T, "request_bound", lambda protocol, params: None)
    assert callable(T.prepare(cfg, tmp_path, BINDINGS, piece))


def test_the_onsite_lines_trusted_device_is_each_memory_attesters_placement(tmp_path, monkeypatch):
    """The verifier's line carries each memory attester's `trusted-device` premise as `placement` reads it from the
    challenger's outcome here: under a committed configuration, ``holds`` for a session signed under the pinned key, and
    rehearsal for one that reports no key."""
    key = b"a stand-in memory attester key"
    cfg = R.load(CONFIG, REG)
    cfg = {**cfg, "pin": [{**p, "key-sha512": hashlib.sha512(key).hexdigest()} if p["protocol"] == "pous" else p
                          for p in cfg["pin"]]}
    monkeypatch.setattr(mc, "committed", lambda: frozenset({K.digest(cfg)}))
    outcomes = {"pous": {"outcome": {"public_key": key.hex()}, "refused": None},
                "wipe": {"outcome": {"labels": 0}, "refused": None}}
    monkeypatch.setattr(T, "prepare", lambda cfg, state, bindings, x: lambda: outcomes[x.component])
    monkeypatch.setattr(T, "registration", lambda cfg, state: None)
    monkeypatch.setattr(T, "onsite_check", lambda cfg, state: {"verdict": "undecided"})
    (tmp_path / T.BINDINGS).write_text(json.dumps(BINDINGS))
    line, = T.live(cfg, tmp_path, [T.MEMORY_ATTESTER, T.VERIFIER]).outcomes[T.VERIFIER]["outcome"]["verdicts"]
    assert set(line[mc.PREMISE]) == {"pous", "wipe"} and line["verdict"] == "undecided"
    assert line[mc.PREMISE]["pous"].startswith("holds")
    assert line[mc.PREMISE]["wipe"].startswith(mc.REHEARSAL)


def test_the_bandwidth_verifiers_start_reads_the_memory_attesters_run_beside_it(tmp_path, monkeypatch):
    """PoUB's verifier pins the memory attester's PoUS key, which is in that party's run on the same host: its start
    gets that run's state directory (``challenger``) beside its bindings, and its own state directory."""
    got = {}

    def start(configuration, bound, state):
        got.update(bound=bound, state=state)
        raise NotImplementedError(T.NOT_YET)

    monkeypatch.setitem(T.POUB.starts, T.P2.VERIFIER, start)
    with pytest.raises(NotImplementedError):
        T.prepare(R.load(CONFIG, REG), tmp_path, BINDINGS, T.PARTIES["bandwidth"][T.BANDWIDTH_VERIFIER])
    assert got["bound"] == {**BINDINGS["bandwidth-verifier"]["bandwidth"],
                            "challenger": str(tmp_path / "runs" / "pous" / T.MEMORY_ATTESTER)}
    assert got["state"] == tmp_path / "runs" / "bandwidth" / T.BANDWIDTH_VERIFIER


def test_the_bandwidth_verifiers_start_is_not_yet_a_path_that_runs(tmp_path):
    start = T.POUB.protocol().starts[T.P2.VERIFIER]
    with pytest.raises(NotImplementedError, match=T.P2.GOAL):
        start({}, {"challenger": str(tmp_path)}, None)


def test_the_runtime_components_channels_are_their_protocols_manifests():
    """The configuration's channels of PoUS's rounds, the wipe and PoUB are their protocols' manifests', each role
    under the configuration's name for its party, with the channel to onsite's verifier beside them."""
    for component, s in (("pous", T.POUS), ("wipe", T.WIPE), ("bandwidth", T.POUB)):
        named = {r: c for c, r in s.roles.items()}
        mine = {(c.sender, c.receiver, c.messages, c.carries_coins) for c in T.MANIFESTS[component].channels
                if T.VERIFIER not in (c.sender, c.receiver)}
        assert mine == {(named[c.sender], named[c.receiver], c.messages, c.carries_coins)
                        for c in s.protocol().manifest.channels}


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
/usr/bin/env | grep -E '^(REPO|REV|SLOTS|LOAD|V3C_LAW|V3C_CLASS|CLASSES|POUS_CMD|POUS_ARGS|SERVE_ARGS|WIPE_[A-Z0-9_]+|H1_[A-Z0-9_]+|CERTIFIER_[A-Z_]+)=' \
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
    assert hooks["SLOTS"] == "4" and not {"POUS_CMD", "POUS_ARGS", "WIPE_CMD"} & set(hooks)
    # the served window's rule is the configuration's, a rate: its audit states it to its stream, which takes no fixed count
    assert SamplingDesign(hooks["V3C_LAW"]).text == K.parse(CONFIG.read_text())["service"]["law"]
    kept = set(VARIABLES) - {"PRECISION", "WINDOW_STEPS", "DEVELOPMENT"}
    assert {k: hooks.get(k) for k in kept} == {k: VARIABLES[k] for k in kept}
    assert {"SERVE_ARGS", "H1_GATE", "CERTIFIER_FRONT", "CERTIFIER_CHAIN_CMD", "CERTIFIER_PERIOD_CMD"} <= set(hooks) \
        and "CERTIFIER_CMD" not in hooks
    assert json.loads((state / "bindings.json").read_text()) == BINDINGS
    assert "192.0.2.1" not in json.dumps(json.loads((state / "run.json").read_text()))
    assert hooks["SERVE_ARGS"] == "--protocol ids --id-payload 503"


#: the prover node's parties whose pieces are the served window's alone: its prover's are runtime parties (`live`)
SERVED_PARTIES = ("--party", "workers", "--party", "firewall")


def test_the_workloads_piece_is_the_prover_nodes_first_and_each_verb_it_names_is_the_runners(capsys):
    rc, got, _ = dry(capsys, "--on", T.PROVER_NODE)
    first = got["plan"][0]["entries"][0]
    assert rc == 0 and (got["plan"][0]["party"], first["component"]) == (T.WORKERS, T.SERVED_WINDOW)
    assert first["verbs"] == ["predict", "serve", "seal", "window", "advice", "circuits", "publish"]
    runner = (ROOT / T.RUNNER).read_text()
    assert [v for v in first["verbs"] if f'if [ "${{1:-}}" = {v} ]; then' not in runner] == []
    assert first["writes"] and all(w.split("/", 1)[0] in ("D", "OUT") for w in first["writes"])
    assert all("verbs" not in e for x in got["plan"] for e in x["entries"] if e is not first)


#: what the adapter's `run.sh settings MODEL -- verity run ...` sets of the node the workload runs on
NODE_SETTINGS = ("VENV", "CUDA_HOME", "HF_HOME", "FLOCK_WORK", "DISK_FLOOR_GB", "HIDDEN_SO", "NORM_TAP_SO",
                 "SERVE_ROOT", "ROW", "ROW_ID")


def test_the_workloads_piece_hands_the_runner_the_nodes_settings(tmp_path, capsys, monkeypatch):
    settings = {k: str(tmp_path / "node" / k.lower()) for k in NODE_SETTINGS}
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    names = "|".join(NODE_SETTINGS)
    monkeypatch.setattr(T, "RUNNER", stand_in(tmp_path, "runner.sh", f'mkdir -p "$OUT"\n/usr/bin/env | grep -E '
                                                                    f"'^({names})=' | sort > \"$OUT/node.txt\"\n"))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, *SERVED_PARTIES, "--bindings", bindings(tmp_path), "--state",
                     str(state)], REG)
    assert rc == 0, capsys.readouterr().err
    assert dict(x.split("=", 1) for x in (state / "out" / "node.txt").read_text().splitlines()) == settings


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


#: `served_verdicts.py record` as the onsite check runs it: its rows, and its verdict the Lean answer of the one onsite
#: composite under its ``--composite`` (``pending`` without one, with a development one or one of another product, or
#: while a row is missing; a Lean reject rejects), with ``lean`` saying where that answer was read or why none was
#: taken, in `served_verdicts.composite_answer`'s words; its arguments go to ``argv.json`` beside its ``--out``
RECORD = """import glob, json, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == "window":
    print(json.dumps({"window": 2, "steps": [16, 32]}))
    sys.exit(0)
out = Path(a[a.index("--out") + 1])
out.mkdir(parents=True)
(out.parent / "argv.json").write_text(json.dumps(a))
rows = {"integrity": "accept", "work": "accept", "space": "accept", "communication": "accept", "wipe": "accept"}
rows.update(json.loads(%r))
said, lean = None, "no composite was read"
if "--composite" in a:
    public = a[a.index("--composite") + 1]
    found = [(Path(f).parent.name, json.load(open(f))) for f in sorted(glob.glob(public + "/*/composite.json"))]
    found = [(d, doc) for d, doc in found if doc["product"] == "verity/onsite/v0"]
    if len(found) != 1:
        lean = f"{len(found)} composites of the onsite check (verity/onsite/v0) in {public}, not one"
    elif found[0][1].get("development"):
        lean = (f"{found[0][0]}/composite.json is a development check, which claims no record: "
                + found[0][1]["development"])
    else:
        said = found[0][1]["verdict"]["verdict"]
        lean = f"verity/onsite/v0's Lean check, compiled, in {found[0][0]}/composite.json"
missing = [] if "bandwidth" in rows else ["bandwidth"]
composed = "reject" if said == "reject" else "accept" if said == "accept" and not missing else "pending"
(out / "verdicts.json").write_text(json.dumps({"protocols": {k: {"verdict": v} for k, v in rows.items()}}))
print("RECORD " + json.dumps({"composed": composed, "missing": missing, "lean": lean,
                              "node": a[a.index("--node") + 1]}))
"""
ALL_ROWS = {"bandwidth": "accept", "inference-only": "accept"}
LEAN_IN = "verity/onsite/v0's Lean check, compiled, in onsite/composite.json"


def onsite_composite(out: Path, product: str, verdict: str, development: str | None = None) -> None:
    """``composite.py``'s public composite.json of ``product`` under ``out/composite``, its Lean answering
    ``verdict``."""
    d = out / "composite" / product.split("/")[1]
    d.mkdir(parents=True)
    doc = {"product": product, "verdict": {"id": product, "verdict": verdict}}
    d.joinpath("composite.json").write_text(json.dumps({**doc, **({"development": development} if development else {})}))


@pytest.mark.parametrize("rows, composite, verdict, rc, reason", [
    ({}, None, "undecided", 2, "missing ['bandwidth']"),
    ({}, ("verity/onsite/v0", "accept"), "undecided", 2, "missing ['bandwidth']"),
    ({"bandwidth": "accept"}, ("verity/onsite/v0", "accept"), "undecided", 2, "no row for ['inference-only']"),
    (ALL_ROWS, None, "undecided", 2, "no composite was read"),
    (ALL_ROWS, ("verity/onsite/v0", "accept"), "accept", 0, None),
    (ALL_ROWS, ("verity/onsite/v0", "reject"), "reject", 1, LEAN_IN),
    (ALL_ROWS, ("verity/onsite/v0", "undecided"), "undecided", 2, LEAN_IN),
    (ALL_ROWS, ("verity/onsite/v0", "accept", "composed on the served node"), "undecided", 2,
     "onsite/composite.json is a development check, which claims no record: composed on the served node"),
    (ALL_ROWS, ("verity/remote/v0", "accept"), "undecided", 2,
     "0 composites of the onsite check (verity/onsite/v0) in {c}, not one"),
])
def test_the_onsite_check_is_the_records_composed_verdict(tmp_path, capsys, monkeypatch, rows, composite, verdict, rc,
                                                          reason):
    """The onsite check hands the record the composite's public output where it is there (``--composite``), so the
    period's verdict is the compiled Lean onsite check's answer as the record takes it: an accept only where that Lean
    accepted and the record has a row for every listed subprotocol, a reject where it rejected. With no composite, a
    development one, one of another product or an undecided one, the line is undecided and its reason is the record's
    ``lean``, or the row it lacks."""
    monkeypatch.setattr(T, "VERDICTS_SCRIPT", stand_in(tmp_path, "record.py", RECORD % json.dumps(rows)))
    state = tmp_path / "state"
    (state / "out").mkdir(parents=True)
    (state / "out" / "machine.txt").write_text("research\nprover-1\n")
    if composite:
        onsite_composite(state / "out", *composite)
    assert R.run_main([str(CONFIG), "--on", T.AUDITOR_HOST, "--party", "verifier", "--state", str(state)], REG) == rc
    out = capsys.readouterr().out
    assert {x["party"]: x["refused"] for x in said(out, "PARTY")} == {"verifier": None}
    (line,) = said(out, "VERDICT")
    assert line["verdict"] == verdict and line["block"] == 2 and line["product"] == T.ONSITE
    assert line["components"] == {**{c: "accept" for c in T.PROTOCOLS},
                                  **{c: rows.get(c, "absent") for c in ("bandwidth", "inference-only")}}
    assert line["coins"] == "development"
    assert line["reason"] == (reason and reason.replace("{c}", str(state / "out" / "composite")))
    argv = json.loads((state / "record" / "argv.json").read_text())
    assert argv[argv.index("--composite") + 1:][:1] == [str(state / "out" / "composite")] if composite else \
        "--composite" not in argv


def test_a_smaller_demos_check_reads_no_onsite_composite(tmp_path, monkeypatch):
    """An application whose product isn't onsite's hands the record no composite: onsite's Lean answer is not its product's."""
    monkeypatch.setattr(T, "VERDICTS_SCRIPT", stand_in(tmp_path, "record.py", RECORD % json.dumps(ALL_ROWS)))
    cfg = T.configuration(R.load(CONFIG, REG)["program"]["commit"], "onsite_compute")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "machine.txt").write_text("research\nprover-1\n")
    onsite_composite(tmp_path / "out", "verity/onsite/v0", "accept")
    got = T.onsite_check(cfg, tmp_path)
    assert "--composite" not in json.loads((tmp_path / "record" / "argv.json").read_text())
    assert got["product"] != T.ONSITE and got["verdict"] == "undecided" and got["reason"] == "no composite was read"


def test_pouws_piece_hands_the_runner_the_main_file_its_challenger_writes():
    cfg, piece = R.load(CONFIG, REG), T.PARTIES["pouw"][T.WORKERS]
    gate = BINDINGS["challenger"]["pouw"]
    assert "MAIN_FILE" not in piece.variables(cfg, BINDINGS)
    assert piece.variables(cfg, {"challenger": {"pouw": {**gate, "main-file": "/run/public/challenger/main.txt"}}}
                           )["MAIN_FILE"] == "/run/public/challenger/main.txt"


def test_a_dry_run_counts_what_a_challenger_started_first_writes_as_bound(capsys):
    start = R.Start({"pouw": ("gate", "peer", "key-sha512", "main-file")}, lambda c, s: None, lambda c, s: None)
    rc, got, err = dry(capsys, reg=dataclasses.replace(REG, starts={T.CHALLENGER: start}))
    assert rc == 0, err
    assert {x["party"]: x.get("writes") for x in got["plan"]}[T.CHALLENGER] == {
        "pouw": ["gate", "peer", "key-sha512", "main-file"]}
    assert got["dry"][0]["unbound"] == sorted(set(UNBOUND) - {f"challenger.pouw.{k}" for k in ("gate", "peer",
                                                                                              "key-sha512")})
