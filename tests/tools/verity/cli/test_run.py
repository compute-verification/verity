"""`verity config` and `verity run` (`verity_cli.run`, `verity.config`): the configuration's refusals by name,
a list of subprotocols that is neither composite's among them and a malformed pin, its hash, which covers the pins and
is unchanged by stating none, the example reproduced by `config new`, and
runs of a step-0 table (`step0.py`) for onsite and remote: its entries in this process, and core's SPOT
parties in the deployment layer's live run, on a machine with stand-in capabilities and a stand-in product executable,
every party on in-memory channels, so no build, GPU, network or clock is involved."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity import config as K
from verity.protocol import Channel, Role
from verity.protocols.resource_accounting.network.verifier import parties as NC
from verity.protocols.resource_accounting.network.verifier.network_attester import NetworkAttesterParams
from verity.protocols.workload_classification.training_detection.inference_only.verifier import parties as IO
from verity.protocols.proofs.spot.verifier import parties as SP
from verity.protocols.proofs.spot.verifier.one_stage import docket as RG
from verity.channel import Refused
import verity_cli as V
from verity_cli import run as R

BUILD_VERIFIER = R.build_verifier
EXAMPLE = Path(__file__).parent / "fixtures" / "config.toml"
TABLE = (Path(__file__).parent, "step0")
REG = R.registry(TABLE)
D = sys.modules[TABLE[1]]
COMMIT = "a" * 40
ONSITE = ("sampled-proofs", "certifier", "pouw", "pous", "bandwidth", "inference-only", "wipe")
REMOTE = ("sampled-proofs", "pearl-plus", "inference-only")
REMOTE_V0 = ("sampled-proofs", "pouw", "inference-only")


def config(protocols=ONSITE, **machines) -> dict:
    cfg = K.new("toy", list(protocols), REG.components, REG.workloads, COMMIT)
    for p in cfg["party"]:
        p["machine"] = machines.get(p["name"].replace("-", "_"), p["machine"])
    return cfg


def refused(cfg) -> str:
    with pytest.raises(Refused) as e:
        K.check(cfg, REG.components, REG.workloads)
    return e.value.name


# --- the configuration -------------------------------------------------------------------------------------------------

def test_the_example_is_config_new_byte_for_byte():
    text = EXAMPLE.read_text()
    commit = K.parse(text)["program"]["commit"]
    cfg = K.new("toy", list(REMOTE), REG.components, REG.workloads, commit)
    header = [h.format(protocols=" and ".join(REMOTE), workload="toy", machine="local") for h in R.HEADER]
    assert K.render(cfg, header) == text
    assert K.parse(text) == cfg


def edit(fn):
    cfg = config()
    fn(cfg)
    return cfg


def party(cfg, name):
    return next(p for p in cfg["party"] if p["name"] == name)


def channel(cfg, source, to):
    return next(c for c in cfg["channel"] if (c["from"], c["to"]) == (source, to))


#: a pin the toy's onsite configuration accepts: the network attester's key for the network attester's sessions
PIN = {"party": "network-certifier", "protocol": "certifier", "scheme": K.SCHEMES[0], "key-sha512": "5a" * 64}


def pinned(**fields) -> dict:
    return edit(lambda c: c.update(pin=[PIN | fields]))


REFUSALS = {
    "unknown-key": lambda c: c.update(extra=1),
    "missing-key": lambda c: c.pop("service"),
    "unknown-version": lambda c: c.update(format="verity/config/v1"),
    "unknown-format": lambda c: c.update(format="other/v0"),
    "no-protocol": lambda c: c.update(protocols=[]),
    "unknown-protocol": lambda c: c.update(protocols=[*ONSITE, "telepathy"]),
    "repeated": lambda c: c.update(protocols=["sampled-proofs", "sampled-proofs"]),
    "requires": lambda c: c.update(protocols=["certifier"]),
    "composite": lambda c: c.update(protocols=["sampled-proofs", "certifier"]),
    "unknown-workload": lambda c: c["workload"].update(name="mainframe"),
    "unknown-params": lambda c: c.update(protocols=list(REMOTE), params={SP.IDS["pous"]: {}}),
    "tie": lambda c: c.update(params={"verity/certifier/v0": {"tie": False}}),
    "law": lambda c: c["service"].update(law="subset:8"),
    "commit": lambda c: c["program"].update(commit="main"),
    "float": lambda c: c["workload"].update(params={"steps": 2.0}),
    "workload": lambda c: c["workload"].update(params={"requests": []}),
    "principal": lambda c: party(c, "challenger").update(principal="developer"),
    "unknown-party": lambda c: c["party"].append({"name": "oracle", "machine": "local", "principal": "anyone"}),
    "missing-party": lambda c: c["party"].remove(party(c, "firewall")),
    "repeated-party": lambda c: c["party"].append(dict(party(c, "firewall"))),
    "machine": lambda c: party(c, "workers").update(machine="Local Box"),
    "listen": lambda c: party(c, "verifier").update(listen=70000),
    "undeclared-channel": lambda c: c["channel"].append({"from": "workers", "to": "challenger", "messages": ["coins"]}),
    "channel": lambda c: channel(c, "challenger", "verifier").update(messages=["coins"]),
    "missing-channel": lambda c: c["channel"].remove(channel(c, "firewall", "verifier")),
    "repeated-channel": lambda c: c["channel"].append(dict(channel(c, "firewall", "verifier"))),
    "dial": lambda c: channel(c, "firewall", "verifier").update(dial="both"),
    "placement": lambda c: party(c, "network-certifier").update(machine="auditor-2"),
    "pin-hash": lambda c: c.update(pin=[PIN | {"key-sha512": "5A" * 64}]),
    "pin-scheme": lambda c: c.update(pin=[PIN | {"scheme": "ed25519"}]),
    "pin-party": lambda c: c.update(pin=[PIN | {"party": "oracle"}]),
    "pin-principal": lambda c: c.update(pin=[PIN | {"party": "workers", "protocol": "sampled-proofs"}]),
    "pin-protocol": lambda c: c.update(pin=[PIN | {"protocol": "pous"}]),
    "repeated-pin": lambda c: c.update(pin=[PIN, PIN | {"key-sha512": "a5" * 64}]),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_configuration_is_refused_by_name(name):
    assert refused(edit(REFUSALS[name])) == name


@pytest.mark.parametrize("protocols", [("sampled-proofs",), ("sampled-proofs", "certifier"), ("pouw",), ONSITE[:-1],
                                       ("sampled-proofs", "certifier", "pouw"), ("sampled-proofs", "wipe"),
                                       ("sampled-proofs", "pouw"), ONSITE[:-2]])
def test_a_set_of_subprotocols_that_is_neither_composite_is_refused(protocols):
    assert refused(edit(lambda c: c.update(protocols=list(protocols)))) == "composite"
    with pytest.raises(K.Refused) as e:
        K.new("toy", list(protocols), REG.components, REG.workloads, COMMIT)
    assert e.value.name in ("composite", "requires")


def test_the_composites_are_remote_and_onsite():
    assert {K.composite(p, REG.components) for p in (ONSITE, REMOTE)} == {"verity/onsite/v0", "verity/remote/v1"}
    assert K.composite(ONSITE[::-1], REG.components) == "verity/onsite/v0"
    assert {cid: sorted(names) for cid, names in K.COMPOSITES.items()} == {
        cid: sorted(subs) for cid, subs in SP.SUBPROTOCOLS.items()}
    assert SP.SUBPROTOCOLS["verity/remote/v1"] == {"inference-only": "verity/inference-only/v0",
                                                   "pearl-plus": "verity/pouw/pearl-plus/v1",
                                                   "sampled-proofs": "verity/sampled-proofs/v1"}
    assert SP.SUBPROTOCOLS["verity/onsite/v0"]["sampled-proofs"] == "verity/sampled-proofs/v1"
    assert SP.SUBPROTOCOLS["verity/onsite/v0"]["pouw"] == "verity/pouw/period/v1"
    assert SP.SUBPROTOCOLS["verity/onsite/v0"]["wipe"] == "verity/wipe/v0"


def test_a_workload_that_lacks_a_need_is_refused_by_name():
    bare = dataclasses.replace(REG.workloads["toy"], provides=frozenset({"windows", "ingress-relay"}))
    with pytest.raises(K.Refused) as e:
        K.check(config(), REG.components, {"toy": bare})
    assert e.value.name == "needs"


def test_two_declarations_of_a_party_agree_and_of_a_channel_are_its_union():
    toy = REG.workloads["toy"]
    a = K.Component("a", "verity/a/v0", (Role("verifier", "auditor", needs=("audit",)),),
                    (Channel("workers", "verifier", ("x",), latency_ms=9),))
    b = K.Component("b", "verity/b/v0", (Role("verifier", "auditor", needs=("product",)),),
                    (Channel("workers", "verifier", ("y",), latency_ms=3, carries_coins=True, after=("x",)),))
    parties, channels = K.declared([a, b], toy)
    assert parties["verifier"].needs == ("audit", "product") and parties["workers"] == Role("workers", "developer")
    assert channels["workers", "verifier"] == Channel("workers", "verifier", ("x", "y"), 3, True, ("x",))
    with pytest.raises(ValueError, match="c declares party verifier as developer per one, another declaration as "
                                         "auditor per one"):
        K.declared([a, K.Component("c", "verity/c/v0", (Role("verifier", "developer"),))], toy)


def test_the_toml_syntax_is_refused_by_name():
    with pytest.raises(Refused) as e:
        K.parse("format = ")
    assert e.value.name == "toml"


def test_the_hash_leaves_out_the_transport_and_fills_the_defaults():
    base = K.digest(K.check(config(), REG.components, REG.workloads))
    moved = edit(lambda c: (party(c, "challenger").update(address="10.0.0.2", listen=7000),
                            channel(c, "challenger", "verifier").update(dial="to")))
    assert K.digest(K.check(moved, REG.components, REG.workloads)) == base
    stated = edit(lambda c: c["workload"].update(params={"steps": 2}))
    assert K.digest(K.check(stated, REG.components, REG.workloads)) == base
    changed = edit(lambda c: c["workload"].update(params={"steps": 3}))
    assert K.digest(K.check(changed, REG.components, REG.workloads)) != base
    assert set(K.TRANSPORT["party"]) <= set(K.PARTY) and set(K.TRANSPORT["channel"]) <= set(K.CHANNEL)


#: The onsite configuration's hash with neither the network attester's bound on a response's tokens nor its lengths
#: rows' form set: a configuration that sets neither keeps it.
UNBOUNDED_ONSITE = ("1cd21b0103bf0bf930af8d251e8d8f0a12153d66d56dfbc7d1583b1d3133c87f"
                    "151b5f774ed98718aabd3453ed9e947fbab982e2b8a107459dbfc1d1d099e17a")


def bounded(tokens):
    return edit(lambda c: c.setdefault("params", {}).setdefault("verity/certifier/v0", {}).update(max_tokens_cap=tokens))


def test_an_unbounded_certifier_configuration_hashes_as_before_the_bound():
    cfg = K.check(config(), REG.components, REG.workloads)
    assert not {"max_tokens_cap", "lengths", "content"} & set(cfg["params"]["verity/certifier/v0"])
    assert K.digest(cfg) == UNBOUNDED_ONSITE


def test_a_bounded_certifier_configuration_round_trips():
    cfg = K.check(K.parse(K.render(bounded(4))), REG.components, REG.workloads)
    table = cfg["params"]["verity/certifier/v0"]
    assert table["max_tokens_cap"] == 4
    assert NetworkAttesterParams(**{k: v for k, v in table.items() if k != "tie"}).max_tokens_cap == 4
    assert K.check(cfg, REG.components, REG.workloads) == cfg
    assert K.digest(cfg) != UNBOUNDED_ONSITE


def with_lengths(**schedule):
    table = {"lengths": 1} | schedule
    return edit(lambda c: c.setdefault("params", {}).setdefault("verity/certifier/v0", {}).update(table))


def test_a_certifier_configuration_with_lengths_rows_round_trips_and_needs_a_schedule_that_fits_them():
    fits = {"cell_bytes": 512, "ingress_buckets": 10, "egress_cells": 2048}
    digests = {UNBOUNDED_ONSITE}
    for form in (1, 2):  # the lengths rows' two forms, each a configuration of its own digest
        cfg = K.check(K.parse(K.render(with_lengths(**fits | {"lengths": form}))), REG.components, REG.workloads)
        table = cfg["params"]["verity/certifier/v0"]
        assert table["lengths"] == form and "max_tokens_cap" not in table
        assert NetworkAttesterParams(**{k: v for k, v in table.items() if k != "tie"}).lengths == form
        assert K.check(cfg, REG.components, REG.workloads) == cfg
        digests.add(K.digest(cfg))
    assert len(digests) == 3
    for schedule in ({}, fits | {"lengths": 3}, fits | {"egress": "bytes"}):
        with pytest.raises(Refused) as e:
            K.check(with_lengths(**schedule), REG.components, REG.workloads)
        assert e.value.name == "params" and "lengths" in e.value.detail


@pytest.mark.parametrize("tokens", [0, -1, True, "4", 4096, 1 << 32])
def test_a_bad_response_bound_is_refused_by_name(tokens):
    with pytest.raises(Refused) as e:
        K.check(bounded(tokens), REG.components, REG.workloads)
    assert e.value.name == "params" and "max_tokens_cap" in e.value.detail


@pytest.mark.parametrize("fields, name", [
    ({"key-sha512": "5a" * 63}, "pin-hash"), ({"key-sha512": "5a" * 65}, "pin-hash"), ({"key-sha512": 5}, "pin-hash"),
    ({"protocol": "telepathy"}, "pin-protocol"), ({"party": "challenger", "protocol": "pouw"}, "pin-protocol"),
    ({"host": "auditor-1"}, "unknown-key"), ({"session": "s-1"}, "unknown-key"),
])
def test_a_malformed_pin_is_refused_by_name(fields, name):
    assert refused(pinned(**fields)) == name


@pytest.mark.parametrize("cfg, name", [
    (edit(lambda c: c.update(pin=[{k: v for k, v in PIN.items() if k != "scheme"}])), "missing-key"),
    (edit(lambda c: c.update(pin=PIN)), "type"),
])
def test_a_pin_that_is_not_an_entry_is_refused_by_name(cfg, name):
    assert refused(cfg) == name


def test_the_schemes_are_the_domain_tables():
    from verity.commitments import domains

    assert K.SCHEMES == (domains.LMS_SHA512_N64_W1_H20,)


def test_pins_are_hashed_and_round_trip_and_none_hashes_as_no_pins():
    base = K.digest(K.check(config(), REG.components, REG.workloads))
    assert K.digest(K.check(edit(lambda c: c.update(pin=[])), REG.components, REG.workloads)) == base
    challenger = PIN | {"party": "challenger", "protocol": "sampled-proofs", "key-sha512": "c3" * 64}
    cfg = edit(lambda c: c.update(pin=[PIN, challenger]))
    text = K.render(cfg)
    assert "[[pin]]" in text and K.parse(text) == cfg
    checked = K.check(K.parse(text), REG.components, REG.workloads)
    assert checked["pin"] == [challenger, PIN] and K.check(checked, REG.components, REG.workloads) == checked
    assert K.parse(K.render(checked)) == checked
    assert K.hashed(checked)["pin"] == checked["pin"]
    digest = K.digest(checked)
    assert digest != base
    assert K.digest(K.check(edit(lambda c: c.update(pin=[challenger, PIN])), REG.components, REG.workloads)) == digest
    assert K.digest(K.check(edit(lambda c: c.update(pin=[challenger, PIN | {"key-sha512": "5b" * 64}])),
                            REG.components, REG.workloads)) != digest


def test_config_new_with_pins_renders_and_checks_back(capsys, tmp_path):
    cfg = K.new("toy", list(ONSITE), REG.components, REG.workloads, COMMIT)
    assert "pin" not in cfg
    cfg["pin"] = [PIN]
    path = tmp_path / "config.toml"
    path.write_text(K.render(cfg))
    assert R.config_main(["hash", str(path)], REG) == 0
    assert capsys.readouterr().out.strip() == K.digest(K.check(cfg, REG.components, REG.workloads))
    assert R.load(path, REG)["pin"] == [PIN]


def test_the_canonical_form_is_verity_canonical_json():
    from verity.circuits.codec import canonical_json

    cfg = K.check(config(), REG.components, REG.workloads)
    assert K.canonical(cfg) == canonical_json(cfg)


def test_config_check_and_hash_print_the_hash(capsys):
    assert R.config_main(["check", str(EXAMPLE)], REG) == 0
    ok = capsys.readouterr().out.split()
    assert R.config_main(["hash", str(EXAMPLE)], REG) == 0
    assert ok == ["ok", capsys.readouterr().out.strip()]
    assert R.config_main(["new", "--workload", "toy", "--protocols", "certifier"], REG) == 2
    assert capsys.readouterr().out.startswith("refused requires:")
    assert R.config_main(["new", "--workload", "toy", "--protocols", "sampled-proofs,certifier"], REG) == 2
    assert capsys.readouterr().out.startswith("refused composite:")


# --- the run -----------------------------------------------------------------------------------------------------------

def stand_in_product(tmp: Path) -> Path:
    """``product IDS id | check PARAMS COMMITMENT EVIDENCE``: the composite of IDS's subprotocols (64 for any other
    set), which accepts when every subprotocol's case evidence says so."""
    composites = {cid: sorted(subs.values()) for cid, subs in SP.SUBPROTOCOLS.items()}
    exe = tmp / "product"
    exe.write_text(f"#!{sys.executable}\n" + f"COMPOSITES = {composites!r}\n" + """import json, sys
ids = sys.argv[1].split(",")
named = [c for c, want in COMPOSITES.items() if sorted(ids) == want]
if not named:
    raise SystemExit(64)
if sys.argv[2] == "id":
    print(named[0])
    raise SystemExit(0)
bad = [c["id"] for c in json.loads(open(sys.argv[5]).read())["components"] if bytes.fromhex(c["evidence"]) != b"accept"]
print(json.dumps({"verdict": "reject" if bad else "accept", "reason": f"refused {bad}" if bad else None}))
raise SystemExit(1 if bad else 0)
""")
    exe.chmod(0o755)
    return exe


def write(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def periods(state: Path) -> range:
    return range(json.loads((state / "run.json").read_text())["periods"])


def case(verdict: str) -> dict:
    return {"commitment": "", "evidence": verdict.encode().hex(), "params": {}}


class StandIn(D.Machine):
    """`step0.Machine` with every capability but the product a stand-in. ``verdicts[(who, period)]`` is what that
    subprotocol's check answers (``accept`` by default; ``silent``: nothing; ``raise``: a fault; ``unavailable``: a step
    it can't do), or for ``workers`` whether they serve; a case's evidence is its verdict, so the stand-in product
    accepts only over accepting components."""

    def __init__(self, config, state, verdicts):
        super().__init__(config, state, {c: functools.partial(self.check, c) for c in D.CHECKS})
        self.verdicts = verdicts

    def _of(self, who, i):
        v = self.verdicts.get((who, i), "accept")
        if v == "raise":
            raise RuntimeError(f"{who}'s check failed")
        if v == "unavailable":
            raise SP.Unavailable(f"{who} can't")
        return v

    def host(self):
        state = self.state

        class Host:
            def start(self):
                write(state / "challenger" / "address.json", {"address": "127.0.0.1:1"})
                return {"address": "127.0.0.1:1", "pin": "f" * 40}

            def stop(self):
                write(state / "challenger" / "host" / "ledger.json", {"closed": True})
        return Host()

    def pin(self):
        return {"sha512": "5a" * 64, "sessions": 1}

    def serve(self, i):
        assert "ingress-relay" not in D.run_of(self.state)["hooks"] or \
            (self.state / "blocks" / "certifier-session" / "relay.json").exists()
        self._of("workers", i)
        write(self.state / "blocks" / f"w{i}" / "period.json", {"period": i})
        write(self.state / "public" / f"w{i}" / "registration.json", {"period": i})
        return json.dumps({"period": i}).encode()

    def audit(self, i, registration, challenger, zk_server):
        v = self._of("sampled-proofs", i)
        answer = {"verdict": v, "commitment": RG.digest(json.loads(registration))}
        write(self.state / "record" / f"w{i}" / "sampled-proofs" / "answer.json", answer)
        return answer, case(v)

    def check(self, who, cfg, state, i):
        v = self._of(who, i)
        return None if v == "silent" else ({"verdict": v, "commitment": "00"}, case(v))


def registry(verdicts: dict | None = None, certifier_rc: int = 0) -> R.Registry:
    def network_attester(cfg, state, local):
        write(state / "blocks" / "certifier-session" / "relay.json", {"periods": len(periods(state))})
        return certifier_rc

    parties = {**REG.parties, "certifier": {**REG.parties["certifier"], "network-certifier": network_attester}}
    return R.Registry(REG.components, REG.workloads, REG.workers, parties, REG.order,
                      functools.partial(D.live, machine=lambda c, s: StandIn(c, s, verdicts or {})))


def run(tmp: Path, reg: R.Registry, *args: str, cfg: dict | None = None, named: bool = False) -> tuple[int, list[dict]]:
    path = tmp / "config.toml"
    path.write_text(K.render(cfg or config()))
    s = tmp / "state"
    s.mkdir(exist_ok=True)
    if not (s / "tools.json").exists():
        (s / "tools.json").write_text(json.dumps({"product": str(stand_in_product(tmp))}))
    rc = R.run_main([path.stem if named else str(path), "--state", str(s), *args], reg, in_process=True)
    lines = (s / "record" / "verdicts.jsonl").read_text().splitlines() if (s / "record" / "verdicts.jsonl").exists() else []
    return rc, [json.loads(x) for x in lines]


def said(out: str, what: str = "PARTY") -> list[dict]:
    return [json.loads(x.split(" ", 1)[1]) for x in out.splitlines() if x.startswith(what + " ")]


def test_every_party_here_accepts(tmp_path, capsys):
    rc, lines = run(tmp_path, registry())
    out = capsys.readouterr().out
    assert rc == 0 and [x["verdict"] for x in lines] == ["accept"]
    line = lines[0]
    assert line["components"] == dict.fromkeys(ONSITE, "accept")
    assert line["product"] == "verity/onsite/v0" and line["coins"] == "simulated" and line["block"] == 0
    assert line["registration"] == RG.digest({"period": 0})
    assert any("Faithful" in u for u in line["simulated"]) and any(COMMIT in u for u in line["simulated"])
    assert said(out, "VERDICT") == lines
    assert said(out) == [{"party": "network-certifier", "exit": 0},
                         *({"party": p, "refused": None} for p in ("challenger", "firewall", "verifier", "workers"))]
    state = tmp_path / "state"
    relay = [json.loads(x) for x in (state / "transcripts" / "network-certifier.jsonl").read_text().splitlines()]
    assert [(x["file"], x["kind"]) for x in relay] == [("blocks/certifier-session/relay.json", "private")]
    assert (state / "challenger" / "host" / "ledger.json").exists()
    kept = state / "runs" / "local" / "runs" / "1"
    assert (kept / "record").exists() and {f.name for f in kept.glob("*.transcript")} == {
        f"{p}.transcript" for p in ("challenger", "firewall", "verifier", "workers")}


@pytest.mark.parametrize("who", ONSITE[1:])
def test_a_subprotocol_that_rejects_rejects(tmp_path, who):
    rc, lines = run(tmp_path, registry({(who, 0): "reject"}))
    assert rc == 1 and lines[0]["verdict"] == "reject"
    assert lines[0]["components"] == {**dict.fromkeys(ONSITE, "accept"), who: "reject"}


@pytest.mark.parametrize("how, status", [("silent", "absent"), ("error", "error")])
def test_a_subprotocol_with_no_answer_never_accepts(tmp_path, how, status):
    rc, lines = run(tmp_path, registry({("certifier", 0): how}))
    assert rc == 1 and lines[0]["verdict"] == "reject" and lines[0]["components"]["certifier"] == status


def test_a_check_that_faults_fails_the_run_with_no_verdict(tmp_path, capsys):
    rc, lines = run(tmp_path, registry({("certifier", 0): "raise"}))
    assert rc == 2 and lines == []
    assert {"party": "verifier", "refused": "party-died"} in said(capsys.readouterr().out)


def test_remotes_verdict_covers_exactly_its_subprotocols(tmp_path, capsys):
    rc, lines = run(tmp_path, registry(), cfg=config(REMOTE))
    assert rc == 0 and lines[0]["components"] == {c: "accept" for c in REMOTE}
    assert lines[0]["product"] == "verity/remote/v1"
    assert "network-certifier" not in capsys.readouterr().out


def test_remotes_development_composite_is_no_composite():
    """H1's development composite (remote with PoUW's period as its work) is no composite a configuration can name:
    protocol soundness covers only onsite and remote v1, so `check` refuses it by name."""
    assert refused(edit(lambda c: c.update(protocols=list(REMOTE_V0)))) == "composite"
    with pytest.raises(K.Refused) as e:
        K.composite(REMOTE_V0, REG.components)
    assert e.value.name == "composite" and "verity/remote/v1 (inference-only, pearl-plus, sampled-proofs)" in str(e.value)
    assert "verity/remote/v0" not in K.COMPOSITES and "verity/remote/v0" not in SP.SUBPROTOCOLS


def verifier_says(line: dict) -> R.Registry:
    def live(cfg, state, here):
        out = {p: {"outcome": {}, "refused": None} for p in here}
        out[R.VERIFIER] = {"outcome": {"verdicts": [line]}, "refused": None}
        return SimpleNamespace(outcomes=out)

    return dataclasses.replace(registry(), live=live)


def accepting(cfg: dict, product: str) -> dict:
    return {"block": 0, "components": dict.fromkeys(cfg["protocols"], "accept"), "product": product, "reason": None,
            "registration": None, "coins": "simulated", "verdict": "accept"}


@pytest.mark.parametrize("cid", sorted(K.UNACCEPTED))
def test_onsites_smaller_sets_are_read_and_hashed_and_their_verdict_is_undecided_under_their_own_id(tmp_path, capsys,
                                                                                                    cid):
    """Each of onsite's two smaller sets is a deployment with an id of its own, never onsite's: `check` reads it, and
    `verity run` runs its parties and records each period's verdict under that id, undecided, since no composite
    takes it, even where every subprotocol's verifier accepts."""
    cfg = config(K.UNACCEPTED[cid])
    assert K.composite(cfg["protocols"], REG.components) == cid and K.digest(cfg) != K.digest(config())
    rc, lines = run(tmp_path, verifier_says(accepting(cfg, cid)), cfg=cfg)
    assert rc == 2 and [(x["product"], x["verdict"]) for x in lines] == [(cid, "undecided")]
    assert lines[0]["components"] == dict.fromkeys(cfg["protocols"], "accept") and cid in lines[0]["reason"]
    assert said(capsys.readouterr().out, "VERDICT") == lines


@pytest.mark.parametrize("cid", sorted(K.UNACCEPTED))
def test_a_smaller_sets_record_never_carries_onsites_id(tmp_path, capsys, cid):
    cfg = config(K.UNACCEPTED[cid])
    rc, lines = run(tmp_path, verifier_says(accepting(cfg, "verity/onsite/v0")), cfg=cfg)
    out = capsys.readouterr().out
    assert rc == 2 and [(x["product"], x["verdict"]) for x in lines] == [(cid, "undecided")]
    assert "verity/onsite/v0" not in (tmp_path / "state" / "record" / "verdicts.jsonl").read_text()
    assert "verity/onsite/v0" not in out


@pytest.mark.parametrize("verdict, rc", [("accept", 0), ("reject", 1)])
@pytest.mark.parametrize("cid", sorted(K.UNACCEPTED))
def test_a_smaller_sets_verdict_the_table_combined_is_passed_through_with_who_combined_it(tmp_path, capsys, cid,
                                                                                          verdict, rc):
    cfg = config(K.UNACCEPTED[cid])
    by = f"python: no Lean verifier takes {cid}"
    line = {**accepting(cfg, "verity/onsite/v0"), "verdict": verdict, "combined_by": by}
    got, lines = run(tmp_path, verifier_says(line), cfg=cfg)
    captured = capsys.readouterr()
    assert got == rc and [(x["product"], x["verdict"], x["combined_by"]) for x in lines] == [(cid, verdict, by)]
    assert f"verity run: block 0's verdict {verdict} is combined by {by}; no composite takes {cid}" in captured.err
    assert said(captured.out, "VERDICT") == lines and "verity/onsite/v0" not in captured.out


def test_verity_run_refuses_a_configuration_that_is_no_composite(tmp_path, capsys):
    cfg = config(REMOTE)
    cfg["protocols"] = ["sampled-proofs"]
    rc, lines = run(tmp_path, registry(), cfg=cfg)
    assert rc == 2 and lines == [] and "refused composite:" in capsys.readouterr().err


def test_each_period_has_its_verdict(tmp_path):
    rc, lines = run(tmp_path, registry({("sampled-proofs", 1): "reject"}), "--periods", "2")
    assert rc == 1 and [(x["block"], x["verdict"]) for x in lines] == [(0, "accept"), (1, "reject")]


def test_an_audit_that_is_unavailable_is_its_periods_error(tmp_path):
    rc, lines = run(tmp_path, registry({("sampled-proofs", 0): "unavailable"}))
    assert rc == 1 and lines[0]["components"]["sampled-proofs"] == "error" and lines[0]["verdict"] == "reject"


def test_an_entry_that_fails_stops_the_run(tmp_path, capsys):
    rc, lines = run(tmp_path, registry(certifier_rc=3))
    assert rc == 2 and lines == [] and said(capsys.readouterr().out) == [{"party": "network-certifier", "exit": 3}]


def test_workers_that_cant_serve_leave_no_verdict(tmp_path, capsys):
    rc, lines = run(tmp_path, registry({("workers", 0): "unavailable"}))
    parties = {x["party"]: x.get("refused") for x in said(capsys.readouterr().out)}
    assert rc == 2 and lines == [] and parties["workers"] == "serve" and parties["verifier"] == "closed"


def test_a_party_placed_elsewhere_is_refused(tmp_path, capsys):
    rc, _ = run(tmp_path, registry(), "--party", "challenger", "--on", "auditor-1")
    assert rc == 2 and "refused machine" in capsys.readouterr().err


def test_parties_apart_meet_only_at_a_configured_listen_port(tmp_path, capsys):
    auditor = config(challenger="auditor-1", network_certifier="auditor-1", verifier="auditor-1")
    rc, _ = run(tmp_path, registry(), "--on", "auditor-1", cfg=auditor)
    captured = capsys.readouterr()
    assert rc == 2 and said(captured.out) == [{"party": "network-certifier", "exit": 0}]
    assert "refused listen: the verifier dials firewall, workers" in captured.err
    prover = tmp_path / "prover"
    prover.mkdir()
    rc, _ = run(prover, registry(), cfg=auditor)
    assert rc == 2 and "refused listen: firewall, workers listen for the verifier" in capsys.readouterr().err
    party(auditor, "firewall").update(address="10.0.0.5", listen=7001)
    party(auditor, "workers").update(listen=7002)
    assert D.listening(K.check(auditor, REG.components, REG.workloads)) == {"firewall": "10.0.0.5:7001",
                                                                           "workers": "127.0.0.1:7002"}


AUDITOR = {"challenger": "auditor-1", "network_certifier": "auditor-1", "verifier": "auditor-1"}


def entry_seeing_local(seen: list) -> R.Registry:
    reg = registry()
    attester = reg.parties["certifier"]["network-certifier"]

    def entry(cfg, state, local):
        seen.append(local)
        return attester(cfg, state, local)

    return dataclasses.replace(reg, parties={**reg.parties, "certifier": {**reg.parties["certifier"],
                                                                         "network-certifier": entry}})


def test_each_on_names_a_machine_this_host_stands_for_and_every_party_there_runs_as_one_host(tmp_path, capsys):
    seen: list = []
    rc, lines = run(tmp_path, entry_seeing_local(seen), "--on", "auditor-1", "--on", "local", cfg=config(**AUDITOR))
    assert rc == 0 and [x["verdict"] for x in lines] == ["accept"] and seen == [True]
    assert {x["party"] for x in said(capsys.readouterr().out)} == {
        "network-certifier", "challenger", "firewall", "verifier", "workers"}
    assert "challenger shares this host (auditor-1, local) with the developer's firewall, workers: Faithful and " \
           "Secret unmet" in lines[0]["simulated"]
    assert json.loads((tmp_path / "state" / "run.json").read_text())["machines"] == ["auditor-1", "local"]


def test_unmet_takes_the_machines_this_host_stands_for_as_one_host():
    cfg = config(**AUDITOR)
    assert R.unmet(cfg, REG, COMMIT, ["auditor-1"]) == []
    assert R.unmet(cfg, REG, COMMIT, ["auditor-1", "local"]) == [
        "challenger shares this host (auditor-1, local) with the developer's firewall, workers: Faithful and Secret unmet"]
    assert R.unmet(config(), REG, COMMIT) == [
        "challenger shares local with the developer's firewall, workers: Faithful and Secret unmet"]


@pytest.mark.parametrize("machines, coins", [(["auditor-1"], "development"), (["auditor-1", "local"], "simulated")])
def test_the_record_says_simulated_coins_when_the_challenger_shares_the_provers_host(tmp_path, machines, coins):
    cfg = config(**AUDITOR)
    line = {**accepting(cfg, "verity/onsite/v0"), "coins": "development"}
    rc, lines = run(tmp_path, verifier_says(line), *(x for m in machines for x in ("--on", m)), cfg=cfg)
    assert rc == 0 and [x["coins"] for x in lines] == [coins]


@pytest.mark.parametrize("machines, local", [(["auditor-1"], False), (["auditor-1", "local"], True)])
def test_a_party_process_runs_as_one_host_with_the_machines_it_is_given(tmp_path, monkeypatch, machines, local):
    seen: list = []
    reg = dataclasses.replace(REG, parties={**REG.parties, "certifier": {
        **REG.parties["certifier"], "network-certifier": lambda cfg, state, local: seen.append(local) or 0}})
    monkeypatch.setattr(R, "registry", lambda *a: reg)
    path, state = tmp_path / "config.toml", tmp_path / "state"
    path.write_text(K.render(config(**AUDITOR)))
    state.mkdir()
    assert R.party_main([str(path), str(state), "network-certifier", *machines]) == 0 and seen == [local]


def test_the_processes_runner_hands_each_party_every_machine_and_the_runs_commit_and_tree(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(R.subprocess, "run", lambda argv, **kw: calls.append((argv, kw["env"])) or
                        SimpleNamespace(returncode=0))
    (tmp_path / "run.json").write_text(json.dumps({"program": COMMIT, "tree": "tree.tar"}))
    assert R.Processes(tmp_path / "config.toml", tmp_path, ["auditor-1", "local"]).run("challenger") == 0
    (argv, env), = calls
    assert argv[-6:] == ["party", str((tmp_path / "config.toml").resolve()), str(tmp_path), "challenger",
                         "auditor-1", "local"]
    assert (env["VERITY_PROGRAM"], env["VERITY_TREE"]) == (COMMIT, str(tmp_path / "tree.tar"))


def lean_package(root: Path) -> Path:
    pkg = root / V.VERIFIER
    pkg.mkdir(parents=True)
    (pkg / "lakefile.toml").write_text(f'[[lean_exe]]\nname = "flock-verify"\nroot = "{V.VERIFY_PROOF_ROOT}"\n')
    return pkg


def test_a_missing_verifier_is_built_with_setup_then_lake_build_and_a_built_one_is_not(tmp_path):
    pkg, calls = lean_package(tmp_path), []

    def fake(argv, cwd):
        calls.append((argv, cwd))
        if argv[1] == "build":
            V.verify_proof_path(tmp_path).parent.mkdir(parents=True)
            V.verify_proof_path(tmp_path).write_text("")
        return SimpleNamespace(returncode=0)

    BUILD_VERIFIER(tmp_path, fake)
    assert [(a[0].rsplit("/", 1)[-1], *a[1:]) for a, _ in calls] == [
        ("bash", "tools/verity/lean/setup.sh", V.VERIFIER), ("lake", "build", "flock-verify")]
    assert [c for _, c in calls] == [tmp_path, pkg]
    BUILD_VERIFIER(tmp_path, fake)
    assert len(calls) == 2


@pytest.mark.parametrize("fails, ran", [("setup.sh", 1), ("build", 2), (None, 2)])
def test_a_verifier_that_doesnt_build_is_refused_by_name(tmp_path, fails, ran):
    lean_package(tmp_path)
    calls: list = []

    def fake(argv, cwd):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if fails and fails in argv[1] else 0)

    with pytest.raises(Refused) as e:
        BUILD_VERIFIER(tmp_path, fake)
    assert e.value.name == "verifier" and len(calls) == ran


def test_verity_run_builds_the_verifier_before_any_party(tmp_path, monkeypatch):
    order: list = []
    monkeypatch.setattr(R, "build_verifier", lambda: order.append("build"))
    assert run(tmp_path, entry_seeing_local(order))[0] == 0 and order == ["build", True]


def test_verity_run_stops_before_any_party_when_the_verifier_doesnt_build(tmp_path, capsys, monkeypatch):
    def unbuilt():
        raise Refused("verifier", "lake build exited 1")

    monkeypatch.setattr(R, "build_verifier", unbuilt)
    rc, lines = run(tmp_path, registry())
    captured = capsys.readouterr()
    assert rc == 2 and lines == [] and said(captured.out) == []
    assert "refused verifier: lake build exited 1" in captured.err


@pytest.mark.parametrize("named", [False, True])
def test_an_application_runs_the_trees_commit_and_a_deployments_own_file_names_a_mismatch_unmet(tmp_path, capsys,
                                                                                                monkeypatch, named):
    ran = "b" * 40
    monkeypatch.setattr(R, "head", lambda: ran)
    monkeypatch.setattr(R, "APPLICATIONS", tmp_path)
    rc, lines = run(tmp_path, registry(), named=named)
    assert R.config_main(["hash", str(tmp_path / "config.toml")], REG) == 0
    file_hash = capsys.readouterr().out.splitlines()[-1]
    assert rc == 0 and [(x["configuration"], x["program"]) for x in lines] == [(file_hash, ran)]
    assert (f"the program is the running tree's {ran}, not the configured {COMMIT}" in lines[0]["simulated"]) is not named
    assert json.loads((tmp_path / "state" / "run.json").read_text())["program"] == ran


def test_a_dry_run_builds_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "build_verifier", lambda: pytest.fail("a dry run built the verifier"))
    assert run(tmp_path, registry(), "--dry-run")[0] == 0


def test_a_subprotocol_with_no_parties_yet_is_refused(tmp_path, capsys):
    reg = registry()
    reg = dataclasses.replace(reg, parties={c: ps for c, ps in reg.parties.items() if c != "wipe"})
    rc, _ = run(tmp_path, reg)
    assert rc == 2 and "refused no-party: wipe" in capsys.readouterr().err


def test_a_state_of_another_configuration_is_refused(tmp_path, capsys):
    assert run(tmp_path, registry())[0] == 0
    rc, _ = run(tmp_path, registry(), "--periods", "2")
    assert rc == 2 and "refused state" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["onsite_toy", "onsite-toy"])
def test_verity_run_name_reads_the_applications_file_of_that_name(tmp_path, capsys, monkeypatch, name):
    monkeypatch.setattr(R, "APPLICATIONS", tmp_path)
    (tmp_path / "onsite_toy.toml").write_text(K.render(config()))
    assert R.run_main([name, "--dry-run"], registry()) == 0
    assert said(capsys.readouterr().out, "DRY")[0]["configuration"] == K.digest(R.load(tmp_path / "onsite_toy.toml", REG))


def test_a_name_that_is_no_application_is_refused_with_the_applications_names(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "APPLICATIONS", tmp_path)
    for name in ("remote", "onsite_toy"):
        (tmp_path / f"{name}.toml").write_text(K.render(config()))
    assert R.run_main(["nosuch", "--dry-run"], registry()) == 2
    assert capsys.readouterr().err == ("verity run: refused application: nosuch is neither a configuration's path nor an "
                                       "application's name: NAME is one of onsite_toy, remote\n")


def test_verity_run_with_no_table_is_refused_by_name(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(R, "REGISTRY", None)
    path = tmp_path / "config.toml"
    path.write_text(K.render(config()))
    assert R.run_main([str(path), "--state", str(tmp_path / "state")]) == 2
    assert "refused no-table" in capsys.readouterr().err
    assert R.config_main(["check", str(path)]) == 2
    assert "refused no-table" in capsys.readouterr().out


def test_the_certifier_and_the_policy_are_declared_from_their_manifests():
    certifier, policy = REG.components["certifier"], REG.components["inference-only"]
    assert (policy.id, policy.parties, policy.channels) == (IO.ID, IO.MANIFEST.parties, IO.MANIFEST.channels)
    assert (certifier.id, certifier.channels) == (NC.ID, NC.MANIFEST.channels)
    assert [r for r in NC.MANIFEST.parties if r not in certifier.parties] == [NC.MANIFEST.role(NC.WORKERS)]
    assert all(r in NC.MANIFEST.parties for r in certifier.parties)
    assert set(D.PARTIES["certifier"]) == {r.name for r in certifier.parties}
    assert [r.name for r in D.TOY.parties] == [NC.WORKERS]
    assert NC.WORKERS not in {p for ps in D.PARTIES.values() for p in ps}


def test_the_table_holds_what_the_listed_components_need():
    assert REG.components["certifier"].needs <= REG.workloads["toy"].provides
    assert os.path.basename(D.__file__) == "step0.py"
    live = {p for ps in D.PARTIES.values() for p, e in ps.items() if e == D.LIVE} | {
        r.name for w, (decl, e) in D.WORKLOADS.items() if e == D.LIVE for r in decl.parties}
    assert live == set(D.ROLES)
    decided_live = {c for c, ps in D.PARTIES.items() if ps.get("verifier") == D.LIVE}
    assert decided_live - {"sampled-proofs"} == set(D.CHECKS)
