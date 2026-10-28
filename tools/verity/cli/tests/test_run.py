"""`verity config` and `verity run` (`verity_cli.run`, `verity.core.deploy.config`): the configuration's refusals by name,
a list of subprotocols that is neither composite's among them, its hash, the example reproduced by `config new`, and
runs of a step-0 table (`step0.py`) for onsite and remote: its entries in this process, and core's sampled-proofs
parties in the deployment layer's live run, on a machine with stand-in capabilities and a stand-in product executable,
every party on in-memory channels, so no build, GPU, network or clock is involved."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
from pathlib import Path

import pytest

from verity.core.deploy import config as K
from verity.core.deploy.manifest import Channel, Role
from verity.core.protocols.sampled_proofs import parties as SP
from verity.core.protocols.sampled_proofs.one_stage import registration as RG
from verity.core.wire import Refused
from verity_cli import run as R

EXAMPLE = Path(__file__).parent / "fixtures" / "config.toml"
TABLE = (Path(__file__).parent, "step0")
REG = R.registry(TABLE)
D = sys.modules[TABLE[1]]
COMMIT = "a" * 40
ONSITE = ("sampled-proofs", "certifier", "pouw", "pous", "bandwidth")
REMOTE = ("sampled-proofs", "pouw")


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
    header = [h.format(protocols="sampled-proofs and pouw", workload="toy", machine="local") for h in R.HEADER]
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
    "unknown-params": lambda c: c.update(params={"verity/wipe/v0": {}}),
    "tie": lambda c: c.update(params={"verity/certifier/v0": {"tie": False}}),
    "law": lambda c: c["service"].update(law="subset:8"),
    "commit": lambda c: c["program"].update(commit="main"),
    "float": lambda c: c["workload"].update(params={"steps": 2.0}),
    "ascii": lambda c: c["workload"].update(name="caf\u00e9"),
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
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_configuration_is_refused_by_name(name):
    assert refused(edit(REFUSALS[name])) == name


@pytest.mark.parametrize("protocols", [("sampled-proofs",), ("sampled-proofs", "certifier"), ("pouw",), ONSITE[:-1],
                                       ("sampled-proofs", "certifier", "pouw"), ("sampled-proofs", "wipe")])
def test_a_set_of_subprotocols_that_is_neither_composite_is_refused(protocols):
    assert refused(edit(lambda c: c.update(protocols=list(protocols)))) == "composite"
    with pytest.raises(K.Refused) as e:
        K.new("toy", list(protocols), REG.components, REG.workloads, COMMIT)
    assert e.value.name in ("composite", "requires")


def test_the_composites_are_remote_and_onsite():
    assert {K.composite(p, REG.components) for p in (ONSITE, REMOTE, (*REMOTE, "wipe"))} == {
        "verity/onsite/v0", "verity/remote/v0"}
    assert K.composite(ONSITE[::-1], REG.components) == "verity/onsite/v0"
    assert {cid: sorted(names) for cid, names in K.COMPOSITES.items()} == {
        cid: sorted(subs) for cid, subs in SP.SUBPROTOCOLS.items()}
    assert SP.SUBPROTOCOLS["verity/remote/v0"] == {"pouw": "verity/pouw/v1", "sampled-proofs": SP.ID}
    assert SP.SUBPROTOCOLS["verity/onsite/v0"]["pouw"] == "verity/pouw/period/v1"


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


def test_the_canonical_form_is_verity_canonical_json():
    from verity.ml.circuits.codec import canonical_json

    cfg = K.check(config(), REG.components, REG.workloads)
    assert K.canonical(cfg) == canonical_json(cfg)
    for v in ({"s": "".join(map(chr, range(0x20, 0x7f))), "c": "\t\n\x00\x1f", "n": [-1, 1 << 63]}, ["x", True, []]):
        assert K.canonical(v) == canonical_json(v)
    for v in ({"s": "d\u00e9"}, {"k\u00e9y": 1}, ["x\x7f"], {"s": "\U0001f600"}):
        with pytest.raises(K.Refused, match="is not ASCII without DEL"):
            K.canonical(v)
        with pytest.raises(ValueError, match="is not ASCII without DEL"):
            canonical_json(v)


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

    def audit(self, i, registration, challenger, firewall):
        v = self._of("sampled-proofs", i)
        answer = {"verdict": v, "commitment": RG.digest(json.loads(registration))}
        write(self.state / "record" / f"w{i}" / "sampled-proofs" / "answer.json", answer)
        return answer, case(v)

    def check(self, who, cfg, state, i):
        v = self._of(who, i)
        return None if v == "silent" else ({"verdict": v, "commitment": "00"}, case(v))


def registry(verdicts: dict | None = None, certifier_rc: int = 0) -> R.Registry:
    def network_certifier(cfg, state, local):
        write(state / "blocks" / "certifier-session" / "relay.json", {"periods": len(periods(state))})
        return certifier_rc

    parties = {**REG.parties, "certifier": {**REG.parties["certifier"], "network-certifier": network_certifier}}
    return R.Registry(REG.components, REG.workloads, REG.workers, parties, REG.order,
                      functools.partial(D.live, machine=lambda c, s: StandIn(c, s, verdicts or {})))


def run(tmp: Path, reg: R.Registry, *args: str, cfg: dict | None = None) -> tuple[int, list[dict]]:
    path = tmp / "config.toml"
    path.write_text(K.render(cfg or config()))
    s = tmp / "state"
    s.mkdir(exist_ok=True)
    if not (s / "tools.json").exists():
        (s / "tools.json").write_text(json.dumps({"product": str(stand_in_product(tmp))}))
    rc = R.run_main([str(path), "--state", str(s), *args], reg, in_process=True)
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
    assert rc == 0 and lines[0]["components"] == {"sampled-proofs": "accept", "pouw": "accept"}
    assert lines[0]["product"] == "verity/remote/v0"
    assert "network-certifier" not in capsys.readouterr().out


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


def test_a_subprotocol_with_no_parties_yet_is_refused(tmp_path, capsys):
    rc, _ = run(tmp_path, registry(), cfg=config([*REMOTE, "wipe"]))
    assert rc == 2 and "refused no-party: wipe" in capsys.readouterr().err


def test_a_state_of_another_configuration_is_refused(tmp_path, capsys):
    assert run(tmp_path, registry())[0] == 0
    rc, _ = run(tmp_path, registry(), "--periods", "2")
    assert rc == 2 and "refused state" in capsys.readouterr().err


def test_verity_run_with_no_table_is_refused_by_name(tmp_path, capsys):
    path = tmp_path / "config.toml"
    path.write_text(K.render(config()))
    assert R.run_main([str(path), "--state", str(tmp_path / "state")]) == 2
    assert "refused no-table" in capsys.readouterr().err
    assert R.config_main(["check", str(path)]) == 2
    assert "refused no-table" in capsys.readouterr().out


def test_the_table_holds_what_the_listed_components_need():
    assert REG.components["certifier"].needs <= REG.workloads["toy"].provides
    assert os.path.basename(D.__file__) == "step0.py"
    live = {p for ps in D.PARTIES.values() for p, e in ps.items() if e == D.LIVE} | {
        r.name for w, (decl, e) in D.WORKLOADS.items() if e == D.LIVE for r in decl.parties}
    assert live == set(D.ROLES)
    decided_live = {c for c, ps in D.PARTIES.items() if ps.get("verifier") == D.LIVE}
    assert decided_live - {"sampled-proofs"} == set(D.CHECKS)
