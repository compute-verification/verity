"""Remote's configuration (`demos/remote.toml`) under the table (`served_table.py`): `verity run --dry-run` resolves
every party, Pearl+'s included, and without bindings names the three Pearl+ lacks; the runner's variables under remote
are Pearl+'s, onsite's carry none of them; Pearl+'s class is refused by name when it disagrees with the workload's slots
or window-steps, and its uniform tie is asked for only by its parameter. Under remote alone, the runner's exit 2 with
the compose hand-off written is pending, and the verifier reads the compose answer's record (Pearl+'s row under
``work``, inference-only's as the answer holds it), undecided without one. The runner and the records are stand-ins
written by each test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verity import config as K
from verity.protocols.proofs.spot.verifier import parties as SP
from verity_cli import run as R

ROOT = Path(R.ROOT)
CONFIG = ROOT / "demos" / "remote.toml"
ONSITE = ROOT / "demos" / "onsite_full.toml"
REG = R.registry()
T = sys.modules["served_table"]
PP = SP.IDS["pearl-plus"]
KEY = "0" * 128
BINDINGS = {"challenger": {"pearl-plus": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}}}
KEYS = ["challenger.pearl-plus.gate", "challenger.pearl-plus.peer", "challenger.pearl-plus.key-sha512"]
PEARLPLUS = {"POUW": "pearlplus", "PEARLPLUS_TIE": "1", "PRODUCT": "remote", "PEARLC_CHALLENGER_GATE": "gate-command",
             "PEARLC_CHALLENGER_PEER": "peer-1", "H1_KEY_SHA512": KEY}
OPTIONAL = {"PEARLC_CHALLENGER_LANE", "MAIN_FILE", "V3C_CLASS", "CLASSES", "PEARLPLUS_TIE_UNIFORM"}


def toml(tables: dict) -> str:
    return "".join(f"[{p}.{c}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in ks.items()) + "\n"
                   for p, cs in tables.items() for c, ks in cs.items())


def bindings(tmp: Path, tables: dict = BINDINGS) -> str:
    f = tmp / "bindings.toml"
    f.write_text(toml(tables))
    return str(f)


def said(out: str, what: str) -> list[dict]:
    return [json.loads(x.split(" ", 1)[1]) for x in out.splitlines() if x.startswith(what + " ")]


def dry(capsys, *args: str, config: Path = CONFIG) -> tuple[int, dict, str]:
    rc = R.run_main([str(config), "--dry-run", *args], REG)
    out = capsys.readouterr()
    return rc, {"plan": said(out.out, "PLAN"), "dry": said(out.out, "DRY")}, out.err


def hooks_of(got: dict, component: str = "pearl-plus") -> dict:
    return next(e["hooks"] for x in got["plan"] for e in x["entries"] if e["component"] == component
                and x["party"] == "workers")


def remote(tmp: Path, params: dict, **workload) -> Path:
    cfg = K.parse(CONFIG.read_text())
    cfg["params"] = {PP: params}
    cfg["workload"]["params"] |= workload
    f = tmp / "remote.toml"
    f.write_text(K.render(cfg))
    return f


def test_a_dry_run_resolves_every_party_of_remote(tmp_path, capsys):
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path))
    cfg = R.load(CONFIG, REG)
    assert rc == 0 and "no-party" not in err, err
    assert sorted(x["party"] for x in got["plan"]) == sorted(p["name"] for p in cfg["party"])
    assert got["dry"][0]["unbound"] == []
    pieces = {(x["party"], e["component"], e["runs"]) for x in got["plan"] for e in x["entries"]}
    assert {(p, r) for p, c, r in pieces if c == "pearl-plus"} == {("challenger", T.GATE),
                                                                   ("workers", T.SERVED_WINDOW), ("verifier", T.CHECK)}
    assert {"from": "challenger", "to": "workers", "messages": ["PearlSalt"], "after": []} in cfg["channel"]


def test_a_dry_run_without_bindings_names_the_three_pearl_plus_lacks(capsys):
    rc, got, _ = dry(capsys)
    assert rc == 0 and got["dry"][0]["unbound"] == sorted(KEYS)
    assert T.unbound(R.load(CONFIG, REG), {}) == KEYS
    assert hooks_of(got)["PEARLC_CHALLENGER_GATE"] == "<challenger.pearl-plus.gate>"


def test_the_runners_variables_under_remote_are_pearl_pluss_and_onsites_carry_none():
    cfg = R.load(CONFIG, REG)
    serve = T.SERVE.variables(cfg, BINDINGS)
    assert {k: serve[k] for k in ("REPO", "PRECISION", "SLOTS", "WINDOW_STEPS", "LOAD")} == {
        "REPO": "verity-int8/Llama-3.2-1B-INT8", "PRECISION": "verity_int8", "SLOTS": "4", "WINDOW_STEPS": "16",
        "LOAD": "varied"}
    variables = serve | {"DEVELOPMENT": "1"} | PEARLPLUS | {"VERITY_CONFIGURATION": K.digest(cfg), "VERITY_PINS": "[]"}
    assert T.variables(cfg, BINDINGS) == variables
    both = {"challenger": {"pearl-plus": {**BINDINGS["challenger"]["pearl-plus"], "lane": "vllm-2",
                                          "main-file": "/tmp/main.json"}}}
    assert T.variables(cfg, both) == variables | {"PEARLC_CHALLENGER_LANE": "vllm-2", "MAIN_FILE": "/tmp/main.json"}
    onsite = T.variables(R.load(ONSITE, REG), {})
    assert not (set(PEARLPLUS) - {"H1_KEY_SHA512"} | {"PEARLC_CHALLENGER_LANE", "MAIN_FILE", "PEARLPLUS_TIE_UNIFORM"}) \
        & set(onsite)
    assert onsite["V3C_CLASS"] == T.SCAN_CLASS and onsite["H1_KEY_SHA512"] == "<challenger.pouw.key-sha512>"


def test_pearl_pluss_class_is_the_runners_and_refused_by_name_where_the_workload_disagrees(tmp_path, capsys):
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path), config=remote(tmp_path, {"class": "4,27,2"},
                                                                                   **{"window-steps": 2}))
    assert rc == 0, err
    assert {k: hooks_of(got).get(k) for k in ("V3C_CLASS", "CLASSES")} == {"V3C_CLASS": "4,27,2", "CLASSES": "4,27,2"}
    for params, workload in (({"class": "4,27,2"}, {}), ({"class": "5,27,2"}, {"window-steps": 2})):
        rc, _, err = dry(capsys, config=remote(tmp_path, params, **workload))
        assert rc == 2 and "refused params: Pearl+'s class" in err, err
    for bad in ("4,27", "4,0,2", "a,b,c", 4):
        rc, _, err = dry(capsys, config=remote(tmp_path, {"class": bad}))
        assert rc == 2 and "refused params: class is SLOTS,PROMPT,OUTPUTS" in err, err
    rc, _, err = dry(capsys, config=remote(tmp_path, {"classes": "4,27,2"}))
    assert rc == 2 and "refused unknown-key" in err and "known: class, tie-uniform" in err


def test_the_uniform_tie_is_set_only_when_asked(tmp_path, capsys):
    assert "PEARLPLUS_TIE_UNIFORM" not in T.variables(R.load(CONFIG, REG), BINDINGS)
    assert R.load(CONFIG, REG)["params"][PP] == {"tie-uniform": False}
    for asked, want in ((True, "1"), (False, None)):
        rc, got, err = dry(capsys, "--bindings", bindings(tmp_path), config=remote(tmp_path, {"tie-uniform": asked}))
        assert rc == 0 and hooks_of(got).get("PEARLPLUS_TIE_UNIFORM") == want, err
    rc, _, err = dry(capsys, config=remote(tmp_path, {"tie-uniform": 1}))
    assert rc == 2 and "refused params: tie-uniform is true or false" in err


#: coreutils' env by path: agent VMs put a redacting `env` ahead of it on PATH (`tools/agent-guard/env_guard.py`)
RUNNER = """mkdir -p "$OUT"
/usr/bin/env | grep -E '^(POUW|PRODUCT|PEARLPLUS_[A-Z_]+|PEARLC_[A-Z_]+|H1_KEY_SHA512|MAIN_FILE|V3C_CLASS|CLASSES|SLOTS|WINDOW_STEPS)=' \
  | sort > "$OUT/hooks.txt"
"""


def test_the_workload_runs_once_with_pearl_pluss_hooks(tmp_path, capsys, monkeypatch):
    for k in OPTIONAL:
        monkeypatch.delenv(k, raising=False)
    runner = tmp_path / "runner.sh"
    runner.write_text(RUNNER)
    monkeypatch.setattr(T, "RUNNER", str(runner))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state", str(state)], REG)
    assert rc == 0, capsys.readouterr()
    here = sorted(p["name"] for p in R.load(CONFIG, REG)["party"] if p["machine"] == T.PROVER_NODE)
    assert said(capsys.readouterr().out, "PARTY") == [{"party": p, "refused": None} for p in here] and "workers" in here
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks == PEARLPLUS | {"SLOTS": "4", "WINDOW_STEPS": "16"}


HANDED = "HANDOFF the final record, composed off this node: on COMPOSE_NODE (auditor-2)"
ONSITE_BINDINGS = {
    "challenger": {"pouw": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}},
    "memory-challenger": {"pous": {"address": "192.0.2.1:7000"}, "wipe": {"address": "192.0.2.1:7001"}},
    "bandwidth-verifier": {"bandwidth": {"address": "192.0.2.2:7002"}},
    "network-certifier": {"certifier": {"gate": "user@gate-1"}},
}


def ends(tmp: Path, monkeypatch, rc: int, *lines: str) -> None:
    runner = tmp / "runner.sh"
    runner.write_text('mkdir -p "$OUT"\n' + "".join(f"echo {json.dumps(x)} >> \"$OUT/verdicts.txt\"\n" for x in lines)
                      + f"exit {rc}\n")
    monkeypatch.setattr(T, "RUNNER", str(runner))


def parties(capsys) -> dict:
    return {x["party"]: x["refused"] for x in said(capsys.readouterr().out, "PARTY")}


def test_the_runners_exit_2_with_the_hand_off_written_is_pending_under_remote(tmp_path, capsys, monkeypatch):
    ends(tmp_path, monkeypatch, 2, "RECORD {}", HANDED)
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state",
                     str(tmp_path / "state")], REG)
    assert rc == 0 and set(parties(capsys).values()) == {None}
    assert T.pending(R.load(CONFIG, REG), tmp_path / "state", 2) == HANDED


def test_any_other_end_of_the_runner_stays_refused_under_remote(tmp_path, capsys, monkeypatch):
    for n, (rc, lines) in enumerate(((2, ()), (2, ("HANDOFF refused: no engine steps",)), (1, (HANDED,)),
                                     (3, (HANDED,)))):
        ends(tmp_path, monkeypatch, rc, *lines)
        got = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--bindings", bindings(tmp_path), "--state",
                          str(tmp_path / f"state-{n}")], REG)
        assert got == 2 and set(parties(capsys).values()) == {T.SERVED_WINDOW}, (rc, lines)


def test_onsites_exit_2_stays_refused_with_the_hand_off_written(tmp_path, capsys, monkeypatch):
    ends(tmp_path, monkeypatch, 2, HANDED)
    rc = R.run_main([str(ONSITE), "--on", T.PROVER_NODE, "--party", "workers", "--party", "firewall", "--bindings",
                     bindings(tmp_path, ONSITE_BINDINGS), "--state", str(tmp_path / "state")], REG)
    assert rc == 2 and parties(capsys) == {"firewall": T.SERVED_WINDOW, "workers": T.SERVED_WINDOW}
    assert T.pending(R.load(ONSITE, REG), tmp_path / "state", 2) is None


def written(base: Path, line: dict | None, rows: dict | None, inference_only: dict | None = None) -> None:
    (base / "record").mkdir(parents=True, exist_ok=True)
    if line is not None:
        (base / "verdicts.txt").write_text(f"SPOT ...\nRECORD {json.dumps(line)}\n")
    if rows is not None:
        (base / "record" / "verdicts.json").write_text(json.dumps({"window": 3, "protocols": rows}))
    if inference_only is not None:
        (base / "record" / "inference-only.json").write_text(json.dumps(inference_only))


ACCEPTED = {"integrity": {"verdict": "accept"}, "work": {"verdict": "accept"}}


def test_remotes_verifier_reads_the_compose_answers_record_pearl_plus_under_work(tmp_path):
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out", {"composed": "pending", "product": "remote"}, {"work": {"verdict": "missing"}})
    written(tmp_path / "out" / "compose", {"composed": "accept", "product": "remote"}, ACCEPTED, {"verdict": "accept"})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["product"], got["block"], got["reason"]) == ("accept", T.REMOTE, 3, None)
    assert got["components"] == {"sampled-proofs": "accept", "pearl-plus": "accept", "inference-only": "accept"}
    written(tmp_path / "out" / "compose", {"composed": "reject", "product": "remote"},
            {**ACCEPTED, "work": {"verdict": "reject"}})
    assert T.remote_check(cfg, tmp_path)["verdict"] == "reject"


def test_inference_onlys_row_is_what_the_compose_answer_holds_and_absent_without_it(tmp_path):
    cfg = R.load(CONFIG, REG)
    base = tmp_path / "out" / "compose"
    written(base, {"composed": "accept", "product": "remote"}, ACCEPTED, {"verdict": "refuse"})
    got = T.remote_check(cfg, tmp_path)
    assert got["components"]["inference-only"] == "refuse" and got["verdict"] == "undecided"
    assert got["reason"] == "not accepted: inference-only refuse"
    (base / "record" / "inference-only.json").unlink()
    got = T.remote_check(cfg, tmp_path)
    assert got["components"]["inference-only"] == "absent" and got["verdict"] == "undecided"
    written(base, None, {**ACCEPTED, "inference-only": {"verdict": "accept"}}, {"verdict": "refuse"})
    assert T.remote_check(cfg, tmp_path)["components"]["inference-only"] == "accept"


def test_without_a_compose_answer_remotes_verdict_is_undecided_and_says_why(tmp_path):
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out", {"composed": "accept", "product": "remote"}, ACCEPTED, {"verdict": "accept"})
    got = T.remote_check(cfg, tmp_path)
    assert got["verdict"] == "undecided" and got["reason"].startswith(f"no compose answer in {tmp_path / 'out' / 'compose'}")
    assert got["components"] == {"sampled-proofs": "accept", "pearl-plus": "accept", "inference-only": "accept"}


def test_remotes_verifier_refuses_by_name_a_record_it_cannot_read_or_of_another_product(tmp_path):
    cfg = R.load(CONFIG, REG)
    for line, rows, why in ((None, None, "no RECORD line"), ({"composed": "accept", "product": "remote"}, None,
                                                            "no record/verdicts.json"),
                            ({"composed": "accept", "product": "onsite"}, ACCEPTED, "composed for 'onsite'")):
        written(tmp_path / "out", line, rows)
        try:
            T.remote_check(cfg, tmp_path)
        except T.Refused as e:
            assert e.name == "record" and why in e.detail
        else:
            raise AssertionError(why)


def test_every_declared_component_has_its_records_row():
    assert set(T.ROWS) == set(T.MANIFESTS) and T.ROWS["pearl-plus"] == T.ROWS["pouw"] == "work"


def test_remotes_verifier_gives_its_verdict_line_on_its_own_machine(tmp_path, capsys):
    state = tmp_path / "state"
    written(state / "out" / "compose", {"composed": "accept", "product": "remote"}, ACCEPTED, {"verdict": "accept"})
    rc = R.run_main([str(CONFIG), "--on", T.AUDITOR_REMOTE, "--state", str(state)], REG)
    out = capsys.readouterr()
    lines = said(out.out, "VERDICT")
    assert len(lines) == 1, out
    assert (rc, lines[0]["product"], lines[0]["verdict"], lines[0]["block"]) == (0, T.REMOTE, "accept", 3)
