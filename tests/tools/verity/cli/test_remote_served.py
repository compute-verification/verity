"""Remote's configuration (`applications/remote.toml`) under the table (`served_table.py`): `verity run --dry-run` resolves
every party, Pearl+'s included, and without bindings Pearl+'s coins are an in-process challenger's on the workload's
machine, which the verifier's line says, with no key blocking and no placeholder reaching the runner; a part of Pearl+'s
live challenger is refused by name; under onsite, Pearl+ in H1's place is refused until onsite's composite lists it,
and then its product is onsite and no compute key blocks; the runner's variables under remote
are Pearl+'s, onsite's carry none of them; Pearl+'s class is refused by name when it disagrees with the workload's slots
or window-steps, and its uniform tie is asked for only by its parameter. Under remote alone, the runner's exit 2 with
the compose hand-off written is pending, and the verifier reads only the compose answer's record (Pearl+'s row under
``work``; inference-only's row holds the verdict while the served node wrote it), undecided with no cell read without
one. The runner and the records are stand-ins written by each test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity import config as K
from verity.protocols.proof.spot.verifier import parties as SP
from verity_cli import run as R
from verity_vllm.inputs import serves

ROOT = Path(R.ROOT)
CONFIG = ROOT / "applications" / "remote.toml"
S = serves.SERVES
DATASET = serves.application(str(CONFIG))[0]
WINDOW = {"SLOTS": str(S.slots), "WINDOW_STEPS": str(S.outputs), "V3C_CLASS": S.window_class, "CLASSES": S.window_class}
ONSITE = ROOT / "applications" / "onsite_full.toml"
REG = R.registry()
T = sys.modules["served_table"]
PP = SP.IDS["pearl-plus"]
KEY = "0" * 128
BINDINGS = {"challenger": {"pearl-plus": {"gate": "gate-command", "peer": "peer-1", "key-sha512": KEY}}}
#: every binding the challenger writes (`served_table.STARTS`), so a run that gives them doesn't start it
STARTED = {"challenger": {"pearl-plus": {**BINDINGS["challenger"]["pearl-plus"], "lane": "lane-1", "main-file": "main.txt"},
                          "sampled-proofs": {"loopback": "loopback-host", "main-file": "main.txt"}}}
STARTED_HOOKS = {"PEARLC_CHALLENGER_LANE": "lane-1", "MAIN_FILE": "main.txt"}
LOCAL = {"POUW": "pearlplus", "PRODUCT": "remote", "PEARLC_CHALLENGER_GATE": "",
         "PEARLC_CHALLENGER_PEER": "", "DEVELOPMENT": "1"}
PEARLPLUS = {"POUW": "pearlplus", "PRODUCT": "remote", "PEARLC_CHALLENGER_GATE": "gate-command",
             "PEARLC_CHALLENGER_PEER": "peer-1", "H1_KEY_SHA512": KEY}
#: the runner's variables a run inherits unless the test clears them; Pearl+'s untied hooks set none of the last four
OPTIONAL = {"PEARLC_CHALLENGER_LANE", "MAIN_FILE", "V3C_CLASS", "CLASSES", "PEARLPLUS_TIE", "PEARLPLUS_TIE_UNIFORM"}


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
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path, STARTED))
    cfg = R.load(CONFIG, REG)
    assert rc == 0 and "no-party" not in err, err
    assert sorted(x["party"] for x in got["plan"]) == sorted(p["name"] for p in cfg["party"])
    assert got["dry"][0]["unbound"] == []
    pieces = {(x["party"], e["component"], e["runs"]) for x in got["plan"] for e in x["entries"]}
    assert {(p, r) for p, c, r in pieces if c == "pearl-plus"} == {("challenger", T.GATE),
                                                                   ("workers", T.SERVED_WINDOW), ("verifier", T.CHECK)}
    assert {"from": "challenger", "to": "workers", "messages": ["PearlSalt"], "after": []} in cfg["channel"]


def test_without_bindings_no_key_blocks_and_pearl_pluss_coins_are_the_workloads_machines(capsys):
    cfg = R.load(CONFIG, REG)
    for on in ((), ("--on", T.PROVER_NODE)):
        rc, got, err = dry(capsys, *on)
        assert rc == 0 and [x["unbound"] for x in got["dry"]] == [[]], err
        assert hooks_of(got) == LOCAL
    assert T.unbound(cfg, {}) == []
    variables = T.variables(cfg, {})
    assert [k for k, v in variables.items() if "<" in v] == []
    assert {k: variables.get(k) for k in LOCAL} == LOCAL and "H1_KEY_SHA512" not in variables


ONSITE_UNBOUND = [k for k in T.unbound(R.load(ONSITE, REG), {}) if k.split(".")[1] != "pouw"]


def onsite_pearlplus(tmp: Path) -> Path:
    f = tmp / "onsite.toml"
    f.write_text(ONSITE.read_text())
    return f


def test_under_onsite_pearl_plus_takes_h1s_place_with_its_product_and_no_compute_key_blocks(tmp_path, capsys):
    assert "pearl-plus" in (listed := K.parse(ONSITE.read_text())["protocols"]) and "pouw" not in listed
    f = onsite_pearlplus(tmp_path)
    rc, got, err = dry(capsys, config=f)
    written = {f"{p}.{c}.{k}" for p, s in T.STARTS.items() for c, ks in s.writes.items() for k in ks}
    assert rc == 0 and got["dry"][0]["unbound"] == sorted(set(ONSITE_UNBOUND) - written), err
    assert hooks_of(got) == LOCAL | {"PRODUCT": "onsite"}
    assert not {"H1_GATE", "H1_PEER"} & set(T.variables(R.load(f, REG), {}))
    tables = json.loads(json.dumps(STARTED))
    for key in ONSITE_UNBOUND:
        p, c, k = key.split(".")
        tables.setdefault(p, {}).setdefault(c, {})[k] = "192.0.2.1:7000"
    rc, got, err = dry(capsys, "--bindings", bindings(tmp_path, tables), config=f)
    assert rc == 0 and hooks_of(got) == PEARLPLUS | STARTED_HOOKS | {"PRODUCT": "onsite"}, err
    assert T.unbound(R.load(f, REG), {}) == ONSITE_UNBOUND


def test_a_part_of_pearl_pluss_live_challenger_is_refused_by_name(tmp_path, capsys):
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, {"challenger": {"pearl-plus": {"gate": "gate-command"}}}))
    assert rc == 2 and "refused unbound: Pearl+'s live challenger takes gate, peer, key-sha512 together" in err, err
    assert "without challenger.pearl-plus.peer, challenger.pearl-plus.key-sha512: bind them" in err, err
    rc, _, err = dry(capsys, "--bindings", bindings(tmp_path, {"challenger": {"pearl-plus": {"peer": "peer-1",
                                                                                              "key-sha512": KEY}}}))
    assert rc == 2 and "refused unbound: challenger.pearl-plus.gate is unbound" in err, err
    assert "bind challenger.pearl-plus.peer, challenger.pearl-plus.key-sha512 beside it" in err, err


def test_the_runners_variables_under_remote_are_pearl_pluss():
    cfg = R.load(CONFIG, REG)
    serve = T.SERVE.variables(cfg, BINDINGS)
    assert {k: serve[k] for k in ("REPO", "PRECISION", "SLOTS", "WINDOW_STEPS", "LOAD", "DATASET", "TIMING", "V3C_CLASS",
                                  "CLASSES")} == {
        "REPO": "unsloth/Llama-3.2-1B", "PRECISION": "bfloat16", "SLOTS": str(S.slots), "WINDOW_STEPS": str(S.outputs),
        "LOAD": "dataset", "DATASET": DATASET, "TIMING": "at-once", "V3C_CLASS": S.window_class, "CLASSES": S.window_class}
    variables = serve | PEARLPLUS | {"VERITY_CONFIGURATION": K.digest(cfg), "VERITY_PINS": "[]",
                                     "PP_DEVELOPMENT_UNDER_HOLD": "1"}
    assert T.variables(cfg, BINDINGS) == variables
    both = {"challenger": {"pearl-plus": {**BINDINGS["challenger"]["pearl-plus"], "lane": "vllm-2",
                                          "main-file": "/tmp/main.json"}}}
    assert T.variables(cfg, both) == variables | {"PEARLC_CHALLENGER_LANE": "vllm-2", "MAIN_FILE": "/tmp/main.json"}


def test_the_retired_tie_params_are_refused_by_name(tmp_path, capsys):
    for retired in ({"class": S.window_class}, {"tie-uniform": True}):
        rc, _, err = dry(capsys, config=remote(tmp_path, retired))
        assert rc == 2 and "refused unknown-key" in err and f"pearl-plus has no setting {sorted(retired)}" in err, err


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
    assert hooks == PEARLPLUS | WINDOW


def test_without_bindings_the_workload_runs_with_no_gate_and_no_placeholder(tmp_path, capsys, monkeypatch):
    for k in OPTIONAL | set(PEARLPLUS):
        monkeypatch.delenv(k, raising=False)
    runner = tmp_path / "runner.sh"
    runner.write_text(RUNNER)
    monkeypatch.setattr(T, "RUNNER", str(runner))
    state = tmp_path / "state"
    rc = R.run_main([str(CONFIG), "--on", T.PROVER_NODE, "--state", str(state)], REG)
    assert rc == 0 and set(parties(capsys).values()) == {None}
    hooks = dict(x.split("=", 1) for x in (state / "out" / "hooks.txt").read_text().splitlines())
    assert hooks == {k: v for k, v in LOCAL.items() if k != "DEVELOPMENT"} | WINDOW


HANDED = "HANDOFF the final record, composed off this node: on COMPOSE_NODE (auditor-2)"


def ends(tmp: Path, monkeypatch, rc: int, *lines: str) -> None:
    runner = tmp / "runner.sh"
    runner.write_text('mkdir -p "$OUT"\n' + "".join(f"echo {json.dumps(x)} >> \"$OUT/verdicts.txt\"\n" for x in lines)
                      + f"exit {rc}\n")
    monkeypatch.setattr(T, "RUNNER", str(runner))


def onsite_bindings() -> dict:
    tables: dict = {}
    for key in T.unbound(R.load(ONSITE, REG), {}):
        p, c, k = key.split(".")
        tables.setdefault(p, {}).setdefault(c, {})[k] = "192.0.2.1:7000"
    return tables


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
    rc = R.run_main([str(ONSITE), "--on", T.PROVER_NODE, "--party", T.WORKERS, "--party", T.ZK_SERVER, "--bindings",
                     bindings(tmp_path, onsite_bindings()), "--state", str(tmp_path / "state")], REG)
    assert rc == 2 and parties(capsys) == {T.WORKERS: T.SERVED_WINDOW, T.ZK_SERVER: T.SERVED_WINDOW}
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
#: inference-only's row as a verifier-side run would write it: its answer recomputed off the served node
CHECKED = {"verdict": "accept", "answer_from": "recomputed on this host by its Lean verifier"}
ALL_ACCEPTED = {c: "accept" for c in K.parse(CONFIG.read_text())["protocols"]}
YES = {"composed": "accept", "product": "remote"}


def test_remotes_verifier_reads_the_compose_answers_record_pearl_plus_under_work(tmp_path):
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out", {"composed": "pending", "product": "remote"}, {"work": {"verdict": "missing"}})
    written(tmp_path / "out" / "compose", YES, {**ACCEPTED, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["product"], got["block"], got["reason"]) == ("accept", T.REMOTE, 3, None)
    assert got["components"] == ALL_ACCEPTED
    written(tmp_path / "out" / "compose", {"composed": "reject", "product": "remote"},
            {**ACCEPTED, "work": {"verdict": "reject"}})
    assert T.remote_check(cfg, tmp_path)["verdict"] == "reject"


@pytest.mark.parametrize("work, verdict", [("accept", "accept"), ("reject", "reject"), ("missing", "undecided")])
def test_pearl_plus_row_says_what_the_auditor_leaves_unchecked_on_every_verdict(tmp_path, work, verdict):
    """Its coins' signatures, and the CUDA-graph rows its verdict doesn't cover."""
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out" / "compose", {**YES, "composed": {"missing": "pending"}.get(work, work)},
            {**ACCEPTED, "work": {"verdict": work}, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert got["verdict"] == verdict and got["answer_from"] == {
        "pearl-plus": "coins' signatures checked on the served node only; not checked by the auditor; Pearl+'s verdict "
                      "covers a workload served eager (vLLM's graph mode NONE): under a CUDA graph (FULL or PIECEWISE) a "
                      "step's linear runs at the capture size, with rows no request owns, and those rows are outside what "
                      "its verdict covers"}


def test_a_component_that_charges_nothing_reads_why_for_its_figure_and_shows_its_verdict(tmp_path, monkeypatch):
    """A stand-in declaration under inference-only's name whose resource is a `Lacks`: the check's line says why it
    has no figure (``charges_nothing``) and keeps its verdict among the components and in the combined verdict, as
    any component's: a reject of it makes the line reject and names it, and a missing row holds the line open."""
    from verity.protocols.accounting import scheme as S
    nothing = S.Lacks("charges nothing: a stand-in that checks a policy")
    standin = S.Scheme("inference-only", "verity/inference-only/v0", nothing, nothing, (), (), ())
    monkeypatch.setattr(T.schemes, "DECLARATIONS", {**T.schemes.DECLARATIONS, "inference-only": "stand-in"})
    monkeypatch.setitem(S.SCHEMES, "inference-only", standin)
    assert T.charges_nothing("inference-only") == nothing.why and T.charges_nothing("pouw") is None
    cfg = R.load(CONFIG, REG)
    base = tmp_path / "out" / "compose"
    written(base, YES, {**ACCEPTED, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["reason"], got["components"]) == ("accept", None, ALL_ACCEPTED)
    assert got["charges_nothing"]["inference-only"] == nothing.why
    written(base, {"composed": "reject", "product": "remote"}, {**ACCEPTED, "inference-only": {"verdict": "reject"}})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["components"]["inference-only"]) == ("reject", "reject")
    assert got["reason"] == "not accepted: inference-only reject"
    written(base, YES, ACCEPTED)
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["components"]["inference-only"]) == ("undecided", "absent")


@pytest.mark.parametrize("name,row", [("sampled-proofs", "integrity"), ("inference-only", "inference-only")])
def test_a_reject_of_a_declared_component_that_charges_nothing_is_named_and_rejects_the_line(tmp_path, name, row):
    """The sampled proofs and the inference-only check as declared (no stand-in): each charges nothing, and a reject
    of it is still its verdict among the components and makes the line reject."""
    assert T.charges_nothing(name)
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out" / "compose", {"composed": "reject", "product": "remote"},
            {**ACCEPTED, "inference-only": CHECKED, row: {"verdict": "reject"}})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["components"][name]) == ("reject", "reject")
    assert name in got["reason"] and name in got["charges_nothing"]


def test_inference_onlys_row_is_the_composed_records_and_no_file_of_the_prover_nodes_side_changes_a_cell(tmp_path):
    cfg = R.load(CONFIG, REG)
    base = tmp_path / "out" / "compose"
    written(base, YES, ACCEPTED, {"verdict": "accept"})
    got = T.remote_check(cfg, tmp_path)
    assert got["components"]["inference-only"] == "absent" and got["verdict"] == "undecided"
    assert got["reason"] == "not accepted: inference-only absent"
    for rows in (ACCEPTED, {**ACCEPTED, "inference-only": CHECKED}, {"work": {"verdict": "reject"}}):
        written(base, YES, rows)
        (tmp_path / "out" / "record").mkdir(parents=True, exist_ok=True)
        for f in ("verdicts.txt", "record/verdicts.json", "record/inference-only.json"):
            (tmp_path / "out" / f).unlink(missing_ok=True)
        alone = T.remote_check(cfg, tmp_path)
        written(tmp_path / "out", YES, {k: {"verdict": "accept"} for k in ("integrity", "work", "inference-only")},
                {"verdict": "accept"})
        (base / "record" / "inference-only.json").write_text(json.dumps({"verdict": "accept"}))
        assert T.remote_check(cfg, tmp_path) == alone, rows


def test_inference_onlys_row_as_the_served_node_wrote_it_holds_the_verdict(tmp_path):
    cfg = R.load(CONFIG, REG)
    for row in ({"verdict": "accept", "answer_from": f"{T.AS_WRITTEN} (inference-only.json)"}, {"verdict": "accept"}):
        written(tmp_path / "out" / "compose", YES, {**ACCEPTED, "inference-only": row})
        got = T.remote_check(cfg, tmp_path)
        assert got["components"] == ALL_ACCEPTED and got["verdict"] == "undecided"
        assert got["reason"] == f"not accepted: inference-only {T.AS_WRITTEN}"
    text = (ROOT / "benchmarks" / "one_stage" / "served_verdicts.py").read_text()
    assert f'AS_WRITTEN = "{T.AS_WRITTEN}"' in text


def test_without_a_compose_answer_or_the_nodes_composite_remotes_line_is_refused_by_name(tmp_path):
    """A record the served node wrote with no composite (``COMPOSITE`` set empty, or nothing composed) is no line:
    refused by name, where it once read silently undecided."""
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out", {**YES, "lean": "no composite was read"}, {**ACCEPTED, "inference-only": CHECKED})
    with pytest.raises(T.Refused) as e:
        T.remote_check(cfg, tmp_path)
    assert e.value.name == "composite"
    assert f"no compose answer in {tmp_path / 'out' / 'compose'}" in e.value.detail
    assert f"no composite in {tmp_path / 'out' / 'composite'}" in e.value.detail
    assert "'accept': no composite was read" in e.value.detail


def nodes_composite(out: Path) -> None:
    (out / "composite" / "remote").mkdir(parents=True, exist_ok=True)
    (out / "composite" / "remote" / "composite.json").write_text(json.dumps({"product": "remote"}))


def test_with_no_gate_answer_the_nodes_own_record_reads_development_never_the_auditors_accept(tmp_path, monkeypatch):
    """Gate's answer absent, the serving node's record and composite present: the record is the prover's own word that
    it passed, so each of its accepts, the line's and each row's, reads `T.DEVELOPMENT_VERDICT`, never accept (under the
    placement hold, undecided); a reject stands, and a Lean that gave no answer reads undecided with its reason."""
    monkeypatch.delenv("VERITY_HOLD_PLACEMENT", raising=False)
    cfg = R.load(CONFIG, REG)
    nodes_composite(tmp_path / "out")
    written(tmp_path / "out", YES, {**ACCEPTED, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["reason"], got["block"], got["source"]) == (
        T.DEVELOPMENT_VERDICT, T.FROM_SERVING_NODE, 3, T.SERVING_NODE)
    assert set(got["components"].values()) == {T.DEVELOPMENT_VERDICT} and "accept" not in json.dumps(got["components"])
    monkeypatch.setenv("VERITY_HOLD_PLACEMENT", "1")
    got = T.remote_check(cfg, tmp_path)
    assert got["verdict"] == "undecided" and set(got["components"].values()) == {"undecided"}
    assert f"the composed verdict {T.AS_WRITTEN}" in got["reason"]
    monkeypatch.delenv("VERITY_HOLD_PLACEMENT")
    written(tmp_path / "out", {"composed": "reject", "product": "remote"}, {**ACCEPTED, "work": {"verdict": "reject"}})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["components"]["pearl-plus"], got["components"]["sampled-proofs"]) == (
        "reject", "reject", T.DEVELOPMENT_VERDICT)
    written(tmp_path / "out", {"composed": "pending", "product": "remote", "lean": "its Lean gave no answer"},
            {**ACCEPTED, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["reason"]) == ("undecided", f"{T.FROM_SERVING_NODE}; its Lean gave no answer")


def test_with_the_gates_answer_and_the_nodes_record_both_present_the_gates_answer_wins(tmp_path, monkeypatch):
    """Both present: the line is the compose gate's answer, its accept the auditor's, with no ``source`` and no
    development sentence, whatever the node's own record says."""
    monkeypatch.delenv("VERITY_HOLD_PLACEMENT", raising=False)
    cfg = R.load(CONFIG, REG)
    nodes_composite(tmp_path / "out")
    for node, gate, want in ((YES, {"composed": "reject", "product": "remote"}, "reject"),
                             ({"composed": "reject", "product": "remote"}, YES, "accept")):
        written(tmp_path / "out", node, {**ACCEPTED, "inference-only": CHECKED})
        written(tmp_path / "out" / "compose", gate,
                {**ACCEPTED, "inference-only": CHECKED, **({"work": {"verdict": "reject"}} if want == "reject" else {})})
        got = T.remote_check(cfg, tmp_path)
        assert got["verdict"] == want and "source" not in got, got
        assert T.FROM_SERVING_NODE not in str(got["reason"]) and T.DEVELOPMENT_VERDICT not in got["components"].values()


def test_a_deferred_composite_is_refused_as_deferred_until_the_follow_ons_record_is_here(tmp_path, monkeypatch):
    """A served node short of the audit's memory defers the tied audit and the composite to a follow-on on a prover
    node: until that follow-on's ``OUT`` is copied to ``out/follow-on``, the line is refused as ``deferred``, naming
    where it reads it; then it is the follow-on's record, `T.FOLLOW_ON` its source, its accepts development."""
    monkeypatch.delenv("VERITY_HOLD_PLACEMENT", raising=False)
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out", {"composed": "pending", "product": "remote", "deferred": "196 GB, under 290"},
            {**ACCEPTED, "work": {"verdict": "missing"}})
    with pytest.raises(T.Refused) as e:
        T.remote_check(cfg, tmp_path)
    assert e.value.name == "deferred" and "196 GB, under 290" in e.value.detail
    assert str(tmp_path / "out" / "follow-on") in e.value.detail
    follow = tmp_path / "out" / "follow-on"
    nodes_composite(follow)
    written(follow, YES, {**ACCEPTED, "inference-only": CHECKED})
    got = T.remote_check(cfg, tmp_path)
    assert (got["verdict"], got["source"], got["reason"]) == (T.DEVELOPMENT_VERDICT, T.FOLLOW_ON, T.FROM_FOLLOW_ON)
    assert set(got["components"].values()) == {T.DEVELOPMENT_VERDICT}


def test_a_line_from_the_serving_nodes_own_record_says_so_and_a_gates_does_not(tmp_path, capsys, monkeypatch):
    """Nothing the prover runs is trusted: a line read from the serving node's own record says so in words, leading
    its reason, and by its ``source``, in the VERDICT line and the record `verity run` writes; a line read from the
    compose gate's answer carries neither."""
    monkeypatch.delenv("VERITY_HOLD_PLACEMENT", raising=False)
    assert T.FROM_SERVING_NODE == ("read from the serving node's own record: a development verdict; the auditor's "
                                   "device did not compose it")
    for gate in (False, True):
        state = tmp_path / str(gate)
        nodes_composite(state / "out")
        written(state / "out", YES, {**ACCEPTED, "inference-only": CHECKED})
        if gate:
            written(state / "out" / "compose", YES, {**ACCEPTED, "inference-only": CHECKED})
        R.run_main([str(CONFIG), "--on", T.AUDITOR_REMOTE, "--bindings", bindings(tmp_path, STARTED), "--state",
                    str(state)], REG)
        line, = said(capsys.readouterr().out, "VERDICT")
        kept, = (json.loads(x) for x in (state / "record" / "verdicts.jsonl").read_text().splitlines())
        for got in (line, kept):
            if gate:
                assert "source" not in got and T.FROM_SERVING_NODE not in str(got["reason"]), got
            else:
                assert got["source"] == T.SERVING_NODE and got["reason"].startswith(T.FROM_SERVING_NODE), got
                assert got["verdict"] == T.DEVELOPMENT_VERDICT, got


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
    written(state / "out" / "compose", YES, {**ACCEPTED, "inference-only": CHECKED})
    rc = R.run_main([str(CONFIG), "--on", T.AUDITOR_REMOTE, "--bindings", bindings(tmp_path, STARTED), "--state",
                     str(state)], REG)
    out = capsys.readouterr()
    lines = said(out.out, "VERDICT")
    assert len(lines) == 1, out
    assert (rc, lines[0]["product"], lines[0]["verdict"], lines[0]["block"]) == (0, T.REMOTE, "accept", 3)
    cfg = R.load(CONFIG, REG)
    gate = {p["name"]: p["machine"] for p in cfg["party"]}[T.CHALLENGER]
    assert lines[0]["coins"] == T.coins_of(cfg)
    assert lines[0]["coins_on"] == {SP.COMPONENT: gate, "pearl-plus": gate}


def test_the_verifiers_line_names_the_challengers_machine_for_a_bound_gate(tmp_path):
    cfg = R.load(CONFIG, REG)
    written(tmp_path / "out" / "compose", YES, {**ACCEPTED, "inference-only": CHECKED})
    spot = SP.COMPONENT
    at = {p["name"]: p["machine"] for p in cfg["party"]}
    for tables, want in ((BINDINGS, {spot: at[T.WORKERS], "pearl-plus": at[T.CHALLENGER]}),
                         ({"challenger": {**BINDINGS["challenger"], spot: {"gate": "user@gate-1"}}},
                          {spot: at[T.CHALLENGER], "pearl-plus": at[T.CHALLENGER]}),
                         (STARTED, {spot: at[T.CHALLENGER], "pearl-plus": at[T.CHALLENGER]})):
        (tmp_path / T.BINDINGS).write_text(json.dumps(tables))
        got = T.remote_check(cfg, tmp_path)
        assert got["coins_on"] == want
        assert got["coins"] == ("simulated" if at[T.WORKERS] in want.values() else T.coins_of(cfg))
    assert T.coins_of(cfg) == "development" and at[T.WORKERS] != at[T.CHALLENGER]
