"""What a served run's audit showed, and what stands between it and fully proved (`verity_bench.served_row`): the
verdict line is the verifier's own and never says an accept it withheld was given; `proved` is printed only where a
Lean verifier decided a verdict and its lock entry lists no open premise, held one fixture each way; the list of what
is open is read from the run's files and the lock, and holds no line on the resampling holes or the cited theorem,
which `spot_status` prints; and the command prints the section from a record that holds what the audit showed."""

from __future__ import annotations

import json
import re

import pytest
from test_served_record import LAYOUT, served

from verity_bench import exfiltration as E
from verity_bench import jobs, served_row as W
from verity_bench import served_record as R
from verity_bench import spot_status

THEOREM = spot_status.THEOREMS[-1]
GIVEN = {"accepted": True, "interior": [], "replay_sample": [], "statements": []}
#: a request's own Definition as a layout names it: none of the stand-in's fields
OWN = {k: v for k, v in LAYOUT.items() if k not in W.STAND_IN} | {"replay": "Request_v1{L=2}"}
CLOSED = dict.fromkeys(spot_status.HOLES, spot_status.CLOSED)
#: a made-up table in which a Lean verifier decides every verdict and every check, each under a lock entry
ALL_LEAN = {
    "verdicts": {k: {"what": f"each session's {k}", "verifier": f"lean-{k}", "decided_by": "lean", "proved_by": "Made.up"}
                 for k in ("lean-verify", "lean-draw", "m0-verifier")},
    "checks": {k: {"what": k, "verifier": f"lean-{k}", "decided_by": "lean"}
               for k in ("checks_before_draw", "pins", "negatives", "registered_reads", "complete", "round_two")},
    "outcome": {"what": "the stream's outcome", "verifier": "lean-outcome", "decided_by": "lean"},
    "withheld": {"two-stage-form": {"why": "the form is not proved"}}}


def tree(tmp_path, *, entry=(), theorem=()):
    """A tree whose lock lists the verdicts' entry and the cited theorem, each with these open premises."""

    (tmp_path / "security").mkdir(exist_ok=True)
    (tmp_path / spot_status.LOCK).write_text(json.dumps({"format": 1, "guarantees": {
        "Made.up": {"premises": {"assumptions": [], "domain": [], "open": list(entry), "reads": []}},
        THEOREM: {"premises": {"assumptions": [], "domain": [], "open": list(theorem), "reads": []}}}}))
    return tmp_path


def record_of(tmp_path, **how):
    """A served run's record whose second stage sampled no proof block, so no planner is read for it."""

    dirs = served(tmp_path / "runs", **how)
    summary = json.loads((dirs["second"] / "out/second.json").read_text())
    summary["second_stage"]["draw"] = []
    (dirs["second"] / "out/second.json").write_text(json.dumps(summary))
    return R.read(**dirs)[0]


def test_proved_is_printed_only_where_every_lock_entry_has_no_open_premise(tmp_path):
    """One fixture each way: the same run, table and files, under a lock whose entries are closed and under one where
    the verdicts' entry takes a premise."""

    record = record_of(tmp_path, outcome=GIVEN, layout=OWN)
    doc = W.account(record, tree(tmp_path), kinds=ALL_LEAN, holes=CLOSED)
    out = W.render(doc)
    assert doc["verdict"]["mark"] == W.PROVED and doc["to_fully_proved"] == []
    assert "- **verdict: proved.** Every sampled proof accepted by the Lean verifier (2 of 2 sessions); the final accept " \
           "given by `lean-outcome`." in out
    assert "- **to fully proved**: nothing in the run's own files or in its verdicts' lock entries." in out
    assert out.count("proved by `Made.up`, which takes no open premise") == 3 and all(k[W.PROVED] for k in doc["verdicts"])
    assert doc["verdict_record"] == {"piece": "second_stage", "control": None, "run": record["run"],
                                     "verifier": {"module": "lean-outcome", "commit": record["commit"]},
                                     "verdict": "accept", "proved_by": THEOREM, "open": None}

    doc = W.account(record, tree(tmp_path, entry=["Made.Premise"]), kinds=ALL_LEAN, holes=CLOSED)
    out = W.render(doc)
    assert doc["verdict"]["mark"] == E.RUN_NOT_PROVED and "- **verdict: run, not proved.**" in out
    assert "proved by" not in out and "verdict: proved" not in out and not any(k[W.PROVED] for k in doc["verdicts"])
    assert [i["said"] for i in doc["to_fully_proved"]] == [
        f"`{k}` (lean-{k}): its lock entry `Made.up` takes 1 open premise" for k in ("lean-draw", "lean-verify", "m0-verifier")]


@pytest.mark.parametrize("how", ["theorem", "hole", "unknown hole", "no entry named", "entry not in the lock", "python",
                                 "stand-in", "no layout"])
def test_one_thing_open_and_the_mark_is_run_not_proved(tmp_path, how):
    record = record_of(tmp_path, outcome=GIVEN, layout=LAYOUT if how == "stand-in" else OWN)
    kinds, holes, root = json.loads(json.dumps(ALL_LEAN)), dict(CLOSED), tree(tmp_path)
    if how == "theorem":
        root = tree(tmp_path, theorem=["Made.Premise"])
    elif how == "hole":
        holes[spot_status.HOLES[0]] = spot_status.OPEN
    elif how == "unknown hole":
        holes[spot_status.HOLES[1]] = spot_status.UNKNOWN
    elif how == "no entry named":
        kinds["verdicts"]["lean-draw"]["proved_by"] = None
    elif how == "entry not in the lock":
        kinds["verdicts"]["lean-verify"]["proved_by"] = "Made.other"
    elif how == "python":
        kinds["checks"]["pins"]["decided_by"] = "python"
    elif how == "no layout":
        record["audit"]["layout"] = None
    doc = W.account(record, root, kinds=kinds, holes=holes)
    out = W.render(doc)
    assert doc["verdict"]["mark"] == E.RUN_NOT_PROVED and "verdict: proved" not in out
    said = [i["said"] for i in doc["to_fully_proved"]]
    assert said == {
        "theorem": [], "hole": [], "unknown hole": [],
        "no entry named": ["`lean-draw` (lean-lean-draw): the table names no lock entry for it"],
        "entry not in the lock": ["`lean-verify` (lean-lean-verify): `Made.other` is not in the lock"],
        "python": ["decided by a verifier that is not Lean: `pins` (lean-pins)"],
        "stand-in": ["the replay block is the stand-in (`Select{V=8}`), not a Definition of the request: the model's "
                     "layers are not covered"],
        "no layout": ["the replay block is not known: no layout is given (the first stage's `stage1.json`, or the serve's "
                      "`commit.json`)"]}[how]
    # the holes and the theorem are the other block's to print: this one reads them for its mark and says nothing of them
    assert doc["mark_reads"] == {"theorem": THEOREM, "open_premises": int(how == "theorem"), "holes": holes}
    # (the verdict's record names the theorem in its own fields, as the command's verdicts of record do)
    lines = out.split("\nThe verdict as a record")[0]
    assert THEOREM not in lines and not any(h in lines for h in spot_status.HOLES)
    # the record names no proof while anything is open, and says what is
    assert (doc["verdict_record"]["verdict"], doc["verdict_record"]["proved_by"]) == ("accept", None)
    assert doc["verdict_record"]["open"] == {
        "theorem": f"{THEOREM} takes 1 open premise", "hole": "not closed at the run's commit: redrawn sample",
        "unknown hole": "not closed at the run's commit: re-committed execution phase"}.get(how, "; ".join(said))


def test_an_accept_the_verifier_withheld_is_never_said_to_be_given(tmp_path, monkeypatch):
    """Under the tree's own table: every check held and the outcome is refused by the two-stage form alone."""

    record = record_of(tmp_path)
    doc = W.account(record, tree(tmp_path), holes=CLOSED)
    out = W.render(doc)
    assert doc["outcome"]["final"] == "withheld" and doc["verdict"]["mark"] == E.RUN_NOT_PROVED
    assert doc["verdict"]["line"] == ("every sampled proof accepted by the Lean verifier (2 of 2 sessions); the final accept "
                                      "withheld, reason `two-stage-form` (the two-round form's soundness is not proved)")
    assert "final accept given" not in out and "| withheld (`two-stage-form`) |" in out
    assert "  - the final accept is withheld by `sampled-proofs-verifier`: `two-stage-form`, the two-round form's " \
           "soundness is not proved" in out
    # who decided each verdict is the table's, and a verdict no Lean verifier decided is listed
    assert {k["kind"]: k["decided_by"] for k in doc["verdicts"]} == {"lean-draw": "lean", "lean-verify": "lean",
                                                                     "m0-verifier": "rust and python"}
    assert "  - decided by a verifier that is not Lean: `m0-verifier` (" in out and "`round_two` (" in out
    # its record takes the command's word for a withheld accept once the command has one
    waits = "the two-round form's soundness is not proved"
    assert (doc["verdict_record"]["verdict"], doc["verdict_record"]["open"]) == (
        "withheld" if "withheld" in E.VERDICT_READS else "refuse", waits)
    monkeypatch.setattr(E, "VERDICT_READS", (*E.VERDICT_READS, "withheld"))
    assert W.verdict_record(doc)["verdict"] == "withheld" and W.verdict_record(doc)["proved_by"] is None
    # no issue's id is printed, and none is in the table
    assert not re.search(r"\b[A-Z]{2,}-\d+\b", out + W.TABLE.read_text())


def test_a_refused_outcome_and_no_outcome_are_said_as_they_are(tmp_path):
    refused = {"accepted": False, "interior": ["statements"], "replay_sample": [], "statements": ["second-wires"]}
    doc = W.account(record_of(tmp_path, outcome=refused, held=False), tree(tmp_path), holes=CLOSED)
    assert doc["verdict"]["line"].endswith("; the final accept refused, `second-wires`")
    assert doc["to_fully_proved"][-1]["said"] == "the final accept is refused"
    assert (doc["verdict_record"]["verdict"], doc["verdict_record"]["open"]) == ("refuse", "the verifier refused it: second-wires")
    # the form's code beside another, or under a check that did not hold, is a refusal, not a withheld accept
    both = refused | {"statements": ["two-stage-form", "second-wires"]}
    assert W.account(record_of(tmp_path, outcome=both), tree(tmp_path), holes=CLOSED)["outcome"]["final"] == "refused"
    form = refused | {"statements": ["two-stage-form"]}
    assert W.account(record_of(tmp_path, outcome=form, held=False), tree(tmp_path), holes=CLOSED)["outcome"]["final"] == "refused"
    # a second stage that reached no outcome holds no verdict; a proof the Lean verifier refused is counted as such
    verdicts = {"lean-verify/m0": True, "lean-verify/m1": False}
    doc = W.account(record_of(tmp_path, outcome=None, held=False, verdicts=verdicts), tree(tmp_path), holes=CLOSED)
    assert doc["verdict"]["line"] == ("1 of 2 sampled proofs accepted by the Lean verifier; no final accept: the second "
                                      "stage reached no outcome")
    assert doc["verdict_record"] is None and doc["to_fully_proved"][-1]["said"] == "the final accept is not reached"


def test_what_the_audit_covered_is_the_layouts_own_account(tmp_path):
    stand_in = W.covered(R.read(**served(tmp_path / "a"))[0]["audit"]["layout"], {"registered": ["logits"]})
    assert stand_in["stand_in"] is True and stand_in["said"] == (
        'the replay block is the stand-in, `Inlined{template="Select{V=8}",N=2}`: instances of the check template '
        "`Select{V=8}`, 1 for each output token, each one in place; what it reads is the replay's own value, `logits`, "
        "registered after the first stage's sample. It checks the choice of each token from a row of scores, under no stop "
        "rule, and nothing of the model's layers that produced the scores")
    stop = W.covered(LAYOUT | {"stop": {"rule": "eos"}, "per_prompt": 2}, {"unlinked": True})
    assert "1 for each output token and 2 for each prompt token" in stop["said"]
    assert "the second stage ran unlinked" in stop["said"] and "and the stop rule the layout registers" in stop["said"]
    own = W.covered(OWN, {})
    assert own["stand_in"] is False and own["said"].startswith("the replay block is `Request_v1{L=2}`, a Definition of the")
    assert W.covered(None, {})["stand_in"] is None


def test_the_provers_seconds_say_which_they_are_and_a_noted_card_is_said_beside_them(tmp_path):
    dirs = served(tmp_path / "runs")
    summary = json.loads((dirs["second"] / "out/second.json").read_text())
    summary["second_stage"]["draw"] = []
    for i, s in enumerate(summary["audit"]["timing"].values()):
        s |= {"gpus": 1, "busy_s": 20.0 + i, R.SOURCE: "busy"}
    summary["audit"]["proof_server"] = {"devices": [0, 1], "busy": {
        "prover-gpu0": {"device": 0, "seconds": 20.0, "sessions": 1}, "prover-gpu1": {"device": 1, "seconds": 21.0, "sessions": 1}}}
    (dirs["second"] / "out/second.json").write_text(json.dumps(summary))
    job = json.loads((dirs["second"] / "job.json").read_text())
    job["run"]["remote"]["machine"] = "made-up-node"
    (dirs["second"] / "job.json").write_text(json.dumps(job))
    kinds = W.table() | {"cards": [{"machine": "made-up-node", "card": 1, "note": "a made-up note"},
                                   {"machine": "another-node", "card": 0, "note": "not this machine's"}]}
    out = W.render(W.account(R.read(**dirs)[0], tree(tmp_path), kinds=kinds, holes=CLOSED))
    assert "- the sessions' proving: 41.0 s over 2 sessions, by the proof server's own clock on each session's prover " \
           "(`busy_s`). 2 of 2 sessions held a GPU, 41.0 GPU-seconds in all." in out
    assert "- card 0 of made-up-node: 1 sessions, 20.0 s busy.\n" in out
    assert "- card 1 of made-up-node: 1 sessions, 21.0 s busy. **A made-up note.**" in out and "not this machine's" not in out
    # a session's seconds, wait included, are an upper bound and said so; sessions on the host's cores hold no GPU-seconds
    for s in summary["audit"]["timing"].values():
        s |= {"gpus": 0, R.SOURCE: "session"}
    (dirs["second"] / "out/second.json").write_text(json.dumps(summary))
    out = W.render(W.account(R.read(**dirs)[0], tree(tmp_path), kinds=kinds, holes=CLOSED))
    assert "- the sessions' proving: at most 90.0 s over 2 sessions, by the audit's clock on each session (`session_s`)" in out
    assert "No session held a GPU: the sessions proved on the host's CPUs, 30 cores, so the proving holds no GPU-seconds." in out


def test_the_security_level_is_the_planners_bound_at_the_runs_own_sizes():
    """One verification phase of the run's sessions and its sampled proof blocks, each session one table, by the
    planner's own function."""

    root = jobs.repository_root()
    got = W.security({"sessions": 20, "proof_blocks": 40}, root)
    binding, headline = E._load(root, "spot_figures", "binding"), E._load(root, "spot_figures/breakeven", "headline")
    want = binding.over_phases(20, headline.CLASSES_A_PROOF_BLOCK * 40, binding.queries(headline.TABLE_SYMBOLS), 1)
    assert got["counted"] == want["counted"] and got["design"] == want["design"] and got["status"] == "modeled"
    assert (got["sessions"], got["sampled_proof_blocks"], got["proof_blocks"], got["tables"]) == (20, 40, 40, 1)
    assert [i["value"] for i in got["assumed"]] == [headline.CLASSES_A_PROOF_BLOCK, headline.TABLE_SYMBOLS, None]
    said = "\n".join(W._security_said(got))
    assert f"**at most {want['counted']['calls']['phase']['bound']}** counted in calls" in said and "(modeled)" in said
    assert "one table a session, the files holding no count of a session's tables" in said
    assert W.security({"sessions": 2, "proof_blocks": 0}, root) is None
    assert "not computed, the second stage sampled no proof block" in W._security_said(None)[0]


def test_the_command_prints_the_section_from_a_record_that_holds_what_the_audit_showed(tmp_path, monkeypatch, capsys):
    record = record_of(tmp_path)
    (served_run := tmp_path / "record.json").write_text(json.dumps(record))
    monkeypatch.setattr(E, "build", lambda **kw: {"runs": [], "served": kw["served"] is not None})
    monkeypatch.setattr(E, "render", lambda doc: "THE FIGURE")
    monkeypatch.setattr(spot_status, "status", lambda runs: {"runs": 0})
    monkeypatch.setattr(spot_status, "render", lambda doc: "THE HOLES")
    monkeypatch.setattr(jobs, "repository_root", lambda *a: tree(tmp_path))
    assert E.main(["--served-run", str(served_run)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("THE FIGURE\n\n## What the audit showed on r20261113-000004-0004, and what stands between it and "
                          "fully proved\n\n- **verdict: run, not proved.**") and out.endswith("\n\nTHE HOLES\n")
    assert E.main(["--served-run", str(served_run), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["served_run"]["schema"] == W.SCHEMA and doc["served_run"]["outcome"]["final"] == "withheld"
    # a record of the share's fields alone prints as it did, and so does the command with no record
    (served_run := tmp_path / "share.json").write_text(json.dumps({k: record[k] for k in R.FIELDS}))
    assert E.main(["--served-run", str(served_run)]) == 0 and capsys.readouterr().out == "THE FIGURE\n\nTHE HOLES\n"
    assert E.main([]) == 0 and capsys.readouterr().out == "THE FIGURE\n\nTHE HOLES\n"


def test_a_table_that_names_no_decider_is_refused(tmp_path):
    (bad := tmp_path / "table.json").write_text(json.dumps({"verdicts": {"k": {"what": "w", "verifier": "v"}}, "checks": {}}))
    with pytest.raises(ValueError, match="`verdicts.k` lacks decided_by"):
        W.table(bad)
    kinds = W.table()
    assert set(kinds) == {"verdicts", "checks", "outcome", "withheld", "cards"}
    # a kind or a check the table does not name is printed as not known, never as a Lean verifier's
    doc = W.account(record_of(tmp_path, verdicts={"new-kind/m0": True}), tree(tmp_path), holes=CLOSED)
    assert doc["verdicts"] == [{"kind": "new-kind", "what": "a kind the table does not name", "verifier": "not known",
                                "decided_by": "not known", "accepted": 1, "of": 1, W.PROVED: False,
                                "proved_by": {"name": None, "listed": False, "open_premises": None, W.PROVED: False}}]
