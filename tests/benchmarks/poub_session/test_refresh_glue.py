"""Refresh as one setting from the command to the result: `run.sh`'s REFRESH is the verifier's `--refresh`
(`test_run_serve.py`), which sets `audit.Params.refresh` a session, tells the prover in the session's message and
names it in the Lean verifier's parameters and in the session's result; and `refresh_result.py` writes the sessions of
a run side by side, one row a rule. Unset, no message, parameter or result holds the word."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "poub_session"))
sys.path.insert(0, str(ROOT))

import refresh_result as R  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import audit  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.session import lean_verifier as lv  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.session import verifier  # noqa: E402

from tests.verity.protocols.resource_accounting.memory.poub import session_harness as H  # noqa: E402


@pytest.fixture
def sent(monkeypatch) -> list[dict]:
    """Every message's head either side sends, in order."""
    heads, send = [], H.Counted.send

    def recording(self, head, body=b""):
        heads.append(head)
        send(self, head, body)

    monkeypatch.setattr(H.Counted, "send", recording)
    return heads


@pytest.fixture
def params_with_refresh(monkeypatch):
    """`audit.Params` with the wire's `refresh` field: the tree's own where it has one, else it with the field added."""
    if "refresh" in {f.name for f in dataclasses.fields(audit.Params)}:
        return audit.Params

    @dataclasses.dataclass(frozen=True, slots=True)
    class Params(audit.Params):
        refresh: bool = False

    monkeypatch.setattr(audit, "Params", Params)
    return Params


@pytest.mark.parametrize("words, sessions, each", [
    (None, 2, (False, False)), ("", 1, (False,)), ("0", 2, (False, False)), ("1", 3, (True, True, True)),
    ("0,1", 2, (False, True)), ("1,0", 2, (True, False))])
def test_refresh_is_one_word_for_every_session_or_one_a_session(words, sessions, each):
    assert verifier.refresh_of(words, sessions) == each


@pytest.mark.parametrize("words, sessions", [("0,1", 1), ("0,1", 3), ("yes", 1), ("1,", 2), ("2", 1)])
def test_a_refresh_word_that_does_not_set_each_session_is_refused(words, sessions):
    with pytest.raises(ValueError, match="--refresh"):
        verifier.refresh_of(words, sessions)


def test_with_refresh_unset_no_message_and_no_result_holds_the_word(sent):
    rs, _ = H.run("honest,honest", sessions=2)
    assert [r["accepted"] for r in rs] == [True, True]
    assert [h["type"] for h in sent].count("session") == 2
    assert "refresh" not in json.dumps(sent) and "refresh" not in json.dumps(rs)
    assert [set(h) for h in sent if h["type"] == "session"] == [{"type", "k", "windows", "lifetime", "pace"}] * 2


def test_a_session_set_to_refresh_on_a_tree_whose_parameters_hold_none_is_refused_before_it_starts(monkeypatch, sent):
    @dataclasses.dataclass(frozen=True)
    class Bare:
        blocks: int

    real = dataclasses.fields
    monkeypatch.setattr(verifier.dataclasses, "fields", lambda c: real(Bare) if c is audit.Params else real(c))
    with pytest.raises(ValueError, match="session 0 is set to refresh, and this tree's audit.Params has no `refresh`"):
        H.run("honest", refresh=(True,))
    assert not [h for h in sent if h["type"] == "session"]


def test_the_session_set_to_refresh_is_told_so_and_the_one_before_it_is_not(params_with_refresh, sent):
    rs, _ = H.run("kv:2,kv:2", sessions=2, refresh=(False, True), kv_every=4, kv_pool=8, lifetime=2)
    first, second = [h for h in sent if h["type"] == "session"]
    assert "refresh" not in first and second["refresh"] is True
    assert rs[0]["accepted"] and "refresh" not in rs[0] and rs[1]["refresh"] is True


def test_the_lean_verifiers_parameters_name_refresh_only_when_it_is_on(params_with_refresh):
    off, on = params_with_refresh(8, 1, 2, lifetime=2), params_with_refresh(8, 1, 2, lifetime=2, refresh=True)
    assert "refresh" not in lv.params_json(off) and lv.params_json(on)["refresh"] is True
    assert {k: v for k, v in lv.params_json(on).items() if k != "refresh"} == lv.params_json(off)


def test_a_refresh_session_the_lean_verifier_ran_without_reading_the_setting_is_a_disagreement(params_with_refresh):
    """A Lean verifier that accepts a refresh session and whose record doesn't say refresh ran it under today's rule:
    the session is refused `disagreement`, never accepted."""
    rs, _ = H.run("kv:2,kv:2", sessions=2, refresh=(False, True), kv_every=4, kv_pool=8, lifetime=2)
    lean_reads_it = (rs[1].get("lean_record") or {}).get("refresh") is True
    if not lean_reads_it:
        assert not rs[1]["accepted"]
        if rs[1]["failure"]["failure"] == "disagreement":
            assert rs[1]["disagreement"]["event"] in ("receive", "refresh", "record")


# ------------------------------------------------------------------------------------------------ refresh_result.py

BLOCK = 4096


def a_session(k: int, *, refresh: bool = False, accepted: bool = True, gamma: float = 1.25, receipts: int = 100,
              post: int = 30, credited: bool = False, probation: list | None = None) -> dict:
    windows = [{"post_seed": [10, 20]} | ({} if probation is None else {"probation": probation})]
    s = {"k": k, "accepted": accepted, "verdict": "lean", "windows": 6, "lifetime": 2, "samples": 16,
         "failure": None if accepted else {"t": 3, "failure": "probation", "at": None},
         "appended": 12, "released": 4, "challenger": {"challenger": "local"}, "lean_check_ns": {"0": 2_000_000, "1": 5_000_000}}
    if refresh:
        s["refresh"] = True
    if accepted:
        s["record"] = {"public": {"block_bytes": BLOCK}, "profile": {"windows": windows}}
        s["ledger"] = {"inputs": {"passes": 7, "receipts": receipts, "post_seed_receipts": post,
                                  "post_seed_credited": credited},
                       "spec": {"gamma": gamma, "gamma_decided_by": "lean", "certified_bytes": 5.0,
                                "terms": {}, **({"probation": 0.5} if probation is not None else {})},
                       "declared": {"gamma": {"spec": {"error": {"windows": 1e-9}}}}}
    return s


def a_run(tmp: Path, name: str, sessions: list[dict], proved: list[dict] | None = None, blocks: int = 64,
          digest: str = "d0") -> Path:
    out = tmp / name / "out"
    out.mkdir(parents=True)
    (out / "verifier.json").write_text(json.dumps({
        "registration": {"blocks": blocks, "codec": "p512/k2", "rows": 16, "digest": digest}, "sessions": sessions}))
    if proved is not None:
        (out / "prover.json").write_text(json.dumps({"sessions": proved}))
    return out


def test_one_run_of_two_sessions_is_two_rows_on_one_registration_each_under_its_rule(tmp_path):
    out = a_run(tmp_path, "r", [a_session(0), a_session(1, refresh=True, credited=True, probation=[5, 25], gamma=1.5)],
                [{"k": 0, "mode": "kv", "fraction": 4.0, "kv": {"rekeyed": 6, "appended": 12}},
                 {"k": 1, "mode": "kv", "fraction": 4.0, "refresh": {"appends": 3, "blocks": 9, "encoded": 12}}])
    doc = R.document([out])
    assert (doc["runs"], doc["same_registration"], doc["trial"], doc["codec"]) == (1, True, True, "p512/k2")
    assert doc["store"] == {"blocks": 64, "block_bytes": BLOCK, "bytes": 64 * BLOCK, "depth": 6, "rows": 16}
    assert "own live coins" in doc["coins"] and "local" in doc["coins"] and doc["plans"] == ["kv:4.0", "kv:4.0"]
    assert "not a bound" in doc["reference"]
    off, on = doc["rules"]
    assert (off["refresh"], off["session"], off["accepted"], off["gamma"], off["error"]) == (False, 0, True, 1.25, 1e-9)
    assert (off["credited_receipts"], off["credited_bytes"], off["probation"]) == (70, 70 * BLOCK, None)
    assert off["rekeys"] == {"appends": None, "blocks": 6, "bytes": 6 * BLOCK, "share_of_encodes": 0.5}
    assert (off["codec"], off["store_depth"], off["lean_check_ms"]) == ("p512/k2", 6, [2.0, 5.0])
    assert (on["refresh"], on["credited_receipts"], on["credited_bytes"]) == (True, 100, 100 * BLOCK)
    assert on["probation"] == {"waiting_bytes": 30 * BLOCK, "written_bytes": 60 * BLOCK, "share": 0.5, "term": 0.5}
    assert on["rekeys"] == {"appends": 3, "blocks": 9, "bytes": 9 * BLOCK, "share_of_encodes": 0.75}


def test_a_refused_session_keeps_its_row_with_its_failures_name_and_no_gamma(tmp_path):
    out = a_run(tmp_path, "r", [a_session(0, gamma=0.5), a_session(1, refresh=True, accepted=False)])
    doc = R.document([out])
    on = doc["rules"][1]
    assert (on["accepted"], on["failure"], on["failure_window"], on["gamma"], on["error"]) == (
        False, "probation", 3, None, None)
    assert doc["reference"] is None, "a share below 1 is a bound, and the line is not said"


def test_two_runs_are_said_to_be_two_and_stores_that_differ_are_not_compared(tmp_path):
    a, b = a_run(tmp_path, "a", [a_session(0)]), a_run(tmp_path, "b", [a_session(0, refresh=True)], digest="d1")
    doc = R.document([a, b])
    assert (doc["runs"], doc["same_registration"], doc["registrations"]) == (2, False, ["d0", "d1"])
    assert [(r["run"], r["refresh"]) for r in doc["rules"]] == [(0, False), (1, True)]
    other = a_run(tmp_path, "c", [a_session(0)], blocks=128)
    with pytest.raises(ValueError, match="differ in blocks or codec"):
        R.document([a, other])


def test_the_result_is_a_trial_that_names_the_document_and_holds_each_rules_share_beside_its_row(tmp_path, monkeypatch,
                                                                                                capsys):
    out = a_run(tmp_path, "r", [a_session(0), a_session(1, refresh=True, accepted=False)])
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(out.parent))
    monkeypatch.delenv("VERITY_BENCH_TRIAL", raising=False)
    assert R.main([str(out)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert json.loads((out / "refresh.json").read_text()) == doc
    result = json.loads((out.parent / "result.json").read_text())
    assert result["trial"] is True and result["measurements"] == [] and result["measurement_files"] == ["out/refresh.json"]
    assert result["workload_fingerprint"]["codec"] == "p512/k2" and result["workload_fingerprint"]["store_depth"] == 6
    off, on = json.loads((out / "trial-measurements.json").read_text())
    assert (off["member"], off["value"], off["rule"]) == ("refresh-off-run-0-session-0", 1.25, doc["rules"][0])
    assert (on["member"], on["value"]) == ("refresh-on-run-0-session-1", None) and "probation" in on["why"]


def test_a_directory_with_no_record_is_said_and_nothing_is_written(tmp_path, capsys):
    assert R.main([str(tmp_path)]) == 2
    assert "refresh_result:" in capsys.readouterr().err and list(tmp_path.iterdir()) == []



def test_a_session_on_a_card_adds_its_card_object_and_the_reference_size_says_it_is_not_a_rate(tmp_path):
    out = a_run(tmp_path, "cpu", [a_session(0)])
    doc = R.document([out])
    assert doc["rules"][0]["card"] is None and "on a processor" in doc["size"] and "not a rate" in doc["size"]
    on = a_session(1, refresh=True, probation=[5, 25])
    on["record"]["public"]["span_ns"] = 4_000_000_000
    on["record"]["profile"]["windows"].append({"post_seed": [10], "probation": [30]})
    carded = a_run(tmp_path, "card", [a_session(0), on],
                   [{"k": 1, "mode": "kv", "fraction": 4.0,
                     "refresh": {"appends": 3, "blocks": 500, "encoded": 600, "encode_ns": 1_000_000_000}}])
    v = json.loads((carded / "verifier.json").read_text())
    v["calibration"] = {"pass_ns": {"p50": 52_900_000}, "encode_bytes_per_s": 2.5e9,
                        "read_tiers": {"hbm": {"bytes_per_s": 1.5e12}}}
    (carded / "verifier.json").write_text(json.dumps(v))
    doc = R.document([carded])
    off, on = doc["rules"]
    assert doc["size"] is None
    assert off["card"] == {"pass_ms_p50": 52.9, "bare_read_TBps": 1.5, "encode_Bps": 2.5e9, "rekey_Bps": None,
                           "rekey_share": None, "probation_share": None, "store_bytes": 64 * BLOCK}
    assert on["card"] | {"pass_ms_p50": 0} == {
        "pass_ms_p50": 0, "bare_read_TBps": 1.5, "encode_Bps": 2.5e9, "rekey_Bps": 500 * BLOCK / 1.0,
        "rekey_share": 0.25, "probation_share": 0.75, "store_bytes": 64 * BLOCK}


def test_a_row_says_what_the_verifiers_disagreed_on(tmp_path):
    s = a_session(0, refresh=True, accepted=False) | {
        "disagreement": {"t": 0, "event": "receive", "lean": {"ok": True}, "python": {"failure": "probation"}, "x": 1}}
    row = R.document([a_run(tmp_path, "a", [a_session(1), s])])["rules"][1]
    assert row["disagreement"] == {"t": 0, "event": "receive", "lean": {"ok": True}, "python": {"failure": "probation"}}

