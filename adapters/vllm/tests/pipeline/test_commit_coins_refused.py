"""What `verity_vllm.pipeline.commit_coins` refuses that no case of `test_commit_coins.py` reaches: a Commit whose coins records
are not one for each run of record, a root of record of another length, a run whose record is not its own coin of the commitment
the challenger signed, a challenge phase reserved for more runs than the Commit has of record, and a gate's open answer that is
no session.  With them, the gate as a process, which the other cases stand in for.  The auditor's challenger, the gate and the
records are `test_commit_coins.py`'s."""
from __future__ import annotations

import json
import re
import shlex
import sys

import pytest
from verity.protocols.proofs.spot.device.challenger import CoinError

import verity_vllm.pipeline.commit_coins as CC
from tests.pipeline.test_commit_coins import (
    ABSENT,
    GATE,
    PEER,
    _Auditor,
    _commit,
    _options,
    _reason,
    _root,
    _with,
)

#: the count's refusal, read around the one word of it the Glossary has retired
COUNT = r"the Commit's coins name {} runs' \w+, not one for each of its {} runs of record"
NOT_ITS_COIN = "run {0}'s coins are not coin {0} of the commitment the auditor's challenger signed"


def test_coins_that_are_not_one_record_a_run_of_record_are_refused_before_a_ledger_is_read(tmp_path, monkeypatch):
    """The verifier takes a coins record for each run of record, at least one, and each root of record whole: a Commit with a record
    too few or too many, with none, or with a root of another length is refused by name, and no ledger is read for it."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0", "run 1"])
    no_ledger = [tmp_path / "no-such-windows.jsonl"]
    for runs, of_record in (([], []), (records[:1], roots), (records, roots[:1]), ([*records, records[1]], roots)):
        why = _reason(auditor, runs, of_record, sid, heads=no_ledger)
        assert re.fullmatch(COUNT.format(len(runs), len(of_record)), why), why
    for short in ([roots[0], roots[1][:32]], [roots[0][:-1], roots[1]], [roots[0] + b"\x00", roots[1]], [b"", roots[1]]):
        assert _reason(auditor, records, short, sid, heads=no_ledger) == (
            f"a run's root of record is not {CC.ROOT_BYTES} bytes (rows-sha512 or the TP run root v2)")
    assert _reason(auditor, records, roots, sid) is None


def test_a_run_whose_coins_are_not_its_own_coin_of_the_signed_commitment_is_refused_by_name(tmp_path, monkeypatch):
    """Run i's coins are coin i of the commitment the challenger signed for the Commit's challenge phase. A record that carries
    another run's index, or names another commitment or none, is refused at the first such run, before any opening is read."""
    auditor = _Auditor(tmp_path, monkeypatch)
    records, roots, sid = _commit(auditor, ["run 0", "run 1"])
    assert _reason(auditor, [records[0], records[1] | {"index": 0}], roots, sid) == NOT_ITS_COIN.format(1)
    assert _reason(auditor, [records[0], _with(records[1], "index", ABSENT)], roots, sid) == NOT_ITS_COIN.format(1)
    assert _reason(auditor, [records[1], records[0]], roots, sid) == NOT_ITS_COIN.format(0)
    for value in (ABSENT, None, "", "00" * 64, records[0]["session"]["commitment"][:-2]):
        named = [records[0], records[1] | {"session": _with(records[1]["session"], "commitment", value)}]
        assert _reason(auditor, named, roots, sid) == NOT_ITS_COIN.format(1), value
    assert _reason(auditor, records, roots, sid) is None


def test_a_challenge_phase_reserved_for_more_runs_than_the_commit_has_of_record_is_refused_with_both_named(tmp_path, monkeypatch):
    """The challenge phase holds a coin of each family for each run of record and no more: one reserved for three runs, two of
    which took their coins and are of record, is refused, with the challenger's families and the Commit's named."""
    auditor = _Auditor(tmp_path, monkeypatch)
    coins = CC.of_options(_options(), 3, gate_run=auditor.run)
    roots = [_root("run 0"), _root("run 1")]
    records = [coins.draw(root) for root in roots]
    assert coins.close() is None
    sid = coins.reserved["session"]
    auditor.end(sid)
    theirs = [(f, 3) for f in CC.FAMILIES]
    assert _reason(auditor, records, roots, sid) == f"the challenger's coins are {CC.SCHEME_TAG} {theirs}, not the Commit's {CC.families(2)}"


def test_the_gate_s_command_takes_each_request_as_one_shell_quoted_word_and_answers_by_its_exit_code_and_output():
    """Without a stand-in the gate is a process: its command's words, then the request as one word, quoted so that the gate's forced
    command splits it into the words it was; the answer is what the process writes, and a refusal is its exit code."""
    echo = "import json, sys; print(json.dumps({'argv': sys.argv[1:]}))"
    gate = CC.ChallengerGate(shlex.join([sys.executable, "-c", echo, "first"]))
    assert gate.call("open", "--session", "s 1", "--root", "ab") == {"argv": ["first", "open --session 's 1' --root ab"]}
    assert [json.loads(line) for line in gate.ledger()] == [{"argv": ["first", "ledger"]}]
    busy = "import json, sys; print(json.dumps({'refused': 'busy'})); sys.exit(3)"
    with pytest.raises(CoinError, match="the challenger gate refused status \\(exit 3\\): busy"):
        CC.ChallengerGate(shlex.join([sys.executable, "-c", busy])).call("status", "s1")


@pytest.mark.parametrize("change", [{"seq": None}, {"seq": -1}, {"root": "ab"}, {"commitment": "AB" * 64}, {"commitment": None},
                                    {"secret": "5e"}])
def test_an_open_answer_that_is_not_a_session_is_refused_as_the_gate_s_answer_without_its_secret(change):
    """The gate's answer to an open is the session the Commit carries, with the hello's secret. One that is not a session (no
    ledger `seq`, a root or a commitment that is not 128 lowercase hex, none from `status` either, a secret of another length)
    is refused as the gate's open answer before any connection, and what is quoted of it holds no secret."""
    secret = "5e" * 32
    answer = {"session": "s1", "seq": 1, "root": _root("run 0").hex(), "commitment": "ab" * 64, "port": 7811, "secret": secret} | change
    asked: list[str] = []

    def gate(argv: list[str]) -> tuple[int, str]:
        asked.append(shlex.split(argv[-1])[0])
        return 0, json.dumps({"reserve": {"session": "s1"}, "open": answer, "status": {"session": "s1", "commitment": None}}[asked[-1]])

    coins = CC.ChallengePhase.reserve(CC.ChallengerGate(GATE, gate), PEER, 1, address=lambda port: pytest.fail("no connection is made"))
    with pytest.raises(CoinError, match="^the challenger gate's open answer: pouw setting 'session' is the challenger gate's open answer") as e:
        coins.draw(_root("run 0"))
    assert secret not in str(e.value) and str(e.value).endswith(" with a secret (not quoted)")
    assert asked == ["reserve", "open"] + ["status"] * (answer["commitment"] is None) and coins.client is None and coins.close() is None
