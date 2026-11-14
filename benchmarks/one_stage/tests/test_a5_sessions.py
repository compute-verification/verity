"""a5's sampled members' sessions run one at a time: outer proofs go through the ZK server one at a time (AGENTS.md), so
``--session-jobs`` takes only 1, kept so the commands that pass it still run, and the proof server an audit proves its
members on behind the ZK server (`a5.proof_server`) verifies none ahead and starts every member's Lean session loop on
the one environment its members agree on. Its verifiers take their sessions' coin trees through the audit's relay where
there is one, and a TP rank's row carries the audit's word for where they came from (``coin_trees``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402


def test_session_jobs_takes_one():
    assert a5.one_session("1") == 1


@pytest.mark.parametrize("jobs", ["0", "2", "16"])
def test_session_jobs_refuses_anything_else_naming_the_rule(jobs):
    with pytest.raises(argparse.ArgumentTypeError, match="through the firewall one at a time"):
        a5.one_session(jobs)


def args(loop: str | None, **more) -> argparse.Namespace:
    """An audit's arguments: ``loop``, the ZK server's Lean loop its sessions run behind, or None for none."""
    return argparse.Namespace(bin="flock-circuit", firewall=loop, **more)


def test_behind_the_zk_server_the_proof_server_proves_one_session_at_a_time_on_the_members_one_environment(tmp_path):
    a = args("loop")
    window = {"partition_file": tmp_path / "partition.json", "program_file": tmp_path / "program.json"}
    server = a5.proof_server(a, tmp_path, [dict(window), dict(window)])
    assert server.verify_ahead == 0 and server.zk and server.out == tmp_path / "proof-server"
    # the environment a member's own prover would run in (`a0.prover_env`), which names the loop and the window's files
    want = a5.a0.prover_env("loop", window)
    assert "loop" in want.values() and any(str(tmp_path / "program.json") in v for v in want.values())
    assert {k: server._env[k] for k in want} == want and want != a5.a0.DIRECT
    server.close()
    # members that would set their loops up differently are not one stream's: each is a process pair, as a plain audit's is
    assert a5.proof_server(a, tmp_path, [dict(window), {**window, "statement": "verity/flock-circuit/tmpl"}]) is None
    assert a5.proof_server(a, tmp_path, []) is None
    assert a5.proof_server(args("loop", plain=True), tmp_path, [window]) is None


@pytest.mark.parametrize("loop", ["loop", None], ids=["behind the ZK server", "with no ZK server"])
def test_on_the_windows_challenger_session_the_proof_servers_verifiers_take_their_trees_through_the_relay(
        tmp_path, monkeypatch, loop):
    """With the audit's relay (`a5.window_trees`) the server starts no challenger of its own, where FLOCK_CHALLENGER=1
    would give each verifier one, and verifies none ahead: a session's rounds are in the session's log before the next
    session's first."""
    monkeypatch.setenv("FLOCK_CHALLENGER", "1")
    relay = SimpleNamespace(flags=["--tree-from", "127.0.0.1:2", "--tree-secret-file", "relay-secret"])
    window = {"partition_file": tmp_path / "partition.json", "program_file": tmp_path / "program.json"}
    server = a5.proof_server(args(loop), tmp_path, [dict(window)], relay)
    assert server.tree_from == relay.flags and not server.challenger and server.verify_ahead == 0
    server.close()
    own = a5.proof_server(args(loop), tmp_path, [dict(window)])
    assert own.challenger and own.tree_from == []
    own.close()


@pytest.mark.parametrize("word", ["window", "own"])
def test_a_tp_ranks_row_carries_the_audits_word_for_where_its_sessions_coin_trees_came_from(tmp_path, monkeypatch, word):
    """`a5.tp_rank`'s row of the step's summary copies the audit's ``coin_trees``. The rank's window of record and its
    audit are stand-ins: the audit itself sets the word (`test_audit_two_members`)."""
    from verity_vllm.pipeline.tp import serving_rows

    monkeypatch.setattr(serving_rows, "window_of_record", lambda *_: (None, {"tp": 2}))
    monkeypatch.setattr(a5, "audit", lambda *_, **__: ({"coin_trees": word}, True))
    a = argparse.Namespace(windows=str(tmp_path), advice_opening=None, tp_commit=None, partition=None, build_root=None,
                           workload=None, port=0, coins=None)
    row = a5.tp_rank(a, {}, tmp_path, 1, 0)
    assert (row["ok"], row["coin_trees"]) == (True, word)


def test_with_no_zk_server_the_proof_server_is_spot_flocks(tmp_path):
    server = a5.proof_server(args(None), tmp_path, [{}])
    # SPOT[Flock]'s own environment (`a0.DIRECT`), where a name whose value is None is unset
    direct = a5.a0.DIRECT
    assert server.verify_ahead == 2 and all(server._env.get(k) == v for k, v in direct.items()) and None in direct.values()
    server.close()
