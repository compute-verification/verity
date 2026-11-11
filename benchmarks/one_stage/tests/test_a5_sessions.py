"""a5's sampled members' sessions run one at a time: outer proofs go through the ZK server one at a time (AGENTS.md), so
``--session-jobs`` takes only 1, kept so the commands that pass it still run, and the proof server an audit proves its
members on behind the ZK server (`a5.proof_server`) verifies none ahead and starts every member's Lean session loop on
the one environment its members agree on."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def test_behind_the_zk_server_the_proof_server_proves_one_session_at_a_time_on_the_members_one_environment(tmp_path):
    a = argparse.Namespace(bin="flock-circuit", firewall="flock-firewall")
    window = {"partition_file": tmp_path / "partition.json", "program_file": tmp_path / "program.json"}
    server = a5.proof_server(a, tmp_path, [dict(window), dict(window)])
    assert server.verify_ahead == 0 and server.zk and server.out == tmp_path / "proof-server"
    want = {"FC_FIREWALL": "1", "FC_FIREWALL_LEAN": "flock-firewall", "FC_FIREWALL_STATEMENT": "verity/flock-circuit",
            "FC_FIREWALL_HELLO_ARGS": f"--partition {tmp_path / 'partition.json'} --program {tmp_path / 'program.json'}"}
    assert {k: server._env[k] for k in want} == want
    server.close()
    # members that would set their loops up differently are not one stream's: each is a process pair, as a plain audit's is
    assert a5.proof_server(a, tmp_path, [dict(window), {**window, "statement": "verity/flock-circuit/tmpl"}]) is None
    assert a5.proof_server(a, tmp_path, []) is None
    assert a5.proof_server(argparse.Namespace(bin="flock-circuit", firewall="flock-firewall", plain=True), tmp_path, [window]) is None


def test_with_no_zk_server_the_proof_server_is_spot_flocks(tmp_path):
    server = a5.proof_server(argparse.Namespace(bin="flock-circuit", firewall=None), tmp_path, [{}])
    assert server.verify_ahead == 2 and server._env["FC_FIREWALL"] == "0" and "FC_HOST_WITNESS" not in server._env
    server.close()
