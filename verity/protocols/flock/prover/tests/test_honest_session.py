"""A recorded honest session held to the honest worker's Lean (`Flock.Firewall.honestReqs`, `FirewallHonest.honestRows`):
every request `rec_live` relayed to Lean's live firewall on an honest recursive session of the unmodified prover, in order,
the ZK server's answer to each with the salts it drew, and the challenger's live-coin part's record (`fixtures/honest-session/`, from
`../pod/93-rec-honest-session.sh`, which keeps `flock-firewall live`'s stdin and stdout through a tee: RoPE d64, one head).

The relayed requests, read back to what the worker brought (`test_honest_reqs.honest_of`), are Lean's `honestReqs` of it
byte for byte (`firewall-honest-difftest`, whose reference is `rec_live`'s encoder); each commitment the ZK server sent is
the worker's row's (`honestRows`, mirrored here, through the bytes `rowBytes` gives its digest) under the salt it drew for
it, in salt order; and the ZK server at those salts gives the live record again on Lean's requests."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import verity_difftest as D
from verity.commitments.rowleaf import ROLE_X, row_bytes, sha512_row_digest, sha512_row_prefix
from verity_flock import rec_live as RL

from test_firewall_agreement import built, lean_run  # noqa: F401 - `built` is a fixture
from test_honest_reqs import PORT, honest_of, server  # noqa: F401 - `server` is a fixture

FIX = Path(__file__).resolve().parent / "fixtures" / "honest-session"

lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def recorded() -> tuple[dict, list[bytes], list[dict]]:
    """The session's schedule, every request `rec_live` relayed, in order, and the ZK server's answer to each. The binary
    starts twice, each start a schedule line and its `ok`: `rec_live serve`'s load check of the schedule, then the
    session."""
    assert (FIX / "firewall-in.txt").is_file(), (
        f"{FIX} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/honest-session`")
    lines = (FIX / "firewall-in.txt").read_text().splitlines()
    answers = [json.loads(a) for a in (FIX / "firewall-out.txt").read_text().splitlines()]
    assert [i for i, line in enumerate(lines) if line.startswith("{")] == [0, 1] and lines[0] == lines[1]
    assert answers[:2] == [{"ok": True}] * 2 and len(answers) == len(lines)
    return json.loads(lines[1])["schedule"], [bytes.fromhex(h) for h in lines[2:]], answers[2:]


def drawn(answers: list[dict]) -> tuple[list[bytes], list[bytes]]:
    """Every salt the ZK server drew, in order, and every commitment it sent under them, in the same order."""
    salts = [bytes.fromhex(x) for a in answers for x in a["salts"]]
    sent = []
    for a in answers:
        for m in a["sent"]:
            if "commit" in m:
                sent += [bytes.fromhex(x) for _, top in m["commit"] for x in top]
            elif "round" in m:
                sent += [bytes.fromhex(x) for top in m["round"]["tops"] for x in top]
                sent += [bytes.fromhex(x) for x in m["round"]["commits"]]
    return salts, sent


def honest_rows(schedule: dict, h: dict) -> list[list[int]]:
    """`FirewallHonest.honestRows`, on `rec_live.top_rows` and `message_rows`: each table's level-0 top in Commit's order,
    then each round's caps after the schedule's copies of level 0's top (`level0`), and its message."""
    rows = [r for _, top in h["roots"] for r in RL.top_rows(bytes.fromhex(top))]
    for i, r in enumerate(schedule["rounds"]):
        items = h["rounds"][i]["items"] if i < len(h["rounds"]) else []
        caps = [bytes.fromhex(it[-1]) for it in items if it[0] in ("bind", "cap")]
        rows += [x for cap in caps[r["level0"]:] for x in RL.top_rows(cap)]
        rows += RL.message_rows([e for it in items if it[0] == "observe" for e in it[2]])
    return rows


def row_message(row: list[int]) -> bytes:
    """`FirewallHonest.rowBytes`: the bytes a row's `sha512/row/v1` digest hashes (role x, 16-bit words)."""
    return sha512_row_prefix(ROLE_X, 16, len(row)) + row_bytes(row, 16)


@lean
def test_rec_lives_relayed_requests_are_honest_reqs_byte_for_byte(server):
    """Every request of the session, its Pings, Register and Proofs included and each Open where the worker sent it: Lean's
    `honestReqs` of what they carry is those requests, and `rec_live`'s encoder agrees."""
    schedule, reqs, answers = recorded()
    assert not any("stop" in a for a in answers)
    assert [b[0] for b in reqs].count(RL.REQ_ROUND) == len(schedule["rounds"]) and reqs[-1][0] == RL.REQ_FINISH
    kind, out = D.replay(PORT, {"schedule": schedule, "honest": honest_of(schedule, reqs)}, server)
    assert kind == "output" and json.loads(out) == [b.hex() for b in reqs]


def test_the_firewalls_commitments_are_of_the_workers_rows():
    """As many rows as the schedule's (`rowCount`), each commitment the ZK server sent `hm96-sha512`'s of its row's digest
    under the salt it drew for it, the digest `sha512_row_digest`'s and of `rowBytes`' bytes."""
    schedule, reqs, answers = recorded()
    salts, sent = drawn(answers)
    rows = honest_rows(schedule, honest_of(schedule, reqs))
    count = sum(schedule["roots"].values()) + sum(sum(r["tops"]) + r["commits"] for r in schedule["rounds"])
    assert len(rows) == len(salts) == len(sent) == count > 0
    msgs = [row_message(r) for r in rows]
    assert all(hashlib.sha512(m).digest() == sha512_row_digest(r, 16, ROLE_X) for m, r in zip(msgs, rows))
    inst = RL.hm96.SHA512
    assert [inst.commit_string(inst.default_key, hashlib.sha512(m).digest(), y) for m, y in zip(msgs, salts)] == sent


@lean
def test_the_firewall_gives_the_live_record_on_lean_honest_reqs(server, built):
    """`flock-firewall` (`Flock.Firewall.run`) on Lean's requests for the session, at the salts the live firewall drew,
    finishes with the challenger's live-coin part's live record (its coins and the link's points, which it samples, left out)."""
    schedule, reqs, answers = recorded()
    salts, _ = drawn(answers)
    kind, out = D.replay(PORT, {"schedule": schedule, "honest": honest_of(schedule, reqs)}, server)
    assert kind == "output"
    [got] = lean_run([("honest", schedule, [bytes.fromhex(x) for x in json.loads(out)])], [salts])
    live = json.loads((FIX / "public.json").read_text())
    live["link"] = {k: v for k, v in live["link"].items() if k != "points"}
    live["rounds"] = [{k: v for k, v in r.items() if k != "coins"} for r in live["rounds"]]
    assert got["record"]["end"] == "finish" and len(got["record"]["rounds"]) == len(schedule["rounds"])
    assert {k: live.get(k) for k in got["record"]} == got["record"]
