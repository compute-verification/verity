"""A stream's failure budget held to the Lean (`core_firewall`: `Firewall.Record` and `Firewall.Release`, served by
`core-firewall-difftest`) where Python's ZK server (`verity_flock.firewall`) keeps it as deployed: an honest stream's ops
on one state directory, its index (``index-py/``) carried across them as `rec_live`'s and the relay's processes carry it,
against the Lean on the record alone, op by op; and the relay's seat of an outer session (`Firewall.seat`: its ``slot`` and
its admission on one read of the record, appended in one write) against the Lean ZK server's (`seat`: ``slot``, then
``admit`` on the record ``slot`` left), on `_release`'s sampled records."""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import strategies as st

import _release as R
import test_firewall_release_agreement as A
import verity_difftest as D
from verity.protocols.spot.verifier import service as S
from verity_flock import firewall as FW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

lean = D.server_fixture(A.port("op", R.python_op, R.op_input()))


@pytest.mark.parametrize("budget", range(FW.BUDGET_MIN, 5))
@pytest.mark.parametrize("rounds", range(1, S.ROUNDS + 1))
def test_an_honest_stream_on_one_state_agrees_with_the_lean_at_every_op(lean, rounds, budget, tmp_path, monkeypatch):
    """`test_firewall_release_agreement.test_an_honest_stream_is_never_refused_an_open`'s streams at R = `rounds` and b =
    `budget`, each run on one Python state directory, a new `Firewall` of it at each op as each process makes one: every
    op's result is the honest flow's, refusing no open, and it and the record it leaves are the Lean's on the record
    before it. Every read is the index's (`_Index`), never the whole record's, so the stream's failures and unended
    sessions the open's rule reads are the tally the index carries from op to op."""
    reads: Counter = Counter()
    read = FW.Firewall._read

    def counted(self, *a, **k):
        rec = read(self, *a, **k)
        reads[type(rec).__name__] += 1
        return rec

    monkeypatch.setattr(FW.Firewall, "_read", counted)
    for k, rng in enumerate((None, *(random.Random(seed) for seed in range(3)))):
        ops, results, _ = A._honest(rounds, budget, 3 * rounds + 2, rng)
        state, text = tmp_path / f"fw-{k}", ""
        log = state / "record.jsonl"
        for op, want in zip(ops, results):
            got = R._op(FW.Firewall(state), op)
            after = log.read_bytes().decode() if log.exists() else ""
            assert A._lean_op(lean)({"op": op, "record": text}) == {"record": after, "result": got}, (rounds, budget, k, op)
            assert got == want, (rounds, budget, k, op, got)
            text = after
    assert set(reads) == {"_Index"}, reads


def _seat(value: dict) -> dict:
    """`{"part", "record", "session"}` through Python's `Firewall.seat` on a state directory holding the record:
    `{"record", "result"}`, its result as the Lean's ``admit`` answers. A record it can't read is a rejection."""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "record.jsonl"
        log.write_bytes(value["record"].encode())
        try:
            with FW.Firewall(Path(d)).seat(value["session"], value["part"]) as why:
                pass
        except json.JSONDecodeError as e:
            raise D.Reject(f"the record: {e}") from None
        result = {"release_ms": None} if why is None else {"refused": why.removeprefix(R.REFUSED)}
        return {"record": log.read_bytes().decode(), "result": result}


@dataclass(frozen=True)
class _SeatPort(D.Port):
    """The Lean ZK server's ``seat`` through the ``op`` port: ``slot``, then ``admit`` on the record ``slot`` left."""

    def lean(self, server: D.Server, value: dict) -> D.Outcome:
        s, p = value["session"], value["part"]
        kind, text = server.call("op", {"op": ["slot", s, p], "record": value["record"]})
        if kind != "output":
            return kind, text
        return server.call("op", {"op": ["admit", s, p], "record": json.loads(text)["record"]})


@st.composite
def seat_input(draw) -> dict:
    """`{"part", "record", "session"}`: an outer session of a scene, on a record none of whose recursive sessions has a
    release schedule (this relay refuses a scheduled session, which the Lean ZK server, keeping its times, admits); half
    the time, on a record whose last line is whole, after a holder of the slot that ended before its outcome: of any
    session, of another one the record opened, or of one opened on a stream without a budget, whose stop the seat
    records (a stop only of a session opened) without stopping the outer session's stream at a budget of 1."""
    text, s, p, _ = draw(R.scene().filter(lambda sc: '"release": {' not in sc[0]))
    if (not text or text.endswith("\n")) and draw(st.booleans()):
        events = [json.loads(line) for line in text.splitlines() if line]
        opened = sorted({e["session"] for e in events if e["event"] == "open"} - {s})
        how, holder = draw(st.sampled_from(["any", "opened", "unbudgeted"])), draw(R.SESSION)
        if how == "opened" and opened:
            holder = draw(st.sampled_from(opened))
        elif how == "unbudgeted":
            out = R.python_op({"op": ["open", "w", ["L0"], None, None], "record": text})
            if isinstance(out["result"], str):
                text, holder = out["record"], out["result"]
        text = R.python_op({"op": ["slot", holder, draw(R.PART)], "record": text})["record"]
    return {"part": p, "record": text, "session": s}


def _seated(value: dict, out: dict) -> str:
    """What the seat came to: admitted or refused, and whether the slot's orphan was recorded as its session's stop."""
    added = [json.loads(line) for line in out["record"][len(value["record"]):].splitlines()]
    orphan = ":orphan" if added and added[0]["event"] == "stop" else ""
    return ("admitted" if out["result"] == {"release_ms": None} else "refused") + orphan


def test_the_relays_seat_is_the_lean_servers(lean):
    """`Firewall.seat` on sampled records gives the results and the record, byte for byte, of the Lean ZK server's ``seat``
    (``slot``, then ``admit``), and the record's rejection where the Lean's rejects it (its last line torn, or without its
    newline, which the slot's line would join): admitted and refused, with the slot's orphan recorded as its session's
    stop and without."""
    tally: Counter = Counter()

    def ref(value):
        out = _seat(value)
        tally[_seated(value, out)] += 1
        return out

    stats = D.run(_SeatPort(A.EXE, "op", ref, seat_input(), package=R.PKG), examples=400, seed=A.SEED, server=lean)
    print(f"seat vs the Lean, seed {A.SEED}: {stats}, {dict(sorted(tally.items()))}")
    assert stats.rejects >= 10, stats
    for k in ("admitted", "refused", "admitted:orphan", "refused:orphan"):
        assert tally[k] >= 5, (k, tally)
