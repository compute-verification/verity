"""R3-7 (red-team-ligerito-3): ``verify-session`` derives a live proof's coin slots from the issuing verifier's own session
record (``run.record_slots``), never from the dump.  Pure python, no torch: the source of ``run.record_slots``,
``live.batch_context`` / ``live.expand_challenge`` and ``protocol.coin_commit`` is exec'd from the working tree.

The red team's attacks (``redteam_live_labels.py``, 87fe4faa on ``lane/red-team-ligerito-3``) wrote a post-hoc label
(A) or an extra trailing commitment (B) into the dump's ``stream_binding`` / ``.coins`` so that ``_file_coins`` re-derived
slots the verifier never issued.  ``record_slots`` has no dump argument: whatever the dump says, the replay uses the
issued coins, so a proof built on the attack's slots is checked against coins it was not built for.  The GPU end of the
same negatives (a proof built on the attack slots, rejected by ``verify_session``) is ``live_session_gpu_test.py``.

    python -m backends.direct.ligerito.live_session_test
"""
from __future__ import annotations

import ast
import hashlib
import os
import struct
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _src(path: Path, names: list[str]) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    out, got = [], set()
    for n in tree.body:
        name = getattr(n, "name", None) or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None))
        if name in names:
            out.append(ast.get_source_segment(text, n))
            got.add(name)
    assert set(names) <= got, (path, set(names) - got)
    return "\n\n".join(out)


pm = types.ModuleType("pm")
pm.hashlib = hashlib
exec(_src(ROOT / "ligero" / "protocol.py", ["COIN_TAG", "COIN_BYTES", "coin_commit"]), pm.__dict__)
live_mod = types.ModuleType("live_mod")
live_mod.hashlib, live_mod.struct, live_mod.pm = hashlib, struct, pm
exec(_src(ROOT / "ligero" / "live.py", ["CHAL_TAG", "batch_context", "expand_challenge"]), live_mod.__dict__)
run_ns = {"live_mod": live_mod, "sha256": lambda b: hashlib.sha256(b).digest()}
exec("from __future__ import annotations\n" + _src(HERE / "run.py", ["record_slots"]), run_ns)
record_slots = run_ns["record_slots"]

R = 6
LABELS = [b"zc/r", b"pcs/beta", b"sc/r/0", b"sc/r/1", b"pcs/S/0", b"pcs/S/1"]


def session():
    """What ``ChallengeServerSession`` records for one batch: its coins, the STMT digest, the label that arrived with MSG k."""
    stmt = b"lgto-stmt|v1|" + os.urandom(96)
    coins = [(os.urandom(32), os.urandom(32)) for _ in range(R)]
    ccom = [pm.coin_commit(r, s) for r, s in coins]
    ctx = live_mod.batch_context(stmt, ccom)
    issued = [live_mod.expand_challenge(ctx, k, LABELS[k], coins[k][0], 32) for k in range(R)]
    rec = {"batch": 0, "stmt_sha256": hashlib.sha256(stmt).hexdigest(),
           "coins": [{"r": r.hex(), "s": s.hex()} for r, s in coins],
           "rounds": [{"k": k, "label": LABELS[k].decode(), "msg_sha256": os.urandom(32).hex()} for k in range(R)]}
    return stmt, coins, ccom, ctx, issued, rec


def _rejects(fn) -> str:
    try:
        fn()
    except ValueError as e:
        return str(e)
    raise AssertionError("accepted")


def test_record_slots_are_the_issued_coins():
    stmt, _, _, _, issued, rec = session()
    assert record_slots(stmt, rec, R) == issued


def test_redteam_attacks_do_not_reach_the_replay():
    """A: label grinding after r_0 opened; B: an extra trailing commitment; C: a random slot.  Each builds the dump's
    slots its way; the replay's slots are the record's, so every attacked position differs from what the proof used."""
    stmt, coins, ccom, ctx, issued, rec = session()
    tries, g = 0, None
    while g is None:
        cand = LABELS[0] + b"#" + str(tries).encode()
        tries += 1
        if live_mod.expand_challenge(ctx, 0, cand, coins[0][0], 32)[:1] == b"\0":
            g = cand
    slots_a = [live_mod.expand_challenge(ctx, 0, g, coins[0][0], 32)] + issued[1:]
    ctx_b = live_mod.batch_context(stmt, ccom + [os.urandom(32)])
    slots_b = [live_mod.expand_challenge(ctx_b, k, LABELS[k], coins[k][0], 32) for k in range(R)]
    slots_c = issued[:2] + [os.urandom(32)] + issued[3:]
    replay = record_slots(stmt, rec, R)
    for name, slots, attacked in (("A", slots_a, [0]), ("B", slots_b, list(range(R))), ("C", slots_c, [2])):
        assert [k for k in range(R) if slots[k] != replay[k]] == attacked, name


def test_record_labels_bind_the_slot():
    """The label is the one the verifier received with MSG k (before it opened r_k); a record whose label was altered
    derives another coin (and verify_session also requires it equal to the protocol label the replay asks for)."""
    stmt, _, _, _, issued, rec = session()
    rec["rounds"][3]["label"] = "sc/r/1#grind"
    got = record_slots(stmt, rec, R)
    assert [k for k in range(R) if got[k] != issued[k]] == [3]


def test_record_shape_is_exact():
    stmt, _, _, _, _, rec = session()
    assert "coins" in _rejects(lambda: record_slots(stmt, rec, R + 1))
    short = dict(rec, coins=rec["coins"][:-1])
    assert "coins" in _rejects(lambda: record_slots(stmt, short, R))
    fewer = dict(rec, rounds=rec["rounds"][:-1])
    assert "rounds" in _rejects(lambda: record_slots(stmt, fewer, R))
    swapped = dict(rec, rounds=[rec["rounds"][1], rec["rounds"][0]] + rec["rounds"][2:])
    assert "rounds" in _rejects(lambda: record_slots(stmt, swapped, R))
    assert "STMT" in _rejects(lambda: record_slots(stmt + b"x", rec, R))
    assert "coins" in _rejects(lambda: record_slots(stmt, rec, 0))


def test_no_dump_argument():
    import inspect
    assert list(inspect.signature(record_slots).parameters) == ["stream_stmt", "rec", "rounds"]


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok {k}")
    print(f"{n} tests passed")
