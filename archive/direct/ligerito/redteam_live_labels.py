"""red-team-ligerito-3 R3-7, retargeted at what ``run.py verify-session`` uses now (87fe4faa on ``lane/red-team-ligerito-3``
exercised ``run._file_coins`` alone, which no longer feeds ``verify-session``, and so printed "1 attack(s) accepted").

``verify_session`` takes a live proof's coin slots from ``run.record_slots(stream_stmt, rec, rounds)``: the issuing
verifier's session record only (its coins, its STMT digest, the labels it received), no dump argument.  An attack wins iff
the slots a cheating prover built its proof on (the dump's ``.coins`` / ``stream_binding``, written after the coins opened)
are the slots ``record_slots`` returns, and those are not the coins the verifier issued.  The cold ``verify-dir`` of a live
dump (it claims nothing) is reported beside it: ``_file_coins`` plus its replay-label and coin-count checks.

Pure python, no torch: the committed source of ``run.record_slots``, ``run._file_coins``, ``live.batch_context``,
``live.expand_challenge`` and ``protocol.coin_commit`` is read from git at ``REV`` and exec'd unchanged; ``verify_session``'s
own source is checked to take its slots from ``record_slots`` and never from ``_file_coins`` / ``stream_binding``.
The verifier's record is simulated as ``ChallengeServerSession`` writes it.  End to end on a GPU (a real proof built on
each attack's slots, rejected by ``verify_session``): ``live_session_gpu_test.py``.

    python redteam_live_labels.py [REV]        # default HEAD
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REV = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def _text(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{REV}:{path}"], text=True)


def _src(path: str, names: list[str]) -> str:
    text = _text(path)
    out, got = [], set()
    for n in ast.parse(text).body:
        name = getattr(n, "name", None) or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None))
        if name in names:
            out.append(ast.get_source_segment(text, n))
            got.add(name)
    assert set(names) <= got, (path, set(names) - got)
    return "\n\n".join(out)


pm = types.ModuleType("pm")
pm.hashlib = hashlib
exec(_src("archive/direct/ligero/protocol.py", ["COIN_TAG", "COIN_BYTES", "coin_commit"]), pm.__dict__)
live_mod = types.ModuleType("live_mod")
live_mod.hashlib, live_mod.struct, live_mod.pm = hashlib, struct, pm
exec(_src("archive/direct/ligero/live.py", ["CHAL_TAG", "batch_context", "expand_challenge"]), live_mod.__dict__)


class LiveCoins:
    def __init__(self, slots):
        self.slots = slots


run_ns = {"live_mod": live_mod, "LiveCoins": LiveCoins, "Path": Path, "sha256": lambda b: hashlib.sha256(b).digest()}
exec("from __future__ import annotations\n" + _src("archive/direct/ligerito/run.py", ["_file_coins"]), run_ns)
_file_coins = run_ns["_file_coins"]
try:
    exec("from __future__ import annotations\n" + _src("archive/direct/ligerito/run.py", ["record_slots"]), run_ns)
    record_slots = run_ns["record_slots"]
except AssertionError:                                          # a REV before R3-7: verify_session used the dump's slots
    record_slots = None


def _calls(fn_src: str) -> set[str]:
    return {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            for n in ast.walk(ast.parse(fn_src)) if isinstance(n, ast.Call)}


vs_src = _src("archive/direct/ligerito/run.py", ["verify_session"])
vs_calls = _calls(vs_src)
structure = {"verify_session_calls_record_slots": "record_slots" in vs_calls,
             "verify_session_calls_file_coins": "_file_coins" in vs_calls,
             "slots_built_from_record_slots": "SessionReplayCoins(slots)" in vs_src and "slots = record_slots(" in vs_src,
             "per_round_label_vs_replay": 'rd["label"] != lb.decode()' in vs_src}

# ---- the verifier's session (what ChallengeServerSession opens and records) ------------------------------------------
R = 6
LABELS = [b"zc/r", b"pcs/beta", b"sc/r/0", b"sc/r/1", b"pcs/S/0", b"pcs/S/1"]   # protocol labels the prover sends with MSG k
stmt = b"lgto-stmt|v1|" + os.urandom(96)
coins = [(os.urandom(32), os.urandom(32)) for _ in range(R)]
ccom = [pm.coin_commit(r, s) for r, s in coins]
ctx = live_mod.batch_context(stmt, ccom)
issued = [live_mod.expand_challenge(ctx, k, LABELS[k], coins[k][0], 32) for k in range(R)]
record = {"batch": 0, "stmt_sha256": hashlib.sha256(stmt).hexdigest(),
          "coins": [{"r": r.hex(), "s": s.hex()} for r, s in coins],
          "rounds": [{"k": k, "label": LABELS[k].decode(), "msg_sha256": os.urandom(32).hex()} for k in range(R)]}


def dump(slots, labels, commitments) -> tuple[Path, dict]:
    d = Path(tempfile.mkdtemp(prefix="rtl3-live-"))
    (d / "p.coins").write_bytes(len(slots).to_bytes(4, "little") + b"".join(slots))
    e = {"coins_file": "p.coins", "stream_binding": {
        "statement": stmt.hex(), "commitments": [c.hex() for c in commitments],
        "openings": [[c["r"], c["s"]] for c in record["coins"]], "labels": [lb.decode() for lb in labels]}}
    return d, e


def run(name, slots, labels, commitments) -> dict:
    d, e = dump(slots, labels, commitments)
    try:
        _file_coins(d, e)
        vd_accepts, vd_why = True, None
    except ValueError as ex:
        vd_accepts, vd_why = False, str(ex)
    if record_slots is not None:
        replay = record_slots(stmt, record, R)                 # verify_session's slots: the record alone
    else:
        replay = slots if vd_accepts else None                 # pre-R3-7 verify_session: whatever _file_coins accepted
    vs_accepts = replay == slots
    if vd_accepts and [lb.decode() for lb in labels] != [lb.decode() for lb in LABELS]:
        vd_accepts, vd_why = False, "stream binding labels are not the protocol's"
    return {"attack": name, "slots_not_issued_by_verifier": [k for k in range(R) if slots[k] != issued[k]],
            "verify_session_slots_differ_at": None if replay is None else [k for k in range(R) if slots[k] != replay[k]],
            "verify_session_accepts": vs_accepts, "verify_dir_accepts": vd_accepts, "verify_dir_why": vd_why}


results = [run("honest", issued, LABELS, ccom)]

# A: per-round label grinding -- after r_0 is opened, try labels until slot 0 has 16 leading zero bits
g_label, tries = None, 0
while g_label is None:
    cand = LABELS[0] + b"#" + str(tries).encode()
    tries += 1
    if live_mod.expand_challenge(ctx, 0, cand, coins[0][0], 32)[:2] == b"\0\0":
        g_label = cand
slots_a = [live_mod.expand_challenge(ctx, 0, g_label, coins[0][0], 32)] + issued[1:]
res_a = run(f"A label grinding (slot 0 chosen: 16 zero bits after {tries} tries)", slots_a, [g_label] + LABELS[1:], ccom)
res_a["ground_label"] = g_label.decode()
results.append(res_a)

# B: context grinding -- a trailing extra "commitment" changes batch_context, so every slot, with the verifier's labels
junk = os.urandom(32)
ctx_b = live_mod.batch_context(stmt, ccom + [junk])
slots_b = [live_mod.expand_challenge(ctx_b, k, LABELS[k], coins[k][0], 32) for k in range(R)]
results.append(run("B extra trailing commitment (batch_context changed)", slots_b, LABELS, ccom + [junk]))

# control: a slot not derived from the stream at all
slots_c = issued[:2] + [os.urandom(32)] + issued[3:]
results.append(run("C control: random slot 2", slots_c, LABELS, ccom))

# record-side: a record whose shape or STMT does not match is refused, not reinterpreted
record_neg = {}
for name, rec, rounds, st in () if record_slots is None else (("one round more than recorded", record, R + 1, stmt),
                              ("coin missing", dict(record, coins=record["coins"][:-1]), R, stmt),
                              ("rounds swapped", dict(record, rounds=[record["rounds"][1], record["rounds"][0]] + record["rounds"][2:]), R, stmt),
                              ("another STMT", record, R, stmt + b"x")):
    try:
        record_slots(st, rec, rounds)
        record_neg[name] = "ACCEPTED"
    except ValueError as ex:
        record_neg[name] = f"refused: {ex}"

print(json.dumps({"rev": subprocess.check_output(["git", "rev-parse", "--short", REV], text=True).strip(),
                  "verify_session_structure": structure, "results": results, "record_negatives": record_neg}, indent=1))
bad = [r["attack"] for r in results if r["slots_not_issued_by_verifier"] and (r["verify_session_accepts"] or r["verify_dir_accepts"])]
bad += [f"record: {k}" for k, v in record_neg.items() if v == "ACCEPTED"]
if not (structure["verify_session_calls_record_slots"] and structure["slots_built_from_record_slots"]
        and structure["per_round_label_vs_replay"]) or structure["verify_session_calls_file_coins"]:
    bad.append("verify_session does not take its slots from record_slots alone")
if not results[0]["verify_session_accepts"] or not results[0]["verify_dir_accepts"]:
    bad.append("honest dump refused (the harness is broken)")
print(f"{len(bad)} attack(s) accepted with coins the verifier never issued: {bad}" if bad else "no attack accepted")
sys.exit(1 if bad else 0)
