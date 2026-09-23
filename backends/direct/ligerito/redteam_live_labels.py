"""red-team-ligerito-3 R3-7: ``run.py verify-session`` re-derives a live dump's coin slots from the dump's own
``stream_binding.labels`` / ``.commitments`` and never compares them with the issuing verifier's record.

Pure python, no torch: the committed source of ``run._file_coins``, ``live.batch_context``, ``live.expand_challenge`` and
``protocol.coin_commit`` is read from git at ``REV`` and exec'd unchanged.  The verifier's session record is simulated as
``ChallengeServerSession`` writes it (``coins[k] = {r, s}``, ``rounds[k].label`` = the label that arrived with MSG k).

    python redteam_live_labels.py [REV]        # default lane/ligerito-relation-2

Prints, per attack, whether ``_file_coins`` accepts slots that differ from the coins the verifier issued.  The rest of
``verify_session`` compares ``rd["label"]`` with ``SessionReplayCoins.labels`` (the labels the replaying verifier asks
for, i.e. the protocol's), not with ``stream_binding.labels``, so it does not see either attack.
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

REV = sys.argv[1] if len(sys.argv) > 1 else "lane/ligerito-relation-2"


def _src(path: str, names: list[str]) -> str:
    text = subprocess.check_output(["git", "show", f"{REV}:{path}"], text=True)
    tree = ast.parse(text)
    out = [ast.get_source_segment(text, n) for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.Assign)) and
           (getattr(n, "name", None) in names or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets)))]
    got = {getattr(n, "name", None) or n.targets[0].id for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.Assign)) and ast.get_source_segment(text, n) in out}
    assert set(names) <= got, (path, set(names) - got)
    return "\n\n".join(out)


pm = types.ModuleType("pm")
pm.hashlib = hashlib
exec(_src("backends/direct/ligero/protocol.py", ["COIN_TAG", "COIN_BYTES", "coin_commit"]), pm.__dict__)
live_mod = types.ModuleType("live_mod")
live_mod.hashlib, live_mod.struct, live_mod.pm = hashlib, struct, pm
exec(_src("backends/direct/ligero/live.py", ["CHAL_TAG", "batch_context", "expand_challenge"]), live_mod.__dict__)


class LiveCoins:
    def __init__(self, slots):
        self.slots = slots


run_ns = {"live_mod": live_mod, "LiveCoins": LiveCoins, "Path": Path}
exec(_src("backends/direct/ligerito/run.py", ["_file_coins"]), run_ns)
_file_coins = run_ns["_file_coins"]

# ---- the verifier's session (what ChallengeServerSession opens and records) ------------------------------------------
R = 6
LABELS = [b"zc/r", b"pcs/beta", b"sc/r/0", b"sc/r/1", b"pcs/S/0", b"pcs/S/1"]   # protocol labels the prover sends with MSG k
stmt = b"lgto-stmt|v1|" + os.urandom(96)
coins = [(os.urandom(32), os.urandom(32)) for _ in range(R)]
ccom = [pm.coin_commit(r, s) for r, s in coins]
ctx = live_mod.batch_context(stmt, ccom)
issued = [live_mod.expand_challenge(ctx, k, LABELS[k], coins[k][0], 32) for k in range(R)]
record = {"coins": [{"r": r.hex(), "s": s.hex()} for r, s in coins],
          "rounds": [{"k": k, "label": LABELS[k].decode()} for k in range(R)]}


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
        accepted = True
        why = None
    except ValueError as ex:
        accepted, why = False, str(ex)
    # verify_session's own checks on the binding: openings == the record's, and (per slot) rd["label"] == the protocol
    # label the replay asks for -- both hold here by construction; stream_binding.labels/commitments are not compared
    openings_ok = e["stream_binding"]["openings"] == [[c["r"], c["s"]] for c in record["coins"]]
    replay_labels_ok = all(rd["label"] == LABELS[k].decode() for k, rd in enumerate(record["rounds"]))
    differs = [k for k in range(R) if slots[k] != issued[k]]
    return {"attack": name, "file_coins_accepts": accepted, "why": why, "openings_match_record": openings_ok,
            "record_labels_match_replay": replay_labels_ok, "slots_not_issued_by_verifier": differs}


results = [run("honest", issued, LABELS, ccom)]

# A: per-round label grinding -- after r_0 is opened, try labels until slot 0 has 16 leading zero bits (any predicate the
# cheating prover needs of its challenge: 2^g SHAKE calls choose among 2^g challenges, every round, post hoc)
g_label, tries = None, 0
while g_label is None:
    cand = LABELS[0] + b"#" + str(tries).encode()
    tries += 1
    if live_mod.expand_challenge(ctx, 0, cand, coins[0][0], 32)[:2] == b"\0\0":
        g_label = cand
slots_a = list(issued)
slots_a[0] = live_mod.expand_challenge(ctx, 0, g_label, coins[0][0], 32)
res_a = run(f"label grinding (slot 0 chosen: 16 zero bits after {tries} tries)", slots_a, [g_label] + LABELS[1:], ccom)
res_a["ground_label"] = g_label.decode()
results.append(res_a)

# B: context grinding -- a trailing extra "commitment" changes batch_context, so every slot, with the verifier's labels
junk = os.urandom(32)
ctx_b = live_mod.batch_context(stmt, ccom + [junk])
slots_b = [live_mod.expand_challenge(ctx_b, k, LABELS[k], coins[k][0], 32) for k in range(R)]
results.append(run("extra trailing commitment (batch_context changed)", slots_b, LABELS, ccom + [junk]))

# control: a slot re-derived from a wrong opening is caught
slots_c = list(issued)
slots_c[2] = os.urandom(32)
results.append(run("control: random slot 2", slots_c, LABELS, ccom))

print(json.dumps({"rev": subprocess.check_output(["git", "rev-parse", "--short", REV], text=True).strip(),
                  "results": results}, indent=1))
bad = [r for r in results if r["file_coins_accepts"] and r["slots_not_issued_by_verifier"]]
print(f"{len(bad)} attack(s) accepted with coins the verifier never issued" if bad else "no attack accepted")
