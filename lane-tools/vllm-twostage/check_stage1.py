"""A stage1.json's coins, as the verifier checks them: `coins.coins_reason` on its evidence, under the deployment's key the
evidence names, at the window root its commit.json names, after its registration's digest, on the bytes its sample
read; and the same evidence held to another digest, which must be refused.

    python check_stage1.py STAGE1.json      # run from the tree's root
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "benchmarks/one_stage")
import a5_coins as AC  # noqa: E402
from verity.core.protocols.spot import coins as CC  # noqa: E402

s = json.loads(Path(sys.argv[1]).read_text())
ev, sess = s.get("evidence"), s.get("session") or {}
out = {"error": s.get("error"), "units": s.get("units"), "draw": s.get("draw"), "run": s["commit"]["run"][:16],
       "registration": s.get("registration", "")[:16], "session": sess.get("plan"), "trusted": sess.get("trusted"),
       "session_error": sess.get("error"), "evidence": ev is not None}
if ev:
    run, reg, read = s["commit"]["run"], bytes.fromhex(s["registration"]), bytes.fromhex(s["read"])
    out["segment"] = {k: (v if len(str(v)) < 80 else f"{str(v)[:40]}…") for k, v in ev["segment"].items()}
    out["ledger_lines"], out["session_files"] = len(ev["ledger"]), sorted(ev["session"])[:12]
    out["coins_reason"] = CC.coins_reason(ev, run, reg, read, AC.lms_verify)
    out["coins_reason_other_digest"] = CC.coins_reason(ev, run, hashlib.sha512(b"another registration").digest(), read,
                                                       AC.lms_verify)
    out["coins_reason_other_window"] = CC.coins_reason(ev, hashlib.sha512(b"another window").hexdigest(), reg, read,
                                                       AC.lms_verify)
print(json.dumps(out, indent=1, default=str))
ok = ev is not None and out["coins_reason"] is None and out["coins_reason_other_digest"] and out["coins_reason_other_window"]
print("GATE-FIRST:", "ACCEPTED, and refused at another digest and another window" if ok else "NOT ACCEPTED")
sys.exit(0 if ok else 1)
