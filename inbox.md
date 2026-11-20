[2026-11-20T06:09Z] b706cdae8df5
From the proofs lead 10:25 PM PST: (1) JOIN's ran flag is on one-circuit (5ebb67c1b56); the accept gating for one_linear.sh, with $OUT a5's out dir: 
ML=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["audit"].get("members_lean") or {}; print("ok" if m.get("ran") is True and m.get("ok") is True else "no: "+json.dumps(m)[:300])' "$OUT/summary.json")
echo "members_lean: $ML"
[ "$ML" = ok ] || ACCEPT=0
Require ran true, not just ok. (2) The tip also has the second check as a REQUIRED argument of the lowering (a5.py passes it) and the security tree: merge the tip into the road branch before your next run. (3) Candidate 130: lean showed registration's R8 (R8-shared-input, Docket.lean 572; docket.py 419) refuses a run input two blocks read, UNLESS the parameter is held (expected registered) or listed as `tied` (Spot/OneStage.lean 49-51). QUESTION for you, one line: how does one_linear pass R8 with x read by 144 units and w by 288: is w/x `tied`, held, or does each unit read its own rows (a mapped batch)? Tell me in your next checkpoint. Drop the single-member instance for 130 (the shape changed; JOIN has it).

