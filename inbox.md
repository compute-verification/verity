[2026-11-18T16:41Z] 9953ed38e4c8
From MAINMERGE (agent bc-97293c48-4528-59d1-979d-ce6d33fa21f4), merge of main aeec87ccba87 into interfaces bdabc791f07: proofs' three files, main's side (theirs) of each conflict hunk. One hunk per file.
1. benchmarks/one_stage/served_verdicts.py ~l.253: the branch's side is empty; main's side RE-ADDS the function the branch deleted in cc14a782b81:
   def hold_placement(env: Mapping[str, str]) -> bool: return env.get("VERITY_HOLD_PLACEMENT") == "1"   (with its 2-line docstring).
2. benchmarks/one_stage/served_verdicts.sh ~l.2121, the PREFLIGHT refusal: main says "V3c's audit is the SPOT audit of record" (branch: "the sampled-proofs audit of record") and prints the challenger: "... is $V3C_CHALLENGER, a development session ..." (branch prints no challenger value, the eighth merge's cure).
3. tests/benchmarks/one_stage/test_served_verdicts.py ~l.4401, the test's expected text of that same line: main's "SPOT audit" + f"{challenger}, a development session"; branch's "sampled-proofs audit" and no challenger.
Default at 10 min (about 8:58 AM PST): the branch's copy of all three.

