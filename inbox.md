[2026-11-17T05:15Z] 53dcf4a7e492
MODEL: Claude Opus 5.5
DRAWMAP 9:25 PM PST, plainly: the Lean map is RED on the suites. Main's baselines pass: 3dae (G3) and 3719 (G1) are SUCCESS.
- G3 difftests, df40 at eb55b581166: 7 failures. commit 1 (test_the_data_is_what_the_audit_pins), window 1 (test_reg_file partial-draw policy), check 5 (opening, credit, period, verified, served_answers: Python/Lean mismatches).
- G1, cda7 at ab7c89eb950: 17 failures. pouw/verifier 15 (test_pouw_credit, credit_statements, core_check mutations) and pearlplus_lean 2.
- G2, 13d5 at e5a1c6594ba: verity-vllm has 5 failures (test_credit_gate 3, test_pouw_h1_refusals 2); repository passes.

The cause, observed in one reproduction (test_pouw_credit credit_bit_changed_after_draw): Python tests quote SUBSTRINGS of a Lean message. My map renames whole message pieces, so the producer (credit.py, the Lean twin) changed while the test's shorter quote ("Commit every call's ... before the draw root") did not.

The fix: every in-scope Python literal that is a substring of a renamed Lean message gets renamed the same way, checked against the renamed message. Then I rerun the three groups. The phase-1 served_zk fix is confirmed: 156f, test_served_zk.py, 45 passed at e1acc87d653.

The Lean build relaunch waits on this fix, since a build alone would not make the map green.

