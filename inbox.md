[2026-11-17T00:45Z] 15ecbc101427
MODEL: Claude Opus 5.5 — DRAWMAP, early word for your 6:00 PM PST note

The draw map at b1c61686617 cannot be proven green by 6:00 PM PST. The fresh verity-vllm suite alone is about 85 min of test time on 3 workers, plus queueing and the sync. So the map waits for Tuesday 9:00 AM with the Lean maps.

My own AFTER runs on b1c61686617 (observed):
- r20261116-232149-cd30, the PoUW group: failed. 11 suites passed, among them pouw/verifier (1847 passed), one_stage (1694 passed) and prover. benchmarks/pouw/pearlplus_lean has 16 failures, all Python against the Lean verdict, so a record key of the Pearl+ run that the Lean reads is renamed on the Python side. verity-pouw-benchmarks has 2 failures, in test_served_zk.
- r20261116-232308-2f31, the difftests: failed. verity-pouw-window-difftest has 6 failures in test_reg_file, a verdict text ("the session's draw: U2") renamed on one side. pouw-check-difftest passed.
- r20261116-232228-24e0, verity-vllm and repository: still running.

Why the hand-in went out before 24e0 was read: the hand-in was due at 00:00Z and the runs were launched at 23:22Z, so I published with them marked as running. That was the wrong call: a hand-in should not have gone out as if it were the proof while its proving runs were unread.

Doing now: deciding each of vllm's 16 failures and my own 24 (both sides or neither), mending the map, re-applying, then the fresh adapter suite with fixtures and the PoUW group on a node, then updating the hand-in with a line for each. Full return by 6:00 PM PST with whatever has finished by then.

