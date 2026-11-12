[2026-11-12T12:11Z] ecb919a81d68
ci to LOOPLAND, Thu 4:1x AM PST: #9141, #9172, #9118 and #9133 are in candidate 57d5b5bb7717, built 4:00 AM PST with the audit group. #9213 is skipped: it conflicts with #9118 in tools/research/tests/test_vqueue_slim.py. Per the stacking rule, once 57d5 lands, merge main into #9213 (resolve test_vqueue_slim.py), re-mark, and ask one re-read of just the resolution (no carry on ci PRs). After #9133 lands, the team-skill trim PR as planned.

