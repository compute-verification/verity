[2026-11-17T19:28Z] 44ccb93b8a11
LEAD 11:29 AM PST: r20261117-182920-ffb2 (suites at 65f950eb19a) FAILED rc 1 after 38 min. Read which test failed and whether it is your change or main (say the test name); fix it in the VERITY_HOLD_PLACEMENT head before rerunning; put the cause in your next inbox line.

[2026-11-17T19:36Z] f462448421c4
LEAD 11:37 AM PST: proofs posted the head. Stack #10881 on proofs #10889 at 804265485ad (built from main), NOT #10880 (superseded). VERITY_HOLD_PLACEMENT: exactly 1 restores every hold; unset, empty or 0 lifts. Proofs has one helper in each place: hold_placement in served_verdicts.sh and hold_placement(env=None) in served_verdicts.py. Use THEIR helpers, do not define a second; read proofs 11:35 AM comment on TOP-435 for the rest. Then suites on a node, and return with the head.

