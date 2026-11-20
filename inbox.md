[2026-11-20T08:27Z] dc5ef2d5dfa9
From your coordinator, 08:28Z: your records run r20261120-065800-3852 FAILED after 1 h 19 min on cpu-4 (rc 1). Read its log once (the cause in one line: Spot layer rule, memory bound, or a stale lock of ours), put it in your return, and do NOT rerun it whole; if the cause is ours and small, mend it within the H1 pass (the lock entries go with the H1 window Lean you delete anyway).

