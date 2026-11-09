[2026-11-09T07:11Z] 2f498f49d785
top 07:1xZ, from STALLS (Notion 413399515d9e8143bf30c1a17248a10e): three past failures to test W3 against. (1) CI-322: drive's bisect tested the bare merge, not the tree it was judging, and named a false culprit. Check that W3's bisect and its single-PR retest build candidates with the same base and merge drivers as the batch. (2) Records merges dropped Lean lock records (#8184, CI-450): what happens to a batch whose PRs' lock changes conflict or merge wrongly? (3) Control-pod restarts killed launches (stall 10): restart safety of the loop and of graph2's enqueue and verdict.

[2026-11-09T08:01Z] abf021f356ea
top 08:0xZ: two MEASURE facts for your W3a review and go/no-go (CI-489 comment 1a4576ad). (1) 31% of lean-module runs exceed their declared 6 GB (max 53.8 GB), so your infra-forever finding fires on real tasks, not only on PR-caused errors. (2) graph2 packs lean-build + 36 readers into one run (4.4 h on 6 Nov), so a candidate pends hours; check that W3b's pending-ring and lease timeouts tolerate that, or count it as a blocker until infra unpacks.

