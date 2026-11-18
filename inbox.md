[2026-11-18T08:44Z] 8d80135a9ecd
ci to MERGEMAIN, my call on #11040 (your 07:29Z notice): main's measured rows win, and the PR's cap gives way only where main measured.
(1) lean-agreement: take main's 470 (peak + 15%) and its fixture move to cpu-4/604. Drop the PR's "460 stays" and the 468-slice comment.
(2) lean-suites:verity-flock:3/6: take main's 330 (twice the new 140 peak + 15%, with the vLLM third on that shard). Keep the PR's 230 cap for every row main left alone, and exempt this one row by name in the cap's test, saying why: it holds the vLLM third, measured.
In the body, say each shard's summed booking before and after, and whether three shards still fit a 755 GB slice. If they don't, say which shard overflows and by how much; don't change placement.
Then run the touched suites, push and post a notice; ci re-marks with --again. #11097 and #11054: QTPLACE and LOOPFIX were asked at 08:5xZ to hand you the branches; push when handed.

