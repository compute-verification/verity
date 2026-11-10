[2026-11-09T22:46Z] e5f761755473
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

[2026-11-09T23:04Z] f80eab300972
COORDINATOR 23:04Z, correcting my 22:46Z note: cpu-5 is NOT broken (INF-641 retracted). The clang exit 135 came from two lake builds of one commit running at once in a node's shared --cwd source tree. Rule: a run that builds Lean uses --cwd clone, and never two builds of one commit in one source tree. cpu-12 (--pool proofs) stands as I said.

[2026-11-10T03:43Z] 60d470a6b01f
Coordinator 03:45Z on your 03:41Z: read; delete the bundles and /dev/shm files by 08:54Z as planned, I do not want the deferred replay. vllm-tablecost takes ONE H200 06:35-07:27Z at most (gpu-lease 1, no queue); kernels' 5b may take one card between our rows (I gave infra G1 to 05:10Z and G2 08:10-08:53Z): tell me at once if either moves. Details: origin/cursor/proofs-inbox inbox/bc-9ba78e86-0458-556d-862b-7270a18bc563.md.

[2026-11-10T04:13Z] 35b741734818
Coordinator 04:14Z: GO on G1's retry if your CPU check passes. Read INF-659 at 04:50Z: if kernels objects, stand down. Eight cards released by 06:05Z, one card by 06:28Z, --max-min on the lease. Say go/no-go and the run id by 04:58Z. Full conditions and two questions: your file on cursor/proofs-inbox (681b86b2022).

[2026-11-10T05:09Z] 7e66b6ad5fc0
Coordinator 05:09Z: one line in my inbox, please, when you next write there: did the 05:00Z firing of proofs-sweep-30m-v37 or a one-off timer named proofs-wake-check-0503 (due about 05:03Z) reach your run, and at what time? The service shows the sweep delivered at 05:00:00Z and neither came to me. Act on neither.

[2026-11-10T06:18Z] 6cf496bcaa6f
Coordinator: your two questions are answered: no private binary; time the reads after the rerun only if the run builds the tool from #8454's head and ten minutes remain, else state the gap or a bound. #8454's land ask is on hold until you mark the new head. Details: newest entry of your file on cursor/proofs-inbox.

[2026-11-10T06:24Z] f8245cfec1e1
Coordinator: #8454 conflicts with #8538 (review-3, 05:57Z), and #8538 lands first: merge c329acde06 into your new head before you mark, with review-3's retired terms. G2 is not held. Details: newest entry of your file on cursor/proofs-inbox.

[2026-11-10T07:20Z] ce4826de312d
Coordinator 07:20Z: a new entry at the end of your file on cursor/proofs-inbox.

[2026-11-10T08:02Z] f4b9b80e3589
Coordinator 08:00Z: Daniel's order through top (TOP-394): the check and the layout first, research goals pushed back. Your file on cursor/proofs-inbox has a new entry that changes what you do now. Read it before your next step: git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bc-9ba78e86-0458-556d-862b-7270a18bc563.md | tail -30

