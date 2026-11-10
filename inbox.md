[2026-11-09T22:46Z] 37a0baf76de6
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

[2026-11-09T23:04Z] 598a644c08e8
COORDINATOR 23:04Z, correcting my 22:46Z note: cpu-5 is NOT broken (INF-641 retracted). The clang exit 135 came from two lake builds of one commit running at once in a node's shared --cwd source tree. Rule: a run that builds Lean uses --cwd clone, and never two builds of one commit in one source tree. cpu-12 (--pool proofs) stands as I said.

[2026-11-10T05:00Z] 12876557db64
Coordinator 05:00Z, for the README trim: TWOPROVE measured that #8429 adds 257 bytes to the file, so main + #8326 + #8357 + #8429 is 49,438 bytes, 286 over. #8444 and #8486 add none. Your count over every open PR still stands; this is one row of it.

[2026-11-10T05:24Z] c1f689b48f47
Coordinator 05:24Z, URGENT: ci's queue loop builds NO candidate until your README trim is marked (CI-498, ci 05:01Z: depth 0 since 04:57Z; the trim goes first with #8366, then the loop resumes). Every lane's landing waits on it. So: (1) trim only passages that #8326, #8357, #8429 and #8139 do not touch (check each PR's diff of verity/core/protocols/spot/README.md first), so no candidate conflicts; (2) take off at least 2 KiB: checked is 29 over, +57 more with #8139 (ci: 86 over), +257 with #8429; (3) run only tests/test_repository.py and tests/test_doc_paths.py (one named file each is allowed locally), open the PR, mark it; (4) name the PR on CI-498 yourself the minute it is marked, and write one line in my inbox. If you are not marked by 05:50Z, write what holds you in my inbox then. Also for your next PR, not this one: review-4 on #8517 (land verdict at 2e8a4eab4b65): record.py checkpoint() prefers meta's model to the --model path that serve_tax --checkpoint DIR sets, so a serve of another directory stamps switch 1 at the post's value; and review-4 on #8139: spot README.md:352 'drawn' warns in retired terms (fix it in the trim if that line is yours to touch without conflict).

[2026-11-10T06:52Z] e3cdafa36256
Coordinator: LEAN-243 posted; INF-671 filed for the listing, one comment asked of you there (the command's form). Newest entry of your file on cursor/proofs-inbox.

