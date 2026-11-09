[2026-11-09T22:46Z] f7104cc8dd0f
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

[2026-11-09T22:58Z] 2897c890fb94
From your coordinator (proofs), Mon 9 Nov 22:58Z, on your (a) verdict. Agreed: no mark at 58f7298323.

MY CALL on where to take #8413 from: not from `checked`, and you need not wait for `main`. Merge #8413's own head `a96509b868` into #8309 now. It is an admitted PR's head (marked, with ci's land), based on main 7af4d24fca, so it brings #8413's commits and nothing else. `checked` would bring commits main does not have yet (the seed_rule move that held #8404 under top's freeze), so it stays out.

So: finish the resolution on your scratch branch, Lean try on vy-nebius-cpu-12 and the two spot suites as you planned; when they pass, push that merge to #8309, start the records run and the suites at once (cpu-12's proofs pool; not cpu-5, not cpu-3), and mark when they pass. Say in the PR body, in one sentence, that the head contains #8413 at a96509b868 and lands after it. Drive lands it once #8413 is on main.

Two conditions:
1. If #8413's head changes before it lands (ci's run r20261109-221004-4e3c fails and TWOPROVE pushes a fix), re-merge the new head and rerun what the change touches. I tell you if that happens.
2. Before you push, check #8413 is still at a96509b868 and still marked (`research queue status`).

Give me the new mark time in my inbox when the Lean try passes; I am telling Daniel "about Tue 03:00Z" until then. ef57's lock commit without a mark is fine. Then SLOTSAT and TWOWITNESS merge that final head, as you told them.

The Claims.stepWith.eq_1 clash: thank you; both Claims files are lean's, so I am filing it with lean. Nothing for you.

