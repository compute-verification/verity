[2026-11-09T22:46Z] 60d44a335ae3
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

[2026-11-09T23:04Z] fc07cc6b43d2
COORDINATOR 23:04Z, correcting my 22:46Z note: cpu-5 is NOT broken (INF-641 retracted). The clang exit 135 came from two lake builds of one commit running at once in a node's shared --cwd source tree. Rule: a run that builds Lean uses --cwd clone, and never two builds of one commit in one source tree. cpu-12 (--pool proofs) stands as I said.

