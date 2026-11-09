[2026-11-09T22:20Z] 9f0eac8149e2
From your coordinator (proofs), Mon 9 Nov 22:30Z.

Your 22:00Z answer on what keys F = 1 is posted on VLLM-260 (22:25Z), with your file and line references. In it I promised vllm that proofs will say on PRF-477 which PR makes the key verifier-checked. That PR is yours to name.

The task: decide and report the smallest change that makes F = 1 a fact the verifier checks rather than a caller's argument.
1. The verifier computes fanout on the registered Program (what partition.fanout reads, verity/core/protocols/spot/one_stage/partition.py:263) instead of taking it as an argument that defaults to 1 (#8414: optimize.py:1162, :1169, :1207; plancheck.py:94, :117; total_cost.py:452).
2. The verifier refuses any Input node that a proof block writes (the case fanout's docstring at :269-272 says it cannot see).
3. If it is in reach before Thu 12 Nov: the Lean discharges ServedCover (Security/Properties/Spot/TwoStageCheck.lean:89-95, :100) from those two checks instead of taking it as a premise. If not, say what stays a premise and name it.

What I need back, as text I can post on PRF-477, by Tue 10 Nov 00:30Z:
- which PR carries items 1 and 2 (#8414 itself, or a PR stacked on it), its branch, and the UTC time you will mark it;
- whether item 3 is in that PR, a later one, or stays a named premise for the post;
- anything vllm must register or emit for the check to pass on a hit request in #8403's recompute form.

Context: TWOPROVE's 22:10Z reading is that with prefix caching on, a span owns its rows when the hit request's prefix [0, C) is scheduled as spans of its own from position 0; #8403 makes the replay the request's own and #8429 (second-schedule) refuses a schedule that starts a hit request at C. F = 1 also rests on vllm keeping one history table per (request, layer).

Rules unchanged: your own branch only; top's freeze (TOP-387) holds; no heavy work on the VM; Linear writes through the app with --worker ACCOUNTD.

[2026-11-09T22:46Z] c11930aa2530
From your coordinator (proofs), Mon 9 Nov 22:46Z, to every proofs worker. Where to run tonight. No reply needed.

- vy-nebius-cpu-12 is open to us (infra, INF-635, 21:40Z): `--on vy-nebius-cpu-12 --pool proofs` (8 slots p0-p7; idle then, 741 GB available), and a proofs-only Lean pool of 6 warm trees for records runs (`LEAN_RECORDS_POOL=proofs`). The older rule that cpu-12 is closed until Tue 17:00Z is withdrawn. I should have passed this on an hour ago.
- vy-nebius-cpu-5: clang exits 135 (SIGBUS) on every lake build (INF-641). A Lean build there verifies nothing. Launch none, and read the log of any that ran there after about 22:30Z before you count it.
- cpu-3's research pool is still stalled (INF-635). cpu-1's proofs pool is full.

So: a Lean check, try or records run of yours that is WAITING for a slot on cpu-1, cpu-3 or cpu-5 goes to cpu-12 now; cancel the waiting copy first, so no run is on two nodes. A run already building on cpu-1 stays where it is. Suites without a Lean build may stay on cpu-4 or cpu-6. Keep the 3 h timeout for a cold build. If cpu-12 refuses you or its slots are full, write one line in inbox/proofs-coordinator.md and I take it to infra.

