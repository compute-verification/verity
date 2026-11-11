[2026-11-11T03:05Z] 67f3af6aad4f
From grade (bc-e5c7e459-1e46-59b9-990b-db32cd1f8f5a), Tue 7:06 PM PST, to your 6:47 PM and 6:24 PM messages. All five items done; I end my turn after this and wait for the move.

1. CORRECTIONS, done 6:53 to 7:00 PM PST. #8778's body: a correction block with the before/after table. #8712's body: a correction under the line that quotes them. Both heads unmoved (6d567e320d04, ae52eca38092), both still "ready" in `research queue status`. TRN-89: a correcting comment (6:54 PM). Note 20261111T0300Z-checkpoint-trn94-correction-two-sensitivities names the six earlier notes it corrects. Each says before and after and that S, rho, p and the bound of record are equal.

2. THE ONE RE-TRACE: it needs no GPU and no te_core release, so I started one, r20261111-025816-77c3 (vy-nebius-cpu-1, source 7f986124ec4), for r20261104-175922-78f9 (the shortest trace of the eleven). It has ended, I ran it. The trace and the comparison passed: the old rule gives the cached record back on every key but the two write times. Its last step failed (rc 1): the builder's code imports matplotlib, which the trace's environment lacks. I ran that step on my VM on the run's fetched records instead:
   row r20261104-175922-78f9@59602, shallow/depthshare, gain 1.1: reduction 510.944968863349 -> 1021.6481275190014 (cell 510.9 -> 1022); update 46.44954262394082 -> 92.87710250172739 (46.45 -> 92.88); fresh_old 340170.8364925555 -> 340171.5504106572 (cell 3.402e5 both). rho_own 16368.93537952727, rho_default 18865.452395765933, rho_served 346.6390494723149: equal. Under the old rule the builder's code gives the published row's cells exactly. Gradient's cut 34,619,392 -> 69,222,400 bits.
   So two of the 12 rows now have their value after; ten do not (r20261104-164930-eb14 has this run's island tree and published cells). None of the ten is started.

3. THE RESTART NOTE, 20261111T0304Z-handoff-trn94-self-copy-restart, answers your four:
   (a) te_core/capture/structure.py lines 508-509, the same in 4.12.11 (b1fd3179a8ca, which made the stored traces), 4.13.4 (origin/cursor/covert-witness-4207) and 4.13.5 (origin/cursor/training-replot-55a1, the published builder). te_core is not on main. I created neither branch and push to neither: the release is the replot worker's or yours.
   (b) No GPU: the trace is fake tensors on CPU, 6 to 33 s a run for these twelve. The scorer's pack step reads the scorer's state on its node; mkpack.py builds a pack without it, from the store and git.
   (c) The command and what it reads, in full. Two of the ten are m-t20 and need m-t20's start.json from the store.
   (d) By reading, not run by me: trace40.py take, trace40.py ledger --over, replot.py node, replot.py check. My job's records name the old te_core tree in their manifest, so they are evidence of the after values, not records to take.

4. THE REVIEWER'S SLOW TEST, in TRN-94's branch at 7f986124ec4 (pushed): test_a_step_among_its_neighbours_is_the_step_alone_with_its_circuit is marked slow. It compares two captures of a two-block model against each other, and a smaller model has no neighbour block to compare, so I marked it rather than shrink it. My own new test is made smaller (one block's two matrices). Suites at that head, r20261111-024801-af54, I ran it: training_slowdown 294 passed, grader 28 passed and 3 skipped, "2 suites: 2 passed (0 cached), 0 failed". The runner still lists two others over 5 s without the mark, which I left: tests/test_against_counter.py::test_a_trace_is_of_its_runs_program_where_its_product_shapes_are_the_counters (6.6 s) and the grader's tests/test_grader.py::test_a_handover_reads_back (6.4 s). Say if you want them marked in this branch.

5. TRN-94: In Progress, one comment (7:05 PM PST). No PR. The body is drafted in my store (trn94/pr-body.md). Retired terms against 6d567e320d0: no file's count rose.

ONE CORRECTION TO MY TABLE OF 6:42 PM: I gave te_core 4.12.11's rule as line 505. It is lines 508-509 there too.

WAITING ON: your message at the move (merge main into #8541, #8712, #8778 and this branch, rerun, mark again), and your word that #8778 is on main before TRN-94's PR.

[2026-11-11T03:06Z] 968f31cb6e7c
From grade, Tue 7:08 PM PST. One sentence of my 7:06 PM message, item 4, is wrong; the change itself stands. I wrote that the neighbours test "compares two captures of a two-block model ... a smaller model has no neighbour block". Its neighbours are steps, not blocks: it captures the step alone and as the second of three, on a top block over a frozen one. It needs the three steps for what its name says and the frozen block for its last assertion (affected share below sigma below 1), so I marked it slow rather than shrink it. No reply needed.

[2026-11-11T03:07Z] ff6b03f67d5a
ONSITE, TRN-95, early word as you asked (all run by me at main a930252c4993, scratch script over benchmarks/rho, seconds on the VM): the remote reference's printed slowdown DOES move by more than its last digit at proofs' p.
reference-remote-h200, week / month / year, one datacenter = the world to every digit:
- registry eps_agg 1e-4: 2.99 / 3.03 / 3.04 (delta printed 2.7e-8)
- p = 3*2^-40, eps' = 2p/delta at delta 1e-6 (eps' 5.5e-6): 2.99 / 3.03 / 3.04, unchanged
- p = 2^-30, same delta (eps' 1.9e-3): 2.97 / 3.01 / 3.02
- p = 2^-25, same delta (eps' 6.0%): 2.54 / 2.57 / 2.57
- eps_agg's own high 1e-3: 2.98 / 3.02 / 3.03; its low 1e-6: unchanged
- holding the calculator's own delta 2.7e-8 instead (eps' = p/delta): p = 2^-30 gives 2.71 / 2.75 / 2.76; p = 2^-25 gives eps' > 1 and the bound is 1.
The printed digit holds up to eps_agg 3.5e-4 (week), 4.8e-4 (month), 6.7e-4 (year): p up to 2^-32.4 / 2^-31.9 / 2^-31.5 at delta 1e-6, or 3.5x / 4.8x / 6.7x the quoted p at the calculator's delta. Set by min-cut throughout; the mover is skip.certs = eps_agg.
Reading so far (read by me): p, delta_s, delta_bind, delta_ks are parameters under hypotheses in DampenerRemotePeriod, PouwPerWindow and CircuitCredit; '3 * 2^-40' is a term only in the honest witness, whose wrong set is False, so nothing backs it there. The calculator takes 2^-40 from core's WINDOW_DELTA (the sample's delta_s) and writes 2 x it as window_delta_crypto for delta_bind + delta_ks, status derived. Write-up follows.

