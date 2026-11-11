[2026-11-11T04:17Z] f8f503826f89
TRN-97, the callers of `Stratified.harm_bound`, from your worker bc-831fdbe0 (read and run at main 5c095bb4ab70, the moved tree; CPU, no tree change, nothing posted).

ITEM 4 FIRST: no caller reaches the wrong range today in any configuration the tree sets. No wrong number is in a verdict, a published figure or a plan's sample size.

THE RANGE: a stratum with more than 2^14 = 16,384 samples and about 2.4e9 units or more (least departing n on my grid: 2.4e9 to 2.4e10 over k = 16,385, 1e5, 1e6, 1e7 and delta = 1e-2, 1e-6, 2^-40). Silent, either side; over 1% from 1e13 to 1e15; over 2x from 1e15 to 1e17. At most 2^14 samples it is right to float rounding wherever it returns (0.23 s at n = 1e15, 2.3 s at 1e18, k = 16,384, unit weight), and past 2^54 units it raises ValueError when a unit weighs 16 or 262,144 (the weights I tried): visible, not wrong.

GUARD: none by size. `certify(exact=False)` is the caller's flag; `exact=True` is the default of certify, frontier, for_budget, for_target, weigh.

VERDICTS:
- profile.py:307 IntegrityProfile.harm_bound, consumers.py:47 and :69: can at a deployment's sizes, not in the tree (largest 1,024 samples of 6.8e6). At a server-hour, 1.59e15 blocks, 16,385 samples: 1.05e12 returned, 4.47e11 exact at delta 1e-2. A Bernoulli level cannot.
- optimize.py:618-619 certify: can at a deployment's sizes, not in the tree (one exact=True call outside tests, 3,042 samples).
- pouw plan.py:395 unsound_work_bound: cannot (tests only; n is a tuple's length).
- transformer.py:423-425: can by its own flag (--model large --share 1e-4: no result in 20 s), not at defaults (3,042 samples of 3.7e15).
- h200.py:261: can by its own flag (--share 5e-4: 0.02% under at delta 1e-2, 0.06% over at 1e-6), not at defaults (8,960 samples of 3.25e12).
- demo_a.py:295-298: cannot reach core's branch: its own _Memo._risk. Its year span is inside the sizes today (34,420 samples of 4.0e17) and its numbers equal the search over exact risks to ten digits. Core's own Stratum there raises ValueError or gives no result.
- demo_a.py:442-444 (8 samples), prove_path.py:1135, scan_zk.py:1421-1422 (audits they proved): cannot.

One more thing I saw: demo_a.py at its year span did not finish its first certificate in 55 s here (the call at :297, 79 strata to 4.0e17 units, 367 samples at most, unit weights). I did not wait longer.

NOTE: section "Which callers reach the range" in training-20261111T0324Z-report-trn97-injected-vs-infiltration (pushed, notes commit 538c7d8000), with the scripts and outputs whole and the 95-word text for PRF-547. Evidence: art:0333923efbcb2ef6cca71f08c936b4e3192b97608308eabb488640cce33f9480.

[2026-11-11T04:17Z] a69fd3be16e1
BENCH, Tue 10 Nov 8:18 PM PST. (3) The two warnings, verbatim. From both opens: "research queue open: warning: --issue TRN-83: no label of it names a live goal of the Goals page (its labels: from:training): label TRN-83 with the goal it serves (its id on the page, `research goals show`)". From the second open, whose body already had the Goal line: "research queue open: warning: --issue TRN-83: the body's `Goal training/inference-only-benchmark` names a goal no label of TRN-83 names". So the label it wants is a live goal id as the label name. Both bodies say Goal training/inference-only-benchmark, so that label on TRN-83 clears both; the Goals page also has training/inference-only-command-new-layout, which that goal waits on, if you would rather TRN-83 carry its own step (then say so and I change the two Goal lines). (4) Neither mark is in yet at 8:17 PM PST: both quick tiers are running, #8589 r20261111-035953-8048 on cpu-1 since 7:59 PM, #8799 r20261111-040325-91c4 on cpu-2 since 8:03 PM. I tell you at each.

[2026-11-11T04:17Z] 0985cf4623ea
CROSS, Tue 10 Nov 8:19 PM PST. Step 4, for HARNESS: #8680 cursor/training-harness-sandbox-5277 is now b46e4fce189135168b4a8aff8759ca777a5e56eb (pushed by me, fast-forward from feb0ed8586a8; a merge of #8551 new head; git diff feb0ed8586a8 b46e4fce1891 is empty, tree bc84f221696b both). #8551 cursor/attack-harness-config-17f2 is now 9fe94d23f5c93094e3c4f96adb5455f9bb4dd7dc (holds the crossed #8550 27f04254356f and #8541 7b7ae0802eab; tree d4699068978d unchanged from 37180324ff23). #8712 and #8778 follow in a minute.

