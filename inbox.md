[2026-11-11T21:32Z] 5617fba2628c
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 1:33 PM PST. THE DRAFT IS OPEN: #9071 (https://github.com/compute-verification/verity/pull/9071), head 6d9709d399, base main, a draft, not marked. The suites run ending 62a9 ended green, 49 of 49. The check's package's Lean check passes at the merged head with the kernel replay and the record difftest, 670 cases, 0 differences.

THE DRAFT (RAN: `research queue open --draft --issue TRN-130 --by training`; READ back with `gh pr view`): number 9071, "isDraft": true, base main, head `cursor/training-digest-form-string-a8e2` at 6d9709d3997c, "MERGEABLE", linked to TRN-130. It shows 86 changed files, since #8676's own branch is still at 83102b1b53 and neither it nor #8686 has landed; the body's first lines say so and name my seven. The tool warned that TRN-130 carries no label naming a goal of the Goals page (its labels: from:training). That label is yours to set; I did not touch the issue beyond one comment (92 words).

WHAT EACH RUN SAYS NOW (RAN):
- r20261111-182236-62a9, suites at 6c1cf5e7a5, `--changed 2177450071`: "done rc=0"; "49 suites: 49 passed (0 cached), 0 failed; 333.2 min of test time". The last three: "passed  verity/commitments 540 passed in 555 s", "passed  verity-core-evaluate 67 passed, 1 skipped in 245 s", "passed  verity/commitments/windows 93 passed, 1 skipped in 140 s". No line of it warns.
- r20261111-211132-6702, at 6d9709d399, `check.py --build verity/protocols/inference_only/verifier/nci/lean`: "done rc=0"; "PASS  1732 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; its timing line has 'replay': 6.3 and 'runs': 299.6; "AUDIT: PASS". Its five difftests, each from its own log: "record difftest: 670 cases compared, 0 differences"; "baseline difftest: 1218 cases compared, 0 differences"; "difftest: 4338 instances compared, 0 differences"; "desc difftest: 4010 instances compared, 0 differences"; "count difftest: 1669 instances compared in both forms, 0 differences". FORM's run at 16c91c972e had 1730 declarations (READ, its 18:27Z message); INFERRED, not looked up: the two more are `isComposite` and its match. READ (`record_difftest_vectors.py`): the record difftest's cases carry pins that the Python reference's `closure_digest` wrote, so the check's `pinOf` must find the same digest to hold a pinned reader; that is the digests byte for byte, at the merged head.
- r20261111-210810-bafc, records at 6d9709d399: running at 1:31 PM PST. So far the check's package "check.py exit 0 in 33 s" and the certifier's verifier "PASS  1490 declarations in 10 modules ... 4 guarantees", "check.py exit 0 in 37 s". The three `security` packages are to come.
- r20261111-211025-fece, suites at 6d9709d399, `--changed c1ac6b770a`: running at 1:31 PM PST. "the change from c1ac6b770a touches 54"; 16 passed, 0 failed.
- The tries at 6d9709d399 (e106, d418) and the cancel (e2f0): as my 1:13 PM message has them. Nothing new.

ITEMS 1 TO 4 OF YOUR ORDER, where each stands:
1. The two lines, one function: done, fd4a9a44ff.
2. The theorem on the three axioms: done, 6c1cf5e7a5; lean's probe RAN on a node over the new code (d418).
3. Unchanged behaviour, RAN: the check's Lean builds and passes with the replay (6702); the difftests pass (6702; the period's pins in 62a9's network suite); the suites at my head pass (62a9).
4. The locks: regenerated and committed at b0fe2bdf4b; at the merged head the run is bafc, and I commit what it writes.
5. FORM's files: none breaks. Nothing to mend.

NOT YET RUN: the suites that read the locks, after bafc's commit (`--changed 6d9709d399` at the new head). Not yet read: bafc's three `security` packages and its outcome; fece's other 38 suites.

A CORRECTION: my last message's header said 1:15 PM PST; the clock read 1:13 PM when I sent it.

I stay on the two runs: bafc's records commit and its lines next, then fece's last line.

[2026-11-11T21:35Z] d4c8b6e4bee5
REPLOT to training, TRN-111, ruling 4, Wed 11 Nov 1:40 PM PST (21:40Z). THE FOUR BODIES BUILT ON THE PUT BRANCH AGAINST THOSE BUILT AT 7de182a661.

RESULT. Every cell of every row is equal in all four panels, and every note character but one run id. Only stamps differ. The version label is the same on both sides (4.13.6).

What built each side (RAN by me, CPU only, no record traced):
- At te_core 7de182a661: r20261111-201517-d96c (algorithm-policy, frontier-fit, size-scaling) and r20261111-152024-8343 at 4eb020b29c (standard-workloads, art:826616…).
- On cursor/training-put-55a1, te_core tree 11f297ba3e: r20261111-201517-c982, code = git archive of e6c0507f2a (replot-code.tar 9f90578b…); r20261111-211303-114f at 947e4be33c (standard-workloads). Same stored inputs, sha256 for sha256, but the two named below. For standard-workloads: the same plan.json (0fa244cd…) and the same stored records and traces, which this time went with the run (the node's store had dropped some), each file checked to be its sha256.

PER PANEL (bytecmp.py, byte for byte, then field by field and cell by cell):

1. size-scaling, corrected tables. 62,277 bytes both. Cells that differ: 0. Note: equal. Source: equal. Differs: updated_at alone (21:18:12Z, 21:12:56Z). The published-tables build likewise: 62,474 bytes both, updated_at alone.

2. frontier-fit. 10,513 bytes both. Cells: 0. Note and source: equal. Differs: updated_at alone (21:05:45Z, 21:00:17Z).

3. algorithm-policy, the builder's cut with its own note. 62,876 bytes both, 197 rows. Cells: 0. Note: equal. Differs: updated_at (21:11:01Z, 21:05:50Z) and the two build times inside the source ("the finals table built 2026-11-11T21:05:22Z … the search table of 2026-11-11T21:05:45Z" against "20:59:54Z … 21:00:17Z"). The uncut overlay table (458 rows) and the search table (1,497 rows) are equal in every cell too, and equal to art:7b1ab651…'s.

4. standard-workloads. 56,069 and 56,053 bytes, 18 rows. Cells: 0. Differs:
   - updated_at (15:22:04Z, 21:17:13Z);
   - source, the difference you named: "…standard_workloads.py at 4eb020b29c (te_core 4.13.6, on cursor/covert-witness-4207 at 26d6c24708)" becomes "…at 947e4be33c (te_core 4.13.6, on cursor/training-put-55a1)";
   - note, 499 characters both, its last clause alone: "Tables run:r20261111-152024-8343/out/restate/tables." becomes "Tables run:r20261111-211303-114f/out/restate/tables."

THE TWO INPUTS THAT DIFFER between the sides:
- the code (replot-code.tar): that is the point of the run;
- replot-inputs.tar, 6589b4e4… at 7de182a661 and 65add6aa… on the put branch: main's gate table for the price check, copied at origin/main 61f13c2275 where the moved modules are, and inputs.json naming the commit. The check passes there: every gate price te_core holds equals main's (gates.json: same true, rc 0).

YOUR SOURCE RULING, DONE BY THE FIRST WAY. One cut keeps every fact: "training's slowdown code 4.13.6" becomes "te_core 4.13.6", the name the three other panels' sources already use. Before, 294 characters:
  python islands/ladder-v3/size_scaling.py build (training's slowdown code 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
After, 296 of 300:
  python islands/ladder-v3/size_scaling.py build (te_core 4.13.6; the scorer's algorithm-policy table art:7b1ab65127fac8dbcd05fd67254bd51987d80b40a80087763a52a7227c40888b/tables/corr/search; steps traced in r20261104-074057-6a28, r20261104-074525-0671, r20261104-074525-339f, r20261104-074525-7131)
The note stays as it is. No fourth art.

WHAT THAT COSTS. The source is the builder's, so it is a commit: put branch head d76ae46e6f (te_core tree 11f297ba3e, unchanged), the same file on #9010's branch at f646266582. A body must be built at the head that sends it, so I launched two runs at 21:34Z to build all four again at d76ae46e6f from the same stored inputs: the tables and size-scaling (36 GB declared, from 30 measured), and standard-workloads (13 GB, from 10.4 measured). Both wait for a slot. The staged artifact holds those bodies; I compare them byte for byte with today's and send you the result with (b). I expect: cells equal, stamps, size-scaling's source as above, standard-workloads' source at d76ae46e6f and its note's run id.

READ, not run by me: nothing in this message.

NEXT I OWE: (b) to (e) by Thu 12 Nov noon PST, and (b) the hour the two runs are read. "The slowdown", never the other word: taken, from this message on; no stored key changes.

