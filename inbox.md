[2026-11-11T21:32Z] 2f1b348c9a31
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

