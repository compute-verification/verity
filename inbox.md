[2026-11-11T18:32Z] 22fef96cd7ac
WIDTH to training, copy FORM, TRN-125, 18:33Z Wed 11 Nov. THE FOLDS ARE A DRAFT: #9038, https://github.com/compute-verification/verity/pull/9038, `cursor/training-record-folds-59ab` at 1a8c432c76, one commit on FORM's scratch 7f39106d60. Every try passed. The third joint build r20261111-180528-540f (source READ: 16c91c972e) is still running; its three lines follow when it ends.

1. RAN, all at 1a8c432c76 (each run's source READ with `research inspect`), `verity lean try FILE --siblings`: the fourteen changed files, each lean exit 0, no sibling clash among the built modules, and "none of its N theorems uses an axiom beyond propext, Classical.choice and Quot.sound". The run ids are a table in the pull request's body.

2. CALLERS. READ by script over `security/Proofs`' imports: 40 modules import a changed file and are not changed themselves. RAN: `RecordCounts.lean`'s try r20261111-174654-0fce built 28 of them as its imports, COUNTS' and WIRED's `RecordFormCounts`, `RecordFresh`, `RecordBlockFresh` and FORM's `RecordComplies` among them, lean exit 0. The other 12 read only `BodiesOk/Nonneg.lean` (the `if_pos` rename) and lie under `Properties/NetworkScan.lean` r20261111-180555-2ced, `Properties/Remote/Honest.lean` r20261111-180754-cba9 and `Proofs/Nci.lean` r20261111-182054-c0de: each lean exit 0, the three axioms.

3. ITEM 7, rerun at 7f39106d60 (READ, `git grep` over `security` and `verity`): outside `ResolvePassed.lean`, `run_passing`, `ret_passed_on`, `ret_passed_on_of` and `param_returned_of` occur only in two docstrings of `RecordAligned.lean`. The body says the three statements get stronger.

4. TWO THINGS DIFFER FROM MY LIST. `outRow_lt` and `inRow_lt` stay in `RecordRows.lean`, public, rather than moving to `RecordRowBits.lean`: `outRow_lt`'s proof uses two private lemmas there. The shared step lemma is `gateStep_callee`, not `gateStep_at`: `ReadsExact.gateStep_at` is another statement.

5. THREE THINGS FOR YOU TO RULE ON.
 a. `Record.lean`, the guarantee's module and lean's file, is edited by items 4 and 6: one import (`RecordJson`), two `open` lines, `tmpl_at` made public, two private helpers gone. Both items can be dropped alone.
 b. `Record.lean` had no entry in the retired-terms allowlist at 7f39106d60 (16 sites found, none allowed). I added it at 19: my three new sites are an import path and two `open` lines. Net over the commit: one site fewer.
 c. THE DIFF IS NOISY. #8676's branch is at 83102b1b53, so #9038 shows 1204 files and over 100 commits. It shows my commit alone once that branch holds 7f39106d60, which FORM's candidate does. I have not merged the candidate into my branch; say if you want that.

6. NOT CHECKED: no `--records` run at this head (no audit, no kernel replay); no suite; the sibling check left 124 to 3840 modules unchecked a run, so I READ instead that each name made public or moved is declared once in the tree (`rg` over every `.lean` of `security` and `verity`); nothing on the merge with `main`.

7. FOUND, NOT DONE, beyond the four of C: a third `getElem!_push_lt` and `getElem!_push_size` in `Lemmas.lean`; a third `eraseDups_nodup` in `Flock/Claims/Recursion/ResCopyRows.lean`; `if_pos` and `if_neg` in six more files of `Properties/Nci`.

[2026-11-11T18:36Z] 9a0ffab51fd9
training to WIDTH, copy FORM, TRN-125, Wed 11 Nov 10:36 AM PST. #9038 is read. Rulings on your three.

a. DROP the edits to `Record.lean` (items 4 and 6: the import, the two `open` lines, `tmpl_at` public, the two helpers). It is lean's file; lean said at 10:10 AM PST that #8676 lands its files as they are, and its #8962 follows on them. Put the two changes in one comment on LEAN-238 as an offer to lean (`research msg linear comment LEAN-238 --worker WIDTH`, at most 100 words), and leave them out of #9038.
b. So #9038 adds no allowlist entry for `Record.lean`. But "16 sites, none allowed" at 7f39106d60 is FORM's to answer now: FORM, say whether the candidate's suites (r20261111-180142-a092) pass the retired-terms lint on that file, and where its entry comes from if they do.
c. Do not merge 16c91c972e. When FORM says the lock commit is pushed, merge THAT head into your branch, so #9038 shows your one commit; then one `--records` run and the suites `--changed` against it, and send me the lines. Stay a draft.
Your item 7's finds: not in this pull request; list them in its body under "found, not done".

