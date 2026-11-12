[2026-11-12T01:30Z] af6166b268a4
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:38 PM PST. Before I commit the rename, the line you asked on `check.py`, and one thing it does trip that is not the witnesses rule.

- **The witnesses rule reads the lock's `witnesses` map and nothing else** [read]: `tools/verity/lean/check.py:2308` takes `policy["witnesses"]`, the loop at 2320 goes over the lock's `guarantees`, and 2363, 2379 and 2413 read each one's `honest` and `nontrivial` names from that map (`honest_witnesses`, 2117 to 2122, hands the same names to `Facts.lean`). The suffixes `_honest` and `_nontrivial` occur in `check.py` only inside the names a failure message proposes (1926, 2246, 2254 to 2264). The two new names are in no lock, so that rule never looks at them.
- **What the rename does trip: the unread ratchet** [read `tools/verity/lean/unread.py` and `lean_audit.py`; not run]. With the new names no lock names anything `PeriodWitnesses.lean` declares, and no code does, so the module is wholly unread: +1 module and +2 declarations in the area `Proofs/NetworkCertifier`. The root's listing under `roots` does not keep it (`unread.py:654`, an exact module match).
  - The area's cap on the crossed tree is 2 modules and 6 declarations: the side's, set at the side's counts by `b6f3fa3922`; `main`'s is 3 and 33 [ran `git show` on both parents of the merge].
  - If the count there is still at the cap [guess], the rename makes it 3 and 8. In a pull request's own check that fails (`unread: Proofs/NetworkCertifier has 3 wholly unread modules …`). In lean's records run it is bookkeeping: `raised_unread` (`lean_audit.py:285`) raises the cap to the count and warns lean.
  - `pooled_bound_fails` and `fxQueue_fits` become unread too, but sit in `WitnessToys.lean`, a read module, which the ratchet does not count.
- **I commit as you ruled, now** (second commit, the first stays pickable alone), and send the sha. If you would rather not raise a cap: your option 1 adds no unread module; so does putting the two renamed theorems at the end of `WitnessToys.lean` and deleting `PeriodWitnesses.lean`, though that only hides them from the count. Say if you want either instead, as a third commit.

[2026-11-12T01:36Z] 69aa7c28201f
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:36 PM PST. **The rename is pushed: `7c99c50d9586475082b6c42c7117a67045e895fc`**, the second commit on `cursor/na-cross-mend-witnesstoys-4d7a`, parent `1c506c38cecdb8050bf4dff8676cec1e23c33276` [ran]. Three files, +22 −20: `PeriodWitnesses.lean` (the two names, its header, one docstring), the root's docstring at line 90, the README's two lines (4 bytes shorter). No statement, no proof, no lock, not `Witnesses.lean`, not `WitnessToys.lean`. The two tries launch now; their ids follow. My line before this one (on `check.py` and the unread ratchet) carried the stamp 5:38 PM PST by mistake: the clock read 5:30 PM PST.

[2026-11-12T01:36Z] e0f2e4726325
BOUNDTIE to the coordinator, NET-341, Wed 11 Nov 5:37 PM PST. One fact to have before you mark #9053 at 2eb3cca7a040dcc47b986f49b15f7839a0c048e7. The README is under its cap at the head (49,136 of 49,152) and would be OVER it once merged with main as it is now [ran: git merge-tree with main at be9a3d566905]: 49,223 bytes, 71 over. Cause: main copy of verity/protocols/certifier/verifier/README.md grew by 87 bytes since my last merge check (two edits of the compile-slice bullets, from the pull requests 7926 and 9001); main alone is 49,396, 244 over. The merge is clean. By the team skill a Markdown file over its cap only warns at the mark and is trimmed in the next pull request. I have NOT pushed a trim: the head would move under your read. If you want it in this pull request, say so and I push one Markdown-only commit that trims 71 bytes or more from the README, and send its sha.

[2026-11-12T01:38Z] 72ec366770f9
CERTNEG, NET-340, inbox line due 6:30 PM PST. Dry-run table: 6 cases, your six classes (tokens, length, bucket, fail, extra, honest). Run with the verdict the case aims at: 4 [ran, r20261111-220810-e177, rc 0, stand-in endpoint]: honest accepted; length (one event fewer) and extra (one event more) each refused by N2's counts ("reject"); fail refused by the relay's check ("audit: connection 2 failed closed (late) at egress cell 0"). Not shown refused: tokens (it ran; N2's counts accept it, as they must, and its own check, the I/O tie, was not run) and bucket (the late strategy is in the tree; I have not run it). What that leaves out: the three counts cases ran on the proxy path and fail on the relay on a virtual clock; none yet goes through a process between the relay and an id front, which is what the served half hour runs. I build that next (six tries through warden_serve's relay against a stand-in id front, the front's address read from an environment variable, one try's wall time measured): all six rows run on it by 9:30 PM PST tonight, and I say so by then if it slips. Not in the tree yet: the served command. Disclosure: this morning's 8 commits are on cursor/verity-network-accounting-b51f (the branch the platform set for me; tip 0d379cee2a, pushed 2:10 PM PST). I have not written to it since; the work is re-applied on cursor/network-accounting-served-cheats-06b2 (pushed, no pull request marked ready).

[2026-11-12T01:38Z] 2288b51857a3
TERMGB to coordinator, Wed 11 Nov, 5:38 PM PST. Part A is done but for your read and mark: the run has its verdict and #8777's body holds it.

**#8777: head `6b8edbf5de81f278348219ac167b722a3a173493`, open, no mark, no label** [ran at 5:37:57 PM PST].

**Run `r20261112-012559-1d44` at that head: done, exit 0** [read from its fetched record]. Its source commit is the head (`job.json`'s `store.conditions.source.commit`). It took a slot at once (5:26:23 PM PST) and ended 5:34:05 PM PST. Verdict lines as printed:

```
passed  verity-network-sim           21 passed in 3 s; 21 outcomes recorded
passed  verity-graph                 208 passed in 23 s; 208 outcomes recorded
passed  verity-bench                 466 passed in 52 s; 466 outcomes recorded
passed  verity-network-e2e           70 passed in 69 s; 70 outcomes recorded
passed  benchmarks/one_stage         537 passed in 150 s; 537 outcomes recorded
passed  verity-network-traces        540 passed in 183 s; 540 outcomes recorded
6 suites: 6 passed (0 cached), 0 failed; 8.0 min of test time
115 passed, 9 warnings in 110.43s (0:01:50)
```

- `outside` is empty for all six suites; each record's counts show 0 failed, 0 errors, 0 skipped [read: `out/logs/*.json`].
- `test_exfiltration.py`: 11 of `verity-bench`'s 466. `test_certifier_term.py`: 20 of `verity-network-traces`' 540 [read].
- The nine warnings are bookkeeping tests'. One names `benchmarks/README.md`, for three rows that are `main`'s (`codec_tensor`, `kv_filler`, `vllm_encoded`), not ours. The two you know are still there: `benchmarks/network_e2e/traces/README.md` lines 50 and 53, and `verity/protocols/certifier/verifier/README.md`, now 244 bytes over its cap (it was 157 at `cf01`; `main` moved it) [read].

**The body** [ran]: fetched from GitHub first (it was byte for byte the 2:04 PM PST one, with `cf01`'s lines), edited, posted 5:37:56 PM PST, read back equal, 48,525 bytes. `Lane`, `Refs NET-326`, `Goal network/term-in-the-number` kept. What changed in it:
- The opening's last paragraph: the branch holds `main` at `be9a3d566905`, six files, 559 added and 116 removed.
- "What it waits on" is now "The merge of `main`, and the one line of #8779's test": the merge commit, no file closed by hand, the one line and why it is this change's, the six-file diff, and "it waits on nothing".
- "Checks" has a new first block, "At the head `6b8edbf5de81`", with the run above and the local result (11 passed with the line; 10 passed and 1 failed at the merge commit without it). The block for `f28f4e4acc2d` now says "two commits below the head", and the three below it five, seven and eight.

**One thing I owe you**: while reading the run's record, one print of mine showed a test id of `benchmarks/harness/tests/test_bench.py` that holds a node class's name in its brackets. It is a string of the tree's test file, not a node of a run; it reached my own tool output only, and no message, file or post carries it. I print counts of that record now, not its entries.

Now the table. No timer is set.

[2026-11-12T01:39Z] 3b2943983b71
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:39 PM PST. The two tries at `7c99c50d9586475082b6c42c7117a67045e895fc`, each `--on auto`, each launched once [ran]:
- `r20261112-013641-056f`: `--try security/Proofs/NetworkCertifier/PeriodWitnesses.lean`, launched 5:36 PM PST.
- `r20261112-013740-39ea`: `--try security/Proofs/NetworkCertifier.lean` (the root), launched 5:37 PM PST.

[2026-11-12T01:40Z] e351a0dc4491
BOUNDTIE to the coordinator, NET-341, Wed 11 Nov 5:40 PM PST. #9053's body is rewritten (replaced at 5:38 PM PST, 21,385 bytes); the head is still 2eb3cca7a040dcc47b986f49b15f7839a0c048e7 and the pull request is unmarked. New in it: 'The premise, open for both forms' (citing the proofs comment on NET-297 of 2:15 PM PST and what would close it, VLLM-370's port and VLLM-339's check), 'What the certifier signs about a connection today' (13 lines, each [read] with file and line), 'The last commit is text only' (the four checks), and the README's size after a merge with main. Run r20261111-220825-e7f8 [read: its out/suites.log]: certifier verifier 1,339 passed, 0 failed, 1 skipped, 1,126 s; repository 307 passed, 0 failed, 333 s; circuit-check suite 404 passed, 0 failed, 5 skipped, 3,657 s; 3 suites passed, 0 cached, 0 failed; the read guard failed none (no 'read files outside its inputs' line). One read you may want from the new section: connection_rows.attest, which commits and signs the lengths and fails tables, has no caller outside the certifier verifier's tests, at this head or on main at be9a3d566905; the runner (warden_serve.py:777-780, :1095-1098) builds its schedule with no lengths rows. Nothing newer than the 2:15 PM PST comment was on NET-297 at 5:21 PM PST, so your withdrawal of the 1:40 PM line is not there yet. My NET-341 comment of 2:13 PM PST still says the premise is the no-stop form's alone; I posted no new comment, none being asked. A slip: my line about the README says 5:37 PM PST and the clock read 5:36 PM when it was sent.

[2026-11-12T01:44Z] 7e148f135958
LEANMERGE to network-accounting's coordinator, Wed 11 Nov 5:44 PM PST. Final report on the rename: both tries have ended, each rc 0. One thing needs your word or lean's: the rename takes the area over its unread cap.

**Head** [ran]: `7c99c50d9586475082b6c42c7117a67045e895fc` on `cursor/na-cross-mend-witnesstoys-4d7a`; its parent is `1c506c38cecdb8050bf4dff8676cec1e23c33276`, on the crossed tip `3ff11d118f93be79dd009c2c4e0d7eae2461d411` (still the tip at 5:43 PM PST).

**The diff** [ran]: 3 files, +22 −20, no statement and no proof.
- `security/Proofs/NetworkCertifier/PeriodWitnesses.lean`: `PooledLeakAccepted_honest` is `PooledLeakAccepted_schedules` (line 31), `PooledLeakAccepted_nontrivial` is `PooledLeakAccepted_fails_at_one_unit` (line 42). The header now says the pair is in `Witnesses` and what the two here add; the first theorem's docstring says "at the fixture `fxQueue`" where it said "at the deployed instance".
- `security/Proofs/NetworkCertifier.lean`, the docstring at lines 90 to 92: it said `PeriodWitnesses` gives the monitoring period's honest and nontrivial lemmas. Your LEAN-249 comment of 5:38 PM PST says this file is lean's by the registry [read]; I did not check the registry before editing it.
- `security/Proofs/NetworkCertifier/README.md`, lines 182 and 183: it said "PooledLeakAccepted's pair". 51,591 bytes before, 51,587 after.
- Left as they are: `WitnessToys.lean:18` and the README's line 178 (they say what the module concludes from, which is still true), `Witnesses.lean`, every lock.
- `git grep` for the two old names over `security/`, `tools/`, `tests/` now finds only `Witnesses.lean` (164, 206, 509) and the lock.

**The tries, each `--on auto`, once, `tree_sha` the head** [ran; outputs read through the mask]:
- `r20261112-013641-056f`, `PeriodWitnesses.lean`: rc 0, ended 5:38 PM PST. "lean exit 0 in 58 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- `r20261112-013740-39ea`, the root `security/Proofs/NetworkCertifier.lean`: rc 0, ended 5:41 PM PST. "Build completed successfully (11066 jobs)", with `Proofs.NetworkCertifier.PeriodWitnesses` and `CountWitnesses` built; "lean exit 0 in 177 s"; "axiom probe: none of its 39 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". No warning names either changed Lean file.
- Not run: any package's Lean check, the replay, a records run. The files elaborate.

**`check.py` and the witnesses** [read]: the rule takes a guarantee's witnesses from the lock's `witnesses` map only (`tools/verity/lean/check.py:2308`, the loop over the lock's guarantees at 2320, the reads at 2363, 2379 and 2413; `honest_witnesses` at 2117). A suffix is read nowhere: `_honest` and `_nontrivial` occur only in the names a failure message proposes (1926, 2246, 2254 to 2264). The two new names are in no lock, so the witnesses rule never sees them, and no other rule of `check.py` fails on a theorem that no lock lists [read: its `unlisted` codes are all about names a lock holds].

**What the rename does trip: the unread ratchet** (`tools/verity/lean/unread.py`, run by `lean_audit.py`).
- No lock and no code names anything `PeriodWitnesses.lean` declares now, so it is a wholly unread module of 2 declarations [read: the seeds; the root under `roots` keeps only its own module, `unread.py:655`; my 5:30 PM PST line said 654].
- The area `Proofs/NetworkCertifier` is capped at 2 modules and 6 declarations on the crossed tree [ran]. The side's own ratchet counted exactly that at `75918e18bbcb`: 2 modules, 6 declarations, 195 lines, one of 120 lines deletable alone [read: `r20261111-034141-2f06`, `out/unread-areas.txt` and `unread.md`]. By their lengths the two are `SampledAuditInstance.lean` (120 lines) and `Warden.lean` (75), three declarations each, and both are byte for byte the same on the crossed tree [ran `sha256sum`; that it is these two is argued, the report does not name them].
- So the area goes to 3 modules and 8 declarations, over its cap [argued; the ratchet reads the audits' facts and I ran none].
- What that does [read `lean_audit.py:285`]: in lean's records run it is bookkeeping, the cap is raised to the count and lean is warned. In a pull request's own check it fails.
- Three ways to close it, none done: lean raises the cap to 3 and 8 in the merge; or the two theorems move to the end of `WitnessToys.lean`, a read module, and `PeriodWitnesses.lean` goes (the count stays 2 and 6, but nothing reads them still); or your option 1.

**For the later issue** [ran `git grep`]: nothing but `PooledLeakAccepted_fails_at_one_unit` reads `pooled_bound_fails` (`PeriodWitnesses.lean:47`; the rest are its declaration and docstrings), and `fxQueue_fits` and `fxRows_wf` are read only by `PooledLeakAccepted_schedules`, `fxQueue_bucket` only by `fxQueue_fits`.

**NET-332** [ran]: my reading of both logs agrees with rc 0, each at `6a285960e2f2`. `r20261111-214832-b9c9`: "lean exit 0 in 34 s", none of its 39 theorems beyond the three axioms. `r20261111-215028-a30c`: 89 passed in 548 s, start to end 11 min 14 s; the slowest test is `test_lean_audit.py::test_service_verify_is_the_same_under_the_adapter` at 72.9 s, the period protocol's slow test 11.3 s.

**Least sure of:** the ratchet's count on the crossed tree, which rests on the side's report and on my reading of the seeds. **Where I did not look:** whether `main`'s merge added a reader of `SampledAuditInstance.lean` or `Warden.lean`; the handle registry.

Everything is pushed and the notes are synced (`tools/na-cross-mend-witnesstoys-leanmerge/`). I posted nothing on Linear.

