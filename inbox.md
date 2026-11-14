[2026-11-14T08:39Z] 3dff43f60aa3
WALKER (session c17bbff0, on A), Sat 14 Nov, 12:39 AM PST.

**An honest reading is refused at the word-by-word point, so I stop on A as you ruled: `test_value_reading.py`'s torch attention step is refused at `Derived[TrainStep]_v1` node 86, operand `x`, rows 0 and 4. Those two rows are all +0 words, the first position of each of the batch's two sequences. The refusal was already there at 5c8d1454, so the second point causes it, not the new member key. Neither PR can be marked until you rule.**

Three sessions of me are live on this VM since about 08:31Z: c17bbff0 (this one, A), 4d808059 (C) and 6d747131 (unknown to me; it has written nothing in /tmp/trn145).

The refusal:
- RAN, at A's head 1637d668 on the VM, `test_value_reading.py -k "distinct_pairs or breaks_the_premise"`: 3 failed, 6 passed.
  - `test_what_the_circuit_alone_charges_reads_distinct_pairs[sdpa]`: `Derived[TrainStep]_v1{SRC={6e2fa37a40d3}}` node 86, `x`, rows (0, 4).
  - `...[relu]`: `Derived[TrainStep]_v1{SRC={a5c37c433e8d}}` node 98, `x`, rows (0, 4).
  - `test_a_batch_that_breaks_the_premise_repeats_pairs_the_reading_charges[sdpa-prefixes in proportion]`: node 86, `x`, rows (0, 4).
- RAN, the sdpa case instrumented: node 86 is a batched f32 `LinearRef_v1{K=4,N=4}`. Its `x` has 8 rows of 4 words in one activation. At the word-by-word point, rows 0 and 4 are `0x0` in every word, and the other six rows are distinct and nonzero.
- RAN: the token input is found by use (node 33, 64 bits, 30 words), drawn as a permutation of the table's 8 rows. So M3 works here, and no member key is involved (A = 1).
- INFERRED: rows 0 and 4 are the first position of each sequence (2 rows × 4 positions). There a softmax's backward is 0 for every input: dS = P ⊙ (dP − Σ P·dP), and P = [1] at a row with one key. So the rows are equal by an identity of an honest circuit. `P_BATCH` itself excludes each row's first position, but the second point compares every row. The specific rule calls `x` specific, the second point refuses it, and the point can't tell a first position from any other.
- RAN: the quick tier at 5c8d1454, r20261114-080235-0f74 (the other session's), failed the same 3 tests (rc 1), so 093eb4b07b7's second point causes this. The honest small model passes at both heads.

Your ruling is needed: whether to exclude each sequence's first position from the second point's comparison (it would need the positions, which the point doesn't have now), to leave such an operand uncharged, or to do something else.

A:
- #9558's head is still 1637d66868708a6a4563ba95958af632e7c1e5a5, as sent at 12:30 AM. The reader's four probes are refused there, and the four mutations fail by name.
- Held, on the backup branch cursor/trn145-a-fix-wip-eb6e only: 4c96f3db79d6, one commit that fixes the quick tier's other failure, which is A's own. `program_reading.py:86` swallowed `(OSError, StopIteration)`, failing `tests/lint/test_repo_silent_excepts.py`. It now reads VmHWM with `next(..., None)`. RAN: that lint passes 3 of 3, and the command's small-case test passes. I push it to #9558 with the fix you rule, as one push.
- The quick tier at 1637d668, r20261114-083224-ce52, is waiting for a slot. It will fail on the two known causes. I leave it to show whether anything else fails.
- A's body (M4) is not written yet. It follows the ruling's head, with the command's small case at 1637d668 (RAN on the VM: refused null, N 6,029,312, claims 44, pass 1.6 s, pass peak 0.84 GiB), and with the memory section's harness figures taken out under ruling 4.

C (session 4d808059's report): its head is 7e94d534f322 (A's 1637d668 merged). Its quick tier is r20261114-083506-23be and its s-t20 reading r20261114-083504-83ab. 9b8b was cancelled at 08:34Z. C holds the same second point, so its quick tier fails the same three tests.

Events not addressed to WALKER: none this turn.

[2026-11-14T08:39Z] 79e44adb8450
READCALC, Sat 14 Nov, 12:39 AM PST. Training's lock entries against the policy's 91 files, on origin/main 7f8579ab056.

**2 files: of the 86 the split gives to network-accounting, two declare an entry the lock names @training for.**

How:
- RAN: in the three locks I took every entry whose owner is `@training`, from `guarantees`, `assumptions`, `open` and `domain`. That is 134 entries with 132 distinct names: 99 in security/, 15 in Properties/ and none in Proofs/.
- RAN: I mapped each name to its declaring file by its full Lean name, with a namespace-tracking parse of every .lean file under security/. Then I confirmed each one with `git grep` of the declaration in that file: 134 of 134 found, each in exactly one file.
- READ: the 91 files are from #9989's body. RAN: all 91 are on main.

Item 3 (the 86 network-accounting files):
security/Properties/Nci/Record.lean 4 (Record.Counts, LaidOut, Owns, Rates) mixed with @lean 5
security/Proofs/Properties/Nci/Agrees.lean 1 (DescAgrees)

Item 4, the training side's files with no training declaration (RAN: no entry of the lock names a declaration in them):
security/Properties/Nci/Threads.lean 0: the Throttles* proofs are in Proofs/Nci/Verdict.lean, owner @lean
security/Properties/Nci/Desc/Cover.lean 0: CheckedCoverTraining is in Proofs/Nci/Cover/Checked.lean
security/Properties/Nci/Desc/Verdict.lean 0: the DescThrottles* proofs are in Proofs/Nci/DescVerdict.lean
(The fourth, Proofs/Properties/Nci/DescVerdict.lean, holds 2 of training's: DescReports and DescWitnesses, which the PR calls the policy's.)

Item 4, files outside the 91 and outside training's ten rows that declare a training-owned entry (RAN):
security/Definitions/Assumptions.lean 5
security/Definitions/Nci.lean 1 mixed with @lean 1
security/Definitions/Nci/Assumptions/Capability/Information.lean 4
security/Definitions/Nci/Assumptions/Cover.lean 1
security/Definitions/Nci/Assumptions/CoverTraining.lean 2
security/Definitions/Nci/Assumptions/Dampener.lean 3 mixed with @lean 4
security/Definitions/Nci/Assumptions/Remote.lean 3 mixed with @lean 3
security/Definitions/Nci/Capability/Consumers.lean 3
security/Definitions/Nci/Capability/Items.lean 2
security/Definitions/Nci/Cover.lean 1
security/Definitions/Nci/Cut.lean 5
security/Definitions/Nci/Formalism/Inference.lean 7
security/Definitions/Nci/Formalism/Runs.lean 4
security/Definitions/Nci/Formalism/Training.lean 3
security/Definitions/Nci/Threads.lean 1 mixed with @lean 2
security/Proofs/Nci/Capability/Budgets.lean 3
security/Proofs/Nci/Capability/Dependency.lean 3
security/Proofs/Nci/Capability/Witnesses.lean 12
security/Proofs/Nci/Cover/Checked.lean 1
security/Proofs/Nci/Cover/Training.lean 1
security/Proofs/Nci/Cut.lean 5
security/Proofs/Nci/CutWitness.lean 7
security/Proofs/Nci/DescVerdict.lean 4
security/Proofs/Nci/Formalism/Fit.lean 3
security/Proofs/Nci/Formalism/Slowdown.lean 5
security/Proofs/Nci/Formalism/Witnesses.lean 12
security/Proofs/Nci/RemotePeriod.lean 3
security/Proofs/Nci/Witnesses.lean 2 mixed with @lean 19

Notes:
- 5 of training's ten rows hold training entries, and 36 files hold them in all.
- 20 of the 134 are `domain` premises (security/ lock), counted because you asked for every premise, open or not. Without them, Definitions/Assumptions.lean and some Definitions/Nci files would drop from item 4's second list.
- One lock entry resolves to no declaration: `Core.Protocol.Accepts` (@lean, Properties lock, domain).
- Nothing written to the tree, no PR, nothing posted. No event meant for you reached me.

[2026-11-14T08:40Z] ce76e631532d
READCALC, Sat 14 Nov, 12:39 AM PST. A correction to my listing's `domain` note.

**Without the 20 `domain` premises, exactly five files drop from item 4's second list (RAN); Definitions/Assumptions.lean stays.**

The five are Definitions/Nci.lean, Definitions/Nci/Cut.lean and Definitions/Nci/Formalism/{Inference,Runs,Training}.lean. Item 3 and the other lines are unchanged.

