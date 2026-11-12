[2026-11-12T20:19Z] 82f93a90d5b7
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 12:20 PM PST. **Job 6 rebuilt on lean's branch: the report asked for by 1:30 PM PST. The branch is pushed with no pull request. One thing is left: run 2 (the lock update at the new head) has waited an hour for a slot and can't end before 1:45 PM PST.**

**The branch**
- `cursor/network-accounting-okbit-onmain-4e7a` at `4be378c712b9`, pushed, no pull request [run: `git rev-parse origin/…`]. It is lean's `ca2e80f1edf7` plus 19 commits: the 18 cherry-picked in their order, then one of mine.
- **No cherry-pick stopped.** All 18 applied with no conflict [run: `git cherry-pick`, one a commit]. `594da4160d53` (the old `55c29a928932`) still builds one theorem short alone; the body says so.
- **The one commit after them, `4be378c712b9`:** hand-written `layers` entries in the certifier verifier's `lean-audit.json`: `NetworkCheck.ScanOk` gets its own layer, and the period check's layer lists it. Run 1's records step found both missing. `--try` never ran that rule, so the old branch had the same gap; the body says so.
- **Test files a cherry-pick changed against `e27a7b84e739`:** one, `verity/protocols/certifier/verifier/network/tests/period_case.py`, 3 lines added and 2 removed, and those are lean's own change of a docstring under my commits [run: `git diff --stat e27a7b84e739 4be378c712b9` over my five test files]. The other four are byte for byte the ones you read.

**The runs** (each launched once, `research run --on auto`)
- **Run 1, `r20261112-174612-20cb`, at `fcdb33673b07`:** status `failed`, exit 1 [run: `research status`].
  - Records step, exit 1. Every Lean package builds, and `security/Proofs` passes: 201,586 declarations in 3,972 modules, only `propext`, `Classical.choice` and `Quot.sound`. Three packages failed (the certifier verifier's Lean, `security`, `security/Properties`): on the two `layers` entries above, closed in `4be378c712b9`, and on lean's lock at `ca2e80f1edf7`, which does not match its own Lean. It kept no record.
  - Suites step, exit 0: 86 suites of 86 passed, none cached, with the reached test modules run whole, slow tests included.
- **Run 2, `r20261112-192107-f9f9`, at `4be378c712b9`** (the records `--update` and the changed suites): status `waiting` at 12:19 PM PST, 58 minutes after its launch [run: `research status`]. The pool's four slots are taken and the GPU node takes no job before 3:30 PM PST. Its Lean step took 84 minutes in run 1. I told you of the wait at 11:23 AM PST and did not relaunch.

**What the lock update moved**
- **Committed: nothing but the hand-written `layers` entry.** No regenerated record is on the branch, as you ruled at 11:44 AM PST.
- **What will move, read off run 1's failures at lean's stale lock** [read: its `report.json` files]: a change of `NetworkCheck.Period.check` moves every record whose statement reads `check`. 23 `Verity.*` records of soundness, `Verity.remote_soundness_of_subprotocols` (deployed) first; proofs' `Flock.SecurityProofs.RecursiveAuditC3`; 8 of ours; and the 11 `Properties.NetworkPeriod` records of your ruling. No statement's text changes. Seven more records read `NetworkCheck.Period` as changed and may be lean's stale lock rather than mine. The body's first screen names them by owner.
- **The exact list waits for lean's lock commit:** at lean's ready line I merge its branch once, run `--update` once and send you what moves by my change alone.

**The body**
- `lanes/network-accounting/20261112T1757Z-checkpoint-net357-okbit-pr-body-onmain.md` in the notes, synced. Its first screen holds your five things, the records finding by owner, and the `--try` line. Its Evidence table has run 1's two rows and a marked place for run 2's.

**Not checked:** the root suite whole (the changed-suites run does not pick it); `verity check` whole; the records at a lock that matches its Lean.

**The v1-id mismatch (your 11:32 AM PST question): sent whole at 11:52 AM PST, read-only.** Its first line again, in case that line is still uncollected:
- The period check and its writers agree on one id, `verity/sampled-proofs/one-stage/v1`, at `main` `0e8d16c3b8be` and at lean's `64ee10f17900`, so no period reads undecided or rejects because of the id [read: `NetworkCheck/Period.lean` `oneStage` and `answered`; `served_integrity.answer`; `scan_zk.window_answer`].
- The mismatch is one layer up: the answer is read out of the period's own evidence, and SPOT, the composites' integrity subprotocol, neither writes nor takes a v1 answer (`VStarRecursive/Sampled.lean`, `V1_REFUSED`). No Lean computed the answers our period accepts.
- The literal can't be flipped today, and the change is an issue of its own, not job 6's pull request. I offered one line for job 6's body under "Not established" if you say the word.

**NET-335, started while run 2 waits. The 6 PM PST hour holds: steps 1 and 2 are already pushed.**
- Scratch `cursor/na-scratch-net335-7c2e` at `a11f58adbfdc`, pushed, no pull request: one commit on proofs' `c2aebff06095`, the head when I cut it at 11:58 AM PST. proofs' `77414839aa90` (your 12:19 PM PST line) is one commit on that, three of proofs' own test files, none that I touch [run: `git diff --stat`]. I take it at proofs' ready line and do not chase it before. Steps 1 and 2 are in it: the record lists each certified row registration as a `registered` value under the registration's own value and is v2, `spec_of` declares each with its public items, and `certified=` goes to `verifier.check`, `stream.register` and the negatives' `register`.
- Ran on the VM: `test_scan_zk.py` whole, 32 passed [run: pytest, the named file]. The honest round on the challenger's coins needs the Lean verifier: it is in run `r20261112-201438-5273` (the changed suites at `a11f58adbfdc`), `waiting` since 12:14 PM PST.
- `git merge-tree` of `a11f58adbfdc`: clean against lean's `64ee10f17900`; against job 6's `4be378c712b9` one conflict, in `test_scan_zk.py`, where job 6 adds the ok ports to the same hand-built registration this change moves the rows out of. `scan_zk.py` merges by itself.
- Nothing of the `certified=` interface has failed to fit the scan so far [read and the VM's tests; the Lean side not run yet]. One thing to know for OPTIN and BOUNDTIE: the record check at proofs' head is still the strict one (`opened != want`, `one_stage/audit.py`), so a round in which a certified value's readers have no session is refused until proofs' relaxed check is in. At a group of one link-window every template's stratum is sampled, so both values are opened [argued from `audit_record` and `session_registered`; no round run].
- What is open for 6 PM PST is the node's run, which waits for a slot. The full line on steps 1 and 2 comes then.

TIMER: none

