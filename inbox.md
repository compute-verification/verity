[2026-11-10T19:10Z] 17a2ce34b46a
**LEANMERGE, NET-312: done. Report, Tue 10 Nov 11:10 AM PST.** Two more messages follow: the citation for proofs with the merge list, and the LEAN-249 comment as posted (11:07 AM PST).

1. **Branch** `cursor/net-leanmerge-b9ae` at `65ed6901d94282c7ade299860a34aab068f65b49`: three commits on `038d5864f7`, clean into the base's `13f54ad455ac` [run: `git merge-tree`]. `0fb03a8fe6` is the Lean (line 7); `872d679e29` and `65ed6901d9` are `Proofs/NetworkCertifier/README.md`, which was main's text over the side's Lean.
2. **Run** `r20261110-182133-4cb3` at `0fb03a8fe61`: `lake build Proofs.NetworkCertifier Proofs.Properties.NetworkPeriod Proofs.Properties.NetworkCertifier.Device`, exit 0, 11,047 jobs from source, no error, no `sorry`; four probes exit 0 [run; `out/summary.txt` and the logs read by me].
3. **Lines 1, 3, 5, 6: confirmed** [read; 6 also built].
4. **Line 2: drop, nothing to carry.** Main's (1) and (2) are `EncardAccSeqsLeAdvice` at one advice type, proved from it word for word [run: probes 1, 2]. What they state that `AdviceBits` does not: a count in the alphabets' own sizes, not rounded up to bits, and no factor at all for a fixed clock. (3) cannot be stated, its schedule is gone; its count at the check's schedule with one sync value is proved [run: probe 3]. No branch of main's names the three [read: #8537, #8040, #8539, #8547, NET-297's `ScratchLaid.lean`].
5. **Yours to decide on line 2:** the lemma that implies them, `EncardAccSeqsLeAdvice`, is proved and in no lock, on main too. So the lock no longer lists a count in a general alphabet. Listing it is one `--update`, not a port.
6. **Line 4: confirmed** [read both]. **No property on the merged tree says what `RowsTiedSampled` said.** The check now accepts a tied audit at any sample and charges it. Proofs' R8 sentence must change, not only its name: next message.
7. **Line 7: my form** [built]. Main's two witnesses of `PeriodLeakAccepted` stay. `SampledCodeRuns` is deleted: no reader, stated over the old record, declared under `Proofs/`. Five toys of the side's with no reader go with it. By the rule's words the side's kind is the stronger evidence (the deployed record); main's is at `toyCode`, which accepts no run, as every witness on the tree is. The side's pair over `Compiled.KS` needs a new assumption: a follow-up for you or Daniel.
8. **`Period.lean`: nothing missing.** #6677, #6902, #6676, #7551, #7675, #7934, #7749, #7962 carried [read]; `Layout` kept as `BucketSchedule`; main's `tied` refusal leaves with line 4. Lean's `slotEntryDigest` on both fixture descriptors gives each list's first pin [run: probe 4]. I ported nothing.
9. **The merge list:** none is on `main` (`5482988e3496`), none ported. Two will not apply as written: #8537 and NET-304. Next message.
10. **Not Lean, not mine:** the certifier's Python and `PROTOCOL.md` still cite the three leaving counts and `CodeSyncSet` (`audit.py:17`, `capacity.py:46,48`, `online.py:23`, `PROTOCOL.md:947,1010,1015,2047`, two tests' docstrings) [read]. No answer of mine depends on CLEANFAIL's row 4.
11. **Least sure of:** deleting the five toys with the assumption (no reader on the tree or in main's branches [run: a script over the file's declarations], but restoring the side's witnesses would want them back), and the merged README's sentences (every name checked by script, the prose by eye).
12. **Where I did not look:** the Python difftests (three test files hold stage markers on the base and do not collect); the Lean check's replay and the locks at my head (ASSEMBLY's); and the six kept names that carry the side's statements (`EncardAccSeqsLeConstantRate` with its four forms, `ExpectedRowCausal`), which I read and did not weigh.

[2026-11-10T19:10Z] 7cd416aa5c8c
**LEANMERGE, NET-312: the citation for proofs, and the merge list.** [read by me on my head `65ed6901d942`, whose `NetworkCheck/Period.lean` and `Properties/NetworkPeriod.lean` are the base's; proofs' text read at #8539's head `353ab190932`]

**For proofs: what replaces `RowsTiedSampled` in R8's docstring**
- **Where:** `Security/Properties/Spot.lean:133-134` on #8539: "`RowsProgWired` equates the scan's `scan/prog` entry's root and domain with an audit's entries port, and `Properties.NetworkPeriod.RowsTiedSampled` has that audit sample every unit." Also #8539's copy of `NetworkCheck/Period.lean:186`.
- **On the merged tree `RowsTiedSampled`, `RowsTiedSampledOf` and `SampledWhole` do not exist, and no property says what they said.** The period check no longer refuses a tied audit that is sampled in part. It accepts it at any sample and charges it.
- **What to cite:**
  - The wiring: `Properties.NetworkPeriod.RowsProgWired`. Each lane's `scan/prog<j>` entry is at the root and domain of one audit's entries port, of a template the check pins (`SlotEntry_v1` or `SelectEntry_v1`). The entry is `scan/prog<j>` a lane now, not one `scan/prog`.
  - The state: `Properties.NetworkPeriod.RowsStateTied`.
  - The charge, in place of "sample every unit": `NetworkCheck.Period.chargedOf` and `faultyEntries`. A tied audit sampled in part, with a bound of at least one incorrect proof block, puts every slot entry bit of its link-window in `K` and its δ in `eps`. `Properties.NetworkPeriod.RowsKBoundsCount` counts it, as its `faulty` factor.
- **A sentence that is true of the merged tree:** "`Properties.NetworkPeriod.RowsProgWired` equates each lane's `scan/prog<j>` entry's root and domain with an audit's entries port. That audit need not sample every proof block: the entry bits its incorrect proof blocks can set are charged in the period's `K` (`NetworkCheck.Period.faultyEntries`, counted by `Properties.NetworkPeriod.RowsKBoundsCount`)."
- **This is weaker than what R8's text leans on, so proofs should read it, not only rename.** Before, a tied value's rows came from an audit proved whole. Now they can come from an incorrect proof block, and the period's bound pays for it.
- **Open, not the merge's:** no property states that the public the check accepts is `rowsPublic` at those terms. That would be the one name to cite in `RowsTiedSampled`'s place.

**The merge list: where each goes on the merged tree.** None is on `main` at `5482988e3496`; I ported none. [each PR's diff read against the head's files]
- **#8547 (NET-285), `c3e183dfab4`:**
  - `Proofs/Properties/NetworkScan/Accept.lean`: as written. The head's file is main's. Its proof is rebuilt against the merged `scanOf`.
  - `NetworkCheck/Period.lean`: `argsApart`, `inputsRead`, `sameLeaves`, `ingressReturn` go after `inputArg`, as written. In `scanOf` the three `need`s go where main has them, with `[pk]` read as `pks`: the merged scan has one entries node a lane. `inputsRead`'s entries are then `nb · ety.leaves` a node, not `s.lanes · nb · ety.leaves`.
  - Its docstrings' `NetTiming.Layout.Fits` is `NetTiming.BoundaryGates.Fits` on the tree.
- **#8537 (NET-289), `c9f02e741c3`: does not apply as written in two places.**
  - Nine definitions and proofs files are main's on the head: as written.
  - Its new `PooledLeakAccepted_nontrivial` in `Witnesses.lean` has the name of the side's in `PeriodWitnesses.lean`. Keep one.
  - `PooledGood`'s `op` gains the closed windows, three binders to four. That breaks the side's `pooled_bound_fails` (`WitnessToys.lean`) and `PooledLeakAccepted_nontrivial` until they are retyped.
- **#8539 (proofs' R8), `353ab190932`:** it changes no file of ours now. Its `Properties/Spot.lean`, the proof and five `Spot/` files differ on the head and are proofs'. R8's docstring takes the citation above.
- **NET-304, `435b8fa13f8`, on #8539, no PR: does not apply as written.**
  - `progParam`, `tiedKey`, `tiedInputWired`, `tiedInputCheck`, `tiedInputChecks` go after `stateChecks`, the last into `check`'s findings.
  - Its "exactly `["prog"]`" must become the lanes' `prog<j>`, so the check takes the schedule's `lanes`.
  - `RowsTiedInputWired` and its `Of` go after `RowsStateTiedOf` in `Properties/NetworkPeriod.lean`. Their docstrings' words about `RowsTiedSampled` go.
  - Its proof needs `otherAnswer`, which left with `rowsTiedSampled_nontrivial`, and an instance at the head's `slotReg` and `tiedWindow` (`"scan/prog0"`).
- **NET-293, `0c2c66856926`, no PR; it now holds #8537's head:**
  - Five new files, and edits to 17 files that are main's on the head: as written.
  - By hand: one import line in `Security/Definitions.lean`, four lines in `Security/Proofs/NetworkCertifier.lean`, and `Witnesses.lean`, which carries #8537's `PooledLeakAccepted_nontrivial` (the clash above).
  - Its ten README hunks go onto the merged README by hand. That file is 269 bytes under its 48 KB cap now, and the hunks add about 3 KB.
- **NET-301, `4b3f7d6966c`, #8686:** the five digest functions on the head are byte-equal to main's, so the move to `NciCheck` applies as on main. One call site on the merged tree (`auditOf`, over `entryIds`), and the docstrings that name `slotEntryDigest`, the side's `selectEntryPins` among them.

**For the lock run, said on LEAN-249 too:** `assumptions`' `NetTiming.Assumptions.SampledCodeRuns` leaves with my branch. The base's locks still hold digests of the three leaving counts and of `RowsTiedSampled` and `SampledWhole`, and one `signature` naming `rowsTiedSampled`.

