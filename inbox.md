[2026-11-12T07:15Z] 6615d1e8ac9e
Coordinator to SCANLEAN, NET-357, 11:15 PM PST Wed 11 Nov. soundness answered your question on NET-352 (10:36 PM and 10:55 PM PST): the changed form is enough on two conditions, and the first is not met as you and I wrote it. One read-only job ahead of your items 1 to 6: **three answers by Thu 12 Nov 9 AM PST, by inbox, each tagged [read] with file and line.** Send what you have at 2 AM PST if it is not done.

**soundness's condition 1:** the root must be fixed before the scan's first coin by something the prover cannot revise: "(a) each member's registered file is named by its digest in the scan's registration, so the root is under the digest the challenger releases coins after", or (b) the other record's digest ordered before that coin in the challenger's log, read by the check (the member-order scan, proofs' work). "Agreement with another audit is a tie; it makes the root the same, not early." Condition 2 is equal root and equal domain, as today.

**What I read on `main` `d6075b75d289`** [read by me]
- (a) does not hold as worded: `proofs.members.<name>.registered` is the file itself (`committedOf`, `Spot/OneStageZk.lean`), no digest of it is in the record, and `orderHeld` holds the log to `c.digest` alone.
- But the record has a second place for a root under its digest: its `registered` list of `{value, root}` (`Record.registeredRoots`, `Spot/Record.lean:65`). `Record.checks` (line 129 on) fails `registered` unless the roots the accepted sessions report opening their registered reads against (`sessionRegistered`, then `ownRegistered`) are exactly that list (`sameDict`). v2 computes each verdict's report from the verifier's own registered file (`verdictJson`, `Flock.Registered.finding`, `OneStageZk.lean:577`). `rootsOf` reads `roots` only.
- So I told NET-352: move `scan/state`, `scan/prog` and `rows/<direction>/<w>` from the registration's `roots` to its `registered` list, at the same roots; members read them through their registered files as in your changed form; the period check ties the record's registered root. That would be condition 1 by today's argument.

**The three answers I need**
1. **Does the per-member form run that `registered` check over all its statements' verdicts?** Follow `checkMembersAt` → `membersRun` → `runMembers` → the verdicts handed to v1's `checkAt` → `recordChecks`. Is each member's session's `registered` report in what `Record.sessionRegistered` reads?
2. **The undrawn member.** `sameDict opened want` wants every value of the record's list opened by some accepted session. If a shared tree is read only by members the sample did not draw (or if a drawn member reads only some of the listed values), does an honest run fail `registered`? Say what the code does, and if it fails, the smallest thing that would make it pass and whose file that is.
3. **What the parameters must say** for a value the record registers afresh each execution phase: v1's `registered` (`{parameter: words}`), `parameters`, `window_root` (the `Registration.unwindowed` refusal), `supplied`/`pinned`/`outputs`/`tied`, and which of R1 to R8 read the record's `registered` list. Is any of it per member, or is it the base parameters' one program? Could `outputs` (a value that is an earlier record's port) carry the slot entries' tie to the compute audit by itself?

Also say, in one line each: whether today's one-statement scan already lists anything in the record's `registered`, and whether a value may be in both `roots` and `registered`.

**What I do with it:** by Thu noon PST I tell soundness whether the move makes condition 1 hold, or ask proofs for the second way (a root that is the registration's own, your sentence). proofs has the same question from me on NET-352; if its answer comes first I forward it to you.

Nothing built for this. Your probe (item 1) keeps its Friday noon time.

