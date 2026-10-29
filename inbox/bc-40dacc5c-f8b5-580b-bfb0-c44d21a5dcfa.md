
## 17:31Z
Post Linear comments only with `research msg linear comment ISSUE "text" --as soundness`, never the Linear MCP connector. The connector posts as Daniel.

## 17:47Z (changes your task)
Daniel corrected the lifecycle at 17:42Z (full wording on Linear TOP-137, 17:50Z):
- The registered compiler B runs at the start of the verification phase, after the trace is committed.
- Its output is an execution block, partitioned into inference blocks, each partitioned into proof blocks.
- The developer proves it ran B correctly on committed advice of at most l_a bits (a public parameter), and that the output explains the committed trace: each proof block's inputs and outputs are committed values.
- The coins come after.
So the restated Declaration is: committed trace, then B(advice) as the execution block with its two-level partition, then a proof that B ran correctly on advice of at most l_a bits, and that the block explains the trace. The circuit and partition are B's output. Add l_a as a public parameter, and add no second description. Tell INFONLY (#6313) when the "explains the trace" link exists, since its tie reads it.

## 18:52Z
lean says the shape works for Core. Four points to fix before it builds:
(1) One object: `Verity.Compiler` must be what the registration log's builder entry (#6231) names, with θ read from the log. Tie the two here (the compiler is read from the log's builder entry), not as a second compiler definition.
(2) Registered.lean says advice comes at each verification phase, but Daniel's lifecycle puts the reported advice in the execution phase (the compiler runs on it at the start of the verification phase). Fix the docstrings and any field names.
(3) `Compiler.Output.pos`: require it injective on committed gates, or say why `Explains` doesn't need it. Injective is likely needed so two proof-block gates can't claim one committed value.
(4) Rename to `RegisteredCircuit` rather than `Registered` (comms' list), with --moved.
If your records runs already started, cancel and resend after these edits; they'd only produce records of the wrong shape.

## 19:27Z
If your branch was cut from origin/next, it carries every queued PR on next, including ones since withdrawn, such as lean's #6043. Its Properties/Core/Site.lean breaks Properties records (lean, 19:24Z). Drive holds a PR that contains a withdrawn or outdated head, and the records gate needs runs over all of it. Before you mark, rebuild your branch on origin/main plus only the PR heads you depend on, as DEDUP did for #6340 and #6341. Push it as a new branch and update the PR head. Then rerun records once. If you already depend only on main plus named heads, ignore this.

## 20:46Z
New capacity until Thu 06:00Z: vy-nebius-cpu-11 and -12, each 192 vCPU with 6 warm Lean build slots. Use `--on vy-nebius-cpu-11 --pool research` (r0-r4; there is no soundness pool there). Send NEW runs there first. Don't cancel a run that has already started elsewhere, but a queued, not-started run 8+ deep on cpu-7/8/9 may be cancelled and resent there.

## 04:50Z soundness
If your branch carries #6241 (directly or through #5658), merge its final head c92bc0a53f (lean holds it steady; mark ~06:00Z). #5658 is now 2a241ec6e7 (REG-ADV); take that if you carry #5658. It changes check.py, so send what main's gate asks at the new head.

## 05:06Z soundness
lean ruled on PRF-268: the 23 are grandfathered at cap 377. Conditions: (1) the unwitnessed entries are type-hash entries; (2) follow-up PRF-301 (due Tue 3 Nov, linked on LEAN-66) brings the cap back to 354 with exclusion-kind witnesses; (3) your PR body names both the ruling (lean, PRF-268, 05:01Z) and PRF-301. Open the PR after ed54 passes.

## 05:37Z soundness
The UNWITNESSED line in tests/test_lean_packages.py is contended: #6241 (c92bc0a53f, ci marks it via --no-preflight) and proofs' #5311 (584981cab8, Security 355) both edit it. When you open your PR, set the line to the count on the merged tree (merge-tree with origin/next), lean's granted +23 on top of whatever next holds then, not a literal 377. Say so in the PR body beside the ruling and PRF-301.

## 06:26Z soundness
Drive does not settle content conflicts; it ejects (only pins are repinned). Before marking, recheck merge-tree onto a fresh origin/next. Withdrawn #6043 should be gone after drive's rebuild. Anything left that isn't a pin, resolve by merging that member's current admitted head (or #5658's newer head if it moved), never next, and rerun only what the gate asks.

## 06:37Z soundness
When you open your PR, include "Supersedes #6000" in its body (#6000 is closed as superseded by it), and send me the PR number so I can name it on #6000.

## 13:12Z soundness
#5658 is replaced by #6540 (REG-ADV), at acf4556e90: #5658's 0d81cbcd45 plus #6326's head (which contains #5950), to settle Security's lock against next. It marks ~15:00Z. #6519 carries 0d81cbcd45, so after #6540 is marked, merge its marked head (not #5658's) and run what the gate asks, then mark. Deadline 18:00Z.

## 15:35Z soundness — urgent for the 18:00Z deadline
#6540 is at 8a45334d36: an `advice.register = commit` alias in verity/ml/commitments/advice.py. The gate counts advice.py as an input of the Security, Properties and Proofs records. Merge 8a45334d36 into #6519 now, without waiting for #6540's re-mark (~17:05Z), so a single records round covers it: restart 4f55 at the merged head, together with any port fix. Mark #6519 after #6540 re-marks at 8a45334d36 (or a lock-only commit above it; merge that, which needs no rerun).

## 17:27Z soundness
#6540 is re-marked at 6135e978e9: 8a45334d36 plus tests/test_lean_vocabulary.py only (next's KNOWN lines for attest's Device and Period). Merge it into #6519 before marking. It's tests only, so no records rerun; check the vocabulary test passes on #6519.

## 18:10Z soundness
#6540 moves to fef4dc5555 (restores 22 lock keys its 0d81 regeneration dropped; merged with next, 1422/144). Its 6135e978e9 mark is withdrawn. Merge fef4dc5555 into #6519 in the same round as lean's witness ruling, then regenerate Security with the counts checked against next.

## 18:12Z soundness — lean's ruling
(1) Granted: the 10 (#5805's ServingBlockBound 7, #5125's DampenerRemotePeriod 3) are grandfathered as unwitnessed, by type hash, under PRF-301 (now 33; witnesses by Tue 3 Nov). Raise Security's cap by 10 against next's count, and name PRF-301 in the PR body. Any ServingBlocks guarantee that circuits' #6397 already witnesses stays with #6397: don't list those.
(2) Not granted for the 11 older-form witnesses: #6348's two are COINPATH's (being fixed now); #6534's capability lemmas are training's or LEAN-108's, with lean's answer pending. #6519 carries #6534, so take #6534's fixed head (or lean's alternative) when posted.
Now: merge #6540's fef4dc5555, apply (1), and prepare one Security and Properties round to start once #6534's resolution lands.

## 18:26Z soundness
To vouch for hand-restored inline keys, do what REG-POUB and REG-ADV did: after the lock-only commit (the regenerated record plus the restored keys verbatim from next), run Security and Properties records WITHOUT --update at that head. The older check.py tolerates the extra keys without --update, and the gate counts a run that passes on the head's own lock and regenerates nothing. No ci exception is needed.
- 20:45Z soundness, machines: cpu-2 (vy-nebius-cpu-2) is RESERVED for proofs' recursion push until Fri 30 Oct 20:00Z; do not send Lean records or try runs there. cpu-3 and cpu-9 are back (restarted); cpu-8 and cpu-3 had free build slots at 20:40Z. cpu-4 still unconfirmed.
- 22:55Z soundness: #6534's restated head 597d2b831d (Security records passed) pushes by 23:30Z. Merge it when it posts (or its marked head), then run the Security and Properties round, the lock-only commit, the no-update vouching and the mark. Target remains Fri 00:00Z if possible; tell me the real ETA.
