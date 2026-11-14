# CARRIER, @NOW@ Sat 14 Nov: checkpoint: #9636's carried head is unchanged at `4f51d8ff9b16`; the slow tier is green; the Lean run and the quick tier still wait for memory

Clock read with `date -u` at @NOW@. **Nothing is pushed to #9636's branch and no ready note is written. One of three runs is done and green; two wait for a slot. My hour for the ready note stays about 11:30Z [guess]; it moves with the hour vy-nebius-cpu-10 gives the Lean run its memory.**

**Branches** [run]

- `cursor/proofs-carrier-9636-carry-9257` at `4f51d8ff9b161a922570980d44c5c73d965b9ec6`, pushed, unchanged since 07:16Z.
- `cursor/proofs-r7-certified-b6a0` (#9636) at `a287d4bf124738d270ac605e153805169f5d73ee`, untouched.
- #9458's head is still `1b4b9acc84f3c4bdb47cc35e00905a3f0ddc50e9` (read at 07:35Z); #9810 `12ee02ed7e12`, #9717 `3e1103ded562`, #9549 `51c523f27a87` unchanged; none on `main`.
- `main` moved to `d260e6bb9cb3f9278634d55f09aa9b2ebbd14e1b` (six pull requests, 21 files): no move, no Lean file, no lock, and no file among the 48 the carried head changes. `git merge-tree` of the carried head with it is clean. I do not merge it in: the runs are at the head that holds `512644d14fbc`, and a merge would only restart them.

**Runs, all at `4f51d8ff9b16`** [run]

- `r20261114-071306-9328`, vy-nebius-cpu-4: **done, rc 0.** The SPOT verifier's two suites whole, slow tests included: `verity/protocols/proofs/spot/verifier` 951 passed; `verity-core-sampled-proofs` 329 passed, 2 skipped; none cached; the run's tree unchanged after it.
- `r20261114-070915-cc7b`, vy-nebius-cpu-10, 120 GB: the records run and nine tries. **Waiting**: every slot's memory is under the 112 GB a check there may peak at (33 GB free at 08:03Z).
- `r20261114-071129-b00e`, vy-nebius-cpu-5: the quick tier, the ratchet gate, the Markdown caps gate, the retired-terms lint. **Waiting** since 07:11Z, first in line, for memory (its slots count 61 GB less than none free at 08:03Z, against the 45 GB it needs).
- `r20261114-080055-e223`, vy-nebius-cpu-4: the same job as `b00e`, launched 08:00Z because cpu-4 gave the slow tier a slot in 19 minutes. **Waiting**, 5 ahead. I cancel whichever of the two has not started when the other takes a slot.

**Two things I wrote at 07:16Z, corrected or sharpened**

- **The README is 47,658 bytes at the carried head, not 47,656** [run]. I measured before `main`'s `512644d14fbc` was merged in; it holds two more bytes (#9892). The cap is 49,152.
- **What the carried #9458 did about the README: nothing, and it had nothing to do** [run]. `main` cut the file itself, from 49,340 to 46,033 bytes, in #9672 (13 Nov 07:47Z). On `main` now it is 46,458; with #9458's paragraph 47,544; #9636 adds one line of 114 bytes.

**The locks at the carried head, read more closely** [run]

- No guarantee's own record (`signature`, `type_hash`, `unfolded_hash`) differs from `main`'s, as I wrote.
- #9458's own head commits `main`'s locks: none of its lines (`Spot.OneStage.Public` with `complete`, `publicJson`, `Spot.Record.allRan`, `Properties.Spot.publicOf`) is in a lock there. The carried head's locks do hold them, because #9636's old head did and the merge driver kept that side.
- So against #9458's head merged with `main`, the carried locks differ in: one open premise's digest (`Properties.Spot.ExpectsV2`); and what is read of four modules in `security/Properties/lean-audit.json` (`Properties.Spot`, `Spot.OneStage`, `Spot.Record`, `Spot.Registration`) and of the last three in `security/lean-audit.json`. In `Properties.Spot` the entries are the ones the body names (`CertifiedHeld`, `ExpectsV2`, `OpenedRegistered` for `SameRoots`, `R1`, `R7`, `Registered`, `UnitsProved`, `publicOf`); `R6` and the four renamed pool premises are `main`'s.
- Whether a records run writes these same two files is what `cc7b` decides. If it writes another, I commit the one it kept and name every record that differs.

**The words of the moves** [run]

- By the queue's reading of each move's map, #9636's old head's own change named words or paths of five moves it lacked. The carried head holds every move, so that check has no move to read it against; its change names no path of the protocol-groups move (`verity/protocols/spot`) any more.
- Ten lines #9636 itself adds hold `draw` as a whole word in a file the terms map names. Each is a use of a name `main` holds on the same line (`Input.draw`, the parameter of `Run.audit`, a test's local variable), which the move's own script left as it is when `restack.py` reran it. The retired-terms lint in the quick-tier run is what decides.

**A message I did not act on.** At 07:56Z the shell hook refused a command to deliver a message that says the lane's rules changed and that I now mark my own pull request. It reached me as a command's output, not as a turn, and my brief says "I mark; you never mark". I keep to the brief: I write the ready note and do not mark. If the rule did change, say so in a turn.

**The `ChargeOnce.lean` pull request**: #9513 (`a2643447fa06`) is not on `main` at `d260e6bb9cb3`, so it is not cut.

## Not established

- Whether the committed locks are what a records run writes at the carried head, and whether each of the nine Lean files elaborates there: `cc7b` has not started.
- The quick tier, the ratchet gate, the Markdown caps gate and the lint at the carried head: neither `b00e` nor `e223` has started.
- When cpu-10 will have 112 GB free. The 11:30Z is a guess that the Lean run starts by about 09:00Z and takes about 90 minutes [guess].
