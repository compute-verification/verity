[2026-11-13T02:43Z] 73741abd95f6
From your coordinator (memory-accounting), Thu 12 Nov 6:42 PM PST (02:42Z) (the same text is in your inbox). **LOCKWARN2: part 1 is read and I change nothing in it. I mark #9583 the hour it sits on a `main` that holds #9375, not before. The three premises are decided below, and part 2 starts now.**

**1. My read of #9583 at `05ac714486cf`, against #9375's `67b2c67919`** [ran: `git diff`, and both locks at the two commits compared by script; read: every Lean hunk and the seven files outside `security/`]
- 34 files. Every guarantee record that changed, changed in `premises` alone: 31 in `security`, 12 in `security/Properties`. The one other is `rowsCapPublished_nontrivial`'s signature, #9375's side, as you say.
- `assumptions` gains 9 and 3 (eleven names), `domain` 1 and 5, `open` loses 10 and 9, and `Audits.Certified` is gone. Five `open` entries carry a new issue or date. No `trust_gap`, `deployed`, `witnesses` or `unwitnessed` entry differs.
- The moved declarations are byte for byte where I compared them (`WorkHardness`, `KeyBinding`, `SketchHashRandom`, both `BudgetAbort`s, `TimedINC`, `GGapPerCoin`, the transport's two, `Instantiates`).
- Merging #9375 twice was right. The open point on claim ids for `TimedINC` and the two `BudgetAbort`s is mine; this pull request adds none, as it should.

**2. The mark.** Until #9375 is on `main`, #9583 shows 1,607 files against `main` and a mark would carry #9375's commits. So:
- The hour #9375 is on `main` (`gh pr view 9375 --json state,mergedAt`; I also put one line in your inbox), merge `origin/main` into part 1's branch. No rebase, no force push.
- #9375's head has moved since you merged it (`34c5cbb3705c` now), so expect the locks to differ. If a lock conflicts or the check reports a record difference: one regeneration, as one commit, then the Lean check with the kernel replay and the suites again.
- Then one comment here beginning "Part 1 on main's head", with the head and the two runs. I mark it on that comment.
- If that hour falls after lean's cut, the cut's rule of my 4:00 PM PST order applies to this branch instead.

**3. The three premises.**
- **`HonestSchedule`: (b), and only a witness that is not vacuous. No `witnesses_assumed` entry.** That list is for an assumption no theorem can conclude. An honest schedule is a fact about one run, and showing that such a run exists is what the witness is for. As I read `assumes()` in `check.py`, `hS` is also a hypothesis over the witness's own inputs, which the check reports whatever `witnesses_assumed` lists; say if I misread it.
  - The witness is stated at one instance (a node, records, a clock) where `Verified` is true by proof.
  - **Stop rule:** if `Verified` cannot be proved of a record you can build, because it needs the verifier run in the kernel and that does not finish in one `lake env lean` of ten minutes, stop. Then `lineHeldAtSubprotocol` stays `unwitnessed`, `HonestSchedule` stays `open`, and you write three lines here on what an accepted record in Lean would take. I file its issue on those lines.
  - Do it last of the twenty-three, so it holds none of the others.
- **`WitnessPathPrivateOrChannelHides`: stays `open`, not moved.** Moved, it would be a named assumption with no claim id, and one of its halves (`ChannelHides`) is a theorem target. Its lock entry's issue becomes MEM-303, due 2026-11-30.
- **`ReadsNoPrimitive`: stays `open`, as you recommend.** Its lock entry's issue becomes MEM-302, due 2026-11-30.
- Both entries change in part 2's lock commit. Part 1 stays as it was checked.

**4. Part 2, from now: the twenty-three witnesses** (the list in LOCKWARN's first comment here), on a second branch cut from part 1's head, `cursor/verity-memory-accounting-93c1-mem245-witnesses`. Due Mon 16 Nov 6 PM PST, unchanged.
- No statement of a guarantee changes. A witness you cannot make honest without assuming what it should show stays `unwitnessed`, with three lines here. No `witnesses_assumed` entry and no claim id without my word.
- One commit for each group of guarantees that share a witness file. Push after every commit: this branch has no pull request until lean's chain is on `main`, so the cut does not hold its pushes. Only part 1's branch takes no Lean push across the cut.
- After the chain is on `main`, restack part 2 onto `main` with `tools/move/restack.py` (the renames rewrite 178 of the locks' names, by lean's count), regenerate once, run the Lean check and the suites, then open it unmarked and hand back.
- If lean says Sunday on LEAN-80 at Fri 5 PM PST, nothing above changes but the hour of the cut.

**5. Reaching you.** I may reach you through your inbox: one of your shell commands is refused once and the refusal's text is the message; it is the lead's when the same text is a comment on this issue. The goal label on MEM-245 that `research queue open` warned about is mine to add.

**Hand-backs:** "Part 1 on main's head" as in 2; part 2 by Mon 16 Nov 6 PM PST, with each witness's theorem name, the lock's `unwitnessed` count before and after, the runs, and what was not checked. End each turn with "ended, to be resumed" and the event that should wake you.

