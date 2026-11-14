# CARRIER, @NOW@ Sat 14 Nov: #9636's carry is under way: what the merge meets, the plan, and the ready note by about 10:30Z [guess]

Clock read with `date -u` at @NOW@. **The carry is started (run `r20261114-055226-0840` on vy-nebius-cpu-5, running). Every hunk I have read so far is a mechanical "both sides". 20:00Z can be made unless a STOP condition comes up; I say so at once if one does.** No branch for #9636 is pushed yet.

**What the branch holds** [run, `git` on `a287d4bf124738d270ac605e153805169f5d73ee`]

- 39 commits over `main` (31 without merges); it holds `main` at `4a7291116e11`.
- Its own change: 19 commits over 20 files.
- #9458's old head `049c5875989c`.
- #9023's three commits (`6b782d972ec4`), left as they are. They change 8 files, 120 lines in and 74 out:
  - `09c7c534a25`: the record's registered check keeps one direction. Each read the verifiers report is held to a registered value at its registered root; a registered value with no reported read is asked nothing. `Properties.Spot.UnitsProved`'s last conjunct becomes `OpenedRegistered` and loses its guard, `SameRoots` is deleted, `Record.lean` and `audit.py` follow, and the tests that encoded the dropped direction expect an accept.
  - `d9225d5a820`: `audit_record`'s docstring.
  - `6b782d972ec`: the two locks' records for it (`security/lean-audit.json`, `security/Properties/lean-audit.json`).

**#9458's carried line, read again** [run]

- #9458's own branch is still at `049c5875989c` at 05:56Z. The hold branch is at `1b4b9acc84f3c4bdb47cc35e00905a3f0ddc50e9`.
- **The hold line has one settlement made by hand, not five files' worth.** It is `a275f10724a9`, where `restack.py` stopped at the terms move's base `92b5727aca40`: `benchmarks/one_stage/tests/test_stop_audit.py`, two docstring lines to the base's side and the helper called by the base's name `_handoff`. The queue's five files are what a plain merge meets; the tool dissolved the other four. I take that commit and settle none of it again.
- **The hold line holds #9810's head `12ee02ed7e12`, which holds #9717's head `3e1103ded562`. Neither is on `main` at 05:57Z.** A carried #9636 holds both until they land.
- The hold line crossed Dither's move (`e4a8682800d6`) and then the clear audit's (`ac8b3b51e490`). Each was made on a pull request's branch and neither holds the other, so `main` has 30 moves that #9636 lacks and `ac8b3b51e490` holds 29 of them.

**What the merge meets** [run, `git merge-tree`, the tree's merge drivers on]

- A plain merge with `main` `5eb2c77ef848`: 9 files. Five are #9458's (the queue's list). Four are #9636's own.
- A plain merge with the carried #9458 line: 5 files, 7 hunks, all #9636's own lines.

| File | Hunks | What meets | Kind |
|---|---|---|---|
| `security/Proofs/Properties/Spot.lean` | 2 | `CertifiedHeld` added beside the terms move's `Window*` to `Run*` | the move's; the tool dissolves it [guess] |
| the SPOT verifier's `README.md` | 1 | the `Certified values` line beside a moved path | the move's; the tool dissolves it [guess] |
| `Spot/Pieces.lean` | 1 | a docstring: "never `certified` entries" beside #9555's `family`, `key`, `main` | both sides |
| `one_stage/verifier.py` | 1 | `audit`'s signature: `certified` beside `family`, `key`, `main` | both sides |
| `service.py` | 2 | two calls: `certified=` beside `family=`, `key=`, `main=` | both sides |

- [derived] None of the seven asks what the check should do.
- [run] `security/Properties/lean-audit.json`: the lock's merge driver takes the removal of `Properties.Spot.SameRoots` and says a lock run regenerates it. The records come from a run, step b.
- [run] #9549 is not on `main` at 05:57Z, so `Spot/OneStage.lean` has no hunk yet.
- [run] No open pull request but #9023 and #9636 names `openedRoots` or `lessOwnReads`: TWOLEG's is not open yet.
- [run] `main` is at `eda01c6a7fe6`: one landing since 05:43Z (#9510), no move. `AGENTS.md` is unchanged.

**The README's cap** [run, bytes of each side]

- The cap is 49,152 bytes. `main`: 46,458. The carried #9458: 47,542.
- **The carried #9458 did nothing about the cap and needed nothing**: it adds 15 lines (the `window_place` item and the `complete` paragraph), and `main`'s README has shrunk by 2,882 bytes since `4a7291116e11`.
- #9636 adds 1,459 bytes over #9458's old head. [derived] The carried #9636 comes to about 49,000, about 150 under. The carried head's own count decides.

**Plan**

1. On vy-nebius-cpu-5, `restack.py` on #9636's head across 29 moves, the 28 up to the protocol groups and the clear audit's (`--main ac8b3b51e490`). Where it stops at `92b5727aca40`, the job merges `a275f10724a9`, the hold line's own settlement there, and goes on. That is the run above.
2. A second job crosses Dither's move (`--move-commit e4a8682800d6`).
3. Here: a plain merge of #9458's carried line (its pull request's head if STOPDRIVER has pushed, else the hold branch), then of `main`. One commit a settlement, each saying which side and why.
4. Pins; the locks by the fast path; the check runs on named nodes; the fast-forward push to `cursor/proofs-r7-certified-b6a0`; the body; the ready note.

The scratch branch is `cursor/proofs-carrier-9636-carry-9257`, pushed when the first job's bundle is in.

**The hour** [guess]

- First job done about 06:40Z. Second job and the merges by 07:45Z. Check runs (a records run replaying `security/Proofs` at 120 GB, the quick suites) by 10:00Z. Ready note about 10:30Z.
- It slips if #9458's head moves after I merge it, if TWOLEG's change lands first, or if a records run waits in line for a Lean slot as one did this morning (32 minutes).

**The `ChargeOnce.lean` pull request**

- [run] #9513 is not on `main` at 05:57Z, so nothing is cut. The Issue line is `Refs PRF-361`, as you answered.

**Not established**

- That the tool dissolves the three hunks I marked [guess]: the run says.
- That `restack.py` crosses Dither's move after the clear audit's. The hold line and my earlier carries crossed them in the other order. If it refuses, I stop and write.
- What the merge with `main` adds beyond the merge with the hold line: the 24 landings since `245da271367f`, #9892's README edit among them. Not read hunk by hunk yet.
- Whether a record beyond `Registered` and `UnitsProved` moves: no run yet.
- Whether the SPOT verifier's slow tier is needed: not decided before the carried diff exists.
