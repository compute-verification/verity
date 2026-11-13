# ADDRROW2: finish the row fact of the address premise (you take over ADDRROW's piece)

Read `COMMON.md` beside this file first, then `ADDRROW.md` (the piece as first briefed), then ADDRROW's last three entries in `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`: search for `## ADDRROW, Fri 13 Nov 07:16Z`, `07:56Z` and `08:29Z`. They hold the state. Check each claim you build on: a line marked `[derived]` or `[guess]` there is not a fact.

**Why you exist.** The worker ADDRROW stopped twice tonight on a platform error and is not resumed again. You continue its piece from its pushed branch and its entries. Nothing of its is unpushed.

**State at Fri 13 Nov 09:00Z** [the coordinator's reading of ADDRROW's entries; the run's state by `research status` at 08:53Z; no log read by the coordinator]
- ADDRROW's branch: `cursor/proofs-addrrow-d318` at `c9ba3542f83693aeb418107d5d57d833be1e64d3`. Create your own, `cursor/proofs-addrrow2-<four hex>`, from that head, and push only to it.
- Compiled (`r20261113-072803-0c37` and earlier runs named in the 07:16Z entry): `Row`, `RowRead`, `RowKind`, `RowDraw`, `RowParse`, `RowOf` and `RowLeaves`. `rowFact_of_runs` takes three premises (`Sits`, `OwnLeaves`, `PublishedRows`), and `rowFact_of_wires` takes `wiresChecked h = true` in place of `OwnLeaves`.
- `AddressFacts/RowTable.lean` is the labels half of `Sits` (`drawn_of_ok`, `labelled_of_ok`, `unitIndices_of_units`). It failed in one `simp` at `fddc438a08` (run `c127`) and was mended at `c9ba3542f8`. The try of the mend, `r20261113-082753-b31c`, ended `done rc 0`.
- Not written: the place half of `Sits`. ADDRROW's plan for it is under "Next" in its 07:56Z entry, with the two facts it had not found.

**Do, in this order.**
1. Read `b31c`'s log. Title an entry that says whether `RowTable.lean` compiles at `c9ba3542f8`, with the probe line. HONEST2 and ASSEMBLY wait on that title.
2. Drop the hypothesis `hc1 : r.ins.read.I.count ≤ 1` from `drawn_of_ok` and `labelled_of_ok`. SESSIONOF2's lemma gives it: `oneTable_of_sessionOk`, compiled at `d45f63a86c` of `cursor/proofs-sessionof2-7c3a` (`r20261113-082915-c289`). From `two h s = some d`, `t < h.ms.length`, `sessionOk h t xS = true` and `setupOf true xS = some (ins, spec, sts)` it gives `sts.size = 1 ∧ ins.read.I.count ≤ 1`. It needs `t < h.ms.length` at your call: say where you get it, or carry it as a named hypothesis.
3. HONEST2's entry of 08:32Z, item 4: `RowLeaves.lean` line 233 needs a two-line mend on FRAME's line. HONEST2 carries it as `da62e83b7a` on `cursor/proofs-honest2-7c4e`. Take the same bytes, so the two branches agree.
4. The place half of `Sits`.

**Not yours:** as `ADDRROW.md` says. No edit of FRAME's files. No merge of `main`. Merge another worker's head only when its entry's title names that head compiled.

**Runs.** Where `COMMON.md` names a node, use `research run --on auto` in its place (it takes no `--pool`), with `--mem-gb` and `--timeout` set, on a pushed commit. Never name vy-nebius-cpu-13.

**Platform faults tonight.** A machine can be replaced in the middle of a turn. Commit and push before every try and before you end a turn. On waking, run `git status` first, and check for `~/wt/.pen`: if it is missing, the machine is new; make it and rebuild your worktree from the pushed branch.

**Owed.** First entry within 20 minutes (item 1). Item 2 compiled, or its obstacle named, by 10:30Z. In that entry: for the place half, the files, lines and theorems left, and what part can compile by Sat 14 Nov 02:00Z.

**Words.** "two-stage", "first stage", "second stage", never "round" for a stage. In plain-words paragraphs: "outer block" for the replay block, "inner block" for the proof block. "Challenges", not "coins", in new prose. Rename nothing that exists.
