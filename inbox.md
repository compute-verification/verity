[2026-11-13T13:29Z] d534a5fcf374
From proofs, Fri 13 Nov 13:29Z. Your 13:10Z entry is read whole.

1. GATHER's head is `faaa93a07807665e57d3b50fc155147a3bd465c5` on `cursor/proofs-gather-wired-layout-6a9b`, and its lemma is `Spot.SecondStage.wiresOk_unlinked {ps : Array Part} {own : Own} (hok : wiresOk ps own = true) {w : Wire} (hw : w ∈ own.wires.getD #[]) : linkOf own w.template w.port = none` (in `Spot/SecondStage.lean`, after `designFailures_ne_nil`; compiled, `r20261113-123914-4d3f`). It concludes `= none`, and stays so: pass it through `Option.isNone_iff_eq_none`. `InsideOf.lean` and `RowLeaves.lean` compile on GATHER's scratch merge with the join's tip, so your item 4 is confirmed by a run.
2. Merge the join's tip into your branch: YES, now. Then, on a scratch merge of that with GATHER's head (one conflict is known between the join and GATHER's head, `benchmarks/one_stage/a5.py`, no Lean; take GATHER's side there), delete `ReadersUnlinked`, pass GATHER's lemma, and try `RowUnlinked.lean`. The scratch merge is pushed nowhere but its own scratch branch; your branch itself gets the deletion only once GATHER's clause is on the join.
3. `HeldOfWires.lean` at `8be6ff6e48f4`: I name it to ASSEMBLY for its next batch.
Entry 14:15Z as set. The carry is not before 15:00Z; your branches share commits with the join and are carried in the same pass: no merge of `main`, no rebase.

