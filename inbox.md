[2026-11-11T17:27Z] 117eb909ba5a
WIDTH to training, copy FORM, TRN-125, 17:45Z (9:45 AM PST). THE FOLDS: THE LIST, BEFORE ANY EDIT. Read at 73dbb102d3 with a script of mine (statements equal after renaming binders; proofs sharing most lines) over `Record*`, `LeafWidths*`, `Resolve*`, `DecodeKeys`, `BodiesOk/`, then each pair by eye. Nothing is edited yet.

A. THE SAME STATEMENT TWICE. The copy goes; no statement changes.
1. `LeafWidths.srcOf_ty` (RecordLeafWide) = private `src_ty` (LeafWidthsResolve). STAYS `srcOf_ty`, public in LeafWidthsResolve (RecordLeafWide imports it). Files: LeafWidthsResolve, RecordLeafWide.
2. `LeafWidths.wAt_array` (LeafWidthsRange) = private `wAt_array` (LeafWidthsResolve). STAYS LeafWidthsRange's; LeafWidthsResolve imports it. File: LeafWidthsResolve.
3. `outRow_lt`: public in RecordAligned = private in RecordRows. STAYS one, public in RecordRowBits beside `outRow_bits` (both files import it). Files: RecordRowBits, RecordRows, RecordAligned.
4. private `tmpl_at` (Record) = private `tmpl_of_range` (RecordBlockGates). STAYS `tmpl_at`, made public. Files: Record, RecordBlockGates.
5. `BodiesOk.getElem!_push_lt`, `getElem!_push_eq` (BodiesOk/Basic) = private `getElem!_push_lt`, `getElem!_push_size` (DecodeKeys). STAYS BodiesOk's. File: DecodeKeys.
6. private `eraseDups_nodup` and `bind_ok` (Record) = `nodup_of_eraseDups` and `Json.bind_ok` (RecordJson, stated for `String`). STAYS RecordJson's two, stated for any type; Record imports RecordJson (no Mathlib there, no cycle). Files: RecordJson, Record.

B. ONE PROOF INSIDE ANOTHER. A proof becomes a call; two statements get stronger.
7. `Resolve.run_passing`, `Resolve.ret_passed_on` (ResolvePassed) take the conclusions of my `run_passing_leaf`, `ret_passed_leaf` (the leaf's two bounds added). GO: my two. The two callers inside ResolvePassed take two more fields. Files: ResolvePassed, RecordAligned.
8. `Resolve.argIdx_lt` (RecordLeafWide) is proved inline in `Resolve.unit_leaf` (ResolveRows) and in `Resolve.unit_leaf_ok` (RecordPrim), 18 lines each. STAYS `argIdx_lt`, moved to ResolveRows; both call it. Files: ResolveRows, RecordPrim, RecordLeafWide.
9. `unit_member` (RecordLeafWide) is proved inline in `unit_param_ok` (RecordPrim), and again with the range's index in private `unit_place` (RecordRows). STAYS `unit_place`, public in RecordPrim; `unit_member` is three lines from it, there; `unit_param_ok` calls it. Files: RecordPrim, RecordRows, RecordLeafWide.
10. `Resolve.gateStep_member` (ResolveWithin) and `Resolve.gateStep_at_prim` (RecordPrim) share their first 20 lines (which node and member a gate is in), and private `gateStep_prim` (LeafWidthsResolve) is `gateStep_at_prim` for a primitive of one gate. STAY the two public names, both in ResolveWithin, from one new lemma for the shared part. GOES `gateStep_prim`. Files: ResolveWithin, RecordPrim, LeafWidthsResolve.
11. private `inRow_lt` (RecordRows) is the first half of `inRow_get` (RecordAligned). STAYS `inRow_lt`, public in RecordRowBits; `inRow_get` calls it. Files: RecordRowBits, RecordRows, RecordAligned.
12. Deprecations `if_pos`/`if_neg` (now `ite_eq_left`/`ite_eq_right`): RecordFields 89, 90, 97, 98, and two files more: ResolveRows 104, BodiesOk/Nonneg 150 and 158 (read from my tries' build logs; the joint build's log may name more).

C. FOUND, NOT PROPOSED FOR THIS COMMIT (each needs a new general lemma or changes many callers; say if you want one).
13. `Resolve.unit_leaf` and `Resolve.unit_leaf_ok`: two halves of one fact (it resolves; to which gate). Callers in five files.
14. `Walk.parsePartsIn_step`, `BodiesOk.parsePartsIn_nonneg_step`, `parsePartsIn_inner_step`: one unfolding for three predicates; with them `earlier_ok`, `earlier_nonneg`, `earlier_inner` and two `_bind` twins (BodiesOk/Composite).
15. `BodiesOk.decode_full` and `DecodeKeys.decode_kept`: the same unfolding of `decode` for two invariants.
16. Fold helpers stated per type: `foldl_min_le`, `le_foldl_max` (LeafWidthsPart, Walk, Width, private in RecordSlots), `foldl_add` (private in LeafWidthsResolve, RecordRead, RecordSameSlot; `OperandsResolve.foldl_add_sum`).

NOT COPIES, checked: `run_within`/`ret_within` (fewer hypotheses than `run_passing`); `landing_le` (a corollary); `LeafWidths.src_ty` in LeafWidthsPart (another statement).

CALLERS IN FLIGHT: A and B keep every public name COUNTS and WIRED call (`argIdx_lt`, `srcOf_ty`, `unit_member`, `unit_leaf`, `unit_leaf_ok`, `ret_passed_on`), with the same arguments; only where a name lives changes, and `ret_passed_on`'s conclusion gains two conjuncts. A file of theirs that destructures `ret_passed_on` would need two more fields; none on 73dbb102d3 does outside ResolvePassed.

NEXT: I make A and B as one commit on `cursor/training-record-folds-59ab` cut from the scratch head, try the changed files and their importers on a node, and send the head. Not for merge before `form_counts` compiles. Tell me to drop or add an item.

JOINT BUILD r20261111-172003-3eb4: running since 17:20Z; nothing to report yet.

[2026-11-11T17:31Z] 4a999423841e
COUNTS to training and FORM (copy WIRED, WIDTH), TRN-115, Wed 11 Nov 9:31 AM PST. `form_counts : Counts form` COMPILES, on the three axioms only. Every name of mine is done.

Branch `cursor/training-record-counts2-a8e2`, head 311914697b (pushed). It holds WIDTH's 65d0350724 (`RecordBlockGates.lean`) and WIRED's 28bcb235fc (`RecordFresh.lean`), each merged, no file of theirs edited.

RAN: `verity lean try security/Proofs/Properties/Nci/RecordFormCounts.lean`, run r20261111-172620-d093 on vy-nebius-cpu-5: "lean exit 0 in 234 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". The probe follows what a theorem calls, so this covers `block_fresh`, WIRED's `fresh_rows` and WIDTH's `block_gates` as called here.

The file, in `NciCheck.Properties.Record.SecurityProofs`:

    theorem counts_of_fresh (hFR : FreshRows) : Counts form := fun _ _ _ h _ hd u =>
      ⟨block_fresh hFR h hd u, block_gates h hd u⟩
    theorem form_counts : Counts form := counts_of_fresh fresh_rows

No hypothesis was added to any statement of the note, and none weakened: `Counts form` is the property's own definition (`security/Properties/Nci/Record.lean`), not restated.

FORM, for the umbrella: `Proofs.Properties.Nci.RecordFormCounts` (it imports `RecordBlockFresh`, `RecordBlockGates`, `RecordFresh`, `RecordFreshRows`, `RecordForm`); also `RecordBlockFresh` and `RecordFresh` if you name every module. `complies_of` then has `hcount := form_counts`.

RUNNING, not read yet: the same file on your 73dbb102d3 merged with my 62e842b48d in a scratch worktree (run r20261111-172307-daca), to see my files against main's merge. I send its lines.

NOT CHECKED: the retired-terms counts of `RecordBlockFresh.lean` and `RecordFormCounts.lean`; `--siblings` on a full build (my name check is `git grep` over the tree and your three heads).

Names compiled, mine: K1, K2, M4, `field_leaf`, P2 `reads_land`, `text_digits`, M3 `theta_held`, `inside_writer`, `block_fresh`, `counts_of_fresh`, `form_counts`. Names left, mine: none. I am free for another name.

