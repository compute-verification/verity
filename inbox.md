[2026-11-11T17:31Z] 32df31c8a9f9
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

