[2026-11-11T17:22Z] 241ca3561357
COUNTS to training and FORM (copy WIRED), TRN-115, Wed 11 Nov 9:22 AM PST. `Counts form` COMPILES FROM ONE HYPOTHESIS, `FreshRows`, WIRED's statement. One line is left.

Branch `cursor/training-record-counts2-a8e2`, head 62e842b48d (pushed). It holds WIDTH's `RecordBlockGates.lean` (65d0350724, merged at 7fd3617700).

RAN, each a `verity lean try` on a node, lean exit 0, its theorems on the three axioms only:
- `RecordBlockFresh.lean`, run r20261111-170505-17a8 (and r20261111-171440-217e), 4 theorems:
  `block_fresh (hFR : FreshRows) (h : read j tr = .ok c) (hd : Served form c d) (u : Fin d.n) : Nci.novelReads d (d.𝓘.gates u) ≤ (form.summarize c).fresh`;
  `landing_bits`: a row some read lands through is a row of a template of the record, inside its port's cut, and the gates its reads land on are no wider than `rowBits`; `cut_outer`; `lookup_of_mem`.
- `RecordFormCounts.lean`, run r20261111-170526-a65f, 1 theorem:
  `counts_of_fresh (hFR : FreshRows) : Counts form := fun _ _ _ h _ hd u => ⟨block_fresh hFR h hd u, block_gates h hd u⟩`.
Its first try, r20261111-170010-b906, failed on a missing import only (`tmpls_named`).

LEFT, mine: `form_counts : Counts form := counts_of_fresh fresh_rows`, one line and one import in `RecordFormCounts.lean`, when WIRED's `RecordFresh.lean` is pushed (its node try is running, WIRED's 17:13Z message). Then one node try and I send the head.

FORM, the modules for the umbrella, all in `NciCheck.Properties.Record.SecurityProofs` unless said: `RecordReads`, `RecordWireRead` and `RecordFreshRows` (both `NciCheck.Record.SecurityProofs`), `RecordKinds`, `RecordBlockFresh`, `RecordFormCounts`; WIDTH's `RecordBlockGates` comes with my head. No name of mine is declared twice in the tree or on your a6de2c785c, WIDTH's or WIRED's heads (READ: `git grep` of each declaration line; `rowBits`, `rowBits_eq` and `find?_one` exist elsewhere under other namespaces).

NOT CHECKED: a build of these on your a6de2c785c; the retired-terms counts of the new files.

Names compiled, mine: K1, K2, M4, `field_leaf`, P2 `reads_land`, `text_digits`, M3 `theta_held`, `inside_writer`, `block_fresh`, `counts_of_fresh`. Names left: `form_counts` (mine, waits on `fresh_rows`), `fresh_rows` (WIRED's).

