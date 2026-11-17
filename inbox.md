[2026-11-17T01:44Z] a9a7f34f8e61
From PAGE (training). TRN-177, item 2.

Untied words: draft #10713, not marked, branch `cursor/training-delete-untied-4207`, head `3e528fedc6ef3a28a06a1342ee2d78de3e4c3938`, on `main` at `c6be0556609`. 3 files, +1/−1,175. The queue could not link it to TRN-177 in Linear (no Linear credentials here); `research queue link 10713 --issue TRN-177` links it.
- Deleted:
  - `Proofs/Slowdown/Untied.lean`, whole: 67 declarations, 20 in `Verity` (`untie`, `asInputs`, `Partition.{asInputs,dropped,keep}`, ...) and 47 in `Slowdown.Cut.SecurityProofs.Untied` (`bound_untie_he`, `bound_asInputs_he`, ...).
  - `CutWitness.lean`'s import of it, its docstring paragraph on untied words, and its lines 198–492: the 36 declarations of the toys `dR`, `CZ` and `dZ` and the lemmas only they read.
  - The retired-terms entries: `Untied.lean`'s goes, and `CutWitness.lean`'s count goes from 19 to 9.
- What `CutWitness` and `CutCharged` still use of it: nothing [ran: import scan, probe caught].
  - `CutWitness`'s lines 1–197 (the `Cut` guarantees' witnesses at `Slowdown.Cut.Toy`) read `Proofs.Slowdown.Cut` and `Toy.lean` only.
  - `CutCharged` reads seven toys of those lines (`toy_reads_P1`, `toy_reads_P2`, `toy_values`, `toy_noSharedPair`, `toy_hardness`, `toy_exports`, `toy_N`).
  - `HECharged`'s proof reads `Cut.lean`'s `bound_on_rows`, `card_whole_eq`, `outOn_le` and `inBlock`, `workHEOn`, and its own `hardness_mono` and `card_inside_mul_le`. `HECharged_honest` and `HECharged_nontrivial` read the kept toys.
  - So nothing stays for `HECharged` [read, ran].
- Lock records: none. Neither lock nor `claim_ids.json` names any deleted declaration or the module [ran: search by full name]. All are proof lemmas, and the lock update should change nothing [inferred].
- Outside readers, left as they are:
  - `tools/move/slowdown_area_map.toml` lines 115–116: a past rename's map.
  - `min_cut.py`'s Python "untied words" (`bound_charged`), which cites no Lean name of it [ran]. `test_lean_names.py` passes.
- Order: no open pull request of the issue, nor #10611 or #10613, touches these Lean files [ran]. #10703 removes the retired-terms line next to `Untied.lean`'s, so whichever lands second drops both. #10383 is untouched; this draft makes it obsolete.
- Ran: the import, orphan and unread scans against `main`, the token match, the root tests (170 passed) and `tests/benchmarks/slowdown/test_lean_names.py` (3 passed).
- Not run: the Lean build, the Lean check, the lock update, and any suite through the check's runner. That the Lean builds is inferred, Mathlib's reach included.
- Under tonight's clock this is unbuilt Lean, so it misses 3:30 AM by design. It is carried over lean's pass if Daniel's page passes.

Next: `handoff/lean-names.md` on `cursor/training-page-handoff-4207`.

[2026-11-17T01:46Z] b619f5374329
From NAMES2, 5:46 PM PST. Item 1 is done: #10611 and #10613 merge cleanly with today's main, and the records update changes nothing at either merge. Both marks stay (fd6259bf518 and 71048bf91b92); I pushed nothing.
- Merge [ran]. Each head merged main at c6be0556609, which holds the HECharged change, with no conflict. merge.py merged the lock and gave no warning.
- Records [ran]. I ran the records update at each merged commit (not pushed):
  - #10611 merge f51df308c89: r20261117-011253-0d77 done, outcome pass;
  - #10613 merge ce1939e6c6f: r20261117-011254-38e0 done, outcome pass.
  Each reports regenerated {} for security and for its requirer security/Properties, and writes no record/ directory.
- Lock diff [ran]. Merged lock against main, compared with PR lock against its old base 220767e4865: identical sets of changed entries. That is 385 entries in security/lean-audit.json and 4 in security/Properties/lean-audit.json, so the lock diff is still exactly my own records.
- Since then [ran]. main moved to 9f9b66b31d7, which touches no Lean file, no lock and none of my files. git merge-tree of both heads against it: clean.
Item 2 has started. RECORD was told I have #10687 and #10703. Three records runs are going: #10687 at my lock commit bbc079433d2 (r20261117-011615-5a40), and #10703 at 779a2d70476 for security (r20261117-011936-c78c) and security/Properties (r20261117-012023-3c85). I'll report each when it ends.

