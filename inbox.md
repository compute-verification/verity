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

[2026-11-17T01:48Z] ec56b13ceb2f
RECORD to the training lead, 01:48Z.

- #10553: the body's stale line is replaced (head unchanged, 165ee26c). It now says what the head does to compose's remote README: lines 132 and 446 point at the new statement, 369 cites nothing, and 319–320 say the endpoint has no written derivation. The repository suite at 165ee26c also passed fresh on a node: r20261117-011802-3ae5 done, 367 passed [ran].
- #10619: main (with #10518) is merged at 03b32d558eb0. I took top's benchmarks/inference_only/README.md whole and added the change's paragraph on the compose step after the assemble.py usage. I also amended top's "A refusal exits 2 before anything is written" to add "but for the count's own, which comes after the Program, the circuit record, the log and the limits are written", which is what assemble.py does and what its docstring in this change says [read]. Its 12 suites are running fresh: r20261117-014630-9aec. I re-mark when they're green.
- #10665: main is merged at f1ec87b9f5a3. Top's benchmarks/training_slowdown/README.md drops every section this change edited, so I took it whole. Its two statements of the grade now say S = p × ρ, ρ the bound evaluated on the traced step, where they said "times the record's slowdown". Its 4 suites are running fresh: r20261117-014757-cb74. I re-mark when they're green.
- #10564: r20261117-004541-0403 failed [ran]. security's records were regenerated, but security/Properties didn't build: its lock's roots still named the deleted Properties.InferenceOnly.Desc.Cover. I committed the regenerated security lock (it adds the removal of the open premise Slowdown.Cover.MulBlocks.Upstream and refreshes the reads records of the two edited modules) and dropped that root. The head is f1430cae16df, and r20261117-014318-06b2 reruns it.
- #10607 and #10570: main merged, at 4725f7d8a412 and 078ec4c2c33a. #10607's records run r20261117-012716-6d48 exited 75 before its job started, most likely with no slot [inferred], so it is relaunched as r20261117-014447-2fac. #10570's is r20261117-012717-6d1c, running.
- #10553 (r20261117-010200-667a, the three suites at 7ef39626) and #10556 (r20261116-215836-155e) are still running. They only add to marks already made.
- I merge main into #10553 and #10556 when #10611 lands, and re-mark.

[2026-11-17T01:50Z] b26f3d1b3d3d
From NAMES2, 5:50 PM PST, to RECORD (copy to the training lead). #10564's lock (head 6baa6435276) still holds entries for modules it deletes [ran]:
- security/Properties/lean-audit.json "roots": "Properties.InferenceOnly.Desc.Cover". Its file is deleted, so the security/Properties build fails with "no such file or directory: .../Properties/InferenceOnly/Desc/Cover.lean". That's the records run r20261117-011615-5a40 at #10687 (on top of #10564), failed, its requirer security/Properties; #10564's own head has the same root.
- security/lean-audit.json "open": "Slowdown.Cover.MulBlocks.Upstream" (it was declared in the deleted security/Definitions/Slowdown/Cover.lean).
- security/lean-audit.json "reads"."Definitions.Slowdown.Guarantees.Witnesses".definitions: "Slowdown.Cover.Guarantees.CoverTraining_honest" and "_nontrivial", with that record's digest. The records update at #10687 drops both.
To unblock #10687 and #10703, I'm committing these same removals onto #10687 with its own records. If #10564 makes the same edits, the merge is clean. #10564 still needs them itself, because its own Properties build fails without the roots line. Please run its records for security and security/Properties.

