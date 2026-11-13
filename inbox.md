[2026-11-13T00:21Z] 8d89b1d54bb9
FIELDS, Thu 12 Nov 16:22 PST.

**Job 1's rename is done and passes its lean try and the retired-terms test; job 2's one definition is `Flock.Program.fieldsOf`, right after `end Ty` in `verity/circuits/lean/Circuits/Program.lean`, and I read the ruling as you wrote it with one addition (#9038 overlaps my proof files).**

Job 1 (branch `cursor/leafwidths-bound-ed69`, head `b50d1bb4f0`)
- The binder is now `bound`, and the two hypotheses `hcap` are now `hbound`. The docstring is rewrapped so it stays within 120 columns. The allowlist entry for the file goes from 10 to 5. READ: the lint's own scan of the file leaves 5 sites, all `Nci`/`NciCheck`.
- RAN r20261113-001227-c7c4: `verity lean try` on the file. lean exit 0; "none of its 12 theorems uses an axiom beyond propext, Classical.choice and Quot.sound".
- RAN r20261113-001416-760a: the `repository` suite on `tests/lint/test_repo_retired_terms.py`. 13 passed.
- READ: no lock lists `residues_mem`. RAN r20261113-001957-d299 (in flight): `lean_changed.py --records security/Proofs --update`, to confirm no record moves. Before it I launched the same command once under a 60 s `timeout`, which cut the client off before it printed a run id; `research status` lists no run for it.
- Next: open the draft.

Job 2: the plan
- The definition: `def fieldsOf : Ty → List Ty` in namespace `Flock.Program`, as `Spot.Bits.fieldsOf` states it (a Tuple's or a Record's fields, else `[t]`). It goes after `end Ty` in §2 and adds nothing else to that file.
- Files I will change:
  - `verity/circuits/lean/Circuits/Program.lean`: add the definition.
  - `verity/protocols/spot/verifier/lean/Spot/Bits.lean`: delete `fieldsOf`; `retWords` reads `Flock.Program.fieldsOf`; fix the module docstring's list. No new import: it already imports `Flock.Program`.
  - `verity/protocols/inference_only/verifier/nci/lean/NciCheck/RecordText.lean`: delete `fieldsOf`. `tmplOf`'s first check becomes `if (fieldsOf defn.ret).isEmpty then throw "templates"`, and its docstring changes to match. I put the check first so the proofs, which unfold `tmplOf` and name hypotheses counted from the end, keep their names.
  - `security/Properties/Nci/Record.lean`: only the `open` lines change. `fieldsOf` leaves `open NciCheck.RecordText (...)` and joins `open Flock.Program (...)`; the statements' text stays the same.
  - The 13 files under `security/Proofs/Properties/Nci/`: the same `open` change, plus `RecordPrim`'s qualified name. `RecordFields.field_leaf` and `RecordRows.fields_leaves` handle a Tuple or Record with no field differently: `fieldsOf` is now `[]` there, so `f < 0` is absurd and the leaves sum to 0. `RecordJson.tmplOf_spec` and `Record.tmplOf_ok` should hold as they are.
  - `scripts/nci_record_reference.py`: change `fields_of`'s hunk, and make `Template` refuse an empty `fields_of` with `templates`.
  - `scripts/record_difftest_vectors.py`: only the comment at its line 815, which says a return of no field is its own one output.
  - The locks, through the tools: `security/Properties/lean-audit.json` (the read of `NciCheck.RecordText.fieldsOf`, and `TemplateOf` and its neighbours now read `Flock.Program.fieldsOf`), plus whatever the check's, Spot's and Proofs' records show.
- `NciCheck/Record.lean` needs no change (READ):
  - It opens `Flock.Program` whole, so line 316 and line 593 read the new definition.
  - Line 316 (`leavesOf`) runs only on the templates `templatesOf` returned (line 426: `ts.map (leavesOf P passed)`), and the new line in `tmplOf` refuses a template with no field first.
  - Line 593 (`layoutOf`) matches `_ :: _ :: _`, so it needs two fields. A return with no field gave `[t]` before and gives `[]` now; both are `none`.
- The difftest already pins the refusal (READ): `BITLESS` has "a template that returns no field", whose return is `Tuple()` and is refused `templates`. It also has a field of no bit beside a 16-bit field. I add no case unless you also want an empty Record.

Open pull requests touching my files (READ: each head diffed against its merge base with `main`)
- #9038: the retired-terms allowlist, in a hunk two lines below job 1's line, so I expect no conflict. Also `RecordAligned`, `RecordFields`, `RecordLeafWide`, `RecordPrim` and `RecordRows`, which job 2 changes. Its `RecordFields.field_leaf` edit (`if_pos`/`if_neg` to `ite_eq_left`/`ite_eq_right`) sits in the lines I change, so whichever of us lands second merges by hand.
- #9071: since its merge base with `main`, it changes only `DefinitionDigest`, `security/Properties/lean-audit.json`, the check's `lean-audit.json` and the allowlist. `gh pr diff` lists more files, but those are already on `main`. The lock is the overlap.
- #9474: the reference, in hunks away from `fields_of` and `Template`; also the check's `lean-audit.json`.
- #9502: `NciCheck/Record.lean`, `RecordText.lean`, the reference, the vectors, the difftest test, `RecordTemplates.lean` and the check's lock. None of its hunks touches `fieldsOf`, `tmplOf`, `fields_of` or `Template`. Its `RecordTemplates` hunk is in `finish_wiring`, where I make no change.
- #9519: none of my files.

Where I read the ruling differently: only in the name. I keep `fieldsOf` as a top-level `Flock.Program.fieldsOf` rather than `Ty.fields`, so every reader that opens `Flock.Program` reads it unchanged. Say if you want `Ty.fieldsOf`.

