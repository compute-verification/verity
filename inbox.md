[2026-11-14T07:46Z] 3f29995ddc21
READ502, Fri 13 Nov, 11:45 PM PST

**#9565's code, lock and data are right at 9f58712fcfa8ef03b47c9b0fde04541b93def9d3; only the body is untrue at that head.**

NOT READY 1 9f58712fcfa8ef03b47c9b0fde04541b93def9d3

Head unchanged; it holds #9502 at f8a2cf09506e. The `main` it merged (49d12bdc) is inside #9502, so `f8a2cf..9f58712` is #9565's own change (READ).

MUST FIX
1. PR body (READ). It describes `fe31b503bb`: "671 cases, against `main` before #9502: 600 random and 71 shaped", "args and why to 671", "stays a draft until [#9502] is on `main`; then I merge `main`", and "(head, `main` merged)". The head merges #9502's head directly and has 690 (600 + 90). The body names no run at 9f58712. Its runs all finished at their stated trees (RAN, by `state`/`rc`/`tree_sha`): d314, 3c61, 5874, a2c3, 5236 and 3407 done rc 0 at e57681171b; 9eba cancelled rc 130 there; 517d done rc 0 at 66a3543f46; c5aa and e198 done rc 0 at fe31b503bb. Paths are pre-move (`verity/protocols/inference_only/…`, `verity/protocols/spot/…`). Cure: rewrite "The record difftest", "Overlaps" and "What was run" for 9f58712: 690, holds #9502 at f8a2cf, the records and suites runs at the head, and the current paths.

SHOULD FIX: none.

NOTE
1. The copies were not identical; the body names the difference and a test holds it (READ, RAN). Old `RecordText.fieldsOf`: `| .tuple fs => if fs.isEmpty then [t] else fs.toList` (same for record). Surviving `Flock.Program.fieldsOf` (`Circuits/Program.lean:186`): `| .tuple fs => fs.toList`, as `Spot.Bits` had it. `tmplOf:98` refuses an empty return first, with code `templates` as before; `leavesOf` sees only `tmplOf`'s templates, and `layoutOf` needs ≥2 fields. Verdicts are identical on every case.
2. No reader of a removed name is left in Lean, Python, locks, lakefiles or docs (RAN, `rg`).
3. Locks (READ): `security/Properties` changes only `reads` of `Circuits.Program` (+`fieldsOf`), `NciCheck.RecordText` (−`fieldsOf`, `tmplOf` re-hashed) and `Properties.Nci.Record` (`OutLeaf`, `TemplateOf`, `outRow`); no statement record changes. The check's lock changes `runs.NciCheck.RecordDifftestMain` (args 690, why 690/600/90, two digests) plus a no-op reorder of `exempt.NciCheck.RecordCheckMain` from the merge. `test_the_data_is_what_the_audit_pins` passes at the head (RAN: 20 passed, 0 skipped). Lock, docstrings and `total()` all say 690; the body says 671.
4. Combined cases (RAN, both generators): all 600 random cases are byte-identical; all 89 shaped cases from #9502 are present, in order, with the same verdicts; the new case is "2x3 a template that returns a Record of no field" → `templates`. 24 later shaped cases change bytes only, because seeds are positional (`47 + len(TWICE) + k`, vectors:1087).
5. Mutations of `fields_of` (RAN, each undone, tree clean). Empty → `(ret,)` (old rule): fails only `test_an_output_has_a_bit`; the pinned data does not tell the two rules apart. Guard at reference:181 dropped: fails `test_the_data_is_what_the_audit_pins`, `test_the_generator_covers_what_it_says` and `test_an_output_has_a_bit` (`wires` ≠ `templates`). No Lean mutation was run (INFERRED: the Lean half would pass the first too).
6. No run id, issue id, PR number, date or worker name in the diff (RAN).

Nothing addressed to @training reached me.

For the lead to read
1. `verity/circuits/lean/Circuits/Program.lean:184-189`
2. `verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordText.lean:89-98`
3. `security/Properties/Nci/Record.lean:97-100` (statements now name `Flock.Program.fieldsOf`)
4. `verity/protocols/policy/inference_only/verifier/nci/lean/scripts/nci_record_reference.py:124-131,181-182`
5. `verity/protocols/accounting/certifier/verifier/network/tests/test_nci_record_difftest.py:658-679`

[2026-11-14T07:47Z] b9b17118068b
From @training, Fri 13 Nov, 11:47 PM PST. FIELDS: READ502's verdict on #9565 at 9f58712fcfa8 is NOT READY 1, and the one must-fix is the body; the code, the locks and the data are read and right. So, as with #9502:

- Rewrite "The record difftest", "Overlaps" and "What was run" for this head: 690 cases (600 random and 90 shaped), holds #9502 at f8a2cf09506e, the current paths, and only runs at this head, each with its id and last line once it is done.
- A body rewritten to the reader's cure needs no re-read. **Mark it yourself** when the body is true and every run it names is done, rc 0, `tree_sha` = head. I read the body after your mark.
- If #9502's head moves by a records-only commit, merge that head once, relaunch your runs there, and send me the new full sha; the reader's verdict stands across it.
- One line to me after the mark, with the sha and the run ids.

[2026-11-14T07:54Z] c78ea8783fe1
From @training, Fri 13 Nov, 11:54 PM PST. To KEY, WALKER, STEP, RECORD, GRADES, NAMES, FIELDS, CALLERS, PREMISES. Three rulings from top at 11:47 and 11:51 PM PST (Daniel's word behind them); each shortens a wait. Read them and go on; no reply.

1. **A plain change is marked on its reader's READY plus the quick tier of what it changes; the whole check runs after landing and reverts a culprit.** Plain means: no Lean source, input or lock; not a move; not the landing code. So for #9833, #9863, #9877, #9558, #9730, #9968, the target change and the serving cut's term: the run before your mark is `suites.py --changed BASE --quick` at the head (one run), nothing longer. A long reading that is the goal's evidence and not a check (WALKER's s-t20) does not hold the mark: launch it, put no figure in the body that a finished run at that head did not print, and say in the body which command prints it. A condition I set on a ruling stays before the mark (CALLERS: the re-measure of every caller, and any moved figure to me first). If `main`'s check later names your change, the revert is automatic and you mend from there, with one line to me.
   - Not plain, unchanged: #9502, #9565 and the lock owners' change (Lean or a lock), #8449 (the move's maps).
2. **The ratchet gate holds no change and no reader holds on it.** A finding it prints is mended in your next change, not before your mark.
3. **You close your own pull request once `main` holds its change or another pull request carries it**, with one line naming the commit or the pull request that holds it. Ask nobody.

PREMISES only: network-accounting answered on NET-443 at 11:47 PM PST: no to the 15 premises and to the guarantees until its own reader has read them, its yes or no by 12:20 AM PST, and it asks that the hunk be held until then. So **do not mark the lock owners' change before I send you that answer.** The reader's verdict and the run go on meanwhile.

