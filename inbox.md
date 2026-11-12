[2026-11-12T21:30Z] 60fde51446a5
KEY, Thu 12 Nov 1:28 PM PST.

**Lean, reference, vectors and the proof edits are pushed on `cursor/trn147-form-reads-formats-10f7` (head 99c3617e12); no PR has opened yet and I have no PR tool in this session, so the draft needs you or the environment; the proofs, the nci Lean check with the lock, and the twelve served cases are all queued on nodes.**

State
- Branch `cursor/trn147-form-reads-formats-10f7`, head 99c3617e12, base main 5f0c008aa0. Two commits: the form (Lean, reference, proofs), then the vectors and the network test. (RAN `git push`)
- No PR exists for the branch (RAN `gh pr list --head ...`: empty). The session has no tool that opens one. The body I'd give it is at the end of this message; open it as a draft, or tell me how.
- `NciCheck/Record.lean` and `RecordText.lean` elaborate: lean exit 0, the axiom probe "no theorems in it" (RAN r20261112-204314-1ac2 on cpu-1).
- Reference and vectors: 687 cases (600 random, 87 shaped). Every broken random case refuses at its stage, every unbroken one is taken, the 13 new ways all come up below seed 500, and the generator's θ list equals the reference's at each format's width (RAN locally, Python only, the reference against itself; the Lean half is the check run below). Data SHA-256: cases d0ec334e56a4…, python 43a29860b51a… (RAN locally); the lock gets them from `check.py --update` on the node.
- The network suite's quick tests in `test_nci_record_difftest.py`: 13 passed, 1 deselected (`slow`) (RAN locally, `pytest -m "not slow"`). Not yet run by the pre-mark recipe on a node.

Decisions I made (decision 6: plainest, telling you now)
1. F1 is `RecordText.formatsOk`, a pure guard right after the tables loop: `formats` an object; each key a registered parameter, not a table's; keys exactly `row_schema`, `word_bits`; schema one of the two names; bits 16 or 32; not v2 at 16; 16·leaves divisible by bits. The key test became `registeredKeyed`: the five, plus `values`, `formats` or both. The read test sits in the reads loop's span test: 16·(row words / cut) divisible by b(writer). All `inputs`.
2. F2 is `Record.formatsHeld`, called in `finish` right after `inputsOf`, code `formats`. For every registered parameter the record has: where the run's own record registers it (its last entry), that entry's root must be the one the run's window commits, its `word_bits` (16 if absent) must equal b(p), and rows·words·b must equal 16·leaves. Where the run's record doesn't register it, a parameter with a format must be θ at b(p), and one with no format passes as today. This holds parameters with no format too, when the run's record registers them. That is new: a record whose run's entry disagrees on root or shape now refuses `formats` even with no format at all. It is what "over every param the run's record registers" says. If the served record has such an entry, case 1 shows it.
3. θ is `isTheta T leaves b p`: rows·words·b = 16·leaves and wordBits = b.
4. Dropping the key refuses through F2 (the 32-bit entry against b = 16). `formats` sits in the module docstring's list after `inputs`, as item 11.
5. Random vectors: where the run's record opens the log (before = 0), the run's record now holds the first record's entries, so F2 reads them. The generator keeps only the entries consistent with the run's commit. This changes which held entries those older random cases carry (Python only, INFERRED from the code). Formats use their own random stream, `random.Random(f"formats/{seed}/{attempt}")`, so every other draw is unchanged.
6. Tenth test: the one predicate is `_formatted(made)`. Its docstring: the interface states no row format and refuses such a record, which the form reads, so the tenth test leaves it out. I also added `formats` to `OWN`: the interface never reads the log, so a `formats` refusal on a record with no `formats` key (the key dropped, a run's entry disagreeing) is the form's own. Without that, the tenth test fails on those cases.
7. `own_rows`: untouched. No new vector refuses there; the new refusals are `inputs` and `formats` only (RAN: the stage check above). My reference hunks are at old lines 71, 430, 442 (the formats guard after the tables loop), 451 (the read test), 515–526 (`theta`, plus new functions after it) and 611/615 in `expand`. None of PROBE's.
8. New shaped cases (`FORMATS` in the vectors, 17): the scan state in 32-bit leaves of the run's record, its format right in segment rows and in v2 rows (accept); a weight registered in 32-bit leaves, its format right (accept), and with no format (taken, its rows fresh: `nci`/`fresh`); another width, no format, a 16-bit run's entry under a 32-bit format, a format the run's record doesn't register, a 32-bit format on a weight registered in 16 bits (each `formats`); and 8 F1 shapes (each `inputs`). The accepted ones equal the served-shape case's verdict and count (RAN locally: the new test).
9. The probe on the evidence branch got `--format P=JSON|P=-` (set or remove one parameter's format; the record re-pointed as `--drop` does), and its θ line takes the width. Tell PROBE's worker: `served_record_case.py` calls `ref.theta(T, x.params, p)`, which now needs a fourth argument, `ref.width_of(rec, p)`. Branch `cursor/trn147-served-cases-10f7` (dbe1d8ede3, never PR'd).

Runs in flight (all queued, waiting for slots)
- r20261112-211519-88a2, r20261112-211519-15b2, r20261112-211518-6aa2: `verity lean try --on auto` on RecordRead, RecordTemplates, RecordInputs (cpu-1).
- r20261112-211705-84f9: `check.py --build --update` on the nci package: its Lean check, the difftest's Lean half on the 687 cases, and the lock's `runs` data (cpu-11, 32 GB).
- r20261112-212235-e658 (cpu-14), r20261112-212342-43a8, r20261112-212342-1920 (cpu-4): the served cases, four a run, each with its reference verdict and its compiled verdict, 64 GB each. A: as served, a `pearl-a/0/0` width 16, a format for `weights/model.layers.0.input_layernorm.weight@2048`, a schema of another name. B: `--drop registered.formats`, `pearl-a/0/0`'s format removed, less smallest, less largest. C: less unread `o_proj`, no pins, α held, α held less smallest. Inputs sent from r20261112-191043-6003.

Left, in order: the proof tries' verdicts (then fix any proof); the lock's data from 84f9, committed; the suites by the pre-mark recipe and the security proofs package's Lean check on a node; the served verdicts into the body. I hold the 6 PM date for everything but the draft's opening, which I can't do from here.

Draft body (for the PR):
---
Lane training
Refs TRN-147
Goal training/check-reads-served-record

The record form reads `registered.formats`, the row format a served record states for the values whose rows are not 16-bit leaves, and holds each width to the registration log.

- Shape (`RecordText.formatsOk`, refusal `inputs`): `registered` may carry `formats`. Each entry names a registered parameter that is not a table's, has exactly `row_schema` and `word_bits`, a schema of `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1`, 16 or 32 bits (not 16 in v2 rows), and leaves that divide the parameter's 16-bit leaves. Each registered read of it reads whole leaves of that width.
- Held to the log (`Record.formatsHeld`, new refusal `formats`, after `inputs`): of each registered parameter the run's own record registers, that entry's root is the one the run commits, its `word_bits` is the parameter's width, and its leaves hold the record's. A parameter with a format that the run's record does not register is θ at its width.
- θ takes the width (`Record.isTheta`).
- The schema's name is held to the set of two names, to nothing registered: no registration entry states it.
- Reference (`nci_record_reference.py`): the same rules at the same stages and codes. Vectors: random formats, 13 new broken ways and 17 made cases (`FORMATS`). The network test's tenth test leaves out records that state formats (`_formatted`), which the interface does not read yet.
- Proofs: `finish_text`, `finish_wiring`, `inputsOf_decls` and `finish_case` take the new stages.
- Lock: `runs.NciCheck.RecordDifftestMain`, 687 cases.
---

