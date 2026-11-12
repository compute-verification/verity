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

[2026-11-12T21:31Z] daf1948d54de
NOTES to training, Thu 12 Nov, 1:40 PM PST (from `date`).

**Two of your ranges say something other than the rows (algorithm-policy's S, size-scaling's split of rho and S); frontier-fit ends at "computes them."; no row of either set is on any of the four tables; three notes need cuts I ask your yes for, and frontier-fit's note can't go into its builder on the put branch.**

Sources: RAN = arithmetic on my VM on fetched files of art:2f75849c… (`part1r/rows-page.tsv`, `build/out/asif/ss/{control,asif-steps}`, `standard-workloads/out/{control,asif}/console`), art:55b35cfb… (`bodies/`, `algorithm-policy/tables/`) and the run record of r20261112-163437-901a (`out/census.json`). Each control equals the staged body row for row (RAN).

## 1. The ranges

- algorithm-policy, the page's 197 rows: 194 have a factor (3 leave the page in the as-if; their search rows are inside the range or have no rho). rho_own 0.399 to 0.628 over 163 rows (rho_default and rho_served the same to 3 digits). **S = p rho_own is 0.432 to 0.667 over the same 163**, because p moves 0.782 to 1.311 with H. Your sentence says "S is 0.40 to 0.63": that is rho_own's range. I propose "at 32 bits S is 0.43 to 0.67 of this" (same length). [RAN] The stored tables (overlay, 288 rows with rho): rho 0.399 to 0.628, S 0.432 to 0.684. [RAN]
- size-scaling, 90 rows: 45 standard-training rows, rho = S, 0.475 to 0.623 (with their traced steps at 32 bits, the third build); 12 traced attack rows, rho 0.432 to 0.552 and S 0.432 to 0.552; 33 rows with no rho. So the rho column runs 0.432 to 0.623 and so does the S column. "rho is 0.475 to 0.623, S 0.432 to 0.552" would be read as the columns' ranges, and an attack row's rho falls below it. I propose "at 32 bits rho and S are 0.43 to 0.62 of this" (12 characters shorter). [RAN]
- standard-workloads: the 16 measured rows 0.2089 to 0.5165 in each of rho_own, rho_default, rho_console and rho_served: yours holds. The two modelled 45M rows (rl, inference) read no record and keep a factor of 1. "a measured rho" would be exact but costs 11 characters and one more cut; I keep yours unless you say. [RAN]
- Gates: 1.526 to 2.553 over the page's 194 rows, which are 161 runs, not 194. [RAN]
- frontier-fit: the v3 fit's 4 points and 6 starts are the starts' own checkpoints: no page run, and the re-pricing didn't rewrite v3's starts (they stay put in it). v2.3-pilot's points include page runs: ref-optim-000096@458 and ref-optim-000097@917 are r20261102-134355-7c3b@458 and @917 (C0 + |C| equal to 6 digits) [INFERRED]; its 4 starts and 10 points move 1.870 to 1.884 [READ, the artifact's README]. By your rule, the sentence ends at "computes them." (69 characters).

## 2. The two sets of rows: none is on the page, in the stored tables, in size-scaling, or in standard-workloads, frontier-fit or the finals table [RAN: every run id each body and table names, against both sets]

- The six: r20261102-135540-8833 and r20261102-142321-a207 (45M · 1.71 tok/p, v2.3-pilot) at g 1.1, 2 and 11, none reached. [READ: note training-20261111T1728Z…six-rows-read; RAN: rho factor 1.000 on each]
- The 42: 14 runs at 36M · 10.8 tok/p, v3, at g 1.1, 2 and 11, none reached: r20261103-051120-be90, -052351-8d97, -055523-9aca, -062249-c1a1, -063939-2c0c, -072518-b6e5, -075914-ae6d, -082447-e30a, -092604-410b, -101524-0d3d, -110633-453e, -120914-5466, -133241-18b3, r20261104-165055-b721. [RAN: the 20 runs with efficient calls in census.json, joined with the search table's rows]

So I write them, by run and row, in `README.md` inside `algorithm-policy/tables`. That changes the stored tables' id, so the note's id changes (12-character form).

## 3. Cuts I ask your yes for (algorithm-policy needs none: your steps 1 and 2 bring it to 498)

**algorithm-policy, 498** (steps 1 and 2 only; S's range as in section 1):
`Corrected: a search record stated one forward's tokens for a step of 2 to 64, so S fell on 24 rows (up to 0.23%) and rose on 37 (up to 72%). Per start, g: finals points and other reached search rows (search split, 1 seed). Finals on the search split: 9 of 15 cells. rho_own: search trace, k = 65,536. S = p rho_own. Attention's products are priced at 16 bits, as the card computes them; at 32 bits S is 0.43 to 0.67 of this. Rotations are at the TF32 product. Cut (tables). Tables art:XXXXXXXXXXXX.`

**size-scaling, 486 -> 495.** Before:
`Per size, gain and policy (gates: AND + XOR + NOT, 3.0x a bf16 GEMM's AND count): standard training (traced at full size, p = 1) and the least S = p*rho over reached attacks, never partial, stub or unmeasured; if only untraced runs reached, the least p, rho null. Headline: grown, the size's own 4,096-in 1,024-out session at k 65,536 (rho at its packed k_bits is at least this). Sensitivities: console (45M), serving (k 1.6e6, its own 100,000-in session). 1B: reference size, no start.`
After:
`Per size, gain and policy: standard training (traced at full size, p = 1) and the least S = p*rho over reached attacks, never partial, stub or unmeasured; untraced alone: least p, rho null. Headline: grown, own 4,096-in 1,024-out session at k 65,536 (rho at packed k_bits is at least this). Sensitivities: console (45M), serving (k 1.6e6, own 100,000-in session). 1B: no start. Attention's products are priced at 16 bits, as the card computes them; at 32 bits rho and S are 0.43 to 0.62 of this.`
Drops the gate unit (the other tables' titles carry it; this title doesn't), "reference size", and shortens three phrases. With your sentence as written it is 508: it needs one more cut.

**standard-workloads, 499 -> 493.** Before:
`Training's slowdown code 4.13.6, reading A. Headline rho_own: own session (4,096 in + 1,024 out on the served model; distill's teacher), k = 65,536 (16-bit ids); at the packed 61,440 rho is at least this. Sensitivities: rho_default = rho_console (45M), rho_served (serving block); distill at its student's session: 20M 111,164, 60M 208,452. rho_HE: theorem of record, given (H), (E); fresh_old rests on P-width extended, false on honest circuits. Tables run:r20261111-213443-46e2/out/restate/tables.`
After:
`Reading A. Headline rho_own: own session (4,096 in + 1,024 out; distill's teacher), k = 65,536; at the packed 61,440 rho is at least this. Sensitivities: rho_default = rho_console (45M), rho_served (serving block); distill at its student's: 20M 111,164, 60M 208,452. fresh_old rests on P-width extended, false on honest circuits. Attention's products are priced at 16 bits, as the card computes them; at 32 bits rho is 0.21 to 0.52 of this. Tables run:r20261111-213443-46e2/out/restate/tables.`
Drops "Training's slowdown code 4.13.6" (every row's te_core cell says it), "on the served model", "(16-bit ids)", "session" after student's, and "rho_HE: theorem of record, given (H), (E)" (no column is named rho_HE; every row's rests_on cell says "theorem of record ... given (H)").

**frontier-fit, 477 -> 486.** Before:
`Per data, held-out bpb L at gates C: E+A*C^-gamma, fit in relative loss to its frontier, the best of the attacks run so far at this size: least L per C (from scratch, starts at C0, checkpoints at C0 + |C|; v3 from scratch only), a point per 1/20 decade. C_eff=(A/(L-E))^(1/gamma); gain C_eff(L)/C at a point, C(L)/C(l0) at a target. v3 provisional (4 from scratch, 5 the target): s-t20 +0.00%, m-t20 -0.01%, l16-t20 +0.00%, xl-t20 +0.00%. Ladder points built from its manifest.`
After:
`Per data, held-out bpb L at gates C: E+A*C^-gamma, fit in relative loss to its frontier, the best of the attacks run so far at this size: least L per C (from scratch, starts at C0, checkpoints at C0 + |C|; v3 from scratch only), a point per 1/20 decade. C_eff=(A/(L-E))^(1/gamma); gain C_eff(L)/C at a point, C(L)/C(l0) at a target. v3 provisional: 4 from scratch, 5 the target. Ladder points built from its manifest. C prices attention's products at 16 bits, as the card computes them.`
Drops the four residuals "s-t20 +0.00%, m-t20 -0.01%, l16-t20 +0.00%, xl-t20 +0.00%" (each is its point row's residual_pct cell).

## 4. Two things you should know before I build

- **frontier-fit's note can't go into its builder on the put branch.** It is te_core/frontier_fit.py's `note`, and `console.refusal` refuses every put (all four, and `--dry` prints it) whose te_core tree differs from cursor/training-replot-55a1's head; today the put branch's te_core equals it [RAN: git diff]. So frontier-fit's new note lives in its body only; the builder's change goes into a te_core version on the replot branch, yours to order. The other three builders (scorer/plot2_final.py, size_scaling.py, standard_workloads.py) are outside te_core, and their notes go there.
- The builders will then hold the factors as literals: measured numbers, on a branch that never lands on main. I do it as you asked unless you say otherwise.
- The read-back: `put_fit` and standard_workloads.py's put both compare with `!=` (and standard-workloads' `--served` guard too). I fix all three to compare as numbers.

I now build the staging with these notes in place (the read-back fix, the tables' README, the bodies, the comparison against the served page); nothing is stored as the staging artifact until your yes on sections 1 and 3, and a changed note is a rerun of one script.

