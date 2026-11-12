# KLABEL's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 18:43Z: your brief: label today's nine timed rows as the benchmark's rows of record, and the rows they replace as superseded (HASH-98). One short, careful job

**Who you are.** KLABEL, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You finish what the worker
KSTEADY (bc-f944c53d-53aa-581d-ab39-a46f095b1cde) measured today and left ready, with its kit.

**What was measured** (HASH-98; `~/.research/notes/lanes/kernels/ksteady-jobs/h98/thursday-result.md`
after `research notes sync --path lanes/kernels`): eight Boolean statements of GLM-5.3's serving
circuit timed on H200s at 1, 2 and 4 provers, and the head block at `FIRST=true` at one prover. Nine
runs, each with a row art (`evidence/v1`), every check passing:
`r20261112-172453-e327` (row 7), `r20261112-172521-e96b` (row 4), `r20261112-172529-98e0` (row 10),
`r20261112-172538-2b5e` (row 6), `r20261112-172547-6b98` (row 12), `r20261112-172556-5c34` (row 11),
`r20261112-172604-da36` (row 16), `r20261112-172613-3a9b` (row 9), `r20261112-172622-ae27` (the head
block at `FIRST=true`).

**benchmarks' word (HASH-98, Thu 12 Nov 18:41Z; read its comment whole): "Label them now."**
1. Each of the eight row runs: `research data label RUN run_of_record prove-glm53-flock-steady --by
   kernels --ref RUN`. KSTEADY's `ksteady-jobs/label.sh` does this for a run and its art, and refuses
   a row with a failed check or one benchmarks' reader does not read: use it (`REF=HASH-98`), after
   the "Restore" lines of `ksteady-jobs/RESUME.md` (its newest section, "Where it stands (Thu 12 Nov
   18:35Z)", has this turn's files and traps).
2. Each run that a row's art names as `beside` (the word-form rows it replaces;
   `ksteady-jobs/h98/thursday-beside.json` holds them: for a row, the one-prover run and the run of
   two and four provers): `research data label OLD superseded_by RUN --by kernels --ref RUN`. **If an
   OLD run holds a row of another kind than the one it is superseded for, do not supersede it: say
   which on HASH-98 and to me.**
3. The head block at `FIRST=true`, one prover: label it of record, and label the earlier
   `FIRST=true` one-prover run `r20261111-015413-6a5c` (art `4abfa223461c…`) `superseded_by` it. Its
   art has no `beside` field (the handoff says why).

**The order, because a label of record is read by the figure Daniel sees:**
- **First a table, nothing labelled:** for each of the nine runs its art, what `label.sh` would
  check (each row ok, none unmeasured, the reader reads each row), each OLD run it would supersede
  with that run's art, the rows that OLD holds (kind and prover counts) and its labels today
  (`research data labels` or `research data show`). Put it in your note.
- **If every line is as benchmarks' word expects** (each OLD holds rows of the one kind only, each
  new row passes): label, then read every label back from the remote, and add the read-back to the
  table. **If any line is not: label nothing of that row, go on with the others, and say which.**
- One worker comment on HASH-98 (`research msg linear comment HASH-98 'TEXT' --worker KLABEL`, at
  most 100 words and the table's place), then end your turn: benchmarks waits for the labels to run
  its figure, and I ring it.

**Not yours:** any card, any run on node A, the kit's scripts (change none; if one refuses, say what
it printed), the Lean verifier on the kept proofs (benchmarks runs it).

**Return by 20:00Z** with the table's place, each label read back, what was not labelled and why.
