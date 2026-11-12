# KAUDIT's inbox (the lead: @kernels, bc-026b7d5e-6674-5e9c-9eef-4aac970acc3b)

Read `inbox/RULES.md` on this branch first, then this file. The newest section wins.

## Thu 12 Nov 19:11Z: your brief: HASH-98's closing audit, run again on today's labels: each done-condition with its evidence, and the types' table refreshed. One job, on CPU only

**Who you are.** KAUDIT, a new worker of the kernels lane on `compute-verification/verity`, on the model
`claude-opus-5-5-high` (Daniel's rule of today: Opus 5.5 does the work). You run again an audit the
worker KSTEADY (bc-f944c53d-53aa-581d-ab39-a46f095b1cde) built and ran on Thu 01:41Z, now that today's
rows are measured and labelled, so that the lead can close HASH-98 and its goal a day early.

**What changed since KSTEADY's audit** (HASH-98's comments of today, read them):
- eight Boolean statements were timed again at 1, 2 and 4 provers and the head block at `FIRST=true`
  at one prover (`ksteady-jobs/h98/thursday-result.md`); they are labelled of record and the 18 runs
  and arts they replace are `superseded_by` them (`20261112T1858Z-report-klabel.md`);
- benchmarks ran the figure on those labels at 19:10Z: SPOT 0.0364 prover GPUs per serving GPU,
  every block 11,576 prover GPU-seconds a token, steady rows 99.5% of proving; and said the 17 older
  arts of record in no `beside` list stay.

**The task.** HASH-98 has seven done-conditions (read the issue whole). For each, say whether it holds
on `main` and the store's labels of now, with its evidence:
1. Run the audit again as KSTEADY's handoff says: `~/.research/notes/lanes/kernels/ksteady-jobs/RESUME.md`
   (after `research notes sync --path lanes/kernels`), the bullet "To run the audit again" and the
   "Restore" lines; the verdicts of the last audit are in the bullet "The verdicts" above it and in
   `h98/result.md`. CPU only, no node A, about ten minutes, most of it the label pull. Use `main` of
   now (write its hash out; the shell hook refuses a variable there).
2. **Compare with the last audit, condition by condition**, and say what moved: the share of proving
   priced from steady rows (Flock and SPOT), the sixteen rows, the fields of each row at 1, 2 and 4
   provers, the labels (how many arts of record, how many superseded, none failing labelled), and for
   condition 6 what proofs has said since (HASH-98 and HASH-115: its confirmation, row 15's pin, the
   head block's Program).
3. **Refresh the comment condition 7 asks for:** each type's GPU-seconds a statement and ms a block
   with its run id, at 1, 2 and 4 provers, for benchmarks' rows 1 to 16 (`h98/post-types-1.md` is the
   last one) with today's re-timed rows in it and what each replaced; the 48 other types only if a
   figure of theirs moved.
4. Post it: one worker comment on HASH-98 with the conditions' verdicts (at most 100 words) and one
   with the table (`research msg linear comment HASH-98 --file F --worker KAUDIT`), never through the
   Linear connector's `save_` tools. **Label nothing and change no script**; if the audit names a row
   that should be labelled or superseded, say which and why.

**Traps the handoff names:** the command's stderr and `research data … --json` name the store's
remote: they stay in `/tmp`, never beside the note or in a comment. Two runs end in `c5c7`. A
`research run` launch line names a pod id: mask it.

**Return by 20:45Z** with each condition's verdict and evidence, what moved since the last audit, the
two comments' places, and anything that stops the issue from closing. End your turn with it: I close
the issue and the goal on your return.
