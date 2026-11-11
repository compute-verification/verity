[2026-11-11T09:24Z] 102064ba1217
grade to training, TRN-102 long runs and the 14. Wed 11 Nov 09:24Z. Ran unless marked.

YOUR 09:07Z RULING, done. Cancelled by run id at 09:07Z, each at 33 to 37 GB against 32 declared: fe07 (r20261111-081546-2281, r20261111-081601-c649), 8452 (r20261111-081640-4aa3, r20261111-081657-ea42), 106f (r20261111-081802-2587, r20261111-081832-8433). Relaunched 09:14Z, the slot reserving the declared figure, and each job stops its own trace at that figure less 1 GB (trace1m.sh), so none passes it unwatched:
- fe07, 66 GB: old r20261111-091355-f79a (cpu-6, running), new r20261111-091423-b223 (cpu-7, in line).
- 8452, 74 GB: old r20261111-091437-176b (cpu-1, in line), new r20261111-091450-a2c7 (cpu-6, running).
- 106f, 56 GB, not 42: old r20261111-091459-78d8 (cpu-9, running), new r20261111-091520-91cc (cpu-9, in line). A program that plans its steps from the budget is recorded three times (base, halved, windows-only) and the base stays; 106f peaked at 32.8 GB after two, so I project 45 to 47.
- e30a kept at 32: its first trace peaked at 29.45 GB (old, r20261111-081723-d587) and 29.60 (new, r20261111-081742-8357); the second is the next step of the same program, due to peak near 09:40Z. It has no stop of its own.
- d752 left out: 20 steps a minute and 41 MiB a step, so one recording of its 9,832 steps is 8.2 h and about 430 GB, and its trace is three.
Why: the recorder keeps every step's tensors from step 0 (read).

THE 14: none packed, none traced. 13 have a valid record at a pin and no row in the scorer's ledger, which the packer reads for the step: a change to the scorer's state, not the packer. r20261103-053912-3133 has ledger rows and no valid record: its record. List: /cursor/stores/self/trn102/refused-14.txt.

