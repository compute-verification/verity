[2026-11-11T11:13Z] 984e183a8d3f
grade to training, TRN-102, Wed 11 Nov 11:13Z. On your 11:08Z. Ran unless marked.

THE NEW CELLS' FILE: /cursor/stores/self/trn102/new-cells.json (your store; 44,932 bytes, sha256 730770e2e434...). It holds the six search rows and the two overlay rows as the build printed them (the panel's columns and the raw rows, per layer; the two layers' rows are equal), the published rows the two replace, the recorder's commit, and the run (r20261111-103704-d1b3).

WHY THE PUBLISHED TABLE HAS NO SUCH ROWS (read: te_core/algorithm_policy.py build and scorer/v1cycle.py REFUSED_ROWS at the replot head; it is in the file). A refused run has no traced rows, so the scorer hands its checkpoints to the builder as untraced. For an untraced run the builder makes a row only at a gain target one of its checkpoints reaches, and the reason is written over "not yet traced". 464a and e55b reach 1.1 and neither reaches 2 or 11: one row each. A traced run has a row at every target, "no" at its last checkpoint where none reaches. So once they trace, each gains a row at 2 and at 11, at its last checkpoint (@951, @1741). The same read explains 464a@759's last digit in C_gates and p: a refused run's |C| is summed from its record without the trace, a traced run's is its scored row's (the cause of the digit: inferred).

TRN-114: commented as training/grade with the lane's command. One run more for it, r20261111-110014-aff5 (cpu-5): the smallest case under torch 2.14.0+cpu, the version the repository's lock pins: 16 of 16 die (SIGSEGV), control 4 of 4 return, the tiny decoder the same. So the fault is in both versions. My jobs never used a node's torch: each made its own venv, 2.15.0+cpu in the first three (cpu-3, cpu-11, cpu-12). Two environments, not two nodes.

NO BUMP: understood. It goes in TRN-102's closing comment with the rest, after f0ed.

f0ed (r20261111-105829-b845) and 106f (r20261111-105852-439e): still in line on cpu-7 at 11:11Z, four checks ahead.

