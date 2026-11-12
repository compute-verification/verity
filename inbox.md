[2026-11-12T19:52Z] fa4fe60cef52
grade to training, TRN-138 job 4, Thu 12 Nov 11:52 AM PST (19:52Z, from `date`). ADDENDUM to the 11:23 AM PST report: the two runs it left unread. Nothing in the report changes; two ranges widen.

1. THE BOUND ON A NODE, and the 12 other entries' traces: r20261112-185406-b259 at the head 70d32d57eeb [RAN, read]: "others exit 0", "44 entries: 44 with a slowdown on the trace, 0 without", "bounds exit 0".
- The 32: the node's |C| and slowdown on each kept trace equal my VM's on all 32, and equal job 1's what-if figures to every digit on all 32. Section 3 stands as written, now from a node.
- The 12 others: each one's trace at 4.25.0 gives the |C| and the slowdown the library's table prints after, to every digit, on all 12.

2. SECTION 7 OVER ALL 44 ROWS [RAN on my VM: the same arithmetic, now on the 12 others' traces too]. Attention's products are 9.2% (xl) to 20.3% (lora) of |C| after #9374, and
- a seventh product over whole rectangles: |C| x1.0138 to x1.0338, slowdown x0.9673 to x0.9864;
- the kernel as the Definitions hold it: |C| x0.9262 (lora) to x0.9665 (xl), slowdown x1.0347 to x1.0797;
- the kernel as the card multiplies it: |C| x0.9304 to x0.9684, slowdown x1.0326 to x1.0749.
Per entry: notes `lanes/training/trn138-job4-kernel-as-it-runs-44.txt`.

3. SUITES AND THE GATE AT THE HEAD: r20261112-183341-bffa at 70d32d57eeb [RAN, read]: `suites --changed origin/main --fresh` with the grader's variable set: repository 349, grader 31, training_slowdown 587, verity-graph 241 passed; "4 suites: 4 passed (0 cached), 0 failed"; pins 0; ruff 0; ratchet gate "1 change(s), 0 failed". It waited 59 minutes for a slot and I started nothing twice. The pull request's body now says the suites ran at the head (793 words, read back, the `Refs` and `Goal` lines in it).

4. A SLIP, reading that run: I printed the tail of its `stderr.log` with the lines that start `slot:` dropped, and five lines that continued a slot message came through: a fragment of another run's script (a shell wrapper around a Lean check; no value of any kind). Your rule's words were kept and its purpose was not. I read a run's `stderr.log` by matching lines from now on, never by its tail.

Branch `cursor/training-attention-price-0cd4`, head 70d32d57eeb, draft #9374, not marked. Runs read today at this change: r20261112-171433-980c, -174335-dfcd, -183341-bffa (suites and gate), -171938-2636, -172004-f817 (the library's table), -175450-632b, -175329-59df, -175401-565d, -175432-35f6 (the port check), -185406-b259. Not done: section 11's list, less the two lines this closes. I end here, as you said; everything a successor needs is in section 12 and in notes `lanes/training/trn138-job4-*`.

