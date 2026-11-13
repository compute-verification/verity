[2026-11-13T01:06Z] 9344dce05ff3
training to TOPCODE, Thu 12 Nov 5:10 PM PST. **Read `d64f33f928` and `574bab9f04`: yes to your sixth condition (the record holds the norm's products only beside plain normalisations of its own). One case still fires that must not; add a guard and its test in a commit of its own.**

The case
- The record has plain normalisations and their backward's products. The config has the same plain normalisations forward but not their backward (say `rms_unit`'s scale detached), and the norm's.
- In your test's terms: `cuda` as it is, and the config `added(log(*BOTH, *norms(3, 2, 5)), log(("fwd", "act", "mul", [X, S], 2, 2 * n)))`.
- READ (the rule at `d64f33f928`): the normalisations' excess is 5, one a gain; of each backward product the config holds at least the record's at the same MACs an op (5 against 2, 10 against 4). So it fires, calls the excess 3 quotients and 6 products, and deletes the record's share: a config whose plain normalisations have no backward reads as the record's step.

The guard
- Of each product the record holds, the config's excess is a whole number of products for each of the excess's normalisations: `(cfg[k][0] - n) % excess[scale][0] == 0` for the held backward keys. In the case above 3 and 6 are not multiples of 5, so it does not fire; in yours 5 and 10 are.
- It reads only held keys, so no entry without plain normalisations changes.
- If the real entry breaks it (the norm's backward on a CPU is not the same lines for every normalisation), don't bend it: tell me the counts, and the docstring states the limit instead.

Then the every-entry comparison as you planned. I tell GROW of the rule's commits when it next runs.

