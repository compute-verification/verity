[2026-11-12T19:47Z] 02aee5373398
SPLIT, Thu 12 Nov, 11:47 AM PST

**Done: no published training figure reads a wrong split of rotary tables from trained weights, so there is nothing to recompute. The full answer is note training-20261112T1946Z-report-split-tables-theta, and the handoff is training-20261112T1947Z-handoff-SPLIT.**

The report's first lines:
- The four panels, served and staged, and the 1,497-row search table all come from te_core's counter. In all 957 records it wrote, that counter logged every rotary product as `fwd/theta`, because `cw.py` binds `net.named_buffers()` as θ. RAN r20261112-193102-3134; READ `te_core/cw.py`, `te_core/run.py`.
- The library came from training_slowdown 4.25.0, which contains the fix. READ.
- No table publishes a ρ from a family's β or from a record with no trace. READ the tables.

Since 3 PM:
- **standard-workloads, standard 20M (r20261103-053241-99bf).** I rebuilt it with the panel's own code (RAN locally, `follow_sw.py` from d76ae46e6f). rho_own is 467,966.900 as recorded, which equals the served 467,967. Restated by e454's plan it is 467,458.462, which equals the staged 467,458. Step 1's `fwd/theta` is 1,610,612,738, which is the rope MACs plus 2.
- **The traces never read the counter's classification.** `capture/script.py` builds with `updated=None`, and `capture/structure.py` treats an unwritten buffer as registered. So every traced share is free of the split. READ.
- **Correction to my 3 PM message.** The 201 records with step-0 `fwd/updated` are not all adapters. Some are whole-matrix copies such as [1408,512], and 21 records have no 2-D fresh tensor. The rope products are `theta` in all 201 regardless. RAN r20261112-193102-3134.

What is done: the report, the handoff, and the 3 PM message.

What is left: nothing the brief requires. If wanted, the one-row reproduction (`follow.py`, in r20261112-193855-d49f's inputs) could be swept over every traced row.

Runs in flight: none. r20261112-193102-3134 and r20261112-193855-d49f are done.

No PR, and no branch pushed. The put branch was only read through a worktree, which I have removed.

