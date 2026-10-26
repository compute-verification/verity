# Inbox: PHYS-INVENTORY (bc-ffd298c4)

## 04:41Z (real UTC), coordinator: top's correction to step 2 (thread 1792988748.528649)
Daniel (9:32 PM PDT) wants the regions of the actual circuit annotated, and causality in time and space assumed. That model already exists on main: the actual-circuit model (`Accounting.Placed`, `Physical.lean`'s `Hardware` and `Feasible`, console's note v2.10). So:
- **No new model in `Security/Definitions/Core/`.**
- **Step 1 continues (due 10:00Z).** For each physical premise, add which of these it refines: one of the model's constraints (causality, links, compute rate, memory), a protocol's bridge (ring, HBM, hash oracle), or trust (the auditor's devices).
- **Step 2's design waits** until Daniel and top settle the layering. Its target is `Hardware`/`Feasible`, with Accel and TS.Machine as what justifies Φ for a device class. Read round 1: note:top/20261026T0440Z-draft-actual-circuit-layers. Don't write step 2 until top posts the outcome.
