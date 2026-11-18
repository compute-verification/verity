lean/WIRED: `shared-unheld` on tied and untied (trial `cursor/lean-classes-trial-fc8d` `44989e65746`, made-up log)

**It is neither our stand-in nor circuits' record. It is the rule (lean's `Record.sharedPlan`, under `SharedHeld`).** The rule accepts only a registered read as holding a shared run. The record holds these rows another way: as a registered table's root and domain (`registered.roots`), which the rule refuses.

**Where the declarations come from:**
- Every declaration the rule reads is in circuits' circuit record: `IF.descriptor_record` of `class_circuit`. Tied's record states 5,380 `registered.reads` and untied's 3,615, beside its `roots`, `openings`, `tables` and `params`.
- My stand-in writes only the registration log (`log0`/`log1`) and the limits, which give θ. θ holds every parameter: 294 of 294 on tied, 102 of 102 on untied.

**The refused rows** (probe `wired-share`, `sharedDecide` replayed row by row):

| class | shared rows | held | refused |
|---|---|---|---|
| tied | 1,933 | 1,877 | 56 |
| untied | 3,725 | 3,669 | 56 |

- **What all 56 are, in each class:** row 0 of `EmbeddingRead_v2{V=128256,H=2048}`'s argument 1 (`root`) or argument 2 (`domain`), a 32-word row, uncut. Each reads a shared run of an `Input1_v1` input node (512 one-bit leaves; base 0, inner 512, count 2,048), and the run is handed to every one of the root's 28 `EmbeddingRead_v2` nodes (4 members each).
- **The first refused, on tied:** node 4331, argument 1 (`root`), row 0, input node 262, template units from 0. Next is node 4331's argument 2 (`domain`, input node 261), then node 5345 (units from 4). On untied: node 107, arguments 1 and 2, input nodes 6 and 5.
- **What fills each row:** the record's table read `Fill.table "table/weights/model.embed_tokens.weight@2048+0"` (`root` true for argument 1, false for `domain`). It covers reader `EmbeddingRead_v2`, ports `root` and `domain`, units 0 to 112, which is all 28 × 4 members.
- **Why the rule refuses it:** `sharedPlan` reads `let .read p := d.fill | none`, so any fill that isn't a registered read makes the row "not held".

**What would pass.** No registered read can: the run is a table's root and domain id, bits the log commits, and no parameter's leaves hold them. A record can't state "a registered read of the parameter whose leaves the run is" without making up a parameter. So the fix is the rule's: hold a shared run's row when every member's row is filled by one registered table read of one table, at one of `root` or `domain` (`Fill.table name root`), as here. Then the rule's other check, one source per input node, reads (input node, table, root) where it reads (input node, parameter, offset).

The Python twin gives the same verdict on both classes: `shared-unheld`.

Runs, each `r20261118-…-<id>`: `2e66` (tied) and `8dac` (untied), the probe; `e2c6` and `93ed`, the twin.
