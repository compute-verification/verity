# Patch status on mainline `b745af3b` (checked 23:00Z, `git apply --check` forward and `-R`)

All three diffs were applied by A3 in ckpt 7 (with A3's edits), so on `b745af3b` none applies forward and none reverse-applies cleanly:

| diff | forward | reverse | content on mainline | action |
|---|---|---|---|---|
| `a3-locator-attention.diff` | no | no | `verifier_v3/locator.py:44 ATTENTION_KINDS`, `:318-323 locate_attention` | **already applied — do not apply** |
| `a3-expected-attention.diff` | no | no | `verifier_v3/expected.py: Deriver.attention_row / .config`, `RecordPartition.locate` head → `head_expectation` (`:377-384`), `kind_program` head (`:420-425`), `coverage_inputs(config=)` (`:447`) | **already applied — do not apply** |
| `prover-rec_p1_obligations-dispatch.diff` | no | no | `scripts/rec_p1_obligations.py:151 BUILDERS/dispatch import`, `:247-260` attention dispatch through `BUILDERS[kind].build(located_unit, record)` | **already applied — do not apply** |

The diffs are kept here as the record of what was proposed; nothing in this directory is to be applied again. `forced_head_run.py` detects the mainline content (`Deriver.attention_row`, `locator.ATTENTION_KINDS`) and runs on the checkout without an overlay.
