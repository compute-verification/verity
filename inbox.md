[2026-11-12T17:21Z] a288f7d00709
training to BACKWARD, Thu 12 Nov 9:21 AM PST (17:21Z). **A new job, starting now while your capture is queued: TRN-141 job 1, the fused backward's Definition. Daniel asked at 9:02 AM PST that what follows a result start before its date; your verdicts came a day early, so the circuit starts today, not after Friday.**

**What.** The Definitions of `aten._scaled_dot_product_flash_attention_backward` as a training step runs it on cc 12.0 (causal, no dropout, the default scale, bf16), one Definition per head, in the forward's pattern (`verity/definitions/registry/fa2_train.py`: `Fa2TrainFirst/Block/Row/Head_v1`), composed from the tree's twins as your CPU twin already composes them.

**Decided by me, reversible; say if the source argues otherwise.**
- The circuit is the kernel as it runs: its own inputs (dO, q, k, v, o, lse), the scores computed again (five products), only the blocks the kernel visits, each rounding where the kernel rounds. It does not wire the forward's S.
- The order is deterministic mode's. One line in the report on what default mode's atomics order means for a job (your 55 words): a statement, not a fix.

**Done when.**
1. The Definitions give every captured word of art:6a0d7e79... at its four sizes. A test beside `verity/definitions/tests/test_fa2_train.py` holds it on a small fixture cut from the capture, kept beside its generator, its `art:` id only in `fixtures/artifacts.json`.
2. The module's sidecar binds them (`<module>.circuit-check.py`, pins) and `uv run circuit-check` passes; its report is in the PR body.
3. A draft PR from main on a `cursor/` branch, not marked. The suites it touches pass on a node (`research run --on auto`), last lines in the body.
4. One table for job 3: at T 1,024, D 64, a head's gates, and the multiply-adds each of the five products holds, beside the full rectangle's.

**Report by inbox today 6 PM PST:** the list of Definitions (name, parameters, what each computes), what is hard, and YOUR date for 1 to 3. I set the goal's date from it.

**Order of your work now:** this job; the queued capture when it runs; Friday noon's TRN-140 report; the efficient kernel's reading last.

**Not yours:** the frontend rule and `export_step` (job 2, SEAT); the grader's test and the counter (job 3, grade). I pass your branch to SEAT when you name it.

