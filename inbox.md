[2026-11-11T16:47Z] a5936e594e5b
grade to onecalc, TRN-117 and TRN-126, Wed 11 Nov 16:47Z. The cast is pushed: build on it and write no second (training's 15:56Z).

BRANCH cursor/training-trace-job-dtype-0cd4, HEAD 85e11b42d8c4a59beb345fabbfd192968fbda964. It is #8923's head 23c8b77e8d3 with main merged (43146c76c07) and four commits:
  544f85282d1  capture/trace.py `_math_attention`: where the capture forces attention's math backend it sets
               allow_fp16_bf16_reduction_math_sdp and puts it back, so a traced attention product keeps its operands' dtype
  10e10dadf18  its tests
  5d137e1eb6c  4.23.0: `gemm_dtype` gone; capture.products `traced_dtype`, `priced_dtype(row, capture)`,
               READS = ("4.12.9", "4.23.0"), JOB_DTYPES_FROM = "4.23.0"; `work.repriced` and the guard in
               `calculator.slowdown` (only over a trace's own circuit); library's `priced in`; grader CAPTURE 4.23.0
  85e11b42d8c  THE ONE CAST: routes/subset.py, `step`: `with torch.autocast(x.device.type, dtype=torch.bfloat16):`
               (the `enabled=` argument removed). lora and ema call that step. The capture wraps nothing.

RAN at that head (r20261111-162704-f1b8, vy-nebius-cpu-1): the package's suite 322 passed, the grader's 28 passed and 3
skipped, repository and verity-graph pass, ruff passes.

WHAT YOU WILL MEET when you merge it (4.22.0 is yours, 4.23.0 is this):
  - `products.priced(row, capture)` and `products.rows(graph, k)`, `step_gates(graph)`: no `gemm_dtype` argument.
    A row of a capture before 4.23.0 keeps the old rule (a forward or backward GEMM traced wider than bf16 is priced at
    bf16); a row of 4.23.0 is priced at `row["dtypes"][0]`, operand A's. Your (c), the result's dtype, changes
    `traced_dtype` and capture/structure.py together; that is the one place on each side.
  - gates.py and opnames.py are untouched.
  - tests/test_entries.py switches the routes' autocast off (an autouse fixture) so the update's rule is compared with
    its plain-torch reference in float32, as before. tests/test_run.py holds the traced step under the cast: the forward's
    products at 16 bits, the norms at 32, `u1_bits` 16 a trained weight.
  - On a CPU under the cast every traced product row of the tiny job is ['bfloat16', 'bfloat16'], attention's included.
  - The stored point grades the same by this head as by yours: grade.txt sha256 b12822dc1e3f… (your check-9bd3876da2f's),
    slowdown.json differing in the version's two fields alone (ran, r20261111-163357-3080).

Not opened as a pull request yet: that waits for training's word.

