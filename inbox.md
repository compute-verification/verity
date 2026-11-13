[2026-11-13T03:51Z] a0acfc0c2636
From benchmarks (KITMARK). #9530 (prove SPOT metrics, benchmarks/prove/run.py + prove_records._mark) is now ON MAIN: merge origin/main, and give its four measurements the required mark. Also BENCH-50 in the same pass: `benchmarks/prover-gpus-per-serving-gpu` is proving total over a modeled serving rate, so its mark must be the WEAKER of proving's and the serving rate's (order measured > partial > modeled), both listed in inputs; pin it in test_prove.py. Name BENCH-50 in the PR body.

