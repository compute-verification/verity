[2026-11-18T05:40Z] 96d9a6e10b30
From the circuits lead: in `benchmarks/compile_cost/run.py`, the `--model` help text (about line 243) still says 'lowered unwired'; the unwired lowering is deleted on the branch (b137a9d04fe) and the benchmark measures the wired one now. Correct that line in your next push. Re-run your command.

