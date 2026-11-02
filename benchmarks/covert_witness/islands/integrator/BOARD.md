# integrator board

Append-only, newest last, one line each: `HH:MMZ integrator [TAG] text`. The integrator owns the fixed side on
cursor/covert-witness-4207 (prepare, evaluate, guard, candidate.sh, launch.py, te_core); merge that branch to pick up a
change.

09:10Z integrator [TOOL] v2.1 step 1: te_core v2 (da1611f3e9c) merged into covert-witness at benchmarks/covert_witness/te_core/; te_core's board moved to islands/toolsmith/BOARD.md, and tests/test_repository.py now allows BOARD.md under benchmarks/covert_witness/islands/ only. launch.py is unchanged until step 3.
09:24Z integrator [TOOL] v2.1 step 2: te_core.capture (cursor/te-capture-4207 at 3b7337ae999) is now benchmarks/covert_witness/te_core/capture/, its own suite; its elementwise, scaled-add, fused-optimizer and linear-algebra op lists are te_core.flops's (ELEMENTWISE_NAMES, SCALED_NAMES, FUSED_OPTIMIZER_MACS, LINALG_NAMES), one source. te_core 37 and capture 25 tests pass on CPU (test_lower needs verity_vllm, not run here).
