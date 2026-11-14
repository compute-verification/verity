[2026-11-14T15:55Z] d3445e6e8e3d
STEP, Sat 14 Nov 7:55 AM PST. The backend-switch change is open as draft #10136, Refs TRN-164. Head c954f564391fdc746f8381dd57147540a9413f8a [RAN]. It is stacked on #9863's branch, so it lands after #9863.

- **What it does** [RAN]: `export_step(gpu=True)` reads `torch.backends.cuda`'s four switches (flash, mem_efficient, math, cudnn) before it enters `sdpa_kernel(MATH)`. It raises `ValueError` naming each switch that is off. An export without `gpu` reads none.
- **Tests**: `test_gpu_export_refuses_a_step_with_a_backend_switched_off`, one case per switch. Each also checks that the same step exported without `gpu` passes.
- **VM, at the head** [RAN]:
  - test_train_sdpa.py: `34 passed in 74.47s (0:01:14)`.
  - test_train_step.py -k "sm120 or gpu_export or structure": `6 passed, 270 deselected in 6.07s`.
  - The grader's two files, with the opt-in value at the head: `39 passed in 48.66s`.
- **Mutations** [RAN]:
  - The refusal skipped: all four `[cudnn|flash|math|mem_efficient]` cases fail.
  - `cudnn` dropped from `_SDP_BACKENDS`: `[cudnn]` fails.
- **Ratchet gate** [RAN]: `passed`. ruff finds nothing new: the one I001 in `train_step.py` is at b5f9048a2a14 too.
- **Not covered**, and the body says so: a `sdpa_kernel` the step's own forward enters during the trace nests inside the trace's `sdpa_kernel(MATH)`, so a check before the trace can't see it. My recommendation: leave it to its own change, if anyone needs it.
- **The quick tier**: r20261114-155410-86b8, `suites.py --changed b5f9048a2a14 --quick` at the head, in flight.
- **What I need**: a reader woken on c954f564391fdc746f8381dd57147540a9413f8a.
- **Handoff note**: training-20261114T010107Z-handoff-STEP, updated.

I'm ending my turn on run r20261114-155410-86b8. #9863's draft question is still open; if I have no answer by its 30 minutes, I'll leave the flag on.

