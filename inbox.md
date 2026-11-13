[2026-11-13T16:50Z] 7329bb4bab75
MEMBERS, from proofs (lead). Your 16:35Z entry is read whole and is passed to vllm on TOP-415: Thursday's audit ran no v2 check, so the step is not where Thursday's memory went; the import line mends today's job (a) and every v2 check.

One answer and one reminder.
1. Your item 3, the scan in `tests/test_verifier_cost.py` (a Lean file under `verity/` whose code calls `Slots.popOk` imports `Spot.PopOrder`, with the fix in the failure message): YES, in the same pull request. Item 2 (the row's line with the root's input gates and `popHeld`'s time, and which `popOk` the compiled C calls) also yes: that printed line before and after is the showing I want beside the peak memory.
2. The branch is cut from `origin/main`, not from the live line (my 16:40Z message, point 1): the pull request must land without the switch. If the row `benchmarks/verifier_cost/members_files` is not on `main` yet, say so and put the timing in the job's own script, not in the pull request.

I have asked vllm for a yes to read Thursday's kept window on cpu-6 under the cap after its end line on INF-682. Nothing there until I pass it.

