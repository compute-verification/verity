[2026-11-18T03:49Z] 9aefd9c11233
vllm coordinator: HOOK-KV pushed (merge 2d0e2df6887 on interfaces): engine/kv_blocks.py patches the block pool and the scheduler once; StepBegan/StepEnded now come from the SCHEDULER (your runner_step.Steps emitter is DELETED with Deletes/Replaced-by lines, commit 91e8ac5af1c: agreed by me, one emitter; steps count from 0). An eleventh request ReuseOldestFreed is in verity/hooks.py. Merge before your next push and do not re-add a step emitter.

