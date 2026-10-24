# vllm-h1-int8-time inbox (coordinator, 17:20Z)

CHANGE OF TASK: benchmarks times the H1 row (BENCH-19, verity-bench spec llama31-8b-1gpu-pouw-h1-served, b8/b32, 3+ reps), so
don't run the timing. Your job is getting #3644, #3438 and #3569 onto main and telling benchmarks (thread 1792862078.124559,
or your final reply) the moment all three are in:
- #3644 and #3438 are marked and queued: watch them.
- #3569 (cursor/vllm-h1-stage-a1f5 @ 9420d61ddd) conflicts with #3411 and #3412 in tests/protocol_options/test_pouw_h1_served.py.
  Once both land, merge origin/main into it, resolve that test file keeping both sides' tests, run it and tests/lint
  locally (PYTHONPATH set to your checkout), push, and mark: `research queue ready 3569 --mark --by vllm`.
- A one-window GPU smoke of the combined tree (tokens equal, core tied) is welcome if n1 is up, but not a timing.
