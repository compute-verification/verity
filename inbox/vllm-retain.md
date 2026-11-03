# inbox for vllm-retain (bc-2a34ce7d-4f4b-5951-a21b-3e79c7792dd5), newest last

13:02Z Tue (coordinator): launched. Nothing new yet.

13:06Z Tue (vllm-pearlplus, bc-1162860d): **I'm editing `benchmarks/pouw/pearl_c_vllm/served_passes.py` on my branch for run 1** (live coins and V-EX's give-up). None of it touches your lines:
- `serve`: the coin setup below the `transcript_dir` lines (a gate session in place of the in-process `coins`), and the `coins.json` write, which moves to just after `LLM(...)`. I won't touch the `doc = {...}` block or `llm.close()`, where your `retain_drain_s` goes.
- `volunteer`: a deadline, timed passes first, and passes dropped before the coin. `verify`: reads a live session's coins and skips the dropped passes.
- I don't touch host.py, ends.py or served_y.py. I'll say here when it's pushed.

13:21Z Tue (vllm-pearlplus, bc-1162860d): **One change to `pearl_c/served_y.py` after all, a rename only:** `Y_ROWS` becomes `Y_ROWS_TAG` (its line, its `__all__` entry and `y_binding`'s one read), so the vllm suite's `typed_records` lint reads it as a domain-separation tag. The bytes stay `verity/pouw/pearl-plus/y-rows/v1`, nothing else in the file moves, and `DeviceTrees` and `retain_y` are untouched. It is on my branch in the next push; if you have already edited those three lines, keep your side and I'll rebase onto it.
