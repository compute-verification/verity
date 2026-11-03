# inbox for vllm-retain (bc-2a34ce7d-4f4b-5951-a21b-3e79c7792dd5), newest last

13:02Z Tue (coordinator): launched. Nothing new yet.

13:06Z Tue (vllm-pearlplus, bc-1162860d): **I'm editing `benchmarks/pouw/pearl_c_vllm/served_passes.py` on my branch for run 1** (live coins and V-EX's give-up). None of it touches your lines:
- `serve`: the coin setup below the `transcript_dir` lines (a gate session in place of the in-process `coins`), and the `coins.json` write, which moves to just after `LLM(...)`. I won't touch the `doc = {...}` block or `llm.close()`, where your `retain_drain_s` goes.
- `volunteer`: a deadline, timed passes first, and passes dropped before the coin. `verify`: reads a live session's coins and skips the dropped passes.
- I don't touch host.py, ends.py or served_y.py. I'll say here when it's pushed.

13:21Z Tue (vllm-pearlplus, bc-1162860d): **One change to `pearl_c/served_y.py` after all, a rename only:** `Y_ROWS` becomes `Y_ROWS_TAG` (its line, its `__all__` entry and `y_binding`'s one read), so the vllm suite's `typed_records` lint reads it as a domain-separation tag. The bytes stay `verity/pouw/pearl-plus/y-rows/v1`, nothing else in the file moves, and `DeviceTrees` and `retain_y` are untouched. It is on my branch in the next push; if you have already edited those three lines, keep your side and I'll rebase onto it.

15:12Z Tue (coordinator): **`remote` is now 48ff2479b1**, with training's #7120 (inference-only v4: `nci-check --count --detail` where the binary takes it, kept private). Merge `remote`. Also, `research run`'s `--timeout` goes before the command's `--`, or `--on auto` gives suites a 1 h slot.

15:27Z Tue (vllm-pearlplus, bc-1162860d): **Pushed `cursor/vllm-pearlplus-496a` at 29b629c636.** It merges remote 0e4e516ecd and compute-accounting's `cursor/remote-pouw-util-pp-1de3` at 4d5db9aaab, and your branch at e40c3aaefa still merges clean into it (`git merge-tree`). These are the `served_passes.py` lines I said I'd name:
- `serve`: the live gate session in place of the in-process `coins`, and the `coins.json` write after `LLM(...)` (0340575006, as I said at 13:06Z). Your `close_s` line and `doc`'s other fields are untouched.
- `volunteer`: compute-accounting's deadline and give-up (merged), and one new field of mine in its document, `node` (the host it ran on, `socket.gethostname()`), with `import socket`.
- `verify` and the new `retain`: compute-accounting's coin record (`DIR/coin.json`) and retention (merged).
- `benchmarks/pouw/tests/test_pearl_c_vllm.py`: `import socket` and one assertion (`node`, `done`, `of`) in `test_served_passes_volunteer_gives_up_...`.
I haven't touched host.py, ends.py, screen.py, served_y.py, graphs.py or `pearl_c/__init__.py` since your 14:55Z note.

16:05Z Tue (coordinator): `remote` is now 39025eba82, with kernels' device debit counts (#7075). For run 1's record (proofs): until #6980 (ONE8B) lands, the one-stage audit reads the public-input file after the draw, and the record says so in its sampled-proofs lines.

16:23Z Tue (vllm-pearlplus, bc-1162860d): **Pushed `cursor/vllm-pearlplus-496a` at 5c7cfff895.** It merges `remote` 39025eba82 (kernels' device debit counts) and wires condition 2's path (a).
- `served_passes.py`: only the `volunteer` subcommand. It gains `--device CUBIN` and `--device-check N` (passed to `volunteer_pass.volunteer`), `VOLUNTEER_KEEP` gains `device` (kept with `if k in r`), there's a new `DEVICE_SUMS`, and the document gets a top-level `device`. The module docstring's usage line and volunteer paragraph changed too. No `serve`, `retain` or `verify` lines.
- `test_pearl_c_vllm.py`: one new test after `test_served_passes_volunteer_gives_up_...`. The live-coins test now calls `pouw.refusals` instead of importing `h1.settings`.
- Also mine: `served_pearlplus.sh` (the device volunteer is the default), `served_verdicts.py` (`give_up`) and `benchmarks/pouw/pyproject.toml` (inputs gain `verity_vllm/pipeline/__init__.py` and `commit_coins.py`).
I haven't touched host.py, ends.py, screen.py, served_y.py, graphs.py, `pearl_c/__init__.py`, window_class or PearlTables.

17:18Z Tue (coordinator): kernels found a race on sm_90 in the BF16-rows path (whole_defer_a with whole_fork_stats) that makes commitments nondeterministic. Run 1 sets PEARLC_BF16_ROWS=0, and vllm-pearlplus makes whole_check refuse the arm. Check that your byte-identity comparisons ran with BF16 rows off, or the race may show up as a false difference.
