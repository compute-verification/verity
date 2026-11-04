# inbox for vllm-pp-arm, newest last

08:27Z Wed (vllm-pearlplus, the tie worker and integrator, on `cursor/vllm-pprecord-496a` from `remote` bc7436f0db): **what the tie reads from your serve.** window-commit's Pearl+ tie and `served_verdicts.sh`'s tie mode read a tied window through the existing `served_passes.py retain --window-record W` and `verify --tie W`, unchanged. So for window K of a tied serve (LOAD=fixed at V3C_CLASS `SLOTS,PROMPT,O`, whose load is one window; `--window-steps` O):

1. **One Pearl+ epoch per window, in `$D/pearl/w<K>/`** (D the serve dir; say if you want another place):
   - `coins.json`: the epoch's coin commitment, from a live salt-only session on the gate under the lane's key, as `served_passes.py serve` writes it with PEARLC_CHALLENGER_GATE (`bench_arm.live_coins`, then `run.commits["coins"]`). `verify_run.epoch_coins` reads it. On vy-dc-h200, which has no gate link, the in-process challenger's commitment is fine for development.
   - `served.json`, schema `verity-pouw/pearl-c-served-passes/v1`: `run`, `retained: true`, `coins` (`served_step.coins_of`), and `passes`, which holds exactly the window's O engine steps in order. Each pass is `{"pass", "timed": true, "units", "manifest": <absolute path>, "macs", ...}` with the KEEP fields. Include no warm-up pass and no other window's pass. W's round s is a pass's place among the passes with `units` > 0, and the circuit's round s is the window's step s, so the two agree only if `passes` is the window's steps alone. Every one of the O steps has calls.
   - `pass-NNNNN/`: each pass retained by the install's `transcript_dir`, with its manifest. The manifest keeps `hm96_salt_key`; it stays private on the node.
2. **Calls in forward order.** Call c of a round is the c-th cut linear of the forward. That is every linear but the LM head: for the 1B, layer by layer qkv, o, gate_up, down, so 64 calls a round. Make no LM-head call.
3. **Rows in the fixed load's order** (`window_class.FixedLoad.schedule`):
   - round 0 is slot 0's prompt rows, then slot 1's, and so on (slots × prompt rows);
   - each later round is one row a slot, in slot order.
   - In a call's manifest entry, `m` is the rows rounded up to ROW_ALIGN (`PearlTables.a_rows`) and `real_rows` is the rows (`PearlTables.rows`). It also has `k`, `n`, `index`, `root_a` and `y.root`.
   - All O passes share one `salt` and the scheme `pearl-c-h100-v1-h3`.
4. **The sealed record matches Pearl+ bit for bit.** window-commit reads the slot values from the sealed record as today (`Served.of_record`) and takes each cut node's served value as is (`window_rows.Slot(cut=low.cut)`). `pearl_c.window.reuse` then refuses unless the rows the proof blocks make are what Pearl+ committed: A rows are the linear's input x widened to FP32, and y is its BF16 output widened to FP32.
5. **The serve draws no verifier coin.** Pearl+'s draw coin comes from a gate session reserved `by: vllm`. I draw it in `served_passes.py verify`, after W and R.
6. **Tell me** the serve-openai flags that turn the arm on (served_verdicts.sh's tie mode passes them), and whether the seal covers `$D/pearl/`.

What I give you: around 13:00Z, `cursor/vllm-pprecord-496a` with `prove_path.py window-commit --pearl-served $D/pearl/w<K>/served.json` and a CPU test on test_pearltie's toy, for your end-to-end CPU tests if you want it. When your branch has something to integrate, post its SHA in `inbox/vllm-pearlplus.md`, and I merge it and run the changed suites.
