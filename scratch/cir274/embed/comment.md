EMBED (piece 4a, the embedding shard): `EmbeddingShard_v1{VS=19360,H=6144,START=0}` is lowered as 32 `EmbeddingRead_v1{V=154880,H=192}` units a member over 32 padded tables of one width, on `cursor/circuits-cir274-embed-6f04` at `cd2c09ea9c22de67244bb1a0fc3fefd37aa12ece`. All four done-conditions are met. Rank 0's window at 43 prompt ids, 2 outputs and 4 layers, with DENSE's and DSAMOE's families beside it, passes `check_proof_block`, `check_wires` and `check_inputs`. Its digest is `630a5f4793a9531b…`.

What holds it:
- (a) The local pytest of `test_node_window_embed.py` at `cd2c09ea` passes 15 tests at LP = 32 and 1 layer: 32 units a member (1,056 in all), one wire pattern a step from `o0` to the all-reduce's `parts` row 0, `EmbeddingShard` gone from `unlowered` and `free`, and a second width refused.
- (b) Run `r20261113-084147-26ee` at source `71f0af81`: 5,138,792 units in 106 templates and 13,239 wire patterns. 1,408 read units, 32 tables of 154,880 rows × 192 words, and `EmbeddingShard` gone from `unlowered` and `free`.
- (c) The same pytest at `cd2c09ea`: each unit is fed only from the circuit's declarations. A token inside the shard gives the shard's words and one outside gives zeros. A swapped table or a wrong row fails. The Boolean read on two units agrees with the reference.
- (d) The ratchet gate at `cd2c09ea` fails only on `node_window_values.py:634`. On an unpushed local merge with `63cc12a8` it gives `failures: []`. No lint list changes. The adapter's lint and mypy tests name neither of my files; the one lint failure is P3 in `node_window_values.py`, which is on the joint head.
- The split (VLLM-417, 08:04Z): `split(params)` keeps both kinds of table apart from the plain weights. At 43 ids and 4 layers the 61 plain weights and their nonces are the same with and without the shard's tables; handed unsplit, 60 of the 61 would move.

Not done:
- vllm's hunk (`SHARD_VOCAB`, `shard_vocab`, `shard_param` and the `unit_templates` case at one width) lives in `node_window_embed.py`. The tables' registration and its one-a-process commit are not written. Both are in `scratch/cir274/embed/needs-vllm.md` on the scratch branch.
- `node_window_values.run` can't check these units: its `Tables` refuses a `/vocab` table. The test file has its own check instead.
- The prefill's 1,376 read units are wired to nothing until `AllReduceMultimem` (piece 2) is lowered.

The statement I trust least: "a second width is refused". The file refuses it only within one window. Two windows or processes, each at one width but different ones, would pass here, and only the tables' registration (vllm's) can refuse that for good.
