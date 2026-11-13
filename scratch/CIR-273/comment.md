**Result.** The top-p mask is now its own unit, `TopPMaskChunk_v1`, and the sampled accept reads the masked row through it. Every accept unit takes the row as whole 32-lane rows, one parameter per mask chunk's output, and no unit takes more than 132. `pass_plan(K, V)` wires a pass of K+1 rows from the top-p pipeline into the accept, and a test holds it to `TopPMaskV4` on each row and to `sampled_accept`. The select's pins did not move. Branch `cursor/circuits-cir273-masked-row-6f04`, head below; the code is at `1d7949bb00`, with only empty checkpoint commits and this file after it.

**What holds it.**

- `TopPMaskChunk_v1{V,N,FIRST,LAST}` is the select's own lines up to `sel(keep, x, -inf)`, through one helper (`_masked`) that `_select_body` also calls.
  - Its parameters are the N bf16 lanes, `temp`, `splits` and the six `sw{S}`, plus the carry unless FIRST.
  - It returns the N masked f32 lanes, plus the carry unless LAST.
  - The carry is the 27 per-split-count position words, the select's carry less its 3 head words.
  - At V=154,880 it is 4,840 units a row (N=32) and 29,040 a pass at K=5.
- `mask_plan(V)` beside `plan(V)` is the 3,319 pipeline units, then the 4,840 chunks. Each chunk is wired to `scalars_unit(V,S)` and, by `cw`, to the chunk before.
- The accept units, all still `_v1`, at K=5 and V=154,880: 5,410 units over 12 Definitions. A pass is 54,364 units, 48,954 of them top-p.
  - `AcceptTileHead_v1{N,AT,FIRST,LAST}` reads 128 rows, or 104 for the last head. Its counts are {4096,0,T,F} ×5, {4096,4096,F,F} ×90, {4096,0,F,F} ×90 and {3328,4096,F,T} ×5. Each head carries the tile's maximum to the next.
  - `AcceptTileSum_v1{2048,AT,LAST}` reads 64 rows: AT 0, 2048 and 4096 ×95 each, and 6144 LAST ×90.
  - `AcceptRowEnd_v1{19,1280,6144}` ×5 reads 40 rows, 81 parameters in all.
  - `AcceptResample_v1` ×4,840 reads one row of each of the 6 rows.
- The row limit is `MAX_ROWS` = 256 at `verity/commitments/windows/interface.py:311`, checked in `check_proof_block` at line 334. The plan test asserts every accept Definition passes it; their blocks are k_log 23 to 27.
- The masked rows a pass commits: 929,280 f32 lane words (29,736,960 bits), plus 783,918 carry words between chunks (25,085,376 bits).
- Runs:
  - `r20261112-224801-3690`: circuit-check on the `topp_stages`, `sampled_accept` and `boolean_topp_stages` sidecars, 13, 35 and 13 targets, 0 failures. `--update-pins` wrote nothing, and the select's `GumbelTopPSelect_v2` pins pass unchanged. `SampledAccept_v1{K=5,V=154880,TILE=8192}` checks by its 12 callees.
  - `r20261112-225525-7370` at `1d7949bb00`: `repository` 352 passed; `verity-vllm` 15,034 passed, 485 skipped; `verity/definitions` 966 passed, 2 skipped; 0 failed.
  - `r20261113-003646-bfb1` at `0a30cfb891`: the 16 new mask tests (S = 1 to 32, and the plan at V = 16, 64, 1000 and 154,880) and the pass composition test pass.
- `ratchet_gate.py origin/main HEAD --json` reports `failures: 0` on 1 change, and `git merge-tree --write-tree origin/main HEAD` is clean. ACCEPT's branch had not moved past `903c545eb3` when I last fetched.

**Not done.**

- No Boolean σ-lift (v2) of `TopPMaskChunk_v1`, so it has no pinned digests.
- The composition test runs at K=2, V=8208. It takes the top-p pipeline units' outputs from the twin and evaluates only the mask chunks and the accept units; evaluating the pipeline is about 200 s a row.
- vllm's stage layout and reader are not changed.
- Whether the mask carry counts in leakage is proofs' call.
- This file sits under `scratch/` and must be dropped before the branch lands.

**The statement I trust least:** that the pass composition test holds the real pipeline. It checks the mask chunks and the accept on the twin's pipeline outputs, so a pipeline unit that disagreed with its twin would pass it unseen. Locally, one row's evaluated pipeline matched the twin with 0 mismatches, but nothing holds that in a test.
