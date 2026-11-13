Lane circuits

Stacked on #9693 (`cursor/circuits-cir264-sampled-accept-6f04`), and lands after it. This PR's base branch is #9693's, so the diff here is this PR's alone.

## What it adds

- **The unit that writes the masked row.** `TopPMaskChunk_v1` (`topp_stages.py`) writes one row of the top-p masked logits. `mask_plan(V)` lists the top-p pipeline's units with the mask chunks in place of the select chunks. The mask's work is `mask_lanes`, one engine-only function that both the Definition and its reference run. The catalog id is in `catalog_ids.json`.
- **The recut accept units.** `AcceptTileHead_v1`, `AcceptTileSum_v1`, `AcceptRowEnd_v1` and `AcceptResample_v1` now read a row's lanes as rows of `ROW = TS.SELECT_N` lanes, one parameter a row. Each read is then one mask chunk's output row, whole. `pass_plan(K, V)` is the plan of a pass of K + 1 rows: each row's top-p pipeline and mask chunks (`("topp", row, …)`), then the accept's units (`("accept", …)`), with every read of a masked row a wire from the chunk that writes it.
- **The resample on 32-bit stream keys.** `AcceptResample_v1` selected the resampled row's 64-bit position with `SelectI64_v1`, then keyed it, and `Q_call v1` refused that gate as wider than its X = 32. Each row's position is now keyed (`GumbelStreamKey(seed, pos[p])`), and the 32-bit key is selected with `SelectI32`. The key is a function of the position alone, so the Call's function is unchanged, and so are its ports and its reference. The cost is K + 1 key primitives in each resample chunk instead of 1.
- **The numpy references.** circuit-check now compares these against the Definitions on random vectors:
  - `TopPMaskChunk_v1` against the twin's `mask_chunk` (`topp_stage_twins.py`). It declines a carry that is no chunk lane's, and it samples chunks of real rows at every split count.
  - `SampledAccept_v1` against `sampled_accept` itself (`accept_kernel`).
- `_CONF` (`exact-model-tested`) names the replay at this branch's head (`evidence.json`'s `sampled_accept_capture.replay_run_record`).

`SampledAccept_v1{K=5,V=154880,TILE=8192}`'s ports are those at #9693's head.

## circuit-check

Run `r20261113-083534-e2a9`, at `6ee3b8d3b9d51e2dde2c92af7acfb4293ba02090`. `--update-pins` on `topp_stages` and `sampled_accept` wrote nothing.

`sampled_accept`:
```
circuit-check: vectors a realization compared with the reference on: 9 random, 0 edge-only, 25 none
circuit-check: 35 target(s), 0 failure(s), 33 unchecked (warnings, which never fail) on 33 Definition(s), report /workspace/research/runs/r20261113-083534-e2a9/out/cc-sampled_accept.json
```
`topp_stages`:
```
circuit-check: vectors a realization compared with the reference on: 5 random, 0 edge-only, 8 none
circuit-check: 13 target(s), 0 failure(s), 9 unchecked (warnings, which never fail) on 9 Definition(s), report /workspace/research/runs/r20261113-083534-e2a9/out/cc-topp_stages.json
```
`boolean_topp_stages`:
```
circuit-check: vectors a realization compared with the reference on: 13 random, 0 edge-only, 0 none
circuit-check: 13 target(s), 0 failure(s), 15 unchecked (warnings, which never fail) on 9 Definition(s), report /workspace/research/runs/r20261113-083534-e2a9/out/cc-boolean_topp_stages.json
```
At #9693's head (`66f957e940d7428d0904626920d265a4f38e93c7`, the same run), `sampled_accept` reads:
```
circuit-check: 31 target(s), 0 failure(s), 30 unchecked (warnings, which never fail) on 29 Definition(s), 10 Definition(s) Q_call v1 refuses, report /workspace/research/runs/r20261113-083534-e2a9/out/cc-accept-head.json
```

## The replay

Run `r20261113-081538-2873`, at `0d2d60aba33c2cc4708c39bc33221a9b95195f69` (the merge of #9693's head). It ran #9693's replay script on the same capture tar, sha256 `4ca7772a558b…`, off the card:
```
cases 410 (requests of 1 + 5 rows; 2460 rows), inside the domain 409, of them 0 differ; words compared 80226, differ 0
definition: 11 cases, 264 words compared, 0 differ
```
The commit after it, `6ee3b8d3b9`, changes only `evidence.json`'s record id.

## What ran where

- circuit-check: `r20261113-083534-e2a9` at `6ee3b8d3b9`.
- The replay: `r20261113-081538-2873` at `0d2d60aba3`.
- Suites at `6ee3b8d3b9`, run `r20261113-084943-f574`: `repository` 352 passed; `verity/definitions` 969 passed, 2 skipped (`2 suites: 2 passed (0 cached), 0 failed`).
- The ratchet gate (`adapters/vllm/tools/ratchet_gate.py 66f957e940d7 HEAD --json`) at `6ee3b8d3b9` gives `failures: []`.
- The whole `--changed` set, `verity-vllm` among it, last ran at `f65b556ae575` (`r20261113-015516-01e6`, before the merge of #9693's last commit), and all three suites passed. It runs again at the head carried across tonight's layout moves.

## What I trust least

- The mask chunk's reference shares `mask_lanes` with the Definition. The comparison therefore checks the Definition's compilation into gates, not the masking logic both run. The only thing independent of it is where the sampled inputs come from: the twin's earlier stages over real rows.
- The nine `AcceptResample_v1` bindings, like every accept unit but the root, are compared on no vector: their lowerings are past circuit-check's budget, and no unit has a per-unit reference. They are held only through the root's reference, `SampledAccept_v1{K=2,V=8208,TILE=512}` against `sampled_accept`, and through the replay at K=5. circuit-check's uniform vectors on middle and last mask chunks are mostly declined, so those targets compare on the sampler's real-row vectors.
