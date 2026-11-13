MASKROW, Fri 13 Nov 01:22 PST (09:22Z): the stacked PR is open as #9720, head `6ee3b8d3b9d51e2dde2c92af7acfb4293ba02090`. It is based on #9693's branch, not a draft and unmarked. My branch holds #9693's head `66f957e940` through the merge commit `0d2d60aba3`.

**The replay at my head.** Run `r20261113-081538-2873`, at `0d2d60aba3`, off the card. It used #9693's replay script on the same capture tar (sha256 `4ca7772a558b…`).
- Capture: 409 requests inside the domain, 80,226 words, 0 differ.
- The registered `SampledAccept_v1{K=5,V=154880,TILE=8192}`: 11 requests, 264 words compared, 0 differ.
- `evidence.json`'s `sampled_accept_capture.replay_run_record` is now this run's record, `art:69f76ea2ce962f904e997d8076d77be09cdbdf2288d4be63b14fbaf068a09555`, and `_CONF` names it (`6ee3b8d3b9`).

**circuit-check.** Run `r20261113-083534-e2a9`, at `6ee3b8d3b9`. `--update-pins` wrote nothing.
- `sampled_accept`: 35 targets, 0 failures, 33 unchecked; `9 random, 0 edge-only, 25 none`.
- `topp_stages`: 13 targets, 0 failures, 9 unchecked; `5 random, 0 edge-only, 8 none`.
- `boolean_topp_stages`: 13 targets, 0 failures, 15 unchecked; `13 random, 0 edge-only, 0 none`.
- `Q_call v1` refuses 0 of the accept's Definitions at my head, and 10 at #9693's head (`66f957e940`, the same run).

**Suites at `6ee3b8d3b9`.** Run `r20261113-084943-f574`: `repository` 352 passed; `verity/definitions` 969 passed, 2 skipped. The ratchet gate against `66f957e940d7` gives `failures: []`.
- A first run, `r20261113-083616-5f0a`, was cancelled when its memory passed 48 GB (46.3 GiB, its own processes, with 4 workers on `verity/definitions`). The rerun used 3 test processes and peaked at 41.1 GB.
- `verity-vllm` and the rest of the `--changed` set were not run tonight. They run at the head carried across the layout moves.
