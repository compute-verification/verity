**The capture of record passed at PR #9317's head: 1,413 planted rows and 127,274 served rows held to `Fp8DsMlaUpconvertRow_v1{DN=512,DR=64}`, 0 differing. `verity/definitions` and the circuit-check suite pass at the head with the sync.**

Head `0978f5af04bb7a35269ce2fdf874f5a675539c97`, unchanged. Nothing was pushed to the PR's branch.

**Capture** `r20261113-031927-2fac`, source the head, node A, card 3, cores 30-39. Exit 0; `capture.json` and `compare.json` both say `ok: true`.

| Rows | Compared | Differ |
|---|---|---|
| Planted, held to the Definition's evaluator | 1,413 | 0 |
| Served, all 7 layers, held to the Definition's numpy kernel inside the capture | 127,274 | 0 |
| Served, first layer, held to the Definition's evaluator | 18,182 | 0 |

- 154 calls of the op over 53 steps, no row unmapped.
- Library sha256 `683d3249f3a3e244105fcb9d966bc3979a3c20e1ad045b745bbee05650b37471`, `matches: true`: the cu129 wheel's.
- **Lease:** one, held 48 s (Thu 12 Nov 7:20:01 PM to 7:20:49 PM PST), declared with `--cores 30-39`; gpu-lease's line reads "CMD pinned to CPUs 30-39". Card 3 held 0 MiB before and after. The compare ended at 7:23:50 PM PST, off the card.
- **Card 3 and cores 30-39 are free from 7:24 PM PST.** I run nothing more in the row (to 8:25 PM PST), so its card 3 part can be released.

**Against `r20261112-192752-fca0`** (at `79cb0b3e28`): two things differ, the cores (30-39 declared on the lease; then 34-43, pinned by `taskset` with none declared) and the source SHA.

- The two runs' records differ in one value of 4,701: the path of the Definition's module in `capture.json`.
- The 22 saved first-layer calls and the 2 planted files are byte for byte the same.

**Suites** `r20261113-003940-657d`, source the head, vy-nebius-cpu-4, after `uv sync -q --all-packages --extra torch-cpu`. rc 0.

| What | Verdict | Passed | Failed | Skipped |
|---|---|---|---|---|
| `verity/definitions` | passed | 986 | 0 | 2 |
| circuit-check suite (`tools/verity/circuit_check`) | passed | 428 | 0 | 5 |

- circuit-check on the sidecar: rc 0, 2 targets, 0 failures, 0 unchecked. Both targets were reused from records, so none was checked anew at the head, which changes no file circuit-check reads. The last run that checked them anew is `r20261112-192652-7776`, at `79cb0b3e28`.
- Already at the head: `repository` 352 passed (`r20261112-221618-0259`); `verity-vllm` 15,016 passed, 0 failed, 485 skipped (`r20261112-233932-8658`).

**Still not run**

- A prefill past `index_topk` (2048): the top-k side of the route is read in the pin's source, not run.
- The SASS listing on node A itself: the node has no `nvdisasm`. That its kernel is the listed one rests on its library's sha256.
- Decode context parallel, prefill context parallel, tensor-parallel 8, and GLM-5.3's own weights.
- The rest of `check`: no other suite, `circuit-check --all` or the Lean check.

The PR's body has all of this. It has no Mark section, and the PR is unmarked.
