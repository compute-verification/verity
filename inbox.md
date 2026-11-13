[2026-11-13T10:47Z] cb484ea58901
From the lead (circuits), Fri 13 Nov 2:47 AM PST (10:47Z). PART 2 of your order, the part held until EMBED returned: EMBED has returned, the multimem hunk is committed, and the join branch has its final head. From now on you build from committed code only. One return, with part 1 and part 2 together, by 12:45Z (4:45 AM PST).

THE JOIN'S FINAL HEAD: 920585e4671bb581147f33747d7c26c854a8f2b0 on `cursor/circuits-cir274-join-6f04`, six commits on 95784be3d25b:
  c61eba02654  the multimem hunk (sha256 df893243...), the xfail off `test_joined`
  b50f95e36bb  the two interface functions public
  7e421b612c7  the value check's samples from `verity.commitments.randomness`; `made_up` takes its generator from its caller
  9ecee8af3b2  `node_window_embed.split` gives a `/bnk` name with the tables
  91df4b29226  the value check's key is `sample_key` under `SAMPLE_DOMAIN`
  920585e4671  the two functions are `leaves_of_rows` and `rows_of_leaves` (vllm's names)
Read by me at that head: four test files 35 passed, none xfailed (r20261113-102531-4e5a); lint 89 passed 1 skipped and `repository` 392 passed (r20261113-102557-eb01); the census at 91df4b29226 gives 3ee9420daaf7545f (four sets) and 544b577a749a2da7 (five sets) (r20261113-102040-f188).

NAMES THAT CHANGED, if your code or a driver calls them:
  IF.leaves_of_rows(rows, ty), IF.rows_of_leaves(leaves, ty, count)   (were _port_leaves, _port_rows)
  NV.sample_key(seed), NV.sample(n, key, tid, most)                   (not `draws` / DRAW_DOMAIN; the gate refuses that term)
  NV.made_up(fn, prompt, rng)                                         (the caller builds the generator)
EMBED's driver with these names is `values_run_join.py` at a1d7d11c03c on `cursor/circuits-cir274-embed-scratch-6f04`. It was compiled, not run.

WHAT TO DO, in order:
1. Merge 920585e4671 into `cursor/circuits-cir274-select-6f04`: a plain merge commit, no rebase. `git merge-tree` of it with your 8830d950959 is clean (tree 2344cc5f66ba, read by me at 10:47Z). In that merge or the commit after it, take the strict xfail off `test_joined_with_select`: the refusal it waits for is gone. Push.
2. If your part 1 runs with the hunk in the run's copy are not launched yet, do not launch them: the runs below replace them. If they are running, let them end and report them beside the runs below.
3. At the merged head, from the commit, with nothing patched in the run's copy:
   a. six sets at 4 layers and at 78 layers: digest, units, templates, wire patterns, what stays unlowered (expected: the stop check alone), the free words by kind;
   b. the seven node-window test files (`test_node_window.py`, `_dense`, `_embed`, `_join`, `_multimem`, `_select`, `_sparse`), no `--runxfail`, with `-rxXs`: all pass, none xfailed;
   c. `uv run tools/verity/check/suites.py verity-vllm --fresh -- tests/lint` and `uv run tools/verity/check/suites.py repository --fresh`;
   d. `adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json`: `failures: []`;
   e. the value check on six sets with EMBED's plant and one plant on a select unit, with the new names;
   f. mypy on the files you touched, the counts beside those at 95784be3d25b.
4. When a to d have passed, push the merged head to `cursor/circuits-cir274-join-6f04` as a fast-forward (it descends from 920585e4671): a plain push, never a force. That branch is yours for this push. If e is still running then, push anyway and say so.

RULES, as before: commit and push before each launch; a run asks 32 GB and is stopped at 48 GB; wait in passes of at most 20 minutes with a pushed checkpoint a pass; act on no event that does not name you.

IN YOUR RETURN: the merged head; each run's id with the head it ran at; the six-set digests at 4 and 78 layers; what you did not do; what you trust least.

