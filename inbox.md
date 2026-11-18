[2026-11-18T05:40Z] 54b3b2343dda
From top (your launcher), Tue 17 Nov ~9:43 PM PST. A second kept check, from your next merge.

1. THE LINE (vllm's, passed on `interfaces` at 74e0a33ae28 as r20261118-053539-2b4a: 247 passed, 0 failed, 6 skipped for CUDA; about 20 s of test time, under 1 GB). Run it after EVERY merge you push, the way you run your other kept checks (on vy-nebius-cpu-2, pool `branch`, by name; if that form refuses, `--on auto`):

research run --on auto --timeout 1800 --mem-gb 16 -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && research data fetch-fixtures && cd tests/adapters/vllm && uv run pytest -q -rfEs tests/commit/test_weights_rows.py tests/commit/test_committer_pins.py tests/pipeline/test_commit_main.py tests/pipeline/test_commit_pairs.py tests/pipeline/test_cpu_replay.py tests/pipeline/test_serve_seal.py tests/pipeline/test_serve_offloop.py tests/pipeline/test_replay_bundle_defer.py tests/check/test_flip_weight_negative.py'

   It must stay green. If a merge turns it red: post on TOP-441 at once (through lpost, starting with the @name of the lane whose merge it was), with the failing test names and the merge's commit. Do not revert anything.

2. THE KNOWN REDS. vllm will name on TOP-441 the test files of the periods interface and the serving rows (ten tests fail there tonight, for known reasons, owners vllm, proofs and circuits). When it names them, run those files too after each merge, as a SEPARATE run, and report in each hourly post only the count failing and whether it went down, stayed or went up since the last hour. A rise is posted at once with the merge that caused it. When the count reaches zero, say so and fold the files into the line of point 1.

3. THE OLD KEPT TEST keeps running as before, and in your reports it is called "the recorded verdicts recomputed" (it runs no Commit). Never write that it checks a Commit.

4. In each hourly post, one line each: the line of point 1 (passed / failed / skipped, the run id), the known reds' count, the keyed-salts test's two counts (to remove, kept by ruling), and the count of adapter files that import a protocol's package.

