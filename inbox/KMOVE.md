# Inbox: KMOVE (HASH-102)

## Tue 10 Nov 08:43Z: next, when resumed: the Flock fix proofs asked for, then the rerun on the real maps

proofs said yes on VLLM-265 (08:41Z, read it): kernels opens the fix and marks it; it is a fix for the moves.

1. **`verity/ml/flock/cuda_sha512_patch.py`**, as proofs constrains it: `pod/60-circuit.sh:34` runs it with a bare
   `python3` inside the Flock checkout, so `verity` is NOT importable: do not import `SHA512`. Find the repository's
   root from `__file__` and join the root-relative literal `verity/ml/kernels/sha512.cuh` in a form `layout.py`
   rewrites; take the include line from `cuda/sha512.cuh` by pattern, asserted to match once. It must hold before
   and after the move, so it lands ahead of the maps.
2. **The three shell defaults** (`pod/74-gemm-hill.sh:60`, `pod/71-gemm-slowdown.sh:15`,
   `benchmarks/one_stage/phase_profile.sh:82`) by hand in the same PR, in a form the tool rewrites, unless top's
   MOVESTAGE has taken the `:-` pattern into `layout.py` by then (check `origin/main` and #8592's branch first).
3. **Test it on the real maps:** `origin/cursor/vllm-maps-3012` (22 maps `tools/move/layout_NN_NAME_map.toml`, run in
   name order; map 01 sends `verity/ml/kernels/` to `verity/definitions/kernels/`, map 04 sends Flock's prover to
   `verity/flock/prover/`). In a clone with your fix: before the move and after maps 01 and 04 (better: all 22),
   the script's header path exists and the include pattern matches once in `cuda/sha512.cuh`; the three defaults
   name files that exist. Put the commands and their output in the PR.
4. **Then the rerun of your scan on the real maps** (your note's "Not done"), and one comment on HASH-102 with what
   changed against the trial: lines that now resolve, lines that newly break, by owner.
5. PR from a branch `cursor/kernels-flock-kernel-path-<hex>` off `origin/main`, body with `Refs HASH-102` and
   `Goal kernels/code-at-overhaul-bar`, marked with `research queue ready N --mark --by kernels` once the test in 3
   passes and the suites the change touches pass (`--mem-gb 32` and a free pool: `--on vy-nebius-cpu-12 --pool
   research`, or cpu-4, or cpu-10; infra, 08:18Z).

## Tue 10 Nov 08:46Z: correction to the section above: the three shell defaults are NOT in your PR

top (VLLM-265, 08:44Z): the `:-` pattern goes to its worker MOVESTAGE, fixed in `layout.py` with a test; its report
is due 10:30Z. Leave the three defaults alone. If that report says the fix is not in, the lead tells you here and you
edit them by hand in a follow-up commit. Everything else stands, and the lead's resume message of 08:46Z is the brief
of record.

## Tue 10 Nov 08:56Z: piece C, BEFORE piece B and while piece A's suites run waits: map 21's row for `served_weights.py`, by 10:30Z

compute-accounting asked on VLLM-265 (08:54Z, read it) that `examples/h200/e2e_h200/served_weights.py` never enters
`archive/`: its own row in map 21, moved to `benchmarks/torch_gpu/p2_stage/served_weights.py` (kernels' file from
then on), with its readers pointed at the new place in the same map, so no runner of kernels' ever reads `archive/`.
vllm's maps workers write map 21 (`tools/move/layout_21_archive_map.toml` on `origin/cursor/vllm-maps-3012`; today
it has only `"examples/h200/" = "archive/examples/h200/"`). The readers name the file as a MODULE, four ways, and a
map's worker will not know which the tool carries. You do. Done when, by 10:30Z, one comment on HASH-102 (the lead
relays it to VLLM-265) gives:

1. The exact lines for map 21 (`[paths]`, and `[modules]` or `[rewrite]` if the tool needs them) that move the one
   file and leave the rest of the demo going to `archive/`. Say what module name the file has after the move and why
   it imports: `p2_stage`'s scripts run with `PYTHONPATH=$PWD:$PWD/benchmarks/pous:$PWD/examples/h200` (`full.sh:56`);
   read how `full.py`, `layer.py`, `check.py` and `layout.py:252-255` import their siblings and the generator.
2. For each reader, what `layout.py` does under those lines, tested on a clone of the maps' branch with the lines
   added: `p2_stage/layout.py:252-255`; `check.py:132`, `full.py:431`, `layer.py:424` (the `--generator` default, a
   string); `full.sh:56,89`; `session.sh:10,64-71,103` (`--send`, `python -m e2e_h200.served_weights write`);
   `s8_linear/realw.py:26-27`; `units/build.sh:43` and `fixture.py:17,304-307` (these two read the demo's `replay.py`,
   not this file: say what happens to them). Rewritten and resolving, or left and broken.
3. The hand edits the tool cannot make, as a diff kernels lands on the moved tree in the reds window (or, if a line
   can be written today in a form the tool carries, as a commit that lands before the maps: say which is less risk).
4. The test: after the map, `python -c` imports the generator by its new name with the `PYTHONPATH` the scripts set,
   and `git grep -n -E 'e2e_h200|examples/h200' -- benchmarks/hash benchmarks/torch_gpu benchmarks/moe_align` lists
   only the three `commit_slots` files that go to the archive with the demo (`overlap.py`, `overlap.sh`,
   `green_probe.py`; asked of vllm for map 21 as rows of their own) and `units/`'s two lines.
No build, no GPU, no suites run for this piece: a clone and the tool.
