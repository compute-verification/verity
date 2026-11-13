For @proofs and @vllm: this changes proofs' files (the long row's two registry modules, their two circuit-check sidecars and the definitions' long-row test) and vllm's: the description-size record and its allowlist, and the long row's check template with its two test files and its circuit-check pin. No partition file names the long row, and `test_served_unit_rows.py`'s pin of main's 38 served statements is unchanged. Each `_v2` the cut now binds has its `_v1`'s signature and function. One revert undoes it: no `_v1` Definition, binding, digest pin or `RANGES` rule is removed or unregistered.

vllm's check template `DsaTopKReadRowsCut_v1` (`adapters/vllm/verity_vllm/pipeline/long_row.py`, which reached main after this branch's base) takes the mask's units from `dsa_topk_mask_cut.plan`, so the switch moved its pick and mask block to `_v2` while it bound the `_v1` positions and merge itself. Its own commit here, `0e1df5b614`, makes it bind `DsaMaskPositions_v2` and `DsaPositionsMerge_v2`. Past `WHOLE_MAX`, the template's index is then the units `dsa_topk_reads.cut` names, as its docstring says, and a new test holds it to that. vllm can reverse the commit: taking it out restores the `_v1` positions and merge.

The head, `05dc686227`, holds `main` at `1e4d5c993c`, which has #9375. It is the earlier head `0e1df5b614` merged with that `main`, 1,544 commits on. The merge had one conflict, in `test_dsa_topk_reads.py`, and moves no pin, no generated file and no `STATEMENTS` (below).

Lane circuits
Refs CIR-251
Goal circuits/programs-not-unrolled

## What changes

- `verity/definitions/registry/dsa_topk_reads.py`: `cut` binds `DsaTopKMaskWhole_v2{N,TOPK}` (up to `WHOLE_MAX`), `DsaMaskPositions_v2{N,AT,K}` per block and `DsaPositionsMerge_v2{M,K}`, in place of the `_v1`s. Its docstring, `index_block`'s and the module docstring say so. The body's own `index` still calls the `_v1`s, so `DsaTopKReadRows_v1`'s Definition and its pins don't move.
- `verity/definitions/registry/dsa_topk_mask_cut.py`: `plan` binds `DsaTopKDigitPick_v2{M,K,TOPK}` and `DsaTopKMaskBlock_v2{N,AT,TOPK}`. `DsaTopKDigitCount_v1` stays, since it has no `_v2`. The module docstring says which units `plan` and `cut` name.
- `verity/definitions/tests/test_dsa_topk_reads.py`: the cut's names in `test_the_cut_counts_its_units`, `INDEX_UNITS` and `MASK_UNITS` are the `_v2`s. Their k_log sets are unchanged.
- `verity/definitions/registry/dsa_topk_reads.circuit-check.py`: `RANGES` adds `DsaMaskPositions_v2{N=2048..40960 step 2048,AT=0..114688 step 16384,K=2048}` and `DsaPositionsMerge_v2{M=1..8,K=2048}`, each at its `_v1` rule's spots. The `_v1` rules stay.
- `verity/definitions/registry/dsa_topk_mask_cut.circuit-check.py`: `RANGES` adds `DsaTopKMaskWhole_v2{N=4096..40960 step 2048,TOPK=2048}` at the `_v1` rule's spots. The `_v1` rule stays.
- No digest or definitions pin was added or moved. Each served `_v2` binding already has its digest pin on main, at the `_v2` statics the cut serves: positions at `N=16384,AT=16384` and `N=40960,AT=0`; merge `M=8`; the picks `M=8,K=0..3`; the mask block at `AT=0` and `AT=16384`; the whole mask at `N=40960`.
- The pick and the mask block get no `RANGES` rule, as their `_v1`s had none.
- `adapters/vllm/tools/description_sizes.json`: against main's record, the 24 long-row `_v1` entries are out and the 24 `_v2` entries the cut names are in; no other entry differs (below).
- `adapters/vllm/tests/tools/allowlists/description_sizes.json`: the six long-row `_v1` entries are out. They were `DsaMaskPositions_v1{N=16384,AT=*}` and `{N=40960}`, `DsaPositionsMerge_v1{M=8}`, `DsaTopKDigitPick_v1{M=8,K=*}`, `DsaTopKMaskBlock_v1{N=16384,AT=*}` and `DsaTopKMaskWhole_v1{N=40960}`. Main's allowlist has 14 entries; this one has 8.

vllm's long-row files, in the commit `0e1df5b614`:

- `adapters/vllm/verity_vllm/pipeline/long_row.py`: `read_calls` and `units` bind `R.DsaMaskPositionsV2` and `R.DsaPositionsMergeV2` (four lines). The docstring names the `_v2` pick, mask block, positions and merge, and says the index is what `dsa_topk_reads.cut` names past `WHOLE_MAX`.
- `adapters/vllm/tests/pipeline/test_long_row.py`: the served row's counts are `DsaTopKDigitPick_v2` 4, `DsaTopKMaskBlock_v2` 8, `DsaMaskPositions_v2` 8 and `DsaPositionsMerge_v2` 1, the values the test printed on the switched tree. The layout test's wire names `DsaPositionsMerge_v2`. The new test is `test_past_whole_max_its_positions_and_merge_are_the_cuts`, at 43,008 keys and at the served 2^17: the template's positions and merge units are the ones `dsa_topk_reads.cut(t)` names, each binding as many times, and they number the blocks plus one. Up to `WHOLE_MAX` the two are not meant to agree, since the cut takes the mask whole there, a block of N keys.
- `adapters/vllm/tests/pipeline/test_serving_rows_bits.py`: the family sets name the four `_v2`s. The slow test's filter leaves out `DsaTopKDigitPick_v2`, the pick, as it left out the `_v1` before.
- `adapters/vllm/verity_vllm/pipeline/long_row.circuit-check.json`: re-pinned by `circuit-check --update-pins`. One pin moves: `DsaTopKReadRowsCut_v1{N=12,TOPK=2,BLOCK=4,READ=EmbeddingRead_v1{V=16,H=1}}`, sha512 `d27c07cc…8e8073` (main's) to `24598b2d…8c389a`.

This branch doesn't touch `boolean_dsa_topk_mask.py` (`_count_sum`, `_count_tree`, `_sum_counts`).

## Size record

The record was regenerated from main's by `description_sizes.py update -j 8` in run `r20261112-175233-95a0` at the merge `2db3d7f5c0` ("92 measured, 8 kept, 0 failed; 8 of 100 past 4,194,304 bytes"). `update` rewrites `run`, `seconds` and `gb` in every entry it measures, so the job then put back main's entry for every template whose `bytes` and `largest` equal main's (68 entries; a step of the job, not committed, which CIR-268 makes the tool's own). Against main's record:

- 24 entries in, 24 out, 0 rewritten (none with the same `bytes` and `largest` and none with different).
- Templates past 4,194,304 bytes: 31 of 100 in main's, 8 of 100 here. 23 of the 24 `_v1` entries out were past it; none of the 24 `_v2` entries in is.

The new entries, in bytes: `DsaMaskPositions_v2{N=16384,AT=*}` 292,306 to 292,390; `{N=40960}` 1,197,199; `DsaPositionsMerge_v2` 9,769 at M=1 and 78,136 at M=8; the four picks 107,351 to 136,274; the eight mask blocks 500,188 to 509,240; the whole mask 738,273. Each built in at most 2.6 s, under the 4 s that marks a rebuild test slow.

## circuit-check

The cut reaches circuit-check in one place: the block check of a Call that serving stages as its cut (`checks.block_range`, through `CUTS`). These runs are at the head before the merge with main, `b5edda0392`, whose circuit and sidecar files this head keeps; main's changes to circuit-check since are an ordering fix in `targets._registered`.

- **The four long-row sidecars** (`dsa_topk_reads`, `dsa_topk_mask_cut`, `dsa_topk_mask`, `boolean_dsa_topk_mask`, `-j 2`, run `r20261112-113220-7e07`): 385 targets, 0 failures, and 103 unchecked (warnings: dead and redundant gates, and 43 targets no realization compared with the reference).
  - `DsaTopKReadRows_v1{N=12,TOPK=2,BLOCK=4,…}`'s block check read the new cut: `DsaTopKMaskWhole_v2{N=12,TOPK=2}`, `DsaMaskPositions_v2{N=12,AT=0,K=2}` and `DsaPositionsMerge_v2{M=1,K=2}`, each at k_log 21, with no failure.
  - The served `DsaTopKReadRows_v1` roots at 4,096, 40,960, 43,008 and 131,072 keys passed by calls. Their calls are the body's, which this doesn't change.
  - Every `_v2` digest pin the cut serves matched.
- **The `_v2` rules' spots** (`--no-as-call --ranges DsaMaskPositions_v2 DsaPositionsMerge_v2 DsaTopKMaskWhole_v2`, `-j 4`, run `r20261112-132237-4074`): 14 targets, 0 failures; all three rules hold. At all 14 spots no realization compared the unit with the reference (`vectors/none`), so the correspondence check there held only the reference's totality.
- **The same spots with `--as-call`** failed 8 of 14 on `partition/unpartitioned`. `--as-call` holds a target to the partition invariant of a Call inside a served Program; these units are the cut's own proof blocks, which circuit-check classifies with `partition.applies: false`, so the `--no-as-call` run is the one that applies.
- No pin of these sidecars was added or moved, and `--update-pins` was not run on them.

## The template's commit, before it was made

These ran on a scratch branch that was never pushed, whose commits hold this head's edits to `long_row.py` and its tests. `4dcfddd223` holds the template's edit and the new test; `ead4cd10c3` adds the two test files' names. Neither commit holds the pin file. A wait is from asking for a slot to taking it, and a hold from taking it to the run's end.

- `r20261112-215227-5074` (cpu-1, 18 cores, 48 GB asked; waited 14 min 50 s, held 348 s), at `4dcfddd223`: the two test files whole, slow tests included.
  - 8 passed, 5 failed and 1 skipped (the opt-in test). The new test passed at 43,008 and 2^17.
  - Each failure printed the `_v2` names where the test held the `_v1`s, and the served row's counts. The five failures are the four from before, plus the layout test, on the merge's name in the wire.
  - The test files' names were set from that output.
- `r20261112-222012-07c9` (cpu-9, 22 cores, 48 GB asked; waited 18 min 3 s, held 278 s), at `ead4cd10c3`: the same files after their edit. 13 passed, 1 skipped (the opt-in `test_every_unit_of_a_long_row_stages_in_the_proved_form`). That includes `test_the_small_units_but_the_pick_stage_in_the_proved_form`, the slow test that stopped at its last assertion before.
- `r20261112-215257-04f3` (cpu-1, 12 cores, 48 GB asked; waited 18 min 20 s, held 5,300 s), at `4dcfddd223`, whose circuit-check inputs are this head's:
  - `circuit-check long_row.circuit-check.py --update-pins`: 16 targets, 1 failure, the pin it rewrote (`d27c07cc…` to `24598b2d…`). It wrote the pin file this head commits.
  - Then `long_row` and the four long-row sidecars (`dsa_topk_reads`, `dsa_topk_mask_cut`, `dsa_topk_mask`, `boolean_dsa_topk_mask`) together, `-j 2`, on the re-pinned file. That gave 387 targets, 0 failures and 105 unchecked (warnings, which never fail). Of the 387, 15 were reused from the step before.
  - The served `DsaTopKReadRowsCut_v1{N=131072,TOPK=2048,BLOCK=16384,…}` passed by its 14 calls. The `_v2` positions and merge at small statics (`DsaMaskPositions_v2{N=8,AT=0,K=4}`, `{N=6,AT=8,K=4}`, `{N=14,AT=0,K=4}`, `DsaPositionsMerge_v2{M=1,K=4}` and `{M=3,K=4}`) were each compared with the reference on 64 random vectors, with no failure.
- **No check of the `_v1` template fails at `_v2`.** Against `r20261112-183858-c97c`, the long-row sidecar alone at the previous head, every target that passed there passes here.
  - The one difference is which positions that sidecar pulls in. With `_v1`, it took the `dsa_topk_reads` sidecar's small `DsaMaskPositions_v1` roots, which it names. With `_v2`, its target is the served `DsaMaskPositions_v2{N=16384,AT=16384,K=2048}`, which no realization compares at that size.
  - The small `_v2` positions are checked in the same run through `dsa_topk_reads`' own roots, above.

## The merge with main at `1e4d5c993c`

The one conflict was the two asserts of `test_the_index_and_read_units_pass_the_window_builders_proof_block_check`. Main raised the slot to `IF.PACK_LOG`, and this branch names the `_v2` merge. The region, as closed:

```diff
-<<<<<<< HEAD
-    assert k["DsaPositionsChunk_v1"] == k["EmbeddingRead_v1"] == {23} and k["DsaPositionsSlot_v1"] == {21}
-    assert max(k["DsaPositionsMerge_v2"]) <= 26
-=======
-    assert k["DsaPositionsChunk_v1"] == k["EmbeddingRead_v1"] == {23} and k["DsaPositionsSlot_v1"] == {IF.PACK_LOG} == {22}
-    assert max(k["DsaPositionsMerge_v1"]) <= 26
->>>>>>> origin/main
+    assert k["DsaPositionsChunk_v1"] == k["EmbeddingRead_v1"] == {23} and k["DsaPositionsSlot_v1"] == {IF.PACK_LOG} == {22}
+    assert max(k["DsaPositionsMerge_v2"]) <= 26
```

What was checked at this head in `r20261113-033216-49da` (cpu-10, 16 cores; a `--cwd clone` job, since the record's step writes into the tree):

- **`test_dsa_topk_reads.py`, every test:** 40 passed, 1 skipped (the one `VERITY_ML_DEFINITIONS_SLOW=1` runs). The test above passed: at every served class, the slot's k_log is 22 and the `_v2` merge's is at most 26.
- **The size record:** `description_sizes.py update -j 8`, main's tool, which now keeps an entry whose description has not moved, rewrote the record byte for byte as committed ("92 measured, 0 of them written; 100 entries kept as they stood, 0 failed"). `test_description_sizes.py`: 101 passed.
- **`STATEMENTS`:** `test_served_unit_rows.py` passed with the pin unchanged.

**Pins.** No pin moved, and `--update-pins` was not run. `r20261113-034020-4471` (cpu-1, 12 cores) ran `circuit-check` on `long_row.circuit-check.py` and `dsa_topk_reads.circuit-check.py` at this head: 87 targets, 0 failures, 39 unchecked (warnings), with every pin matching.

**What the template commit rests on.** None of the five long-row files changed on main between `15662e7c40` and `1e4d5c993c`: `long_row.py`, `long_row.circuit-check.py`, `long_row.circuit-check.json`, `test_long_row.py` and `test_serving_rows_bits.py`. Nor did `dsa_topk_reads.py` or `dsa_topk_mask_cut.py`, which hold the `_v2` Definitions it binds. #9375 renames `Registry`, `Node`, `interface.Opening`, `interface.rows_k_log` and `interface.server_ends`; no Python file this change touches names them.

## Runs at this head

- `r20261113-033454-e466` (cpu-1, 12 cores, 64 GB asked; took its slot at 03:41Z), source `05dc686227` (its `head.txt`). It runs `scratch`'s `headsuites2.sh`, in three steps:
  - `suites.py --changed 1e4d5c993c` whole, not `--quick`: the 60 suites the change from that `main` touches, `verity-vllm` and `repository` among them, with every slow test.
  - `repository` by name.
  - The two long-row files with `-rA`.
- Its first step, read at 06:02Z: 48 of the 60 suites had passed, none had failed and none was cached. `repository` had 392 passed.
- 12 suites had not ended at 06:02Z and are not read: `verity-vllm`, `verity-numerical`, `research`, circuit-check's suite, `verity/circuits`, `verity/commitments`, `verity/commitments/windows`, `verity-core-evaluate`, `verity/definitions/ops`, `verity/definitions`, `verity-flock` and `verity/protocols/pouw/verifier`. The two long-row files are in `verity-vllm`. The run's last two steps had not started.
- `ratchet_gate.py origin/main HEAD --json` at `05dc686227`, base `1e4d5c993c08`: `"failures": []`. The change from `1e4d5c993c08` passed, 14,952 to 14,946 ratchet exceptions.

## Runs at `0e1df5b614`, before the merge with main at `1e4d5c993c`

- `r20261112-234237-75ca` (cpu-14, 16 cores, 48 GB asked; waited 1 min 19 s), source `0e1df5b614` (its `head.txt`). It runs `scratch`'s `headsuites.sh`, in three steps:
  - `suites.py --changed 4176404df6 --quick`: the 23 suites the change touches, `verity-vllm` and `repository` among them. Under `--quick` the two long-row test files keep their slow tests.
  - `verity-vllm` and `repository` by name.
  - The two long-row files with `-rA`.
  - 23 suites: 23 passed. `verity-vllm` had 14,362 passed, 401 skipped and none failed; `repository` had 352 passed.
  - Its last two steps (the named suites, then the long-row files with `-rA`) were still running at 00:45Z and are not read.
- `ratchet_gate.py origin/main HEAD --json` at `0e1df5b614`, base `7e70df68bd18`: `"failures": []`. The change from `15662e7c40` passed, 15,002 to 14,996 ratchet exceptions.

At the previous head `4176404df6`, before the template's commit:
- `r20261112-180706-ae82`: 57 suites, 56 passed. `verity-vllm` failed on the three long-row tests that are not slow.
- `r20261112-183858-c97c`: the long-row tests, of which four failed, and the pin `d27c07cc…` to `877f04d2…` for the mixed template, which was never pinned.

## Not run

- `circuit-check --all` at this head. It ran at #9371's head `a62164a32a`, which holds this one: 17,340 targets, 0 failures (#9371's body).
- The whole uncached check, the Lean check and lean-agreement. No Lean or Flock file of this change's own moves.
- The `_v2` rules' spots again at this head: they ran at `b5edda0392` (above).
- At the earlier heads, the slow tests of suites whose test modules the change doesn't reach (`--quick`); at this head they are in the run above, in its 12 suites not yet read. Among them are `adapters/vllm/tests/program/test_dsa_long_row_calls.py`'s slow tests, which hold each `_v2` to its `_v1`.
- No circuit-check target checks the cut at served sizes. The served roots are checked by their calls, which are the body's. The served cut's blocks are held by `test_dsa_topk_reads.py`'s proof-block tests (every unit at every served class, at the k_log sets in `MASK_UNITS` and `INDEX_UNITS`).
- Staging or proving any unit at served size: not this PR's.
- #9254 and #9075 are still open and regenerate the record too. Whichever reaches `main` second, this or they, takes one more merge.

