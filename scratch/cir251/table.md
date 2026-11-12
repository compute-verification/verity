# CIR-251: what moves when the served ids switch

Scratch, never merged. Read on `main` at `19fb19a21e`. Switch (a): the long row's cut (`dsa_topk_reads.cut`,
`dsa_topk_mask_cut.plan`) binds the five `_v2` units. Switch (b): the DSA norm's three `opt_in` marks go, and the partition
files name the `_v3` parts.

Tags in brackets: **ran** = a script or test in this directory, run on this VM; **read** = I read the line; **record** =
a run record I fetched (`research fetch --all`); **inferred** = not checked.

## 1. Files that change

**Must** means a test fails, or a stored value is wrong, without the edit. **Should** means everything passes, but the file
then names forms that are no longer served.

| # | Path | Owner | Part | Must/should | What changes | How I know |
|---|---|---|---|---|---|---|
| 1 | `benchmarks/one_stage/glm53_prefill_tp8_step0.json` | @proofs | b | must | `DsaRMSNormSumSq_v2{N=512}`, `{N=2048}` and `DsaRMSNormApply_v2{G=64}` become `_v3`, in `templates` and `chunk_cap`. The caps stay 8192, 2048 and 16384. | read; caps equal the `_v3` caps in r20261111-175248-7290 [record] |
| 2 | `benchmarks/one_stage/glm53_prefill_tp8_step0_core.json` | @proofs | b | must | `SumSq{N=512}` and `Apply{G=64}` become `_v3`. Caps 8192 and 16384 stay. | read; record as above |
| 3 | `benchmarks/one_stage/glm53_all_families_tp8_step16.json` | @proofs | b | must | `SumSq{N=512}` and `Apply{G=64}` become `_v3`. Caps 1 and 8 stay. | read; the caps staying is inferred |
| 4 | `adapters/vllm/tests/pipeline/test_served_unit_rows.py` | @vllm | b | must | `STATEMENTS` goes from `eed75699…0d7110a` to `36b89ee9…6b3934ac` (full value in `served_statements.py`'s output). The count stays 43. | ran `served_statements.py`: today's digest equals the pin |
| 5 | `verity/definitions/registry/dsa_topk_reads.py` | @proofs | a | must | `cut` lines 473–475 bind `DsaTopKMaskWholeV2`, `DsaMaskPositionsV2` and `DsaPositionsMergeV2`. Docstrings at line 47 and in `cut` change. The body's `index` (lines 390, 394) may stay `_v1`. | read |
| 6 | `verity/definitions/registry/dsa_topk_mask_cut.py` | @proofs | a | must | `plan` lines 470 and 474 bind `DsaTopKDigitPickV2` and `DsaTopKMaskBlockV2`. The docstring at line 36 changes. | read |
| 7 | `verity/definitions/registry/boolean_dsa_norm.py` | @proofs | b | must | Lines 165, 200 and 211 (`opt_in = True`) go. Docstrings at lines 8 and 24–25 change. | read |
| 8 | `verity/definitions/tests/test_dsa_topk_reads.py` | @proofs | a | must | 4 of 39 tests fail: `test_the_cut_counts_its_units` (lines 323–330), `test_every_served_unit_has_the_rows_flocks_composer_proves` (`INDEX_UNITS`, line 339), `…index_and_read_units…` (line 376) and `…mask_units…` (`MASK_UNITS`, lines 382–383). | ran `switched_pytest.py a` |
| 9 | `adapters/vllm/tests/program/test_dsa_norm.py` | @circuits | b | must | `test_registrations` fails at line 150, which asserts `boolean_version` = `DsaRMSNorm_v2{N=512}`. | ran `switched_pytest.py b` |
| 10 | `adapters/vllm/tests/program/test_dsa_norm_calls.py` | @circuits | b | must | Line 56 asserts `boolean_version(word).id == traced.id` (the `_v2`). Docstrings at lines 1–2 and 49 change. | read; a slow test, not run |
| 11 | `adapters/vllm/tests/properties/glm53_fused_rmsnorm_capture_gpu.py` | @circuits | b | must | Line 356 records whether `boolean_version` = `DsaRMSNorm_v2`, which turns False. A GPU tool, in no suite. | read |
| 12 | `adapters/vllm/tools/description_sizes.json` | @vllm | a+b | must | 27 entries out (24 long-row `_v1`, 3 norm `_v2`) and 27 in. Past 4 MiB: 33 → 7. Needs `update` as a node job with `--mem-gb 16`. | ran (size test fails on a: "24 served templates have no entry"); sizes in §4 |
| 13 | `adapters/vllm/tests/tools/allowlists/description_sizes.json` | @vllm | a+b | must | 9 entries go: 6 long-row patterns and 3 norm entries. | ran (a: 6 named stale); the 3 norm entries are read |
| 14 | `verity/definitions/registry/dsa_topk_reads.circuit-check.py` | @proofs | a | should | `RANGES` names the served ranges as `DsaMaskPositions_v1{…}` and `DsaPositionsMerge_v1{…}`. | read |
| 15 | `verity/definitions/registry/dsa_topk_mask_cut.circuit-check.py` | @proofs | a | should | `RANGES` names `DsaTopKMaskWhole_v1{N=4096..40960}`. | read |
| 16 | `adapters/vllm/tests/pipeline/test_serving_rows_norm_split.py` | @vllm | b | should | Lines 55–56 bind `DsaRMSNormSumSqV2` and `ApplyV2` as the Boolean path. It passes after (b). | read; ran (b alone passes) |

Counts: 13 must and 3 should, 16 files in all. By owner: @proofs 9, @vllm 4, @circuits 3. By group: partition files 3,
pins and digests 1 (+2 should), code and tests 8 (+1 should), size record and list 2, benchmarks and census 0, docs 0 (the
docstrings are inside files 5–7 and 10).

### Checked, no change

| Path | Owner | Why it stays | How I know |
|---|---|---|---|
| `verity/definitions/registry/{dsa_topk_reads,dsa_topk_mask_cut,boolean_dsa_topk_mask,boolean_dsa_norm}.circuit-check.json` | @proofs | They pin Definitions at their own bindings, never the cut. They already pin the `_v2`/`_v3` at 10 of the 24 new long-row bindings and at all 5 served norm bindings. The other 14 long-row bindings could get pins (optional). | read |
| `verity/definitions/tests/catalog_ids.json` | @proofs | It names family ids. It moves only if a `_v1` is deleted. | read |
| `verity/definitions/tests/test_dsa_topk_mask_cut.py` | @proofs | 30 of 30 non-slow tests pass with (a). The `_v2`s share the `_v1`s' signatures and functions. | ran |
| `adapters/vllm/tests/program/test_dsa_long_row_calls.py` | @circuits | It binds both versions by name and never reads the cut. | read; the run went past a minute, so it is not run here |
| `adapters/vllm/tools/description_sizes.py` | @vllm | It reads `R.cut`. Only its output (file 12) moves. | read |
| `benchmarks/served_units/stage.py`, `tests/test_names.py` | — | They name `DsaPositionsChunk`/`Slot` (kept) and census `DsaRMSNorm_v1`/`_v2` strings. | read |
| `verity/definitions/tests/test_mnnvl_and_dsa_norm_splits.py` | @proofs | It evaluates `boolean_version` on bits, so it will run the `_v3`s, which have the same function. | inferred; it hit the 1 GB guard here with and without the switch |
| `benchmarks/sampling_plan/glm53_blocks.json` (and its builder `benchmarks/one_stage/block_census.py`) | @proofs | It names the word `DsaRMSNorm_v1{N=2048}`/`{N=512}` only, from a census pinned by `art:` id. None of the 26. | read |
| `benchmarks/spot_figures/cards/*.glm-5.3.json`, `benchmarks/harness/tests/fixtures/{prove,exfiltration}/…` | @proofs | They name word kinds only. | read |
| `tools/verity/circuit_check/src/circuit_check/heavy.json` | @circuits | It is keyed by the sidecar root `DsaRMSNorm_v2` at N=256. | read |
| `adapters/vllm/manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json` | — | A record of a past capture. | read |
| Flock (`verity/protocols/flock/**`) | — | No file names the seven families, under any spelling. | `git grep` |

### Stored digests: found by reading vs found only by a run

- **Found by reading:** `test_served_unit_rows.py`'s `STATEMENTS` (new value computed). No circuit-check pin moves. The Lean
  locks are not touched.
- **Only a run names:** any pin of a served Program's or partition's digest in vllm's or benchmarks' suites. vllm runs its
  suites on the switch branch. I did not guess these.

## 1b. For each of the 26 templates: the files of @proofs' that name its id or family

None of the 26 exact ids is in a Flock file, the card, or any proofs file except where listed. `description_sizes` (the
record and the list) is vllm's and names all 26.

| Template(s) today (count) | @proofs' files that name the id or family | Edit |
|---|---|---|
| `DsaMaskPositions_v1{N=16384,AT=0..114688 step 16384,K=2048}` (8), `{N=40960,AT=0,K=2048}` (1) | `dsa_topk_reads.py` (`cut` line 474), `dsa_topk_reads.circuit-check.py` (`RANGES`), `test_dsa_topk_reads.py` (lines 330, 383) | yes, should, yes |
| `DsaPositionsMerge_v1{M=8,K=2048}` (1) | `dsa_topk_reads.py` (line 475), `dsa_topk_reads.circuit-check.py` (`RANGES`), `dsa_topk_reads.circuit-check.json` (exact-id digest pin), `test_dsa_topk_reads.py` (lines 327, 339, 376) | yes, should, keep (it pins the `_v1` itself), yes |
| `DsaTopKDigitPick_v1{M=8,K=0..3,TOPK=2048}` (4) | `dsa_topk_mask_cut.py` (`plan` line 470), `dsa_topk_mask_cut.circuit-check.py` + `.json` (`_v1` roots and exact-id pins), `test_dsa_topk_reads.py` (line 382) | yes, keep, yes |
| `DsaTopKMaskBlock_v1{N=16384,AT=0..114688 step 16384,TOPK=2048}` (8) | `dsa_topk_mask_cut.py` (`plan` line 474), `test_dsa_topk_reads.py` (line 383) | yes, yes |
| `DsaTopKMaskWhole_v1{N=40960,TOPK=2048}` (1) | `dsa_topk_reads.py` (`cut` line 473), `dsa_topk_mask_cut.circuit-check.py` (`RANGES`), `test_dsa_topk_reads.py` (lines 323–324, 382) | yes, should, yes |
| `DsaRMSNormSumSq_v2{N=512,EPS=1e-5}` (1) | the three partition files, `boolean_dsa_norm.py` (`opt_in`) | yes, yes |
| `DsaRMSNormSumSq_v2{N=2048,EPS=1e-5}` (1) | `glm53_prefill_tp8_step0.json`, `boolean_dsa_norm.py` | yes, yes |
| `DsaRMSNormApply_v2{G=64}` (1) | the three partition files, `boolean_dsa_norm.py` | yes, yes |

## 2. Readers of `boolean_version` that see the norm move

Partition files do not follow `boolean_version`: `serving_templates.template` resolves a partition's id itself. So dropping
`opt_in` alone moves no served statement [ran: with (b) alone, `test_served_unit_rows.py` and the size test pass].

- **Assert today's answer:** `adapters/vllm/tests/program/test_dsa_norm.py:150` fails [ran].
  `adapters/vllm/tests/program/test_dsa_norm_calls.py:56` fails [read; slow].
- **Record it:** `adapters/vllm/tests/properties/glm53_fused_rmsnorm_capture_gpu.py:356` [read].
- **Report it, nothing stored on main:** `verity/definitions/circuits/census.py` (a class's `boolean_version` field), so
  KSTEADY's census kinds; `benchmarks/one_stage/a11_kinds.py:222`; `benchmarks/one_stage/a11.py:565`;
  `benchmarks/verifier_cost/lift.py:45`; `benchmarks/unit_block/run.py:57`;
  `adapters/vllm/verity_vllm/program/predict/block_counts.py`; `adapters/vllm/verity_vllm/pipeline/chunk_rows.py:131` (a
  refusal's text) [read].
- **Pick the `_v3` and still pass:** `verity/definitions/tests/test_mnnvl_and_dsa_norm_splits.py:92–96` [inferred].
- **Not the norm:** `serving_templates.py:492` (the `DOT` only). `test_serving_rows_gates.py` and
  `test_zk_partition_names.py` cover other families.
- `boolean_version` for the five served norm words goes from `_v2` to `_v3` [ran `norm_versions.py`].

## 3. KSTEADY's 45 kinds

Notes `lanes/kernels/ksteady-jobs/h98/boolean-versions.md` (notes commit `53126c6695`): 11 numbered rows + 34 unnumbered = 45.
Switch (b) moves 2: `DsaRMSNorm_v1{N=2048}` (0.013%) and `DsaRMSNorm_v1{N=512}` (0.006%). Their Boolean version goes from
`DsaRMSNorm_v2` (traced) to `DsaRMSNorm_v3` (written as Calls). At N=2048 the ANDs go from 12,356,735 to 12,393,600, and
the description from 723,273,584 to 1,355,569 bytes [record r20261111-175248-7290]. No long-row unit is among the 45, so
switch (a) moves none. The census row is the whole norm row, while the served layers prove its two parts.

## 4. Sizes, and staging per unit

New description bytes, all under 4 MiB [ran `new_sizes.py` at `19fb19a21e`; the norm's scale from the record]:

| Template | Bytes |
|---|---|
| `DsaMaskPositions_v2{N=16384,AT=*}` | 292,306–292,390 |
| `DsaMaskPositions_v2{N=40960,AT=0}` | 1,197,199 |
| `DsaPositionsMerge_v2{M=1}` | 9,769 |
| `DsaPositionsMerge_v2{M=8}` | 78,136 |
| `DsaTopKDigitPick_v2{M=8,K=0..3}` | 107,351–136,274 |
| `DsaTopKMaskBlock_v2{N=16384,AT=*}` | 500,188–509,240 |
| `DsaTopKMaskWhole_v2{N=40960}` | 738,273 |
| `DsaRMSNormApply_v3{G=64}` | 269,724 (also in the record) |
| `DsaRMSNormSumSq_v3{N=512}` | 1,088,488 [record; past the 1 GB guard here] |
| `DsaRMSNormSumSq_v3{N=2048}` | 1,090,688 [record] |

Staging (`UnitStager`, as `chunk_rows.chunk_cap` stages) at each unit's largest served binding:

| Unit | Largest served binding | ANDs | Staging peak / time | Run |
|---|---|---|---|---|
| `DsaMaskPositions_v2` | N=40960, AT=0 (k_log 25) | 10,919,817 | **not staged**: cancelled at 492.3 GiB after 32 min, still growing | r20261111-203604-3b94 |
| — same, next size down | N=16384, AT=16384 | 4,272,029 | 172.86 GB / 1,299 s | same run |
| `DsaPositionsMerge_v2` | M=8, K=2048 | 6,046,233 | 9.06 GB / 176 s | r20261111-214928-87b8 |
| `DsaTopKMaskWhole_v2` | N=40960 (k_log 27, 0.9941 of the block) | 6,550,553 | **not staged**: killed at 312.72 GiB | r20261111-193600-4237 |
| `DsaTopKMaskBlock_v2` | N=16384, AT=16384 | 1,544,137 | 48.37 GB / 778 s | r20261111-214822-ffea |
| `DsaTopKDigitPick_v2` | M=8, K=3 | 132,532 | 11.28 GB / 141 s (K=0..2: 11.2–11.5 GB, 138–144 s) | r20261111-200014-2e92 |
| `DsaRadixTies_v2` | N=40960, inside the whole mask; not a unit | 1,265,513 | description 0.2 s / 0.06 GB; not staged alone | r20261111-193600-4237 |
| `DsaRadixPass_v2`, `DsaRowValid_v2` | N=40960, inside the whole mask; not units | — | none in the runs I opened | — |
| `DsaRMSNormSumSq_v3` | N=2048 (cap 2048) | 6,151,262 | 7.56 GB / 218 s | r20261111-175248-7290 |
| `DsaRMSNormApply_v3` | G=64 (cap 16384) | 195,106 | 1.12 GB / 18 s | same run |
| `DsaRMSNorm_v3` | N=2048; served by no partition file | 12,393,600 | 15.4 GB / 441 s | same run |

The two killed whole-row-mask cases are both in r20261111-193600-4237. It ran on `vy-nebius-cpu-6`, tree `5edf71778c08`,
as `cir215_figures_longrow.py all -j 3` over 8 cases, in check slot r1 with a 161.1 GB reservation. Process peaks and
times come from its `resources.jsonl`, matched to cases by launch order and the logs:

- `whole.v2.40960` built its description (10,496,840 bytes at that tree, 10.4 s, 0.92 GB) and read its block (k_log 27).
  It died in staging at 20:12:01Z, at 312.72 GiB, 23 min in. The job's cgroup had reached 604.66 GiB, with `whole.v1`
  beside it (`oom_kill` +2).
- `whole.v1.40960` built its description (148,488,690 bytes, 104.5 s, 7.3 GB). It died in staging at 20:45:22Z, at
  603.42 GiB, alone, 57 min in.

What to try next (no run started):

- **(i)** Stage `whole.v2.40960` alone (`-j 1`, one case) on a CPU node, with `--mem-gb` near that node's cap. Then
  `positions.v2.40960` the same way. Each holds a node for up to about 1–2 h, at up to about 600 GiB, and may still die.
- **(ii)** Without a run: a cut that leaves no row whole past 16,384 keys would serve only bindings already staged
  (48 GB and 173 GB). That changes the cut's shape, so it is proofs' design call.
- **(iii)** Staging memory tracks neither ANDs nor `nnz_a`. The merge block holds 37M nonzeros and staged in 9 GB. The
  mask block, at 1.36G, took 48 GB, and the positions block, at 0.67G + 0.70G, took 173 GB. That fits a benchmark of its
  own.

## 5. Two ways to switch

**Outright.** One PR changes five bindings in `cut`/`plan`, drops three `opt_in` lines and renames the norm in the three
partition files. It re-pins `STATEMENTS`, remakes the size record and list, and fixes the tests in files 8–11. It
moves every served statement with the norm, plus the long row's check, at once. The `_v1`/`_v2` Definitions and their pins
stay registered, so one revert undoes it.

**By a flag for a while.** Nothing on `main` reads a flag here: no setting in `cut`, `plan`, `boolean_dsa_norm`,
`serving_templates` or `description_sizes`. `plan` is `@cache`d, and `CUTS` maps one id to one function, so a flag would be
fixed per process. The partition files name ids, so the norm would need two sets of partition files. The other option,
serving rewriting ids by the flag, breaks proofs' condition that a statement names exactly its partition file's Definition.
The size test wants exactly the served set, so the record would hold both sets or read the flag. It moves the same files,
plus the flag, both arms' records and tests, and a second PR to remove it.

**Which the code favours: outright.** The modules state that a version "serves only a caller that binds it". The norm's
split moved its partition files and re-pinned in one commit (`ab618b481a`). Both the size test and the served-rows pin
demand exactly the served set.
