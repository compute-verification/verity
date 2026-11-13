# CIR-274 join trial: the token select with the all-reduce and the embedding shard (part 1)

Select branch `cursor/circuits-cir274-select-6f04`:
- `026c3f93d4aa3ba8e29b8765d9fc8495ca56f7db`: the merge of the join head `95784be3d25b`, parents `c3f66d06e5` and
  `95784be3d25b`, tree `fe07c882dedb`, which `git merge-tree` predicted.
- `89535c2cc3`, then `8830d950959675e1df3d4169d1b5dece2350ddd3` (head): `test_joined_with_select` in
  `test_node_window_select.py`.

Nothing was pushed to the join branch. EMBED's `multimem-join.diff` (sha256 `df8932430747de41…62df42`, copied here as
`multimem-join.diff`) is applied only in each run's copy; each run records the head, the diff's sha256 and the applied diff
stat in `$OUT`.

## The six-set window: 43 ids, 2 outputs, 4 layers, rank 0

Run `r20261113-101728-9378` (`--cwd clone`, head `026c3f93d4` with the hunk), `join6_census.py`.

- Five sets, as a control: digest `544b577a749a2da7…`, 5,190,049 units, 109 templates, 13,268 wire patterns. That is EMBED's
  window, on `glm_request`'s Program `7f36f17ae005fc05…`.
- Six sets: digest
  `04fcba3603a6c80515b75d6a4376a22a0441b4e7f0796c338cb52e304eeaac45c031e37b2da6ec43351bcd66b33e0035e42c0120f11c9763b50d4fd7a5173bf3`;
  5,190,089 units, 129 templates, 13,448 wire patterns.
  - `settle`, `check_wires` and `check_proof_block` (all 129, under their cuts) passed; nothing refused.
  - Against five sets: +40 units, +20 templates, +180 wire patterns, all of them the `TokenSelect` family's.
- Unlowered: `StopCheck` 1, and nothing else.
- Free words by kind: peers 17,573,248, prompt 2,752, weights 322,961,408. `TokenSelect` is gone.
  - Five sets had `TokenSelect` 64 and peers 17,302,208; the six sets' peers add the select's 2,240 rows of 121 words
    (271,040).
- The decode's shard reads take `tok` by wire from the select: one wire from `GreedyCombine_v2{V=154880,N=7744}` `o0` into
  `EmbeddingRead_v1{V=154880,H=192}` `tok`, at 1,376, radix (32), base 0, step (0). Units 1,376..1,407 all read combine
  unit 0, the prefill's select.
  - The read's only free rows are the prefill's `tok`: 1,376 units from the `prompt` parameter.
  - The combine's `o0` has that one reader: the decode's select (unit 1) feeds only the unlowered stop check.
- The prefill's shard reads still feed row 0 of the `SIZE=192` combine's `parts`: one wire, at 0, radix (43, 32), base 0,
  step (1, 43), cut row 0. It covers 1,376 reader units of `AllReduceMultimem_v1{WORLD=8,SIZE=192,…}`, which has 2,752 units
  with `parts` cut 8 ways at 4 layers.

Nothing refused, so no cure was needed in my files and no diff was needed for anyone else's.

## The test

`test_node_window_select.py::test_joined_with_select` builds the six sets at one decoder layer and holds the three facts above.
It is a strict xfail on the multimem chunk refusal, as `test_joined` is.

- Run `r20261113-102238-9d26` (head `8830d95095`, `--cwd clone`), ruff clean:
  - Without the hunk: the select file's 5 tests passed, and both join tests are expected failures.
  - With the hunk, under `--runxfail`: `test_joined_with_select` and `test_joined` passed.
- Run `r20261113-102435-57de` (head `8830d95095`, no hunk): the seven node-window test files gave 60 passed and 2 expected
  failures.
- `r20261113-102238` was preceded by `r20261113-102042-891a`, at `89535c2cc3`. There my last assertion took the 4-layer
  combine's units (2,752) where one layer has 1,376; I fixed it in `8830d95095`.

## The value check

Run `r20261113-101643-b602` (head `026c3f93d4` with the hunk), `values_join6.py --ids 43 --layers 4`. The driver is EMBED's
`values_run_join.py` with the select's family added, the chunks anchored by `greedy_stages.twin` (as in d5b0) and a plant on
select chunk 0's logits row beside EMBED's read plant.

This run was launched with `--cwd source`, and its `git apply` edited the shared source tree of `026c3f93d4` on
vy-nebius-cpu-8. The census launched beside it was refused (SourceNotReady); I relaunched it with `--cwd clone`, as every later
run is. The launcher sets that tree aside on the next run at that commit.

- Window `04fcba36…` both ways, 2,381 stood ports, unlowered `StopCheck` 1.
- 42,785 units checked, 0 mismatches, 0 undeclared, 0 rows two pieces hold, 788 tables committed.
- `unchecked_unevaluated` is empty: 52 of 129 templates are not checked directly, and all 52 are evaluated.
- The select: 20 templates, 40 units checked, 0 mismatches; tokens 2639 and 569, each the twin's.
- The read: 1,024 of 1,408 units checked, 0 mismatches.
- Plants: the read plant gave 30 mismatches in 30 sampled planted units, and the select plant 1 in 2 (unit
  `[16512, 2639, 0]` expected, `[16479, 7438, 0]` got).
- 2,071.7 s, 11.42 GiB.

# Part 2: the join's final head `920585e4671`

`19a3c311bba193ed62e7e7365f0f55fa0ec6c7d2` merges `920585e4671` into the select branch: parents `8830d95095` and `920585e4671`,
tree `2344cc5f66ba`. `f2bd805cd886e15feaf25d5642259d96330f5055` takes the strict xfail off `test_joined_with_select`.
Every run below is at `f2bd805cd8`, `--cwd source`, nothing patched (`status.txt` empty), 32 GB asked.

## The six sets at 4 and 78 layers

`join6_census.py --ids 43`.

| | 4 layers (`r20261113-105451-05a2`) | 78 layers (`r20261113-105450-8415`) |
|---|---|---|
| digest | `04fcba3603a6c805…` | `2c9671f29867b9da…` |
| units | 5,190,089 | 124,207,397 |
| templates | 129 | 129 |
| wire patterns | 13,448 | 946,308 |
| unlowered | `StopCheck` 1 | `StopCheck` 1 |
| free peers | 17,573,248 | 297,641,344 |
| free prompt | 2,752 | 2,752 |
| free weights | 322,961,408 | 6,297,747,456 |

- Full digests: 4 layers
  `04fcba3603a6c80515b75d6a4376a22a0441b4e7f0796c338cb52e304eeaac45c031e37b2da6ec43351bcd66b33e0035e42c0120f11c9763b50d4fd7a5173bf3`
  (the part-1 window with the hunk applied); 78 layers
  `2c9671f29867b9da1f56e5a13b38ae5dd6f33ed53ef9d7c90069537cac0f4c00233b5cfac031be01fb53ca3711dc2dd5391461f7374caacd74456563a805f1a5`.
- At both: `check_wires` passed and `check_proof_block` refused none of the 129 templates. No row is free of `TokenSelect`;
  the decode's 32 reads take `tok` by one wire from the combine `o0` at 1,376; the prefill's reads are row 0 of the `SIZE=192`
  combine's `parts`.
- The 4-layer five-set control is `544b577a749a2da7…`, 5,190,049 units.
- Programs `7f36f17ae005fc05…` (319 nodes) and `3c99109308321b3b…` (6,979 nodes). The 78-layer build took 969 s at a
  14.45 GB peak.

## Tests, suites and the gate

- `r20261113-105450-b7ff`: the seven node-window files, `-rxXs`, no `--runxfail`: 63 passed, none xfailed, none skipped;
  3:55, 3.5 GB peak.
- `r20261113-105450-216d`, against `origin/main` `053b8e50fe8a`:
  - `ratchet_gate.py origin/main HEAD --json` gave `failures: []`;
  - `suites.py verity-vllm --fresh -- tests/lint` gave 89 passed, 1 skipped;
  - `suites.py repository --fresh` gave 359 passed.
- The join branch was pushed from `920585e4671` to `f2bd805cd8` as a fast-forward at 11:16Z (3:16 AM PST), while the value
  check was still running.

## mypy

`216d`, one file at a time under `[tool.mypy]`, mypy 1.20.2:
- `test_node_window_select.py`: no issues.
- `node_window_select.py`: 1 error, line 130, `arg-type`. `got` keeps its `_Free` in its type on the branch where it has
  none.

Neither file exists at `95784be3d25b`. The lint suite passed with the error because `test_the_list_matches_or_warns`
turns a new error into a warning. The cure, typed rows on that branch with the same values, is `392fa75023b` on the select
branch; it is not on the join branch.

MYPY_RUN

## The value check

Run `r20261113-105451-a1b3` (`f2bd805cd8`, nothing patched): `values_join6.py --ids 43 --layers 4`, seed 274, at most
1,024 units a template. It uses the new names (`made_up(fn, prompt, rng)`, `sample_key`, `sample`).

- Window `04fcba36…` both ways, 129 templates, 5,190,089 units, 2,381 stood ports, unlowered `StopCheck` 1.
- Units: 42,737 checked, 0 mismatches, 0 undeclared, 0 rows two pieces hold, 788 tables committed. Part 1 checked
  42,785 under the old sampling.
- 52 of 129 templates are not checked directly, and `unchecked_unevaluated` is empty.
- The select: 20 templates, 40 units checked, 0 mismatches; tokens 2639 and 569, each the twin's and `TokenSelect_v4`'s.
- The read: 1,024 of 1,408 units checked, 0 mismatches.
- EMBED's read plant (block 1's table over block 0's) gave 33 mismatches in 33 sampled planted units, none with undeclared
  inputs.
- The select plant (chunk 0's logits wire moved to the row beside its best lane's) gave 1 mismatch in 2: unit 0 expected
  `[16512, 2639, 0]` and got `[16479, 7438, 0]`.
- 2,095.5 s, 11.42 GiB peak.
