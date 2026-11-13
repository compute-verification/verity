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

RESULT
