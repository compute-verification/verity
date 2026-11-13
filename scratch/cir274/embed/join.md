# CIR-274 join: the embedding shard into the prompt's all-reduce

Join branch `cursor/circuits-cir274-join-6f04`: created at `cb65dbaa73793dddc6811bfe419100c53c89a0ad`, merged
`cd2c09ea9c22de67244bb1a0fc3fefd37aa12ece` (`3d756dac43e8fa6a570fd84d306f6918e9c06788`), then:

- `ff26be85a3b1e1aa76973388f77fb6519d7d2683`: `table_words`' array annotated (mypy, imports skipped).
- `9bcdba92eaeaf72cad8db01557794dfc28a920bc`: `node_window_values.Tables` commits `/b<b>/vocab` tables; its test.
- `95784be3d25bb851cf63c51e1d94d7e9813a5dc3`: `test_node_window_join.py`.

## The blocker and the hunk (not applied on the join branch)

`grain.lower` takes the embedding shard's producer at coordinate grain, K = chunk(6144) = 96. `_coordinate` then cuts
rank 0's partial into 96-word chunks, and the shard's 192-word rows cross them:

    Unexpressible: a writer row of 192 words crosses a 96-word chunk of AllReduceMultimem_v1{WORLD=8,SIZE=264192,...}

`multimem-join.diff` (sha256 `df8932430747de412a85ba22f41c820cc96526d95dae604b567293510c62df42`) takes such a
site, one whose rank-0 rows are wider than a chunk, through `_row` at the site's token row (`low.row`): one
`AllReduceMultimem_v1{SIZE=192}` unit per 192-word block, rank 0's block wired, the peers' free. A site whose partial is
free or stood keeps `_coordinate`, so the 4-set window is unchanged (run B1: digest `3ee9420daaf7545f...`, 5,191,393
units, 108 templates, as REQBLOCK's).

## Runs (join head 95784be3, the hunk applied in the run's clone for B1 and B2)

- A `r20261113-091550-ec0f`: the five test files, without the hunk.
- B1 `r20261113-091616-0dbd`: census at 4 and 5 sets, 43 ids, 4 layers; the join test with `--runxfail`; the three
  test files with the hunk.
- B2 `r20261113-091646-4d44`: `values_run_join.py --ids 43 --layers 4`, with the read's plant.
