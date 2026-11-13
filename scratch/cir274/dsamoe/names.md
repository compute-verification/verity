# Names: the window's expert tables against `expert_tables.tables_of`

**Verdict: equal.** 756 table parameter names on each side, the same set, and each name has the same row width and row
count on both sides.

## The comparison

The Build is GLM-5.3 at TP8, rank 0, 4 layers (one MoE layer, layer 3), eager, at 43 prompt ids and 2 outputs. It is
`benchmarks/compile_cost/run.py`'s `registered(0, 4, "eager")`, with request Program `7f36f17ae005…`.

**Ours.** The `IF.Table`s that `node_window_sparse._table_reads` declares for the experts, lowered by
`NodeWindow(...).lower()`. For each table: its name (= its `param`), its `words` and its `rows`.

**Theirs.** `expert_tables.tables_of(fields, {"w13": 1, "w2": 32})`, over the Build's weight fields. Each table's row
width and row count come from `expert_rows.table_words` on a zero stack of the field's shape. The fields are read two
ways, and both give the same 113 fields:
- **`record_fields`.** The weights record as `window_weights._fields` reads it, taken off the request Program's `weights`
  struct.
- **`program_dir_params`.** `weights_of_record.program_weight_fields` over `[["weights", str(type)]]`, the Program dir's
  `instances["params"]` that `expert_tables.register(program_dir, ...)` reads.

I didn't read the record through `window_weights._fields(reg)` itself. It builds the class Program, which needs a
`sweep` that this synthetic Build doesn't have (`KeyError: 'sweep'`, run `r20261113-041114-3cb4`).

| expert field | shape | bits | tables: ours | tables: theirs | row width (words): ours / theirs | rows: ours / theirs |
|---|---|---|---|---|---|---|
| `model.layers.3.mlp.experts.routed_experts.w13_weight` | (256, 512, 6144) | 8 | 512 | 512 | 3,072 / 3,072 | 256 / 256 |
| `….w13_weight_scale_inv` | (256, 4, 48) | 32 | 4 | 4 | 96 / 96 | 256 / 256 |
| `….w2_weight` | (256, 6144, 256) | 8 | 192 | 192 | 4,096 / 4,096 | 256 / 256 |
| `….w2_weight_scale_inv` | (256, 48, 2) | 32 | 48 | 48 | 4 / 4 | 256 / 256 |
| **total** | | | **756** | **756** | | |

- **Names:** of the form `weights/model.layers.3.mlp.experts.routed_experts.<field>@<H>/b<block>/table`.
  - Only ours: none.
  - Only theirs: none.
  - Width or rows differing: none.
- **Each table's registered parameter size** (`w.b.params[param]`) is words × rows on every one of the 756
  (`leaves_check` true).

## A registration made `served=`

**Not run.** A capture's `params.json` is in neither the tree nor the store.
- **The tree:** no tracked file is named `params.json`. The served pins in `adapters/vllm/verity_vllm/pins/served/` are
  for Llama 3.2 1B, Qwen3 8B and Qwen3 30B-A3B, not GLM-5.3. The writer of a capture's `params.json` is
  `benchmarks/spot_proved/capture.py`.
- **The store:** I searched every artifact's manifest. 11 name a `params.json`, all run records, and none is a capture's:
  - the network-certifier dry run's four cases;
  - the certifier period runs;
  - an 81-byte `day-params.json` input.

  No vLLM build, row-dir or match-record artifact names one.

## Runs

Every run is from source `30dac672d2bb682dd3a81c80bcf4915f2577fd2e`, the joint branch's head.
- **The comparison:** `r20261113-042918-6435` (`p3_names.py`, `run_names_only.sh`), output `names_l4.json`.
- **The store searches:** `r20261113-041114-3cb4` (end of its log), `r20261113-041701-6f10`, `r20261113-041932-e876` and
  `r20261113-042207-cc7d` (`store_params*.sh`).
