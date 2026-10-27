# tp2 fixtures

Small, committed inputs for `tests/engine/collectives/test_tp_collective.py`, all derived from ONE captured run, `tp2_b1`
(Qwen2.5-1.5B, `tensor_parallel_size=2`, eager, bf16, batch-invariant, PyNccl, custom all-reduce off).

| file | what | from |
|---|---|---|
| `collective_shard.steps03.rank{0,1}.json.gz` | per-rank collective launch shards (`verity-capture/tp2-collective-shard/v1`) for engine steps 0 (prefill, 32 rows) and 3 (decode): 118 rows per rank = 2 × (57 all_reduce + 1 all_gather + 1 vocab_shard_lookup); operand touches capped at 2, result consumers at 3 | `python -m tests.engine.collectives.collective_link extract log.rank{r}.jsonl.gz --steps 0,3 --max-touches 2 --max-consumers 3 --compact` over the run's per-rank raw logs (`runs/tp2_b1/capture/`, 590 KB gz each, not committed) |
| `fold_gap.shard.rank{0,1}.json` | the run's fold-gap report per rank (diagnostic shard profile): 944 Unsupported, 0 instances, first insufficiency point = step 0 event 92 reading `a384` written by event 88 | the run's evidence (copied) |
| `instances.shard.rank0.jsonl` | the fold's instances file of that fold: `n_instances 0` (the empty Program) | idem |
| `tp2_body.summary.json` | wave-1 body checker over the snapshotted collectives: 114 all-reduce `bf16(sum_r f32(shard_r))` ok, 2 all-gather concat ok, both ranks' outputs equal | idem |

The shards are gzipped only for size (308 KB → 12 KB each); `collective_link.load_shard` reads both forms.
