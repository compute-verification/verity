# verity_vllm.pipeline.tp

A run that several ranks serve together (tensor parallelism): the commands that capture it, commit it, match it and
replay it, the extension each rank's worker loads (`rank_worker`), and the wires of a window over every rank (`wires`).
`pipeline/row_tp.py` is the row that runs the commands in order.

## Interface

What the adapter's other packages and the benchmarks import from this package, module by module. The command line loads
six modules by name and takes `Options` and `main` from each: `capture` (`tp-capture`), `commit` (`tp-commit`),
`commit_replay` (`tp-commit-replay`), `fold_match` (`tp-fold-match`), `match` (`tp-match`) and `serving_rows`
(`tp-serving-rows`). The adapter's other packages read `TpRow` of `pipeline/row_tp.py` and `DEFER_WORLDS` of `c2` (the
row); `emit`, `finish` and `parse_ranks` of `emit_ranks` and the four names of `links` (the Build); `GROUP` of `grain`
and `request_program_dirs` of `rank_replay` (`program.predict`); and `load` of `rank_replay`, `Refused` of
`commit_replay`, which is the adapter's own refusal handed on, and `from_bundle`, `record` and `window_of` of
`serving_rows` (the advice's windows). The benchmarks read `parse_engine_args` of `capture`; `EXTENSION` of
`rank_worker`, the path the engine loads each rank's extension by; `engine` and `warmups` of `phases`; `apply`,
`as_json` and `require_ranks` of `target`; `cpu_delta` and `thread_cpu` of `trace`; `run` of `warmup`; `load` and
`manifest_of_rank` of `rank_replay`; and `BINDING`, `ROWS_SCHEME`, `record`, `source` and `window_of_record` of
`serving_rows`. Each of these modules declares its names in `__all__`, except `commit`, `serving_rows` and
`pipeline/row_tp.py`, which declare none yet. Two reads go around the declared names: a driver calls the extension's
methods by name over the engine's `collective_rpc`, as `benchmarks/served_commit/tp_decode.py` does for
`tp2_commit_install`, and `benchmarks/one_stage/a10.py` reads `commit_replay`'s private `_of_record`. The package's
other modules are read only inside it and by its tests.
