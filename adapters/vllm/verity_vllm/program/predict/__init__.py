"""Ahead-of-time prediction of the Build's Programs from a row's declared facts φ, without tracing, and its comparison with the traces.

`phi` (φ: checkpoint config, target, TP, shape, sampler, execution), `build` (BuildProgram(φ): the `Derived[vLLM*]` Program the
Build would derive, from a hand-written vocabulary, with the Llama-family decoder), `model_shape` (the shape facts of a
checkpoint's config it builds from, and its weights record), `dsa` and `dsv4` (the family decoders it
dispatches to by `FAMILY_FACTS[...]["arch"]`), `node` (a tensor-parallel node's Program: its ranks' forwards in one body, joined at
their collectives, with no peer Input), `common` (what every decoder writes with), `naming` (what a request Program's Build
says about it beyond its digest, written down from φ: the audit's names without a trace), `score` (φ of a row from its declared
inputs, the predicted workload Program and the predictor's code version), `aot` (the
comparison that switches a row, and a switched row's AOT Build), `compiled` (the served-compiled Build from a model's config and
its pins), `cli` (`verity-vllm predict`), `domain` (what of φ a predicted Program can't be the prediction of), `steps` (I2,
the compiler from a served run's advice to the Programs its committed rows must pass) with `batch` (its batch invariance),
`segment` (a class-G request's history and launches in a window), `reads` (every field it reads, with its account), `hits` (a
prefix-cache hit's sources as it reads them), `ingress_policy` (a request's certified body as it reads it), `traced` (the Build
traces it holds its compiled digests to), `tp_steps` (a node's advice compiled at every rank and composed into one TP workload)
and `template_counts` (each template's instances in a `dsa` request Program, counted without building it),
`window_circuit` (a served window as its class's one Circuit) with `window_class` (the class it is of, and H1's tables as it
reads them), `window_builder` (its proof blocks), `window_rows` (its units given their output rows from a served window's
values), `scan_state` (the network certifier's scan state as a class's Programs read it), `inference_blocks` (its
partition into inference blocks, a private table) and `window_weights` (its registered weights), `block_counts`
(`verity-vllm block-counts`: a day's inference blocks counted from its windows' circuit records and judged against the day's
declared numbers), `node_registration` (θ of a node's Program, registered from its TP Commit of record),
and `step_needs` (what `dsa` calls for one served engine step at its launch, each refusal counted in its place).

The record path reaches this package only through a switched row (its `aot` in `manifests/records.json`): its Build's derives are
`predict emit` (`aot`), each refused unless the predicted Program is the one its comparison's trace holds.  No module of the Build,
Match or Commit imports it, and the trace that checks it forbids it as an input (`program.frontend.inputs_trace`).  The served
Commit binds a row's Build of record as predict writes it (`aot.write_row`, ``predict row``), through its record
(`records.predicted_row`, `pipeline.serve_row.row_refusal`), and a traced Build only cross-checks it (``predict compare
--predicted``).  A served Commit may take `compiled`'s Build as its Build of record: its treatment rebuilds the Build from its
own compile artifact, and `compiled verify` holds the two equal.
"""
