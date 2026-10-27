"""Ahead-of-time prediction of the Build's Programs from a row's declared facts φ, without tracing, and its score against the traces.

`phi` (φ: checkpoint config, target, TP, shape, sampler, execution), `build` (BuildProgram(φ): the `Derived[vLLM*]` Program the
Build would derive, from a hand-written vocabulary, with the Llama-family decoder), `dsa` and `dsv4` (the family decoders it
dispatches to by `FAMILY_FACTS[...]["arch"]`), `node` (a tensor-parallel node's Program: its ranks' forwards in one body, joined at
their collectives, with no peer Input), `common` (what every decoder writes with), `inventory` (the traced rows and their
Builds), `score` (predicted vs traced, per row and granularity: digest, equivalence verdict, first divergence), `aot` (the
comparison that switches a row, and a switched row's AOT Build), `compiled` (the served-compiled Build from a model's config and
its pins), `cli` (`verity-vllm predict`), `domain` (what of φ a predicted Program can't be the prediction of), `steps` (I2,
the compiler from a served run's advice to the Programs its committed rows must pass) with `batch` (its batch invariance),
`window_circuit` (a served window as its class's one Circuit) with `window_builder` (its proof blocks) and `window_weights`
(its registered weights), and `node_registration` (θ of a node's Program, registered from its TP Commit of record).

The record path reaches this package only through a switched row (its `aot` in `manifests/records.json`): its Build's derives are
`predict emit` (`aot`), each refused unless the predicted Program is the one its comparison's trace holds.  No module of the Build,
Match or Commit imports it, and the trace that checks it forbids it as an input (`program.frontend.inputs_trace`).  The served
Commit binds a row's Build of record as predict writes it (`aot.write_row`, ``predict row``), through its record
(`records.predicted_row`, `pipeline.serve_seal.row_refusal`), and a traced Build only cross-checks it (``predict compare
--predicted``).  A served Commit may take `compiled`'s Build as its Build of record: its treatment rebuilds the Build from its
own compile artifact, and `compiled verify` holds the two equal.
"""
