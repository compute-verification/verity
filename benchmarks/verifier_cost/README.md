# verifier_cost

A verifier of record's time against the size of the description it reads and against the size of the circuit that
description unrolls to. The rule it measures (AGENTS.md "Design rules", Daniel, 18 Oct): a verifier's cost is
O(circuit description), never O(unrolled circuit).

The workload is Serve_v1 (`benchmarks/network_sim/workload/served.py`) at a ladder of configs up to B0 (`Config.b0()`,
30 layers), word-level and on bits. On bits it is `lift.on_bits`: every Definition Serve_v1 reaches, lifted to bits all
the way down (`ir="boolean"` can't derive Serve_v1; `lift.py`'s docstring says why). Two verifiers are measured:

* `measure`, the NCI check: `nci-check` (`NciCheck.Desc`, one summary per Definition) on an instance whose θ is the
  registered inputs and whose IUs are the root's Calls, one each. Where the unrolled circuit is small enough
  (`--unrolled-max`), the instance expanded gate by gate also goes through `nci-check --unrolled` (`NciCheck.check`, the
  gate-by-gate reference), and the two verdicts must agree.
* `partition`, the one-stage verifier's partition: `sampled-proofs-difftest`'s `partition` (`Partition.units`, `strata`,
  the expanded law and the digest, as the verifier of record computes them) on Serve_v1 under the template query of its
  Calls (`Q_template_instances`), under `Q_word` v1 and `Q_call` v1 at `X = 16` where its Calls' distinct Definitions are
  small enough (`--unrolled-max`), and on the served shape: the population program of Serve_v1's largest Definition at
  1 to 10⁶ instances under `Q_template_instance`, as a served window registers one. A template query must apply.

~~~sh
verity bench verifier_cost measure --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
verity bench verifier_cost partition --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES]
research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone -- \
  bash benchmarks/verifier_cost/job.sh measure|partition --unrolled-max 40000000   # every config, on a node
verity bench verifier_cost store RUN                     # its results.json or partition.json to the store; prints the art: id
verity bench verifier_cost plot ART --to PNG             # from the store
~~~

`results.json` (`verity/verifier-cost/v1`) has one row per config and IR: the description's bytes, Definitions and body
nodes, the unrolled circuit's gates, and per check its wall seconds (the best of `--repeat`), peak RSS and the verdict.
`partition.json` (`verity/verifier-cost-partition/v1`) has one row per config, IR, shape and query: the description's
bytes, the unrolled gates, the gates of the Calls' distinct Definitions (what `Q_word` and `Q_call` lay out), the wall
seconds, peak RSS and the population (null where the verifier refuses the query). `store` puts either in the store as
an `evidence/v1` record naming the run and its commit; it runs where the store's keys are (a VM), since a node's shell
has none.

Nothing here is a served Program: `on_bits`'s output is this benchmark's construction, and nothing outside this directory
reads it.
