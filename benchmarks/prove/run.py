"""`verity bench prove`: what a proof system's prover costs on a workload, `verity-bench prove` (its README beside this
file).

    verity bench prove --system flock --workload glm53               every block proved: a row a block type, the totals
    verity bench prove --system spot --workload glm53 [--sampling ART]
                                                                     Flock at the workload's sampling plan of record
    verity bench prove --system flock --workload glm53 --provers 2   each type priced at two provers (default: one)
    verity bench prove --system flock --workload glm53 --cores 24    each type priced on a host of 24 CPU cores a card
                                                                     (default: 10)
    verity bench prove --system flock --workload glm53 --rows 12     the 12 costliest block types, the rest in one row
    verity bench prove --system flock --workload glm53 --json        the table as JSON
    verity bench prove lean-verify --lean VERIFIER --proof TREE      the Lean verifier timed on one kept proof, as the
                                                                     record the table's Lean verifier figures read
    verity bench prove layers --flock DIR|RUN --zk DIR|RUN --sampling DIR|RUN --recursion DIR|RUN
                                                                     the proof benchmark by layer, the benchmark result
                                                                     for each layer read from its run's output
                                                                     (`verity-bench prove-layers`)

Under a research run, `--system spot` also writes `METRICS` into the run's result (`prove_records.record`), and `layers`
writes `LAYER_METRICS`.
"""

from __future__ import annotations

import sys

from verity_bench.cli import main

#: the metrics `benchmarks/prove/metrics.toml` holds this benchmark to, by id: each the figure its value names
#: (`prove_records.FIGURES`), a total of SPOT's table at the workload's sampling plan of record, or the prover GPUs a
#: serving GPU at the exfiltration run of record's design (`prove_records.AT_RECORD`)
METRICS = {"benchmarks/prover-gpus-per-serving-gpu": "gpus_per_serving_gpu",
           "benchmarks/prover-gpu-seconds-per-token": "gpu_seconds_per_token",
           "benchmarks/lean-verifier-seconds-per-token": "lean_verifier_seconds_per_token",
           "benchmarks/proof-bytes-per-token": "proof_bytes_per_token",
           "benchmarks/prover-gpus-per-serving-gpu-at-record": "gpus_per_serving_gpu_at_record"}
#: the metrics `layers` writes, by id: each the figure a proof block it holds (`proof_layers.FIGURES`), one member a
#: layer (`proof_layers.MEMBERS`)
LAYER_METRICS = {"benchmarks/proof-layer-prover-seconds": "prover_s",
                 "benchmarks/proof-layer-lean-seconds": "lean_s",
                 "benchmarks/proof-layer-proof-bytes": "proof_bytes"}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["lean-verify"]:
        sys.exit(main(argv))
    if argv[:1] == ["layers"]:
        sys.exit(main(["prove-layers", *argv[1:]], metrics=LAYER_METRICS))
    sys.exit(main(["prove", *argv], metrics=METRICS))
