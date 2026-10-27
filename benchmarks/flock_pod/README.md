# C-Flock's pod scripts

The jobs that build C-Flock (`verity/ml/flock/`) on a node and measure or exercise it there: link and patch an upstream Flock
checkout (`10-link.sh`, `20-gpu-link.sh`), prove and verify its benchmark relations (`20-pure.sh`, `30-cell.sh`,
`60-circuit.sh`), sweep the unit shapes of a served row (`70-class-sweep.sh`), hill-climb the GEMM statement
(`74-gemm-hill.sh`, `gemm_hill.py`), stage and price the recursive session (`80-rec-inner.sh`, `85-rec-reprice.sh`,
`rec_inner.py`, `rec_reprice.py`), and record the sessions whose verdicts C-Flock's Lean tests replay (`89-served-registered.sh`,
`91-plain-hidden.sh`, `92-zk-sessions.sh`). Each script states its settings, taken as `K=V` arguments, at its top.

They run on a node through `research run`, from the repository root (`--cwd source`, or `--cwd clone` where a script's header
says so). `tool.py` declares the `flock_pure` and `flock_class_sweep` tools
(`research run --tool flock_pure -- bash benchmarks/flock_pod/20-pure.sh ...`), and `agreement_tool.py` the `flock_agreement`
job, which runs `verity/ml/flock/verifier/ci-pod.sh`. A script writes its results under the run's `out/`
(`$RESEARCH_RUN_DIR/out`), which `research run` publishes. C-Flock's tests (`verity/ml/flock/tests/`) load the Python here by
path.
