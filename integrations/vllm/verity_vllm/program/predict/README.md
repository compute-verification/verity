# verity_vllm.program.predict

A Build's Programs predicted ahead of time from a row's facts, without tracing, and what is built on the prediction. The
package's docstring lists every module.

## Interface

- **Provides:** the predicted Program (`build`) and its score against the traced one (`score`); a row's ahead-of-time
  Build (`aot`, `compiled`); `steps`, which compiles a served run's advice into the Programs its committed rows must pass;
  a served window as its class's one Circuit (`window_circuit`); and the commands `verity-vllm predict` and
  `verity-vllm block-counts`.
- **Loaded by:** `benchmarks/prove_path`, `benchmarks/compile_cost`, `benchmarks/one_stage`, `benchmarks/route_capture`,
  `benchmarks/spot_proved`, `integrations/vllm/tools` and `circuit-check`. No module of the Build, the Match or the
  Commit imports it.
- **Loads:** `verity.ml.circuits`, `verity.ml.definitions`, `verity.ml.windows` and `verity.ml.commitments.gates`; the
  certifier's and SPOT's modules under `verity.core.protocols` and `verity.ml.protocols`; and the integration's
  `program`, `pipeline`, `engine`, `acquire`, `check` and `records`.
