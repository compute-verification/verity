# verity_vllm.program.predict

A Build's Programs predicted ahead of time from a row's facts, without tracing, and what is built on the prediction. The
package's docstring lists every module.

## Interface

- **Provides:** the predicted Program (`build`) of a row's facts φ (`phi`; `score.phi_of` reads a row's from its
  declared inputs) and its comparison with the traced one (`aot`); a row's ahead-of-time Build (`aot`, `compiled`),
  with the names the audit reads written down from φ (`naming`); `steps`, which compiles a served run's advice into the
  Programs its committed rows must pass, with what one served engine step calls for (`step_needs`); a served window as
  its class's one Circuit (`window_circuit`, with `window_class`, `window_rows`, `window_weights` and `scan_state`); a
  day's inference blocks counted from its windows' circuit records (`block_counts`); and the commands
  `verity-vllm predict` and `verity-vllm block-counts`.
- **What others import:** code outside the integration imports the seven modules `tests/test_vllm_public_api.py` names
  public (`block_counts`, `naming`, `scan_state`, `step_needs`, `window_class`, `window_rows` and `window_weights`),
  from each only the names its `__all__` lists, and the seven more that test lists until their importers move:
  `batch`, `build`, `common`, `phi`, `score`, `steps` and `window_circuit`. Of these, `build`, `phi` and
  `window_circuit` list in `__all__` the names that library code and files outside the integration import from them;
  `batch`, `common`, `score` and `steps` state none yet. Besides the command line's two, library code reads two names
  as text, so they stay where they are: `steps.compile_steps`, the registered compiler's name that a run's advice is
  bound to (`pipeline.advice.COMPILER`), and the package's own, which `program.frontend.inputs_trace` resolves without
  importing it.
- **Loaded by:** `benchmarks/prove_path`, `benchmarks/inference_only`, `benchmarks/compile_cost`,
  `benchmarks/one_stage`, `benchmarks/route_capture`, `benchmarks/spot_proved`, `adapters/vllm/tools` and a test of
  `verity/definitions/requirement`; the integration's command line (`pipeline.cli`) loads `cli` and `block_counts` by
  name. No module of the Build, the Match or the Commit imports it.
- **Loads:** `verity.circuits`, `verity.definitions`, `verity.commitments.windows` and `verity.commitments`; the
  certifier's modules under `verity.protocols.certifier`, and SPOT's under `verity.protocols.spot.verifier`;
  and the integration's `program`, `config`, `pipeline`, `engine`, `check`, `records`, `observe`, `acquire`, `query`,
  `protocols`, `correspondence`, `target_family` and `errors`, with `commit` inside a function.
