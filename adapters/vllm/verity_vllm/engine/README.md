# verity_vllm.engine

The pinned vLLM engine: its build, the environment and the switches it is served under, the facts of the models and
cards it serves, and the seams through which the rest of the integration reaches vLLM. The package's docstring says
what it holds.

## Interface

- **Provides:** to code outside the integration, the modules `tests/test_vllm_public_api.py` lists as public:
  engine specs (`spec`), a built engine's vLLM objects (`handle`), a stock engine at the pin (`stock`), a served
  request's seed (`served_seed`), the engine core's seam between steps (`core_gate`), the step trace
  (`step_trace`), the sampler's capture (`sampler_capture`), the carried pool's row (`pool_row`) and the MoE
  alignment (`moe_align`). To the integration's other packages also: the engine's profile and run facts
  (`engine_profile`, `run_facts`), its served switches (`served`), a request's own id through the engine
  (`engine_driver`), the one owner of runtime patches (`hooks`), the int8 kernels and the exact-int8 parts of every
  process that builds the model (`int8_exact`, `int8_ops`, `int8_attention`, `int8_experts`, `int8_ranks`), sparse
  MLA at one split and the top-p split at its registered count (`sparse_mla`, `topp_pin`), a served engine class's
  admission (`class_admission`), the collectives' sites and the fused norm as one collective
  (`collectives.collective_sites`, `collectives.fused_norm`), the static serving schedule (`static_schedule`), the
  two checkpoint makers' commands (`fp8_checkpoint`, `int8_checkpoint`), a workload's load and its requests' run
  through a built engine, at once or at declared steps (`vllm_adapter`, `arrivals`), the generated kernels'
  identity (`kernel_identity`), a checkpoint's modules on the meta device and a serving rank's (`meta_inventory`),
  the one wrap of the model runner's step and the parts that run around it (`runner_step`), and the facts of
  models, cards and transports (`profiles.cases`, `profiles.coverage_roles`, `profiles.dense_generic`,
  `profiles.devices`, `profiles.engine_classes`, `profiles.family_facts`, `profiles.transport`,
  `profiles.admission_calibration`). Each of these lists in `__all__` the names that other packages, the
  benchmarks and the command line read from it. Five more modules that other packages or the benchmarks read
  state no `__all__` yet: `build`, `env`, `int8_fused`, `int8_vllm` and `profiles.layer_classes`; `step_trace`'s
  does not yet list `SERVED` and `scheduler_config`, which other packages read; and
  `observe.fold.profiles.generic` reads one private name that no `__all__` lists,
  `profiles.dense_generic._sha256`.
- **Loaded by:** the integration's `pipeline`, `acquire`, `program`, `protocols`, `observe`, `properties`, `check`
  and `commit` packages, its `llm` and `glue`; and the benchmarks that build, serve or time the engine
  (`benchmarks/hash`, `benchmarks/pouw`, `benchmarks/serve_tax`, `benchmarks/torch_gpu`, `benchmarks/vllm_encoded`
  and a dozen more directories).
- **Loads:** vLLM, torch, triton, numpy and, in `meta_inventory`, `huggingface_hub`; the integration's `config`,
  `settings`, `errors`, `log`, `target_family`, `records.logits_capture`, `program.frontend` (the target profile,
  the kernel read-back, the export's compatibility) and, in `vllm_adapter`, `program.prefix_plan` and `commit`;
  `verity.definitions.registry`, `verity.definitions.commitments`, `verity.commitments`, `verity.circuits.codec`
  and `verity_config`.
