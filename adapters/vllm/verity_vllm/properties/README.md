# verity_vllm.properties

Property harnesses over the integration: non-interference, the kernel census, the golden and holdout corpora and the
admission difftests. The tap exactness checks are test code, in `tests/properties/` beside the GPU scripts that
run them.

## Interface

- **Provides:** the commands `verity-vllm census`, `golden`, `holdout`, `noninterference` and `properties-admission`,
  and the difftest adapters that `properties-admission --adapter` loads by name (`difftests`).
- **Loaded by:** the integration's `check`, `observe` and `pipeline` packages. Otherwise it is run, as those commands and
  by the GPU tests.
- **Loads:** `verity.circuits` and `verity.definitions`; the integration's `records`, `observe`, `config`, `settings`,
  `engine`, `commit`, `check` and `program`; and torch and vLLM inside functions.
