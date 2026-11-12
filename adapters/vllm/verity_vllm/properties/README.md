# verity_vllm.properties

Property harnesses over the integration: non-interference, the kernel census, the golden and holdout corpora, the
admission difftests and the exactness checks.

## Interface

- **Provides:** the commands `verity-vllm census`, `golden`, `holdout`, `noninterference` and `properties-admission`,
  and the difftest adapters that `properties-admission --adapter` loads by name (`difftests`).
- **Loaded by:** the integration's `check`, `observe` and `pipeline` packages. Otherwise it is run, as those commands and
  by the GPU tests.
- **Loads:** `verity.circuits` and `verity.definitions`; the integration's `observe`, `config`, `settings`,
  `engine`, `commit`, `acquire`, `check`, `query` and `program`; and torch and vLLM inside functions.
