# verity_vllm.properties

Property harnesses over the integration: the admission difftests, the quarantine lint and the kernel allowlist. The
tap exactness checks are test code, in `tests/properties/` beside the GPU scripts that run them.

## Interface

- **Provides:** the command `verity-vllm properties-admission` and the difftest adapters its `--adapter` loads by name
  (`difftests`), `quarantine_lint` and `kernel_allowlist`.
- **Loaded by:** the integration's `check`, `observe` and `pipeline` packages. Otherwise it is run, as those commands and
  by the GPU tests.
- **Loads:** `verity.circuits` and `verity.definitions`; the integration's `records`, `observe`, `config`, `settings`,
  `engine`, `commit`, `check` and `program`; and torch and vLLM inside functions.
