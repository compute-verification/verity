# verity_vllm.properties

The quarantine lint and the kernel allowlist. The tap exactness checks are test code, in `tests/properties/` beside the
GPU scripts that run them.

## Interface

- **Provides:** `quarantine_lint` (gate G4's purity and identity lint of `program/quarantine/`) and `kernel_allowlist`.
- **Loaded by:** the integration's `observe` package (the sm_89 eager profile reads `KERNEL_ALLOWLIST`) and tests.
- **Loads:** the integration's `config`, `program` and `records`.
