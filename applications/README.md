# Applications

An application is one file: a configuration (`verity/config/v0`, `verity.config`) that `verity run` reads. An application
holds no code and is never tuned; the table it runs from is `tools/verity/cli/verity_cli/served_table.py`, which holds no
copy of it, and `tools/verity/cli/verity_cli/configurations.toml` holds each file's hash. `verity run NAME` runs `applications/NAME.toml`. Each file's machines are
names the deploy bindings bind (`tools/verity/cli/README.md`): `prover-node` holds the developer's parties (workers, the
ZK server, the prover), `gate-host` the challenger and the network attester, `auditor-host` the memory attester, the
bandwidth verifier and the verifier, and remote's `auditor-remote` the auditor's parties off the site.

| file | what it shows | command | how far it runs and is proved |
| --- | --- | --- | --- |
| `remote.toml` | remote (`verity/remote/v1`): SPOT, PoUW's Pearl+ run verdict and the inference-only policy on the served execution phase, with no auditor's device at the site | `verity run applications/remote.toml --on MACHINE [--bindings FILE]` | `verity run --dry-run` resolves every party of the file, and needs no bindings: unbound, SPOT's and Pearl+'s coins are flipped on the workload's machine, a development run, which the verifier's line says (`coins_on`). The workload it stands for (`adapters/vllm/tools/served_run/`, H1's development executable): run `moved`, proof `none`; no run of record of the remote verdict has passed, since PoUW's verdict there is a development run that credits nothing, and the remote protocol has no guarantee in Lean (`verity/protocols/run/verifier/remote/README.md`, "The open gap") |
| `onsite_full.toml` | onsite (`verity/onsite/v1`): SPOT, PoUW's Pearl+ run verdict, the inference-only policy, the network attester and the memory wipe, PoUS at 320 challenges and PoUB, on the BF16 workload's execution phase (Llama 3.2 1B at 1 slot and 1 window step, the input dataset `llama32_1b_fixed_1_1_1` at once); the memory attester's keys for PoUS and the wipe pinned | `verity run applications/onsite_full.toml --on MACHINE --bindings FILE` | run `moved`, proof `none`: `verity run --dry-run` resolves it without a GPU, and it has no run of record. Pearl+'s piece is its own verdict over its own passes, held to no window record, a development piece that credits nothing; the network attester's piece is then whole, on its front's own window |
