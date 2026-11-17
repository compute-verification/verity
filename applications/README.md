# Applications

An application is one file: a configuration (`verity/config/v0`, `verity.config`) that `verity run` reads. An application
holds no code and is never tuned; the table it runs from is `tools/verity/cli/verity_cli/served_table.py`, which holds each
file byte for byte (`tests/tools/verity/cli/test_onsite_served.py`), and `tools/verity/cli/verity_cli/configurations.toml`
holds each file's hash. `verity demo NAME` is `verity run applications/NAME.toml`; with no NAME it runs `onsite_compute`. Each file's machines are
names the deploy bindings bind (`tools/verity/cli/README.md`): `prover-node` holds the developer's parties (workers, the
ZK server, the prover), `gate-host` the challenger and the network attester, `auditor-host` the memory attester, the
bandwidth verifier and the verifier, and remote's `auditor-remote` the auditor's parties off the site.

| file | what it shows | command | how far it runs and is proved |
| --- | --- | --- | --- |
| `remote.toml` | remote (`verity/remote/v1`): SPOT, PoUW's Pearl+ run verdict and the inference-only policy on the served execution phase, with no auditor's device at the site | `verity run applications/remote.toml --on MACHINE --bindings FILE` | not accepted by `verity run` yet: it refuses the file by name (`no-party`), since Pearl+ has no piece. The served run it stands for (`adapters/vllm/tools/served_run/`, H1's development executable): run `moved`, proof `none`; no run of record of the remote verdict has passed, since PoUW's verdict there is a development run that credits nothing, and the remote protocol has no guarantee in Lean (`verity/protocols/compose/verifier/remote/README.md`, "The open gap") |
| `onsite_network.toml` | the network attester, SPOT and the memory wipe (`verity/onsite-network/v0`, a configuration of its own, never onsite) | `verity run applications/onsite_network.toml --on MACHINE --bindings FILE` | not accepted by `verity run` yet: it stops with `NotImplementedError` naming the goal demos/onsite-demo; proof `none` |
| `onsite_compute.toml` | the default: those three, PoUW and the inference-only policy (`verity/onsite-compute/v0`, a configuration of its own, never onsite) | `verity run applications/onsite_compute.toml --on MACHINE --bindings FILE` | not accepted by `verity run` yet: it stops with `NotImplementedError` naming the goal demos/onsite-demo; proof `none` |
| `onsite_full.toml` | onsite (`verity/onsite/v0`): those five, PoUS at 320 challenges and PoUB, on the served int8 execution phase; the memory attester's keys for PoUS and the wipe pinned | `verity run applications/onsite_full.toml --on MACHINE --bindings FILE` | run `moved`, proof `none`: `verity run --dry-run` resolves it without a GPU, and it has no run of record yet |
