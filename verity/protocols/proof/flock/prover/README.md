# Flock's prover (`verity-flock`)

Flock proves a committed relation, the commitment's hash leaves included, in one binary-field proof with upstream Flock
(`succinctlabs/flock` at `b684b12`, patched here). The proof layer's README (`../../README.md`) is the architecture.

- `python/verity_flock/`: the lowering of a proof block to a Flock statement (`circuit.py`, CLI `flock-circuit`,
  statements `verity/flock-circuit` and `verity/flock-tables`), the proof server (`proof_server.py`), and the recursion
  provers (`rec_*.py`), kept until they go.
- `live/`: the Rust crate `flock-live`, dropped into a Flock checkout as `crates/flock-live`: the prover and verifier
  binaries and the live-coin session; `live/PROTOCOL.md` says what each side of the ZK server computes.
- `*.patch`, `cuda_*_patch.py`, `cuda/`: what the pod scripts apply to the upstream checkout, for the GPU architecture
  `SM` names (90 or 120; `GPU=0` builds for the CPU).
- `pod/`: the pod scripts, run through `research run`; `74-gemm-hill.sh`'s header and `benchmarks/proof_throughput/gemm_hill.py` are the GPU session procedure.
- `agreement/`: the Lean verifier held to upstream's Rust verifier.

Python returns no accept of its own: each verdict is the Lean verifier's (`../verifier/`, `verity-verify-proof`), and a
Python path that once answered is a difftest's reference (`_twin`) or raises `NotImplementedError` naming its goal.

~~~sh
uv run pytest tests/verity/protocols/proof/flock/prover
~~~
