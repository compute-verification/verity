# C-Flock (`verity-flock`)

The active backend family. C-Flock proves a committed relation, the commitment's hash leaves included, in one
binary-field proof with upstream Flock (`succinctlabs/flock` at `b684b12`, patched here). Where the frozen families are
and what the merge gate runs per family: `../README.md`.

## Layout

- `python/verity_flock/`: the Python package.
  - `backend.py`: the configured backends over the GEMM-coordinate subcircuits (`BACKENDS`: `FLOCK_FRAME_V3_BLAKE3`,
    `FLOCK_BLOCK`, `FLOCK_VLLM_V1`); `lowering.py` and `unit.py`: the pinned lowering of one tensor-core transition unit.
  - `ir_lower.py`: the generic IR lowering, an IR Definition composed from the per-op circuit pieces of `fp.py` (over
    `gf2.py`); `tail_pieces.py`: circuits for the primitives a template's units leave outside.
  - `templates/`: one lowering per subcircuit template; `ir_frame.py` (`verity/flock-ir-frame/v3`) and `ir_sampling.py`
    (`verity/flock-ir-sampling/v1`): their frame-v3 statements.
  - `partition_units.py`: every unit class of an arbitrary partition (`verity.ir.units`) lowered and checked against the
    reference evaluator; `boolean_export.py`: a served row's leaf templates as Boolean circuits (a derived format, not a
    statement); `circuit_types.py` (`verity/circuit-type/v1`): each distinct sub-circuit once, the object the Boolean
    export, the docs site and the prover's layouts share.
  - `bench.py`, `ir_bench.py`, `negatives.py`, `register.py`: the benchmark sweeps, statement negatives and cell
    registration.
- `live/`: the Rust crate `flock-live`, which the pod scripts drop into a Flock `b684b12` checkout as
  `crates/flock-live`: the prover and verifier binaries (`flock-pure`, `flock-ir-frame`, `flock-ir-sampling`,
  `flock-ir-block`, `flock-vllm-v1`, …) and the live-coin session.
- `*.patch`, `cuda_*_patch.py`, `cuda/`: what the pod scripts apply to the upstream checkout.
- `pod/`: the pod scripts, run through `research run` (`tool.py` declares the `flock_pure` tool).
- `verifier/`: the verifier of record: the specification `PROTOCOL.md`, the executable Lean verifier (`verifier/lean/`,
  `flock-verify`), its Lean proofs (`lean/level3/`, `lean/soundness/`) and the agreement scripts. See its README.

C-Flock's circuit prover (M0: the module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit`)
is on draft PR #83, not on `main`. `main` holds its consumers: the Lean verifier reads its format, serving writes rows in
it (`verity_vllm.commit.serving_rows`), and `benchmarks/one_stage` runs it from PR #83's tree. C-Flock ZK (M1/M2) is on
draft PRs #123 and #138, stacked on #83.

## Tests

~~~sh
uv run pytest backends/flock/tests                          # 181 tests, in `check`; several minutes on 4 cores
FLOCK_IR_SLOW=1 uv run pytest backends/flock/tests/test_ir_lowering.py   # the five 2^19/2^21-row units
cd backends/flock/verifier/lean && lake build               # the Lean verifier; `check` builds and audits it
~~~

`tests/test_lean_verifier.py` skips its ten Lean-backed tests without `lake` on the PATH. `check` runs none of the
`live/` crate's Rust tests; they build only inside the upstream checkout.
