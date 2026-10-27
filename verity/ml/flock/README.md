# C-Flock (`verity-flock`)

The active backend family. C-Flock proves a committed relation, the commitment's hash leaves included, in one
binary-field proof with upstream Flock (`succinctlabs/flock` at `b684b12`, patched here).

## Layout

- `python/verity_flock/`: the Python package.
  - `backend.py`: the configured backends over the GEMM-coordinate subcircuits (`BACKENDS`: `FLOCK_FRAME_V3_BLAKE3`,
    `FLOCK_BLOCK`, `FLOCK_VLLM_V1`); `lowering.py` and `unit.py`: the pinned lowering of one tensor-core transition unit.
  - `ir_lower.py`: the generic IR lowering, an IR Definition composed from the per-op circuit pieces of `fp.py` (over
    `gf2.py`); `tail_pieces.py`: circuits for the primitives a template's units leave outside.
  - `templates/`: one lowering per subcircuit template; `ir_frame.py` (`verity/flock-ir-frame/v3`) and `ir_sampling.py`
    (`verity/flock-ir-sampling/v1`): their frame-v3 statements.
  - `partition_units.py`: every unit class of an arbitrary partition (`verity.ml.circuits.units`) lowered and checked against the
    reference evaluator; `boolean_export.py`: a served workload's leaf templates, or one census subcircuit (`subcircuit`, the catalog's blob), as Boolean
    circuits (a derived format, not a statement); `circuit_types.py` (`verity/circuit-type/v1`): each distinct sub-circuit once, the object the Boolean
    export, the docs site and the prover's layouts share.
  - M0: `circuit.py` (`verity/flock-circuit`), a lowered template proved with its serving rows committed in the circuit,
    the rows' SHA-512 and the template's tail in the circuit too (`sha512_circuit.py`, `tail.py`), or with `layout=MODE`
    the templated statement (`verity/flock-circuit/tmpl`: the unit as `tmpl.py`'s template, its slot from `tmpl.layout`,
    Lean's `Tmpl.layoutM`), which the verifier refuses until its proof lands; `tables.py`
    (`verity/flock-tables`), several tables in one commitment with private glue; `class_statement.py`, a unit class of
    any partition as an M0 statement (`UnitStager(..., layout=MODE, structural=True)`: the unit as `struct_lower.py`'s
    template, the flat unit never built, its outputs from the template, `tmpl.evaluate`), and `class_sweep.py`, the
    backend sweep's work lists; `circuit_bench.py`,
    `flock-circuit`'s sweep.
  - `bench.py`, `ir_bench.py`, `negatives.py`, `register.py`: the benchmark sweeps, statement negatives and cell
    registration.
- `live/`: the Rust crate `flock-live`, which the pod scripts drop into a Flock `b684b12` checkout as
  `crates/flock-live`: the prover and verifier binaries (`flock-circuit`, `flock-pure`, `flock-ir-frame`,
  `flock-ir-sampling`, `flock-ir-block`, `flock-vllm-v1`, …) and the live-coin session. Under `--zk` with
  `FC_FIREWALL=1`, `flock-circuit` is the prover workers' side of the ZK server, which is Lean
  (`../../core/service/flock/Flock/Firewall`, `flock-firewall`, and its release side `../../core/service/firewall/`):
  `live/PROTOCOL.md` §10.4 says what each side computes.
- `*.patch`, `cuda_*_patch.py`, `cuda/`: what the pod scripts apply to the upstream checkout.
- `pod/`: the pod scripts, run through `research run` (`tool.py` declares the `flock_pure` tool).
- `verifier/`: the verifier of record: the specification `PROTOCOL.md`, the executable Lean verifier (`../../core/service/flock/`,
  `flock-verify`) and the agreement scripts; its Lean proofs are under `Security/Proofs/Flock/` (`Level3/`,
  `Soundness/`). See its README.

C-Flock's circuit prover is M0 in the Layout above (module `verity_flock.circuit`, CLI `flock-circuit`, statements
`verity/flock-circuit` and `verity/flock-tables`). The Lean verifier reads its format, serving writes rows in it
(`verity.ml.windows.rows`), and `benchmarks/one_stage` runs it from a pinned tree sent with the run
(`--m0-pythonpath`), not from the checkout. C-Flock ZK is `flock-circuit prove --zk` (`live/PROTOCOL.md`).
`flock-circuit` also reads the typed statement (`verity/flock-circuit/types`, `verity_flock.typed_statement`) of a
flat class and of a template whose instance type places calls (GEMM) and reads tables (attention, as placed `table/v2`
reads): `live/src/typed.rs` derives the rows from the types and layouts, as the verifier does, and `flock-circuit rows`
writes them as staging's `rows/`, to diff the two mirrors.

## Hill-climb points in GPU sessions

A BF16 hill-climb point (`pod/74-gemm-hill.sh`, its records by `pod/gemm_hill.py`) runs in a GPU session: one job on node 1,
one preflight check (byte identity at B = 16), then `POINTS=N` points of one statement. A session's points share one tree,
one binary, one `FC_*` set and one K; a setting that differs needs a session of its own. A point whose key (the binaries, the
`FC_*` settings, the GPU) differs from its check's fails, and with it the session.

1. Stage with no GPU (`gpus: 0`; 64 GiB at K <= 4096, 128 GiB at K >= 8192):
   `bash verity/ml/flock/pod/74-gemm-hill.sh K=8192 CPUS=16 STAGE_ONLY=1 SM=120 FLOCK_WORK=/workspace/jobs/<lane>`.
2. Run the session with one GPU and the same settings plus `POINTS=N`: a step and its confirming re-run is `POINTS=2`, a
   mean of three `POINTS=3`. With a verifier pod (`VERIFIERS=DIR`), the serving job gets the same `POINTS`.
3. Once the attempt is in the store, roll up its points, each citing its session's art. On node 1 (no `research` on the
   PATH; `T` is the run's tree):

   ~~~sh
   RESEARCH_STORE=/workspace/jobs/store PYTHONPATH=$T/tools/research/src \
   RESEARCH_BIN="/workspace/jobs/bin/uv run --no-project --python 3.12 python -m research" \
     python3 $T/verity/ml/flock/pod/gemm_hill.py rollup --attempt RUN --run-dir /workspace/jobs/runs/RUN \
       --file /workspace/usage/hillclimb/bf16-GemmCoordinate_v2-K8192.json
   ~~~

   Each point lands with `arts`: its own record's art (`point`), the attempt's `result.json` (`session`, a
   `hillclimb-session/v0`) and the preflight check's record (`preflight`). `--run-dir` reads the points from the run's
   directory and checks their bytes against their arts; without it they are fetched from the store. A point rolled up
   before gets its `arts` and is not appended again. `--point FILE` still appends one point that isn't in the store,
   citing no art.

A point's `gpu_held_s_amortized` is its sweep plus an equal share of the session's other GPU-held seconds (the check, the
pod's start); compare it with a one-point job's `gpu_held_s_job`. The session's `result.json` passes only when the check and
every point pass. `POINTS=0` runs the preflight check alone, and `STAGE_ONLY=1 POINTS=0` stages only its statement. The
check's three GPU cases run at once, each in its own process, while three times one case's device memory (measured per K in
`74-gemm-hill.sh`) fits the GPU's free memory, else one after another; its record (`selftest.per_case`, `selftest.run`) has
each case's wall time, exit code and peak device memory, and why it fell back.

## Tests

~~~sh
uv run pytest verity/ml/flock/tests                          # in `check`; several minutes on 4 cores
FLOCK_IR_SLOW=1 uv run pytest verity/ml/flock/tests/test_ir_lowering.py   # the five 2^19/2^21-row units
cd verity/core/service/flock && lake build               # the Lean verifier; `check` builds and audits it
~~~

`tests/test_lean_verifier.py` skips its Lean-backed tests without `lake` on the PATH. `check` runs none of the
`live/` crate's Rust tests; they build only inside the upstream checkout.
