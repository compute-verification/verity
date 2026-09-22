# a-verifier: where the independent Rust verifier and the a-gpu Python verifier disagree

Lane `a-verifier` (track A), 2026-09-22.  The verifier is `backends/gkr/verifier/` (`verity-gkr-verify`, no
dependencies: its own BabyBear^6 over X^6 - 31, SHA-256, NTT, `proof.bin` parser, GKR / LogUp / Ligero checks),
written from `backends/gkr/gpu/README.md` + `PROTOCOL.md` and the a-gpu Python source, sharing no code with it.
Every proof below was produced by the a-gpu prover on `vy-g5` (H100) and checked by both verifiers.

## Findings

### F1. The a-gpu Triton-kernel path currently produces proofs that BOTH verifiers reject (prover bug, not a verifier one)

`python -m gpu.run prove /workspace/bb/pos4096 --device cuda` (the configuration of the recorded 4.43 s ledger row,
r20260922-074455-d0b6, tree c4d472a) now yields a proof that the Python verifier rejects with
`LogUp level 0: final check` and the Rust verifier rejects with `LogUp POW level 0: final check`, at B = 2 (synth),
64 and 4096, deterministically (runs r20260922-084329-7c80, -085845-2a26, -090318-2864, -091240-4bf9), and at the
a-gpu lane's own recorded tree c4d472a with its own `PYTHONPATH` (r20260922-091831-ef04, -092229-641b).  With
`VERITY_GPU_TORCH_ONLY=1` (every kernel replaced by its torch reference) the same command produces proofs both
verifiers accept (B = 64: r20260922-090318-2864; B = 4096 x 3 reps: r20260922-091240-4bf9, prover 24.9-27.1 s,
Python verifier 9.7-9.8 s).

The level-0 check is `lambda * q_root == p0 q1 + p1 q0 + lambda q0 q1` at the root of the POW table's fractional-sum
tree, i.e. the committed numerator `sum m_i/(t_i + z) - sum 1/(q_j + z)` is not zero: the kernel path's
multiplicities / leaf build (`_fast_phase1`, `_fast_combine_level`, `_fast_scatter`) disagree with the torch path
for the first table.  Two verifiers written independently reject the proof at the same place, so this is a prover
fault.  The recorded 4.43 s run verified at 07:44Z on the same pod and tree, so the fault is environmental or
non-deterministic (a race in a fused kernel, a Triton cache state) rather than a source change -- for the a-gpu lane
to bisect.  **Consequence for the ledger**: the 4.43 s prover row is not reproducible as of 09:20Z; the reproducible
GPU prover today is the torch path at ~25 s.  The e2e numbers below are given for both.

### F2. `proof.bin` did not exist

`gpu/README.md` documented the transcript and the in-memory `Proof` (root, messages, opening) but no serialisation;
the a-gpu lane never wrote a proof to disk (its verifier ran in-process).  This lane added `Proof.to_bytes /
from_bytes` and the README "Proof file" section (self-delimiting little-endian; every count is *checked* against the
statement by the verifier, none is *used* to size anything).  Not a disagreement, but the reason no independent
check existed.

### F3. Statement binding is by the circuit files, not the manifest

The Rust verifier takes the chain columns (`x.c_lo/hi`, `y_lo/hi`, `y32_lo/hi`, `y16`) by *name* from
`circuit.txt` / `epilogue.txt`; the Python `load_instance` takes their indices from `manifest.json["chain"]`.  Both
absorb the same circuit text digest, so a manifest that points at other columns would make Python accept a proof of
a different statement than the one the circuit text describes.  Not exploitable by a prover (the manifest is the
verifier's own input) but a footgun: the manifest's `chain` block should be checked against the names.

### F4. Reported `slots` differ run to run

`gpu.run` reports `slots` (rounds.protocol) 1920 at B = 4096 and 1246 or 1247 at B = 64 for different proofs of the
same statement: Ligero's `open_set` redraws on a duplicate column index, so the count is proof-dependent (expected
excess ~ t^2 / 2n = 1.1).  The Rust verifier's `slots` agrees with Python's for every proof both accepted.  The
depth 438 the latency lane uses does not depend on it.

### None otherwise

Every proof the Python verifier accepted, the Rust verifier accepted; every negative and mutation the Python
verifier rejected (or would reject), the Rust verifier rejected -- see the table in `../../gkr/verifier/README.md`
and the ledger entries.
