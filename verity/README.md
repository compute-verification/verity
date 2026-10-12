# Verity core (`verity`)

One distribution, one module per concern. Import from the module that owns an object
(`from verity.ml.circuits import Program`); `verity` itself re-exports nothing. The architecture is the root
`README.md`, and the repository map is the root `AGENTS.md`.

| module | what it owns |
|---|---|
| `verity.ml.circuits` | Circuits: types, Definitions, Programs, the canonical layout, the descriptor codec, constants' binding times (`constants`), annotations (`verity/annotations/v0`). Stdlib only, and it imports no other core module. Its format (descriptor v1, canonical JSON = RFC 8785, layout, scope resolution) is specified in `ml/circuits/PROTOCOL.md`, with every rule pinned by `ml/circuits/tests/format_vectors.json`. |
| the partition references, in `verity.ml.circuits` | `cut` (the query `Q_word` v1's canonical evaluator, its algorithm in the docstring), `partition.validate_unit_cut` (the invariant) and `partition_object` (`verity/partition/v1`, a program digest plus a query: `evaluate`, `locate`, `validate`, `verify`). They are pinned by `ml/circuits/tests/partition_vectors.json` and `template_instance_vectors.json`. `units` cuts each proof unit out as a standalone circuit. |
| `verity.ml.circuits.boolean` | The Boolean profile (PROTOCOL §9): the gates `And<n>_v1`, `Xor<n>_v1`, `Not_v1`, the constant bits, `is_boolean` and the word view. Also C-Flock's linear-form builder (`forms`), the multiplexer tree (`gather`) and `trace.definition`, a builder traced into gates. |
| `verity.ml.circuits.library`, `.operations` | The library of hardware and protocol objects core owns. The pinned list of operations, whose guard (`program_data`) lists the data a program still holds. |
| `verity.ml.circuits.evaluation` | The one evaluator: the plain-integer reference (`evaluate`), `evaluate_batch` over numpy arrays with registered kernels (each listed in `ml/circuits/tests/evaluation/test_evaluation.py`, which self-checks it), and `bits`, the bit-sliced evaluator of Boolean Definitions. |
| `verity.ml.commitments` | The commitment schemes the records use (`frame_v3`, `vllm_v1`, `hm96`, Merkle trees and openings), each with a specification and conformance vectors. SHA-512 and the Merkle tree as Boolean gates are its `gates`, and Merkle multiproofs its `multiproof`. |
| `verity.ml.crypto` | Shared hash code (`blake3`, `turboshake`). Under one hash, the BLAKE3 and TurboSHAKE code leaves the served path for `archive/`. |
| `verity.ml.randomness` | `derive(source, domain, context)`, the one challenge derivation, and exact samplers over counter-mode streams. Stdlib only, and it imports no other core module. Its byte formats are pinned by `ml/randomness/tests/vectors.json`. |
| `verity.ml.private_circuits` | The universal unit: one public Boolean Definition that runs any private circuit of a size class (protocol 2, private circuits). |
| `verity.core.protocols.profile` | `IntegrityProfile`: what an accepted audit established (the exhaustive facts, a verifier's sampling design over named levels of a partition hierarchy, and δ). |
| `verity.core.protocols.sampled_proofs` | Sampled proofs as the one-stage audit (`one_stage`) and two-stage sampling (`two_stage`). Plans and consumers are `verity.core.protocols.sampled_proofs`. |
| `verity.core.protocols.pouw` | Proof of useful work: the `PoUWScheme` interface, its lifecycle (`audit`) and the named schemes (`schemes.get`), each with pinned vectors. |
| `verity.core.protocols.pous` | Proof of useful space: the reference codecs, setup, and the memory challenger's timed audit. Its grader is the Lean package `lean/`. |
| `verity.core.protocols.bandwidth` | The bandwidth audit v0: a window's tags over a PoUS store before a deadline, the verifier's sampled check, and the ingress it certifies (`PROTOCOL.md`). Its GPU tagger is `benchmarks/bandwidth/`. |
| `verity.core.protocols.remote_bandwidth` | The remote bandwidth audit v0: a node's registered store under kv/v0's codec, passes chained by their roots, each over a public plan of distinct blocks, receipts bound to PoUW's tile output, a deadline per window, the verifier's sampled openings, and the record with its γ terms (`PROTOCOL.md`). `timing` simulates the remote schedule. |
| `verity.core.protocols.certifier` | The network certifier: its audit and calibration. |
| `verity.core.claims` | The ids of security claims (guarantees, and assumptions as `property/instance` such as `cr/sha-512`) that the tables cite. Stdlib only. |
| `verity/core/`, in Lean | The Lean ports in `primitives`, `devices`, `service` and `protocols`, each held to today's Python by `difftest`, and C-Flock's executable verifier of record (`service/flock/`). |
| `Security/` | The Lean every protocol states its guarantees in, and their proofs (root `AGENTS.md`, "Lean"). |

`tests/test_boundaries.py` checks which core units each may import (`ALLOWED`) and which packages core may not import, and
it keeps the plain-integer references (`ml/circuits`, `ml/randomness`, `core/claims`) free of numpy.

~~~sh
uv run tools/verity/check/suites.py   # each directory's tests are a suite of their own; `check` runs them all
~~~
