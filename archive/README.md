# Backends

A backend establishes Verity verification obligations (root `README.md`, "Backends"). Each directory here is a backend
family or a component the families share. Policy for the proof campaign (labels, workloads, the decision rule):
`AGENTS.md`.

## Families

C-Flock is the one active backend family. The others are frozen: kept, still in the tables, not developed.

| directory | family | status | what is here |
|---|---|---|---|
| `../verity/ml/flock/`, `../verity/core/service/flock/` | C-Flock | active | binary-field proofs of committed relations, hash leaves included, with upstream Flock: the Python package `verity_flock`, the Rust crate `live/` (built inside an upstream Flock checkout), and the verifier of record: its specification, `../verity/ml/flock/verifier/PROTOCOL.md`, and the Lean verifier, `../verity/core/service/flock/` |
| `gkr/` | A-GKR | frozen | data-parallel GKR with LogUp-GKR lookups (`gkr/PROTOCOL.md`): the CPU prover `verity-gkr` (Rust), the GPU prover `gkr/gpu/`, kernel studies (`gkr/packed/`, `gkr/tensor/`), the independent Rust verifier `gkr/verifier/` |
| `direct/`, `ligero-verify/`, `ligerito-verify/` | B-Ligero | frozen | direct local constraints (`direct/PROTOCOL.md`): the Ligero prover `direct/ligero/` (torch), the Plonky3 diagnostic crate `verity-direct`, the encoders `direct/encode/`, Ligerito `direct/ligerito/`, and independent Rust verifiers for the Ligero and Ligerito proofs |
| `sp1/` | D-SP1 | frozen | the SP1 zkVM checker guest, its host and the Python driver `verity_sp1` |
| `vole/` | VOLE (QuickSilver, designated verifier) | frozen | the WP2 checker exported for Diet Mac'n'Cheese, a patch to swanky, the bench driver |
| `numerical/` | none | frozen | `verity_numerical`: independent references, the WP2 checker, the soundness calculator, and the bench spine (`verity_numerical.bench`) that C-Flock and the table renderers still use |
| `shared/` | none | frozen | GPU Merkle hashing (`../verity/ml/kernels/hash_gpu/`) and the A100 re-pricing (`shared/anchor_a100/`) for A-GKR and B-Ligero |
| `redteam/` | none | frozen | red-team forgery harnesses against A-GKR and B-Ligero |
| `pouw-blake3/` | none | frozen | Pearl-C's BLAKE3 hashing formats (-h1, -h2, the BLAKE3 -h3), their kernels, benches and vectors, kept after the 2026-10-06 ruling that standardized PoUW on SHA-512 (`pouw-blake3/README.md`); no tests |

## What `check` runs

`check` (`tools/verity/check/check.py`) is the merge gate. Its pytest step runs every test suite (`tools/verity/check/suites.py`, skill
`testing`): each family below with Python tests is one suite, configured in its own `pyproject.toml`, and a pass is reused
while the suite's inputs are unchanged. It has no `cargo` step, so no Rust test under `backends/` runs in it. The frozen
crates' proofs, systems and vectors are fetched fixtures (`fixtures/artifacts.json`): run
`uv run research data fetch-fixtures backends/<crate>` before their `cargo test` (`include_bytes!` needs the files to compile). Its Lean steps build C-Flock's Lean verifier, check
its partition check against `verity.ml.circuits.partition` on random cuts, and audit every Lake package; the Lean verifier's
agreement with upstream's Rust verifier runs only when the run is sent its input bundle.

Python counts are pytest's collected items on a CPU machine after `uv sync --all-packages --extra torch-cpu` (CPU-only torch,
no GPU). Rust counts are `#[test]` functions. C-Flock's row gives no counts, since most C-Flock PRs change them;
`cd verity/ml/flock && uv run pytest --collect-only -q | tail -1` gives the current one.

| family | status | Python tests `check` collects | Python tests outside `check` | Rust tests (`check` runs none) |
|---|---|---|---|---|
| C-Flock | active | `../verity/ml/flock/tests`, of which some skip: without `lake` on the PATH, without `FLOCK_IR_SLOW=1`, or without a captured sampling set (`VERITY_CAPTURED_101_GUMBEL`) | none | `../verity/ml/flock/live` |
| A-GKR | frozen | `gkr/tests`: 61 | `gkr/packed/tests` 17, `gkr/gpu/v2/tests` 9, `gkr/tensor/tests` 22, `gkr/tensor/fused/tests` 25; 3 more files need torch to import | `gkr`: 23; `gkr/verifier`: 43 |
| B-Ligero | frozen | `direct/ligero/tests`: 12 | `direct/`: 427; 14 more files need torch to import | `direct`: 47 (157 more in vendored Plonky3); `ligero-verify`: 74; `ligerito-verify`: 99 |
| D-SP1 | frozen | `sp1/tests`: 10, of which 3 skip without `VERITY_SP1_RELATION_ONLY_HOST` or `VERITY_SP1_PROVE=1`; `test_typed.py` skips whole without a built host | none | `sp1/common`: 135; `sp1/host`: 1 |
| VOLE | frozen | none: `vole/` has no tests | none | none (the swanky patch is applied outside the repository) |
| numerical | frozen | `../benchmarks/numerical/tests`: 1,059, of which 9 skip | none | none |
| shared | frozen | none | `../verity/ml/kernels/hash_gpu/tests`: 132 | none |

Outside `backends/`, `check` also runs `integrations/vllm`'s suite on the CPU; its GPU, vLLM-runtime and pod tests skip
themselves, and its known failures on `main` are listed in `integrations/vllm/tests/conftest.py`.

## A-GKR and B-Ligero: shared analysis

| document | backend family | what it is |
|---|---|---|
| `gkr/PROTOCOL.md` | A-GKR: data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer | the specification: numbered transcript, masking, hash assumptions, leakage rules, error decomposition, unproved obligations |
| `direct/PROTOCOL.md` | B-Ligero: direct local constraints in an interactive ZK Ligero argument with logUp lookups | the parameterisation; same sections |
| `../benchmarks/numerical/python/verity_numerical/security/accounting.py` | both | soundness calculator: exact rationals and log2 per component (batching, sumcheck, lookup, encoding/opening, commitment/hash), conditioned logUp challenge space, `union_over_proofs` / `per_proof_target`, `first_configuration` per operating point, `rounds_and_depth_*` / `e2e_sweep` (latency axis), `census_estimate_*` (contract `census.*` names, estimates); tests in `../benchmarks/numerical/tests/security/` |
| `../benchmarks/numerical/python/verity_numerical/security/census.py` | both | census of the checker **from the WP2 gadget contracts** (`verity_numerical.checker`, not `verity.ml.circuits.silicon.relation`): per transition unit 358 `mul`, 137 `lin`, 306 lookups (284 into one 16-bit range table, 22 into `ALIGN`/`POW`/`LEAD`/`NORM`), 321 committed columns (34 words + 225 hints + 62 row columns), 244 products, multiplicative depth 5; per \(K=1536\) proof unit 34 370 / 13 059 / 29 379; `diff_vs_relation` shows the relation's undercount (+212 `mul`, +33 `lin`, +134 lookups per unit). Source of every per-unit default in the accountant (`CHECKER_TABLES`, `DEFAULT_LAYERS_A`, `CandidateA.inputs_per_unit`, `CandidateB.witness_per_unit`) |

Labels: an implementation of either backend family is `NON_ZK_PROOF_DIAGNOSTIC` until its interactive, private,
hash-based composition (protocol sections 5–8) is implemented and its budget is reproduced by the calculator;
with honest-verifier privacy only (no verifier-coin commitment, protocol section 5.5 / 5.3) it is
`COMPLETE_HVZK_BACKEND`, never `COMPLETE_ZK_BACKEND`; a plain Plonky3 AIR/STARK is `NON_ZK_PROOF_DIAGNOSTIC`
by construction (Fiat–Shamir, unmasked openings). The calculator refuses to derive "bits of security" from a
field size; only a complete `Budget` has a level.

Census headline (WP2 counts, \(K=1536\), \(B=4096\), Goldilocks cubic at \(2^{-128}\); all rows but committed
elements and depth are cost-model estimates): committed elements A **181M** (321 + 140 committed query wires per
unit; 126M without them at depth 630) vs B **780M** (1 064 slots + 306 \(F_c\) helpers per unit), ratio 1:4.3
(1:4.0 on the superseded relation census: 101M vs 405M); bytes encoded 6.2 GB vs 26.6 GB; sequential depth 381
vs 3 (`rounds.protocol` 1 589 vs 7); peak memory 14 GB vs 33 GB; field operations 121G vs 64G mult-equivalents
(A's under a carry-everything layer-width bound). Quadratic extension at \(2^{-80}\): 181M vs 659M. No operating
point moved (quadratic first meets \(2^{-80}\)/\(2^{-96}\), cubic \(2^{-112}\)/\(2^{-128}\)). The gap is an
explanatory variable, not a verdict.
