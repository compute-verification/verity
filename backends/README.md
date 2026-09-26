# Backends

Policy: `AGENTS.md` (labels, workloads, decision rule). Numerical references and campaign-side tooling:
`numerical/`.

## Cryptographic-composition lane (specifications, no provers)

| document | backend family | status |
|---|---|---|
| `gkr/PROTOCOL.md` | A-GKR: data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer | proposed research; numbered transcript, masking, hash assumptions, leakage rules, error decomposition, unproved obligations |
| `direct/PROTOCOL.md` | B-Ligero: direct local constraints in an interactive ZK Ligero argument with logUp lookups | proposed parameterisation; same sections |
| `numerical/python/verity_numerical/security/accounting.py` | both | soundness calculator: exact rationals and log2 per component (batching, sumcheck, lookup, encoding/opening, commitment/hash), conditioned logUp challenge space, `union_over_proofs` / `per_proof_target`, `first_configuration` per operating point, `rounds_and_depth_*` / `e2e_sweep` (latency axis), `census_estimate_*` (contract `census.*` names, estimates); tests in `numerical/tests/security/` |
| `numerical/python/verity_numerical/security/census.py` | both | census of the checker **from the WP2 gadget contracts** (`verity_numerical.checker`, not `verity.ml.tc.relation`): per transition unit 358 `mul`, 137 `lin`, 306 lookups (284 into one 16-bit range table, 22 into `ALIGN`/`POW`/`LEAD`/`NORM`), 321 committed columns (34 words + 225 hints + 62 row columns), 244 products, multiplicative depth 5; per \(K=1536\) VU 34 370 / 13 059 / 29 379; `diff_vs_relation` shows the relation's undercount (+212 `mul`, +33 `lin`, +134 lookups per unit). Source of every per-unit default in the accountant (`CHECKER_TABLES`, `DEFAULT_LAYERS_A`, `CandidateA.inputs_per_unit`, `CandidateB.witness_per_unit`) |

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
