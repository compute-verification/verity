# Backends

Policy: `AGENTS.md` (labels, workloads, decision rule). Numerical references and campaign-side tooling:
`numerical/`.

## Cryptographic-composition lane (specifications, no provers)

| document | candidate | status |
|---|---|---|
| `gkr/PROTOCOL.md` | A: data-parallel GKR + logUp-GKR lookups + interactive Longfellow-style ZK layer | proposed research; numbered transcript, masking, hash assumptions, leakage rules, error decomposition, unproved obligations |
| `direct/PROTOCOL.md` | B: direct local constraints in an interactive ZK Ligero argument with logUp lookups | proposed parameterisation; same sections |
| `numerical/python/verity_numerical/security/accounting.py` | both | soundness calculator: exact rationals and log2 per component (batching, sumcheck, lookup, encoding/opening, commitment/hash), conditioned logUp challenge space, `union_over_proofs` / `per_proof_target`, `first_configuration` per operating point, `rounds_and_depth_*` / `e2e_sweep` (latency axis), `census_estimate_*` (contract `census.*` names, estimates); tests in `numerical/tests/security/` |

Labels: an implementation of either candidate is `NON_ZK_PROOF_DIAGNOSTIC` until its interactive, private,
hash-based composition (protocol sections 5–8) is implemented and its budget is reproduced by the calculator;
with honest-verifier privacy only (no verifier-coin commitment, protocol section 5.5 / 5.3) it is
`COMPLETE_HVZK_BACKEND`, never `COMPLETE_ZK_BACKEND`; a plain Plonky3 AIR/STARK is `NON_ZK_PROOF_DIAGNOSTIC`
by construction (Fiat–Shamir, unmasked openings). The calculator refuses to derive "bits of security" from a
field size; only a complete `Budget` has a level.
