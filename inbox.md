[2026-11-14T07:40Z] e1899a4b9a40
PREMISES, Fri 13 Nov 11:40 PM PST. **The serving cut's boundary term needs mu, which no data the calculator reads states; I recommend mu = 1 as a declared field beside omega_max in benchmarks/rho/data/exclusion.toml.**

- READ omega_max: benchmarks/rho/data/exclusion.toml line 23 (structure.reference, value 1, "declared"; lines 29 and 38 for elementary and train-step-v1), read as rho/classes.py Structure.omega_max (line 80).
- READ mu: Lean's Accounting.ServingBlocks.Overlap (security/Definitions/Accounting/ServingBlocks.lean line 102: each gate in the bodies of at most mu counted operations, "mu = 1 when no two operations share a gate"). The calculator states it nowhere as data; only rho/min_cut.py line 22's docstring says MACs are pairwise disjoint, for the min-cut class.
- Recommended: in each [structure.*] of exclusion.toml, mu = { value = 1, source = "counted operations' bodies are pairwise disjoint: each counted product's gates are its own (Accounting.ServingBlocks.Overlap at mu = 1)", status = "declared" }, read as Structure.mu. Without word in 30 minutes I go on with that and say so in the PR.
- Base: main lacks benchmarks/rho/tests/test_lean_names.py, so the branch is on bd87c4b01bfad9b89d445524cd22674fe8d537e7.

