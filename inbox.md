[2026-11-16T20:30Z] cdd7777b3bcc
From PAGE to the training lead: your (ii), and what formalism.py's readers take. Read at main 3a621d22e61 ([read] unless marked).

**(ii) The Remote joins: DELETE all 8. None is a small restatement over the page.**
- `SlowdownRemote` and `SlowdownRemoteMonitoringPeriod` (the two that join a count to PoUW and SPOT; `_honest` and `_nontrivial` are their witnesses) take the page's count only as `Cut.workHE n k (βw βx) b O` (CountHE's MAC bound). They conclude a compute-efficiency bound: η ≤ s + (1 − s)(r̂ + ε_W W/C + ε' + JΔ/H + θ) + ν + χ, and the tiers' hardware multiple max(1, 1/(s + x/MFU₀)).
- That conclusion rests on the tiers' vocabulary in `Definitions/Slowdown/Slowdown.lean` (`efficiency`, `unregisteredShare`, `residualShare`; classed delete), on five named premises of the tiers' compute model in `Assumptions/Slowdown.lean` (`SourcesCover`, `ResidualAtMost`, `WithinCompute`, `UnmonitoredAtMost`, `CheatingAtMost`), and on `Remote.lean`'s `hardwareMultiple` and `remoteSlowdown`.
- Restating over the page means replacing that model by ρ = nc/|C|: the conclusion changes kind, from efficiency over C + U to a gate ratio. That is a new theorem, not a restatement.
- `CertificateSlack` (Markov over PoUW's windows) and `VerdictLag` (lag over identities) state no slowdown at all. They are PoUW and SPOT accounting lemmas. They go with the tiers, or move to ResourceAccounting if lean or compute-accounting wants them.
- Stated consequence: the join of the page's count to PoUW and the sampled proofs is not in Lean.
- Readers to edit [ran]: `Proofs/InferenceOnly.lean` imports `Proofs.Slowdown.RemoteMonitoringPeriod` (lean's aggregator line); `Proofs/Slowdown/Escape.lean` and the two Witnesses files are D's or C's. Compose's `verifier/remote/README.md` describes `SlowdownRemote`.

**What formalism.py's readers take** [ran, an AST pass over every importer]
- `grade.py` takes 12 names: `Refused`, `STAND_IN`, `CELL_SENSITIVITIES`, `of_record`, `traced_label`, `measure`, `bound`, `published`, `of_trace`, `products_of_log`, `TRACED_STRUCTURE` and `READ_BACKS`. The grade's number of record is `formalism.published` at (c, k). So deleting formalism.py means restating the grade's slowdown over `min_cut.py` (the page's theorem), not just deleting.
  - `grade` is read by `benchmarks/training_slowdown/run.py` (`config_of`, `read`, `report`, `SPLITS`) and by `library.py`, which takes only `grade.per_period`.
  - `library.py` does not import formalism.py.
- `capture/circuit.py` and `capture/products.py` do NOT import formalism.py; the map's "import" was a docstring grep, corrected here. They compute formalism v2's quantities on a traced step: `capture/circuit.measure` gives σ, τ, a, u_in, the tail, Δ_grad and the cuts, plus `gates` and `products`. `capture/structure.py` stores that dict as the graph's `formalism` block.
  - `products.py` reads only `graph.formalism["gates"]`.
  - `traced.py` writes the block as the record's `circuit`.
  - `grade.py` reads `row["formalism"]` (`parts`).
  - So the page needs the block's `gates` and `products` (|C| and the MACs), and nothing of σ, τ, a, u_in, the tail or Δ_grad.
- The tests take [ran]:
  - `test_formalism.py`: everything (about 40 names); delete it whole.
  - `test_grade.py`: `published`, `measure`, `of_record`, `Refused`, `STAND_IN`, `PLAIN_PRODUCTS`, `TRACED_PRODUCTS`, `TRACED` and `log_products`.
  - `test_one_calculator.py`: `at`, `even`, `published`, `Part`, `Measured` and `bound`. It also names formalism.py's `at` and `published` in its one-calculator table.
  - `test_policy.py`: `at`, `even`, `Part` and `Measured`.
  - `test_repeats.py`: `bound`, `measure` and `COUNTED_PRODUCTS`, and `he_row` from `test_formalism`.
  - `test_run.py`: `other_program`, `TRACED_STRUCTURE` and `READ_BACKS`.
  - `test_circuit.py`: `READ_BACKS`.
  - `test_library.py`: `he_rests_on`.
  - That is eight test files, not six. Also `tests/test_one_slowdown.py` lists `test_formalism.py`; `tests/lint/allowlists/repo_retired_terms.json` has formalism.py's count row; `tools/move/slowdown_{harness,python}_map.toml` name it (data).
- Since STEP drafts on library.py: library.py reads formalism v2 only through `grade.per_period` and the `circuit` block's `gates` and `products`.

**PR 2 (formalism v2's Lean leaf) next, as a draft.** Its Lean check and lock `--update` can't run here (no secrets: `research run` can't reach a node). So the lock edit will be the 20 records and 3 assumptions removed by hand and checked locally by grep, and the draft will say the Lean check is unrun. Tell me to stop at the reading if you'd rather a worker on a fresh VM take it.

