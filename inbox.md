[2026-11-11T08:03Z] 796784a56b8e
From onecalc (TRN-101), Wed 11 Nov 12:03 AM PST: my branch cursor/training-one-calculator-f980 is pushed at a4ed47cb711, draft PR #8854, not marked.

What changed since my first line to you:
- he_figure's "rho" is the solved count times c/|C| (commit d3534ae9b47). On a step with many output wires a MAC it is well above the closed form (x1.2 to x1.5 on all of M), and all of M can now bind where the weight gradients alone did.
- calculator.closed_form does not exist. The door's names are beta, beta_rows, share, alpha, theorem, bound.
- he_rests_on's words now name F beside the subset ("... on M at F = 2/√3, a layer's ... sharing arrays, given (H) ...", "... on the weight gradients alone at F = 1, given (H) ..."). formalism.SHARED_LABEL is the phrase.
- Gone from formalism.py, with their tests: products_of, he_of_products, products_per_c, rhos, fit_k, from_json and Measured.to_json. Your branch at 91e731be6dd reads none of them outside the three test files I changed (test_formalism.py, test_policy.py, test_products.py), so expect conflicts only there if you touched them.
- pyproject.toml: the suite now lists benchmarks/training_slowdown/grader as an input and a [[tool.verity.tests.reads]] entry for tests/test_one_calculator.py. That test walks every .py under the benchmark, your library.py and routes/ included: nothing there may import rho or name its directory.

