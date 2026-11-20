[2026-11-20T16:34Z] 43083346e927
Coordinator 16:40Z: benchmarks/pouw/credited_in_c.py was left where it is because your branch changes it. It has 9 direct imports of the scheme package against an entry of 0 in the ratchet (tests/interfaces_ratchet.json, scheme_package_imports). Do NOT move it now; say in your return what your branch changes in it, so the move under verity/protocols/resource_accounting/compute/pouw/prover/ can follow your merge. The tip is 17bc63d2ba3.

