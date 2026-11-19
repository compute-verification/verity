[2026-11-19T00:23Z] 026b19008a6e
From the circuits lead, one small hunk to carry in your next commit onto one-circuit (a docs line in a file of ours): benchmarks/mempool/difftest.py line 23, the docstring names the encodings replay_input module, which the network lane deleted on one-circuit (93855cc1ebe): the phrase "as replay input (the encodings replay_input ...)" becomes "as replay input". Check tests/test_doc_paths.py does not name it elsewhere in our files (git grep replay_input in verity/ml/circuits, benchmarks/mempool, tools/verity/circuit_check).

