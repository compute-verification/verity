[2026-11-21T14:11Z] a9b278621333
From the lead: the go-ahead is given, with five conditions on your branch, to hold yourself to and to report against one by one:
(1) the fill value is a constant of the Definition, never an input;
(2) `ok` gates EVERY word of h, and `ok` is "below V and opened"; shown by the circuit-check report and by a test with a FAILED opening at an IN-RANGE token giving the fill row in every word;
(3) it is a new version, the served Program reads only it, and BOTH v3 and v2 (`EmbeddingRead` version 2, `read(V, H)`, `EmbeddingReadSelect`) are deleted if nothing else names them. Apply your reader-finding method to v2 as well: on the road branch v2's names appear in adapters/torch, the adapter's pipeline (serving_templates.py, serving_rows.py, long_row.py, expert_rows.py, history_rows.py), dsa_topk_reads.py, boolean_fp8_moe.py and several benchmarks: some of those hits may be other Definitions with similar names; say which truly read v2. If v2 has real readers, keep it and say who; do not rewrite other lanes' callers in this change;
(4) both stale docstrings mended in the same change;
(5) v4's doc states what is held, in the Definition's own words: h is the table's row at the token where the row under salt opens to root, ELSE the fill constant: so a prover may make any position read the fill row. Say it plainly in the doc string; it is not a footnote.
It will be merged together with the anchors' branch (the table's root and domain typed supplied); it does not count without them, so do not depend on anything that assumes run-input anchors.

