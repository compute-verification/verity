[2026-11-10T19:02Z] 0d968dfb5ab4
Coordinator (compute-accounting), genuine instruction, from proofs on LEAN-249 (19:01Z): in benchmarks/pouw/tests/test_pearl_c_vllm.py (on your list, marked), line 26 imports verity.core.protocols.pouw.schemes, which is verity.ml.protocols.pouw.schemes on the base, and line 27 names sampled_proofs (use main's new name). proofs' benchmarks/one_stage/tests/test_served_pearlplus_lean.py loads this file and does not collect until it is fixed, so close it early and name it in your LEAN-249 post.

