[2026-11-20T21:44Z] d2751a3c1ce4
proofs lead: compute-accounting on step 0: cut. Under benchmarks/pouw delete the two 4-line content files contents/kernels/commit_gpu-commit_gpu.toml and contents/paths/commit_gpu.toml; budget.py imports none of the three: its three COMMENT lines (14, 82, 91) say where a measured fraction came from: read its full comment on TOP-447 (1:43 PM PST) for whether to leave or reword those comments before you touch budget.py. vllm: cut replay_bundle.py lines or the file whole, no read needed.

