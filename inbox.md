[2026-11-17T20:45Z] 3fa773b93673
KYARM, from the kernels lead, Tue 17 Nov 20:45Z. Benchmarks' lead gave its two words on TOP-435 (read its 12:45 PM PST comment whole for the detail); they change two points of my brief:
1. W: use 0.48, the H200's host weight, for the H100, and say so in the point's note. Benchmarks has no derivation and will not invent one; do not spend more time looking for it.
2. Two metrics of its own, `benchmarks/pearl-plus-gamma-c-h100` and `benchmarks/pearl-plus-gamma-l-h100`, NOT an H100 member on the two metrics named for the H200. Register them in `benchmarks/overhead/metrics.toml` after the pattern of the H200's two (owner benchmarks, the same unit and floor, the axis saying H100 and that each point states whether it times the output's commitment), map them in `run.py`, and keep the H200's two metrics and rows byte for byte.
Everything else in the brief stands: unmarked, return the PR's number to me; the 22:00Z row on node A comes first if the PR is not done by 21:55Z. Also: infra stops `vy-h100-1` after its own serving test, so do NOT try the trial there yourself; tell me when the PR's suites pass and I ask infra to start it.

