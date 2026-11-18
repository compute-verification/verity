[2026-11-18T02:37Z] 2e1498533bf0
From the proofs lead, one caution for the Foundation job from another worker's night: a prover binary built on our nebius nodes (glibc 2.39) did not run on node A (glibc 2.35). Before you hand infra the binary built on vy-h100-1, compare its glibc with the Foundation node's (ask infra for `ldd --version` there, or read it from the serving job's output in the notes), and say in your return whether the binary can run there or must be built in a container matching the node.

