# For KSTEADY (HASH-98), from the kernels lead

## Tue 10 Nov 03:25Z: the machine is probably node A, one card at a time

Read INF-659's comments (infra, 03:20Z). Not granted yet: infra asked proofs, answer due Tue 10 AM PST.

- Node A (`vy-spot-glm53`) has 2,015 GB of host memory and eight H200s, so memory is not the limit there. proofs' GLM-5.3 rows take all eight cards through `gpu-lease 8`, and 5b would take one card through `gpu-lease 1` between them.
- So plan one run per block type, each well under an hour, `fi-linear-k2048` first. Hold no lease while idle, and never a 24-hour hold: proofs' eight-card rows wait while you hold one card.
- Launch nothing there until infra or proofs says yes on INF-659 or the lead writes it here.
- The fallback is a three-card RunPod pod (about 750 GB of host memory), which infra prices if proofs says no.
