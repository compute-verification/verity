
## 26 Oct 01:15Z, from the coordinator: the hashing cost bounds the credited rate on the H200 (compute-accounting's finding, CMP-107, note:compute-accounting/20261025T2330Z-report-h200-pouw)

- The only pinned INT8 tile, `PouwH1Tile_v1{K=64,R=4,C=8}`, costs 17 SHA-512 compressions per 2,048 MACs. That cost, not the kernel, sets the credited rate.
- At the H200's measured 4.79 G compressions/s, covering the frozen shape's static matmuls would take about 83 min of hashing per step, against a 3.7–7.8 s step. Even giving the GPU's whole SHA-512 rate to H1 credits only 0.075–0.15% of the static MACs. So Demo A credits a fixed public subset of cells, and the covert-compute term is nearly vacuous there.
- There are two alternatives. One is per-call noise, pinned for the window check (J10-R's `PouwH1Tile_v5/v6`): the period credit already accepts it on the served Llama path, and kernels say an sm_90a build of their mma.sync s8 units is a recompile. The other is a 32×32 tile at K ≥ 4,096: 612 compressions per 4.2 M MACs, 57× less hashing per MAC. hash's model still puts it at γ ≥ 19 on the H200 (note:hash/20261019T0427Z-draft-dither-hb).
- top has put the choice to Daniel, recommending that M3 run the honest subset and report it as is, with CMP-108 as the principled route.

**For you (PROTO-DESIGN):** every candidate's γ must include its commitment and proof hashing: SHA-512 compressions per credited MAC, at 4.79 G/s on the H200. Treat "credit covers what fraction of the served MACs, at γ ≤ 10" as a first-class column. Include per-call-noise J10-R (v5/v6) and the large-K tile as candidates.

## 26 Oct 01:20Z, from the coordinator: parallel bets
Daniel (01:05Z): "Let's take a bunch of parallel bets and see what pays off." The brief's section 5 now lists six bets, each with its own worker: BET-HYBRID CMP-120, BET-PEARL-I8 CMP-121, BET-BANDWIDTH CMP-122, BET-K-DISCHARGE CMP-123, BET-COVERAGE CMP-124 and BET-ATTACK-LAB CMP-125.

**For you:** you are the scorer. Your ranked table includes every bet as a candidate row, and you update it as bets report (I relay their results). Score each bet on its γ, its coverage of the served MACs, the threats it closes, the proof route it enables and what's left unproved, and its kill status.

## 26 Oct 01:25Z, from the coordinator: the per-call-noise tile is being built for the H200
Daniel (6:05 PM PDT, via top): kernels has started the sm_90a build of `PouwH1Tile_v5/v6` (per-call noise, J10-R) with KCOMMIT-FB: the recompile, the SASS gate and the exactness gates, on vy-dc-h200 (not the demo card). Whether M3's credit cites it is still Daniel's decision. Treat v5/v6 on the H200 as the near-term baseline row: get its SHA-512 compressions per credited MAC from the J10-R code (`benchmarks/hash/dither`, h1u), and say what coverage it reaches at γ ≤ 10. When kernels posts measured numbers, I'll relay them.

## 26 Oct 01:42Z, from the coordinator: avoid "Activity task timed out"
Two ACCEL workers (O-IO and O-CHARGE) died 30–40 min into their first turn with "Activity task timed out", before pushing or writing a note, so their work was lost. Do these:
- **If you haven't yet,** write and sync your note now, even if it only holds your plan and findings so far, and sync it again at every result.
- **Keep every tool call short** (well under 10 min).
- **Read big files in slices.**
- **Push your branch early.**
- **End your turn** rather than run one very long step.
