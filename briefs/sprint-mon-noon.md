# Sprint: the end state by Mon 26 Oct 19:00Z (noon PDT)

Written 26 Oct 03:00Z by the coordinator (bc-ebc09a9a, @compute-research). Daniel, 8:51 PM PDT 25 Oct: "Ok make sure we
are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and it should be getting red-teamed in
parallel." Linear: CMP-135 (sprint), CMP-136 WORKSOUND, CMP-137 SKIP-FIX, CMP-138 COVER-ATTN, CMP-139 RT-UNITS, CMP-140 RT-PROTO, CMP-141 RT-LEAN, CMP-142 RT-MODEL. The goal is `briefs/accel-program.md` section 0. Every ACCEL worker reads this file at each inbox read.

> **Clock correction (03:52Z real UTC, from `date -u`):** the coordinator's inbox stamps between about 03:15Z and 05:25Z ran up to 1.5 h ahead of real UTC. Every deadline in this file is real UTC, so there's more time than those stamps implied. From now on, all stamps come from `date -u`.

## What "done at 19:00Z" means

| # | Deliverable | Due |
|---|---|---|
| D1 | **Protocol frozen**: one spec with its parameters, γ, coverage and ε at θ for the H200 and the RTX PRO 6000. v1 at 06:30Z, v2 (after red team) at 12:00Z | 06:30Z / 12:00Z |
| D2 | **`WorkSound`, the headline theorem in Lean**, proved, its premises exactly section 0's trust base, lock entries, PR marked | statement 07:00Z, proof 15:00Z, marked 17:00Z |
| D3 | **The model in Lean** (`Accel`, both instances, θ tables with run ids), PR marked | θ 08:00Z, marked 14:00Z |
| D4 | **Naive tiled INT8 beats Strassen under the model**, Lean | paper 10:00Z, Lean 16:00Z |
| D5 | **The public-best artifact**: every exact method we know, priced at θ on the frozen protocol's operands; gives \(c_{mm}\) and \(c_{sha}\) | v1 12:00Z, v2 17:00Z |
| D6 | **The number table**: ε, γ, coverage, perplexity; world model (rule L2); week, month, year; δ = 1% and \(10^{-6}\) | 17:00Z |
| D7 | **Red-team reports**: what each red team tried, what broke, what was fixed, what stands open | 18:00Z |

Not by noon: the GPU demonstration (the served kernel's γ on real H200s), which waits for M3 (Mon 23:00Z). Until then γ is
by data sheet, and the table says so.

## Where it stands (03:00Z)

- **Protocol candidate (PROTO-DESIGN, `note:20261026T0133Z-draft-accel-proto-design`, "No-fold receipt"):** HYB noise
  (BET-HYBRID-2) with every word a SHA-512 Merkle leaf, no fold. Binding = collision resistance + Theorem S. At skip
  budget e = 1%: γ 7.2 static MAC-weighted (8.9 at K 16384, 4.75 at 65536), ε ≈ 1.1%, coverage 95.2% of all matmul MACs
  at average context 4k (90.9% at 8k), QKᵀ not creditable. **The serving skip caps L** (\(\varepsilon_{skip} = L/(\gamma K)\)):
  designing it away lets L → K, γ_hash → C/K ≈ 0.7 at K = 16384, frees the budget for QKᵀ, and puts ε < 1% in reach.
- **Theorem (DITHER-THM, `note:20261026T0130Z-draft-accel-dither-thm`, head 9c280aaad6):** restated under R-ACCEL-6;
  one computational premise `SAKMatmul` (Theorem O at \(c = \mathrm{pub}/H\) plus the direct-sum cut), `ShaSpeed` in its
  own slot, `FreedOfFloor` proved, \(\varepsilon = (1-h)(1-c_{mm}) + h(1-c_{sha}) + \ln(1/\delta)/(\eta k) + \beta K/(kH)\).
  **Open: commit-and-open.** Collision resistance gives binding, not that the committed words were computed: a root
  produced some cheaper way and lazily opened is not excluded. Closing it needs SHA-512 as a random oracle for the
  commitments (each query priced at the public-best speed, R-ACCEL-7), or SAK(matmul) in a commit-and-open reading.
  Asked of Daniel 03:05Z with the random oracle recommended; state the headline with a named `Extract` premise now, so
  either answer slots in.
- **Model (MODEL-SPEC, head 5f9f1bb594 on #4389):** P1Int, P2Local/Distinct/Frame/Priced and both refinements proved;
  looped wgmma witness, P1Float, P1Captured in progress. CALIB's θ table in progress.
- **Public best (BET-ALGO-LAB, 68d6a42285):** closest alternative Strassen d1 at A = B = 0, 3.82× honest, priced with
  the old fold design. Must be redone on the no-fold operands, where words are sparser.

> **R-ACCEL-8 (04:41Z):** no shadow GEMMs and no noised copies. Credited work is the useful product with O bound. See brief section 1.

> **Status 06:50Z (real UTC).**
> - **D1:** protocol v3 is adopted (PROTO-DESIGN-3, CMP-118 e0cd4061): H200 t = 2, L′ 512–640; RTX PRO 6000 t = 3, L′ 256. Its overlapped γ is at most 6.06, so P-HYB3 is only the alternative.
> - **D6:** PROTO-DESIGN-4 bc-9bcba447, due 17:00Z.
> - **D2:** WORKSOUND-3 bc-85e32442 owns the statement and proof: W1/L37, L39, L27–L29, P13–P17 and P20. WORKSOUND-WIT bc-e7fce412 owns the witness (L7, L38). WORKSOUND-INST-2 bc-e039ac1a owns #4567.
> - **D3:** MODEL-SPEC-4 bc-a2fef9e0, mark #4553 at 14:00Z. It is blocked on lane lean's #4389, which conflicts with main (LEAN-59).
> - **D4:** NAIVE-STRASSEN-2 bc-f3e9dd67, #4607 (depth 3 proved), due 16:00Z.
> - **D5 v2:** BET-ALGO-LAB bc-f53e4d35, posting early.
> - **Red teams:** ALGO-COST-3 bc-1cd0aa7d (reconciles A25 with v3, checks D5 v2), RT-PROTO-2 bc-d03b7138, RT-LEAN-2 bc-6a8c9f0f.
> - Decisions on P15–P20 are in `inbox/ACCEL-PROTO-DESIGN.md` 06:45Z.
>
> **256 GB Lean checks (infra, 07:03Z): use `research run --on vy-nebius-cpu-2 --mem-gb 256`.** It has 837 GB free and runs to 28 Oct 06:00Z. vy-nebius-2 and -3 stop at about 08:00Z, so don't use them. If cpu-2 refuses the reservation, give the coordinator the run id; infra clears the ledger.

## Who builds what, and who red-teams it

| Deliverable | Builder | Red team |
|---|---|---|
| D1 protocol v1/v2 + D6 numbers | PROTO-DESIGN bc-7a403fcc | RT-PROTO, RT-ALGO |
| Serving skip designed away (into D1 v2) | SKIP-FIX (new) | RT-PROTO |
| Attention, QKᵀ and PV credited within γ ≤ 10 (into D1 v2) | COVER-ATTN (new) | RT-PROTO |
| D2 `WorkSound` | WORKSOUND (new; takes DITHER-THM's branch) | RT-LEAN |
| `Extract` (commit-and-open), its Lean discharge from the random oracle | ABSORB-THM bc-2b4ffb7f | RT-LEAN |
| D3 model + θ | MODEL-SPEC bc-8e07fa19, CALIB bc-85ce2d11 | RT-MODEL, RT-UNITS |
| D4 naive vs Strassen | NAIVE-STRASSEN bc-96d70fa4 | RT-ALGO |
| D5 public best | BET-ALGO-LAB bc-f53e4d35 | RT-ALGO, RT-UNITS |
| Unconditional ring part (shrinks SAK's load; not on the critical path) | BET-K-DISCHARGE-2 bc-272d7c2a | RT-LEAN |
| A fold with a provable in-model minimum (fallback lever on γ_hash) | BET-FOLD bc-a34e3d35 | RT-ALGO |
| PoUW for Demo A (M3; separate path, unchanged) | J10R-CREDIT, POUW-LAND | — |

Fold workers FOLD-SAT, FOLD-LUCK and O-CHARGE are paused: under the no-fold design their results are a fallback. The
one computational slot `Pouw.Assumptions.SAKMatmul` moves to WORKSOUND (single owner of the headline).

## Red-team charters (each red team is independent of the builders it attacks)

- **RT-ALGO** (O-REDTEAM bc-3d215ce6 on its last turn, then a fresh RT-ALGO): beat the honest kernel on the frozen
  protocol's operands at θ with any exact method: Strassen and Winograd forms with overflow fix-ups, noise stripping
  (separating the data product from the noise terms), packing, Four Russians, CRT/Ozaki splits, digit tricks, choice of
  block pairing and K-permutation (free on fixed weights), precomputation on fixed weights.
- **RT-UNITS** (new): every unit on the H200 and the RTX PRO 6000 that does arithmetic or a reduction: tensor cores in
  every format (INT8, FP8, FP16/BF16, TF32, FP4/NVFP4/MXFP, b1) and the 2:4 sparse forms, DP4A/IDP, IMAD/IMAD.HI/IMAD.WIDE,
  texture filtering, L2 and shared-memory atomics and reductions, TMA bulk reductions, distributed shared memory in
  clusters, copy engines, NVLink/NVSwitch in-network reductions, RT cores (RTX PRO 6000), video and JPEG engines, clock
  limits, memory compression. For each: is it in the model with a rate, and can it compute any part of the committed
  words or hashes more cheaply than the model prices?
- **RT-PROTO** (new): protocol-level: the serving skip; the prover's own degenerate workload (A = B = 0, chosen
  shapes); precomputation before coins; reuse across units, periods and nodes; coin timing; gaming the stratified draw;
  lazy opening of Merkle commitments; period boundaries; network import and outsourcing; collusion across nodes; the LP's
  honest optimum (R-ACCEL-1); duplicate requests; decode-sized batches.
- **RT-LEAN** (new): `WorkSound`'s Lean statement against section 0: hidden hypotheses, vacuity (is the witness
  meaningful?), whether it quantifies over every program of `Accel`, the θ instantiation, whether its ε is the number
  table's, and whether the lock lists exactly the trust base and nothing else.
- **RT-MODEL** (new): θ against data sheets and microbenchmarks; missing resources or stores; P1 and P2 edge cases;
  `AccelProductForm` on each card; whether `Accel`'s programs cover everything the device can do (with RT-UNITS).

## Sprint rules

0a. **Your inbox is on a git branch, not in the notes repo:** `git fetch -q origin cursor/ca-research-inbox-ea24 && git show origin/cursor/ca-research-inbox-ea24:inbox/<YOUR-FILE>.md` (several workers missed it).
0. **Find the findings addressed to you at every checkpoint** (two builders missed theirs in the first round):
   `research notes sync && rg -n "(FINDING|RESULT) [A-Z]+[0-9]+ → [^:]*<YOUR-NAME>" ~/.research/notes/lanes/compute-research/`
   Red-team notes so far: `20261026T0310Z-draft-accel-rt-lean`, `20261026T0310Z-draft-accel-rt-model`,
   `20261026T0315Z-draft-accel-rt-proto`, `20261026T0320Z-draft-accel-rt-units`, plus ALGO-COST's once written.
1. **Statement first, then straight to its red team.** Red teams read builders' notes and branches directly; no relay.
   A red team writes each finding in its note as `FINDING <id> → <builder>: <one line>` and comments it on the builder's
   Linear issue (rule O prefix). The builder answers in its note within an hour: fixed (commit), refuted (why), or open.
2. **A break of the frozen protocol opens the next version at once.** PROTO-DESIGN owns versions; every builder states
   which version it is on.
3. **Turns are 30 minutes**, with a checkpoint line at least every 30 minutes; the coordinator resumes you. Write the
   note within your first 10 minutes.
4. **Decide yourself.** What a guarantee rests on (a claim) goes to the coordinator for Daniel, with your recommendation;
   keep working on the recommended branch meanwhile.
5. **CPU only before M3.** GPU items go in the coordinator's GPU queue for after Mon 23:00Z.
6. Times are Z. 06:30Z is 11:30 PM PDT; 12:00Z 5 AM; 15:00Z 8 AM; 17:00Z 10 AM; 19:00Z noon.
