# Inbox: ACCEL-WORKSOUND

## 03:20Z, coordinator: relays from O-REDTEAM pass 3 (note:20261026T0219Z-draft-accel-o-redteam-pass2, branch 82069feca3), BET-HYBRID-2 (6bf82586be), O-CHARGE (4e0e97ef6c), FOLD-LUCK (d66381f251)
- **FOLD-LUCK** (paused) left `Pouw.TheoremO.Slot` in `Security/Definitions/Pouw/TheoremO.lean` on `cursor/compute-research-accel-fold-luck-ea24` at d66381f251, built on MODEL-SPEC's `Machine`. It is a moment-form round slot with `Tail`, `TailOfSlot`, `HonestMeets` and `HonestTight`, unelaborated. It is yours to reuse for `SAKMatmul`'s round half, or to drop.
- **O-CHARGE (4e0e97ef6c):** the Merkle receipt needs extraction, either SHA-512 as a random oracle (its recommendation, matching ours) or SAK in commitment form. Its WRITES premise retires under `Extract`.
- **O-REDTEAM:** NF5, no deadline from coin release to root, means nothing ties the words to the device's window. The period's cut needs a deadline in the protocol: PROTO-DESIGN v1 adds it, and the statement reads it. NF4: hashing can move off-device to a host CPU, which the model doesn't price. Either the protocol binds hashing to the device, or the LP includes host resources.

## 03:25Z, coordinator: protocol v1 is frozen (PROTO-DESIGN, note:20261026T0133Z-draft-accel-proto-design, section "Protocol v1"; code nofold.py v1() at 1bfece4240)
HYB U ±80, V ±64, with no fold. Every 4-byte word is a SHA-512 leaf in a binary tree, and the served output is each cell's last committed word, so post-processing can't be skipped. e = 0.75%, ε_S = 0.1% (13,816 draws). L is 1280/2688 on the H200 and 640/1664 on the RTX PRO 6000. ε 0.85% (δ 1e-6) / 0.78% (δ 1%). Step γ 8.79 on the H200 (the K=16384 matmuls at 10.54) and 7.19 on the RTX PRO 6000. Coverage at 4k context 97.6% and 100%. Coins: a live seed expanded with SHA-512 in counter mode (a PRG use, named; going to Daniel with the Merkle random-oracle question). PROTO-DESIGN-2 (bc-2cdf8501, CMP-118) owns v2 (12:00Z) and D6 (17:00Z).

PROTO-DESIGN posted two findings on CMP-136 for you. W1: e (the serving-skip budget) is covered only if c_mm prices the public best for the whole transcript, the served output included; if it's per product, the headline needs +e as its own term. W2: h should be the priced share at θ, not a time share at η = 0.85. Answer both in your note. State the ε that PROTO-DESIGN-2's D6 uses.

## 03:35Z, coordinator: O-IO's result (note:20261026T0127Z-draft-accel-o-io, art:092f373e6da01e9fac8c436c20a8b2e19dbd3eb1ab812c0ab61f2c7753e74a5b: io_lp.py, candidate_lp.py)
O-IO's results:
- **Lookup routes at A = B = 0.** At A = B = 0, every design so far credits a little fresh randomness plus periodic noise. The adversary computes the periodic part once, and the fresh part from shared-memory lookup tables (Four Russians) on CUDA cores that full-rate GEMM leaves idle. These are in-model LP lower bounds on ε, not GPU runs:
  - today's Dither: 8.75% (data sheet) / 5.94% (with the shared-memory port);
  - HYB as specified: 1.84% / 1.65%;
  - v1-like no-fold HYB at the down shape (K 65536, L 3110): 1.33% / 1.06% against the best public algorithm, and 2.5–3.2% against the reference kernel. That's on top of v1's 0.85%.
- **Lemma S (proved on paper).** If the credited instances' distribution doesn't depend on the prover's choices (SYM), then ε_model = 0 against the best honest program, with no lower bound, ABSORB or fold needed. SAK(matmul) then only has to hold on uniform int8, the most-studied class.
- **P-SYM, the twin.** Credit a fresh PRF-uniform int8 GEMM RS (seed expanded with SHA-512) beside each useful GEMM, 1:1 in MACs. The useful GEMM is untouched, so there's no perplexity cost. γ as a ratio is about 2.1–2.5: the twin adds 1, the PRG 6460/n, plus the receipt. With a no-fold receipt, twin shapes can have large K, so attention's MACs get credited cheaply, and there's no serving skip. **Whether crediting a shadow GEMM instead of the useful product is acceptable is Daniel's call; it's asked.** O-IO (resumed) writes P-SYM as a full v2-alt spec by 08:00Z; PROTO-DESIGN-2 scores both.
- **P-HYB3:** HYB with 3-bit fresh dither leaves ε ≤ 0.63% against the reference kernel, with γ unchanged. The perplexity cost is unmeasured (21× dither variance), and its class is unstudied for SAK.
- **HelpWithinCertifier (help ≤ K/b) is false as stated.** About 1 Kbit through the certifier (seed out, root or receipt in) outsources a whole unit whenever the unit's inputs are seed-derived: the twin, or any design at the prover's own A = B = 0 workload. The provable form is per unit; bounding words needs timing or bulk coins.
- **The coordinator decided OPT_honest:** it's the best public algorithm (OPT*), as the agreed end state has it. The deployed kernel's gap from it is reported as its own term, ε_alg (0.53–1.76% for the reference kernel per O-IO).

Two consequences for the statement. (1) `ImportWithinCertifier` in the per-bit form (help ≤ βK) is false: state import per unit (units outsourced ≤ K/b_unit, b_unit ≈ 1 Kbit when inputs are seed-derived) and keep the bound's dependence on b_unit explicit. (2) Lemma S gives a second, very clean headline for a SYM design: if Daniel takes the twin, `WorkSound` for SYM is a coupling plus SAK at uniform int8. Keep the statement parametric enough to instantiate either.

## 03:45Z, coordinator: BET-HYBRID-2's large-L noise (note section K, head 3da2547c1a, largel.py; posted on CMP-120 and CMP-118)
Recommended for large L:
- u_X = u_W = 80;
- a noise period of exactly one segment (rank L per side, as the skip bound needs);
- U's columns freshly permuted every segment, which blocks O-REDTEAM's HY1 because the cross term never repeats;
- the cheapest Strassen-family method on paper is depth-1 Winograd at (7 + 0.5·8)/8 = 1.375× naive. X at ±64 fails (1.125); u_X = 72 is borderline (1.22);
- the served product is recovered exactly in largel.py's six cases (uniform, zero, HY1's A = 0 with periodic B; ±80 and ±127).

Honest cost beyond the hash: wraps (W on 2:4, +0.5 tensor; or both on the ALU, +0.22 issue / +0.43 pipe) and peels, γ_rm ≈ 6L/K (0.53 at L = 1450, K = 16384). These hide behind C/L under max-accounting, but add about 1 under additive accounting.

Open risks:
- HY4: Laderman rank 23 needs at least 19 summed operands to stay above 1.2, and the ⟨4,4,4⟩ rank 48–49 schemes are uncounted. The criterion: a scheme beats 1.2 if R + 0.5(s_A + s_B) < 1.2·mkn.
- 1.375 assumes the adversary's overflow fix costs at least 2:4's 0.5 per summed operand.
- The 70B densities at ±80 are missing.

HY2: wrap corrections and peels are serving work outside the words. Under R-ACCEL-1 they're other work, not credit, so skipping them isn't a soundness issue. Say so explicitly in the statement's docstring, or answer BET-HYBRID-2's smell that they may enter ε through the skip bound when tensor time binds.

## 03:50Z, coordinator: decisions on RT-LEAN's and RT-PROTO's first findings (note:20261026T0310Z-draft-accel-rt-lean, note:20261026T0315Z-draft-accel-rt-proto)
- **You own the adversary type** (L18). Make it per round: round w's dither coins, then the prover's work, then round w's commitment, then round w's check coins (RT-LEAN L1; RT-PROTO P8 on overlapping calls: order rounds per node, or state the cut per group of overlapping rounds). Push the type first, before 05:00Z, so ABSORB-THM writes `Extract` against it.
- **L2:** take `Extract` out of `SAKMatmul`'s `DirectSum` half. `SAKMatmul` speaks only of matrix products; extraction is its own named premise.
- **L3 and L5/L19:** tie c to the public best's priced cost (`pub/H`), read `ShaSpeed`, and add the share of honest work that is neither matmul nor SHA-512.
- **L4:** read MODEL-SPEC's `Accel` (branch `cursor/compute-research-accel-model-spec-ea24`), not a placeholder.
- **P2 (outsourcing) and P9 (host offload):** state import per unit, and make the period's devices include host CPUs and DPUs (P9's fix), priced in the model. Whether outsourcing is attribution only (the global-compute framing) or a break (the per-fleet framing) goes to Daniel. Keep both readings expressible.
- **P5:** a deadline from coin to root, and no coins near the period's end.
- **P6:** the credit form for batched decode (credit the product's MACs, not cost as served).
