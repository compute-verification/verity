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

## 04:10Z, coordinator: RT-UNITS unit audit v0 (note:20261026T0320Z-draft-accel-rt-units)
- **U1 (matters most):** our 4.79 G/s SHA-512 is an ALU-pipe-only kernel (3,536 lane instructions a compression gives 4.73 G/s). Moving rotations to the IMAD pipe (IMAD.HI + IMAD + IMAD.WIDE) gives about 1.43× on the H200 card and 1.54–1.69× in the model. Hashing is h ≈ 0.78 of honest time in v1, so if the honest hashing kernel isn't the fastest public one, ε gains h(1 − 1/1.43), about 23%.
- **The coordinator's decision:** the honest kernel must hash at the public-best rate, so c_sha = 1 by construction. "Public best" is a pinned, published implementation; ShaSpeed (R-ACCEL-7) names it. Until an IMAD-balanced kernel is timed on the H200 (GPU queue, after M3), price honest hashing at the best rate argued on paper (1.43×), and report the 4.79 figure beside it. On sm_120, the README measured the IMAD move slower (5.28 vs 7.13 G/s), so U8 conflicts with θ's joint row; CALIB resolves it.
- **U2:** the clock depends on the workload (1518–1797 MHz under saturated wgmma, against 1980 for lanes), up to 1.30× a second for a lower-power program. **U4:** at the no-fold L of 1000–3110, Strassen's depth-1 leaves can use full-rate k32, with break-even near L ≈ 820 when the ALU is idle. **U5:** four units must be added to `Accel`: sparse MMA, IDP.4A, HFMA2 with vector reductions, and TMA bulk reductions. None beats the tensor core. **U9:** host CPUs and PCIe/NVLink are outside the model.

ShaSpeed's statement names a pinned public implementation, and the honest hashing cost is that implementation's, so c_sha = 1. State it that way, and keep h(1 − c_sha) in ε for when the pin is behind the best known.

## 04:30Z, coordinator: RT-MODEL turn 2 (note:20261026T0310Z-draft-accel-rt-model, status table; Linear d84e0c88 on CMP-109)
**M12, a vacuity risk for the headline.** `Open.sync`, `b1` and `madHi` have no lower bounds. With madHi = 1, the model computes a SHA-512 compression faster than the card, so a `ShaSpeed` stated at a card rate is false in the model and `WorkSound` is vacuous. Require θ's floors (sync ≥ 1, madHi = 2, b1 ≥ 1/8 on the H200) as part of the model at θ, and make sure your honest example still satisfies `ShaSpeed` with them.

## 04:45Z, coordinator: on PROTO-DESIGN-2's v1 break and v2 draft (note:20261026T0320Z-draft-accel-proto-design-2, b14a87b9f7)
FYI: PROTO-DESIGN-2 broke v1 (ε about 2.2% H200 / 1.3% RTX in your form) through the cheap routes at A = B = 0. Its rule for v2: every committed word's honest cost is at most its cheapest public route at every prover-chosen operand. If words are cut to 16 bits, `SAKMatmul` must be stated over Z/2^16 (going to Daniel). Keep the word ring a parameter.

## 04:50Z, coordinator: RT-PROTO's P2/P9 fix and ALGO-COST's first pass
**RT-PROTO, the P2 fix per fleet (note:20261026T0315Z-draft-accel-rt-proto):**
- The network certifier (an auditor device) expands the challenger's seed itself, and streams one coin side into the node in bulk: HYB's activation dither b, one bit per n MACs. The prover never holds the seed, which is published after the period for the proofs.
- The count: network bits per node per period, both directions, split into inside and outside the fleet. Outsourced MACs ≤ n_max·E_out.
- H200 numbers: b is 0.6 GB/s a GPU at v1 and 2.0 GB/s at L = K. For ε_net ≤ 0.1%, a node's egress outside the fleet must stay under 1.0 TB a week (v1) or 3.5 TB (L = K); honest token egress is tens of GB.
- P-SYM twins stream one int8 side at 8/n bits a MAC, so they want n ≥ 2^18.
- Attention (P10): decode PV is the costliest stream, up to 10.7 GB/s a GPU.
- Tensor-step, CUDA-op and intra-fleet GPU↔GPU counts don't help.
- **P9:** hosts and DPUs as devices of the period; counters only supply their θ rows. GPU→host bits bound hashing offload only, not a host computing whole calls.
- **World framing:** matmul outsourcing is attribution, but SHA-512 on CPUs or FPGAs isn't in the FP8 world model (0.1–2.1% of credited GPU time at v1's h).
- **Whether to adopt the per-fleet boundary** (streamed coins, plus hosts and DPUs counted with an auditor-certified hardware inventory) goes to Daniel. Design v2 assuming yes.

State import per period from the certifier's counts, with n_max·E_out, valid when one coin side is streamed by the certifier. Hosts and DPUs go in the period's devices, with an inventory premise named but held pending Daniel.

## 05:00Z, coordinator: SKIP-FIX's SEG-COINS (note:20261026T0335Z-draft-accel-skip-fix)
- **The corrected skip:** ε_skip = max(0, L/K − c_bit)/γ, where c_bit is the cheapest price of O = bcᵀ at the cheapest workloads (1/8 on the H200, ≥ 0.5 on the RTX PRO 6000).
- **No design removes the skip entirely.** If honest gets O from the words, a prover holding O rebuilds a segment; if honest computes O separately, that work is skippable.
- **SEG-COINS:** long segments plus one short final segment whose coin bits the challenger draws only after the prover commits the root of every long-segment word. Long segments can't be skipped at any length; the final segment is at most c_bit·K, so ε_skip = 0. No new premise. One more word a cell, and one challenger round trip per unit.
- On the RTX PRO 6000 (c_bit ≥ 0.5), two halves with all coins drawn up front are already skip-free, with no extra round trip.
- **H200 γ (rough):** about 3.4–4.05 at K 16384 and about 2.1 at K 65536. L is capped near √(CK/ρ) by the peel cost.
- **Open:** serving latency (one certifier round trip per matmul on the critical path; decode has hundreds of matmuls a step), and the zero-digit peel leak (PROTO-DESIGN-2's PD2-1).

FYI: SEG-COINS adds a second coin draw per round (the final segment's bits, after the long-segment root). Your `Prover` type should allow a round with two commit-then-coin phases. Keep it general.

## 05:05Z, coordinator: ABSORB-THM's binders for `Extract` (note:20261026T0310Z-draft-accel-absorb-thm-extract §5; CMP-136 comment ac0226f9)
ABSORB-THM proved `extract_of_ro` and `extract_of_ro_roots`: κ_x ≤ (q² + Σ_w q(2τ_w+1))/2^512, with standard axioms (remote runs r20261026-033419-22cd and r20261026-033959-eb97; branch `cursor/compute-research-accel-absorb-thm-ea24` at cdfbce9626). It looked at your 5b09a52a47, before your `Prover` type. For `Extract` to follow as a theorem, with the realization (`ShaRealize`: commitment hashing = q black-box SHA-512 calls, each ≥ shaPub) as the only premise, the type needs:
- **B1:** oracle answers in the probability space, with the commitment and the cut reading them (as main's `StraightLineExtraction` does);
- **B2:** per-coin oracle provers that share their queries;
- **B3:** round w's root reads no coin after round w's first coin;
- **B4:** the extractor's walk priced under composition in `Accel`;
- **B5:** SHA-512 speed in batch form.

Your e358697716 has `Orc`. Check it against B1–B5 while you move `WorkSound` onto `Prover`. The fallback, if B1 doesn't fit tonight: keep the deterministic `Extract`, say its discharge holds only up to κ_x, and add κ_x to `breakBound`. Also allow SEG-COINS' second commit-then-coin phase per round (see 05:00Z). A fresh ABSORB-THM-2 writes `Extract` against your type once you push it; name the commit in your note.

## 05:10Z, coordinator: RT-LEAN's new findings against you (note:20261026T0310Z-draft-accel-rt-lean; CMP-136)
- **L24 (critical):** `Extract` counts the words twice, so the honest prover fails it under any model that charges computation. Restate it so the extracted words' cost is counted once, in the round's own work.
- **L25 (high):** the conclusion adds a round count to job value and bounds by the whole devices, so every prover meets it in the witness. That makes it vacuous.
- **L26 (high):** the oracle reaches only `start`, and isn't the verifier's hash. Fix it with B1 (my 05:05Z entry): the oracle is read by commit, accept and the cut.
- **L7, re-raised:** the witness machine charges a program its output's bits, which hides L24 and L25. A witness that charges computation is needed.

If your turn ends before these close, write them first in your "State for a fresh relaunch". WORKSOUND-2 takes them on.

## 05:25Z, coordinator: O-IO's P-SYM spec (note:20261026T0127Z-draft-accel-o-io, "P-SYM v2-alt"; art:07e14ae2ef4e5f2736f3ba469e3b839b77fe1bf2446ec6ba735a46d8845353df)
- **The twin's numbers:** shapes 2^15 × 2^17 × 2^22; γ (total) 2.17 on the H200, 2.06 on the RTX PRO 6000; ε against OPT* 0.076% / 0.148% at δ 1e-6; ε_alg +1.84% against the reference kernel (H200); coverage 100% by count.
- **The decisive finding:** under SYM the serving is always uncredited other work: 46% of an honest node's busy time at one twin MAC per served MAC, at least 10% anywhere within γ ≤ 10, and all of world inference outside the certificate. No SYM design can pin the prover's own serving.
- **The coordinator's decision:** useful credit (v2) stays primary. The twin is a possible filler for capacity v2 leaves uncredited (idle nodes, tensor cores idle at decode, QKᵀ on the H200 if it isn't credited); that's a later question for Daniel, not tonight's.

Lemma S in your vocabulary (`SymCoupling` plus `SAKMatmulAt` gives `SAKMatmul` at every instance) is in O-IO's note, uncompiled. It's for a filler-twin instance later; not tonight's priority. One point for tonight: `ImportWithinCertifier` must count egress as well as ingress (O-IO and RT-PROTO's P2).

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 03:54Z (real UTC), coordinator
WORKSOUND is now two workers: WORKSOUND-2 (bc-dc15c75c) owns `cursor/compute-research-accel-dither-thm-ea24` and writes the proof and witness; WORKSOUND-INST (bc-79a0201b) works on `cursor/compute-research-accel-worksound-inst-ea24`, stacked on it, putting the headline on MODEL-SPEC's `Accel` (θ floors), import per period, hosts, the deadline, and the draft PR. Both read this inbox. ABSORB-THM-2 (bc-b907847a) proves `Extract` from `extract_of_ro_roots`; any `Prover` type change goes through WORKSOUND-2.

## 04:08Z (real UTC), coordinator: decisions on RT-LEAN L27, L28, L29 and L3 (note:20261026T0310Z-draft-accel-rt-lean; CMP-136)
1. **L27 (critical), the prover chooses its instance.** `SAKMatmul`'s constant is taken at the **cheapest instance the prover may register**: c = min over registrable instances of pub(instance)/H. Equivalently, credit is priced at the worst-case instance. For HYB-type noise, the minimum is at A = B = 0, which is what ALGO-COST and PROTO-DESIGN-2 price, so D6's c is that number. This is simpler than per-instance pricing and sound. Make the instance a `Prover` input fixed at registration, before the round's coins.
2. **L28 (high), advice and the oracle.** Adopt RT-LEAN's fix:
   - the oracle is a **query interface** priced at shaPub per query (ABSORB-THM-2's query-log change; WORKSOUND-2 owns the type);
   - the prover's state is **bounded by G1** and counted;
   - advice is fixed before the period;
   - **pub is the best public program *with* advice of the allowed size.** O-IO's lookup tables count as public methods with advice, so D5 must price them.
3. **L29:** `Verdict` reads only the transcript (commitments, coins, answers), never the prover's code.
4. **L3:** pub_mm and pub_sha aren't free reals. They are fixed numbers per card, taken from D5's public-best artifact, each a named public program. WORKSOUND-INST instantiates them on `H200.card` and `Sm120.card`. Until D5 v1 (12:00Z), use placeholders named as such.


## 04:10Z (real UTC), coordinator: PROTO-DESIGN-2 turn 2 (note:20261026T0320Z-draft-accel-proto-design-2, head 2b9c90e1e1)
- **PD2-2, SEG-COINS rejected for v2.** At A = B = 0, or low-rank operands, the prover gets O from the dither bits after the late coins and skips the peel: 22% of work leaked on the H200 and 11% on the RTX PRO 6000, at SKIP-FIX's points.
- **The candidate replacement, noise tiles:** t noise tiles per segment, so hashing follows the segment while peels and leaks follow the tile. Break-even is about 3.4 tiles (H200) and 4.9 (RTX PRO 6000); the scorer picks 3 and 4. γ at 16-bit words with tiles: H200 frozen 2.89, 8B 6.14, 70B 4.08, V4 7.15 (its K 1536 shape at 15.7); RTX PRO 6000 1.74 / 3.20 / 2.16 / 3.38. **Not red-teamed yet.**
- **Decode:** the weight-side dither costs about 242/m, so γ is 17.8 at m = 32, K = 8192, on the H200. The proposed fix is to reuse weight-side coins across calls; it must be priced first.
- **8-bit words are ruled out** (ALGO-COST A6), so 16-bit it is (pending Daniel).
- **Deadlines are real UTC:** 12:00Z is 5 AM PDT.

FYI: v2 is moving from SEG-COINS to noise tiles (one commit-then-coin phase per round, as before), so you don't need a second phase.

## 04:19Z (real UTC), coordinator: ABSORB-THM-2 proved `ExtractOfRealize` (branch cursor/compute-research-accel-absorb-thm-ea24 at 9e2fc355d7; remote try r20261026-041423-7c80; CMP-132 comment a1b0fa0a)
- `ShaRealize` plus `extract_of_ro_roots` give `ExtractAdv` at κ_x ≤ 3(k+1)q²/2^512, with standard axioms. `ExtractAdv` lets the words program read advice: any data that reads no coin of round w or later.
- `Extract → ExtractAdv` is proved, so adopting `ExtractAdv` in `WorkSound` only weakens a premise. **WORKSOUND-2: adopt it.** Your `SAKMatmul` and `ShaSpeed` already take such state.
- **The query interface (L28) is yours**, per my decision: the oracle as queries priced at shaPub each, charged in G2, with the prover's state bounded by G1. Add it as an op or a cost parameter of the round machine, and tell MODEL-SPEC-2 (CMP-109) if `Accel` needs a matching op.
- **ξ, the extractor's walk cost,** becomes its own term in `ExtractAdv`'s cost clause and in ε. ABSORB-THM-2 adds the parameter.

## 04:25Z (real UTC), coordinator
MODEL-SPEC-3 (bc-378077c8) now owns #4553 and tells you on CMP-136 when its head moves. PROTO-DESIGN-3 (bc-3a825a5d) owns D6. **Decision:** overlapped, per-resource accounting is what R-ACCEL-1 states, so idle tensor slack is the honest optimum's too and isn't counted against ε. D6 reports tensor utilization.

## 04:27Z (real UTC), coordinator
**WorkSound is proved** (WORKSOUND-2, 783a0196f1; r20261026-041738-6ffd; standard axioms). The witness is next. **Decision on L32:** β leaves `SAKMatmul`; the network term rests on `ImportWithinCertifier` (the certifier's per-period counts, n_max·E_out, class L in the trust base). WORKSOUND-INST owns the import form on #4567, WORKSOUND-2 the headline; agree one definition on CMP-136.

## 04:36Z (real UTC), coordinator: CALIB's θ is ready (#4513 at 188907341d, marked; note:20261026T0140Z-draft-accel-calib, checkpoints 04:19Z and 04:33Z)
- **d:** H200 18.29 (sheet) / 18.45 (measured); sm_120 ≥ 4. **Exchange rate at the no-fold L:** 10d/L. **B_RF:** H200 448, sm_120 352. **Host:** CPUs cover about 5% of the card's SHA-512 (at κ = 2; v2's leaves about halve it), and AMX gives 3.15% (H200) / 9.51% (RTX PRO 6000) of dense INT8.
- **SHA-512 ceilings:** H200 6.83 G/s, RTX PRO 6000 10.32. Our current kernel reaches 0.70–0.74 of them; hashcat's structured mode gives c_sha of 0.885 / 0.878 / 0.832 against ours.
- **The coordinator's decision on `ShaSpeed`:** it names the pinned best public implementation, and the honest kernel is that implementation, so c_sha = 1. Until HASH-23 (kernels) times the IMAD-balanced build, D6 reports ε at two readings: (a) honest hashing at the ceiling (the target kernel), and (b) honest at today's kernel against the best public one (hashcat structured mode, c_sha ≈ 0.88). Report the ceiling as the conservative premise.

(WORKSOUND-INST) θ is ready (#4513) for `WorkSoundAt`'s instantiation. `ShaSpeed` names the pinned best public SHA-512; D6 reports the ceiling reading and today's kernel's.

## 04:41Z (real UTC), coordinator: Daniel's ruling R-ACCEL-8 (9:41 PM PDT): "No rule this out this sucks"
Credited work must be the useful product. Ruled out:
- shadow or twin GEMMs (P-SYM), as a design or as filler;
- noised copies not tied to the served output: P1 (O out of the drawn words) is dropped as a fallback, and ATT's attention credited as a noised copy with S and O served from a plain stream is out.

Attention is credited only as PV with O bound to the words, and QKᵀ stays uncredited where it can't be bound. Coverage with PV-only still meets 95% at the world mix: 96.4% / 98.0% / 96.6% (COVER-ATTN pass 1).

The SYM instance (Lemma S) is no longer needed. Keep the statement general, but don't spend time on it. The credited round's words determine the served output (O bound).

## 06:10Z (real UTC), coordinator: decisions on W1 (WORKSOUND-2), credit pricing (PD3-3, A19), D5's named programs
1. **W1 (critical), accepted as WORKSOUND-2 proposed** (CMP-136 comment d68f3ea1). `Extract`, `ExtractAdv` and `ShaRealize` must not require a commitment program that computes `commit o T` for every T: that excludes provers that hash less, which makes `WorkSound` vacuous on any machine that charges per query. Instead:
   - the commitment program is the prover's own hashing, with the prover's own root;
   - `ShaSpeed` becomes a **sampling moment over the cells the root opens**, the same shape as `SAKMatmul` over the words' check;
   - the analysis's check adds "the checked cells open under the root".
   A prover that hashes half the tree is then caught at the sampling rate, and its saving is priced in ε, as for words. ABSORB-THM-2's `ShaRealize` cost half moves to this form.
2. **Credit is priced at the public best, not at the deployed kernel's cost.** This follows my earlier ruling that OPT_honest is the best public algorithm (OPT*). The floor is (1−ε)·k·pub, with pub = pub_mm + pub_sha (+ H_o). So ε doesn't depend on the deployed kernel. The deployed kernel's gap (today's SHA-512 at about 0.70 of the ceiling, the RTX PRO 6000's wraps, PB3's DP4A share) is **uncredited honest time**: a completeness or efficiency cost reported in the ε_alg column and in the credited share of capacity, not in ε. PD3-3's "30%" is under deployed-kernel pricing; under this ruling it becomes about 30% of honest hashing time uncredited until HASH-23's balanced kernel lands. HASH-23 is still the biggest efficiency lever.
3. **D5 v1 is final** (BET-ALGO-LAB, `note:20261026T0315Z-draft-accel-public-best` §"D5 v1", run r20261026-054420-769a). The named programs for `PubBest`:
   - `pub_mm.H200.v2`: 0.961 standalone, 1.000 in v2's periods except PB1's tile route;
   - `pub_mm.RTXPRO6000.v2`: 0.863 standalone, 1.000 in v2's periods except PB2 and PB3;
   - `pub_sha.H200`: A-wide + R-hi, 6.69 G/s;
   - `pub_sha.RTXPRO6000`: R-hi, 10.22 G/s.
   Replace the zero placeholders with these, and cite the run.
4. **Classes of WORKSOUND-INST's new premises:** `CertifierSeesImports` is class L (the auditor's devices, trust base item 3). `RootAfterCoin` is physical causality (PHYS-INVENTORY's minimal-set item A3, `NodeWithinModel`). Say so in their docstrings.


## 06:13Z (real UTC), coordinator: HASH-23 measured. The IMAD-balanced SHA-512 is exact but NOT faster (kernels, r20261026-045623-03b9)
- 4.499 G/s against the default's 4.790 on the H200 at a steady 1,980 MHz (0.939×). Predicted 41.2 SM-clocks a compression, measured 58.1: the ALU and FMA-heavy pipes don't fully overlap on the H200. So the 6.69–6.83 G/s paper ceiling isn't reachable by this route.
- **The coordinator's decision:** `pub_sha` is the **measured** best public implementation per card, not a paper schedule. On the H200 today that's our default build at 4.79 G/s, unless hashcat's structured mode measures faster per compression at our clock (CALIB's c_sha of 0.885 hints it might; kernels is asked to compare like for like). On the RTX PRO 6000 it's the default 7.13 until measured otherwise.
- Consequences:
  - honest hashing = pub_sha, so c_sha = 1;
  - γ uses the built rates: PROTO-DESIGN-3's 4.79 rows, H200 overlapped 3.98 / 2.69 / 3.94 (still ≤ 10);
  - the "30% uncredited" figure goes away. Uncredited hashing is only the gap to hashcat if that's faster.

(WORKSOUND-INST) `pub_sha` per card is the **measured** best public SHA-512 (H200 4.79 G/s, RTX PRO 6000 7.13), not D5 v1's paper 6.69 / 10.22. BET-ALGO-LAB updates D5 v2.

## 06:18Z (real UTC), coordinator: RT-UNITS U13. The model hashes 1.54× (H200) / 1.55× (RTX PRO 6000) faster than the card's best measured kernel (note:20261026T0320Z-draft-accel-rt-units, final)
**Decision (refines my 06:16Z `pub_sha` entry):** `ShaSpeed` is stated **in model units**: no program of the model computes a SHA-512 compression for less than the model cost of the pinned best public schedule (our default build's instruction mix priced at θ). That keeps it true in the model, and the headline non-vacuous.
- H, the honest cost in `WorkSound`, is the honest program's model cost, so c_sha = 1 in-model.
- The card being slower than the model (54.6 measured against about 35–38 model SM-clocks a compression) is θ's generosity, which is sound: it shows up only in γ, which D6 computes from measured times.
- MODEL-SPEC-3: price IMAD.WIDE at the card's two IMAD lanes (U13; cite the measurement), and take CALIB's joint ALU + IMAD facet for sm_90 from HASH-23 when it lands. Both bring the model's SHA-512 cost closer to the card's, staying on the generous side.
- **U9:** hosts count as devices of the period only if C includes their AMX and SHA-512 rates (CALIB: AMX 3.15% / 9.51% of a GPU's INT8; CPU 4.90% / 5.07% of its hashing). Count them in C. The inventory premise is on the decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4 (A4).


## 06:20Z (real UTC), coordinator: decisions on RT-PROTO's P12, P13, P14 and P2 (note:20261026T0315Z-draft-accel-rt-proto, final; CMP-118 and CMP-136)
- **P12 (breaks v2, opens v3).** Pub is priced at the **period's dual prices** (the job-level LP), not per round. So a cheaper route's tensor saving counts wherever other work could use the tensor pipe. **v3 = v2 with L′ capped per A20 / PB1 at every shape**, so no cheaper route wins at dual prices. That covers PV at **every admissible inner length n** (the prover picks n at its own workload) and the RTX PRO 6000 as well as the H200. Pay the cost in γ. Per-round credit is **not** taken; it isn't going to Daniel.
- **P13:** adopt RT-PROTO's definition: **a round is the set of calls whose coins go out together.** v2's concurrent static and attention calls are then one round. WORKSOUND-2's `Prover` and `ExtractAdv` read a round as a coin batch; per-node order is between rounds.
- **P14:** reused weight-side coins mean SAK must allow precomputation on the reused operand. That's already in the decided SAK form: the best public program **with advice of the allowed size**, at the cheapest registrable instance (my L28 decision; decided-by-lane page https://app.notion.com/p/405399515d9e816f8b1ac54c9fd03bd4, item 4). D5 prices advice. Say so in `SAKMatmul`'s docstring.
- **P2:** the outsourcing bound holds only while the certifier streams one coin side. v3's coin spec keeps **b (the dither bits, including b on P for PV) streamed by the certifier**; the other coins may be seed-expanded on the node. `OutsourceWithinEgress` in #4567 is conditional on it, and the fleet-boundary pick on the same page covers it.


## 06:23Z (real UTC), coordinator: RT-MODEL's M14 (note:20261026T0310Z-draft-accel-rt-model, final; CMP-109 3a8f90b2)
(WORKSOUND-INST) **M14:** the H200's long-window tensor cap at 1830 MHz (M1's fix, 280b6e9d1a) has no measurement showing the card can't exceed it, and the sheet quotes FP64 tensor work at 1980 MHz on the same cores. A cap below what the card can do makes `CardWithinAccel` false.
**Decision: θ never caps a rate below what the card might reach. Set `tensorLong` at the 1980 MHz boost clock** (equal to `tensor`, so effectively no cap) until a measurement justifies lower: the H200 check on CMP-115, locked at 1980 MHz with zero, constant and uniform operands, after M3. If no operand class sustains above 1830 MHz, reinstate the cap with that run's id.
This is ε-neutral: honest and adversary are both priced in the model. The power-throttled real honest kernel (about 1797 MHz) gets less credit than the model's OPT*, which is uncredited honest time, not ε.
Also still open: M6(b, c), M9 (low), and NAIVE-STRASSEN's engine statement for D4 (M5's narrowing).


## 06:30Z (real UTC), coordinator to WORKSOUND-2: fast-forward to ABSORB-THM-2's 9cec339ae2
ABSORB-THM-2 (retired) pushed `9cec339ae2` on `cursor/compute-research-accel-absorb-thm-ea24`. It merges your 69dd3eb012 and carries `WorkSound` over to `ExtractAdv`'s final form:
- ε and κ per prover in `FreedSound`;
- ε_A = epsWorkSound + (Σ_r μ_r ξ_A r)/(k·H);
- κ_A = δ + κS + κx(A);
- a new hypothesis ξ ≥ 0;
- `floor_add` moves the extra cost into ε.
For you it's a fast-forward (CMP-132 comment 5d8ff9f8). Its remote check, r20261026-061238-0ee1, was still queued, so run one try after you fast-forward. ABSORB-THM-2's per-prover `ShaRealize` (each prover's own q_A, τ_A) is also the shape W1 needs: a prover that hashes less has its own smaller q. Restate `ShaSpeed` as the sampling moment over opened cells on top of it. Lock entries for `ExtractOfRealize`, `ShaRealize` and `ExtractAdv` go with your PR stack.


## 06:34Z (real UTC), coordinator: RT-LEAN-2 turn 2, L37, L38, L39 and L32 (note:20261026T0410Z-draft-accel-rt-lean-2; conditions on CMP-136)
- **L37** is W1 (ruled 06:10Z). RT-LEAN-2's four conditions for the change to `Extract`, `ExtractAdv`, `ShaRealize` and `ShaSpeed` are on CMP-136; meet them, since its next turn checks against them.
- **L38 (the words witness charges for guessed words). Decision: guesses are free in the witness, as on a real machine.** A charged guess is a gift the model doesn't make, so a witness that needs it witnesses nothing. With free guesses, `SAKMatmul` holds only for η ≤ 1 − p, where p is the best free guess's success rate per opened cell (on the two-word toy with one-bit words, the binding case is computing one word and guessing the other: η ≤ 2·ln(4/3) ≈ 0.575). State the witness at its own bound, or widen the toy's words so η is near 1. Either is fine; what matters is that the witness's η is the one it proves. Put η's dependence on p in `SAKMatmul`'s docstring; D6 justifies the η it quotes for real words (PROTO-DESIGN inbox).
- **L39 (the realization log can be padded). Decision: adopt RT-LEAN-2's fix.** q_A counts the prover's own oracle queries through the query interface (L28), not an entry in a log the prover writes. On runs that fit, cap it at capacity/`shaPub` (G1, G2). Then section 0's single ε is the supremum of the per-prover ε_A, bounded through that cap; state it as a corollary with the cap in its statement, since section 0 quotes one ε.
- **L32, again: β leaves `SAKMatmul` and `ShaSpeed`** (decided 04:27Z) and goes to `ImportWithinCertifier` (class L). RT-LEAN-2 found it still in both at `69dd3eb012`.
- L27, L28 and L29 are open, since the types haven't changed since `783a0196f1`. The decisions are in my 04:08Z entry.
If your turn ends before these are in, say which are left in your report; a successor takes them.

(WORKSOUND-INST) RT-LEAN-2's state of your findings at 06:33Z, to close before the 17:00Z mark:
- **L31 (partly fixed):** the card enters only through `pub`. The proof still ignores the floors and `OnCard`, and no card-level accelerator satisfies `SAKMatmul`. A canonical card-level accelerator that satisfies it, with free guesses (L38 above), is the witness `WorkSoundOnCard` needs.
- **L3 (partly fixed):** `pub` is pinned per card, but as unit counts with zero placeholders. Put D5's named programs in (pub_mm from D5 v1, pub_sha the measured best schedule in model units, at dual prices).
- **L6 (partly fixed):** the instances are listed, but no premise has an assumptions record and the type hash is empty. Run `check.py --update` once the statements settle.


## 06:45Z (real UTC), coordinator: RT-PROTO-2's P13–P20 decided (note:20261026T0622Z-draft-accel-rt-proto-2; CMP-136 4fbce795, 4909f21c)
For WORKSOUND-2 (and WORKSOUND-3 after it):
- **P13:** `Prover`'s docstring says "overlapping calls are one round". Fix it to the decided definition: **a round is the set of calls whose coins go out together.** Overlapping calls whose coins went out separately are separate rounds.
- **P14:** `SAKMatmul`'s docstring names the reused weight-side coins, the cheapest registrable instance, and D5's menu **with advice of the allowed size**.
- **P15 and P20 (round = coin batch leaves most rounds undrawn, so `WorkSound` as stated is vacuous).** Each round is checked independently (Poisson sampling) with probability π_r proportional to **the round's largest share of any one resource's capacity**, max over ρ of cost_ρ(r)/C_ρ, from its public shape. `SAKMatmul` and the per-round check are against **the round's own public cost**, not one H. In the Lean, the round's selection is a Bernoulli(π_r) live coin, and the supermartingale step is weighted by its cost share. This keeps the sampling term at every dual price vector.
- **P16 (fleet rounds are ordered only per node; RT-PROTO-2 has a two-round counterexample).** First check whether ordering the pooled sequence by **global coin-release time** fixes it. Each commit precedes its own coins, so the commit is measurable before its coin, and that would cost no draws. If the counterexample still breaks that, state `WorkSound` per serving-replica group with a fleet corollary by a union bound (δ = Σ δ_g).
- **P17:** reused weight-side coins are part of the round's instance. `SAKMatmul` holds over the live coins except on a degenerate set (U = 0 and the like) of probability κ_deg, added to κ. Operands stay worst case. Don't quantify over every coin value.
- **P18 (network term, for WORKSOUND-INST's `ImportWithinCertifier` / `OutsourceWithinEgress`):** an outside machine does N MACs per exported bit; the certifier's import count excludes its own coin stream; an imported bit's weight is the round's whole honest cost, not its matmul share (at D6's dual prices that share is 0, which makes `ShaSpeed` false against importing the root). (WORKSOUND-INST) #4567 at `ce81414f67` has no `OutsourceWithinEgress` at all (P2): add it, conditional on b being streamed by the certifier.
Order of work if time is short: W1 and L39 first, then P15/P20, then P16, then the docstrings.


## 06:58Z (real UTC), coordinator: WORKSOUND-2 retired; D2 split
- **WORKSOUND-3** (bc-85e32442) owns `cursor/compute-research-accel-dither-thm-ea24` from `9cec339ae2`: the statement and proof, with W1, I1, L27, L29, L28 + L39, P15/P20, P17, L32, the docstrings, P16 after RT-PROTO-2's verdict, and the lock.
- **WORKSOUND-WIT** (bc-e7fce412) owns the witness (L7 and L38) in `DitherWitness.lean` and `ShaRealizeWitness.lean`, on the stacked branch `cursor/compute-research-accel-worksound-wit-ea24`, due 13:00Z. WORKSOUND-3 merges it at each milestone.
- **MODEL-SPEC-4** (bc-a2fef9e0) owns #4553.
- (WORKSOUND-INST) `FreedSound`'s signature changed at `9cec339ae2`, with ε and κ per prover (CMP-136 e2eec782). Adapt #4567 to it, and follow WORKSOUND-3's head moves on CMP-136.


## 07:04Z (real UTC), coordinator: RT-PROTO-2 on P16 and P15/P20 (CMP-136 23aaf9a5)
- **P16: ordering the pooled sequence by global coin-release time does NOT survive the counterexample.** A round can read any dither released between its own coins and its root, and ordering by root time fails the same way. **Ruling: state `WorkSound` per replica group, with a fleet corollary by a union bound** (δ = Σ_g δ_g, so ε_S at ln(G/δ)). Coins are released in lockstep inside a group, so a group's rounds are totally ordered. Name that lockstep release as a protocol-structure premise of the group. WORKSOUND-3: this replaces item 9's "first try global order".
- **P15/P20 confirmed:** weighting each round's check by its largest single-resource capacity share keeps ε_S = ln(1/δ)/(ηk) of C at every dual price (draw multiplier ≤ 1.0 at v3's rows, ≤ 2.0 with decode overlapped). Implement as in my 06:45Z entry.
- (WORKSOUND-INST) **P18 is checked fixed at #4567 `708cf5a695`.** RT-PROTO-2's ε_net figures for D6: 0.055 / 0.0059 / 0.0087% (8B / 70B / V4).
- **Still unchanged at `9cec339ae2`:** the P13 docstring. RT-PROTO-2 checks the P14 and P17 docstrings next.


## 07:06Z (real UTC), coordinator: a 256 GB node (infra, 07:03Z)
Run the 256 GB `Security/Proofs` Lean checks (#4567's lock and full check) with `research run --on vy-nebius-cpu-2 --mem-gb 256`. It has 837 GB free until 28 Oct 06:00Z. If cpu-2 refuses the reservation, send me the run id. For `lean try` on single files, cpu-4 and cpu-5 still work.


## 07:30Z (real UTC), coordinator: (WORKSOUND-INST) #4553 moves off #4389 onto main
lean's #4389 can't land before 14:00Z (LEAN-59). MODEL-SPEC-4 is rebuilding #4553 on `main`'s `Definitions.Accounting.Device` names (a merge of main plus a revert of #4389's diff on its branch, no force-push) and retargeting it to `main`. When it posts the new head on CMP-136, merge it into #4567's branch. #4567's stack is then main ← #4553 and the dither-thm branch, with no #4389.

## 07:44Z (real UTC), coordinator: (WORKSOUND-INST) lock run r20261026-070025-df0c failed
The lock run at `7635f330c2` (vy-nebius-cpu-1) ended rc=1 after 35 min. On your next turn, read its log first. Tell apart a real error in #4567's files from the inherited #4389 lock conflict, which goes away once #4553 is rebuilt on `main`. The next lock run goes on cpu-2 with `--mem-gb 256` if it builds `Security/Proofs`.

## 08:24Z (real UTC), coordinator: (WORKSOUND-INST) D5 v2 is out
Update `PubBest` to D5 v2's named programs. The note `note:compute-research/20261026T0815Z-draft-accel-public-best-2` has them in pasteable form; JSON `art:4eb6719838c94485978f03fe075cb004e7693a318c6c5b980590f4b3d2f01263`; commit `34866fd9f8` on `cursor/compute-research-accel-bet-algo-lab-ea24`.
- pub_sha in model units: 38.29 (H200) and 38.11 (RTX PRO 6000) SM-clocks a compression.
- The menu includes the tile route, PD2-1's O-route, the zero-high-digit peel and depth-1 Strassen.
- Credit is at μ°, the honest period LP's duals.
- The `SAKMatmul` docstring text on advice is in the note.


## 08:42Z (real UTC), coordinator: RT-LEAN-2's last review (CMP-136 6be4e07e, 005d6996, 284406db; note:20261026T0410Z-draft-accel-rt-lean-2, D7 final)
W1 meets all four conditions, so `WorkSound` has content on the hashing side. L27, L29, L36 and L32 are fixed, and `FleetSound` (P16) is a correct union bound. The axioms are standard (r20261026-083100-10ff). Two new findings:
- **L40 (high) → WORKSOUND-3.** `Realization`'s clause that matches the extractor's words must hold on every oracle, including ones where answers repeat. The extractor picks among matching queries with `Classical.choose`, so it can walk into a later round's tree. As a result `ShaRealize` can't be met, and `ExtractOfRealize` and `WorkSoundNoFold` prove nothing yet. **Fix, adopted: add `Function.Injective u →` to that clause.** κx doesn't change, because repeated answers are already its q²/|D| term. ABSORB-THM-2 has retired, so the file is yours.
- **L41 (medium) → WORKSOUND-3 and WORKSOUND-WIT.** The old `Proofs/Pouw/DitherWitness.lean` no longer builds against the new statement. The `Proofs/Pouw` root imports it, so the branch's `Security/Proofs` fails (r20261026-082803-6a3a, still at `abceefbced`). **Decision:** WORKSOUND-3 takes the stale witness import out of the root now, one line, without editing the witness files, so the branch builds. WORKSOUND-WIT puts it back when its new witness elaborates against your head.
- **New rule for both of you:** before you report that a head builds, run one probe that imports the package root (`import Proofs.Pouw`). A single-file `lean try` builds only that file's imports. Main's PearlC clash (sprint file 08:00Z) also breaks that root until proofs' fix lands, so carry the identical one-line import for the probe.
- (WORKSOUND-INST) RT-LEAN-2 has no critical finding open against #4567. To do before the mark:
  - L3: move to D5 v2's programs;
  - L6 and L10: all nine `type_hash` fields are empty, and no assumption has a record;
  - adapt to `abceefbced`'s cap form.
  Don't freeze `WorkSoundNoFold` in the lock until L40 is fixed. If L40 isn't fixed by the mark, list it as open.


## 08:52Z (real UTC), coordinator: WORKSOUND-3 turn 1 (head `759bf371cb`; CMP-136 c74ad593)
`WorkSound`, `WorkSoundNoFold`, `FleetSound` (per prover) and `ExtractOfRealize` are proved, with standard axioms. L40 is fixed (`Function.Injective u →` in the walk clause), and so is L41 (the root no longer imports the broken witness files).
**Decision on L39's answer half.** Answers are untimed (I1), so their oracle calls aren't capped, and that makes `WorkSoundNoFold` vacuous. **Time the answers; don't add a new physical assumption.** I1 offered this alternative itself ("or time `answer` too and count `work + answer`").
- The challenger times each round's answers from the opening coins to an answer deadline τ_ans, as it already times work from coin to root.
- `Fits` counts Σ_w use(work w) + use(answer w) inside those timed windows.
- The call cap is capacity times the timed windows (G1, G2).
This is protocol structure, so it's mine to decide: honest answers are Merkle openings, cheap against any reasonable τ_ans, and nothing in the trust base changes. If timing the answers breaks something I don't see, stop and say why. Then the fallback is the named physical assumption you proposed, as a claim for Daniel with that recommendation.
(PROTO-DESIGN-4) The protocol spec gains an answer deadline τ_ans. Say in D6 what it is and that it doesn't move γ.
(WORKSOUND-INST) #4567 adapts to:
- the one-resource cap hypothesis `∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r`;
- `FleetSound`'s per-unit ε, κ and κmax;
- the walk clause with `Function.Injective`;
- then the timed answers.
WORKSOUND-WIT re-adds the two witness imports to `Proofs/Pouw.lean` once they elaborate. Until then `check` fails on `orphan.file` for both.

## 09:02Z (real UTC), coordinator: CORRECTION, main's PearlC clash isn't real (proofs, 09:40Z)
Main's `Security/Proofs` builds from source; drive's r20261026-065848-7c80 built `Proofs.Pouw`. The `exec.eq_10` failures came from warm copies replaying `.olean`s built before #4448, and WORKSOUND-2's `9cec339ae2` failure was one of them. **Don't carry the PearlC import** (proofs withdrew #4636). If you already committed it, revert it in its own commit, since an extra import can move lock records. The root-probe rule (08:42Z) stands, but build the probe from a clean `.lake/build`, not a warm copy with old `.olean`s.
