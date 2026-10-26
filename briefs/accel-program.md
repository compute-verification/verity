# ACCEL: a minimal accelerator model, and Dither's security under it

Program brief, compute-research lane (coordinator bc-ebc09a9a, handle @compute-research). Written 26 Oct 00:50Z. Every ACCEL
worker reads all of it, then does its own section. Rules: `briefs/brief-rules.md` on this branch (cloud overrides first).

## 0. The target end state (agreed with Daniel, 26 Oct 02:15Z; trust base restated 02:50Z with R-ACCEL-6/7, and Daniel's go at 02:51Z: "make sure we are pushing hard to achieve this goal -- I would love to be done by tomorrow ~noon, and it should be getting red-teamed in parallel")

The deliverable is one theorem in Lean, with nothing else in the lock. Every workstream works toward it. **The sprint to
Mon 26 Oct 19:00Z (noon PDT) is `briefs/sprint-mon-noon.md`: who builds what, who red-teams it, and when.**

**The trust base (R-ACCEL-4, -6, -7). These and nothing else:**

1. **The accelerator model at θ.** (G1) capacity, (G2) rate on every resource including issue, and the instruction set's
   documented exact semantics, from which P1 and P2 follow; the one named wiring fact is `AccelProductForm` (each
   tensor-core multiplier feeds one accumulator). θ is on the sound side: never below what the card can reach (data-sheet peaks at boost clock, or measured maxima), cross-checked by microbenchmarks with run ids. A cap below a documented peak needs a measurement (RT-MODEL M14).
2. **Standard cryptography.** SHA-512 collision resistance, and C-Flock sound and zero-knowledge (its named claims).
   SHA-512 as a random oracle only where a result names it and Daniel confirms that use (R-ACCEL-3).
3. **Live coins and the auditor's devices.** Uniform coins drawn by the auditor's challenger after the commitments they
   perturb; the network certifier on hardware the prover doesn't control (`trusted-nebius-host`).
4. **Symmetric algorithmic knowledge, two closely studied algorithms only.** (a) Matrix multiplication (R-ACCEL-6): no
   program computes exact matrix products more cheaply, per priced unit, than the best public method, including alongside
   other work. (b) SHA-512 (R-ACCEL-7): no one computes SHA-512 meaningfully faster than the best public implementation.

Nothing else: no circuit class, no "this fold is optimal", no empirical premise. ASSUMPTION-AUDIT's defect list is
empty. Off-matmul pieces (receipt, fold, lucks) are proved or designed away, never assumed (R-ACCEL-6).

**The guarantee (R-ACCEL-1).** For a fleet over a monitoring period \(T\), with \(C\) the fleet's capacity, and for
every prover strategy:
\[\Pr\big[\text{accepted} \wedge W_{\text{cred}} + W_{\text{other}} > \mathrm{OPT}_{\text{honest}} + \varepsilon C\big] \le \delta,\]
\[\varepsilon = \varepsilon_{\text{model}}(\theta) + \varepsilon_{\text{sample}}(\delta,k) + \varepsilon_{\text{net}}(K).\]
It is proved for the deployed protocol's Lean verifier (tied to the code by difftest) and instantiated at the H200's
and the RTX PRO 6000's θ.

**The adversary resource model: the world's compute.** The source is Daniel's 3-pager, "Computational integrity via
sampled zero-knowledge proofs" (Notion `3e6399515d9e80bb86a6fea6920829ec`), and its figure script; the census entry is
coming. The figures:

- all inference worldwide over a one-year horizon (to Sept 2027): \(R \approx 2.1\times10^{16}\) requests;
- DeepSeek-V4-Pro (49B active parameters), with about 4,500 tokens per request at 2 FLOP per active parameter per
  token, so about \(4.4\times10^{14}\) FLOP per request;
- **total about \(9.4\times10^{30}\) FP8 FLOP a year**, about \(1.8\times10^{29}\) a week;
- 1 H100-hour is \(7.1\times10^{18}\) FLOP (FP8 dense peak).

The base stated in the script is 30M H100-equivalents at \(2\times10^{15}\) FLOP/s, growing 3.3× a year. Every
result is stated against this model, for a week, a month and a year (rule L).

**Targets:**

| Quantity | Target |
|---|---|
| \(\varepsilon\) | ≤ 1% at \(\delta = 10^{-6}\), also reported at 1% |
| \(\gamma\) (honest serving overhead) | ≤ 10 |
| Coverage | ≥ 95% of the served workload's matmul MACs, static and dynamic (Daniel: "that's the goal for now") |
| Usefulness | perplexity < 5% for now, optimized last (Daniel) |
| Proving cost | not a constraint (R-ACCEL-5) |

**Demonstrated, beside the proof:**

- the served kernel meets γ, coverage and perplexity on real H200s;
- θ holds on the cards;
- every alternative algorithm we can build, priced in the model, loses to honest.

## 1. The goal (Daniel, 25 Oct; quotes are his)

"We need to develop an elegant, minimalistic accelerator model and we need to prove the security of Dither under it."
Success is three things:

1. physical assumptions about GPUs, "parameters of which are empirically estimated (e.g. starting just from datasheet
   speeds/capacities of things)";
2. a protocol: Dither;
3. "a proof that under (1) + standard cryptographic assumptions, (2) gives us (1-ε)-work-soundness over a week-long
   monitoring period", or over the compute of one week of global compute.

**Ruling, binding on every workstream: no algorithmic assumptions.** "The whole point of the physical model of the GPUs
is to restrict the circuit that the adversary actually runs ... We make no algorithmic assumptions. I just want to make
assumptions about the physical data flow between things." So no premise of the form "no algorithm does X in less than
Y". That rules out NaiveOptimal, an A_mm, the circuit class 𝒦 as an assumption, `ChargeVF3j16`, and a per-word work
premise. The lower bound has to be a **theorem inside the model**. The only trust left:

- that real GPUs obey the model at its stated parameters;
- standard cryptography: SHA-512 collision resistance, and C-Flock sound and zero-knowledge;
- live coins from the auditor-trusted challenger.

**The intermediate result he named:** "some GPU model under which we can prove that the naive tiled matrix
multiplication algorithm is faster than Strassen."

**Rulings, 25 Oct 5:59 PM PDT (Daniel):**

- **R-ACCEL-1, freed compute.** \((1-\varepsilon)\)-work-soundness with parallel resources means this: no prover strategy
  gets more credited work plus other work out of the period's devices than an honest prover could on the same devices,
  plus \(\varepsilon C\). Its dual is a price vector. Daniel: "sounds good but may need to revisit over time". Build on
  it, and keep the definition in one place so a later revision is one edit.
- **R-ACCEL-2, the protocol may change.** "Yes may change Dither for sure." Then: "we really need to choose some version
  of dither or maybe even something else such that we get provable security under our accelerator model." So the
  deliverable is a *co-designed* triple of model, protocol and proof. Every workstream that hits an obstacle proposes the
  smallest protocol change that removes it, with its honest overhead (γ) and its effect on usefulness (perplexity),
  rather than stopping. PROTO-DESIGN collects these proposals and ranks them.
- **Daniel also asked** whether Pearl-C gets provable security under the model. Today it doesn't: its γ rests on the
  named conjecture TT_OUT(ρ) over W1 programs, an algorithmic assumption. PEARLC-MODEL owns the question.

**Rulings, 25 Oct 7:00 PM PDT (Daniel):**

- **R-ACCEL-4, the only assumptions are the model's parameters.** "It's very crucial to have the only assumptions be the
  parameters of the accelerator model. Let's make this work and invest heavily in it if need be."
  - The trust base is the model's parameters \(\theta\), standard cryptography (R-ACCEL-3 included), and live coins
    from the auditor's devices.
  - Every other premise is a **defect to remove**: FOLD-OPT, FILL-OPT and ABSORB; H-μ⁺'s "errors can't be made to
    cancel"; 𝒦; `ChargeVF3j16`; WORD-WORK's per-word premise; TT_OUT.
  - ASSUMPTION-AUDIT keeps the burn-down list. A statement may hold a theorem-to-be slot inside the model, but never a
    premise about algorithms.
- **R-ACCEL-3, SHA-512 as a random oracle.** Daniel: "I could see this being potentially reasonable". A route may use the
  random-oracle model for SHA-512, but only as an explicitly named assumption. Say which results use it; Daniel
  confirms each one that depends on it. Where collision resistance suffices, prefer it.

**Ruling, 25 Oct 7:06 PM PDT (Daniel):**

- **R-ACCEL-5, proving cost is not a constraint.** "don't worry about proving, I know for a fact that won't be the actual constraint in deployment." Never trade coverage, ε or γ for proving cost. γ is the honest serving overhead.

**OPEN DECISION, asked of Daniel at 26 Oct 03:00Z (the barrier).** ABSORB-THM's Theorem B: over an instruction set containing LOP3, a proved "work ≥ c·n³" for the transcript is an explicit superlinear circuit lower bound, beyond known techniques. The options are (A) a narrow named computational assumption, cryptanalyzed like SHA-512's; (B) SHA-512 per word as a random oracle (γ ≈ 134–223); (C) credit only the provable floor (c ≈ 1/3 on the H200); (D) keep searching. The coordinator recommends (A) with (C) beside it. Until Daniel answers, every result is stated as its unconditional part plus the weakest assumption it needs.

**Ruling, 25 Oct 7:37 PM PDT (Daniel): R-ACCEL-6, SAK scoped to matmul.** The one computational premise is symmetric algorithmic knowledge for exact matrix products only, "because these are so closely studied". Everything that isn't a matmul (receipt, fold, ABSORB, lucks) must be proved or designed away, e.g. a SHA-512 commitment of every word at large L (γ_hash ≈ 6460/L, γ_rm ≈ L/K).

**Ruling, 25 Oct 7:46 PM PDT (Daniel): R-ACCEL-7, SHA-512 implementation speed.** The second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. Daniel: "sure". With per-word hashing, an adversary with a faster SHA-512 would free part of honest's hashing time, so this premise bounds that share.

**The adversary resource model, reconciled (coordinator, 26 Oct 03:55Z):** the 3-pager's 30M H100-equivalents at 2e15 FLOP/s is the base-date rate, 1.9e30 FLOP a year. Grown 3.3× a year for about 16 months to Sept 2027 (about 4.9×), it gives R × FLOP per request = 2.12e16 × 4.41e14 = **9.36e30 FLOP a year, the annual rate at Sept 2027: the figure of record**.

**Wording:**

- Don't use "completeness" for a model property; the word is overloaded.
- "Known semantics" and "one charge" are protocol-level, not GPU assumptions.
- Use LaTeX in anything written for Daniel.
- Say "network certifier", never "warden".
- Results are per monitoring period over all compute (rule L).


**R-ACCEL-8 (Daniel, 9:41 PM PDT 25 Oct, on "may PoUW credit a shadow GEMM instead of the useful product?"): "No rule this out this sucks."** Credited work must be the useful product. Ruled out:
- shadow or twin GEMMs (P-SYM), including as filler;
- crediting a noised copy that isn't tied to the served output: RT-PROTO's P1 fallback, and ATT's plain-stream attention.

Attention is credited only as PV with the served output bound to the words; QKᵀ is uncredited where it can't be bound. PV-only coverage meets 95% at the world mix: 96.4% (Llama-3.1-8B), 98.0% (70B), 96.6% (DeepSeek-V4).

## 2. The draft model (refine it; keep it this small)

A device is \(\mathcal G = (\Sigma, M, R, \rho, \mathcal I, u)\):

- stores \(\Sigma\) (registers and accumulators, shared memory and L1, L2, HBM) with capacities \(M_s\);
- resources \(R\) (each tensor-core format's pipe, each ALU pipe, each memory path) with rates \(\rho_r\);
- \(\mathcal I\), the card's whole instruction set, with each instruction's exact semantics and the units \(u_r(i) \ge 0\)
  it uses of each resource.

The model has two assumptions:

- **(G1) Capacity:** \(\mathrm{held}_s(t) \le M_s\) at every time.
- **(G2) Rate:** over any interval of length \(T\), \(\sum_{i \text{ run}} u_r(i) \le \rho_r T\) for every \(r\).

The **instance obligation** is that \(\mathcal I\) is the card's whole instruction set and that every parameter is at
most the data sheet's (or measured, with a run id).

Two data-flow principles restrict the adversary:

- **(P1) Exact products only through fixed operand formats.** Each multiplier unit has an exact domain: operand formats
  plus the range in which its accumulation is exact.
  - The INT8 path is s8×s8→s32 at rate \(\rho_8\).
  - FP16, BF16, TF32, FP8 and FP4 are exact only within stated ranges. `Security/Definitions/Core/TC/*` already
    transcribes BF16 wgmma accumulation exactly; reuse it.
  - A wider operand takes \(\lceil w/8 \rceil\) slices per side on the INT8 path.
- **(P2) A product lives in one accumulator.** Inside an MMA, each multiplier's output is summed into exactly one output
  element's adder tree. Getting an accumulator's value into any other word costs data movement on the register file,
  ALU or shared-memory resources, which is (G2) on those resources.

**The elegant target:** P1 and P2 aren't separate axioms but properties of \(\mathcal I\)'s semantics: an MMA's exact
domain, and its data flow as a reduction tree. Then the whole model is (G1), (G2) and a faithful instruction set, and
everything is parameterized by \(\theta = (M, \rho, u)\), with the H200 SXM and the RTX PRO 6000 (sm_120) as instances.
Keep P1 and P2 as named Props only where they can't be derived from the semantics.

## 3. What exists (read it, don't restate it)

**Dither.** Notion `073336e7f7c44d1cb16de39d55025860` ("Dither protocol"). Read the whole page, including top's 23 Oct
"Extra context" on the 𝒦-machine, the sm_120 machine \(B_\theta\), the `ChargeVF3j16` bridge and Theorem O.

- Protocol: \(\tilde A = A + R\) with \(R \in \{0,1\}^{n\times n}\), and \(\tilde B = B + S\) with
  \(S_{tj} = U_{j,\,t \bmod L}\) and \(U \in \{\pm1..\pm64\}^{n\times L}\), at \(L = 32\), \(A \le 126\), \(|B| \le 63\).
- Every 32-step prefix sum is committed through the duo receipt fold, and live draws pick the words to check by
  zero-knowledge proof.

**Theorem K and Theorem S (proved, within 𝒦).** Theorem K counts by mod-2 rank: each word's cross term has rank \(L\),
and a product of the covered forms contributes rank ≤ 1 (one-sided, \(p = 0.854\)) or ≤ 2 (affine, \(p = 3/4\)), under
purity rules. Theorem S pools over instances: \(D = \ln(1/\delta)/(1-p) \approx 190\) at \(\delta = 2^{-40}\).

- `Security/Proofs/Pouw/RelativeSecurity.lean`: `theoremK` (~l.2984), `theoremS_G` (~l.509) and `parity_luck_short`
  (~l.5712); from ~l.7087, Model B's `machine`, `price120`, `price120θ`, `gridMatMachine120θ` and `serviceFloor_θ`.
- `Security/Definitions/Pouw/RelativeSecurity.lean`: `OneSided`, `AffProd`, `RunMap`, `ServiceFloor`,
  `ServiceFloorPerProver` and `breakBound`.
- `Security/Models/Pouw/Assumptions/RelativeSecurity.lean`: `KScopeAffine` (~l.159), `KScopeOneSided` (~l.189),
  `ChargeVF3j16` (~l.207) and `ChargeVF3j16Overlap` (~l.339).
- `Security/Models/Pouw/Guarantees/RelativeSecurity.lean`: the floors, among them `StagedFloorKOneSided`.

**Device models.**

- `Security/Definitions/Accounting/Device.lean`, moving to Core as `Verity.Device` in #4389 (lean's One-vocabulary wave
  0, branch `cursor/lean-vocab-device-fc8d`). Build on it; don't fork it.
- `Security/Definitions/Accounting/TS/Sheet.lean`: the data-sheet machine.
- W1, `Security/Definitions/Pouw/PearlC/W1.lean`: the instruction machine with exact formats; #4448 adds `Fmt.s8`. W1
  makes oblivious moves free, which contradicts P2: that's the modeling change.

**Measured side.** `benchmarks/pouw/machine_model/` (the census of instruction prices, with the vector of record in
`data/model.toml`) and `benchmarks/census/hardware.json`.

**The deployed receipt.** `verity/core/protocols/pouw/circuit/segment.py` (`_duo_absorb`, `fold_final`, `fx1_key`) and
`benchmarks/pouw/served_h1/receipts.py`.

**Paper notes** on Theorem O and its attacks are in `~/.research/notes` (`research notes` CLI):

- `lanes/compute-accounting/20261015T1816Z-draft-rs-lean`
- `20261016T1135Z-draft-rs-p4-ldmatrix`
- `20261016T0713Z-draft-e2e-slice`
- `20261016T0101Z-draft-rs-d2-wide-peel`
- `20261015T2035Z-draft-cjl-round2`
- `20261016T0213Z-draft-stride-under-new-prices`
- `lanes/proofs/20261018T2245Z-draft-appendix-oblivious`

Grep the notes for "Theorem O", "CoincCharged", "P2-FF" and "RS-E-REF". Theorem O is stated as \(c \ge 1-\delta\) given
CoincCharged\(_\theta\), and \(\tfrac14\) without it. Find out what that \(\tfrac14\) is: it may be the output-bandwidth
bound in 4(g) below.

## 4. What the coordinator already worked out (start here; check it, don't rediscover it)

**(a) Strassen under P1, in the worst case over int8 inputs.** Each recursion level widens operands by a bit. Of
Strassen's 7 products, 3 are wide on both sides and 2 on each single side. With \(\lceil w/8\rceil\) slices per side,
the relative tensor-core cost at depth \(r\) is
\(E[s_A s_B]\,(7/8)^r\), which is

- 2.5 at \(r = 1\) and \((4 - 4(2/7)^r)(7/8)^r \ge 1.37\) for \(r \le 8\);
- above 1 up to about \(r = 16\) (a binomial estimate gives ≈ 1.03 at \(r = 16\)), and below 1 from about
  \(r = 17\), i.e. \(n \gtrsim 4\cdot 10^6\). The coordinator's earlier "≥ 1.06" assumed every product wide on both
  sides; compute it exactly.

That's before counting any addition. FP16 (11-bit exact, half rate) allows \(r \le 3\) at \(\ge 2(7/8)^3 = 1.34\), and
BF16 (exact for \(|x| \le 256\), half rate) allows \(r \le 1\) at 1.75. Check every exact path on each card:

- INT8;
- FP8, with Hopper's limited accumulation precision;
- FP16, BF16 and TF32;
- FP4/NVFP4 on sm_120;
- b1 and INT4 where the ISA still has them.

**(b) P1 isn't enough for Dither.**

- The adversary picks \(A\) and \(B\), and \(A = B = 0\) is admissible. The noise is narrow: \(R\) is bits, and \(U\)
  is 7-bit.
- One Strassen level on the cross term \(R\,U^\top\) fits INT8. The R-combinations lie in \(\{-1,0,1,2\}\) and the
  U-combinations in \([-128,128]\); +128 comes up about once in 16k entries and is fixed by a sparse correction. So
  depth-1 Strassen needs about 7/8 of naive's tensor-core work.
- What stops it is P2. Five of the 7 products feed two output blocks, so each level costs \(5n^3/(4L) = 5n^3/128\)
  extra deliveries, which is 1.25× the honest word count. A delivery costs 0.5–1 IADD3 (an IADD3 has three inputs).
  One ALU op is about 64 INT8 MACs of time on Hopper: 4096 INT8 MACs per SM per clock at the data-sheet clock,
  against 64 INT32 lanes.
- So depth-1 Strassen trades ALU for tensor-core time at roughly 10–19 MAC-times of ALU per MAC saved.

**(c) Linear receipts, checked.**

- duo is nonlinear in each word: \(u = y \oplus p \oplus K\), \(y \leftarrow uK_A + z\), \(z \leftarrow
  \mathrm{rotl}(u) + y\). So an adversary can't fold Strassen's products into the receipt without forming each word,
  and the P2 charge stands for duo.
- Check that xin, fx1 and li are likewise nonlinear in each word, and that the 8 chains can't be merged linearly.

- **Correction (02:32Z, BET-BANDWIDTH):** that nonlinearity forces materialization is a *claim*. As a premise, "every accepted run writes each committed word into a register" (ABSORB) is a statement about all programs. Every route that leans on it (P2's delivery charge, register-write bounds, a fold minimum) must prove it in the model or name it as a theorem-to-be.

**(d) Theorem O's engine, if every instruction obeys it.** Suppose a product contributes rank ≤ 1 to the mod-2
bilinear part of each of the \(f_p\) words it feeds. Each word needs rank \(L\), so \(\sum_p f_p \ge n^3\). With a
delivery price \(d\) in MAC units,
\[\mathrm{cost} = \#\mathrm{products} + d\sum_p (f_p - 1) \ge \min(1, d)\cdot n^3,\]
so \(c = 1\) whenever \(d \ge 1\). The research is making "≤ 1 (or 2) rank per word fed" true for every instruction
in \(\mathcal I\).

**(e) Known threats to (d).**

- *Bitwise ops are cheap mod-2 multipliers.* LOP3 does 32 bit-ANDs a lane, about 2048 per SM per clock against about
  4096 INT8 MACs.
- *IMAD packing defeats the mod-2 reduction itself.* Pack 32 R-bits into \(a\) and 32 U-bits reversed into \(b\): bit
  31 of \(ab\) is \(\sum_i r_i u_i\) plus carries, a whole word's mod-2 form from one IMAD. IMAD.WIDE reaches about
  1024 pair-terms. Shifts and bit extraction aren't ring homomorphisms mod 2, so Theorem K's technique stops at the
  ring ops. A potential that sees every bit of the exact word is needed, or a proof that the Boolean glue costs more
  than it saves: an exact 8×8-bit multiply in LOP3s is about 80 bit-ops, 160× a tensor-core MAC.
- *Concurrency.* Resources run in parallel, so Strassen's extra additions can run on an idle ALU while the freed tensor
  cores do other work. The honest Dither kernel is ALU-bound, which is what makes that trade cost wall time.
  **The figures are per device** (LIT's smell, 02:20Z):
  - The measured 1.47–1.55× plain INT8 with duo is the **RTX PRO 6000's**.
  - On the **H200**, data-sheet arithmetic (one ALU op ≈ 64 INT8 MACs of time, duo ≈ 3 ops per 32-MAC word) predicts
    ALU time ≈ 6× tensor time with duo. That means γ ≈ 5, with about 5/6 of the tensor cores idle.
  - The H200 headline design uses the LI fold, which O-REDTEAM reads as balanced against the census row JI90_0.
  - CALIB states r_H and the measured overhead per instance. Until then, quote γ and the idle share only with their device.
- *Packing* narrow products into one int8 lane with spaced accumulation.
- *Four Russians* for the binary \(R\): table lookups in place of MACs.
- *Reuse and pre-σ work.* \(AB\) can be computed once for a repeated \((A,B)\). Under rule L, the monitoring period
  counts all compute on every counted device, so pre-σ work counts. Say how work off the counted devices is handled
  (the network certifier's \(K\)).
- *Ballard–Demmel–Holtz–Schwartz:* Strassen moves fewer words than classical, so plain HBM or shared-memory I/O
  bounds favour Strassen. The leverage has to come from the accumulator level (P2), not from memory traffic.

**(f) What "freed compute" means** (DITHER-THM decides; the lane owns the credit form).

- With parallel resources, a run's time satisfies \(T \ge u_r/\rho_r\) for every \(r\). So for weights \(\lambda\) in
  the simplex, the priced cost \(\sum_r \lambda_r u_r/\rho_r \le T\), and a passing run \(X\) with concurrent work
  \(Y\) has \(\mathrm{priced}(X) + \mathrm{priced}(Y) \le T\).
- The coordinator's recommended definition: no prover strategy gets more credited work plus other work on the period's
  devices than the honest optimum on the same devices, plus \(\varepsilon C\).
  - Its LP dual is a price vector \(\lambda\). \(B_\theta\)'s ⅘·B + ⅕·R6 has this form.
  - Per-resource "tensor-core MACs used ≥ (1−ε)·credited" is the wrong notion, because honest Dither leaves tensor cores idle: about ⅓ on the RTX PRO 6000 (measured), and possibly far more on the H200 with duo (see 4(e)).
- Hypothesis to test: with duo forcing per-word ALU work, ε is governed by the honest kernel's ALU slack over the
  minimal per-word fold cost \((r_H - r_{\min})\), divided by Strassen's ALU-for-tensor-core exchange rate in (b).

**(g) A fully general but weak bound.**

- The transcript is \(n^3/L\) words of 32 bits, \(n^3\) bits at \(L = 32\), and duo reads each word from a register.
  So time ≥ \(n^3\) bits divided by the register-file write bandwidth from all units.
- A k32 INT8 MMA writes 1 bit per MAC of time. Small-k MMAs and the ALU pipes write more in aggregate, so this gives
  \(c \approx \rho_8/B_{RF}\), perhaps ¼.
- It covers every instruction with no algebra. Combining it with (d) is a candidate route.

## 5. Workstreams (one cloud worker each)

Every worker owns one Linear sub-issue of the ACCEL parent (ids in its launch message) and comments its status, run ids
and PRs there. A worker owns only its own files.

### MODEL-SPEC

Write the model as Lean definitions, statement first.

- **The definitions:** \(\mathcal G\), (G1) and (G2), with P1 and P2 derived from the instruction semantics where
  possible (§2). They go in `Security/Definitions/Core/` on #4389's `Verity.Device`: stack on
  `cursor/lean-vocab-device-fc8d` until it lands, extend Device rather than forking it, and reuse `Core/TC` for float
  exactness.
- **The instances:** the H200 and RTX PRO 6000 as data, with data-sheet placeholders until CALIB delivers.
- **A non-vacuity witness (rule I):** the honest naive tiled INT8 kernel satisfies the model on the H200 instance at its
  data-sheet rate.
- **W1:** say how W1's free oblivious moves map onto P2's priced moves. Don't change W1's readers in this PR.
- **Size:** count your assumptions. Each one must be a physical data-flow fact about the card, checkable against its
  documentation or a microbenchmark.

Statement pushed by **Mon 26 Oct 08:00Z**, then stop for the red team.

### NAIVE-STRASSEN

The intermediate result, paper first, then Lean.

- **The theorem:** in the model, for exact int8 matmul in the worst case over inputs, every Strassen–Winograd schedule
  of depth \(r \ge 1\) takes strictly longer than naive tiled on both instances, up to the stated \(n\). Any mix of exact
  paths and slicing counts; the bound comes from P1 and G2, before additions.
- **The generalization:** every integer-coefficient bilinear scheme \(\langle m,k,n\rangle\) of rank \(R\) (Laderman,
  AlphaTensor's, flip-graph schemes), as one inequality in \(R/(mkn)\) and the operands' coefficient ℓ1 norms.
- **Its limit:** say exactly where Dither's adversarial-data case escapes the theorem (4b). That case is O-CHARGE's.
- **Lean:** in `Security/Proofs` (or `Properties`) on MODEL-SPEC's definitions. Until those exist, use a minimal local
  structure that MODEL-SPEC's definitions will instantiate.

Paper by **Mon 10:00Z**, Lean statement by **Mon 16:00Z**.

### O-CHARGE

Theorem O by P2 charging: every program the model allows that produces Dither's transcript exactly pays at least
\(c\,n^3\) in priced MAC units, with \(c \ge 1 - \delta_O\), reducing to Theorem K's per-word rank count (4d).

- **Your part:** ring instructions (MMA by format, IMAD as a ring op, IADD3) plus deliveries.
- **Threats to answer:** 4b, 4c, 4e's concurrency, packing and Four Russians, and 4f's exchange rate.
- **Protocol changes:** if the bound needs one (a wider \(U\), a different \(L\), a different fold), propose the
  smallest.

Paper first by **Mon 12:00Z**, the Lean statement after the red team.

### O-BITS

The non-ring half of Theorem O: LOP3, SHF, PRMT, POPC, IMAD's high bits and IMAD.WIDE, BMSK, byte permutes, and table
lookups through shared memory.

- **The goal:** a progress measure that sees the exact word, not just bit 0 (4e's IMAD packing), together with a proof
  that each such instruction's progress per priced unit is at most the tensor core's. Or a counterexample, with its cost
  on the H200 instance.
- **The weak bound:** work out 4g in full, both as a fallback and as an ingredient.

Paper by **Mon 12:00Z**. You and O-CHARGE meet at one potential function; I relay.

### O-IO

An independent route: a pebbling or data-movement game whose innermost level is the accumulators (P2), with the register
file, shared memory, L2 and HBM as outer levels at G1's capacities and G2's bandwidths.

- **The bound:** the priced data movement any exact computation of Dither's transcript needs, compared with O-CHARGE's.
- **Mind 4e's last item:** classical I/O bounds favour Strassen, so the leverage has to come from the accumulator level.
- **The comparison:** if this game is the cleaner presentation of the model, which is Daniel's "physical data flow", say
  so.

Paper by **Mon 12:00Z**.

### O-REDTEAM

Break the model and the lower bounds.

- **The threats:** own 4b–4g and find more. For each, give a concrete exact algorithm, its priced cost on the H200
  instance under G1, G2, P1 and P2, and the principle it violates or exploits.
- **Checks:** small exact CPU checks run locally (under ~2 GB and ~10 min), and anything larger through `research run`.
  GPU measurements are batched on the H200 per rule G, only when a paper attack needs one; the demo has priority.
- **Reviews:** red-team MODEL-SPEC's statement and O-CHARGE's and O-BITS's papers as each lands (I relay them).

First report by **Mon 06:00Z**: the triaged threat list.

### CALIB

\(\theta\) for the H200 SXM (141 GB) and the RTX PRO 6000 (sm_120): every rate and capacity the model reads. Sources go
in order: data sheet (document and page), then census, then measured run id.

- **Rates:**
  - tensor-core dense rates per format and shape (k32 against k16);
  - ALU pipes per SM per clock (IMAD, IADD3, LOP3, SHF, PRMT, dp4a, FFMA);
  - register-file read and write bandwidth from each unit;
  - shared memory and L1, L2 and HBM bandwidth.
- **Capacities and clocks:** register, shared-memory, L2 and HBM capacities, and the clocks, saying which clock the
  data-sheet TOPS assume.
- **Exact paths:** which exist on each card (b1, INT4, FP4).
- **Derived values:**
  - \(d\), one delivery in INT8 MAC units, by each route;
  - the ALU-for-tensor-core exchange rate of 4b;
  - 4g's \(B_{RF}\);
  - honest duo's ALU ops per word \(r_H\), against your best minimal \(r_{\min}\).

The data goes where layout rule 9 says (census or machine_model), with README tables generated. First table by **Mon
06:00Z**.

### DITHER-THM

The final guarantee, statement first, with no algorithmic premise. Under (G1), (G2), P1 and P2 at \(\theta\), SHA-512
collision resistance, C-Flock's soundness and zero knowledge, and live coins from the auditor's challenger, Dither over a
monitoring period is \((1-\varepsilon)\)-work-sound with \(\varepsilon = f(\theta, \delta, k)\). The terms of
\(\varepsilon\) are:

- the model's lower-bound slack, from O-CHARGE's and O-BITS's \(c\);
- sampling luck, \(D/k\);
- work imported over the network, the network certifier's \(K\).

What the statement must do:

- **Freed compute:** decide and define it (4f), and say why.
- **Rule L:** state it at a week, a month and a year of an H200 fleet (the census), at \(\delta = 1\%\) and \(10^{-6}\),
  plus one week of global compute.
- **Theorem O's slot:** until O-CHARGE's statement exists, it holds a named Prop **inside the model**. Its docstring
  says it's a theorem to be proved, not an assumption. Never put a premise about algorithms there.
- **What it replaces:** 𝒦, `ChargeVF3j16` and WORD-WORK's per-word premise. Show the map from today's
  `StagedFloorKOneSided` and #4363's `H1CheckedCreditPeriod` to it.
- **A non-vacuity witness (rule I).**

Statement by **Mon 12:00Z**, then stop.

### LIT

Survey what bears on P1, P2 and Theorem O, saying what each result implies for us:

- Hong–Kung and red-blue pebbling; Irony–Toledo–Tiskin; Ballard–Demmel–Holtz–Schwartz, for classical and for Strassen;
- Raz's bounded-coefficient bound and the Bläser and Landsberg rank bounds (why rank alone can't give \(n^3\)), and the
  bilinear complexity of \(\langle n, L, n\rangle\) at small \(L\);
- Strassen on GPU tensor cores, and exact integer Strassen;
- the Ozaki scheme and INT8 emulation of FP64 (what operand splitting costs);
- accumulation precision: DeepSeek-V3 on Hopper FP8, and BF16 wgmma;
- Four Russians, AlphaTensor and flip graphs;
- proofs of useful work and memory-hard-function lower bounds in pebbling models, the nearest analogues of what we're
  proving.

A note with citations, plus a one-page "what this means for the model". By **Mon 10:00Z**.

### PROTO-DESIGN (added 26 Oct 01:10Z, after R-ACCEL-2)

Choose the protocol so that the lower bound is provable in the model: a Dither variant, or something else.

**The tension to resolve.** The noise has to be removable cheaply, which forces structure on it: Dither's rank-\(L\)
\(S = \Pi U^\top\) and its 1-bit \(R\), or Pearl-C's low-rank \(N = EF^\top\). Structured, narrow noise is exactly what
fast algorithms exploit (4b: \(A = B = 0\) leaves narrow operands, and one INT8 Strassen level fits). Pearl-C escapes by
putting the noise *inside* the FP8 quantization, \(\tilde X = \mathrm{fp8}(\alpha X + \beta N)\). Its operands are then
full-width at any \(A\). On the H100 its per-product truncation also breaks algebraic shortcuts, but on sm_120 the FP8 atom is the exact sum's truncation in about 99.97% of atoms, so only P1's operand widths protect it there (PEARLC-MODEL, 03:40Z). But it also defeats Theorem K's
rank counting.

**Levers:**

- \(L\);
- \(U\)'s range, and where the noise sits;
- the fold: its per-word nonlinearity, and its minimal ALU cost \(r_{\min}\), which by 4f may set \(\varepsilon\);
- bits committed per MAC, which strengthens 4g's general bound;
- integer exactness against bit-exact float;
- hybrids, e.g. integer exactness with full-width operands at every \(A, B\) plus a nonlinear per-word fold. If that's
  achievable at low γ, only O-BITS's non-ring instructions remain.

**Deliverable:** a ranked table of candidate protocols. For each candidate:

- the honest overhead γ, from data-sheet rates plus measured where it exists;
- its usefulness cost, as perplexity or output error;
- which threats of 4b–4g it removes;
- which proof route it enables, and what's still unproved;
- the smallest change from Dither or Pearl-C.

Work from the other workstreams' obstacles as they report; I relay them. Paper by **Mon 12:00Z**, updated as reports land.

### PEARLC-MODEL (added 26 Oct 01:10Z, Daniel's question)

Does Pearl-C get provable security under the model?

**Read first:**

- `verity/core/protocols/pouw/schemes/pearl_c.py`'s docstring: the forming, the HOPPER_E4M3_K32 chains, the peel, the
  leaves and the tickets;
- `Security/Models/Pouw/Assumptions/PearlC/` (TTOut) and `Security/Definitions/Pouw/Assumptions/PearlC/TTOutW1.lean`,
  where γ rests on TT_OUT(ρ) and "nothing here is proved by counting";
- W1 (`Security/Definitions/Pouw/PearlC/W1.lean`: the whole sm_120 instruction machine with prices; moves are free,
  against P2);
- the Pearl paper and its own security claim.

**Questions to answer:**

1. Which part of TT_OUT the model (G1, G2, P1, P2 at θ) discharges, and which part is still a statement about all
   programs.
2. P1 for FP8 atoms. Is the FP8 tensor core the only unit with Hopper's and sm_120's exact E4M3 atom semantics
   (per-product alignment truncation, FP32 promotion)? What does emulating it cost on every other exact path?
3. A proof route that needs no rank argument: data flow, or the dependency of each atom on its own products' truncated
   values.
4. Which threats of 4b–4g vanish for Pearl-C (narrow noise, Strassen), which survive (bitwise glue, concurrency, cheaper
   tickets or leaves), and which are new (FP exactness windows, the peel).
5. A recommendation, Dither against Pearl-C against a hybrid, for "provable under the model".

Paper by **Mon 12:00Z**.

### Parallel bets (Daniel, 26 Oct 01:05Z: "Let's take a bunch of parallel bets and see what pays off.")

Each bet is one worker that pursues one concrete protocol-and-proof path end to end. Every bet has a hypothesis, a **kill
criterion** (what would show it can't work, so we learn fast) and a deliverable due **Mon 26 Oct 12:00Z**. PROTO-DESIGN
scores the bets against each other. A bet that's killed reports why: that is a result too.

- **BET-HYBRID** (CMP-120).
  - *Hypothesis:* an integer protocol exists with three properties. (i) Every tensor-core operand is full-width at every
    admissible \(A, B\), so a Strassen-level combination overflows int8 with high probability. (ii) The noise is
    removable for at most about \(0.1\,n^3\) extra MACs. (iii) Each word's cross term is still a rank-\(L\) mod-2 form in
    fresh coins, so Theorem K and P2 charging apply.
  - *Directions to start from:* per-segment fresh noise, mod-\(2^8\) wraparound noise, structured noise on both sides,
    and splitting an operand.
  - *Kill:* a proof that (i), (ii) and (iii) are incompatible.
  - *Deliverable:* a spec, the paper argument, a CPU reference (python prover and verifier), γ, and a usefulness plan.
- **BET-PEARL-I8** (CMP-121).
  - *Hypothesis:* put Pearl-C's noise inside an **INT8** quantization, \(\tilde X = q_8(\alpha X + \beta N)\) with
    \(N = EF^\top\) low-rank, keep exact s32 accumulation, and add a pinned integer peel. The operands are then
    full-width at any \(A\), arithmetic stays exact, and no float accumulation subtleties arise. The lower bound reduces
    to exact segment sums of a product of two pseudorandom full-width int8 matrices. NAIVE-STRASSEN's bound holds there in
    the average case, so the gap left is only the non-ring instructions.
  - *Kill:* the integer peel can't be exact and cheap, or the model allows a cheaper exact algorithm on pseudorandom
    full-width int8.
  - *Deliverable:* a spec, a CPU reference, γ, and a perplexity plan.
- **BET-BANDWIDTH** (CMP-122).
  - *Hypothesis:* a protocol whose honest run is both tensor-core-bound and register-write-bound, because it commits more
    bits per MAC (smaller \(L\), or more words). Then 4g's fully general output-bandwidth bound plus the tensor-core rate
    give \(c \approx 1\) under R-ACCEL-1's LP, with no rank argument.
  - *Kill:* it needs \(\gamma > 10\), or the LP shows tensor-core savings still pay.
  - *Deliverable:* the paper, plus numbers at H200 rates.
- **BET-K-DISCHARGE** (CMP-123).
  - *Lean first.* On MODEL-SPEC's definitions, or a local stand-in, prove that every program in the ring fragment of the
    ISA pays at least Theorem K's count, reusing `theoremK`. The ring fragment is MMA by format, IMAD and IADD3 as ring
    operations, and moves priced by P2. That discharges `KScopeOneSided` and `ChargeVF3j16` for ring-fragment programs.
  - This is O-CHARGE's claim pursued in Lean, in parallel. It owns its own module.
  - *Kill:* a ring-fragment program that's cheaper. That would be a counterexample for O-CHARGE too.
  - *Deliverable:* the Lean statement, then a stop for the red team.
- **BET-COVERAGE** (CMP-124).
  - *Hypothesis:* a tile and commitment design credits at least half of the frozen shape's served INT8 MACs on the H200 at
    \(\gamma \le 10\), within C-Flock's limits. The candidates are receipts folded per word with SHA-512 only per
    receipt, large-K tiles, and per-call noise (J10-R's `PouwH1Tile_v5/v6`).
  - *Context:* today's H1 tile, at 17 compressions per 2,048 MACs, credits 0.075–0.15% (CMP-107).
  - *Kill:* a hashing floor per credited MAC that exceeds the budget for every design within C-Flock's limits.
  - *Deliverable:* a table, plus a recommendation.
- **BET-ALGO-LAB** (CMP-125; relaunched 02:08Z under this name).
  - *What it is:* an algorithm-cost lab. Implement in exact CPU emulation the reference Dither kernel and the cheaper-looking alternative exact algorithms for the same transcript: depth-1 Strassen on narrow inputs with sparse correction, packed integer multiplies, bit-sliced arithmetic, Four Russians tables, and alternative receipt-fold implementations.
  - *Method:* count each algorithm's instructions by model class, and price them at data-sheet rates.
  - *Deliverable:* the code, the counted costs, and each algorithm's cost relative to the reference. CPU only.

### Assumption removal (added 26 Oct 02:40Z, after R-ACCEL-4)

- **ASSUMPTION-AUDIT.** Inventory every premise in the chain, from the deployed Dither/H1 (and Pearl-C) to the period
  guarantee and JointHorizon's PoUW part:
  - the floors and their premises (`KScope*`, `ChargeVF3j16`, `ServiceSoundStaged*`);
  - #4363's `H1CheckedCreditPeriod` premises and CMP-127's J10-R ones;
  - H-μ⁺;
  - the honest-design TOML's FOLD-OPT and FILL-OPT;
  - ABSORB;
  - WORD-WORK's premise;
  - TT_OUT.

  Classify each as model parameter, standard crypto (random oracle per R-ACCEL-3 marked as such), live coins or trusted
  device, protocol-code fact (provable from code), or **defect** (an algorithmic or other premise). For each defect,
  name the workstream that removes it, or say "unowned". Keep the table current as workers report.
- **FOLD-SAT.** Mechanical lower bounds by SMT superoptimization over the GPU's exact instruction semantics: LOP3,
  IADD3, IMAD with .HI and .WIDE, SHF, PRMT and the rest of what's relevant. The targets:
  - the minimal instruction count, and issue-slot count, of one fold step for li, duo and BET-FOLD's candidates;
  - the same from split-word inputs, word = a + b, which puts ABSORB in cost form;
  - amortized multi-word steps.

  This makes \(r_{\min}\) a checked fact about the ISA, not an assumption.
- **ABSORB-THM.** Prove in the model that any program producing the receipt pays at least \(r\) per word for absorbing
  it: the cost form of ABSORB, from the fold's algebraic structure (dependency, fan-in, degree). Do it for li, duo and
  BET-FOLD's candidates. The random oracle (R-ACCEL-3) is an optional route, via fold keys or blocks through SHA-512,
  but hashing every word costs γ ≈ 6455/L, which is too much on its own.

## 6. Coordination

- **Red team:** I red-team every statement before its proof (rule 6). Report to me in your final message, and I relay
  between workstreams.
- **Files:**
  - MODEL-SPEC owns the new module in `Definitions/Core`;
  - DITHER-THM owns its guarantee module;
  - NAIVE-STRASSEN, O-CHARGE and O-BITS own their Proofs or Properties modules;
  - nobody edits another's files;
  - notes go in `lanes/compute-research/`, as `<UTC>-draft-accel-<name>.md`.
- **Program targets:**
  - the model and naive-vs-Strassen proved, in the lock, by Tue 27 Oct;
  - Dither's theorem stated, with Theorem O's slot named, by Mon 18:00Z;
  - Theorem O argued on paper, with the red team's verdict, by Tue 27 Oct 12:00Z.

  The 24-hour demo (P-CMP-1) keeps priority on shared machines.
