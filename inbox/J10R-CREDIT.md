
## 26 Oct 02:02Z, from the coordinator: the draw is the sampling block's (Daniel, 6:36 PM PDT, via top)
- PoUW's draw isn't special. The v5/v6 tile units go into the sampling block's stratified draw as one more subcircuit, with their ANDs in the work half. The planner sets their k under the 1% proving budget. There is no hand-set K′ = 22 at ε = 1/2.
- proofs recommends that the tiles join the integrity registration: one sampling block and one draw, so LEAN-56's single stratified bound (`JointStratified`) covers them with no new form. The tiles must be registered, circuit-checked Definitions of at most k_log 27. compute-accounting and proofs confirm on CMP-126.
- **For you:** state the strata DrawIsCoin with k_s read from that draw, not as a parameter you pick, and make it compose with LEAN-56's `JointStratified`. I'm asking lean for the interface, and I'll relay it. PoUW's number is the tile stratum's `eps_at` (ε at δ) from the same draw.
- Serving overhead (commits, per-call noise) is a separate number from the 1% proving budget. Never fold it in.
- Wording: don't say "leg" (Daniel: "it's polluting everything"). Say "protocol", or the protocol's name (PoUW, sampled proofs, ...).

## 26 Oct 02:12Z, from the coordinator: lean's interface for the stratified draw (lean, 02:25Z in thread 1792970372.621909; final names confirmed when STRATJOINT/LEAN-56's statement is up, by Mon 06:00Z)
1. **A stratum's draw:** `FlockSoundness.Audit.stratified σ k hk : Law n`, in `Security/Proofs/Flock/Soundness/Audit/Stratified.lean`.
   - `σ : Fin n → Fin m` puts each unit of the sampling block in its subcircuit's stratum.
   - `k : Fin m → ℕ` is the per-stratum draw size, with `hk : ∀ s, k s ≤ (stratum σ s).card`.
   - The population is `stratum σ s`.
   - The executable side is `FlockSoundness.PlanDraw.Stratum` (name, units, k).
2. **What JointStratified consumes:** no per-stratum guarantee. It takes one budget, Σ_s `stratumRisk σ k s x_s` ≤ ln(1/δ), and `stratified_isHarmBound_exact` (`Audit/HarmBound.lean`) gives the harm bound. PoUW's tile number is the tile stratum's risk at its k_s, `stratumRisk σ k s_tile c`, read from the same draw.
3. **For you:** state J10R's DrawIsCoin over `stratified σ k hk`, with the tile stratum as one s. Don't restate the joint bound. lean has asked STRATJOINT to export a per-stratum projection lemma (one stratum's escape from the joint draw) for PoUW to cite. If it isn't there yet, name the projection you need as a lemma-to-be in your statement, and say so in your final message.

## 26 Oct 02:40Z, from the coordinator: two more rulings from Daniel (25 Oct 7:00 PM PDT); brief section 1 updated

- **R-ACCEL-4.** "It's very crucial to have the only assumptions be the parameters of the accelerator model. Let's make this work and invest heavily in it if need be."
  - The trust base is the model's parameters θ, standard crypto, and live coins from the auditor's devices.
  - Every other premise is a defect to remove: FOLD-OPT, FILL-OPT, ABSORB, H-μ⁺'s cancellation premise, 𝒦, ChargeVF3j16, the per-word premise and TT_OUT.
  - A theorem-to-be slot inside the model is fine. A premise about algorithms never is.
  - ASSUMPTION-AUDIT keeps the burn-down list. Send any premise you introduce or find to the coordinator, and say which class it's in.
- **R-ACCEL-3.** On SHA-512 as a random oracle, Daniel: "I could see this being potentially reasonable."
  - You may use the random-oracle model for SHA-512, but only as an explicitly named assumption.
  - Say which results use it; Daniel confirms each one that depends on it.
  - Prefer collision resistance where it suffices.
- **New workers:** ASSUMPTION-AUDIT, FOLD-SAT (SMT superoptimization lower bounds for fold steps over the exact ISA) and ABSORB-THM (the absorption cost per word, proved in the model).

## 26 Oct 02:55Z, from the coordinator: BET-COVERAGE's result (note compute-research/20261026T0136Z-draft-accel-bet-coverage; branch cursor/compute-research-accel-bet-coverage-ea24 at 114ce998dd)

- **The bet is alive, on projection.** The best coverage at γ ≤ 10 without a new claim is **about 73%** of the frozen shape's static INT8 MACs on one H200. This is a model projection, not a measurement. The design is v5/v6 (J10-R, per-call noise) on 32×32 units at full K, run as one stream, with the noised weight W′ = B + c + U formed once per call (a kernel-only change). v5/v6 as built projects to 65%. The one compute-bound H200 measurement (an older, unsound J10-R kernel) gives about 50%.
- **Hashing isn't the limit.** SHA-512 costs 5.70e-6 compressions per credited MAC (1.2 H200 MAC-times). The binding cost is the units' integer work, the fold and the B unpack, at **16.5 MAC-times per MAC**.
- **Killed by hashing:** per-unit noise (at most 25%), the pinned v1 tile (0.4%), operand rows committed per tile, and single lane-chunk draws (under 40%).
- **Higher coverage needs Daniel.** Folding at L = 64 reaches 94–100% but needs a **new ReceiptLuckAt claim**. **The li fold rests on KLI₂₄, an empirical premise. That's a defect under R-ACCEL-4.**
- **Proving is the real constraint.** ε < 1% needs 3,857 sampling blocks a period (δ = 1%) or 11,571 (10⁻⁶). That is 3,628 or 10,883 process-days of proving a period, scaled from one card's model, or about 518 processes for a week. Chunking units at K = 4,096 cuts it 5.6× and costs 14 points of coverage.
- **Gaps for M3 (compute-accounting's).** `h1zy` has no J10-R lanes, so a full-K v5/v6 unit (6e9 ANDs) can't be proved inside C-Flock's 2^27 limit. A verifier `range` equal to the workload's unit count would reveal the batch size M; it should be a fixed, padded 2^27 with filler units and a public bound on ρ.
- **A smell for PROTO-DESIGN.** Its "100% at γ ≤ 10" rests on the machine model, not measured units. Measured units run at 0.45 of BET-COVERAGE's floor, which is already 1.4× the machine model's figure.

**For you:** Two inputs for your statement. The verifier's range should be a fixed, padded 2^27 with filler units and a public bound on ρ, so it doesn't reveal M; state ρ against that public bound. And `h1zy` has no J10-R lanes yet, so a full-K unit exceeds 2^27. Say which unit size your statement is at.

## 26 Oct 03:02Z, from the coordinator: R-ACCEL-5, proving cost is not a constraint
Daniel (25 Oct 7:06 PM PDT): "don't worry about proving, I know for a fact that won't be the actual constraint in deployment."
- Drop proving cost from scoring and from every limit.
- Never trade coverage, ε or γ for proving cost. For example, don't chunk units smaller just to prove faster.
- γ means the honest serving overhead only.
- You may still report proving cost as information.

## 26 Oct 02:20Z, from the coordinator: the target end state is agreed with Daniel (brief section 0, new)
- Read `briefs/accel-program.md` section 0. It holds the trust base, the guarantee, **the adversary resource model (the world's compute: about 9.4e30 FP8 FLOP a year, about 1.8e29 a week)** and the targets: ε ≤ 1% at δ = 10⁻⁶, γ ≤ 10, coverage ≥ 95% of all matmul MACs, perplexity < 5% for now (optimized last), and proving cost not a constraint.
- State every figure against the world model, for a week, a month and a year. A datacenter or single-card figure goes beside it, never instead of it.

## 26 Oct 02:40Z, from the coordinator: owners and heads
ASSUMPTION-AUDIT assigns you UnitFixed: change its definition inside the J10-R statement so it's provable from the code. Also, StagedPins relates two descriptions of one circuit, which main's 25 Oct one-description rule forbids, so state StagedPins at J10-R without a second description.

## 26 Oct 03:00Z, from the coordinator: the barrier, an open decision with Daniel (ABSORB-THM, note compute-research/20261026T0215Z-draft-accel-absorb-thm; O-BITS, note compute-research/20261026T0152Z-draft-accel-o-bits)

- **Theorem B (ABSORB-THM; O-BITS agrees).** The instruction set contains LOP3, which computes any 3-input Boolean function. So a proved bound of "work ≥ c·n³" (or r per absorbed word) for computing Dither's transcript from (A, B, σ), over the full instruction set, would be an explicit superlinear Boolean circuit lower bound, beyond every known technique. This hits Theorem O, O-CHARGE, ABSORB and FOLD-OPT alike.
- **What is provable from the model's parameters alone:**
  - output and word counting: O-BITS shows every non-ring instruction costs at least the k32 INT8 tensor core's time per word on the H200, but in aggregate over parallel units 4(g) gives c ≈ 0.333 on the H200 and about 0.11–0.14 on sm_120;
  - costs when the words are given as inputs (ABSORB-THM's Theorem D: r ≈ 1/766 SM-clock per word, about 5.3 INT8 MAC-times, 0.17 of LI's honest cost);
  - ring-fragment counts (BET-K-DISCHARGE).
- **The random oracle doesn't rescue it cheaply** (Theorem B′): one SHA-512 compression forces at most 48 words, so a hybrid costs γ ≳ 4300/L, about 134 at L = 32.
- **The decision with Daniel** (asked 03:00Z):
  - **(A)** one named computational assumption, as narrow as possible (e.g. "bit-level tricks on coin-dependent data don't beat the ring fragment"), with the same standing as SHA-512's collision resistance: stated, cryptanalyzed by BET-ALGO-LAB and O-REDTEAM, with ε_alg measured;
  - **(B)** SHA-512 per word as a random oracle, at γ ≈ 134–223;
  - **(C)** credit only the provable floor (c ≈ 1/3 on the H200), so ε ≈ 0 on what's credited but coverage is about 1/3;
  - **(D)** keep searching for a protocol that sidesteps it. The barrier applies to any explicit function, so none is in sight.
  - The coordinator recommends (A) with (C)'s floor stated beside it as the unconditional part.
- **Until Daniel answers:** keep going. Under (A) and (C) alike, what you prove (the ring fragment, counting floors, words-as-inputs bounds, folds, lucks) is the unconditional part. Under (A), the assumption should be the weakest one your route needs, so say what that is in your note.
- **Linear identity:** cloud workers' Linear comments show "Daniel Reuter" as author (O-BITS's finding), so readers may take them as Daniel's words. **Start every Linear comment with `[ACCEL <NAME>]`** (or your worker name).

## 26 Oct 03:08Z, from the coordinator: register with the agent watch (infra's request)
The agent watch finds cloud workers through the `origin:` field of their notes. Put this exact line in your note's front matter and keep it there:
`origin: J10R-CREDIT, bc-7efe38bd-091c-56db-b1fd-b015a5eaa75b, launched by compute-research bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac; code cursor/compute-research-j10r-credit-ea24`
Then sync. Without it, nobody hears when your `research run`s end.

## 26 Oct 03:18Z, from the coordinator: lean said yes to both (03:05Z)
1. STRATJOINT (LEAN-56) exports the per-stratum projection lemma (stratum s's hypergeometric factor in `stratified σ k hk`'s escape). The name comes when its statement is up, by Mon 06:00Z.
2. STRATJOINT moves `stratum`, `stratified` and `stratumRisk` into Definitions, in `Core/Draw.lean` beside `Verity.Law` (#4371). Then drop your restatement and import them.

## 26 Oct 03:22Z, from the coordinator: FOLD-LUCK's result (note compute-research/20261026T0222Z-draft-accel-fold-luck)

- **The receipt-side lucks are computational, not luck.** This covers `ReceiptLuckAt`, `GuessLuckAt` and H-μ⁺'s correlated-view point mass. Every coin and key that enters the honest receipt (dither and noise coins, salt, the fold's start state and keys) is read by the prover before it commits, because it needs them to compute the product. The only unseen coins are the draw and C-Flock's proof coins, and neither enters the receipt.
  - So "a deficient round keeps the honest receipt with probability ≤ p" is a claim about how cheaply a program can compute the receipt.
  - SHA-512-derived keys and keyed universal hashes don't change this: the cheater knows the key before choosing its words.
  - These rows therefore collapse into **the one computational slot**: Theorem O's, which is also what Daniel's symmetric-algorithmic-knowledge proposal would cover.
- **li has concrete cheap parts** (from `LiCheckpointLowBit`, proved on main; checked on the deployed LA/LK keys):
  - flipping bit 31 of any two words that enter one checkpoint leaves the checkpoint unchanged at every key, so receipt luck is 1;
  - bit 0 of every li checkpoint is the parity of the words' low bits, computable in O(n²).
- **What a fold needs:** it must be injective in each word (li is) and have **no cheap part** (li fails). Rotating the state, as xin and duo do, removes both failures but costs 1.5 lane-ops a word, over the 0.54-per-pipe budget.
- **li's KLI₂₄, (RT), (G₃) and (HO_U)** are not read under #2352 (pinned openings plus a checkpoint per segment). They survive only in the li design TOMLs' `[mand.fold]` text, which is stale: `LiWindowUnbound` proves the window receipt they price is unbound at every key.
- **A smell for ASSUMPTION-AUDIT:** `HMuPlus.PointMass.coins` looks like a live-coins fact, but `CoinRound` assumes the committed value never reads the fresh bits. In Dither the cheater does read them (B′ = B + f is the matmul's input), so in the deployed protocol it is computational.

**For you:** Your lucks with shared row and column noise are, by FOLD-LUCK's argument, also computational. State them through the single slot rather than as a separate claim. FOLD-LUCK handles their form next turn.

## 26 Oct 03:35Z, from the coordinator: R-ACCEL-6, the computational assumption is scoped to matmul (Daniel, 25 Oct 7:37 PM PDT)

- **The ruling.** Daniel: "I'm most happy with scoping this down to matmuls because these are so closely studied. If we make this public algorithmic knowledge thing for every single algorithm, that just feels way weaker because a lot of these are completely unstudied." The single computational premise (symmetric algorithmic knowledge, SAK) covers only **exact matrix products**: no program computes them on the card more cheaply, per priced unit, than the best public matmul algorithm.
  - That includes the segment products and prefix sums of a product.
  - It includes computing all but a small fraction of a product's entries correctly. Matmul's random self-reducibility makes that no easier.
  - Everything on the tensor core and in ring arithmetic is still proved.
- **Consequence 1: nothing else can lean on SAK.** The receipt, the fold, ABSORB and the lucks are not matmuls, so they must be **provable** (from θ, standard crypto and live coins) or designed away.
- **Consequence 2: a receipt without a fold** (the coordinator's proposal; PROTO-DESIGN is checking it). Commit every word with SHA-512 (a Merkle tree, collision resistance only) and drop the fold.
  - Hashing costs γ_hash ≈ 6460/L on the H200; noise removal costs γ_rm ≈ L/K.
  - At L in the thousands (L ≤ K), both are small. At K = 16384: L = 8192 gives about 0.79 + 0.5. At K = 65536: L = 16384 gives about 0.39 + 0.25.
  - Binding is then collision resistance plus sampling luck (Theorem S). The receipt-side defects (FOLD-OPT, ABSORB, ReceiptLuckAt, GuessLuckAt, H-μ⁺) disappear.
  - What remains about hashing is its speed. An adversary with a faster SHA-512 frees part of honest's hashing time, so SHA-512's implementation speed is a second closely studied algorithm. It goes to Daniel to confirm.
- **HYB** (BET-HYBRID-2; note compute-research/20261026T0225Z-draft-accel-bet-hybrid-2; branch cursor/compute-research-accel-bet-hybrid-ea24 at 58cc8a6a9f) is H1 with three changes:
  - U widened to ±96;
  - activation-side noise V of ±64;
  - 8-bit data on both sides, with sparse mod-2⁸ wrap corrections.

  The served product is exactly H1's. INT8 overflow stays at 11.3% or more for two-term and 26.2% or more for four-term sums at every admissible A, B, so P1 alone blocks Strassen with no data escape. Each word's F₂ rank is 64. γ ≈ 1.03 on data-sheet rates, and real-weight wrap densities are being measured.

## 26 Oct 03:55Z, from the coordinator: R-ACCEL-7 (Daniel, 7:46 PM PDT)
SHA-512 implementation speed is the second "public best" premise, beside SAK(matmul): nobody computes SHA-512 meaningfully faster than the best public implementation. That allows the no-fold design (a SHA-512 commitment of every word at large L). The adversary model's figures are reconciled in the brief's section 1: 9.36e30 FLOP a year is the Sept 2027 rate (the base grown about 4.9×), and it is the figure of record.

## 04:35Z, coordinator (to J10R-CREDIT-2)
J10R-CREDIT's records run r20261026-025306-70e3, on a merge with next bf70ad8430, failed rc 1 after 32 min. Its logs aren't fetchable from here. WORD-WORK's similar records-on-merge-with-next run (r20261026-022946-7593) failed in next's own modules, `Proofs.Flock.Bridge.Count` (type mismatches at lines 44–52) and `Proofs.Pous.Pooled`, not in its files. So rerun the records fast path on your own head, not on a merge with next. Drive checks the merge. If your head's run fails in modules that aren't yours, say so on CMP-127 and carry on.

## 03:52Z (real UTC, date -u), coordinator
Clock correction: my stamps above, from about 03:15Z on, ran up to 1.5 h ahead of real UTC. Deadlines in `briefs/sprint-mon-noon.md` are real UTC. Use `date -u`.

## 04:30Z (real UTC), coordinator to J10R-CREDIT-2
RT-LEAN-2 L35 (low): W3 only shows consistency on a toy circuit. Say so in its docstring and #4541's body. It's a follow-up commit if the mark would otherwise restart its checks.

## 07:15Z (real UTC), coordinator, for J10R-CREDIT-3 (launches after #4541 and #4618 land)
lean, 07:20Z, Slack 1792998847.018959: the projection lemma for discharging #4541's `TileStratumEscape` is #4618 (marked, stacked on #4590): `FlockSoundness.Audit.Law.stratified_proj_escape σ k hk s B : prCoin (fun ω => Disjoint (ω s).1 B) = C(n_s − |B∩s|, k_s) / C(n_s, k_s)`, in `Security/Proofs/Flock/Soundness/Audit/StratifiedProj.lean`. Its companions are `stratified_proj_escape_eq_exp` (= exp(−stratumRisk σ k s |B∩s|), given |B∩s| + k_s ≤ n_s) and `stratified_escape_eq_exp`.
`Law` still lives in security_proofs. One of three PRs will move it (#4371 → Core/Draw, or #4229/#4303 → Flock/Audit/Law), with names unchanged. Until then, keep our restatement of `stratum` on the Properties side. Also to do: the L22 `RegisteredBeforeDraw` docstring.
