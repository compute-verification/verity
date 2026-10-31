# HASH: securely reduce PoUW's hashing (from 18:35Z, 29 Oct)

**Daniel, 11:27 AM PDT (18:27Z):** the LEAN workstream lands the lead's implementation and proof on `main`. A second
workstream explores "all of the ways that we can securely reduce the hashing requirements ... hash fewer steps ...
jointly explore the space of noising and hashing ... random down projections or just stuff like this, which is cheaper
than hashing ... other assumptions we can make. Let's just keep searching ... how can we securely reduce hashing? That's
the core research question."

## The problem, in numbers

- Under DistinctWrites (cost ≥ 32 · #committed, distinct, coin-dependent 32-bit words) credit needs about one committed
  word per 28–32 MACs. Each word is SHA-512'd at about 3,500 registers a compression (32 words), so about 110 register
  writes a word.
- **h**, honest hashing over stock in W1, is about 120–230 (N-SHALB: one register a compression is 0.066× stock). About
  99.2–99.6% of the credited work is hashing.
- T2 is 10.9× (64 rows) to 62× sync / 75× async (8,192 rows).
- **The figure of merit is h, at fixed T1 ≤ 1%, T2 ≤ 100× and T3 ≤ 5%, and the assumption's rank.** Report h for every
  design.

## Already known (don't repeat it; extend it or break its premise)

- **Sparse commitments:** road (b) (N-DENSE) found neither a word every 64 nor every 128 steps creditable at rank ≤ 3,
  because of the middle-segment skip (N-SAKU, proved premise-free): a prover skips the uncommitted middle. The same
  skip kills every exact-output design (NCP, per2, os2).
- **Linear sketches fold** (H12, N-LIT): Σ r_t w_t over running words equals Σ_τ κ_τ Σ_{t≥τ} r_t, computable without the
  words. Any commitment linear in the words is cheap for the adversary.
- **Module-SIS** (GD-SIS, 28 Oct): h = A·x mod 2³² costs 896 int8 MACs a committed byte, and its opening in C-Flock is
  24–36× SHA-512's rows. It never beats SHA-512, at any row size.
- **SAK on literally uniform operands** (road (c), N-SAKU): no reduction gives credit, and the obstruction is proved.
- **Seeded coins** (N-EGRESS): expanding coins from a seed lets an outside machine take the seed, so coins come raw.
- **R1** (the certifier hashes every word) moves the hashing off the prover, but ships the words: T2 97.5×–665×.
- The one-hash rule (SHA-512 everywhere, AGENTS.md) and "everything zero-knowledge" stand. A design that needs another
  primitive or reveals a sketch names the rule change, and the coordinator takes it to Daniel. A sketch is a value: it
  is committed under a hiding leaf, or its leakage is bounded.

## Directions (each lane owns one; cross-pollinate through notes)

| lane | agent | direction |
|---|---|---|
| SPARSE | N-FLOOR | hash fewer steps: which mechanism makes skipped steps unskippable. Candidates: coins for later steps drawn after an earlier commitment (chained rounds, and their latency), checkpoint positions drawn after commitment, noise that compounds through the running word yet stays removable. Price each and find its floor |
| SKETCH | N-LIT | random projections and sketches cheaper than hashing: non-linear sketches (Σ r·g(w) with cheap non-linear g, byte lookups, squaring mod p), sketches keyed by coins drawn after the words are held, and polynomial or universal hashing sealed by one SHA-512. Survey the literature (Freivalds, universal and almost-universal hashing, sumcheck-style batching, proofs of space and time) and say exactly where each folds |
| SHA | N-SHALB | fewer SHA-512 bytes per credited MAC within the rules: truncated or packed words (commit k bits of a word), wider credit per word, tree and chunk shapes, and one compression over more credited work. The true floor of h under ShaWrites |
| NOISE | N-NOISE | the joint noise × commitment grammar: extend last night's 400,505-design grammar with commitment axes (which words, how many bits, sketch or hash), and search for noise that makes sparse, truncated or sketched commitments creditable |
| ASSUME | N-SALT | the assumption ladder: the weakest premise that lets a design commit only outputs, or every k-th step. Candidates: SAK variants scoped to our noise class, fine-grained conjectures, a tensor-core-only cost model, memory or bandwidth floors, sequential work. Rank each, give its falsification test, and say what h it buys |
| ALT | N-COVER | other cost bases and placements: credit HBM traffic or MMA issue, not register writes; split hashing between card and certifier with sketches shipped instead of words; price the network for each |
| HIDE | N-HIDE | hiding and zero-knowledge for every proposal: what a sketch or truncation leaks, the hiding leaf it needs, and its cost; check each against the rules |
| COUNTER | N-FP8 | counterexample search: for each proposal's claim, search small instances on CPU for a cheaper prover strategy, and report each with a run id |
| CARD | N-CARD | the card: h, T1, T2, T3, assumption rank and status for every design, on one basis with the lead as the baseline, plus a ranked frontier at each checkpoint |

## Done when

A design that cuts h by at least 10× from the lead's, holding T1 ≤ 1%, T2 ≤ 100× and T3 ≤ 5%:
- its every premise named and ranked;
- its credit checked on CPU and its cost measured or bounded;
- with a falsification test for each new premise.

Or a proof that no design in a stated class can. Short of that, the frontier of h against assumption rank is the
answer, with the best design at each rank.

## Rules

- No PR, no Slack, no Linear. Push your branch after each commit (`cursor/compute-research-hash-<lane>-<suffix>`).
  Long jobs go through `research run`, with scratch outside the run directory. **Use your own node (19:30Z): cpu-5
  is crowded** (its check slots queue six deep, and its /workspace is under its inode floor). SPARSE cpu-2, SKETCH
  cpu-3, SHA cpu-4, NOISE cpu-6, ASSUME cpu-7, ALT cpu-8, HIDE cpu-12 (cpu-9 is down since about 19:30Z, ssh timing out), COUNTER cpu-10, CARD cpu-11.
- GPU only on vy-dc-h200-5 through gpu-lease, today until 20:40Z, after N-KERNEL's and N-W1's runs. Never use
  vy-dc-h200.
- Each result is labelled proved (Lean, axioms printed), checked (CPU, with a run id), measured (GPU, with a run id)
  or estimate.
- Checkpoint notes go under `lanes/compute-research` (`kind: report`), published with `research notes sync --path
  lanes/compute-research` (or `research notes push compute-research` if your CLI has that instead).
- Report to the coordinator by finishing your task. **Interim 1 is due 21:00Z** (2 PM PDT): your direction's map, what
  closes, what's open, and h for anything priced. Then keep going.

## FOLD: design the fold, and the mildest assumption behind it (Daniel, 20:31Z: "swarm that shit")

Daniel asked whether the fold can be computed dynamically, and how creative we've been. So far we haven't been, and
that is the problem. We have four hand-picked fixed folds, each resting on "this specific construction has no cheaper
program": FOLD-OPT, ThinFoldOpt, XorWrites, SketchWrites2 and PairWrites. That is SAK's shape, narrower. COUNTER has
already refuted XorWrites_16 and the XOR fold, and SKETCH's NH and NOISE's quadratic-decomposition attack disagree.

N-SALT's zero-data bound settles the frame: any cut needs a premise about the function that consumes the words. **The
design question is which function gives the mildest premise.** Lanes:

| lane | agent | direction |
|---|---|---|
| DYNAMIC | N-W1 | Fold families whose structure, not just constants, is drawn by live coins per call or per step after the data are committed: random op sequences, rotations, constants, small S-boxes, multipliers. The premise then is "a random member of a large family has no shortcut on average", which is close to the symmetric knowledge on random operands Daniel accepts. Give each family's honest cost and its credit at zero data. Run an average-case superoptimizer search over random members (single and multi-step), and look for counting arguments: if a shortcut must work for most draws, the number of short programs bounds it. |
| ROUNDS | new | Folds built from a standard primitive's round function, so the premise is about SHA-512 as ShaWrites is: a few SHA-512 rounds per word, the message schedule as a fold, ShaWrites per round. Price h (perhaps about 5×) and say exactly what premise it reads. |
| LIT | N-LIT | Survey functions with cost lower bounds: proofs of sequential work, memory-hard functions, VDFs (iterated squaring), lower bounds in restricted models (ARX, algebraic, branching programs), and universal-hash families with computational bounds. Map each to a fold, its premise and its rank. First, settle your NH sketch against NOISE's finding that quadratic sketches decompose cheaply on the lead's words (note 20261029T2000Z-report-hash-noise), with COUNTER. |
| TENSOR | new | Make the tensor-core work itself the fold: chained random matrix products of the words with a nonlinearity between them, so the fold's cost is MMA work the hardware does anyway. Give the credit under H32 and TileGrain, h and T2. |
| COUNTER | N-FP8 | Add an average-case harness over families: for a fold family and a draw distribution, find the cheapest program that works for most draws, at single and multi-step windows. |

Each lane: h, T1, T2, T3 (estimate if unmeasured), the premise exactly with its rank and falsification test, and
whether it is about our construction, a random family, or a standard primitive. Checkpoint at 23:00Z.

## §interim-1 (coordinator, 21:10Z)

**Answer: nothing yet cuts the hashing at a premise Daniel has accepted.** At ShaWrites' rank (4a) and below, the
lead's h = 127.5 stands. h ≥ 108.6 is proved in core Lean for credit by register writes alone (N-SHALB,
`HashFloor.lean`). Every real cut needs a premise about the function that consumes the words, by N-SALT's zero-data
bound (proved): no premise about the operands can let a design commit only outputs or every k-th step. The only
exception is a threat-model change, public weights.

**The lesson that shapes the rest** (COUNTER, note 20261029T2053Z-report-hash-counter-nh, checked):
- At zero data the lead's words are w_t(i,j) = x + R_t(i) + C_t(class(i), j): a data term, a row walk and a column
  walk.
- So any credited function that decomposes over those sums is cheap. Row-only sums are shared by every column,
  column-only sums by a class's rows, and the cross terms batch onto IMMA as u8 digits.
- A fold or sketch must mix non-linearly across the sum, or the noise must not leave words that are row-plus-column
  at zero data.

**Closed (checked unless marked):**
- sparse commits, every k-th step (SPARSE, NOISE);
- chained coins, and checkpoints drawn after commitment (SPARSE);
- truncated words (NOISE, SHA, COUNTER);
- linear sketches and projections, which fold (SKETCH);
- Module-SIS (earlier);
- xor-m (SHA: the served IMMA works in place, so the margin is 3%, and with near-free predicates it is false);
- the unkeyed XOR fold (COUNTER `pack`: 15.8% at K 2,048 and 23.7% at K 8,192, corrected 22:05Z from 28–36%);
- **NH/nh2, refuted by `decomp`** (COUNTER: PairWrites fails T1 by 68–71% of stock);
- crediting HBM traffic or MMA issue (ALT);
- the card/certifier split, which only moves the hashing (about 62 Genoa cores a GPU).

**Still standing, each about our own construction unless noted:**

| design | h | premise and rank | status |
|---|---|---|---|
| multiply–xor fold (SPARSE) | 1.57–2.94 | FOLD-OPT, 4, our construction (SAK's shape) | single steps searched; multi-step and `decomp` not yet; T2 est. 6.7–9.7× (standalone) |
| thin fold, alternating XOR3/IADD3 (ASSUME → EGRESS) | 2.31–3.50 | FoldMaterial + ThinFoldOpt, 5 | two-instruction windows hold; one packing route is 1.9× honest; `decomp` not yet; T2 unmeasured |
| mirror-π quadratic sketch (NOISE) | 2.37–4.55 | SketchWrites2, 4b (SAK's rung) | network term closed with tiled coins; decode stream 72.7 MB a token; `decomp` not yet |
| D-OUT, outputs-only, public weights (ALT) | 0.91–1.39 with attention uncredited (64–98% credited), 3.4–43 with all credited | SAK-UF on the registered weights, 4 (not about our noise); needs public weights, against everything-ZK | survives every priced attack at 1-bit noise (T1 0.74%); T3 pending |

**Rules (confirmed):** a fold committed under SHA-512 and credited only by a cost premise binds nothing, so it isn't a
hash. The one-hash rule needs no exception for it (HIDE, SKETCH). What does change is PoUW's statement: it commits a
function of the words, which is a claims change for Daniel when a candidate is ready. Hiding is free inside the
forward's tree if sketches and folds stay inside it, in small chunks (HIDE).

**No ask to Daniel yet.** Every surviving candidate's premise is about our own construction, except D-OUT, which needs
public weights. The FOLD swarm (dynamic families, SHA-512 rounds, tensor folds, the literature) is aiming at a premise
about random families or a standard primitive, and it checkpoints at 23:00Z.

**Re-targeted:**
- COUNTER (N-FP8): `decomp` against the multiply–xor fold, the thin fold and mirror-π; multi-step windows; D-OUT's
  routes; then the family harness.
- EGRESS: the thin fold's three-instruction and cross windows and its 32-bit fibres, plus `decomp`.
- NOISE: noise layouts whose zero-data words aren't row-plus-column (higher-rank coins at a priced bandwidth), the
  lever every fold needs.
- Every FOLD lane: design against `decomp` first.

**Rulings on CARD's three asks** (coordinator, 21:15Z; note 20261029T2110Z-report-hash-card):
1. **Predicate writes stay free in W1.** Pricing them as register writes would be a stronger hardware premise, and it
   buys only trunc-16's 2.0×. xor-m and trunc-16 stay closed. The card drops the priced column, or labels it
   information only.
2. **An unkeyed fold sealed by SHA-512 sits within the one-hash rule.** SHA-512 does all the binding, and the fold
   only carries the credit's cost premise. That premise goes to Daniel with the candidate, as the change to PoUW's
   statement.
3. **The Duo exception is moot here.** It covers folds that bind in SHA-512's place, under H-μ⁺. Every HASH design
   seals its fold under SHA-512, so by ruling 2 none of them needs the exception, keyed or not.

**Card basis:** h counts hashing wherever it runs. ALT's card/certifier split moves the hashing to the auditor's
device and cuts none, so its 98.09 leaves the frontier.

**21:30Z, FOLD-ROUNDS (note 20261029T2120Z-report-hash-fold-rounds), the first cut at rank 4a:**
- **R80X:** run SHA-512's 80 rounds over the raw words, each XORed first with a per-position constant (one LOP3), and
  seal each state with one SHA-512. h 17.7–18.1, a 7.0–7.2× cut.
- **Premises:** ShaWrites-R80, which is ShaWrites carried over to raw round words, and AddXor (no single sm_120
  instruction computes (a + b) ⊕ c), which isn't enumerated yet.
- **RDX** adds a second word into e′: h 9.0–9.4, a 14× cut, but at rank 4 as a construction of our own.
- `decomp` breaks every plain round fold, because at zero data a word enters only h + W + K_t. The XOR is what fixes
  that.
- T1 is about 0.0002% with chains carried across calls. T2 is an estimate (9.1/10.7 against the lead's 54.9/66.0),
  and T3 is the lead's.

Queue:
- COUNTER attacks R80X and RDX after its `decomp` round.
- FOLD-ROUNDS enumerates AddXor (making it checked), states ShaWrites-R80 exactly with its falsification test, and
  searches for a fold past 10× that stays inside SHA-512's round function.

**21:40Z, LIT (note 20261029T2131Z-report-hash-fold-lit, checked):**
- NH is refuted independently, with matching numbers: nh2 is credited correctly only up to G = 16 words a value (h ≈ 8.3, and even there PairWrites is false).
- **The survey's structural point:** at zero data the only input is the coins, so every cost lower bound stated in input length credits coins, not words. Only query bounds against an ideal oracle or permutation survive. A fold's premise has to be about a standard permutation or round, which is R80X's direction.
- Dominated: proofs of sequential work, memory-hard functions, VDFs, BLAKE3 (h 44.7) and SipHash (15–28).
- Rank-5 survivors: an ARX round a word (2.5–10.9), a degree-4 power (2.53, thin against COUNTER's expansion attack), a known-order squaring chain (7.75–21.3), and a BRW tree hash (1.5–3.1, unattacked).
- Six shortcuts every fold must avoid: linear folds, additive fusion, long sums of products, packed narrow words, rotation-free programs, and step-invariant terms.

LIT becomes SI-RED, a general attack harness. BRW goes first. COUNTER keeps R80X and RDX.

**22:05Z, NOISE and TENSOR:**
- **The thin fold falls** (NOISE, note 20261029T2159Z-report-hash-noise, checked). `mma_pairsum` computes an IADD3 pair step at zero data with one IMMA from σ + 2w, writing neither word. It wins by 25–33% of stock on all 15 layouts. ASSUME's check had no IMMA.
- **The multiply–xor fold:** nothing beats honest from a live N₀ (checked); from N₀ = 0, `edge_any` wins by 1.1–2.5%. N₀ must be live.
- **The pure XOR fold** over full-width salted words survives every prover in the grid: h 2.31–3.50 (estimate), rank 5.
- **Sketches:** on mirror-π with four coin lanes, the quadratic sketch and one-entry NH lose nothing under every feature model. On row-plus-column words, `decomp_sq` wins by 50%.
- **Wire:** per-entry coins cost 35 MB a token, and bilinear coins fail T3.
- **Decode-stream discrepancy (it matters for the lead's T3):** NOISE's formula gives the lead 92 MB a token at decode, against EGRESS's 11.7–13.2 MB. NOISE reconciles it now.
- **The tensor-core fold** (TENSOR, note 20261029T2155Z-report-hash-fold-tensor, checked on CPU):
  - h ≈ 1.51, an 84.5× cut, with T2 2.2–3.2× (estimate).
  - It holds under credit = stock only at rank 5 (TensorMaterial). Crediting the fold's own cost is false: a row-sharing prover gets T1 = 102%.
  - **The lever:** a wide rank-2 term from live coins in each word's starting value raises the write-count floor from 0.77× to 1.94× stock. Credit = stock could then rest on a count, provable as HashFloor was, not on a premise.
- **Ruling on TENSOR's ask:** the frozen lead's word form doesn't change (freeze 1). A HASH design may define its own word form, as a v3 candidate, with T3, the network term and honest's cost priced.
- Queue:
  - TENSOR makes the lever's count rigorous, in Lean if it can.
  - COUNTER's falsification target is any prover below 0.867 registers a word.
  - NOISE prices the lever's wire cost.

**22:10Z, COUNTER (note 20261029T2145Z-report-hash-counter-checkpoint):**
- **R80X's constants are constrained (proved by algebra).** With 0x7FFF_FFFF the XORed word splits into a row term plus a column term, so `decomp` breaks it again. AddXor at zero data is proved for constants with low nibble 0x1; the other nibbles are open. R80X uses only constants with low nibble 0x1 until the sweep (r20261029-220256-36ba) settles the rest.
- **The family harness** (`fold_family.py`) finds three 32-bit identities a fold must avoid: an add before a multiply distributes, adjacent multiplies merge, and runs of ring-affine steps collapse to 2 instructions. Excluding all three leaves the multiply–xor fold's shape.
- **The multiply–xor fold needs a live, nonzero N₀** (it agrees with NOISE).
- The thin fold survives `pack` in COUNTER's price model, but NOISE's `mma_pairsum` breaks it anyway, so it stays closed.
- Queue: COUNTER finishes its two runs, then attacks R80X (with low-nibble-0x1 constants) and RDX in full, then TENSOR's fold plus the rank-2 lever (target: any prover below 0.867 registers a word). FOLD-ROUNDS takes the constant rule when it reports.

**22:15Z, the denominator smell (SI-THEORY, note 20261029T2203Z-report-silicon-theory; coordinator's arithmetic):**
- The lead's gap and T1 are stated as shares of stock (useful MACs), while honest work is (1 + h)·stock with h = 127.5.
- So an error δ in ShaWrites' per-compression floor moves the gap by δ·h·stock. T1 ≤ 1% of stock needs ShaWrites exact to 1%/h, about 0.008%. N-SMELL's R14 is an instance: 0.44% between 3,515.38 and 3,530.92 is 56% of stock.
- As a share of the device's capacity, (1 + h)·stock, the same error is about δ: robust.
- Lower h makes the useful-work reading robust too: about 0.06% at R80X and about 1% at h ≈ 1. That is a second reason for HASH beside T2.
- The card should carry "premise precision needed = T1/h" for every design.
- Which reading the lead's claim uses is Daniel's (asked 22:15Z). R14's ruling (3,530.92) stands meanwhile; under the capacity reading both figures lie inside the margin.

## 22:25Z: Daniel chose capacity, and what that does to HASH (coordinator)

Daniel approved A: the gap is stated over capacity, (1 + h)·stock.

**The consequence.** On stock hardware no standard premise credits the MACs: SI-THEORY, and N-SALT's zero-data bound. A cheater serving zero-data requests pays for the matmul almost nothing; only the hashing or fold is forced. So the share of capacity a design can account for is

  a ≤ h_forced / (1 + h),

where h_forced ≤ h is the hashing or fold work a cheater can't skip. So 99% needs h ≥ 99; 95% needs h ≥ 19; 90% needs h ≥ 9.
- The lead's 127.5 gives 99.2%. This is why every design last night was nearly all hashing: the 99% target plus no MAC credit forces it.
- On stock hardware at 99%, h can fall at most from 127.5 to about 108.6 (N-SHALB's DistinctWrites floor), a 1.17× cut.
- **Every sub-99 design on stock hardware lowers a.** R80X (h ≈ 18) gives about 94.7%, the tensor fold with the lever about 77%, and EGRESS's k = 4 thin fold (h 2.45) about 71%.
- **Cutting h at 99% needs MAC credit:**
  - (i) a premise about the matmul itself (the SAK family; ALT's D-OUT, which rests on SAK-UF over public weights);
  - (ii) silicon: SI-THEORY's SHA-512 round in the MMA unit, or closure.
- The 99% target is the question to Daniel (card, 22:25Z).
- Until he answers, every HASH and SILICON result is reported with a (capacity accounted) beside h. The card computes a = cheapest prover's cost / honest cost.
- EGRESS's k = 4 thin fold (note 20261029T2210Z-report-hash-assume) is recorded, not re-targeted.

**22:30Z, SPARSE (note 20261029T2220Z-report-hash-sparse-window):**
- The multiply–xor fold (c4) starting at N₀ = 0 fails T1, at 2.59%. Interims 1 and 2's 0.42% is withdrawn.
- One live word a call (κ) still fails, at 1.51%. Starting at N₀ = a_i ⊕ b_j (live words a row and a column) leaves nothing known below honest; SAT finds no 2-instruction step 1.
- One-step windows are `unsat` below honest; two steps are `unsat` at random data. Two- and three-step windows at zero and chosen data are still running on cpu-2. Don't relay this as "checked over windows" yet.
- Under Daniel's capacity reading (22:25Z) the fold accounts at most h/(1 + h), about 61–75%. It is recorded, and the lane stays idle until Daniel settles the fork: a premise about the matmul, the hashing lead, or silicon.

**22:45Z, ALT's D-OUT under the capacity reading (note 20261029T2229Z-report-hash-alt):**
- **T3 is checked on CPU with 5 seeds.** Served as specified, D-OUT fails at +430% (2K): noise on the BOS row swamps its scale. `sink` (BOS row exact) gives +3.19% / +2.91%; `sink_nodown` +0.62%, but it credits only 72.4% of the linears.
- **The lead's hashing on attention and the BOS row, outputs-only on the rest:** h 3.4 / 26.8 / 43.1 at T 512 / T 8,192 / decode, at a = 99.45% / 99.20% / 99.18%. That is 35×, 4.4× and 2.7× less hashing than the lead, at 99%.
- It needs public weights (an exception to everything zero-knowledge) and SAK-UF (rank 4, about the matmul on registered weights).
- It is the concrete "credit the multiply" option for Daniel's fork. It waits on N-PPL's confirmation: our int8 LUT attention rather than fp32, a second model, full windows.
- Attention is 96% of the remaining h at T 8,192.

**22:48Z, TENSOR checkpoint 2 (note 20261029T2232Z-report-hash-fold-tensor) retracts the 22:05Z lever entry:**
- With one R a call, the cheapest route (last step's product plus the carries, `mma_pairsum` on bytes) floors a prover at 0.37× stock, not 1.94×.
- **A fresh R each k32 step** floors it at 1.43×/1.44× stock. That uses `TensorFloor.lean` (core, sorry-free, arithmetic only) plus H32 and **ByteEntry**, a new rank-5 premise that each byte is formed by one of six named routes. It is not a count. h ≈ 1.58 / 1.53.
- **Under the capacity reading** it accounts a ≈ 1.43/2.55 ≈ 56%. It is shelved with the other stock-hardware folds pending Daniel's fork. The coordinator's 22:05Z "credit could rest on a count" is withdrawn.

## 22:50Z: RETRACTION of the 22:15Z and 22:25Z entries (coordinator)

- **Retracted: "the multiply isn't credited" and a ≤ h/(1 + h).** Under W1, committing every k32 word forces one write a word, priced at no less than the MMA's 32 units a word. That is DistinctWrites under H32, the rank bound. So the lead credits stock through forced writes (CORE.md §2a: only the final word of each dot product is unforced, 32/K of stock), and hashing on top of it under ShaSpeed. The noise is what makes every word distinct and unpredictable, so each must be written. `decomp` breaks designs that don't consume every word: sketches over aggregates.
- **Retracted: the capacity denominator.** Daniel's §2a ruling (Wed 7:01 PM PDT) is that the gap is additive against stock, cost(H) − min cost(adv) ≤ 1% of stock, precisely so that a large hashing overhead can't hide a skippable matmul. The card 2034fe69 approval is being reversed with Daniel. The 99% card 7f2658c2 is withdrawn.
- **What stands under §2a:**
  - A design's whole honest cost, hashing or fold included, must be forced to within 1% of stock. "Credit = stock" designs whose overhead isn't forced fail (TENSOR's).
  - Overhead premises need precision of about 1%/h: lead 0.008%, R80X about 0.05%, h ≈ 1 about 1%. That is HASH's second motivation beside T2.
  - Co-issue is judged in work, not wall time (§2a), which answers SI-THEORY's 14–28% time reading: report it, don't judge on it.
- HASH re-targets at 23:00Z as planned, under §2a.

**08:05Z, ROUNDS checkpoint 3 (note 20261030T0755Z-report-hash-fold-rounds): R80X specified as a v3 candidate.**
- **Function:** SHA-512's rounds over free 64-bit words, each XORed with `xc(t)`, plus the feed-forward per block. Honest cost is 30j + 14 writes a block of j words.
- **h** 18.10 / 17.72 (7.0–7.2× under the lead), within 0.6% of the structural best inside the unmodified round.
- **Word layout:** each 64-bit word is the accumulator pair at rows r and r+8 of one MMA fragment, so no shuffle.
- **Constants:** K at (t+40) mod 80, rotated a byte at a time until each half has a nonzero bit in 4..6, then low nibble 0x1. This fixes 8 halves at 0x81.
- **Cadence:** one chain a thread slot, started a window from SHA-512(tag ‖ salt ‖ chain id), carried across calls, its state sealed once a call as a hiding leaf.
- **Under §2a,** the fold adds no gap only if Daniel extends ShaSpeed ("hashing credited, with honest taken as the best") from SHA-512 to its round function on raw words. Without that, the gap is the premise's slack, which must hold to 0.055% (1.39 writes a block).
- **T2, estimate:** 9.1/10.7 at t8192 against the lead's 54.9/66.0. The measurement plan is sm_120 tiers 1–3, then an H200 run of record.
- **Lean draft:** `Security/Definitions/Pouw/Assumptions/ShaWritesR80.lean` (type-checks; pinned to `xc` and to SHA-512("abc")). It is weaker than the protocol's premise until W1 gains carry predicates. The falsification test is `hash_r80_premise.py falsify`.
- **Next:** when COUNTER's R80X attack reports (09:00Z), the coordinator puts the ShaSpeed-R80 extension to Daniel as a card.

**09:00Z, COUNTER (note 20261030T0826Z-report-hash-counter-r80x-arxm), judged under §2a:**
- **arxm fails §2a at every layout (checked).** A zero-data prover pays 3.09–3.53 eq a step against honest's 4, a gap of 53–104% of stock. The counting theorem covers about 1 of honest's 4 eq; the cross-output sharing of the row and column terms is what fails. Shelved.
- **R80X: not broken, not yet certified.** Width-4 sweeps hold only against one free term a side. The 35.7% bound quoted at 08:05Z is withdrawn as a width artifact; the ceiling, if every half had a route, is 114% of stock (estimate).
- **Plan:** ROUNDS sets bit 4 in every `xc` half (so all 160 certify at 8 bits), then COUNTER runs the 160 certificate solves at 8–10 bits on pods.
- **The R80X v3 ask to Daniel (ShaSpeed-R80) waits on those certificates.**
- RDX's raw V skip ties, at 34 = 34 writes a round.

ROUNDS constants for COUNTER's certificates: 7408679cbad22b9ffdb5ae4d476390ed3b84a76e

**19:05Z, COUNTER (note 20261030T1900Z-report-hash-counter-r80x-certificates; ea3fb28192):**
- **R80X's word entry is certified over LOP3, IADD3, IMAD and LEA.** None of the 160 `xc` halves (ROUNDS' 7408679cba) has a two-instruction program at 32 bits, against honest's three, hi carry-in included: z3 at 8 bits, `word_entry_rank` (all 93.6M IADD3/IMAD/LEA programs) and `word_entry_mux`. So §2a's word-entry term is 0 within that set (checked); four steps are hand arguments.
- **Open:** PRMT, SHF, IMAD.WIDE hi, IDP.4A and IMNMX. The ceiling if every half had a route is 114% of stock (estimate; none is known). N-FLOOR certifies those pairs now.
- **Then the coordinator takes the R80X v3 ask to Daniel:** extend ShaSpeed (\"honest taken as best\") from SHA-512's compression to its round function on raw words (`ShaWritesR80`), with ROUNDS' spec, the certificates and the T2 estimate.


## D-OUT and Strassen in s8 (coordinator, 00:30Z Sat 31 Oct)

HT-PREMISE v4 (note:compute-research/20261031T0022Z-report-hardness-premise-v4) finds that D-OUT's premise, outputs only
over public natural weights at full K, has a Strassen-in-s8 hole:
- if an overflow correction costs about 1 register write, the gap is 1.84% (3.6× SAK-UF's 0.516%), and capacity falls to
  99.15–99.16%, the same as the lead alone;
- at 2 writes, 99.2–99.45% holds;
- the crossover is about 1.6 writes on the H200 and 1.9 on sm_120.

D-OUT's "99.2–99.45% capacity" is therefore **conditional** until the correction is timed on a GPU (running, sm_120).
The lead itself is not exposed (1.93–3.12× stock under its per-word commitments).
- **Update, 01:55Z (timed on sm_120, vy-nebius-1):** the overflow correction measures 0.84 writes for 2-bit forms and
  2.36 for int8 forms, under the 1.70 crossover. **D-OUT is exposed on sm_120:** a gap of 2.18% (4.2× SAK-UF), and
  capacity of 99.08–99.16%, at or under the lead's 99.16%. So its "99.2–99.45%" does not survive there.
  - **On the H200 it's ambiguous.** The kernel scaled to the H200 is about 3.3 writes (gap 0.000%), but the issue floor of
    the same instructions is 1.48 (gap 0.75%), against a crossover of about 1.68.
  - **To settle it:** about 2 minutes of one H200, in the next window that doesn't take the demo's node: `leaf_corr.cu` at
    c09e05e8c6 on `cursor/compute-research-hardness-premise-7f3a`, with `ARCH=sm_90a PEAK=4096 CORRS=34578`
    (note:compute-research/20261031T0022Z-report-hardness-premise-v4 §3).
