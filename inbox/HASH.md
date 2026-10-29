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
- the unkeyed XOR fold (COUNTER `pack`: 28–36% of stock);
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
