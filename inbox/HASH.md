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
