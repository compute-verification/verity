# PoUW night program (Daniel, Thu 29 Oct 06:16Z = Wed 11:16 PM PDT)

Coordinator: compute-research (bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac). Issue: CMP-294. Daniel, going to bed:

> "Go ham tonight. We need to find a solution. Organize a really impressive research program. I have a feeling we have not
> gone down every single possible road. Exhaust the full space of assumptions we can make, of noisy strategies, of
> denoising strategies. Now that you know the proof of useful work scheme is really modular, this might help you search
> the space more parsimoniously. Use all of your resources."

Read first: AGENTS.md (`git show origin/main:AGENTS.md` wins), then CORE.md §2a and §6–§10, SWEEP.md, and the sweep's
notes (`~/.research/notes/lanes/compute-research/*-report-sw-*.md`, after `research notes sync`). This file extends
SWEEP.md. Where they differ, this file holds.

## 1. The spec (unchanged; CORE.md §2a, Daniel's rulings)

- **One matmul** O = A·Bᵀ, with A and B committed before the call's noise. Each call is atomic and gets fresh noise from
  live coins (no reuse). Workloads: Llama-3.2-1B int8 on one H200 (sm_90) at T 512 (the demo's 64 real tokens; report
  512 rows too) and T 8,192; a V4-like dense layer (K = n = 7,168) and its routed experts.
- **T1 (hard):** cost(H) − min over adversaries of cost(adv) ≤ 1% of cost(stock), in priced work (W1 / MH100 units: a dense
  int8 MAC = 1 unit; every instruction at its measured price per 32-bit register written; H32 is measured on the H200 with
  zero slack, h200Measured = h100Floor). Not wall time. The honest program is the best possible honest program. Hashing
  is credited work.
- **T2:** honest overhead ≤ 100× stock. **T3:** perplexity rise ≤ 5% (WikiText-2 at 2K and 8K).
- **Assumptions as mild as possible.** Ranks: 0 a theorem; 1 a measured hardware floor; 2 a small CPU-checkable fact;
  3 model faithfulness or protocol premises; 4 algorithmic knowledge; 5 a hardness conjecture. SAK is the bar to beat.
  Daniel: SAK "is secretly assuming that whatever structured noise we added is good"; "I would happily say symmetric
  knowledge on what literally random operands were, but ours aren't literally random."
- **Terms:** a dot product is one output element; a proof block is what C-Flock proves (its inputs and outputs are
  committed, never its interior: Daniel, 3:57 PM PDT). Never "cell".

### Coordinator defaults for tonight (each to be confirmed by Daniel in the morning)

1. **Rank 4 is split.** 4a is algorithmic knowledge about a standardized public primitive (ShaSpeed on SHA-512). 4b is
   SAK on our own noised operands. Daniel's objection is to 4b. Report both; the target stays rank ≤ 3 everywhere.
   SAK on literally uniform operands (SAK-U) is 4b's mildest form; report it as its own row.
2. **Hiding is required** (design rule: every value under a hiding leaf), so honest-only hiding work is a T1 term.
3. **The epilogue and the step's nonlinear operations** (dequant, norms, softmax, activations) are a coverage term,
   reported beside T1 for the whole served step (N-COVER), not dropped and not folded into the matmul's T1.
4. **Coins** are reported both credited and uncredited.
5. **An open premise ranks 5 until settled** (SW-SMELL P1). Report a design's matmul part and hashing part separately,
   and its overall rank as the max.

## 2. The design space, factorized

A design is one choice per axis. T1, T2 and T3 are sums of terms that each read a few axes, so axes can be searched
separately and recombined. Use this vocabulary in every note and card.

| axis | choices so far | reads into |
|---|---|---|
| **N** noise | where (A lanes, B lanes, both, accumulator init, k order, scales, sparsity metadata); alphabet and bits a lane; public code; per call | forming, coins, distinctness, T3 |
| **D** de-noise | served (noise stays in O′); exact cancellation in the GEMM (stack width α); post-hoc correction; folded into the epilogue | final word, de-noise work, α, T3 |
| **C** commitment | which words (every k32 accumulator word; formed operands C1; noise E₁F₁; epilogue intermediates); density L; width (32, 16); packing | forced work, bytes, uncommitted steps |
| **H** hashing | hm96 rows, row size, tree; hiding (salt PRF, mask M_key·y); separate, beside or fused | T2, ShaSpeed's δ, hiding term |
| **P** premises | the set closing each term, each with a rank | the design's rank |
| **F** format | s8 × s8, u8 × s8, int7, FP8 E4M3, BF16 | sign fix, exactness, T3 |
| **W** workload | linears, QKᵀ, PV, decode (< 64 rows), experts, lm_head, epilogue and nonlinear ops | coverage, shape prices |

T1 = forming(N, F) − forced(C) + de-noise(D) + final word(D, P) + uncommitted steps(C, P) + sign fix(F, N) + coins(N)
+ hashing slack(H, P) + hiding(H) + hashing before the noise(P) [+ coverage(W)].

## 3. Where we are (06:20Z)

**The lead is `ssn_1l`, served sparse noise** (SW-NOISE final, note 20261029T0551Z-report-sw-noise; art:d90a3d9d…):

- Each k32 step carries 31 data entries and one fresh coin bit a row against a public weight of 1. Step 1 carries 28 data
  entries and 4 bits against an injective public code of the output column. The noise is served (O′ = A·Bᵀ + small
  noise), so every committed word, the final one included, is noise-dependent and pairwise distinct for every A and B.
- **T1 0.049%** on Llama at every T (0.255% with coins uncredited); 0.029% on V4 dense. α 1.044. No final-word lemma, no
  C1, no int7, no sign fix.
- **T2 62.9× serial / 34.2× overlapped at T 8,192; 6.5× / 4.0× at T 512** (estimates on SW-GPU's measured parts).
- **T3 measured (SW-PPL final, note 20261029T0725Z-report-sw-ppl): −0.09% [−0.34, +0.15] at 2K and −0.02% [−0.11, +0.06]
  at 8K** against stock int8, all 16 windows, one noise seed. Every 1-bit SSN format is within noise of stock; 4-bit lanes cost
  +0.17–0.26% at 8K.
- **Premises:** H32 (1, measured), W1 (3, unnamed), ShaSpeed (4a) for the credited hashing, hash-compose (4a, unstated),
  atomic call + PeriodInventory for hashing A before its noise (3), the distinctness lemma (0 once proved; open), a
  grinding bound (open).
- **It fails T1 today on hiding:** hm96's salt PRF, mask and c are honest-only: 7.2× stock at 16 KiB rows, about 1.2× at
  256 KiB (118% at SW-PRICE's best row) and 0.15× at 1 MiB, in W1. Salts from the coins are necessary, not sufficient
  (SW-SMELL).
- **Since 06:20Z (the sweep's finals; note 20261029T0745Z-report-pouw-sweep):**
  - SSN's distinctness is proved in scratch Lean (SW-LEAN) for the coin-bit map SW-NOISE stated. It is unlocked and on
    MH100w semantics marked draft.
  - SW-NOISE's 5-lane code covers 371,292 columns, enough for lm_head and V4-Pro.
  - With T3 measured, SSN leads at T 512 too (SW-PRICE: 12.6× / 24.9×). NCP was first there only while SSN's T3 was
    unknown.

**The tradeoff that sets every design's rank (coordinator's reading; attack it):**

- Under DistinctWrites at rank ≤ 3, a committed 32-bit word is credited at most about 64 units (SW-SEM: 32 at H32 alone,
  64 − 1024/L with every class semantic). SSN's words are cheap to compute from precomputed data dots, so SSN gets 32
  and must commit every k32.
- So committed bytes are at least 1/16 to 1/8 B a useful MAC. SHA-512 costs about 908 W1 units a byte at the floor, so
  hashing is roughly 60–125× stock in W1. T1's additive 1% then needs ShaSpeed to about 10⁻⁴ (SW-HASH: δ 2.7 × 10⁻⁵ at
  NCP's volume).
- **So every word-commit design is rank 4a overall, through ShaSpeed, whatever its noise.** The roads out:
  - **(a) Demote ShaSpeed** to a tight, CPU-checked instruction lower bound for SHA-512 on sm_90 (rank 2), with no
    sharing across instances and hash-compose as a counting theorem (N-SHALB, N-COMPOSE).
  - **(b) Raise the credit a committed byte at rank ≤ 3**: SW-SEM's saturating-noise route to L 64 or 128 by
    `CoverCount`; packed codes for quadratic noise (N-DENSE).
  - **(c) An assumption Daniel accepts that needs fewer commitments**: SAK on literally uniform operands, with designs
    that reduce to it exactly (N-SAKU).
  - **(d) Something outside the paradigm** (N-ALT), or a hash outside the one-hash rule (a what-if needing Daniel's
    ruling; label it so).
- **Road (c) is closed** (N-SAKU final, 08:51Z, note 20261029T0850Z-report-np-saku). SAK-U alone leaves at least the whole
  stock work uncertified, in every class of design (exact, served, padded). Beside the per-word floor it adds nothing at
  k32. Sparse commitments in M0 cost an exact design L/K of stock (proved in Lean), and committing outputs only fails
  for every design. N-SAKU moves to hiding as N-SALT (§4).
- **Road (d) (N-ALT final, 09:03Z, note 20261029T0903Z-report-np-alt):** no road outside DistinctWrites lowers the matmul
  part. Two roads move the hashing part, and both put the network certifier (auditor-trusted) on the GPU's port.
  - **R1:** every word is hashed by the certifier, so the prover needs no hashing premise. Rank 3 overall. Its T2 at
    400 Gb/s is 97.5× at T 512 and 665× at T 8,192, so it works for decode and small batches only.
  - **R1b:** the lead, with the salt and coins coming from the certifier after A_j arrives. T1 0.000%, T2 19× / 59× / 60×,
    rank 4a. It ties for first in N-CARD's space.
- **New term outside the core T1: egress.** The core spec gives the adversary no outside machine. In the full guarantee,
  OutsourceWithinEgress bounds outsourced work by egress bits. One coin bit of the lead carries about 350,000 MACs, so
  that bound is vacuous: 138–268% of stock at a 32-bit-a-token cap. It is the largest term of every design that hashes
  on the prover. Closing it to 1% needs a coin bit every 1,300–2,500 MACs. That's N-EGRESS's (N-ALT resumed), and a
  question for Daniel: does T1 include the network term?

## 4. The workers

"From" names the sweep worker resumed into the role (it keeps its context); "new" is a fresh agent.

| name | from | task | done when |
|---|---|---|---|
| **N-CARD** | SW-PRICE (07:35Z) | One modular calculator over §2's axes: every design is a composition of modules, every term traced to a run or art. Enumerate the product space, print the Pareto frontier at each rank (matmul and hashing parts separately) for Llama T 512 / 512 rows / T 8,192, V4 dense and experts, decode. Ingest every N-* card. Own the ranking. | `pouw_space.py` reproduces SW-NOISE's, SW-PREMISES' and SW-HASH's numbers on their designs and prints the ranked frontier |
| **N-PREM** | SW-PREMISES (now) | The assumption space, round 2, and the premise ledger: every premise's statement, rank (4a/4b), falsification, status, owner and the terms it closes; minimal premise sets per frontier design; new classes (memory capacity against precomputation and advice, energy as a pricing, information flow and fan-in cones, determinism, ISA completeness, SAK-U, protocol premises) | a ledger art and "what rank ≤ 3 overall needs", term by term |
| **N-SHALB** | new | ShaSpeed → rank 2: a tight instruction-count lower bound for a SHA-512 compression on sm_90's integer ISA, the best honest count, and the sharing argument across hm96 leaves (ASICBoost-style) | a CPU-checked bound, its gap to the best kernel, and the T1 residual at SSN's volume |
| **N-COMPOSE** | new | hash-compose as a counting theorem in MH100w (committed words must be materialized; SHA-512's writes aren't targets), and the atomic call and PeriodInventory as named Props closing hashing before the noise. Lean, or a counterexample | statements, proofs or a counterexample |
| **N-W1** | new | W1 stated as a refinement of Core/Accel (CardWithinAccel is locked); an opcode census of compiled sm_90a SASS; measured prices for every opcode, multi-output forms included (IMAD.WIDE, carry and predicate outputs, LDSM, uniform datapath) | the census art, the Lean bridge statement, and every opcode under 32 a register (or none) |
| **N-DENSE** | SW-SEM (now) | Raise credit a committed byte at rank ≤ 3: your saturating-noise route (full-range s8 lanes so `CoverCount` applies; the wide lanes as a finite rank-2 computation), `BamFloor`'s unchecked step, packed 16-bit codes for quadratic noise | whether L 64 or 128 is creditable at rank ≤ 3, with the design and its T1/T2 |
| **N-SALT** | N-SAKU (08:55Z) | Hiding through forced work, in parallel with N-HIDE: hiding randomness that the forced computation consumes, so no separate honest-only salt work remains. For example, a prover-secret noise lane per step, PRF-derived from a committed key and checked in the proof, giving each leaf its own entropy; or anything else that makes hm96's hiding come from committed, forced words | a hiding design whose honest-only work is ≤ 0.1% of stock, with what the proof checks and the premises it reads |
| **N-NOISE** | SW-NOISE (07:35Z) | The noise and de-noise space, exhaustively: a grammar over N × D × F, automated search for designs dominating `ssn_1l` (α, forming, coins, T3), lower bounds on what any design must pay (noise lanes a step, coin bits), attention and decode variants | the grammar, the search's output, and a dominance proof or a better design |
| **N-EXACT** | new | Exact-output designs (served O = A·Bᵀ) and every de-noising strategy: one-sided stacks, low-rank, cancelling stacks (prove NCP's 3K minimal), post-hoc and epilogue-folded corrections, paired calls; their premise sets (M1 needs the atomic call) | the best exact design at each T with its premises |
| **N-FP8** | new | FP8 E4M3 and BF16 serving on the H200: SSN-like lanes with exact small values, the accumulator's real semantics, distinctness under rounding | a verdict and the best floating-point design |
| **N-HIDE** | SW-HASH (now) | Close the hiding term: salt derivation forced in the drawn unit's proof, the mask's price, rows against proof cost, hiding randomness consumed by forced work. Then the hashing frontier at SSN's volume (bytes a MAC, T2, δ) | a hiding design at ≤ 0.1% of stock with its premises |
| **N-KERNEL** | SW-GPU (now) | `ssn_1l`'s kernel on the H200: s8 wgmma with 31 + 1 lanes, every k32 word exported and hm96-hashed (separate and overlapped), words checked against a CPU reference; T2 at Llama's shapes; the forming pass and coin latency timed | measured T2 with run ids |
| **N-FLOOR** | new | Cost-floor audit on the hardware: implement the most efficient program you can that produces the lead's committed words (and their hashing), and measure its priced cost against honest's; measure the price per register written of every alternative instruction route and engine the model's floor must cover | the measured cost per committed word of the most efficient program, and every route priced below the floor (or none) |
| **N-RED** | new | Red team on the theory: grinding and the success set, chosen inputs, free-set loopholes, reuse across calls, advice, coin expansion, the span caveat, hash-compose loopholes, ZK leakage of served noise, served output against committed output | a ranked attack list with severities and fixes |
| **N-COVER** | new | The whole served step: attention (QKᵀ, PV, softmax), decode, experts, lm_head, epilogue and nonlinear ops, each in W1; the best design per component; the whole step's T1 and T2 | a coverage table and the step's T1 |
| **N-LEAN** | SW-LEAN (07:35Z) | Lean for the lead: SSN's distinctness lemma, the grinding bound, and the path to locking DistinctWrites | proofs, axioms printed |
| **N-PPL** | SW-PPL (07:35Z) | T3 for every SSN variant (b1, b4, ssn_a, ssn2) and for N-NOISE's and N-COVER's attention variants, at 2K and 8K | measured deltas with intervals |
| **N-LIT** | SW-OLD (now) | The outside literature, mapped to our ranks: proofs of useful work from matmul, average-case and fine-grained matmul hardness and random self-reduction, proof-of-work hash optimization (ASICBoost) and SHA circuit lower bounds, bilinear lower bounds at small sizes, memory- and energy-based work models | an assumption catalogue with what each would buy here |
| **N-EGRESS** | N-ALT (09:10Z) | The network term N-ALT found: designs whose coins are dense enough (one bit every 1,300–2,500 MACs, perhaps per output element) that OutsourceWithinEgress bounds outsourcing to ≤ 1% of stock, with their T1, T2 and T3; how R1b's certifier coin release interacts; which of these could become the lead. Outside the core T1, for Daniel's ruling | the best design that closes the network term, priced in full, or the obstruction |
| **N-SMELL** | SW-SMELL (07:35Z) | Smell-check every N-* note and the ranking as they land | a running smell catalogue |

## 5. How designs are scored

- **Cards.** Every design goes to N-CARD as a card: its module per axis, its terms with sources, its premises. N-CARD
  posts the card format in its inbox by 08:15Z; until then, use SW-NOISE §5's card table.
- **Ranking:** overall rank (with 4a/4b, matmul part and hashing part shown), then T1 at T 512 and T 8,192, then T2, then
  T3. An open premise ranks 5. A design with an open finding from N-RED or N-FLOOR can't rank first.
- **Labels:** proved (Lean, with `#print axioms`), checked (CPU, with run id), measured (GPU, with run id), estimate.
  Never "locked" unless the lock has it.
- **Check every lead.** N-RED (on paper) and N-FLOOR (measured on the H200) test whatever leads at each interim.
- **Per monitoring period:** state any new bound per window, week, month and year at one H200, the reference datacenter
  and the world, on the one basis N-CARD fixes in its card format (SW-SMELL: five bases were in use, up to 4.5× apart).

## 6. Rules (every worker)

- **Your inbox is inbox/<NAME>.md.** Re-read it every 30 minutes:
  `git fetch origin cursor/compute-research-golden-inbox-ea24 && git show FETCH_HEAD:inbox/<NAME>.md`.
- **Note:** `~/.research/notes/lanes/compute-research/<UTC>-report-np-<slug>.md`, front-matter id, campaign: pouw-night,
  lane: compute-research, kind: report, status: interim|final, repo: compute-verification/verity, origin: <NAME>. It
  ends with a Smells section (.agents/skills/smell-check/SKILL.md) and an Asks section (what you need from whom; the
  coordinator routes it). Publish with `research notes sync --path lanes/compute-research`.
- **Data:** `research data put --kind evidence/v1 --file|--tree P --preserve`; cite the art: id. Results go in the store,
  never in Git.
- **Code:** push to cursor/compute-research-np-<slug>-<4 hex>. Never push to main, and open no PR.
- **Compute:**
  - CPU through `research run`. cpu-1..9 are up to Sat 31 Oct 20:00Z; cpu-10..12 are stopped. **cpu-1 is crowded**
    (`--on auto` lands most runs there): pick `--on vy-nebius-cpu-N` for N in 2..9, spread by your name's position in
    §4's table.
  - GPU only on vy-dc-h200-5, through gpu-lease and `research run`. Infra extended it to 15:00Z (lease 14:55Z; INF-362).
    Finish GPU runs by 14:45Z. Never the demo node vy-dc-h200.
  - Lean: the lean-proofs skill. Never share a writable `.lake` with another agent.
- **Labels:** estimates labelled; every measurement with its run id.
- **Messaging:** no Slack and no Linear (the handle is the coordinator's); your note and its Asks are your report.
  Never print secrets.
- **Timeline:** interim 1 by 09:00Z, interim 2 by 11:30Z, final by 13:30Z. The coordinator re-targets workers at 09:30Z
  and 12:00Z from what the interims show.

FINAL MESSAGE: a 10-line summary, your note id, art ids and branch.
