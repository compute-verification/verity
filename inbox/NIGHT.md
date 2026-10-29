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

## 7. Interim 1 synthesis and re-targets (coordinator, 09:40Z)

The platform paused every worker at about 09:21Z. All are resumed at 09:40Z on the rows below. Interim 2 is due by 11:30Z,
the final by 13:30Z.

### Where we are

- **The lead is now `ssn-1l-tag`** (N-CARD interim 1). It is ssn_1l plus N-RED's four row-tag lanes in step 1, against a
  coin-drawn private G.
  - T1 is 0.000% (0.420% with coins uncredited).
  - ssn_1l's kernel is built and measured (N-KERNEL; the tag kernel isn't built yet). T2 at T 8,192 is 57.5× beside and
    65.8× serial; at T 512 it is 11.1× / 30.4× with 64 rows and 36.0× / 61.6× with 512 rows. Every word is exact.
  - T3 is unmeasured for the tag lanes; ssn_1l's is −0.09% / −0.02%.
  - Runners-up: N-PREM's `ssn-1l-hw` (a 16-bit hash code a row, with no tag and no G), and N-NOISE's
    `AorB/rgap1/sv4+tag+nz` (0.030% / 0.012%).
- **Attacks that re-ranked everything:**
  - **The coin branch** (N-RED A1/A2, N-PREM). A prover that branches on the coins copies repeated rows, and on chosen
    inputs ssn_1l loses 19–100% of stock. Row tags fix it.
  - **The middle-segment skip** (N-SAKU, proved premise-free): every exact-output design fails on Llama in M0.
  - N-LEAN: SW-NOISE's g·n bound is false, and against a coin-reading prover today's fixed-program bound is vacuous.
  - **So `adaptive-dw` (DistinctWrites for programs that branch on the coins) is the lead's critical open premise.**
    It ranks 5 until proved.
- **N-FLOOR's incidents:**
  - The host CPU is uncounted: it can take 4.83× stock of credited hashing. Fix: PeriodInventory with HostsCounted, in T1.
  - SHA-512 is credited at 3,632 registers a compression where 3,532 suffice for chained compressions (N-SHALB). Fix:
    credit at the best known count.
  - Otherwise no route on the card is under the floor (N-W1: 58 probes; N-FLOOR: every engine, NVDEC at ≥ 353k a word).
- **Roads:** (a) is closed at rank ≤ 3 (N-SHALB: MH100w's bound for SHA-512 is 16 registers). (b) is closed at rank ≤ 3
  (N-DENSE: neither L 64 nor L 128). (c) is closed (N-SAKU). (d) gives R1 / R1b and the egress term (N-ALT).
  - So the lead is rank 4a through **ShaWrites**. N-COMPOSE proved hash-compose as a counting theorem given ShaWrites,
    which is strictly stronger than ShaSpeed, and the rank-3 premise CarriedWords.
- **Hiding:** N-HIDE's **L1** (one hm96 instance a call, over a subtree root) plus **(a)** (the PRF checked in the
  drawn unit's proof) passes at every point, 0.025% / 0.050% at 64 rows. **L2** (private roots, one instance a window)
  is ≈ 0 everywhere, but it is a protocol change. N-LIT's counter-mode pad (H20) drops M_key·y and c; that is an
  argument, not yet checked.
- **Exactness for free:** de-noising inside stock's epilogue costs no extra instruction (N-EXACT, proved). A served
  design can serve A·Bᵀ exactly, with its committed words still noisy.
- **Floating point:** FP8 fails (N-FP8): no pinned design passes both T1 and T3.
- **The lead's open premises** (N-CARD), each to an owner: adaptive-dw, PeriodInventory with hosts, ssn-tag-distinct,
  g-private, copy-floor, ssn-success-set, salt-forced, mask-optimal, lane-semantics and one-instruction-minimal; and
  ShaWrites (4a), CarriedWords (3) and W1 (3).

### Re-targets (09:40Z)

| worker | until 11:30Z |
|---|---|
| **N-DENSE → N-ADAPT** | `adaptive-dw`: state and prove DistinctWrites for programs that branch on the coins (branches, predicates and copies priced as N-W1 / N-FLOOR measure), for the tag lead and `ssn-1l-hw`. Lean with N-LEAN. Or the exact obstruction, and the premise that closes it |
| **N-LEAN** | `ssn-tag-distinct` and `ssn-success-set` for the tag lead and hw; the adaptive lifting with N-ADAPT; the locking path |
| **N-RED** | Attack `ssn-1l-tag`, `ssn-1l-hw` and `AorB/rgap1/sv4+tag+nz`: G's privacy, tile-level copies (`copy-floor`), adaptive branches, the epilogue de-noise, L1 hiding |
| **N-FLOOR** | Measure what the adaptive and copy premises need on the H200: the price of branching and predicates per coin bit read, TMA / bulk tile copies per committed word (`copy-floor`), and the host's θ rows (HostsCounted) |
| **N-KERNEL** | Build the tag lead (4 tag lanes + G) with L1 hiding and (a)'s PRF in the drawn unit; measure T2; fix root(A_j)'s latency |
| **N-PPL** | T3 for `ssn-1l-tag`, `ssn-1l-hw`, `AorB/rgap1/sv4+tag+nz`, and the epilogue-de-noised lead (should equal stock exactly) |
| **N-EXACT** | Apply your epilogue de-noise to the served lead: exact served output, noisy committed words. Show the skip doesn't apply and the adversary gains nothing; price it |
| **N-HIDE** | Settle L1 + (a) as the lead's hiding module (spec, what the proof checks, premises), with L2 as the protocol option; with N-SALT on the pad |
| **N-SALT** | N-LIT's counter-mode pad (H20): construction, hiding argument, what the proof checks, cost; against L1 + (a) |
| **N-SHALB** | Road (a) is closed at rank ≤ 3. Make 4a sharp: ShaWrites' constant (3,532 chained), sharing across leaves under L1 (midstates, ASICBoost-style), and the premise text with its δ |
| **N-COMPOSE** | PeriodInventory with HostsCounted inside T1 (N-FLOOR's incident 1), the atomic call, CarriedWords; proofs |
| **N-W1** | lane-semantics and one-instruction-minimal; finish the census and the Lean bridge; prices for predicates and branches |
| **N-PREM** | The lead's premise ledger with owners; minimal sets for tag / hw / AorB; the rank verdict |
| **N-CARD** | Re-rank with hashing credited at 3,532 (chained), the host term, L1 + (a) hiding, epilogue de-noise and the network column. Ranked frontier by 11:15Z |
| **N-NOISE** | Interim 2 with tiles; tag and hw dominance; the best A1/A2-closed design; the densest coins for N-EGRESS |
| **N-COVER** | The whole step with the tag lead and epilogue de-noise; attention; decode with N-FP8 |
| **N-FP8** | Finish the BF16 verdict and the int8-served FP8 route; then the lead at decode on the H200 (m 1–64, mma.sync against padded wgmma; T1's shape term and T2) |
| **N-EGRESS** | As tasked |
| **N-LIT** | Adaptive lower bounds (decision trees, branching programs, adaptive algorithms with oracle access); the pad's literature |
| **N-SMELL** | As tasked; a pre-read of the final's claims by 13:00Z |

### Addendum (10:45Z): N-RED's final (note 20261029T1039Z-report-np-red)

- **A2r breaks the lead and both runners-up.** Chosen data cancels each step's public part and alternates the
  cancel, so each row walks back over values it already holds. It writes about 7 of 67 words (K 2,048) and copies the
  rest: **94.8% of stock on `ssn-1l-tag`, 94.6% on `ssn-1l-hw`** (checked), and by argument AorB and `+tag+nz` too.
- **What survives at cap 32:** the lead plus **`blc4a-t64-e7`**, a private coin lane per tile and column with 7-bit
  coins, at 0.32% / 0.35% including egress. With 4-bit coins (`blc1a-t128-e4`) it leaves 3.2%. Its requirements:
  - the coins come after A_j: the certifier hashes A_j before releasing any coin, one draw a slot. Released before A_j,
    the attack takes 103%;
  - R1b's per-leaf certifier salt (one a 16 KiB leaf) bounds egress at 0.097% / 0.19%;
  - N-HIDE's L1 + (a) needs the words leaf in the record before the window's draws, or about 119× stock of hashing is
    skippable. Prefix binding costs +11% of word hashing.
- **The alternative fix:** R1's certifier check (b) credits only an entry's distinct values. It over-credits nothing,
  and costs honest 0.09–0.20%.
- **Copy models** decide e4 against e7. Model P lets copies of one word be free with free branches; it is T1's
  current reading. Model C is N-LEAN's CopyGrain(4), and model W copies a whole 32-lane tile. **Coordinator default:
  model P, so e = 7** (conservative; N-RED recommends it). Under C or W, e4 suffices. That is for Daniel, and N-FLOOR's
  copy prices bear on it.
- **New lead to build and measure:** `ssn-1l-tag + blc4a-t64-e7` with coins after A_j.

### Addendum (11:30Z): adaptive-dw (N-ADAPT final, note 20261029T1124Z-report-np-adapt)

- **Proved in Lean** (`adaptive_entries_le`, `adaptive_registers_le`, `adaptive_lanes_le`; axioms clean; r20261029-105546-5885):
  for any program that branches on the coins, cost ≥ check (b)'s credit (an entry's distinct values), under models
  P, C and W. Two conditions: (i) no two entries share a value at the model's copy grain; (ii) the initial memory holds
  none of the words (A5).
- **On the new lead,** model W pays everything (b) credits. Bare model P doesn't: masked copies across identical chosen
  rows are credited but not paid, **about 8.9% of stock** (checked, r20261029-111608-21b8). CopyGrain(4) is false on
  sm_90 (N-FLOOR), so C reduces to P.
- **New coordinator default (replaces 10:45Z's "model P"): T1 rests on model P plus N-FLOOR's measured floors.** A
  selection on a lane's own bit costs ≥ 2.24 writes (`select-floor`), and an on-SM copy ≥ 1 write a word
  (`copy-floor`, word grain); both are rank 1, measured. Under them the masked channel is 0, by argument; the Lean step
  that prices selections is N-ADAPT's next task. Keep e = 7 unless N-RED shows e4 survives under these prices. For
  Daniel: P plus measured floors, or W (lane semantics, rank 3).
- **Correction (N-ADAPT's smell):** "(b) over-credits nothing" (10:45Z addendum) holds only with N-RED's cross-entry
  condition (i).

## 8. Interim 2 synthesis and the final push (coordinator, 11:55Z)

### The lead: `ssn-1l-tag + blc4a-t64-e7`

It is the tag lead plus three odd 7-bit coin lanes for each 64-row tile, step and column, on step 1 too. The
certifier streams them raw after A_j (R1b). Hiding is L1 + (a), and de-noise is N-EXACT's fold.

| target | value | source |
|---|---|---|
| T1 (core) | 0.31–0.33% under bare model P (all of it the A2c / A8 + walk residual p*); coins credited 0; forming ≤ 0.02%. Under P plus measured floors, N-ADAPT's masked channel is 0 (argument; Lean step in progress) | N-CARD 1100Z, N-EGRESS final |
| T1 with N-LIT's weighting A′ = (1, 127, 1) | p* falls 94.5×, to about 0.0035% (estimate; same coins and stream) | N-LIT 1108Z, **to check** |
| network term (beside T1) | **Corrected (N-EGRESS final 2, 1224Z):** as stated, 0.097% / 0.188% at cap 32 and 0.97% / 1.88% at cap 320 (fails at the bar); with A′ = (1, 127, 1), 0.056% / 0.108% and 0.56% / 1.08% (fails at the bar). **`blc4a-t64-e7-rot`** (A′ = (127, 1, 1) on even rows, (1, 127, 1) on odd) gives 0.037% / 0.072% and 0.37% / 0.72% at every point; it needs a per-row-class fold (unbuilt) and +1.1% stream | N-EGRESS final 2 |
| T2 | 10.2× / 19.9× (T 512, 64 rows), 36.8× / 64.1× (512 rows), 54.9× / 66.0× (T 8,192), sync / async (estimate). The kernel is built, and its H200 timing r20261029-110957-0b45 is queued | N-CARD, N-KERNEL 1115Z |
| T3 | **served noisy (decision 10), coin lanes folded exactly.** N-PPL's CPU bench, which doesn't draw the coins and takes the fold as exact: −0.164% / −0.021% (2K / 8K, 5 seeds) against the bench's int8, ≤ +0.14% / +0.09% at 95% against vLLM's int8. xu (exact) is the 22-bit recipe bit for bit. The fold's exactness is checked on the H200 by N-KERNEL | N-PPL 1259Z, N-KERNEL, N-SMELL |
| hiding | 0.0002–0.05% at every serving point (L1 + (a)); decode 0.10% at 16 rows; keyed leaf 0 on a ruling | N-HIDE, N-SALT |
| hashing credit | **per form:** words 3,515.38 registers a compression (16 KiB chunks), A 3,449. A flat 3,532 over-credits 1.11× stock | N-SHALB 1126Z |
| rank | matmul part 3 once settled (H32 1; select-floor and copy-floor 1, measured; W1 bridge proved locally; lane-semantics 3; one-instruction-minimal checked). Hashing part 4a (ShaWrites, which hash-compose needs, N-COMPOSE final). Protocol premises 3 (trusted-host, certifier-coins, certifier-salt, atomic call, PeriodInventory with hosts). **Effective 5** until the open ones close | N-PREM, N-COMPOSE, N-W1, N-FLOOR |

### Unreconciled, to settle by 13:15Z

1. N-LEAN's "the floor survives only if b ≥ c·M, so pricing branches can't close the adaptive premise", against
   N-ADAPT's "P plus measured floors closes it". Are a coin lane's merges per digit read small enough that both
   hold? Owners: N-LEAN and N-ADAPT.
   **Resolved (N-ADAPT final 2, 1227Z):** both hold, because they count different things. N-LEAN's floor counts every
   word; check (b) never credits a kept word. On blc4a a word is kept with probability ≤ 3/1024 and a whole tile with
   ≤ 3.6e-21 a tile and step. The tile-grain instance (`adaptive_registers_le` with tile registers) is proved: residual 0
   given (i) at tile grain (checked: 0 excess in 8 cases) and A5.
2. N-NOISE's lower bound: a served design with independent noise steps loses ≥ 14.3% to A8 at T3-safe variance. Which
   class is the lead in (exact fold, so A5)? Owners: N-NOISE, N-RED and N-EXACT (A5).
3. Serving noisy or exact. N-COVER: exact serving frees the step's other lanes (+8.4 points of coverage). N-EGRESS and
   N-EXACT: the lead serves exactly through the fold. Owners: N-COVER and N-CARD.
4. N-SMELL's open items: the step-1 hole in the layout the card ranks, so rank the layout with lane coins on step 1
   (103); the egress statement's amendment and A_j's bytes (92, 95); the fold's range requirement (105); sampling's skip
   at today's K (99); hashing R per form (101).

### Decisions for Daniel (gathered; each with a default)

**Daniel, 20:27Z (ASK-189 A′, through top): "Okay, sure. On all the things above."** That takes the night program's
seven defaults and builds the lead. It is consistent with his 16:00Z (ShaWrites) and 19:37Z (tile grain, the keyed
leaf) decisions, and with the LEAN workstream.

1. Is rank 4a, ShaWrites / ShaSpeed on SHA-512, "milder than SAK"? Default: yes, reported as 4a.
   **Decided, Daniel 16:00Z: yes** ("I'm fine with it taking 3,500 registers"). The LEAN program (inbox/LEAN.md) takes
   ShaWrites as a named assumption.
   **Measured (N-SHALB final, 1234Z): the best honest committer built tonight** writes 3,517.36 registers a compression
   against R = 3,515.38, bit for bit with the host. That is 0.056% over R, short of δ (4.3e-5), so it **leaves 0.131×
   stock of honest hashing uncredited, 13× T1's budget** (the served committer: 5.0–7.7×). Only Daniel's "the honest
   program is the best possible honest program" waives it, so the hashing part's T1 rests on that ruling as well as on
   ShaWrites. It is also 19% slower than served.
2. The copy model. **Default revised at 12:05Z: `tile-grain`**, model W on the tensor pipe (N-FLOOR final,
   20261029T1148Z-report-np-floor). The lead's words leave the tensor pipe in tiles of ≥ 128 words, so a copy saves
   time only when a whole tile repeats; a lane's predicated copy costs 13.5–50.5 units more than computing the word.
   **P plus measured floors is withdrawn**: select-floor and copy-floor hold only where issue slots bind, and both fail
   beside the lead's wgmma. Under tile-grain, N-ADAPT's proved W case applies, its masked copies save nothing, and
   4-bit coins suffice (N-RED §4). Bare P stays in the frontier as the conservative column. N-FLOOR restated
   tile-grain as a work premise at **rank 3** (its shapes are rank 1, its counting rule rank 3) and measured the lead's
   words leaving the tensor pipe in 2,048–4,096-word tiles a warp.
   **N-RED final 2 (1227Z) sharpens it:** literal P certifies no design. The lead's open hole is rows meeting within a tile
   (birthday pools, because the coin cancels between a tile's rows): 46.1% under literal P, 16.5% with each steering
   decision at the 32 floor. N-FLOOR's measured prices close it, and so does the wide tag at the floor, so **the wide tag
   is a stated requirement of the weighted lead**. Tile grain holds at the lead's own wgmma width (no whole tile repeats at
   256 rows). At m16n8 tiles it would leave 3.9–4.2%, which needs an `mma.sync` adversary whose cost is unpriced (open).
   The options for Daniel: literal P (nothing passes), priced decisions, tile grain (rank 3) or measured prices.
3. model-public (chosen weights): the coin lanes close it, so the lead doesn't need it (default: not taken).
4. Hiding: `pad-leaf-ruled` (keyed leaf) and L2 (private roots). Default: L1 + (a), no ruling needed.
5. Does T1 include the network term? Default: reported beside T1; the lead closes it either way.
6. Whole-step coverage, softmax mostly: it misses 1% at Llama 1 × 8K and decode (N-COVER). Default: reported beside T1.
7. The certifier as coin source (R1b), which needs trusted-host. Default: taken.
8. Exactness past 2²² (the fold). Default: "exact on the stated range".
9. **Stock's price at decode** (N-COVER final): T1 against stock's cheapest instruction at one row (12.02 units a MAC)
   or one unit a real MAC. At batch 1 that moves Llama's step from 0.31% / 0.81% to 0.68% / 1.49%. Default: report
   both.
   It now decides one-row decode (N-FP8 final 2, `cert-call`, G's forming 0): the linears' T1 core is 1.21% at m 1 in
   W1 (fails; pad64's shape 0.60% + N-NOISE's term 0.60%) and 0.65% against stock's instruction (passes). From m 2 it
   passes in W1 (~0.93%); m 16 0.67%, m 64 0.19%.
10. **Serving: noisy (coordinator, 12:25Z, settles §8 item 3).** N-COVER: exact serving (xu) adds +2.0 to +9.2 points of
    coverage on Llama, and noisy serving's T3 shows no detectable loss: at most +0.14% (2K) and +0.09% (8K) at 95%
    against vLLM's int8 (N-PPL final, 20261029T1259Z-report-np-ppl; the lead as served is −0.048% / +0.026% there). The lead serves noisy, with its coin lanes
    folded exactly; xu stays the option for exactness.

### Final push (12:00–13:30Z)

Finals by 13:30Z. **N-CARD's ranked frontier by 13:15Z.** Each inbox has its line.
