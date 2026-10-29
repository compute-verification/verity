# PoUW sweep: work bounds milder than SAK on the H200 (Daniel, Thu 29 Oct 04:15Z = Wed 9:15 PM PDT)

Coordinator: compute-research (bc-ebc09a9a-ce64-52b5-b3b6-2c8862c7e9ac). Issue: CMP-294. This extends inbox/CORE.md; read it,
especially §2a and §6–§10.

## The spec (Daniel's rulings, CORE.md §2a)

- **One matmul** O = A·Bᵀ, both operands committed before fresh noise. Each call is atomic, with fresh noise every call (no
  reuse). Llama-3.2-1B int8 on one H200, at T 512 (the demo's 64 real tokens) and T 8,192. Report a DeepSeek-V4-like shape
  too.
- **T1, hard:** cost(H) − min over adversaries of cost(adv) ≤ 1% of cost(stock), in priced work (MH100 / W1 units: one dense
  int8 MAC = 1 unit, every instruction at its measured price a register). Not wall time.
- **The honest program H is the best possible honest program.** Hashing is credited, under ShaSpeed.
- **T2:** honest overhead ≤ 100× stock. **T3:** perplexity rise ≤ 5% (WikiText-2, 2K and 8K).
- **Assumption ranks:** 0 a theorem; 1 a measured hardware floor; 2 a small CPU-checkable fact; 3 model faithfulness;
  4 SAK; 5 a hardness conjecture. **Rank ≤ 3 beats SAK.** Daniel: SAK "is secretly assuming that whatever structured noise
  we added is good."
- **Terms:** a dot product is one output element of a matmul (never "cell"), and a proof block is what C-Flock proves.

## Where we are (tonight)

- **DistinctWrites under H32 is the mildest bound we have.** `DistinctWritesMH100(Floor)` is in
  Security/Definitions/Pouw/Assumptions/Dimension/H100W.lean, and H32 is measured on the H200 (h200Measured = h100Floor,
  run r20261029-021534-a5a3):

      cost(adv) ≥ 32 · #{committed, pairwise-distinct, noise-dependent 32-bit words}

  - **It is modular in the noise.** NCP and Dither differ only in how they noise and de-noise.
  - **Commitment density is the knob.** Committed words are forced. Anything else is gap, unless another premise covers it.
- **NCP with a word every k32** (CORE-TIME, note 20261029T0240Z-report-core-time; CORE-NCP, note 20261029T0110Z-report-core-ncp):
  - its gap is the final word (32/K) plus forming (24/n + 24/m): 2.10% at T 8,192 and 38–39% at T 512;
  - T2 is 155–186× at T 8,192, with hashing 98%;
  - T3 is +3.68%.
- **Candidate fixes (the coordinator's; unverified):**
  1. **Commit the formed operands X and Y**, so DistinctWrites forces forming. That is 32 units per 4-entry register, the
     floor price p = 8 an entry. The extra hashing is 3/n + 3/m B a MAC.
  2. **A premise on what one tensor instruction computes:** a k32 partial sum, not "any function". Committing every L would
     then force L/32 steps a word, cutting hashing L/32 times.
  3. **The final-word lemma:** the free set is the call's own words, not their span (CORE-TIME).
- **ShaSpeed:** with about 150× hashing credited, ShaSpeed has to hold to about 0.0015% (CORE-SMELL C27). Less hashing
  also lightens that burden.

## Goal

Rank designs by assumption rank, then T1, then T2, then T3, on the H200. Every design is (noise and de-noise) × (what is
committed and how densely) × (the premise set). Find the best one at T 512 and at T 8,192, and say what it still needs.

## Workers

| Name | Task |
|---|---|
| SW-LEAN | Lean: candidate 1 (commit X and Y; DistinctWrites forces forming) and the final-word lemma. Prove them, or find the counterexample. |
| SW-SEM | Candidate 2: tensor and CUDA-core instruction semantics as named premises against MH100w; the bound for committing every L; the counting lemma; margins; how to falsify each. |
| SW-PREMISES | Sweep reasonable accelerator-model assumptions: rate and issue caps, register-file and operand delivery, fan-in and data flow, minimal k-steps, memory (G1), clocks, determinism. For each, what gap term it closes, its rank, and how to measure it on the H200. |
| SW-NOISE | Sweep noising and de-noising compatible with DistinctWrites: NCP's 3-block cancellation, 2-block stacks, Dither's periodic noise and correction, low-rank, Kronecker, block-diagonal, permutation, int7 and wrap-free designs, designs whose final word is the useful output. Minimize honest work not forced. |
| SW-HASH | Sweep interior hashing and commitment: density, word width and packing, committing operands, tree and row layout, where hashing runs, cheaper commitments, ShaSpeed's precision burden. Target T2 ≤ 100× at T 8,192. |
| SW-PRICE | A calculator: (noise × commitment × premises) → T1 terms (final word, forming, de-noise, uncommitted steps, hashing slack), T2 and T3 inputs, for Llama-3.2-1B at both T and a V4-like shape. Builds on `ncp_time.py` (cursor/compute-research-core-time-60d4) and CORE-NCP's pricing. Writes the ranking. |
| SW-GPU | H200 measurements for SW-PRICE: forming cost an entry, de-noise corrections, per-call coin expansion, SHA-512 row hashing, a prototype every-k32 commit kernel. vy-dc-h200-5 only. |
| SW-PPL | T3 for the candidate noise and quantization formats SW-NOISE proposes, on Llama-3.2-1B (WikiText-2, 2K and 8K, against stock int8). |
| SW-OLD | Archaeology: every work-bound argument in the notes and the Lean not yet used (Steiner rigidity / Theorems B and C, Theorem W's ½, fan-in floor, scalar route, FOLD-OPT barriers, RelSec K/S, TS headlines, the Sep research plan, hardness-shaped matrices, new-crypto, PHYS-INVENTORY CMP-146). Say what each bounds, its premises, and whether it combines with DistinctWrites. |
| SW-SMELL | Smell-check every SW note and the ranking as they evolve. |

## Rules (every worker)

- **Reading:**
  - AGENTS.md first; `git show origin/main:AGENTS.md` wins.
  - Then this file, CORE.md, and the catalogs (catalog-lean.md, catalog-notes.md, catalog-code.md):
    `git fetch origin cursor/compute-research-golden-inbox-ea24 && git show FETCH_HEAD:inbox/SWEEP.md`.
  - Your inbox is inbox/<NAME>.md. Re-read it every 30 minutes.
  - Other workers' notes: ~/.research/notes/lanes/compute-research/*-report-sw-*.md (`research notes sync`).
- **Note:** ~/.research/notes/lanes/compute-research/<UTC>-report-sw-<slug>.md, with front-matter id, campaign: pouw-sweep,
  lane: compute-research, kind: report, status: interim|final, repo: compute-verification/verity, origin: <NAME>. It ends
  with a Smells section (.agents/skills/smell-check/SKILL.md). Publish with `research notes sync --path lanes/compute-research`.
- **Data:** `research data put --kind analysis --file|--tree P --preserve`; cite the art: id.
- **Code:** push to cursor/compute-research-sw-<slug>-<4 hex>. Never push to main, and open no PR.
- **Compute:**
  - CPU through `research run --on auto`, or `--on vy-nebius-cpu-12 --pool research`. If one node's slots are full, use
    another.
  - GPU only on vy-dc-h200-5, through gpu-lease. Never the demo node vy-dc-h200.
- **Labels:** label estimates as estimates, and give every measurement its run id.
- **Messaging:** Linear writes only via `research msg linear --as compute-research comment CMP-294 "TEXT"` (options before the
  subcommand). No Slack. Never print secrets.
- **Timeline:** interim note by 05:45Z, final by 07:30Z.

FINAL MESSAGE: a 10-line summary, your note id, art ids and branch.
