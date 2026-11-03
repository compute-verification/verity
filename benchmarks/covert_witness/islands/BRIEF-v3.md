# Training-efficiency search, campaign v3: the brief (formalism v2, the OLMo mix)

You are a search agent in the training lane (@training), in the Verity repo (/workspace). If your VM is new, first run
`git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh | bash -s -- /workspace`.
Never print secrets or the environment. No Slack, Linear or PRs to main. No training on your VM, only seconds-long CPU
checks (AGENT_GUARD_LOCAL_OK=1). GPU work goes through the harness's launcher, never vy-dc-h200*.

This brief replaces the v1-era one. Campaign v3 starts at the integrator's board line `[TOOL] v3 pinned at <commit>`.
Until then, keep working at the v2.3 pin, whose results are now the FineWeb-Edu pilot.

## The spec

- The formalism is v2: note:training-20261102T2215Z-report-training-formalism-v2 (research-notes repo,
  lanes/training/). Its words are the ones to use: θ, θ′, token blocks, affected gates A, reachable gates R, σ, w,
  s = c/w, τ, x₀, s*, u, u_in, a, Δ, d, Δ_grad, A1, A1′, A2, the counting lemma, Slowdown, Fit, Security. Retired: φ,
  Dep, covers, classes, the r sweep.
- The experiment is note:training-20261103T0030Z-draft-experiment-consolidated. The two plots are
  `training/frontier-branches` (loss against total compute: the dashed frontier, a blue and a red branch per start)
  and `training/algorithm-policy` (each run's p = |C|/H against its ρ, with iso-S diagonals).

## What you are optimizing

For a start (θ, ℓ₀) and a target ℓ* < ℓ₀:

- H(ℓ*) is the least compute after the start of any valid unconstrained run reaching ℓ*: anyone's, any method.
- W*(ℓ*; c, k) is the least |C|·ρ over valid runs reaching ℓ*, where ρ is the run's slowdown under the policy (c, k).
- S = W*/H = p·ρ, with p = |C|/H.

Two jobs:

- Lower H: a better honest method. That's the reference directions' job, and anyone's.
- Lower W*: a run whose p·ρ is small at some (c, k). That means a low ρ at a modest compute overhead.

A reported S is the best attack found, so an upper bound on the true S. ρ = 1 means the theorem proves no slowdown
there, never "no slowdown".

ρ per checkpoint (te_core 4.0, `te_core.formalism.at`), from quantities measured on the run's traced step:

- **Fit (a construction):** if Δ + sτ ≤ k (compute), 2Δ_grad ≤ k (reduction) and Δ_t + Δ_grad ≤ k (update
  application), then ρ = c/(c − d), basis `fit`.
  - Δ_t is the weights in the form the update rule reads and writes: a straight-through run's fp32 latents, 32 bits a
    value; for every other run Δ_t = Δ.
  - Record latent_bits in notes.delta_grad (coordinator's ruling, 3 Nov 00:47Z).
- **Slowdown (a lower bound under A1 and A1′, basis `target`):** ρ ≥ σ(1 − x₀)·s/s*, with x₀ ≤ max{x : u_in(x) ≤ k}
  and s* ≤ max_x x(k − u(x))/a.
- **A known weakness:** v2's A1 doesn't charge a block's reads of earlier tokens' keys and values. If a route's ρ rests
  on that (a narrow cut, as with an adapter on the embedding), tag it `loose-end-attention` in your DIGEST line.
  Daniel decides whether A1 changes.

The attack surfaces:

- Δ, Δ_t and Δ_grad, for Fit.
- σ, u, u_in and a, for Slowdown, at the compute p they cost.
- A code of yours that lowers a or u (an activation code, an update encoding) is a finding: it refutes A1 at the old
  value.

## Inputs

- **Data:** OLMo 2's stage-1 pretraining mix, tokenized with one 32,000-token byte-level BPE trained on the training
  split. Held-out documents are split into a search pool, a promotion shard and a sealed final shard. The paths come in
  the pin line. Loss is held-out bits per byte.
- **Starts:** ref-optim's v3 ladder, one dense Llama-style recipe with tied embeddings, published in batches. Each comes
  with its digest, shape and C0.
  - On the frontier, at about 20 tokens per parameter: s-t20 (20M non-embedding), m-t20 (60M) and l16-t20 (200M).
  - Over-trained, off it by design: s-t200, m-t200, s-t2000, and l16-t180 if it fits.
  - Start from what's published; the board says when each lands.
- **θ is free information, not a fixed architecture.** Grow the model, prune it, distill it, upcycle it into experts,
  add adapters. Every new parameter is θ′, and the formalism prices it.
- **Evaluation cost:** the harness records each scored checkpoint's forward gates per token (active experts only for
  MoE). Scoring caps it at the larger of the start's cost and the cost of the compute-optimal model at that loss,
  pending Daniel's OK. No cap at run time.
- **Targets:** ℓ₀ less 0.5%, 1%, 2% and 5%, plus the scaling plot's compute-equivalent gains (a tenth of, and all of,
  the start's own training compute).
- **The dial:** c in tokens of the start's own forward pass (one session per block), k in bits. The default is 4,096
  tokens and 10 kB, and the served demo's measured block (3.0e13 gates, 145,568 bits) is marked.

## Families to try

Screen each mechanism on s-t20, confirm on m-t20, and give each its own `--note` tag. For every new tag, put a
plain-words label and a description of at most 200 characters in your next DIGEST line; the methods table is built
from them.

- **Full training, optimizers and schedules:** AdamW, Muon, Lion, Adafactor, SOAP and Shampoo; warmup-stable-decay,
  rewarm handling, learning rate and batch. This sets H.
- **Growth and upcycling:** width or depth growth from θ, function-preserving, and MoE upcycling. Now allowed.
- **Distillation:** θ as the teacher, into a larger or a differently shaped student.
- **Training from scratch:** a point on the dashed line.
- **Adapters and PEFT, merged or not:** LoRA, DoRA, adapters, prefix tuning, IA³.
- **Shallow and frozen-tail updates:** only the top blocks or the head.
- **Compressed training state:** 1-bit or sign updates with error feedback, top-k, PowerSGD, low-precision momentum
  and messages, rotated codes. These attack Δ_grad and Δ_t.
- **Update codes:** quantized, sparse, low-rank or entropy-coded deltas, and seed logs. These attack Δ and u.
- **Activation codes:** low-precision or entropy-coded activations at a cut, and bottlenecks. These attack a.
- **Seed and zeroth-order methods:** MeZO and shared-seed perturbations.
- **Data and curriculum:** selection, ordering, deduplication, synthetic data from θ, repetition.

## Long runs

Daniel wants more decades of the branches.

- **10× runs:** each family runs one continuation from s-t20 at about 10× its C0 (about 3 GPU-hours at the ladder's
  throughput).
- **100× runs:** the two families the board names run one each at about 100× (about 30 GPU-hours).
- **The long tier:** both use the harness's long tier, whose flag the pin line names, and only when the board assigns
  them.

## Harness v3 and how to launch

- Merge `origin/cursor/covert-witness-4207` at the pin commit into your branch `cursor/te-<direction>-4207`. Then
  launch from a committed checkout:
  `python benchmarks/covert_witness/launch.py START --island NAME --note "MECHANISM ..." [--node NODE] -- TRAIN_ARGS`.
  There's no class; `prepare.submit` takes none.
- **Nodes:**
  - vy-nebius-1 and vy-nebius-3 run the search, one GPU a run, at most two of your runs in flight.
  - vy-nebius-2 trains the starts. The ladder outranks the search for GPUs.
- **Fixed side:** never edit the evaluator, the checker, the launcher, the counter or te_core. If one is wrong, or you
  find a way to score without really lowering loss within the rules, post a `[LOOPHOLE]` line and don't use it. Shared
  tools go to the toolsmith (`[TOOL]`).

## Board

- **Writing:** add lines to `benchmarks/covert_witness/islands/<direction>/BOARD.md` on your own branch and push.
  Every line starts with the UTC date and time, `YYYY-MM-DD HH:MMZ` (for example `2026-11-03 05:10Z`), then your
  direction and the kind: FINDING, NEGATIVE (failed ideas matter as much), TOOL, LOOPHOLE, QUESTION or DIGEST, with
  record ids. Undated lines are taken as Mon 2 Nov from 08:45Z and Tue 3 Nov before.
- **Reading:** the merged board is `benchmarks/covert_witness/BOARD.md` on cursor/te-digest-4207. The coordinator's
  DIGEST lines there are rulings.
- **Pace:** explore first, at least three distinct mechanisms an hour, each screened cheaply (a 3-point learning-rate
  sweep at most). Then deepen what works, keeping a fifth of your runs on new ideas.
- **DIGEST every 90 minutes:** your best records (ids, start, bits per byte, |C| in gates, Δ, Δ_t, Δ_grad, ρ and S
  at the default dial), what you learned, and what's next.

## Schedule

- **Last launch:** Wed 4 Nov 08:00Z.
- **FINAL line:** your best three record ids per start and target, by 08:30Z.
- **Sealed finals:** every point that sets an envelope reruns with three seeds on the sealed shard, 08:30–14:00Z.
- **The final panel:** by 16:00Z. The nodes end at 18:45Z.
