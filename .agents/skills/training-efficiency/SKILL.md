---
name: training-efficiency
description: >
  How to run, or search in, a training-efficiency search in Verity: the least compute that lowers a registered start's
  loss to a target ℓ*, honestly and under a policy (c, k), scored in training formalism v1's terms (Fit, Slowdown,
  Security; |C| in Boolean gates; ρ, p = |C|/H, S = W*/H). Covers the covert_witness harness (launch.py, the pin, an
  island shipped alone, --class, --arms), the boards (an island's BOARD.md, merge_board.py's digest, DIGEST, FINAL and
  LOOPHOLE lines), scoring (honest = least compute of any method by a complete schedule, checkpoints as labelled upper
  bounds), and the coordinator's side: one pin per campaign, 45-minute checkpoints, nodes and leases (CW_NODES, --node,
  drain_s, the evictor, research keep) and final reruns on the sealed set. Use when launching, joining or coordinating a
  search direction, scoring or reporting an honest or covert record, merging the board, or planning a campaign's pin,
  nodes and finals.
---

# Training-efficiency search

A search measures, from registered weights θ (a start and its loss ℓ0), the least compute that reaches a target loss ℓ*,
honestly and under a policy's slowdown. Its question and quantities are Daniel's training formalism v1
(`~/.research/notes/lanes/training/20261102T1817Z-report-training-formalism-v1.md`, with the experiments and plots it
feeds in `20261102T1715Z-draft-training-efficiency-experiments.md` beside it). The harness is `benchmarks/covert_witness/`
and its counter `benchmarks/covert_witness/te_core/`: their READMEs own the commands, files and checks. This skill is how
to work in a search, and the rules no code enforces yet.

## The question

- **Circuits.** C is a set of gates, each reading one or two values and computing a bit; θ ⊆ in(C) are the registered
  weights. A policy (c, k, n) allows blocks of at most c gates that read at most k bits beyond θ, n blocks a period, and
  the slowdown ρ(C) is the least |𝓘|·c/|C| over (c, k)-partitions 𝓘 of C.
- **Training's annotation**: the new slow weights θ' (values gates compute) and the token blocks 𝒯. From them: Dep, the
  gates depending on θ' (everything downstream of the update); φ = |Dep ∩ ⋃𝒯|/|C|; w, gates per token; s = c/w; τ, data
  bits per token; a, the cut cost; u(x), the computed bits a block reads to do a share x of a token's dependent work, with
  u(1) the per-token update and x0 the largest x with u(x) ≤ k; Δ, the bits describing θ' given θ with a decoder of at
  most c gates; Δ_grad, the gradient and optimizer state passed between blocks.
- **Theorems.** Fit: Δ + sτ ≤ k gives ρ = 1 for the updated model, and training it also needs 2Δ_grad + sτ ≤ k.
  Slowdown, under A1 (cut cost): ρ ≳ φ(1 − x0)·s·a / max_x x(k − u(x)). Security, under A1 and A2 (efficiency floor):
  reaching ℓ* covertly costs at least H(ℓ*)·min ρp, with at most nc available per period.
- **What a search measures**: H(ℓ*), the least compute of any method, and for each attack A its p_A = |C_A|/H and ρ_A,
  so S(c, k; θ, ℓ*) = min_A ρ_A·p_A = W*/H, with W* the least |C|·ρ over attacks reaching ℓ*. The inputs are the start,
  ℓ* and the policy; the red team chooses everything else (method, schedule, update form, placement, encoding). Score
  several c, and the reference policy first: c = 4096 w (one 4,096-token session per block) and k = 10 kB.
- **Units**: |C|, c and w in Boolean gates, each operation's multiply-accumulates converted with its circuit's Definition
  (the gate counts circuit-check reports), never a rule of thumb; k, τ, a, u, Δ and Δ_grad in bits.

## Taking part: a direction

- **Your island.** A direction is a branch `cursor/te-<direction>-NNNN` and an island
  `benchmarks/covert_witness/islands/<direction>/`: its `train.py`, a README and its BOARD.md. The launcher ships the
  pinned commit's fixed side and, from your checkout, only the island as HEAD has it, as a tar: commit before launching,
  and keep everything the program imports inside the island (a file named like a fixed module never replaces it).
- **Launch** with the command in the harness README (CLASS START, a note whose first word is the mechanism, then the
  program's arguments after `--`), under your own checkout's environment (`uv run`): an older checkout's `research` on
  the PATH fails on the pool's name. While the pinned harness has classes, a submission names the class it ran under
  (`class_named`), so an island launched under several takes `--class` among its arguments and passes it on.
- **At most two candidates in flight** per direction. Pack a sweep into one candidate with an island's `--arms` (each arm
  restarts from θ on the same data order, with 1/n of the budget) instead of queueing more.
- **Records.** Each launch is preserved in the store and ingested into te_core's ledger; cite a record by its `art:` id
  and run id. Only records at the pin's version count, and another direction's E relaunched as yours doesn't
  (`duplicate-e`).
- **The board.** Append one line per entry, `HH:MMZ <agent> [TAG] text (record ids)`, to your island's BOARD.md, and push
  your branch. Tags: FINDING, NEGATIVE (a route that didn't work, with its records), TOOL, LOOPHOLE, QUESTION (naming
  whom it is for), PROMOTED, DIGEST, FINAL. At the start of every loop read the digest, BOARD.md at the harness's root on
  the digest branch: its coordinator DIGEST lines are the campaign's rulings, and they override your brief.
- **A loophole is reported, never used.** A way to score better than the work is worth (an uncounted route, held-out
  bytes the training job can reach, a check that passes the wrong thing) is a LOOPHOLE line with the smallest probe that
  shows it, for the toolsmith (te_core) or the integrator (the fixed side). A record that uses one is withdrawn, and an
  audit direction's probes never enter H or W*.
- **FINAL.** When your direction is done, or at the coordinator's call, a FINAL line names at most three records at the
  pin, each with what it shows; the scorer reruns them.

## Scoring a record

- **Valid** is every check of the harness's witness checker passing; an invalid record counts for nothing, whatever its
  loss.
- **Fit** is checked on each record by construction. **Slowdown's ρ** is reported as "target" until Lean proves
  Slowdown. u(x) and a are the best values a witness shows (an encoding or activation code the red team exhibits),
  labelled upper bounds, never measured truth. Δ is Fit's quantity and u(1) Slowdown's: they agree only when every
  token's path reads the whole update, so they are separate columns.
- **Honest.** H(ℓ*) is the least |C| over every valid record, of any method or direction (a covert one at its |C|), that
  reaches ℓ* by a complete schedule; each branch end of cooldowns forked from one trunk is a complete schedule. W*(ℓ*) is
  the least |C|·ρ over the same records, so S = W*/H ≥ 1 by construction: a ratio below 1 is a scoring bug, never a
  covert advantage. A column of the reference directions alone may stand beside it as context.
- **Checkpoints.** A mid-schedule checkpoint stands only where no complete schedule reaches ℓ*, labelled "upper bound
  (checkpoint)".
- **Aim complete schedules at each target.** The honest side, and any direction whose record may set W*, runs short
  complete schedules sized to end at each ℓ* (say 0.5%, 1% and 2% below the start's loss), never one long run read at its
  checkpoints.
- **S is what the records found show**, not a bound: H and W* are each the least over the records found, ρ is
  Slowdown's target under A1, and a method found later moves S either way (a cheaper honest one up, a cheaper attack
  down). Report S with the records that set it.
- A record of a harness that priced the retired theorem (v2.3's ρ, U, D, β and r) is read for its compute (counted in
  multiply-accumulates, to convert to gates), its E and its witnesses, never for its ρ.

## Running a search (the coordinator)

- **One pin per campaign.** Pin the fixed side once it is complete (`benchmarks/covert_witness/pin.py`, which reads
  `pinned.json` from the branch its `BRANCH` names); every record is measured by the pinned commit. A fix to the
  final-score path alone is a new commit of the same version (`same_version`), so no record changes regime; any other
  change is a new version, which leaves every earlier record out. Later pricing reaches records by re-scoring them, never
  by a re-pin. Announce the pin, and the merge every direction needs, in a DIGEST line.
- **Checkpoint every 45 minutes.** Merge the boards with `python3 benchmarks/covert_witness/tools/merge_board.py --push`
  from the lane's worktree (it fetches every search branch and the lane's branch, keeps the digest's own lines, and
  dates each line by its own board, so a night past midnight stays in order). Read every line since the last checkpoint,
  answer each QUESTION, post each ruling as a DIGEST line in your own board on the lane's branch, cancel what exceeds two
  in flight, and write the checkpoint line (`.agents/skills/team/SKILL.md`). The stall rule: a direction with no new
  line and nothing in flight since the last checkpoint is resumed with the digest; one resumed twice without a new record
  is relaunched fresh from its board, or closed with its FINAL.
- **Nodes.** The launcher takes the next node of the pinned commit's `benchmarks/covert_witness/nodes.txt`, or of the
  file `CW_NODES` names (the same `NODE pool=GPU_POOL cpu=CPU_POOL` lines), in turn, and `--node` names one. The turn is
  counted per agent VM under a lock no other VM sees, so directions on different VMs pile onto one node: give each VM
  its own order, or place with `--node`. A node is listed only once `benchmarks/covert_witness/prep.sh` has prepared it.
- **Leases and the stop.** A node's cluster agent admits no lease past its admission end, the node's stop (its
  deadline) less `drain_s` (`infra/cluster/README.md`; the drain is in `infra/cluster/descriptions/nebius.toml`). A
  candidate's lease is 95 minutes (candidate.sh), so the last launch that runs starts 95 minutes before the admission
  end; later ones wait in the line and never start. Before a search, ask infra for the stop it needs and a drain of
  minutes, not hours, and set the cutoffs up front: the last full-budget launch, the last search launch, and the node
  kept for finals.
- **The evictor deletes run directories** on a node short of disk. A checkpoint another run reads (a cooldown branch's
  trunk) is fetched from the store, or kept with `research keep PATH --owner @training --expires … --priority keep
  --preserve` (`.agents/skills/writing-runs/SKILL.md`, norm 4). An agent VM fills too, with fetched runs under
  `~/.research/runs`, and `research` refuses a launch below its free-space floor.
- **When infra stops a node**, its runs die with rc 143, which is no candidate's failure: relaunch them at the pin on
  another node, and rerun `prep.sh` on the stopped one (on a prepared node it only checks) before launching there again.
- **Finals.** After the last search launch the scorer reruns in final mode, on the sealed final set
  (`launch.py --final`, direction scorer), the records that set H or W* at some target and every direction's FINAL
  records, at 3 seeds each, shortest budgets first. A final run is preserved, not ingested; the table gives every point
  its spread, and the search's numbers beside the final set's.

## Smells that are rules here

The smell check (`.agents/skills/smell-check/SKILL.md`) applies to every claim; in a search it takes these forms.

- **No assumed parameter swept as an axis.** Every quantity the theorem reads is measured from the run's circuit or
  witnessed by the red team's own circuit; a sweep over an unwitnessed one (a compressibility r, β at full width) is no
  result.
- **No caps the score doesn't read.** The policy and the budget are the constraints. An update-size tier is a bin of Δ
  for the A2 curve, never a cap a record must fit, and a class is a launcher argument, never a cap.
- **The measurement is the theorem's quantity.** Dep is every gate downstream of the update, activations included, not
  the products that read changed weights: an adapter scores the same merged or not. Rule from the theorem's definitions,
  never from the board's last line.
- **The honest baseline is the best method of any kind**, and is re-derived whenever any record beats it.
- **ρ = 1 means the theorem proves no slowdown there**, never that there is none: say "guaranteed slowdown".
- **Read a record before relaying its label**, a table before forwarding it, and a FINAL before the scorer reruns it.
- **Retired terms stay out** of new lines and tables (the formalism note's "Retired"): covers and a Cover theorem,
  (Γ, α) and a block size that includes width(θ), counting only the products that read updated weights, the old deep
  share and depth, per-MAC β, and the r sweep.
