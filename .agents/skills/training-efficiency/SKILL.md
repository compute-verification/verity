---
name: training-efficiency
description: >
  How to run, or search in, a training-efficiency search in Verity: the least compute that lowers a registered start's
  loss to a target ℓ*, honestly and under a policy (c, k), scored in training formalism v2's terms (Fit, Slowdown,
  Security; |C| in Boolean gates; ρ, p = |C|/H, S = W*/H). Explains the covert_witness harness (launch.py, the pin, an
  island shipped alone, --island, --arms), the boards (an island's BOARD.md, merge_board.py's digest, DIGEST, FINAL and
  LOOPHOLE lines), scoring (honest = least compute of any method at any checkpoint, W* over the same checkpoints), and the coordinator's side: one pin per campaign, 45-minute checkpoints, nodes and leases (CW_NODES, --node,
  drain_s, the evictor, research keep) and final reruns on the sealed set. Use when launching, joining or coordinating a
  search direction, scoring or reporting an honest or covert record, merging the board, or planning a campaign's pin,
  nodes and finals.
---

# Training-efficiency search

A search measures, from registered weights θ (a start and its loss ℓ0), the least compute that reaches a target loss ℓ*,
honestly and under a policy's slowdown. Its question and quantities are Daniel's training formalism v2, which
`GLOSSARY.md` ("Training") states in the repository and which wins over the summary below; the note is
`~/.research/notes/lanes/training/20261102T2215Z-report-training-formalism-v2.md`, with the experiments and plots it
feeds in `20261102T1715Z-draft-training-efficiency-experiments.md` beside it. The harness is `benchmarks/covert_witness/`
and its counter `benchmarks/covert_witness/te_core/`: their READMEs own the commands, files and checks. This skill is how
to work in a search, and the rules no code enforces yet.

## The question

- **Circuits.** C is a set of gates, each reading one or two values and computing a bit; θ ⊆ in(C) are the registered
  weights. A policy (c, k, n) allows blocks of at most c gates that read at most k bits beyond θ, n blocks a period, and
  the slowdown ρ(C) is the least |𝓘|·c/|C| over (c, k)-partitions 𝓘 of C.
- **Training's annotation**: the new slow weights θ' (values gates compute) and the token blocks 𝒯. From them: A, the
  affected gates, those depending on θ' (everything downstream of the update), and R ⊆ A, the reachable ones a block can
  do while computing the token from its inputs; σ, the token blocks' share of |C|; w, gates per token; s = c/w; τ, data
  bits per token; x0 = |T ∩ R|/|T ∩ A|; s*, the most tokens' worth of unreachable work a block does; a, the cut cost;
  u(x), the update bits a block reads to do a share x of a token's affected work (u(1) the per-token update), and
  u_in(x), the same for the first share x in dependency order while computing the token from its inputs; Δ, the bits
  describing θ' given θ with a decoder of d ≤ c gates; Δ_grad, the gradient and optimizer state passed between blocks;
  t, a frozen tail's share of the affected work. Update bits are values of θ' or an encoding of it, never cut values.
- **Theorems.** Fit: running θ' with Δ + sτ ≤ k gives ρ ≤ c/(c − d) (honest serving exactly 1), and training needs
  three kinds of block: compute Δ + sτ ≤ k, reduction 2Δ_grad ≤ k, update application Δ + Δ_grad ≤ k. Slowdown is the
  counting lemma at A ∖ R, in token form ρ ≥ σ(1 − x0)·s/s* (the largest per-token affected count where token blocks
  vary), with the bounds s* ≤ max_x x(k − u(x))/a (A1, cut cost) and x0 ≤ max{x : u_in(x) ≤ k} (A1′, update cost from
  the inputs). Security, under A1, A1′ and A2 (efficiency floor): reaching ℓ* covertly costs at least H(ℓ*)·min ρp, with
  at most nc available per period.
- **What a search measures**: H(ℓ*), the least compute of any method, and for each attack A its p_A = |C_A|/H and ρ_A,
  so S(c, k; θ, ℓ*) = min_A ρ_A·p_A = W*/H, with W* the least |C|·ρ over attacks reaching ℓ*. The inputs are the start,
  ℓ* and the policy; the red team chooses everything else (method, schedule, update form, placement, encoding). Score
  several c (`python -m te_core.policy`: the (c, k) grid in decades, from one token's inference block to the start's
  server block), and the reference policy first: c = 4096 w (one 4,096-token session per block) and k = 10 kB.
- **Units**: |C|, c and w in Boolean gates, each operation's multiply-accumulates converted with its circuit's Definition
  (the gate counts circuit-check reports, `python -m te_core.gates`), never a rule of thumb, and d too; k, τ, a, u, u_in,
  Δ and Δ_grad in bits.

## Taking part: a direction

- **Your island.** A direction is a branch `cursor/te-<direction>-NNNN` and an island
  `benchmarks/covert_witness/islands/<direction>/`: its `train.py`, a README and its BOARD.md. The launcher ships the
  pinned commit's fixed side and, from your checkout, only the island as HEAD has it, as a tar: commit before launching,
  and keep everything the program imports inside the island (a file named like a fixed module never replaces it).
- **Launch** with the command in the harness README, `launch.py START --note TEXT -- TRAIN_ARGS` (`--island NAME` for an
  island other than your direction's; the note's first word is the mechanism), from a checkout at the pinned harness
  version, since the launcher refuses a pin of another, and under that checkout's environment (`uv run`): an older
  checkout's `research` on the PATH fails on the pool's name. v3 has no classes and caps nothing; v2.3 took a CLASS
  before START and checked the class a submission names (`class_named`), so an island launched under several passed
  `--class` on.
- **At most two candidates in flight** per direction. Pack a sweep into one candidate with an island's `--arms` (each arm
  restarts from θ on the same data order, with 1/n of the budget) instead of queueing more.
- **Records.** Each launch is preserved in the store and ingested into te_core's ledger; cite a record by its `art:` id
  and run id. Only records at the pin's version count, and another direction's E relaunched as yours doesn't
  (`duplicate-e`).
- **The board.** Append one line per entry, `HH:MMZ <agent> [TAG] text (record ids)`, to your island's BOARD.md, and push
  your branch. Tags: FINDING, NEGATIVE (a route that didn't work, with its records), TOOL, LOOPHOLE, QUESTION (naming
  whom it is for), PROMOTED, DIGEST, FINAL. At the start of every loop read the digest, BOARD.md at the harness's root on
  the digest branch (merge_board.py's `--digest`): its coordinator DIGEST lines are the campaign's rulings, and they
  override your brief.
- **A loophole is reported, never used.** A way to score better than the work is worth (an uncounted route, held-out
  bytes the training job can reach, a check that passes the wrong thing) is a LOOPHOLE line with the smallest probe that
  shows it, for the toolsmith (te_core) or the integrator (the fixed side). A record that uses one is withdrawn, and an
  audit direction's probes never enter H or W*.
- **FINAL.** When your direction is done, or at the coordinator's call, a FINAL line names at most three records at the
  pin, each with what it shows; the scorer reruns them.

## Scoring a record

- **Valid** is every check of the harness's witness checker passing; an invalid record counts for nothing, whatever its
  loss.
- **Fit** is checked on each record by construction, per kind of block. **Slowdown's ρ** is reported as "target" until
  Lean proves Slowdown. The claim takes the assumptions' upper bounds, never best-known values: x0 and s* are maxima over
  every block, so a score reads them from u_in and u (A1′, A1), which are labelled assumptions, and a and Δ are the best
  values a witness shows (an encoding or activation code the red team exhibits), labelled upper bounds. The best-known
  x0 and s*, from blocks someone exhibits, are the attack side (`x0_known`, `s_star_known`): they say how tight the bound
  is, never what it is. Δ is Fit's quantity and u(1) Slowdown's: they agree only when every token's path reads the whole
  update, so they are separate columns.
- **A frozen tail is always charged.** With frozen layers above the changed ones (a share t of the
  affected work, the changed update > k), u_in jumps to the changed update at once, so x0 ≈ 0, s* ≤ t·k/a and
  ρ ≳ σ·s·a/(t·k): the tail's work is never exempt.
- **Honest.** H(ℓ*) is the least |C| over every checkpoint of every valid record, of any method or direction (a covert
  one at its |C|), that reaches ℓ*: a checkpoint is a model someone reached, and the sealed final reruns guard against
  picking noise. W*(ℓ*) is the least |C|·ρ over the same checkpoints, so S = W*/H ≥ 1 by construction: a ratio below 1
  is a scoring bug, never a covert advantage. A column of the reference directions alone may stand beside it as
  context. The table is `python -m te_core.frontier --formalism` (W*, H and S per target and (c, k) cell, with W*'s
  bounds and attack side), the five panels `python -m te_core.plots --out DIR` (P1-P4 and A2, off the console), and
  both over v2.3's records `python -m te_core.rescore --out DIR`; the console's training panels,
  `training/algorithm-policy` (p = C/H against ρ per run, start and gain target, a gain being effective compute over
  the start's own) and `training/frontier-fit` (each data's scaling law, through which a loss is effective compute),
  are `python -m te_core.algorithm_policy --out DIR --publish` (all from benchmarks/covert_witness).
- **Aim short complete schedules at each target.** A schedule sized to end at ℓ* (say 0.5%, 1% and 2% below the start's
  loss) usually reaches it for about half the compute of a checkpoint inside one long run, so both sides run them.
- **A reported W* is the best attack known**, an upper bound (`.agents/skills/friction/SKILL.md`), and H the best
  method known, so S is no bound: a record found later moves it either way (one that lowers H alone raises S, one that
  lowers W* alone lowers it). Report S with the records that set it.
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
- **The measurement is the theorem's quantity.** A is every gate downstream of the update, activations included, not
  the products that read changed weights: an adapter scores the same merged or not. Rule from the theorem's definitions,
  never from the board's last line.
- **The honest baseline is the best method of any kind**, and is re-derived whenever any record beats it.
- **ρ = 1 means no slowdown is proved there**, never that there is none: call ρ the guaranteed slowdown, so ρ = 1 reads
  as none guaranteed.
- **Retired terms stay out** of board lines and tables too: the Glossary's older names (`GLOSSARY.md`, "Older names",
  which the retired-terms test enforces in docs only), and "Cover", which the formalism's correction replaced by
  Slowdown.
- **Check before it goes out.** Read a record before relaying its label, and a FINAL before the scorer reruns it. A table
  goes out with S ≥ 1 in every cell, each H and W* naming the record that sets it, |C| in gates, each ρ labelled
  "target", u(x) and u_in(x) labelled assumptions, a and Δ upper bounds, the attack side beside the bound, and each
  checkpoint figure "upper bound (checkpoint)".
