---
name: friction
description: >
  What to do when a tool, rule, check or step gets in your way in Verity: fix it yourself when the change is clearly yours,
  ask Daniel when you can't tell, record his ruling so nobody asks the same question twice, or leave a short friction note
  when you can't fix it now. Holds the rulings so far on which changes an agent makes without asking (spend, gates in
  `check`, tests, shared tools, other lanes' work) and the `kind: friction` note format with its `severity: incident` flag.
  Use when you worked around something, a step bit you twice, you see a better abstraction, you're unsure whether a change
  is yours to make, you just got a ruling from Daniel, you're the coordinator reading friction notes, or a command fails for
  missing credentials.
---

# Friction

When something gets in your way (a wrong command, a clumsy step, a check that fails late, a missing abstraction), take the
first of these that applies.

If `research data`, `research queue` or node ssh fail for missing credentials, rerun from a fresh tmux login shell before
anything else: on a cloud VM a long-lived tool shell can lose the injected secrets (store, node SSH key, Slack token) that a
fresh login shell still has. `research` itself takes them from the tmux server first; a plain `ssh` or script doesn't.
If git push fails because the VM's GitHub token lapsed, hand the branch over through the evidence store (git bundle create B
origin/BASE..BRANCH, then research data put --kind evidence/v1 --file B --preserve), and send the art id, the branch and its
head sha: no agent can read another agent's Cursor store.

## 1. Fix it, when it's clearly yours

Most friction is. Fix it at the source, in the PR you're working on or in a small one of its own, instead of working around
it:

- anything in the files your task owns: your package, its tests, your lane's scripts;
- a bug in a shared tool that stops it doing what its own docs or help say;
- a wrong or missing command, flag or gotcha in a skill or a `kb/` doc;
- a failure message that didn't say how to fix the failure;
- a nondeterministic or needlessly slow test: make it deterministic or smaller, or delete it.

Say what you fixed in the PR description. It needs no note.

## 2. Ask, when you can't tell

A change is near the line when it changes what other agents must do or what a gate accepts, touches another lane's
in-flight files, spends money, deletes others' work, or changes agreed semantics or acceptance criteria. Read the rulings
below first, and follow one if it covers the case. Otherwise ask:

- one stand-alone question to your coordinator (a handle holder: on Slack, `research slack ask`; a worker: its report or
  reply to the coordinator that launched it). The coordinator answers from the rulings when one applies, and
  otherwise puts the question to Daniel as written;
- the case, the change, and your recommendation, in a few lines;
- meanwhile, prepare the change on your branch, and land it only after the ruling.

For a new rule, gate or required review, price it first with `.agents/skills/process-design/SKILL.md`, and put the
proposal in the question.

## 3. Record the ruling

A ruling is a labelled example near the line. Record it where the next agent at the same spot will read it, in the PR that
makes the change or in a one-line PR of its own:

- a norm every agent needs: a line in `AGENTS.md`;
- a norm for one kind of work: a line in that work's skill (testing, Lean, ...);
- otherwise: a line under "Rulings so far" below, with its date.

Write it for the class of case, not the one instance, in one line. Delete it once code enforces it.

## 4. Note it, when you can't fix it now

Write a friction note when the fix isn't yours or can't happen now, and the friction cost something: a failed run or check,
about half an hour, or the second time you hit it. Not for what you fixed, and not for one-off slips. Search the notes first:
if one covers it, write a two-line note naming it under `recurs:`, because repeats are what get fixed.

A note is a file, never a message, so it pings nobody. A blocker still goes to your coordinator as a message (Slack, or
your report to it); the note is its record.

~~~markdown
---
id: <lane>/<YYYYMMDDTHHMMZ>-friction-<slug>
lane: <lane>
kind: friction
status: open
severity: incident     # only when a result, money, `main`, a gate or a deadline was lost; otherwise leave the line out
recurs: note:<id>      # optional: the earlier note about the same thing
---

# <one line: what got in the way>

What happened, with ids (run, PR, commit), and what it cost. What you did instead. The better abstraction, if you see one.
~~~

Save it as `~/.research/notes/lanes/<lane>/<YYYYMMDDTHHMMZ>-friction-<slug>.md`. A Project worker without the notes clone
writes it under the Project store's `internal/lanes/<lane>/`, which the mirror forwards. The notes are public: nothing
private (lane contract §5b).

**The coordinator's pass.** The root coordinator reads new friction notes once a day in its existing loop
(`research notes mcp`, `search_notes` with `kind: friction` and `since`). It turns a repeat into a fix assigned to someone, or
into a question for Daniel.

## Rulings so far

Newest first. Each is Daniel's unless it names the coordinator.

- 2026-10-05 (1:27 PM PDT): a coordinator never loses contact with a worker it launched. A running cloud agent can't take a
  `Task` resume, and the Lean move lead went 85 minutes without a word. So every brief names the worker's lane, so
  `research notes status` flags it STALE, and the worker checkpoints at least every 20 minutes (contract §3). At each
  checkpoint it reads its follow-up queue (`cursor-cloud` `get-message-queue`). When something is queued, it ends its
  turn soon, leaving long runs going on their pods, so the message is delivered. Only the agent that launched a worker
  owns it: its coordinator's `SendToAgent` refuses it ("not a worker of this coordinator"). Until there's a channel, the
  one way to reach a running worker the launcher can't queue to is a human's follow-up box on the worker's dashboard
  page. So as soon as a worker goes STALE, the launcher asks Daniel, with the page's link and the message to paste.
  It starts a parallel lane only if that gets no answer.
- 2026-10-04 (2:21 PM PDT): the repository migration finishes in short serial time. A lead launches workers for every
  piece that doesn't wait on another (Lean ones in separate checkouts), and spends compute wherever it shortens the
  critical path: reruns, check capacity, extra CPU pods. Pods still bound themselves under a `budgets.toml` line, and GPU
  work still names its question.
- 2026-10-04 (2:21 PM PDT): during the migration, ci resolves a conflict that is only a rename (a moved file that another
  PR edited) in the train tip's merge commit, instead of sending the PR back for a restack, and names each resolution in
  the handoff. A conflict that needs judgment still goes to the PR's owner, and `check` still runs on the merged tree.
- 2026-10-04 (8:42 AM PDT): the proof system proves batches of arbitrary Boolean circuits, so soundness is **one
  theorem over everything the compiled Lean verifier accepts**, with no hypothesis on the circuit file or the statement's
  form (no `scopeOk`, no `pub.tables = none`). The verifier fails closed: it refuses every form the proof doesn't cover,
  and rejecting an unproved form is a completeness limit, never a soundness gap. A circuit's inputs and outputs use one
  canonical format, proved once in general (any row length, any input groups, any output placement), and frontends
  normalize to it. Adding a workload never needs a verifier or proof change. The verifier gains no new form ahead of its
  proof, and progress is never measured by which workloads' files fall inside a proof's scope. Pushback is welcome:
  send it to Daniel through @proofs.
- 2026-10-03: a test tier too slow for `check` (the vLLM reference regression's manifest rebuilds, about an hour) runs as a
  detached job after each train lands on `main` (`tools/check/post_train.py`, started by `research merge --push`) and labels
  the landing's check attempt `note=regression pass` or `fail:<rows>`, which circuits sweeps daily; `check` blocks on the
  cheap tier only (its T0 checks, about four minutes), and under `VERITY_STORE_REQUIRED=1` a fixture or input that doesn't
  resolve fails rather than skips.
- 2026-10-03 (the coordinator's): data you don't own on a node is deleted only with `research retention gc` or `research
  retention rm --approved-by @OWNER`, which refuses unless every retention record on the path names that owner; never a bare
  `rm`, and never on another lane's word about who owns it.
- 2026-10-03 (card `d9eb2f28`): PoUW may import sampled proofs, in Python (`ALLOWED` in
  `verity/tests/test_boundaries.py`) as in Lean. Any other protocol → protocol import is still Daniel's call, and the
  edges stay acyclic. The layout that follows (lean with compute-accounting): a protocol guarantee that needs a proof
  system (PoUW's `TileProofSoundAll`) is stated over sampled proofs' statements of what a backend proves (its soundness on
  registered inputs, and the honest prover's membership in the backend's class), which the backend's own package proves
  for itself. No package requires both a protocol and a backend, there is no `bridges/` layer, and a citation names both
  guarantees.
- 2026-10-02 (10:27 PM PDT): a statement review or red-team grant on a PR carries across a restack or a merge of `main`
  when the restacked head's `lean-audit.json` change against its base is byte-identical to the one that was granted.
  Any other change to `lean-audit.json` needs a new grant.
- 2026-10-02 (10:27 PM PDT): `hash-derived-key` (`Hm96Hiding` at the pinned key) stays a named assumption. Hiding is
  proved at a fresh uniform key, and nobody works on proving that the pinned key behaves like one.
- 2026-10-02 (10:27 PM PDT): the lander starts every merge check (`check --record`, `research merge`). The PR captain
  (`ci`) trial-merges, preflights and hands over train tips, but never starts a merge check itself.
- 2026-10-02 (7:23 AM PDT): an overhead reported to Daniel is the `--zk` prover's only (no M0 number and no `--zk`/M0
  ratio), normalized to the chip's peak: prover seconds over the time the computation's FLOPs take at the census's dense
  datasheet peak for its dtype, never over a measured runtime such as a vLLM run's, written to one significant digit
  (6e7, 1e8).
- 2026-10-01 (4:31 PM PDT): a table that tabulates a fixed function is a value even with no program parameter
  (`gelu_tanh_bf16`, `tanh_rn`). The only tables in an operation's own definition are the hardware's measured ones (the
  MUFU tables, `tanh_mufu` included), which stay gates with the floating-point formats' constants. The hidden read's cost
  is accepted for conceptual clarity unless it proves a bottleneck.
- 2026-10-01 (3:32 PM PDT): no protocol deals with public values, so none gets an exemption; PoUW's tile check moves into
  a proof. During development a proof may be replaced by replay, or verification skipped, for research velocity: that's
  a shortcut, never the protocol. Shapes such as batch size and sequence length may stay public for now: the first
  protocol decouples values from circuits, and private-circuit protocols come later.
- 2026-10-01 (2:43 PM PDT): a decision Daniel must make goes where Slack notifies him, never only in a note or a lead's
  thread. Use one question per post, with its options, your recommendation, and its deadline or default:
  `research msg ask-daniel` (the card mentions him), or, if the card fails, `research msg send --to @daniel`.
- 2026-10-01 (1:14 PM PDT; this wording 3:06 PM): the circuit holds only gates, and values are always hidden, the core
  guarantee of even the most basic protocol. An operation's own definition is gates: the floating-point formats'
  constants, SHA-512's, the measured GPU tables for exp2, rcp and tanh. Everything a program or a request brings is a value:
  the weights, a program's constants (eps, the attention scale, a softcap) and tables (`gelu_tanh_bf16`, `tanh_rn`), a
  request's parameters (temperature, top-p, Philox positions), every intermediate and every output. A value is committed
  under a hiding leaf and touched only inside a zero-knowledge proof, never framed as part of the program the verifier
  already has. The hill-climb optimizes the hiding protocol (`--zk`), not the non-ZK circuit prover.
- 2026-10-01 (12:37 PM PDT): lanes work around the clock, so timelines are planned in hours: a pilot, trial or rollout gets
  hours (about eight for a process pilot), not the days or weeks a human team would give it.
- 2026-10-01 (10:56 AM PDT): only leads get handles (a name in `research msg`), never individual workers; a worker is
  reached through its lead. A new lead is a line in `.agents/skills/using-slack/registry.json`, and `names open`
  refuses any other name.
- 2026-10-01 (9:50 AM PDT): lane notes are records (reports, checkpoints, findings, drafts, friction), never a channel.
  A handoff, an order or an ask goes on Slack (`research msg`), or to your own Cursor workers as a follow-up; a note
  written as a message reaches nobody.
- 2026-10-01 (9:12 AM PDT): agents message each other directly, without a coordinator relaying, and adding a handle
  needs no Slack admin. Comms owns the routing (`@<name>` at the start of a post wakes that lead's agent).

- 2026-10-01 (12:05 AM PDT): a protocol's README is an outline, signposts and its core API, and Lean is the source of
  truth for a mature protocol. A protocol keeps a `PROTOCOL.md` only while it is experimental (Python is the truth until
  the Lean catches up), and the PR whose Lean covers it deletes the file.
- 2026-10-01: whoever lands a train runs `research merge --sweep` after each push, which closes the landed PRs, retargets
  their children and deletes landed branches. `research merge --push` pushes the merge and then sweeps, so a lander passes it.
- 2026-09-30 (2:52 PM PDT; at all hours since the coordinator's 06:33Z Oct 4): GPU work runs only as items its lane owner
  said yes to, each naming its research question (`--question`, or a `question` label on its run). There is no filler:
  idle GPUs beat padded ones, a steward reports idle GPUs instead of offering them as backfill, and approved backfill runs
  on preemptible leases.
- 2026-09-30: the Lean infrastructure lane changes `tools/lean`, the `lean-proofs` skill, `check`'s Lean steps, shared
  Lean foundations and Lean or Mathlib bumps (in a PR of their own, announced first) without asking. It speeds up another
  lane's proofs when no guarantee's record changes; a changed record in another lane's package goes through that lane. What a
  headline theorem may rest on, what "proved" means, retiring a theorem a table cites, spend past a budget line and any
  new human gate go to Daniel.
- 2026-09-30: a lane fixes a bug it hits in `tools/research` in a small PR of its own, and the research coordinator only
  reviews and merges it; when an open PR already changes the same function, the fix goes to that PR's owner.
- 2026-09-30: a run's author declares what custody persists with `--declared-output`; undeclared files are hash-listed in
  `trimmed.json`, not uploaded, except that a transitional default uploads up to 256 MiB of them until lanes declare their
  outputs. How to declare them is the `writing-runs` skill (`.agents/skills/writing-runs/`, in progress).
- 2026-09-29: merge velocity outranks CI spend: a larger CI pool (about $120 a day standing, more on backlog days) beats
  trains waiting on pods.
- 2026-09-29: RunPod tops itself up, so a low balance is not a blocker and needs no request to Daniel; spend caps and guards
  are the controls.
- 2026-09-29: cache test results aggressively, per test where the key allows: velocity outranks gate purity there, with the
  nightly uncached check of `main` as the backstop.
- 2026-09-29: after a train lands, the coordinator closes each stacked PR whose code is on `main` but that GitHub didn't
  mark merged, with a "Merged into main at {sha} (train {name})" comment, without asking.
- 2026-09-28: a timing-dependent test is made deterministic or deleted.
- 2026-09-28: changes that speed up checks or merges land ahead of other work.

## Before you finish

- Each fix you made is in a PR description.
- Each question you asked has its ruling recorded, or is still open in your checkpoint or report.
- Each friction note is a few lines, names its ids, and carries `severity: incident` only when something was lost.
