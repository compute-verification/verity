---
name: friction
description: >
  What to do when a tool, rule, check or step gets in your way in Verity: fix it yourself when the change is clearly yours,
  ask Daniel when you can't tell, record his ruling so nobody asks the same question twice, or leave a short friction note
  when you can't fix it now. Holds the rulings so far on which changes an agent makes without asking (spend, gates in
  `check`, tests, shared tools, other lanes' work) and the `kind: friction` note format with its `severity: incident` flag.
  Use when you worked around something, a step bit you twice, you see a better abstraction, you're unsure whether a change
  is yours to make, you just got a ruling from Daniel, or you're the coordinator reading friction notes.
---

# Friction

When something gets in your way (a wrong command, a clumsy step, a check that fails late, a missing abstraction), take the
first of these that applies.

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

- one stand-alone question, through the channel you already use to reach your coordinator (a lane: a handoff to
  `lanes/coordinator/`; a Project worker: its report). The coordinator answers from the rulings when one applies, and
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

A note is a file, never a message, so it pings nobody. A blocker still goes to your coordinator as a handoff (lane contract
§3a); the note is its record.

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

Newest first. Each is Daniel's.

- 2026-10-01: whoever lands a train runs `research merge --sweep` after each push, which closes the landed PRs, retargets
  their children and deletes landed branches. `research merge --push` pushes the merge and then sweeps, so a lander passes it.
- 2026-09-30: the Lean infrastructure lane changes `tools/lean`, the `lean-proofs` skill, `check`'s Lean steps, shared
  Lean foundations and Lean or Mathlib bumps (in a PR of their own, announced first) without asking. It speeds up another
  lane's proofs when no pinned record changes; a changed record in another lane's package goes through that lane. What a
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
