---
name: writing-goals
description: >
  How a lead writes and keeps its goals in the plan, the one graph Daniel reads on the console's Goals page: what a
  goal is (an outcome with one owner and a date), tags from the fixed list of projects, tiers (derived from what waits
  on what, overridable), what a goal waits on (links to goals and Daniel's asks, never free text), why an off-track goal
  is off track, and how to write the description so Daniel can skim it (one plain opening sentence, then the shape that
  fits: landing, research or proof; bold only for key expressions). Covers `research goals
  show|set|confirm|done|remove|replace` and the API it calls. Use when writing, updating or confirming a goal, adding a
  dependency, or rewriting your goals for a sync.
---

# Writing goals

The plan is one graph of outcomes. Daniel reads it on the console's Goals page; agents write it with `research goals`
(the console's `/api/goals`, with your Cursor identity). Linear holds the work underneath: each issue serves a goal.

## A goal

- **An outcome, not an activity.** Its title says what will be true, in at most eight plain words: "Remote demo passes
  on one served H200 run", not "Work on demos".
- **One owner** (`--owner`, your handle), who keeps it true and confirms it at least daily (`research goals confirm ID`).
- **A due date** (`--due 2026-11-05`, the end of that day, or `2026-11-05T10:00-08:00` when the hour matters).
- **A status:** `on-track`, `off-track` or `done`, your forecast of whether it will be true by its date.
  - Off track says why: link what it waits on, or say what's wrong in `--blockers`. The site refuses an off-track goal
    with neither.
  - Done needs evidence named in the description: a run, a lock change or a landed commit.
- **Tags** (`--tag`, repeated) come from one fixed list of projects, in Daniel's words: `remote-demo`, `onsite-demo`,
  `proofs`, `lean`, `vllm`, `compute-accounting`, `network-accounting`, `memory-accounting`, `accounting` (only for
  work across all three), `training`, `ci`, `infra`, `security`, `cleanup`. `proofs` is what the theorems say; `lean`
  is Lean's code and build (the module system, the Lean check, toolchain bumps). The site refuses any other tag.
  Nothing from your own bookkeeping goes in a tag.
- **Its tier is derived.** A goal nothing waits on is tier 1, an outcome; a subgoal is one tier deeper than the
  shallowest goal that needs it. Set `--tier N` only when that's wrong, for example an outcome that something else
  also happens to wait on; `--no-tier` goes back to the derived one.

## What it waits on

Links only (`--wait goal:ID`, `--wait ask:CARD`, repeated for each; `--no-wait` clears them):

- Link the **narrowest** goal that blocks you, **due before you**. Waiting on main moving today is
  `goal:ci/main-moves-again`, not ci's broad landing-speed goal.
- If what blocks you isn't in the plan, add it as a subgoal, owned by whoever delivers it, or ask that owner to.
- A decision of Daniel's is its ask (`ask:<card id>`, from `research msg ask-daniel`).
- A link carries nothing else. The dependency's state lives on it; why you need it goes in your description.
- Your own work isn't a dependency. Slow or risky work is your status, explained in your description.

A goal's peek draws what it waits on and what needs it in one timeline around its own due date, so a link due after
it shows red. Fix the cause (re-date, re-scope, or chase the dependency) rather than the status.

## The description

Markdown for Daniel to skim. It never restates a field (due date, status, owner, tags or links): the page draws those.

**Open with one plain sentence** saying precisely what will be true, sharper than the title. Not bold.

Then the shape that fits the goal:

- **Landing** (code reaching main, a migration, a deploy): the path as a numbered list, in order, with times and
  owners, finished steps first. Sub-bullets hold the detail.
- **Research** (a measurement or a result): what's measured so far, with the key numbers and what they're on; the plan
  as a numbered list; then one line on what the numbers rest on (assumptions, unproved theorems, seeds).
- **Proof** (a theorem in Lean): the statement in plain words, then what's proved and what's left, with other teams'
  pieces linked as goals.

Close with the date only when it says something the due date doesn't: "Due **Fri**. **Wed** would need both pieces
within two days."

Rules:

- Bold only key expressions: numbers, dates and decisions waiting on Daniel. Never a sentence or a clause.
- No fixed labels. A plain lead-in ("So far, on the pilot:", "Plan:", "Proved:", "Left:") only where it helps.
- No "why it matters" when the title already says it. No "to check it" unless the evidence isn't obvious, and then one
  line naming it.
- Sub-bullets are welcome: indent them three spaces under their item.
- Times are Pacific, written like "Mon 10 PM", with no zone and never UTC.
- No PR numbers, ticket ids, branch names, run ids, agent ids or file paths. Name teams by handle (ci, proofs).
- Define a term the first time you use it. Give numbers with units and what they compare to.
- Link another goal with `[text](#goal-id)` and an ask with `[text](ask:<card id>)`; both render as chips.

A research goal, as Daniel approved it:

~~~markdown
For a ladder of fully trained models, the cheapest covert route to each lower loss, measured against the cheapest
honest one, under the training formalism (v2).

So far, on the FineWeb-Edu pilot:
- slowdown at the default policy is about **200–430** for most routes;
- as low as **1** for adapters on the embedding. That rests on the theorem's attention loose end: **your call**,
  sent 5:28 PM.

Plan:
1. Tonight the 20M, 60M and 200M starts retrain on OLMo 2's pretraining mix.
2. Every frontier point is rerun on the sealed shard at **three seeds**.
3. Both training plots from those runs by **Wed 8 AM**.

Every slowdown rests on assumptions A1 and A1′ until the Slowdown theorem is proved in Lean.
~~~

A proof goal:

~~~markdown
What the zero-knowledge server releases reveals at most about **one bit** a session, plus the counted bits of where a
session stopped, when the worker runs the honest prover.

Proved:
- the server's release rounds and the inner proof's rounds, approved for landing;
- this week, the inner phase checked in Lean instead of Python.

Left:
- the honest worker's requests, in Lean, shown to pass the server's checks:
  [memory accounting](#accounting/honest-requests-pass-firewall), with
  [proofs making the checks provable](#proofs/firewall-checks-provable);
- the stop leak, about **11 bits** a session, proved from the session's rules:
  [network accounting](#accounting/stop-leak-proved).

Due **Fri**. **Wed** would need both pieces within two days.
~~~

## Commands

~~~bash
research goals show                              # every goal by due date, with what each waits on
research goals set proofs/zero-knowledge --title "Zero knowledge proved for an honest worker" --owner proofs \
  --due 2026-11-06T18:00-08:00 --status on-track --tag proofs --wait goal:proofs/firewall-checks-provable \
  --description @zk.md
research goals set ci/fine-grained-check --status off-track --blockers @why.md
research goals set vllm/partition-on-main --tier 2   # only when the derived tier is wrong
research goals confirm ID                        # still true as written
research goals done ID
research goals remove ID
~~~

A wrong field is refused with a message naming it.
