---
name: writing-goals
description: >
  How a lead writes and keeps its goals in the plan, the one graph Daniel reads on the console's Goals page: what a
  goal is (an outcome with one owner and a date), headline goals versus steps, what a goal waits on (links to goals,
  steps and Daniel's asks, never free text), the status rule the page enforces, and how to write the description for
  Daniel (flexible prose that never restates the fields). Covers `research goals show|set|confirm|done|remove|replace`
  and the API it calls. Use when writing, updating or confirming a goal, adding a dependency, or rewriting your goals
  for a sync.
---

# Writing goals

The plan is one graph of outcomes. Daniel reads it on the console's Goals page; agents write it with `research goals`
(the console's `/api/goals`, with your Cursor identity). Linear holds the work underneath: each issue serves a goal.

## A goal

- **An outcome, not an activity.** Its title says what will be true, in at most eight plain words: "Both demos pass on
  one served H200 run", not "Work on demos".
- **One owner** (`--owner`, your handle), who keeps it true and confirms it at least daily (`research goals confirm ID`).
- **A due date** (`--due 2026-11-05`, the end of that day, or `2026-11-05T10:00-08:00` when the hour matters).
- **A status:** `on-track`, `off-track` or `done`, your forecast of whether it will be true by its date. Done needs
  evidence named in the description: a run, a lock change or a landed commit.
- **Headline or step.** A headline goal is one Daniel would recognize and could check himself: at most two per lead a
  week. Anything smaller is a step (`--step`): the same fields, off the Goals page, shown where something links to it.

## What it waits on

Links only (`--wait goal:ID`, `--wait ask:CARD`, repeated for each; `--no-wait` clears them):

- Link the **narrowest** goal or step that blocks you, **due before you**. Waiting on main moving today is
  `goal:ci/main-moves-again`, not ci's broad landing-speed goal.
- If what blocks you isn't in the plan, add it as a step, owned by whoever delivers it, or ask that owner to.
- A decision of Daniel's is its ask (`ask:<card id>`, from `research msg ask-daniel`).
- A link carries nothing else. The dependency's state lives on it; why you need it goes in your description.
- Your own work isn't a dependency. Slow or risky work is your status, explained in your description.

**The page enforces one rule:** a goal shows as off track, whatever you wrote, while it waits on a goal due after it,
a goal that is off track, or an ask still open past its date. Fix the cause (re-date, re-scope, or chase the
dependency) rather than the status.

## The description

Markdown for Daniel. Flexible prose, but it never restates a field: no due date, status, owner or dependency list,
since the page draws those. It says what the fields can't:

- exactly what will be true, more precisely than the title;
- why it matters, when that isn't obvious;
- how Daniel can check it (the record, lock change or run to look at);
- where it stands, in a sentence, when that adds to the steps.

Style:

- Lead with the claim, in bold. Then short bullets with bold labels where they help: **Why it matters**, **Done**,
  **Left**, **Now**, **To check it**. One idea per bullet.
- Bold the key numbers and dates ("**450** changes", "**Wed 10 AM**").
- Times are Pacific, written like "Mon 10 PM", with no zone and never UTC.
- No PR numbers, ticket ids, branch names, run ids, agent ids or file paths. Name teams by handle (ci, proofs).
- Define a term the first time you use it. Give numbers with units and what they compare to.
- Link another goal with `[text](#goal-id)` and an ask with `[text](ask:<card id>)`; both render as chips.

## Commands

~~~bash
research goals show [--steps]                    # the plan by due date, with what each goal waits on
research goals set proofs/zero-knowledge --title "Zero knowledge proved, end to end" --owner proofs \
  --due 2026-11-13 --status on-track --tag proofs --wait goal:proofs/zk-server-lean --description @zk.md
research goals set vllm/partition-on-main --step --owner vllm --title "Partition into proof blocks on main" \
  --due 2026-11-02T22:00-08:00 --status off-track --tag demos --wait goal:ci/main-moves-again --description @p.md
research goals confirm ID                        # still true as written
research goals done ID
research goals remove ID
~~~

A wrong field is refused with a message naming it.
