---
name: writing-goals
description: >
  How a lead writes and keeps its goals in the plan, the one graph Daniel reads on the console's Goals page: what a
  goal is (something that finishes, with one owner and a date), how to title it as a headline Daniel reads cold (three
  to six plain words, no number, none of our own labels), tags from the fixed list of areas, goals and their subgoals
  (a goal under another is its subgoal), what a goal waits on (links to goals and Daniel's asks, never free text), why
  an off-track goal is off track, and how to write the description so Daniel can skim it (one plain opening sentence,
  then the shape that fits: landing, research or proof; measured numbers in LaTeX, literals as code, leads as
  @handle; a lead-in line for every list and no word he has said he doesn't know, refused by the site). Covers
  `research goals show|set|confirm|done|remove|replace` and the API it calls. Use when writing, titling, updating or
  confirming a goal, adding a dependency, or rewriting your goals for a sync.
---

# Writing goals

The plan is one graph of outcomes. Daniel reads it on the console's Goals page; agents write it with `research goals`
(the console's `/api/goals`, with your Cursor identity). Linear holds the work underneath: each issue serves a goal.

## Keeping it current

The page is how Daniel sees what is being done, so it moves when the work does:

- **When a goal's state changes**, rewrite where it stands, and its status the moment it slips; don't wait for a sync.
- **Work serves a goal on the page:** its issue carries the goal's label (the `goal` label group, one label per id),
  and its PR names that issue and the goal: `research queue open --issue KEY` writes a `Goal <id>` line for each live
  goal KEY's labels name, and `open` and `research queue ready` warn, never refusing, of a body naming no live goal so
  and not saying `Issue: none: REASON`. Work that serves none starts by writing its goal, or asking top.
- **One outcome, one goal.** Before adding one, look for it under another owner's goal, and make yours its subgoal.

Once a day top rings the owner of a goal past its due, for a new date or its evidence.

## A goal

- **Something that finishes, never an ongoing activity** ("Work on demos"). "X measured" isn't a goal until it names
  what's delivered: "Second swarm's training results out".
- **A title is a headline.** Daniel reads the page by its titles, and a title he has to open to understand is
  shorthand. Read yours cold before you save it; top reads every one.
  1. Short: usually three to six words, never more than about ten. It names the outcome at the level Daniel would
     say it. What the goal tests, counts or accepts goes in the description, which he clicks through to.
  2. No numbers on goals. A goal is named by its title, never "goal 5" or "the eleven", in titles or descriptions.
  3. Plain words and names Daniel uses: demo, benchmark, metrics, proof, exfiltration capacity, GLM-5.3, SPOT, Lean.
     Not our own labels: slice, layout, overhaul, re-cut, the term, the tight figure, the serve.
  4. No leading "The" or "A".
  5. A subgoal doesn't repeat what the goal above it already says, such as the model or the demo.
  6. No ticket ids, and no lead's handle as the subject ("ci's …").

  His examples, each a title as it was and as it should read:

  | Was | Headline |
  | --- | --- |
  | 3 Repository reorganized: one layout, clean slices, public | Repository reorganized and published |
  | 11 Onsite demo: all four protocols accept | Onsite demo done |
  | The check reads what the serve registers | Inference-only demo done |
  | The eleven's numbers live on the console | Console shows core system metrics |
  | The certifier's term is zero without a failure | Network certification adds no exfiltration capacity unless a connection fails |
  | ci's slice reads clean | CI code cleaned up for review |
  | A candidate's check launch registers in three minutes | Check launches within three minutes |
  | GLM-5.3's long DSA rows in provable units | 128k-context attention fits in proof blocks |
  | Remote theorem rests on no open proof | Remote demo's soundness fully proved |
  | The Commit's share of the served step, measured | Commitment overhead measured |
- **One owner** (`--owner`): the lead of the area the goal mostly builds, who keeps it true. Other areas that do
  parts of it are @handle in its plan; a check the remote demo runs is the demo lead's goal even when proofs writes
  it in Lean.
- **A due date** (`--due YYYY-MM-DD`, the end of that day, or `YYYY-MM-DDTHH:MM-08:00` when the hour matters).
- **A priority**, 1 to 3: anyone may set one (`research goals priority ID 1|2|3|clear`), and the page shows
  who set it and when. A goal nobody set takes the highest priority of the goals waiting on it.
- **Today's work, as a subgoal.** An open goal due within a week waits on a subgoal due within a day: write it
  (`research goals set SUB ...`), then list it among what the goal waits on (`--wait goal:SUB`, repeated for every
  link, since `--wait` replaces the list). A goal has no separate next-step field, and the site refuses one.
- **A status:** `on-track`, `off-track` or `done`, your forecast of whether it will be true by its date.
  - Off track says why: link what it waits on, or say what's wrong in `--blockers`. The site refuses an off-track goal
    with neither.
  - Done needs evidence named in the description: a run, a lock change or a landed commit.
- **Tags** (`--tag`, repeated) are the development areas responsible for the goal, from one fixed list in
  Daniel's words: `remote-demo`, `onsite-demo`, `proofs`, `circuits`, `lean`, `vllm`, `compute-accounting`,
  `network-accounting`, `memory-accounting`, `accounting` (work across all three), `training`, `ci`, `infra`,
  `security`, `cleanup`. The site refuses any other tag.
  - An area is a scope of responsibility in Verity's automated development, usually one lead's, often a part of
    the architecture. The demos assemble the others; `ci`, `infra`, `security` and `cleanup` belong to the system that
    develops Verity.
  - Tag the areas a goal builds or changes, never one it only uses. The trivial PoUW's Lean guarantee uses the
    proof system, so it's `compute-accounting`, not `proofs`. The daily-program check builds part of the remote demo,
    so it's `remote-demo`.
  - One tag is work inside an area; several mark joint work that can't be done at their interfaces.
  - `proofs` is the proof system; `lean` is Lean's code and build (the module system, the Lean check, toolchain bumps).
  - The list is at Daniel's level. Finer areas, such as kernels, benchmarks, compute-research, soundness
    and comms, are real to the leads but below it: their goals take the tag of the area they belong to at his
    level. A kernel the demo serves is `compute-accounting` or `vllm`, and the demo's cost measurements are
    `remote-demo`.
  - Why a goal matters isn't a tag: it's the goal above it.
- **Goals and subgoals.** Everything is a goal, and a goal under another is its subgoal: the goal waits on it
  (`--wait goal:ID`). Don't call a goal a "project", a "piece", a "step" or a "tier", in a title or a description.
  - A goal nothing waits on is one of Daniel's own: about a dozen across the plan, each one he would name himself.
    Its description opens with the state that finishes it.
  - A subgoal is a dated state you can check: the soundness theorem and the H200 accounting are subgoals of the
    remote demo.
  - **Before adding a goal nothing waits on,** ask whether it's a subgoal of one already in the plan. If nothing
    holds it only because an umbrella is missing, add the umbrella or ask its owner to. A merge isn't a goal of
    Daniel's.
  - The page derives how deep a goal sits from what waits on what, down to three levels. The command's `--tier N`,
    which keeps its old name, overrides that, and `--no-tier` goes back to the derived one.

## What it waits on

Links only (`--wait goal:ID`, `--wait ask:CARD`, repeated for each; `--no-wait` clears them):

- Link the **narrowest** goal that blocks you, **due before you**. Waiting on main moving today is
  `goal:ci/main-moves-again`, not ci's broad landing-speed goal.
- If what blocks you isn't in the plan, add it as a subgoal, owned by whoever delivers it, or ask that owner to.
- A decision of Daniel's is its ask (`ask:<card id>`, from `research msg ask-daniel`).
- A link carries nothing else. The dependency's state lives on it; why you need it goes in your description.
- Your own work isn't a dependency. Slow or risky work is your status, explained in your description.

A goal's peek shows what it waits on and what needs it in one timeline around its own due date, so a link due after
it shows red. Fix the cause (re-date, re-scope, or chase the dependency) rather than the status.

## The description

Markdown for Daniel to skim. It never restates a field (due date, status, owner, tags or links): the page shows those.

**Open with one plain sentence** saying precisely what will be true, sharper than the title. Not bold.

Then the shape that fits the goal:

- **Landing** (code reaching main, a migration, a deploy): the path as a numbered list, in order, with times and
  owners, finished steps first. Sub-bullets hold the detail.
- **Research** (a measurement or a result): what's measured so far, with the key numbers and what they're on; the plan
  as a numbered list; then one line on what the numbers rest on (assumptions, unproved theorems, seeds).
- **Proof** (a theorem in Lean): the statement in plain words, then what's proved and what's left, with other teams'
  parts linked as goals.

Close with the date only when it says something the due date doesn't: "Due **Fri**. **Wed** would need both subgoals
within two days."

Rules:

- Write what Daniel needs to judge the forecast: what will be true, where it stands, the plan. A derivation, a proof
  sketch or a list of runs goes in the linked issue or note; the goal links it.
- Current truth only: no corrections, no "moved to", no history of the goal. The page shows when it was last written.
- Say where work stands in outcome words: running, in review, waiting to merge, on main. Not the pipeline's words:
  "approved for landing", "marked", "the landing loop", "drive", "a train".
- Name who does a step as @handle in its sentence ("@ci moves main on it"), never as a tag in parentheses.
- A step's day is usually enough ("Tue", "by Wed noon"). Give the hour only when another goal waits on it.
- Leave out what Daniel doesn't act on: run counts, restarts, idle machines, GPU counts.
- Plain words over coined ones. "Fresh products", "whole-making", "the sealed shard" and "a live coin" are terms you
  made; say what they mean ("matrix products whose two inputs were both just computed"), or define the term once.

- Measured numbers are LaTeX: $9.5$ ms, $98.75\%$, $0$ of $6{,}256$ tiles. Bold only dates and decisions waiting on
  Daniel, never a sentence or a clause.
- Code (backticks) is for a literal someone would type, flip or match: a GPU target (`sm_90a`), a number format
  (`fp8`, `bf16`), a served option (`--enable-chunked-prefill`), a command (`research goals set`). Name the literal
  when it's the thing someone flips or checks, not "one fast path". Names of things stay plain: vLLM, cuBLASLt,
  Llama 3.2 1B, the H200. The site refuses a word with an underscore outside code.
- No fixed labels, but every list has a lead-in line ending in a colon that says what it is, in the goal's own words:
  "So far, on the pilot:", "Plan:", "Proved:", "Left:", "By Tue 6 AM:". The site refuses a list without one.
- Math is LaTeX between dollar signs: $\rho$, $p = C/H$, $2^{-256}$. The site refuses a Greek character (ρ, ε) or a
  superscript (2⁻²⁵⁶) outside math or code.
- No "why it matters" when the title already says it. No "to check it" unless the evidence isn't obvious, and then one
  line naming it.
- Sub-bullets are welcome: indent them three spaces under their item.
- Times are Pacific, written like "Mon 10 PM", with no zone and never UTC.
- No PR numbers, ticket ids, branch names, run ids, agent ids, file paths or code names (a package, module or
  function: "te_core", "page_rho"). Say what the thing does. No goal by number either ("goal 5", "the eleven"):
  link it by its title.
- Name a lead as @handle (@ci, @proofs): the page sets it as the lead's chip, apart from the area tags and the
  software a handle is named after.
- The site refuses a word Daniel has said he doesn't know, and its refusal says what to write instead. The list is
  `UNKNOWN` in the website repository's goals check (`plain.ts`); when he says he doesn't know a word, tell dashboard,
  who adds it with his words.
- Give numbers with units and what they compare to.
- Link another goal with `[text](#goal-id)` and an ask with `[text](ask:<card id>)`; both render as chips. Never a
  bare id.

A research goal, as Daniel approved it:

~~~markdown
For a ladder of fully trained models, the cheapest covert route to each lower loss, measured against the cheapest
honest one, under the training formalism (v2).

So far, on the FineWeb-Edu pilot:
- slowdown at the default policy is about $200$ to $430$ for most routes;
- as low as $1$ for adapters on the embedding. That rests on the theorem's attention loose end: **your call**,
  sent 5:28 PM.

Plan:
1. Tonight the 20M, 60M and 200M starts retrain on OLMo 2's pretraining mix.
2. Every frontier point is rerun on the sealed shard at $3$ seeds.
3. Both training plots from those runs by **Wed 8 AM**.

Every slowdown rests on assumptions A1 and A1′ until the Slowdown theorem is proved in Lean.
~~~

A proof goal:

~~~markdown
What the zero-knowledge server releases reveals at most about $1$ bit a session, plus the counted bits of where a
session stopped, when the worker runs the honest prover.

Proved:
- the server's release rounds and the inner proof's rounds, approved for landing;
- this week, the inner phase checked in Lean instead of Python.

Left:
- the honest worker's requests, in Lean, shown to pass the server's checks:
  [memory accounting](#accounting/honest-requests-pass-firewall), with
  [proofs making the checks provable](#proofs/firewall-checks-provable);
- the stop leak, about $11$ bits a session, proved from the session's rules:
  [network accounting](#accounting/stop-leak-proved).

Due **Fri**. **Wed** would need both subgoals within two days.
~~~

## Commands

~~~bash
research goals show                              # every goal by due date, with what each waits on
research goals set proofs/zero-knowledge --title "Zero knowledge proved for an honest worker" --owner proofs \
  --due 2026-11-06T18:00-08:00 --status on-track --tag proofs --wait goal:proofs/firewall-checks-provable \
  --description @zk.md
research goals set ci/fine-grained-check --status off-track --blockers @why.md
research goals set vllm/partition-on-main --tier 2   # only when the derived depth is wrong
research goals confirm ID                        # still true as written
research goals done ID
research goals remove ID
~~~

A wrong field is refused with a message naming it.
