---
name: writing-goals
description: >
  How a lead writes and keeps its goals in the plan, the one graph Daniel reads on the console's Goals page: what the
  plan is for, what makes something a goal (one outcome he would track on its own; when to split a goal into
  subgoals, and when a step, a pull request or a run folds into one), how to title it as a headline Daniel reads cold
  (three to six plain words, no number, none of our own labels), tags from the fixed list of areas, what a goal waits
  on (links to goals and Daniel's asks, never free text), which goal is set off track (the one where work is late),
  the top goals (a list Daniel chooses), and how to write the description for a reader who knows the architecture and
  has not followed the work (what will be true, what is true today instead, then a target, limits or risks only when
  they change a decision; measured numbers in LaTeX, literals as code, no word he has said he doesn't know). Covers
  `research goals show|set|confirm|done|remove|replace` and the API it calls. Use when writing, titling, updating or
  confirming a goal, splitting one, adding a dependency, or rewriting your goals for a sync.
---

# Writing goals

## What the plan is for

Daniel steers the work from the console's Goals page without following the work. He reads it cold to learn what we
are making true and by when, which of those will be late, and what stands in each one's way. Then he acts: he moves a
date, sets a priority, answers an ask, or changes who has it.

Every rule below follows from that. Where none covers your case, ask what he could see and do with the goal as you
wrote it. Agents write the page with `research goals`; Linear holds the work underneath, each issue serving a goal.

## What makes something a goal

One outcome he would track on its own: something that will be true of the system, a result or a decision made, that
can be done, late or stuck independently of the goals beside it. It has one owner, one date and one status, so it
has to be one thing. Three ways to miss:

- **Too coarse.** Work that finishes separately is separate goals. The sign is an "and" in where it stands: he can't
  see which part is late. Each part is a subgoal with its own owner, date and reason, and the goal waits on them.
  What they must agree on first, such as an interface, is a subgoal they wait on.
- **Too fine.** If its slipping would change nothing he does, it folds into the goal it serves, where anything he
  would decide on becomes a limit or a risk: a day's step; a pull request's progress (in review, waiting to merge,
  tests passing, rebased across a move); a run, which is evidence for the goal that needs its result; a worker
  started, a brief written, a review fix, a handoff.
- **Not an outcome.** An ongoing activity ("Work on demos") never finishes, and "X measured" isn't one until it names
  what's delivered: "Second swarm's training results out".

His case: "Core benchmarks run from main on a small model" covered three benchmark commands, and its one reason for
being off track listed three unrelated states. It stays his goal and waits on a subgoal for each benchmark, and those
on one for what every benchmark command takes and prints.

## Keeping it current

He acts on what the page says today, so it moves when the work does:

- **What Daniel asks for shows until all of it is true.** The page's first job is that nothing he asks for is started
  and then dropped. The day he asks, it is a goal: the outcome he wants, interpreted in plain words, never quoted or
  marked as his, with its key points as bullets (the main renames, not every name). It sits directly under the goal
  of his it serves, titled as he would look for it, not inside a group or one sentence of a broader goal. Today's
  sentence says which points are not yet true; it is done when all are, and he confirms nothing. Whoever regroups,
  parks or folds goals checks that each still shows.
- **When a goal's state changes**, rewrite the sentence that says what is true today, and its status the moment it
  slips; don't wait for a sync. Progress, plans and steps live in the goal's Linear issues, never in the goal.
- **Work serves a goal on the page:** its issue carries the goal's label (the `goal` label group, one label per id),
  and its PR names that issue: `research queue open --issue KEY` writes a `Goal <id>` line for each live goal KEY's
  labels name, and warns of a body that names none. Work that serves none starts by writing its goal, or asking top.

`research goals audit` names, and once a day top rings the owner of, a goal past its due, an off-track goal not written
in 24 hours, and a goal due within 24 hours not written in the 24 before its due. A parked goal is left out.

## A goal's fields

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
  | The network attester's term is zero without a failure | Network attestation adds no exfiltration capacity unless a connection fails |
  | Remote theorem rests on no open proof | Remote demo's soundness fully proved |
- **One owner** (`--owner`): the lead of the area the goal mostly builds, who keeps it true. Another area's part of
  it is that lead's goal, linked; a check the remote demo runs is the demo lead's goal even when proofs writes it in
  Lean.
- **A due date**, more precise as it nears (`--due YYYY-MM-DD`, the end of that day, or `YYYY-MM-DDTHH:MM-08:00`):
  - more than a week away, a day;
  - inside a week, the day you now expect, moved the day your estimate moves;
  - inside 24 hours, the hour its last step ends (a run's or a check's end), set that day: 6 PM only when that is
    the estimate. `research goals confirm ID` if the hour you gave stands.
- **A priority**, 1 to 3: anyone may set one (`research goals priority ID 1|2|3|clear`), and the page shows
  who set it and when. A goal nobody set takes the highest priority of the goals waiting on it.
- **A status:** `on-track`, `off-track`, `parked` or `done`, your forecast of whether it will be true by its date.
  - **Only the goal where work is late is set off track:** one whose own work will miss its date, with no open goal
    under it that explains why. Its `--blockers` say what stops it today, in one or two plain sentences. Each day it
    stays off track, rewrite them, or `confirm` it if every word holds.
  - **A goal that waits on open goals takes its status from them:** off track when one under it is off track, past
    its date or parked. The page's Bottlenecks view lists those. Leave the goal on track with no blockers; the site
    refuses a new off-track status or reason on it. If it will be late and nothing under it is, the late work has
    no goal yet: write it as a subgoal with its owner and date, link it, and set that one off track.
  - A goal due before one it waits on shows off track too, though no work is late: one date is wrong. Move one, or
    unlink what it doesn't need.
  - Parked: nobody works toward it for now; its `--blockers` say what it is parked behind.
  - Done needs evidence named in the description: a run, a lock change or a landed commit.
- **Tags** (`--tag`, repeated) are the development areas responsible for the goal, from one fixed list in
  Daniel's words: `remote-demo`, `onsite-demo`, `proofs`, `circuits`, `lean`, `vllm`, `compute-accounting`,
  `network-accounting`, `memory-accounting`, `accounting` (work across all three), `training`, `ci`, `infra`,
  `security`, `cleanup`.
  - An area is a scope of responsibility, usually one lead's.
  - Tag the areas a goal builds or changes, never one it only uses. The trivial PoUW's Lean guarantee uses the
    proof system, so it's `compute-accounting`, not `proofs`. The daily-program check builds part of the remote demo,
    so it's `remote-demo`.
  - One tag is work inside an area; several mark joint work that can't be done at their interfaces.
  - `proofs` is the proof system; `lean` is Lean's code and build (the module system, the Lean check, toolchain bumps).
  - Finer areas (kernels, benchmarks, compute-research, soundness, comms) take the tag of the area they belong to at
    Daniel's level. A kernel the demo serves is `compute-accounting` or `vllm`, and the demo's cost measurements are
    `remote-demo`.
  - Why a goal matters isn't a tag: it's the goal above it.
- **Goals and subgoals.** Everything is a goal, and a goal under another is its subgoal: the goal waits on it
  (`--wait goal:ID`). Don't call a goal a "project", a "piece", a "step" or a "tier".
  - **The top goals are a list Daniel chooses,** about a dozen. Only the page's editor (`@dashboard-goals`) adds
    one, and the site refuses anyone else's `--tier 1`: send it the title, owner and date. A top goal's description
    opens with the state that finishes it.
  - **Every other goal sits under a top goal being worked on,** linked directly or through a subgoal. One that
    nothing waits on shows under "Under no top goal" until it is linked under the goal it serves, or parked.
  - **Before adding a goal,** look for it under every owner, and find the goal it serves. If that goal is missing,
    add it or ask its owner to, or the editor if it would be a top goal. A merge isn't a goal of Daniel's.

## What it waits on

Links only (`--wait goal:ID`, `--wait ask:CARD`, repeated for each; `--no-wait` clears them):

- Link the **narrowest** goal that blocks you, **due before you**. Waiting on main moving today is
  `goal:ci/main-moves-again`, not ci's broad landing-speed goal.
- If what blocks you isn't in the plan, add it as a subgoal, owned by whoever delivers it, or ask that owner to.
- A decision of Daniel's is its ask (`ask:<card id>`, from `research msg ask-daniel`).
- A link carries nothing else. The dependency's state lives on it; why you need it goes in your description.
- Your own work isn't a dependency: it is your status, or one more subgoal on a goal that waits on open goals.

A link due after you shows red in your list of what you wait on. Fix the cause: re-date, re-scope, or chase the
dependency.

## The description

Markdown. It never restates a field (due date, status, owner, tags or links): the page shows those.

**The reader** knows the architecture and its glossary and has not followed your work. He reads a goal to learn what
will be true and whether to worry.

1. **Open with what will be true when the goal is done.** One sentence about a thing the reader can identify. If he
   couldn't identify it, identify it in that same sentence by what it does and where it sits ("the test that the vLLM
   adapter's int8 matrix multiplication is bit-exact"), never in sentences before it. Not bold.
2. **Next to it, say what is true today instead, and what that costs.** Skip this when the opening makes it plain.
3. **Add more only if it changes what the reader would decide:**
   - `### Target`: a number the outcome must hit, and what it compares to.
   - `### Limits`: what "done" does not include, which a reader would otherwise assume it does.
   - `### Risks`: what could make it false or late, that is specific to this goal.

   Use a section only when it has real content. Most goals need none or one.

Test: after the first two sentences, could that reader say what changes and why it matters? If not, fix those two
sentences. Do not add more around them.

Rules:

- **Do not write what you cannot explain** to that reader: what the thing is, and what breaks without it.
- **Say what, never who.** No handle and no lead as the subject; the owner is a field. The exception is when who
  holds something is itself the content: which lead owns a directory, or an outcome that is one team's own number
  ("ci's changes refused for a merge conflict"). With no goal to link, ask that lead for one and name the lead until
  it exists.
- **No status diary:** no "where it stands", "so far" or "plan", no dated steps, nothing about what is in review or
  waiting to merge, no run counts, restarts, machines or GPU counts, no corrections, no history of the goal, no who
  approved what. The page shows when it was last written.
- **Anything you refer to is a goal he can click, or it is cut.** Link another goal with `[its title](#goal-id)` and
  an ask with `[text](ask:<card id>)`; both render as chips. Never a bare id, and never a goal by number.
- **Use the architecture's word for a thing.** No coined synonym ("PoUW's check" for PoUW's output) and no lane
  shorthand ("the swarm", "the serve"). A term the reader may not know is said in plain words, or added to the
  page's glossary: tell dashboard.
- **Parallel items are bullets** under a lead-in line ending in a colon that says what they are, in the goal's own
  words. Never an inline list with counts. Sub-bullets are indented three spaces.
- Measured numbers are LaTeX, with units and what they compare to: $9.5$ ms, $98.75\%$, $0$ of $6{,}256$ tiles. Keep
  a number only if it changes what the reader thinks. Math is LaTeX between dollar signs: $\rho$, $2^{-256}$.
- Code (backticks) is for a literal someone would type, flip or match: a GPU target (`sm_90a`), a number format
  (`fp8`), a path, a command (`research goals set`). Names of things stay plain: vLLM, Llama 3.2 1B, the H200.
- No PR numbers, ticket ids, branch names, run ids, agent ids or code names (a package, module or function:
  "te_core", "page_rho"). Say what the thing does.
- Bold only a decision waiting on Daniel. A date appears only when it is the fact ("defaults on **Mon 16 Nov** at
  6 PM"), written in Pacific time like "Mon 10 PM", with no zone and never UTC.
- When Daniel says he doesn't know a word, tell dashboard, who adds it to the words the site refuses.

Goals as Daniel approved them. An outcome whose problem today needs saying:

~~~markdown
The network attester certifies every execution phase of a GLM-5.3 serving run.

On the last served try it did not sign the first execution phase, because it stalled while the operating system moved
its memory pages. It signed every later phase, and the try as a whole was still refused, since one phase
was unsigned. The attester's node is now set not to move those pages.

### Risks

The new setting was tried only on a nearly idle node, where the stall is less likely to show than under serving load.
~~~

Parallel items, and a limit:

~~~markdown
One list covers the inference-only policy's Lean:

- each guarantee, with what Lean proves of it today
- each open part, with its owner and date
- what the statement for served traffic still lacks

### Limits

Of the $25$ open parts:

- $8$ have no date from their owner
- $2$ have no owner
~~~

## Commands

~~~bash
research goals show                              # every goal by due date
research goals set proofs/zero-knowledge --title "Zero knowledge proved for an honest worker" --owner proofs \
  --due 2026-11-06T18:00-08:00 --status on-track --tag proofs --wait goal:proofs/firewall-checks-provable \
  --description @zk.md
research goals set ci/fine-grained-check --status off-track --blockers @why.md
research goals confirm ID                        # still true as written
research goals done ID
research goals remove ID
~~~

The site checks every write, and a refusal names the field and what to write instead.
