---
name: process-design
description: >
  How to propose, change or retire a process in Verity, and how to assess its effect on research velocity first. A process
  is anything that runs on other people's work: a rule in AGENTS.md or the lane contract (LANE-CONTRACT.md), a check or gate
  in `check` or `research merge`, a required review, approval or sign-off, a file or document every PR must keep current, a
  convention every lane follows. Covers the scarce resources to price (the human's attention, the merge path, shared files
  and queues, always-read rules, pods), ten tests of a good process, a walkthrough on the last weeks' real changes from git
  and gh, a default budget, a proposal template, and a retirement pass that deletes rules or moves them into code. Use when
  adding a rule after an incident, adding a gate or a required reviewer, making PRs update a shared file, editing AGENTS.md
  or the lane contract, when a process feels heavy or slow, or when pruning rules.
---

# Process design

A **process** is anything that runs on work other than the change that introduces it: a rule agents read, a check in
`check`, a gate in `research merge`, a required reviewer, a file every PR must update, a convention every lane follows. A
test in your own package that guards your own code is not one. Every process costs something on each change it touches
and pays off only on the few where it catches an error, so assess it before proposing it, and retire it once it no
longer pays.

## What research velocity is

Research velocity is the rate at which Verity produces results someone can act on: security claims proved under named
assumptions, benchmark measurements, design decisions, working integrations. It is not the rate of commits: a fast result
that is later retracted costs velocity. A process moves it through five channels:

- **Loop latency:** the time from an idea to a trusted result. Steps, builds, queues and reviews on the critical path add
  to it.
- **Parallelism:** how many lanes can work at once without colliding. Shared files, single queues and global rules
  subtract from it.
- **Decision latency:** the time a result waits for a human. Every blocking question adds to it; defaults and delegation
  cut it.
- **Yield:** how many results stay true. A check at the right point raises it; an error caught late costs rework and
  retractions.
- **Reuse:** how easily a result is found, cited and built on: the evidence store, labels, generated views.

The first three are paid on every change the process touches. The last two pay off only on the changes where an error
would have happened or a result gets reused. A process earns its place when its benefit on those few changes exceeds its
cost on all of them.

**Price only the scarce resources.** Machine time and agent time are cheap and run in parallel. These are not:

1. **the human's attention**, the bottleneck;
2. **the merge path:** trains, `check` on a train's tip, the coordinator;
3. **files and queues many lanes pass through;**
4. **the words every agent reads before working:** the root `AGENTS.md`, the lane contract, the `AGENTS.md` of the
   subtree;
5. **pods and money.**

Agents follow rules literally and at scale, and they don't push back on a bad one. A rule's cost is multiplied by the
number of agents and how often it fires, and nothing prunes it unless someone does.

## The ten tests

Ask each of these of a proposed process. A "no" costs velocity unless the proposal says why it's worth it.

1. **Is it priced in scarce resources, not in steps?** Ten automatic checks of a few seconds cost less than one approval
   per change. Good: suites cached by input, so only what changed reruns. Bad: a named reviewer on every change, when the
   reviewer is always the human.
2. **Does it fire where results are relied on, not where they're made?** The reliance points are landing on `main`,
   publishing a table or a Notion page, citing a theorem as proved, and handing a spec to another implementer. Most
   exploration results are thrown away. Good: records only for cited theorems (guarantees); `experimental` schemes.
   Bad: a `PROTOCOL.md` kept current on every behavior change, which cost 64 commits to the protocols' prose against 70
   to their Python in the week of Sep 22–29.
3. **Is it code, with the fix in its failure message?** A check costs nothing until it fires, and it teaches when it
   does. Prose taxes every agent's context and is followed unreliably; the lane contract says it: "a rule you can't see
   enforced is not protection". Good: the Lean check's `escapes` list; `tests/test_boundaries.py`.
4. **Does it label rather than block, unless the error would pass unnoticed?** A label lets work go on and lets readers
   filter. Block only for errors nobody would notice until they cost a lot: #118 made `merkle_binding` vacuous by
   redefining `MerkleScheme.Collision`, the axiom audit passed it, and only the guarantees' records and a person reading
   them catch that. Good: `(alg.)` on algebraic-hash results.
5. **Does it keep one source and generate the rest?** Every hand-kept copy doubles edits and drifts. Bad: the dense
   scheme's Lean re-certified as "matching the code" after the tag format had diverged between Lean and Python;
   hand-written proved/open tables.
6. **Does it add no serialization point?** A file every PR edits, one queue or one reviewer caps how many lanes can work
   at once and turns one failure into many waits. Bad: soundness's flat root import list, which conflicted across trains.
7. **Is following it the cheapest way to explore?** If it isn't, work moves outside it and the cost comes back at
   integration. Bad: research Lean kept outside the repository, whose landing needed flattened files and renames by
   regex checked by hand.
8. **Does every decision it asks for stand alone, with a default that proceeds?** Waits grow sharply once a reviewer is
   near capacity, so one blocking question stalls many lanes. A provisional default keeps them moving and is cheap to
   reverse. Good: POUS's provisional `Meets` clauses.
9. **Does it have an owner, a named failure and an exit?** Rules accrete one per incident and rarely leave.
10. **Is it a rare change, made in code?** Rule churn has a cost of its own: every agent's instructions and every
    in-flight brief go stale. The root `AGENTS.md` took 25 edits on Sep 27.

In one sentence: a good process makes the right thing the easy thing while exploring, and makes the wrong thing loud
where a result is relied on.

## Assess a proposed process

1. **Name the failure:** an incident (a PR, commit or note id) or a concrete risk, how often it happens, and what it costs
   when found late: rework, a retracted table, a false claim to a colleague. No failure, no process.
2. **Find the smallest mechanism.** Try these in order, and stop at the first that works: nothing (the failure is rare
   and cheap to fix late); a label; a code check in the suite that owns the files it reads (see
   `.agents/skills/testing/SKILL.md`); a generated artifact in place of a hand-kept one; and only then a prose rule or a
   human gate.
3. **Pull the real changes.** Take the last one or two weeks of history in the area the process touches, and group the
   changes by type, with counts: behavior changes, refactors, proofs, experiments, promotions (a scheme becomes the
   default, a format gets a new version), publications.

   ~~~sh
   git fetch origin main
   git log origin/main --since=2.weeks --date=format:'%m-%d %H:%M' --format='%h %ad %s' -- verity/core/protocols/pous   # changes by path
   git log origin/main --since=1.week --format=%h -- AGENTS.md | wc -l                                        # how often a file changes
   gh pr list --state merged --limit 300 --json number,createdAt,mergedAt,files --jq '
     .[] | select(any(.files[]; .path | startswith("verity/core/protocols/")))
     | [.number, (((.mergedAt|fromdate) - (.createdAt|fromdate)) / 3600 * 10 | floor / 10)] | @tsv'         # hours to merge
   wc -w AGENTS.md ~/.research/notes/kb/LANE-CONTRACT.md                                                      # always-read words
   ~~~

4. **Walk each type through the process.** For each, record the steps it adds, the wall-clock time it adds on the
   critical path, whether it blocks or labels, and who acts: the machine, an agent, the coordinator or the human. Walk the
   bursts too: research moves in them (POUS changed its default scheme three times in nine hours on Sep 27–28), and a
   gate that is cheap on average can serialize a burst.
5. **Price it per week** in the scarce resources: the human's minutes, merge-path minutes, new files or queues many lanes
   touch, words added to always-read rules, pod money. Compare the total with the failure's expected cost per week: how
   often it happens times what it costs when found late.
6. **Run the ten tests,** with a one-line answer to each.
7. **Simplify and walk again.** Drop every piece whose cost exceeds its benefit, and repeat steps 4 to 6 on what's left.
8. **Pilot.** Apply it to one package or one lane for about eight hours, measure the same numbers, then extend it or retire
   it. Lanes work around the clock, so a pilot gets hours, not the week a human team would give it.

### Default budget

Exceeding any of these needs a stated reason in the proposal:

- no new step on the common path (branch edits, exploratory runs) unless it's automatic and takes under a minute;
- no new blocking human approval on the common path: human gates sit at reliance points, as stand-alone questions with a
  default;
- no new file or queue that most PRs must touch;
- a few lines at most added to `AGENTS.md` or the lane contract: the detail goes in a skill or in the check's failure
  message;
- an owner, a named failure with its incident, and an exit.

### Worked example

The Lean-first protocol norms, as first proposed: `check` regenerates each protocol's vectors from its Lean spec on every
commit; the Python passes every vector except those in a hand-kept lag list; and it must match at three points
(publishing a result, citing a theorem for a run, and making a scheme the default or bumping a wire format). Walked
through a week of changes to POUS, PoUW, the one-stage audit and sampled proofs:

- about 30 Python changes that don't change behavior, 3 experimental scheme versions and about 14 Lean proofs: no cost;
- about 6 changes of a default or a law: each would wait 1 to 3 hours for an executable spec, new vectors and a
  guarantee review, which would have serialized POUS's three default switches in nine hours;
- the lag list: a second source, and a file lanes would collide on.

The walkthrough justifies neither the lag list nor the default-switch match point, because a result built on an
unmatched default is labelled already. The lighter version keeps one match point, publishing or citing a result as the
protocol's, and `check` detects lag by itself: the Python may fail a vector only if that vector's expected value changed
since the Python last passed it.

## Write the proposal

Put it in the PR description. When the process needs a decision before any code is written, put it in a `kind: draft`
note under `~/.research/notes/lanes/<lane>/` instead, for the coordinator to forward. Keep it under a page:

~~~markdown
**Failure.** <incident id or concrete risk>; <how often>; <cost when found late>.
**Mechanism.** <the smallest one; the smaller ones tried, and why they fall short>.
**Walkthrough** (<area>, <dates>):

| Change type | Count | What the process adds | Blocks or labels | Who acts |
|---|---|---|---|---|

**Price per week.** Human: <minutes>. Merge path: <minutes>. Shared files or queues: <which>. Always-read words: <+n>.
Pods: <$>. Against the failure's expected cost: <per week>.
**Tests.** <one line for each of the ten that isn't a clear yes>.
**Budget.** <within it, or which line is exceeded and why>.
**Owner and exit.** <who owns it; what retires it>.
**Pilot.** <package or lane>; <the numbers to measure, and when>.
**Decisions.** <each a stand-alone question with a recommended default that proceeds if nobody answers>.
~~~

## Retire a process

Rules only accumulate unless someone removes them. Run a retirement pass:

- when asked, or when a process feels heavy;
- before adding a rule to `AGENTS.md` or the lane contract, over the section you're adding it to;
- weekly over the root `AGENTS.md` and the lane contract; the coordinator owns this one.

For each rule, find where it came from (`git log origin/main -S'<phrase>' --format='%h %ad %s' --date=short -- AGENTS.md`;
the lane contract's rules carry a "Why:"), then take the first of these that applies:

- **enforced in code already:** delete the prose, or cut it to a pointer at the check;
- **enforceable in code:** write the check, with the fix in its failure message, and delete the prose;
- **needed by one kind of work only:** move it into that work's skill, or into the subtree's `AGENTS.md`;
- **its failure hasn't recurred and would be cheap to catch late, or it fails most of the ten tests:** delete it;
- **otherwise:** keep it, and give it an owner and an exit if it lacks them.

Make the removals in one PR, with one line of reason per rule. A removal that changes agreed semantics or acceptance
criteria goes to the human as a stand-alone question with a default.

## Before you land it

- The failure is an incident, or a concrete risk with a rate, not "it could happen".
- The numbers come from the commands above, not from estimates.
- The budget is met, or each line it exceeds has a reason.
- A code check fails with a message that says how to fix the failure, and runs in the suite that owns the files it reads.
- What the change adds to `AGENTS.md` or the lane contract is routing and invariants only, and any rule it replaces is
  removed in the same change.
- The proposal names an owner, an exit and a pilot.
