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
`check`, a gate in `research merge`, a required reviewer, a file every PR must update, a convention every lane follows (a
test guarding your own package's code is not one). It costs something on every change it touches and pays only on the
few where it catches an error, so assess it before proposing it, and retire it once it no longer pays.

## What it costs

Research velocity is the rate of results someone can act on (proved claims, measurements, design decisions, working
integrations), not of commits: a retracted result costs it. A process costs **loop latency** (idea to trusted result),
**parallelism** (lanes working at once without colliding) and **decision latency** (time waiting on a human) on every
change it touches, and pays in **yield** (results that stay true) and **reuse** (results found, cited, built on) only
where an error or a reuse happens. It earns its place when that benefit exceeds its cost on all of them.

Price only the scarce resources: the human's attention (the bottleneck); the merge path (`next`, its reviewers, a Lean
change's Lean check, the coordinator); files and queues many lanes pass through; the words every agent reads first (the
root `AGENTS.md`, the lane contract, a subtree's `AGENTS.md`); pods and money. Machine and agent time are cheap. Agents
follow rules literally and at scale and never push back, so a rule's cost multiplies by the agents and how often it
fires, and nothing prunes it unless someone does.

## The ten tests

A "no" costs velocity unless the proposal says why it's worth it.

1. **Priced in scarce resources, not steps?** Ten cached automatic checks cost less than one approval per change.
2. **Fires where results are relied on** (landing on `main`, publishing a table or Notion page, citing a theorem as
   proved, handing a spec to another implementer), **not where they're made?** Most exploration is thrown away.
3. **Code, with the fix in its failure message?** A check costs nothing until it fires; prose taxes every agent's context
   and is followed unreliably.
4. **Labels rather than blocks**, unless the error would pass unnoticed until it cost a lot (a redefinition that makes a
   guarantee vacuous while the axiom audit passes)?
5. **One source, the rest generated?** A hand-kept copy doubles edits and drifts.
6. **No serialization point?** A file every PR edits, one queue or one reviewer caps parallel lanes and turns one failure
   into many waits.
7. **Following it is the cheapest way to explore?** Otherwise work moves outside it and the cost returns at integration.
8. **Each decision it asks for stands alone, with a default that proceeds?** Waits grow sharply near a reviewer's
   capacity; a provisional default is cheap to reverse.
9. **An owner, a named failure and an exit?** Rules accrete one per incident and rarely leave.
10. **A rare change, made in code?** Rule churn stales every agent's instructions and in-flight brief.

A good process makes the right thing easy while exploring, and the wrong thing loud where a result is relied on.

## Assess a proposed process

1. **Name the failure:** an incident (PR, commit or note id) or a concrete risk with its rate, and its cost when found
   late (rework, a retracted table, a false claim to a colleague). No failure, no process.
2. **Find the smallest mechanism**, stopping at the first that works: nothing (rare and cheap to fix late); a label; a
   code check in the suite that owns the files it reads (the testing skill); a generated artifact in place of a hand-kept
   one; only then a prose rule or a human gate.
3. **Pull the real changes** of the last one or two weeks in the area, grouped by type with counts: behavior changes,
   refactors, proofs, experiments, promotions (a new default scheme or format version), publications.

   ~~~sh
   git fetch origin main
   git log origin/main --since=2.weeks --date=format:'%m-%d %H:%M' --format='%h %ad %s' -- verity/core/protocols/pous  # changes by path
   git log origin/main --since=1.week --format=%h -- AGENTS.md | wc -l  # how often a file changes
   gh pr list --state merged --limit 300 --json number,createdAt,mergedAt,files --jq '
     .[] | select(any(.files[]; .path | startswith("verity/core/protocols/")))
     | [.number, (((.mergedAt|fromdate) - (.createdAt|fromdate)) / 3600 * 10 | floor / 10)] | @tsv'  # hours to merge
   wc -w AGENTS.md ~/.research/notes/kb/LANE-CONTRACT.md  # always-read words
   ~~~

4. **Walk each type through it:** the steps and critical-path time it adds, whether it blocks or labels, and who acts
   (machine, agent, coordinator or human). Walk the bursts too: a gate cheap on average can serialize one.
5. **Price it per week**, from those commands' numbers, not estimates, in the scarce resources (human minutes,
   merge-path minutes, files or queues many lanes touch, always-read words, pod money) against the failure's expected
   cost per week (rate times late cost).
6. **Run the ten tests**, one line each.
7. **Simplify and walk again:** drop every piece whose cost exceeds its benefit, and repeat steps 4 to 6.
8. **Pilot** it on one package or lane for about eight hours, measure the same numbers, then extend or retire it.

**Default budget**; exceeding a line needs a stated reason:

- no new step on the common path (branch edits, exploratory runs) unless automatic and under a minute;
- no new blocking human approval on the common path: human gates sit at reliance points, as stand-alone questions with a
  default;
- no new file or queue most PRs must touch;
- a few lines at most added to `AGENTS.md` or the lane contract, routing and invariants only, removing any rule they
  replace in the same change; the detail goes in a skill or a check's failure message, and a check runs in the suite that
  owns the files it reads;
- an owner, a named failure with its incident, an exit and a pilot.

## Write the proposal

In the PR description, under a page; when it needs a decision before any code, in a `kind: draft` note under
`~/.research/notes/lanes/<lane>/` for the coordinator to forward:

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

Run a retirement pass when asked or a process feels heavy, over the section a new rule joins before adding it, and
weekly over the root `AGENTS.md` and the lane contract (the coordinator's). For each rule, find its origin (`git log origin/main -S'<phrase>' --format='%h %ad %s' --date=short -- AGENTS.md`; the lane
contract's rules carry a "Why:") and take the first that applies:

- **enforced in code already:** delete the prose, or cut it to a pointer at the check;
- **enforceable in code:** write the check, with the fix in its failure message, and delete the prose;
- **needed by one kind of work only:** move it into that work's skill or the subtree's `AGENTS.md`;
- **its failure hasn't recurred and would be cheap to catch late, or it fails most of the ten tests:** delete it;
- **otherwise:** keep it, with an owner and an exit.

Make the removals in one PR, one line of reason per rule. A removal that changes agreed semantics or acceptance criteria
goes to the human as a stand-alone question with a default.
