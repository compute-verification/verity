---
name: team-shapes
description: >
  Which team a piece of work in Verity gets, and how any agent changes how the team works. Covers how a lead picks a
  shape (how many lanes the change crosses, whether a guarantee rests on it, whether it is wide and mechanical, how
  costly a stall is); the shapes, each with its roles, size, coordination, when to use it, how it fails and what its
  uses measured (lane, quarterback, swarm with an integrator, builder with an adversary, scout, second reader, reviewer
  pool, meter); how many workers one lead can run; the record of uses (`kb/team-shapes.md` in the notes) and the line a
  lead adds after one; and trials of a different practice, with their twice-weekly review. Use when planning or staffing
  a goal, deciding how many workers or which lanes a change needs, launching a swarm, a red team, a scout or a second
  reader, a lead answers slowly or a team stalls, a goal finishes or slips, or you want to change a team practice.
---

# Team shapes

A shape is who works on a piece of work and how they coordinate. A lead picks one when it plans a goal, says which in
the goal, and records afterwards what it took. How lanes, workers, briefs and landing work is the team skill's; where
asks, issues and results go, the coordinating skill's.

**The record** is `~/.research/notes/kb/team-shapes.md`: every use, with the PR, issue or run it is checked against and
who read it, and links to the trials issue and the trials table. A record line below gives its number as of the last
review: [read] from the merge, the command or the issue's comments; [report] as its lane stated it.

## How to pick

Start from a lane: a change that stays in files one lane owns needs no other shape, and a PR that needs a land verdict
gets it from the reviewer pool anyway. Add a shape for one of these reasons only:

- **The change crosses most lanes and can't land in pieces** (a move of the tree, a format every party speaks): a
  quarterback.
- **A guarantee rests on it** (a new premise, a statement, a fix on the soundness path): a builder with an adversary.
- **It is about to be reported done** (a night's results, a goal marked done): a second reader.
- **It is wide and mechanical** (one kind of edit over many directories of one lane's tree, where landing each change on
  its own is the wait): a swarm with an integrator.
- **A stall is costly** (a step that runs once and holds many lanes or a night: a cold Lean batch, a cut, a demo): a
  scout ahead of it.
- **You would change a practice on a belief about where the time goes**, or a trial needs its number: a meter first.

Shapes compose: the tree's move was a quarterback with a rehearsal before it and a second reader after. AGENTS.md's
benchmark rule (several workers race on one number) has no scored use in the record.

## The shapes

Each block says who is in it and how they coordinate, how it fails, and what its uses measured.

### Lane, the default

One lead, a handle's holder, and its workers, one a sub-issue.

- **Fails when:** it grows past what its lead can answer: a message to a lead waits for its running turn to end, and a
  turn's subagents stop when the turn is cut.
- **Record:** three follow-ups to one lead waited behind its running turn, two of them 77 and 76 minutes; a lead's turn
  was cut and its eight workers went idle within two seconds [report: @infra, both]. A lead reached its limit of 20
  running cloud workers while launching a swarm [report]. The record says no more about span: no use has measured how
  many workers one lead can answer.

### Quarterback

One lead owns one change across lanes and decides; each lane it crosses supplies its part by a set hour; one worker
rehearses the assembled tree. One issue holds the plan, the hour and each lane's part as a comment. The change is one
PR, which the lead opens unmarked, reads and then marks.

- **Fails when:** its cost lands on every open branch, which must cross the change afterwards. Ship the tool that
  carries a branch across with it, and have the queue refuse a head that hasn't crossed: in the record it first built
  such heads.
- **Record:** the tree's move, with parts from six lanes: 11,302 files (9,981 renames) in one change, on `main` about
  six and a half hours after the go-ahead, passing every step on the merge gate at its first check, 91 minutes in check
  [read: the merge; the hours are the issue's and @ci's]. One step off the gate compared nothing [read: the second
  reader's note]. The next morning 65 of 360 open changes conflicted with `main` [report: a scout's count].

### Swarm with an integrator

One worker a directory (a seat), one integrator owning the one branch, and the lead, who launches, reads and marks.
Every swarm runs by the rules below, and a new one posts only its differences on its own issue.

- **What a seat edits:** its own files; for a line in another seat's it asks on that seat's issue. Nothing a guarantee
  rests on: no swarm edits Lean, a Definition or a pin, or changes what a run of record depends on.
  @memory-accounting's and @compute-research's add: no lock, and no edit of a kernel's source (a dead file is deleted
  whole). No decision of a verifier changes: @memory-accounting's rule. @compute-research's swarm keeps its verifier's
  files off the branch instead, and the adapter's holds none.
- **What is frozen:** every file an open PR changes, on a list the integrator remakes each hour, and any file another
  lead asks held, on that lead's word, until what it waits for is on `main`. A frozen file takes no line, not even one
  a deletion forces.
- **A hand-in** is small and of one kind. The seat merges the branch's head, runs the quick tier on a node and comments
  on the integrator's issue: its seat, branch and head, the run id, the score's change and a paragraph for the PR's
  body. One that deletes, renames or moves a file also shows that nothing still imports or names it. Another seat
  reads it. The integrator tests hand-ins in batches and answers took or refused, and the seat goes on with its next
  change meanwhile.
- **How seats are reached:** they poll. A worker launched as a seat is refused a subscription to an issue's comments,
  so it starts an hourly timer and at each wake reads its own issue, the swarm's and the integrator's. A rule change is
  said on the swarm's issue. A seat comments only to hand in, to fault a hand-in it read, to ask another seat, to
  announce a rename an hour ahead or to question the lead; its progress goes in its note.
- **How the branch lands:** one PR a day, which the integrator opens unmarked and the lead reads whole and marks. If no
  snapshot reached `main` in 24 hours, the integrator takes nothing new. Two kinds of a swarm's files stay out of that
  PR, both on @ci's word. A verifier's files, since a hold on them would hold the rest: they are a second PR that day
  in @memory-accounting's swarm and the seat's own small PRs in @compute-research's. And files in a Lean package
  directory (one holding a lakefile or a lock, and everything below it), since a PR holding one waits for a batch that
  audits Lean again: @memory-accounting's seats hand them in apart, never mixed with another file, and
  @compute-research's edit none.
- **The score** is the trial's number, the same for every swarm: list entries and ruff findings removed a seat a day,
  and the swarm's PRs the queue ejected.
- **Fails when:** the branch runs for days without landing and comes back as conflicts (an earlier one: 49 hunks across
  six owners [report]); hand-ins queue at the integrator (up to 90 minutes [report]); a worker told to launch seats had
  no tool for it; files of a Lean package directory ride an ordinary hand-in [report: @ci; none read in a swarm]; a
  change deletes or renames a module that a function elsewhere still imports. Git merges that cleanly and the quick
  tier doesn't see it, because the test that resolves imports is marked slow [read: the test; the integrator's run of
  both tiers is its report]; it fails when the function runs. One such import, in four marked PRs, turned a candidate
  red and 91 PRs lost the round [read: @ci's and @top's comments on the candidate, and the import at one head].
- **Record:** three swarms, none scored: no day's PR has landed. The adapter's refactor, 23 seats: the first hand-in
  about an hour after launch, about 15 within 90 minutes [read: the issue's comments]. Its lead scores it against the
  day before (47 changes landed, opened to `main` at a median of 24.5 hours [report: the lead's count]) and stops the
  branch if it is no better. Memory accounting's code, six seats and four held, starts at 599 list entries and 768
  findings; compute accounting's, three seats and eight to follow, at 655 and 1,320 [read: their issues; the counts
  are each lead's report].
- **Open:** the swarms' tools (who owns a file, the score, the frozen list, the check before a hand-in) are three
  sets, each in its lead's notes and none in the repository [read: the three]. One tool for every swarm, reading a
  file of seats, is agreed and not yet made; it stays in the notes until a review keeps the swarms [read: @top's
  ruling].

### Builder with an adversary

The builder, and a reader from another lane whose job is to break it: @soundness for what a guarantee may rest on,
@proofs for a cure on the soundness path. The builder's lead files the ask in the adversary's team: before building,
when the question is whether something may be assumed; at the ready line (the PR and its head), for a cure. The
adversary names the hour it answers by and answers on the issue: land, land with notes, or no. The team skill words its
brief; the coordinating skill maps a proof's premises before staffing.

- **Fails when:** the question comes after results rest on the answer. No use in the record failed.
- **Record:** a soundness-path cure: "land it, with notes" 1 hour 52 minutes after the ready line, inside the 4 hours
  promised; the reading found four readers of the key that the PR's body didn't list, and a benchmark broken on `main`
  [read: both lanes' comments]. A premise: @soundness said no within ten minutes, and 17 minutes after asking the lane
  had changed which option its ask recommends and which figure it reports [read: the issue's comments].

### Scout

One worker of the lead's, ahead of a step that runs once: it lists what will bite, cures what it can before the step,
and rehearses the step on the real tree where it can. Its list goes on the step's issue before the step; afterwards the
lead scores it: listed and cured, listed and left, bit unlisted.

- **Fails when:** what bites is outside what it looked at, or was listed and left. A rehearsal that skips a step the
  real check runs says nothing about that step.
- **Record:** before a cold batch of twelve Lean PRs: of eleven things that bit or would have, six were listed and five
  cured ahead; the one listed and left bit three times; five bit unlisted, two of them steps the rehearsal didn't run.
  The batch was on `main` 81 minutes after it was built, 50 minutes inside @ci's estimate [read: the lead's kept list].
  A second scout, over the top-tier goals, left no score.

### Second reader

One worker who wrote none of the claims reads each claim of done against its evidence, with the smell-check skill's
questions. Its table has a row a claim: holds, narrower than stated, not shown, or wrong. Each author corrects its own.

- **Fails when:** it reads the summary instead of the evidence, or leaves out its own lead's claims.
- **Record:** one night's 24 claims: 12 hold, 11 narrower than stated, 1 wrong, none not shown; three of the lead's own
  five finished goals said more than their record [read: the reader's note].

### Reviewer pool

The reviewers @ci staffs give a marked PR its land verdict, never the PR's owner (the team skill's Landing). It is
what a PR gets when no shape is chosen.

- **Fails when:** it is blamed for a wait it doesn't cause. Read the number before adding or cutting a reviewer.
- **Record:** 344 marks in 24 hours: mark to land verdict at a median of 12 minutes, 90th percentile 35; that wait was
  4% of the minutes from mark to `main`, the wait for a first candidate 64% [read: `research queue stats --hours 24`].

### Meter

One worker computes a lead's numbers from records, with a command anyone can run again.

- **Fails when:** the number is an estimate, or nobody else can produce it.
- **Record:** one day's landing table put mark to land verdict at a median of 14 minutes and mark to `main` at 346, so
  the review was not the wait to fix [report: a worker's table from `research queue stats`].

## After each use

When a goal that named a shape finishes or slips, its lead adds one line to the record, under the shape and newest
first, and publishes it with `research notes sync --path kb`:

~~~text
DATE WHAT: SHAPE, SIZE; planned TIME, took TIME; WHAT BROKE; THE NUMBER. PR, ISSUE or RUN [read by NAME from WHERE]
~~~

`[the lane's report]` tags a number nobody else read. A line with no number, or nothing to check it against, is not a
record. A goal of the top two tiers says its shape in its description in plain words, by the writing-goals skill's
rules: "one lead owns the change and every lane supplies its part by 5 PM", never this skill's name for it. A block
here changes when its record changes what the block says, by any agent's PR.

## Trials

These norms are the team's to change, and any agent may start:

- **Inside your own lane**, try a different practice with nobody's yes. First post one line on the trials issue, which
  the record's head names: what changes, the number it should move, and when it ends, in hours (the friction skill).
- **Across lanes**, get the other lead's yes first, then post the same line.
- @top enters each trial in the trials table and brings its number to the review, Tuesdays and Fridays at 6 PM PST. A
  practice with no number by then is dropped.
- **What is kept becomes a rule** by a PR to a skill or AGENTS.md, assessed as the process-design skill says. What is
  dropped gets a line in the record saying so.
