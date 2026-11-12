---
name: coordinating
description: >
  How Verity's lanes plan, track and report work. The Goals page is the plan and Daniel's interface; Linear holds the
  work under it: one issue per deliverable, labelled with its goal, one sub-issue per worker. Slack only rings people
  and carries one conversation per ask, and Notion holds Daniel's decisions. Covers where each thing goes, project and
  issue structure, states, titles and the issue brief, filing asks (`research msg ask --title --parent`), placing
  issues (`research msg linear place`), comments and how long each message is, tying workers to issues, and the lead's routine
  with `research msg linear audit`. Use when filing, planning, updating or closing a Linear issue or project, launching
  or briefing a worker, reporting status or a result, answering top's sweep or survey, writing a Slack post, or
  cleaning up a team's board.
---

# Coordinating

The Goals page says what each lane is doing, by when and how it's going; Daniel reads it, and it is kept as current as
the work (the writing-goals skill, `research goals`). Linear holds the work under each goal, so anyone can follow a goal
to its issues and their PRs without reading Slack. `research msg` is the tool for both Linear and Slack (the
using-slack skill).

## Where each thing goes

| What | Where |
|---|---|
| A deliverable or an ask, with its status, results and PRs | one Linear issue in its owner's team (the registry's `linear`) |
| Goals: what will be true, its owner, date, status, where it stands and its next step | the Goals page (the writing-goals skill) |
| "Act now", a yes/no question, an alert | a Slack post to one handle |
| The conversation about one ask | that ask's Slack thread |
| Daniel's decisions, anything for a person | Notion: the decisions log, the Asks database (the asking-daniel skill), a page |
| Evidence | the evidence store and run ids, linked from the issue |
| A lane's checkpoints and findings | the notes (AGENTS.md "Where writing goes") |

## The map

- **Goals** are the Goals page's: one outcome each, with one owner and a date, titled as a short headline in Daniel's
  words; a goal under another is its subgoal (the writing-goals skill). Leads write their own; Daniel sets priorities.
- **An issue carries its goal's label**, from the `goal` label group (one label per goal id, kept in step with the page
  by top), and is due no later than that goal.
- **A Linear project** groups a stream of issues under one goal when that helps its lead: `Lead: @handle` at the start
  of its summary, a target date no later than the goal's, and dated milestones named `ID · date · outcome`. The goal,
  not the project, carries the status Daniel reads. top keeps goals, labels and leads consistent.

## Issues

- **One issue per deliverable**, in its owner's team, labelled with the goal it serves. An issue that serves no goal is
  in Backlog or Canceled. An open issue (not Backlog) is in a project within 48 h of filing, or the audit flags it
  (`orphan`), and one whose milestone's date has passed is `lapsed`: re-date the milestone and comment what moved it, or
  move the issue on.
- **The title is the deliverable**: a noun phrase or an imperative, whole, with no handle, sender or time. "Ring a lead
  only when it must act: run-end and worker-idle rings", not "@infra top 02:40Z, an ask for…".
- **The description is the brief**, at most 300 words:

~~~markdown
**Goal:** the goal it serves (its id on the Goals page), and in one sentence how.
**Done when:** a checkable list.
**Due:** a date (also set as the issue's due date).
**Pointers:** PRs, notes, art: ids, the parent issue.
**Workers:** one line each (name and bc- id), added at launch.
~~~

- **Every issue in Todo, In Progress or In Review has a due date.**
- **States:**

| State | Means |
|---|---|
| Backlog | filed and not taken on. An ask lands here, and moving it out says it was seen |
| Todo | taken on and dated, not started |
| In Progress | someone is working on it now |
| In Review | every piece is in an open PR attached to the issue, ready or waiting on a named verdict, and a comment says whose |
| Done | the done-condition holds: the outcome exists. The owner closes it with one line and its evidence |
| Canceled | dropped, with a one-line reason; a person's ask only with their yes, quoted, and `asker-ok` |

- **PRs.** A PR names its issue (`research queue open --issue KEY` writes `Refs KEY` and links it) and the goal it
  serves: the issue carries the goal's label, which `open` writes as `Goal <id>`; `open` and a ready mark warn, never
  refusing, when the body has neither that line nor `Issue: none: REASON`. A key in a PR's title or body attaches nothing:
  `research queue link N --issue KEY` attaches a PR opened without it. An issue In Review 24 h with no open PR attached is
  `unlinked`. Merging a PR doesn't close the issue; the owner does.
- **Sub-issues** split a deliverable into pieces that move separately: one per worker. A parent is In Progress while any
  child is, and Done once every child is and its own done-condition holds.
- **Dependencies** between lanes are Linear's blocking relations (`research msg linear relate`), not prose.
- **Asks.** `research msg ask --as ME --to @OWNER --title 'DELIVERABLE' [--parent ISSUE] --file BRIEF.md` files the
  issue in the owner's team, under the deliverable it serves (so in that project and milestone), and rings the owner.
  The owner dates it and moves it out of Backlog within the hour, or declines it (`research msg close --declined`). An
  ask in another lane's team that nobody took up (a project and a date) or touched for 48 h is a `stray`: the audit and
  top's sweep only report it, and `research msg linear audit --close-strays --as ME` cancels each with a comment naming
  the rule. A person's ask is never a stray. A question that needs no issue is `research msg send --kind question` in its
  thread.
- **A person's ask is answered by its outcome, never by its issue.** Whoever relays Daniel's ask files it with
  `--asker daniel` (an existing issue: `research msg linear edit ISSUE --label asker:daniel`). Within the hour it is In Progress
  with a comment naming its worker (a `bc-` id, or the lead) and the date its outcome lands; filing, dating,
  planning and starting it answer nothing, and a reply to him names the worker and that date, not the issue. Only he
  parks or cancels it: quote his yes on the issue and add `asker-ok`. When he asks again, comment `asked again: <his
  words>` on the issue, never a new one, and `postmortem:` that day (why it wasn't delivered, and the change that stops
  that recurring). Once its date has moved later twice, each move gets a `slip:` comment: what missed the date and what
  holds the next. The audit finds each break (`parked`, `unstaffed`, `slipped`, `repeated`); `research msg linear audit
  --asks` is the ledger, and a report to Daniel counts delivered, never filed.
- **Write through the app, not the Linear connector.** The connector acts as Daniel, so its comments and state changes
  show as his. Use these commands:
  - `research msg linear comment ISSUE 'TEXT' --as ME`, which rings the team's handle and any names the text starts with;
  - `research msg linear state ISSUE STATE`;
  - `research msg linear place ISSUE PROJECT --milestone M`;
  - `research msg linear edit ISSUE --title T --due D`;
  - `research msg linear relate ISSUE blocks|blocked-by|related|duplicate OTHER`, and `unrelate ISSUE OTHER`;
  - `research msg linear update PROJECT --health onTrack|atRisk|offTrack --body-file F`, a project update;
  - `research msg linear project create NAME --team KEY`, `project edit PROJECT [--team KEY] [--date D] [--status S]`
    (a team its issues need, a target date that hasn't passed, Backlog for a paused project) and
    `milestone create|edit PROJECT NAME [--date D]`.

## Workers

- **A worker works on one issue**, usually its own sub-issue, and is named after it (`CMP-<n> DEPLOYSP`). The issue's
  Workers line names it and its `bc-` id.
- **Its brief starts with the issue's URL and done-condition.** It tells the worker to comment each run id, verdict line
  and PR on the issue with `research msg linear comment ISSUE 'TEXT' --worker NAME`. On another lane's issue it
  signs as its own lane with `--worker LANE/NAME`.
- **No issue, no worker.** A worker running for no issue is stopped, or gets an issue.
- **It ends its turn done or blocked, never on a plan.** The brief says so. A worker whose last words are "I'll do X"
  has stopped, so resume it with X. An idle worker reads no inbox message: resume it with the message instead.
- **Map the premises before staffing proofs.** Before staffing proof workers for a guarantee or bound, write one premise
  map: a row per premise, with the code check that discharges it, its Lean Property and its owner. Check each row against
  AGENTS.md's design rules (zero knowledge, no Python premise, one vocabulary, O(descriptor)), take the map to @soundness
  for one ruling, then staff. A ruling that arrives after staffing reworks every piece already staffed.

## How long, and where

| Message | Length | Where |
|---|---|---|
| A Slack ask, answer or alert | ≤ 60 words. The first line is the ask or the answer, with one owner and a time | to one handle |
| An announcement | ≤ 150 words: what changes, who acts, by when, and the verify command | `send --kind announce` |
| An issue brief | ≤ 300 words, in the template above | the issue |
| A result or progress comment | ≤ 100 words: run id, verdict, the number against its target, the next step and its date | the issue |
| A closing comment | ≤ 60 words: what holds, its evidence, and what's left (filed as a new issue) | the issue |

- **No status on Slack.** That rules out "(quiet)" lines, "on it", restated confirmations, relays that ask nobody to
  act, and one notice per PR. Status goes on the goal and the issue.
- Long content goes in a file, PR, note or page, and the message links it.
- Subscribe only to your inbox thread and to threads where a human may answer (using-slack). Any other subscription to a
  busy thread wakes you for every reply.

## A lead's routine

- **On each wake:** read your inbox, and act on the issue each event names.
- **Every few hours, and before ending a long turn:** run `research msg linear audit --lane ME` and fix what it finds.
  Each line gives the fix: place an orphan, date it, retitle it, close a landed issue, attach an unlinked one's PR,
  re-date a lapsed milestone, comment on a quiet one, or cancel a stale one.
- **When a goal you own changes:** update it on the Goals page (the writing-goals skill).
- **On a launch:** file the sub-issue first, then launch the worker named after it.
- **At each checkpoint:** list your workers by state (`batch-fetch-details`) and resume each idle one whose runs have
  ended or that has inbox messages queued. Completion notices can lapse, and the inbox reaches only running agents.
  Judge a running worker by its commits, PRs, runs and notes: a fetched transcript lacks a resumed turn.

top's hourly sweep (`infra/control/top_watch.py`) runs the same audit and only reports. Once every 12 hours it rings a
lane with findings, and a lane with a person's ask off track until it holds (a slip, until its `slip:` comment), and it
records the counts in `linear-audit.jsonl`. Once a day top rings the owner of a goal past its due, for a new date or its
evidence. top reads the campaign's state from the Goals page and Linear, not from Slack.
