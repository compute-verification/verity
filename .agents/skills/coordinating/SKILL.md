---
name: coordinating
description: >
  How Verity's lanes plan, track and report work. Linear is the campaign map and the status: Goal labels, projects,
  milestones, one issue per deliverable, one sub-issue per worker. Slack only rings people and carries one conversation
  per ask, and Notion holds what Daniel reads. Covers where each thing goes, project and issue structure, states,
  titles and the issue brief, filing asks (`research msg ask --title --parent`), placing issues (`research msg linear
  place`), comments and project updates and how long each message is, tying workers to issues, and the lead's routine
  with `research msg linear audit`. Use when filing, planning, updating or closing a Linear issue or project, launching
  or briefing a worker, reporting status or a result, answering top's sweep or survey, writing a Slack post, or
  cleaning up a team's board.
---

# Coordinating

Linear holds the work and its status. Anyone who reads Linear should see what each lane is doing, for which goal, by
when and how it's going, without reading Slack. `research msg` is the tool for both Linear and Slack (the using-slack
skill).

## Where each thing goes

| What | Where |
|---|---|
| A deliverable or an ask, with its status, results and PRs | one Linear issue in its owner's team (the registry's `linear`) |
| Goals, projects, milestones and their health | Linear's Projects view, grouped by the Goal label |
| "Act now", a yes/no question, an alert | a Slack post to one handle |
| The conversation about one ask | that ask's Slack thread |
| Daniel's decisions, anything for a person | Notion: the decisions log, the Asks database (the asking-daniel skill), a page |
| Evidence | the evidence store and run ids, linked from the issue |
| A lane's checkpoints and findings | the notes (the team skill's "Outside the tree") |

## The map

- **Goals** are the project label group *Goal*: `A · Verified deployment, end to end`, `B · Accounting bounds on real
  hardware`, `C · Training` and `D · Research velocity`. Daniel adds a goal.
- **A project** is one outcome with one lead and a date. It has:
  - one Goal label;
  - `Lead: @handle` at the start of its summary, with that handle's team as its lead team;
  - a target date, and dated milestones named `ID · date · outcome` (`M3 · Mon 4 PM PDT · composed verdict`);
  - priority Urgent for this week's Goal A work, High for Goal D and Medium for the rest;
  - project updates from its lead (below).
- A stream of three or more issues toward one outcome that fits no project gets its own project. Its lead creates it
  (the Linear connector is fine for projects, milestones and labels), labels it and tells top. top keeps the goals,
  priorities and leads consistent.

## Issues

- **One issue per deliverable**, in its owner's team, in a project and the milestone it serves. An issue in no project
  is in Backlog or Canceled.
- **The title is the deliverable**: a noun phrase or an imperative, whole, with no handle, sender or time. "Ring a lead
  only when it must act: run-end and worker-idle rings", not "@infra top 02:40Z, an ask for…".
- **The description is the brief**, at most 300 words:

~~~markdown
**Goal:** one sentence, naming whose ruling it serves.
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
| In Review | every piece is in a ready PR or waits on a named verdict, and a comment says whose |
| Done | the done-condition holds. The owner closes it with one line and its evidence |
| Canceled | dropped, with a one-line reason |

- **PRs.** A PR names its issue in its body (`Part of CMP-<n>`), which attaches it in Linear. Merging a PR doesn't close
  the issue; the owner does.
- **Sub-issues** split a deliverable into pieces that move separately: one per worker. A parent is In Progress while any
  child is, and Done once every child is and its own done-condition holds.
- **Dependencies** between lanes are Linear's blocking relations (`research msg linear relate`), not prose.
- **Asks.** `research msg ask --as ME --to @OWNER --title 'DELIVERABLE' [--parent ISSUE] --file BRIEF.md` files the
  issue in the owner's team, under the deliverable it serves (so in that project and milestone), and rings the owner.
  The owner dates it and moves it out of Backlog within the hour, or declines it (`research msg close --declined`). A
  question that needs no issue is `research msg send --kind question` in its thread.
- **Write through the app, not the Linear connector.** The connector acts as Daniel, so its comments and state changes
  show as his. Use these commands:
  - `research msg linear comment ISSUE 'TEXT' --as ME`, which rings the team's handle and any names the text starts with;
  - `research msg linear state ISSUE STATE`;
  - `research msg linear place ISSUE PROJECT --milestone M`;
  - `research msg linear edit ISSUE --title T --due D`;
  - `research msg linear relate ISSUE blocks|blocked-by|related|duplicate OTHER`, and `unrelate ISSUE OTHER`;
  - `research msg linear update PROJECT --health onTrack|atRisk|offTrack --body-file F`, a project update;
  - `research msg linear project create NAME --team KEY` and `milestone create|edit PROJECT NAME [--date D]`.

## Workers

- **A worker works on one issue**, usually its own sub-issue, and is named after it (`CMP-<n> DEPLOYSP`). The issue's
  Workers line names it and its `bc-` id.
- **Its brief starts with the issue's URL and done-condition.** It tells the worker to comment each run id, verdict line
  and PR on the issue with `research msg linear comment ISSUE 'TEXT' --worker NAME`.
- **No issue, no worker.** A worker running for no issue is stopped, or gets an issue.

## How long, and where

| Message | Length | Where |
|---|---|---|
| A Slack ask, answer or alert | ≤ 60 words. The first line is the ask or the answer, with one owner and a time | to one handle |
| An announcement | ≤ 150 words: what changes, who acts, by when, and the verify command | `send --kind announce` |
| An issue brief | ≤ 300 words, in the template above | the issue |
| A result or progress comment | ≤ 100 words: run id, verdict, the number against its target, the next step and its date | the issue |
| A closing comment | ≤ 60 words: what holds, its evidence, and what's left (filed as a new issue) | the issue |
| A project update | its health (on track, at risk or off track) and ≤ 5 bullets, ≤ 120 words | the project, daily and at each milestone |

- **No status on Slack.** That rules out "(quiet)" lines, "on it", restated confirmations, relays that ask nobody to
  act, and one notice per PR. Status goes on the issue or in the project update.
- Long content goes in a file, PR, note or page, and the message links it.
- Subscribe only to your inbox thread and to threads where a human may answer (using-slack). Any other subscription to a
  busy thread wakes you for every reply.

## A lead's routine

- **On each wake:** read your inbox, and act on the issue each event names.
- **Every few hours, and before ending a long turn:** run `research msg linear audit --lane ME` and fix what it finds.
  Each line gives the fix: place an orphan, date it, retitle it, close a landed issue, comment on a quiet one, or cancel
  a stale one.
- **Daily and at each milestone:** post a project update on each project you lead.
- **On a launch:** file the sub-issue first, then launch the worker named after it.

top's hourly sweep (`infra/control/top_watch.py`) runs the same audit. It rings a lane with findings once every 12
hours and records the counts in `linear-audit.jsonl`. top reads the campaign's state from Linear, not from Slack.
