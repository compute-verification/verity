"""The agent watch (INF-69): free a cloud agent whose run is held, and record what it sees, from the control pod.

    python3 agent_watch.py loop --log /data/agent-watch/events.jsonl     $CURSOR_API_KEY set; standard library only

An agent hears Slack, Linear and timer events only between runs, and a run can hang after a tool call returns: proofs from
05:29 to 15:33Z and circuits from 09:49Z on 18 Oct, three coordinators at once about 15:26Z on 15 Oct. Until now only Daniel
pressing Stop freed one. Every `POLL_S` the watch lists the agents (`GET /v1/agents`, which shows every agent on the account:
it keeps those whose `repos` name compute-verification/verity; ACTIVE ones, and IDLE ones changed in the last `IDLE_RECENT_S`,
since an ERROR leaves an agent IDLE). For each ACTIVE agent it lists its runs (`GET .../runs`),
page by page back to the newest that started: its `latestRunId` is often a run cancelled on arrival, created and cancelled
within half a second while the run doing the work is older (memory-accounting, 19:32Z 18 Oct; 52 such runs in 50 min by 20:40Z). It follows the event stream (`GET .../runs/{run}/stream`,
server-sent events) of the agent's newest run that is RUNNING or WAITING_FOR_BACKGROUND_WORK, until that run ends or another
takes its place. Events carry no time, so it times them as they arrive. A heartbeat counts as nothing, and so does the `status`
the API re-sends on every reconnection. Each simple event (`assistant`, `thinking`, `tool_call`) has an `interaction_update`
twin, which counts as activity but never as what the agent did last. A cloud subagent's `task` call stays `running` while the
subagent works and the parent goes on (the parent thinks or starts a step while it is open), so such a call counts as
background work, and its progress (`tool-call-delta`) as the subagent's, not the parent's; so does a `task` call, or in a
background wait any call, finishing whose start the watch didn't see (it lay before the stream's replay). A run is

- **held** when its last event is a finished tool call, or the end of a turn while it still reads RUNNING, with no tool call
  of its own open, and the parent has streamed nothing since for `HELD_S` (RUNNING) or `HELD_BG_S` (WAITING_FOR_BACKGROUND_WORK,
  counting its subagents' progress, and longer because cancelling it may take its background workers with it; a wait after its
  turn ended is the normal state, never held). The watch cancels it (`POST .../runs/{run}/cancel`) and starts a new run on the
  agent (`POST /v1/agents/{id}/runs`) that says from when it was held and to read its inbox. A refused new run (409 agent_busy)
  is tried again every sweep for `CONTINUE_RETRY_S`, then left for a person. A cancel refused with 409 is not asked again;
  a run still read as going `REFUSED_S` later is recorded for a person. It cancels at most once per agent every
  `RECANCEL_S`, so a run that hangs again is left for a person;
- **stalled** when it reads RUNNING and has streamed nothing at all, its subagents' progress included, for `STALL_S`, and the
  agent's own `updatedAt` (which moves while it works; a run's doesn't) hasn't moved for as long: a hung tool call or a
  suspended VM, which the API doesn't show. It is cancelled and continued as a held
  run is, under the same once-an-hour limit; a quiet stream alone, while the agent's `updatedAt` moves, is a broken stream,
  not a stall. A followed background wait is stalled the same way after `STUCK` (2 h) with nothing streamed and the agent's
  `updatedAt` still: cancelling a wait may take its local subagents, but a suspended VM has lost them already (compute-accounting's
  research lane sat in one from 05:26 to 23:40Z on 23 Oct, INF-192);
- **held unread** when the agent is any holder a registry handle's `holder` names (its lanes that post as it included,
  `all_holders`), its run RUNNING or waiting on background work, and it has done nothing itself for `UNREAD_S` (1 h) while a
  worker it waits on streamed within `WORKER_LIVE_S`: a resumed worker streaming in the foreground hides it from every rule
  above, and its follow-ups sit unread (compute-accounting's research lane 17:25-21:45Z, network-accounting 11:30-13:36Z, 24
  Oct). For the handle's inbox holder a doorbell must have rung after its last own event an hour ago, or its timer lapsed; a
  lane's subscriptions the watch can't see, so the hour alone. It is cancelled and continued as a held run is (the worker, a
  cloud agent, runs on), owner top, under the same once-an-hour limit (top, 24 Oct 21:50Z);
- **in a long call** when a tool call the parent waits on has been running for `LONG_CALL_S` (recorded, never cancelled: a
  build or a wait may be legitimate, unless it is a holder's held unread);
- **crashed** when the agent's latest run ended in ERROR: the watch starts a new run, at most once every `RESTART_S` and
  `MAX_RESTARTS` times in a row (vllm's runs errored within a minute, run after run, on 18 Oct), then records it as
  **erroring**, for a person. Only a latest run that FINISHED ends the row: a restarted run reads RUNNING for a while before
  it errors again.

A run among an agent's newest left in CREATING for 10 min by the API's own `updatedAt` with no newer run started (only ones
cancelled on arrival) and no run followed (a run queued behind a followed one waits its turn, and that one is judged), or a followed background wait with nothing at all streamed for 2 h while its agent's `updatedAt` moves, is recorded once as **stuck**: on 18 Oct proofs' and circuits' runs sat in CREATING behind a
held one while every later one was cancelled on arrival, network-accounting's behind a goal run from 18:27Z, and the API refuses
to cancel a CREATING run. A goal run (`run-goal-...`) can't be read or streamed on its own (400), so it is never followed; an
agent whose latest run the API won't read is recorded once as unreadable, and again, with `since`, once it has been for
`UNREADABLE_S`.

A coordinator holds a handle of `--registry` (the Slack registry): the holder of the handle's open inbox in `--routes` (the notes'
`routes.d`, which a handover moves first; on 18 Oct the registry still named vllm's predecessor), else the first agent id in the
handle's `holder`. Its inbox thread rings it for each doorbell, and it may run a recurring duty timer, which it declares in
`--timers` (`infra/control/agent_watch_timers.json`, deployed as `/data/agent-watch/timers.json`: `{handle: interval_s}`; comms
has none by design, waking on its inbox, issue events and one-shot timers). It is judged by the run the
sweep judges it by (the run followed, else its newest that started: a doorbell mid-turn leaves a run cancelled on arrival as its
`latestRunId`, vllm at 20:23Z 18 Oct). One whose run's turn is
over (FINISHED, EXPIRED, or WAITING_FOR_BACKGROUND_WORK, where coordinators mostly sit between wakes) is **ready to wake** when
a doorbell for its handle rang after its last activity and `DOORBELL_S` has passed with nothing run (its subscription didn't),
or, for a handle that declares a timer, when nothing has run for twice its interval since then (its timer didn't fire); a
handle that declares none is never woken, or recorded, on idle time alone. Its last activity is the later of the run's `updatedAt`
(read each sweep) and its last stream event: whether a wake starts a new run or resumes a background-waiting one, it moves. The
doorbells come from `doorbells.json` beside `--log` (`{"t", "inboxes": {handle: {"agent", "last"}}}`, each inbox's latest
doorbell as a Slack ts), which infra's relay writes, since only it reads Slack; one older than `BELLS_STALE_S` is not read. The
watch records it at most once per agent an hour, kept in `state.json`, and with `--wake` starts a run that says so; a wake
refused with 409 agent_busy is recorded and counts as the hour's (zombie CREATING runs kept memory-accounting busy for hours),
except that one whose run is WAITING_FOR_BACKGROUND_WORK has that run cancelled and the wake started again: Cursor refuses
any run on an agent waiting on background work, and proofs' wakes were refused so hourly from 10:46 to 16:12Z on 19 Oct.
`--wake` stays off by default.

A cloud worker (a coordinator's Task subagent) is not in `GET /v1/agents`, lists no runs, and can't be started (POST runs:
400 legacy workflow), so only its coordinator can resume it; on 18 Oct all 20 of proofs' workers stopped mid-step 20:06-20:47Z
and nobody noticed until 23:08Z. The watch finds workers in the `origin:` of the notes (`--notes`, a research-notes clone)
under `lanes/` modified in the last `WORKER_NOTES_S`, and in `workers.d/*.json` beside `--log` (`{agent_id, lane,
launched_by}`, for a coordinator that registers its workers), and in the `task` tool calls of every run it follows (the
call's `args.agentId` is the worker launched or resumed: on 19 Oct 16:49-17:03Z 14 of proofs' workers, named only in a
handover note's body, stopped mid-step and none was watched), minus the coordinators and every agent the list has. Each sweep
it reads each one once (`GET /v1/agents/{id}`; `WORKERS_NEW` first reads a sweep; one IDLE for `WORKER_IDLE_S` only when a
note of its changes; a 404 never again), and records **worker_idle** when it went from ACTIVE to IDLE (once per transition,
at its `updatedAt`, which moves on a status change and not during work), unless its parent's Task call completed as it
stopped (`reported_back`: it reported back, INF-277), and **worker_long** once when it has been ACTIVE for
`--worker-long-h` (6). It records only; the relay rings the lane's handle. A worker's worker_idle and idle_after_run are
held with its parent's others (`bursts` in `state.json`) and posted once none has joined for `BURST_QUIET_S`, or `BURST_S`
after the first: one alone as itself, several as one **workers_failed** naming them all. Each names the wake-up the worker
says it set (`subscribe_timer`, "a wake-up in 15 minutes") in its last messages, `GET /v0/agents/{id}/conversation`. That
endpoint is the only conversation the API has, and it carries only user and assistant text: a tool call left without its
result, the shape of those stops, is not visible to the watch, so a worker that stops is judged by its IDLE alone.

An agent that works but says nothing is **silent**: a worker ACTIVE for `SILENT_S` (90 min), or a coordinator for
`COORD_SILENT_S` (60 min, its record's `owner` top, whom the relay rings), with no note of
its (a note whose `origin:` names it, new or changed by its mtime; for a coordinator also any note of its lane, under
`lanes/<handle>/` or with `lane: <handle>`, since it signs with its lane) and no push to its branches. Its branches are the API's
(`git.branches` of its runs, or of a worker's GET), a worker's workers.d `branch`, those its notes' origins name, and every
head ending in its id's last four characters (a coordinator's `cursor/...-1037`); pushes are read every `HEADS_EVERY_S` with
`git ls-remote --heads` of `--git-remote` (the deploy key fixtures_export.sh fetches with), a branch's push time being when the
watch first saw its sha (`heads.json` beside `--log`). The 90 min count from the latest of its ACTIVE spell's start, its newest
note, its newest push and the first heads read, so a fresh deploy judges nobody for 90 min. It is recorded once, and again
after a second window (`SILENT_RINGS`), then not until one of those moves; the relay rings its handle (a coordinator's own, a
worker's launcher's). A silent agent still showing nothing `NO_STEP_S` (6 h) after its last activity is recorded once more,
as **no_step**, rung to @top: on 26 Oct nine workers sat ACTIVE with no step for 30-64 h, each rung twice and then by no one.
Its last activity there also counts its agent's own `updatedAt` and, for a coordinator, its judged run's `updatedAt` and last
stream event, since no_step advises stopping and archiving it: on 26 Oct at 21:05Z it said so of compute-accounting's and
network-accounting's coordinators, each working in Slack within two minutes, with no note or push for 6 h.
Without `--notes` and heads read in the last `HEADS_STALE_S`, nobody is judged.

When a `research run` on a node ends, its sender learns of it only by polling, and a worker that ended its turn to wait never
does. Every `every_s` of `--nodes` (`infra/control/agent_watch_nodes.json`, deployed as `/data/agent-watch/nodes.json`: the
node names, the ssh options with the key, optionally `machines_d`), the watch reads each node's runs that ended since its last
read over one ssh (`RUNS_READER`, host, port, user and runs root from the node's machines.d entry in the notes clone), and
records each once as **run_ended** for whoever sent it (the request's `submitted_by`, else the command's `--by`, else the job's
lane): a coordinator's own handle, or a former one's (`lineage`), a worker's launcher's (its coordinator resumes it with Task:
Cursor refuses a run on a worker), or its parent's when the watch learned it from a Task call, else the lane's (`--by`, the
job's lane, or a handle `submitted_by` names) when that is a handle, else the handle of the branch whose head the run's tree was
(`branch_handle`: the registry's `branch_suffixes`, or `cursor/<handle>-...`). The runs of one read whose sender no handle owns are one
`run_ended` with role `unknown`, each run with its `sender`, for the relay to ring @infra once; a run whose handle
`quiet_handles` lists (ci: drive reads its own train checks' ends) is skipped. `run_ends.json` keeps each node's last read and the runs
seen (`RUN_ENDS_KEEP_S`); runs that ended before the first read are never rung. `agent_watch.py runs` prints what it would read.

The run's end also starts a clock (`awaiting` in `state.json`) on whoever must act on it: the sender when the list has it, a
worker that sent it, else the coordinator of the lane's handle. An agent whose turn is over and that has done nothing since
the run ended is **idle_after_run** `RESUME_S` after it: the watch starts a run on it that carries the run's result (with
the background wait cancelled, as for a wake, when that is what refuses it), and a refused one is tried again each sweep as
a continue run is. A worker can't be started, so its coordinator is rung (the record's `owner`) and, if the worker is still
idle `WORKER_RELAUNCH_S` later, with its task unfinished, it is **relaunch**ed (INF-141; 19 Oct, three that had reported back
were relaunched within a minute): never while its coordinator (its parent, or a holder of its handle) is ACTIVE, never on a
conversation the API won't read, and once per idle at most. Unfinished is a last message that doesn't read as a report back
(`reported`) and either a newest run that failed (a run its own lane cancelled with `research cancel`, its reason or BY naming
the sender, or its BY the handle or an agent of it, is posted quietly and starts no clock at all) or a stop mid-step
(`mid_step`); any other is recorded quietly as **not_relaunched**. The fresh agent (`POST /v1/agents`) is named as a worker
(`LANE worker NAME (relaunch of XXXX)`), told its branches (with none recorded, the `cursor/...` branches its newest note
names), to report in its coordinator's inbox thread with `research msg send --as LANE`, and that the watch wakes it after its
runs, and given its newest note; it is written to workers.d as the lane's worker, so its runs route as a worker's. A sender no
route knows is learned from its brief (`learn`). A coordinator the watch can't resume (erroring, a continue refused for 30 min, a cancel the API refuses) is relaunched
from the newest handover note under `lanes/<handle>/` with `launch_coordinator.py launch --predecessor` at its next
generation, its inbox thread as the pairing thread. Each is relaunched at most once every `RELAUNCH_EVERY_S`. Relaunching
needs a key the GitHub broker trusts (Daniel's user key; a service account's agents can't push), which the watch reads,
only when it relaunches, from `launch.env` beside `--log` (root, 0600, one `CURSOR_API_KEY=` line) and passes only in the
launch's environment; until that file exists every relaunch is recorded as not done. Each failure record names its `owner`
(`top` for a coordinator, the launcher's handle for a worker or an agent a coordinator launched), whose inbox the relay rings.

Each finding is a line of JSON in `--log`; each sweep ends by replacing `status.json` beside it atomically, in the schema
`verity/agent-watch-status/v1` (`snapshot`: its time `t`, `poll_s`, and per agent read its id, the run the watch judges,
statuses, watch state, since and restart count, in categories only), so a reader can tell a quiet watch from a dead one; a
sweep that fails leaves the last one. With `$AGENT_WATCH_PG_URL` set, the same bytes are then inserted, never
overwriting, into Neon's `store_private.agent_watch_status` (`Publisher`, through `psql`; the newest `STATUS_ROWS` kept), for
readers off the pod. `state.json` beside it keeps what must outlive a restart (the watch restarts on every
deploy): when it last cancelled or restarted each agent, and recorded it ready to wake, the restart counts, the continue runs still to start, the runs
already recorded, and since when each coordinator has been ACTIVE and each silent spell's rings. `watch.lock` beside it keeps a second watch from running. The control pod holds no Slack credential, so
infra's relay reads the findings over ssh and posts each in #agent-alerts. Cancelling and restarting don't wait for it.
`--once --dry-run` runs one sweep that follows no stream and sends no POST (each is printed).
"""
from __future__ import annotations

import argparse
import base64
import calendar
import fcntl
import json
import os
import re
import subprocess
import sys
import shlex
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Iterator

API = "https://api.cursor.com"
POLL_S = 60
HELD_S = 15 * 60
HELD_BG_S = 60 * 60         # held while WAITING_FOR_BACKGROUND_WORK: cancelling it may take its local background workers with it
LONG_CALL_S = 45 * 60
STALL_S = 30 * 60           # RUNNING with nothing streamed, subagents' progress included, and the agent's updatedAt still
UNREAD_S = 60 * 60          # a holder that did nothing itself this long while a worker it waits on streamed: held (unread)
WORKER_LIVE_S = 15 * 60     # ... a worker counts as streaming when its progress came this recently
RECANCEL_S = 3600
RESTART_S = 10 * 60         # an ERROR run is restarted at most this often
AGENTS_PAGE = 10            # GET /v1/agents page size, halved on each 503 down to AGENTS_PAGE_MIN for the process's life: the
AGENTS_PAGE_MIN = 2         # largest page Cursor answers keeps shrinking (100 and 50 503'd from 26 Oct 16:25Z, 25 from 27 Oct
#                             ~00:00Z, which blinded the watch for 3.5 h; 10 answers)
SWEEP_READS_S = 12 * 60     # a sweep's reads stop after this long (a sweep takes ~6 min; with reads near their 60 s timeout, one
#                             took 24 min, 26 Oct, and the cloud watch alerts after 20 min without a finished sweep)
RESUME_S = 10 * 60          # idle this long after a research run it must act on ended, with nothing run since: resumed
WORKER_RELAUNCH_S = 20 * 60 # a worker still idle this long after its coordinator was rung for it: relaunched fresh
RELAUNCH_EVERY_S = 6 * 3600 # an agent is relaunched at most once in this long
AWAIT_KEEP_S = 6 * 3600     # a run end nobody acted on this long ago is forgotten
CONTINUE_RETRY_S = 30 * 60  # how long a refused continue run is tried again before it is left for a person
REFUSED_S = 5 * 60          # a run whose cancel got 409 still read as going this long after is left for a person
MAX_RESTARTS = 3            # ERROR runs in a row the watch restarts before leaving the agent for a person
UNREADABLE_S = 3600         # a latest run the API won't read this long is recorded again, for a person
IDLE_RECENT_S = 6 * 3600    # an IDLE agent is read only if it changed this recently (an ERROR leaves an agent IDLE)
RUNS_PAGE = 50              # an ACTIVE agent's runs are read a page at a time, back to the newest that started
RUNS_PAGES = 10             # (memory-accounting had 52 runs cancelled on arrival in 50 min, 18 Oct), at most this many pages
FOLLOW = {"RUNNING": HELD_S, "WAITING_FOR_BACKGROUND_WORK": HELD_BG_S}
TERMINAL = {"FINISHED", "ERROR", "CANCELLED", "EXPIRED"}
# recorded once, never cancelled: a run CREATING this long by its `updatedAt`, a followed background wait with nothing streamed
# while its agent's `updatedAt` moves (with that still too, the wait is stalled: cancelled and continued)
STUCK = {"CREATING": 10 * 60, "WAITING_FOR_BACKGROUND_WORK": 2 * 3600}
GOAL = "run-goal-"              # a goal run's id: the runs endpoint and its stream take only run-<uuid> (400)
SUBAGENT = {"tool-call-delta"}          # interaction_update types that report a subagent's progress, not the parent's
CONTINUE = ("continue: your run was held from {since} (the agent watch saw nothing for {mins} min after {after}, and "
            "cancelled it). Read your inbox and carry on from where you stopped.")
AFTER = {"tool_done": "a finished tool call", "turn_ended": "the end of your turn"}
AFTER_ERROR = ("continue: your last run ended in ERROR (the agent watch restarted you). Read your inbox and carry on from where "
               "you stopped.")
UNREAD = ("continue: your run did nothing itself for {mins} min from {since} while a worker it waits on streamed, so your "
          "follow-ups sat unread ({why}; the agent watch cancelled it). The worker runs on: check it by its id, read your inbox "
          "and carry on from where you stopped, without waiting on a worker in the foreground.")
STALLED = ("continue: your run streamed nothing for {mins} min from {since} while it read {status} (a hung tool call or a "
           "suspended VM; the agent watch cancelled it). Check what that step was doing, read your inbox and carry on from where "
           "you stopped.")
RESUME = ("resume: {runs} ended and nothing has run on you since (the agent watch started this run {mins} min after). Read the "
          "result (`research inspect RUN --on NODE`){inbox} and carry on from where you stopped.")
WORKER_RELAUNCH = (
    "You are a fresh launch of Verity worker {old} ({short}, lane {lane}, a worker of the @{handle} coordinator), which the "
    "agent watch found idle with its task unfinished ({why}) after {runs} ended, with nobody resuming it within {mins} min, so "
    "it launched you in its place (fast failure detection). You are a worker, not a coordinator: you hold no "
    "handle, and @{handle}'s coordinator still owns the lane and your task. Read origin/main's AGENTS.md, its team skill "
    "(`.agents/skills/team/SKILL.md`) and the lane contract first. Report to your coordinator in its inbox thread, your checkpoints, your final report and any question: "
    "`research msg send --as {handle} --thread {inbox} --text '...'`. End your turn while a research run goes on: when it ends, "
    "the agent watch rings your coordinator's inbox if it failed (`research msg ring-run`) and, if you are still idle 10 min later, starts a "
    "run on you with its result. Your branches: {branches}; fetch the one you work on and push only to it. Read your runs' "
    "results with `research inspect RUN --on NODE`. Your predecessor's newest note ({note}) follows; carry on from where it "
    "stopped, and name both {old} and your own id in your notes' origin.\n"
    "--- newest note ---\n{text}")
HANDOVER = (
    "The agent watch relaunched you (fast failure detection): your predecessor {old} {why}, at {at}, and the "
    "watch could not resume it. It is left as it was: once you hold the inbox, ask @top to archive it. Its newest handover "
    "note ({note}) follows.\n\n{text}")
NOTE_CHARS = 30_000             # a note given to a relaunched agent is cut to this many characters
#: a branch a note names, `cursor/...` in backticks: a relaunched worker with no recorded branch is told these, the first first
NOTE_BRANCH_RE = re.compile(r"`(cursor/[A-Za-z0-9._/-]+)`")
ENV_NAME = "Verity compute-verification"     # the one-repo environment every coordinator and worker runs on
MODEL = {"id": "claude-opus-5-5", "params": [{"id": "effort", "value": "high"}]}
COORDINATORS = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/compute-verification/verity/coordinators")  # launch_coordinator's
DOORBELL_S = 10 * 60
BELLS_STALE_S = 30 * 60
# a run whose turn is over; coordinators mostly sit in background work between wakes (12 of 14 at 19:38Z, 18 Oct). CANCELLED is a
# person's stop or the watch's own; ERROR is `restart`'s
IDLE_DONE = {"FINISHED", "EXPIRED", "WAITING_FOR_BACKGROUND_WORK"}
AGENT_RE = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
WAKE = ("wake: your timer or your inbox subscription seems not to have fired ({why}; the agent watch started this run). Read "
        "your doorbell (`research msg read --as {handle}`), run your duties, and set again whichever lapsed.")
SCHEMA = "verity/agent-watch-status/v1"
AGENT_STATUSES = ("ACTIVE", "IDLE", "ARCHIVED", "unknown")
RUN_STATUSES = ("CREATING", "RUNNING", "WAITING_FOR_BACKGROUND_WORK", "FINISHED", "ERROR", "CANCELLED", "EXPIRED", "unreadable",
                "unknown")
WATCH_STATES = ("ok", "followed", "held", "held_again", "continued", "continue_failed", "cancel_refused", "long_call", "stuck",
                "crashed", "erroring", "unreadable", "ready_to_wake", "worker_idle", "worker_long", "silent", "run_ended",
                "stalled", "idle_after_run", "relaunch", "workers_failed", "not_relaunched", "held_unread", "no_step")
BURST_S = 15 * 60               # workers of one parent failing within this of the first are one alert (proofs' 14: 16:49-17:03Z)
BURST_QUIET_S = 3 * 60          # ... posted once none has joined for this long (so a lone failure waits about a sweep)
CONV_MSGS = 6                   # a failed worker's last assistant messages read for a wake-up it set itself
# a wake-up an agent says it set (subscribe_timer's tool call is not in the API's conversation, only what it says of it)
WAKE_RE = re.compile(r"[^.!?\n]*\b(?:wake[- ]?up|subscribe_timer|timer)s?\b[^.!?\n]*[.!?]?", re.I)
WAKE_IN_RE = re.compile(r"\bin (\d+) ?(min(?:ute)?s?|h(?:ours?)?)\b", re.I)
# a worker's last message that reads as a report back (STRAND's "Nothing changed: ...", DSV4LAND's "... or you can file it"):
# it finished its turn on purpose, and is never relaunched; a report runs to several paragraphs, or ends on a question
REPORT_RE = re.compile(r"\b(?:final|report(?:s|ed)?|summary|nothing (?:changed|to do|left|new)|handed (?:over|off|back)|"
                       r"when you resume me|you can|please|over to you|blocked|waiting (?:on|for) (?:you|your))\b", re.IGNORECASE)
# ... and one that stopped mid-step: its last sentence announces a step ("Committing the fix and rerunning the check on cpu-3.",
# "One more direct query of the node:") and no wait it set itself
STEP_RE = re.compile(r"(?:now |next,? |then |first,? |last,? |ok(?:ay)?,? )?(?:let me|let's|i'll|i will|i'm going to|"
                     r"going to|(?!\w*thing\b)\w+ing\b)", re.IGNORECASE)
WAIT_RE = re.compile(r"\b(?:wait(?:s|ing)?|wake[- ]?up|wakes?|timer|resumes?|ring(?:s|-run)?|until)\b", re.IGNORECASE)
LEARN_PER_READ = 4              # unknown senders whose conversation one node read reads for the coordinator their brief names
LEARN_RETRY_S = 6 * 3600        # a sender whose brief named none is read again after this long
WORKER_NOTES_S = 12 * 3600      # a note modified this recently names its author as a worker to watch
WORKER_IDLE_S = 12 * 3600       # a worker IDLE this long is read again only when a note of its changes
#: how far before a worker's IDLE read its parent's Task call may complete and still be its report back
TASK_DONE_SLACK_S = 120
WORKER_LONG_S = 6 * 3600        # ACTIVE this long is recorded once (--worker-long-h)
WORKER_KEEP_S = 2 * 86400       # a worker idle or gone this long, and in no recent note, is forgotten
WORKERS_NEW = 20                # workers read for the first time per sweep: a fresh notes clone names hundreds
SILENT_S = 90 * 60              # ACTIVE with no note of its and no push to its branches this long: rung, again a window later
COORD_SILENT_S = 60 * 60        # ... a coordinator's window, rung to @top (TOP-90 row 37: memory-accounting hung 01:47-04:39Z)
SILENT_RINGS = 2                # rings per silent spell; then none until it shows activity
NO_STEP_S = 6 * 3600            # ... and still nothing this long: one `no_step`, rung to @top (top, 26 Oct: nine workers ACTIVE
                                # with no step for 30-64 h, each rung twice as silent and then by no one)
HEADS_EVERY_S = 5 * 60          # the branch heads (`git ls-remote --heads`) are read this often
HEADS_STALE_S = 3 * HEADS_EVERY_S   # heads older than this tell no push apart from none: nobody is judged silent
RUN_ENDS_EVERY_S = 5 * 60       # the nodes' ended runs are read this often (nodes.json `every_s`)
RUN_ENDS_SLACK_S = 10 * 60      # each node is read back this far past its last read: a run's status lands late, clocks differ
RUN_ENDS_KEEP_S = 3 * 86400     # a run rung (or skipped) this long ago is forgotten
SAY_EVERY_S = 3600              # a failing ls-remote or node read is printed at most this often
BRANCH_RE = re.compile(r"\b(?:cursor|lane)/[A-Za-z0-9._/-]*[A-Za-z0-9]")


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def epoch(s: str | None) -> float | None:
    """The API's ISO time (2026-10-18T09:49:50.225Z) in epoch seconds; None when it doesn't parse."""
    try:
        return calendar.timegm(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, TypeError):
        return None


def read_json(p: Path | None) -> dict:
    try:
        d = json.loads(p.read_text()) if p else {}
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def holders(registry: Path | None, routes: Path | None) -> dict[str, str]:
    """agent id -> handle, for each registry handle: its open inbox's holder (`routes/<handle>.json`, which a handover moves
    first), else the first agent id in its `holder` (later ones are lanes that post as it)."""
    out = {}
    for h in read_json(registry).get("handles") or []:
        r = read_json(routes / f"{h.get('handle')}.json") if routes else {}
        m = AGENT_RE.fullmatch(str(r.get("holder") or "")) if not r.get("closed") else None
        if m := m or AGENT_RE.search(str(h.get("holder", ""))):
            out[m[0]] = h["handle"]
    return out


def all_holders(registry: Path | None, routes: Path | None) -> dict[str, str]:
    """agent id -> handle for every agent id in each registry handle's `holder` (its lanes that post as it included) and each
    open inbox's holder: the agents a held run of which leaves a handle's follow-ups unread (`unread`)."""
    out = {}
    for h in read_json(registry).get("handles") or []:
        for m in AGENT_RE.finditer(str(h.get("holder", ""))):
            out.setdefault(m[0], h["handle"])
    return {**out, **holders(registry, routes)}


def lineage(registry: Path | None, coords: dict[str, str], agent: str) -> str | None:
    """The handle `agent` coordinates, or coordinated before a handover moved its inbox: the watch's coordinators (`holders`),
    else the first agent id in a registry handle's `holder` (proofs' generation 3 after generation 4 took the inbox, 19 Oct),
    else a generation `launch_coordinator.py` made of a registry handle."""
    if agent in coords:
        return coords[agent]
    handles = [h for h in read_json(registry).get("handles") or [] if h.get("handle")]
    return next((h["handle"] for h in handles if (m := AGENT_RE.search(str(h.get("holder", "")))) and m[0] == agent),
                None) or next((h["handle"] for h in handles if generation(h["handle"], agent)), None)


def timers(path: Path | None) -> dict[str, float]:
    """handle -> its recurring duty timer's interval in seconds, from `--timers` (`{"infra": 900}`), read each sweep; a
    handle absent, or a value that isn't a positive number, declares none. A missing file declares none for anyone."""
    return {str(h): float(v) for h, v in read_json(path).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}


def front_matter(p: Path) -> dict[str, str]:
    """A note's YAML front-matter, its `key: value` lines only (all a note's origin and lane need)."""
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except OSError:
        return {}
    if not head.startswith("---\n") or (end := head.find("\n---", 4)) < 0:
        return {}
    out = {}
    for line in head[4:end].splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() and not k.startswith((" ", "-")):
            out.setdefault(k.strip(), v.strip().strip("'\""))
    return out


def recent_notes(notes: Path | None, t: float) -> Iterator[tuple[Path, float, dict[str, str]]]:
    """(path, mtime, front-matter) of each note under `notes/lanes/` modified in the last `WORKER_NOTES_S`. The notes clone is
    synced every 2 min, and git gives a file it adds or changes the time it did so, so a note's mtime is when it arrived."""
    for p in sorted((notes / "lanes").glob("*/*.md")) if notes else []:
        try:
            heard = p.stat().st_mtime
        except OSError:
            continue
        if t - heard < WORKER_NOTES_S:
            yield p, heard, front_matter(p)


def lane_handle(lane: str, names: set[str]) -> str | None:
    """The handle a lane belongs to: the lane itself when it is one, else the longest handle it extends with a `-`
    (`kernels-hashproof` is @kernels'), else None."""
    if lane in names:
        return lane
    return max((h for h in names if h and lane.startswith(h + "-")), key=len, default=None)


def workers(notes: Path | None, workers_d: Path, registry: Path | None, coords: dict[str, str], t: float) -> dict[str, dict]:
    """agent id -> {"lane", "handle", "heard", "branches"}: each cloud worker (a Task subagent, which `GET /v1/agents` doesn't
    list) named by `workers_d/*.json` (`{agent_id, lane, launched_by, branch}`, which a coordinator writes for the workers it
    launches) or by the `origin:` of a note under `notes/lanes/` modified in the last `WORKER_NOTES_S`, minus the coordinators
    (`coords`). Its handle is its launcher's (`launched_by`), else its lane's when the lane is a handle, else that of a
    coordinator its note's origin names (`started by proofs bc-...`); `heard` is the newest such file's mtime; `branches` the
    `branch` its workers.d file names; `parent` its launcher (`launched_by`, or that coordinator), if known."""
    names = {str(h.get("handle")) for h in read_json(registry).get("handles") or []}
    out: dict[str, dict] = {}

    def add(agent: str, lane: str, handle: str | None, heard: float, branches: tuple[str, ...] = (),
            parent: str | None = None) -> None:
        w = out.setdefault(agent, {"lane": lane, "handle": handle, "heard": heard, "branches": [], "parent": parent})
        w["lane"], w["handle"], w["heard"] = w["lane"] or lane, w["handle"] or handle, max(w["heard"], heard)
        w["branches"], w["parent"] = sorted({*w["branches"], *branches}), w["parent"] or parent

    for p in sorted(workers_d.glob("*.json")) if workers_d.is_dir() else []:
        d = read_json(p)
        if (m := AGENT_RE.fullmatch(str(d.get("agent_id") or ""))) and m[0] not in coords:
            lane, by = str(d.get("lane") or ""), AGENT_RE.search(str(d.get("launched_by") or ""))
            handle = (lineage(registry, coords, by[0]) if by else None) or lane_handle(lane, names)
            add(m[0], lane, handle, p.stat().st_mtime, (str(d["branch"]),) if d.get("branch") else (), by[0] if by else None)
    for p, heard, fm in recent_notes(notes, t):
        ids, lane = AGENT_RE.findall(fm.get("origin", "")), fm.get("lane") or p.parent.name
        launcher = next((i for i in ids if i in coords), None)
        for agent in ids:
            if agent not in coords:
                add(agent, lane, lane_handle(lane, names) or coords.get(launcher or ""), heard, parent=launcher)
    return out


def note_times(notes: Path | None, t: float) -> tuple[dict[str, float], dict[str, set[str]], dict[str, float]]:
    """(agent id -> its newest note's mtime, agent id -> the branches its notes' origins name, lane -> its newest note's mtime),
    over the notes `recent_notes` gives: a note is an agent's when its `origin:` names it (`ZKS10, bc-...; code
    cursor/zk-frame-c7a9`), and its lane's by its directory under `lanes/` and its `lane:`."""
    heard: dict[str, float] = {}
    named: dict[str, set[str]] = {}
    lanes: dict[str, float] = {}
    for p, mtime, fm in recent_notes(notes, t):
        origin = fm.get("origin", "")
        for agent in AGENT_RE.findall(origin):
            heard[agent] = max(heard.get(agent, mtime), mtime)
            named.setdefault(agent, set()).update(BRANCH_RE.findall(origin))
        for lane in {p.parent.name, fm.get("lane", "")} - {""}:
            lanes[lane] = max(lanes.get(lane, mtime), mtime)
    return heard, named, lanes


def pushed_branches(rec: dict) -> set[str]:
    """The Verity branches an API record names: a run's or an agent's `git.branches[]` (`{repoUrl, branch}`, the agent's pushed
    branches, the same on every run of it), and its `repos[].startingRef` when it works on that branch (`workOnCurrentBranch`)."""
    out = {str(b["branch"]) for b in (rec.get("git") or {}).get("branches") or []
           if isinstance(b, dict) and b.get("branch") and VERITY_REPO.search(str(b.get("repoUrl") or ""))}
    if rec.get("workOnCurrentBranch"):
        out |= {str(r["startingRef"]) for r in rec.get("repos") or [] if isinstance(r, dict) and r.get("startingRef")
                and VERITY_REPO.search(str(r.get("url") or ""))}
    return out


def run_cmd(argv: list[str], stdin: str | None, timeout: float) -> tuple[int, str, str]:
    """(rc, stdout, stderr) of a command; a command that can't start or runs past `timeout` is rc 255."""
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 255, "", f"{type(e).__name__}: {e}"
    return r.returncode, r.stdout, r.stderr


def duration(s: object) -> str:
    """A run's wall time as a person reads it: `40 s`, `12 min`, `2 h 05 min`; `?` when unknown."""
    if not isinstance(s, (int, float)) or isinstance(s, bool) or s < 0:
        return "?"
    if s < 60:
        return f"{s:.0f} s"
    m = int(s // 60)
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def ended_runs(runs: list[dict]) -> str:
    """The research runs an agent was resumed or relaunched for, as its prompt and the relay name them."""
    return "; ".join(f"run {x['run']} ({x.get('what') or '?'}) on {x.get('node')}: {x.get('state')} rc={x.get('rc')} after "
                     f"{x.get('after')} at {iso(x['ended'])}" for x in runs)


def reported(text: str) -> bool:
    """Whether an agent's last message reads as a report back: several paragraphs, a question, `DONE`, or report words
    (`REPORT_RE`)."""
    s = text.strip()
    return bool(s) and ("\n" in s or s.endswith("?") or bool(re.search(r"\bDONE\b", s) or REPORT_RE.search(s)))


def mid_step(text: str) -> bool:
    """Whether an agent's last message stops mid-step: it ends on a colon, or its last sentence announces a step
    (`STEP_RE`), and it names no wait it set itself (`WAIT_RE`)."""
    s = text.strip()
    last = re.split(r"(?<=[.!?])\s+", s)[-1] if s else ""
    return bool(s) and not WAIT_RE.search(s) and (s.endswith(":") or bool(STEP_RE.match(last)))


def generation(handle: str, agent: str) -> int | None:
    """The generation `launch_coordinator.py` gave `agent` as `handle`'s coordinator (its id is uuid5 of `handle/N`)."""
    return next((g for g in range(1, 100) if f"bc-{uuid.uuid5(COORDINATORS, f'{handle}/{g}')}" == agent), None)


def run_env(argv: list[str], env: dict[str, str], timeout: float = 300) -> tuple[int, str, str]:
    """`run_cmd` with `env` added to the watch's environment: the launch key reaches only the launch, never a command line."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env={**os.environ, **env})
    except (OSError, subprocess.TimeoutExpired) as e:
        return 255, "", f"{type(e).__name__}: {e}"
    return r.returncode, r.stdout, r.stderr


def note_text(p: Path | None) -> str:
    try:
        return p.read_text(errors="replace")[:NOTE_CHARS] if p else ""
    except OSError:
        return ""


def handover_note(notes: Path | None, handle: str) -> Path | None:
    """The newest note under `lanes/<handle>/` whose name says handover or handoff (not a reply, an ack or an order), else
    its newest note; notes are named by their UTC time, so the newest sorts last."""
    found = sorted((notes / "lanes" / handle).glob("*.md")) if notes else []
    hand = [p for p in found if re.search(r"hand(?:over|off)", p.name) and not re.search(r"-(?:reply|ack|order)-", p.name)]
    return (hand or found or [None])[-1]


# Run on a node over ssh (`python3 - ROOT SINCE`), standard library only: each run under ROOT/runs whose status.json changed at
# or after SINCE (epoch s) and reads a terminal state, with who sent it (its request's `submitted_by`, else the `--by` its command
# passes), what it was, and, for one that was cancelled or SIGTERMed, the newest `research cancel` of it in its events.jsonl's last
# MiB (`cancel_reason`, and `cancelled_by`, the BY of its "(research cancel by BY)"), as one JSON list.
RUNS_READER = r'''
import json, os, re, sys
root, since = sys.argv[1], float(sys.argv[2])
def load(p):
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}
def cancel_of(d):
    try:
        with open(os.path.join(d, "events.jsonl"), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 1048576))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return "", ""
    for line in reversed(tail.splitlines()):
        try:
            e = json.loads(line) if '"cancel_intent"' in line else None
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("kind") == "cancel_intent" and e.get("actor") == "manual":
            why = str(e.get("reason") or "")
            m = re.search(r"\(research cancel by ([^)]*)\)\s*$", why)
            return (m.group(1).strip() if m else ""), why[:300]
    return "", ""
out, runs = [], os.path.join(root, "runs")
try:
    names = sorted(os.listdir(runs))
except OSError:
    names = []
for n in names:
    p = os.path.join(runs, n, "status.json")
    try:
        mtime = os.stat(p).st_mtime
    except OSError:
        continue
    if not n.startswith("r") or mtime < since:
        continue
    st = load(p)
    if st.get("state") not in ("done", "failed", "cancelled"):
        continue
    tr = st.get("transitions") or [{}]
    job, req = load(os.path.join(runs, n, "job.json")), load(os.path.join(root, "requests", n, "request.json"))
    cmd = [str(a) for a in (req.get("command") or (job.get("execution") or {}).get("command") or [])]
    own, comp = job.get("ownership") or {}, job.get("computation") or {}
    by, why = cancel_of(os.path.join(runs, n)) if st.get("state") == "cancelled" or st.get("rc") == 143 else ("", "")
    out.append({"run": n, "state": st.get("state"), "rc": st.get("rc"), "wall_s": st.get("wall_s"),
                "ended": tr[-1].get("t_utc"), "mtime": mtime, "submitted_by": req.get("submitted_by") or "",
                "by": cmd[cmd.index("--by") + 1] if "--by" in cmd[:-1] else "", "lane": own.get("lane") or "",
                "job": own.get("job") or "", "stage": comp.get("stage") or "", "command": " ".join(cmd)[:200],
                "cancelled_by": by, "cancel_reason": why, "tree": comp.get("tree_sha") or ""})
print(json.dumps(out))
'''


#: how long `read_heads` remembers which branch a pushed sha was the head of (`Watch.branch_of`)
HEADS_SHA_KEEP_S = 2 * 86400


def branch_handle(branch: str, registry: Path | None, handles: set[str]) -> str | None:
    """The handle a branch belongs to, for a run whose sender no route knows (TOP-90 row 21, 25 Oct: seven workers traced by
    hand from tree to branch to lane): the registry's `branch_suffixes` of a handle naming the branch's last `-` part, else a
    handle the branch's name starts with after `cursor/` (`cursor/kernels-kp2s8-8c40`), the longest such."""
    if not branch:
        return None
    suffix = branch.rsplit("-", 1)[-1]
    for h in read_json(registry).get("handles") or [] if registry else []:
        if suffix in (h.get("branch_suffixes") or []) and str(h.get("handle")) in handles:
            return str(h["handle"])
    name = branch.split("/", 1)[-1]
    return max((x for x in handles if name.startswith(x + "-")), key=len, default=None)


def machine(machines_d: Path, name: str) -> dict | None:
    """A node's `machines.d/<name>.toml` (`research pods register --host`): host, port, user, root and python, with research's
    defaults; None when it is missing or names no host."""
    try:
        m = tomllib.loads((machines_d / f"{name}.toml").read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    if not isinstance(m.get("host"), str) or not m["host"]:
        return None
    return {"port": 22, "user": "root", "root": "/workspace/research", "python": "python3", **m}


def node_runs(cfg: dict, notes: Path | None, node: str, since: float,
              sh: Callable[[list[str], str | None, float], tuple[int, str, str]]) -> tuple[list[dict] | None, str]:
    """(the node's runs `RUNS_READER` lists from `since`, "") over one ssh (`cfg["ssh"]`, then `-p PORT user@host`), or (None,
    why) when the node has no machines.d entry (`cfg["machines_d"]`, else the notes clone's) or doesn't answer."""
    md = Path(cfg["machines_d"]) if cfg.get("machines_d") else notes / "machines.d" if notes else None
    if (m := machine(md, node) if md else None) is None:
        return None, f"no host for {node} in {md}/{node}.toml"
    ssh = [str(a) for a in cfg.get("ssh") or ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]]
    argv = [*ssh, "-p", str(m["port"]), f"{m['user']}@{m['host']}",
            shlex.join([str(m["python"]), "-", str(m["root"]), f"{since:.0f}"])]
    rc, out, err = sh(argv, RUNS_READER, 120)
    try:
        rows = json.loads(out) if rc == 0 else None
    except ValueError:
        rows = None
    if not isinstance(rows, list):
        return None, f"{node} unreadable (ssh exited {rc}: {err.strip()[-200:]})"
    return rows, ""


def print_runs(nodes: Path, notes: Path | None, hours: float) -> int:
    """`agent_watch.py runs`: each node's runs that ended in the last `hours`, one JSON line each, as the watch would read them."""
    cfg, rc = read_json(nodes), 0
    for node in cfg.get("nodes") or []:
        rows, why = node_runs(cfg, notes, str(node), time.time() - hours * 3600, run_cmd)
        if rows is None:
            print(why, file=sys.stderr)
            rc = 1
        for row in rows or []:
            print(json.dumps({"node": node, **row}, sort_keys=True))
    return rc


def doorbells(path: Path, t: float) -> tuple[dict[str, float], str]:
    """handle -> its latest doorbell (a Slack ts), from the relay's file, and whether that was fresh, stale or missing."""
    d = read_json(path)
    if not d:
        return {}, "missing"
    if t - (epoch(d.get("t")) or 0) >= BELLS_STALE_S:
        return {}, "stale"
    out = {}
    for handle, i in (d.get("inboxes") or {}).items():
        try:
            out[str(handle)] = float(i["last"])
        except (KeyError, TypeError, ValueError):
            continue
    return out, "fresh"


VERITY_REPO = re.compile(r"github\.com[/:]compute-verification/verity(?:\.git)?/?$")


def verity(a: dict) -> bool:
    """Whether an agent's `repos` (`[{"url": "https://github.com/compute-verification/verity"}]`) name Verity's repository."""
    return any(VERITY_REPO.search(str(r.get("url") if isinstance(r, dict) else r)) for r in a.get("repos") or [])


def started(x: dict) -> bool:
    """Whether a run record ever ran: a run cancelled on arrival reads CANCELLED with no `durationMs`."""
    return x.get("status") not in ("CREATING", "CANCELLED") or bool(x.get("durationMs"))


class Api:
    """Cursor's Cloud Agents API, v1. Past `reads_until` (by `clock`), a GET raises TimeoutError without asking; POSTs, the
    watch's actions, always go."""

    def __init__(self, key: str, base: str = API, timeout: float = 60, clock: Callable[[], float] = time.monotonic):
        self.auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
        self.base, self.timeout, self.clock = base, timeout, clock
        self.reads_until: float | None = None
        self.page = AGENTS_PAGE

    def _req(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Authorization": self.auth, "Accept": "application/json", **(headers or {})}
        if data is not None:
            h["Content-Type"] = "application/json"
        return urllib.request.Request(self.base + path, data=data, method=method, headers=h)

    def get(self, path: str, **params) -> dict:
        q = ("?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})) if params else ""
        timeout = self.timeout
        if self.reads_until is not None:
            if (timeout := min(timeout, self.reads_until - self.clock())) <= 0:
                raise TimeoutError(f"this sweep's reads are spent ({SWEEP_READS_S // 60} min); the next sweep reads again")
        with urllib.request.urlopen(self._req("GET", path + q), timeout=timeout) as r:
            return json.load(r)

    def post(self, path: str, body: dict | None = None) -> dict:
        with urllib.request.urlopen(self._req("POST", path, body if body is not None else {}), timeout=self.timeout) as r:
            return json.load(r)

    def agents(self) -> Iterator[dict]:
        cursor = None
        while True:
            try:
                page = self.get("/v1/agents", limit=self.page, cursor=cursor, includeArchived="false")
            except urllib.error.HTTPError as e:
                if e.code != 503 or self.page <= AGENTS_PAGE_MIN:
                    raise
                self.page = max(AGENTS_PAGE_MIN, self.page // 2)
                print(f"{iso(time.time())} GET /v1/agents 503: page size now {self.page}", file=sys.stderr, flush=True)
                continue
            yield from page.get("items") or []
            cursor = page.get("nextCursor")
            if not cursor:
                return

    def stream(self, agent: str, run: str, stop: Callable[[], bool] = lambda: False) -> Iterator[tuple[str, dict]]:
        """(event, data) from the run's stream, until it ends or `stop()`; resumes from the last event id when the connection
        drops. Returns on 400, 404 or 410 (410 stream_expired past the retention window)."""
        last_id = None
        while not stop():
            headers = {"Accept": "text/event-stream", **({"Last-Event-ID": last_id} if last_id else {})}
            try:
                with urllib.request.urlopen(self._req("GET", f"/v1/agents/{agent}/runs/{run}/stream", headers=headers),
                                            timeout=10 * 60) as r:
                    event, data = "message", []
                    for raw in r:
                        line = raw.decode("utf-8", "replace").rstrip("\r\n")
                        if not line:
                            if data:
                                try:
                                    payload = json.loads("\n".join(data))
                                except ValueError:
                                    payload = {"text": "\n".join(data)}
                                yield event, payload if isinstance(payload, dict) else {"value": payload}
                                if event == "done":
                                    return
                            event, data = "message", []
                        elif line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif line.startswith("id:"):
                            last_id = line[3:].strip()
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 410):
                    return
                raise
            except (TimeoutError, OSError):
                continue


@dataclass
class Run:
    agent: str
    run: str
    name: str = ""
    status: str = "RUNNING"
    last_kind: str = "start"            # start | text | call | tool_done | turn_ended
    last_t: float = 0.0                 # the parent's last event
    bg_t: float = 0.0                   # its subagents' last progress
    agent_t: float = 0.0                # its agent's `updatedAt` as the last sweep listed it
    calls: dict[str, float] = field(default_factory=dict)   # open tool calls the parent waits on: callId -> started
    bg: set[str] = field(default_factory=set)               # open calls it went on past: background work
    told: set[str] = field(default_factory=set)
    ended: str | None = None
    streaming: bool = False
    refused_t: float | None = None                  # when its cancel got 409


def lock(path: Path) -> IO | None:
    """An exclusive lock on `path`, held while the returned file stays open; None when another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


class Watch:
    def __init__(self, api, log: Path, clock: Callable[[], float] = time.time, registry: Path | None = None,
                 routes: Path | None = None, wake: bool = False, timers: Path | None = None,
                 publish: Callable[[bytes, float], None] | None = None,
                 notes: Path | None = None, worker_long_s: float = WORKER_LONG_S, git_remote: str | None = None,
                 nodes: Path | None = None, sh: Callable[[list[str], str | None, float], tuple[int, str, str]] = run_cmd,
                 launcher: Path | None = None, launch: Callable[[list[str], dict[str, str]], tuple[int, str, str]] = run_env,
                 launch_api: Callable[[str], object] = lambda key: Api(key)):
        self.api, self.log, self.clock, self.wake, self.publish = api, log, clock, wake, publish
        self.notes, self.worker_long_s = notes, worker_long_s
        self.git_remote, self.nodes, self.sh = git_remote, nodes, sh
        self.launch_env, self.launcher, self.launch, self.launch_api = log.with_name("launch.env"), launcher, launch, launch_api
        self.coords: dict[str, str] = {}            # coordinator -> its handle, as this sweep read them
        self.holders_all: dict[str, str] = {}       # every agent a registry holder names (its lanes too) -> the handle
        self.heads_path, self.ends_path = log.with_name("heads.json"), log.with_name("run_ends.json")
        self.heads = read_json(self.heads_path)     # {"t0", "t", "heads": {branch: [sha, when the watch first saw it]}}
        self.ends = read_json(self.ends_path)       # {"since", "nodes": {node: last read}, "seen": {node/run: when}}
        self.heads_read = self.ends_read = -float("inf")
        self.said: dict[str, float] = {}            # what failed -> when it was last printed
        self.names: dict[str, str] = {}             # listed agent -> its name, for run_ended
        self.branches: dict[str, set[str]] = {}     # listed agent -> the branches its run records name
        self.worker_status: dict[str, str] = {}             # worker -> its status as this sweep read it, for status.json
        self.agent_status: dict[str, str] = {}              # listed agent -> its status as this sweep read it, for run_ended
        self.registry, self.routes, self.timers = registry, routes, timers
        self.doorbells = ""                         # whether doorbells.json was fresh, stale or missing at the last sweep
        self.runs: dict[str, Run] = {}
        self.states: dict[str, tuple[str, str, str]] = {}   # agent -> (run, its last finding's kind, since), for status.json
        self.status_of: dict[str, tuple[str, str]] = {}     # agent -> (run, its status when the watch last read it)
        self.lock = threading.Lock()
        self.kids_lock = threading.Lock()           # `children`, which streams add to and `save` reads, under `lock` or not
        self.state_path = log.with_name("state.json")
        try:
            s = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            s = {}
        self.freed: dict[str, float] = s.get("freed", {})         # agent -> when the watch last cancelled or restarted it
        self.errors: dict[str, int] = s.get("errors", {})         # agent -> ERROR runs in a row the watch restarted
        self.left: set[str] = set(s.get("left", []))              # agents past MAX_RESTARTS, recorded once
        self.pending: dict[str, dict] = s.get("pending", {})      # agent -> a continue run the API refused, tried again
        self.done: dict[str, float] = s.get("done", {})           # runs judged once (ERROR, stuck, unreadable) -> when
        self.woken: dict[str, float] = s.get("woken", {})         # agent -> when the watch last recorded it ready to wake
        self.workers: dict[str, dict] = s.get("workers", {})      # worker -> lane, handle, status, updated, since, read, ...
        self.active: dict[str, float] = s.get("active", {})       # coordinator -> since when the watch has seen it ACTIVE
        self.silent: dict[str, dict] = s.get("silent", {})        # agent -> {"from": its last activity, "rings"} while silent
        self.awaiting: dict[str, dict] = s.get("awaiting", {})    # agent -> the research runs that ended for it to act on
        self.relaunched: dict[str, float] = s.get("relaunched", {})   # agent -> when the watch last tried to relaunch it
        self.owners: dict[str, str] = s.get("owners", {})         # listed agent a coordinator launched -> that coordinator's handle
        self.children: dict[str, dict] = s.get("children", {})    # worker a followed run's Task call named -> parent, name, t
        self.bursts: dict[str, dict] = s.get("bursts", {})        # parent -> its workers' failures held to post as one
        self.final: dict[str, dict] = {}                          # run -> its terminal record, never read again (not saved)
        self.learn_tried: dict[str, float] = {}                   # sender whose brief named no coordinator -> when (not saved)
        self.learn_left = 0                                       # briefs the current node read may still read

    def save(self) -> None:
        t = self.clock()
        self.done = {k: v for k, v in self.done.items() if v >= t - 2 * 86400}
        self.woken = {k: v for k, v in self.woken.items() if v >= t - RECANCEL_S}
        self.relaunched = {k: v for k, v in self.relaunched.items() if v >= t - 2 * 86400}
        with self.kids_lock:
            self.children = {k: v for k, v in self.children.items() if v["t"] >= t - WORKER_KEEP_S}
            children = dict(self.children)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"freed": self.freed, "errors": self.errors, "left": sorted(self.left),
                                   "pending": self.pending, "done": self.done, "woken": self.woken,
                                   "workers": self.workers, "active": self.active, "silent": self.silent,
                                   "awaiting": self.awaiting, "relaunched": self.relaunched, "owners": self.owners,
                                   "children": children, "bursts": self.bursts}) + "\n")
        tmp.replace(self.state_path)

    def owned(self, agent: str) -> dict[str, str]:
        """`{"owner": handle}` for a failure record: whose inbox the relay rings. A coordinator's owner is top; an agent a
        coordinator launched (a worker, or one the list has) is that coordinator's handle's."""
        owner = ("top" if agent in self.coords or agent in self.holders_all else
                 self.owners.get(agent) or (self.workers.get(agent) or {}).get("handle"))
        return {"owner": owner} if owner else {}

    def launch_key(self) -> str | None:
        """The key a relaunch creates agents with, from `launch.env` (one `CURSOR_API_KEY=` line); None while it is absent."""
        try:
            lines = self.launch_env.read_text().splitlines()
        except OSError:
            return None
        return next((v.strip().strip("\"'") for k, _, v in (ln.partition("=") for ln in lines)
                     if k.strip().removeprefix("export ").strip() == "CURSOR_API_KEY" and v.strip()), None)

    def say(self, what: str, msg: str, t: float) -> None:
        """Print a failure, at most once every `SAY_EVERY_S` for the same `what`."""
        if t - self.said.get(what, -SAY_EVERY_S) >= SAY_EVERY_S:
            self.said[what] = t
            print(f"{iso(t)} {msg} (said at most once an hour)", file=sys.stderr, flush=True)

    def record(self, kind: str, r: Run, *, judged: bool = True, **detail) -> None:
        """A finding as a line of `--log`; one about a run the watch judges the agent by (not a `research run`'s end) is also
        its status row's state."""
        rec = {"t": iso(self.clock()), "kind": kind, "agent": r.agent, "name": r.name, "run": r.run, **detail}
        if judged:
            self.states[r.agent] = (r.run, kind, str(detail.get("since") or rec["t"]))
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def once(self, run: str, t: float) -> bool:
        """True the first time `run` is judged (an ERROR restart, a stuck or unreadable record), across restarts."""
        if run in self.done:
            return False
        self.done[run] = t
        self.save()
        return True

    def child(self, parent: str, data: dict, t: float) -> None:
        """A followed run's Task call names its worker (`args.agentId` when it launches or resumes one; the result's
        `cloudAgentBcId` when the stream has it): a worker of `parent`, watched like one a note names."""
        args, res = data.get("args") or {}, data.get("result")
        m = AGENT_RE.fullmatch(str(args.get("agentId") or "") if isinstance(args, dict) else "") or AGENT_RE.search(
            json.dumps(res) if isinstance(res, (dict, list)) else "")
        if m and m[0] != parent:
            with self.kids_lock:
                c = self.children.setdefault(m[0], {"parent": parent, "name": ""})
                c.update(parent=parent, t=t, name=str(args.get("description") or c["name"])[:120] if isinstance(args, dict)
                         else c["name"])
                if data.get("status") == "completed":
                    c["done"] = t                   # the parent has the worker's result: it knows the worker stopped
                else:
                    c.pop("done", None)

    def reported_back(self, agent: str, idle_t: float) -> bool:
        """Whether `agent`'s parent got its Task result as it stopped (`TASK_DONE_SLACK_S` either side of its IDLE read): it
        reported back. A Task call that completed earlier is no report: INF-126's 14 workers had theirs complete, then stopped
        mid-step."""
        with self.kids_lock:
            done = (self.children.get(agent) or {}).get("done")
        return done is not None and done >= idle_t - TASK_DONE_SLACK_S

    def hold(self, kind: str, r: Run, t: float, w: dict, **detail) -> None:
        """A worker's failure, held with its parent's others (`bursts`) to be posted as one by `flush_bursts`."""
        key = w.get("parent") or w.get("handle") or w.get("lane") or r.agent
        b = self.bursts.setdefault(key, {"first": t, "parent": w.get("parent") or "", "handle": w.get("handle") or "",
                                         "lane": w.get("lane") or "", "items": []})
        b["items"] = [x for x in b["items"] if x["agent"] != r.agent] + [
            {"kind": kind, "agent": r.agent, "name": r.name, "run": r.run, "idle_t": w.get("updated", t), **detail}]
        b["last"] = t
        self.save()

    def conversation(self, agent: str) -> list[dict] | None:
        """An agent's messages (`GET /v0/agents/{id}/conversation`: user and assistant text only, no tool calls); None when
        the API won't read it."""
        try:
            msgs = (self.api.get(f"/v0/agents/{agent}/conversation") or {}).get("messages") or []
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(self.clock())} conversation of {agent}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return None
        return [m for m in msgs if isinstance(m, dict)]

    def wake_of(self, agent: str, idle_t: float) -> dict:
        """The wake-up a worker says it set before it stopped (`wake`, its words), and when it is due (`wake_due`) if it says
        in how long, from its last assistant messages."""
        if (msgs := self.conversation(agent)) is None:
            return {}
        turn = []
        for m in reversed(msgs):
            if m.get("type") == "user_message" or len(turn) >= CONV_MSGS:
                break
            if m.get("type") == "assistant_message":
                turn.append(str(m.get("text") or ""))
        said = next((s.group(0).strip() for text in turn for s in [WAKE_RE.search(text)] if s), None)
        if said is None:
            return {}
        out = {"wake": said[:200]}
        if n := WAKE_IN_RE.search(said):
            out["wake_due"] = iso(idle_t + int(n[1]) * (3600 if n[2].lower().startswith("h") else 60))
        return out

    def flush_bursts(self, t: float) -> None:
        """Post each parent's held worker failures once none has joined for `BURST_QUIET_S`, or `BURST_S` after the first:
        one alone as its own record, several as one **workers_failed** to the parent's lane naming them all; each with the
        wake-up it set itself, if it says so."""
        due = [k for k, b in self.bursts.items() if t - b.get("last", b["first"]) >= BURST_QUIET_S or t - b["first"] >= BURST_S]
        for key in sorted(due):
            b = self.bursts.pop(key)
            items = []
            for x in b["items"]:
                if x["kind"] == "worker_idle" and self.reported_back(x["agent"], x.get("idle_t", t)):
                    continue
                idle_t = x.pop("idle_t", t)
                items.append({**x, **self.wake_of(x["agent"], idle_t)})
            if not items:
                continue
            if len(items) == 1:
                x = items[0]
                self.record(x.pop("kind"), Run(x.pop("agent"), x.pop("run"), x.pop("name")), **x)
                continue
            parent = b["parent"]
            span = max(epoch(x.get("since") or "") or t for x in items) - min(epoch(x.get("since") or "") or t for x in items)
            self.record("workers_failed", Run(parent, "", self.names.get(parent, "")), judged=False, role="workers",
                        lane=b["lane"], count=len(items), minutes=int(span // 60), members=items,
                        **({"handle": b["handle"], "owner": b["handle"]} if b["handle"] else {}))
        if due:
            self.save()

    def restart(self, a: dict, run: str, t: float) -> None:
        """A new run on an agent whose latest run ended in ERROR, at most once every `RESTART_S` and `MAX_RESTARTS` times in a
        row; then a coordinator is relaunched."""
        r = Run(a["id"], run, a.get("name") or "")
        n = self.errors.get(a["id"], 0)
        if run in self.done:                        # restarted already: only the next ERROR run counts
            return
        if n >= MAX_RESTARTS:
            if a["id"] not in self.left:
                self.left.add(a["id"])
                self.save()
                handle = self.coords.get(a["id"])
                self.record("erroring", r, status="ERROR", restarts=n,
                            action="relaunching it" if handle else "none: left for a person", **self.owned(a["id"]))
                if handle:
                    self.relaunch_coordinator(a, handle, f"ended in ERROR after {n} restarts in a row", run, t)
            return
        if t - self.freed.get(a["id"], -RESTART_S) < RESTART_S or not self.once(run, t):
            return
        self.freed[a["id"]], self.errors[a["id"]] = t, n + 1
        self.save()
        try:
            new = self.api.post(f"/v1/agents/{a['id']}/runs", {"prompt": {"text": AFTER_ERROR}})
            self.record("crashed", r, status="ERROR", action=f"restarted ({n + 1} of {MAX_RESTARTS})",
                        new_run=((new or {}).get("run") or {}).get("id"), **self.owned(a["id"]))
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # agent_busy: one of its own events already started a run; not ours to count
                self.freed.pop(a["id"], None)
                self.errors[a["id"]] = n
                self.done.pop(run, None)
                self.save()
            else:
                self.record("crashed", r, status="ERROR", action=f"restart failed: HTTP {e.code}", **self.owned(a["id"]))
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.record("crashed", r, status="ERROR", action=f"restart failed: {type(e).__name__}: {e}", **self.owned(a["id"]))

    @staticmethod
    def ready(status: str | None, last: float | None, bell: float | None, t: float, every: float | None = None) -> str | None:
        """Why a coordinator whose latest run is in `status`, last active at `last`, is ready to wake. Only a doorbell after
        `last` counts: one rung during the turn may have been read in it, and the timer rule catches one that wasn't. Idle
        time alone counts only for a handle that declares a recurring timer of `every` seconds, after twice that."""
        if status not in IDLE_DONE or last is None:
            return None
        if bell is not None and bell > last and t - bell >= DOORBELL_S:
            return f"a doorbell rang at {iso(bell)} and nothing has run since"
        if every and t - last >= 2 * every:
            return (f"nothing has run for {int((t - last) // 60)} min since its latest run went {status} at {iso(last)} "
                    f"(its timer is every {int(every // 60)} min)")
        return None

    def wake_up(self, a: dict, handle: str, run: str, why: str, t: float, status: str | None = None) -> None:
        """Record a coordinator ready to wake, and with `wake` start a run on it; at most once per agent an hour. A wake
        refused because its run waits on background work cancels that run and is started again (`unblock`)."""
        if t - self.woken.get(a["id"], -RECANCEL_S) < RECANCEL_S:
            return
        self.woken[a["id"]], new, action = t, None, "recorded (--wake is off)"
        self.save()
        if self.wake:
            text = WAKE.format(why=why, handle=handle)
            new, err = self.cont(a["id"], text)
            unblocked = new is None and err == "HTTP 409" and status == "WAITING_FOR_BACKGROUND_WORK"
            if unblocked:
                new, err = self.unblock(a["id"], run, text)
            # 409 agent_busy otherwise counts too: zombie CREATING runs keep an agent busy for hours
            action = (f"cancelled its background wait ({run}) and started a run" if new is not None and unblocked else
                      "started a run" if new is not None else
                      f"wake refused after cancelling its background wait: {err}" if unblocked else
                      "wake refused: HTTP 409 (agent busy); tried again in an hour" if err == "HTTP 409" else f"wake failed: {err}")
        self.record("ready_to_wake", Run(a["id"], run, a.get("name") or ""), handle=handle, reason=why, action=action,
                    **({"new_run": new} if new else {}))

    def unblock(self, agent: str, run: str, text: str) -> tuple[str | None, str]:
        """Cancel `run`, which waits on background work (Cursor refuses any new run on the agent meanwhile), and start `text`
        on the agent: `cont`'s answer, or (None, why) when the cancel is refused."""
        try:
            self.api.post(f"/v1/agents/{agent}/runs/{run}/cancel")
        except urllib.error.HTTPError as e:
            if e.code != 409:                       # run_not_cancellable: it ended meanwhile, so the run may start now
                return None, f"cancel refused: HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return None, f"cancel failed: {type(e).__name__}: {e}"
        return self.cont(agent, text)

    def on_event(self, run_id: str, event: str, data: dict) -> None:
        with self.lock:
            r = self.runs.get(run_id)
            if r is None:
                return
            t = self.clock()
            kind = data.get("type") if event == "interaction_update" else None
            if event in ("status", "result"):
                if str(data.get("status")) in TERMINAL:
                    r.ended = str(data.get("status"))
                return
            if event not in ("tool_call", "assistant", "thinking", "interaction_update"):
                return                              # heartbeat, error, done
            cid = str(data.get("callId") or data.get("name") or "?") if event == "tool_call" else data.get("callId")
            if event == "tool_call" and data.get("name") == "task":
                self.child(r.agent, data, t)
            if kind in SUBAGENT or (cid is not None and str(cid) in r.bg):
                r.bg_t = t                          # a subagent's progress, or its task call finishing: not the parent's doing
                return
            if event == "tool_call":
                if data.get("status") == "completed" and cid not in r.calls and (
                        data.get("name") == "task" or r.status == "WAITING_FOR_BACKGROUND_WORK"):
                    r.bg_t = t                      # a worker finishing, its start before the replay: the parent's turn is over
                    return
                if data.get("status") == "completed":
                    r.calls.pop(cid, None)
                    r.last_kind = "call" if r.calls else "tool_done"
                else:
                    r.calls.setdefault(cid, t)
                    r.last_kind = "call"
            elif event in ("assistant", "thinking") or kind in ("step-started", "turn-ended", "user-message-appended"):
                r.bg.update(r.calls)                # it went on past them: background work (a cloud subagent's task)
                r.calls.clear()
                if event in ("assistant", "thinking"):
                    r.last_kind = "text"
                elif kind == "turn-ended":
                    r.last_kind = "turn_ended"
                elif kind == "user-message-appended":
                    r.last_kind = "start"
            r.last_t = t

    def classify(self, r: Run, t: float) -> str | None:
        if r.ended:
            return "ended"
        waiting = r.status == "WAITING_FOR_BACKGROUND_WORK"
        quiet = t - (max(r.last_t, r.bg_t) if waiting else r.last_t)
        if not r.calls and quiet >= FOLLOW[r.status] and (r.last_kind == "tool_done" or (r.last_kind == "turn_ended" and not waiting)):
            return "held"
        if r.status == "RUNNING" and t - max(r.last_t, r.bg_t) >= STALL_S and t - r.agent_t >= STALL_S:
            return "stalled"
        if waiting and t - max(r.last_t, r.bg_t) >= STUCK[r.status] and t - r.agent_t >= STUCK[r.status]:
            return "stalled"
        if r.calls and t - min(r.calls.values()) >= LONG_CALL_S:
            return "long_call"
        if waiting and t - max(r.last_t, r.bg_t) >= STUCK[r.status]:
            return "stuck"
        return None

    def unread(self, r: Run, t: float, bell: float | None, every: float | None) -> str | None:
        """Why a holder's run (`holders_all`) is held while a worker streams: RUNNING or waiting on background work, it has done
        nothing itself for `UNREAD_S` while a worker it waits on streamed within `WORKER_LIVE_S`, so it reads no follow-ups
        (compute-accounting's research lane 17:25-21:45Z, network-accounting 11:30-13:36Z, 24 Oct: a resumed worker streamed
        in the foreground, no rule fired). For the handle's inbox holder, a doorbell rang after its last own event `UNREAD_S`
        ago or its timer lapsed; a lane that posts as the handle has subscriptions the watch can't see, so the hour alone."""
        if r.ended or r.status not in FOLLOW or r.agent not in self.holders_all:
            return None
        if t - r.last_t < UNREAD_S or t - r.bg_t >= WORKER_LIVE_S:
            return None
        if r.agent not in self.coords:
            return f"a lane of @{self.holders_all[r.agent]}: nothing of its own for {int((t - r.last_t) // 60)} min"
        if bell is not None and bell > r.last_t and t - bell >= UNREAD_S:
            return f"a doorbell rang at {iso(bell)}"
        if every and t - r.last_t >= max(UNREAD_S, 2 * every):
            return f"its {int(every // 60)}-min timer lapsed"
        return None

    def free(self, r: Run, t: float, kind: str = "held", why: str = "") -> None:
        """Cancel a held, stalled or held-unread run and start a continue run on its agent, at most once per agent an hour."""
        quiet = r.last_t if kind in ("held", "held_unread") else max(r.last_t, r.bg_t)
        since, mins, owner = iso(quiet), int((t - quiet) // 60), self.owned(r.agent)
        if why:
            owner = {**owner, "handle": self.holders_all.get(r.agent), "reason": why}
        if t - self.freed.get(r.agent, -RECANCEL_S) < RECANCEL_S:
            if "held-again" not in r.told:
                r.told.add("held-again")
                self.record("held_again", r, since=since, minutes=mins, rule=kind, action="none: freed once within the hour",
                            **owner)
            return
        if r.refused_t is not None:
            if t - r.refused_t >= REFUSED_S and "refused" not in r.told:
                r.told.add("refused")
                handle = self.coords.get(r.agent)
                self.record("cancel_refused", r, since=since, minutes=mins, rule=kind,
                            action=f"{'relaunching it' if handle else 'none'}: the API refused the cancel (409) and still "
                                   f"reads it {r.status}", **owner)
                if handle:
                    self.relaunch_coordinator({"id": r.agent, "name": r.name}, handle,
                                              f"had run {r.run} {kind}, and the API refused to cancel it", r.run, t)
            return
        try:
            self.api.post(f"/v1/agents/{r.agent}/runs/{r.run}/cancel")
        except urllib.error.HTTPError as e:
            if e.code == 409:                       # run_not_cancellable: it ended meanwhile, and the runs list lags
                r.refused_t = t
                return
            self.freed[r.agent] = t
            self.save()
            self.record(kind, r, since=since, minutes=mins, action=f"cancel failed: HTTP {e.code}", **owner)
            r.ended = "CANCELLED"
            return
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.freed[r.agent] = t
            self.save()
            self.record(kind, r, since=since, minutes=mins, action=f"cancel failed: {type(e).__name__}: {e}", **owner)
            r.ended = "CANCELLED"
            return
        self.freed[r.agent] = t
        r.ended = "CANCELLED"
        text = (CONTINUE.format(since=since, mins=mins, after=AFTER.get(r.last_kind, "a finished tool call")) if kind == "held"
                else UNREAD.format(since=since, mins=mins, why=why) if kind == "held_unread"
                else STALLED.format(since=since, mins=mins, status=r.status))
        new, err = self.cont(r.agent, text)
        if new is not None:
            self.save()
            self.record(kind, r, since=since, minutes=mins, status=r.status, action="cancelled and continued", new_run=new,
                        **owner)
        else:
            self.pending[r.agent] = {"text": text, "run": r.run, "name": r.name, "since": t}
            self.save()
            self.record(kind, r, since=since, minutes=mins, action=f"cancelled; the continue run was refused ({err}), "
                        "tried again each sweep", **owner)

    def cont(self, agent: str, text: str) -> tuple[str | None, str]:
        """Start the continue run: (its id or "", "") once started, (None, why) when refused."""
        try:
            new = self.api.post(f"/v1/agents/{agent}/runs", {"prompt": {"text": text}})
            return ((new or {}).get("run") or {}).get("id") or "", ""
        except urllib.error.HTTPError as e:
            return None, f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return None, f"{type(e).__name__}: {e}"

    def retry(self, current: dict[str, tuple[dict, dict]], t: float) -> None:
        """Start each continue run the API refused, unless the agent has a run of its own going again; a coordinator's refused
        for `CONTINUE_RETRY_S` is relaunched."""
        going = {a["id"] for a, x in current.values() if x["id"] != self.pending.get(a["id"], {}).get("run")}
        for agent, p in list(self.pending.items()):
            r = Run(agent, p["run"], p.get("name", ""))
            if agent in going:
                self.pending.pop(agent)
            elif t - p["since"] >= CONTINUE_RETRY_S:
                self.pending.pop(agent)
                handle = self.coords.get(agent)
                self.record("continue_failed", r, since=iso(p["since"]),
                            action="relaunching it" if handle else "none: left for a person", **self.owned(agent))
                if handle:
                    self.relaunch_coordinator({"id": agent, "name": r.name}, handle,
                                              f"had a continue run refused for {CONTINUE_RETRY_S // 60} min", p["run"], t)
            else:
                new, _ = self.cont(agent, p["text"])
                if new is None:
                    continue
                self.pending.pop(agent)
                self.record("continued", r, new_run=new)
            self.save()

    def read(self, a: dict, run: str) -> dict | None:
        """The run's record (a terminal one is read once); None, and recorded once, when the API won't read it."""
        if run in self.final:
            return self.final[run]
        try:
            info = {"id": run, **self.api.get(f"/v1/agents/{a['id']}/runs/{run}")}
        except urllib.error.HTTPError as e:         # a goal run's id (run-goal-...) isn't one the runs endpoint takes
            t, first, r = self.clock(), self.done.get(run), Run(a["id"], run, a.get("name") or "")
            if self.once(run, t):
                self.record("unreadable", r, status=f"HTTP {e.code}")
            elif first is not None and t - first >= UNREADABLE_S and self.once(run + "#unreadable", t):
                self.record("unreadable", r, status=f"HTTP {e.code}", since=iso(first), minutes=int((t - first) // 60))
            return None
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(self.clock())} run {run} of {a['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return None
        if info.get("status") in TERMINAL:
            self.final[run] = info
        return info

    def runs_of(self, a: dict, t: float) -> list[dict] | None:
        """The agent's newest runs, newest first, back to the newest that started: an ACTIVE agent's paged from the list, else
        its latest, paged behind only when that was cancelled on arrival. None when the list fails."""
        latest = a["latestRunId"]
        runs: list[dict] = []
        if a.get("status") != "ACTIVE":
            if not (info := self.read(a, latest)) or started(info):
                return [info] if info else []
            runs = [info]
        try:
            listed, cursor = [], None
            for _ in range(RUNS_PAGES):
                page = self.api.get(f"/v1/agents/{a['id']}/runs", limit=RUNS_PAGE, cursor=cursor)
                listed += page.get("items") or []
                cursor = page.get("nextCursor")
                if not cursor or any(started(x) for x in listed):
                    break
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(t)} runs of {a['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return runs or None
        if not runs and latest not in {x.get("id") for x in listed} and (info := self.read(a, latest)):
            runs = [info]
        return runs + [x for x in listed if x.get("id") not in {y["id"] for y in runs}]

    def watch_workers(self, coords: dict[str, str], listed: set[str], t: float) -> None:
        """Read each known worker (`GET /v1/agents/{id}`, at most once a sweep): record `worker_idle` when it went from ACTIVE
        to IDLE, and `worker_long` once when it has been ACTIVE `worker_long_s`. A worker has no runs the API lists and can't
        be started (400 legacy workflow), so only its coordinator can resume it (Task); the watch records, never acts."""
        before = json.dumps([self.workers, self.owners], sort_keys=True)
        for agent, d in workers(self.notes, self.log.with_name("workers.d"), self.registry, coords, t).items():
            if agent in listed and d["handle"]:                 # an agent a coordinator launched through the API: it owns it
                self.owners[agent] = d["handle"]
            w = self.workers.setdefault(agent, {"heard": 0.0})
            w.update(lane=d["lane"] or w.get("lane") or "", handle=d["handle"] or w.get("handle"),
                     heard=max(w["heard"], d["heard"]))
            if d["branches"]:
                w["branches"] = sorted({*w.get("branches", []), *d["branches"]})
            if d.get("parent") and not w.get("parent"):
                w["parent"] = d["parent"]
        with self.kids_lock:
            kids = dict(self.children)
        parents: dict[str, str | None] = {}         # parent -> its handle, read once a sweep
        for agent, c in kids.items():
            if agent in coords:
                continue
            p = c["parent"]
            if p not in parents:
                parents[p] = self.parent_handle(p, coords)
            handle = parents[p]
            if agent in listed and handle:
                self.owners[agent] = handle
            w = self.workers.setdefault(agent, {"heard": 0.0})
            w.update(parent=p, lane=w.get("lane") or handle or "", handle=w.get("handle") or handle,
                     heard=max(w["heard"], c["t"]), name=w.get("name") or c.get("name") or "")
        self.worker_status, new = {}, 0
        for agent, w in sorted(self.workers.items()):
            if agent in listed or agent in coords:              # an agent the sweep reads already, or a coordinator now
                self.workers.pop(agent)
                continue
            idle_long = w.get("status") == "IDLE" and t - w.get("updated", t) >= WORKER_IDLE_S
            if w.get("gone") or (idle_long and w["heard"] <= w.get("read", 0)):
                continue
            if "read" not in w:
                if new >= WORKERS_NEW:
                    continue
                new += 1
            w["read"] = t
            try:
                info = self.api.get(f"/v1/agents/{agent}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    w["gone"] = True
                print(f"{iso(t)} worker {agent}: HTTP {e.code}" + ("; not read again" if e.code == 404 else ""),
                      file=sys.stderr, flush=True)
                continue
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"{iso(t)} worker {agent}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue
            if info.get("repos") and not verity(info):         # not Verity's: never acted on (a worker's GET may carry none)
                w["gone"] = True
                print(f"{iso(t)} worker {agent}: not a Verity agent; not read again", file=sys.stderr, flush=True)
                continue
            status, updated, prev = str(info.get("status") or "unknown"), epoch(info.get("updatedAt")) or t, w.get("status")
            w.update(status=status, updated=updated, name=str(info.get("name") or w.get("name") or ""))
            if pushed := pushed_branches(info):
                w["branches"] = sorted({*w.get("branches", []), *pushed})
            self.worker_status[agent] = status
            r = Run(agent, "", w["name"])
            detail = {"lane": w["lane"], **({"handle": w["handle"]} if w.get("handle") else {})}
            if status == "ACTIVE":
                if prev != "ACTIVE":                            # updatedAt moves on a status change, not during work
                    w["since"], w["long"] = updated, False
                if not w["long"] and t - w["since"] >= self.worker_long_s:
                    w["long"] = True
                    self.record("worker_long", r, since=iso(w["since"]), hours=int((t - w["since"]) // 3600), **detail)
            elif status == "IDLE" and prev == "ACTIVE":
                w["idle_at"] = updated
                if self.reported_back(agent, updated):
                    print(f"{iso(t)} worker {agent} idle: its parent got its Task result, so no one is rung", file=sys.stderr,
                          flush=True)
                else:
                    self.hold("worker_idle", r, t, w, at=iso(updated), since=iso(updated), **detail)
            elif status == "ARCHIVED":
                w["gone"] = True
        for agent in [a for a, w in self.workers.items() if (w.get("gone") or w.get("status") == "IDLE")
                      and t - max(w.get("updated", t), w["heard"]) >= WORKER_KEEP_S]:
            self.workers.pop(agent)
        self.owners = {a: h for a, h in self.owners.items() if a in listed}
        if json.dumps([self.workers, self.owners], sort_keys=True) != before:
            self.save()

    def read_heads(self, t: float) -> None:
        """Every `HEADS_EVERY_S`, Verity's branch heads (`git ls-remote --heads`, the deploy key fixtures_export.sh fetches with):
        a branch whose sha moved, or that appeared, was pushed when the watch saw it; `heads.json` keeps them across restarts."""
        if not self.git_remote or t - self.heads_read < HEADS_EVERY_S:
            return
        self.heads_read = t
        rc, out, err = self.sh(["git", "ls-remote", "--heads", self.git_remote], None, 120)
        if rc != 0:
            self.say("heads", f"git ls-remote {self.git_remote} exited {rc}: {err.strip()[-200:]}; no one is judged silent "
                              "while the heads are stale", t)
            return
        old, new = self.heads.get("heads") or {}, {}
        shas = {k: v for k, v in (self.heads.get("shas") or {}).items() if isinstance(v, list) and v[1] >= t - HEADS_SHA_KEEP_S}
        for line in out.splitlines():
            sha, _, ref = line.strip().partition("\t")
            if ref.startswith("refs/heads/") and sha:
                b = ref[len("refs/heads/"):]
                new[b] = old[b] if isinstance(old.get(b), list) and old[b][0] == sha else [sha, t]
                shas.setdefault(sha, [b, t])
        self.heads = {"t0": self.heads.get("t0", t), "t": t, "heads": new, "shas": shas}
        write_atomic(self.heads_path, (json.dumps(self.heads, sort_keys=True) + "\n").encode())

    def watch_silence(self, known: dict[str, dict], t: float) -> None:
        """Record `silent` for each agent in `known` (agent -> {name, handle, lane, run, since: ACTIVE since, branches}) with no
        note of its (`note_times`) and no push to its branches (its own, the ones its notes name, and every head ending in its
        id's last four characters, as a coordinator's `cursor/...-1037`) for `SILENT_S`, counted from the latest of those, its
        ACTIVE spell's start and the first heads read; again after a second window, then not until one of those moves. Needs
        `--notes` and heads read in the last `HEADS_STALE_S`."""
        changed = False
        for agent in [a for a in self.silent if a not in known]:
            self.silent.pop(agent)
            changed = True
        if self.notes is None or not self.git_remote or t - self.heads.get("t", -HEADS_STALE_S) >= HEADS_STALE_S:
            if changed:
                self.save()
            return
        heard, named, lanes = note_times(self.notes, t)
        heads = self.heads.get("heads") or {}
        for agent, k in sorted(known.items()):
            coord = agent in self.coords
            mine = {*k["branches"], *named.get(agent, ()), *(b for b in heads if b.endswith("-" + agent[-4:]))}
            pushed = max((heads[b][1] for b in mine if b in heads), default=None)
            # a coordinator signs its notes with its lane, not its id (top, 27 Oct: four false silent rings in 70 min)
            lane_note = max((lanes[x] for x in (k["handle"], k["lane"]) if x and x in lanes), default=None) if coord else None
            last = max(x for x in (k["since"], heard.get(agent), lane_note, pushed, self.heads["t0"]) if x is not None)
            s = self.silent.get(agent)
            if s and s["from"] != last:
                self.silent.pop(agent)
                s, changed = None, True
            n = s["rings"] if s else 0
            if n >= SILENT_RINGS:
                still = max(last, k.get("acted") or last)
                if s and not s.get("no_step") and t - still >= NO_STEP_S:
                    s["no_step"] = True
                    changed = True
                    self.record("no_step", Run(agent, k["run"], k["name"]), since=iso(still), hours=int((t - still) // 3600),
                                lane=k["lane"], **({"handle": k["handle"]} if k["handle"] else {}), owner="top",
                                role="coordinator" if coord else "worker")
                continue
            if t - last < (COORD_SILENT_S if coord else SILENT_S) * (n + 1):
                continue
            self.silent[agent] = {"from": last, "rings": n + 1}
            changed = True
            self.record("silent", Run(agent, k["run"], k["name"]), since=iso(last), minutes=int((t - last) // 60),
                        active_minutes=int((t - k["since"]) // 60), window=n + 1, lane=k["lane"],
                        branches=sorted(b for b in mine if b in heads) or sorted(mine),
                        **({"handle": k["handle"]} if k["handle"] else {}), **({"owner": "top"} if coord else {}))
        if changed:
            self.save()

    def parent_handle(self, parent: str, coords: dict[str, str]) -> str | None:
        """The handle a worker's `parent` answers to: its own (`lineage`), the coordinator that launched it, or its own launcher's."""
        return lineage(self.registry, coords, parent) or self.owners.get(parent) or (self.workers.get(parent) or {}).get("handle")

    def learn(self, agent: str, coords: dict[str, str], t: float) -> dict:
        """A sender no route knows, learned from its brief (its conversation's first user message): a worker of the first
        agent id there that is a coordinator (`lineage`: WIDTHS' "a cloud worker of the circuits coordinator (agent bc-...)"),
        else of the current coordinator of the first registry handle it names (ZKS10's "a worker for the proofs lane
        (@proofs)"), kept in `children` as a Task call's worker is; {} when it names neither, read again `LEARN_RETRY_S` later."""
        if t - self.learn_tried.get(agent, -LEARN_RETRY_S) < LEARN_RETRY_S:
            return {}
        self.learn_tried[agent], self.learn_left = t, self.learn_left - 1
        msgs = self.conversation(agent) or []
        brief = next((str(m.get("text") or "") for m in msgs if m.get("type") == "user_message"), "")
        parent = next((i for i in AGENT_RE.findall(brief) if i != agent and lineage(self.registry, coords, i)), None)
        if parent is None:
            names = {str(h.get("handle")) for h in read_json(self.registry).get("handles") or []}
            handle = next((h for h in re.findall(r"@([a-z][a-z0-9-]*)", brief) if h in names), None)
            parent = next((a for a, h in sorted(coords.items()) if h == handle), None) if handle else None
        if parent is None:
            return {}
        try:
            name = str((self.api.get(f"/v1/agents/{agent}") or {}).get("name") or "")[:120]
        except (urllib.error.URLError, OSError, ValueError):
            name = ""
        self.learn_tried.pop(agent, None)
        with self.kids_lock:
            kid = self.children[agent] = {"parent": parent, "name": name, "t": t}
        print(f"{iso(t)} run ends: learned {agent} ({name or 'unnamed'}) as a worker of {parent} from its brief",
              file=sys.stderr, flush=True)
        return dict(kid)

    def own_cancel(self, row: dict, agent: str, handle: str, name: str, parent: str | None, coords: dict[str, str]) -> bool:
        """Whether `row`'s run was cancelled by its own lane (`research cancel`, its events' cancel_intent): its BY or reason
        names the sender (its id, or its name's first word, SALT-WIRE's "superseded ... at SALT-WIRE's final head (research
        cancel by ubuntu)"), or its BY names the run's handle or an agent of that handle (its coordinator, a worker of it)."""
        why, by = str(row.get("cancel_reason") or ""), str(row.get("cancelled_by") or "")
        if not why:
            return False
        said = f"{by} {why}"
        short = {s[:11] for s in AGENT_RE.findall(said)} | set(re.findall(r"bc-[0-9a-f]{8}\b", said))
        if agent and agent[:11] in short:
            return True
        first = re.split(r"[:(]", name)[0].strip()
        if len(first) >= 4 and re.search(rf"(?<![\w-]){re.escape(first)}(?![\w-])", said):
            return True
        if not handle:
            return False
        mine = {a for a, h in coords.items() if h == handle} | {parent or ""} | {a for a, h in self.owners.items() if h == handle}
        mine |= {a for a, w in self.workers.items() if w.get("handle") == handle}
        named = {s[:11] for s in AGENT_RE.findall(by)} | set(re.findall(r"bc-[0-9a-f]{8}\b", by))
        return handle in re.findall(r"[a-z0-9-]+", by.lower()) or any(a[:11] in named for a in mine if a)

    def run_ends(self, coords: dict[str, str], t: float) -> None:
        """Every `every_s` of `--nodes` (`{"nodes": [...], "ssh": [...], "machines_d"?, "every_s"?}`), each node's runs that
        ended since its last read (`RUNS_READER`, one ssh per node, host and root from its machines.d entry), each recorded once
        as `run_ended` for the handle of whoever sent it: a coordinator's own or a former one's, a worker's launcher's or its
        parent's (only its coordinator can resume it, with Task), else its lane's (`--by`, the job's lane, or a handle named in
        `submitted_by`) when that is a handle. The read's runs no handle owns are one record, with no handle and role `unknown`.
        `run_ends.json` keeps each node's last read and the runs already seen; runs that ended before the first read are not
        rung."""
        cfg = read_json(self.nodes)
        nodes = [str(n) for n in cfg.get("nodes") or []]
        every = cfg.get("every_s") if isinstance(cfg.get("every_s"), (int, float)) else RUN_ENDS_EVERY_S
        if not nodes or t - self.ends_read < every:
            return
        self.ends_read = t
        st = {"since": self.ends.get("since", t), "nodes": dict(self.ends.get("nodes") or {}),
              "seen": {k: v for k, v in (self.ends.get("seen") or {}).items() if v >= t - RUN_ENDS_KEEP_S}}
        handles = {str(h.get("handle")) for h in read_json(self.registry).get("handles") or []}
        quiet = {str(h) for h in cfg.get("quiet_handles") or []}
        unknown: list[dict] = []
        for node in nodes:
            since = max(st["since"], st["nodes"].get(node, st["since"])) - RUN_ENDS_SLACK_S
            rows, why = node_runs(cfg, self.notes, node, since, self.sh)
            if rows is None:
                self.say(f"node {node}", f"run ends: {why}", t)
                continue
            st["nodes"][node], self.learn_left = t, LEARN_PER_READ
            for row in rows:
                if not isinstance(row, dict) or (key := f"{node}/{row.get('run')}") in st["seen"]:
                    continue
                st["seen"][key] = t
                ended = epoch(row.get("ended")) or row.get("mtime") or t
                if ended >= st["since"] and (u := self.ring_run(node, row, coords, handles, ended, quiet)):
                    unknown.append(u)
        if unknown:
            self.record("run_ended", Run("", unknown[0]["run"] if len(unknown) == 1 else f"{len(unknown)} runs"), judged=False,
                        handle=None, role="unknown", count=len(unknown), runs=unknown,
                        senders=sorted({u["sender"] for u in unknown}))
        self.ends = st
        write_atomic(self.ends_path, (json.dumps(st, sort_keys=True) + "\n").encode())

    def branch_of(self, sha: str) -> str:
        """The branch whose head was ``sha`` when `read_heads` saw it (kept `HEADS_SHA_KEEP_S`), or ""."""
        if not sha:
            return ""
        got = (self.heads.get("shas") or {}).get(sha)
        if isinstance(got, list) and got:
            return str(got[0])
        return next((b for b, v in (self.heads.get("heads") or {}).items() if isinstance(v, list) and v[0] == sha), "")

    def sender_status(self, agent: str) -> str:
        """The sender's status as the watch last read it (`ACTIVE` while it is in a turn, so it sees its run end itself), else
        `unknown`."""
        return (self.agent_status.get(agent) or self.worker_status.get(agent) or str((self.workers.get(agent) or {}).get("status")
                or "") or "unknown") if agent else "unknown"

    def ring_run(self, node: str, row: dict, coords: dict[str, str], handles: set[str], ended: float,
                 quiet: set[str] = frozenset()) -> dict | None:
        """Record `row`'s end for its sender's handle; a run whose sender no handle owns is returned instead, for `run_ends` to
        record with the read's others as one (93 such senders sent 753 runs in 14 h on 19 Oct: one ring each would bury @infra)."""
        sent = str(row.get("submitted_by") or "")
        by = AGENT_RE.search(sent)
        agent = by[0] if by else ""
        words = re.findall(r"[a-z0-9-]+", re.sub(r"https?://\S+", " ", sent))
        lane = next((x for x in (row.get("by"), row.get("lane"), *words) if x in handles), None)
        w = self.workers.get(agent) or {}
        with self.kids_lock:
            kid = self.children.get(agent) or {}
            if kid:
                kid["t"] = self.clock()             # a worker still sending runs is never forgotten
        known = agent in coords or agent in self.owners or w.get("handle") or w.get("parent") or kid
        if agent and not known and not lineage(self.registry, coords, agent) and self.learn_left > 0:
            kid = self.learn(agent, coords, self.clock())
        parent = w.get("parent") or kid.get("parent")
        if agent in coords:
            handle, role, name = coords[agent], "coordinator", self.names.get(agent, "")
        elif agent and (h := lineage(self.registry, coords, agent)):
            handle, role, name = h, "coordinator", self.names.get(agent, "")
        elif agent in self.owners:                 # an agent a coordinator launched through the API, a relaunched worker
            handle, role, name = self.owners[agent], "worker", self.names.get(agent, "") or str(kid.get("name") or "")
        elif w.get("handle"):
            handle, role, name = w["handle"], "worker", str(w.get("name") or "")
        elif parent and (h := self.parent_handle(parent, coords)):
            handle, role, name = h, "worker", str(w.get("name") or kid.get("name") or self.names.get(agent, ""))
        elif lane:
            handle, role, name = lane, "lane", ""
        elif h := branch_handle(self.branch_of(str(row.get("tree") or "")), self.registry, handles):
            handle, role, name = h, "lane", ""
        else:
            handle, role, name = "", "unknown", ""
        if handle in quiet:
            return None
        what = row.get("job") or row.get("stage") or " ".join(str(row.get("command") or "").split()[:4]) or "?"
        if self.own_cancel(row, agent, handle, name if role == "worker" else "", parent, coords):
            print(f"{iso(self.clock())} run ends: {node} {row.get('run')} was cancelled by its own lane "
                  f"({row.get('cancelled_by') or '?'}): posted quietly, nothing to resume", file=sys.stderr, flush=True)
            if role != "unknown":
                self.record("run_ended", Run(agent, str(row.get("run")), name), judged=False, handle=handle, role=role,
                            node=node, what=what[:80], rc=row.get("rc"), state=row.get("state"),
                            after=duration(row.get("wall_s")), at=iso(ended), lane=w.get("lane") or lane or "", quiet=True,
                            cancelled_by=str(row.get("cancelled_by") or ""), cancel_reason=str(row.get("cancel_reason")),
                            sender_status=self.sender_status(agent), **({"parent": parent} if role == "worker" and parent else {}))
            if self.awaiting.pop(agent, None) is not None:
                self.save()
            return None
        if role == "unknown":
            print(f"{iso(self.clock())} run ends: {node} {row.get('run')} sent by {sent or row.get('by') or '?'}, which no "
                  "handle owns: recorded for @infra to route", file=sys.stderr, flush=True)
            return {"run": str(row.get("run")), "node": node, "sender": sent or str(row.get("by") or ""), "what": what[:80],
                    "rc": row.get("rc"), "state": row.get("state"), "after": duration(row.get("wall_s")), "at": iso(ended)}
        self.record("run_ended", Run(agent, str(row.get("run")), name), judged=False, handle=handle, role=role, node=node,
                    what=what[:80],
                    rc=row.get("rc"), state=row.get("state"), after=duration(row.get("wall_s")), at=iso(ended),
                    lane=w.get("lane") or lane or "", sender_status=self.sender_status(agent),
                    **({"parent": parent} if role == "worker" and parent else {}),
                    **({"listed": True} if role == "worker" and agent in self.names else {}))
        # who must act on it: its sender when the list has it or it is a worker, else the lane's handle's coordinator
        target = agent if agent in self.names or role == "worker" else next((a for a, h in coords.items() if h == handle), None)
        if target:
            e = self.awaiting.setdefault(target, {"handle": handle, "runs": []})
            e["runs"] = [*e["runs"], {"run": str(row.get("run")), "node": node, "what": what[:80], "rc": row.get("rc"),
                                      "state": row.get("state"), "after": duration(row.get("wall_s")), "ended": ended}][-10:]
            self.save()
        return None

    def turn(self, a: dict, judges: dict[str, dict], live: dict[str, dict], t: float) -> tuple[dict | None, float | None]:
        """(the run a listed agent is judged by, its last activity): the later of that run's `updatedAt` and its last stream
        event; an agent the sweep didn't read is read here."""
        if (info := judges.get(a["id"])) is None:
            runs = [] if a["id"] in live else self.runs_of(a, t) or []
            if not (info := next((x for x in runs if started(x)), runs[0] if runs else None)):
                return None, None
        with self.lock:
            r = self.runs.get(info["id"])
            heard = r.last_t if r and r.last_kind != "start" else None
        return info, max((x for x in (epoch(info.get("updatedAt") or info.get("createdAt")), heard) if x is not None), default=None)

    def watch_resumes(self, agents: dict[str, dict], judges: dict[str, dict], live: dict[str, dict], t: float) -> None:
        """Act on each agent that must act on a research run that ended (`awaiting`, from `ring_run`) and hasn't. A listed one
        whose turn is over with nothing run since the run ended is resumed with its result `RESUME_S` after (`idle_after_run`);
        a worker IDLE since before it ended has its coordinator rung then, and is relaunched `WORKER_RELAUNCH_S` later."""
        before = json.dumps(self.awaiting, sort_keys=True)
        for agent, e in list(self.awaiting.items()):
            runs = [x for x in e["runs"] if t - x["ended"] < AWAIT_KEEP_S]
            w = self.workers.get(agent)
            if agent in agents:
                info, last = self.turn(agents[agent], judges, live, t)
                if info is None:
                    continue
                runs = [x for x in runs if last is None or x["ended"] > last]     # one it ran after was its to read
            elif w is not None and not w.get("gone"):
                if w.get("status") is None:
                    continue
                runs = [x for x in runs if x["ended"] > w.get("updated", 0)]    # it changed status after: it could read it
            elif w is None and agent in self.children and runs:
                e["runs"] = runs                    # learned this sweep (`learn`): judged once `watch_workers` reads it
                continue
            else:
                runs = []
            if not runs:
                self.awaiting.pop(agent)
                continue
            e["runs"], first = runs, min(x["ended"] for x in runs)
            mins = int((t - first) // 60)
            if agent in agents:
                if (info.get("status") not in IDLE_DONE or t - first < RESUME_S
                        or self.woken.get(agent, -RECANCEL_S) > first or agent in self.pending):
                    continue
                self.awaiting.pop(agent)
                a, handle = agents[agent], self.coords.get(agent)
                text = RESUME.format(runs=f"your research {ended_runs(runs)}", mins=mins,
                                     inbox=f", read your inbox (`research msg read --as {handle}`)" if handle else "")
                new, err = self.cont(agent, text)
                unblocked = new is None and err == "HTTP 409" and info.get("status") == "WAITING_FOR_BACKGROUND_WORK"
                if unblocked:
                    new, err = self.unblock(agent, info["id"], text)
                if new is None:
                    self.pending[agent] = {"text": text, "run": info["id"], "name": a.get("name") or "", "since": t}
                action = ((f"cancelled its background wait ({info['id']}) and " if unblocked and new is not None else "")
                          + ("resumed it with the run's result" if new is not None else
                             f"resume refused ({err}{', after cancelling its background wait' if unblocked else ''}); tried "
                             "again each sweep"))
                self.woken[agent] = t
                self.save()
                self.record("idle_after_run", Run(agent, info["id"], a.get("name") or ""), role="coordinator" if handle else "agent",
                            status=info.get("status"), since=iso(last) if last else iso(first), minutes=mins,
                            ended_run=", ".join(x["run"] for x in runs), result=ended_runs(runs[-1:]), action=action,
                            **({"new_run": new} if new else {}), **({"handle": handle} if handle else {}), **self.owned(agent))
            elif w.get("status") == "IDLE":
                if w.get("relaunched_for") is not None and w.get("relaunched_for") == w.get("updated"):
                    self.awaiting.pop(agent)                # judged once for this idle: never relaunched twice
                    continue
                if "rung" not in e:
                    if t - first < RESUME_S:
                        continue
                    e["rung"] = t
                    handle = w.get("handle")
                    self.hold("idle_after_run", Run(agent, "", str(w.get("name") or "")), t, w, role="worker",
                              lane=w.get("lane") or "", since=iso(w.get("updated", first)), minutes=mins,
                              ended_run=", ".join(x["run"] for x in runs), result=ended_runs(runs[-1:]),
                              action=(f"rang @{handle} to resume it with Task" if handle else "none: no coordinator owns it")
                              + f"; relaunched fresh in {WORKER_RELAUNCH_S // 60} min if it stays idle with its task "
                                "unfinished and its coordinator isn't ACTIVE",
                              **({"handle": handle} if handle else {}), **self.owned(agent))
                elif t - e["rung"] >= WORKER_RELAUNCH_S and not self.coordinator_active(w, agents):
                    if (why := self.unfinished(agent, w, runs)) is None:
                        continue                            # its conversation unreadable: never relaunched on a guess
                    self.awaiting.pop(agent)
                    w["relaunched_for"] = w.get("updated")
                    if why[0]:
                        self.relaunch_worker(agent, w, runs, t, why[1])
                    else:
                        self.record("not_relaunched", Run(agent, "", str(w.get("name") or "")), judged=False, role="worker",
                                    lane=w.get("lane") or "", since=iso(w.get("updated", first)),
                                    ended_run=", ".join(x["run"] for x in runs), why=why[1], quiet=True,
                                    **({"handle": w["handle"]} if w.get("handle") else {}))
        if json.dumps(self.awaiting, sort_keys=True) != before:
            self.save()

    def coordinator_active(self, w: dict, agents: dict[str, dict]) -> bool:
        """Whether a worker's coordinator (its parent, or an agent holding its handle) is ACTIVE: it may be about to resume it."""
        mine = {w.get("parent") or ""} | {a for a, h in self.coords.items() if w.get("handle") and h == w["handle"]}
        return any((agents.get(a) or {}).get("status") == "ACTIVE" for a in mine if a)

    def unfinished(self, agent: str, w: dict, runs: list[dict]) -> tuple[bool, str] | None:
        """(whether to relaunch an idle worker, why), from its conversation and its newest ended run: never one whose last
        message reads as a report back (`reported`); else one whose newest run failed (rc not 0: `ring_run` keeps a cancel by
        its own lane out of `runs`) or that stopped mid-step (`mid_step`, or a prompt it never answered). None when its
        conversation can't be read."""
        if not w.get("handle"):
            return False, "no coordinator owns it, so a relaunch would have no one to report to"
        if (msgs := self.conversation(agent)) is None:
            return None
        said = next((str(m.get("text") or "") for m in reversed(msgs) if m.get("type") == "assistant_message"), "")
        quote = " ".join(said.split())[-160:]
        if reported(said):
            return False, f"its last message reads as a report back (\u201c\u2026{quote}\u201d)"
        x = max(runs, key=lambda r: r["ended"])
        if x.get("rc") != 0 or x.get("state") not in ("done", None):
            return True, f"its run {x['run']} ended {x.get('state')} rc={x.get('rc')} and its last message reports nothing"
        if not msgs or msgs[-1].get("type") != "assistant_message":
            return True, "it never answered its last prompt"
        if mid_step(said):
            return True, f"it stopped mid-step (its last message: \u201c{quote}\u201d)"
        return False, (f"its run {x['run']} passed (rc=0) and its last message (\u201c\u2026{quote}\u201d) reads as neither a "
                       "report nor a stop mid-step")

    def relaunch_coordinator(self, a: dict, handle: str, why: str, run: str, t: float) -> None:
        """Launch `handle`'s next coordinator from its newest handover note (`launch_coordinator.py launch --predecessor`, its
        inbox thread as the pairing thread), with the launch key; recorded as not done while there is none."""
        agent = a["id"]
        if t - self.relaunched.get(agent, -RELAUNCH_EVERY_S) < RELAUNCH_EVERY_S:
            return
        self.relaunched[agent] = t
        self.save()
        note, gen = handover_note(self.notes, handle), generation(handle, agent)
        inbox = str(read_json(self.routes / f"{handle}.json").get("inbox_ts") or "") if self.routes else ""
        new, key = None, self.launch_key()
        if key is None:
            action = f"not relaunched: no launch key in {self.launch_env} (relaunching is off until it exists)"
        elif self.launcher is None or not self.launcher.is_file():
            action = f"not relaunched: no launch_coordinator.py at {self.launcher}"
        elif gen is None:
            action = f"not relaunched: {agent} is no generation launch_coordinator.py makes for {handle}"
        elif not inbox:
            action = f"not relaunched: routes.d names no inbox thread for {handle}"
        else:
            brief = self.log.with_name(f"relaunch-{handle}.md")
            brief.write_text(HANDOVER.format(old=agent, why=why, at=iso(t), note=note.name if note else "none found",
                                             text=note_text(note) or f"(no note under lanes/{handle}/)"))
            argv = [sys.executable, str(self.launcher), "launch", handle, "--predecessor", agent, "--pairing-thread", inbox,
                    "--generation", str(gen + 1), "--text-file", str(brief)]
            rc, out, err = self.launch(argv, {"CURSOR_API_KEY": key,
                                              **({"RESEARCH_SLACK_REGISTRY": str(self.registry)} if self.registry else {})})
            m = AGENT_RE.search(out) if rc == 0 else None
            new = m[0] if m else None
            action = (f"relaunched it from its handover note as {new} (generation {gen + 1})" if new else
                      f"relaunch failed: launch_coordinator.py exited {rc}: {(err or out).strip()[-200:]}")
        self.record("relaunch", Run(agent, run, a.get("name") or ""), role="coordinator", handle=handle, reason=why, action=action,
                    note=note.name if note else "", **({"new_agent": new} if new else {}), owner="top")

    def relaunch_worker(self, agent: str, w: dict, runs: list[dict], t: float, why: str = "") -> None:
        """Launch a fresh agent in place of a worker nobody resumed (`POST /v1/agents` with the launch key), named as a worker
        (`LANE worker NAME (relaunch of XXXX)`: a coordinator's name is `verity: LANE ...`), told its branches, how to report
        to its coordinator's inbox and that the watch wakes it after its runs, and given its newest note; written to
        `workers.d` as its lane's worker, launched by its coordinator, so its runs route as a worker's. Recorded as not done
        while there is no key."""
        if t - self.relaunched.get(agent, -RELAUNCH_EVERY_S) < RELAUNCH_EVERY_S:
            return
        self.relaunched[agent] = t
        self.save()
        handle, lane = w.get("handle"), str(w.get("lane") or "")
        short = re.split(r"[:(]", str(w.get("name") or ""))[0].strip() or agent[:11]
        mine = [(mtime, p) for p, mtime, fm in recent_notes(self.notes, t) if agent in AGENT_RE.findall(fm.get("origin", ""))]
        note = max(mine)[1] if mine else None
        new, key = None, self.launch_key()
        named = [] if w.get("branches") else list(dict.fromkeys(NOTE_BRANCH_RE.findall(note_text(note))))
        launcher = next((a for a, h in sorted(self.coords.items()) if h == handle), "") or str(w.get("parent") or "")
        inbox = str(read_json(self.routes / f"{handle}.json").get("inbox_ts") or "") if self.routes and handle else ""
        if key is None:
            action = f"not relaunched: no launch key in {self.launch_env} (relaunching is off until it exists)"
        else:
            text = WORKER_RELAUNCH.format(old=agent, short=short, lane=lane or "?", handle=handle or "?", why=why or "?",
                                          runs=ended_runs(runs), mins=int((t - min(x["ended"] for x in runs)) // 60),
                                          inbox=inbox or f"<its inbox thread: inbox_ts in the notes' routes.d/{handle}.json>",
                                          branches=", ".join(w.get("branches") or []) or "none recorded"
                                          + (f"; its newest note names {', '.join(named)}" if named else ""),
                                          note=note.name if note else "none in the last 12 h", text=note_text(note))
            body = {"name": f"{handle or lane or 'verity'} worker {short} (relaunch of {agent[-4:]})"[:100],
                    "prompt": {"text": text}, "env": {"type": "cloud", "name": ENV_NAME}, "model": MODEL}
            try:
                new = ((self.launch_api(key).post("/v1/agents", body) or {}).get("agent") or {}).get("id")
                action = f"relaunched it fresh as {new}"
            except urllib.error.HTTPError as e:
                action = f"relaunch failed: HTTP {e.code}"
            except (urllib.error.URLError, OSError, ValueError) as e:
                action = f"relaunch failed: {type(e).__name__}: {e}"
            if new:
                write_atomic(self.log.with_name("workers.d") / f"{handle or lane or 'worker'}-{new}.json",
                             (json.dumps({"agent_id": new, "lane": handle or lane, "launched_by": launcher, "role": "worker",
                                          "name": body["name"], "relaunch_of": agent,
                                          **({"branch": b} if (b := (w.get("branches") or named or [None])[0]) else {})})
                              + "\n").encode())
                if handle:
                    self.owners[new] = handle
                    self.save()
        self.record("relaunch", Run(agent, "", str(w.get("name") or "")), role="worker", lane=lane, reason=(
            f"idle since {iso(w.get('updated', t))} after its run {', '.join(x['run'] for x in runs)} ended, not resumed "
            f"{WORKER_RELAUNCH_S // 60} min after its coordinator was rung, and {why or 'unfinished'}"), action=action,
            note=note.name if note else "",
            **({"new_agent": new} if new else {}), **({"handle": handle} if handle else {}), **self.owned(agent))

    def sweep(self, start_stream: Callable[[Run], None]) -> None:
        """One pass: find each live agent's current run and follow it, record runs stuck in CREATING, restart an ERROR one,
        start refused continue runs, then judge every followed run."""
        t = self.clock()
        live, agents = {}, list(self.api.agents())
        # the key sees every agent on the account (its owner's own too);
        # only Verity's are followed, restarted or woken, and only they are status rows
        listed, agents = {a["id"] for a in agents}, [a for a in agents if verity(a)]
        self.coords = coords = holders(self.registry, self.routes)
        self.holders_all = all_holders(self.registry, self.routes)
        for a in agents:
            recent = t - (epoch(a.get("updatedAt")) or 0) < IDLE_RECENT_S
            if a.get("latestRunId") and (a.get("status") == "ACTIVE" or (a.get("status") == "IDLE" and recent)):
                live[a["id"]] = a
        current: dict[str, tuple[dict, dict]] = {}      # run -> (agent, run record)
        judged: dict[str, dict] = {}                    # run -> agent: the run each agent read is judged by
        judges: dict[str, dict] = {}                    # agent -> the record of the run it is judged by
        for a in live.values():
            latest = a["latestRunId"]
            if (runs := self.runs_of(a, t)) is None:
                continue
            if pushed := set().union(*(pushed_branches(x) for x in runs)):
                self.branches[a["id"]] = pushed
            follow = next((x for x in runs if x.get("status") in FOLLOW and not str(x.get("id")).startswith(GOAL)), None)
            newer_started = False                       # a CREATING run with a newer one started after it blocks nothing,
            for x in runs if follow is None else ():    # and one behind a followed run waits for it, judged in its place
                since = epoch(x.get("updatedAt") or x.get("createdAt"))
                if (x.get("status") == "CREATING" and not newer_started and since is not None
                        and t - since >= STUCK["CREATING"] and self.once(x["id"], t)):
                    self.record("stuck", Run(a["id"], x["id"], a.get("name") or ""), status="CREATING", since=iso(since),
                                minutes=int((t - since) // 60))
                newer_started |= started(x)
            last = next((x for x in runs if x.get("id") == latest), None)
            if follow:
                current[follow["id"]] = (a, follow)
            judge = follow or next((x for x in runs if started(x) or x.get("status") == "CREATING"), None) or last
            judged[judge["id"] if judge else latest] = a
            if judge:
                judges[a["id"]] = judge
                self.status_of[a["id"]] = (judge["id"], str(judge.get("status")))
            if last and last.get("status") == "FINISHED":   # not a run going: a restart runs a minute before it errors
                if self.errors.pop(a["id"], None) is not None or a["id"] in self.left:
                    self.left.discard(a["id"])
                    self.save()
            elif last and last.get("status") == "ERROR":
                self.restart(a, latest, t)
        self.retry(current, t)
        bells, fresh = doorbells(self.log.with_name("doorbells.json"), t)
        every = timers(self.timers)
        if fresh != self.doorbells and (coords or self.doorbells):
            print(f"{iso(t)} doorbells.json: {fresh}; {len(coords)} coordinators", file=sys.stderr, flush=True)
        self.doorbells = fresh
        for a in agents:
            handle = coords.get(a["id"])
            if not handle or not a.get("latestRunId") or a.get("status") == "ARCHIVED":
                continue
            info, last = self.turn(a, judges, live, t)
            if info is None:
                continue
            if why := self.ready(info.get("status"), last, bells.get(handle), t, every.get(handle)):
                self.wake_up(a, handle, info["id"], why, t, info.get("status"))
        self.watch_workers(coords, listed, t)
        self.names = {a["id"]: str(a.get("name") or "") for a in agents}
        self.agent_status = {a["id"]: str(a.get("status") or "") for a in agents}
        known, active = {}, {}
        for a in agents:
            if a["id"] in coords and a.get("status") == "ACTIVE":
                active[a["id"]] = since = self.active.get(a["id"], t)
                judge = judges.get(a["id"]) or {}
                with self.lock:
                    r = self.runs.get(judge.get("id") or "")
                acted = max((x for x in (epoch(a.get("updatedAt")), epoch(judge.get("updatedAt") or judge.get("createdAt")),
                                         r.last_t if r and r.last_kind != "start" else None) if x is not None), default=None)
                known[a["id"]] = {"name": self.names[a["id"]], "handle": coords[a["id"]], "lane": coords[a["id"]],
                                  "run": judge.get("id") or a.get("latestRunId") or "", "since": since,
                                  "branches": sorted(self.branches.get(a["id"], set()) | pushed_branches(a)), "acted": acted}
        if active != self.active:
            self.active = active
            self.save()
        for agent, w in self.workers.items():
            if w.get("status") == "ACTIVE" and not w.get("gone") and w.get("since") is not None:
                known[agent] = {"name": str(w.get("name") or ""), "handle": w.get("handle"), "lane": w.get("lane") or "",
                                "run": "", "since": w["since"], "branches": list(w.get("branches") or []),
                                "acted": w.get("updated")}
        self.read_heads(t)
        self.watch_silence(known, t)
        self.run_ends(coords, t)
        self.watch_resumes({a["id"]: a for a in agents if a.get("status") != "ARCHIVED"}, judges, live, t)
        self.flush_bursts(t)
        for run, (a, x) in current.items():
            with self.lock:
                r = self.runs.get(run)
                if r is None:
                    waiting = x["status"] == "WAITING_FOR_BACKGROUND_WORK"
                    since = min(t, epoch(x.get("updatedAt")) or t) if waiting else t
                    r = self.runs[run] = Run(a["id"], run, a.get("name") or "", last_t=since)
                r.status, r.agent_t = x["status"], epoch(a.get("updatedAt")) or 0.0
                start = not r.streaming
                r.streaming = True
            if start:
                start_stream(r)
        with self.lock:
            for run in [run for run in self.runs if run not in current]:
                self.runs.pop(run)
            runs = list(self.runs.values())
        for r in runs:
            with self.lock:
                kind = self.classify(r, t)
                handle = self.holders_all.get(r.agent)
                if kind in ("held", "stalled"):
                    self.free(r, t, kind)
                elif why := self.unread(r, t, bells.get(handle or ""), every.get(handle or "")):
                    self.free(r, t, "held_unread", why)
                elif kind in ("long_call", "stuck") and kind not in r.told:
                    r.told.add(kind)
                    if kind == "long_call":
                        self.record(kind, r, since=iso(min(r.calls.values())))
                    elif self.once(r.run + "#stuck", t):
                        quiet = max(r.last_t, r.bg_t)
                        self.record(kind, r, status=r.status, since=iso(quiet), minutes=int((t - quiet) // 60))
                if r.ended:
                    self.runs.pop(r.run, None)
        with self.lock:
            write_atomic(self.log.with_name("status.json"), body := status_bytes(snapshot(self, judged, t)))
        if self.publish:
            self.publish(body, t)


def snapshot(w: Watch, live: dict[str, dict], t: float) -> dict:
    """`status.json`, schema `verity/agent-watch-status/v1`: one entry per agent the sweep read, in categories only (no name,
    prompt, error text, branch or Slack identity), keyed in `live` by the run the watch judges it by (the run it follows, else
    its newest run that started or is CREATING, else its latest). `watch_state` is the last finding on that run (a long call
    only while it lasts), else `followed` or `ok`; `since` is when that finding began; `run_status` is the run's status when
    the watch last read it. A worker the sweep read is a row too, with no run (`run_id` "", `run_status` unknown): its
    `watch_state` is `worker_idle` while idle after the watch saw it ACTIVE, `worker_long` while that long spell lasts, else
    `ok`. A value outside the pinned enums raises, so the last valid file stays."""
    agents = []
    for run, a in live.items():
        agent, r = a["id"], w.runs.get(run)
        state_run, state, since = w.states.get(agent, ("", "", None))
        long_call = r is not None and bool(r.calls) and t - min(r.calls.values()) >= LONG_CALL_S
        if state_run != run or (state == "long_call" and not long_call):
            state, since = ("followed" if r is not None else "ok"), None
        read_run, read_status = w.status_of.get(agent, ("", ""))
        run_status = ("unreadable" if state == "unreadable" else
                      read_status if read_run == run and read_status in RUN_STATUSES else "unknown")
        agent_status = a.get("status") if a.get("status") in AGENT_STATUSES else "unknown"
        if state not in WATCH_STATES:
            raise ValueError(f"status.json: watch state {state!r} is not one of WATCH_STATES; add it there")
        agents.append({"agent_id": agent, "run_id": run, "agent_status": agent_status, "run_status": run_status,
                       "watch_state": state, "since": since, "restart_count": int(w.errors.get(agent, 0))})
    for agent, status in w.worker_status.items():
        k = w.workers.get(agent, {})
        state, since = (("worker_idle", iso(k["idle_at"])) if status == "IDLE" and k.get("idle_at") is not None else
                        ("worker_long", iso(k["since"])) if status == "ACTIVE" and k.get("long") else ("ok", None))
        agents.append({"agent_id": agent, "run_id": "", "agent_status": status if status in AGENT_STATUSES else "unknown",
                       "run_status": "unknown", "watch_state": state, "since": since, "restart_count": 0})
    return {"schema": SCHEMA, "t": iso(t), "poll_s": POLL_S, "agents": sorted(agents, key=lambda e: e["agent_id"])}


def status_bytes(snap: dict) -> bytes:
    """The same snapshot always gives the same bytes."""
    return (json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_atomic(path: Path, data: bytes) -> None:
    """Replace `path` with `data` or leave it as it was: a temporary file in its directory, fsynced, renamed over it, then the
    directory fsynced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


STATUS_TABLE = "store_private.agent_watch_status"
STATUS_ROWS = 60                # the newest sweeps kept: an hour at POLL_S
PUBLISH_DDL = f"CREATE TABLE IF NOT EXISTS {STATUS_TABLE} (t timestamptz PRIMARY KEY, body text NOT NULL);\n"


def pg_env(url: str) -> dict[str, str]:
    """libpq's environment for a `postgresql://user:password@host/db?sslmode=...` URL, so the URL never sits on a command line
    (`research.store.bucket_index.pg_env`, which this file, run alone on the pod, can't import)."""
    u = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(u.query)
    if u.scheme not in ("postgres", "postgresql") or not (u.hostname or query.get("host")):
        raise ValueError(f"not a postgres URL with a host (scheme {u.scheme!r})")
    env = {"PGHOST": u.hostname or "", "PGPORT": str(u.port or 5432), "PGDATABASE": urllib.parse.unquote(u.path.lstrip("/")),
           "PGUSER": urllib.parse.unquote(u.username or ""), "PGPASSWORD": urllib.parse.unquote(u.password or "")}
    for key, var in (("host", "PGHOST"), ("sslmode", "PGSSLMODE"), ("channel_binding", "PGCHANNELBINDING"), ("options", "PGOPTIONS"),
                     ("connect_timeout", "PGCONNECT_TIMEOUT")):
        if v := query.get(key):
            env[var] = v[-1]
    return {k: v for k, v in env.items() if v}


class Publisher:
    """Each sweep's `status.json` bytes inserted into Neon's `STATUS_TABLE` at its `t`, never overwriting: a row already at that
    `t` stays, and one whose body differs is printed as a FAULT, once per `t` (console reads a row as final once written). The
    newest `STATUS_ROWS` are kept, all through `psql` in one transaction. The table is created on first use. A failure is
    printed at most once an hour and never stops a sweep; the next sweep publishes its own snapshot."""

    def __init__(self, url: str, psql: str = "psql"):
        self.url, self.psql, self.created, self.said = url, psql, False, -RECANCEL_S
        self.faulted: dict[str, None] = {}          # the t of each differing body already printed, the newest STATUS_ROWS

    def __call__(self, body: bytes, t: float) -> None:
        lit, at = "'" + body.decode().replace("'", "''") + "'", iso(t)
        script = ((PUBLISH_DDL if not self.created else "")
                  + f"INSERT INTO {STATUS_TABLE} (t, body) VALUES ('{at}', {lit}) ON CONFLICT (t) DO NOTHING;\n"
                    f"SELECT CASE WHEN body = {lit} THEN 'same' ELSE 'differs' END FROM {STATUS_TABLE} WHERE t = '{at}';\n"
                    f"DELETE FROM {STATUS_TABLE} WHERE t NOT IN (SELECT t FROM {STATUS_TABLE} ORDER BY t DESC "
                    f"LIMIT {STATUS_ROWS});\n")
        try:
            r = subprocess.run([self.psql, "-X", "-q", "-A", "-t", "-1", "-v", "ON_ERROR_STOP=1", "-f", "-"], input=script,
                               env={**os.environ, **pg_env(self.url)}, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError(f"psql exited {r.returncode}: {r.stderr.strip()[-500:]}")
            self.created = True
        except Exception as e:  # noqa: BLE001 (a failed publish never stops a sweep)
            if t - self.said >= RECANCEL_S:
                self.said = t
                print(f"{iso(t)} status not published to {STATUS_TABLE} (said at most once an hour): {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            return
        if (r.stdout or "").split() != ["same"] and at not in self.faulted:
            self.faulted[at] = None
            while len(self.faulted) > STATUS_ROWS:
                self.faulted.pop(next(iter(self.faulted)))
            print(f"{at} FAULT: {STATUS_TABLE} already holds a different body at t={at}; left as it was, this sweep's not "
                  f"published (psql read {(r.stdout or '').strip()[:40]!r})", file=sys.stderr, flush=True)


def pump(watch: Watch, r: Run) -> None:
    """Feed the run's stream to the watch until the run ends or the watch stops following it; then the next sweep may follow
    it again."""
    def stop() -> bool:
        return r.ended is not None or watch.runs.get(r.run) is not r
    try:
        for event, data in watch.api.stream(r.agent, r.run, stop=stop):
            watch.on_event(r.run, event, data)
            if stop():
                break
    except Exception as e:  # noqa: BLE001 (a stream that fails is followed again by the next sweep)
        print(f"{iso(time.time())} stream {r.run} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        r.streaming = False


def follow(watch: Watch, r: Run) -> None:
    threading.Thread(target=pump, args=(watch, r), daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_watch.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--log", type=Path, default=Path("/data/agent-watch/events.jsonl"))
    lp.add_argument("--registry", type=Path, default=None, help="the Slack registry (registry.json): the handles")
    lp.add_argument("--routes", type=Path, default=None, help="the notes' routes.d: who holds each handle's inbox")
    lp.add_argument("--timers", type=Path, default=Path("/data/agent-watch/timers.json"),
                    help="handle -> its recurring timer's interval (s): only these are woken on idle time (2x it)")
    lp.add_argument("--wake", action="store_true", help="start a run on a coordinator ready to wake (default: record it only)")
    lp.add_argument("--once", action="store_true", help="one sweep, following no stream")
    lp.add_argument("--dry-run", action="store_true", help="send no POST; print each instead")
    lp.add_argument("--notes", type=Path, default=None, help="a research-notes clone: its recent notes' origins name workers")
    lp.add_argument("--worker-long-h", type=float, default=WORKER_LONG_S / 3600, help="hours ACTIVE before a worker is recorded")
    lp.add_argument("--git-remote", default=None,
                    help="Verity's remote, for `git ls-remote --heads` (pushes; without it nobody is judged silent)")
    lp.add_argument("--nodes", type=Path, default=Path("/data/agent-watch/nodes.json"),
                    help="the nodes whose ended runs ring their sender's handle (no file: none)")
    lp.add_argument("--launcher", type=Path, default=None,
                    help="launch_coordinator.py, for a coordinator's relaunch (default: the --registry checkout's "
                         "infra/cursor/launch_coordinator.py); relaunching also needs launch.env beside --log")
    rp = sub.add_parser("runs", help="print each node's runs that ended in the last --hours, as the watch reads them; record nothing")
    rp.add_argument("--nodes", type=Path, default=Path("/data/agent-watch/nodes.json"))
    rp.add_argument("--notes", type=Path, default=None, help="the notes clone whose machines.d names the nodes' hosts")
    rp.add_argument("--hours", type=float, default=2.0)
    ns = ap.parse_args(argv)
    if ns.cmd == "runs":
        return print_runs(ns.nodes, ns.notes, ns.hours)
    key = os.environ.get("CURSOR_API_KEY")
    if not key:
        print("agent_watch: no CURSOR_API_KEY in the environment", file=sys.stderr)
        return 2
    held = lock(ns.log.with_name("watch.lock"))
    if held is None:
        print(f"{iso(time.time())} another watch holds {ns.log.with_name('watch.lock')}", file=sys.stderr, flush=True)
        return 0
    api = Api(key)
    if ns.dry_run:
        def post(path: str, body: dict | None = None) -> dict:
            print(f"dry-run: POST {path} {json.dumps(body)}", file=sys.stderr, flush=True)
            return {"run": {"id": "dry-run"}}
        api.post = post
    publish = None
    if url := os.environ.get("AGENT_WATCH_PG_URL"):
        publish = Publisher(url) if not ns.dry_run else (
            lambda body, t: print(f"dry-run: would publish {len(body)} bytes to {STATUS_TABLE}", file=sys.stderr, flush=True))
    launcher = ns.launcher or (ns.registry.resolve().parents[3] / "infra/cursor/launch_coordinator.py" if ns.registry else None)
    watch = Watch(api, ns.log, registry=ns.registry, routes=ns.routes, timers=ns.timers, wake=ns.wake, publish=publish, notes=ns.notes,
                  worker_long_s=ns.worker_long_h * 3600, git_remote=ns.git_remote, nodes=ns.nodes, launcher=launcher)
    if ns.dry_run:
        watch.launch = lambda argv, env: (print(f"dry-run: {shlex.join(argv)}", file=sys.stderr, flush=True), (1, "", "dry-run"))[1]
        watch.launch_api = lambda key: api
    if ns.once:
        watch.sweep(lambda r: None)
        return 0
    while True:
        api.reads_until = api.clock() + SWEEP_READS_S
        try:
            watch.sweep(lambda r: follow(watch, r))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"{iso(time.time())} sweep failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
