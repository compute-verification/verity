# The control pod

`vy-control-verity` is the RunPod CPU pod (`p1ugdh9hfgqe5d`, US-CA-2, cpu3g, 4 vCPU, 16 GB, 40 GB container disk) that runs the budget
guard, the resource steward, the console's store panels and every loop that watches the system (the watchers supervisor). Its state lives on network volume `4ue28s1c6g` (1000 GB, NFS),
mounted at `/data`. This directory holds the master copies of its boot script and its loops: deploy from here rather than
editing on the pod.

## The persistence contract

- **Everything that must survive lives under `/data`.** The container disk holds links and packages that `post_start.sh`
  recreates, drive's two clones under `/root/fast` (seeded from the volume), the nightly check's clone and worktrees
  `/root/fast/nightly-check` (fetched again on first use), and the local store caches `/root/drive-store`,
  `/root/watchers-preflight-store`, `/root/watchers-nightly-store`, `/root/watchers-challenger-store` and
  `/root/watchers-labels-pack-store` (SQLite on NFS was too slow). A RunPod reset wipes the
  container disk, `/post_start.sh` included. Any `PATCH /pods/{id}` resets the pod about 30 s later, even one that only renames
  it. So never edit the pod; make a new one.
- **To resize, move to a new pod.** The services need more than 4 GB: at 4 GB the cgroup sat at its cap and killed processes.
  - create it with the RunPod REST API under the same name, on volume `4ue28s1c6g` mounted at `/data`, with the old pod's image
    and `PUBLIC_KEY`; it boots with sshd only;
  - `drive.sh stop` on the old pod, then copy the store caches above to `/data/handover/<old id>/`, each `catalog.sqlite` with
    sqlite3's `.backup`;
  - stop the old pod (never start it again: its `/post_start.sh` would start a second guard), copy the caches into the new
    pod's `/root`, and run `sh /data/boot/post_start.sh`, then `drive.sh start`;
  - `research pods register vy-control --pod-id NEW --project verity --replace --any-name`, and terminate the old pod, since two pods of one
    name make `research pods ssh vy-control-verity` and the guard host's lookup refuse.
- **After a reset, run `sh /data/boot/post_start.sh` once.** It relinks every path below, installs rsync, sqlite3, git and psql when
  they're missing, and starts each service detached (`setsid -f`). It also puts `/post_start.sh` back as a link to itself.
- **A service is added in a PR here.** Its files go under `/data/<name>`; its link and its start line go into
  `post_start.sh`. A process started by hand on the pod, or a file kept only on the container disk, is lost at the next reset.
- **This is the only pod that may run the guard** (`budgets.GUARD_HOST`). `post_start.sh` starts it with the other services, so
  a restart never leaves the budgets unguarded. Never run two guards.

## Services

| Service | Started by | Code | Log |
|---|---|---|---|
| Budget guard (`research pods guard --supervise`) | `post_start.sh` | `/data/guard-src` (a `main` tree; its `COMMIT` file names it) | `/root/.research/pods/guard-budgets.log` |
| Resource steward (`research notes watch ... --lease --standby`) | `steward-run.sh`, deployed as `/data/steward/run.sh` | `/data/steward/verity` (a clone of this repo) | `/data/steward/watch.log` |
| `/tmp` sweep, hourly | `tmp_sweep.sh`, deployed as `/data/infra/tmp_sweep.sh` | the script | `/data/infra/tmp-sweep.log` |
| Drive: `research queue drive`, ci's landing automation, one tick every 2 min as lane ci, flags from main's `tools/verity/check/drive.toml`, their one source, plus `/data/drive/args` (only the flags its `command_line` lets override, `--on` and `--first`; a `--config FILE` there is refused unless every key it sets has main's value); sources `r2.env` and `relay.env` (Slack); posts as ci's handle with `RESEARCH_AGENT_URL` from `/data/drive/agent-url` (ci's coordinator URL, unless the caller sets one); the verity-drive App's installation token comes from `gh_app_token.sh` (`/data/state/secrets/verity-drive.pem`), reused 45 min from memory (`--cached`): both clones' credential helper is `gh_app_token.sh git`, which hands it to git when it pushes `next`, `checked` and `main`, and `gh` is `/data/drive/ghw/gh`, which sets `GH_TOKEN` per call before running `/data/drive/bin/gh`, so a tick longer than the token's hour still pushes; nodes as `research run --on` reaches them, from the steward's notes clone's `machines.d` | `drive.sh`, deployed as `/data/infra/drive.sh`; no tick while `/data/drive/stopped` exists: ci runs `/data/infra/drive.sh stop` (waits for the tick in progress), `start`, `status`, `once --dry-run`, `deploy SHA|main`, or `queue ARGS` (`research queue ARGS` between ticks with drive's env and state: the lander's `queue eject N --by ci --reason ...` runs here; `queue promote`, a loop's every minute or so, and ci's `queue baseline SHA --by ci` fast-forward main beside a tick, without its lock or a checkout, and write nothing of drive's) | `/root/fast/verity` on the container disk (a checkout of origin/main it fetches each tick, at the landed commit `/data/drive/deployed` pins when it names one (`drive.sh deploy SHA`; `drive.sh deploy main` follows main's tip again), else at main's tip; its `tools/research`), seeded from `/data/drive/verity` when the container is new | `/data/drive/drive.log`; `verity.state.json` in `/root/.research/queue/`, whose `verity` is a symlink to the queue's clone in `/root/fast/queue-verity` (seeded from `verity.volume` beside it); the store's cache in `/root/drive-store`, its catalog there and its `objects/` and `attempts/` links into `/data/drive/store` (`store_links`); its temp dirs (`TMPDIR`, the trains' `check-lean-deps-*`) in `/data/drive/tmp`, and the trains' checkouts (`TRAIN_TREES`) in `/root/fast/trains` on the container disk, each removed an hour after its last change once nothing runs in it; `RESEARCH_EXECUTOR_ENQUEUE` names the task executor's enqueue command (`/root/executor/infra/executor.py`), which `research executor enqueue` runs |
| Promote, every minute: `drive.sh queue promote` fast-forwards main to a `checked` commit whose after-landing run passed; skipped while `/data/drive/stopped` exists | `promote-loop.sh`, deployed as `/data/infra/promote-loop.sh`, started by `post_start.sh` | the script | `/data/drive/promote.log` |
| Store index, every 5 min (`research data index --postgres` into Neon's `store_private`, and `--public` into `store_public` once `store-export.env` exists) | `store_index.sh`, deployed as `/data/infra/store_index.sh` | `/data/infra/store-index-src` (a `tools/research` tree; its `COMMIT` file names it) | `/data/infra/store-index.log` |
| Fixture export, every 5 min (`research data export` of main's `fixtures/artifacts.json`, then next's (admitted PRs, so a train finds the fixtures its PRs register), to the public export, so nobody exports a new fixture by hand; the relay's `fixtures-public` monitor alerts on one still missing) | `fixtures_export.sh`, deployed as `/data/infra/fixtures_export.sh` | `/data/infra/fixtures-export-repo` (a checkout of origin/main it fetches each pass with the deploy key) | `/data/infra/fixtures-export.log` |
| Store GC, every 6 h (`research data gc`, the store README §7.4: blobs only uncited artifacts past their kind's TTL hold; plans only until `/data/store-gc/apply` exists, then marks and deletes; skips a pass while the notes clone or the fixtures checkout was last fetched over 6 h ago) | `store_gc.sh`, deployed as `/data/infra/store_gc.sh` | `/data/infra/store-gc-src` (a `tools/research` tree; its `COMMIT` file names it) | `/data/infra/store-gc.log`; each pass's report in `/data/store-gc/reports/`, the marks in `marks.json`, the manifests, labels and attempts it read in `cache/` |
| Store panels, every 10 min | `/data/console/console-loop.sh` (`tools/research/console/`) | `/data/console` | `/data/console/console.log` |
| Agent watch: follows every running cloud agent's run stream and cancels and continues a run held 15 min after a finished tool call or stalled (RUNNING with nothing streamed and its agent's `updatedAt` still for 30 min: a hung call or a suspended VM), restarts an ERROR run (10 min apart, 3 in a row), resumes with the run's result an agent idle 10 min after a `research run` it must act on ended (`idle_after_run`; a background wait that refuses it, or a wake, is cancelled first), relaunches a coordinator it can't resume from its newest handover note and a worker nobody resumed from its branch and newest note (`relaunch`; off until `/data/agent-watch/launch.env` holds a launch key), records an agent out of a turn (IDLE, or ERROR where the platform reports it) that changed 5 min after its latest run finished, with no newer run, has sat as long since, and whose conversation ends on a message it wrote nothing after (`dropped`: a turn a Task resume ran is no run the API lists and moves `updatedAt` the same way, so the conversation, read once per such change, tells the two apart; one the API won't read is tried every 10 min and recorded unchecked after 60), and records a coordinator ready to wake (a doorbell pending 10 min, or, for a handle in `agent_watch_timers.json`, no run for twice its timer's interval; with `--wake`, on while `/data/agent-watch/wake` exists, it starts a run), and records a cloud worker (a Task subagent) that went from ACTIVE to IDLE, unless its parent got its Task result as it stopped, or has been ACTIVE 6 h (`worker_idle`, `worker_long`; record only: only its coordinator can resume it, so its lane is its launcher's coordinator's, however many workers up that is (the agent whose Task call the watch saw launch or resume it, else its registration's `launched_by`, else its registration's lane, else the agent its brief names; one rule for a worker and for a run's sender), and one none of these places is recorded once as `unplaced`, rung to infra, its alerts then ringing no one; several of one parent's within 15 min are one `workers_failed`, naming each and the wake-up it says it set, from `GET /v0/agents/{id}/conversation`), a worker ACTIVE 90 min with no note and no push to its branches, or a coordinator in one turn (its run RUNNING, read again before it is recorded) with neither and no assistant text for 60 min (`silent`, twice at most per spell; a coordinator's text is its run's stream's, else what a read of its conversation finds new), and each `research run` that ended on a node of `nodes.json`, for its sender's handle (`run_ended`; a sender no handle owns whose name names one lane (the issue key it starts with when one lane has its Linear team, else a lane's handle it starts with, else its issue keys further in when they are all one team's) becomes that lane's worker, written to `workers.d` with `placed_by: name` or `placed_by: handle`, and its lane is rung, a lane being a registry handle with a Linear team that a cloud agent holds; a read's runs whose sender is still nobody's are one record, rung to infra once for each sender) (`verity_ops/agent_watch.py`) | `agent_watch.sh`, deployed as `/data/infra/agent_watch.sh`; waits for `/data/agent-watch/cursor.env` (root, 0600, `CURSOR_API_KEY=`) | `/data/infra/agent_watch.py` (a copy of `infra/ops/verity_ops/agent_watch.py`); coordinators from `/data/infra/fixtures-export-repo`'s registry and `/data/steward/research-notes/routes.d`, recurring timers from `/data/agent-watch/timers.json` (a copy of `agent_watch_timers.json`), doorbells from the relay's `/data/agent-watch/doorbells.json`; workers from the `origin:` of `/data/steward/research-notes/lanes/*/*.md` modified in the last 12 h (`--notes`), the `task` calls of every run it follows, and `/data/agent-watch/workers.d/*.json` (`{agent_id, lane, launched_by, branch}`, a coordinator's; `placed-<id>.json`, with `placed_by`, the watch's own, which any other source overrides); pushes from `git ls-remote --heads` with `/data/state/secrets/verity-deploy`; the remote's open PRs from GitHub with `relay.env`'s `GH_TOKEN` (`--gh-env`, the file read at each read: a push to a PR's branch counts for every agent its body names, and for an agent no branch is known for, a push to a PR whose `Lane` line names its lane); nodes from `/data/agent-watch/nodes.json` (a copy of `agent_watch_nodes.json`, read each sweep: names and ssh options), their hosts from the notes clone's `machines.d` | `/data/agent-watch/watch.log`; findings in `/data/agent-watch/events.jsonl`; each sweep's snapshot in `/data/agent-watch/status.json` (schema `verity/agent-watch-status/v1`, read by console); what must outlive a restart in `/data/agent-watch/state.json`, the branch heads it saw in `heads.json` and the nodes' runs already seen in `run_ends.json`; with `/root/.config/verity/store-index.env` present, the same snapshot in Neon's `store_private.agent_watch_status` (newest 60) |
| Cloud watch, every 15 min: each loop above's success marker, R2's size and growth by top-level prefix and the Lean cache's last publish, Neon, spend against budgets.toml's lines, the GitHub rate limit, jobs nobody claims and the Vercel site's endpoints; posts only anomalies in #agent-alerts naming infra's user group (once when one opens, once when it clears) and a digest at 15:00Z, naming each check a missing key leaves unmonitored (`cloud_watch.py`'s docstring) | `cloud_watch.sh`, deployed as `/data/infra/cloud_watch.sh`; sources `r2.env`, `relay.env` and `store-index.env` | `/data/infra/cloud_watch.py` and `/data/infra/cloud_watch.toml` (copies of this directory's) | `/data/cloud-watch/watch.log`; state in `/data/cloud-watch/state.json` |
| Control disk relay, every 5 min: the root disk and the steward's notes sync | `control_disk.sh`, deployed as `/data/infra/control-disk/run.sh`; sources `relay.env` | `/data/infra/control-disk/control_disk_alert.py` (a copy of this directory's), `research-src` (a `tools/research` tree) and `registry.json` beside it | `/data/infra/control-disk/relay.log`; state in `seen.json` |
| Watchers: one supervisor (`watchers.py`) for every loop that watches the system and used to run in tmux on an agent VM, each in its own process group, started again a minute after it exits, waiting while a credential or seeded state it `needs` is missing: `vm-stop` (`vm_stop.py`: the H200 idle stop and the deadline and lease backstops through the Nebius API), `fleet` (`fleet.py loop`), `monitors-relay` (`research monitors relay`: the relay-side monitors and CPU node 1's watchdog alerts; while node 2 is stopped it reads no node-side monitors and leaves `node-heartbeat` out, as `watchers.toml` says), `agent-watch-relay`, `preflight-probe`, `nightly-check` (`nightly_check.py`: once a day from 09:00Z, one uncached check of `main`'s tip, `check.py --record --no-cache` with every path counted as changed, on the node and in the pool its arguments name; it reads the run's result, compares its verdicts part by part with the recorded check's of the same commit (the store's attempts there: each suite, Lean package and step, with its key and Lean report), posts the outcome and the comparison in #agent-alerts, ringing infra unless it passed and infra and ci when a part disagrees, and labels `commit:<sha>` with `nightly_check`; the relay's `nightly-check` monitor rings infra when its state shows no outcome for 36 h), top's `top-asks` and `top-sweep` (`top_watch.py`), `challenger-publish` (the challenger host's ledger and ended sessions to the store, by the `publish.sh` installed in `/data/watchers/challenger/`), `ci-priority-from-goals` (every 10 min, the landing order's priority list `/data/drive/priority` rebuilt from the goals its PRs serve, or only proposed while it runs dry, by ci's script installed as `/data/drive/bin-ci/priority_from_goals_loop.sh`), and `labels-pack` (`research data labels-pack`: every hour it pulls the store's new labels into `/root/watchers-labels-pack-store` and replaces the labels pack on the remote); a one-off timed job is an executable `/data/watchers/extra.d/NAME.sh`, done once it exits 0. Every minute it writes `status.json` and upserts it into Neon's `store_private.control_heartbeat`, which the dead-man's switch (`deadman/`, a Vercel cron in console's team) reads, posting in #agent-alerts when the pod stops | `watchers.sh`, deployed as `/data/watchers/run.sh`; the loops and their credentials in `/data/watchers/watchers.toml` (a copy of `watchers.toml`), the stops in `/data/watchers/vm_stop.toml` | `/data/watchers/src` (a `git archive` of a pushed commit; its `COMMIT` file names it); ssh through `bin/ssh` with `watchers_ssh_config` and the RunPod key | `/data/watchers/supervisor.log`; each loop's in `/data/watchers/log/NAME.log`; their states in `/data/watchers/status.json` (schema `verity/watchers-status/v1`) |

The disk alert, `control_disk_alert.py`, ran on infra's VM (tmux `control-disk-alert`) while the pod held no Slack credential.
The pod has `relay.env` (`SLACK_BOT_TOKEN`, `GH_TOKEN`), and a copy runs on the pod from
`/data/infra/control-disk/run.sh`, which `post_start.sh` starts (the cloud watch
alerts when its `seen.json` goes stale). It reads `df` every 5 minutes and posts in #agent-alerts at 90% and 95%, again under
85%, and when the pod has been unreadable for 30 minutes. The same pass alerts when the steward's notes sync stalls (HEAD
detached, an `index.lock` older than 10 minutes, or a change unpushed for an hour, on two reads in a row) and notes when it
flows again. The agent watch's findings reach Slack through `agent_watch_relay.py`, the watchers' `agent-watch-relay` loop on this pod,
which posts each new line of `events.jsonl` in #agent-alerts every 2 minutes, leading with top's handle and ringing top's inbox for what the watch leaves for a person (erroring, stuck, a refused cancel, held again, a run unreadable for an hour); a failure the watch acted on (held, stalled, crashed, idle after its run ended, relaunched), or a dropped message, leads with the record's `owner` (top for a coordinator, the launcher's handle for a worker) and rings that inbox, a dropped message's line saying Cursor shows the agent in ERROR only where the record's `status` is; a worker's alert and a silent worker lead with the record's handle and ring that inbox, and a silent coordinator rings top; an ended run leads with its sender's handle, and its doorbell is `research msg ring-run`, unless its sender was in a turn (ACTIVE) at its end, when it is posted unrung (one per run, with `--to` that handle for all but a coordinator's own run); a worker's run that ended rc 0 is posted but rings no one, since the watch resumes the worker. The loop runs from the deployed tree (`/data/watchers/src`, main's `research`); a record it can't render goes to infra as its raw JSON; the watch frees runs without it.

A coordinator whose duty timer recurs opts in to the watch's idle rule by adding its handle and the timer's interval in seconds
to `agent_watch_timers.json` (`{"infra": 900}`) in a PR; infra copies the file to `/data/agent-watch/timers.json`, which the
watch reads each sweep, so no restart is needed. It is then woken (or, with `--wake` off, recorded) once nothing has run for
twice that interval. A handle that isn't listed (comms wakes on its inbox, issue events and one-shot timers) is woken only by
a doorbell pending in its inbox, never on idle time alone.

Drive runs on this pod. `post_start.sh` starts its loop, and the loop ticks only while `/data/drive/stopped`
is absent, so a reset brings drive back in whichever state ci left it. Two drives must never tick against one `next`: ci
stops its own loop before `drive.sh start` here. Its checkout was cloned over https with `--filter=blob:none`, with
`notes_sync.CREDENTIAL_HELPER`'s helper for `GH_TOKEN` as its local `credential.helper`.
`drive.sh start` refuses until a dry-run push of `next` with that token goes through: the token is verity-drive's (App
5385601, installation 173153984), minted by `gh_app_token.sh` with Contents and Pull requests write on this repository only.

The steward holds a lease, so a second copy stands by instead of running. Its `--reap` reads `lane/*` tips from origin over https,
its clone's local `credential.helper` being `gh_app_token.sh git`; if `watch.log` shows `FETCH-FAILED`, remove `--reap` until the fetch works again.

## `/data`

| Path | Linked from | Holds |
|---|---|---|
| `/data/boot` | `/post_start.sh` | `post_start.sh` |
| `/data/state/root-research` | `/root/.research` | the guard's and the steward's state |
| `/data/state/runpod` | `/root/.runpod` | the RunPod CLI config and its ssh key |
| `/data/state/config-verity` | `/root/.config/verity` | `r2.env` (R2 credentials), `relay.env` (`SLACK_BOT_TOKEN`, `GH_TOKEN`), `store-index.env` (`STORE_INDEXER_DATABASE_URL`), `store-export.env` (`RESEARCH_EXPORT_BUCKET`; optionally a key pair of its own, `RESEARCH_EXPORT_ACCESS_KEY_ID` / `RESEARCH_EXPORT_SECRET_ACCESS_KEY`, which must be able to list the bucket, else `store_index.sh` uses `r2.env`'s; optionally `RESEARCH_EXPORT_ENDPOINT`) and `panels.key` |
| `/data/state/secrets` | | `verity-drive.pem` (the App key `gh_app_token.sh` signs with: every clone of verity here fetches over https with its token; GitHub refuses the old deploy key `verity-deploy`), `panels-key-push`, `known_hosts`, `msg.env` (`SLACK_BOT_TOKEN`, a copy of `relay.env`'s, for `msg.sh`; root, 0600) |
| `/data/slack-app` | | `token.json`: the Slack app's configuration token pair, the one copy, which `research msg slack-app` rotates in place (root, 0600; the directory 0700) |
| `/data/uv` | `/root/.local/bin/uv`, `/root/.local/share/uv/python`, `/usr/local/bin/python3.12` | uv and CPython 3.12 |
| `/data/steward` | `/workspace/steward` | the verity clone, the research-notes clone, the steward's store, `run.sh`, `render-src` |
| `/data/guard-src` | `/workspace/guard-src` | the guard's code |
| `/data/drive` | | drive's checkout (`verity`), `bin/gh`, `gh-config`, `args`, the stop file `stopped` and `drive.log`; `bin-ci/` (ci's scripts, which the watchers' `ci-priority-from-goals` runs) and `priority`, the landing order's priority list |
| `/data/console` | `/workspace/console` | the console's loop, its code and its outbox |
| `/data/chats` | `/workspace/chats` | the chats archive |
| `/data/research`, `/data/research-control` | `/workspace/research`, `/workspace/research-control` | `research run` state and control worktrees |
| `/data/infra` | `/workspace/infra` | infra's pod scripts and their logs; `msg-repo` and `msg-notes`, `msg.sh`'s checkout and notes clone |
| `/data/watchers` | | the watchers supervisor: `run.sh`, `watchers.toml`, `vm_stop.toml`, `src`, `log/`, `status.json`, each loop's state (`vm_stop.json`, `fleet/`, `monitors/`, `agent-watch-offset`, `preflight-probe.json` with `preflight-tree` (its store index is `/root/watchers-preflight-store`, on the pod's own disk), `nightly-check.json` (schema `verity/nightly-check/v1`: the night in progress and the last outcome) with `nightly-check.launch.log` (the last launch's output) and `nightly-check/tmp` (the launch's temp dirs; its clone, worktrees and store index are on the pod's own disk, `/root/fast/nightly-check` and `/root/watchers-nightly-store`), `top-asks.json`, `top-sweep.json` with `linear-audit.jsonl` and `goals/` (the Goals page each hour, which `research velocity` reads), `challenger/` with `publish.sh` and its `state`), `repo` (a clone without a working tree, for `research deploy drift`'s `RESEARCH_DEPLOY_REPO`; drift fetches it each run), `bin/gh` (GitHub's CLI release, checksum-checked, on every loop's `PATH`; it reads `relay.env`'s `GH_TOKEN`), `extra.d/`, `known_hosts`; `notion.env` (`NOTION_API_KEY`, for `top-asks`), `goals.key` (a site key with `events:read`, one line, for `top-sweep`'s read of the Goals page) and `nebius-stop.env` (a Nebius account that may stop instances, for `vm-stop`) once placed (root, 0600) |
| `/data/backup`, `/data/tmp` | | backups and transfer bundles |

Secrets stay on the volume and never in git: this README names the files, never their contents. To see which variables an env
file here sets, run `research pods ssh vy-control-verity -- python3 - FILE... < tools/agent-guard/envnames.py` from a checkout: it
prints the names and nothing of a file that isn't an env file, so `'/root/.config/verity/*'`, which also matches `panels.key`, is
safe to pass. Never `cat`, `sed` or `cut` these files, which the shell hook refuses; check a key with `stat`, `sha256sum` or
`openssl pkey -in FILE -pubout`.

## Deploy

A loop reads its script at each start, so after a copy restart the loop, or let it pick the file up at its next pass:

~~~sh
CP="research pods ssh p1ugdh9hfgqe5d --"
$CP "cat > /data/boot/post_start.sh" < infra/control/post_start.sh
$CP "cat > /data/steward/run.sh" < infra/control/steward-run.sh
$CP "cat > /data/infra/store_index.sh && chmod +x /data/infra/store_index.sh" < infra/control/store_index.sh
$CP "cat > /data/infra/store_gc.sh && chmod +x /data/infra/store_gc.sh" < infra/control/store_gc.sh
$CP "cat > /data/infra/fixtures_export.sh && chmod +x /data/infra/fixtures_export.sh" < infra/control/fixtures_export.sh
$CP "cat > /data/infra/cloud_watch.sh && chmod +x /data/infra/cloud_watch.sh" < infra/control/cloud_watch.sh
$CP "cat > /data/infra/cloud_watch.py" < infra/control/cloud_watch.py
$CP "cat > /data/infra/cloud_watch.toml" < infra/control/cloud_watch.toml
$CP "/data/infra/cloud_watch.sh once --dry-run --digest"     # what it would post, posting nothing and writing no state
$CP "cat > /data/infra/gh_app_token.sh && chmod +x /data/infra/gh_app_token.sh" < infra/control/gh_app_token.sh
$CP "cat > /data/infra/gh_notes_token.sh && chmod +x /data/infra/gh_notes_token.sh" < infra/control/gh_notes_token.sh   # the same App, scoped to research-notes: the steward's notes clones
$CP "cat > /data/infra/drive.sh.new && chmod +x /data/infra/drive.sh.new && mv /data/infra/drive.sh.new /data/infra/drive.sh" < infra/control/drive.sh   # a new inode: the running loop keeps reading its own
$CP "/data/infra/drive.sh once --dry-run"     # one tick that pushes, launches and posts nothing
$CP "cat > /data/infra/agent_watch.sh && chmod +x /data/infra/agent_watch.sh" < infra/control/agent_watch.sh
$CP "cat > /data/infra/agent_watch.py" < infra/ops/verity_ops/agent_watch.py
$CP "cat > /data/agent-watch/timers.json" < infra/control/agent_watch_timers.json
$CP "cat > /data/agent-watch/nodes.json" < infra/control/agent_watch_nodes.json
$CP "python3.12 /data/infra/agent_watch.py runs --notes /data/steward/research-notes --hours 2"   # each node's ended runs, recording nothing
git archive HEAD tools/research | $CP "rm -rf /data/infra/store-index-src.new && mkdir /data/infra/store-index-src.new && tar -x --strip-components=2 -C /data/infra/store-index-src.new && echo $(git rev-parse HEAD) > /data/infra/store-index-src.new/COMMIT && rm -rf /data/infra/store-index-src && mv /data/infra/store-index-src.new /data/infra/store-index-src"
$CP "cat > /data/infra/control-disk/run.sh && chmod +x /data/infra/control-disk/run.sh" < infra/control/control_disk.sh
$CP "cat > /data/infra/control-disk/control_disk_alert.py" < infra/control/control_disk_alert.py
$CP "/data/infra/control-disk/run.sh --once --dry-run"      # one pass, posting nothing and writing no state
$CP "cat > /data/infra/msg.sh && chmod +x /data/infra/msg.sh" < infra/control/msg.sh
~~~

`monitors-relay`'s `drift` reads git history the tree lacks from `/data/watchers/repo`, cloned once over https with `gh_app_token.sh` as its credential helper, and
`fixtures-public`, `unverified-main` (which also reads drive's `/root/.research/queue/verity.state.json`), the failed-check posts and
the router call `gh`, installed once into `/data/watchers/bin`:

~~~sh
$CP "git clone -q --no-checkout -c credential.helper='!/data/infra/gh_app_token.sh git' \
  https://github.com/compute-verification/verity.git /data/watchers/repo"
V=2.102.0; $CP "set -e; d=\$(mktemp -d /data/tmp/gh.XXXX); cd \$d; B=https://github.com/cli/cli/releases/download/v$V; \
  curl -fsSLO \$B/gh_${V}_linux_amd64.tar.gz; curl -fsSLO \$B/gh_${V}_checksums.txt; grep ' gh_${V}_linux_amd64.tar.gz\$' gh_${V}_checksums.txt | sha256sum -c -; \
  tar --no-same-owner -xzf gh_${V}_linux_amd64.tar.gz gh_${V}_linux_amd64/bin/gh; install -D -m 0755 gh_${V}_linux_amd64/bin/gh /data/watchers/bin/gh; cd /; rm -rf \$d"
~~~

The watchers run from a pushed commit's tree. Deploy it, keeping the tree it replaces, then restart the supervisor (`run.sh`
starts it again 10 s later, and it stops every loop first). The new `watchers.toml` replaces the pod's only once `watchers.py
check --replaces` has passed on it: the check prints each loop resolved, then each loop that is new or whose argv, env,
env_files or needs differ from the pod's, and refuses a config that lacks a loop the pod's has, which the restart would stop.
Such a loop was added on the pod alone: add it to `watchers.toml` in a PR and deploy that commit, or, to stop it on purpose,
add `--allow-drop NAME` to the check. The last line restarts nothing while a config that didn't pass the check is left:

~~~sh
git archive HEAD tools/research infra/control infra/ops/verity_ops .agents/skills/using-slack | $CP "set -e; cd /data/watchers; \
  rm -rf src.new; mkdir src.new; tar -x -C src.new; echo $(git rev-parse HEAD) > src.new/COMMIT; \
  if [ -e src ]; then rm -rf src.prev; mv src src.prev; fi; mv src.new src"
$CP "cat > /data/watchers/watchers.toml.new && PYTHONPATH=/data/watchers/src/tools/research/src python3.12 \
  /data/watchers/src/infra/control/watchers.py check --config /data/watchers/watchers.toml.new --replaces /data/watchers/watchers.toml \
  && mv /data/watchers/watchers.toml.new /data/watchers/watchers.toml" < infra/control/watchers.toml
$CP "cat > /data/watchers/vm_stop.toml" < infra/control/vm_stop.toml
$CP "cat > /data/watchers/run.sh && chmod +x /data/watchers/run.sh" < infra/control/watchers.sh
$CP 'if [ -e /data/watchers/watchers.toml.new ]; then echo "not restarted: watchers.toml.new did not pass the check, so the supervisor still runs watchers.toml as it was"; exit 1; fi; kill "$(cat /data/watchers/supervisor.pid)"'
~~~

A config that doesn't pass the check stays beside the pod's as `watchers.toml.new`: the pod's `watchers.toml` is as it was, and
the last line says so and restarts nothing, as it does on every later run while that file is there. The rest of the block has
run all the same: `src` is the new tree, which a loop runs from whenever the supervisor starts it again, `src.prev` is the tree
that run of the block replaced, and `vm_stop.toml`, read every pass, and `run.sh` are the new ones. To finish the deploy, run
the block again from the commit that keeps the loop, or the check's line with `--allow-drop NAME` added and then the last line:
a check that passes moves `watchers.toml.new` into place. To give the deploy up, remove the file and put the tree back:

~~~sh
$CP "rm /data/watchers/watchers.toml.new"
$CP "cd /data/watchers && [ -d src.prev ] && rm -rf src && mv src.prev src"
~~~

A loop that takes over from a copy elsewhere starts where that copy is, never posting everything again: the relay with `research
monitors relay ... --seed` (its loop's arguments), the agent-watch relay once `/data/watchers/agent-watch-offset` holds the
count of complete lines in `/data/agent-watch/events.jsonl` (`grep -c '' events.jsonl`, less one for a last line without its
newline), and top's loops with `top_watch.py asks --seed` and `sweep --seed`, whose state files they wait for. A seed runs in its
loop's environment and directory with `watchers.py exec`:

~~~sh
W="PYTHONPATH=/data/watchers/src/tools/research/src python3.12 /data/watchers/src/infra/control/watchers.py exec"
$CP "$W top-asks -- python3.12 -s /data/watchers/src/infra/control/top_watch.py asks --state /data/watchers/top-asks.json --seed"
~~~

The nightly check needs no seed and no clone made by hand: its first pass writes `/data/watchers/nightly-check.json`, and
its first night makes its clone under `/root/fast/nightly-check`, as does the first after a reset. A first pass after 09:00Z
starts no night that day, so the first night is the next day's. Its log's first line says the command it launches; each night
adds a `launch:` line, the run's id, the outcome and the comparison (`compared:`), and the state file holds the night in
progress, with its comparison, and the last outcome. Its
arguments are the loop's in `watchers.toml` (`watchers.py check` prints them); run with `--dry-run` added, under `$W
nightly-check --`, they say whether a night would start now and start nothing:

~~~sh
$CP "tail -n 20 /data/watchers/log/nightly-check.log; cat /data/watchers/nightly-check.json"
~~~

A change to `post_start.sh` takes effect at the next reset. To apply a new link or service now, run the new line by hand,
or the whole script: each link is `ln -sfn`, so a rerun only repeats them. It does start a second copy of every loop,
though: the steward's lease makes its copy stand by, but kill the old guard first.

## The door

Root here is for an admin key that no agent holds. Every other key belongs to the account `research` and is forced to
`vy-control-gate`, which runs one verb with checked arguments, as root through one sudo rule, or refuses: the budgets guard's
state and a wake for any agent (`research pods create`, `research run --on`), RunPod's list, get, create and terminate for a
machine that holds no RunPod key, and for ci's own key `drive.sh`'s listed forms (`... -- drive queue baseline SHA --by ci`).
The gate's docstring names the verbs, and a request is never given to a shell. `research` sends verbs once the machine
registry's entry for `vy-control-verity` is that account (`provider = "ssh"`, `user = "research"`, written with `research pods
register --host --port --user`); until then it asks root's shell as before and says so (`tools/research/src/research/pods/door.py`).

`door_setup.sh`, deployed as `/data/infra/vy-control-setup`, makes the account, its sudo rule and the gate's copy in
`/usr/local/sbin` from two files of public keys on the volume: `/data/boot/door/gate-keys` (`LANE TYPE BASE64` a line, LANE `-`
for the key every agent holds, `ci` for ci's) and `/data/boot/door/admin-keys` (root's). Run without a flag it adds the admin
keys to root's; `--lock`, refused before that, leaves root the admin keys only, and `--unlock` puts the earlier file back.
`post_start.sh` runs it after a reset, when the image has given root the pod's `PUBLIC_KEY` again, so a locked door locks
again; until that run the key the pod was created with is root. Once the door is locked, every command in this file that
reaches root (`$CP`, `msg.sh`) is run with the admin key.

## Running `research msg` as root

`msg.sh` runs `research msg ARGS` on the pod for a handle holder who needs a credential that lives only here: the Slack app's
configuration token pair, `/data/slack-app/token.json` (`slack-app export|diff|apply`), and the bot token in
`/data/state/secrets/msg.env` (`chat open|close|list`). It
fetches origin's `$REF` (default `main`; `REF=pull/N/head` runs a PR's head) into its own sparse checkout,
`/data/infra/msg-repo`, on every call, and passes the credentials only through the environment. Its notes are its own clone,
`/data/infra/msg-notes` (made once from the steward's, with the same origin and credential helper, and pulled on every call), so
`chat open`'s push of `people.d/NAME.json` never writes the clone the budgets guard reads. A command that writes the registry
prints the registry's diff for the caller to commit in a registry PR.

~~~sh
research pods ssh vy-control-verity -- env RESEARCH_AGENT_URL=https://cursor.com/agents/<your bc id> \
  /data/infra/msg.sh chat open luke --user U0CA897T7MK --start-here <the person's start page URL> --dry-run
research pods ssh vy-control-verity -- /data/infra/msg.sh slack-app diff
~~~
