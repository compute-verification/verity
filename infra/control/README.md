# The control pod

`vy-control-verity` is the RunPod CPU pod (`nv7h6w1pairkcu`, US-CA-2, 2 vCPU, 4 GB, 20 GB container disk) that runs the budget
guard, the resource steward and the console's store panels. Its state lives on network volume `4ue28s1c6g` (1000 GB, NFS),
mounted at `/data`. This directory holds the master copies of its boot script and its loops: deploy from here rather than
editing on the pod.

## The persistence contract

- **Everything that must survive lives under `/data`.** The container disk holds only links and packages that `post_start.sh`
  recreates. A RunPod reset wipes the container disk, `/post_start.sh` included. Any `PATCH /pods/{id}` resets the pod about
  30 s later, even one that only renames it. So never edit the pod; make a new one.
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
| Store index, every 5 min (`research data index --postgres` into Neon's `store_private`) | `store_index.sh`, deployed as `/data/infra/store_index.sh` | `/data/infra/store-index-src` (a `tools/research` tree; its `COMMIT` file names it) | `/data/infra/store-index.log` |
| Store panels, every 10 min | `/data/console/console-loop.sh` (`tools/research/console/`) | `/data/console` | `/data/console/console.log` |

The pod holds no Slack credential, so its disk alert runs elsewhere: `control_disk_alert.py` runs on infra's VM (tmux
`control-disk-alert`), reads `df` on the pod every 5 minutes over ssh, and posts in #agent-alerts at 90% and 95%, again under
85%, and when the pod has been unreadable for 30 minutes. It stops when that VM is suspended.

The steward holds a lease, so a second copy stands by instead of running. Its `--reap` reads `lane/*` tips from origin with the
read-only deploy key; if `watch.log` shows `FETCH-FAILED`, remove `--reap` until the fetch works again.

## `/data`

| Path | Linked from | Holds |
|---|---|---|
| `/data/boot` | `/post_start.sh` | `post_start.sh` |
| `/data/state/root-research` | `/root/.research` | the guard's and the steward's state |
| `/data/state/runpod` | `/root/.runpod` | the RunPod CLI config and its ssh key |
| `/data/state/config-verity` | `/root/.config/verity` | `r2.env` (R2 credentials), `store-index.env` (`STORE_INDEXER_DATABASE_URL`) and `panels.key` |
| `/data/state/secrets` | | `notes-git-credentials`, `verity-deploy` (read-only deploy key), `panels-key-push`, `known_hosts` |
| `/data/uv` | `/root/.local/bin/uv`, `/root/.local/share/uv/python`, `/usr/local/bin/python3.12` | uv and CPython 3.12 |
| `/data/steward` | `/workspace/steward` | the verity clone, the research-notes clone, the steward's store, `run.sh`, `render-src` |
| `/data/guard-src` | `/workspace/guard-src` | the guard's code |
| `/data/console` | `/workspace/console` | the console's loop, its code and its outbox |
| `/data/chats` | `/workspace/chats` | the chats archive |
| `/data/research`, `/data/research-control` | `/workspace/research`, `/workspace/research-control` | `research run` state and control worktrees |
| `/data/infra` | `/workspace/infra` | infra's pod scripts and their logs |
| `/data/backup`, `/data/tmp` | | backups and transfer bundles |

Secrets stay on the volume and never in git: this README names the files, never their contents.

## Deploy

A loop reads its script at each start, so after a copy restart the loop, or let it pick the file up at its next pass:

~~~sh
CP="research pods ssh nv7h6w1pairkcu --"
$CP "cat > /data/boot/post_start.sh" < infra/control/post_start.sh
$CP "cat > /data/steward/run.sh" < infra/control/steward-run.sh
$CP "cat > /data/infra/store_index.sh && chmod +x /data/infra/store_index.sh" < infra/control/store_index.sh
git archive HEAD tools/research | $CP "rm -rf /data/infra/store-index-src.new && mkdir /data/infra/store-index-src.new && tar -x --strip-components=2 -C /data/infra/store-index-src.new && echo $(git rev-parse HEAD) > /data/infra/store-index-src.new/COMMIT && rm -rf /data/infra/store-index-src && mv /data/infra/store-index-src.new /data/infra/store-index-src"
~~~

A change to `post_start.sh` takes effect at the next reset. To apply a new link or service now, run the new line by hand,
or the whole script: each link is `ln -sfn`, so a rerun only repeats them. It does start a second copy of every loop,
though: the steward's lease makes its copy stand by, but kill the old guard first.
