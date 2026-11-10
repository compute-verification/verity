# MACHINES: are our machines durable? A read-only audit, and a ranked list of fixes.

**Why.** Daniel, Tue 10 Nov 3:35 PM PST: "look after our machines, make sure they're durable". top wants facts before it orders anything of infra. **You change nothing on any machine.**

**The machines.** The control pod `vy-control-verity` (RunPod; it runs the queue loop that lands `main`, the watchers, the agent watch, the Slack router's relay). The Nebius CPU check nodes `vy-nebius-cpu-1` to `-13`. The GPU nodes and pods (`vy-nebius-1` to `-3`, `vy-dc-h200-*`, proofs' RunPod pod of 8 H200s, the H8 pod if it exists). The auditor's hosts (`vy-challenger-1`, `vy-auditor-1`, `vy-auditor-2`). Find the true list yourself: `python3 -m research pods list`, `tools/research/src/research/pods/nebius/`, `infra/cluster/`, `infra/README.md`.

**For each machine, read (cheap commands only; the check nodes are running tonight's candidates, so nothing that takes cores, memory or disk I/O for more than a second or two):**
- up or down, uptime, cores, memory total and used, load.
- every disk and volume: size, used, percent, inodes percent; the five largest directories two levels down under the work and cache roots, by `du` with a depth limit and a timeout of 20 s (skip one that times out and say so).
- out-of-memory kills and disk or filesystem errors in the kernel log for the last 24 h (count and newest time).
- processes older than 24 h that are not daemons (name, age, owner by path); orphaned worktrees and scratch trees (count and total size).
- what holds state that exists nowhere else: for a check node, runs not yet preserved in the evidence store; for the control pod, the queue's state, logs and the pod-only scripts and settings under `/data` and `/root` (name them; which are in git, which exist only there).
- how it comes back: is it rebuilt from code (`research deploy install`, a boot script) or by hand, and is that written down. Do not test it.

**Then answer, for top:**
1. **Single points of failure, ranked by what stops if each dies now** (landing, checks, agents' messages, a lane's runs), with how long it would take to bring back and whether that is written down or a guess.
2. **What is close to full or failing today** (over 80% disk or inodes, memory pressure, a growing directory nobody cleans), with the owner by path.
3. **What is paid for and idle** (a table: machine, $/h if you can find it, load, whose).
4. **Ten fixes at most, ranked by risk removed for the work it takes,** each with its owner (infra, ci, a lane), and which top could order tonight without stopping tonight's landing.

**Deliver.**
- Your note, with the tables.
- One Linear issue's text for infra's team in a file under your note's directory (title, the ten fixes, the evidence). **Do not file it**; top files it.
- A first cut in your final message **by 01:15Z Wed (5:15 PM PST)**. If the full audit takes longer, give the control pod and the nine check nodes the loop uses first (`cpu-13, -7, -11, -6, -9, -2, -1, -4, -5`), and end your turn; top resumes you for the rest.

**Known already (check, do not trust):** the control pod's `/data` is one NFS network volume, 932 GB at 61%; its root disk is 40 GB; `git worktree remove` fails on that volume and left four `plan-plan-*` trees under `/data/vqueue-tmp/drive-launches/` today; ci keeps hand backups of the loop's arguments in `/data/vqueue/live/`.
