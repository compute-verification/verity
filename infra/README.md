# infra

Our machines: the shared CPU and GPU nodes, the control pod that watches them, and the tools that launch the coordinators.
Nothing outside `infra/` imports it, apart from the files `INFRA_IMPORTERS` in `tests/test_boundaries.py` names; the
`research` command reaches `ops` and `cluster` by running them, never by import. `cluster`, `control`, `cursor` and
`slack-events` each have a README with the detail; `ops` and `nebius` are described here. This page maps each machine
and loop to its code.

## The shared nodes

| What | Code |
|---|---|
| Node provisioning: create, set up, stop and tear down a Nebius VM | `tools/research/src/research/pods/nebius/` (`launch.sh`, `vm_setup.sh`, `vm_stop.sh`, `teardown.sh`) |
| What is installed on each node, from a pushed commit (`research deploy install`) | `tools/research/src/research/pods/nebius/deploy.toml` |
| A node's systemd units (GPU adoption, custody, keepers, the IPv4-options drop, MPTCP off) | `tools/research/src/research/pods/nebius/units/` and the `vy-*.service` files beside `deploy.toml` |
| The fleet's 5-minute usage samples and the compute panels | `tools/research/src/research/pods/nebius/fleet.py` |
| The node list each placer reads | the notes' `machines.d`, one file per machine (`research pods register`) |
| Policy, planner and allocation ledger for the shared nodes (`cluster`, stdlib only) | `infra/cluster/` |
| GPU-hours held, busy and useful per node and hour (`cluster usage`, which `research usage` runs) | `infra/cluster/src/cluster/usage.py` |
| Deploys to the nodes and their drift (`ops deploy`), and keepers (`ops keeper`) | `infra/ops/verity_ops/` |
| The console's panels on node 1 | `infra/nebius/` |

`infra/ops` is the distribution `verity-ops`, whose command is `ops`. `research deploy` and `research keeper` hand off
to it by exec, so those are the commands to run:

~~~text
research deploy drift [--on NODE]        how far each node's installed files are from origin/main
research deploy install NODE --by WHO    install them from a pushed commit; a dry run without --apply
research keeper status --on NODE         a node's standing targets (`set` and `stop` change them)
~~~

`ops/verity_ops/deploy.py` reads the manifest `tools/research/src/research/pods/nebius/deploy.toml`, `keeper.py` sets
the targets a node's keeper fills, and `agent_watch.py` is the control pod's agent watch (below). The `tests/` beside
them hold the control pod's tests too.

`infra/nebius` holds two units, `nebius/verity-console.service` and `nebius/verity-console.timer`: every five minutes
node 1 runs the console's `node1` group (`tools/research/console/`). The manifest installs both.

## The control pod

`vy-control-verity` runs every loop that must not depend on an agent VM: an agent VM can be suspended at any moment.
`infra/control/README.md` has its services table, its `/data` layout and how to deploy each piece.

| Loop | Code |
|---|---|
| Boot: links and services from the network volume | `infra/control/post_start.sh` |
| Drive, the landing automation's ticks, and promote | `infra/control/drive.sh`, `infra/control/promote-loop.sh` |
| The agent watch: frees held runs, records idle workers and run ends | `infra/ops/verity_ops/agent_watch.py`, run by `infra/control/agent_watch.sh` |
| The agent watch's relay to Slack | `infra/control/agent_watch_relay.py` |
| The cloud watch: loop health, R2, Neon, spend against budgets | `infra/control/cloud_watch.py` and `cloud_watch.toml` |
| The watchers supervisor and its loops (VM stops and deadlines, fleet, monitors, top's asks relay, the nightly uncached check of `main`) | `infra/control/watchers.py`, `watchers.toml`, `vm_stop.py`, `top_watch.py`, `nightly_check.py` |
| The control disk alert and the notes sync check | `infra/control/control_disk_alert.py`, `control_disk.sh` |
| Store index, garbage collection and fixture export | `infra/control/store_index.sh`, `store_gc.sh`, `fixtures_export.sh` |
| The resource steward | `infra/control/steward-run.sh` |
| GitHub App tokens for drive and the notes | `infra/control/gh_app_token.sh`, `gh_notes_token.sh` |
| `research msg` as root, for credentials that live only on the pod | `infra/control/msg.sh` |
| The dead-man's switch, which alerts when the pod stops | `infra/control/deadman/` |

## Agents and messaging

| What | Code |
|---|---|
| Launching, handing over and archiving the coordinators | `infra/cursor/launch_coordinator.py`, `infra/cursor/README.md` |
| The Slack events service, which routes people's posts to lanes | `infra/slack-events/` |
| The agent VM's secret guard and shell hook, and the reader that lists an env file's variable names and no value (`envnames.py`) | `tools/agent-guard/` |
| The run recorder and the run-store client (`research`) | `tools/research/`, the store's contract in `tools/research/src/research/store/README.md` |
