# infra

Our machines: the shared CPU and GPU nodes, the control pod that watches them, and the tools that launch the coordinators.
Nothing outside `infra/` imports it, apart from the files `INFRA_IMPORTERS` in `tests/test_boundaries.py` names. Each
subdirectory has its own README with the detail; this page maps each machine and loop to its code.

## The shared nodes

| What | Code |
|---|---|
| Node provisioning: create, set up, stop and tear down a Nebius VM | `tools/research/src/research/pods/nebius/` (`launch.sh`, `vm_setup.sh`, `vm_stop.sh`, `teardown.sh`) |
| What is installed on each node, from a pushed commit (`research deploy install`) | `tools/research/src/research/pods/nebius/deploy.toml` |
| A node's systemd units (GPU adoption, custody, keepers, the IPv4-options drop, MPTCP off) | `tools/research/src/research/pods/nebius/units/` and the `vy-*.service` files beside `deploy.toml` |
| The fleet's 5-minute usage samples and the compute panels | `tools/research/src/research/pods/nebius/fleet.py` |
| The node list each placer reads | the notes' `machines.d`, one file per machine (`research pods register`) |
| Policy, planner and allocation ledger for the shared nodes (`cluster`, stdlib only) | `infra/cluster/` |
| Deploys to the nodes and their drift (`ops deploy`), and keepers (`ops keeper`) | `infra/ops/verity_ops/` |
| The console's panels on node 1 | `infra/nebius/` |

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
| The watchers supervisor and its loops (VM stops and deadlines, fleet, monitors, top's asks relay) | `infra/control/watchers.py`, `watchers.toml`, `vm_stop.py`, `top_watch.py` |
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
