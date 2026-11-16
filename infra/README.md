# infra

`infra/` is the team's own machines and the code that keeps them running: the shared CPU and GPU nodes, the control
pod that runs every loop that must outlive an agent VM, and the services that launch the coordinators and carry their
messages. Nothing in `verity/` reads any of it, and nothing outside `infra/` imports it, apart from the ties
`tests/import_order.toml` lists into its row, the last. The `research` command reaches `ops` and `cluster` by running them,
never by import.

## What you run

The nodes are installed from a pushed commit, and `research` is the command for them. It hands off to `ops`, this
directory's command:

~~~text
research deploy drift [--on NODE]        how far each node's installed files are from origin/main
research deploy install NODE --by WHO    install them from a pushed commit; a dry run without --apply
research deploy roll PATH --by WHO       a file made from the register of machines, on every node that lists it, one at a time
research keeper status --on NODE         a node's standing targets (`set` and `stop` change them)
~~~

Each reaches the nodes, so it needs the team's keys and the nodes' addresses.

## What is here

| Directory | What it is |
| --- | --- |
| [`cluster/`](cluster/README.md) | The shared nodes' policy, planner and allocation ledger, and their GPU-hours by node and hour. |
| [`control/`](control/README.md) | The control pod, `vy-control-verity`: its loops, its services and how each is deployed. |
| [`cursor/`](cursor/README.md) | Launching, handing over and archiving the coordinators. |
| [`slack-events/`](slack-events/README.md) | The service that routes people's Slack posts to lanes. |
| `ops/` | The `ops` command: deploys to the nodes and their drift, and each node's keepers. |
| `nebius/` | The units that run the console's panels on node 1. |
| `dashboard/` | The Metrics dashboard's own number: how many top goals have a number that runs. |

The agent VM's secret guard and shell hook are [`tools/agent-guard/`](../tools/agent-guard/README.md), and the run
recorder, the evidence store's client and `research` itself are [`tools/research/`](../tools/research/README.md).
