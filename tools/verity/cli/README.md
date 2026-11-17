# `verity`

`verity` is the repository's one command. It does no work of its own: each subcommand hands off to the tool that does
it, such as the check, a benchmark's runner, Flock's Lean verifier or a configuration's parties.

## Try it

~~~sh
uv run verity --help
uv run verity run applications/onsite_full.toml --dry-run
~~~

The first prints every subcommand with a line each. The second resolves the onsite application's eight parties and the
channels between them and runs nothing: it prints a `PLAN` line for each party, a `CHANNEL` line for each channel, and a
`DRY` line with the counts and the bindings still missing. Neither needs a GPU.

## What it runs

| Command | What it hands off to |
| --- | --- |
| `verity check` | The whole check, or with `tests`, `circuits` or `lean` only those parts of it. |
| `verity ci` | The check's recent runs, and one run's steps. |
| `verity tasks`, `verity graph` | The check as a graph of tasks: listed, or one task run here. |
| `verity bench NAME` | A benchmark's runner. |
| `verity demo NAME`, `verity run CONFIG` | A configuration's parties on this machine, and its verdict. An application is one configuration file. |
| `verity config` | Writes, checks or hashes a configuration. |
| `verity verify proof` | Flock's Lean verifier, with its own arguments. |
| `verity lean try FILE`, `verity lean deployed` | One Lean file elaborated, the edit loop; the deployed guarantees' rows from the locks. |

In a NAME, a `-` stands for a `_`. `verity run`'s options, what it prints and its exit codes are in its module's
docstring (`verity_cli/run.py`).

## Running an application on machines

A configuration names roles, not hosts. The onsite application places the developer's parties on `prover-node`, and the
auditor's on `gate-host` and `auditor-host`. A bindings file (`--bindings FILE`) gives each party's hosts, addresses and
gate commands, so binding another host changes no hash. On the developer's node, then on the auditor device:

    verity run applications/onsite_full.toml --on prover-node --bindings BINDINGS --state STATE
    verity run applications/onsite_full.toml --on auditor-host --state STATE

The gate's sides start when the developer's pieces open them.
