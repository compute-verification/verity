# The served window's onsite run

`config.toml` is one deployment configuration (`verity/config/v0`, `verity.core.deploy.config`) of onsite's seven
subprotocols (`verity/onsite/v0`): sampled proofs, PoUW, the inference-only policy, PoUS, PoUB (`bandwidth`), the network
certifier and the wipe, on the `served-window` workload: one window of Llama-3.2-1B in int8 served through the
vLLM integration's committing endpoint. `verity run` executes it, reading `served_table.py` (`verity_cli.run.REGISTRY`).
The file is `served_table.configuration` rendered by `verity.core.deploy.config.render`, and
`tools/verity/cli/tests/test_onsite_served.py` holds it to that byte for byte. It names no host and no address: those are
the deploy bindings' (below), so binding another host changes no hash.

## The phases

- **Registration**: the site (the prover node the configuration places the developer's parties on), θ (the served
  weights, registered before the first request) and the builder B (the committer's registered compiler).
- **Execution**: the served window: serve-openai under the node's GPU lease, the requests, every closed window sealed.
- **Verification**: each component's commitment, challenge, proof or timed response (PoUS's and PoUB's sessions), and
  validation: the pieces' answers, then the onsite check, `served_verdicts.py record`, whose composed verdict is the
  period's `VERDICT` line.

## The parties

| party | principal | machine | its pieces (`served_table.PARTIES`) |
| --- | --- | --- | --- |
| `workers` | developer | `prover-node` | the served window, `benchmarks/one_stage/served_verdicts.sh`, run once with every piece's hooks; sampled proofs' commitment and proofs; PoUW's H1 on the engine |
| `firewall` | developer | `prover-node` | the Lean firewall of the records audit, inside the served window |
| `prover` | developer | `prover-node` | PoUS's served piece, PoUB's served session and the wipe, inside the served window |
| `challenger` | auditor | `gate-host` | the gate: sampled proofs' live challenger and PoUW's H1 coins |
| `network-certifier` | auditor | `gate-host` | the warden in front of the endpoint and its piece, reached through the gate |
| `memory-challenger` | auditor | `auditor-host` | PoUS's and the wipe's timed reads, on its own coins and clock, signing the session |
| `bandwidth-verifier` | auditor | `auditor-host` | PoUB's verifier, on its own coins and clock |
| `verifier` | auditor | `auditor-host` | the onsite check over the served window's public record |

The channels are the components' declarations (`served_table.MANIFESTS`, sampled proofs' and the wipe's from core's
`MANIFEST`s), merged per sender and receiver; `verity run --dry-run` prints each.

## The hooks

The served window's runner reads each piece from its variables, which `verity run` sets from the configuration and the
deploy bindings:

| variables | component | from |
| --- | --- | --- |
| `REPO`, `REV`, `PRECISION`, `SLOTS`, `WINDOW_STEPS`, `LOAD` | the workload | `[workload.params]` |
| `V3C_LAW` | sampled proofs | `[service] law` |
| `PP_CHALLENGER`, `PP_PEER`, `PP_CHALLENGER_KEY`, `PP_GATE_KEY` | sampled proofs | `challenger.sampled-proofs`: `gate`, `peer`, `key`, `gate-key`, each optional; with no `gate` the records audit draws on its own live coins |
| `H1_GATE`, `H1_PEER`, `H1_KEY_SHA512` | PoUW | `challenger.pouw`: `gate` (the gate command), `peer`, `key-sha512`; `POUW_CMD` is the runner's default then |
| `POUS_CMD`, `POUS_ARGS` | PoUS | `memory-challenger.pous`: `host`, `address`, `command`, `key-sha512`; `--rounds` is `[params."verity/pous/p2/v4/run"] challenges`, 320 (ε = 2^-20) |
| `SERVE_ARGS` (`--remote-verifier`) | PoUB | `bandwidth-verifier.bandwidth`: `address` |
| `CERTIFIER_FRONT`, `CERTIFIER_CMD` | the network certifier | `network-certifier.certifier`: `gate` (`USER@HOST`) |
| `WIPE_CMD`, `WIPE_CHALLENGER`, `WIPE_ADDRESS`, `WIPE_COMMAND` | the wipe | `memory-challenger.wipe`: `host`, `address`, `command` |
| `OUT` | the run | `STATE/out`, the served window's public record |

`BANDWIDTH_CMD` has no piece in the tree yet, so the record names PoUB's row missing and composes `pending` while it is.
The inference-only policy has no hook: no served window produces the declared division yet, so the record has no row
for it, and the onsite check's verdict stays `undecided` while a listed subprotocol has no row.
The serve dir stays where the runner keeps it on the node (`D`, `SERVE_ROOT` and the runner's other variables pass
through from the environment).

## The deploy bindings

A TOML file, a table per party and in it a table per listed component that declares it (`verity_cli.run`'s docstring);
a key a piece needs and the file lacks is refused `unbound` before the served window starts:

~~~toml
[challenger.pouw]
gate = "GATE-COMMAND"            # what reaches the gate, e.g. an ssh command line
peer = "PEER"
key-sha512 = "HEX"               # the SHA-512 of the challenger key the verifier trusts

[memory-challenger.pous]
host = "HOST"
address = "HOST:PORT"
command = "COMMAND"              # what starts its side on its host
key-sha512 = "HEX"

[memory-challenger.wipe]
host = "HOST"
address = "HOST:PORT"
command = "COMMAND"

[bandwidth-verifier.bandwidth]
address = "HOST:PORT"

[network-certifier.certifier]
gate = "USER@HOST"
~~~

## Running it

Every party resolved, every channel and piece, and nothing run (no GPU); with `--bindings`, a missing binding refused:

    verity run examples/onsite_served/config.toml --dry-run --bindings BINDINGS

On the prover node, from the commit `research deploy install` installed (set `[program] commit` to it, else the verdict
names the difference under `simulated`): the served window, every piece inside it, the auditor's sides opened through
their bindings:

    verity run examples/onsite_served/config.toml --on prover-node --bindings BINDINGS --state STATE

The gate's and the auditor host's sides are started by their gates when the prover's pieces open them, so
`verity run --on gate-host` starts nothing. On the auditor host, with `STATE/out` holding the served window's public
record, the onsite check and the period's verdict:

    verity run examples/onsite_served/config.toml --on auditor-host --state STATE
