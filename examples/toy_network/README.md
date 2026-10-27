# The toy network example

Sampled proofs and the network certifier on the toy, every party on one machine. `config.toml` is the deployment
configuration `verity config new --workload toy --protocols sampled-proofs,certifier` writes: it states only its choices,
and every parameter it leaves out is its declaration's default (`benchmarks/all_protocols/verity_all_protocols/deployment.py`,
step 0's table). `verity run` runs its parties and prints each period's verdict.

## Workload

The toy (`integrations/torch_loop`): a two-layer W8A8 MLP on CPU, serving requests 3 and 7 for 2 decode steps each,
at most 8 units a member, a period at a time. It provides the two hooks the configuration's protocols need:

- `windows`, for sampled proofs: each period's circuit registered before any coin, its rows and salts committed;
- `ingress-relay`, for the network certifier: each output unit reads the output the network certifier relayed and signed
  for the period (`tie = true`).

## Placement

Every party is on `local`, so the challenger, which sends the coins, shares a machine with the developer's workers and
firewall. Its `Faithful` and `Secret` premises are unmet there, and every verdict line says `"coins": "simulated"`.

The network certifier's party runs first, in a process of its own. The other four are sampled proofs' parties
(`verity/core/protocols/sampled_proofs/parties.py`): next-message machines on the deployment layer (`verity.core.deploy`),
their messages the wire's, run together in one live run. What the machine does for them is a capability it gives them,
and on the toy those are `deployment.py`'s `Machine`, on `benchmarks/all_protocols/verity_all_protocols/parties.py`'s.

| Party | Principal | What it does |
| --- | --- | --- |
| `network-certifier` | auditor | relays each period's requests to the model, commits the outputs at the reading circuit's out shape and signs the record |
| `challenger` | auditor | starts a development challenger host (`benchmarks/one_stage/challenger_party.py`, with its own key, ledger and pins log) on loopback and tells the verifier where it is (`Challenger`); stops it on the verifier's `Done` |
| `firewall` | developer | tells the verifier the SHA-512 of the Lean `flock-firewall` that every outer proof goes through, one at a time (`Firewall`) |
| `workers` | developer | the toy serving each period, tied to the relayed outputs, and the period's registration sent to the verifier (`Registration`) |
| `verifier` | auditor | each period, once its registration is in: the sampled-proofs audit on the challenger's coins, then the network certifier's period check, then the verdict: onsite's `product` over exactly the two listed components |

## Commands

From a fresh clone with `origin/main` fetched (the challenger pins the newest commit of main's first-parent line that
the tree is built on), and Rust's `cargo` on the path:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run verity config new --workload toy --protocols sampled-proofs,certifier --out config.toml
uv run verity config check config.toml
uv run verity run config.toml --state /tmp/toy-network
~~~

`verity run examples/toy_network/config.toml` and `verity example toy_network` run this file instead. `config check`
prints `ok` and the configuration's hash. The first party that needs the tools builds them into `STATE/build`
(`benchmarks/all_protocols/build.sh`: C-Flock's prover, the Lean verifier, firewall and product). `verity run` prints a
`RUN` line, a `PARTY` line as each party finishes (the network certifier's exit code, then each live party's refusal,
`null` when it had none), and one `VERDICT` line a period, also written to `STATE/record/verdicts.jsonl`. It exits 0
when every period accepts, 1 when one rejects, and 2 otherwise.

Run `r20261027-064850-9e47` ran these commands from a clean clone of fc8938eb47, with `--periods 2`, on vy-nebius-cpu-4
(192 cores). `verity run` took 5441 s, and both periods accept:

~~~text
PARTY {"exit": 0, "party": "network-certifier"}
PARTY {"party": "challenger", "refused": null}
PARTY {"party": "firewall", "refused": null}
PARTY {"party": "verifier", "refused": null}
PARTY {"party": "workers", "refused": null}
VERDICT {"block": 0, "coins": "simulated", "components": {"certifier": "accept", "sampled-proofs": "accept"}, "product": "verity/product/v0", "reason": null, "registration": "a32f4b31592a91c4c9e47e8f7c4150c9cccc246ce0cafa60757eb998f05e6d712ee6de744a47eb3f75007e29eadb33b2be17c840c2838d4f3e5c39f50dfe8181", "run": "r20261027-064850-9e47", "simulated": ["challenger shares local with the developer's firewall, workers: Faithful and Secret unmet"], "verdict": "accept"}
VERDICT {"block": 1, "coins": "simulated", "components": {"certifier": "accept", "sampled-proofs": "accept"}, "product": "verity/product/v0", "reason": null, "registration": "293481f5a4062f05ab4439ca62637b8f16516472c32b6f9b7cf312e62a5d4190abf0d7faff463c2a015099f47c990b97f71ae567b4464616f0cb3361e6cbb89a", "run": "r20261027-064850-9e47", "simulated": ["challenger shares local with the developer's firewall, workers: Faithful and Secret unmet"], "verdict": "accept"}
~~~

## Parties on different machines

`--machine NAME`, or `--party ROLE`, runs only the parties placed there. The challenger, the firewall and the workers
then listen for the verifier at their `listen` port (on their `address`, loopback by default), and the verifier dials
the ones that run elsewhere. A party that has to listen or be dialed and has no `listen` is refused by name
(`listen`). Each channel's handshake is signed with a party key made for the run, and the configuration's hash leaves
`address` and `listen` out, so moving a listener changes no hash. Until the audit's draw, witnesses and proofs are
messages of their own, the verifier's audit runs the prover's sessions itself, so the verifier and the workers share
the machine the rows are on.

## The sampled-proofs parties alone

Their own entry point runs them on the toy, with no network certifier, on
`benchmarks/all_protocols/tests/fixtures/sampled-proofs.toml` (a uniform 8 of each block's members):

~~~sh
PYTHONPATH=benchmarks/all_protocols:integrations/torch_loop uv run python -m verity_all_protocols.parties \
    benchmarks/all_protocols/tests/fixtures/sampled-proofs.toml --state STATE --local
~~~

`--tamper 0` makes block 0's workers record outputs that are not the model's, and the verifier rejects it. `--party
ROLE` with `--at ROLE=HOST:PORT` runs a party in a process of its own over TCP. `benchmarks/all_protocols/parties.sh`
runs honest, tampered, an absent certifier and the four parties apart (runs `r20261027-013034-fbf3` and
`r20261027-053345-b27b`).

## The state

Each entry's output is in `STATE/logs/<party>.log`, and what it wrote is its transcript,
`STATE/transcripts/<party>.jsonl`: one line a file, with its bytes, SHA-512 and kind, `public` under `record/` and
`public/`, else `private`. The live parties' transcripts and their run's record are the deployment layer's, under
`STATE/runs/local/runs/1/`. The record, `STATE/record/`, keeps the entries' public lines (`record/transcripts/`), each
period's `w<I>/<component>/answer.json` and `case.json`, and the network certifier's public session files. The workers'
rows and salts stay in `STATE/blocks/`.

## What is simulated

- The coins: the challenger's host is a process on the developer's machine, with a key made for the run.
- The network certifier also runs on the developer's machine, beside the model it relays for. Its relay reaches the
  workers as files in the shared state before the live run, not as messages on the wire.
- The witnesses and proofs: the verifier's audit runs the prover's sessions itself, on the rows in the shared state.
- The program: `[program] commit` is the commit `config new` ran at. A run on another tree names the difference in each
  verdict line's `simulated`.
