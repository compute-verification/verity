# The toy-network example

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

| Party | Principal | What it does |
| --- | --- | --- |
| `challenger` | auditor | a development challenger host (`benchmarks/one_stage/challenger_party.py`), listening on loopback; it opens and finishes each period's draw for the verifier, with its own coins |
| `firewall` | developer | builds the tools on first use (`benchmarks/all_protocols/build.sh`: C-Flock's prover, the Lean verifier, firewall and product) and pins the Lean `flock-firewall` every outer proof goes through |
| `network-certifier` | auditor | relays each period's requests to the model, commits the outputs at the reading circuit's out shape and signs the record |
| `workers` | developer | the toy serving each period, tied to the relayed outputs |
| `verifier` | auditor | each period's sampled-proofs audit on the challenger's coins, then the network certifier's period check, then the verdict: onsite's `product` over exactly the two listed components |

## Commands

From a fresh clone with `origin/main` fetched (the challenger pins the newest commit of main's first-parent line that
the tree is built on), and Rust's `cargo` on the path:

~~~sh
uv sync --all-packages --extra torch-cpu
uv run verity config new --workload toy --protocols sampled-proofs,certifier --out config.toml
uv run verity config check config.toml
uv run verity run config.toml --state /tmp/toy-network
~~~

`verity run examples/toy-network/config.toml` and `verity example toy-network` run this file instead. `config check`
prints `ok` and the configuration's hash. `verity run` prints a `RUN` line, a `PARTY` line as each party exits (the
challenger's twice: ready, first, and stopped, last), and one `VERDICT` line a period, also written to
`STATE/record/verdicts.jsonl`. It exits 0 when every period accepts, 1 when one rejects, and 2 otherwise.

The parties can also run one command at a time on the same state, in the order above. The challenger runs beside the
verifier, which is the one party that talks to it:

~~~sh
uv run verity run config.toml --state /tmp/toy-network --party firewall
uv run verity run config.toml --state /tmp/toy-network --party network-certifier
uv run verity run config.toml --state /tmp/toy-network --party workers
uv run verity run config.toml --state /tmp/toy-network --party challenger --party verifier
~~~

`--replay /tmp/toy-network --party challenger --party verifier --state DIR` reruns the challenger and the verifier over
that run's served periods in a new state, without what they wrote: the same records, on new coins.

## The state

Each party's output is in `STATE/logs/<party>.log`. What each party wrote is its transcript, `STATE/transcripts/<party>.jsonl`:
one line a file, with its bytes, SHA-512 and kind, `public` under `record/` and `public/`, else `private`. The record,
`STATE/record/`, keeps the public lines (`record/transcripts/`), each period's `w<I>/<component>/answer.json` and
`case.json`, and the network certifier's public session files. The workers' rows and salts stay in `STATE/blocks/`.

## What is simulated

- The coins: the challenger is a process on the developer's machine, with a key made for the run.
- The network certifier also runs on the developer's machine, beside the model it relays for.
- The messages: at step 0 the parties exchange them as files in the shared state, one party after another, not on the
  wire. The transcripts are what each party wrote.
- The program: `[program] commit` is the commit `config new` ran at. A run on another tree names the difference in each
  verdict line's `simulated`.
