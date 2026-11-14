# The verifier's auditor host

`host.py` is vy-auditor-1's verifier service: the verifier of PoUB
(`verity/protocols/accounting/poub/verifier/session/verifier.py`) run only as `auditor`, from a tree infra pinned, one run at a time, behind the
gate `vy-verifier-gate` that every key of the host's `research` is forced to. Its docstring is the reference: who reaches
the gate, what runs, a run, a replay and the store. The host's deployed files are
`tools/research/src/research/pods/nebius/verifier/`, and infra deploys `host.py` and owns their entries in
`tools/research/src/research/pods/nebius/deploy.toml`.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `host.py gate VERB ...` (on the host, `vy-verifier-gate VERB ...`) | The gate, the one way into the host's verifier. `run` starts a run from its whitelisted parameters (A run, below); `status [RUN]`, `fetch RUN [--transcripts]`, `replay RUN`, `stop RUN` and `close NODE_HEX` read, replay, stop a run or close a node's registration. It answers on stdout, and a refusal is exit 3 with `{"refused": WHY}`. |
| `host.py gate wipe_party [--reads K] [--native [--threads T]]`, `challenger_party`, `bw_party (--pous-session SID \| --fixture) [--minutes M]`, `party_key wipe\|pous`, `party_log wipe\|pous [LOG]` | The memory challenger's two parties and the party of PoUB over PoUS's P2 store, each run as `auditor` from the pinned tree, one session per connection: the request line on the gate's stdin, then `{"listening": PORT}` and the signed report on its stdout. `party_key` prints a party's public key and `party_log` its sessions as the host kept them. |
| `host.py run RUN`, `host.py replay RUN`, `host.py publish RUN` | What the host's units run as `auditor`. A run is the verifier's process with the request's parameters and the host's own, and leaves `runs/RUN/out/` (`verifier.json`, `challenger/`, `verifier.log`, `registry.json`, and the transcripts gzipped with their `SHA256SUMS`) and `end.json`. A replay checks the transcripts against what the run recorded and replays them through both verifiers into `runs/RUN/replay/`. `publish` puts `runs/RUN/` in the evidence store as a tree of kind `verifier-run/v1`. |
| `host.py pinned COMMIT` and `host.py close NODE_HEX` | `pinned` appends a pin (the commit and the UTC time) to the host's pins log, which a run starts from, and prints it. `close` closes a node's registration in the verifier's registry. |

Nothing imports this directory: every entry point is `host.py` run by its path.

`party_log wipe|pous|bw [LOG]` answers root on the host alone (`sudo vy-verifier-gate party_log ...`): the gate refuses
it over ssh to every key, named or not, since a party's log holds what its device received and signed. The parties write
their files as before.

A party verb (`wipe_party`, `challenger_party`, `bw_party`) answers only a caller from a registered node, so every answer
of a party goes into a node. The wrapper `vy-verifier-gate` passes the ssh caller's address (the first word of
`SSH_CONNECTION`, which sudo would otherwise drop) as `--from`. The gate refuses the verb unless that address is one of a
peer's `addresses` in `/etc/vy/verifier.json` (`node_of`), whatever the key. A peer lists the addresses its ssh to the
host comes from:

~~~json
"peers": {"NODE": {"host_bytes": 1843153879040, "ingress_bytes_per_s": 4.4e9, "addresses": ["ADDRESS"]}}
~~~

## A run

~~~text
vy-verifier-gate run --peer NAME --port P --period P --pin SHA512 (--paced SPEC ... [--pace-pass-ms MS]
                     | --sessions N [--windows W] [--window-ms MS] [--samples S])
                     [--minutes M] [--jitter-ms MS] [--quantile Q] [--lifetime L]
~~~

A lane names a run's parameters, each held to a whitelist, and never a path, a host, a key or a node's capacity: those
are the host's, from `/etc/vy/verifier.json`. The verifier holds no checkpoint: a run names no model. `--pin` is the SHA-512 of the challenger's public key, passed to
verifier.py as `--challenger-pin`.

The verifier credits a session's coins only from a challenger that ran main's code (`coins.trusted`, through
`challenger.pinned_session`). It fetches the challenger's pins log through the challenger's gate (`pins`) at each
session's end, beside the ledger, and it reads main's first-parent history from `/opt/vy-verifier/main-first-parent.txt`
(`--challenger-main`). Infra writes that file, one 40-hex commit a line, whenever it pins either host:

~~~sh
git fetch origin main && git rev-list --first-parent origin/main > main-first-parent.txt
install -m 0644 -o root -g root main-first-parent.txt /opt/vy-verifier/main-first-parent.txt   # on the auditor host
~~~

The gate refuses a run before starting anything while the file is missing. A challenger pinned after the file was
written isn't on it, so its sessions are refused as `coins` until infra writes the file again.

## The compose gate

The same `host.py`, pinned under `/opt/vy-compose/<commit>` (its `pins.jsonl`, `/var/lib/vy-compose`), is
`vy-compose-gate`, which serves one verb: `compose`, a served run's final record and composite composed off the served
node from its compose hand-off. The lane's own gate key is forced to `vy-compose-gate --ssh "$SSH_ORIGINAL_COMMAND"
--key LANE` for `research` on the auditor host's public address (`restrict`, and `from=` the hosts that send
hand-offs), and the compose itself runs as `auditor`, and the sender is
`served_verdicts.sh gate-compose`, which runs:

~~~sh
tar --create --files-from=private/compose/inputs.txt --file=- | $COMPOSE_GATE compose
~~~

`COMPOSE_GATE` is the sender's setting: `ssh` with the lane's gate key, as `research` on that address.

At pin time, as `auditor`, infra builds the tree's venv (`uv sync --all-packages --extra torch-cpu`) and the Lean
packages the compose runs (`verity/protocols/proofs/recursion/verifier/recursive`, whose `vstar-recursive` the record runs, and the
composite's), and records the pin with the branch it is on (`host.py pinned COMMIT BRANCH`). The compose runs as
`auditor` with `HOME=/var/lib/vy-compose`, so `lake` and `elan` resolve from `/usr/local/bin` or that home's
`.elan/bin`. The docstring says what the gate refuses and what it answers.

## Tests

`tests/test_host.py` runs the gate in-process with its root steps injected. It holds the whitelist to verifier.py's own
parser, including the host's main list, and the compose gate's
receive and compose on toy hand-offs (`benchmarks/one_stage/tests/test_served_verdicts.py` sends one through it with
`gate-compose`). It also runs one session through `tests/stub_verifier.py` (slow). Run the suite as
`uv run tools/verity/check/suites.py verity/protocols/accounting/poub/device/live_verifier`, on a node (the testing skill).
