# The party runtime (`verity.configuration.deploy`)

The party runtime runs a protocol's parties on the machines a configuration assigns them, over channels between them,
and keeps each party's log of the run. It knows no protocol: a protocol hands it a manifest and its parties' code, and it
reads no message's meaning (the frames and the handshake are `verity.configuration.wire`'s). A party does no I/O and reads no
ambient clock or randomness: its clock readings and coins are messages from sources the runtime attaches, so the same
party code runs live, simulated, replayed, or live against recorded counterparts, and a replay reproduces it exactly.
Nothing is retried: every failure the runtime sees finishes the party refused by name, and a party that dies fails its run.

## Interface

What code outside this directory may use. Everything else here is internal.

| Entry point | What it is |
|---|---|
| `party.Party` (`party.py`) | A party as a sans-IO next-message machine: `start(self, params: Params, identity: Identity) -> list[Action]` and then `on(self, msg: Received) -> list[Action]`, each returning its actions (`Send`, `Timer`, `LogEntry`, `Finish`), as `security/Definitions/Core/Interaction.lean` states a prover and a verifier. `Identity` is who the party is and what its machine gives it, and `Received` a message that reached it. |
| `manifest.Manifest`, with `Role` and `Channel`, and `manifest.place(manifest: Manifest, site: Topology, bindings: Mapping[str, str], *, simulated: bool = False) -> PlaceResult` (`manifest.py`) | What a protocol declares (its parties as `Role`s, its channels, its messages and what it needs of the workload), and the check of a placement of its parties on a site's machines: owners, capabilities, coins never crossing a developer-owned machine, and latency bounds. |
| `config.check(cfg: Mapping[str, Any], components: Mapping[str, Component], workloads: Mapping[str, Workload]) -> dict[str, Any]`, `config.digest(cfg: Mapping[str, Any]) -> str` and `config.new(workload: str, protocols: Sequence[str], components: Mapping[str, Component], workloads: Mapping[str, Workload], commit: str, machine: str = LOCAL) -> dict[str, Any]` (`config.py`) | The configuration, `verity/config/v0`: checked against what each listed subprotocol and the workload declare (`Component`, `Workload`), each refusal by name; its hash, SHA-512 of its canonical JSON without the transport-only fields; and a new one that states only its choices. `parse` and `render` read and write its TOML, `declared` gives the parties and channels the declarations make between them, and `composite` the id of the composite the listed protocols make: onsite, remote or one of onsite's two smaller lists (`COMPOSITES`). |
| `runtime.run(config: Mapping[str, Any], parties_here: Sequence[str], mode: str, state_dir: str \| os.PathLike \| None, *, protocols: Mapping[str, ProtocolDef], ...) -> RunResult` (`runtime.py`) | The parties placed here, run in one of the modes `simulated`, `live`, `mixed` or `replay`. `runtime.ProtocolDef` is a protocol as the runtime runs it: its manifest and a factory for each of its parties. The result holds each party's outcome and its log, a `transcript.PartyLog` (`transcript.py`): a party's append-only log of a run, each entry a wire frame. `transcript.record` is the run's record, the logs' entries of the public kinds. |
| `stream.run(party: Party, params: Params, identity: Identity, f: BinaryIO, codec: Codec, *, peer: str, clock: Callable[[], int] \| None = None, coins: Callable[[int], bytes] \| None = None, wait: Callable[[Callable[[], int], int], None] = sleep_until) -> tuple[Finish, list[LogEntry]]` and `stream.Side` (`stream.py`) | A party over a byte stream in its protocol's own codec, until it finishes: its finish and what it recorded. `stream.Side` is a party's side as another process or machine runs it. |

Internal, reached only through `runtime.run`'s arguments:

- `channels` (`channels.py`): the pipes, untrusted by both ends: in memory, over TCP, `Playback` (a recorded side),
  `Tee` (the log writer) and `Adversary` (drops, delays, reorders or corrupts frames).
- `sources` (`sources.py`): clocks and coins as parties on the same machine: `VirtualClock` or `RealClock`,
  `SeededCoins` or `OsCoins`.
- `process.spawn` (`process.py`): a party in a process of its own on this machine, over TCP to the run that spawned it.

The served table of `verity run` doesn't go through `runtime.run`: it runs `served_verdicts.sh`.

## Where to look

- `tools/verity/cli/verity_cli/run.py`: `verity run` and `verity config`, the commands over this package.
- `verity/protocols/pous/verifier/wipe_parties.py` and `verity/protocols/poub/verifier/session_parties.py`: protocols'
  parties on the runtime.
- `tests/deploy_example.py`: a two-party protocol the tests run every way the runtime runs one; `tests/test_deploy.py`
  and `tests/test_stream.py` are the suite (`uv run tools/verity/check/suites.py verity/configuration/deploy`).
