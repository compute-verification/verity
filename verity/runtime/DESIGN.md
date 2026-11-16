# The party runtime: design

What code outside `verity.runtime` may use, with each entry point's signature, and what is internal. `README.md` says
what the runtime is and how to try it.

## Interface

What code outside this directory may use. Everything else here is internal.

The names, in four groups. A protocol's parties are written against `verity.party`: `Party`, `Params`, `Identity` and
`Received`, and the actions (`Action`) `Send`, `Timer`, `LogEntry` and `Finish`. What a protocol declares is
`verity.protocol`'s `Manifest`, `Role`, `Channel` and `ProtocolDef` and `verity.config`'s `Component` and `Workload`,
with `verity.protocol`'s `place` and `site_of` for a placement (`Topology`, `PlaceResult`). The configuration is the
rest of `verity.config`: `check`, `digest`, `new`, `parse`, `render`, `canonical`, `declared` and `composite`, with
`COMPOSITES`, `LOCAL`, `PARTY`, `CHANNEL` and `TRANSPORT`. The runtime is `runner`'s `run` and `RunResult`, `stream`'s
`Side`, and the sources a caller hands `runner.run` (`sources`): `VirtualClock` on a `VirtualTime`, `RealClock` (a
machine's clock and its name, `MONOTONIC` by default, which the party reads as `Identity.clock`), `SeededCoins`,
`seeded` and `OWNER`. A refusal by name is raised as the wire's `Refused`: import it from `verity.channel`.

| Entry point | What it is |
|---|---|
| `verity.party.Party` (`verity/party.py`) | A party as a sans-IO next-message machine: `start(self, params: Params, identity: Identity) -> list[Action]` and then `on(self, msg: Received) -> list[Action]`, each returning its actions (`Send`, `Timer`, `LogEntry`, `Finish`), as `security/Definitions/Core/Interaction.lean` states a prover and a verifier. `Identity` is who the party is and what its machine gives it, and `Received` a message that reached it. |
| `verity.protocol.Manifest`, with `Role` and `Channel`, and `verity.protocol.place(manifest: Manifest, site: Topology, bindings: Mapping[str, str], *, simulated: bool = False) -> PlaceResult` (`verity/protocol.py`) | What a protocol declares (its parties as `Role`s, its channels and its messages), and the check of a placement of its parties on a site's machines: owners, capabilities, coins never crossing a developer-owned machine, and latency bounds. `verity.protocol.ProtocolDef` is a protocol as a run takes it: its manifest, a factory for each of its parties, and a starter for each party a deployment starts on its machine (`starts`), which gives its `Start`: its parameters, capabilities and clock reading from the configuration, its deploy bindings and its state directory. |
| `verity.config.check(cfg: Mapping[str, Any], components: Mapping[str, Component], workloads: Mapping[str, Workload]) -> dict[str, Any]`, `verity.config.digest(cfg: Mapping[str, Any]) -> str` and `verity.config.new(workload: str, protocols: Sequence[str], components: Mapping[str, Component], workloads: Mapping[str, Workload], commit: str, machine: str = LOCAL) -> dict[str, Any]` (`verity/config.py`) | The configuration, `verity/config/v0`: checked against what each listed subprotocol and the workload declare (`Component`, `Workload`), each refusal by name, the keys the auditor pinned for its devices (`[[pin]]`) among what it checks; its hash, SHA-512 of its canonical JSON without the transport-only fields; and a new one that states only its choices. `parse` and `render` read and write its TOML, `declared` gives the parties and channels the declarations make between them, and `composite` the id of the configuration, onsite or remote, that the listed protocols make (`COMPOSITES`). |
| `runner.run(config: Mapping[str, Any], parties_here: Sequence[str], mode: str, state_dir: str \| os.PathLike \| None, *, protocols: Mapping[str, ProtocolDef], ...) -> RunResult` (`runner.py`) | The parties placed here, run in one of the modes `simulated`, `live`, `mixed` or `replay`. `protocols` holds each protocol's `verity.protocol.ProtocolDef`. The result holds each party's outcome and its log, a `transcript.PartyLog` (`transcript.py`): a party's append-only log of a run, each entry a wire frame. `transcript.record` is the run's record, the logs' entries of the public kinds. |
| `stream.Side(host: str, command: tuple[str, ...], address: tuple[str, int] \| None = None)`, with `Side.of(period: Mapping[str, Any], node: str, key: str, local: Sequence[str], what: str \| None = None) -> Side` and `Side.talk(request: Mapping[str, Any], drive: Callable[[socket.socket], T]) -> tuple[dict[str, Any], T]` and `Side.ask(request: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]` (`stream.py`) | A party's side as another process or machine runs it: where `period.json` puts it, one session with it, its stream handed to the protocol's own code, and one request it answers with no session. |

Internal, which no code outside this directory imports and `runner.run` reaches from its arguments:

- `channels` (`channels.py`): the pipes, untrusted by both ends: in memory, over TCP, `Playback` (a recorded side)
  and `Adversary` (drops, delays, reorders or corrupts frames).
- `transcript` (`transcript.py`): `PartyLog` and `record`, which outside code reads as a `RunResult`'s `transcripts`
  and `record()`.
- the rest of `sources` (`sources.py`, clocks and coins as parties on the same machine): their messages, and
  `OsCoins`, which a run with no seed attaches itself, as it attaches `RealClock()`.
- `process.spawn` (`process.py`): a party in a process of its own on this machine, over TCP to the run that spawned it.
- `framing` (`framing.py`): a message's frame under its registry (`encode`) and the parse of one as adversarial
  (`decode`, `split`, `parse`, `FrameReader`, `frames`), each rule broken refused by name; and `handshake`
  (`handshake.py`), the signed exchange that opens every channel between parties.

The served table of `verity run` runs `served_verdicts.sh` for the served window, and its runtime pieces (the wipe's,
PoUS's rounds' and PoUB's parties) through `runner.run`.

## Where to look

- `tools/verity/cli/verity_cli/run.py`: `verity run` and `verity config`, the commands over this package.
- `verity/protocols/resource_accounting/memory/poub/verifier/session_parties.py` with `verity/protocols/resource_accounting/memory/poub/prover/session_prover.py`,
  `verity/protocols/resource_accounting/memory/pous/attester/wipe_challenger.py` with `verity/protocols/resource_accounting/memory/pous/prover/wipe_prover.py`,
  and `verity/protocols/resource_accounting/compute/pouw/verifier/parties.py`: protocols' parties on the runtime, the two sides of the
  first two each in its own file.
- `tests/verity/runtime/deploy_example.py`: a two-party protocol the tests run every way the runtime runs one;
  `tests/verity/runtime/test_deploy.py` and `tests/verity/runtime/test_stream.py` are the suite (`uv run tools/check/suites.py verity/runtime`).
