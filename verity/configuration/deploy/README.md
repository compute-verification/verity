# The party runtime (`verity.configuration.deploy`)

The party runtime runs a protocol's parties on the machines a configuration assigns them, over channels between them,
and keeps each party's log of the run. It knows no protocol: a protocol hands it a manifest and its parties' code, and it
reads no message's meaning (the frames and the handshake are `verity.configuration.wire`'s). A party does no I/O and reads no
ambient clock or randomness: its clock readings and coins are messages from sources the runtime attaches, so the same
party code runs live, simulated, replayed, or live against recorded counterparts, and a replay reproduces it exactly.
Nothing is retried: every failure the runtime sees finishes the party refused by name, and a party that dies fails its run.

## Top-level API

- `party.Party` (`party.py`): a party as a sans-IO next-message machine, `start(params, identity)` and then
  `on(received)`, each returning its actions (`Send`, `Timer`, `LogEntry`, `Finish`), as
  `Security/Definitions/Core/Interaction.lean` states a prover and a verifier.
- `manifest.Manifest` and `manifest.place` (`manifest.py`): what a protocol declares (its parties as `Role`s, its
  channels, its messages and what it needs of the workload), and the check of a placement of its parties on a site's
  machines: owners, capabilities, coins never crossing a developer-owned machine, and latency bounds.
- `config.check`, `config.digest` and `config.new` (`config.py`): the configuration, `verity/config/v0`, checked against
  what each listed subprotocol and the workload declare, each refusal by name, and its hash, SHA-512 of its canonical JSON
  without the transport-only fields.
- `runtime.run` (`runtime.py`): the parties placed here, run in one of the modes `simulated`, `live`, `mixed` or `replay`.
  `process.spawn`'s child, `demos/lifecycle` and the toy table of `verity run`'s step-0 test call it; the served
  table of `verity run` runs `served_verdicts.sh` instead.
- `transcript.PartyLog` and `transcript.record` (`transcript.py`): a party's append-only log of a run, each entry a wire
  frame, and the run's record, the logs' entries of the public kinds.
- `channels` (`channels.py`): the pipes, untrusted by both ends: in memory, over TCP, `Playback` (a recorded side),
  `Tee` (the log writer) and `Adversary` (drops, delays, reorders or corrupts frames).
- `sources` (`sources.py`): clocks and coins as parties on the same machine: `VirtualClock` or `RealClock`,
  `SeededCoins` or `OsCoins`.
- `process.spawn` (`process.py`): a party in a process of its own on this machine, over TCP to the run that spawned it.
- `stream.run` and `stream.Side` (`stream.py`): a party over a byte stream in its protocol's own codec, and a party's side
  as another process or machine runs it.

## Where to look

- `tools/verity/cli/verity_cli/run.py`: `verity run` and `verity config`, the commands over this package.
- `verity/protocols/pous/verifier/wipe_parties.py` and `verity/protocols/poub/verifier/session_parties.py`: protocols'
  parties on the runtime.
- `tests/deploy_example.py`: a two-party protocol the tests run every way the runtime runs one; `tests/test_deploy.py`
  and `tests/test_stream.py` are the suite (`uv run tools/verity/check/suites.py verity/configuration/deploy`).
