# The party runtime (`verity.runtime`)

A protocol is a set of parties sending each other messages: a prover, a verifier, a challenger, an attester. The runtime
is what puts those parties on real machines. It starts each party on the machine a configuration assigns it, carries
its messages over channels, and keeps each party's log of the run.

It knows no protocol. A protocol hands it a manifest and its parties' code, and the runtime reads no message's meaning.

A party does no I/O and reads no ambient clock or randomness: its clock readings and coins are messages from sources the
runtime attaches. So the same party code runs live, simulated, replayed, or live against recorded counterparts, and a
replay reproduces it exactly. Nothing is retried: every failure the runtime sees finishes the party refused by name, and
a party that dies fails its run.

## Try it

~~~sh
uv run tools/check/suites.py verity/runtime
~~~

This runs a two-party protocol every way the runtime runs one, and needs no GPU. Its last line counts the suites that
passed. The protocol is `tests/verity/runtime/deploy_example.py`, the one to read first: a challenger samples a nonce,
and the responder must answer its SHA-512 before a deadline.

`verity run` and `verity config` are the commands over this package.

## Interface

Code outside this directory uses these names, and nothing else here:

| To write | Import |
| --- | --- |
| A protocol's parties | `verity.party`: `Party`, `Params`, `Identity`, `Received`, and the actions (`Action`) `Send`, `Timer`, `LogEntry` and `Finish` |
| What a protocol declares | `verity.protocol`: `Manifest`, `Role`, `Channel`, `ProtocolDef`, and `place` and `site_of` for a placement (`Topology`, `PlaceResult`); `verity.config`: `Component` and `Workload` |
| The configuration | the rest of `verity.config`: `check`, `digest`, `new`, `parse`, `render`, `canonical`, `declared` and `composite`, with `COMPOSITES`, `LOCAL`, `PARTY`, `CHANNEL` and `TRANSPORT` |
| A run | `runner`'s `run` and `RunResult`, `stream`'s `Side`, and the sources a caller hands `runner.run` (`sources`): `VirtualClock` on a `VirtualTime`, `RealClock`, `SeededCoins`, `seeded` and `OWNER` |

A refusal by name is raised as the wire's `Refused`: import it from `verity.channel`. [`DESIGN.md`](DESIGN.md) gives
each entry point's signature and what is internal.
