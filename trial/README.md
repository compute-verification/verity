# Layout trial

A toy of the proposed layout, with integrity end to end on a real circuit:

~~~bash
uv sync && uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]
lake build    # the Lean beside the Python
~~~

## The model

Every party (a prover, a verifier, a replayer, a warden, a pinger) is a transducer over one ordered log of events: it
receives the events addressed to it and sends events in reply (`verity.primitives.events`). What it observes and
computes in between is its own state and never enters the log. A party's name starts with its domain
(`verifier/integrity`). `verity.protocol.run.run(parties, view)` plays the parties against each other and yields one
domain's view. The auditor trusts the verifier, so the verifier's view is the output: its profiles are events in it.
Replayers, wardens and pingers are trusted too but sealed: what they see stays their own, and reaches the verifier only
as what they send it.

A construction is a frozen dataclass of its parameters, with a method per domain it occupies that returns that
domain's parties (`Replay(k, delta).verifier(scheme)`). Choosing a construction is choosing a value.

There are no epochs: each subprotocol keeps its own cadence over the log. Order in the log is logical; physical time
enters only as trusted readouts in what a trusted party sends.

## Layers

Imports point down only, in Python and in Lean:

~~~text
verity/protocol        run; later, the composition: accounting per resource, then policy
verity/subprotocols    each an interactive protocol that establishes its own statement; never imports another
verity/primitives      events, circuits and partitions, commitments, randomness
~~~

`examples/` sit on top. An integration (vLLM) is a prover.

## Conventions

- **Lean sits beside the Python it states.** `verity/primitives/Circuit.lean` sits beside `circuit.py`. There is one
  Lake package at the root (`lakefile.toml`), and a module's name is its path. The wheel excludes `*.lean` and
  `.lake`.
- **`__init__.py` holds only a docstring.** Code goes in named modules, so the files can be dropped (making the
  folders namespace packages) once `packages/verity` is gone and nothing else named `verity` is on the path.
- **Heavy constructions have their own `pyproject.toml`.** It sits beside the source and declares the dependencies
  (`integrity/proofs/sp1/`); the source is imported only by whoever asks for it.
