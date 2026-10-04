# Layout trial

A toy of the proposed layout: every protocol's interface, remote and onsite Verity end to end on a real circuit, with
integrity implemented and the other protocols standing in for theirs:

~~~bash
uv sync
uv run python examples/remote.py [f32_add|bf16_gt_strict]
uv run python examples/onsite.py [f32_add|bf16_gt_strict]
uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]
lake build    # the Lean beside the Python
~~~

## The model

The principals are the prover, the verifier, and the trusted hardware: replayers, wardens, pingers. An actor is one
principal's part in one protocol, a transducer over one ordered log of events: it receives the events addressed to it
and sends events in reply (`verity.primitives.events`). What it observes and computes in between is its own state and
never enters the log. An actor's name starts with its principal (`verifier/integrity`). `verity.run.run(actors, view)`
plays the actors against each other and yields one principal's view. The auditor trusts the verifier, so the
verifier's view is the output: what its actors establish are events in it. Replayers, wardens and pingers are trusted
too but sealed: what they see stays their own, and reaches the verifier only as what they send it.

A construction is a frozen dataclass of its parameters, with a method per principal it involves that returns that
principal's actors (`Replay(k, delta).verifier()`). A trusted principal's method takes its readings (`pinger(elapsed)`).
Choosing a construction is choosing a value. A composite is a construction whose fields are constructions: its methods
put their actors in one mapping, wired by name (a verifier's `to` names who reads what it establishes), and its
guarantee is proved from theirs. Anyone can address anyone, so an actor takes each kind of event only from the
principal it trusts for it (`receive(kind, sender)`). Nothing orchestrates the actors: each moves when an event reaches
it, so each protocol keeps its own cadence over the log. Order in the log is logical; physical time enters only as
trusted readouts in what a trusted principal sends.

## Layers

Imports point down only, in Python and in Lean:

~~~text
verity/run.py          the runner
verity/protocols       flat; each establishes its own statement, from the prover (integrity) or from other protocols'
                       statements and physical facts (accounting, policy); imports another only for the events it
                       reads, or to compose it (remote, onsite)
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
