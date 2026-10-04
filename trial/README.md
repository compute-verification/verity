# Layout trial

A toy of the proposed layout: remote and onsite Verity end to end on real circuits, with integrity implemented and
the rest standing in for the main repository's protocols:

~~~bash
uv sync
uv run python examples/remote.py [f32_add|bf16_gt_strict]
uv run python examples/onsite.py [f32_add|bf16_gt_strict]
uv run python examples/integrity.py [f32_add|bf16_gt_strict] [replay|transparent|sp1]
uv run python examples/datacenter.py [scenario ...]    # a site over a period of epochs, honest and otherwise
lake build    # the Lean beside the Python
~~~

## The model

The principals are the prover, the verifier, and the trusted hardware: replayers, wardens, pingers. An actor is one
principal's part in one protocol, a transducer over one ordered log of events (`verity.primitives.events`). A send goes
to a principal, and every other actor of that principal receives it; a record (a send to no one) goes to the sender's own
principal, which is how one protocol's verifier reads what another's established. What an actor observes and computes
in between is its own state and never enters the log. `verity.run.run(principals, view)` takes each principal's actors,
plays them against each other and yields one principal's view. The auditor trusts the verifier, so the verifier's view
is the output: what its actors establish are records in it. Replayers, wardens and pingers are trusted too but sealed:
what they see stays their own, and reaches the verifier only as what they send it.

Everything over the prover's work reads one stream (`verity.primitives.registration`). First, before any coins, the
registration: the prover's builder, the program that builds each execution's circuit and partition from a bounded
bitstring of advice (the runtime choices no one could fix ahead of time: how requests were batched, which step a
session continues, a preemption, an autotune pick), holding the baseline (weights, constants) as every circuit's first
inputs. Then per execution its advice and a root over the boundary wires of its partition's regions, until the prover
closes the stream: one epoch. A partition has two levels (`verity.primitives.partition`): isolation units, which NCI
bounds, each split into regions, which proofs sample. An execution may read state an earlier one wrote (as a decode
step reads its KV cache): that wire is committed once, in the writer's root, so the step that reads it reads what the
step that wrote it committed, and the stream's isolation units join through it: a session is one isolation unit, which
reads only its prompt however many steps it took. The verifier rebuilds each execution's circuit from its advice
(`Stream`). Every statement over the stream names its digest.

What the verifier establishes splits in two. **Accounting** says what happened, in three families: integrity and
utilization read the stream, and coverage reads them both.
- **integrity**: the executions are what their circuits compute, up to a bound on wrong units;
- **utilization**: at least so much compute (work, keyed by a salt) and memory (space, a timed audit of an encoding)
  was used between a challenge and a clock's reading;
- **coverage**: of each resource's capacity (from an inventory of devices and the census's facts per device, or from
  what the warden saw leave), how much the other two account for, and the headroom left, where anything else could
  have run.

**Compliance** says whether that was allowed: a policy's parameters are its limits, and its statement says whether the
epoch kept to them. NCI is the first; its limits include how much headroom it tolerates, since whatever ran in the
headroom went unchecked. An epoch's verdict says nothing outside its window, so a period of epochs complies only if
each does and their windows leave no gap (`nci.period`).

The two products, `remote` and `onsite`, run the same accounting and policies, and differ in who reports each
reading. Remotely the verifier has nothing at the site: its own clock times the work, and the prover declares its
inventory, so remote shows that the declared fleet was busy with declared work and never that there was no other.
Onsite the verifier's hardware counts the fleet, times work and memory challenges, replays executions and watches the
links, so it also bounds memory and network, and its inventory is a measurement.

A construction is a frozen dataclass of its parameters, with a method per principal it involves that returns that
principal's actor (`Replay(law, delta).verifier()`). A trusted principal's method takes its readings
(`pinger(fleet, work_window, space_seconds)`), and an actor that reads one takes the principal it trusts for it
(`Work.verifier(clock_from)`). Choosing a construction is choosing a value. A product is a construction whose fields
are constructions: its methods return the lists of its parts' actors, and its guarantee is proved from theirs. Actors
have no names and are not wired: anyone can send to anyone, so an actor waits for each kind of event only from the
principal it trusts for it (`receive(kind, sender)`, `gather`, `collect`), and an event keeps in its mailbox until it
waits for it. Nothing orchestrates the actors: each moves when an event it waits for reaches it, so each keeps its own
cadence over the log. Order in the log is logical; physical time enters only as trusted readouts in what a trusted
principal sends.

## Layers

Imports point down only, in Python and in Lean:

~~~text
verity/run.py              the runner
verity/remote.py, onsite.py   the products: who reports each reading, and which policies judge the account
verity/compliance          whether it was allowed: a module per policy (nci)
verity/accounting          what happened; coverage reads the families below it:
    coverage               the headroom per resource (compute, memory, network), from the inventory and the warden
    utilization            work and space used between a challenge and a clock's reading
    integrity              whether the executions are what their circuits compute (replay, proofs)
verity/primitives          events, circuits and partitions, commitments, randomness, the registration and its stream
~~~

`examples/` sit on top: a run of each product, integrity alone, and a datacenter over a period of epochs. An
integration (vLLM) is a prover.

## Conventions

- **Lean sits beside the Python it states.** `verity/primitives/Circuit.lean` sits beside `circuit.py`. There is one
  Lake package at the root (`lakefile.toml`), and a module's name is its path. The wheel excludes `*.lean` and
  `.lake`.
- **`__init__.py` holds only a docstring.** Code goes in named modules, so the files can be dropped (making the
  folders namespace packages) once `packages/verity` is gone and nothing else named `verity` is on the path.
- **Heavy constructions have their own `pyproject.toml`.** It sits beside the source and declares the dependencies
  (`integrity/proofs/sp1/`); the source is imported only by whoever asks for it.
